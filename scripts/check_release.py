#!/usr/bin/env python3
"""Validate release inputs and generate the archive checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SUBMODULES = ("transplat", "mvsplat", "depthsplat")
REQUIRED_THIRD_PARTY = (
    "TranSplat",
    "MVSplat",
    "DepthSplat",
    "DINOv2",
    "iFlow",
    "ASAP7",
    "DeepScaleTool",
    "Ramulator 2",
    "DRAMPower",
    "Re10K",
    "ACID",
    "DL3DV",
)
LOCAL_PATH_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"[A-Za-z]:\\Users\\"),
)
DEEPSCALE_ASSETS = {
    "hardware/scaling/vendor/DeepScaleTool.xlsm": "561a3f8f5e91a3c496d6e0f4262c09412209f3bbc6d22b714cde323ebd958df8",
    "hardware/scaling/vendor/LICENSE-GPL-3.0.txt": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
}


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_files() -> list[Path]:
    from scripts.build_archive import include_in_source_release, normalize_archive_relative

    release_manifest = ROOT / "release-manifest.json"
    if release_manifest.is_file():
        record = json.loads(release_manifest.read_text(encoding="utf-8"))
        files = record.get("files")
        if (
            record.get("bundle_kind") != "source"
            or record.get("validation", {}).get("pass") is not True
            or not isinstance(files, dict)
            or not files
        ):
            raise RuntimeError("release manifest has an invalid source file set")
        selected = []
        normalized_paths = set()
        non_normalized = []
        for relative in files:
            normalized = normalize_archive_relative(relative)
            if normalized in normalized_paths:
                raise RuntimeError(
                    f"release manifest has a duplicate normalized file: {normalized}"
                )
            normalized_paths.add(normalized)
            if normalized != relative:
                non_normalized.append(relative)
                continue
            pure = PurePosixPath(normalized)
            path = ROOT / pure
            if (
                not include_in_source_release(pure)
                or path.is_symlink()
                or not path.is_file()
            ):
                raise RuntimeError(f"release manifest has an invalid file: {relative}")
            selected.append(path)
        if non_normalized:
            raise RuntimeError(
                "release manifest has non-normalized files: "
                + ", ".join(sorted(non_normalized)[:10])
            )
        return sorted(selected)
    output = git("ls-files", "--recurse-submodules", "-z")
    selected = []
    for name in output.split("\0"):
        if not name:
            continue
        normalized = normalize_archive_relative(name)
        if normalized != name:
            raise RuntimeError(f"tracked release path is not normalized: {name}")
        path = ROOT / normalized
        if not include_in_source_release(PurePosixPath(normalized)):
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"tracked release path is not a regular file: {name}")
        selected.append(path)
    return sorted(selected)


def check_local_paths(files: list[Path]) -> list[str]:
    failures = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in LOCAL_PATH_PATTERNS:
            if pattern.search(text):
                try:
                    display = path.relative_to(ROOT)
                except ValueError:
                    display = path
                failures.append(f"author-local path in {display}")
                break
    return failures


def submodule_record() -> tuple[dict[str, str], list[str]]:
    records = {}
    failures = []
    for name in SUBMODULES:
        index_line = git("ls-files", "--stage", name)
        fields = index_line.split()
        if len(fields) < 4 or fields[0] != "160000":
            failures.append(f"{name} is not a gitlink")
            continue
        expected = fields[1]
        actual = git("-C", str(ROOT / name), "rev-parse", "HEAD")
        records[name] = expected
        if actual != expected:
            failures.append(f"{name} checkout {actual} does not match gitlink {expected}")
    return records, failures


def check_third_party() -> list[str]:
    path = ROOT / "THIRD_PARTY.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return [f"THIRD_PARTY.md is missing {name}" for name in REQUIRED_THIRD_PARTY if name not in text]


def check_dataset_sources(path: Path | None = None) -> list[str]:
    path = path or ROOT / "artifact/manifests/datasets.json"
    if not path.is_file():
        return ["artifact/manifests/datasets.json is missing"]
    record = json.loads(path.read_text(encoding="utf-8"))
    datasets = record.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"re10k", "acid", "dl3dv"}:
        return ["dataset source manifest does not cover Re10K, ACID, and DL3DV"]
    failures = []
    claim_status = json.loads((ROOT / "artifact/claim_status.json").read_text(encoding="utf-8"))
    claimed_datasets = {
        pair.split("/", 1)[1]
        for pair, status in claim_status["software_pairs"].items()
        if status == "CLAIMED"
    }
    # Re10K and ACID are locally prepared, release-checked Functional inputs even
    # when their paper-result rows are suspended.  DL3DV remains optional while
    # gated and is required only if a corresponding result is claimed.
    verified_datasets = {"re10k", "acid"} | claimed_datasets
    for name, item in datasets.items():
        revision = item.get("prepared_source_revision")
        if name in verified_datasets:
            revision_valid = (
                bool(re.fullmatch(r"archive-sha256:[0-9a-f]{64}", str(revision)))
                if name in {"re10k", "acid"}
                else bool(re.fullmatch(r"[0-9a-f]{40}", str(revision)))
            )
            if not revision_valid:
                failures.append(f"{name} has no verified prepared source revision")
        representations = item.get("representations")
        if representations is not None:
            if name != "dl3dv" or not isinstance(representations, dict) or set(representations) != {"native", "re10k"}:
                failures.append(f"{name} has an invalid prepared representation contract")
            else:
                for representation, prepared in sorted(representations.items()):
                    if not isinstance(prepared, dict):
                        failures.append(
                            f"{name}/{representation} has invalid representation metadata"
                        )
                        continue
                    tree_hash = prepared.get("expected_tree_sha256")
                    if name in verified_datasets and (not isinstance(tree_hash, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", tree_hash
                    )):
                        failures.append(
                            f"{name}/{representation} has no verified dataset tree SHA256"
                        )
                    if not prepared.get("path") or not prepared.get("schema"):
                        failures.append(
                            f"{name}/{representation} has incomplete representation metadata"
                        )
        else:
            tree_hash = item.get("expected_tree_sha256")
            if name in verified_datasets and (not isinstance(tree_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", tree_hash
            )):
                failures.append(f"{name} has no verified dataset tree SHA256")
        license_record = item.get("license", {})
        license_status = license_record.get("status")
        terms_verified = license_status in {
            "verified",
            "verified_no_redistribution",
        }
        redistribution_safe = (
            license_status != "verified_no_redistribution"
            or license_record.get("redistribution_in_zenodo") is False
        )
        if (
            not terms_verified
            or not license_record.get("terms_url")
            or not redistribution_safe
        ):
            failures.append(f"{name} license terms are not verified")
    return failures


def check_runtime_asset_manifest(path: Path | None = None) -> list[str]:
    path = path or ROOT / "artifact/manifests/runtime_assets.json"
    if not path.is_file():
        return ["runtime asset manifest is missing"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if manifest.get("schema_version") != "1.0" or not isinstance(files, list) or not files:
        return ["runtime asset manifest has an invalid schema"]
    failures = []
    paths = set()
    names = set()
    for ordinal, record in enumerate(files):
        label = record.get("name", f"entry {ordinal}") if isinstance(record, dict) else f"entry {ordinal}"
        if not isinstance(record, dict):
            failures.append(f"invalid runtime asset record: {label}")
            continue
        relative = record.get("path")
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or relative in paths
        ):
            failures.append(f"invalid or duplicate runtime asset path: {relative}")
        else:
            paths.add(relative)
        name = record.get("name")
        if not isinstance(name, str) or not name or name in names:
            failures.append(f"invalid or duplicate runtime asset name: {name}")
        else:
            names.add(name)
        profiles = record.get("profiles")
        if (
            not isinstance(profiles, list)
            or not profiles
            or not set(profiles) <= {"quick", "classic", "depthsplat"}
        ):
            failures.append(f"invalid runtime asset profiles: {label}")
        models = record.get("models")
        if models is not None and (
            not isinstance(models, list)
            or not models
            or len(models) != len(set(models))
            or not set(models) <= {"transplat", "mvsplat", "depthsplat"}
        ):
            failures.append(f"invalid runtime asset models: {label}")
        if not isinstance(record.get("url"), str) or not record["url"].startswith("https://"):
            failures.append(f"runtime asset has no HTTPS source: {label}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            failures.append(f"runtime asset has no valid SHA256: {label}")
        size = record.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            failures.append(f"runtime asset has no positive size: {label}")
        kind = record.get("kind")
        if kind not in {"file", "tar_gz"}:
            failures.append(f"unsupported runtime asset kind: {label}")
        elif kind == "tar_gz":
            if not re.fullmatch(r"[0-9a-f]{40}", str(record.get("commit", ""))):
                failures.append(f"runtime source archive has no pinned commit: {label}")
            if not record.get("archive_root") or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("license_sha256", ""))
            ):
                failures.append(f"runtime source archive provenance is incomplete: {label}")
    return failures


def check_checkpoint_manifest(path: Path | None = None) -> list[str]:
    path = path or ROOT / "artifact/manifests/checkpoints.json"
    if not path.is_file():
        return ["checkpoint manifest is missing"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    expected = {
        ("transplat", "re10k"): ("classic", ["re10k", "dl3dv"]),
        ("transplat", "acid"): ("classic", ["acid"]),
        ("mvsplat", "re10k"): ("classic", ["re10k", "dl3dv"]),
        ("mvsplat", "acid"): ("classic", ["acid"]),
        ("depthsplat", "re10k"): ("depthsplat", ["re10k", "acid"]),
        ("depthsplat", "dl3dv"): ("depthsplat", ["dl3dv"]),
    }
    if manifest.get("schema_version") != "1.0" or not isinstance(records, list):
        return ["checkpoint manifest has an invalid schema"]
    failures = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"invalid checkpoint record: entry {ordinal}")
            continue
        pair = (record.get("model"), record.get("dataset"))
        if pair not in expected or pair in seen_pairs:
            failures.append(f"invalid or duplicate checkpoint pair: {pair[0]}/{pair[1]}")
            continue
        seen_pairs.add(pair)
        profile, evaluation_datasets = expected[pair]
        label = f"{pair[0]}/{pair[1]}"
        relative = record.get("path")
        expected_path = f"{pair[0]}/checkpoints/{pair[1]}.ckpt"
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or relative != expected_path
            or relative in seen_paths
        ):
            failures.append(f"invalid or duplicate checkpoint path: {label}")
        else:
            seen_paths.add(relative)
        if record.get("profile") != profile:
            failures.append(f"checkpoint profile mismatch: {label}")
        if record.get("evaluation_datasets") != evaluation_datasets:
            failures.append(f"checkpoint evaluation mapping mismatch: {label}")
        url = record.get("url")
        if not isinstance(url, str) or re.fullmatch(
            r"https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/[^/]+",
            url,
        ) is None:
            failures.append(f"checkpoint source is not commit-pinned HTTPS: {label}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            failures.append(f"checkpoint has no valid SHA256: {label}")
        size = record.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            failures.append(f"checkpoint has no positive size: {label}")
    missing = sorted(set(expected) - seen_pairs)
    if missing:
        failures.append(
            "checkpoint manifest is missing pairs: "
            + ", ".join(f"{model}/{dataset}" for model, dataset in missing)
        )
    return failures


def check_deepscale_assets(root: Path = ROOT) -> list[str]:
    failures = []
    for relative, expected in DEEPSCALE_ASSETS.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing pinned DeepScaleTool asset: {relative}")
        elif sha256_file(path) != expected:
            failures.append(f"DeepScaleTool asset hash mismatch: {relative}")
    return failures


def check_orin_contract(path: Path | None = None) -> list[str]:
    path = path or ROOT / "environments/orin/contract.json"
    if not path.is_file():
        return ["Orin environment contract is missing"]
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "runtime_autodetect":
        return ["Orin environment contract does not use runtime autodetection"]
    required = (
        "device_model_contains",
        "power_mode_contains",
        "require_jetson_clocks",
        "maximum_allowed_temperature_c",
        "timing_repetitions",
    )
    missing = [key for key in required if contract.get(key) in (None, "")]
    return ["Orin environment contract is missing " + ", ".join(missing)] if missing else []


def check_evaluation_protocol(path: Path | None = None) -> list[str]:
    from scripts.compile_protocol import compile_protocol

    path = path or ROOT / "artifact/evaluation_protocol.json"
    expected_path = ROOT / "artifact/expected_results.json"
    if not path.is_file():
        return ["evaluation sample protocol is missing"]
    try:
        compiled = compile_protocol(path, ROOT)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"evaluation protocol source contract is invalid: {exc}"]
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if set(compiled["pairs"]) != set(expected["table1"]):
        return ["evaluation protocol does not cover every claimed model/dataset pair"]
    claim_status = json.loads((ROOT / "artifact/claim_status.json").read_text(encoding="utf-8"))
    claimed_pairs = {
        pair for pair, state in claim_status["software_pairs"].items() if state == "CLAIMED"
    }
    if any(
        not isinstance(record.get("dataset_tree_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", record["dataset_tree_sha256"])
        for pair, record in protocol["pairs"].items()
        if pair in claimed_pairs
    ):
        return ["evaluation protocol has unresolved prepared dataset tree SHA256 values"]
    return []


def check_claim_status(path: Path | None = None) -> list[str]:
    path = path or ROOT / "artifact/claim_status.json"
    if not path.is_file():
        return ["artifact claim status is missing"]
    record = json.loads(path.read_text(encoding="utf-8"))
    expected_pairs = {
        f"{model}/{dataset}"
        for model in ("transplat", "mvsplat", "depthsplat")
        for dataset in ("re10k", "acid", "dl3dv")
    }
    failures = []
    allowed_pair_states = {
        "software_pairs": {
            "CLAIMED",
            "NOT_CLAIMED_GATED_DATA",
            "NOT_CLAIMED_SAES_SPARSE_QUALITY_MISMATCH",
        },
        "mechanism_pairs": {
            "CLAIMED",
            "NOT_CLAIMED_GATED_DATA",
            "NOT_CLAIMED_SAES_PROTOCOL_MISMATCH",
            "NOT_CLAIMED_FSDR_LSH_AND_SAES_PROTOCOL_MISMATCH",
        },
    }
    for field in ("software_pairs", "mechanism_pairs"):
        states = record.get(field)
        if not isinstance(states, dict) or set(states) != expected_pairs:
            failures.append(f"claim status {field} does not cover all nine pairs")
            continue
        invalid = sorted(set(states.values()) - allowed_pair_states[field])
        if invalid:
            failures.append(f"claim status {field} has invalid states: {invalid}")
    aggregate_states = {
        "figure8": {
            "CLAIMED",
            "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION",
            "NOT_CLAIMED_NO_ORIN_EVIDENCE",
        },
        "figure11": {"CLAIMED", "NOT_CLAIMED_INCOMPLETE_NINE_PAIR_MATRIX"},
        "sensitivity": {"CLAIMED", "NOT_CLAIMED_INCOMPLETE_NINE_PAIR_MATRIX"},
        "rtl": {"CLAIMED"},
        "dram": {"FUNCTIONAL_ONLY"},
        "physical_asap7": {"CLAIMED", "NOT_CLAIMED_RESOURCE_LIMIT"},
        "deepscale": {"CLAIMED", "NOT_CLAIMED_NO_PHYSICAL_INPUT"},
    }
    for field, allowed in aggregate_states.items():
        if record.get(field) not in allowed:
            failures.append(f"claim status {field} is invalid")
    if (
        record.get("physical_asap7") != "CLAIMED"
        and record.get("deepscale") == "CLAIMED"
    ):
        failures.append("DeepScale cannot be claimed without claimed ASAP7 input")
    return failures


def _binding_failures(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, subject: str
) -> list[str]:
    """Return field-level failures for one staged source/config binding."""
    from scripts.stage_reference_results import (
        SOURCE_PROVENANCE_FIELDS,
        canonical_sha256,
    )

    failures: list[str] = []
    actual_source = actual.get("source")
    expected_source = expected.get("source")
    if not isinstance(actual_source, Mapping):
        failures.append(f"{subject} is missing source provenance binding")
    elif not isinstance(expected_source, Mapping):
        failures.append(f"cannot determine current source provenance for {subject}")
    else:
        for field in SOURCE_PROVENANCE_FIELDS:
            if field not in actual_source:
                failures.append(f"{subject} source provenance missing {field}")
            elif actual_source[field] != expected_source.get(field):
                failures.append(f"{subject} source provenance mismatch {field}")

    actual_mechanism = actual.get("mechanism")
    expected_mechanism = expected.get("mechanism")
    if not isinstance(actual_mechanism, Mapping):
        failures.append(f"{subject} is missing mechanism provenance binding")
    elif not isinstance(expected_mechanism, Mapping):
        failures.append(f"cannot determine current mechanism provenance for {subject}")
    else:
        for field in (
            "mechanism_config_sha256",
            "calibration_status",
            "calibration_provenance_sha256",
        ):
            if field not in actual_mechanism:
                failures.append(f"{subject} mechanism provenance missing {field}")
            elif actual_mechanism[field] != expected_mechanism.get(field):
                failures.append(f"{subject} mechanism provenance mismatch {field}")
        calibration = actual_mechanism.get("calibration_provenance")
        digest = actual_mechanism.get("calibration_provenance_sha256")
        if not isinstance(calibration, Mapping):
            failures.append(f"{subject} mechanism provenance missing calibration_provenance")
        elif canonical_sha256(calibration) != digest:
            failures.append(
                f"{subject} mechanism calibration provenance digest is inconsistent"
            )
    return failures


def check_reference_results(
    root: Path = ROOT,
    *,
    reference_results: Path | None = None,
    expected_binding: Mapping[str, Any] | None = None,
) -> list[str]:
    from scripts.stage_reference_results import (
        release_evidence_binding,
        required_categories,
        validate_generated_records,
    )

    reference_root = (
        Path(reference_results).resolve()
        if reference_results is not None
        else root / "artifact/reference_results"
    )
    manifest_path = reference_root / "manifest.json"
    if not manifest_path.is_file():
        return ["reference evidence manifest is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    if manifest.get("status") != "complete" or manifest.get("validation_status") != "PASS":
        failures.append("reference evidence has not been staged from a passing validation run")
    if manifest.get("validation_require_key_results") is not True:
        failures.append("reference evidence was not validated with --require-key-results")
    required = required_categories()
    if not required <= set(manifest.get("categories", [])):
        failures.append("reference evidence categories are incomplete")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        failures.append("reference evidence file manifest is empty")
        return failures
    for relative, record in files.items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.parts[:1] != ("evidence",):
            failures.append(f"unsafe reference evidence path: {relative}")
            continue
        path = reference_root / pure
        if not isinstance(record, dict):
            failures.append(f"invalid reference evidence record: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing reference evidence: {relative}")
            continue
        if path.stat().st_size != record.get("size") or sha256_file(path) != record.get("sha256"):
            failures.append(f"reference evidence hash mismatch: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if any(pattern.search(text) for pattern in LOCAL_PATH_PATTERNS):
            failures.append(f"author-local path in reference evidence: {relative}")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        failures.append(
            "reference evidence manifest has no provenance binding (legacy Functional-era evidence)"
        )
        return failures
    try:
        current_binding = (
            dict(expected_binding)
            if expected_binding is not None
            else release_evidence_binding(root)
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"cannot determine current release provenance: {exc}")
        return failures
    failures.extend(
        _binding_failures(
            provenance,
            current_binding,
            subject="reference evidence manifest",
        )
    )

    generated_records, generated_failures = validate_generated_records(
        {
            reference_root / PurePosixPath(relative): PurePosixPath(relative).relative_to(
                "evidence"
            )
            for relative in files
            if PurePosixPath(relative).parts[:1] == ("evidence",)
            and (reference_root / PurePosixPath(relative)).is_file()
        },
        expected_binding=current_binding,
    )
    failures.extend(f"reference {failure}" for failure in generated_failures)
    declared_records = provenance.get("generated_records")
    if not isinstance(declared_records, Mapping):
        failures.append("reference evidence manifest is missing generated-record bindings")
    elif not generated_failures:
        if set(declared_records) != set(generated_records):
            failures.append(
                "reference evidence manifest generated-record bindings do not cover the staged results"
            )
        else:
            for relative, binding in generated_records.items():
                if declared_records.get(relative) != binding:
                    failures.append(
                        f"reference evidence manifest generated-record binding mismatch: {relative}"
                    )
    return failures


def check_doi() -> tuple[str | None, list[str]]:
    path = ROOT / "artifact/release.json"
    if not path.is_file():
        return None, ["artifact/release.json is missing"]
    record = json.loads(path.read_text(encoding="utf-8"))
    doi = record.get("zenodo_doi")
    if not isinstance(doi, str) or not re.fullmatch(r"10\.5281/zenodo\.\d+", doi):
        return None, ["release metadata has no valid Zenodo DOI"]
    return doi, []


def build_manifest(
    require_doi: bool = False, *, reference_results: Path | None = None
) -> dict[str, Any]:
    files = archive_files()
    submodules, failures = submodule_record()
    failures.extend(check_local_paths(files))
    failures.extend(check_third_party())
    failures.extend(check_dataset_sources())
    failures.extend(check_checkpoint_manifest())
    failures.extend(check_runtime_asset_manifest())
    failures.extend(check_deepscale_assets())
    failures.extend(check_orin_contract())
    failures.extend(check_claim_status())
    failures.extend(check_evaluation_protocol())
    failures.extend(check_reference_results(ROOT, reference_results=reference_results))
    doi = None
    if require_doi:
        doi, doi_failures = check_doi()
        failures.extend(doi_failures)
    return {
        "schema_version": "1.0",
        "git_commit": git("rev-parse", "HEAD"),
        "submodules": submodules,
        "zenodo_doi": doi,
        "files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in files
        },
        "validation": {"pass": not failures, "failures": failures},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--require-doi", action="store_true")
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.archive):
        parser.error("specify exactly one of --manifest or --archive")
    try:
        if args.archive:
            from scripts.build_archive import verify

            record = verify(args.archive.resolve())
            if args.require_doi and not record.get("zenodo_doi"):
                raise ValueError("archive manifest has no Zenodo DOI")
            print(json.dumps(record, indent=2, sort_keys=True))
            return 0
        manifest = build_manifest(args.require_doi)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, RuntimeError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not manifest["validation"]["pass"]:
        for failure in manifest["validation"]["failures"]:
            print(f"error: {failure}", file=sys.stderr)
        return 2
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
