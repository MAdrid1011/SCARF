#!/usr/bin/env python3
"""Validate and import a platform claim-evidence export.

The platform export is validated in place before any release output is
modified.  This command accepts only real claim records: format templates,
simulation rehearsals, diagnostic traces, and preregistered configurations are
rejected.  Use ``--check-only`` while transferring a large export; pass
``--install-config`` only after the report is ready to be installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("transplat", "mvsplat", "depthsplat")
DATASETS = ("re10k", "acid", "dl3dv")
WORKFLOWS = ("quality", "mechanisms", "speedup")
COPY_DIRECTORIES = (
    "calibration",
    "timing-backend",
    "quality",
    "mechanisms",
    "speedup",
    "environments",
    "reports",
    "rtl",
    "dram",
    "datasets",
)
MARKER_KEYS = {
    "simulation_only",
    "synthetic_fixture",
    "format_only",
    "not_for_release",
}
MARKER_STRINGS = {
    "pending_validation",
    "predicted",
    "simulation_prediction",
    "diagnostic",
    "diagnostic_device_timing",
    "diagnostic-interface-fixture",
    "not_computed",
    "PAPER_REFERENCE_ONLY",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _marker_path(value: Any, path: str = "record") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in MARKER_KEYS and (
                child is True or (key == "format_only" and child == "template")
            ):
                return f"{path}.{key}"
            found = _marker_path(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _marker_path(child, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str):
        if value in MARKER_STRINGS or value.startswith("<fill-"):
            return path
    return None


def _assert_real_record(record: Mapping[str, Any], label: str) -> None:
    marker = _marker_path(record)
    if marker:
        raise ValueError(f"{label} contains a non-claim marker at {marker}")
    if record.get("claim_eligible") is False:
        raise ValueError(f"{label} is marked claim_eligible=false")


def _expected_counts(root: Path, profile: str) -> dict[str, int]:
    if profile == "full":
        protocol = _load(root / "artifact/evaluation_protocol.json", "evaluation protocol")
        pairs = protocol.get("pairs")
        if not isinstance(pairs, Mapping):
            raise ValueError("evaluation protocol has no pair records")
        return {
            dataset: int(pairs[f"transplat/{dataset}"]["sample_count"])
            for dataset in DATASETS
        }
    if profile == "reviewer":
        manifest = _load(
            root / "artifact/protocol/reviewer/manifest.json",
            "reviewer evidence profile",
        )
        datasets = manifest.get("datasets")
        if not isinstance(datasets, Mapping):
            raise ValueError("reviewer evidence profile has no datasets")
        return {dataset: int(datasets[dataset]["sample_count"]) for dataset in DATASETS}
    raise ValueError(f"unsupported profile: {profile}")


def _safe_sample_path(parent: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe")
    path = (parent / Path(*relative.parts)).resolve()
    parent = parent.resolve()
    if path != parent and parent not in path.parents:
        raise ValueError(f"{label} path escapes its aggregate directory")
    return path


def _environment_digest(path: Path, profile: str) -> str:
    record = _load(path, f"{profile} environment")
    details = record.get("details")
    if not isinstance(details, Mapping):
        raise ValueError(f"{profile} environment has no details")
    canonical = {
        "profile": profile,
        "python": details.get("python"),
        "implementation": details.get("implementation"),
        "torch": details.get("torch"),
        "torchvision": details.get("torchvision"),
        "torch_cuda": details.get("torch_cuda"),
        "lock_sha256": details.get("lock_sha256"),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_sample_lineage(
    aggregate_path: Path, record: Mapping[str, Any], *, workflow: str
) -> int:
    provenance = record.get("provenance")
    evaluation = provenance.get("evaluation") if isinstance(provenance, Mapping) else None
    if not isinstance(evaluation, Mapping) or evaluation.get("kind") != "dataset_aggregate":
        raise ValueError(f"{workflow}/{aggregate_path.parent.name} is not a dataset aggregate")
    entries = evaluation.get("sample_results")
    count = evaluation.get("sample_count")
    if not isinstance(entries, list) or not isinstance(count, int) or len(entries) != count:
        raise ValueError(f"{workflow}/{aggregate_path.parent.name} has incomplete sample lineage")
    indices: list[int] = []
    from scripts.validate_result import validate

    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{workflow}/{aggregate_path.parent.name} has an invalid sample entry")
        sample_path = _safe_sample_path(aggregate_path.parent, entry.get("path"), "sample")
        if not sample_path.is_file():
            raise ValueError(f"sample result is missing: {sample_path}")
        if entry.get("sha256") != sha256_file(sample_path):
            raise ValueError(f"sample result hash mismatch: {sample_path}")
        sample = _load(sample_path, "sample result")
        _assert_real_record(sample, str(sample_path))
        if sample.get("schema_version") != "2.1":
            raise ValueError(f"sample result is not schema 2.1: {sample_path}")
        validate(sample)
        sample_eval = sample.get("provenance", {}).get("evaluation", {})
        index = sample_eval.get("sample_index")
        if not isinstance(index, int) or index != entry.get("sample_index"):
            raise ValueError(f"sample selection mismatch: {sample_path}")
        indices.append(index)
    if indices != sorted(indices) or len(set(indices)) != count:
        raise ValueError(f"{workflow}/{aggregate_path.parent.name} sample indices are not complete")
    return count


def validate_export(
    source: Path,
    *,
    root: Path = ROOT,
    profile: str = "full",
    timing_manifest: Path | None = None,
) -> dict[str, Any]:
    """Validate a complete external export and return an import summary."""
    source = Path(source).resolve()
    root = Path(root).resolve()
    if not source.is_dir():
        raise ValueError(f"evidence export is not a directory: {source}")
    counts = _expected_counts(root, profile)

    config_path = source / "calibration/mechanism_config.json"
    if not config_path.is_file():
        config_path = source / "artifact/mechanism_config.json"
    if not config_path.is_file():
        raise ValueError("calibration/mechanism_config.json is missing")
    config = _load(config_path, "mechanism configuration")
    _assert_real_record(config, str(config_path))
    from scripts.mechanism_config import require_calibrated_mechanism

    try:
        _calibrated, mechanism = require_calibrated_mechanism(config_path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"calibration configuration is not claim-ready: {exc}") from exc
    config_sha256 = sha256_file(config_path)
    if mechanism.get("status") != "calibrated":
        raise ValueError("calibration configuration must have status=calibrated")

    manifest_path = Path(timing_manifest or source / "timing-backend/manifest.json").resolve()
    if not manifest_path.is_file():
        raise ValueError(f"timing backend manifest is missing: {manifest_path}")
    from scripts.claim_timing_backend import load_claim_timing_manifest

    try:
        backend = load_claim_timing_manifest(manifest_path, root=source)
    except (OSError, ValueError) as exc:
        raise ValueError(f"timing backend is not claim-ready: {exc}") from exc
    timing_sha256 = sha256_file(manifest_path)
    expected_samples = len(MODELS) * sum(counts.values())
    if len(backend.samples) != expected_samples:
        raise ValueError(
            f"timing backend has {len(backend.samples)} samples, expected {expected_samples}"
        )

    pair_summary: dict[str, Any] = {}
    from scripts.execution_contract import execution_contract
    from scripts.validate_ae import orin_evidence_reason
    from scripts.validate_result import validate

    for workflow in WORKFLOWS:
        for model in MODELS:
            for dataset in DATASETS:
                pair = f"{model}/{dataset}"
                result_path = source / workflow / f"{model}_{dataset}" / "results.json"
                if not result_path.is_file():
                    raise ValueError(f"missing {workflow} aggregate: {result_path}")
                result = _load(result_path, f"{workflow} aggregate")
                _assert_real_record(result, str(result_path))
                if result.get("schema_version") != "2.1":
                    raise ValueError(f"{workflow}/{pair} must use result schema 2.1")
                validate(result)
                provenance = result.get("provenance", {})
                contract = execution_contract(result)
                if not isinstance(contract, Mapping) or contract.get("run_class") != "claim":
                    raise ValueError(f"{workflow}/{pair} was not executed in claim mode")
                evaluation = provenance.get("evaluation", {})
                dataset_provenance = provenance.get("dataset")
                environment_provenance = provenance.get("environment")
                if (
                    provenance.get("model") != model
                    or not isinstance(dataset_provenance, Mapping)
                    or dataset_provenance.get("name") != dataset
                    or evaluation.get("sample_count") != counts[dataset]
                ):
                    raise ValueError(f"{workflow}/{pair} identity or sample count is wrong")
                timing_binding = provenance.get("timing_backend", {})
                if timing_binding.get("manifest", {}).get("sha256") != timing_sha256:
                    raise ValueError(f"{workflow}/{pair} is bound to a different timing manifest")
                if provenance.get("mechanism_config_sha256") != config_sha256:
                    raise ValueError(f"{workflow}/{pair} is bound to a different mechanism config")
                if not isinstance(environment_provenance, Mapping):
                    raise ValueError(f"{workflow}/{pair} has no environment provenance")
                profile_name = environment_provenance.get("profile")
                if not isinstance(profile_name, str) or not profile_name:
                    raise ValueError(f"{workflow}/{pair} has an invalid environment profile")
                environment_path = source / "environments" / f"{profile_name}.json"
                if not environment_path.is_file():
                    raise ValueError(f"environment record is missing: {environment_path}")
                if _environment_digest(environment_path, profile_name) != provenance["environment"].get(
                    "digest_sha256"
                ):
                    raise ValueError(f"environment digest mismatch: {result_path}")
                sample_count = _validate_sample_lineage(result_path, result, workflow=workflow)
                if workflow == "speedup":
                    if result.get("evidence_class") != "independent_measurement":
                        raise ValueError(f"speedup/{pair} is not independent measurement evidence")
                    reason = orin_evidence_reason(
                        result,
                        evaluation.get("sample_selection_sha256"),
                        result_path.parent,
                    )
                    if reason is not None:
                        raise ValueError(f"speedup/{pair} Orin evidence is incomplete: {reason}")
                pair_summary[f"{workflow}:{pair}"] = {"sample_count": sample_count, "status": "PASS"}

    validation_path = source / "validation.json"
    validation = _load(validation_path, "AE validation")
    if validation.get("status") != "PASS" or validation.get("require_key_results") is not True:
        raise ValueError("validation.json must be PASS with require_key_results=true")
    return {
        "schema_version": "scarf-claim-import-v1",
        "status": "READY",
        "profile": profile,
        "source_root": str(source),
        "timing_manifest_sha256": timing_sha256,
        "mechanism_config_sha256": config_sha256,
        "timing_sample_count": len(backend.samples),
        "pairs": pair_summary,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def _copy_export(source: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for directory in COPY_DIRECTORIES:
        origin = source / directory
        if not origin.is_dir():
            continue
        target = destination / directory
        shutil.copytree(origin, target, dirs_exist_ok=True, symlinks=False)
        copied.append(directory)
    for path in sorted(source.glob("manifest-*.json")):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
            copied.append(path.name)
    validation = source / "validation.json"
    if validation.is_file():
        shutil.copy2(validation, destination / validation.name)
        copied.append(validation.name)
    return copied


def import_export(
    source: Path,
    *,
    output_root: Path,
    root: Path = ROOT,
    profile: str = "full",
    timing_manifest: Path | None = None,
    install_config: bool = False,
    check_only: bool = False,
) -> dict[str, Any]:
    report = validate_export(
        source,
        root=root,
        profile=profile,
        timing_manifest=timing_manifest,
    )
    if check_only:
        report["mode"] = "check-only"
        return report
    output_root = Path(output_root).resolve()
    if output_root == root or root in output_root.parents:
        raise ValueError("output root must not be the checkout root")
    copied = _copy_export(Path(source).resolve(), output_root)
    if install_config:
        config_source = Path(source).resolve() / "calibration/mechanism_config.json"
        if not config_source.is_file():
            config_source = Path(source).resolve() / "artifact/mechanism_config.json"
        artifact_config = Path(root).resolve() / "artifact/mechanism_config.json"
        temporary = artifact_config.with_suffix(".json.import-tmp")
        temporary.write_bytes(config_source.read_bytes())
        os.replace(temporary, artifact_config)
        report["installed_config"] = str(artifact_config)
    report["mode"] = "imported"
    report["copied"] = copied
    report_path = output_root / "claim-import.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="external platform evidence export")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/ae")
    parser.add_argument("--root", type=Path, default=ROOT, help="release checkout root")
    parser.add_argument("--profile", choices=("full", "reviewer"), default="full")
    parser.add_argument("--timing-manifest", type=Path)
    parser.add_argument("--install-config", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = import_export(
            args.input,
            output_root=args.output_root,
            root=args.root,
            profile=args.profile,
            timing_manifest=args.timing_manifest,
            install_config=args.install_config,
            check_only=args.check_only,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
