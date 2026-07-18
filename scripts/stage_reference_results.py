#!/usr/bin/env python3
"""Stage a validated AE result tree for deterministic archival release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "artifact/reference_results"
BASE_CATEGORIES = {
    "quick",
    "datasets",
    "rtl",
    "dram",
    "reports",
    "environments",
    "validation",
}
SOURCE_PROVENANCE_FIELDS = (
    "git_commit",
    "git_dirty",
    "source_identity",
    "source_tree_sha256",
    "submodules",
)
RAW_ASAP7_PPA_PATH = "physical/asap7/ppa.json"
DEEPSCALE_PPA_PATH = "physical/asap7/ppa_28nm_estimated.json"
PHYSICAL_RESULT_PATHS = {RAW_ASAP7_PPA_PATH, DEEPSCALE_PPA_PATH}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the stable form used by provenance records."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def release_evidence_binding(root: Path = ROOT) -> dict[str, Any]:
    """Return the source and mechanism identity required for staged evidence."""
    from scripts.mechanism_config import load_mechanism_config
    from scripts.result_record import source_identity

    root = Path(root).resolve()
    source = source_identity(root)
    _, mechanism = load_mechanism_config(root / "artifact/mechanism_config.json")
    calibration = {
        key: value
        for key, value in mechanism.items()
        if key != "mechanism_config_sha256"
    }
    return {
        "source": {
            "git_commit": source["git_commit"],
            "git_dirty": source["git_dirty"],
            "source_identity": source["source"],
            "source_tree_sha256": source["source_tree_sha256"],
            "submodules": source["submodules"],
        },
        "mechanism": {
            "mechanism_config_sha256": mechanism["mechanism_config_sha256"],
            "calibration_status": calibration["status"],
            "calibration_provenance_sha256": canonical_sha256(calibration),
            "calibration_provenance": calibration,
        },
    }


def _record_binding(
    provenance: Mapping[str, Any], *, calibration: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "source": {
            field: provenance.get(field) for field in SOURCE_PROVENANCE_FIELDS
        },
        "mechanism": {
            "mechanism_config_sha256": provenance.get("mechanism_config_sha256"),
            "calibration_status": calibration.get("status"),
            "calibration_provenance_sha256": canonical_sha256(calibration),
        },
    }


def is_generated_execution_record(relative: Path) -> bool:
    """Return whether a staged file carries direct execution provenance."""
    return relative.name == "results.json" or relative.as_posix() in PHYSICAL_RESULT_PATHS


def validate_deepscale_derivations(
    files: Mapping[Path, Path], parsed_records: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Bind a staged 28 nm estimate to its exact staged raw ASAP7 input."""
    by_relative = {
        relative.as_posix(): path for path, relative in files.items()
    }
    estimate_path = by_relative.get(DEEPSCALE_PPA_PATH)
    if estimate_path is None:
        return []
    label = str(Path("evidence") / DEEPSCALE_PPA_PATH)
    raw_path = by_relative.get(RAW_ASAP7_PPA_PATH)
    if raw_path is None:
        return [f"DeepScale estimate is missing staged raw ASAP7 PPA: {label}"]
    raw = parsed_records.get(RAW_ASAP7_PPA_PATH)
    estimate = parsed_records.get(DEEPSCALE_PPA_PATH)
    if raw is None or estimate is None:
        # The caller already reports malformed JSON for either record.
        return []
    try:
        from hardware.scaling.deepscale import validate_scaled_record

        validate_scaled_record(
            estimate,
            raw,
            raw_input_sha256=sha256_file(raw_path),
        )
    except (OSError, ValueError) as exc:
        return [f"DeepScale estimate input binding is invalid: {label} ({exc})"]
    return []


def validate_generated_records(
    files: Mapping[Path, Path], *, expected_binding: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Check every staged execution result against one source/config identity.

    Non-result artifacts such as images, traces, and reports are hash-bound by the
    enclosing manifest.  Every ``results.json`` is independently provenance-bound
    here so an old Functional tree cannot be restaged as current claim evidence.
    """
    expected_source = expected_binding.get("source")
    expected_mechanism = expected_binding.get("mechanism")
    if not isinstance(expected_source, Mapping) or not isinstance(
        expected_mechanism, Mapping
    ):
        raise ValueError("release evidence binding has an invalid schema")

    records: dict[str, dict[str, Any]] = {}
    parsed_records: dict[str, Mapping[str, Any]] = {}
    failures: list[str] = []
    for path, relative in sorted(files.items(), key=lambda item: str(item[1])):
        if not is_generated_execution_record(relative):
            continue
        label = str(Path("evidence") / relative)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"generated evidence is not valid JSON: {label} ({exc})")
            continue
        if isinstance(record, Mapping):
            parsed_records[relative.as_posix()] = record
        provenance = record.get("provenance") if isinstance(record, dict) else None
        if not isinstance(provenance, Mapping):
            failures.append(f"generated evidence provenance is missing: {label}")
            continue

        source_valid = True
        for field in SOURCE_PROVENANCE_FIELDS:
            if field not in provenance:
                failures.append(
                    f"generated evidence source provenance missing {field}: {label}"
                )
                source_valid = False
            elif provenance[field] != expected_source.get(field):
                failures.append(
                    f"generated evidence source provenance mismatch {field}: {label}"
                )
                source_valid = False

        mechanism_valid = True
        if "mechanism_config_sha256" not in provenance:
            failures.append(
                f"generated evidence mechanism provenance missing mechanism_config_sha256: {label}"
            )
            mechanism_valid = False
        elif provenance["mechanism_config_sha256"] != expected_mechanism.get(
            "mechanism_config_sha256"
        ):
            failures.append(
                f"generated evidence mechanism provenance mismatch mechanism_config_sha256: {label}"
            )
            mechanism_valid = False

        calibration = provenance.get("calibration_provenance")
        if not isinstance(calibration, Mapping):
            failures.append(
                f"generated evidence mechanism provenance missing calibration_provenance: {label}"
            )
            mechanism_valid = False
        else:
            if calibration.get("status") != expected_mechanism.get(
                "calibration_status"
            ):
                failures.append(
                    f"generated evidence calibration status mismatch: {label}"
                )
                mechanism_valid = False
            if canonical_sha256(calibration) != expected_mechanism.get(
                "calibration_provenance_sha256"
            ):
                failures.append(
                    f"generated evidence calibration provenance digest mismatch: {label}"
                )
                mechanism_valid = False

        if source_valid and mechanism_valid:
            records[label] = _record_binding(provenance, calibration=calibration)

    failures.extend(validate_deepscale_derivations(files, parsed_records))
    if not records and not failures:
        failures.append("validated output has no generated execution result records")
    return records, failures


def required_categories() -> set[str]:
    claims = json.loads((ROOT / "artifact/claim_status.json").read_text(encoding="utf-8"))
    categories = set(BASE_CATEGORIES)
    if any(state == "CLAIMED" for state in claims.get("software_pairs", {}).values()):
        categories.add("quality")
    if any(state == "CLAIMED" for state in claims.get("mechanism_pairs", {}).values()):
        categories.add("ablation")
    if claims.get("figure8") == "CLAIMED":
        categories.add("speedup")
    if claims.get("sensitivity") == "CLAIMED":
        categories.add("sensitivity")
    if claims.get("physical_asap7") == "CLAIMED" or claims.get("deepscale") == "CLAIMED":
        categories.add("physical")
    return categories


def required_environment_profiles() -> set[str]:
    claims = json.loads((ROOT / "artifact/claim_status.json").read_text(encoding="utf-8"))
    claimed_models = {
        pair.split("/", 1)[0]
        for pair, state in claims.get("software_pairs", {}).items()
        if state == "CLAIMED"
    }
    # The strict Functional quick run always exercises MVSplat in classic.
    profiles = {"classic"}
    if claimed_models & {"transplat", "mvsplat"}:
        profiles.add("classic")
    if "depthsplat" in claimed_models:
        profiles.add("depthsplat")
    return profiles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_execution_manifest(
    path: Path,
    *,
    repository_root: Path = ROOT,
    orchestrator_python: Path = Path(sys.executable),
) -> bytes:
    """Return a path-normalized archival copy of a workflow manifest."""
    record = json.loads(path.read_text(encoding="utf-8"))
    replacements = {
        str(repository_root.resolve()): "$SCARF_ROOT",
        str(orchestrator_python.resolve()): "$PYTHON",
        str(orchestrator_python): "$PYTHON",
    }
    declared_root = record.get("root")
    if isinstance(declared_root, str) and declared_root:
        source_root = Path(declared_root)
        if source_root.is_absolute() and source_root != Path(source_root.anchor):
            replacements[declared_root] = "$SCARF_ROOT"
            replacements[str(source_root.resolve())] = "$SCARF_ROOT"

    def command_lists(value):
        if not isinstance(value, list) or not value:
            return []
        if all(isinstance(item, str) for item in value):
            return [value]
        return [
            command
            for command in value
            if isinstance(command, list)
            and command
            and all(isinstance(item, str) for item in command)
        ]

    for field in ("commands", "dataset_commands"):
        for command in command_lists(record.get(field)):
            executable = Path(command[0])
            if executable.name in {"python", "python3"}:
                replacements[str(executable)] = "$PYTHON"
    profiles = set()
    for experiment in record.get("experiments", []):
        if not isinstance(experiment, dict):
            continue
        profile = experiment.get("environment_profile")
        command = experiment.get("command")
        if not isinstance(profile, str) or not isinstance(command, list) or not command:
            continue
        interpreter = Path(str(command[0]))
        variable = f"$SCARF_PYTHON_{profile.upper()}"
        profiles.add(profile)
        for name in ("python", "python3"):
            replacements[str(interpreter.parent / name)] = variable

    ordered = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            for source, target in ordered:
                value = value.replace(source, target)
        return value

    portable = normalize(record)
    portable["archival_path_normalization"] = {
        "applied": True,
        "repository_root": "$SCARF_ROOT",
        "orchestrator_python": "$PYTHON",
        "profile_interpreters": [
            f"$SCARF_PYTHON_{profile.upper()}" for profile in sorted(profiles)
        ],
    }
    return (json.dumps(portable, indent=2, sort_keys=True) + "\n").encode()


def selected_files(source: Path) -> dict[Path, Path]:
    patterns = (
        "manifest-*.json",
        "quick/*/results.json",
        "quick/*/pair-execution.json",
        "quick/*/progress.jsonl",
        "quick/*/samples/*/results.json",
        "quality/*/results.json",
        "quality/*/pair-execution.json",
        "quality/*/progress.jsonl",
        "quality/*/samples/*/results.json",
        "speedup/*/results.json",
        "speedup/*/samples/*/results.json",
        "speedup/*/samples/*/orin-evidence/measurement.json",
        "speedup/*/samples/*/orin-evidence/cuda-events.json",
        "speedup/*/samples/*/orin-evidence/tegrastats.log",
        "speedup/*/samples/*/orin-evidence/nsight*.nsys-rep",
        "speedup/*/orin-profile/*",
        "ablation/*/results.json",
        "ablation/*/pair-execution.json",
        "ablation/*/progress.jsonl",
        "ablation/*/samples/*/results.json",
        "sensitivity/results.json",
        "sensitivity/plan.json",
        "datasets/*.json",
        "rtl/results.json",
        "rtl/*.log",
        "rtl/rtl/*.sv",
        "rtl/rtl/*.f",
        "rtl/traces/*.vcd",
        "dram/*.json",
        "dram/*.csv",
        "dram/*.jsonl",
        "dram/*.yaml",
        "dram/*.log",
        "dram/*.trace",
        "physical/asap7/*.json",
        "physical/asap7/*.log",
        "physical/asap7/*.tcl",
        "physical/asap7/runtime/iflow/result/ScarfTop.droute.*/droute_drc.rpt",
        "physical/asap7/runtime/iflow/rtl/ScarfTop/sram-proxies.json",
        "reports/*",
        "environments/*.json",
        "validation.json",
    )
    selected = {}
    for pattern in patterns:
        for path in source.glob(pattern):
            if path.is_file():
                selected[path] = path.relative_to(source)
    for category in ("quick", "quality", "ablation"):
        category_root = source / category
        if not category_root.is_dir():
            continue
        for pair_root in sorted(path for path in category_root.iterdir() if path.is_dir()):
            sample_roots = sorted(
                path
                for path in (pair_root / "samples").glob("sample_*")
                if (path / "results.json").is_file()
            )
            if not sample_roots:
                continue
            for path in sample_roots[0].glob("*.png"):
                if path.is_file():
                    selected[path] = path.relative_to(source)
    return selected


def stage(source: Path, destination: Path = DESTINATION) -> dict:
    source = source.resolve()
    validation_path = source / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError("reference evidence requires validation status PASS")
    if validation.get("require_key_results") is not True:
        raise ValueError(
            "reference evidence requires validation with --require-key-results"
        )
    evidence = destination / "evidence"
    if evidence.exists():
        raise FileExistsError(f"reference evidence already exists: {evidence}")
    files = selected_files(source)
    categories = {
        relative.parts[0] if relative.name != "validation.json" else "validation"
        for relative in files.values()
    }
    missing = sorted(required_categories() - categories)
    if missing:
        raise ValueError("validated output is missing archive categories: " + ", ".join(missing))
    environment_profiles = {
        relative.stem
        for relative in files.values()
        if relative.parts[:1] == ("environments",) and relative.suffix == ".json"
    }
    missing_profiles = sorted(required_environment_profiles() - environment_profiles)
    if missing_profiles:
        raise ValueError(
            "validated output is missing environment profiles: "
            + ", ".join(missing_profiles)
        )
    provenance = release_evidence_binding(ROOT)
    generated_records, provenance_failures = validate_generated_records(
        files, expected_binding=provenance
    )
    if provenance_failures:
        raise ValueError(
            "generated evidence does not match the current release provenance: "
            + "; ".join(provenance_failures)
        )
    records = {}
    for path, relative in sorted(files.items(), key=lambda item: str(item[1])):
        target = evidence / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(relative.parts) == 1 and relative.name.startswith("manifest-"):
            target.write_bytes(portable_execution_manifest(path))
        else:
            shutil.copy2(path, target)
        records[str(Path("evidence") / relative)] = {
            "size": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    manifest = {
        "schema_version": "2.0",
        "status": "complete",
        "validation_status": "PASS",
        "validation_require_key_results": True,
        "categories": sorted(categories),
        "files": records,
        "provenance": {
            **provenance,
            "generated_records": generated_records,
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    try:
        manifest = stage(args.input, args.destination.resolve())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"staged {len(manifest['files'])} reference evidence files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
