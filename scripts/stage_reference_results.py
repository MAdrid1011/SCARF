#!/usr/bin/env python3
"""Stage a validated AE result tree for deterministic archival release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Collection, Mapping


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
REFERENCE_ONLY_REPORT_FILENAMES = frozenset(
    {
        "paper_reference.md",
        "table1_reference.csv",
        "mechanisms_reference.csv",
        "summary_reference.csv",
    }
)
REFERENCE_ONLY_MARKER = b"PAPER_REFERENCE_ONLY"


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


def is_scarf_execution_result(relative: Path) -> bool:
    """Return whether a result must satisfy the traceable SCARF result schema."""
    return relative.name == "results.json" and bool(relative.parts) and relative.parts[0] in {
        "quick",
        "quality",
        "speedup",
        "ablation",
        "mechanisms",
        "utilization",
    }


def has_reference_only_marker(path: Path) -> bool:
    """Detect the explicit preview marker without loading large evidence files."""
    trailing = b""
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(64 * 1024), b""):
                if REFERENCE_ONLY_MARKER in trailing + block:
                    return True
                trailing = (trailing + block)[-(len(REFERENCE_ONLY_MARKER) - 1) :]
    except OSError:
        return False
    return False


def is_reference_only_report(relative: Path, source: Path | None = None) -> bool:
    """Keep reference-only previews out of reviewer-facing evidence."""
    return (
        relative.name in REFERENCE_ONLY_REPORT_FILENAMES
        or relative.name.endswith("_reference.csv")
        or (source is not None and has_reference_only_marker(source))
    )


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


def _aggregate_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude only the invocation text when comparing a rebuilt aggregate."""
    provenance = dict(record["provenance"])
    provenance.pop("command", None)
    return {
        "schema_version": record.get("schema_version"),
        "evidence_class": record.get("evidence_class"),
        "provenance": provenance,
        "quality": record.get("quality"),
        "performance": record.get("performance"),
        "events": record.get("events"),
        "energy": record.get("energy"),
        "ablation": record.get("ablation"),
        "fsdr_saes": record.get("fsdr_saes"),
        "hardware": record.get("hardware"),
        "validation": record.get("validation"),
    }


def _sample_result_path(
    aggregate_relative: Path, declared_path: Any
) -> Path | None:
    if not isinstance(declared_path, str) or not declared_path:
        return None
    pure = PurePosixPath(declared_path)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    candidate = aggregate_relative.parent / Path(*pure.parts)
    try:
        candidate.relative_to(aggregate_relative.parent)
    except ValueError:
        return None
    return candidate


def validate_aggregate_lineage(
    files: Mapping[Path, Path], parsed_records: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Rebuild every staged SCARF aggregate from its staged sample records."""
    from scripts.aggregate_results import aggregate

    paths_by_relative = {
        relative.as_posix(): path for path, relative in files.items()
    }
    failures: list[str] = []
    for aggregate_relative_text, aggregate_record in sorted(parsed_records.items()):
        provenance = aggregate_record.get("provenance")
        evaluation = provenance.get("evaluation") if isinstance(provenance, Mapping) else None
        if not isinstance(evaluation, Mapping) or evaluation.get("kind") != "dataset_aggregate":
            continue
        aggregate_relative = Path(aggregate_relative_text)
        label = str(Path("evidence") / aggregate_relative)
        sample_entries = evaluation.get("sample_results")
        sample_count = evaluation.get("sample_count")
        if (
            not isinstance(sample_entries, list)
            or not sample_entries
            or not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count != len(sample_entries)
        ):
            failures.append(f"aggregate sample lineage is incomplete: {label}")
            continue
        sample_paths: list[Path] = []
        lineage_valid = True
        for entry in sample_entries:
            if not isinstance(entry, Mapping):
                lineage_valid = False
                break
            sample_relative = _sample_result_path(
                aggregate_relative, entry.get("path")
            )
            if sample_relative is None:
                lineage_valid = False
                break
            sample_path = paths_by_relative.get(sample_relative.as_posix())
            sample_record = parsed_records.get(sample_relative.as_posix())
            if sample_path is None or not isinstance(sample_record, Mapping):
                lineage_valid = False
                break
            sample_evaluation = sample_record.get("provenance", {}).get("evaluation")
            if (
                not isinstance(sample_evaluation, Mapping)
                or sample_evaluation.get("kind") != "sample"
                or sample_evaluation.get("sample_index") != entry.get("sample_index")
                or sha256_file(sample_path) != entry.get("sha256")
            ):
                lineage_valid = False
                break
            sample_paths.append(sample_path)
        if not lineage_valid:
            failures.append(f"aggregate sample lineage does not match staged samples: {label}")
            continue
        try:
            aggregate_path = paths_by_relative.get(aggregate_relative.as_posix())
            if aggregate_path is None:
                raise ValueError("aggregate result is not staged")
            rebuilt = aggregate(
                sample_paths, sample_count, output=aggregate_path
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            failures.append(f"aggregate sample lineage cannot be rebuilt: {label} ({exc})")
            continue
        if _aggregate_projection(aggregate_record) != _aggregate_projection(rebuilt):
            failures.append(
                f"aggregate result does not match its staged sample results: {label}"
            )
    return failures


def validate_claim_timing_artifacts(
    files: Mapping[Path, Path], parsed_records: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Require every claim timing descriptor to resolve to a staged raw trace."""
    paths_by_relative = {
        relative.as_posix(): path for path, relative in files.items()
    }
    failures: list[str] = []
    for relative_text, record in sorted(parsed_records.items()):
        provenance = record.get("provenance")
        contract = (
            provenance.get("execution_contract")
            if isinstance(provenance, Mapping)
            else None
        )
        if not isinstance(contract, Mapping) or contract.get("run_class") != "claim":
            continue
        backend = provenance.get("timing_backend") if isinstance(provenance, Mapping) else None
        if not isinstance(backend, Mapping):
            failures.append(
                f"claim timing backend provenance is missing: evidence/{relative_text}"
            )
            continue
        source = backend.get("source")
        manifest = backend.get("manifest")
        if (
            not isinstance(source, Mapping)
            or not isinstance(manifest, Mapping)
            or not isinstance(source.get("path"), str)
            or not isinstance(manifest.get("path"), str)
            or not isinstance(source.get("sha256"), str)
            or not isinstance(manifest.get("sha256"), str)
        ):
            failures.append(
                f"claim timing backend provenance is malformed: evidence/{relative_text}"
            )
            continue
        result_parent = Path(relative_text).parent
        source_relative = PurePosixPath(source["path"])
        manifest_relative = PurePosixPath(manifest["path"])
        if source_relative.is_absolute() or ".." in source_relative.parts:
            failures.append(
                f"claim timing RTL source path is unsafe: evidence/{relative_text}"
            )
            continue
        if manifest_relative.is_absolute() or ".." in manifest_relative.parts:
            failures.append(
                f"claim timing manifest path is unsafe: evidence/{relative_text}"
            )
            continue
        source_key = (result_parent / Path(*source_relative.parts)).as_posix()
        source_path = paths_by_relative.get(source_key)
        if source_path is None or sha256_file(source_path) != source["sha256"]:
            failures.append(
                f"claim timing RTL source is not present or hash-bound: evidence/{relative_text}"
            )
            continue
        manifest_key = (result_parent / Path(*manifest_relative.parts)).as_posix()
        manifest_path = paths_by_relative.get(manifest_key)
        if manifest_path is None or sha256_file(manifest_path) != manifest["sha256"]:
            failures.append(
                f"claim timing manifest is not present or hash-bound: evidence/{relative_text}"
            )
            continue
        try:
            from scripts.claim_timing_backend import load_claim_timing_manifest

            load_claim_timing_manifest(
                manifest_path,
                root=manifest_path.parent,
            )
        except (OSError, ValueError) as exc:
            failures.append(
                f"claim timing backend bundle is invalid: evidence/{relative_text} ({exc})"
            )
            continue
        timing = record.get("performance", {}).get("claim_timing")
        trace = timing.get("trace") if isinstance(timing, Mapping) else None
        trace_path = trace.get("path") if isinstance(trace, Mapping) else None
        if not isinstance(trace_path, str):
            failures.append(
                f"claim timing raw trace is missing: evidence/{relative_text}"
            )
            continue
        trace_relative = Path(relative_text).parent / Path(*PurePosixPath(trace_path).parts)
        raw_trace = paths_by_relative.get(trace_relative.as_posix())
        if raw_trace is None or sha256_file(raw_trace) != trace.get("sha256"):
            failures.append(
                f"claim timing raw trace is not staged or hash-bound: evidence/{relative_text}"
            )
    return failures


def validate_report_catalog(
    output: Path,
    included_relatives: Collection[Path],
    *,
    project_root: Path = ROOT,
) -> list[str]:
    """Validate report sources before a catalog can enter staged evidence."""
    from scripts.generate_report import _source_records, validate_figure_catalog
    from scripts.validate_ae import figure8_catalog_evidence_check

    catalog_path = output / "reports" / "figure_catalog.json"
    if not catalog_path.is_file():
        return ["generated report catalog is missing"]
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        contract = json.loads(
            (project_root / "artifact" / "evaluation_catalog.json").read_text(
                encoding="utf-8"
            )
        )
        claims = json.loads(
            (project_root / "artifact" / "claim_status.json").read_text(
                encoding="utf-8"
            )
        )
        protocol = json.loads(
            (project_root / "artifact" / "evaluation_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        validate_figure_catalog(catalog, contract, require_key_results=False)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"generated report catalog is invalid: {exc}"]

    failures: list[str] = []
    included = {Path(relative).as_posix() for relative in included_relatives}
    for required_report in (
        "reports/figure_catalog.json",
        "reports/reproduction_report.md",
    ):
        if required_report not in included:
            failures.append(
                f"generated report artifact is not included in staged evidence: {required_report}"
            )
    requirements = {item["id"]: item for item in contract["results"]}
    for row in catalog["results"]:
        result_id = row["id"]
        if not isinstance(row.get("selected"), bool):
            failures.append(f"report catalog {result_id} has no boolean selected state")
            continue
        expected_sources = (
            _source_records(output, requirements[result_id]["raw_inputs"])
            if row["selected"]
            else []
        )
        if row["source_data"] != expected_sources:
            failures.append(
                f"report catalog sources do not match regenerated inputs: {result_id}"
            )
        for source in row["source_data"]:
            relative = source.get("path") if isinstance(source, Mapping) else None
            if not isinstance(relative, str) or relative not in included:
                failures.append(
                    f"report catalog source is not included in staged evidence: {relative}"
                )
        for export in row["exports"]:
            pure = PurePosixPath(export) if isinstance(export, str) else None
            if (
                pure is None
                or pure.is_absolute()
                or ".." in pure.parts
                or not pure.parts
                or pure.parts[0] == "reports"
            ):
                failures.append(f"report catalog has unsafe export path: {export}")
                continue
            relative = (Path("reports") / Path(*pure.parts)).as_posix()
            if relative not in included or not (output / relative).is_file():
                failures.append(
                    f"report catalog export is not included in staged evidence: {export}"
                )
                continue
            if pure.parts[:1] == ("status",) and pure.suffix == ".json":
                try:
                    status_record = json.loads((output / relative).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(
                        f"report catalog status export is invalid: {export} ({exc})"
                    )
                    continue
                if (
                    not isinstance(status_record, Mapping)
                    or status_record.get("id") != result_id
                    or status_record.get("status") != row["status"]
                    or status_record.get("required_evidence_class")
                    != row["evidence_class"]
                ):
                    failures.append(
                        f"report catalog status export does not match catalog row: {export}"
                    )

    check = figure8_catalog_evidence_check(
        output, catalog, protocol, claims.get("figure8")
    )
    if check["pass"] is not True:
        failures.append(
            "Figure 8 catalog does not satisfy independent Orin evidence: "
            f"{check['actual']}"
        )
    return failures


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

    from scripts.validate_result import reject_reference_only_record, validate

    records: dict[str, dict[str, Any]] = {}
    parsed_records: dict[str, Mapping[str, Any]] = {}
    lineage_records: dict[str, Mapping[str, Any]] = {}
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
        schema_valid = True
        if isinstance(record, Mapping):
            parsed_records[relative.as_posix()] = record
            try:
                reject_reference_only_record(record)
                if is_scarf_execution_result(relative):
                    validate(dict(record))
                    if relative.parts[0] != "quick" and record.get(
                        "schema_version"
                    ) != "2.1":
                        raise ValueError(
                            "claim-facing SCARF results require schema 2.1 "
                            "execution-trace bindings"
                        )
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"generated SCARF result is invalid: {label} ({exc})")
                schema_valid = False
        else:
            schema_valid = False
            failures.append(f"generated evidence root is not an object: {label}")
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

        if source_valid and mechanism_valid and schema_valid:
            records[label] = _record_binding(provenance, calibration=calibration)
            if is_scarf_execution_result(relative):
                lineage_records[relative.as_posix()] = record

    failures.extend(validate_deepscale_derivations(files, parsed_records))
    failures.extend(validate_aggregate_lineage(files, lineage_records))
    failures.extend(validate_claim_timing_artifacts(files, lineage_records))
    if not records and not failures:
        failures.append("validated output has no generated execution result records")
    return records, failures


def required_categories() -> set[str]:
    claims = json.loads((ROOT / "artifact/claim_status.json").read_text(encoding="utf-8"))
    categories = set(BASE_CATEGORIES)
    if any(state == "CLAIMED" for state in claims.get("software_pairs", {}).values()):
        categories.add("quality")
    if any(state == "CLAIMED" for state in claims.get("mechanism_pairs", {}).values()):
        categories.add("mechanisms")
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


def evidence_categories(relatives: Collection[Path]) -> set[str]:
    """Derive archive categories from the same relative paths used for staging."""
    categories: set[str] = set()
    for relative in relatives:
        path = Path(relative)
        if not path.parts:
            continue
        categories.add("validation" if path.name == "validation.json" else path.parts[0])
    return categories


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
        "quick/*/timing-trace/*",
        "quick/*/timing-trace/**/*",
        "quick/*/timing-backend/**/*",
        "quick/*/samples/*/timing-trace/*",
        "quick/*/samples/*/timing-trace/**/*",
        "quick/*/samples/*/timing-backend/**/*",
        "quality/*/results.json",
        "quality/*/pair-execution.json",
        "quality/*/progress.jsonl",
        "quality/*/samples/*/results.json",
        "quality/*/timing-trace/*",
        "quality/*/timing-trace/**/*",
        "quality/*/timing-backend/**/*",
        "quality/*/samples/*/timing-trace/*",
        "quality/*/samples/*/timing-trace/**/*",
        "quality/*/samples/*/timing-backend/**/*",
        "speedup/*/results.json",
        "speedup/*/samples/*/results.json",
        "speedup/*/timing-trace/*",
        "speedup/*/timing-trace/**/*",
        "speedup/*/timing-backend/**/*",
        "speedup/*/samples/*/timing-trace/*",
        "speedup/*/samples/*/timing-trace/**/*",
        "speedup/*/samples/*/timing-backend/**/*",
        "speedup/*/samples/*/orin-evidence/measurement.json",
        "speedup/*/samples/*/orin-evidence/cuda-events.json",
        "speedup/*/samples/*/orin-evidence/tegrastats.log",
        "speedup/*/samples/*/orin-evidence/nsight*.nsys-rep",
        "speedup/*/orin-profile/*",
        "ablation/*/results.json",
        "ablation/*/pair-execution.json",
        "ablation/*/progress.jsonl",
        "ablation/*/samples/*/results.json",
        "mechanisms/*/results.json",
        "mechanisms/*/pair-execution.json",
        "mechanisms/*/progress.jsonl",
        "mechanisms/*/samples/*/results.json",
        "ablation/*/timing-trace/*",
        "ablation/*/timing-trace/**/*",
        "ablation/*/timing-backend/**/*",
        "ablation/*/samples/*/timing-trace/*",
        "ablation/*/samples/*/timing-trace/**/*",
        "ablation/*/samples/*/timing-backend/**/*",
        "mechanisms/*/timing-trace/*",
        "mechanisms/*/timing-trace/**/*",
        "mechanisms/*/timing-backend/**/*",
        "mechanisms/*/samples/*/timing-trace/*",
        "mechanisms/*/samples/*/timing-trace/**/*",
        "mechanisms/*/samples/*/timing-backend/**/*",
        "utilization/*/timing-trace/*",
        "utilization/*/samples/*/timing-trace/*",
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
        "reports/**/*",
        "environments/*.json",
        "validation.json",
    )
    selected = {}
    for pattern in patterns:
        for path in source.glob(pattern):
            if path.is_file():
                relative = path.relative_to(source)
                if not is_reference_only_report(relative, path):
                    selected[path] = relative
    for category in ("quick", "quality", "ablation", "mechanisms"):
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
                    relative = path.relative_to(source)
                    if not is_reference_only_report(relative, path):
                        selected[path] = relative
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
    categories = evidence_categories(files.values())
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
    report_failures = validate_report_catalog(source, files.values())
    if report_failures:
        raise ValueError(
            "generated evidence report catalog is invalid: " + "; ".join(report_failures)
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
