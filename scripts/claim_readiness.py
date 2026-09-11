#!/usr/bin/env python3
"""Audit whether a SCARF output tree is ready for claim-mode evidence.

This command is intentionally an audit, not a result generator.  It reports
the exact release-state blockers so a real calibration run and real
source-bound timing export can be staged without guesswork.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS = ("transplat", "mvsplat", "depthsplat")
DATASETS = ("re10k", "acid", "dl3dv")
WORKFLOWS = ("quality", "mechanisms", "speedup")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _expected_counts(root: Path, profile: str) -> dict[str, int]:
    if profile == "full":
        protocol = _load_json(root / "artifact/evaluation_protocol.json", "evaluation protocol")
        pairs = protocol.get("pairs")
        if not isinstance(pairs, dict):
            raise ValueError("evaluation protocol has no pair records")
        counts = {}
        for dataset in DATASETS:
            counts[dataset] = int(pairs[f"transplat/{dataset}"]["sample_count"])
        return counts
    if profile == "reviewer":
        manifest = _load_json(
            root / "artifact/protocol/reviewer/manifest.json",
            "reviewer evidence profile",
        )
        datasets = manifest.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError("reviewer evidence profile has no datasets")
        return {dataset: int(datasets[dataset]["sample_count"]) for dataset in DATASETS}
    raise ValueError(f"unsupported evidence profile: {profile}")


def _audit_calibration(root: Path) -> tuple[dict[str, Any], list[str]]:
    path = root / "artifact/mechanism_config.json"
    blockers: list[str] = []
    try:
        record = _load_json(path, "mechanism configuration")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "missing", "path": str(path)}, [f"CALIBRATION_CONFIG_MISSING: {exc}"]
    status = record.get("status")
    calibration = record.get("calibration")
    selected = record.get("selected")
    ready = status == "calibrated" and isinstance(selected, dict)
    if status != "calibrated":
        blockers.append("CALIBRATION_NOT_FROZEN: mechanism_config.json must have status=calibrated")
    if not isinstance(selected, dict):
        blockers.append("CALIBRATION_SELECTION_MISSING: selected tuple is empty")
    if not isinstance(calibration, dict):
        blockers.append("CALIBRATION_PROVENANCE_MISSING: calibration object is absent")
    else:
        for field in ("manifest_sha256", "candidate_records_sha256"):
            value = calibration.get(field)
            if not isinstance(value, str) or len(value) != 64:
                blockers.append(f"CALIBRATION_{field.upper()}_MISSING: expected SHA256")
        if calibration.get("evaluation_disjoint") is not True:
            blockers.append("CALIBRATION_NOT_DISJOINT: evaluation_disjoint must be true")
        if calibration.get("protocol") not in {
            "dl3dv_train_holdout_v1",
            "acid_train_holdout_v1",
        }:
            blockers.append("CALIBRATION_PROTOCOL_MISSING: expected a supported train/holdout protocol")
    config_sha256 = None
    if not blockers:
        try:
            from scripts.mechanism_config import load_mechanism_config

            _config, provenance = load_mechanism_config(path)
            config_sha256 = provenance.get("mechanism_config_sha256")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"CALIBRATION_CONFIG_INVALID: {exc}")
    return {
        "path": str(path),
        "status": status,
        "selected": selected,
        "sha256": config_sha256,
        "ready": ready and not blockers,
    }, blockers


def _audit_timing(
    source_root: Path,
    output_root: Path,
    manifest_path: Path | None,
    counts: dict[str, int],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    path = (manifest_path or output_root / "timing-backend/manifest.json").resolve()
    if not path.is_file():
        return {"path": str(path), "status": "missing", "sample_counts": {}}, [
            "TIMING_MANIFEST_MISSING: provide a source-bound RTL timing manifest"
        ]
    try:
        from scripts.claim_timing_backend import load_claim_timing_manifest

        backend = load_claim_timing_manifest(path, root=source_root)
    except (OSError, ValueError) as exc:
        return {"path": str(path), "status": "invalid", "error": str(exc)}, [
            f"TIMING_MANIFEST_INVALID: {exc}"
        ]
    sample_counts = {
        f"{model}/{dataset}": sum(
            1 for key in backend.samples if key[0] == model and key[1] == dataset
        )
        for model in MODELS
        for dataset in DATASETS
    }
    for dataset, expected in counts.items():
        for model in MODELS:
            key = f"{model}/{dataset}"
            actual = sample_counts[key]
            if actual != expected:
                blockers.append(
                    f"TIMING_COVERAGE_INCOMPLETE: {key} has {actual}, expected {expected}"
                )
    return {
        "path": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "valid" if not blockers else "incomplete",
        "clock_mhz": backend.clock_mhz,
        "sample_counts": sample_counts,
        "source": backend.backend_provenance["source"],
    }, blockers


def _audit_workflows(
    root: Path,
    counts: dict[str, int],
    *,
    calibration: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    workflows: dict[str, Any] = {}
    for workflow in WORKFLOWS:
        pairs: dict[str, Any] = {}
        for model in MODELS:
            for dataset in DATASETS:
                pair = f"{model}/{dataset}"
                pair_directory = f"{model}_{dataset}"
                result = root / workflow / pair_directory / "results.json"
                # Performance uses the speedup directory; quality and mechanism
                # use their matching directory names in the generated tree.
                if not result.is_file():
                    pairs[pair] = {"status": "missing", "path": str(result)}
                    blockers.append(
                        f"RESULT_MISSING: {workflow}/{pair_directory}/results.json"
                    )
                    continue
                try:
                    record = _load_json(result, f"{workflow} result")
                    provenance = record.get("provenance")
                    evaluation = (
                        provenance.get("evaluation", {})
                        if isinstance(provenance, dict)
                        else {}
                    )
                    actual = evaluation.get("sample_count")
                    expected = counts[dataset]
                    if evaluation.get("kind") != "dataset_aggregate" or actual != expected:
                        blockers.append(
                            f"RESULT_INCOMPLETE: {workflow}/{pair} has sample_count={actual}, expected {expected}"
                        )
                        pairs[pair] = {"status": "incomplete", "sample_count": actual, "expected": expected}
                    else:
                        from scripts.execution_contract import execution_contract
                        from scripts.validate_result import validate

                        try:
                            if record.get("schema_version") != "2.1":
                                raise ValueError("claim aggregate must use schema 2.1")
                            validate(record)
                            contract = execution_contract(record)
                            if contract is None or contract.get("run_class") != "claim":
                                raise ValueError("aggregate was not executed in claim mode")
                            if record.get("provenance", {}).get("mechanism_config_sha256") != calibration.get("sha256"):
                                raise ValueError("mechanism config SHA256 does not match release config")
                            if workflow != "quality":
                                timing_binding = record.get("provenance", {}).get("timing_backend", {})
                                if timing_binding.get("manifest", {}).get("sha256") != timing.get("manifest_sha256"):
                                    raise ValueError("timing manifest SHA256 does not match release timing bundle")
                            from scripts.import_claim_evidence import _environment_digest, _validate_sample_lineage

                            environment = record["provenance"]["environment"]
                            profile = environment.get("profile")
                            environment_path = root / "environments" / f"{profile}.json"
                            if not environment_path.is_file():
                                raise ValueError(f"environment/{profile}.json is missing")
                            if _environment_digest(environment_path, profile) != environment.get("digest_sha256"):
                                raise ValueError("environment digest does not match record")
                            _validate_sample_lineage(result, record, workflow=workflow)
                            if workflow == "speedup":
                                if record.get("evidence_class") != "independent_measurement":
                                    raise ValueError("speedup aggregate is not independent measurement evidence")
                                from scripts.validate_ae import orin_evidence_reason

                                reason = orin_evidence_reason(
                                    record,
                                    evaluation.get("sample_selection_sha256"),
                                    result.parent,
                                )
                                if reason is not None:
                                    raise ValueError(f"Orin evidence incomplete: {reason}")
                        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                            blockers.append(f"RESULT_NOT_CLAIM_READY: {workflow}/{pair}: {exc}")
                            pairs[pair] = {"status": "invalid", "error": str(exc), "sample_count": actual}
                        else:
                            pairs[pair] = {"status": "complete", "sample_count": actual}
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    blockers.append(f"RESULT_INVALID: {workflow}/{pair}: {exc}")
                    pairs[pair] = {"status": "invalid", "error": str(exc)}
        workflows[workflow] = pairs
    validation = root / "validation.json"
    if not validation.is_file():
        blockers.append("VALIDATION_MISSING: run validate_ae.py --require-key-results")
    else:
        try:
            record = _load_json(validation, "AE validation report")
            if record.get("status") != "PASS" or record.get("require_key_results") is not True:
                blockers.append("VALIDATION_NOT_FINAL: validation.json must be PASS with require_key_results=true")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"VALIDATION_INVALID: {exc}")
    return {"validation_path": str(validation), "workflows": workflows}, blockers


def _audit_reference_manifest(root: Path) -> tuple[dict[str, Any], list[str]]:
    """Detect stale manifests that name evidence absent from the checkout."""
    path = root / "artifact/reference_results/manifest.json"
    if not path.is_file():
        return {"path": str(path), "status": "not_staged"}, []
    try:
        record = _load_json(path, "reference evidence manifest")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"path": str(path), "status": "invalid", "error": str(exc)}, [
            f"REFERENCE_MANIFEST_INVALID: {exc}"
        ]
    declared = record.get("files")
    missing = []
    if isinstance(declared, dict):
        for relative in declared:
            if not (root / "artifact/reference_results" / relative).is_file():
                missing.append(relative)
    else:
        missing.append("<files mapping>")
    blockers = []
    if missing:
        blockers.append(
            "REFERENCE_MANIFEST_STALE: missing "
            + str(len(missing))
            + " file(s) named by artifact/reference_results/manifest.json"
        )
    if record.get("status") != "complete" or record.get("validation_require_key_results") is not True:
        blockers.append("REFERENCE_MANIFEST_NOT_FINAL: stage only after validate_ae.py --require-key-results")
    return {
        "path": str(path),
        "status": "complete" if not blockers else "stale",
        "declared_files": len(declared) if isinstance(declared, dict) else 0,
        "missing_files": missing[:20],
    }, blockers


def audit(
    root: Path = ROOT,
    *,
    output_root: Path | None = None,
    profile: str = "full",
    timing_manifest: Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_root = Path(output_root or root).resolve()
    counts = _expected_counts(root, profile)
    calibration, calibration_blockers = _audit_calibration(root)
    timing, timing_blockers = _audit_timing(
        root, output_root, timing_manifest, counts
    )
    workflows, workflow_blockers = _audit_workflows(
        output_root,
        counts,
        calibration=calibration,
        timing=timing,
    )
    reference, reference_blockers = _audit_reference_manifest(root)
    blockers = (
        calibration_blockers
        + timing_blockers
        + workflow_blockers
        + reference_blockers
    )
    return {
        "schema_version": "scarf-claim-readiness-v1",
        "root": str(root),
        "output_root": str(output_root),
        "profile": profile,
        "expected_sample_counts": counts,
        "claim_ready": not blockers,
        "blockers": blockers,
        "calibration": calibration,
        "timing_backend": timing,
        "reference_evidence": reference,
        **workflows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="release checkout root")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="generated AE output root (defaults to --root)",
    )
    parser.add_argument("--profile", choices=("full", "reviewer"), default="full")
    parser.add_argument("--timing-manifest", type=Path)
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)
    try:
        report = audit(
            args.root,
            output_root=args.output_root,
            profile=args.profile,
            timing_manifest=args.timing_manifest,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(f"{'PASS' if report['claim_ready'] else 'BLOCKED'}: {len(report['blockers'])} blocker(s)")
    for blocker in report["blockers"]:
        print(f"- {blocker}")
    if args.output:
        print(args.output)
    return 0 if report["claim_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
