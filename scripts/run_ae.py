#!/usr/bin/env python3
"""Run, inspect, and validate SCARF artifact-evaluation workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CALIBRATION_ROOT = ROOT / "downloads" / "calibration" / "prepared"
DEFAULT_DL3DV_CALIBRATION_MANIFEST = (
    ROOT / "outputs" / "calibration" / "dl3dv-protocol" / "manifest.json"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ae_config import (
    CLAIMED_MATRIX,
    ClaimSelection,
    resolve_claim_selection,
    resolve_experiment,
)
from scripts.compile_protocol import canonicalize_index
from scripts.evidence_profiles import resolve_evidence_selection


PAPER_SOFTWARE_MODES = {"quality", "performance", "mechanisms", "utilization"}
CORE_RESULT_SELECTION = "figure8,table1,figure11,table2,table3"
ORIN_WORKFLOWS = {"orin", "speedup", "performance"}
PENDING_EVALUATOR = "PENDING_EVALUATOR"
ORIN_PENDING_REASON = "requires a Jetson Orin NX host"
SAES_QUALITY_MODE = "saes-quality"
DL3DV_GATE_PROFILE = "dl3dv-gate"
SAES_QUALITY_GATE_STATUS_BLOCKED = "BLOCKED_CALIBRATION_NOT_FROZEN"
SAES_QUALITY_GATE_STATUS_INPUTS_UNAVAILABLE = (
    "BLOCKED_CALIBRATION_INPUTS_UNAVAILABLE"
)
SAES_QUALITY_GATE_STATUS_NO_COMPATIBLE_EXECUTION = (
    "BLOCKED_NO_COMPATIBLE_QUALITY_EXECUTION"
)
SOFTWARE_MODES = {
    "quick",
    "quality",
    "performance",
    "mechanisms",
    "utilization",
    "speedup",
    "ablation",
    "orin",
    "fsdr",
}
CLAIM_ONLY_SOFTWARE_MODES = PAPER_SOFTWARE_MODES | {
    "speedup",
    "ablation",
    "orin",
    "fsdr",
}
FSDR_MATRIX = tuple(
    (model, dataset)
    for model in ("transplat", "mvsplat", "depthsplat")
    for dataset in ("re10k", "acid")
)
MODES = tuple(
    sorted(
        SOFTWARE_MODES
        | {
            "all",
            "all-eval",
            "dram",
            "figures",
            "worstcase",
            "sensitivity",
            "rtl",
            "physical",
            "scale",
            "report",
            "validate",
            "calibrate",
            "pilot",
            "proxy",
            SAES_QUALITY_MODE,
        }
    )
)


def load_claim_status() -> dict[str, Any]:
    path = ROOT / "artifact/claim_status.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "1.0":
        raise ValueError("artifact claim status has an invalid schema")
    return record


def _claimed_pairs() -> tuple[tuple[str, str], ...]:
    status = load_claim_status().get("software_pairs", {})
    return tuple(
        (model, dataset)
        for model, dataset in CLAIMED_MATRIX
        if status.get(f"{model}/{dataset}") == "CLAIMED"
    )


def _python_for(profile: str, override: str | None) -> str:
    if override:
        return override
    variable = f"SCARF_PYTHON_{profile.upper()}"
    configured = os.environ.get(variable)
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file() or not os.access(configured_path, os.X_OK):
            raise ValueError(
                f"{variable} must name an executable Python interpreter: {configured}"
            )
        return str(configured_path)
    local_profile = ROOT / ".venv" / profile / "bin" / "python"
    if local_profile.is_file() and os.access(local_profile, os.X_OK):
        return str(local_profile)
    raise ValueError(
        f"no Python interpreter configured for {profile}; pass --python, set "
        f"{variable}, or create {local_profile}"
    )


def _is_orin_host() -> bool:
    try:
        model = Path("/proc/device-tree/model").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False
    return "Jetson Orin NX" in model.rstrip("\x00")


def _pairs_for_mode(mode: str) -> tuple[tuple[str, str], ...]:
    if mode == "quick":
        return (("mvsplat", "re10k"),)
    if mode == "fsdr":
        return FSDR_MATRIX
    if mode in PAPER_SOFTWARE_MODES:
        return CLAIMED_MATRIX
    if mode in {"quality", "ablation"}:
        return _claimed_pairs()
    if mode in {"speedup", "orin"}:
        return (
            _claimed_pairs()
            if load_claim_status().get("figure8") == "CLAIMED"
            else ()
        )
    return ()


def parse_pair_filter(value: str | None) -> tuple[tuple[str, str], ...] | None:
    """Parse an explicit, ordered subset of canonical model/dataset pairs."""
    if value is None:
        return None
    if value.strip().lower() == "all":
        return CLAIMED_MATRIX
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--pairs must be a comma-separated model/dataset list")
    pairs: list[tuple[str, str]] = []
    for token in tokens:
        parts = token.split("/")
        if len(parts) != 2 or tuple(parts) not in CLAIMED_MATRIX:
            raise ValueError(f"unknown protocol pair in --pairs: {token}")
        pair = (parts[0], parts[1])
        if pair in pairs:
            raise ValueError(f"duplicate protocol pair in --pairs: {token}")
        pairs.append(pair)
    return tuple(pairs)


def _protocol_sample_count(model: str, dataset: str) -> int:
    path = ROOT / "artifact/evaluation_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "finalized":
        raise ValueError(
            "full claim runs require --num-samples until artifact/evaluation_protocol.json is finalized"
        )
    record = protocol.get("pairs", {}).get(f"{model}/{dataset}", {})
    count = record.get("sample_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f"evaluation protocol has no positive sample count for {model}/{dataset}")
    return count


def _quick_selection() -> ClaimSelection:
    path = ROOT / "artifact/quick/evaluation_index.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _, summary = canonicalize_index(path, digest)
    return ClaimSelection(
        model="mvsplat",
        dataset="re10k",
        index_path=path,
        source_index_sha256=digest,
        sample_count=summary["sample_count"],
        sample_selection_sha256=summary["sample_selection_sha256"],
    )


def saes_quality_gate_preflight() -> dict[str, Any]:
    """Check whether the current candidate has a safe sample-0 quality path."""
    from scripts.mechanism_config import DEFAULT_CONFIG, require_calibrated_mechanism

    preflight: dict[str, Any] = {
        "model": "transplat",
        "dataset": "dl3dv",
        "sample_index": 0,
        "candidate": {
            "materialization": "representative",
            "l1_anchor_count": 12,
            "context_safety_guard": True,
        },
        "target_rgb_quality_execution_scheduled": False,
        "paper_result_eligible": False,
        "requires_calibrated_evaluation_disjoint_configuration": True,
        "quality_execution_status": "UNAVAILABLE",
        "quality_execution_reason": (
            "the fixed representative quality wrapper is unavailable until the "
            "global evaluation-disjoint mechanism configuration is calibrated"
        ),
    }
    try:
        # This diagnostic gate reports the preregistered root baseline and is
        # intentionally separate from reviewer claim execution, which resolves
        # the verified frozen bundle through the default claim path.
            _config, provenance = require_calibrated_mechanism()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        if not DEFAULT_DL3DV_CALIBRATION_MANIFEST.is_file():
            return {
                **preflight,
                "status": SAES_QUALITY_GATE_STATUS_INPUTS_UNAVAILABLE,
                "ready": False,
                "reason": (
                    "missing official DL3DV calibration manifest "
                    "outputs/calibration/dl3dv-protocol/manifest.json; prepare "
                    "the evaluation-disjoint inputs described in artifact/CALIBRATION.md "
                    "before running calibrate"
                ),
                "mechanism_config_reason": str(exc),
            }
        return {
            **preflight,
            "status": SAES_QUALITY_GATE_STATUS_BLOCKED,
            "ready": False,
            "reason": str(exc),
        }
    wrapper = SCRIPT_DIR / "saes_representative_quality_gate.py"
    if not wrapper.is_file():
        return {
            **preflight,
            "status": SAES_QUALITY_GATE_STATUS_NO_COMPATIBLE_EXECUTION,
            "ready": False,
            "reason": "fixed representative quality wrapper is missing",
            "mechanism_config_sha256": provenance["mechanism_config_sha256"],
            "calibration_status": provenance["status"],
            "evaluation_disjoint": provenance["evaluation_disjoint"],
            "saes_execution_route_sha256": provenance[
                "saes_execution_route_sha256"
            ],
        }
    return {
        **preflight,
        "status": "READY_FIXED_NONCLAIM_DIAGNOSTIC",
        "ready": True,
        "target_rgb_quality_execution_scheduled": True,
        "quality_execution_status": "FIXED_NONCLAIM_DIAGNOSTIC",
        "quality_execution_reason": (
            "fixed TranSplat/DL3DV sample-0 representative diagnostic; "
            "its result remains non-claim evidence"
        ),
        "reason": "evaluation-disjoint mechanism configuration is calibrated",
        "mechanism_config_sha256": provenance["mechanism_config_sha256"],
        "calibration_status": provenance["status"],
        "evaluation_disjoint": provenance["evaluation_disjoint"],
        "saes_execution_route_sha256": provenance[
            "saes_execution_route_sha256"
        ],
    }


def build_saes_quality_plan(
    output_root: Path, python_override: str | None = None
) -> dict[str, Any]:
    """Build the one fixed non-claim sample-0 quality diagnostic."""

    preflight = saes_quality_gate_preflight()
    if not preflight["ready"]:
        return {"preflight": preflight, "commands": []}
    try:
        classic_python = _python_for("classic", python_override)
    except ValueError as exc:
        return {
            "preflight": {
                **preflight,
                "status": SAES_QUALITY_GATE_STATUS_NO_COMPATIBLE_EXECUTION,
                "ready": False,
                "target_rgb_quality_execution_scheduled": False,
                "quality_execution_status": "UNAVAILABLE",
                "quality_execution_reason": str(exc),
                "reason": str(exc),
            },
            "commands": [],
        }
    return {
        "preflight": preflight,
        "commands": [
            [
                classic_python,
                str(SCRIPT_DIR / "saes_representative_quality_gate.py"),
                "--output-dir",
                str(output_root / SAES_QUALITY_MODE / "transplat_dl3dv_sample0"),
                "--device",
                "cuda",
            ]
        ],
    }


def build_software_plan(
    mode: str,
    output_root: Path,
    python_override: str | None = None,
    num_samples: int | None = None,
    pair_filter: tuple[tuple[str, str], ...] | None = None,
    allow_partial_filter: bool = False,
    evidence_profile: str = "full",
    claim_execution: bool = True,
    orin_available: bool | None = None,
    claim_timing_manifest: Path | None = None,
    selection_file: Path | None = None,
    run_selection_sha256: str | None = None,
    export_rtl_payload: bool = False,
) -> list[dict[str, Any]]:
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive")
    mode_pairs = _pairs_for_mode(mode)
    if pair_filter is not None:
        unsupported = [pair for pair in pair_filter if pair not in mode_pairs]
        if unsupported and not allow_partial_filter:
            names = ", ".join(f"{model}/{dataset}" for model, dataset in unsupported)
            raise ValueError(f"--pairs not supported by {mode}: {names}")
        mode_pairs = tuple(pair for pair in pair_filter if pair in mode_pairs)
    experiments = []
    for model, dataset in mode_pairs:
        # ``performance`` is the public Figure 8 command name.  Its evidence
        # has always been named ``speedup`` by the validator, report generator,
        # and release stager; retain that one canonical on-disk workflow name.
        result_workflow = "speedup" if mode == "performance" else mode
        config = resolve_experiment(model, dataset, ROOT)
        selection = (
            _quick_selection()
            if mode == "quick"
            else resolve_evidence_selection(model, dataset, ROOT, evidence_profile)
        )
        sample_count = (
            num_samples
            if num_samples is not None
            else (1 if mode == "quick" else selection.sample_count)
        )
        profile_python = _python_for(config.environment_profile, python_override)
        output_dir = output_root / result_workflow / f"{model}_{dataset}"
        dataset_root = (
            ROOT / "datasets/quick-re10k"
            if mode == "quick"
            else config.dataset_root
        )
        dataset_representation = (
            "re10k-synthetic-functional-v1"
            if mode == "quick"
            else config.dataset_representation
        )
        demo_command = [
            profile_python,
            str(SCRIPT_DIR / "demo.py"),
            "--model",
            model,
            "--dataset",
            dataset,
            "--checkpoint",
            str(config.checkpoint),
            "--dataset-root",
            str(dataset_root),
            "--evaluation-index",
            str(selection.index_path),
            (
                "--functional-run"
                if mode == "quick"
                else "--claim-run" if claim_execution else "--diagnostic-run"
            ),
            "--device",
            "auto",
            "--seed",
            "0",
        ]
        if claim_execution:
            claim_surface = (
                "quality"
                if mode == "quality"
                else "performance"
                if mode in {"performance", "speedup", "orin", "utilization"}
                else "mechanisms"
            )
            demo_command.extend(
                ("--claim-workflow", claim_surface)
            )
            if claim_timing_manifest is not None:
                demo_command.extend(
                    ("--claim-timing-manifest", str(claim_timing_manifest))
                )
        if mode in {"ablation", "mechanisms", "all", "all-eval"}:
            demo_command.append("--ablation")
        if mode == "quality":
            demo_command.extend(("--image-output-policy", "all"))
        if mode == "fsdr":
            demo_command.extend(("--fsdr-only", "--image-output-policy", "none"))
        command = [
            profile_python,
            str(SCRIPT_DIR / "run_pair.py"),
            "--evaluation-index",
            str(selection.index_path),
            "--source-index-sha256",
            selection.source_index_sha256,
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--num-samples",
            str(sample_count),
            "--resume",
        ]
        if selection_file is not None:
            command.extend(
                [
                    "--selection-file",
                    str(selection_file),
                    "--pair",
                    f"{model}/{dataset}",
                ]
            )
        if export_rtl_payload:
            command.append("--export-rtl-payload")
        command.extend(["--", *demo_command])
        commands = [command]
        if mode == "fsdr":
            aggregate_command = [
                sys.executable,
                str(SCRIPT_DIR / "aggregate_fsdr.py"),
                "--input-dir",
                str(output_dir / "samples"),
                "--expected-count",
                str(sample_count),
                "--output",
                str(output_dir / "results.json"),
            ]
        else:
            aggregate_command = [
                sys.executable,
                str(SCRIPT_DIR / "aggregate_results.py"),
                "--input-dir",
                str(output_dir / "samples"),
                "--expected-count",
                str(sample_count),
                "--output",
                str(output_dir / "results.json"),
            ]
        experiments.append(
            {
                "workflow": mode,
                "result_workflow": result_workflow,
                "model": model,
                "dataset": dataset,
                "experiment": config.experiment,
                "hydra_overrides": list(config.hydra_overrides),
                "dataset_root": str(dataset_root),
                "dataset_representation": dataset_representation,
                "evaluation_index": str(selection.index_path),
                "source_index_sha256": selection.source_index_sha256,
                "sample_selection_sha256": selection.sample_selection_sha256,
                "environment_profile": config.environment_profile,
                "command": commands[0],
                "commands": commands,
                "aggregate_command": aggregate_command,
                "sample_count": sample_count,
                "evidence_profile": evidence_profile,
                "claim_timing_manifest": (
                    str(claim_timing_manifest)
                    if claim_execution and claim_timing_manifest
                    else None
                ),
                "run_selection_sha256": run_selection_sha256,
                "result": str(output_dir / "results.json"),
            }
        )
    if mode in ORIN_WORKFLOWS:
        if orin_available is None:
            orin_available = _is_orin_host()
        if not orin_available:
            for item in experiments:
                item["execution_status"] = PENDING_EVALUATOR
                item["skip_reason"] = ORIN_PENDING_REASON
        else:
            for item in experiments:
                original = item["commands"][0]
                pair_dir = Path(original[original.index("--output-dir") + 1])
                wrapped = [[
                    original[0],
                    str(ROOT / "hardware/orin/run.py"),
                    "--pair-dir",
                    str(pair_dir),
                    "--evidence-dir",
                    str(pair_dir / "orin-profile"),
                    "--sample-selection-sha256",
                    item["sample_selection_sha256"],
                    "--expected-count",
                    str(item["sample_count"]),
                    "--",
                    *original,
                ]]
                item["commands"] = wrapped
                item["command"] = wrapped[0]
    return experiments


def build_dataset_validation_commands(
    experiments: list[dict[str, Any]], output_root: Path
) -> list[list[str]]:
    commands = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in experiments:
        if item.get("execution_status") == PENDING_EVALUATOR:
            continue
        key = (
            item["dataset_root"],
            item["dataset_representation"],
            item["evaluation_index"],
            item["source_index_sha256"],
        )
        if key in seen:
            continue
        seen.add(key)
        representation = item["dataset_representation"]
        safe_name = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in representation
        )
        commands.append(
            [
                item["command"][0],
                str(ROOT / "data/verify_prepared_dataset.py"),
                "--root",
                item["dataset_root"],
                "--evaluation-index",
                item["evaluation_index"],
                "--source-index-sha256",
                item["source_index_sha256"],
                "--representation",
                representation,
                "--output",
                str(output_root / "datasets" / f"{safe_name}-validation.json"),
            ]
        )
    return commands


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    num_samples = getattr(args, "num_samples", None)
    evidence_profile = getattr(args, "profile", "full")
    claim_timing_manifest = getattr(args, "claim_timing_manifest", None)
    pair_filter = parse_pair_filter(getattr(args, "pairs", None))
    if pair_filter is not None and args.mode not in SOFTWARE_MODES | {"all", "all-eval", "pilot"}:
        raise ValueError(f"--pairs is not supported by {args.mode}")
    if args.mode in {SAES_QUALITY_MODE, "proxy"}:
        experiments = []
    elif args.mode == "calibrate":
        experiments = []
    elif args.mode == "pilot":
        experiments = build_software_plan(
            "mechanisms",
            args.output_root,
            args.python,
            1,
            pair_filter,
            False,
            "full",
            False,
        )
    elif args.mode == "all":
        experiments = []
        claimed_pair_filter = _claimed_pairs()
        if pair_filter is not None:
            claimed_pair_filter = tuple(
                pair for pair in pair_filter if pair in claimed_pair_filter
            )
        for workflow in ("quality", "speedup", "mechanisms"):
            experiments.extend(
                build_software_plan(
                    workflow,
                    args.output_root,
                    args.python,
                    num_samples,
                    claimed_pair_filter,
                    True,
                    evidence_profile,
                    claim_timing_manifest=claim_timing_manifest,
                )
            )
    elif args.mode == "all-eval":
        experiments = []
        for workflow in ("quality", "performance", "mechanisms"):
            experiments.extend(
                build_software_plan(
                    workflow,
                    args.output_root,
                    args.python,
                    num_samples,
                    pair_filter,
                    True,
                    evidence_profile,
                    claim_timing_manifest=claim_timing_manifest,
                )
            )
    else:
        experiments = build_software_plan(
            args.mode,
            args.output_root,
            args.python,
            num_samples,
            pair_filter,
            False,
            evidence_profile,
            claim_timing_manifest=claim_timing_manifest,
        )
    effective_profile = (
        "calibration"
        if args.mode == "calibrate"
        else "pilot"
        if args.mode == "pilot"
        else "not-applicable"
        if args.mode in {SAES_QUALITY_MODE, "proxy"}
        else evidence_profile
    )
    allow_low_memory_attempt = bool(
        getattr(args, "allow_low_memory_attempt", False)
    )
    calibration_root = getattr(args, "calibration_root", DEFAULT_CALIBRATION_ROOT)
    calibration_manifest = getattr(
        args, "calibration_manifest", DEFAULT_DL3DV_CALIBRATION_MANIFEST
    )
    if args.mode == "calibrate" and calibration_root != DEFAULT_CALIBRATION_ROOT:
        raise ValueError(
            "--calibration-root is legacy Functional-only input and cannot freeze "
            "a DL3DV Results Reproduced configuration; use --calibration-manifest"
        )
    plan: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": args.mode,
        "root": str(ROOT),
        "output_root": str(args.output_root),
        "experiments": experiments,
        "evidence_profile": effective_profile,
        "calibration_contract": "artifact/CALIBRATION.md",
        "claim_status": load_claim_status(),
        "requested_device": getattr(args, "device", "auto"),
        "requested_pairs": (
            [f"{model}/{dataset}" for model, dataset in pair_filter]
            if pair_filter is not None
            else None
        ),
        "requested_figures": getattr(args, "figures", "all"),
        "require_key_results": bool(getattr(args, "require_key_results", False)),
        "allow_low_memory_attempt": allow_low_memory_attempt,
        "calibration_root": str(calibration_root),
        "calibration_manifest": str(calibration_manifest),
        "claim_timing_manifest": (
            str(claim_timing_manifest) if claim_timing_manifest is not None else None
        ),
    }
    plan["software_claim_scope"] = {
        "status": (
            "DIAGNOSTIC_PREFLIGHT"
            if args.mode == SAES_QUALITY_MODE
            else "PROXY_ONLY"
            if args.mode == "proxy"
            else
            "NO_CLAIMED_PAIRS"
            if args.mode in CLAIM_ONLY_SOFTWARE_MODES | {"all", "all-eval"}
            and not experiments
            else "ACTIVE"
        ),
        "pair_count": len(experiments),
        "diagnostic_results_are_claim_evidence": False,
    }
    pending_evaluators: list[dict[str, str]] = []
    for item in experiments:
        if item.get("execution_status") != PENDING_EVALUATOR:
            continue
        pending = {
            "workflow": item["workflow"],
            "status": PENDING_EVALUATOR,
            "reason": item["skip_reason"],
        }
        if pending not in pending_evaluators:
            pending_evaluators.append(pending)
    plan["pending_evaluators"] = pending_evaluators
    plan["dataset_commands"] = build_dataset_validation_commands(
        experiments, args.output_root
    )
    sensitivity_command = [
        sys.executable,
        str(SCRIPT_DIR / "sensitivity_sweep.py"),
        "--output-dir",
        str(args.output_root / "sensitivity"),
        "--profile",
        evidence_profile,
    ]
    if num_samples is not None:
        sensitivity_command.extend(("--num-samples", str(num_samples)))
    if args.mode == "proxy":
        proxy_input = getattr(args, "proxy_input", None)
        if proxy_input is None:
            raise ValueError("proxy workflow requires --proxy-input")
        plan["proxy_scope"] = {
            "status": "PROXY_ONLY",
            "claim_eligible": False,
            "evidence_class": "hardware_proxy",
        }
        plan["commands"] = [
            [
                sys.executable,
                str(SCRIPT_DIR / "normalize_orin_proxy.py"),
                "--input",
                str(Path(proxy_input).resolve()),
                "--output",
                str(args.output_root / "reports" / "orin-nx-proxy.json"),
            ]
        ]
    elif args.mode == "calibrate":
        sweep_dir = args.output_root / "sweep"
        calibration_python = _python_for("classic", args.python)
        plan["calibration_python"] = calibration_python
        plan["commands"] = [
            [
                calibration_python,
                str(SCRIPT_DIR / "calibration_sweep.py"),
                "--manifest",
                str(calibration_manifest),
                "--output-dir",
                str(sweep_dir),
            ],
            [
                calibration_python,
                str(SCRIPT_DIR / "calibrate_mechanisms.py"),
                "--candidate-records",
                str(sweep_dir / "candidates.json"),
                "--output-dir",
                str(args.output_root),
            ],
        ]
    elif args.mode == SAES_QUALITY_MODE:
        gate = build_saes_quality_plan(args.output_root, args.python)
        plan["profile_selector"] = DL3DV_GATE_PROFILE
        plan["saes_quality_gate"] = gate["preflight"]
        plan["commands"] = gate["commands"]
    elif args.mode == "sensitivity":
        plan["commands"] = [sensitivity_command]
    elif args.mode == "rtl":
        plan["commands"] = [["bash", str(SCRIPT_DIR / "run_rtl.sh"), "--output-dir", str(args.output_root / "rtl")]]
    elif args.mode == "dram":
        plan["commands"] = [[
            "bash",
            str(ROOT / "hardware/dram/run.sh"),
            "--events",
            str(ROOT / "hardware/dram/test_vectors/scarf_smoke_events.jsonl"),
            "--output-dir",
            str(args.output_root / "dram"),
        ]]
    elif args.mode == "physical":
        command = [
            "bash",
            str(ROOT / "hardware" / "iflow" / "run.sh"),
            "--platform",
            "asap7",
            "--stage",
            "all",
            "--output-dir",
            str(args.output_root / "physical" / "asap7"),
        ]
        if allow_low_memory_attempt:
            command.append("--allow-low-memory-attempt")
        plan["commands"] = [command]
    elif args.mode == "scale":
        plan["commands"] = [[sys.executable, str(ROOT / "hardware/scaling/deepscale.py"), "--source-node", "7", "--target-node", "28", "--input", str(args.output_root / "physical/asap7/ppa.json"), "--output", str(args.output_root / "physical/asap7/ppa_28nm_estimated.json")]]
    elif args.mode in {"report", "figures"}:
        command = [
            sys.executable,
            str(SCRIPT_DIR / "generate_report.py"),
            "--input",
            str(args.output_root),
            "--output-dir",
            str(args.output_root / "reports"),
        ]
        if args.mode == "figures":
            command.extend(("--figures", getattr(args, "figures", "all")))
        plan["commands"] = [command]
    elif args.mode == "worstcase":
        plan["commands"] = [[
            sys.executable,
            str(SCRIPT_DIR / "generate_report.py"),
            "--input",
            str(args.output_root),
            "--output-dir",
            str(args.output_root / "reports"),
            "--figures",
            "figure10",
        ]]
    elif args.mode == "validate":
        command = [
            sys.executable,
            str(SCRIPT_DIR / "validate_ae.py"),
            "--input",
            str(args.output_root),
        ]
        if getattr(args, "require_key_results", False):
            command.append("--require-key-results")
        plan["commands"] = [command]
    elif args.mode == "all":
        plan["commands"] = [
            ["bash", str(SCRIPT_DIR / "run_rtl.sh"), "--output-dir", str(args.output_root / "rtl")],
            [
                sys.executable,
                str(SCRIPT_DIR / "generate_report.py"),
                "--input",
                str(args.output_root),
                "--output-dir",
                str(args.output_root / "reports"),
                "--figures",
                CORE_RESULT_SELECTION,
            ],
            [sys.executable, str(SCRIPT_DIR / "validate_ae.py"), "--input", str(args.output_root)],
        ]
    elif args.mode == "all-eval":
        plan["commands"] = [
            ["bash", str(SCRIPT_DIR / "run_rtl.sh"), "--output-dir", str(args.output_root / "rtl")],
            [
                sys.executable,
                str(SCRIPT_DIR / "generate_report.py"),
                "--input",
                str(args.output_root),
                "--output-dir",
                str(args.output_root / "reports"),
                "--figures",
                CORE_RESULT_SELECTION,
            ],
            [
                sys.executable,
                str(SCRIPT_DIR / "validate_ae.py"),
                "--input",
                str(args.output_root),
                "--profile",
                "evaluator-final",
                "--require-key-results",
                "--allow-missing-quick",
            ],
        ]
    return plan


def execute_plan(plan: dict[str, Any], output_root: Path) -> int:
    if plan.get("mode") == SAES_QUALITY_MODE:
        preflight = saes_quality_gate_preflight()
        if not preflight["ready"]:
            print(
                "error: saes-quality is blocked before target-RGB quality execution: "
                f"{preflight['reason']}",
                file=sys.stderr,
            )
            return 2
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(plan)
    manifest["started_at"] = datetime.now(timezone.utc).isoformat()
    (output_root / f"manifest-{plan['mode']}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if plan.get("software_claim_scope", {}).get("status") == "NO_CLAIMED_PAIRS":
        print(
            "notice: no software pairs are currently claimed; "
            "skipping claim-only software execution",
            flush=True,
        )
    profile_pythons: dict[str, str] = {}
    for item in plan.get("environment_checks", []):
        profile = item["profile"]
        profile_python = item["python"]
        previous = profile_pythons.setdefault(profile, profile_python)
        if previous != profile_python:
            print(
                f"error: {profile} experiments use multiple Python interpreters",
                file=sys.stderr,
            )
            return 2
    for item in plan["experiments"]:
        if item.get("execution_status") == PENDING_EVALUATOR:
            continue
        profile = item["environment_profile"]
        profile_python = item["command"][0]
        previous = profile_pythons.setdefault(profile, profile_python)
        if previous != profile_python:
            print(
                f"error: {profile} experiments use multiple Python interpreters",
                file=sys.stderr,
            )
            return 2
    environment_dir = output_root / "environments"
    for profile, profile_python in sorted(profile_pythons.items()):
        environment_dir.mkdir(parents=True, exist_ok=True)
        command = [
            profile_python,
            str(SCRIPT_DIR / "check_environment.py"),
            "--profile",
            profile,
            "--output",
            str(environment_dir / f"{profile}.json"),
        ]
        print("+", " ".join(command), flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT)
        except FileNotFoundError:
            print(
                f"error: Python interpreter for {profile} does not exist: {profile_python}",
                file=sys.stderr,
            )
            return 2
        if result.returncode != 0:
            return result.returncode
    commands = list(plan.get("dataset_commands", []))
    for item in plan["experiments"]:
        if item.get("execution_status") == PENDING_EVALUATOR:
            print(
                "notice: skipping "
                f"{item['workflow']} {item['model']}/{item['dataset']}: "
                f"{item['skip_reason']}",
                flush=True,
            )
            continue
        commands.extend(item["commands"])
        commands.append(item["aggregate_command"])
    commands.extend(plan.get("commands", []))
    for command in commands:
        print("+", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            return result.returncode
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", help="Override Python for every model profile")
    parser.add_argument(
        "--profile",
        choices=("full", "reviewer", DL3DV_GATE_PROFILE),
        default="full",
        help=(
            "Select the complete or reviewer evidence protocol; dl3dv-gate is "
            "the fixed non-claim saes-quality selector"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override the finalized protocol and run deterministic sample indices [0, N)",
    )
    parser.add_argument(
        "--pairs",
        help="Run an ordered comma-separated subset such as transplat/re10k,mvsplat/acid",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "ae",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "orin"),
        default="auto",
        help="Select the real-hardware contract for performance execution",
    )
    parser.add_argument(
        "--figures",
        default="all",
        help="Comma-separated figure/table IDs or all",
    )
    parser.add_argument(
        "--require-key-results",
        action="store_true",
        help="Fail validation unless every mandatory catalog result passes",
    )
    parser.add_argument(
        "--allow-low-memory-attempt",
        action="store_true",
        help="Run the unchanged ASAP7 flow below the recommended 48 GiB threshold.",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=DEFAULT_CALIBRATION_ROOT,
        help="Legacy Functional-only training root; cannot freeze a DL3DV claim config",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=DEFAULT_DL3DV_CALIBRATION_MANIFEST,
        help="Prepared 24-train/8-holdout DL3DV calibration protocol manifest",
    )
    parser.add_argument(
        "--claim-timing-manifest",
        type=Path,
        help=(
            "Source-bound RTL timing manifest for mechanisms and performance "
            "claim workflows; quality claims do not require timing"
        ),
    )
    parser.add_argument(
        "--proxy-input",
        type=Path,
        help="filled reviewer-GPU proxy JSON (only for the proxy workflow)",
    )
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.mode == SAES_QUALITY_MODE:
        if args.profile != DL3DV_GATE_PROFILE:
            parser.error("saes-quality requires --profile dl3dv-gate")
        if args.num_samples is not None:
            parser.error("saes-quality is fixed to DL3DV sample index 0")
        if args.pairs is not None:
            parser.error("saes-quality does not accept --pairs")
        if args.device != "auto":
            parser.error("saes-quality uses the fixed CUDA diagnostic device")
    elif args.profile == DL3DV_GATE_PROFILE:
        parser.error("--profile dl3dv-gate is only valid with saes-quality")
    if args.allow_low_memory_attempt and args.mode != "physical":
        parser.error("--allow-low-memory-attempt is only valid with physical")
    if args.calibration_root != DEFAULT_CALIBRATION_ROOT and args.mode != "calibrate":
        parser.error("--calibration-root is only valid with calibrate")
    if (
        args.calibration_manifest != DEFAULT_DL3DV_CALIBRATION_MANIFEST
        and args.mode != "calibrate"
    ):
        parser.error("--calibration-manifest is only valid with calibrate")
    if args.claim_timing_manifest is not None and args.mode not in (
        CLAIM_ONLY_SOFTWARE_MODES | {"all", "all-eval"}
    ):
        parser.error(
            "--claim-timing-manifest is only valid with claim result workflows"
        )
    if args.mode == "proxy" and args.proxy_input is None:
        parser.error("proxy workflow requires --proxy-input")
    if args.mode == "proxy" and args.device != "auto":
        parser.error("proxy workflow does not accept --device orin")
    if args.mode != "proxy" and args.proxy_input is not None:
        parser.error("--proxy-input is only valid with the proxy workflow")
    return args


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(args)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    return execute_plan(plan, args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
