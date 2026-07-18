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
        return configured
    local_profile = ROOT / ".venv" / profile / "bin" / "python"
    return str(local_profile) if local_profile.is_file() else sys.executable


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


def build_software_plan(
    mode: str,
    output_root: Path,
    python_override: str | None = None,
    num_samples: int | None = None,
    pair_filter: tuple[tuple[str, str], ...] | None = None,
    allow_partial_filter: bool = False,
    evidence_profile: str = "full",
    claim_execution: bool = True,
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
        output_dir = output_root / mode / f"{model}_{dataset}"
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
            "--",
            *demo_command,
        ]
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
                "result": str(output_dir / "results.json"),
            }
        )
    if mode in {"orin", "speedup", "performance"}:
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
    pair_filter = parse_pair_filter(getattr(args, "pairs", None))
    if pair_filter is not None and args.mode not in SOFTWARE_MODES | {"all", "all-eval", "pilot"}:
        raise ValueError(f"--pairs is not supported by {args.mode}")
    if args.mode == "calibrate":
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
        for workflow in ("quality", "speedup", "fsdr"):
            experiments.extend(
                build_software_plan(
                    workflow,
                    args.output_root,
                    args.python,
                    num_samples,
                    claimed_pair_filter,
                    True,
                    evidence_profile,
                )
            )
    elif args.mode == "all-eval":
        experiments = []
        for workflow in ("quality", "performance", "mechanisms", "utilization"):
            experiments.extend(
                build_software_plan(
                    workflow,
                    args.output_root,
                    args.python,
                    num_samples,
                    pair_filter,
                    True,
                    evidence_profile,
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
        )
    effective_profile = (
        "calibration"
        if args.mode == "calibrate"
        else "pilot" if args.mode == "pilot" else evidence_profile
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
    }
    plan["software_claim_scope"] = {
        "status": (
            "NO_CLAIMED_PAIRS"
            if args.mode in CLAIM_ONLY_SOFTWARE_MODES | {"all", "all-eval"}
            and not experiments
            else "ACTIVE"
        ),
        "pair_count": len(experiments),
        "diagnostic_results_are_claim_evidence": False,
    }
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
    sensitivity_claimed = plan["claim_status"].get("sensitivity") == "CLAIMED"
    if args.mode == "calibrate":
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
        physical_commands = []
        if plan["claim_status"].get("physical_asap7") == "CLAIMED":
            physical_commands.append(
                ["bash", str(ROOT / "hardware/iflow/run.sh"), "--platform", "asap7", "--stage", "all", "--output-dir", str(args.output_root / "physical/asap7")]
            )
        if plan["claim_status"].get("deepscale") == "CLAIMED":
            physical_commands.append(
                [sys.executable, str(ROOT / "hardware/scaling/deepscale.py"), "--source-node", "7", "--target-node", "28", "--input", str(args.output_root / "physical/asap7/ppa.json"), "--output", str(args.output_root / "physical/asap7/ppa_28nm_estimated.json")]
            )
        plan["commands"] = [
            *([sensitivity_command] if sensitivity_claimed else []),
            ["bash", str(SCRIPT_DIR / "run_rtl.sh"), "--output-dir", str(args.output_root / "rtl")],
            ["bash", str(ROOT / "hardware/dram/run.sh"), "--events", str(ROOT / "hardware/dram/test_vectors/scarf_smoke_events.jsonl"), "--output-dir", str(args.output_root / "dram")],
            *physical_commands,
            [sys.executable, str(SCRIPT_DIR / "generate_report.py"), "--input", str(args.output_root), "--output-dir", str(args.output_root / "reports")],
            [sys.executable, str(SCRIPT_DIR / "validate_ae.py"), "--input", str(args.output_root)],
        ]
    elif args.mode == "all-eval":
        plan["commands"] = [
            sensitivity_command,
            ["bash", str(SCRIPT_DIR / "run_rtl.sh"), "--output-dir", str(args.output_root / "rtl")],
            [
                "bash",
                str(ROOT / "hardware/dram/run.sh"),
                "--events",
                str(ROOT / "hardware/dram/test_vectors/scarf_smoke_events.jsonl"),
                "--output-dir",
                str(args.output_root / "dram"),
            ],
            [
                sys.executable,
                str(SCRIPT_DIR / "generate_report.py"),
                "--input",
                str(args.output_root),
                "--output-dir",
                str(args.output_root / "reports"),
                "--figures",
                "all",
            ],
            [
                sys.executable,
                str(SCRIPT_DIR / "validate_ae.py"),
                "--input",
                str(args.output_root),
                "--require-key-results",
            ],
        ]
    return plan


def execute_plan(plan: dict[str, Any], output_root: Path) -> int:
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
    for item in plan["experiments"]:
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
        choices=("full", "reviewer"),
        default="full",
        help="Select the complete or frozen reviewer evidence protocol",
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
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.allow_low_memory_attempt and args.mode != "physical":
        parser.error("--allow-low-memory-attempt is only valid with physical")
    if args.calibration_root != DEFAULT_CALIBRATION_ROOT and args.mode != "calibrate":
        parser.error("--calibration-root is only valid with calibrate")
    if (
        args.calibration_manifest != DEFAULT_DL3DV_CALIBRATION_MANIFEST
        and args.mode != "calibrate"
    ):
        parser.error("--calibration-manifest is only valid with calibrate")
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
