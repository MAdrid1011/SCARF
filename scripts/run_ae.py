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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ae_config import (
    CLAIMED_MATRIX,
    ClaimSelection,
    resolve_claim_selection,
    resolve_experiment,
)
from scripts.compile_protocol import canonicalize_index


SOFTWARE_MODES = {"quick", "quality", "speedup", "ablation", "orin"}
MODES = tuple(
    sorted(
        SOFTWARE_MODES
        | {"all", "dram", "sensitivity", "rtl", "physical", "scale", "report", "validate"}
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
    if mode in {"quality", "ablation"}:
        return _claimed_pairs()
    if mode in {"speedup", "orin"}:
        return (
            _claimed_pairs()
            if load_claim_status().get("figure8") == "CLAIMED"
            else ()
        )
    return ()


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
) -> list[dict[str, Any]]:
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive")
    experiments = []
    for model, dataset in _pairs_for_mode(mode):
        sample_count = (
            num_samples
            if num_samples is not None
            else (1 if mode == "quick" else _protocol_sample_count(model, dataset))
        )
        config = resolve_experiment(model, dataset, ROOT)
        selection = (
            _quick_selection()
            if mode == "quick"
            else resolve_claim_selection(model, dataset, ROOT)
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
            "--functional-run" if mode == "quick" else "--claim-run",
            "--device",
            "auto",
            "--seed",
            "0",
        ]
        if mode in {"ablation", "all"}:
            demo_command.append("--ablation")
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
                "result": str(output_dir / "results.json"),
            }
        )
    if mode in {"orin", "speedup"}:
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
    num_samples = getattr(args, "num_samples", 1)
    if args.mode == "all":
        experiments = []
        for workflow in ("quality", "speedup"):
            experiments.extend(
                build_software_plan(workflow, args.output_root, args.python, num_samples)
            )
    else:
        experiments = build_software_plan(
            args.mode, args.output_root, args.python, num_samples
        )
    plan: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": args.mode,
        "root": str(ROOT),
        "output_root": str(args.output_root),
        "experiments": experiments,
        "claim_status": load_claim_status(),
    }
    plan["dataset_commands"] = build_dataset_validation_commands(
        experiments, args.output_root
    )
    sensitivity_command = [
        sys.executable,
        str(SCRIPT_DIR / "sensitivity_sweep.py"),
        "--output-dir",
        str(args.output_root / "sensitivity"),
    ]
    if num_samples is not None:
        sensitivity_command.extend(("--num-samples", str(num_samples)))
    sensitivity_claimed = plan["claim_status"].get("sensitivity") == "CLAIMED"
    if args.mode == "sensitivity":
        plan["commands"] = [sensitivity_command] if sensitivity_claimed else []
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
        plan["commands"] = [["bash", str(ROOT / "hardware" / "iflow" / "run.sh"), "--platform", "asap7", "--stage", "all", "--output-dir", str(args.output_root / "physical" / "asap7")]]
    elif args.mode == "scale":
        plan["commands"] = [[sys.executable, str(ROOT / "hardware/scaling/deepscale.py"), "--source-node", "7", "--target-node", "28", "--input", str(args.output_root / "physical/asap7/ppa.json"), "--output", str(args.output_root / "physical/asap7/ppa_28nm_estimated.json")]]
    elif args.mode == "report":
        plan["commands"] = [[sys.executable, str(SCRIPT_DIR / "generate_report.py"), "--input", str(args.output_root), "--output-dir", str(args.output_root / "reports")]]
    elif args.mode == "validate":
        plan["commands"] = [[sys.executable, str(SCRIPT_DIR / "validate_ae.py"), "--input", str(args.output_root)]]
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
    return plan


def execute_plan(plan: dict[str, Any], output_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(plan)
    manifest["started_at"] = datetime.now(timezone.utc).isoformat()
    (output_root / f"manifest-{plan['mode']}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
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
        "--num-samples",
        type=int,
        default=None,
        help="Override the finalized protocol and run deterministic sample indices [0, N)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "ae",
    )
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
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
