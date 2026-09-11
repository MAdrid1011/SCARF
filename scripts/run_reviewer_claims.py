#!/usr/bin/env python3
"""Run all three claim workflows from validated prerequisites.

The usual form accepts a prerequisite directory.  For a clean-checkout
workflow, ``--calibration-config`` plus ``--stimulus-root`` generates the
source-bound timing bundle in that directory before running the same gates.
No command in this script fabricates calibration, model, or hardware records.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _pair_filter(pairs: list[str]) -> tuple[tuple[str, str], ...]:
    return tuple(tuple(pair.split("/", 1)) for pair in pairs)  # type: ignore[return-value]


def build_key_results_plan(
    run_config: Path,
    *,
    output_root: Path | None = None,
    figure8_device: str | None = None,
    proxy_input: Path | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Build the ordered reviewer plan and materialize its shared selection."""
    from scripts.reviewer_run_config import build_selection, load_run_config, write_selection
    from scripts.run_ae import build_dataset_validation_commands, build_software_plan

    config, config_sha256 = load_run_config(run_config)
    if figure8_device is not None:
        if figure8_device not in {"auto", "orin", "proxy"}:
            raise ValueError("--figure8-device must be auto, orin, or proxy")
        config["figure8_device"] = figure8_device
        config_sha256 = __import__("hashlib").sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    output = (output_root or ROOT / config["output_root"]).resolve()
    workflow_python = python or sys.executable
    selection_path = write_selection(
        output / "run-selection.json", build_selection(ROOT, config, config_sha256)
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pairs = list(selection["pairs"])
    pair_filter = _pair_filter(pairs)
    count = config["samples_per_pair"]
    num_samples = None if count == "all" else int(count)

    quality = build_software_plan(
        "quality", output, python, num_samples, pair_filter, True,
        config["profile"], True, selection_file=selection_path,
        run_selection_sha256=selection["selection_sha256"], export_rtl_payload=True,
    )
    device = config["figure8_device"]
    if device == "auto":
        device = "orin" if _is_orin_host() else "proxy"
    timing_events = output / "timing-backend" / "rtl-events.json"
    timing_manifest = output / "timing-backend" / "manifest.json"
    stimuli_root = output / "rtl-stimuli"
    mechanism = build_software_plan(
        "mechanisms", output, python, num_samples, pair_filter, True,
        config["profile"], True, claim_timing_manifest=timing_manifest,
        selection_file=selection_path,
        run_selection_sha256=selection["selection_sha256"],
    )
    phases: list[dict[str, Any]] = []
    phases.append({
        "name": "quality",
        "commands": [
            *build_dataset_validation_commands(quality, output),
            *[command for item in quality for command in item["commands"]],
            *[item["aggregate_command"] for item in quality],
        ],
    })
    phases.append({
        "name": "source_rtl_timing",
        "commands": [
            [workflow_python, "scripts/build_reviewer_stimuli.py", "--quality-root",
             str(output / "quality"), "--selection-file", str(selection_path),
             "--output-root", str(stimuli_root)],
            [workflow_python, "scripts/run_rtl_timing.py", "--stimulus-root", str(stimuli_root),
             "--source-rtl", str(ROOT / "hardware/orin/rtl/ScarfTop.sv"),
             "--output", str(timing_events)],
            [workflow_python, "scripts/export_claim_timing.py", "--events", str(timing_events),
             "--source-rtl", str(ROOT / "hardware/orin/rtl/ScarfTop.sv"),
             "--output-dir", str(output / "timing-backend")],
        ],
    })
    phases.append({
        "name": "mechanisms",
        "commands": [
            *[command for item in mechanism for command in item["commands"]],
            *[item["aggregate_command"] for item in mechanism],
        ],
    })
    if device == "orin":
        performance = build_software_plan(
            "performance", output, python, num_samples, pair_filter, True,
            config["profile"], True, orin_available=True,
            claim_timing_manifest=timing_manifest, selection_file=selection_path,
            run_selection_sha256=selection["selection_sha256"],
        )
        performance_commands = [
            *[command for item in performance for command in item["commands"]],
            *[item["aggregate_command"] for item in performance],
        ]
    else:
        proxy_output = output / "reviewer-gpu-proxy.json"
        proxy_capture_input = (
            proxy_input.resolve()
            if proxy_input is not None
            else output / "reviewer-gpu-proxy-input.json"
        )
        if proxy_input is None:
            # The local smoke configuration deliberately omits a hand-filled
            # Orin contract. Keep the result non-claim and derive a compact
            # Figure 8 proxy from the quality records already produced above.
            performance_commands = [[
                workflow_python, "scripts/generate_local_proxy_report.py",
                "--quality-root", str(output / "quality"),
                "--output-dir", str(output / "reports"),
            ]]
        else:
            performance_commands = [[
                workflow_python, "scripts/capture_gpu_proxy.py", "--selection-file", str(selection_path),
                "--input", str(proxy_capture_input), "--output", str(proxy_output),
                "--continue-if-gpu-contended" if config["continue_if_gpu_contended"] else "--fail-if-gpu-contended",
            ], [
                workflow_python, "scripts/normalize_orin_proxy.py", "--input", str(proxy_output),
                "--output", str(output / "reports/orin-nx-proxy.json"),
            ], [
                workflow_python, "scripts/generate_proxy_report.py", "--input",
                str(output / "reports/orin-nx-proxy.json"), "--output-dir",
                str(output / "reports"),
            ]]
    phases.append({"name": "performance_or_proxy", "device": device, "commands": performance_commands})
    report_commands = [
        [workflow_python, "scripts/generate_report.py", "--input", str(output),
         "--output-dir", str(output / "reports"),
         "--figures", "table1,figure8,figure11"],
    ]
    if device == "proxy" and proxy_input is None:
        # Standard claim reporting intentionally filters local smoke evidence.
        # Restore the explicitly non-claiming local rows after that filter so a
        # reviewer can inspect all three models from the documented one-command
        # proxy workflow without mistaking them for paper results.
        report_commands.append([
            workflow_python, "scripts/generate_local_proxy_report.py",
            "--quality-root", str(output / "quality"),
            "--output-dir", str(output / "reports"),
        ])
    report_commands.append([
        workflow_python, "scripts/validate_ae.py", "--input", str(output),
        "--profile", "author-preflight", "--workflow-smoke",
    ])
    phases.append({
        "name": "reports_and_validation",
        "commands": report_commands,
    })
    return {
        "schema_version": "scarf-reviewer-key-results-plan-v1",
        "run_config": str(Path(run_config).resolve()),
        "run_config_sha256": config_sha256,
        "config": config,
        "output_root": str(output),
        "selection": str(selection_path),
        "selection_sha256": selection["selection_sha256"],
        "device": device,
        "phases": phases,
        "claim_scope": "workflow_smoke" if config["samples_per_pair"] != "all" else "profile_subset",
    }


def _is_orin_host() -> bool:
    try:
        return "Jetson Orin NX" in Path("/proc/device-tree/model").read_text(
            encoding="utf-8", errors="replace"
        ).rstrip("\x00")
    except OSError:
        return False


def execute_key_results_plan(plan: dict[str, Any]) -> int:
    """Run the ordered plan after enforcing the release calibration gate."""
    from scripts.mechanism_config import require_calibrated_mechanism

    try:
        require_calibrated_mechanism()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(
            "BLOCKED before model execution: release frozen calibration bundle is "
            f"required ({ROOT / 'artifact/calibration/frozen'}): {exc}",
            file=sys.stderr,
        )
        return 2
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "reviewer-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for phase in plan["phases"]:
        print(f"== {phase['name']} ==", flush=True)
        for command in phase["commands"]:
            run(command)
    return 0


def build_prerequisites(args: argparse.Namespace, output: Path) -> Path:
    """Create a validated prerequisite tree from local calibrated inputs."""
    generated = output / "generated-prerequisites"
    calibration = generated / "calibration/mechanism_config.json"
    calibration.parent.mkdir(parents=True, exist_ok=True)
    if args.run_calibration:
        run(
            [
                args.python,
                "scripts/run_calibration.py",
                "--output-root",
                str(output / "calibration"),
            ]
        )
        source_config = output / "calibration/calibration/mechanism_config.json"
    elif args.calibration_config is not None:
        source_config = args.calibration_config.resolve()
    else:
        raise ValueError(
            "generated prerequisites require --calibration-config or --run-calibration"
        )
    if not source_config.is_file():
        raise FileNotFoundError(f"calibrated configuration not found: {source_config}")
    shutil.copy2(source_config, calibration)

    timing_dir = generated / "timing-backend"
    if args.stimulus_root is not None:
        events = generated / "rtl-events.json"
        run(
            [
                args.python,
                "scripts/run_rtl_timing.py",
                "--stimulus-root",
                str(args.stimulus_root.resolve()),
                "--source-rtl",
                str(args.source_rtl.resolve()),
                "--output",
                str(events),
            ]
        )
        event_input = events
    elif args.timing_events is not None:
        event_input = args.timing_events.resolve()
    else:
        raise ValueError(
            "generated prerequisites require --stimulus-root or --timing-events"
        )
    run(
        [
            args.python,
            "scripts/export_claim_timing.py",
            "--events",
            str(event_input),
            "--source-rtl",
            str(args.source_rtl.resolve()),
            "--output-dir",
            str(timing_dir),
        ]
    )
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prerequisites", type=Path, help="directory containing real calibration/ and timing-backend/")
    parser.add_argument("--calibration-config", type=Path, help="frozen status: calibrated mechanism_config.json")
    parser.add_argument("--run-calibration", action="store_true", help="run the pinned DL3DV download/prepare/sweep/freeze pipeline")
    parser.add_argument("--stimulus-root", type=Path, help="directory of packed scarf-rtl-stimulus-v1 files")
    parser.add_argument("--timing-events", type=Path, help="existing claim-eligible RTL event export")
    parser.add_argument("--source-rtl", type=Path, default=ROOT / "hardware/orin/rtl/ScarfTop.sv")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--profile", choices=("full", "reviewer"), default="reviewer")
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Override the Python interpreter for every model profile; by default "
            "SCARF_PYTHON_CLASSIC and SCARF_PYTHON_DEPTHSPLAT are resolved per model"
        ),
    )
    parser.add_argument("--run-config", type=Path, help="artifact/key_results_run.json")
    parser.add_argument("--figure8-device", choices=("auto", "orin", "proxy"))
    parser.add_argument(
        "--proxy-input",
        type=Path,
        help="filled reviewer-GPU proxy JSON (required when Figure 8 device is proxy)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the ordered run-config plan")
    args = parser.parse_args(argv)
    if args.run_config is not None:
        try:
            plan = build_key_results_plan(
                args.run_config,
                output_root=args.output_root,
                figure8_device=args.figure8_device,
                proxy_input=args.proxy_input,
                python=args.python,
            )
            if args.dry_run:
                print(json.dumps(plan, indent=2, sort_keys=True))
                return 0
            return execute_key_results_plan(plan)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            print(f"BLOCKED: reviewer key-results plan failed: {exc}", file=sys.stderr)
            return 2
    if args.prerequisites is None and args.stimulus_root is None and args.timing_events is None:
        parser.error("provide --prerequisites, or generated inputs (--stimulus-root/--timing-events)")
    if args.stimulus_root is not None and args.timing_events is not None:
        parser.error("--stimulus-root and --timing-events are mutually exclusive")
    if args.prerequisites is not None and (args.calibration_config or args.run_calibration or args.stimulus_root or args.timing_events):
        parser.error("generated inputs cannot be combined with --prerequisites")
    if args.calibration_config is not None and args.run_calibration:
        parser.error("--calibration-config and --run-calibration are mutually exclusive")
    output = (args.output_root or ROOT / "outputs/ae").resolve()
    try:
        prerequisites = args.prerequisites.resolve() if args.prerequisites is not None else build_prerequisites(args, output)
        manifest = output / "timing-backend/manifest.json"
        run([args.python, "scripts/install_claim_prerequisites.py", "--input", str(prerequisites), "--profile", args.profile, "--output-root", str(output)])
        run([args.python, "scripts/claim_timing_backend.py", str(manifest)])
        for mode, extra in (("quality", []), ("mechanisms", []), ("performance", ["--device", "orin"])):
            command = ["bash", "scripts/run_ae.sh", mode, *extra]
            if mode in {"mechanisms", "performance"}:
                command.extend(["--claim-timing-manifest", str(manifest)])
            command.extend(["--output-root", str(output)])
            run(command)
        run([args.python, "scripts/validate_ae.py", "--input", str(output), "--require-key-results"])
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"BLOCKED: reviewer claim workflow failed: {exc}", file=sys.stderr)
        return 2
    print("PASS: claim workflows and validate_ae.py --require-key-results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
