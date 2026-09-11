#!/usr/bin/env python3
"""Single entry point for reviewer-side evidence generation and validation.

The command exposes the complete sequence without hiding platform-dependent
inputs.  ``calibrate`` uses public DL3DV data; ``stimulus`` and ``timing``
consume outputs from a real model run and RTL simulator respectively;
``proxy`` handles a reviewer GPU without Jetson hardware; ``validate`` runs
the release gates.  No subcommand creates synthetic claim evidence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> int:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    calibrate = sub.add_parser("calibrate", help="download, prepare, sweep, and freeze DL3DV configuration")
    calibrate.add_argument("--output-root", type=Path, default=ROOT / "outputs/calibration")
    calibrate.add_argument("--evaluation-index", type=Path)
    calibrate.add_argument("--revision")
    calibrate.add_argument("--skip-download", action="store_true")
    stimulus = sub.add_parser("stimulus", help="convert one model sample result to RTL adapter input")
    stimulus.add_argument("sample", type=Path)
    stimulus.add_argument("--output", type=Path, required=True)
    stimulus.add_argument("--input-root", type=Path)
    stimulus.add_argument("--payload", type=Path, action="append")
    timing = sub.add_parser("timing", help="convert real RTL simulator events into a validated timing bundle")
    timing.add_argument("--events", type=Path, required=True)
    timing.add_argument("--source-rtl", type=Path, required=True)
    timing.add_argument("--output-dir", type=Path, required=True)
    proxy = sub.add_parser("proxy", help="normalize a filled reviewer-GPU input")
    proxy.add_argument("--input", type=Path, required=True)
    proxy.add_argument("--output-root", type=Path, default=ROOT / "outputs/ae")
    validate = sub.add_parser("validate", help="audit claim prerequisites and result tree")
    validate.add_argument("--input", type=Path, default=ROOT / "outputs/ae")
    validate.add_argument("--profile", choices=("full", "reviewer"), default="reviewer")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        command = [sys.executable, "scripts/run_calibration.py", "--output-root", str(args.output_root)]
        for flag, value in (("--evaluation-index", args.evaluation_index), ("--revision", args.revision)):
            if value:
                command.extend([flag, str(value)])
        if args.skip_download:
            command.append("--skip-download")
        return _run(command)
    if args.command == "stimulus":
        command = [sys.executable, "scripts/export_rtl_stimulus.py", str(args.sample), "--output", str(args.output)]
        if args.input_root:
            command.extend(["--input-root", str(args.input_root)])
        for payload in args.payload or []:
            command.extend(["--payload", str(payload)])
        return _run(command)
    if args.command == "timing":
        return _run([sys.executable, "scripts/export_claim_timing.py", "--events", str(args.events), "--source-rtl", str(args.source_rtl), "--output-dir", str(args.output_dir)])
    if args.command == "proxy":
        return _run(["bash", "scripts/run_ae.sh", "proxy", "--proxy-input", str(args.input), "--output-root", str(args.output_root)])
    return _run([sys.executable, "scripts/claim_readiness.py", "--root", str(ROOT), "--output-root", str(args.input), "--profile", args.profile])


if __name__ == "__main__":
    raise SystemExit(main())
