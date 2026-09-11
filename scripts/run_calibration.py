#!/usr/bin/env python3
"""Run the documented DL3DV calibration pipeline end to end.

This is a convenience wrapper around the release's audited download,
preparation, sweep, and freeze commands.  It never fabricates calibration
records: any missing data, checkpoint, or runtime dependency is reported as a
failure and no calibrated configuration is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVISION = "5902ed6d707cc13a7779907c1e096676f7707971"
DEFAULT_INDEX = ROOT / "depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/calibration")
    parser.add_argument("--evaluation-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-download", action="store_true", help="reuse an existing prepared DL3DV tree")
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    index = args.evaluation_index.resolve()
    plan = output / "dl3dv-download-plan.json"
    source_root = output / "dl3dv"
    protocol = output / "dl3dv-protocol"
    sweep = output / "dl3dv-sweep"
    try:
        output.mkdir(parents=True, exist_ok=True)
        if not args.skip_download:
            run([args.python, "data/download_dl3dv_calibration.py", "--write-plan", str(plan), "--evaluation-index", str(index), "--revision", args.revision])
            run([args.python, "data/download_dl3dv_calibration.py", "--plan", str(plan), "--evaluation-index", str(index), "--revision", args.revision, "--output-root", str(source_root)])
        preparation = source_root / ".scarf-dl3dv-calibration-source.json"
        run([args.python, "data/prepare_dl3dv_calibration_inputs.py", "--raw-root", str(source_root / "prepared"), "--plan", str(plan), "--preparation-record", str(preparation), "--output-dir", str(protocol)])
        manifest = protocol / "manifest.json"
        run([args.python, "scripts/calibration_sweep.py", "--manifest", str(manifest), "--output-dir", str(sweep)])
        config_path = ROOT / "artifact/mechanism_config.json"
        run([args.python, "scripts/calibrate_mechanisms.py", "--candidate-records", str(sweep / "candidates.json"), "--output-dir", str(sweep), "--config-output", str(config_path)])
        # Keep a self-contained prerequisite export beside the calibration
        # provenance. The claim installer consumes this frozen copy.
        frozen = output / "calibration" / "mechanism_config.json"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, frozen)
        (output / "calibration" / "SHA256SUMS").write_text(
            f"{hashlib.sha256(frozen.read_bytes()).hexdigest()}  mechanism_config.json\n",
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"BLOCKED: calibration pipeline failed: {exc}", file=sys.stderr)
        return 2
    print(f"CALIBRATED: {ROOT / 'artifact/mechanism_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
