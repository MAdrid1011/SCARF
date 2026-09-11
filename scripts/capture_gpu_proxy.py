#!/usr/bin/env python3
"""Validate a same-selection reviewer-GPU timing capture for the Orin proxy.

The capture file is supplied by the reviewer after running the CUDA-event
measurement command.  This helper only binds it to ``run-selection.json`` and
records whether a competing GPU process was observed; it never invents
latencies or converts diagnostic timings into claim evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Allow ``python scripts/capture_gpu_proxy.py`` from a clean checkout, just
# like the other repository entry points.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reviewer_run_config import load_selection


def _contended() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return bool(result.stdout.strip())


def capture(
    input_path: Path,
    output_path: Path,
    selection_file: Path,
    *,
    continue_if_gpu_contended: bool,
) -> Path:
    selection = load_selection(selection_file)
    source = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("GPU proxy capture must be a JSON object")
    if source.get("schema_version") != "scarf-gpu-proxy-input-v1":
        raise ValueError("GPU proxy capture must use scarf-gpu-proxy-input-v1")
    if source.get("claim_eligible") is not False:
        raise ValueError("GPU proxy capture must be claim_eligible=false")
    workload = source.get("workload")
    if not isinstance(workload, dict) or workload.get("selection_sha256") != selection["selection_sha256"]:
        raise ValueError("GPU proxy capture is not bound to the shared run selection")
    contended = _contended()
    if contended and not continue_if_gpu_contended:
        raise RuntimeError("a competing GPU process is active; rerun with continue_if_gpu_contended")
    capture_record = dict(source)
    capture_metadata = dict(capture_record.get("capture", {}))
    capture_metadata["timing_quality"] = "contended" if contended else "isolated"
    capture_metadata["contention_policy"] = (
        "continue_and_mark" if continue_if_gpu_contended else "fail_on_contention"
    )
    capture_record["capture"] = capture_metadata
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(capture_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--continue-if-gpu-contended", action="store_true")
    parser.add_argument("--fail-if-gpu-contended", action="store_true")
    args = parser.parse_args(argv)
    if args.continue_if_gpu_contended == args.fail_if_gpu_contended:
        parser.error("choose exactly one contention policy")
    try:
        print(capture(
            args.input,
            args.output,
            args.selection_file,
            continue_if_gpu_contended=args.continue_if_gpu_contended,
        ))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
