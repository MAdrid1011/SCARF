#!/usr/bin/env python3
"""Record an incomplete iFlow attempt without promoting it to PPA evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.iflow.resource_guard import memory_snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_files(output_dir: Path) -> list[dict[str, str]]:
    patterns = (
        "manifest.json",
        "resource-*.json",
        "stage-*-validation.json",
        "iflow-*.log",
        "time-*.log",
    )
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in output_dir.glob(pattern)
            if path.is_file()
        }
    )
    return [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def completed_stages(output_dir: Path) -> list[str]:
    stages = []
    for path in sorted(output_dir.glob("stage-*-validation.json")):
        stage = path.name.removeprefix("stage-").removesuffix("-validation.json")
        stages.append(stage)
    return stages


def build_record(
    output_dir: Path,
    *,
    incomplete_stage: str,
    reason: str,
    termination: str,
    sample_seconds: float | None = None,
    swap_in_pages: int | None = None,
    swap_out_pages: int | None = None,
) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise ValueError(f"physical output directory does not exist: {output_dir}")
    if not incomplete_stage:
        raise ValueError("incomplete_stage must be non-empty")
    if not reason:
        raise ValueError("reason must be non-empty")
    if not termination:
        raise ValueError("termination must be non-empty")
    if sample_seconds is not None and sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if swap_in_pages is not None and swap_in_pages < 0:
        raise ValueError("swap_in_pages must be non-negative")
    if swap_out_pages is not None and swap_out_pages < 0:
        raise ValueError("swap_out_pages must be non-negative")
    return {
        "schema_version": "1.0",
        "status": "NOT_CLAIMED_RESOURCE_LIMIT",
        "physical_valid": False,
        "incomplete_stage": incomplete_stage,
        "reason": reason,
        "termination": termination,
        "completed_stages": completed_stages(output_dir),
        "resource_snapshot_at_recording": memory_snapshot(),
        "swap_observation": {
            "sample_seconds": sample_seconds,
            "swap_in_pages": swap_in_pages,
            "swap_out_pages": swap_out_pages,
        },
        "evidence": evidence_files(output_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--incomplete-stage", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--termination", required=True)
    parser.add_argument("--sample-seconds", type=float)
    parser.add_argument("--swap-in-pages", type=int)
    parser.add_argument("--swap-out-pages", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record = build_record(
            args.output_dir,
            incomplete_stage=args.incomplete_stage,
            reason=args.reason,
            termination=args.termination,
            sample_seconds=args.sample_seconds,
            swap_in_pages=args.swap_in_pages,
            swap_out_pages=args.swap_out_pages,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    output = args.output_dir / "attempt-outcome.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
