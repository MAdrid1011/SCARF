#!/usr/bin/env python3
"""Reject physical runs that would contend with Vivado or insufficient RAM."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


MINIMUM_AVAILABLE_BYTES = 48 * 1024**3


def check_resources(*, available_bytes: int, active_processes: list[str]) -> dict:
    normalized = [name.lower() for name in active_processes]
    if any("vivado" in name for name in normalized):
        raise RuntimeError("Vivado is active; the SCARF physical flow must wait")
    if available_bytes < MINIMUM_AVAILABLE_BYTES:
        available_gib = available_bytes / 1024**3
        raise RuntimeError(
            f"physical flow requires 48 GiB available memory, found {available_gib:.1f} GiB"
        )
    return {
        "pass": True,
        "available_bytes": int(available_bytes),
        "minimum_available_bytes": MINIMUM_AVAILABLE_BYTES,
        "active_processes": list(active_processes),
    }


def available_memory_bytes(path: Path = Path("/proc/meminfo")) -> int:
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        fields[key] = value.strip()
    text = fields.get("MemAvailable")
    if text is None or not text.endswith(" kB"):
        raise ValueError("/proc/meminfo has no MemAvailable field")
    return int(text[:-3].strip()) * 1024


def process_names() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "args="], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError("cannot inspect active processes")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        record = check_resources(
            available_bytes=available_memory_bytes(), active_processes=process_names()
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
