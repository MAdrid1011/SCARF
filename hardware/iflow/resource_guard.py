#!/usr/bin/env python3
"""Reject physical runs that would contend with Vivado or insufficient RAM."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


MINIMUM_AVAILABLE_BYTES = 48 * 1024**3


def vivado_processes(active_processes: list[str]) -> list[str]:
    normalized = [name.lower() for name in active_processes]
    return [
        process
        for process, normalized_process in zip(active_processes, normalized)
        if "vivado" in normalized_process
    ]


def require_no_vivado(active_processes: list[str]) -> None:
    if vivado_processes(active_processes):
        raise RuntimeError("Vivado is active; the SCARF physical flow must wait")


def check_resources(
    *,
    available_bytes: int,
    active_processes: list[str],
    allow_low_memory_attempt: bool = False,
) -> dict:
    """Validate the host before a physical run.

    The default path requires enough free memory for an unconstrained full
    implementation. An explicit low-memory attempt is permitted for evidence
    collection, but remains visibly labeled in its resource record.
    """

    require_no_vivado(active_processes)
    threshold_met = available_bytes >= MINIMUM_AVAILABLE_BYTES
    if not threshold_met and not allow_low_memory_attempt:
        available_gib = available_bytes / 1024**3
        raise RuntimeError(
            f"physical flow requires 48 GiB available memory, found {available_gib:.1f} GiB"
        )
    return {
        "pass": True,
        "mode": "standard" if threshold_met else "low_memory_attempt",
        "available_bytes": int(available_bytes),
        "minimum_available_bytes": MINIMUM_AVAILABLE_BYTES,
        "available_memory_threshold_met": threshold_met,
        "allow_low_memory_attempt": bool(allow_low_memory_attempt),
        "active_vivado_processes": vivado_processes(active_processes),
    }


def memory_snapshot(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        fields[key] = value.strip()
    values = {}
    for name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
        text = fields.get(name)
        if text is None or not text.endswith(" kB"):
            raise ValueError(f"/proc/meminfo has no {name} field")
        values[name] = int(text[:-3].strip()) * 1024
    return {
        "mem_total_bytes": values["MemTotal"],
        "mem_available_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_free_bytes": values["SwapFree"],
        "swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
    }


def available_memory_bytes(path: Path = Path("/proc/meminfo")) -> int:
    return memory_snapshot(path)["mem_available_bytes"]


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
    parser.add_argument(
        "--allow-low-memory-attempt",
        action="store_true",
        help="Run unchanged physical stages below 48 GiB and label the evidence.",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Record host memory and reject newly active Vivado without a RAM threshold.",
    )
    args = parser.parse_args()
    try:
        snapshot = memory_snapshot()
        processes = process_names()
        if args.snapshot_only:
            require_no_vivado(processes)
            record = {
                "pass": True,
                "mode": "snapshot",
                "active_vivado_processes": vivado_processes(processes),
            }
        else:
            record = check_resources(
                available_bytes=snapshot["mem_available_bytes"],
                active_processes=processes,
                allow_low_memory_attempt=args.allow_low_memory_attempt,
            )
        record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        record["memory"] = snapshot
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
