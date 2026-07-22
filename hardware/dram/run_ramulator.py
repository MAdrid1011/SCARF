#!/usr/bin/env python3
"""Run pinned Ramulator 2.1 on a SCARF LoadStoreTrace."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
VERSIONS = json.loads((HERE / "versions.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_checkout(root: Path, component: str, license_name: str) -> str:
    expected = VERSIONS[component]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    commit = result.stdout.strip()
    if result.returncode or commit != expected["commit"]:
        raise ValueError(f"{component} commit mismatch: expected {expected['commit']}, got {commit or 'unknown'}")
    license_path = root / license_name
    if not license_path.is_file() or sha256_file(license_path) != expected["license_sha256"]:
        raise ValueError(f"{component} license hash mismatch")
    return commit


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def output_artifact(path: Path, output_dir: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(output_dir.resolve()).as_posix(),
        "path_base": "output_dir",
        "sha256": sha256_file(path),
    }


def validate_metrics(metrics: dict[str, Any], command_text: str) -> None:
    cycles = metrics.get("memory_cycles")
    submitted = metrics.get("submitted_requests")
    completed = metrics.get("completed_requests")
    reads = metrics.get("read_requests")
    writes = metrics.get("write_requests")
    latency = metrics.get("average_read_latency_cycles")
    if not _positive_number(cycles):
        raise RuntimeError("Ramulator reported zero memory cycles")
    if not _positive_number(submitted) or submitted != completed:
        raise RuntimeError("Ramulator request queues were not fully drained")
    if not isinstance(reads, int) or isinstance(reads, bool) or reads < 0:
        raise RuntimeError("Ramulator emitted an invalid read count")
    if not isinstance(writes, int) or isinstance(writes, bool) or writes < 0:
        raise RuntimeError("Ramulator emitted an invalid write count")
    if reads + writes != completed:
        raise RuntimeError("Ramulator completion counters are inconsistent")
    if reads and not _positive_number(latency):
        raise RuntimeError("Ramulator reported zero read latency")
    if reads and ",RD," not in command_text:
        raise RuntimeError("Ramulator command trace has no completed read command")
    if writes and ",WR," not in command_text:
        raise RuntimeError("Ramulator command trace has no completed write command")


def run(root: Path, trace: Path, output_dir: Path, driver: Path) -> dict[str, Any]:
    commit = check_checkout(root, "ramulator2", "LICENSE")
    if not driver.is_file() or not os.access(driver, os.X_OK):
        raise FileNotFoundError("SCARF Ramulator driver is not built; run hardware/dram/build_driver.sh")
    python_root = root / "python"
    sys.path.insert(0, str(python_root))
    try:
        ramulator = importlib.import_module("ramulator")
    except ImportError as exc:
        raise RuntimeError("Ramulator Python configuration package is unavailable") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    command_prefix = Path("ramulator_commands.csv")
    frontend = ramulator.frontend.External(clock_ratio=1)
    dram = ramulator.dram.LPDDR5(
        org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400", rank=1
    )
    controller = ramulator.controller.LPDDR5(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        controller_plugins=[
            ramulator.controller_plugin.CmdTraceRecorder(path=str(command_prefix))
        ],
    )
    memory = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[controller],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )
    config = {
        "frontend": frontend.to_config(),
        "memory_system": memory.to_config(),
    }
    config_path = output_dir / "ramulator_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path = output_dir / "ramulator_metrics.json"
    raw_stats = output_dir / "ramulator_stats.yaml"
    log = output_dir / "ramulator.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(driver), str(config_path), str(trace), str(metrics_path), str(raw_stats)],
            cwd=output_dir,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"Ramulator external driver failed; inspect {log}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    command_trace = output_dir / f"{command_prefix}.ch0"
    if not command_trace.is_file():
        raise RuntimeError("Ramulator did not emit complete timing evidence")
    command_text = command_trace.read_text(encoding="utf-8")
    validate_metrics(metrics, command_text)
    cycles = metrics["memory_cycles"]
    reads = metrics["read_requests"]
    writes = metrics["write_requests"]
    latency = metrics["average_read_latency_cycles"]
    return {
        "schema_version": "1.0",
        "evidence_type": "ramulator2_lpddr5_proxy",
        "memory_standard": "LPDDR5-6400 public proxy; not paper LPDDR4X measurement",
        "metrics": {
            "memory_cycles": cycles,
            "average_read_latency_cycles": latency,
            "read_requests": reads,
            "write_requests": writes,
        },
        "artifacts": {
            "input_trace": output_artifact(trace, output_dir),
            "raw_stats": output_artifact(raw_stats, output_dir),
            "driver_metrics": output_artifact(metrics_path, output_dir),
            "driver_log": output_artifact(log, output_dir),
            "resolved_config": output_artifact(config_path, output_dir),
            "command_trace": output_artifact(command_trace, output_dir),
        },
        "provenance": {
            "ramulator_commit": commit,
            "configuration": {
                "dram": "LPDDR5_8Gb_x16",
                "timing": "LPDDR5_6400",
                "controller": "LPDDR5/FRFCFS/Open/AllBank",
                "channels": 1,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ramulator-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--driver", type=Path, default=HERE / "build/ramulator_driver")
    args = parser.parse_args()
    try:
        record = run(
            args.ramulator_root.resolve(),
            args.trace.resolve(),
            args.output_dir.resolve(),
            args.driver.resolve(),
        )
        output = args.output_dir / "ramulator.json"
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
