#!/usr/bin/env python3
"""Validate and collect the public DRAM proxy evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.result_record import source_identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_artifact(path: Path, output_dir: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(output_dir.resolve()).as_posix(),
        "path_base": "output_dir",
        "sha256": sha256_file(path),
    }


def collect(output_dir: Path) -> dict:
    paths = {
        "trace_manifest": output_dir / "trace-manifest.json",
        "ramulator": output_dir / "ramulator.json",
        "drampower": output_dir / "drampower.json",
    }
    records = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    requests = records["trace_manifest"].get("request_count")
    cycles = records["ramulator"].get("metrics", {}).get("memory_cycles")
    energy = records["drampower"].get("metrics", {}).get("offchip_energy_j")
    values = (requests, cycles, energy)
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
        for value in values
    ):
        raise ValueError("DRAM evidence contains missing, zero, or non-finite metrics")
    source = source_identity()
    workload = records["trace_manifest"].get("workload")
    if workload is not None:
        if not isinstance(workload, dict):
            raise ValueError("DRAM workload binding is invalid")
        count = workload.get("sample_count")
        software = workload.get("software_result")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not isinstance(software, dict)
            or software.get("path_base") != "output_dir"
        ):
            raise ValueError("DRAM workload binding is incomplete")
        software_path = output_dir / str(software.get("path", ""))
        if (
            not software_path.is_file()
            or sha256_file(software_path) != software.get("sha256")
        ):
            raise ValueError("DRAM workload software result hash mismatch")
    claim = (
        "per_inference_workload_proxy"
        if workload is not None
        else "functional public proxy only"
    )
    metrics = {
        "trace_requests": requests,
        "ramulator_memory_cycles": cycles,
        "ramulator_average_read_latency_cycles": records["ramulator"]["metrics"].get(
            "average_read_latency_cycles"
        ),
        "drampower_offchip_energy_j": energy,
    }
    if workload is not None:
        metrics["drampower_offchip_energy_per_inference_j"] = energy / workload[
            "sample_count"
        ]
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "provenance": {
            "git_commit": source["git_commit"],
            "git_dirty": source["git_dirty"],
            "source_identity": source["source"],
            "submodules": source["submodules"],
        },
        "evidence_type": "public_memory_system_proxy",
        "paper_lpddr4x_reproduced": False,
        "metrics": metrics,
        "artifacts": {
            key: output_artifact(path, output_dir) for key, path in paths.items()
        },
        "scope": {
            "timing_model": "Ramulator 2.1 LPDDR5-6400",
            "energy_model": "DRAMPower 6.0.2 LPDDR5",
            "paper_interface": "LPDDR4X",
            "claim": claim,
        },
    }
    if workload is not None:
        result["workload"] = workload
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = collect(args.output_dir.resolve())
        output = args.output_dir / "results.json"
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
