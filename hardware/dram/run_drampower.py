#!/usr/bin/env python3
"""Run pinned DRAMPower on a converted Ramulator command trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
VERSIONS = json.loads((HERE / "versions.json").read_text(encoding="utf-8"))
LPDDR5_TIMING_COMPATIBILITY_FIELDS = (
    "RTW_L_32",
    "RTW_L_16",
    "RTW_S_32",
    "RTW_S_16",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_artifact(path: Path, base: Path, path_base: str) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(base.resolve()).as_posix(),
        "path_base": path_base,
        "sha256": sha256_file(path),
    }


def materialize_memspec(root: Path, output_dir: Path, expected: dict) -> tuple[Path, Path]:
    source = root / "tests/tests_drampower/resources/lpddr5.json"
    if sha256_file(source) != expected["lpddr5_memspec_sha256"]:
        raise ValueError("DRAMPower LPDDR5 memspec hash mismatch")
    record = json.loads(source.read_text(encoding="utf-8"))
    timing = record.get("memspec", {}).get("memtimingspec")
    if not isinstance(timing, dict):
        raise ValueError("DRAMPower LPDDR5 memspec has no timing object")
    present = [field for field in LPDDR5_TIMING_COMPATIBILITY_FIELDS if field in timing]
    if present:
        raise ValueError(f"pinned LPDDR5 memspec unexpectedly defines compatibility fields: {present}")
    for field in LPDDR5_TIMING_COMPATIBILITY_FIELDS:
        timing[field] = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    derived = output_dir / "lpddr5_proxy_memspec.json"
    derived.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return source, derived


def run(root: Path, command_trace: Path, output_dir: Path) -> dict:
    expected = VERSIONS["drampower"]
    commit_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    commit = commit_result.stdout.strip()
    if commit_result.returncode or commit != expected["commit"]:
        raise ValueError(f"DRAMPower commit mismatch: expected {expected['commit']}, got {commit or 'unknown'}")
    license_path = root / "LICENSE.txt"
    executable = root / "build/bin/cli"
    if sha256_file(license_path) != expected["license_sha256"]:
        raise ValueError("DRAMPower license hash mismatch")
    if not executable.is_file():
        raise FileNotFoundError("DRAMPower CLI is not built; follow the pinned build instructions")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_memspec, memspec = materialize_memspec(root, output_dir, expected)
    raw_json = output_dir / "drampower_raw.json"
    raw_json.write_text("", encoding="utf-8")
    log = output_dir / "drampower.log"
    config = HERE / "drampower_config.json"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(executable), "-c", str(config), "-m", str(memspec), "-t", str(command_trace), "-j", str(raw_json)],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"DRAMPower failed; inspect {log}")
    raw = json.loads(raw_json.read_text(encoding="utf-8"))
    energy = raw.get("TotalEnergy")
    if not isinstance(energy, (int, float)) or not math.isfinite(energy) or energy <= 0:
        raise ValueError("DRAMPower result has no positive TotalEnergy")
    return {
        "schema_version": "1.0",
        "evidence_type": "drampower_lpddr5_proxy",
        "memory_standard": "LPDDR5 public proxy; not paper LPDDR4X measurement",
        "metrics": {"offchip_energy_j": float(energy)},
        "assumptions": {"read_write_toggle_rate": 0.5, "data_payload_source": "toggle-rate model"},
        "artifacts": {
            "command_trace": relative_artifact(command_trace, output_dir, "output_dir"),
            "raw_result": relative_artifact(raw_json, output_dir, "output_dir"),
            "log": relative_artifact(log, output_dir, "output_dir"),
            "memspec": relative_artifact(memspec, output_dir, "output_dir"),
            "source_memspec": relative_artifact(
                source_memspec, root, "drampower_root"
            ),
            "config": relative_artifact(config, HERE.parents[1], "repo_root"),
            "executable": relative_artifact(executable, root, "drampower_root"),
        },
        "provenance": {
            "drampower_commit": commit,
            "dramutils_version": expected["dramutils_version"],
            "memspec_compatibility_fields": {
                field: 0 for field in LPDDR5_TIMING_COMPATIBILITY_FIELDS
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drampower-root", type=Path, required=True)
    parser.add_argument("--command-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = run(args.drampower_root.resolve(), args.command_trace.resolve(), args.output_dir.resolve())
        output = args.output_dir / "drampower.json"
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
