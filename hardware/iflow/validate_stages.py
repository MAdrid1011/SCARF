#!/usr/bin/env python3
"""Reject iFlow runs that did not materialize every requested stage output."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.iflow.preflight import parse_stages


DESIGN = "ScarfTop"
SRAM_INSTANCES = (
    "featureBuf/bank0_ext",
    "featureBuf/bank1_ext",
    "tileBuf/mem_ext",
    "weightBuf/mem_ext",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pdn(runtime: Path, pdn_def: Path) -> tuple[dict, list[str]]:
    checks = {
        "special_nets": {"VDD": False, "VSS": False},
        "macro_grids": {instance: False for instance in SRAM_INSTANCES},
    }
    failures = []
    in_special_nets = False
    declared_special_nets = 0
    with pdn_def.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("SPECIALNETS "):
                in_special_nets = True
                try:
                    declared_special_nets = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
            elif in_special_nets and line.startswith("END SPECIALNETS"):
                break
            elif in_special_nets:
                stripped = line.strip()
                if stripped.startswith("- VDD ") and "+ USE POWER" in stripped:
                    checks["special_nets"]["VDD"] = True
                if stripped.startswith("- VSS ") and "+ USE GROUND" in stripped:
                    checks["special_nets"]["VSS"] = True
    checks["declared_special_nets"] = declared_special_nets
    if declared_special_nets < 2 or not all(checks["special_nets"].values()):
        failures.append("pdn:VDD/VSS-special-nets")

    logs = sorted((runtime / "log").glob("ScarfTop.pdn.*.AE.log"))
    if not logs:
        failures.append("pdn:log")
        return checks, failures
    log = logs[-1]
    log_text = log.read_text(encoding="utf-8", errors="replace")
    for instance in SRAM_INSTANCES:
        marker = f"grid for instance {instance}"
        checks["macro_grids"][instance] = marker in log_text
        if marker not in log_text:
            failures.append(f"pdn:macro-grid:{instance}")
    checks["log"] = {
        "path": str(log.relative_to(runtime)),
        "path_base": "runtime_root",
        "sha256": sha256_file(log),
    }
    return checks, failures


def validate_stage_outputs(runtime: Path, stage_spec: str) -> dict:
    runtime = runtime.resolve()
    stages = parse_stages(stage_spec)
    records = []
    missing = []
    for stage in stages:
        directories = sorted(
            path
            for path in (runtime / "result").glob(f"ScarfTop.{stage}.*.AE")
            if path.is_dir()
        )
        directory = directories[-1] if directories else None
        extensions = (
            ("gds",)
            if stage == "layout"
            else (("v",) if stage == "synth" else ("def", "v"))
        )
        artifacts = {}
        if directory is not None:
            for extension in extensions:
                path = directory / f"ScarfTop.{extension}"
                if path.is_file() and path.stat().st_size > 0:
                    artifacts[extension] = {
                        "path": str(path.relative_to(runtime)),
                        "path_base": "runtime_root",
                        "sha256": sha256_file(path),
                    }
                else:
                    missing.append(f"{stage}:ScarfTop.{extension}")
        else:
            missing.append(f"{stage}:result-directory")
        record = {"stage": stage, "artifacts": artifacts}
        if stage == "pdn" and directory is not None and "def" in artifacts:
            checks, pdn_failures = validate_pdn(runtime, directory / f"{DESIGN}.def")
            record["checks"] = checks
            missing.extend(pdn_failures)
        records.append(record)
    return {
        "schema_version": "1.0",
        "design": "ScarfTop",
        "requested_stages": list(stages),
        "stages": records,
        "complete": not missing,
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = validate_stage_outputs(args.runtime, args.stage)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not record["complete"]:
        print("error: incomplete iFlow stages: " + ", ".join(record["missing"]))
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
