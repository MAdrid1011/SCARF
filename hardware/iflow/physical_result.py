#!/usr/bin/env python3
"""Parse routed iFlow/ASAP7 evidence into a strict PPA record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def last_float(patterns: tuple[str, ...], text: str) -> float | None:
    values = []
    for pattern in patterns:
        values.extend(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
    if not values:
        return None
    value = values[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return float(value)


def report_time_unit(text: str) -> str:
    matches = re.findall(r"^SCARF_TIME_UNIT\s+(ps|ns)\s*$", text, re.MULTILINE)
    return matches[-1] if matches else "ns"


def parse_ppa_report(text: str, clock_period_ns: float = 1.0) -> dict[str, float | None]:
    area = last_float(
        (
            r"Design area\s+([0-9.eE+-]+)",
            r"Total cell area:\s*([0-9.eE+-]+)",
        ),
        text,
    )
    utilization = last_float((r"([0-9.eE+-]+)%\s+utilization",), text)
    wns = last_float(
        (
            r"worst\s+slack\s+(-?[0-9.eE+]+)",
            r"report_wns[^\n]*\n(?:[^\n]*\n){0,3}?\s*(-?[0-9.eE+]+)\s*$",
            r"^wns\s+(-?[0-9.eE+]+)",
        ),
        text,
    )
    total_power = last_float(
        (
            r"^Total\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+([0-9.eE+-]+)\s*$",
            r"Total Power\s*=\s*([0-9.eE+-]+)",
        ),
        text,
    )
    power_row = re.findall(
        r"^Total\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    dynamic_power = static_power = None
    if power_row:
        internal, switching, leakage, total = map(float, power_row[-1])
        dynamic_power = internal + switching
        static_power = leakage
        total_power = total
    time_unit = report_time_unit(text)
    if wns is not None and time_unit == "ps":
        wns /= 1000.0
    delay = clock_period_ns - wns if wns is not None else None
    frequency = 1000.0 / delay if delay is not None and delay > 0 else None
    return {
        "area_um2": area,
        "utilization_pct": utilization,
        "wns_ns": wns,
        "critical_path_ns": delay,
        "max_frequency_mhz": frequency,
        "total_power_w": total_power,
        "dynamic_power_w": dynamic_power,
        "static_power_w": static_power,
    }


def parse_drc(text: str) -> int | None:
    matches = re.findall(
        r"(?:number of violations|total violations|drc violations)\s*[:=]\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return int(matches[-1])
    violations = re.findall(
        r"^\s*violation type\s*:", text, flags=re.IGNORECASE | re.MULTILINE
    )
    if violations:
        return len(violations)
    return 0 if not text.strip() else None


def find_last(runtime: Path, pattern: str) -> Path | None:
    matches = sorted(path for path in runtime.glob(pattern) if path.is_file())
    return matches[-1] if matches else None


def build_record(runtime: Path, manifest_path: Path) -> dict[str, Any]:
    runtime = runtime.resolve()
    output_dir = runtime.parent.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = runtime.parent.parent / "ppa-report.log"
    routed_def = find_last(runtime, "result/ScarfTop.droute.*/ScarfTop.def")
    routed_netlist = find_last(runtime, "result/ScarfTop.droute.*/ScarfTop.v")
    route_drc_report = find_last(runtime, "result/ScarfTop.droute.*/droute_drc.rpt")
    gds = find_last(runtime, "result/ScarfTop.layout.*/ScarfTop.gds")
    proxy_manifest = runtime / "rtl/ScarfTop/sram-proxies.json"
    iflow_log = runtime.parent.parent / "iflow.log"

    report_text = report.read_text(encoding="utf-8", errors="replace") if report.is_file() else ""
    log_text = iflow_log.read_text(encoding="utf-8", errors="replace") if iflow_log.is_file() else ""
    metrics = parse_ppa_report(report_text)
    route_drc_text = (
        route_drc_report.read_text(encoding="utf-8", errors="replace")
        if route_drc_report is not None
        else ""
    )
    drc = parse_drc(route_drc_text) if route_drc_report is not None else parse_drc(log_text)
    required_metrics = (
        "area_um2",
        "critical_path_ns",
        "max_frequency_mhz",
        "total_power_w",
        "dynamic_power_w",
        "static_power_w",
    )
    complete_metrics = all(
        isinstance(metrics[key], (int, float)) and math.isfinite(metrics[key]) and metrics[key] > 0
        for key in required_metrics
    )
    complete_files = all(
        path is not None and path.stat().st_size > 0 for path in (routed_def, routed_netlist, gds)
    )
    proxy_record = (
        json.loads(proxy_manifest.read_text(encoding="utf-8"))
        if proxy_manifest.is_file()
        else None
    )
    proxy_area_mm2 = proxy_record.get("total_proxy_area_mm2") if proxy_record else None
    total_area_mm2 = metrics.get("area_um2")
    if total_area_mm2 is not None:
        total_area_mm2 /= 1_000_000.0
    logic_area_mm2 = (
        total_area_mm2 - proxy_area_mm2
        if total_area_mm2 is not None and proxy_area_mm2 is not None
        else None
    )
    area_partition_valid = logic_area_mm2 is not None and logic_area_mm2 > 0
    physical_valid = bool(
        manifest.get("stages") and manifest["stages"][-1] == "layout"
        and complete_metrics
        and complete_files
        and route_drc_report is not None
        and proxy_manifest.is_file()
        and area_partition_valid
        and drc == 0
    )
    artifacts = {}
    for label, path in (
        ("routed_def", routed_def),
        ("routed_netlist", routed_netlist),
        ("gds", gds),
        ("route_drc_report", route_drc_report),
        ("ppa_report", report if report.is_file() else None),
        ("sram_proxy_manifest", proxy_manifest if proxy_manifest.is_file() else None),
    ):
        if path is not None:
            artifacts[label] = {
                "path": path.resolve().relative_to(output_dir).as_posix(),
                "path_base": "output_dir",
                "sha256": sha256_file(path),
            }
    metrics.pop("area_um2")
    return {
        "schema_version": "1.0",
        "evidence_type": "asap7_predictive_postroute",
        "physical_valid": physical_valid,
        "process": {"name": "ASAP7", "node_nm": 7, "predictive": True},
        "metrics": {
            "area_mm2": total_area_mm2,
            "logic_area_mm2": logic_area_mm2,
            "sram_proxy_area_mm2": proxy_area_mm2,
            **metrics,
            "drc_violations": drc,
        },
        "scope": {
            "logic_included": True,
            "sram_proxy_included": proxy_manifest.is_file(),
            "io_phy_included": False,
            "power_activity": "vectorless logic only",
            "sram_proxy_power_modeled": False,
            "drc_scope": "OpenROAD detailed-routing violations; not foundry signoff DRC",
        },
        "units": {
            "report_time": report_time_unit(report_text),
            "normalized_time": "ns",
            "area": "mm^2",
            "power": "W",
        },
        "artifacts": artifacts,
        "validation": {
            "complete_metrics": complete_metrics,
            "complete_routed_files": complete_files,
            "zero_drc": drc == 0,
            "route_drc_report_present": route_drc_report is not None,
            "area_partition_valid": area_partition_valid,
        },
        "provenance": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = build_record(args.runtime, args.manifest)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not record["physical_valid"]:
        print(f"error: incomplete or invalid physical evidence; inspect {args.output}")
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
