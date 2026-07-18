#!/usr/bin/env python3
"""Parse routed iFlow/ASAP7 evidence into a strict PPA record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.iflow.paper_table4 import (
    BUFFER_PROXY_COMPONENTS,
    REQUIRED_REPORT_COMPONENTS,
    TABLE4_AGGREGATES,
    TABLE4_COMPONENTS,
)
from scripts.mechanism_config import load_mechanism_config
from scripts.result_record import source_identity


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
    aggregate_text = text.split("SCARF_HIER_AREA", 1)[0]
    area = last_float(
        (
            r"Design area\s+([0-9.eE+-]+)",
            r"Total cell area:\s*([0-9.eE+-]+)",
        ),
        aggregate_text,
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
        aggregate_text,
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


def parse_hierarchy_report(text: str) -> dict[str, dict[str, float]]:
    areas = {
        name: float(value) / 1_000_000.0
        for name, value in re.findall(
            r"^SCARF_HIER_AREA\s+(\S+)\s+([0-9.eE+-]+)\s*$",
            text,
            flags=re.MULTILINE,
        )
    }
    powers = {}
    for name, block in re.findall(
        r"^SCARF_HIER_POWER_BEGIN\s+(\S+)\s*$\n(.*?)^SCARF_HIER_POWER_END\s+\1\s*$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    ):
        rows = re.findall(
            r"^Total\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if rows:
            internal, switching, leakage, total = map(float, rows[-1])
            powers[name] = {
                "dynamic_power_w": internal + switching,
                "static_power_w": leakage,
                "total_power_w": total,
            }
    return {
        name: {"area_mm2": areas[name], **powers[name]}
        for name in REQUIRED_REPORT_COMPONENTS
        if name in areas and name in powers
    }


def parse_hierarchy_instance_counts(text: str) -> dict[str, int]:
    """Read the routed leaf-instance bindings for each named paper component."""
    counts = {
        name: int(value)
        for name, value in re.findall(
            r"^SCARF_HIER_INSTANCE_COUNT\s+(\S+)\s+(\d+)\s*$",
            text,
            flags=re.MULTILINE,
        )
    }
    return {
        name: counts[name]
        for name in REQUIRED_REPORT_COMPONENTS
        if name in counts
    }


def parse_def_die_area_mm2(text: str) -> float | None:
    """Return the routed DEF die area in mm^2, or None for an invalid DEF."""
    units = re.findall(
        r"^\s*UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    dieareas = re.findall(
        r"^\s*DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*"
        r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not units or not dieareas:
        return None
    dbu = int(units[-1])
    x0, y0, x1, y1 = map(int, dieareas[-1])
    area_dbu2 = abs(x1 - x0) * abs(y1 - y0)
    if dbu <= 0 or area_dbu2 <= 0:
        return None
    return area_dbu2 / float(dbu * dbu * 1_000_000.0)


def _blank_component() -> dict[str, None]:
    return {
        "area_mm2": None,
        "dynamic_power_w": None,
        "static_power_w": None,
        "total_power_w": None,
    }


def _finite_component(component: dict[str, Any]) -> bool:
    return all(
        isinstance(component.get(field), (int, float))
        and math.isfinite(float(component[field]))
        and float(component[field]) >= 0
        for field in ("area_mm2", "dynamic_power_w", "static_power_w", "total_power_w")
    )


def _sum_components(
    components: dict[str, dict[str, float | None]], names: tuple[str, ...]
) -> dict[str, float]:
    fields = ("area_mm2", "dynamic_power_w", "static_power_w", "total_power_w")
    result: dict[str, float] = {}
    for field in fields:
        values = [components[name].get(field) for name in names]
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError(f"cannot aggregate incomplete Table 4 field: {field}")
        result[field] = sum(float(value) for value in values)
    return result


def _proxy_components(proxy_record: dict[str, Any] | None) -> dict[str, dict[str, float | None]]:
    components = {
        component: _blank_component() for component in BUFFER_PROXY_COMPONENTS.values()
    }
    if not isinstance(proxy_record, dict):
        return components
    for macro in proxy_record.get("macros", []):
        if not isinstance(macro, dict):
            continue
        component = BUFFER_PROXY_COMPONENTS.get(macro.get("name"))
        if component is None:
            continue
        per_instance = macro.get("area_per_instance_mm2")
        instances = macro.get("instances")
        if not isinstance(per_instance, (int, float)) or not isinstance(instances, int):
            continue
        components[component] = {
            "area_mm2": float(per_instance) * instances,
            "dynamic_power_w": None,
            "static_power_w": None,
            "total_power_w": None,
        }
    return components


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
    resource_validation = output_dir / "resource-validation.json"

    report_text = report.read_text(encoding="utf-8", errors="replace") if report.is_file() else ""
    log_text = iflow_log.read_text(encoding="utf-8", errors="replace") if iflow_log.is_file() else ""
    routed_def_text = (
        routed_def.read_text(encoding="utf-8", errors="replace")
        if routed_def is not None
        else ""
    )
    metrics = parse_ppa_report(report_text)
    hierarchy = parse_hierarchy_report(report_text)
    hierarchy_bindings = parse_hierarchy_instance_counts(report_text)
    die_area_mm2 = parse_def_die_area_mm2(routed_def_text)
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
    cell_area_mm2 = metrics.get("area_um2")
    if cell_area_mm2 is not None:
        cell_area_mm2 /= 1_000_000.0
    logic_area_mm2 = (
        cell_area_mm2 - proxy_area_mm2
        if cell_area_mm2 is not None and proxy_area_mm2 is not None
        else None
    )
    routing_filler_area_mm2 = (
        die_area_mm2 - cell_area_mm2
        if die_area_mm2 is not None and cell_area_mm2 is not None
        else None
    )
    reported_hierarchy_complete = (
        set(hierarchy) == set(REQUIRED_REPORT_COMPONENTS)
        and all(_finite_component(component) for component in hierarchy.values())
    )
    hierarchy_bindings_complete = (
        set(hierarchy_bindings) == set(REQUIRED_REPORT_COMPONENTS)
        and all(count > 0 for count in hierarchy_bindings.values())
    )
    table_hierarchy: dict[str, dict[str, float | None]] = {}
    proxy_components = _proxy_components(proxy_record)
    proxy_component_areas = [
        component["area_mm2"] for component in proxy_components.values()
    ]
    proxy_components_complete = all(
        isinstance(value, (int, float)) and value > 0
        for value in proxy_component_areas
    )
    proxy_area_valid = (
        isinstance(proxy_area_mm2, (int, float))
        and proxy_components_complete
        and math.isclose(
            float(proxy_area_mm2),
            sum(float(value) for value in proxy_component_areas),
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
    )
    area_partition_valid = (
        isinstance(logic_area_mm2, (int, float))
        and logic_area_mm2 > 0
        and proxy_area_valid
        and isinstance(die_area_mm2, (int, float))
        and die_area_mm2 > 0
        and isinstance(routing_filler_area_mm2, (int, float))
        and routing_filler_area_mm2 >= 0
    )
    hierarchy_complete = False
    if (
        reported_hierarchy_complete
        and hierarchy_bindings_complete
        and area_partition_valid
        and complete_metrics
    ):
        table_hierarchy = {
            name: dict(hierarchy[name]) for name in REQUIRED_REPORT_COMPONENTS
        }
        for aggregate, children in TABLE4_AGGREGATES.items():
            if aggregate != "on_chip_buffers":
                table_hierarchy[aggregate] = _sum_components(table_hierarchy, children)
        table_hierarchy.update(proxy_components)
        table_hierarchy["on_chip_buffers"] = {
            "area_mm2": float(proxy_area_mm2),
            "dynamic_power_w": None,
            "static_power_w": None,
            "total_power_w": None,
        }
        aggregate_names = ("mvu", "ggu_array", "fsdr_subsystem")
        known_area = sum(float(table_hierarchy[name]["area_mm2"]) for name in aggregate_names)
        known_dynamic = sum(
            float(table_hierarchy[name]["dynamic_power_w"]) for name in aggregate_names
        )
        known_static = sum(
            float(table_hierarchy[name]["static_power_w"]) for name in aggregate_names
        )
        control_area = float(logic_area_mm2) - known_area
        control_dynamic = float(metrics["dynamic_power_w"]) - known_dynamic
        control_static = float(metrics["static_power_w"]) - known_static
        if min(control_area, control_dynamic, control_static) >= 0:
            # Residual synthesized logic is exactly the paper's Control +
            # Interconnect row.  There is no separately synthesizable PLL
            # or commercial PHY in the public ASAP7 collateral.
            table_hierarchy["control_interconnect"] = {
                "area_mm2": control_area,
                "dynamic_power_w": control_dynamic,
                "static_power_w": control_static,
                "total_power_w": control_dynamic + control_static,
            }
            table_hierarchy["control_and_clock"] = _blank_component()
            table_hierarchy["pll_clock_tree"] = _blank_component()
            table_hierarchy["io_phy"] = _blank_component()
            table_hierarchy["io_lpddr4x_phy"] = _blank_component()
            table_hierarchy["routing_filler"] = {
                "area_mm2": float(routing_filler_area_mm2),
                "dynamic_power_w": None,
                "static_power_w": None,
                "total_power_w": None,
            }
            table_hierarchy["total_die"] = {
                "area_mm2": float(die_area_mm2),
                "dynamic_power_w": None,
                "static_power_w": None,
                "total_power_w": float(metrics["total_power_w"]),
            }
            hierarchy_complete = set(table_hierarchy) == set(TABLE4_COMPONENTS)
    physical_valid = bool(
        manifest.get("stages") and manifest["stages"][-1] == "layout"
        and complete_metrics
        and complete_files
        and route_drc_report is not None
        and proxy_manifest.is_file()
        and area_partition_valid
        and hierarchy_bindings_complete
        and hierarchy_complete
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
        ("resource_validation", resource_validation if resource_validation.is_file() else None),
    ):
        if path is not None:
            artifacts[label] = {
                "path": path.resolve().relative_to(output_dir).as_posix(),
                "path_base": "output_dir",
                "sha256": sha256_file(path),
            }
    metrics.pop("area_um2")
    host_resources = (
        json.loads(resource_validation.read_text(encoding="utf-8"))
        if resource_validation.is_file()
        else None
    )
    source = source_identity(ROOT)
    _, mechanism = load_mechanism_config()
    calibration_provenance = {
        key: value
        for key, value in mechanism.items()
        if key != "mechanism_config_sha256"
    }
    provenance = {
        **dict(manifest),
        "git_commit": source["git_commit"],
        "git_dirty": source["git_dirty"],
        "source_identity": source["source"],
        "source_tree_sha256": source["source_tree_sha256"],
        "submodules": source["submodules"],
        "mechanism_config_sha256": mechanism["mechanism_config_sha256"],
        "calibration_provenance": calibration_provenance,
    }
    if host_resources is not None:
        provenance["host_resources"] = host_resources
    return {
        "schema_version": "1.0",
        "evidence_type": "asap7_predictive_postroute",
        "physical_valid": physical_valid,
        "process": {"name": "ASAP7", "node_nm": 7, "predictive": True},
        "hierarchy_bindings": {
            "source": "routed DEF leaf-instance names matched by the Table 4 mapping",
            "instance_counts": hierarchy_bindings,
        },
        "metrics": {
            "area_mm2": die_area_mm2,
            "cell_area_mm2": cell_area_mm2,
            "logic_area_mm2": logic_area_mm2,
            "sram_proxy_area_mm2": proxy_area_mm2,
            "routing_filler_area_mm2": routing_filler_area_mm2,
            "hierarchy": table_hierarchy,
            **metrics,
            "drc_violations": drc,
        },
        "scope": {
            "logic_included": True,
            "sram_proxy_included": proxy_manifest.is_file(),
            "io_phy_included": False,
            "power_activity": "vectorless logic only",
            "sram_proxy_power_modeled": False,
            "pll_clock_tree_separately_modeled": False,
            "total_area_basis": "routed DEF DIEAREA; excludes commercial I/O/PHY macro",
            "routing_filler_area_basis": "routed DEF DIEAREA minus reported placed-cell area",
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
            "proxy_area_partition_valid": proxy_area_valid,
            "die_area_present": die_area_mm2 is not None,
            "reported_hierarchy_complete": reported_hierarchy_complete,
            "hierarchy_bindings_complete": hierarchy_bindings_complete,
            "hierarchy_complete": hierarchy_complete,
        },
        "provenance": provenance,
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
