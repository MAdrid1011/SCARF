#!/usr/bin/env python3
"""Run a final OpenROAD timing, area, and vectorless-power report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.iflow.sram_proxies import PROXIES
from hardware.iflow.paper_table4 import REPORT_COMPONENTS


LIBRARIES = (
    "asap7sc7p5t_AO_RVT_TT_nldm_201020.lib",
    "asap7sc7p5t_INVBUF_RVT_TT_nldm_201020.lib",
    "asap7sc7p5t_OA_RVT_TT_nldm_201020.lib",
    "asap7sc7p5t_SEQ_RVT_TT_nldm_201020.lib",
    "asap7sc7p5t_SIMPLE_RVT_TT_nldm_201020.lib",
)

def one_match(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"missing {label}: {pattern}")
    return matches[-1]


def build_tcl(runtime: Path, final_def: Path, path_root: Path | None = None) -> str:
    file_root = path_root or runtime
    final_def_for_tcl = file_root / final_def.relative_to(runtime)
    tech_lef = file_root / "foundry/asap7/lef/asap7_tech_1x_201209.lef"
    cell_lef = file_root / "foundry/asap7/lef/asap7sc7p5t_27_R_1x_201211.lef"
    sdc = file_root / "rtl/ScarfTop/ScarfTop.sdc"
    rc = file_root / "foundry/asap7/setRC.tcl"
    lines = [
        f"read_lef {tech_lef}",
        f"read_lef {cell_lef}",
    ]
    lines.extend(
        f"read_lef {file_root / 'foundry/asap7/proxy' / (proxy.name + '.lef')}"
        for proxy in PROXIES
    )
    lines.extend(f"read_liberty {file_root / 'foundry/asap7/lib' / name}" for name in LIBRARIES)
    lines.extend(
        f"read_liberty {file_root / 'foundry/asap7/proxy' / (proxy.name + '.lib')}"
        for proxy in PROXIES
    )
    hierarchy_groups = " ".join(
        "{" + label + " {" + " ".join(patterns) + "}}"
        for label, patterns in REPORT_COMPONENTS.items()
    )
    lines.extend(
        (
            f"read_def {final_def_for_tcl}",
            f"read_sdc {sdc}",
            f"source {rc}",
            "set_wire_rc -layer M3",
            'puts "SCARF_PPA_BEGIN"',
            'puts "SCARF_TIME_UNIT ps"',
            "report_checks -path_delay max -format full_clock_expanded",
            "report_wns",
            "report_tns",
            "report_design_area",
            "report_power",
            f"set scarf_hierarchy_groups {{{hierarchy_groups}}}",
            "set scarf_block [[[[::ord::get_db] getChip] getBlock] getInsts]",
            "set scarf_dbu [[[::ord::get_db] getTech] getDbUnitsPerMicron]",
            "foreach group $scarf_hierarchy_groups {",
            "  set label [lindex $group 0]",
            "  set patterns [lindex $group 1]",
            "  set area_dbu2 0",
            "  set matched_instances 0",
            "  foreach inst $scarf_block {",
            "    set name [$inst getName]",
            "    set matched 0",
            "    foreach pattern $patterns { if {[string match $pattern $name]} { set matched 1 } }",
            "    if {$matched} {",
            "      incr matched_instances",
            "      set master [$inst getMaster]",
            "      if {[lsearch -exact {mem_1365x768 bank_65536x16 mem_16384x32} [$master getName]] < 0} {",
            "        set area_dbu2 [expr {$area_dbu2 + ([$master getWidth] * [$master getHeight])}]",
            "      }",
            "    }",
            "  }",
            "  puts \"SCARF_HIER_AREA $label [expr {double($area_dbu2) / ($scarf_dbu * $scarf_dbu)}]\"",
            "  puts \"SCARF_HIER_INSTANCE_COUNT $label $matched_instances\"",
            "  set cells [get_cells -hierarchical -quiet $patterns]",
            "  puts \"SCARF_HIER_POWER_BEGIN $label\"",
            "  if {[sizeof_collection $cells] > 0} { report_power -instances $cells }",
            "  puts \"SCARF_HIER_POWER_END $label\"",
            "}",
            'puts "SCARF_PPA_END"',
            "exit",
        )
    )
    return "\n".join(lines) + "\n"


def run_report(
    runtime: Path,
    output: Path,
    *,
    path_root: Path | None = None,
    prepare_only: bool = False,
) -> Path:
    runtime = runtime.resolve()
    final_def = one_match(
        runtime, "result/ScarfTop.droute.openroad_1.2.0.asap7.HS.TYP.*/ScarfTop.def", "routed DEF"
    )
    tcl = output.with_suffix(".tcl")
    tcl.write_text(build_tcl(runtime, final_def, path_root), encoding="utf-8")
    if prepare_only:
        return tcl
    openroad = runtime / "tools/OpenROADae191807/bin/openroad"
    if not openroad.is_file():
        raise FileNotFoundError(f"OpenROAD executable not found: {openroad}")
    with output.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [str(openroad), str(tcl)], cwd=runtime, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"OpenROAD PPA report failed; inspect {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run_report(
        args.runtime,
        args.output,
        path_root=args.path_root,
        prepare_only=args.prepare_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
