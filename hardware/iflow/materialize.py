#!/usr/bin/env python3
"""Materialize the SCARF design overlay in an isolated iFlow worktree."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.iflow.sram_proxies import (
    blackbox_memories,
    generate_views,
    patch_foundry_config,
)


DESIGN = "ScarfTop"
RC_AWARE_STAGES = (
    "gplace.openroad_1.2.0.tcl",
    "resize.openroad_1.2.0.tcl",
    "cts.openroad_1.2.0.tcl",
    "groute.openroad_1.2.0.tcl",
)


def patch_asap7_rc(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    anchor = "set_wire_rc -layer $WIRE_RC_LAYER"
    if anchor not in text:
        raise ValueError(f"failed to locate wire RC anchor in {script.name}")
    source = 'source "$PROJ_PATH/foundry/$FOUNDRY/setRC.tcl"'
    if source in text:
        raise ValueError(f"ASAP7 RC source is already present in {script.name}")
    script.write_text(text.replace(anchor, source + "\n" + anchor, 1), encoding="utf-8")


def materialize(runtime: Path, rtl: Path) -> None:
    runtime = runtime.resolve()
    rtl = rtl.resolve()
    if not rtl.is_file():
        raise FileNotFoundError(f"emitted SystemVerilog not found: {rtl}")
    template_dir = runtime / "scripts" / "gcd"
    if not template_dir.is_dir():
        raise FileNotFoundError(f"iFlow gcd template is missing: {template_dir}")

    design_scripts = runtime / "scripts" / DESIGN
    if design_scripts.exists():
        raise FileExistsError(f"runtime design overlay already exists: {design_scripts}")
    shutil.copytree(template_dir, design_scripts)

    for script_name in RC_AWARE_STAGES:
        patch_asap7_rc(design_scripts / script_name)

    flow_cfg = runtime / "scripts" / "cfg" / "flow_cfg.py"
    flow_text = flow_cfg.read_text(encoding="utf-8")
    declaration = "scarf_top       = Flow('ScarfTop','asap7','HS','TYP','')"
    if "Flow('ScarfTop'" not in flow_text:
        flow_cfg.write_text(flow_text.rstrip() + "\n" + declaration + "\n", encoding="utf-8")

    synth = design_scripts / "synth.yosys_0.9.tcl"
    synth_text = synth.read_text(encoding="utf-8")
    synth_text = synth_text.replace("$RTL_PATH/gcd.v", "$RTL_PATH/ScarfTop.sv")
    synth_text = re.sub(
        r'set CLOCK_PERIOD\s+"[^"]+"', 'set CLOCK_PERIOD            "1.0"', synth_text
    )
    # ABC's read_constr format is not SDC. Timing-driven mapping still uses
    # CLOCK_PERIOD through the explicit `abc -D` argument below this script.
    synth_text = synth_text.replace("+read_constr,$SDC_FILE;", "+")
    synth_text = re.sub(
        r'(?m)^[ \t]*-constr[ \t]+"\$SDC_FILE"[ \t]+\\[ \t]*\r?\n',
        "",
        synth_text,
        count=1,
    )
    if "read_constr,$SDC_FILE" in synth_text:
        raise ValueError("failed to remove incompatible ABC constraint input")
    if '-constr "$SDC_FILE"' in synth_text:
        raise ValueError("failed to remove incompatible Yosys ABC constraint input")
    if "$RTL_PATH/ScarfTop.sv" not in synth_text:
        raise ValueError("failed to update iFlow synthesis RTL input")
    synth.write_text(synth_text, encoding="utf-8")

    floorplan = design_scripts / "floorplan.openroad_1.2.0.tcl"
    floorplan_text = floorplan.read_text(encoding="utf-8")
    floorplan_text = re.sub(
        r'(?m)^set (DIE_AREA|CORE_AREA)\s+"[^"]+"\s*$', r'# SCARF uses utilization-based floorplanning', floorplan_text
    )
    anchor = 'set TRACKS_INFO_FILE    "$PROJ_PATH/foundry/$FOUNDRY/tracks_1.2.0.info"'
    settings = "\n".join(
        (
            "set CORE_UTILIZATION 45",
            "set CORE_ASPECT_RATIO 1.0",
            "set CORE_MARGIN 10",
        )
    )
    if anchor not in floorplan_text:
        raise ValueError("failed to locate iFlow floorplan configuration anchor")
    placement_anchor = "# pre report"
    placement_commands = "\n".join(
        (
            "# Place the four abstract SRAM proxies deterministically at core corners.",
            "proc placeScarfMacro {inst_name corner} {",
            "    set db [::ord::get_db]",
            "    set block [[$db getChip] getBlock]",
            "    set inst [$block findInst $inst_name]",
            "    if {$inst == \"NULL\"} { error \"missing SRAM proxy instance $inst_name\" }",
            "    set master [$inst getMaster]",
            "    set die [$block getDieArea]",
            "    set margin [expr 10 * [[$db getTech] getDbUnitsPerMicron]]",
            "    set left [expr [$die xMin] + $margin]",
            "    set bottom [expr [$die yMin] + $margin]",
            "    set right [expr [$die xMax] - $margin - [$master getWidth]]",
            "    set top [expr [$die yMax] - $margin - [$master getHeight]]",
            "    if {$corner == \"LL\"} { set x $left; set y $bottom }",
            "    if {$corner == \"LR\"} { set x $right; set y $bottom }",
            "    if {$corner == \"UL\"} { set x $left; set y $top }",
            "    if {$corner == \"UR\"} { set x $right; set y $top }",
            "    $inst setOrigin $x $y",
            "    $inst setOrient R0",
            "    $inst setPlacementStatus FIRM",
            "}",
            "placeScarfMacro featureBuf/bank0_ext LL",
            "placeScarfMacro featureBuf/bank1_ext LR",
            "placeScarfMacro tileBuf/mem_ext UL",
            "placeScarfMacro weightBuf/mem_ext UR",
            "place_pins -random \\",
            "           -hor_layer $IO_H_LAYER \\",
            "           -ver_layer $IO_V_LAYER",
        )
    )
    if placement_anchor not in floorplan_text:
        raise ValueError("failed to locate iFlow floorplan placement anchor")
    floorplan.write_text(
        floorplan_text.replace(anchor, settings + "\n" + anchor, 1).replace(
            placement_anchor,
            placement_commands + "\n\n" + placement_anchor,
            1,
        ),
        encoding="utf-8",
    )

    pdn = design_scripts / "pdn_asap7.cfg"
    pdn_text = pdn.read_text(encoding="utf-8")
    pdn_overlay = """

# SCARF abstract SRAM proxy power connectivity.
set pdngen::global_connections {
  VDD {{inst_name .* pin_name ^VDD$}}
  VSS {{inst_name .* pin_name ^VSS$}}
}

pdngen::specify_grid macro {
    orient {R0}
    power_pins "VDD"
    ground_pins "VSS"
    blockages "M1 M2 M3 M4"
    connect {{M5_PIN_ver M6}}
}
"""
    if "SCARF abstract SRAM proxy power connectivity" in pdn_text:
        raise ValueError("SCARF macro PDN overlay is already present")
    pdn.write_text(pdn_text.rstrip() + pdn_overlay, encoding="utf-8")

    rtl_dir = runtime / "rtl" / DESIGN
    rtl_dir.mkdir(parents=True, exist_ok=False)
    synthesis_rtl = blackbox_memories(rtl.read_text(encoding="utf-8"))
    (rtl_dir / "ScarfTop.sv").write_text(synthesis_rtl, encoding="utf-8")
    shutil.copy2(
        Path(__file__).resolve().parent / "templates" / "ScarfTop.sdc",
        rtl_dir / "ScarfTop.sdc",
    )
    proxy_dir = runtime / "foundry" / "asap7" / "proxy"
    proxy_manifest = generate_views(proxy_dir, Path("/usr/bin/klayout"))
    (rtl_dir / "sram-proxies.json").write_text(
        json.dumps(proxy_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    patch_foundry_config(runtime / "scripts" / "cfg" / "foundry_cfg.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--rtl", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.runtime, args.rtl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
