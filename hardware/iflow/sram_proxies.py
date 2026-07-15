#!/usr/bin/env python3
"""Generate explicitly labeled abstract SRAM proxy views for ASAP7 runs."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


BITCELL_AREA_UM2 = 0.03
ARRAY_EFFICIENCY = 0.60


@dataclass(frozen=True)
class Port:
    name: str
    width: int
    direction: str
    clock: bool = False


@dataclass(frozen=True)
class Proxy:
    name: str
    depth: int
    width: int
    instances: int
    ports: tuple[Port, ...]

    @property
    def area_um2(self) -> float:
        return self.depth * self.width * BITCELL_AREA_UM2 / ARRAY_EFFICIENCY

    @property
    def side_um(self) -> float:
        return math.sqrt(self.area_um2)


PROXIES = (
    Proxy(
        "mem_1365x768",
        1365,
        768,
        1,
        (
            Port("R0_addr", 11, "input"), Port("R0_en", 1, "input"),
            Port("R0_clk", 1, "input", True), Port("R0_data", 768, "output"),
            Port("W0_addr", 11, "input"), Port("W0_en", 1, "input"),
            Port("W0_clk", 1, "input", True), Port("W0_data", 768, "input"),
        ),
    ),
    Proxy(
        "bank_65536x16",
        65536,
        16,
        2,
        (
            Port("R0_addr", 16, "input"), Port("R0_en", 1, "input"),
            Port("R0_clk", 1, "input", True), Port("R0_data", 16, "output"),
            Port("R1_addr", 16, "input"), Port("R1_en", 1, "input"),
            Port("R1_clk", 1, "input", True), Port("R1_data", 16, "output"),
            Port("W0_addr", 16, "input"), Port("W0_en", 1, "input"),
            Port("W0_clk", 1, "input", True), Port("W0_data", 16, "input"),
        ),
    ),
    Proxy(
        "mem_16384x32",
        16384,
        32,
        1,
        (
            Port("R0_addr", 14, "input"), Port("R0_en", 1, "input"),
            Port("R0_clk", 1, "input", True), Port("R0_data", 32, "output"),
            Port("W0_addr", 14, "input"), Port("W0_en", 1, "input"),
            Port("W0_clk", 1, "input", True), Port("W0_data", 32, "input"),
        ),
    ),
)


def blackbox_memories(systemverilog: str) -> str:
    result = systemverilog
    for proxy in PROXIES:
        pattern = re.compile(
            rf"(?:// VCS coverage exclude_file\s*)?module {re.escape(proxy.name)}\((.*?)\);.*?endmodule",
            flags=re.DOTALL,
        )
        match = pattern.search(result)
        if not match:
            raise ValueError(f"cannot find generated memory module {proxy.name}")
        declaration = f"(* blackbox *) module {proxy.name}({match.group(1)});\nendmodule"
        result = result[: match.start()] + declaration + result[match.end() :]
    if "reg [767:0] Memory" in result or "reg [31:0] Memory" in result:
        raise ValueError("generated SRAM arrays remain in synthesis RTL")
    return result


def liberty(proxy: Proxy) -> str:
    bus_widths = sorted({port.width for port in proxy.ports if port.width > 1})
    types = []
    for width in bus_widths:
        types.append(
            f"  type (bus{width}) {{ base_type : array; data_type : bit; "
            f"bit_width : {width}; bit_from : {width - 1}; bit_to : 0; downto : true; }}"
        )
    pins = []
    for port in proxy.ports:
        clock = " clock : true;" if port.clock else ""
        if port.width == 1:
            pins.append(f"    pin ({port.name}) {{ direction : {port.direction};{clock} }}")
        else:
            pins.append(
                f"    bus ({port.name}) {{ bus_type : bus{port.width}; "
                f"direction : {port.direction}; }}"
            )
    return "\n".join(
        (
            f"library (scarf_sram_proxy_{proxy.name}) {{",
            "  delay_model : table_lookup;",
            '  time_unit : "1ns";',
            '  voltage_unit : "1V";',
            '  current_unit : "1mA";',
            '  capacitive_load_unit (1,pf);',
            "  nom_process : 1.0;",
            "  nom_temperature : 25.0;",
            "  nom_voltage : 0.7;",
            "  input_threshold_pct_fall : 50.0;",
            "  input_threshold_pct_rise : 50.0;",
            "  output_threshold_pct_fall : 50.0;",
            "  output_threshold_pct_rise : 50.0;",
            "  slew_lower_threshold_pct_fall : 10.0;",
            "  slew_lower_threshold_pct_rise : 10.0;",
            "  slew_upper_threshold_pct_fall : 90.0;",
            "  slew_upper_threshold_pct_rise : 90.0;",
            *types,
            f"  cell ({proxy.name}) {{",
            f"    area : {proxy.area_um2:.6f};",
            "    dont_use : true;",
            "    pg_pin (VDD) { pg_type : primary_power; voltage_name : VDD; }",
            "    pg_pin (VSS) { pg_type : primary_ground; voltage_name : VSS; }",
            *pins,
            "  }",
            "}",
            "",
        )
    )


def lef(proxy: Proxy) -> str:
    side = proxy.side_um
    power_width = 0.120
    power_pins = []
    for name, use, center in (
        ("VDD", "POWER", side * 0.25),
        ("VSS", "GROUND", side * 0.75),
    ):
        x0 = center - power_width / 2
        x1 = center + power_width / 2
        power_pins.extend(
            (
                f"  PIN {name}",
                "    DIRECTION INOUT ;",
                f"    USE {use} ;",
                "    PORT",
                "      LAYER M5 ;",
                f"      RECT {x0:.6f} 0.000 {x1:.6f} {side:.6f} ;",
                "    END",
                f"  END {name}",
            )
        )
    pins = []
    names = []
    for port in proxy.ports:
        names.extend(
            [port.name] if port.width == 1 else [f"{port.name}[{index}]" for index in range(port.width)]
        )
    pitch = max(side / (len(names) + 1), 0.002)
    for index, name in enumerate(names, 1):
        y = min(index * pitch, side - 0.002)
        base_name = name.split("[", 1)[0]
        direction = next(port.direction.upper() for port in proxy.ports if port.name == base_name)
        pins.extend(
            (
                f"  PIN {name}",
                f"    DIRECTION {direction} ;",
                "    USE SIGNAL ;",
                "    PORT",
                "      LAYER M4 ;",
                f"      RECT 0.000 {y:.6f} 0.002 {min(y + 0.002, side):.6f} ;",
                "    END",
                f"  END {name}",
            )
        )
    return "\n".join(
        (
            "VERSION 5.8 ;",
            'BUSBITCHARS "[]" ;',
            'DIVIDERCHAR "/" ;',
            f"MACRO {proxy.name}",
            "  CLASS BLOCK ;",
            "  ORIGIN 0 0 ;",
            f"  SIZE {side:.6f} BY {side:.6f} ;",
            "  SYMMETRY X Y ;",
            "  SITE asap7sc7p5t ;",
            *power_pins,
            *pins,
            f"  OBS\n    LAYER M4 ;\n      RECT 0 0 {side:.6f} {side:.6f} ;\n  END",
            f"END {proxy.name}",
            "END LIBRARY",
            "",
        )
    )


def generate_gds(klayout: Path, proxy: Proxy, output: Path) -> None:
    script = output.with_suffix(".rb")
    script.write_text(
        "include RBA\n"
        "layout = Layout.new\n"
        "layout.dbu = 0.001\n"
        f'cell = layout.create_cell("{proxy.name}")\n'
        "layer = layout.layer(1, 0)\n"
        f"cell.shapes(layer).insert(Box.new(0, 0, {(proxy.side_um * 1000):.0f}, {(proxy.side_um * 1000):.0f}))\n"
        f'layout.write("{output}")\n',
        encoding="utf-8",
    )
    result = subprocess.run([str(klayout), "-b", "-r", str(script)], capture_output=True, text=True)
    if result.returncode or not output.is_file():
        raise RuntimeError(result.stderr or f"failed to generate {output}")


def generate_views(output_dir: Path, klayout: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for proxy in PROXIES:
        (output_dir / f"{proxy.name}.lib").write_text(liberty(proxy), encoding="utf-8")
        (output_dir / f"{proxy.name}.lef").write_text(lef(proxy), encoding="utf-8")
        generate_gds(klayout, proxy, output_dir / f"{proxy.name}.gds")
        records.append(
            {
                "name": proxy.name,
                "depth": proxy.depth,
                "width": proxy.width,
                "instances": proxy.instances,
                "area_per_instance_mm2": proxy.area_um2 / 1_000_000.0,
            }
        )
    return {
        "schema_version": "1.0",
        "evidence_type": "abstract_sram_proxy",
        "model": {
            "bitcell_area_um2": BITCELL_AREA_UM2,
            "array_efficiency": ARRAY_EFFICIENCY,
            "power_modeled": False,
            "foundry_compiler_output": False,
        },
        "macros": records,
        "total_proxy_area_mm2": sum(
            proxy.area_um2 * proxy.instances for proxy in PROXIES
        ) / 1_000_000.0,
    }


def patch_foundry_config(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index("# asap7")
    end = text.index("# SMIC110", start)
    section = text[start:end]
    libs = "\n".join(f"            'foundry/asap7/proxy/{p.name}.lib'," for p in PROXIES)
    lefs = "\n".join(f"            'foundry/asap7/proxy/{p.name}.lef'," for p in PROXIES)
    gds = "\n".join(f"            'foundry/asap7/proxy/{p.name}.gds'," for p in PROXIES)
    section = section.replace("'macro,TYP' : (\n        ),", f"'macro,TYP' : (\n{libs}\n        ),", 1)
    section = section.replace("'macro'     : (\n        )", f"'macro'     : (\n{lefs}\n        )", 1)
    section = section.replace("'macro'     : (\n        )", f"'macro'     : (\n{gds}\n        )", 1)
    if "foundry/asap7/proxy" not in section:
        raise ValueError("failed to add SRAM proxy views to ASAP7 configuration")
    path.write_text(text[:start] + section + text[end:], encoding="utf-8")
