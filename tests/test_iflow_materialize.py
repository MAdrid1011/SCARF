from pathlib import Path


def test_materialize_creates_isolated_design_overlay(tmp_path):
    from hardware.iflow.materialize import materialize

    runtime = tmp_path / "iflow"
    template = runtime / "scripts" / "gcd"
    template.mkdir(parents=True)
    (runtime / "scripts" / "cfg").mkdir()
    (runtime / "scripts" / "cfg" / "flow_cfg.py").write_text("gcd = Flow('gcd','sky130','HS','TYP','')\n")
    (runtime / "scripts" / "cfg" / "foundry_cfg.py").write_text(
        "# asap7\n"
        "asap7 = Foundry(lib = {'macro,TYP' : (\n        ),}, "
        "lef = {'macro'     : (\n        )}, gds = {'macro'     : (\n        )})\n"
        "# SMIC110\n"
    )
    (template / "synth.yosys_0.9.tcl").write_text(
        'set CLOCK_PERIOD "20.0"\n'
        'set abc_script "+read_constr,$SDC_FILE;strash;map,{D};"\n'
        'set VERILOG_FILES " $RTL_PATH/gcd.v "\n'
        'abc -D [expr $CLOCK_PERIOD * 1000] \\\n'
        '    -constr "$SDC_FILE" \\\n'
        '    -script $abc_script\n'
    )
    (template / "floorplan.openroad_1.2.0.tcl").write_text(
        'set DIE_AREA "0 0 20 20"\n'
        'set CORE_AREA "1 1 19 19"\n'
        'set TRACKS_INFO_FILE    "$PROJ_PATH/foundry/$FOUNDRY/tracks_1.2.0.info"\n'
        '# pre report\n'
    )
    (template / "pdn_asap7.cfg").write_text(
        "pdngen::specify_grid stdcell { name top }\n"
    )
    for name in (
        "gplace.openroad_1.2.0.tcl",
        "resize.openroad_1.2.0.tcl",
        "cts.openroad_1.2.0.tcl",
        "groute.openroad_1.2.0.tcl",
    ):
        (template / name).write_text("set_wire_rc -layer $WIRE_RC_LAYER\n")
    rtl = tmp_path / "ScarfTop.sv"
    rtl.write_text(
        "module mem_1365x768(input R0_clk); reg [767:0] Memory[0:1364]; endmodule\n"
        "module bank_65536x16(input R0_clk); reg [15:0] Memory[0:65535]; endmodule\n"
        "module mem_16384x32(input R0_clk); reg [31:0] Memory[0:16383]; endmodule\n"
        "module ScarfTop(input clock); endmodule\n"
    )

    materialize(runtime, rtl)

    flow = (runtime / "scripts" / "cfg" / "flow_cfg.py").read_text()
    synth = (runtime / "scripts" / "ScarfTop" / "synth.yosys_0.9.tcl").read_text()
    floorplan = (runtime / "scripts" / "ScarfTop" / "floorplan.openroad_1.2.0.tcl").read_text()
    pdn = (runtime / "scripts" / "ScarfTop" / "pdn_asap7.cfg").read_text()
    assert "Flow('ScarfTop','asap7','HS','TYP','')" in flow
    assert "$RTL_PATH/ScarfTop.sv" in synth
    assert 'CLOCK_PERIOD            "1.0"' in synth
    assert "read_constr,$SDC_FILE" not in synth
    assert '-constr "$SDC_FILE"' not in synth
    assert 'set abc_script "+strash' in synth
    assert "CORE_UTILIZATION 45" in floorplan
    assert "placeScarfMacro featureBuf/bank0_ext LL" in floorplan
    assert "placeScarfMacro weightBuf/mem_ext UR" in floorplan
    assert "place_pins -random" in floorplan
    assert "set pdngen::global_connections" in pdn
    assert "pdngen::specify_grid macro" in pdn
    assert 'connect {{M5_PIN_ver M6}}' in pdn
    for name in (
        "gplace.openroad_1.2.0.tcl",
        "resize.openroad_1.2.0.tcl",
        "cts.openroad_1.2.0.tcl",
        "groute.openroad_1.2.0.tcl",
    ):
        stage_script = (runtime / "scripts" / "ScarfTop" / name).read_text()
        assert 'source "$PROJ_PATH/foundry/$FOUNDRY/setRC.tcl"' in stage_script
        assert stage_script.index("source ") < stage_script.index("set_wire_rc ")
    assert not (runtime / "scripts" / "gcd" / "ScarfTop.sdc").exists()
    assert (runtime / "rtl" / "ScarfTop" / "ScarfTop.sdc").is_file()
    sdc = (runtime / "rtl" / "ScarfTop" / "ScarfTop.sdc").read_text()
    assert "set_units -time ps" in sdc
    assert "create_clock -name scarf_clock -period 1000" in sdc
    assert "set_clock_uncertainty 50" in sdc
    assert "set_input_delay 100" in sdc
    assert "set_output_delay 100" in sdc
    synthesis_rtl = (runtime / "rtl" / "ScarfTop" / "ScarfTop.sv").read_text()
    assert synthesis_rtl.count("(* blackbox *)") == 3
    assert "reg [767:0] Memory" not in synthesis_rtl
    assert (runtime / "rtl" / "ScarfTop" / "sram-proxies.json").is_file()
    assert (runtime / "foundry" / "asap7" / "proxy" / "mem_16384x32.lef").is_file()
    proxy_lef = (runtime / "foundry" / "asap7" / "proxy" / "mem_16384x32.lef").read_text()
    assert "PIN VDD" in proxy_lef and "USE POWER" in proxy_lef
    assert "PIN VSS" in proxy_lef and "USE GROUND" in proxy_lef
