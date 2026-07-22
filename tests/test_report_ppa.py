from pathlib import Path


def test_final_ppa_report_loads_asap7_rc_before_wire_model(tmp_path: Path):
    from hardware.iflow.report_ppa import build_tcl

    runtime = tmp_path / "runtime"
    final_def = runtime / "result/ScarfTop.droute.test/ScarfTop.def"
    final_def.parent.mkdir(parents=True)
    final_def.write_text("VERSION 5.8 ;\n", encoding="utf-8")

    text = build_tcl(runtime, final_def)

    source = f"source {runtime / 'foundry/asap7/setRC.tcl'}"
    assert source in text
    assert text.index(source) < text.index("set_wire_rc -layer M3")
    assert 'puts "SCARF_TIME_UNIT ps"' in text
    for component in (
        "mvu_mmcu",
        "ggu_position_calc",
        "fsdr_cam_array",
        "control_interconnect",
    ):
        assert f"SCARF_HIER_AREA $label" in text
        assert "SCARF_HIER_INSTANCE_COUNT $label $matched_instances" in text
        assert component in text
    assert "mvu_aux" not in text
    assert "control_buffer_logic" not in text
