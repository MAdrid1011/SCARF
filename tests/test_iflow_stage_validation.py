from pathlib import Path


def test_stage_validation_requires_nonempty_outputs(tmp_path: Path):
    from hardware.iflow.validate_stages import validate_stage_outputs

    runtime = tmp_path / "iflow"
    synth = runtime / "result/ScarfTop.synth.yosys.asap7.HS.TYP.AE"
    floorplan = runtime / "result/ScarfTop.floorplan.openroad.asap7.HS.TYP.AE"
    synth.mkdir(parents=True)
    floorplan.mkdir(parents=True)
    (synth / "ScarfTop.v").write_text("module ScarfTop; endmodule\n")
    incomplete = validate_stage_outputs(runtime, "synth,floorplan")
    assert not incomplete["complete"]
    assert "floorplan:ScarfTop.def" in incomplete["missing"]
    assert "floorplan:ScarfTop.v" in incomplete["missing"]

    (floorplan / "ScarfTop.def").write_text("VERSION 5.8 ;\n")
    (floorplan / "ScarfTop.v").write_text("module ScarfTop; endmodule\n")
    complete = validate_stage_outputs(runtime, "synth,floorplan")
    assert complete["complete"]
    assert not complete["missing"]


def test_pdn_validation_requires_power_nets_and_all_macro_grids(tmp_path: Path):
    from hardware.iflow.validate_stages import SRAM_INSTANCES, validate_stage_outputs

    runtime = tmp_path / "iflow"
    pdn = runtime / "result/ScarfTop.pdn.openroad.asap7.HS.TYP.AE"
    pdn.mkdir(parents=True)
    (pdn / "ScarfTop.v").write_text("module ScarfTop; endmodule\n")
    (pdn / "ScarfTop.def").write_text(
        "VERSION 5.8 ;\n"
        "SPECIALNETS 2 ;\n"
        "    - VDD ( * VDD ) + USE POWER\n"
        "    - VSS ( * VSS ) + USE GROUND\n"
        "END SPECIALNETS\n",
        encoding="utf-8",
    )
    log_dir = runtime / "log"
    log_dir.mkdir()
    log = log_dir / "ScarfTop.pdn.openroad.asap7.HS.TYP.AE.log"
    log.write_text(
        "\n".join(f"[INFO PDN-0034] - grid for instance {name}" for name in SRAM_INSTANCES),
        encoding="utf-8",
    )

    record = validate_stage_outputs(runtime, "pdn")

    assert record["complete"]
    checks = record["stages"][0]["checks"]
    assert checks["declared_special_nets"] == 2
    assert all(checks["special_nets"].values())
    assert all(checks["macro_grids"].values())

    log.write_text("grid for instance featureBuf/bank0_ext\n", encoding="utf-8")
    incomplete = validate_stage_outputs(runtime, "pdn")
    assert not incomplete["complete"]
    assert "pdn:macro-grid:weightBuf/mem_ext" in incomplete["missing"]
