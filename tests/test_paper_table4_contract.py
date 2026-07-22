import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_released_table4_rows_match_the_public_contract_in_order():
    from hardware.iflow.paper_table4 import TABLE4_COMPONENTS, TABLE4_LABELS

    expected = json.loads((ROOT / "artifact/expected_results.json").read_text())
    hierarchy = expected["hardware_comparison_target"]["hierarchy"]
    public_labels = (
        "MVU",
        "MMCU",
        "VectorALU",
        "BilinearUnit",
        "NormUnit",
        "ActivationUnit",
        "GGU Array (x32 PEs)",
        "PositionCalc",
        "CovBuilder",
        "SH_OPGenerator",
        "FSDR Subsystem",
        "LSHHashUnit",
        "CAM Array (32-entry)",
        "FSDR Controller",
        "On-chip Buffers",
        "Weight Buffer (128 KB)",
        "Feature Buffer (256 KB)",
        "Tile Buffer (64 KB)",
        "Control & Clock",
        "Control + Interconnect",
        "PLL + Clock tree",
        "I/O & PHY",
        "I/O + LPDDR4X PHY",
        "Routing / filler",
        "Total die",
    )

    assert tuple(hierarchy) == TABLE4_COMPONENTS
    assert tuple(TABLE4_LABELS[component] for component in TABLE4_COMPONENTS) == public_labels


def test_paper_table4_groups_and_targets_are_internally_consistent():
    from hardware.iflow.paper_table4 import PAPER_TABLE4_GROUPS

    expected = json.loads((ROOT / "artifact/expected_results.json").read_text(encoding="utf-8"))
    hierarchy = expected["hardware_comparison_target"]["hierarchy"]

    for parent, children in PAPER_TABLE4_GROUPS.items():
        for metric in ("area_mm2", "dynamic_power_w", "static_power_w"):
            assert hierarchy[parent][metric] == pytest.approx(
                sum(hierarchy[child][metric] for child in children)
            )

    logic_and_memory_groups = (
        "mvu",
        "ggu_array",
        "fsdr_subsystem",
        "on_chip_buffers",
        "control_and_clock",
        "io_phy",
    )
    assert hierarchy["total_die"]["total_power_w"] == pytest.approx(
        sum(
            hierarchy[group]["dynamic_power_w"] + hierarchy[group]["static_power_w"]
            for group in logic_and_memory_groups
        )
    )
    assert hierarchy["total_die"]["area_mm2"] == pytest.approx(
        sum(hierarchy[group]["area_mm2"] for group in logic_and_memory_groups)
        + hierarchy["routing_filler"]["area_mm2"],
        abs=0.03,
    )


def test_rtl_instance_names_implement_every_named_paper_component():
    from hardware.iflow.paper_table4 import REPORT_COMPONENTS

    top = (ROOT / "chisel/src/main/scala/scarf/ScarfTop.scala").read_text(encoding="utf-8")
    ggu = (ROOT / "chisel/src/main/scala/scarf/ggu/GGUArray.scala").read_text(encoding="utf-8")
    config = (ROOT / "chisel/src/main/scala/scarf/Config.scala").read_text(encoding="utf-8")

    expected_instances = {
        "mvu_mmcu": "val mmcu       = Module(new MMCU",
        "mvu_vector_alu": "val vectorALU  = Module(new VectorALU",
        "mvu_bilinear_unit": "val bilinear   = Module(new BilinearUnit",
        "mvu_norm_unit": "val normUnit   = Module(new NormUnit)",
        "mvu_activation_unit": "val activation = Module(new ActivationUnit",
        "ggu_array": "val gguArray   = Module(new GGUArray",
        "fsdr_lsh_hash_unit": "val lshHash    = Module(new LSHHashUnit",
        "fsdr_cam_array": "val fsdrCache  = Module(new FSDRCache",
        "fsdr_controller": "val fsdrCtrl   = Module(new FSDRController)",
        "control_interconnect": "val dramIF     = Module(new DRAMInterface)",
    }
    for component, instance in expected_instances.items():
        assert instance in top, component
        assert component in REPORT_COMPONENTS

    assert "val GGUPECount: Int       = 32" in config
    assert "val pes = Seq.fill(numPEs)(Module(new GGUPE))" in ggu
    assert "val posCalc = Module(new PositionCalc)" in ggu
    assert "val covBld  = Module(new CovBuilder)" in ggu
    assert "val shOpGen = Module(new SHOPGenerator)" in ggu
    assert "val WeightBufferBytes: Int    = 128 * 1024" in config
    assert "val FeatureBufferBytes: Int   = 256 * 1024" in config
    assert "val TileBufferBytes: Int      = 64 * 1024" in config
