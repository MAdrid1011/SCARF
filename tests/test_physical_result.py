import math
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_physical_result_script_runs_from_its_own_directory():
    result = subprocess.run(
        [sys.executable, str(ROOT / "hardware/iflow/physical_result.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_parse_openroad_ppa_report():
    from hardware.iflow.physical_result import parse_ppa_report

    text = """
Design area 250000.0 u^2 50.0% utilization.
wns 0.125
Total 1.0e-2 2.0e-2 3.0e-3 3.3e-2
"""
    metrics = parse_ppa_report(text)
    assert metrics["area_um2"] == 250000.0
    assert metrics["utilization_pct"] == 50.0
    assert metrics["wns_ns"] == 0.125
    assert metrics["critical_path_ns"] == 0.875
    assert math.isclose(metrics["max_frequency_mhz"], 1000 / 0.875)


def test_parse_openroad_ppa_report_normalizes_picoseconds():
    from hardware.iflow.physical_result import parse_ppa_report

    text = """SCARF_TIME_UNIT ps
report_wns
wns 125.0
Design area 12345 u^2 45% utilization.
Total 1.0e-2 2.0e-2 3.0e-3 3.3e-2
"""
    metrics = parse_ppa_report(text)

    assert math.isclose(metrics["wns_ns"], 0.125)
    assert math.isclose(metrics["critical_path_ns"], 0.875)
    assert math.isclose(metrics["max_frequency_mhz"], 1000 / 0.875)
    assert metrics["dynamic_power_w"] == 0.03
    assert metrics["static_power_w"] == 0.003
    assert metrics["total_power_w"] == 0.033


def test_missing_metrics_remain_missing():
    from hardware.iflow.physical_result import parse_ppa_report

    metrics = parse_ppa_report("route completed without reports")
    assert all(value is None for value in metrics.values())


def test_parse_hierarchical_area_power_and_instance_binding_markers():
    from hardware.iflow.physical_result import (
        parse_hierarchy_instance_counts,
        parse_hierarchy_report,
    )
    from hardware.iflow.paper_table4 import REQUIRED_REPORT_COMPONENTS

    blocks = []
    for index, name in enumerate(REQUIRED_REPORT_COMPONENTS, start=1):
        blocks.extend(
            (
                f"SCARF_HIER_AREA {name} {index * 1000}",
                f"SCARF_HIER_INSTANCE_COUNT {name} {index}",
                f"SCARF_HIER_POWER_BEGIN {name}",
                f"Total 0.001 0.002 0.0001 0.0031",
                f"SCARF_HIER_POWER_END {name}",
            )
        )
    hierarchy = parse_hierarchy_report("\n".join(blocks))

    assert hierarchy["mvu_mmcu"]["area_mm2"] == 0.002
    assert hierarchy["ggu_position_calc"]["dynamic_power_w"] == 0.003
    assert hierarchy["fsdr_cam_array"]["static_power_w"] == 0.0001
    assert parse_hierarchy_instance_counts("\n".join(blocks))["mvu_mmcu"] == 2
    assert "mvu_aux" not in hierarchy
    assert "control_buffer_logic" not in hierarchy


def test_parse_def_die_area_uses_def_units():
    from hardware.iflow.physical_result import parse_def_die_area_mm2

    assert parse_def_die_area_mm2(
        "UNITS DISTANCE MICRONS 1000 ;\nDIEAREA ( 0 0 ) ( 200000 300000 ) ;\n"
    ) == 0.06
    assert parse_def_die_area_mm2("DIEAREA ( 0 0 ) ( 1 1 ) ;\n") is None


def test_route_drc_parser_is_strict_about_unknown_nonempty_reports():
    from hardware.iflow.physical_result import parse_drc

    assert parse_drc("") == 0
    assert parse_drc("number of violations = 0\n") == 0
    assert parse_drc("number of violations = 17\n") == 17
    assert parse_drc("violation type: Short\nviolation type: MinArea\n") == 2
    assert parse_drc("route finished but report format is unknown\n") is None


def test_complete_routed_fixture_sets_physical_valid(tmp_path, monkeypatch):
    import hardware.iflow.physical_result as physical_result

    build_record = physical_result.build_record
    from hardware.iflow.paper_table4 import REQUIRED_REPORT_COMPONENTS

    identity = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "source": "git",
        "source_tree_sha256": "b" * 64,
        "submodules": {
            "transplat": "c" * 40,
            "mvsplat": "d" * 40,
            "depthsplat": "e" * 40,
        },
    }
    mechanism = {
        "status": "preregistered",
        "manifest_sha256": "f" * 64,
        "candidate_records_sha256": None,
        "evaluation_disjoint": False,
        "expected_results_accessed": False,
        "global_configuration": True,
        "mechanism_config_sha256": "1" * 64,
    }
    monkeypatch.setattr(physical_result, "source_identity", lambda *_: identity)
    monkeypatch.setattr(
        physical_result, "load_mechanism_config", lambda: ({}, mechanism)
    )

    output = tmp_path / "physical"
    runtime = output / "runtime/iflow"
    droute = runtime / "result/ScarfTop.droute.openroad.asap7.HS.TYP.AE"
    layout = runtime / "result/ScarfTop.layout.klayout.asap7.HS.TYP.AE"
    proxy_dir = runtime / "rtl/ScarfTop"
    droute.mkdir(parents=True)
    layout.mkdir(parents=True)
    proxy_dir.mkdir(parents=True)
    (droute / "ScarfTop.def").write_text(
        "VERSION 5.8 ;\nUNITS DISTANCE MICRONS 1000 ;\n"
        "DIEAREA ( 0 0 ) ( 600000 600000 ) ;\n",
        encoding="utf-8",
    )
    (droute / "ScarfTop.v").write_text("module ScarfTop; endmodule\n", encoding="utf-8")
    (droute / "droute_drc.rpt").write_text("", encoding="utf-8")
    (layout / "ScarfTop.gds").write_bytes(b"GDS")
    (proxy_dir / "sram-proxies.json").write_text(
        json.dumps(
            {
                "total_proxy_area_mm2": 0.05,
                "macros": [
                    {
                        "name": "mem_1365x768",
                        "instances": 1,
                        "area_per_instance_mm2": 0.01,
                    },
                    {
                        "name": "bank_65536x16",
                        "instances": 2,
                        "area_per_instance_mm2": 0.01,
                    },
                    {
                        "name": "mem_16384x32",
                        "instances": 1,
                        "area_per_instance_mm2": 0.02,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    hierarchy_report = "\n".join(
        f"SCARF_HIER_AREA {name} 10000\n"
        f"SCARF_HIER_INSTANCE_COUNT {name} 1\n"
        f"SCARF_HIER_POWER_BEGIN {name}\n"
        "Total 1.0e-3 2.0e-3 1.0e-4 3.1e-3\n"
        f"SCARF_HIER_POWER_END {name}"
        for name in REQUIRED_REPORT_COMPONENTS
    )
    (output / "ppa-report.log").write_text(
        "SCARF_TIME_UNIT ps\n"
        "wns -500\n"
        "Design area 250000 u^2 50% utilization.\n"
        "Total 5.0e-2 5.0e-2 1.0e-2 1.1e-1\n"
        + hierarchy_report
        + "\n",
        encoding="utf-8",
    )
    (output / "iflow.log").write_text("layout complete\n", encoding="utf-8")
    (output / "resource-validation.json").write_text(
        json.dumps(
            {
                "mode": "low_memory_attempt",
                "available_memory_threshold_met": False,
                "memory": {"mem_available_bytes": 12 * 1024**3},
            }
        ),
        encoding="utf-8",
    )
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"stages": ["droute", "layout"]}), encoding="utf-8")

    record = build_record(runtime, manifest)

    assert record["physical_valid"]
    assert record["metrics"]["wns_ns"] == -0.5
    assert record["metrics"]["drc_violations"] == 0
    assert record["validation"]["route_drc_report_present"]
    assert record["validation"]["hierarchy_bindings_complete"]
    assert record["validation"]["hierarchy_complete"]
    assert "mvu_mmcu" in record["metrics"]["hierarchy"]
    assert "on_chip_buffers_feature_buffer" in record["metrics"]["hierarchy"]
    assert "io_lpddr4x_phy" in record["metrics"]["hierarchy"]
    assert record["metrics"]["hierarchy"]["io_lpddr4x_phy"]["area_mm2"] is None
    assert record["metrics"]["hierarchy"]["total_die"]["area_mm2"] == 0.36
    assert record["metrics"]["hierarchy"]["total_die"]["total_power_w"] == 0.11
    assert "other_logic" not in record["metrics"]["hierarchy"]
    assert record["hierarchy_bindings"]["instance_counts"]["ggu_position_calc"] == 1
    assert "route_drc_report" in record["artifacts"]
    assert "resource_validation" in record["artifacts"]
    assert record["artifacts"]["routed_def"]["path"].startswith("runtime/iflow/")
    assert record["artifacts"]["routed_def"]["path_base"] == "output_dir"
    assert record["provenance"]["host_resources"]["mode"] == "low_memory_attempt"
    assert record["provenance"]["source_tree_sha256"] == identity["source_tree_sha256"]
    assert (
        record["provenance"]["mechanism_config_sha256"]
        == mechanism["mechanism_config_sha256"]
    )
    assert record["provenance"]["calibration_provenance"]["status"] == "preregistered"
