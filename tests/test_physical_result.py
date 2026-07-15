import math
import json


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


def test_route_drc_parser_is_strict_about_unknown_nonempty_reports():
    from hardware.iflow.physical_result import parse_drc

    assert parse_drc("") == 0
    assert parse_drc("number of violations = 0\n") == 0
    assert parse_drc("number of violations = 17\n") == 17
    assert parse_drc("violation type: Short\nviolation type: MinArea\n") == 2
    assert parse_drc("route finished but report format is unknown\n") is None


def test_complete_routed_fixture_sets_physical_valid(tmp_path):
    from hardware.iflow.physical_result import build_record

    output = tmp_path / "physical"
    runtime = output / "runtime/iflow"
    droute = runtime / "result/ScarfTop.droute.openroad.asap7.HS.TYP.AE"
    layout = runtime / "result/ScarfTop.layout.klayout.asap7.HS.TYP.AE"
    proxy_dir = runtime / "rtl/ScarfTop"
    droute.mkdir(parents=True)
    layout.mkdir(parents=True)
    proxy_dir.mkdir(parents=True)
    (droute / "ScarfTop.def").write_text("VERSION 5.8 ;\n", encoding="utf-8")
    (droute / "ScarfTop.v").write_text("module ScarfTop; endmodule\n", encoding="utf-8")
    (droute / "droute_drc.rpt").write_text("", encoding="utf-8")
    (layout / "ScarfTop.gds").write_bytes(b"GDS")
    (proxy_dir / "sram-proxies.json").write_text(
        json.dumps({"total_proxy_area_mm2": 0.05}), encoding="utf-8"
    )
    (output / "ppa-report.log").write_text(
        "SCARF_TIME_UNIT ps\n"
        "wns -500\n"
        "Design area 250000 u^2 50% utilization.\n"
        "Total 1.0e-2 2.0e-2 3.0e-3 3.3e-2\n",
        encoding="utf-8",
    )
    (output / "iflow.log").write_text("layout complete\n", encoding="utf-8")
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"stages": ["droute", "layout"]}), encoding="utf-8")

    record = build_record(runtime, manifest)

    assert record["physical_valid"]
    assert record["metrics"]["wns_ns"] == -0.5
    assert record["metrics"]["drc_violations"] == 0
    assert record["validation"]["route_drc_report_present"]
    assert "route_drc_report" in record["artifacts"]
    assert record["artifacts"]["routed_def"]["path"].startswith("runtime/iflow/")
    assert record["artifacts"]["routed_def"]["path_base"] == "output_dir"
