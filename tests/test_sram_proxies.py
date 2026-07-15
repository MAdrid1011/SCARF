import math


def test_blackbox_replacement_removes_behavioral_arrays():
    from hardware.iflow.sram_proxies import PROXIES, blackbox_memories

    source = "\n".join(
        f"module {proxy.name}(input R0_clk); reg [{proxy.width - 1}:0] Memory[0:{proxy.depth - 1}]; endmodule"
        for proxy in PROXIES
    )
    result = blackbox_memories(source)
    assert result.count("(* blackbox *)") == len(PROXIES)
    assert " Memory[" not in result


def test_proxy_area_is_separate_and_deterministic():
    from hardware.iflow.sram_proxies import (
        ARRAY_EFFICIENCY,
        BITCELL_AREA_UM2,
        PROXIES,
    )

    expected_um2 = sum(
        proxy.depth * proxy.width * proxy.instances * BITCELL_AREA_UM2 / ARRAY_EFFICIENCY
        for proxy in PROXIES
    )
    assert math.isclose(expected_um2, 183488.0)


def test_proxy_liberty_has_required_sta_thresholds():
    from hardware.iflow.sram_proxies import PROXIES, liberty

    text = liberty(PROXIES[0])
    assert "input_threshold_pct_rise : 50.0" in text
    assert "output_threshold_pct_fall : 50.0" in text
    assert "slew_lower_threshold_pct_rise : 10.0" in text
    assert "slew_upper_threshold_pct_fall : 90.0" in text
    assert "dont_use : true" in text
    assert "pg_pin (VDD)" in text
    assert "pg_pin (VSS)" in text


def test_proxy_lef_has_abstract_power_stripes():
    from hardware.iflow.sram_proxies import PROXIES, lef

    text = lef(PROXIES[0])
    assert "PIN VDD" in text
    assert "USE POWER" in text
    assert "PIN VSS" in text
    assert "USE GROUND" in text
    assert text.count("LAYER M5") == 2
