from pathlib import Path

import pytest


def test_orin_mode_is_empty_without_real_device_evidence(tmp_path: Path):
    from argparse import Namespace
    from scripts.run_ae import build_plan

    plan = build_plan(
        Namespace(mode="orin", output_root=tmp_path, python=None, num_samples=1)
    )
    assert not plan["experiments"]
    assert plan["claim_status"]["figure8"] == "NOT_CLAIMED_NO_ORIN_EVIDENCE"


def test_tegrastats_temperature_summary_is_strict():
    from hardware.orin.run import parse_tegrastats

    summary = parse_tegrastats(
        "RAM 100/1000 CPU@42.5C GPU@48.0C\nRAM 110/1000 CPU@43.0C GPU@51.5C\n"
    )
    assert summary == {
        "sample_lines": 2,
        "maximum_temperature_c": 51.5,
        "minimum_temperature_c": 42.5,
    }
    with pytest.raises(ValueError, match="no temperature"):
        parse_tegrastats("RAM 100/1000\n")


def test_orin_contract_uses_runtime_autodetection():
    from scripts.check_environment import load_orin_contract

    contract = load_orin_contract()
    assert contract["status"] == "runtime_autodetect"
    assert contract["timing_repetitions"] == 5
