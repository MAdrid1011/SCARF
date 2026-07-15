import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "hardware" / "scaling" / "deepscale.py"


def run_scale(tmp_path: Path, payload: dict, *args: str):
    source = tmp_path / "ppa.json"
    target = tmp_path / "scaled.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-node",
            "7",
            "--target-node",
            "28",
            "--input",
            str(source),
            "--output",
            str(target),
            *args,
        ],
        capture_output=True,
        text=True,
    )
    return result, target


def test_paper_reference_factors():
    sys.path.insert(0, str(ROOT))
    from hardware.scaling.deepscale import scaling_factor

    assert scaling_factor("area", 130, 45) == pytest.approx(8.3)
    assert scaling_factor("power", 45, 32) == pytest.approx(0.78 / 0.63)
    assert scaling_factor("area", 7, 28) == pytest.approx(0.011 / 0.35)
    assert scaling_factor("delay", 7, 28) == pytest.approx(0.53 / 0.67)


@pytest.mark.parametrize(
    "metric,value",
    [
        ("area_mm2", 2.5),
        ("delay_ns", 0.8),
        ("power_w", 1.2),
        ("energy_mj", 3.4),
        ("throughput_ips", 9.0),
        ("throughput_per_area", 3.6),
    ],
)
def test_round_trip(metric, value):
    sys.path.insert(0, str(ROOT))
    from hardware.scaling.deepscale import scale_value

    scaled = scale_value(metric, value, 7, 28)
    restored = scale_value(metric, scaled, 28, 7)
    assert restored == pytest.approx(value)


def test_cli_preserves_raw_record_and_writes_provenance(tmp_path):
    raw = {
        "schema_version": "1.0",
        "evidence_type": "asap7_predictive_postroute",
        "physical_valid": True,
        "metrics": {
            "logic_area_mm2": 1.0,
            "critical_path_ns": 0.5,
            "total_power_w": 2.0,
        },
    }
    result, output = run_scale(tmp_path, raw)
    assert result.returncode == 0, result.stderr
    scaled = json.loads(output.read_text(encoding="utf-8"))
    assert scaled["raw"] == raw
    assert scaled["evidence_type"] == "28nm_equivalent_estimate"
    assert scaled["source_process"] == "ASAP7 predictive 7 nm"
    assert scaled["target_process"] == "28 nm equivalent"
    assert scaled["scaling"]["tool"] == "DeepScaleTool"
    assert scaled["scaled_metrics"]["logic_area_mm2"] == pytest.approx(31.8181818)


def test_cli_rejects_invalid_or_incomplete_physical_record(tmp_path):
    result, output = run_scale(
        tmp_path,
        {"physical_valid": False, "metrics": {"logic_area_mm2": 1.0}},
    )
    assert result.returncode != 0
    assert not output.exists()
    assert "physical_valid" in result.stderr


def test_unsupported_node_is_rejected():
    sys.path.insert(0, str(ROOT))
    from hardware.scaling.deepscale import scaling_factor

    with pytest.raises(ValueError, match="unsupported technology node"):
        scaling_factor("area", 7, 12)


def test_pinned_workbook_is_present_and_verified(tmp_path):
    from hardware.scaling.deepscale import WORKBOOK_SHA256, verify_workbook

    assert verify_workbook() == WORKBOOK_SHA256
    corrupt = tmp_path / "DeepScaleTool.xlsm"
    corrupt.write_bytes(b"not the pinned workbook")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_workbook(corrupt)
