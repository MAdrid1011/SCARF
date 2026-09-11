import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_calibration_validation_fixture.py"


def test_calibration_fixture_is_deterministic_and_non_claim(tmp_path: Path):
    output = tmp_path / "calibration"
    command = [sys.executable, str(SCRIPT), "build", "--output-dir", str(output)]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    manifest = output / "calibration-fixture.json"
    first_bytes = manifest.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "non-claim calibration fixture" in first.stdout
    assert "non-claim calibration fixture" in second.stdout
    assert manifest.read_bytes() == first_bytes
    record = json.loads(first_bytes)
    assert record["claim_eligible"] is False
    assert record["synthetic_fixture"] is True
    assert record["selection"]["train_count"] == 24
    assert record["selection"]["holdout_count"] == 8
    assert record["candidate_parameters"] == {
        "beta_d": 1.0,
        "beta_f": 0.1,
        "beta_x": 0.5,
        "gamma_depth": 0.075,
    }


def test_calibration_fixture_rejects_tampered_selection(tmp_path: Path):
    output = tmp_path / "calibration"
    subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--output-dir", str(output)],
        check=True,
    )
    selection = output / "selection.json"
    selection.write_text(
        selection.read_text(encoding="utf-8").replace("fixture-train-000", "tampered"),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "validate",
            "--manifest",
            str(output / "calibration-fixture.json"),
            "--root",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "selection hash mismatch" in result.stdout


def test_calibration_fixture_is_rejected_by_real_config_builder(tmp_path: Path):
    output = tmp_path / "calibration"
    subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--output-dir", str(output)],
        check=True,
    )
    from scripts.calibrate_mechanisms import build_config

    with pytest.raises(ValueError, match="not a split-aware"):
        build_config(output / "calibration-fixture.json")


def test_calibration_fixture_output_is_rejected_inside_release_root(
    tmp_path: Path, monkeypatch
):
    import scripts.build_calibration_validation_fixture as module

    monkey_root = tmp_path / "checkout"
    monkey_root.mkdir()
    monkeypatch.setattr(module, "ROOT", monkey_root)
    with pytest.raises(ValueError, match="outside artifact"):
        module.build_fixture(monkey_root / "artifact" / "reference_results")
