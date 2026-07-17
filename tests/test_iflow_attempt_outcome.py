import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_attempt_outcome_script_runs_from_its_own_directory():
    result = subprocess.run(
        [sys.executable, str(ROOT / "hardware/iflow/attempt_outcome.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_incomplete_attempt_record_preserves_stage_and_resource_evidence(tmp_path, monkeypatch):
    from hardware.iflow import attempt_outcome

    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "resource-before-groute.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "stage-cts-validation.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "iflow-groute.log").write_text("routing\n", encoding="utf-8")
    (tmp_path / "time-groute.log").write_text("time\n", encoding="utf-8")
    monkeypatch.setattr(
        attempt_outcome,
        "memory_snapshot",
        lambda: {"mem_available_bytes": 1024, "swap_used_bytes": 2048},
    )

    record = attempt_outcome.build_record(
        tmp_path,
        incomplete_stage="groute",
        reason="persistent_swap_thrashing",
        termination="controller_sigterm",
        sample_seconds=15,
        swap_in_pages=172774,
        swap_out_pages=3044,
    )

    assert record["status"] == "NOT_CLAIMED_RESOURCE_LIMIT"
    assert record["physical_valid"] is False
    assert record["incomplete_stage"] == "groute"
    assert record["completed_stages"] == ["cts"]
    assert record["swap_observation"]["swap_in_pages"] == 172774
    assert {item["path"] for item in record["evidence"]} == {
        "iflow-groute.log",
        "manifest.json",
        "resource-before-groute.json",
        "stage-cts-validation.json",
        "time-groute.log",
    }


def test_incomplete_attempt_rejects_invalid_swap_observation(tmp_path):
    from hardware.iflow.attempt_outcome import build_record

    with pytest.raises(ValueError, match="swap_in_pages"):
        build_record(
            tmp_path,
            incomplete_stage="groute",
            reason="persistent_swap_thrashing",
            termination="controller_sigterm",
            swap_in_pages=-1,
        )
