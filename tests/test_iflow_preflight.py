import os
import subprocess
from pathlib import Path

import pytest


def init_checkout(tmp_path: Path) -> Path:
    from hardware.iflow.preflight import COLLATERAL, EXPECTED_COMMIT, REQUIRED

    root = tmp_path / "iflow"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "ae@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "AE"], cwd=root, check=True)
    for relative in set(REQUIRED) | set(COLLATERAL):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    for relative in (
        "scripts/run_flow.py",
        "tools/yosys4be891e8/bin/yosys",
        "tools/OpenROADae191807/bin/openroad",
    ):
        os.chmod(root / relative, 0o755)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert actual != EXPECTED_COMMIT
    return root


def test_stage_contract_is_complete_and_ordered():
    from hardware.iflow.preflight import STAGES, PreflightError, parse_stages

    assert parse_stages("all") == STAGES
    assert parse_stages("synth,floorplan") == ("synth", "floorplan")
    with pytest.raises(PreflightError, match="physical-flow order"):
        parse_stages("floorplan,synth")
    with pytest.raises(PreflightError, match="unsupported"):
        parse_stages("synth,magic")


def test_wrong_commit_is_rejected(tmp_path):
    from hardware.iflow.preflight import PreflightError, validate_checkout

    root = init_checkout(tmp_path)
    with pytest.raises(PreflightError, match="commit mismatch"):
        validate_checkout(root)


def test_pinned_but_dirty_checkout_is_rejected(tmp_path, monkeypatch):
    import hardware.iflow.preflight as preflight

    root = init_checkout(tmp_path)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setattr(preflight, "EXPECTED_COMMIT", commit)
    tracked = root / "scripts/run_flow.py"
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(preflight.PreflightError, match="dirty"):
        preflight.validate_checkout(root)
