import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_claim_timing_validation_fixture.py"


def test_fixture_is_deterministic_and_explicitly_non_claim(tmp_path: Path):
    output = tmp_path / "fixture"
    command = [sys.executable, str(SCRIPT), "build", "--output-dir", str(output)]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    manifest = output / "manifest.json"
    first_bytes = manifest.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "claim_eligible=false" in first.stdout
    assert "claim_eligible=false" in second.stdout
    assert manifest.read_bytes() == first_bytes
    record = json.loads(first_bytes)
    assert record["bundle_kind"] == "non-claim-validation-fixture"
    assert record["synthetic_fixture"] is True
    assert record["not_for_release"] is True
    assert record["samples"][0]["variants"]["asic_fsdr_saes"]["total_cycles"] == 305


def test_fixture_validator_accepts_and_claim_loader_rejects(tmp_path: Path):
    output = tmp_path / "fixture"
    subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--output-dir", str(output)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "validate",
            "--manifest",
            str(output / "manifest.json"),
            "--root",
            str(output),
        ],
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/claim_timing_backend.py"),
            str(output / "manifest.json"),
            "--root",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "invalid field set" in result.stdout


def test_fixture_trace_cannot_be_promoted_by_rewriting_manifest_shell(tmp_path: Path):
    output = tmp_path / "fixture"
    subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--output-dir", str(output)],
        check=True,
    )
    source = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    promoted = {
        "schema_version": "source-bound-timing-backend-v1",
        "backend": source["backend"],
        "clock_mhz": source["clock_mhz"],
        "samples": source["samples"],
    }
    manifest = output / "promoted-shell.json"
    manifest.write_text(json.dumps(promoted), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/claim_timing_backend.py"),
            str(manifest),
            "--root",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "synthetic/non-claim" in result.stdout


def test_fixture_output_is_rejected_inside_release_roots(tmp_path: Path, monkeypatch):
    from scripts.build_claim_timing_validation_fixture import build_fixture

    # The check is based on the checkout root, so use a temporary path that is
    # deliberately made to look like a release subtree through monkeypatching.
    import scripts.build_claim_timing_validation_fixture as module

    monkey_root = tmp_path / "checkout"
    monkey_root.mkdir()
    monkeypatch.setattr(module, "ROOT", monkey_root)
    with pytest.raises(ValueError, match="outside artifact"):
        build_fixture(monkey_root / "artifact" / "reference_results")
