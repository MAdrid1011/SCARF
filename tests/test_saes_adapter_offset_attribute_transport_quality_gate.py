import pytest
import subprocess
import sys
from pathlib import Path


def test_attribute_transport_quality_gate_is_fixed_and_rejects_overrides():
    from scripts import saes_adapter_offset_attribute_transport_quality_gate as gate

    assert gate.MATERIALIZATION == (
        "conditional-adapter-offset-attribute-transport-diagnostic"
    )
    assert gate.PILOT_KIND == (
        "saes_adapter_offset_attribute_transport_selected_output_quality_pilot"
    )
    assert gate.SEED == 0
    for option, value in (
        ("--seed", "1"),
        ("--materialization", "conditional-adapter-offset-transport-diagnostic"),
    ):
        with pytest.raises(SystemExit) as exc:
            gate.main(
                [
                    "--output-dir",
                    "outputs/fixed",
                    option,
                    value,
                ]
            )
        assert exc.value.code == 2


def test_attribute_transport_quality_gate_runs_as_a_direct_script():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "saes_adapter_offset_attribute_transport_quality_gate.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--seed" not in completed.stdout
    assert "--materialization" not in completed.stdout
