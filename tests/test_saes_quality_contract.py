import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_representative_quality_gate_is_fixed_to_the_registered_route():
    from scripts import saes_representative_quality_gate as gate
    from scripts.saes_execution_identity import build_saes_execution_identity

    assert gate.EXECUTION_IDENTITY == build_saes_execution_identity()
    assert gate.MATERIALIZATION == "representative"
    assert gate.EXECUTION_IDENTITY["context_safety_guard"] is True
    assert gate.EXECUTION_IDENTITY["context_guard_policy"] == (
        "projected-anchor-coverage-occlusion-v3"
    )
    assert gate.EXECUTION_IDENTITY["context_guard_max_center_mahalanobis"] == 2.0
    assert gate.EXECUTION_IDENTITY["cross_check_threshold"] == 0.015
    assert gate.PILOT_KIND == "saes_representative_candidate_quality_pilot"
    assert gate.SEED == 0
    for option, value in (
        ("--seed", "1"),
        ("--context-safety-guard", None),
        ("--materialization", "dense-diagnostic"),
    ):
        arguments = ["--output-dir", "outputs/fixed", option]
        if value is not None:
            arguments.append(value)
        with pytest.raises(SystemExit) as exc:
            gate.main(arguments)
        assert exc.value.code == 2


def test_quality_route_rejects_any_candidate_identity_drift():
    from scripts.saes_execution_identity import build_saes_execution_identity
    from scripts.saes_selected_output_quality_gate import resolve_quality_route

    identity = build_saes_execution_identity()
    resolved = resolve_quality_route(
        materialization="representative",
        context_safety_guard=True,
        execution_identity=identity,
    )
    assert resolved["route_sha256"] == identity["route_sha256"]
    with pytest.raises(ValueError, match="context safety guard"):
        resolve_quality_route(
            materialization="representative",
            context_safety_guard=False,
            execution_identity=identity,
        )
    with pytest.raises(ValueError, match="materialization"):
        resolve_quality_route(
            materialization="dense-diagnostic",
            context_safety_guard=True,
            execution_identity=identity,
        )


def test_representative_quality_gate_direct_help_hides_fixed_overrides():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "saes_representative_quality_gate.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--seed" not in completed.stdout
    assert "--context-safety-guard" not in completed.stdout
    assert "--materialization" not in completed.stdout
