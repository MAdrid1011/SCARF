from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _gaussians():
    values = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    return SimpleNamespace(
        means=values.clone(),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 6, 1, 1),
        harmonics=values.unsqueeze(-1).clone(),
        opacities=torch.full((1, 6, 1), 0.2),
    )


def test_selected_only_guard_rejects_skipped_s3_reads_and_reports_selected_reads():
    from saes.frozen_audit_preflight import (
        selected_s3_read_evidence,
        wrap_selected_s3_reads,
    )

    guarded, fields = wrap_selected_s3_reads(_gaussians(), torch.tensor((1, 4)))
    _ = guarded.means[0, torch.tensor((1, 4))]
    _ = guarded.covariances[0, torch.tensor((1, 4))]
    _ = guarded.harmonics[0, torch.tensor((1, 4))]
    _ = guarded.opacities[0, torch.tensor((1, 4))]
    evidence = selected_s3_read_evidence(fields, label="fixture")
    assert all(value["selected_only"] for value in evidence.values())
    with pytest.raises(AssertionError, match="skipped S3 descriptor read"):
        _ = guarded.means[0, torch.tensor((0,))]


def test_posthoc_full_s3_exception_is_nonruntime_and_observes_dense_reads():
    from saes.frozen_audit_preflight import (
        bind_posthoc_full_s3_observation,
        require_posthoc_full_s3_exception,
    )

    contract = require_posthoc_full_s3_exception(
        materialization="same-budget-dense-oracle-diagnostic",
        runtime_execution=False,
        paper_result_eligible=False,
        quality_gate_authorized=False,
    )
    record = bind_posthoc_full_s3_observation(contract, full_stage3_reads=12)
    assert record["selected_only_s3_instrumentation_verdict"] == "not-applicable"
    assert record["observed_full_stage3_reads"] == 12


def test_selected_only_commit_rejects_trace_counter_and_output_drift():
    from saes.frozen_audit_preflight import (
        assert_selected_only_commit_equal,
        selected_only_commit_payload,
    )

    gaussians = _gaussians()
    mask = torch.tensor((False, True, False, True, False, True))
    reference = selected_only_commit_payload(
        gaussians=gaussians,
        modified_mask=mask,
        stats={"full_tiles": 1},
        tile_trace=[{"level": "L0"}],
    )
    identical = selected_only_commit_payload(
        gaussians=gaussians,
        modified_mask=mask,
        stats={"full_tiles": 1},
        tile_trace=[{"level": "L0"}],
    )
    assert assert_selected_only_commit_equal(reference, identical, label="fixture") == {
        "means": 0.0,
        "covariances": 0.0,
        "harmonics": 0.0,
        "opacities": 0.0,
    }
    drift = {**identical, "stats": {"full_tiles": 2}}
    with pytest.raises(RuntimeError, match="stats"):
        assert_selected_only_commit_equal(reference, drift, label="fixture")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"materialization": "different"},
        {"runtime_execution": True},
        {"paper_result_eligible": True},
        {"quality_gate_authorized": True},
    ),
)
def test_posthoc_full_s3_exception_fails_closed_for_wrong_mode_or_eligibility(kwargs):
    from saes.frozen_audit_preflight import require_posthoc_full_s3_exception

    baseline = {
        "materialization": "same-budget-dense-oracle-diagnostic",
        "runtime_execution": False,
        "paper_result_eligible": False,
        "quality_gate_authorized": False,
    }
    with pytest.raises(RuntimeError):
        require_posthoc_full_s3_exception(**{**baseline, **kwargs})
