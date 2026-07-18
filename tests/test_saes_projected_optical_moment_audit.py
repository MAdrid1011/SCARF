"""Focused synthetic tests for context-only projected optical-moment audits."""

from __future__ import annotations

import inspect

import pytest


torch = pytest.importorskip("torch")


def _camera(*, origin=(0.0, 0.0, 0.0)):
    extrinsic = torch.eye(4, dtype=torch.float64)
    extrinsic[:3, 3] = torch.tensor(origin, dtype=torch.float64)
    return extrinsic, torch.eye(3, dtype=torch.float64)


def _descriptors(*, offscreen=False, opacity=0.45, covariance_scale=1.0):
    x = 3.0 if offscreen else 0.2
    means = torch.tensor(((x, 0.1, 2.0), (x + 0.1, -0.1, 2.4)), dtype=torch.float64)
    covariance = torch.diag(torch.tensor((0.10, 0.08, 0.04), dtype=torch.float64))
    covariances = covariance.unsqueeze(0).repeat(2, 1, 1) * covariance_scale
    opacities = torch.tensor((opacity, opacity * 0.8), dtype=torch.float64)
    return means, covariances, opacities


def _moment(**kwargs):
    from saes.projected_optical_moment_audit import project_context_optical_moment

    means, covariances, opacities = _descriptors(**kwargs)
    extrinsic, intrinsic = _camera()
    return project_context_optical_moment(
        means=means,
        covariances=covariances,
        opacities=opacities,
        context_extrinsic=extrinsic,
        context_intrinsic=intrinsic,
    )


def test_context_projected_optical_moment_keeps_finite_offscreen_points():
    moment = _moment(offscreen=True)

    assert moment.valid
    assert moment.reason is None
    assert moment.offscreen_count == 2
    assert moment.mass is not None and moment.mass > 0.0
    assert moment.center is not None and moment.center[0] > 1.0
    assert moment.covariance is not None
    assert moment.determinant is not None and moment.determinant > 0.0
    assert moment.log_determinant is not None
    assert moment.footprint_log_area == pytest.approx(0.5 * moment.log_determinant)
    assert moment.psd
    assert moment.condition_number is not None and moment.condition_number >= 1.0


@pytest.mark.parametrize(
    ("mutator", "reason"),
    (
        (lambda means, covariances, opacities: (means, covariances, torch.ones_like(opacities)), "invalid-alpha"),
        (
            lambda means, covariances, opacities: (
                means,
                torch.diag(torch.tensor((-0.1, 0.08, 0.04), dtype=torch.float64)).unsqueeze(0).repeat(2, 1, 1),
                opacities,
            ),
            "invalid-covariance-psd",
        ),
        (
            lambda means, covariances, opacities: (means, torch.zeros_like(covariances), opacities),
            "invalid-projected-determinant",
        ),
        (lambda means, covariances, opacities: (means, covariances, torch.zeros_like(opacities)), "zero-optical-mass"),
        (
            lambda means, covariances, opacities: (
                means - torch.tensor((0.0, 0.0, 3.0), dtype=torch.float64),
                covariances,
                opacities,
            ),
            "nonpositive-camera-depth",
        ),
    ),
)
def test_context_projected_optical_moment_reports_strict_failures(mutator, reason):
    from saes.projected_optical_moment_audit import project_context_optical_moment

    means, covariances, opacities = mutator(*_descriptors())
    extrinsic, intrinsic = _camera()
    moment = project_context_optical_moment(
        means=means,
        covariances=covariances,
        opacities=opacities,
        context_extrinsic=extrinsic,
        context_intrinsic=intrinsic,
    )

    assert not moment.valid
    assert moment.reason == reason
    assert moment.mass is None
    assert moment.center is None
    assert moment.covariance is None


def test_tile_comparison_exposes_dense_errors_psd_condition_and_fit_residual():
    from saes.projected_optical_moment_audit import compare_tile_directionally

    dense = _moment()
    current = _moment(covariance_scale=1.6)
    candidate = _moment()
    comparison = compare_tile_directionally(
        tile_key=(0, 3, 7),
        level="L0",
        context_index=1,
        dense=dense,
        current=current,
        candidate=candidate,
    )

    assert comparison.valid
    assert comparison.current is not None and comparison.candidate is not None
    assert comparison.current.mass_absolute_error > 0.0
    assert comparison.current.center_absolute_error >= 0.0
    assert comparison.current.covariance_absolute_error > 0.0
    assert comparison.current.footprint_determinant_absolute_error > 0.0
    assert comparison.current.footprint_log_ratio_absolute > 0.0
    assert comparison.current.psd
    assert comparison.current.condition_number >= 1.0
    assert comparison.current.fit_residual == pytest.approx(
        comparison.current.covariance_relative_error
    )
    assert comparison.candidate.mass_absolute_error == pytest.approx(0.0)
    assert comparison.candidate.covariance_relative_error == pytest.approx(0.0)


def _synthetic_error(*, mass, covariance, other=0.9):
    from saes.projected_optical_moment_audit import DenseErrors

    return DenseErrors(
        mass_absolute_error=mass,
        mass_relative_error=mass,
        center_absolute_error=other,
        covariance_absolute_error=covariance,
        covariance_relative_error=covariance,
        footprint_determinant_absolute_error=other,
        footprint_log_ratio=other,
        footprint_log_ratio_absolute=other,
        psd=True,
        condition_number=2.0,
        fit_residual=covariance,
    )


def _synthetic_record(*, tile, level, context, current_mass=1.0, candidate_mass=1.0, current_covariance=1.0, candidate_covariance=0.75):
    from saes.projected_optical_moment_audit import (
        DirectionalTileComparison,
        ProjectedOpticalMoment,
    )

    dense = ProjectedOpticalMoment(
        valid=True,
        reason=None,
        mass=1.0,
        center=torch.zeros(2, dtype=torch.float64),
        covariance=torch.eye(2, dtype=torch.float64),
        determinant=1.0,
        log_determinant=0.0,
        footprint_log_area=0.0,
        psd=True,
        condition_number=1.0,
        descriptor_count=1,
        offscreen_count=0,
    )
    return DirectionalTileComparison(
        tile_key=tile,
        level=level,
        context_index=context,
        valid=True,
        reason=None,
        dense=dense,
        current=_synthetic_error(mass=current_mass, covariance=current_covariance, other=1.0),
        candidate=_synthetic_error(mass=candidate_mass, covariance=candidate_covariance, other=0.9),
    )


def test_aggregation_reports_p50_p95_and_max_by_level_context_cell():
    from saes.projected_optical_moment_audit import aggregate_directional_comparisons

    records = tuple(
        _synthetic_record(
            tile=index,
            level="L0",
            context=0,
            current_covariance=float(index),
            candidate_covariance=float(index) * 0.75,
        )
        for index in (1, 2, 3)
    )
    summaries = aggregate_directional_comparisons(records)

    assert len(summaries) == 1
    summary = summaries[0]
    current = summary.current["covariance_relative_error"]
    candidate = summary.candidate["covariance_relative_error"]
    assert current.count == 3
    assert current.p50 == pytest.approx(2.0)
    assert current.p95 == pytest.approx(2.9)
    assert current.maximum == pytest.approx(3.0)
    assert candidate.p95 == pytest.approx(2.175)
    assert summary.percentile_nonworse


def test_directional_gate_requires_all_cell_percentiles_covariance_drop_and_mass_nonworsening():
    from saes.projected_optical_moment_audit import evaluate_directional_gate

    records = (
        _synthetic_record(tile="a", level="L0", context=0),
        _synthetic_record(tile="b", level="L1", context=1),
    )
    passing = evaluate_directional_gate(records)
    assert passing.passed
    assert passing.aggregate_covariance_decrease == pytest.approx(0.25)

    insufficient_covariance = evaluate_directional_gate(
        (
            _synthetic_record(
                tile="a", level="L0", context=0, candidate_covariance=0.85
            ),
            _synthetic_record(
                tile="b", level="L1", context=1, candidate_covariance=0.85
            ),
        )
    )
    assert not insufficient_covariance.passed
    assert insufficient_covariance.reason == "insufficient-aggregate-covariance-decrease"

    mass_worse = evaluate_directional_gate(
        (
            _synthetic_record(tile="a", level="L0", context=0, candidate_mass=1.005),
            _synthetic_record(tile="b", level="L1", context=1, candidate_mass=1.005),
        )
    )
    assert not mass_worse.passed
    assert mass_worse.reason == "aggregate-mass-error-worse"

    percentile_worse = evaluate_directional_gate(
        (
            _synthetic_record(tile="a", level="L0", context=0, candidate_covariance=1.02),
            _synthetic_record(tile="b", level="L1", context=1, candidate_covariance=1.02),
        )
    )
    assert not percentile_worse.passed
    assert percentile_worse.reason == "percentile-regression-or-invalid-record"


def test_public_helpers_have_no_target_parameter_and_invalid_records_fail_the_gate():
    from saes.projected_optical_moment_audit import (
        DirectionalTileComparison,
        evaluate_directional_gate,
        project_context_optical_moment,
        compare_tile_directionally,
    )

    for helper in (
        project_context_optical_moment,
        compare_tile_directionally,
        evaluate_directional_gate,
    ):
        assert all("target" not in parameter for parameter in inspect.signature(helper).parameters)

    dense = _moment()
    invalid = _moment(opacity=0.0)
    comparison = compare_tile_directionally(
        tile_key="invalid",
        level="L1",
        context_index=0,
        dense=dense,
        current=dense,
        candidate=invalid,
    )
    assert not comparison.valid
    assert comparison.reason == "candidate:zero-optical-mass"
    gate = evaluate_directional_gate((comparison,))
    assert not gate.passed
    assert gate.reason == "percentile-regression-or-invalid-record"
    assert isinstance(comparison, DirectionalTileComparison)
