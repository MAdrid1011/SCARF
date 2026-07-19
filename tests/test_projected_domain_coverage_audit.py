"""Focused tests for posthoc dense projected-domain coverage diagnostics."""

from __future__ import annotations

import inspect

import pytest


torch = pytest.importorskip("torch")


def _camera():
    return torch.eye(4, dtype=torch.float64), torch.eye(3, dtype=torch.float64)


def _descriptors(*, dense_x: float = 0.1, candidate_x: float = 0.1, candidate_scale: float = 4.0):
    dense_means = torch.tensor(((dense_x, 0.0, 2.0),), dtype=torch.float64)
    candidate_means = torch.tensor(((candidate_x, 0.0, 2.0),), dtype=torch.float64)
    dense_covariances = torch.diag(torch.tensor((0.01, 0.01, 0.01), dtype=torch.float64)).unsqueeze(0)
    candidate_covariances = dense_covariances * candidate_scale
    dense_opacities = torch.tensor((0.5,), dtype=torch.float64)
    candidate_opacities = torch.tensor((0.5,), dtype=torch.float64)
    return (
        dense_means,
        dense_covariances,
        dense_opacities,
        candidate_means,
        candidate_covariances,
        candidate_opacities,
    )


def _audit(**kwargs):
    from saes.projected_domain_coverage_audit import audit_projected_dense_domain_coverage

    extrinsic, intrinsic = _camera()
    values = _descriptors(**kwargs)
    return audit_projected_dense_domain_coverage(
        dense_means=values[0],
        dense_covariances=values[1],
        dense_opacities=values[2],
        candidate_means=values[3],
        candidate_covariances=values[4],
        candidate_opacities=values[5],
        context_extrinsic=extrinsic,
        context_intrinsic=intrinsic,
    )


def test_two_sigma_dense_ellipse_containment_reports_full_recall():
    result = _audit()

    assert result.valid
    assert result.reason is None
    assert result.sigma == 2.0
    assert result.active_dense_descriptor_count == 1
    assert result.active_candidate_descriptor_count == 1
    assert result.hole_count == 0
    assert result.count_recall == pytest.approx(1.0)
    assert result.mass_weighted_recall == pytest.approx(1.0)
    assert result.contained_optical_mass == pytest.approx(result.dense_optical_mass)


def test_uncontained_dense_ellipse_is_reported_as_a_hole():
    result = _audit(dense_x=0.7, candidate_x=0.0, candidate_scale=1.0)

    assert result.valid
    assert result.hole_count == 1
    assert result.count_recall == pytest.approx(0.0)
    assert result.mass_weighted_recall == pytest.approx(0.0)
    assert result.contained_optical_mass == pytest.approx(0.0)


def test_invalid_dense_covariance_fails_closed():
    from saes.projected_domain_coverage_audit import audit_projected_dense_domain_coverage

    extrinsic, intrinsic = _camera()
    values = list(_descriptors())
    values[1] = torch.diag(torch.tensor((-0.01, 0.01, 0.01), dtype=torch.float64)).unsqueeze(0)
    result = audit_projected_dense_domain_coverage(
        dense_means=values[0],
        dense_covariances=values[1],
        dense_opacities=values[2],
        candidate_means=values[3],
        candidate_covariances=values[4],
        candidate_opacities=values[5],
        context_extrinsic=extrinsic,
        context_intrinsic=intrinsic,
    )

    assert not result.valid
    assert result.reason == "dense:invalid-covariance-psd"
    assert result.mass_weighted_recall is None
    assert result.count_recall is None
    assert result.hole_count is None


def test_public_audit_has_no_target_parameter():
    from saes.projected_domain_coverage_audit import audit_projected_dense_domain_coverage

    assert all("target" not in name for name in inspect.signature(audit_projected_dense_domain_coverage).parameters)
