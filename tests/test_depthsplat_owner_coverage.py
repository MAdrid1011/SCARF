"""Focused contracts for source-only, owner-strict DepthSplat coverage."""

from __future__ import annotations

import inspect
import json

import pytest


torch = pytest.importorskip("torch")


def _camera_set(owner_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.eye(4, dtype=torch.float64).reshape(1, 4, 4).repeat(owner_count, 1, 1),
        torch.eye(3, dtype=torch.float64).reshape(1, 3, 3).repeat(owner_count, 1, 1),
    )


def _covariances(count: int) -> torch.Tensor:
    covariance = torch.diag(torch.tensor((0.01, 0.01, 0.01), dtype=torch.float64))
    return covariance.reshape(1, 3, 3).repeat(count, 1, 1)


def _audit(
    *,
    virtual_means: torch.Tensor,
    owners: torch.Tensor,
    merged_means: torch.Tensor,
    owner_source_means: torch.Tensor | None = None,
    virtual_covariances: torch.Tensor | None = None,
    owner_source_covariances: torch.Tensor | None = None,
    merged_covariances: torch.Tensor | None = None,
):
    from saes.depthsplat_owner_coverage import audit_depthsplat_owner_coverage

    owner_count = int(merged_means.shape[0])
    extrinsics, intrinsics = _camera_set(owner_count)
    if owner_source_means is None:
        owner_source_means = merged_means.clone()
    if virtual_covariances is None:
        virtual_covariances = _covariances(int(virtual_means.shape[0]))
    if owner_source_covariances is None:
        owner_source_covariances = _covariances(owner_count)
    if merged_covariances is None:
        merged_covariances = _covariances(owner_count)
    return audit_depthsplat_owner_coverage(
        virtual_means=virtual_means,
        virtual_covariances=virtual_covariances,
        virtual_opacities=torch.full(
            (virtual_means.shape[0],), 0.5, dtype=torch.float64
        ),
        virtual_owner_indices=owners,
        owner_source_means=owner_source_means,
        owner_source_covariances=owner_source_covariances,
        owner_source_opacities=torch.full((owner_count,), 0.5, dtype=torch.float64),
        merged_means=merged_means,
        merged_covariances=merged_covariances,
        merged_opacities=torch.full((owner_count,), 0.5, dtype=torch.float64),
        owner_source_extrinsics=extrinsics,
        owner_source_intrinsics=intrinsics,
    )


def test_owner_assigned_coverage_passes_and_is_json_safe():
    means = torch.tensor(((0.0, 0.0, 2.0), (0.7, 0.0, 2.0)), dtype=torch.float64)
    virtual_covariances = _covariances(2)
    merged_covariances = _covariances(2)
    virtual_covariances_before = virtual_covariances.clone()
    merged_covariances_before = merged_covariances.clone()

    from saes.depthsplat_owner_coverage import audit_depthsplat_owner_coverage

    extrinsics, intrinsics = _camera_set(2)
    result = audit_depthsplat_owner_coverage(
        virtual_means=means,
        virtual_covariances=virtual_covariances,
        virtual_opacities=torch.full((2,), 0.5, dtype=torch.float64),
        virtual_owner_indices=torch.tensor((0, 1), dtype=torch.int64),
        owner_source_means=means.clone(),
        owner_source_covariances=_covariances(2),
        owner_source_opacities=torch.full((2,), 0.5, dtype=torch.float64),
        merged_means=means.clone(),
        merged_covariances=merged_covariances,
        merged_opacities=torch.full((2,), 0.5, dtype=torch.float64),
        owner_source_extrinsics=extrinsics,
        owner_source_intrinsics=intrinsics,
    )

    assert result["passed"] is True
    assert result["summary"]["hole_count"] == 0
    assert result["summary"]["all_active_virtual_2sigma_supports_contained"] is True
    assert result["summary"]["all_active_owner_anchor_and_virtual_2sigma_supports_contained"] is True
    assert [record["passed"] for record in result["owners"]] == [True, True]
    assert [record["source_anchor_support_included"] for record in result["owners"]] == [True, True]
    assert result["ownership"]["owner_assignment_counts"] == [1, 1]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert torch.equal(virtual_covariances, virtual_covariances_before)
    assert torch.equal(merged_covariances, merged_covariances_before)
    assert all(
        "target" not in parameter and "scale" not in parameter
        for parameter in inspect.signature(audit_depthsplat_owner_coverage).parameters
    )


def test_owner_assigned_coverage_reports_a_hole_for_its_own_merged_gaussian():
    result = _audit(
        virtual_means=torch.tensor(((0.7, 0.0, 2.0),), dtype=torch.float64),
        owners=torch.tensor((0,), dtype=torch.int64),
        merged_means=torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float64),
    )

    assert result["passed"] is False
    assert result["summary"]["hole_count"] == 1
    assert result["summary"]["failed_owner_count"] == 1
    assert result["owners"][0]["reason"] == "uncontained-assigned-virtuals"
    assert result["owners"][0]["coverage"]["hole_count"] >= 1


def test_owner_assigned_coverage_fails_closed_for_an_invalid_owner_index():
    result = _audit(
        virtual_means=torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float64),
        owners=torch.tensor((1,), dtype=torch.int64),
        merged_means=torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float64),
    )

    assert result["passed"] is False
    assert result["summary"]["input_valid"] is False
    assert result["summary"]["reason"] == "owner-index-out-of-range"
    assert result["summary"]["invalid_owner_assignment_count"] == 1
    assert result["owners"] == []


def test_owner_assigned_coverage_rejects_cross_owner_masking():
    """A global candidate set covers both virtuals, but each owner is wrong."""

    virtual_means = torch.tensor(
        ((0.0, 0.0, 2.0), (0.7, 0.0, 2.0)), dtype=torch.float64
    )
    swapped_merged_means = torch.tensor(
        ((0.7, 0.0, 2.0), (0.0, 0.0, 2.0)), dtype=torch.float64
    )
    result = _audit(
        virtual_means=virtual_means,
        owners=torch.tensor((0, 1), dtype=torch.int64),
        merged_means=swapped_merged_means,
    )

    from saes.projected_domain_coverage_audit import audit_projected_dense_domain_coverage

    extrinsic, intrinsic = _camera_set(1)
    globally_masked = audit_projected_dense_domain_coverage(
        dense_means=virtual_means,
        dense_covariances=_covariances(2),
        dense_opacities=torch.full((2,), 0.5, dtype=torch.float64),
        candidate_means=swapped_merged_means,
        candidate_covariances=_covariances(2),
        candidate_opacities=torch.full((2,), 0.5, dtype=torch.float64),
        context_extrinsic=extrinsic[0],
        context_intrinsic=intrinsic[0],
    )

    assert globally_masked.valid is True
    assert globally_masked.hole_count == 0
    assert result["passed"] is False
    assert result["summary"]["hole_count"] == 2
    assert [record["coverage"]["hole_count"] for record in result["owners"]] == [1, 1]


def test_owner_assigned_coverage_rejects_a_replaced_source_anchor_hole():
    """A fixed-scale merge must certify the source anchor it replaces."""

    covariance = torch.diag(torch.tensor((0.01, 0.01, 0.01), dtype=torch.float64))
    merged_covariance = torch.diag(
        torch.tensor((0.0541, 0.01, 0.01), dtype=torch.float64)
    )
    result = _audit(
        virtual_means=torch.tensor(((0.7, 0.0, 2.0),), dtype=torch.float64),
        owners=torch.tensor((0,), dtype=torch.int64),
        owner_source_means=torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float64),
        merged_means=torch.tensor(((0.63, 0.0, 2.0),), dtype=torch.float64),
        virtual_covariances=covariance.reshape(1, 3, 3),
        owner_source_covariances=covariance.reshape(1, 3, 3),
        merged_covariances=merged_covariance.reshape(1, 3, 3),
    )

    assert result["passed"] is False
    assert result["owners"][0]["source_anchor_support_included"] is True
    assert result["owners"][0]["source_anchor_support_passed"] is False
    assert result["owners"][0]["coverage"]["hole_count"] >= 1


def test_owner_assigned_coverage_checks_a_source_anchor_without_virtuals():
    result = _audit(
        virtual_means=torch.empty((0, 3), dtype=torch.float64),
        owners=torch.empty((0,), dtype=torch.int64),
        owner_source_means=torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float64),
        merged_means=torch.tensor(((0.7, 0.0, 2.0),), dtype=torch.float64),
    )

    assert result["passed"] is False
    assert result["summary"]["virtual_primitive_count"] == 0
    assert result["owners"][0]["dense_descriptor_count"] == 1
    assert result["owners"][0]["source_anchor_support_passed"] is False
