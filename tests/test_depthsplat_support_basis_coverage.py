"""Focused contracts for the source-only DepthSplat support-basis guard."""

from __future__ import annotations

import inspect
import json

import pytest


torch = pytest.importorskip("torch")


def _covariances(count: int, *, scale: float = 0.01) -> torch.Tensor:
    covariance = torch.diag(torch.tensor((scale, scale, scale), dtype=torch.float64))
    return covariance.reshape(1, 3, 3).repeat(count, 1, 1)


def _cameras(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.eye(4, dtype=torch.float64).reshape(1, 4, 4).repeat(count, 1, 1),
        torch.eye(3, dtype=torch.float64).reshape(1, 3, 3).repeat(count, 1, 1),
    )


def _audit(
    *,
    virtual_means: torch.Tensor,
    source_means: torch.Tensor,
    merged_means: torch.Tensor,
    spatial: torch.Tensor,
    bilateral: torch.Tensor,
    source_opacities: torch.Tensor | None = None,
    virtual_opacities: torch.Tensor | None = None,
    context_extrinsics: torch.Tensor | None = None,
    context_intrinsics: torch.Tensor | None = None,
):
    from saes.depthsplat_support_basis_coverage import (
        audit_depthsplat_tile_support_basis,
    )

    anchor_count = int(source_means.shape[0])
    virtual_count = int(virtual_means.shape[0])
    if source_opacities is None:
        source_opacities = torch.full((anchor_count,), 0.5, dtype=torch.float64)
    if virtual_opacities is None:
        virtual_opacities = torch.full((virtual_count,), 0.5, dtype=torch.float64)
    if context_extrinsics is None or context_intrinsics is None:
        context_extrinsics, context_intrinsics = _cameras(anchor_count)
    return audit_depthsplat_tile_support_basis(
        virtual_means=virtual_means,
        virtual_covariances=_covariances(virtual_count),
        virtual_opacities=virtual_opacities,
        virtual_origin_slots=torch.arange(
            100, 100 + virtual_count, dtype=torch.int64
        ),
        virtual_source_spatial_weights=spatial,
        bilateral_assignment_weights=bilateral,
        anchor_source_means=source_means,
        anchor_source_covariances=_covariances(anchor_count),
        anchor_source_opacities=source_opacities,
        anchor_dense_slots=torch.arange(anchor_count, dtype=torch.int64),
        merged_means=merged_means,
        merged_covariances=_covariances(anchor_count),
        merged_opacities=torch.full((anchor_count,), 0.5, dtype=torch.float64),
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
    )


def test_support_basis_passes_through_a_composed_same_tile_anchor_path():
    """A source anchor can be covered by its actual S@R output path."""

    source_means = torch.tensor(((0.0, 0.0, 2.0), (0.7, 0.0, 2.0)), dtype=torch.float64)
    result = _audit(
        virtual_means=torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float64),
        source_means=source_means,
        # Output 0 has drifted; output 1 contains the active source/virtual.
        merged_means=torch.tensor(((0.7, 0.0, 2.0), (0.0, 0.0, 2.0)), dtype=torch.float64),
        spatial=torch.tensor(((1.0, 0.0),), dtype=torch.float64),
        bilateral=torch.tensor(((0.0, 1.0),), dtype=torch.float64),
        source_opacities=torch.tensor((0.5, 0.0), dtype=torch.float64),
    )

    assert result["passed"] is True
    assert result["summary"]["hole_count"] == 0
    assert result["anchors"][0]["basis_candidate_anchor_indices"] == [0, 1]
    assert result["anchors"][0]["passed"] is True
    assert result["virtuals"][0]["basis_candidate_anchor_indices"] == [1]
    assert result["virtuals"][0]["passed"] is True
    assert result["source_only"]["target_rgb_accessed"] is False
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_support_basis_rejects_a_candidate_without_a_positive_virtual_ledger_path():
    source_means = torch.tensor(((0.0, 0.0, 2.0), (0.7, 0.0, 2.0)), dtype=torch.float64)
    result = _audit(
        virtual_means=torch.tensor(((0.7, 0.0, 2.0),), dtype=torch.float64),
        source_means=source_means,
        # Candidate 1 would cover the virtual, but R only permits candidate 0.
        merged_means=torch.tensor(((0.0, 0.0, 2.0), (0.7, 0.0, 2.0)), dtype=torch.float64),
        spatial=torch.tensor(((1.0, 0.0),), dtype=torch.float64),
        bilateral=torch.tensor(((1.0, 0.0),), dtype=torch.float64),
        source_opacities=torch.tensor((0.0, 0.0), dtype=torch.float64),
    )

    assert result["passed"] is False
    assert result["virtuals"][0]["basis_candidate_anchor_indices"] == [0]
    assert result["virtuals"][0]["reason"] == "uncontained-same-tile-support"
    assert result["summary"]["failed_virtual_count"] == 1


def test_support_basis_rejects_piecewise_coverage_without_one_containing_candidate():
    result = _audit(
        virtual_means=torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float64),
        source_means=torch.tensor(((0.0, 0.0, 2.0), (0.0, 0.0, 2.0)), dtype=torch.float64),
        merged_means=torch.tensor(((-0.08, 0.0, 2.0), (0.08, 0.0, 2.0)), dtype=torch.float64),
        spatial=torch.tensor(((0.5, 0.5),), dtype=torch.float64),
        bilateral=torch.tensor(((0.5, 0.5),), dtype=torch.float64),
        source_opacities=torch.tensor((0.0, 0.0), dtype=torch.float64),
    )

    assert result["passed"] is False
    assert result["virtuals"][0]["basis_candidate_anchor_indices"] == [0, 1]
    assert result["virtuals"][0]["coverage"]["hole_count"] == 1


def test_support_basis_rejects_non_normalized_ledger_and_camera_drift():
    source = torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float64)
    bad_ledger = _audit(
        virtual_means=source.clone(),
        source_means=source.clone(),
        merged_means=source.clone(),
        spatial=torch.tensor(((1.0,),), dtype=torch.float64),
        bilateral=torch.tensor(((0.5,),), dtype=torch.float64),
    )
    extrinsics, intrinsics = _cameras(1)
    extrinsics = extrinsics.repeat(2, 1, 1)
    intrinsics = intrinsics.repeat(2, 1, 1)
    extrinsics[1, 0, 3] = 1.0
    camera_drift = _audit(
        virtual_means=source.clone(),
        source_means=torch.cat((source, source), dim=0),
        merged_means=torch.cat((source, source), dim=0),
        spatial=torch.tensor(((0.5, 0.5),), dtype=torch.float64),
        bilateral=torch.tensor(((0.5, 0.5),), dtype=torch.float64),
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )

    assert bad_ledger["summary"]["reason"] == "bilateral-assignment-normalization"
    assert camera_drift["summary"]["reason"] == "same-tile-camera-mismatch"


def test_support_basis_binds_both_ledgers_without_mutating_input_covariances():
    from saes.depthsplat_support_basis_coverage import audit_depthsplat_tile_support_basis

    means = torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float64)
    covariances = _covariances(1)
    before = covariances.clone()
    extrinsics, intrinsics = _cameras(1)
    result = audit_depthsplat_tile_support_basis(
        virtual_means=means.clone(),
        virtual_covariances=covariances,
        virtual_opacities=torch.tensor((0.5,), dtype=torch.float64),
        virtual_origin_slots=torch.tensor((100,), dtype=torch.int64),
        virtual_source_spatial_weights=torch.tensor(((1.0,),), dtype=torch.float64),
        bilateral_assignment_weights=torch.tensor(((1.0,),), dtype=torch.float64),
        anchor_source_means=means.clone(),
        anchor_source_covariances=covariances,
        anchor_source_opacities=torch.tensor((0.5,), dtype=torch.float64),
        anchor_dense_slots=torch.tensor((0,), dtype=torch.int64),
        merged_means=means.clone(),
        merged_covariances=covariances,
        merged_opacities=torch.tensor((0.5,), dtype=torch.float64),
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )

    assert result["passed"] is True
    assert result["binding"]["virtual_source_spatial_weights_sha256"]
    assert result["binding"]["bilateral_assignment_weights_sha256"]
    assert result["binding"]["anchor_candidate_basis_mask_sha256"]
    assert torch.equal(covariances, before)
    assert all(
        "target" not in parameter and "scale" not in parameter
        for parameter in inspect.signature(audit_depthsplat_tile_support_basis).parameters
    )
