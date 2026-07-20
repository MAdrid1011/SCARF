"""Focused contracts for the source-only DepthSplat mixture kernel guard."""

from __future__ import annotations

import copy

import pytest
import torch

from saes.depthsplat_mixture_kernel_guard import (
    assess_depthsplat_tile_kernel_closure,
)


def _inputs(
    *,
    anchor_means: list[list[float]],
    spatial_weights: list[list[float]],
    bilateral_assignment_weights: list[list[float]],
    covariance_scale: float = 0.03,
    opacity: float = 0.5,
) -> dict[str, torch.Tensor | tuple[int, int, int]]:
    """Build an exact S/R moment replay fixture for the kernel-only guard."""

    dtype = torch.float64
    means = torch.tensor(anchor_means, dtype=dtype)
    spatial = torch.tensor(spatial_weights, dtype=dtype)
    bilateral = torch.tensor(bilateral_assignment_weights, dtype=dtype)
    anchor_count = int(means.shape[0])
    virtual_count = int(spatial.shape[0])
    covariances = (
        torch.eye(3, dtype=dtype).repeat(anchor_count, 1, 1) * covariance_scale
    )
    anchor_opacities = torch.full((anchor_count,), opacity, dtype=dtype)

    virtual_means = spatial @ means
    virtual_deltas = means.unsqueeze(0) - virtual_means.unsqueeze(1)
    virtual_covariances = (
        spatial[:, :, None, None]
        * (
            covariances.unsqueeze(0)
            + virtual_deltas.unsqueeze(-1) @ virtual_deltas.unsqueeze(-2)
        )
    ).sum(dim=1)
    virtual_opacities = spatial @ anchor_opacities

    normalizers = 1.0 + bilateral.sum(dim=0)
    merged_means = (
        means + bilateral.mT @ virtual_means
    ) / normalizers.unsqueeze(1)
    source_deltas = means - merged_means
    source_terms = covariances + (
        source_deltas.unsqueeze(-1) @ source_deltas.unsqueeze(-2)
    )
    virtual_to_output_deltas = (
        virtual_means.unsqueeze(1) - merged_means.unsqueeze(0)
    )
    virtual_terms = virtual_covariances.unsqueeze(1) + (
        virtual_to_output_deltas.unsqueeze(-1)
        @ virtual_to_output_deltas.unsqueeze(-2)
    )
    merged_covariances = (
        source_terms
        + (bilateral[:, :, None, None] * virtual_terms).sum(dim=0)
    ) / normalizers[:, None, None]
    merged_covariances = (merged_covariances + merged_covariances.mT) * 0.5
    merged_opacities = (
        anchor_opacities + bilateral.mT @ virtual_opacities
    ) / normalizers

    return {
        "tile_key": (0, 0, 0),
        "anchor_dense_slots": torch.arange(anchor_count, dtype=torch.int64),
        "virtual_origin_slots": torch.arange(
            anchor_count, anchor_count + virtual_count, dtype=torch.int64
        ),
        "bilateral_assignment_weights": bilateral,
        "anchor_source_means": means,
        "anchor_source_covariances": covariances,
        "anchor_source_opacities": anchor_opacities,
        "virtual_means": virtual_means,
        "virtual_covariances": virtual_covariances,
        "virtual_opacities": virtual_opacities,
        "merged_means": merged_means,
        "merged_covariances": merged_covariances,
        "merged_opacities": merged_opacities,
        "context_extrinsics": torch.eye(4, dtype=dtype).repeat(
            anchor_count, 1, 1
        ),
        "context_intrinsics": torch.eye(3, dtype=dtype).repeat(
            anchor_count, 1, 1
        ),
    }


def _assess(
    values: dict[str, torch.Tensor | tuple[int, int, int]], *, strict: float
) -> dict[str, object]:
    return assess_depthsplat_tile_kernel_closure(
        **values,  # type: ignore[arg-type]
        strict_maximum_relative_risk=strict,
    )


def _per_output_risks(assessment: dict[str, object]) -> list[dict[str, object]]:
    summary = assessment["summary"]
    assert isinstance(summary, dict)
    risks = summary["per_output"]
    assert isinstance(risks, list)
    assert all(isinstance(record, dict) for record in risks)
    return risks  # type: ignore[return-value]


def test_one_component_exact_kernel_closure_passes_with_near_zero_risk() -> None:
    assessment = _assess(
        _inputs(
            anchor_means=[[0.0, 0.0, 3.0]],
            spatial_weights=[[1.0]],
            bilateral_assignment_weights=[[1.0]],
        ),
        strict=1.0e-7,
    )

    assert assessment["passed"] is True
    summary = assessment["summary"]
    assert isinstance(summary, dict)
    assert summary["maximum_kernel_risk"] <= 1.0e-9
    assert summary["maximum_world_kernel_risk"] <= 1.0e-9
    assert summary["maximum_source_kernel_risk"] <= 1.0e-9
    assert _per_output_risks(assessment)[0]["passed"] is True


def test_symmetric_separated_two_component_mixture_fails_strict_closure() -> None:
    # Output zero becomes the equal-weight mixture of these separated anchors.
    assessment = _assess(
        _inputs(
            anchor_means=[[-3.0, 0.0, 4.0], [3.0, 0.0, 4.0]],
            spatial_weights=[[0.0, 1.0]],
            bilateral_assignment_weights=[[1.0, 0.0]],
        ),
        strict=1.0e-6,
    )

    assert assessment["passed"] is False
    summary = assessment["summary"]
    assert isinstance(summary, dict)
    assert summary["reason"] == "kernel-closure-risk-exceeds-strict-limit"
    assert summary["maximum_kernel_risk"] > 0.1
    assert _per_output_risks(assessment)[0]["passed"] is False


def test_virtual_contributor_permutation_preserves_kernel_risk() -> None:
    values = _inputs(
        anchor_means=[[-2.0, 0.0, 3.0], [2.0, 0.0, 3.0]],
        spatial_weights=[[1.0, 0.0], [0.0, 1.0]],
        bilateral_assignment_weights=[[0.25, 0.75], [0.75, 0.25]],
        covariance_scale=0.1,
        opacity=0.4,
    )
    permuted = copy.copy(values)
    permutation = torch.tensor([1, 0], dtype=torch.long)
    for field in (
        "virtual_origin_slots",
        "bilateral_assignment_weights",
        "virtual_means",
        "virtual_covariances",
        "virtual_opacities",
    ):
        value = values[field]
        assert torch.is_tensor(value)
        permuted[field] = value.index_select(0, permutation)

    baseline = _assess(values, strict=10.0)
    replayed = _assess(permuted, strict=10.0)

    assert baseline["passed"] is True
    assert replayed["passed"] is True
    baseline_summary = baseline["summary"]
    replayed_summary = replayed["summary"]
    assert isinstance(baseline_summary, dict)
    assert isinstance(replayed_summary, dict)
    for field in (
        "maximum_kernel_risk",
        "maximum_world_kernel_risk",
        "maximum_source_kernel_risk",
    ):
        assert replayed_summary[field] == pytest.approx(
            baseline_summary[field], rel=1.0e-12, abs=1.0e-12
        )
    for baseline_output, replayed_output in zip(
        _per_output_risks(baseline), _per_output_risks(replayed), strict=True
    ):
        assert replayed_output["world_kernel_risk"] == pytest.approx(
            baseline_output["world_kernel_risk"], rel=1.0e-12, abs=1.0e-12
        )
        assert replayed_output["source_kernel_risk"] == pytest.approx(
            baseline_output["source_kernel_risk"], rel=1.0e-12, abs=1.0e-12
        )


def test_tiny_positive_bilateral_weight_changes_risk_numerically() -> None:
    common = {
        "anchor_means": [[-3.0, 0.0, 4.0], [3.0, 0.0, 4.0]],
        "spatial_weights": [[0.0, 1.0]],
    }
    no_contribution = _assess(
        _inputs(**common, bilateral_assignment_weights=[[0.0, 1.0]]),
        strict=10.0,
    )
    tiny_contribution = _assess(
        _inputs(**common, bilateral_assignment_weights=[[1.0e-6, 1.0 - 1.0e-6]]),
        strict=10.0,
    )

    assert no_contribution["passed"] is True
    assert tiny_contribution["passed"] is True
    baseline_risk = _per_output_risks(no_contribution)[0]["maximum_kernel_risk"]
    tiny_risk = _per_output_risks(tiny_contribution)[0]["maximum_kernel_risk"]
    assert baseline_risk == pytest.approx(0.0, abs=1.0e-9)
    assert tiny_risk > 1.0e-4


def test_malformed_nonfinite_or_invalid_camera_fails_closed() -> None:
    valid = _inputs(
        anchor_means=[[0.0, 0.0, 3.0]],
        spatial_weights=[[1.0]],
        bilateral_assignment_weights=[[1.0]],
    )
    malformed = copy.copy(valid)
    malformed["bilateral_assignment_weights"] = torch.ones((1, 2), dtype=torch.float64)
    nonfinite = copy.copy(valid)
    means = valid["anchor_source_means"]
    assert torch.is_tensor(means)
    nonfinite["anchor_source_means"] = means.clone()
    nonfinite["anchor_source_means"][0, 0] = float("nan")
    invalid_camera = copy.copy(valid)
    intrinsics = valid["context_intrinsics"]
    assert torch.is_tensor(intrinsics)
    invalid_camera["context_intrinsics"] = intrinsics.clone()
    invalid_camera["context_intrinsics"][0, 2, 2] = 0.0

    for values in (malformed, nonfinite, invalid_camera):
        assessment = _assess(values, strict=1.0e-7)
        assert assessment["passed"] is False
        summary = assessment["summary"]
        assert isinstance(summary, dict)
        assert summary["input_valid"] is False
        assert summary["maximum_kernel_risk"] is None


def test_nonfinite_analytic_kernel_risk_is_an_unscorable_full_promotion(monkeypatch) -> None:
    import saes.depthsplat_mixture_kernel_guard as kernel_guard

    monkeypatch.setattr(
        kernel_guard, "_relative_kernel_risk", lambda **_kwargs: float("inf")
    )
    assessment = kernel_guard.assess_depthsplat_tile_kernel_closure(
        **_inputs(
            anchor_means=[[0.0, 0.0, 3.0]],
            spatial_weights=[[1.0]],
            bilateral_assignment_weights=[[1.0]],
        ),
        strict_maximum_relative_risk=1.0e-7,
    )

    assert assessment["passed"] is False
    assert assessment["binding"] is None
    summary = assessment["summary"]
    assert isinstance(summary, dict)
    assert summary["input_valid"] is False
    assert summary["reason"] == "kernel-integral"
    assert summary["maximum_kernel_risk"] is None
