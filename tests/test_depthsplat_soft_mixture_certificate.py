"""Focused contracts for source-only DepthSplat soft S/R replay."""

from __future__ import annotations

import copy

import pytest
import torch

from saes.depthsplat_soft_mixture_certificate import certify_depthsplat_tile_soft_mixture


def _inputs(*, assignment: torch.Tensor | None = None) -> dict[str, torch.Tensor | tuple[int, int, int]]:
    dtype = torch.float32
    means = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype)
    covariances = torch.eye(3, dtype=dtype).repeat(2, 1, 1) * 0.1
    harmonics = torch.tensor([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=dtype)
    opacities = torch.tensor([0.2, 0.4], dtype=dtype)
    spatial = torch.tensor([[0.5, 0.5]], dtype=dtype)
    bilateral = (
        assignment
        if assignment is not None
        else torch.tensor([[0.5, 0.5]], dtype=dtype)
    )
    virtual_means = spatial @ means
    virtual_deltas = means.unsqueeze(0) - virtual_means.unsqueeze(1)
    virtual_covariances = (
        spatial[:, :, None, None]
        * (
            covariances.unsqueeze(0)
            + virtual_deltas.unsqueeze(3) @ virtual_deltas.unsqueeze(2)
        )
    ).sum(dim=1)
    virtual_harmonics = (spatial @ harmonics.reshape(2, -1)).reshape_as(harmonics[:1])
    virtual_opacities = spatial @ opacities
    normalizers = 1.0 + bilateral.sum(dim=0)
    merged_means = (means + bilateral.mT @ virtual_means) / normalizers[:, None]
    source_deltas = means - merged_means
    source_terms = covariances + source_deltas.unsqueeze(2) @ source_deltas.unsqueeze(1)
    virtual_deltas = virtual_means.unsqueeze(1) - merged_means.unsqueeze(0)
    virtual_terms = virtual_covariances.unsqueeze(1) + (
        virtual_deltas.unsqueeze(3) @ virtual_deltas.unsqueeze(2)
    )
    merged_covariances = (
        source_terms + (bilateral[:, :, None, None] * virtual_terms).sum(dim=0)
    ) / normalizers[:, None, None]
    merged_harmonics = (
        harmonics.reshape(2, -1) + bilateral.mT @ virtual_harmonics.reshape(1, -1)
    ) / normalizers[:, None]
    merged_opacities = (opacities + bilateral.mT @ virtual_opacities) / normalizers
    return {
        "tile_key": (0, 0, 0),
        "anchor_dense_slots": torch.tensor([0, 1], dtype=torch.int64),
        "virtual_origin_slots": torch.tensor([2], dtype=torch.int64),
        "spatial_weights": spatial,
        "bilateral_assignment_weights": bilateral,
        "anchor_source_means": means,
        "anchor_source_covariances": covariances,
        "anchor_source_harmonics": harmonics,
        "anchor_source_opacities": opacities,
        "virtual_means": virtual_means,
        "virtual_covariances": virtual_covariances,
        "virtual_harmonics": virtual_harmonics,
        "virtual_opacities": virtual_opacities,
        "merged_means": merged_means,
        "merged_covariances": merged_covariances,
        "merged_harmonics": merged_harmonics.reshape_as(harmonics),
        "merged_opacities": merged_opacities,
        "context_extrinsics": torch.eye(4, dtype=dtype).repeat(2, 1, 1),
        "context_intrinsics": torch.eye(3, dtype=dtype).repeat(2, 1, 1),
    }


def _certificate(values: dict[str, torch.Tensor | tuple[int, int, int]]) -> dict[str, object]:
    return certify_depthsplat_tile_soft_mixture(**values)  # type: ignore[arg-type]


def test_soft_mixture_replays_exact_s_and_r_moments() -> None:
    certificate = _certificate(_inputs())

    assert certificate["passed"] is True
    assert certificate["source_only"] == {
        "source_camera_only": True,
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "omitted_s3_attributes_accessed": False,
        "input_covariances_mutated": False,
        "fixed_covariance_scale": True,
        "boolean_owner_assignment_used": False,
        "projected_domain_guard_used": False,
    }


@pytest.mark.parametrize(
    "field",
    (
        "spatial_weights",
        "bilateral_assignment_weights",
        "virtual_means",
        "virtual_covariances",
        "virtual_harmonics",
        "virtual_opacities",
        "merged_means",
        "merged_covariances",
        "merged_harmonics",
        "merged_opacities",
    ),
)
def test_soft_mixture_rejects_replay_tampering(field: str) -> None:
    values = _inputs()
    tampered = copy.copy(values)
    value = tampered[field]
    assert torch.is_tensor(value)
    value = value.clone()
    value.reshape(-1)[0] += 0.05
    tampered[field] = value

    certificate = _certificate(tampered)

    assert certificate["passed"] is False


def test_soft_mixture_replays_tiny_nonzero_bilateral_contributor_numerically() -> None:
    assignment = torch.tensor([[1.0e-8, 1.0 - 1.0e-8]], dtype=torch.float32)
    certificate = _certificate(_inputs(assignment=assignment))

    assert certificate["passed"] is True
    binding = certificate["binding"]
    assert isinstance(binding, dict)
    assert binding["bilateral_assignment_weights_sha256"]


def test_soft_mixture_rejects_tile_camera_or_psd_tampering() -> None:
    values = _inputs()
    camera_tampered = copy.copy(values)
    cameras = values["context_extrinsics"]
    assert torch.is_tensor(cameras)
    camera_tampered["context_extrinsics"] = cameras.clone()
    camera_tampered["context_extrinsics"][1, 0, 3] = 1.0
    assert _certificate(camera_tampered)["passed"] is False

    psd_tampered = copy.copy(values)
    covariances = values["merged_covariances"]
    assert torch.is_tensor(covariances)
    psd_tampered["merged_covariances"] = covariances.clone()
    psd_tampered["merged_covariances"][0, 0, 0] = -1.0
    assert _certificate(psd_tampered)["passed"] is False
