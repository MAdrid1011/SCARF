"""Source-only replay certificate for DepthSplat's soft S/R moment merge.

The paper defines a spatial virtual field followed by soft bilateral
assignment and first/second-moment matching. This module verifies that the
committed selected-anchor updates reproduce exactly those equations. It is
not a projected-domain containment guard, router, renderer, or target-side
metric surface.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch


SCHEMA_VERSION = "depthsplat-soft-mixture-moment-certificate-v1"
KIND = "depthsplat-source-only-soft-mixture-moment-replay"
POLICY = "selected-anchor-spatial-s-bilateral-r-exact-moment-replay-v1"
NUMERICAL_TOLERANCE_MULTIPLIER = 1024.0

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("soft-mixture certificate tensor digest requires a tensor")
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _valid_tile_key(value: Any) -> tuple[int, int, int] | None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        return None
    return tuple(int(item) for item in value)


def _matrix_summary(covariances: torch.Tensor) -> tuple[float | None, str | None]:
    if (
        not torch.is_tensor(covariances)
        or covariances.ndim != 3
        or covariances.shape[1:] != (3, 3)
    ):
        return None, "covariance-shape"
    symmetric = (covariances + covariances.mT) * 0.5
    try:
        eigenvalues = torch.linalg.eigvalsh(symmetric)
    except RuntimeError:
        return None, "covariance-eigendecomposition"
    if not bool(torch.isfinite(eigenvalues).all()):
        return None, "nonfinite-covariance"
    minimum = float(eigenvalues.min().item()) if eigenvalues.numel() else 0.0
    if minimum < -1e-6:
        return minimum, "non-psd-covariance"
    return minimum, None


def _residual(observed: torch.Tensor, expected: torch.Tensor) -> tuple[dict[str, float], bool]:
    if observed.shape != expected.shape:
        return {"maximum_absolute": float("inf"), "tolerance": 0.0}, False
    maximum = float((observed - expected).abs().amax().item()) if observed.numel() else 0.0
    scale = float(
        torch.maximum(observed.abs().amax(), expected.abs().amax()).item()
    ) if observed.numel() else 0.0
    tolerance = float(
        NUMERICAL_TOLERANCE_MULTIPLIER
        * torch.finfo(observed.dtype).eps
        * max(1.0, scale)
    )
    return {"maximum_absolute": maximum, "tolerance": tolerance}, maximum <= tolerance


def _source_only_metadata() -> dict[str, bool]:
    return {
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


def _failure(
    *, reason: str, tile_key: Any, anchor_count: int, virtual_count: int
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "policy": POLICY,
        "source_only": _source_only_metadata(),
        "tile_key": list(tile_key) if _valid_tile_key(tile_key) is not None else None,
        "passed": False,
        "summary": {
            "input_valid": False,
            "reason": reason,
            "anchor_count": anchor_count,
            "virtual_count": virtual_count,
            "normalizer_min": None,
            "normalizer_max": None,
            "minimum_source_covariance_eigenvalue": None,
            "minimum_virtual_covariance_eigenvalue": None,
            "minimum_merged_covariance_eigenvalue": None,
            "residuals": None,
        },
        "binding": None,
    }


def certify_depthsplat_tile_soft_mixture(
    *,
    tile_key: tuple[int, int, int],
    anchor_dense_slots: torch.Tensor,
    virtual_origin_slots: torch.Tensor,
    spatial_weights: torch.Tensor,
    bilateral_assignment_weights: torch.Tensor,
    anchor_source_means: torch.Tensor,
    anchor_source_covariances: torch.Tensor,
    anchor_source_harmonics: torch.Tensor,
    anchor_source_opacities: torch.Tensor,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_harmonics: torch.Tensor,
    virtual_opacities: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_harmonics: torch.Tensor,
    merged_opacities: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
) -> dict[str, Any]:
    """Replay the exact selected-anchor S/R soft-mixture equations.

    ``S`` builds the virtual field from selected anchors. ``R`` supplies the
    numerical contribution of every virtual to every retained output. No
    boolean owner partition or projected-domain proxy participates.
    """

    anchor_count = int(anchor_source_means.shape[0]) if torch.is_tensor(anchor_source_means) and anchor_source_means.ndim else 0
    virtual_count = int(virtual_means.shape[0]) if torch.is_tensor(virtual_means) and virtual_means.ndim else 0
    checked_tile_key = _valid_tile_key(tile_key)
    descriptor_tensors = (
        spatial_weights,
        bilateral_assignment_weights,
        anchor_source_means,
        anchor_source_covariances,
        anchor_source_harmonics,
        anchor_source_opacities,
        virtual_means,
        virtual_covariances,
        virtual_harmonics,
        virtual_opacities,
        merged_means,
        merged_covariances,
        merged_harmonics,
        merged_opacities,
        context_extrinsics,
        context_intrinsics,
    )
    if checked_tile_key is None or not all(
        torch.is_tensor(value) and torch.is_floating_point(value)
        for value in descriptor_tensors
    ):
        return _failure(
            reason="input-contract",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
        )
    if (
        anchor_count < 1
        or virtual_count < 1
        or anchor_source_means.shape != (anchor_count, 3)
        or anchor_source_covariances.shape != (anchor_count, 3, 3)
        or anchor_source_harmonics.ndim != 3
        or anchor_source_harmonics.shape[0] != anchor_count
        or anchor_source_opacities.shape not in {(anchor_count,), (anchor_count, 1)}
        or virtual_means.shape != (virtual_count, 3)
        or virtual_covariances.shape != (virtual_count, 3, 3)
        or virtual_harmonics.shape != (virtual_count, *anchor_source_harmonics.shape[1:])
        or virtual_opacities.shape not in {(virtual_count,), (virtual_count, 1)}
        or merged_means.shape != (anchor_count, 3)
        or merged_covariances.shape != (anchor_count, 3, 3)
        or merged_harmonics.shape != anchor_source_harmonics.shape
        or merged_opacities.shape not in {(anchor_count,), (anchor_count, 1)}
        or spatial_weights.shape != (virtual_count, anchor_count)
        or bilateral_assignment_weights.shape != (virtual_count, anchor_count)
        or context_extrinsics.shape != (anchor_count, 4, 4)
        or context_intrinsics.shape != (anchor_count, 3, 3)
        or not torch.is_tensor(anchor_dense_slots)
        or not torch.is_tensor(virtual_origin_slots)
        or anchor_dense_slots.shape != (anchor_count,)
        or virtual_origin_slots.shape != (virtual_count,)
        or anchor_dense_slots.dtype not in _INTEGER_DTYPES
        or virtual_origin_slots.dtype not in _INTEGER_DTYPES
    ):
        return _failure(
            reason="shape-contract",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
        )
    if any(
        value.device != anchor_source_means.device
        or value.dtype != anchor_source_means.dtype
        for value in descriptor_tensors
    ) or (
        anchor_dense_slots.device != anchor_source_means.device
        or virtual_origin_slots.device != anchor_source_means.device
    ):
        return _failure(
            reason="cross-family-contract",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
        )
    if not all(bool(torch.isfinite(value).all()) for value in descriptor_tensors):
        return _failure(
            reason="nonfinite-input",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
        )
    if (
        bool((spatial_weights < 0.0).any())
        or bool((bilateral_assignment_weights < 0.0).any())
        or not torch.allclose(
            spatial_weights.sum(dim=1),
            torch.ones(virtual_count, device=spatial_weights.device, dtype=spatial_weights.dtype),
            rtol=1e-5,
            atol=1e-5,
        )
        or not torch.allclose(
            bilateral_assignment_weights.sum(dim=1),
            torch.ones(virtual_count, device=bilateral_assignment_weights.device, dtype=bilateral_assignment_weights.dtype),
            rtol=1e-5,
            atol=1e-5,
        )
    ):
        return _failure(
            reason="soft-ledger-normalization",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
        )
    anchor_slots = anchor_dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    virtual_slots = virtual_origin_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if (
        len(set(anchor_slots)) != anchor_count
        or len(set(virtual_slots)) != virtual_count
        or set(anchor_slots) & set(virtual_slots)
        or not torch.equal(context_extrinsics, context_extrinsics[:1].expand_as(context_extrinsics))
        or not torch.equal(context_intrinsics, context_intrinsics[:1].expand_as(context_intrinsics))
    ):
        return _failure(
            reason="same-tile-binding",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
        )
    source_minimum, source_error = _matrix_summary(anchor_source_covariances)
    virtual_minimum, virtual_error = _matrix_summary(virtual_covariances)
    merged_minimum, merged_error = _matrix_summary(merged_covariances)
    if source_error or virtual_error or merged_error:
        return _failure(
            reason=source_error or virtual_error or merged_error or "covariance-contract",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
        )
    anchor_alpha = anchor_source_opacities.reshape(anchor_count)
    virtual_alpha = virtual_opacities.reshape(virtual_count)
    output_alpha = merged_opacities.reshape(anchor_count)
    if (
        bool((anchor_alpha < 0.0).any())
        or bool((virtual_alpha < 0.0).any())
        or bool((output_alpha < 0.0).any())
        or bool((anchor_alpha >= 1.0).any())
        or bool((virtual_alpha >= 1.0).any())
        or bool((output_alpha >= 1.0).any())
    ):
        return _failure(
            reason="alpha-contract",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
        )

    expected_virtual_means = spatial_weights @ anchor_source_means
    virtual_deltas = anchor_source_means.unsqueeze(0) - expected_virtual_means.unsqueeze(1)
    expected_virtual_covariances = (
        spatial_weights.reshape(virtual_count, anchor_count, 1, 1)
        * (
            anchor_source_covariances.unsqueeze(0)
            + virtual_deltas.unsqueeze(3) @ virtual_deltas.unsqueeze(2)
        )
    ).sum(dim=1)
    expected_virtual_covariances = (
        expected_virtual_covariances + expected_virtual_covariances.mT
    ) * 0.5
    expected_virtual_harmonics = (
        spatial_weights @ anchor_source_harmonics.reshape(anchor_count, -1)
    ).reshape_as(virtual_harmonics)
    expected_virtual_opacities = spatial_weights @ anchor_alpha

    normalizers = 1.0 + bilateral_assignment_weights.sum(dim=0)
    expected_merged_means = (
        anchor_source_means + bilateral_assignment_weights.mT @ virtual_means
    ) / normalizers.unsqueeze(1)
    source_deltas = anchor_source_means - expected_merged_means
    source_terms = anchor_source_covariances + (
        source_deltas.unsqueeze(2) @ source_deltas.unsqueeze(1)
    )
    virtual_to_output_deltas = (
        virtual_means.unsqueeze(1) - expected_merged_means.unsqueeze(0)
    )
    virtual_terms = virtual_covariances.unsqueeze(1) + (
        virtual_to_output_deltas.unsqueeze(3)
        @ virtual_to_output_deltas.unsqueeze(2)
    )
    expected_merged_covariances = (
        source_terms
        + (
            bilateral_assignment_weights.reshape(virtual_count, anchor_count, 1, 1)
            * virtual_terms
        ).sum(dim=0)
    ) / normalizers.reshape(anchor_count, 1, 1)
    expected_merged_covariances = (
        expected_merged_covariances + expected_merged_covariances.mT
    ) * 0.5
    expected_merged_harmonics = (
        anchor_source_harmonics.reshape(anchor_count, -1)
        + bilateral_assignment_weights.mT @ virtual_harmonics.reshape(virtual_count, -1)
    ) / normalizers.unsqueeze(1)
    expected_merged_harmonics = expected_merged_harmonics.reshape_as(merged_harmonics)
    expected_merged_opacities = (
        anchor_alpha + bilateral_assignment_weights.mT @ virtual_alpha
    ) / normalizers

    residuals: dict[str, dict[str, float]] = {}
    checks: list[bool] = []
    for name, observed, expected in (
        ("virtual_means", virtual_means, expected_virtual_means),
        ("virtual_covariances", virtual_covariances, expected_virtual_covariances),
        ("virtual_harmonics", virtual_harmonics, expected_virtual_harmonics),
        ("virtual_opacities", virtual_alpha, expected_virtual_opacities),
        ("merged_means", merged_means, expected_merged_means),
        ("merged_covariances", merged_covariances, expected_merged_covariances),
        ("merged_harmonics", merged_harmonics, expected_merged_harmonics),
        ("merged_opacities", output_alpha, expected_merged_opacities),
    ):
        residual, passed = _residual(observed, expected)
        residuals[name] = residual
        checks.append(passed)
    range_valid = (
        not bool((output_alpha < anchor_alpha.min() - 1e-6).any())
        and not bool((output_alpha > anchor_alpha.max() + 1e-6).any())
        and not bool(
            (merged_harmonics < anchor_source_harmonics.amin(dim=0) - 1e-5).any()
        )
        and not bool(
            (merged_harmonics > anchor_source_harmonics.amax(dim=0) + 1e-5).any()
        )
    )
    passed = all(checks) and range_valid and bool((normalizers >= 1.0).all())
    binding = {
        "tile_key_sha256": _canonical_sha256(list(checked_tile_key)),
        "anchor_dense_slots_sha256": _tensor_sha256(anchor_dense_slots),
        "virtual_origin_slots_sha256": _tensor_sha256(virtual_origin_slots),
        "spatial_weights_sha256": _tensor_sha256(spatial_weights),
        "bilateral_assignment_weights_sha256": _tensor_sha256(
            bilateral_assignment_weights
        ),
        "anchor_source_means_sha256": _tensor_sha256(anchor_source_means),
        "anchor_source_covariances_sha256": _tensor_sha256(anchor_source_covariances),
        "anchor_source_harmonics_sha256": _tensor_sha256(anchor_source_harmonics),
        "anchor_source_opacities_sha256": _tensor_sha256(anchor_alpha),
        "virtual_means_sha256": _tensor_sha256(virtual_means),
        "virtual_covariances_sha256": _tensor_sha256(virtual_covariances),
        "virtual_harmonics_sha256": _tensor_sha256(virtual_harmonics),
        "virtual_opacities_sha256": _tensor_sha256(virtual_alpha),
        "merged_means_sha256": _tensor_sha256(merged_means),
        "merged_covariances_sha256": _tensor_sha256(merged_covariances),
        "merged_harmonics_sha256": _tensor_sha256(merged_harmonics),
        "merged_opacities_sha256": _tensor_sha256(output_alpha),
        "context_extrinsics_sha256": _tensor_sha256(context_extrinsics),
        "context_intrinsics_sha256": _tensor_sha256(context_intrinsics),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "policy": POLICY,
        "source_only": _source_only_metadata(),
        "tile_key": list(checked_tile_key),
        "passed": passed,
        "summary": {
            "input_valid": True,
            "reason": None if passed else "soft-mixture-moment-replay-mismatch",
            "anchor_count": anchor_count,
            "virtual_count": virtual_count,
            "normalizer_min": float(normalizers.min().item()),
            "normalizer_max": float(normalizers.max().item()),
            "minimum_source_covariance_eigenvalue": source_minimum,
            "minimum_virtual_covariance_eigenvalue": virtual_minimum,
            "minimum_merged_covariance_eigenvalue": merged_minimum,
            "residuals": residuals,
            "range_valid": range_valid,
        },
        "binding": binding,
    }


__all__ = (
    "KIND",
    "POLICY",
    "SCHEMA_VERSION",
    "certify_depthsplat_tile_soft_mixture",
)
