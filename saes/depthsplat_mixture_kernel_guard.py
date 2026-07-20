"""Source-only kernel-closure guard for DepthSplat soft mixture updates."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

from saes.projected_optical_moment_audit import (
    _project_context_points,
    _valid_context_geometry,
)


SCHEMA_VERSION = "depthsplat-mixture-kernel-closure-v1"
KIND = "depthsplat-source-only-mixture-kernel-closure"
POLICY = "actual-anchor-virtual-mixture-analytic-l2-abstention-v1"
NUMERICAL_CLOSURE_TOLERANCE_MULTIPLIER = 1024.0

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_only_metadata() -> dict[str, bool]:
    return {
        "source_camera_only": True,
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "omitted_s3_attributes_accessed": False,
        "boolean_owner_assignment_used": False,
        "projected_domain_guard_used": False,
        "covariance_expansion_used": False,
        "alpha_union_used": False,
    }


def _failure(
    *,
    reason: str,
    tile_key: Any,
    anchor_count: int,
    virtual_count: int,
    strict_maximum_relative_risk: float | None,
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
            "strict_maximum_relative_risk": strict_maximum_relative_risk,
            "maximum_world_kernel_risk": None,
            "maximum_source_kernel_risk": None,
            "maximum_kernel_risk": None,
            "maximum_source_log_depth_rms": None,
            "per_output": [],
        },
        "binding": None,
    }


def _valid_tile_key(value: Any) -> tuple[int, int, int] | None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        return None
    return tuple(int(item) for item in value)


def _flatten_alpha(value: torch.Tensor, count: int) -> torch.Tensor | None:
    if value.shape == (count,):
        return value
    if value.shape == (count, 1):
        return value[:, 0]
    return None


def _stabilize_spd(
    covariances: torch.Tensor,
) -> tuple[torch.Tensor | None, float | None, str | None]:
    """Return a local SPD view without mutating the bound input tensor."""

    symmetric = (covariances + covariances.mT) * 0.5
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    except RuntimeError:
        return None, None, "covariance-eigendecomposition"
    if not bool(torch.isfinite(eigenvalues).all()):
        return None, None, "nonfinite-covariance"
    scale = torch.maximum(symmetric.abs().amax(), symmetric.new_ones(()))
    tolerance = NUMERICAL_CLOSURE_TOLERANCE_MULTIPLIER * torch.finfo(
        symmetric.dtype
    ).eps * scale
    if bool((eigenvalues < -tolerance).any()):
        return None, None, "non-psd-covariance"
    floor = torch.finfo(symmetric.dtype).eps * scale
    stabilized_values = eigenvalues.clamp_min(floor)
    stabilized = eigenvectors @ torch.diag_embed(stabilized_values) @ eigenvectors.mT
    return (stabilized + stabilized.mT) * 0.5, float(floor.item()), None


def _gaussian_gram(
    first_means: torch.Tensor,
    first_covariances: torch.Tensor,
    second_means: torch.Tensor,
    second_covariances: torch.Tensor,
) -> torch.Tensor | None:
    """Integrate pairs of unnormalized Gaussian kernels analytically."""

    dimension = first_means.shape[1]
    sums = first_covariances[:, None] + second_covariances[None, :]
    deltas = first_means[:, None] - second_means[None, :]
    try:
        sign_first, logdet_first = torch.linalg.slogdet(first_covariances)
        sign_second, logdet_second = torch.linalg.slogdet(second_covariances)
        sign_sum, logdet_sum = torch.linalg.slogdet(sums)
        solved = torch.linalg.solve(sums, deltas.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        return None
    quadratic = (deltas * solved).sum(dim=-1)
    if (
        bool((sign_first <= 0.0).any())
        or bool((sign_second <= 0.0).any())
        or bool((sign_sum <= 0.0).any())
        or not bool(torch.isfinite(quadratic).all())
    ):
        return None
    log_integral = (
        0.5 * float(dimension) * math.log(2.0 * math.pi)
        + 0.5 * (logdet_first[:, None] + logdet_second[None, :] - logdet_sum)
        - 0.5 * quadratic
    )
    integral = torch.exp(log_integral)
    return integral if bool(torch.isfinite(integral).all()) else None


def _relative_kernel_risk(
    *,
    component_means: torch.Tensor,
    component_covariances: torch.Tensor,
    component_coefficients: torch.Tensor,
    merged_mean: torch.Tensor,
    merged_covariance: torch.Tensor,
    merged_coefficient: torch.Tensor,
) -> float | None:
    if bool((component_coefficients < 0.0).any()) or bool(merged_coefficient < 0.0):
        return None
    merged_means = merged_mean.unsqueeze(0)
    merged_covariances = merged_covariance.unsqueeze(0)
    component_gram = _gaussian_gram(
        component_means,
        component_covariances,
        component_means,
        component_covariances,
    )
    cross_gram = _gaussian_gram(
        component_means,
        component_covariances,
        merged_means,
        merged_covariances,
    )
    output_gram = _gaussian_gram(
        merged_means,
        merged_covariances,
        merged_means,
        merged_covariances,
    )
    if component_gram is None or cross_gram is None or output_gram is None:
        return None
    epp = torch.einsum("i,ij,j->", component_coefficients, component_gram, component_coefficients)
    epq = torch.dot(component_coefficients, cross_gram[:, 0]) * merged_coefficient
    eqq = output_gram[0, 0] * merged_coefficient.square()
    if not bool(torch.isfinite(torch.stack((epp, epq, eqq))).all()):
        return None
    numerical_floor = torch.finfo(component_means.dtype).eps
    if bool(epp <= numerical_floor):
        return 0.0 if bool(eqq <= numerical_floor) else float("inf")
    residual = (epp - 2.0 * epq + eqq).clamp_min(0.0)
    value = (residual / epp).sqrt()
    return float(value.item()) if bool(torch.isfinite(value)) else None


def _source_projection(
    means: torch.Tensor,
    covariances: torch.Tensor,
    *,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, str]:
    projected = _project_context_points(
        means,
        context_extrinsic=extrinsic,
        context_intrinsic=intrinsic,
    )
    if projected[0] is None:
        return None, None, str(projected[1])
    centers, jacobians = projected
    projected_covariances = jacobians @ covariances @ jacobians.mT
    projected_covariances, _floor, error = _stabilize_spd(projected_covariances)
    if error is not None or projected_covariances is None:
        return None, None, f"projected:{error or 'covariance'}"
    rotation = extrinsic[:3, :3]
    origin = extrinsic[:3, 3]
    depths = ((means - origin) @ rotation)[:, 2]
    if not bool(torch.isfinite(depths).all()) or bool((depths <= 0.0).any()):
        return None, None, "nonpositive-source-depth"
    return centers, projected_covariances, depths


def assess_depthsplat_tile_kernel_closure(
    *,
    tile_key: tuple[int, int, int],
    anchor_dense_slots: torch.Tensor,
    virtual_origin_slots: torch.Tensor,
    bilateral_assignment_weights: torch.Tensor,
    anchor_source_means: torch.Tensor,
    anchor_source_covariances: torch.Tensor,
    anchor_source_opacities: torch.Tensor,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_opacities: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_opacities: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    strict_maximum_relative_risk: float | None = None,
) -> dict[str, Any]:
    """Assess whether each actual anchor-plus-virtual mixture closes exactly.

    The contributor field mirrors the materializer's current order: the
    retained source anchor followed by the already S-materialized virtual
    Gaussians. The guard is source-only and performs no sampling or rendering.
    """

    anchor_count = (
        int(anchor_source_means.shape[0])
        if torch.is_tensor(anchor_source_means) and anchor_source_means.ndim == 2
        else 0
    )
    virtual_count = (
        int(virtual_means.shape[0])
        if torch.is_tensor(virtual_means) and virtual_means.ndim == 2
        else 0
    )
    checked_tile_key = _valid_tile_key(tile_key)
    if strict_maximum_relative_risk is None:
        strict_maximum_relative_risk = float(
            NUMERICAL_CLOSURE_TOLERANCE_MULTIPLIER
            * torch.finfo(anchor_source_means.dtype).eps
        ) if torch.is_tensor(anchor_source_means) and torch.is_floating_point(anchor_source_means) else None
    if (
        isinstance(strict_maximum_relative_risk, bool)
        or not isinstance(strict_maximum_relative_risk, (int, float))
        or not math.isfinite(float(strict_maximum_relative_risk))
        or float(strict_maximum_relative_risk) < 0.0
    ):
        return _failure(
            reason="risk-threshold",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
            strict_maximum_relative_risk=None,
        )
    threshold = float(strict_maximum_relative_risk)
    tensors = (
        bilateral_assignment_weights,
        anchor_source_means,
        anchor_source_covariances,
        anchor_source_opacities,
        virtual_means,
        virtual_covariances,
        virtual_opacities,
        merged_means,
        merged_covariances,
        merged_opacities,
        context_extrinsics,
        context_intrinsics,
    )
    if checked_tile_key is None or not all(
        torch.is_tensor(value) and torch.is_floating_point(value) for value in tensors
    ):
        return _failure(
            reason="input-contract",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
            strict_maximum_relative_risk=threshold,
        )
    anchor_alpha = _flatten_alpha(anchor_source_opacities, anchor_count)
    virtual_alpha = _flatten_alpha(virtual_opacities, virtual_count)
    merged_alpha = _flatten_alpha(merged_opacities, anchor_count)
    if (
        anchor_count < 1
        or virtual_count < 1
        or anchor_alpha is None
        or virtual_alpha is None
        or merged_alpha is None
        or anchor_source_means.shape != (anchor_count, 3)
        or anchor_source_covariances.shape != (anchor_count, 3, 3)
        or virtual_means.shape != (virtual_count, 3)
        or virtual_covariances.shape != (virtual_count, 3, 3)
        or merged_means.shape != (anchor_count, 3)
        or merged_covariances.shape != (anchor_count, 3, 3)
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
            strict_maximum_relative_risk=threshold,
        )
    if any(
        value.device != anchor_source_means.device
        or value.dtype != anchor_source_means.dtype
        for value in tensors
    ) or (
        anchor_dense_slots.device != anchor_source_means.device
        or virtual_origin_slots.device != anchor_source_means.device
    ):
        return _failure(
            reason="cross-family-contract",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
            strict_maximum_relative_risk=threshold,
        )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        return _failure(
            reason="nonfinite-input",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
            strict_maximum_relative_risk=threshold,
        )
    if (
        bool((anchor_alpha < 0.0).any())
        or bool((virtual_alpha < 0.0).any())
        or bool((merged_alpha < 0.0).any())
        or bool((anchor_alpha >= 1.0).any())
        or bool((virtual_alpha >= 1.0).any())
        or bool((merged_alpha >= 1.0).any())
        or bool((bilateral_assignment_weights < 0.0).any())
        or not torch.allclose(
            bilateral_assignment_weights.sum(dim=1),
            torch.ones(
                virtual_count,
                device=anchor_source_means.device,
                dtype=anchor_source_means.dtype,
            ),
            rtol=1e-5,
            atol=1e-5,
        )
    ):
        return _failure(
            reason="soft-ledger-contract",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
            strict_maximum_relative_risk=threshold,
        )
    anchor_slots = anchor_dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    virtual_slots = virtual_origin_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if (
        len(set(anchor_slots)) != anchor_count
        or len(set(virtual_slots)) != virtual_count
        or set(anchor_slots) & set(virtual_slots)
        or not torch.equal(context_extrinsics, context_extrinsics[:1].expand_as(context_extrinsics))
        or not torch.equal(context_intrinsics, context_intrinsics[:1].expand_as(context_intrinsics))
        or not _valid_context_geometry(context_extrinsics[0], context_intrinsics[0])
    ):
        return _failure(
            reason="same-tile-camera-binding",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
            strict_maximum_relative_risk=threshold,
        )
    source_covariances, source_floor, source_error = _stabilize_spd(
        anchor_source_covariances
    )
    virtual_covariances_checked, virtual_floor, virtual_error = _stabilize_spd(
        virtual_covariances
    )
    merged_covariances_checked, merged_floor, merged_error = _stabilize_spd(
        merged_covariances
    )
    if (
        source_error is not None
        or virtual_error is not None
        or merged_error is not None
        or source_covariances is None
        or virtual_covariances_checked is None
        or merged_covariances_checked is None
    ):
        return _failure(
            reason=source_error or virtual_error or merged_error or "covariance-contract",
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
            strict_maximum_relative_risk=threshold,
        )
    all_source_means = torch.cat((anchor_source_means, virtual_means), dim=0)
    all_source_covariances = torch.cat(
        (source_covariances, virtual_covariances_checked), dim=0
    )
    source_projection = _source_projection(
        all_source_means,
        all_source_covariances,
        extrinsic=context_extrinsics[0],
        intrinsic=context_intrinsics[0],
    )
    merged_projection = _source_projection(
        merged_means,
        merged_covariances_checked,
        extrinsic=context_extrinsics[0],
        intrinsic=context_intrinsics[0],
    )
    if source_projection[0] is None or merged_projection[0] is None:
        return _failure(
            reason=str(source_projection[2] if source_projection[0] is None else merged_projection[2]),
            tile_key=tile_key,
            anchor_count=anchor_count,
            virtual_count=virtual_count,
            strict_maximum_relative_risk=threshold,
        )
    source_centers, source_projected_covariances, source_depths = source_projection
    merged_centers, merged_projected_covariances, merged_depths = merged_projection
    output_records: list[dict[str, Any]] = []
    maximum_world_risk = 0.0
    maximum_source_risk = 0.0
    maximum_depth_rms = 0.0
    guard_passed = True
    for output_index in range(anchor_count):
        masses = torch.cat(
            (
                torch.ones(1, device=anchor_source_means.device, dtype=anchor_source_means.dtype),
                bilateral_assignment_weights[:, output_index],
            )
        )
        normalizer = masses.sum()
        if not bool(torch.isfinite(normalizer)) or bool(normalizer <= 0.0):
            return _failure(
                reason="invalid-normalizer",
                tile_key=tile_key,
                anchor_count=anchor_count,
                virtual_count=virtual_count,
                strict_maximum_relative_risk=threshold,
            )
        component_indices = torch.cat(
            (
                torch.tensor(
                    [output_index], device=anchor_source_means.device, dtype=torch.long
                ),
                torch.arange(
                    anchor_count,
                    anchor_count + virtual_count,
                    device=anchor_source_means.device,
                    dtype=torch.long,
                ),
            )
        )
        component_alpha = torch.cat(
            (anchor_alpha[output_index : output_index + 1], virtual_alpha)
        )
        coefficients = masses * component_alpha / normalizer
        world_risk = _relative_kernel_risk(
            component_means=all_source_means[component_indices],
            component_covariances=all_source_covariances[component_indices],
            component_coefficients=coefficients,
            merged_mean=merged_means[output_index],
            merged_covariance=merged_covariances_checked[output_index],
            merged_coefficient=merged_alpha[output_index],
        )
        source_risk = _relative_kernel_risk(
            component_means=source_centers[component_indices],
            component_covariances=source_projected_covariances[component_indices],
            component_coefficients=coefficients,
            merged_mean=merged_centers[output_index],
            merged_covariance=merged_projected_covariances[output_index],
            merged_coefficient=merged_alpha[output_index],
        )
        log_depths = source_depths[component_indices].log()
        depth_rms = (
            (masses / normalizer * (log_depths - merged_depths[output_index].log()).square())
            .sum()
            .clamp_min(0.0)
            .sqrt()
        )
        if (
            world_risk is None
            or source_risk is None
            or not math.isfinite(world_risk)
            or not math.isfinite(source_risk)
            or not bool(torch.isfinite(depth_rms))
        ):
            return _failure(
                reason="kernel-integral",
                tile_key=tile_key,
                anchor_count=anchor_count,
                virtual_count=virtual_count,
                strict_maximum_relative_risk=threshold,
            )
        maximum_world_risk = max(maximum_world_risk, world_risk)
        maximum_source_risk = max(maximum_source_risk, source_risk)
        maximum_depth_rms = max(maximum_depth_rms, float(depth_rms.item()))
        output_risk = max(world_risk, source_risk)
        output_passed = output_risk <= threshold
        guard_passed = guard_passed and output_passed
        output_records.append(
            {
                "output_index": output_index,
                "normalizer": float(normalizer.item()),
                "world_kernel_risk": world_risk,
                "source_kernel_risk": source_risk,
                "maximum_kernel_risk": output_risk,
                "source_log_depth_rms": float(depth_rms.item()),
                "passed": output_passed,
            }
        )
    maximum_risk = max(maximum_world_risk, maximum_source_risk)
    binding = {
        "tile_key_sha256": _canonical_sha256(list(checked_tile_key)),
        "anchor_dense_slots_sha256": _tensor_sha256(anchor_dense_slots),
        "virtual_origin_slots_sha256": _tensor_sha256(virtual_origin_slots),
        "bilateral_assignment_weights_sha256": _tensor_sha256(
            bilateral_assignment_weights
        ),
        "anchor_source_means_sha256": _tensor_sha256(anchor_source_means),
        "anchor_source_covariances_sha256": _tensor_sha256(anchor_source_covariances),
        "anchor_source_opacities_sha256": _tensor_sha256(anchor_alpha),
        "virtual_means_sha256": _tensor_sha256(virtual_means),
        "virtual_covariances_sha256": _tensor_sha256(virtual_covariances),
        "virtual_opacities_sha256": _tensor_sha256(virtual_alpha),
        "merged_means_sha256": _tensor_sha256(merged_means),
        "merged_covariances_sha256": _tensor_sha256(merged_covariances),
        "merged_opacities_sha256": _tensor_sha256(merged_alpha),
        "context_extrinsics_sha256": _tensor_sha256(context_extrinsics),
        "context_intrinsics_sha256": _tensor_sha256(context_intrinsics),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "policy": POLICY,
        "source_only": _source_only_metadata(),
        "tile_key": list(checked_tile_key),
        "passed": guard_passed,
        "summary": {
            "input_valid": True,
            "reason": None if guard_passed else "kernel-closure-risk-exceeds-strict-limit",
            "anchor_count": anchor_count,
            "virtual_count": virtual_count,
            "strict_maximum_relative_risk": threshold,
            "maximum_world_kernel_risk": maximum_world_risk,
            "maximum_source_kernel_risk": maximum_source_risk,
            "maximum_kernel_risk": maximum_risk,
            "maximum_source_log_depth_rms": maximum_depth_rms,
            "covariance_numerical_floors": {
                "source": source_floor,
                "virtual": virtual_floor,
                "merged": merged_floor,
            },
            "per_output": output_records,
        },
        "binding": binding,
    }


__all__ = (
    "KIND",
    "NUMERICAL_CLOSURE_TOLERANCE_MULTIPLIER",
    "POLICY",
    "SCHEMA_VERSION",
    "assess_depthsplat_tile_kernel_closure",
)
