"""Source-only same-tile support-basis coverage for DepthSplat.

DepthSplat materializes every selected anchor from the same virtual field via
the paper's spatial and soft-bilateral ledgers.  A post-merge guard must
therefore certify the retained tile as a local basis, rather than inventing a
single hard owner for each virtual primitive.  This module checks each source
anchor and selected-only virtual two-sigma ellipse against the committed
merged anchors reachable through those existing ledgers.  It never routes,
mutates descriptors, reads target state, or accepts a covariance scale.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from saes.projected_domain_coverage_audit import (
    FIXED_SIGMA,
    ProjectedDomainCoverage,
    audit_projected_dense_domain_coverage,
)


AUDIT_SCHEMA_VERSION = "depthsplat-tile-support-basis-coverage-audit-v1"
AUDIT_KIND = "depthsplat-source-only-same-tile-support-basis-coverage"
SUPPORT_BASIS_POLICY = "same-tile-composed-soft-ledger-support-v1"

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _descriptor_count(value: object) -> int:
    if torch.is_tensor(value) and value.ndim >= 1:
        return int(value.shape[0])
    return 0


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash an immutable tensor binding without coercing its values."""

    if not torch.is_tensor(value):
        raise TypeError("DepthSplat support-basis tensor digest requires a tensor")
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _coverage_payload(coverage: ProjectedDomainCoverage) -> dict[str, Any]:
    return {
        "valid": coverage.valid,
        "reason": coverage.reason,
        "schema_version": coverage.schema_version,
        "sigma": coverage.sigma,
        "dense_descriptor_count": coverage.dense_descriptor_count,
        "active_dense_descriptor_count": coverage.active_dense_descriptor_count,
        "candidate_descriptor_count": coverage.candidate_descriptor_count,
        "active_candidate_descriptor_count": coverage.active_candidate_descriptor_count,
        "dense_optical_mass": coverage.dense_optical_mass,
        "contained_optical_mass": coverage.contained_optical_mass,
        "mass_weighted_recall": coverage.mass_weighted_recall,
        "count_recall": coverage.count_recall,
        "hole_count": coverage.hole_count,
        "dense_offscreen_count": coverage.dense_offscreen_count,
        "candidate_offscreen_count": coverage.candidate_offscreen_count,
    }


def _source_only_metadata() -> dict[str, bool]:
    return {
        "source_camera_only": True,
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "input_covariances_mutated": False,
        "same_tile_candidate_basis_only": True,
        "fixed_covariance_scale": True,
    }


def _failure_result(
    *, reason: str, virtual_count: int, anchor_count: int
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "support_basis_policy": SUPPORT_BASIS_POLICY,
        "source_only": _source_only_metadata(),
        "sigma": FIXED_SIGMA,
        "passed": False,
        "summary": {
            "input_valid": False,
            "reason": reason,
            "anchor_count": anchor_count,
            "source_anchor_count": anchor_count,
            "candidate_anchor_count": anchor_count,
            "virtual_primitive_count": virtual_count,
            "checked_source_anchor_count": 0,
            "checked_virtual_count": 0,
            "active_source_anchor_count": None,
            "active_virtual_primitive_count": None,
            "active_dense_primitive_count": None,
            "failed_source_anchor_count": 0,
            "failed_virtual_count": 0,
            "hole_count": None,
            "dense_optical_mass": None,
            "contained_optical_mass": None,
            "mass_weighted_recall": None,
            "count_recall": None,
            "all_active_same_tile_supports_contained": False,
        },
        "binding": {
            "anchor_dense_slots_sha256": None,
            "virtual_origin_slots_sha256": None,
            "virtual_means_sha256": None,
            "virtual_covariances_sha256": None,
            "virtual_opacities_sha256": None,
            "virtual_source_spatial_weights_sha256": None,
            "bilateral_assignment_weights_sha256": None,
            "virtual_candidate_basis_mask_sha256": None,
            "anchor_candidate_basis_mask_sha256": None,
            "source_to_output_ledger_sha256": None,
            "anchor_source_means_sha256": None,
            "anchor_source_covariances_sha256": None,
            "anchor_source_opacities_sha256": None,
            "merged_means_sha256": None,
            "merged_covariances_sha256": None,
            "merged_opacities_sha256": None,
            "context_extrinsics_sha256": None,
            "context_intrinsics_sha256": None,
        },
        "anchors": [],
        "virtuals": [],
    }


def _global_contract_reason(
    *,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_opacities: torch.Tensor,
    virtual_origin_slots: torch.Tensor,
    virtual_source_spatial_weights: torch.Tensor,
    bilateral_assignment_weights: torch.Tensor,
    anchor_source_means: torch.Tensor,
    anchor_source_covariances: torch.Tensor,
    anchor_source_opacities: torch.Tensor,
    anchor_dense_slots: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_opacities: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
) -> str | None:
    """Validate descriptor families and the one-camera same-tile boundary."""

    descriptor_tensors = (
        virtual_means,
        virtual_covariances,
        virtual_opacities,
        virtual_source_spatial_weights,
        bilateral_assignment_weights,
        anchor_source_means,
        anchor_source_covariances,
        anchor_source_opacities,
        merged_means,
        merged_covariances,
        merged_opacities,
        context_extrinsics,
        context_intrinsics,
    )
    if not all(
        torch.is_tensor(value) and torch.is_floating_point(value)
        for value in descriptor_tensors
    ):
        return "input-contract"
    if not torch.is_tensor(virtual_origin_slots) or not torch.is_tensor(anchor_dense_slots):
        return "slot-input-contract"

    virtual_count = _descriptor_count(virtual_means)
    anchor_count = _descriptor_count(merged_means)
    if anchor_count < 1:
        return "input-contract"
    if (
        virtual_means.shape != (virtual_count, 3)
        or virtual_covariances.shape != (virtual_count, 3, 3)
        or virtual_opacities.shape not in {(virtual_count,), (virtual_count, 1)}
        or virtual_origin_slots.shape != (virtual_count,)
        or virtual_source_spatial_weights.shape != (virtual_count, anchor_count)
        or bilateral_assignment_weights.shape != (virtual_count, anchor_count)
        or anchor_source_means.shape != (anchor_count, 3)
        or anchor_source_covariances.shape != (anchor_count, 3, 3)
        or anchor_source_opacities.shape not in {(anchor_count,), (anchor_count, 1)}
        or anchor_dense_slots.shape != (anchor_count,)
        or merged_means.shape != (anchor_count, 3)
        or merged_covariances.shape != (anchor_count, 3, 3)
        or merged_opacities.shape not in {(anchor_count,), (anchor_count, 1)}
        or context_extrinsics.shape != (anchor_count, 4, 4)
        or context_intrinsics.shape != (anchor_count, 3, 3)
    ):
        return "input-contract"
    if (
        virtual_origin_slots.dtype not in _INTEGER_DTYPES
        or anchor_dense_slots.dtype not in _INTEGER_DTYPES
        or virtual_origin_slots.device != virtual_means.device
        or anchor_dense_slots.device != virtual_means.device
    ):
        return "slot-input-contract"
    if any(
        value.device != virtual_means.device or value.dtype != virtual_means.dtype
        for value in descriptor_tensors
    ):
        return "cross-family-input-contract"
    for weights, label in (
        (virtual_source_spatial_weights, "spatial"),
        (bilateral_assignment_weights, "bilateral"),
    ):
        if not bool(torch.isfinite(weights).all()):
            return f"nonfinite-{label}-assignment"
        if bool((weights < 0.0).any()):
            return f"negative-{label}-assignment"
        if virtual_count and not torch.allclose(
            weights.sum(dim=1),
            torch.ones(
                virtual_count, device=virtual_means.device, dtype=virtual_means.dtype
            ),
            rtol=1e-5,
            atol=1e-5,
        ):
            return f"{label}-assignment-normalization"
    if virtual_count and bool((bilateral_assignment_weights > 0.0).sum(dim=1).eq(0).any()):
        return "empty-virtual-support-basis"
    anchor_values = anchor_dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    virtual_values = virtual_origin_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if (
        len(set(anchor_values)) != anchor_count
        or len(set(virtual_values)) != virtual_count
        or set(anchor_values) & set(virtual_values)
    ):
        return "same-tile-slot-partition"
    if not torch.equal(context_extrinsics, context_extrinsics[:1].expand_as(context_extrinsics)):
        return "same-tile-camera-mismatch"
    if not torch.equal(context_intrinsics, context_intrinsics[:1].expand_as(context_intrinsics)):
        return "same-tile-camera-mismatch"
    return None


def _support_record(
    *,
    source_kind: str,
    source_index: int,
    source_slot: int,
    source_means: torch.Tensor,
    source_covariances: torch.Tensor,
    source_opacities: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_opacities: torch.Tensor,
    context_extrinsic: torch.Tensor,
    context_intrinsic: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[dict[str, Any], int, float, float, bool]:
    """Audit one source ellipse against all committed anchors in its tile."""

    candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return (
            {
                "source_kind": source_kind,
                "source_index": source_index,
                "source_slot": source_slot,
                "basis_candidate_anchor_indices": [],
                "checked": True,
                "audit_valid": False,
                "active": None,
                "passed": False,
                "reason": "empty-source-support-basis",
                "coverage": None,
            },
            0,
            0.0,
            0.0,
            False,
        )
    coverage = audit_projected_dense_domain_coverage(
        dense_means=source_means[source_index : source_index + 1],
        dense_covariances=source_covariances[source_index : source_index + 1],
        dense_opacities=source_opacities[source_index : source_index + 1],
        candidate_means=merged_means[candidate_indices],
        candidate_covariances=merged_covariances[candidate_indices],
        candidate_opacities=merged_opacities[candidate_indices],
        context_extrinsic=context_extrinsic,
        context_intrinsic=context_intrinsic,
    )
    payload = _coverage_payload(coverage)
    if coverage.valid:
        if (
            coverage.hole_count is None
            or coverage.active_dense_descriptor_count != 1
            or coverage.dense_optical_mass is None
            or coverage.contained_optical_mass is None
        ):
            return (
                {
                    "source_kind": source_kind,
                    "source_index": source_index,
                    "source_slot": source_slot,
                    "basis_candidate_anchor_indices": [
                        int(index) for index in candidate_indices.detach().cpu().tolist()
                    ],
                    "checked": True,
                    "audit_valid": True,
                    "active": None,
                    "passed": False,
                    "reason": "incomplete-support-coverage-evidence",
                    "coverage": payload,
                },
                0,
                0.0,
                0.0,
                True,
            )
        holes = int(coverage.hole_count)
        passed = holes == 0
        return (
            {
                "source_kind": source_kind,
                "source_index": source_index,
                "source_slot": source_slot,
                "basis_candidate_anchor_indices": [
                    int(index) for index in candidate_indices.detach().cpu().tolist()
                ],
                "checked": True,
                "audit_valid": True,
                "active": True,
                "passed": passed,
                "reason": None if passed else "uncontained-same-tile-support",
                "coverage": payload,
            },
            holes,
            float(coverage.dense_optical_mass),
            float(coverage.contained_optical_mass),
            True,
        )
    if coverage.reason == "zero-dense-optical-mass":
        return (
            {
                "source_kind": source_kind,
                "source_index": source_index,
                "source_slot": source_slot,
                "basis_candidate_anchor_indices": [
                    int(index) for index in candidate_indices.detach().cpu().tolist()
                ],
                "checked": True,
                "audit_valid": False,
                "active": False,
                "passed": True,
                "reason": "no-active-source-support",
                "coverage": payload,
            },
            0,
            0.0,
            0.0,
            False,
        )
    return (
        {
            "source_kind": source_kind,
            "source_index": source_index,
            "source_slot": source_slot,
            "basis_candidate_anchor_indices": [
                int(index) for index in candidate_indices.detach().cpu().tolist()
            ],
            "checked": True,
            "audit_valid": False,
            "active": None,
            "passed": False,
            "reason": coverage.reason or "support-coverage-audit-failed",
            "coverage": payload,
        },
        0,
        0.0,
        0.0,
        False,
    )


def audit_depthsplat_tile_support_basis(
    *,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_opacities: torch.Tensor,
    virtual_origin_slots: torch.Tensor,
    virtual_source_spatial_weights: torch.Tensor,
    bilateral_assignment_weights: torch.Tensor,
    anchor_source_means: torch.Tensor,
    anchor_source_covariances: torch.Tensor,
    anchor_source_opacities: torch.Tensor,
    anchor_dense_slots: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_opacities: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
) -> dict[str, Any]:
    """Certify a committed same-tile merged Gaussian support basis.

    Each positive-alpha source anchor and selected-only virtual is checked in
    the one source camera shared by the tile.  A source ellipse must be fully
    contained by at least one merged anchor reached through the committed
    ``S`` then ``R`` ledger.  This is a local cooperative basis check, not an
    arbitrary union approximation.
    """

    virtual_count = _descriptor_count(virtual_means)
    anchor_count = _descriptor_count(merged_means)
    contract_reason = _global_contract_reason(
        virtual_means=virtual_means,
        virtual_covariances=virtual_covariances,
        virtual_opacities=virtual_opacities,
        virtual_origin_slots=virtual_origin_slots,
        virtual_source_spatial_weights=virtual_source_spatial_weights,
        bilateral_assignment_weights=bilateral_assignment_weights,
        anchor_source_means=anchor_source_means,
        anchor_source_covariances=anchor_source_covariances,
        anchor_source_opacities=anchor_source_opacities,
        anchor_dense_slots=anchor_dense_slots,
        merged_means=merged_means,
        merged_covariances=merged_covariances,
        merged_opacities=merged_opacities,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
    )
    if contract_reason is not None:
        return _failure_result(
            reason=contract_reason,
            virtual_count=virtual_count,
            anchor_count=anchor_count,
        )

    source_anchor_opacities = anchor_source_opacities.reshape(anchor_count)
    virtual_opacity_values = virtual_opacities.reshape(virtual_count)
    anchor_slots = anchor_dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    virtual_slots = virtual_origin_slots.detach().to(device="cpu", dtype=torch.int64).tolist()

    anchor_records: list[dict[str, Any]] = []
    virtual_records: list[dict[str, Any]] = []
    anchor_holes = 0
    virtual_holes = 0
    dense_optical_mass = 0.0
    contained_optical_mass = 0.0
    active_anchor_count = 0
    active_virtual_count = 0
    failed_anchor_count = 0
    failed_virtual_count = 0
    virtual_candidate_masks = bilateral_assignment_weights > 0.0
    source_to_output_ledger = (
        virtual_source_spatial_weights.mT @ bilateral_assignment_weights
    )
    anchor_candidate_masks = (
        source_to_output_ledger > 0.0
    ) | torch.eye(anchor_count, device=virtual_means.device, dtype=torch.bool)

    for anchor_index, slot in enumerate(anchor_slots):
        record, holes, dense_mass, contained_mass, active = _support_record(
            source_kind="anchor",
            source_index=anchor_index,
            source_slot=int(slot),
            source_means=anchor_source_means,
            source_covariances=anchor_source_covariances,
            source_opacities=source_anchor_opacities,
            merged_means=merged_means,
            merged_covariances=merged_covariances,
            merged_opacities=merged_opacities,
            context_extrinsic=context_extrinsics[0],
            context_intrinsic=context_intrinsics[0],
            candidate_mask=anchor_candidate_masks[anchor_index],
        )
        anchor_records.append(record)
        anchor_holes += holes
        dense_optical_mass += dense_mass
        contained_optical_mass += contained_mass
        active_anchor_count += int(active)
        failed_anchor_count += int(not record["passed"])

    for virtual_index, slot in enumerate(virtual_slots):
        record, holes, dense_mass, contained_mass, active = _support_record(
            source_kind="virtual",
            source_index=virtual_index,
            source_slot=int(slot),
            source_means=virtual_means,
            source_covariances=virtual_covariances,
            source_opacities=virtual_opacity_values,
            merged_means=merged_means,
            merged_covariances=merged_covariances,
            merged_opacities=merged_opacities,
            context_extrinsic=context_extrinsics[0],
            context_intrinsic=context_intrinsics[0],
            candidate_mask=virtual_candidate_masks[virtual_index],
        )
        virtual_records.append(record)
        virtual_holes += holes
        dense_optical_mass += dense_mass
        contained_optical_mass += contained_mass
        active_virtual_count += int(active)
        failed_virtual_count += int(not record["passed"])

    active_dense_count = active_anchor_count + active_virtual_count
    hole_count = anchor_holes + virtual_holes
    passed = failed_anchor_count == 0 and failed_virtual_count == 0 and hole_count == 0
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "support_basis_policy": SUPPORT_BASIS_POLICY,
        "source_only": _source_only_metadata(),
        "sigma": FIXED_SIGMA,
        "passed": passed,
        "summary": {
            "input_valid": True,
            "reason": None if passed else "same-tile-support-basis-coverage-failed",
            "anchor_count": anchor_count,
            "source_anchor_count": anchor_count,
            "candidate_anchor_count": anchor_count,
            "virtual_primitive_count": virtual_count,
            "checked_source_anchor_count": anchor_count,
            "checked_virtual_count": virtual_count,
            "active_source_anchor_count": active_anchor_count,
            "active_virtual_primitive_count": active_virtual_count,
            "active_dense_primitive_count": active_dense_count,
            "failed_source_anchor_count": failed_anchor_count,
            "failed_virtual_count": failed_virtual_count,
            "hole_count": hole_count,
            "dense_optical_mass": dense_optical_mass,
            "contained_optical_mass": contained_optical_mass,
            "mass_weighted_recall": (
                contained_optical_mass / dense_optical_mass
                if dense_optical_mass > 0.0
                else None
            ),
            "count_recall": (
                (active_dense_count - hole_count) / active_dense_count
                if active_dense_count > 0
                else None
            ),
            "all_active_same_tile_supports_contained": passed,
        },
        "binding": {
            "anchor_dense_slots_sha256": _tensor_sha256(anchor_dense_slots),
            "virtual_origin_slots_sha256": _tensor_sha256(virtual_origin_slots),
            "virtual_means_sha256": _tensor_sha256(virtual_means),
            "virtual_covariances_sha256": _tensor_sha256(virtual_covariances),
            "virtual_opacities_sha256": _tensor_sha256(virtual_opacities),
            "virtual_source_spatial_weights_sha256": _tensor_sha256(
                virtual_source_spatial_weights
            ),
            "bilateral_assignment_weights_sha256": _tensor_sha256(
                bilateral_assignment_weights
            ),
            "virtual_candidate_basis_mask_sha256": _tensor_sha256(
                virtual_candidate_masks.to(dtype=torch.uint8)
            ),
            "anchor_candidate_basis_mask_sha256": _tensor_sha256(
                anchor_candidate_masks.to(dtype=torch.uint8)
            ),
            "source_to_output_ledger_sha256": _tensor_sha256(
                source_to_output_ledger
            ),
            "anchor_source_means_sha256": _tensor_sha256(anchor_source_means),
            "anchor_source_covariances_sha256": _tensor_sha256(
                anchor_source_covariances
            ),
            "anchor_source_opacities_sha256": _tensor_sha256(
                anchor_source_opacities
            ),
            "merged_means_sha256": _tensor_sha256(merged_means),
            "merged_covariances_sha256": _tensor_sha256(merged_covariances),
            "merged_opacities_sha256": _tensor_sha256(merged_opacities),
            "context_extrinsics_sha256": _tensor_sha256(context_extrinsics),
            "context_intrinsics_sha256": _tensor_sha256(context_intrinsics),
        },
        "anchors": anchor_records,
        "virtuals": virtual_records,
    }


__all__ = (
    "AUDIT_KIND",
    "AUDIT_SCHEMA_VERSION",
    "SUPPORT_BASIS_POLICY",
    "audit_depthsplat_tile_support_basis",
)
