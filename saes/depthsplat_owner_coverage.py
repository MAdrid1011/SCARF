"""Source-only, owner-strict projected coverage diagnostics for DepthSplat.

This post-commit diagnostic checks each owner-local virtual field and its
pre-merge source anchor only against the merged Gaussian that replaces them.
It intentionally delegates the fixed two-sigma ellipse test to
:mod:`saes.projected_domain_coverage_audit` and never changes a descriptor or
participates in routing.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

import torch

from saes.projected_domain_coverage_audit import (
    FIXED_SIGMA,
    ProjectedDomainCoverage,
    audit_projected_dense_domain_coverage,
)


AUDIT_SCHEMA_VERSION = "depthsplat-owner-coverage-audit-v2"
AUDIT_KIND = "depthsplat-source-only-owner-assigned-coverage"
OWNER_ASSIGNMENT_POLICY = "maximum-bilateral-weight-owner-v1"

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
    """Hash one immutable tensor binding without coercing its contents."""

    if not torch.is_tensor(value):
        raise TypeError("owner coverage tensor digest requires a tensor")
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _coverage_payload(coverage: ProjectedDomainCoverage) -> dict[str, Any]:
    """Return the generic audit result without tensors or dataclasses."""

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
        "owner_source_anchor_support_included": True,
    }


def _failure_result(
    *,
    reason: str,
    virtual_count: int,
    owner_count: int,
    invalid_owner_assignment_count: int = 0,
) -> dict[str, Any]:
    """Return a fail-closed, JSON-safe result before owner-wise auditing."""

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "source_only": _source_only_metadata(),
        "sigma": FIXED_SIGMA,
        "passed": False,
        "summary": {
            "input_valid": False,
            "reason": reason,
            "owner_count": owner_count,
            "source_anchor_count": owner_count,
            "virtual_primitive_count": virtual_count,
            "assigned_owner_count": 0,
            "empty_owner_count": 0,
            "checked_owner_count": 0,
            "audit_valid_owner_count": 0,
            "inactive_owner_count": 0,
            "failed_owner_count": 0,
            "invalid_owner_assignment_count": invalid_owner_assignment_count,
            "active_virtual_primitive_count": None,
            "active_source_anchor_count": None,
            "active_dense_primitive_count": None,
            "hole_count": None,
            "dense_optical_mass": None,
            "contained_optical_mass": None,
            "mass_weighted_recall": None,
            "count_recall": None,
            "failure_reason_counts": {reason: 1},
            "all_active_virtual_2sigma_supports_contained": False,
            "all_active_owner_anchor_and_virtual_2sigma_supports_contained": False,
        },
        "ownership": {
            "policy": OWNER_ASSIGNMENT_POLICY,
            "virtual_owner_indices_sha256": None,
            "owner_assignment_counts": None,
        },
        "owners": [],
    }


def _global_contract_reason(
    *,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_opacities: torch.Tensor,
    virtual_owner_indices: torch.Tensor,
    owner_source_means: torch.Tensor,
    owner_source_covariances: torch.Tensor,
    owner_source_opacities: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_opacities: torch.Tensor,
    owner_source_extrinsics: torch.Tensor,
    owner_source_intrinsics: torch.Tensor,
) -> str | None:
    """Validate indexing and family alignment before splitting by owner."""

    descriptor_tensors = (
        virtual_means,
        virtual_covariances,
        virtual_opacities,
        owner_source_means,
        owner_source_covariances,
        owner_source_opacities,
        merged_means,
        merged_covariances,
        merged_opacities,
        owner_source_extrinsics,
        owner_source_intrinsics,
    )
    if not all(
        torch.is_tensor(value) and torch.is_floating_point(value)
        for value in descriptor_tensors
    ):
        return "input-contract"
    if not torch.is_tensor(virtual_owner_indices):
        return "owner-indices-input-contract"

    virtual_count = _descriptor_count(virtual_means)
    owner_count = _descriptor_count(merged_means)
    if owner_count < 1:
        return "input-contract"
    if (
        virtual_means.shape != (virtual_count, 3)
        or virtual_covariances.shape != (virtual_count, 3, 3)
        or virtual_opacities.shape not in {(virtual_count,), (virtual_count, 1)}
        or owner_source_means.shape != (owner_count, 3)
        or owner_source_covariances.shape != (owner_count, 3, 3)
        or owner_source_opacities.shape not in {(owner_count,), (owner_count, 1)}
        or merged_means.shape != (owner_count, 3)
        or merged_covariances.shape != (owner_count, 3, 3)
        or merged_opacities.shape not in {(owner_count,), (owner_count, 1)}
        or owner_source_extrinsics.shape != (owner_count, 4, 4)
        or owner_source_intrinsics.shape != (owner_count, 3, 3)
    ):
        return "input-contract"
    if (
        virtual_owner_indices.ndim != 1
        or virtual_owner_indices.shape[0] != virtual_count
        or virtual_owner_indices.dtype not in _INTEGER_DTYPES
        or virtual_owner_indices.device != virtual_means.device
    ):
        return "owner-indices-input-contract"
    if any(
        value.device != virtual_means.device or value.dtype != virtual_means.dtype
        for value in descriptor_tensors
    ):
        return "cross-family-input-contract"
    return None


def audit_depthsplat_owner_coverage(
    *,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_opacities: torch.Tensor,
    virtual_owner_indices: torch.Tensor,
    owner_source_means: torch.Tensor,
    owner_source_covariances: torch.Tensor,
    owner_source_opacities: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_opacities: torch.Tensor,
    owner_source_extrinsics: torch.Tensor,
    owner_source_intrinsics: torch.Tensor,
) -> dict[str, Any]:
    """Audit assigned virtual supports against exactly one merged owner each.

    All tensors are already committed source-side descriptors or source-camera
    geometry.  Each owner is audited against a dense set containing its own
    pre-merge anchor plus only the virtual primitives assigned to it.  This
    forbids both neighboring merged Gaussians and an owner-local virtual field
    from masking support lost by the replaced anchor itself.
    """

    virtual_count = _descriptor_count(virtual_means)
    owner_count = _descriptor_count(merged_means)
    contract_reason = _global_contract_reason(
        virtual_means=virtual_means,
        virtual_covariances=virtual_covariances,
        virtual_opacities=virtual_opacities,
        virtual_owner_indices=virtual_owner_indices,
        owner_source_means=owner_source_means,
        owner_source_covariances=owner_source_covariances,
        owner_source_opacities=owner_source_opacities,
        merged_means=merged_means,
        merged_covariances=merged_covariances,
        merged_opacities=merged_opacities,
        owner_source_extrinsics=owner_source_extrinsics,
        owner_source_intrinsics=owner_source_intrinsics,
    )
    if contract_reason is not None:
        return _failure_result(
            reason=contract_reason,
            virtual_count=virtual_count,
            owner_count=owner_count,
        )

    owner_values = virtual_owner_indices.detach().to(
        device="cpu", dtype=torch.int64
    ).tolist()
    invalid_owner_assignment_count = sum(
        owner < 0 or owner >= owner_count for owner in owner_values
    )
    if invalid_owner_assignment_count:
        return _failure_result(
            reason="owner-index-out-of-range",
            virtual_count=virtual_count,
            owner_count=owner_count,
            invalid_owner_assignment_count=invalid_owner_assignment_count,
        )
    virtual_opacity_vector = virtual_opacities.reshape(virtual_count)
    owner_source_opacity_vector = owner_source_opacities.reshape(owner_count)

    records: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()
    assigned_owner_count = 0
    checked_owner_count = 0
    audit_valid_owner_count = 0
    inactive_owner_count = 0
    failed_owner_count = 0
    active_virtual_count = 0
    active_source_anchor_count = 0
    active_dense_count = 0
    hole_count = 0
    dense_optical_mass = 0.0
    contained_optical_mass = 0.0
    owner_assignment_counts: list[int] = []

    for owner_index in range(owner_count):
        assigned = virtual_owner_indices == owner_index
        assigned_count = int(assigned.sum().item())
        owner_assignment_counts.append(assigned_count)
        if assigned_count:
            assigned_owner_count += 1
        checked_owner_count += 1
        dense_means = torch.cat(
            (owner_source_means[owner_index : owner_index + 1], virtual_means[assigned]),
            dim=0,
        )
        dense_covariances = torch.cat(
            (
                owner_source_covariances[owner_index : owner_index + 1],
                virtual_covariances[assigned],
            ),
            dim=0,
        )
        dense_opacities = torch.cat(
            (
                owner_source_opacity_vector[owner_index : owner_index + 1],
                virtual_opacity_vector[assigned],
            ),
            dim=0,
        )
        coverage = audit_projected_dense_domain_coverage(
            dense_means=dense_means,
            dense_covariances=dense_covariances,
            dense_opacities=dense_opacities,
            candidate_means=merged_means[owner_index : owner_index + 1],
            candidate_covariances=merged_covariances[owner_index : owner_index + 1],
            candidate_opacities=merged_opacities[owner_index : owner_index + 1],
            context_extrinsic=owner_source_extrinsics[owner_index],
            context_intrinsic=owner_source_intrinsics[owner_index],
        )
        payload = _coverage_payload(coverage)
        if coverage.valid:
            if (
                coverage.hole_count is None
                or coverage.dense_optical_mass is None
                or coverage.contained_optical_mass is None
            ):
                owner_active_count = None
                owner_active_source_anchor_count = None
                owner_active_virtual_count = None
                owner_passed = False
                owner_reason = "incomplete-owner-coverage-evidence"
                failure_reasons[owner_reason] += 1
                failed_owner_count += 1
                records.append(
                    {
                        "owner_index": owner_index,
                        "source_anchor_count": 1,
                        "source_anchor_support_included": True,
                        "source_anchor_support_passed": False,
                        "assigned_virtual_count": assigned_count,
                        "dense_descriptor_count": assigned_count + 1,
                        "active_virtual_count": owner_active_virtual_count,
                        "active_source_anchor_count": owner_active_source_anchor_count,
                        "active_dense_descriptor_count": owner_active_count,
                        "checked": True,
                        "audit_valid": coverage.valid,
                        "passed": owner_passed,
                        "reason": owner_reason,
                        "coverage": payload,
                    }
                )
                continue
            audit_valid_owner_count += 1
            owner_active_count = coverage.active_dense_descriptor_count
            owner_active_source_anchor_count = int(
                (owner_source_opacity_vector[owner_index : owner_index + 1] > 0.0)
                .sum()
                .item()
            )
            owner_active_virtual_count = owner_active_count - owner_active_source_anchor_count
            if owner_active_virtual_count < 0:
                raise RuntimeError("owner coverage active descriptor count is inconsistent")
            owner_hole_count = coverage.hole_count
            active_virtual_count += owner_active_virtual_count
            active_source_anchor_count += owner_active_source_anchor_count
            active_dense_count += owner_active_count
            hole_count += owner_hole_count
            dense_optical_mass += float(coverage.dense_optical_mass)
            contained_optical_mass += float(coverage.contained_optical_mass)
            owner_passed = owner_hole_count == 0
            owner_reason = None if owner_passed else "uncontained-assigned-virtuals"
        elif coverage.reason == "zero-dense-optical-mass":
            # No positive-alpha owner anchor or assigned virtual support exists.
            # It cannot create a hole, while the generic result remains visible.
            inactive_owner_count += 1
            owner_active_count = 0
            owner_active_source_anchor_count = 0
            owner_active_virtual_count = 0
            owner_passed = True
            owner_reason = "no-active-owner-support"
        else:
            owner_active_count = None
            owner_active_source_anchor_count = None
            owner_active_virtual_count = None
            owner_passed = False
            owner_reason = coverage.reason or "owner-coverage-audit-failed"
            failure_reasons[owner_reason] += 1

        if not owner_passed:
            failed_owner_count += 1
        records.append(
            {
                "owner_index": owner_index,
                "source_anchor_count": 1,
                "source_anchor_support_included": True,
                "source_anchor_support_passed": owner_passed,
                "assigned_virtual_count": assigned_count,
                "dense_descriptor_count": assigned_count + 1,
                "active_virtual_count": owner_active_virtual_count,
                "active_source_anchor_count": owner_active_source_anchor_count,
                "active_dense_descriptor_count": owner_active_count,
                "checked": True,
                "audit_valid": coverage.valid,
                "passed": owner_passed,
                "reason": owner_reason,
                "coverage": payload,
            }
        )

    passed = failed_owner_count == 0 and hole_count == 0
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "source_only": _source_only_metadata(),
        "sigma": FIXED_SIGMA,
        "passed": passed,
        "summary": {
            "input_valid": True,
            "reason": None if passed else "owner-coverage-failed",
            "owner_count": owner_count,
            "source_anchor_count": owner_count,
            "virtual_primitive_count": virtual_count,
            "assigned_owner_count": assigned_owner_count,
            "empty_owner_count": owner_count - assigned_owner_count,
            "checked_owner_count": checked_owner_count,
            "audit_valid_owner_count": audit_valid_owner_count,
            "inactive_owner_count": inactive_owner_count,
            "failed_owner_count": failed_owner_count,
            "invalid_owner_assignment_count": 0,
            "active_virtual_primitive_count": active_virtual_count,
            "active_source_anchor_count": active_source_anchor_count,
            "active_dense_primitive_count": active_dense_count,
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
            "failure_reason_counts": dict(sorted(failure_reasons.items())),
            "all_active_virtual_2sigma_supports_contained": passed,
            "all_active_owner_anchor_and_virtual_2sigma_supports_contained": passed,
        },
        "ownership": {
            "policy": OWNER_ASSIGNMENT_POLICY,
            "virtual_owner_indices_sha256": _tensor_sha256(virtual_owner_indices),
            "owner_assignment_counts": owner_assignment_counts,
        },
        "owners": records,
    }


__all__ = (
    "AUDIT_KIND",
    "AUDIT_SCHEMA_VERSION",
    "audit_depthsplat_owner_coverage",
)
