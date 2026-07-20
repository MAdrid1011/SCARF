"""Source-only coverage diagnostics for literal DepthSplat T=4 virtual merges.

The literal T=4 materializer deliberately retains its fixed-scale first/second
moment merge.  This module observes that committed preflight result without
changing routing, materialization, or the frozen V16T4 guard.  It rebuilds
only the selected-probe virtual Gaussian field and projects it through the
source camera carried by the selected packet.  No target-view input is
accepted by the public APIs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import torch

from saes.depthsplat_l0_l1_materializer import (
    DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
    DepthSplatCompactMaterializationPreflight,
    _require_plan,
    _route_anchor_positions,
    _spatial_weights,
    _tile_feature_variance,
    _tile_slots,
    _validate_literal_paper_t4_plan,
    _validate_routing_inputs,
    _validate_source,
)
from saes.depthsplat_selected_output import (
    DepthSplatPackedGaussianAttributes,
    DepthSplatSparseRawPacket,
)
from saes.probe_first_schedule import PAPER_KP_ANCHOR_SEMANTICS
from saes.projected_domain_coverage_audit import (
    FIXED_SIGMA,
    ProjectedDomainCoverage,
    audit_projected_dense_domain_coverage,
)


AUDIT_SCHEMA_VERSION = "depthsplat-literal-t4-virtual-anchor-coverage-audit-v1"
AUDIT_KIND = "depthsplat-literal-t4-source-only-virtual-anchor-coverage"
VIRTUAL_GEOMETRY_SOURCE = "selected-probe-gaussian-spatial-moment-v1"
FIXED_SCALE = 1.0


def _coverage_payload(coverage: ProjectedDomainCoverage) -> dict[str, Any]:
    """Copy only JSON-safe scalar evidence from the generic coverage audit."""

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


def audit_literal_virtual_anchor_set(
    *,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_opacities: torch.Tensor,
    merged_mean: torch.Tensor,
    merged_covariance: torch.Tensor,
    merged_opacity: torch.Tensor,
    context_extrinsic: torch.Tensor,
    context_intrinsic: torch.Tensor,
) -> dict[str, Any]:
    """Check all virtual ellipses against one fixed-scale merged anchor.

    The generic audit evaluates every positive-alpha virtual primitive at its
    fixed two-sigma support.  ``continuity`` is deliberately strict: it is
    true only when every active virtual support is contained by this one
    merged anchor, not by a neighboring output anchor.
    """

    if (
        not torch.is_tensor(merged_mean)
        or merged_mean.shape != (3,)
        or not torch.is_tensor(merged_covariance)
        or merged_covariance.shape != (3, 3)
        or not torch.is_tensor(merged_opacity)
        or merged_opacity.numel() != 1
    ):
        raise ValueError("literal virtual coverage candidate has an invalid shape")
    candidate_means = merged_mean.reshape(1, 3)
    candidate_covariances = merged_covariance.reshape(1, 3, 3)
    candidate_opacities = merged_opacity.reshape(1)
    coverage = audit_projected_dense_domain_coverage(
        dense_means=virtual_means,
        dense_covariances=virtual_covariances,
        dense_opacities=virtual_opacities,
        candidate_means=candidate_means,
        candidate_covariances=candidate_covariances,
        candidate_opacities=candidate_opacities,
        context_extrinsic=context_extrinsic,
        context_intrinsic=context_intrinsic,
    )
    payload = _coverage_payload(coverage)
    all_contained = (
        coverage.valid
        and coverage.hole_count == 0
        and coverage.active_dense_descriptor_count > 0
    )
    return {
        "fixed_moment_covariance_scale": FIXED_SCALE,
        "virtual_primitive_count": int(virtual_means.shape[0]),
        "coverage": payload,
        "continuity": {
            "checked": coverage.valid,
            "all_active_virtual_2sigma_supports_contained": all_contained
            if coverage.valid
            else None,
            "broken": (not all_contained) if coverage.valid else None,
        },
    }


def _slot_to_update_index(
    preflight: DepthSplatCompactMaterializationPreflight,
) -> dict[int, int]:
    slots = preflight.update_dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(set(slots)) != len(slots):
        raise ValueError("literal virtual coverage preflight has duplicate update slots")
    count = len(slots)
    if (
        preflight.means.shape != (count, 3)
        or preflight.covariances.shape != (count, 3, 3)
        or preflight.opacities.shape != (count,)
    ):
        raise ValueError("literal virtual coverage preflight update tensors are inconsistent")
    return {int(slot): index for index, slot in enumerate(slots)}


def _literal_virtual_field(
    *,
    source_means: torch.Tensor,
    source_covariances: torch.Tensor,
    source_opacities: torch.Tensor,
    source_positions: list[tuple[int, int]],
    target_positions: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild the exact selected-only literal virtual geometry field."""

    spatial = _spatial_weights(
        target_positions,
        source_positions,
        device=source_means.device,
        dtype=source_means.dtype,
    )
    virtual_means = spatial @ source_means
    mean_deltas = source_means.unsqueeze(0) - virtual_means.unsqueeze(1)
    virtual_covariances = (
        spatial.reshape(spatial.shape[0], spatial.shape[1], 1, 1)
        * (
            source_covariances.unsqueeze(0)
            + mean_deltas.unsqueeze(3) @ mean_deltas.unsqueeze(2)
        )
    ).sum(dim=1)
    virtual_covariances = (virtual_covariances + virtual_covariances.mT) * 0.5
    virtual_opacities = spatial @ source_opacities
    return virtual_means, virtual_covariances, virtual_opacities


def _validate_literal_preflight(
    preflight: DepthSplatCompactMaterializationPreflight,
) -> None:
    if not isinstance(preflight, DepthSplatCompactMaterializationPreflight):
        raise TypeError("literal virtual coverage requires a materialization preflight")
    events = preflight.events
    if (
        events.get("execution_profile")
        != DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        or events.get("maximum_coverage_covariance_scale") != FIXED_SCALE
        or events.get("literal_finite_psd_moment_merge") is not True
        or events.get("literal_support_containment_guard") is not False
        or events.get("target_rgb_accessed") is not False
        or events.get("target_camera_accessed_before_commit") is not False
        or events.get("skipped_s3_attributes_accessed") is not False
        or events.get("source_nonprobe_s3_attribute_reads") != 0
    ):
        raise ValueError("literal virtual coverage preflight is not fixed-scale source-only")


def audit_literal_t4_virtual_anchor_coverage(
    *,
    packet: DepthSplatSparseRawPacket,
    packed: DepthSplatPackedGaussianAttributes,
    plan: Any,
    routing_features: torch.Tensor,
    routing_z_depths: torch.Tensor,
    preflight: DepthSplatCompactMaterializationPreflight,
) -> dict[str, Any]:
    """Audit every committed literal T=4 merged anchor from source inputs only.

    The descriptor reference is the twelve virtual Gaussians reconstructed from
    the four selected source probes in each compact tile.  The candidate is the
    corresponding fixed-scale preflight update.  Full/promoted tiles are
    deliberately excluded because they have no merged candidate to diagnose.
    """

    views, height, width, semantics = _require_plan(plan)
    _validate_literal_paper_t4_plan(
        plan, views=views, height=height, width=width, semantics=semantics
    )
    if semantics != PAPER_KP_ANCHOR_SEMANTICS:
        raise ValueError("literal virtual coverage anchor semantics changed")
    _validate_literal_preflight(preflight)
    slot_to_index = _validate_source(
        packet, packed, plan, views=views, height=height, width=width
    )
    assignment_features, assignment_semantics = _validate_routing_inputs(
        routing_features,
        routing_z_depths,
        packet,
        plan,
        views=views,
        height=height,
        width=width,
        execution_profile=DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
    )
    if assignment_semantics != "raw-bilinear-s1-v1":
        raise ValueError("literal virtual coverage assignment semantics changed")
    update_by_slot = _slot_to_update_index(preflight)
    plan_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in plan.tile_trace
    }
    preflight_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in preflight.tile_trace
    }
    if len(plan_records) != views * (height // 4) * (width // 4) or set(plan_records) != set(
        preflight_records
    ):
        raise ValueError("literal virtual coverage tile trace is incomplete")

    records: list[dict[str, Any]] = []
    compact_tile_count = 0
    accepted_compact_tile_count = 0
    full_or_promoted_tile_count = 0
    broken_tiles: set[tuple[int, int, int]] = set()
    reason_counts: Counter[str] = Counter()
    valid_anchor_count = 0
    active_virtual_count = 0
    hole_count = 0
    dense_optical_mass = 0.0
    contained_optical_mass = 0.0

    for key in sorted(plan_records):
        view, tile_y, tile_x = key
        plan_record = plan_records[key]
        preflight_record = preflight_records[key]
        level = plan_record.get("pre_guard_route")
        if level == "Full":
            full_or_promoted_tile_count += 1
            continue
        if level not in {"L0", "L1"}:
            raise ValueError("literal virtual coverage route is invalid")
        compact_tile_count += 1
        if preflight_record.get("accepted") is not True:
            full_or_promoted_tile_count += 1
            continue
        accepted_compact_tile_count += 1
        anchors = _route_anchor_positions(plan_record, level=str(level), semantics=semantics)
        if anchors != [(0, 0), (0, 3), (3, 0), (3, 3)]:
            raise ValueError("literal virtual coverage does not have four selected probes")
        anchor_slots = _tile_slots(
            view=view,
            tile_y=tile_y,
            tile_x=tile_x,
            height=height,
            width=width,
            tile_size=4,
            positions=anchors,
        )
        if any(slot not in slot_to_index or slot not in update_by_slot for slot in anchor_slots):
            raise ValueError("literal virtual coverage compact anchor is unbound")
        anchor_indices = [slot_to_index[slot] for slot in anchor_slots]
        source_positions = [(tile_y * 4 + row, tile_x * 4 + column) for row, column in anchors]
        target_positions = [
            (tile_y * 4 + row, tile_x * 4 + column)
            for row in range(4)
            for column in range(4)
            if (row, column) not in set(anchors)
        ]
        if len(target_positions) != 12:
            raise RuntimeError("literal virtual coverage virtual position count changed")
        virtual_means, virtual_covariances, virtual_opacities = _literal_virtual_field(
            source_means=packed.means[anchor_indices],
            source_covariances=packed.covariances[anchor_indices],
            source_opacities=packed.opacities[anchor_indices],
            source_positions=source_positions,
            target_positions=target_positions,
        )
        if (
            not bool(torch.isfinite(virtual_means).all())
            or not bool(torch.isfinite(virtual_covariances).all())
            or not bool(torch.isfinite(virtual_opacities).all())
        ):
            raise ValueError("literal virtual coverage field is non-finite")
        camera_extrinsic = packet.extrinsics[anchor_indices[0]]
        camera_intrinsic = packet.intrinsics[anchor_indices[0]]
        if not torch.equal(packet.extrinsics[anchor_indices], camera_extrinsic.expand(4, -1, -1)) or not torch.equal(
            packet.intrinsics[anchor_indices], camera_intrinsic.expand(4, -1, -1)
        ):
            raise ValueError("literal virtual coverage tile crosses source cameras")
        feature_variance = _tile_feature_variance(
            plan_record, str(plan.events["feature_statistic"])
        )
        if feature_variance < 0.0:
            raise ValueError("literal virtual coverage feature variance is invalid")
        for anchor_offset, slot in enumerate(anchor_slots):
            update_index = update_by_slot[slot]
            evidence = audit_literal_virtual_anchor_set(
                virtual_means=virtual_means,
                virtual_covariances=virtual_covariances,
                virtual_opacities=virtual_opacities,
                merged_mean=preflight.means[update_index],
                merged_covariance=preflight.covariances[update_index],
                merged_opacity=preflight.opacities[update_index],
                context_extrinsic=camera_extrinsic,
                context_intrinsic=camera_intrinsic,
            )
            coverage = evidence["coverage"]
            valid = coverage["valid"] is True
            if valid:
                valid_anchor_count += 1
                active_virtual_count += int(coverage["active_dense_descriptor_count"])
                hole_count += int(coverage["hole_count"])
                dense_optical_mass += float(coverage["dense_optical_mass"])
                contained_optical_mass += float(coverage["contained_optical_mass"])
                if coverage["hole_count"]:
                    broken_tiles.add(key)
            else:
                reason_counts[str(coverage["reason"] or "unknown")] += 1
                broken_tiles.add(key)
            records.append(
                {
                    "view": view,
                    "tile_y": tile_y,
                    "tile_x": tile_x,
                    "level": str(level),
                    "anchor_offset": anchor_offset,
                    "anchor_dense_slot": int(slot),
                    "preflight_update_index": update_index,
                    "source_camera_view": view,
                    "virtual_local_positions": [list(position) for position in target_positions],
                    "virtual_geometry_source": VIRTUAL_GEOMETRY_SOURCE,
                    "assignment_feature_variance": feature_variance,
                    **evidence,
                }
            )

    invalid_anchor_count = len(records) - valid_anchor_count
    summary = {
        "planned_compact_tile_count": compact_tile_count,
        "accepted_compact_tile_count": accepted_compact_tile_count,
        "full_or_promoted_tile_count": full_or_promoted_tile_count,
        "merged_anchor_count": len(records),
        "valid_merged_anchor_count": valid_anchor_count,
        "invalid_merged_anchor_count": invalid_anchor_count,
        "invalid_reason_counts": dict(sorted(reason_counts.items())),
        "active_virtual_primitive_count": active_virtual_count,
        "hole_count": hole_count,
        "dense_optical_mass": dense_optical_mass,
        "contained_optical_mass": contained_optical_mass,
        "mass_weighted_recall": (
            contained_optical_mass / dense_optical_mass if dense_optical_mass > 0.0 else None
        ),
        "count_recall": (
            (active_virtual_count - hole_count) / active_virtual_count
            if active_virtual_count > 0
            else None
        ),
        "continuity": {
            "broken_merged_anchor_count": sum(
                record["continuity"]["broken"] is True for record in records
            ),
            "broken_tile_count": len(broken_tiles),
            "all_active_virtual_2sigma_supports_contained": (
                invalid_anchor_count == 0 and hole_count == 0 and active_virtual_count > 0
            ),
        },
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "source_only": {
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "skipped_s3_attributes_accessed": False,
            "source_nonprobe_s3_attribute_reads": 0,
            "source_camera_only": True,
        },
        "literal_fixed_scale": FIXED_SCALE,
        "sigma": FIXED_SIGMA,
        "virtual_geometry_source": VIRTUAL_GEOMETRY_SOURCE,
        "bindings": {
            "plan_tile_trace_sha256": plan.events.get("tile_trace_sha256"),
            "preflight_tile_trace_sha256": preflight.events.get("tile_trace_sha256"),
            "preflight_update_binding": dict(preflight.events.get("update_binding", {})),
            "packet_source_trace_sha256": packed.source_trace_sha256,
            "packet_attribute_binding_sha256": packed.attribute_binding_sha256,
        },
        "summary": summary,
        "records": records,
    }


__all__ = (
    "AUDIT_KIND",
    "AUDIT_SCHEMA_VERSION",
    "FIXED_SCALE",
    "VIRTUAL_GEOMETRY_SOURCE",
    "audit_literal_t4_virtual_anchor_coverage",
    "audit_literal_virtual_anchor_set",
)
