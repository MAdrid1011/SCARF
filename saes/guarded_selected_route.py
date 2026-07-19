"""Resolve SAES materialization guards from source-bound selected attributes.

This module is deliberately a target-free control primitive.  It does not
materialize merged Gaussians or claim an S2/S3 saving.  Its purpose is to
close the gap between a probe-first raw-head plan and the later guard decision:
if a selected L0/L1 candidate fails, it records exactly which Full positions
must be requested before a downstream producer can continue.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from saes.probe_first_schedule import (
    ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    ADAPTIVE_L1_15_MAX_BEST_TO_SECOND_RESIDUAL_RATIO,
    ADAPTIVE_L1_15_SELECTION_SEMANTICS,
    BALANCED_L1_ANCHOR_SEMANTICS,
    LEGACY_L1_ANCHOR_SEMANTICS,
    PAPER_KP_ANCHOR_SEMANTICS,
    IncrementalProbeFirstPlan,
    build_incremental_probe_first_plan,
    l1_local_positions_for_tile,
)
from saes.probe_layout import compute_probe_positions
from saes.progressive_saes import (
    DELETION_CERTIFICATE_POLICY,
    DELETION_CERTIFICATE_SOURCE_KIND,
    PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
    ProgressiveSAES,
)
from saes.sparse_gaussian_consumer import PackedGaussianAttributes


FIXED_TILE_SIZE = 4
FIXED_FEATURE_THRESHOLD = 0.20
FIXED_DEPTH_THRESHOLD = 0.10
FIXED_CROSS_CHECK_THRESHOLD = 0.015
FIXED_DECISION_SEMANTICS = "probe-normalized-std-first-hit"
PAPER_DECISION_SEMANTICS = "paper-probe-feature-variance-first-hit"
ROUTE_SCHEMA_VERSION = "saes-guarded-selected-route-v1"
STRICT_LOSSLESS_ZERO_POLICY = "strict-lossless-zero"
PAPER_NONZERO_DEV_POLICY = "paper-nonzero-dev"
ENGINEERING_L1_12_NONZERO_DEV_POLICY = "engineering-nonzero-l1-12-dev"
ENGINEERING_L1_12_CONTINUITY_DEV_POLICY = "engineering-nonzero-l1-12-continuity-dev"
ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY = (
    "engineering-nonzero-l1-12-balanced-continuity-dev"
)
ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY = (
    "engineering-nonzero-l1-15-adaptive-continuity-dev"
)
ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY = (
    "engineering-nonzero-l1-15-adaptive-confidence-dev"
)
ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY = (
    "engineering-nonzero-l1-15-adaptive-absolute-residual-dev"
)
ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY = (
    "engineering-nonzero-l1-15-adaptive-absolute-residual-v4-attribute-loo-dev"
)
GUARD_ONLY_POLICY = "guard-only"


@dataclass(frozen=True)
class GuardedSelectedRoute:
    """Guard-resolved selected outputs plus any required Full extension.

    ``selected_output_mask`` is the output set after L0/L1 guard resolution.
    ``additional_full_mask`` identifies descriptors not present in the current
    packet which must be produced in a later Full request.  A nonempty
    extension is intentionally not treated as already executed work.
    """

    selected_output_mask: torch.Tensor
    additional_full_mask: torch.Tensor
    raw_head_request_mask: torch.Tensor
    tile_trace: tuple[dict[str, Any], ...]
    events: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("guarded selected-route trace must be JSON-serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def _mask_sha256(mask: torch.Tensor) -> str:
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise ValueError("guarded selected-route masks must be bool tensors")
    value = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _require_hash(trace: Mapping[str, Any], name: str, expected: str, label: str) -> None:
    value = trace.get(name)
    if value != expected:
        raise ValueError(f"selected packet {label} binding does not match the route plan")


def _validate_fixed_plan(
    plan: IncrementalProbeFirstPlan,
    *,
    features: torch.Tensor,
    depths: torch.Tensor,
) -> IncrementalProbeFirstPlan:
    if not isinstance(plan, IncrementalProbeFirstPlan):
        raise TypeError("guarded selected-route requires an incremental probe-first plan")
    if not torch.is_tensor(features) or not torch.is_tensor(depths):
        raise ValueError("guarded selected-route requires S1 features and S2 depths")
    masks = (plan.primary_mask, plan.secondary_mask, plan.full_mask, plan.selection_mask)
    if any(not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 3 for mask in masks):
        raise ValueError("guarded selected-route plan masks must be [V,H,W] bool tensors")
    if any(mask.shape != plan.selection_mask.shape for mask in masks[:3]):
        raise ValueError("guarded selected-route plan masks do not share one layout")
    views, height, width = plan.selection_mask.shape
    if height % FIXED_TILE_SIZE or width % FIXED_TILE_SIZE:
        raise ValueError("guarded selected-route requires an exact T=4 tile layout")
    l1_anchor_semantics = plan.events.get(
        "l1_anchor_semantics", LEGACY_L1_ANCHOR_SEMANTICS
    )
    if l1_anchor_semantics not in {
        PAPER_KP_ANCHOR_SEMANTICS,
        LEGACY_L1_ANCHOR_SEMANTICS,
        BALANCED_L1_ANCHOR_SEMANTICS,
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    }:
        raise ValueError("guarded selected-route received an unknown L1 anchor semantics")
    expected_l1_count = (
        4
        if l1_anchor_semantics == PAPER_KP_ANCHOR_SEMANTICS
        else 15
        if l1_anchor_semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS
        else 12
    )
    allowed_decision_semantics = (
        {
            PAPER_DECISION_SEMANTICS,
            PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
        }
        if l1_anchor_semantics == PAPER_KP_ANCHOR_SEMANTICS
        else {
            FIXED_DECISION_SEMANTICS,
            PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
        }
    )
    decision_semantics = plan.events.get("decision_semantics")
    if (
        plan.events.get("contract_version") != "saes-incremental-probe-first-plan-v1"
        or plan.events.get("tile_size") != FIXED_TILE_SIZE
        or plan.events.get("feature_threshold") != FIXED_FEATURE_THRESHOLD
        or plan.events.get("depth_threshold") != FIXED_DEPTH_THRESHOLD
        or decision_semantics not in allowed_decision_semantics
        or plan.events.get("l0_anchor_count") != 4
        or plan.events.get("l1_anchor_count") != expected_l1_count
    ):
        raise ValueError("guarded selected-route received a non-fixed SAES candidate")
    if l1_anchor_semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS and (
        plan.events.get("l1_anchor_selection")
        != ADAPTIVE_L1_15_SELECTION_SEMANTICS
        or plan.events.get("l1_anchor_selection_uses_s1_only") is not True
    ):
        raise ValueError("guarded selected-route adaptive L1 selection is not source-bound")
    if not torch.equal(
        plan.selection_mask, plan.primary_mask | plan.secondary_mask | plan.full_mask
    ):
        raise ValueError("guarded selected-route plan union is inconsistent")
    if plan.selection_mask.device != features.device or depths.device != features.device:
        raise ValueError("guarded selected-route S1/S2 tensors must share the plan device")

    rebuilt = build_incremental_probe_first_plan(
        features,
        depths,
        height=height,
        width=width,
        tile_size=FIXED_TILE_SIZE,
        feature_threshold=FIXED_FEATURE_THRESHOLD,
        depth_threshold=FIXED_DEPTH_THRESHOLD,
        decision_semantics=decision_semantics,
        l1_anchor_semantics=l1_anchor_semantics,
    )
    for name in ("primary_mask", "secondary_mask", "full_mask", "selection_mask"):
        if not torch.equal(getattr(plan, name), getattr(rebuilt, name)):
            raise ValueError(f"guarded selected-route {name} does not match its S1/S2 plan")
    if plan.events.get("tile_trace_sha256") != rebuilt.events["tile_trace_sha256"]:
        raise ValueError("guarded selected-route tile trace does not match its S1/S2 plan")
    if len(plan.tile_trace) != views * (height // FIXED_TILE_SIZE) * (width // FIXED_TILE_SIZE):
        raise ValueError("guarded selected-route has an incomplete tile trace")
    return rebuilt


def _validate_packed_binding(
    packed: PackedGaussianAttributes,
    plan: IncrementalProbeFirstPlan,
) -> None:
    if not isinstance(packed, PackedGaussianAttributes):
        raise TypeError("guarded selected-route requires packed Gaussian attributes")
    trace = packed.source_trace
    if not isinstance(trace, Mapping):
        raise ValueError("selected packet has no source trace")
    if (
        trace.get("source_bound") is not True
        or trace.get("execution_scope") != "s3_raw_gaussian_head_only"
        or trace.get("adapter_side_inputs_source_bound") is not True
        or trace.get("adapter_side_inputs_same_scoped_invocation") is not True
        or trace.get("head_forward_invocations") != 1
    ):
        raise ValueError("selected packet is not bound to one scoped head invocation")
    if packed.source_trace_sha256 != _canonical_sha256(dict(trace)):
        raise ValueError("selected packet source trace digest is inconsistent")
    _require_hash(
        trace,
        "route_plan_contract_version",
        str(plan.events["contract_version"]),
        "route contract",
    )
    _require_hash(
        trace,
        "route_tile_trace_sha256",
        str(plan.events["tile_trace_sha256"]),
        "route tile trace",
    )
    for name, mask in (
        ("route_primary_mask_sha256", plan.primary_mask),
        ("route_secondary_mask_sha256", plan.secondary_mask),
        ("route_full_mask_sha256", plan.full_mask),
        ("route_selection_mask_sha256", plan.selection_mask),
    ):
        _require_hash(trace, name, _mask_sha256(mask), name.replace("route_", "route "))


def _expected_dense_slots(plan: IncrementalProbeFirstPlan) -> torch.Tensor:
    positions = plan.selection_mask.nonzero(as_tuple=False).to(dtype=torch.int64)
    _, height, width = plan.selection_mask.shape
    return positions[:, 0] * (height * width) + positions[:, 1] * width + positions[:, 2]


def _validate_packed_layout(
    packed: PackedGaussianAttributes, plan: IncrementalProbeFirstPlan
) -> torch.Tensor:
    expected_slots = _expected_dense_slots(plan).to(packed.dense_slots.device)
    slots = packed.dense_slots
    count = expected_slots.numel()
    if (
        slots.ndim != 1
        or slots.dtype != torch.int64
        or slots.shape != expected_slots.shape
        or not torch.equal(slots, expected_slots)
        or packed.batch_indices.shape != (count,)
        or packed.batch_indices.dtype != torch.int64
        or not bool((packed.batch_indices == 0).all())
        or packed.means.shape != (count, 3)
        or packed.covariances.shape != (count, 3, 3)
        or packed.harmonics.ndim != 3
        or packed.harmonics.shape[:2] != (count, 3)
        or packed.opacities.shape != (count,)
    ):
        raise ValueError("packed dense slots do not match the route selection")
    tensors = (packed.means, packed.covariances, packed.harmonics, packed.opacities)
    if any(value.device != slots.device for value in tensors):
        raise ValueError("packed selected attributes must share one device")
    if any(not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("packed selected attributes must be finite floating tensors")
    return expected_slots


def _sentinel_dense_view(
    packed: PackedGaussianAttributes,
    *,
    total_slots: int,
) -> SimpleNamespace:
    """Map selected attributes into a NaN-sentinel dense layout for guard calls.

    The guard implementation only indexes declared anchor slots.  A NaN-filled
    layout makes any accidental read of a skipped descriptor observable and
    preserves the selected-only input boundary without changing that guard's
    existing scalar implementation.
    """

    count = packed.dense_slots.numel()
    if count < 1 or total_slots < count:
        raise ValueError("selected attributes cannot form a guarded dense view")
    device = packed.dense_slots.device
    dtype = packed.means.dtype
    slots = packed.dense_slots
    means = torch.full((1, total_slots, 3), torch.nan, device=device, dtype=dtype)
    covariances = torch.full(
        (1, total_slots, 3, 3), torch.nan, device=device, dtype=packed.covariances.dtype
    )
    harmonics = torch.full(
        (1, total_slots, 3, packed.harmonics.shape[2]),
        torch.nan,
        device=device,
        dtype=packed.harmonics.dtype,
    )
    opacities = torch.full(
        (1, total_slots), torch.nan, device=device, dtype=packed.opacities.dtype
    )
    means[0, slots] = packed.means
    covariances[0, slots] = packed.covariances
    harmonics[0, slots] = packed.harmonics
    opacities[0, slots] = packed.opacities
    return SimpleNamespace(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )


def _tile_slots(
    *,
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    positions: list[tuple[int, int]],
) -> list[int]:
    return [
        view * (height * width)
        + (tile_y * FIXED_TILE_SIZE + row) * width
        + tile_x * FIXED_TILE_SIZE
        + column
        for row, column in positions
    ]


def _mark_tile_positions(
    mask: torch.Tensor,
    *,
    view: int,
    tile_y: int,
    tile_x: int,
    positions: list[tuple[int, int]],
) -> None:
    for row, column in positions:
        mask[
            view,
            tile_y * FIXED_TILE_SIZE + row,
            tile_x * FIXED_TILE_SIZE + column,
        ] = True


def _guard_summary(
    router: ProgressiveSAES,
    dense_view: SimpleNamespace,
    *,
    anchor_slots: list[int],
    primary_anchor_slots: list[int],
    depths: torch.Tensor,
    view: int,
    tile_y: int,
    tile_x: int,
    level: str,
) -> tuple[bool, bool, dict[str, Any]]:
    materialization = router.probe_materialization_validity(
        dense_view, anchor_slots, level=level
    )
    cross_check = (
        router.probe_cross_check_validity(
            dense_view,
            primary_anchor_slots,
            level=level,
        )
        if materialization["passed"]
        else {
            "level": level,
            "anchor_count": len(primary_anchor_slots),
            "error": None,
            "threshold": router.cross_check_threshold,
            "passed": True,
            "nonprobe_s3_attribute_reads": 0,
        }
    )
    primary_depths = router._primary_probe_depth_values(
        depths,
        tile_row=tile_y,
        tile_column=tile_x,
        view_index=view,
        primitive_slot=0,
    )
    context = router.context_coverage_occlusion_safety(
        dense_view,
        anchor_slots,
        primary_depths,
        view_index=view,
        level=level,
    )
    force_full = context["reason"] in {
        "missing_geometry",
        "invalid_probe_depth",
        "invalid_footprint",
        "center_separation",
    }
    return (
        bool(materialization["passed"] and cross_check["passed"] and context["passed"]),
        force_full or not cross_check["passed"],
        {
            "level": level,
            "anchor_count": len(anchor_slots),
            "anchor_dense_slots_sha256": _canonical_sha256(anchor_slots),
            "materialization": {
                key: materialization[key]
                for key in (
                    "passed",
                    "covariance_cosine_minimum",
                    "harmonic_cosine_minimum",
                    "opacity_distance_maximum",
                    "nonprobe_s3_attribute_reads",
                )
            },
            "probe_cross_check": {
                "checked": bool(materialization["passed"]),
                "passed": (
                    bool(cross_check["passed"])
                    if materialization["passed"]
                    else None
                ),
                "error": cross_check["error"],
                "threshold": cross_check["threshold"],
                "anchor_count": cross_check["anchor_count"],
                "nonprobe_s3_attribute_reads": cross_check[
                    "nonprobe_s3_attribute_reads"
                ],
            },
            "context_safety": {
                key: context[key]
                for key in (
                    "passed",
                    "coverage_footprint_ratio",
                    "relative_depth_span",
                    "projected_center_mahalanobis_max",
                    "center_overlap_passed",
                    "reason",
                    "nonprobe_s3_attribute_reads",
                )
            },
        },
    )


def _prepare_source_opacities_for_deletion_certificate(
    source_opacities: torch.Tensor | None,
    *,
    source_kind: str | None,
    views: int,
    height: int,
    width: int,
    packed: PackedGaussianAttributes,
) -> tuple[torch.Tensor | None, str]:
    """Validate the only source-only deletion certificate input."""
    if source_kind != DELETION_CERTIFICATE_SOURCE_KIND:
        return None, "untrusted_source_kind"
    if source_opacities is None:
        return None, "missing_source_opacities"
    if not torch.is_tensor(source_opacities):
        return None, "invalid_source_opacities"
    if tuple(source_opacities.shape) != (1, views, height * width, 1, 1):
        return None, "invalid_source_layout"
    if (
        source_opacities.device != packed.opacities.device
        or source_opacities.dtype != packed.opacities.dtype
        or not torch.is_floating_point(source_opacities)
    ):
        return None, "invalid_source_tensor_type"
    if not bool(torch.isfinite(source_opacities).all()) or bool(
        ((source_opacities < 0.0) | (source_opacities > 1.0)).any()
    ):
        return None, "invalid_source_opacity_values"
    return source_opacities.reshape(1, -1), "ready"


def _exact_source_opacity_zero_certificate(
    *,
    source_opacities: torch.Tensor | None,
    source_status: str,
    dense_view: SimpleNamespace,
    anchor_slots: list[int],
    nonprobe_slots: list[int],
    level: str,
) -> dict[str, Any]:
    """Certify deletion without reading a skipped Stage-3 attribute."""
    result = {
        "policy": DELETION_CERTIFICATE_POLICY,
        "level": level,
        "anchor_count": len(anchor_slots),
        "nonprobe_count": len(nonprobe_slots),
        "source_status": source_status,
        "passed": False,
        "reason": source_status,
        "anchor_s3_opacity_reads": 0,
        "nonprobe_s3_attribute_reads": 0,
    }
    if source_opacities is None:
        return result
    if (
        not anchor_slots
        or len(anchor_slots) != len(set(anchor_slots))
        or len(nonprobe_slots) != len(set(nonprobe_slots))
        or set(anchor_slots) & set(nonprobe_slots)
    ):
        result["reason"] = "invalid_tile_indices"
        return result
    total_slots = source_opacities.shape[1]
    if any(
        not isinstance(slot, int) or slot < 0 or slot >= total_slots
        for slot in (*anchor_slots, *nonprobe_slots)
    ):
        result["reason"] = "invalid_tile_indices"
        return result
    selected_source = source_opacities[0, anchor_slots]
    selected_output = dense_view.opacities[0, anchor_slots]
    result["anchor_s3_opacity_reads"] = len(anchor_slots)
    if not torch.equal(selected_source, selected_output):
        result["reason"] = "source_anchor_mapping_mismatch"
        return result
    nonprobe_source = source_opacities[0, nonprobe_slots]
    result["nonprobe_source_opacity"] = {
        "count": int(nonprobe_source.numel()),
        "strictly_positive_count": int((nonprobe_source > 0.0).sum().item()),
        "minimum": float(nonprobe_source.min().item()),
        "p50": float(torch.quantile(nonprobe_source, 0.50).item()),
        "p95": float(torch.quantile(nonprobe_source, 0.95).item()),
        "maximum": float(nonprobe_source.max().item()),
        "sum": float(nonprobe_source.sum().item()),
    }
    if not bool((nonprobe_source == 0.0).all()):
        result["reason"] = "nonzero_source_opacity"
        return result
    result.update({"passed": True, "reason": "exact_zero_alpha"})
    return result


def resolve_guarded_selected_route(
    plan: IncrementalProbeFirstPlan,
    packed: PackedGaussianAttributes,
    *,
    features: torch.Tensor,
    depths: torch.Tensor,
    context_extrinsics: torch.Tensor | None,
    context_intrinsics: torch.Tensor | None,
    source_opacities: torch.Tensor | None = None,
    source_opacity_certificate_kind: str | None = None,
    require_deletion_certificate: bool = False,
    execution_policy: str | None = None,
    adaptive_l1_maximum_leave_one_out_residual: float | None = None,
) -> GuardedSelectedRoute:
    """Resolve the fixed candidate's materialization guard without target data.

    The returned Full extension is an ordered *request*, not a synthetic
    descriptor.  Callers must execute it through the same source-bound head
    producer before they can consume those newly required slots.
    """

    if not isinstance(require_deletion_certificate, bool):
        raise ValueError("deletion certificate requirement must be boolean")
    if execution_policy is None:
        execution_policy = (
            STRICT_LOSSLESS_ZERO_POLICY
            if require_deletion_certificate
            else GUARD_ONLY_POLICY
        )
    if execution_policy not in {
        STRICT_LOSSLESS_ZERO_POLICY,
        PAPER_NONZERO_DEV_POLICY,
        ENGINEERING_L1_12_NONZERO_DEV_POLICY,
        ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
        ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY,
        ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY,
        ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY,
        ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
        ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
        GUARD_ONLY_POLICY,
    }:
        raise ValueError("guarded selected-route received an unknown execution policy")
    if (
        execution_policy
        in {
            PAPER_NONZERO_DEV_POLICY,
            ENGINEERING_L1_12_NONZERO_DEV_POLICY,
            ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
            ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
        }
        and require_deletion_certificate
    ):
        raise ValueError("paper nonzero materialization cannot claim a lossless certificate")
    if (
        execution_policy == STRICT_LOSSLESS_ZERO_POLICY
        and not require_deletion_certificate
    ):
        raise ValueError("strict lossless policy requires an exact-zero certificate")
    rebuilt = _validate_fixed_plan(plan, features=features, depths=depths)
    if (
        execution_policy == PAPER_NONZERO_DEV_POLICY
        and rebuilt.events.get("l1_anchor_semantics") != PAPER_KP_ANCHOR_SEMANTICS
    ):
        raise ValueError("paper nonzero materialization requires paper-kp-v1 anchors")
    if (
        execution_policy
        in {
            ENGINEERING_L1_12_NONZERO_DEV_POLICY,
            ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
        }
        and rebuilt.events.get("l1_anchor_semantics") != LEGACY_L1_ANCHOR_SEMANTICS
    ):
        raise ValueError("engineering L1 nonzero materialization requires 12 anchors")
    if (
        execution_policy == ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY
        and rebuilt.events.get("l1_anchor_semantics")
        != BALANCED_L1_ANCHOR_SEMANTICS
    ):
        raise ValueError("balanced L1 continuity materialization requires balanced anchors")
    if (
        execution_policy
        in {
            ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
        }
        and rebuilt.events.get("l1_anchor_semantics")
        != ADAPTIVE_L1_15_ANCHOR_SEMANTICS
    ):
        raise ValueError("adaptive L1 continuity materialization requires adaptive anchors")
    adaptive_l1_absolute_residual_guard = (
        execution_policy
        in {
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
        }
    )
    if adaptive_l1_absolute_residual_guard:
        if (
            isinstance(adaptive_l1_maximum_leave_one_out_residual, bool)
            or not isinstance(adaptive_l1_maximum_leave_one_out_residual, (int, float))
            or not torch.isfinite(
                torch.tensor(float(adaptive_l1_maximum_leave_one_out_residual))
            )
            or float(adaptive_l1_maximum_leave_one_out_residual) < 0.0
        ):
            raise ValueError("adaptive L1 absolute-residual policy requires a finite threshold")
    elif adaptive_l1_maximum_leave_one_out_residual is not None:
        raise ValueError("adaptive L1 residual threshold requires its dedicated policy")
    _validate_packed_binding(packed, rebuilt)
    expected_slots = _validate_packed_layout(packed, rebuilt)
    views, height, width = rebuilt.selection_mask.shape
    total_slots = views * height * width
    dense_view = _sentinel_dense_view(packed, total_slots=total_slots)
    if require_deletion_certificate:
        certificate_source_opacities, certificate_source_status = (
            _prepare_source_opacities_for_deletion_certificate(
                source_opacities,
                source_kind=source_opacity_certificate_kind,
                views=views,
                height=height,
                width=width,
                packed=packed,
            )
        )
    else:
        certificate_source_opacities = None
        certificate_source_status = "not_checked"
    router = ProgressiveSAES(
        height,
        width,
        initial_tile_size=FIXED_TILE_SIZE,
        feature_var_threshold=FIXED_FEATURE_THRESHOLD,
        depth_std_threshold=FIXED_DEPTH_THRESHOLD,
        view_count=views,
        primitives_per_pixel=1,
        materialization="representative",
        decision_semantics=rebuilt.events["decision_semantics"],
        cross_check_threshold=FIXED_CROSS_CHECK_THRESHOLD,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        materialization_guard=True,
        context_safety_guard=True,
    )
    primary_positions = compute_probe_positions(FIXED_TILE_SIZE)
    l1_anchor_semantics = rebuilt.events.get("l1_anchor_semantics")
    full_positions = [
        (row, column)
        for row in range(FIXED_TILE_SIZE)
        for column in range(FIXED_TILE_SIZE)
    ]
    expected_l1_count = int(rebuilt.events["l1_anchor_count"])
    if len(primary_positions) != 4 or expected_l1_count not in {4, 12, 15}:
        raise RuntimeError("guarded selected-route probe layout is invalid")

    route_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in rebuilt.tile_trace
    }
    if len(route_records) != len(rebuilt.tile_trace):
        raise ValueError("guarded selected-route tile trace has duplicate coordinates")
    selected_output_mask = torch.zeros_like(rebuilt.selection_mask)
    additional_full_mask = torch.zeros_like(rebuilt.selection_mask)
    trace: list[dict[str, Any]] = []
    route_counts = {"L0": 0, "L1": 0, "Full": 0}
    guard_anchor_reads = 0
    full_passthrough_slots: list[int] = []
    certificate_checks = 0
    certificate_accepted_tiles = 0
    certificate_rejected_tiles = 0
    certificate_zero_opacity_gaussians = 0
    certificate_rejection_reasons: dict[str, int] = {}
    literal_paper_formula_route = execution_policy in {
        PAPER_NONZERO_DEV_POLICY,
        ENGINEERING_L1_12_NONZERO_DEV_POLICY,
    }
    l0_depth_continuity_guard = execution_policy in {
        ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
        ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY,
        ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY,
        ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY,
        ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
        ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
    }
    adaptive_l1_confidence_guard = (
        execution_policy == ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY
    )

    for view in range(views):
        for tile_y in range(height // FIXED_TILE_SIZE):
            for tile_x in range(width // FIXED_TILE_SIZE):
                plan_record = route_records.get((view, tile_y, tile_x))
                if plan_record is None:
                    raise ValueError("guarded selected-route has a missing tile trace record")
                l1_positions = l1_local_positions_for_tile(
                    plan_record,
                    tile_size=FIXED_TILE_SIZE,
                    l1_anchor_semantics=str(l1_anchor_semantics),
                )
                if len(l1_positions) != expected_l1_count:
                    raise ValueError("guarded selected-route tile has an invalid L1 layout")
                pre_guard_route = plan_record.get("pre_guard_route")
                depth_uniform = plan_record.get("depth_uniform")
                if pre_guard_route not in {"L0", "L1", "Full"} or not isinstance(depth_uniform, bool):
                    raise ValueError("guarded selected-route tile trace is invalid")
                guard_checks: list[dict[str, Any]] = []
                l0_depth_continuity: dict[str, Any] | None = None
                adaptive_l1_confidence: dict[str, Any] | None = None
                adaptive_l1_absolute_residual: dict[str, Any] | None = None
                if adaptive_l1_confidence_guard:
                    residual = plan_record.get("adaptive_l1_leave_one_out_residual")
                    ratio = plan_record.get("adaptive_l1_best_to_second_residual_ratio")
                    if (
                        isinstance(residual, bool)
                        or not isinstance(residual, (int, float))
                        or isinstance(ratio, bool)
                        or not isinstance(ratio, (int, float))
                        or not torch.isfinite(torch.tensor((residual, ratio))).all()
                        or residual < 0.0
                        or ratio < 0.0
                    ):
                        raise ValueError("adaptive L1 confidence trace is invalid")
                    adaptive_l1_confidence = {
                        "checked": True,
                        "leave_one_out_residual": float(residual),
                        "best_to_second_residual_ratio": float(ratio),
                        "maximum_ratio": ADAPTIVE_L1_15_MAX_BEST_TO_SECOND_RESIDUAL_RATIO,
                        "passed": float(ratio)
                        <= ADAPTIVE_L1_15_MAX_BEST_TO_SECOND_RESIDUAL_RATIO,
                        "action": "not_applicable",
                    }
                if adaptive_l1_absolute_residual_guard:
                    residual = plan_record.get("adaptive_l1_leave_one_out_residual")
                    if (
                        isinstance(residual, bool)
                        or not isinstance(residual, (int, float))
                        or not torch.isfinite(torch.tensor(float(residual)))
                        or float(residual) < 0.0
                    ):
                        raise ValueError("adaptive L1 absolute residual trace is invalid")
                    maximum = float(adaptive_l1_maximum_leave_one_out_residual)
                    adaptive_l1_absolute_residual = {
                        "checked": True,
                        "leave_one_out_residual": float(residual),
                        "maximum_residual": maximum,
                        "passed": float(residual) <= maximum,
                        "action": "not_applicable",
                    }
                adaptive_l1_retention_passed = (
                    (adaptive_l1_confidence is None or adaptive_l1_confidence["passed"] is True)
                    and (
                        adaptive_l1_absolute_residual is None
                        or adaptive_l1_absolute_residual["passed"] is True
                    )
                )
                final_route = "Full"
                retained_positions = full_positions
                if l0_depth_continuity_guard and pre_guard_route == "L0":
                    # L0's feature-only hit is safe to retain only when the
                    # already scheduled primary depths agree. A depth-uniform
                    # tile widens to its pre-fetched L1 packet;
                    # otherwise the missing positions are requested on Full.
                    final_route = "L1" if depth_uniform and adaptive_l1_retention_passed else "Full"
                    retained_positions = (
                        l1_positions if final_route == "L1" else full_positions
                    )
                    l0_depth_continuity = {
                        "checked": True,
                        "depth_uniform": depth_uniform,
                        "action": (
                            "widen_l1"
                            if final_route == "L1"
                            else "promote_full_l1_confidence"
                            if depth_uniform and adaptive_l1_confidence is not None
                            and adaptive_l1_confidence["passed"] is not True
                            else "promote_full_l1_absolute_residual"
                            if depth_uniform
                            and adaptive_l1_absolute_residual is not None
                            and adaptive_l1_absolute_residual["passed"] is not True
                            else "promote_full"
                        ),
                    }
                    if adaptive_l1_confidence is not None:
                        adaptive_l1_confidence["action"] = (
                            "retain_l1"
                            if final_route == "L1"
                            else "promote_full"
                            if depth_uniform
                            else "not_evaluated_depth_nonuniform"
                        )
                    if adaptive_l1_absolute_residual is not None:
                        adaptive_l1_absolute_residual["action"] = (
                            "retain_l1"
                            if final_route == "L1"
                            else "promote_full"
                            if depth_uniform
                            else "not_evaluated_depth_nonuniform"
                        )
                elif literal_paper_formula_route or l0_depth_continuity_guard:
                    # Section 3 defines the L0/L1 first-hit decisions and
                    # probe aggregation, but no post-adaptor attribute guard.
                    # The compact materializer below remains the sole
                    # fail-closed numerical/geometry validation boundary.
                    final_route = (
                        "Full"
                        if pre_guard_route == "L1" and not adaptive_l1_retention_passed
                        else pre_guard_route
                    )
                    retained_positions = (
                        primary_positions
                        if final_route == "L0"
                        else l1_positions
                        if final_route == "L1"
                        else full_positions
                    )
                    if adaptive_l1_confidence is not None and pre_guard_route == "L1":
                        adaptive_l1_confidence["action"] = (
                            "retain_l1" if final_route == "L1" else "promote_full"
                        )
                    if (
                        adaptive_l1_absolute_residual is not None
                        and pre_guard_route == "L1"
                    ):
                        adaptive_l1_absolute_residual["action"] = (
                            "retain_l1" if final_route == "L1" else "promote_full"
                        )
                elif pre_guard_route == "L0":
                    l0_slots = _tile_slots(
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        height=height,
                        width=width,
                        positions=primary_positions,
                    )
                    passed, force_full, l0_trace = _guard_summary(
                        router,
                        dense_view,
                        anchor_slots=l0_slots,
                        primary_anchor_slots=l0_slots,
                        depths=depths,
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        level="L0",
                    )
                    guard_checks.append(l0_trace)
                    guard_anchor_reads += len(l0_slots)
                    if passed:
                        final_route = "L0"
                        retained_positions = primary_positions
                    elif (
                        not force_full
                        and depth_uniform
                        and rebuilt.events.get("l1_anchor_semantics")
                        != PAPER_KP_ANCHOR_SEMANTICS
                    ):
                        l1_slots = _tile_slots(
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            height=height,
                            width=width,
                            positions=l1_positions,
                        )
                        passed, _force_full, l1_trace = _guard_summary(
                            router,
                            dense_view,
                            anchor_slots=l1_slots,
                            primary_anchor_slots=l0_slots,
                            depths=depths,
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            level="L1",
                        )
                        guard_checks.append(l1_trace)
                        guard_anchor_reads += len(l1_slots)
                        if passed:
                            final_route = "L1"
                            retained_positions = l1_positions
                elif pre_guard_route == "L1":
                    l1_slots = _tile_slots(
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        height=height,
                        width=width,
                        positions=l1_positions,
                    )
                    passed, _force_full, l1_trace = _guard_summary(
                        router,
                        dense_view,
                        anchor_slots=l1_slots,
                        primary_anchor_slots=_tile_slots(
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            height=height,
                            width=width,
                            positions=primary_positions,
                        ),
                        depths=depths,
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        level="L1",
                    )
                    guard_checks.append(l1_trace)
                    guard_anchor_reads += len(l1_slots)
                    if passed:
                        final_route = "L1"
                        retained_positions = l1_positions

                certificate_trace = None
                if require_deletion_certificate and final_route in {"L0", "L1"}:
                    anchor_slots = _tile_slots(
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        height=height,
                        width=width,
                        positions=retained_positions,
                    )
                    tile_slots = _tile_slots(
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        height=height,
                        width=width,
                        positions=full_positions,
                    )
                    anchor_set = set(anchor_slots)
                    nonprobe_slots = [slot for slot in tile_slots if slot not in anchor_set]
                    certificate_trace = _exact_source_opacity_zero_certificate(
                        source_opacities=certificate_source_opacities,
                        source_status=certificate_source_status,
                        dense_view=dense_view,
                        anchor_slots=anchor_slots,
                        nonprobe_slots=nonprobe_slots,
                        level=final_route,
                    )
                    certificate_checks += 1
                    if certificate_trace["passed"]:
                        certificate_accepted_tiles += 1
                        certificate_zero_opacity_gaussians += len(nonprobe_slots)
                    else:
                        certificate_rejected_tiles += 1
                        reason = str(certificate_trace["reason"])
                        certificate_rejection_reasons[reason] = (
                            certificate_rejection_reasons.get(reason, 0) + 1
                        )
                        final_route = "Full"
                        retained_positions = full_positions
                _mark_tile_positions(
                    selected_output_mask,
                    view=view,
                    tile_y=tile_y,
                    tile_x=tile_x,
                    positions=retained_positions,
                )
                if final_route == "Full":
                    full_tile = torch.zeros_like(rebuilt.selection_mask)
                    _mark_tile_positions(
                        full_tile,
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        positions=full_positions,
                    )
                    additional_full_mask |= full_tile & ~rebuilt.selection_mask
                    available_full = full_tile & rebuilt.selection_mask
                    available_positions = available_full.nonzero(as_tuple=False)
                    full_passthrough_slots.extend(
                        (
                            available_positions[:, 0] * (height * width)
                            + available_positions[:, 1] * width
                            + available_positions[:, 2]
                        )
                        .to(device="cpu", dtype=torch.int64)
                        .tolist()
                    )
                route_counts[final_route] += 1
                trace.append(
                    {
                        "view": view,
                        "tile_y": tile_y,
                        "tile_x": tile_x,
                        "pre_guard_route": pre_guard_route,
                        "depth_uniform": depth_uniform,
                        "l0_depth_continuity": l0_depth_continuity,
                        "adaptive_l1_confidence": adaptive_l1_confidence,
                        "adaptive_l1_absolute_residual": adaptive_l1_absolute_residual,
                        "guard_checks": guard_checks,
                        "deletion_certificate": certificate_trace,
                        "retained_local_positions": [
                            list(position) for position in retained_positions
                        ],
                        "final_route": final_route,
                    }
                )

    if bool((additional_full_mask & rebuilt.selection_mask).any()):
        raise RuntimeError("guarded selected-route Full extension overlaps produced slots")
    raw_head_request_mask = rebuilt.selection_mask | additional_full_mask
    if bool((selected_output_mask & ~raw_head_request_mask).any()):
        raise RuntimeError("guarded selected-route output is not covered by its head requests")
    cross_check_records = [
        check["probe_cross_check"]
        for record in trace
        for check in record["guard_checks"]
        if check["probe_cross_check"]["checked"]
    ]
    l0_depth_continuity_records = [
        record["l0_depth_continuity"]
        for record in trace
        if record["l0_depth_continuity"] is not None
    ]
    adaptive_l1_confidence_records = [
        record["adaptive_l1_confidence"]
        for record in trace
        if record.get("adaptive_l1_confidence") is not None
    ]
    adaptive_l1_absolute_residual_records = [
        record["adaptive_l1_absolute_residual"]
        for record in trace
        if record.get("adaptive_l1_absolute_residual") is not None
    ]
    events = {
        "schema_version": ROUTE_SCHEMA_VERSION,
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "target_rgb_accessed_before_commit": False,
        "target_camera_metadata_accessed": False,
        "quality_metrics_computed": False,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "scope": "target_free_selected_route_guard_only",
        "materialization": "representative",
        "execution_policy": execution_policy,
        "not_lossless_deletion": execution_policy
        in {
            PAPER_NONZERO_DEV_POLICY,
            ENGINEERING_L1_12_NONZERO_DEV_POLICY,
            ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
            ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
        },
        "source_nonprobe_s3_attribute_reads": 0,
        "context_safety_guard": not literal_paper_formula_route
        and not l0_depth_continuity_guard,
        "l0_depth_continuity_guard": l0_depth_continuity_guard,
        "l0_depth_continuity_widened_tiles": sum(
            record["action"] == "widen_l1" for record in l0_depth_continuity_records
        ),
        "l0_depth_continuity_promoted_full_tiles": sum(
            record["action"] == "promote_full" for record in l0_depth_continuity_records
        ),
        "adaptive_l1_confidence_guard": adaptive_l1_confidence_guard,
        "adaptive_l1_confidence_maximum_ratio": (
            ADAPTIVE_L1_15_MAX_BEST_TO_SECOND_RESIDUAL_RATIO
            if adaptive_l1_confidence_guard
            else None
        ),
        "adaptive_l1_confidence_checked_tiles": len(adaptive_l1_confidence_records),
        "adaptive_l1_confidence_retained_l1_tiles": sum(
            record["action"] == "retain_l1"
            for record in adaptive_l1_confidence_records
        ),
        "adaptive_l1_confidence_promoted_full_tiles": sum(
            record["action"] == "promote_full"
            for record in adaptive_l1_confidence_records
        ),
        "adaptive_l1_absolute_residual_guard": adaptive_l1_absolute_residual_guard,
        "adaptive_l1_absolute_residual_maximum": (
            float(adaptive_l1_maximum_leave_one_out_residual)
            if adaptive_l1_absolute_residual_guard
            else None
        ),
        "adaptive_l1_absolute_residual_checked_tiles": len(
            adaptive_l1_absolute_residual_records
        ),
        "adaptive_l1_absolute_residual_retained_l1_tiles": sum(
            record["action"] == "retain_l1"
            for record in adaptive_l1_absolute_residual_records
        ),
        "adaptive_l1_absolute_residual_promoted_full_tiles": sum(
            record["action"] == "promote_full"
            for record in adaptive_l1_absolute_residual_records
        ),
        "attribute_guard_mode": (
            "paper-formula-no-extra-attribute-guard"
            if literal_paper_formula_route
            else "l1-15-s1-dominance-widen-or-full-v1"
            if adaptive_l1_confidence_guard
            else "l1-15-v4-attribute-loo-preflight-widen-or-full-v1"
            if execution_policy
            == ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
            else "l1-15-absolute-s1-loo-widen-or-full-v1"
            if adaptive_l1_absolute_residual_guard
            else "l0-depth-consistency-widen-or-full-v1"
            if l0_depth_continuity_guard
            else "context-and-attribute-guard-v1"
        ),
        "deletion_certificate_required": require_deletion_certificate,
        "deletion_certificate_policy": DELETION_CERTIFICATE_POLICY,
        "deletion_certificate_source_kind": source_opacity_certificate_kind,
        "deletion_certificate_source_status": certificate_source_status,
        "deletion_certificate_checks": certificate_checks,
        "deletion_certificate_accepted_tiles": certificate_accepted_tiles,
        "deletion_certificate_rejected_tiles": certificate_rejected_tiles,
        "deletion_certificate_zero_opacity_gaussians": certificate_zero_opacity_gaussians,
        "deletion_certificate_anchor_s3_opacity_reads": sum(
            int(record["deletion_certificate"]["anchor_s3_opacity_reads"])
            for record in trace
            if record["deletion_certificate"] is not None
        ),
        "deletion_certificate_nonprobe_s3_attribute_reads": 0,
        "deletion_certificate_rejection_reasons": certificate_rejection_reasons,
        "uncertified_deletion_fallback_tiles": certificate_rejected_tiles,
        "probe_cross_check_threshold": FIXED_CROSS_CHECK_THRESHOLD,
        "probe_cross_check_checks": len(cross_check_records),
        "probe_cross_check_rejections": sum(
            not record["passed"] for record in cross_check_records
        ),
        "tile_size": FIXED_TILE_SIZE,
        "l0_anchor_count": len(primary_positions),
        "l1_anchor_count": expected_l1_count,
        "l1_anchor_semantics": rebuilt.events.get("l1_anchor_semantics"),
        "source_selection_mask_sha256": _mask_sha256(rebuilt.selection_mask),
        "selected_output_mask_sha256": _mask_sha256(selected_output_mask),
        "additional_full_mask_sha256": _mask_sha256(additional_full_mask),
        "raw_head_request_mask_sha256": _mask_sha256(raw_head_request_mask),
        "source_tile_trace_sha256": rebuilt.events["tile_trace_sha256"],
        "route_counts": route_counts,
        "guard_anchor_attribute_reads": guard_anchor_reads,
        "guard_nonprobe_s3_attribute_reads": 0,
        "available_selected_descriptor_count": int(expected_slots.numel()),
        "unavailable_dense_descriptor_count": int(total_slots - expected_slots.numel()),
        "selected_output_descriptor_count": int(selected_output_mask.sum().item()),
        "additional_full_descriptor_count": int(additional_full_mask.sum().item()),
        "requires_incremental_full_dispatch": bool(additional_full_mask.any()),
        "full_passthrough_dense_slots": sorted(full_passthrough_slots),
        "full_tile_native_identity_verified": False,
        "tile_trace_sha256": _canonical_sha256(trace),
    }
    return GuardedSelectedRoute(
        selected_output_mask=selected_output_mask,
        additional_full_mask=additional_full_mask,
        raw_head_request_mask=raw_head_request_mask,
        tile_trace=tuple(trace),
        events=events,
    )


__all__ = [
    "FIXED_CROSS_CHECK_THRESHOLD",
    "FIXED_DECISION_SEMANTICS",
    "FIXED_DEPTH_THRESHOLD",
    "FIXED_FEATURE_THRESHOLD",
    "FIXED_TILE_SIZE",
    "ENGINEERING_L1_12_NONZERO_DEV_POLICY",
    "ENGINEERING_L1_12_CONTINUITY_DEV_POLICY",
    "ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY",
    "ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY",
    "ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY",
    "ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY",
    "ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY",
    "GUARD_ONLY_POLICY",
    "GuardedSelectedRoute",
    "PAPER_NONZERO_DEV_POLICY",
    "PAPER_DECISION_SEMANTICS",
    "ROUTE_SCHEMA_VERSION",
    "STRICT_LOSSLESS_ZERO_POLICY",
    "resolve_guarded_selected_route",
]
