"""DepthSplat-native nonzero L0/L1 Gaussian materialization.

DepthSplat has a different producer and geometry contract from TranSplat and
MVSplat.  Its raw descriptor starts with an opacity logit, its Adapter derives
the SH DC term from selected context RGB, and its depth is camera *z-depth*.
This module therefore intentionally does not consume the classic packet or
coordinate helpers.  It uses a target-free probe plan only to choose anchors,
constructs skipped-domain virtual Gaussians from selected native attributes
and the already source-bound S2 z-depth map, then absorbs them into retained
anchors by first/second moment matching.

The materializer is fail-closed.  Any provenance, RGB/SH, z-depth, PSD,
continuity, opacity, or support-coverage inconsistency promotes the whole tile
to its source-native Full path before the final packet is committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from saes.depthsplat_backend import canonical_json_sha256
from saes.depthsplat_selected_output import (
    DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
    DepthSplatPackedGaussianAttributes,
    DepthSplatSparseRawPacket,
    depthsplat_attribute_binding_sha256,
)
from saes.probe_first_schedule import (
    ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    BALANCED_L1_ANCHOR_SEMANTICS,
    LEGACY_L1_ANCHOR_SEMANTICS,
    LITERAL_PAPER_T4_PLAN_CONTRACT,
    PAPER_KP_ANCHOR_SEMANTICS,
    IncrementalProbeFirstPlan,
    build_literal_paper_t4_probe_first_plan,
    build_incremental_probe_first_plan,
    literal_paper_t4_route_config_sha256,
    l1_local_positions_for_tile,
)
from saes.probe_layout import compute_probe_positions
from saes.progressive_saes import ProgressiveSAES, paper_assignment_weights


DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION = "saes-depthsplat-l0-l1-materializer-v1"
DEPTHSPLAT_COMPACT_AGGREGATION = (
    "depthsplat-selected-rgb-sh-z-depth-conditional-moment-merge-v1"
)
DEPTHSPLAT_COVERAGE_CERTIFICATE = (
    "depthsplat-selected-z-depth-3d-2sigma-ellipsoid-support-v2"
)
DEPTHSPLAT_COMPACT_COVERAGE_MAX_COVARIANCE_SCALE = 16.0
DEPTHSPLAT_FEATURE_INTERPOLATION_MAX_RELATIVE_RESIDUAL = 1.0
DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE = (
    "depthsplat-development-omitted-z-alpha-union-v1"
)
DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE = (
    "depthsplat-literal-paper-t4-selected-probe-moment-v1"
)
DEPTHSPLAT_LITERAL_PAPER_T4_DECISION_SEMANTICS = (
    "paper-probe-feature-variance-first-hit"
)
DEPTHSPLAT_LITERAL_PAPER_T4_V16_RECORD_KIND = (
    "depthsplat-nonzero-l0-l1-acid-disjoint-v16l-t4"
)
NATIVE_OPACITY_ENDPOINT_FULL_REASON = "native_opacity_endpoint_requires_full"
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE = (
    "depthsplat-selected-rgb-sh-opacity-all-anchor-loo-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY = (
    "all-retained-l0-l1-anchors-selected-labels-only-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_AGGREGATE_SCHEMA = (
    "depthsplat-selected-anchor-attribute-loo-aggregate-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_GUARD_SCHEMA = (
    "depthsplat-selected-anchor-attribute-loo-frozen-v16-guard-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC = (
    "maximum-held-out-anchor-risk-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_ENDPOINT_UNSCORABLE = (
    "native-opacity-endpoint-promoted-full-v1"
)


@dataclass(frozen=True)
class DepthSplatCompactMaterializationPreflight:
    """Selected-only updates and Full promotions for one probe plan."""

    update_dense_slots: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor
    promote_full_mask: torch.Tensor
    tile_trace: tuple[dict[str, Any], ...]
    events: dict[str, Any]


@dataclass(frozen=True)
class DepthSplatCompactFinalRoute:
    """Final decoder output and producer request masks after preflight."""

    selected_output_mask: torch.Tensor
    additional_full_mask: torch.Tensor
    raw_head_request_mask: torch.Tensor
    full_passthrough_mask: torch.Tensor
    tile_trace: tuple[dict[str, Any], ...]
    events: dict[str, Any]


def _tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("DepthSplat materializer tensor digest requires a tensor")
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _mask_sha256(mask: torch.Tensor) -> str:
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise ValueError("DepthSplat materializer mask must be boolean")
    return _tensor_sha256(mask.to(dtype=torch.uint8))


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("DepthSplat materializer trace must be JSON-serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"DepthSplat materializer has no valid {label} hash")
    return value


def _full_positions(tile_size: int) -> list[tuple[int, int]]:
    return [(row, column) for row in range(tile_size) for column in range(tile_size)]


def _tile_slots(
    *,
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    tile_size: int,
    positions: list[tuple[int, int]],
) -> list[int]:
    slots: list[int] = []
    for row, column in positions:
        if not 0 <= row < tile_size or not 0 <= column < tile_size:
            raise ValueError("DepthSplat materializer local anchor is invalid")
        absolute_row = tile_y * tile_size + row
        absolute_column = tile_x * tile_size + column
        if not 0 <= absolute_row < height or not 0 <= absolute_column < width:
            raise ValueError("DepthSplat materializer anchor falls outside its image")
        slots.append(view * (height * width) + absolute_row * width + absolute_column)
    return slots


def _mark_tile(
    mask: torch.Tensor,
    *,
    view: int,
    tile_y: int,
    tile_x: int,
    tile_size: int,
    positions: list[tuple[int, int]] | None = None,
) -> None:
    for row, column in positions if positions is not None else _full_positions(tile_size):
        mask[view, tile_y * tile_size + row, tile_x * tile_size + column] = True


def _require_plan(plan: IncrementalProbeFirstPlan) -> tuple[int, int, int, str]:
    if not isinstance(plan, IncrementalProbeFirstPlan):
        raise TypeError("DepthSplat materializer requires an incremental probe plan")
    events = plan.events
    if events.get("contract_version") not in {
        "saes-incremental-probe-first-plan-v1",
        LITERAL_PAPER_T4_PLAN_CONTRACT,
    }:
        raise ValueError("DepthSplat materializer plan contract changed")
    tile_size = events.get("tile_size")
    if tile_size != 4:
        raise ValueError("DepthSplat materializer supports exactly T=4")
    masks = (plan.primary_mask, plan.secondary_mask, plan.full_mask, plan.selection_mask)
    if any(not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 3 for mask in masks):
        raise ValueError("DepthSplat materializer plan masks are invalid")
    if not torch.equal(plan.selection_mask, plan.primary_mask | plan.secondary_mask | plan.full_mask):
        raise ValueError("DepthSplat materializer plan mask union is inconsistent")
    views, height, width = plan.selection_mask.shape
    if min(views, height, width) < 1 or height % tile_size or width % tile_size:
        raise ValueError("DepthSplat materializer plan geometry is invalid")
    semantics = events.get("l1_anchor_semantics", LEGACY_L1_ANCHOR_SEMANTICS)
    if semantics not in {
        PAPER_KP_ANCHOR_SEMANTICS,
        LEGACY_L1_ANCHOR_SEMANTICS,
        BALANCED_L1_ANCHOR_SEMANTICS,
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    }:
        raise ValueError("DepthSplat materializer plan L1 semantics are invalid")
    if len(plan.tile_trace) != views * (height // tile_size) * (width // tile_size):
        raise ValueError("DepthSplat materializer plan tile trace is incomplete")
    return views, height, width, str(semantics)


def _validate_literal_paper_t4_plan(
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
    semantics: str,
) -> None:
    """Reject any engineering routing behavior from the formal T=4 profile."""

    events = plan.events
    if (
        events.get("contract_version") != LITERAL_PAPER_T4_PLAN_CONTRACT
        or events.get("formal_paper_kp4") is not True
        or events.get("decision_semantics")
        != DEPTHSPLAT_LITERAL_PAPER_T4_DECISION_SEMANTICS
        or events.get("feature_statistic") != "raw-probe-mean-channel-variance"
        or semantics != PAPER_KP_ANCHOR_SEMANTICS
        or events.get("l0_anchor_count") != 4
        or events.get("l1_anchor_count") != 4
        or events.get("l1_anchor_selection") != "fixed-layout-v1"
        or events.get("l1_anchor_selection_uses_s1_only") is not False
        or events.get("depth_checked_after_l0_miss_only") is not True
        or int(plan.secondary_mask.sum().item()) != 0
        or not torch.equal(plan.selection_mask, plan.primary_mask | plan.full_mask)
    ):
        raise ValueError("DepthSplat formal paper T=4 routing contract changed")
    corners = [list(position) for position in compute_probe_positions(4)]
    expected_tiles = views * (height // 4) * (width // 4)
    if len(plan.tile_trace) != expected_tiles:
        raise ValueError("DepthSplat formal paper T=4 tile trace is incomplete")
    for record in plan.tile_trace:
        if not isinstance(record, Mapping):
            raise ValueError("DepthSplat formal paper T=4 tile record is invalid")
        route = record.get("pre_guard_route")
        if (
            route not in {"L0", "L1", "Full"}
            or record.get("primary_local_positions") != corners
            or record.get("secondary_local_positions") != []
            or record.get("depth_checked_after_l0_miss_only") is not True
            or "adaptive_l1_omitted_local_position" in record
            or "adaptive_l1_leave_one_out_residual" in record
        ):
            raise ValueError("DepthSplat formal paper T=4 route is not literal")
        if route == "L0" and record.get("depth_uniform") is not None:
            raise ValueError("DepthSplat formal L0 route inspected depth before a miss")
        if route in {"L1", "Full"} and not isinstance(record.get("depth_uniform"), bool):
            raise ValueError("DepthSplat formal L1/Full route lacks conditional depth evidence")


def _validate_source(
    packet: DepthSplatSparseRawPacket,
    packed: DepthSplatPackedGaussianAttributes,
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
) -> dict[int, int]:
    if not isinstance(packet, DepthSplatSparseRawPacket) or not isinstance(
        packed, DepthSplatPackedGaussianAttributes
    ):
        raise TypeError("DepthSplat materializer requires native selected packet and attributes")
    count = int(packet.dense_slots.numel())
    if (
        count < 1
        or packet.descriptor_keys.shape != (count, 4)
        or packet.descriptor_keys.dtype != torch.int64
        or packet.raw_head_descriptors.ndim != 2
        or packet.raw_head_descriptors.shape[0] != count
        or packet.raw_head_descriptors.shape[1] < 4
        or packet.extrinsics.shape != (count, 4, 4)
        or packet.intrinsics.shape != (count, 3, 3)
        or packet.coordinates.shape != (count, 2)
        or packet.depths.shape != (count,)
        or packet.mapped_opacities.shape != (count,)
        or packet.source_rgb.shape != (count, 3)
        or packet.dense_slots.shape != (count,)
        or packet.dense_slots.dtype != torch.int64
        or packed.dense_slots.shape != (count,)
        or not torch.equal(packet.dense_slots, packed.dense_slots)
        or packed.means.shape != (count, 3)
        or packed.covariances.shape != (count, 3, 3)
        or packed.harmonics.ndim != 3
        or packed.harmonics.shape[:2] != (count, 3)
        or packed.opacities.shape != (count,)
    ):
        raise ValueError("DepthSplat materializer selected packet tensors are inconsistent")
    values = (
        packet.raw_head_descriptors,
        packet.extrinsics,
        packet.intrinsics,
        packet.coordinates,
        packet.depths,
        packet.mapped_opacities,
        packet.source_rgb,
        packed.means,
        packed.covariances,
        packed.harmonics,
        packed.opacities,
    )
    if any(value.device != packet.raw_head_descriptors.device for value in values):
        raise ValueError("DepthSplat materializer inputs must share one device")
    if any(value.dtype != torch.float32 for value in values):
        raise ValueError("DepthSplat materializer requires strict float32 native attributes")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("DepthSplat materializer selected inputs must be finite")
    if not bool((packet.depths > 0.0).all()):
        raise ValueError("DepthSplat materializer requires positive selected z-depths")
    if not torch.allclose(
        packet.mapped_opacities,
        packet.raw_head_descriptors[:, 0].sigmoid(),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("DepthSplat materializer packet opacity logit drifted")
    if not bool((packed.opacities >= 0.0).all()) or not bool((packed.opacities <= 1.0).all()):
        raise ValueError("DepthSplat materializer selected opacities are invalid")
    covariance = (packed.covariances + packed.covariances.mT) * 0.5
    if float((packed.covariances - packed.covariances.mT).abs().amax().item()) > 1e-4:
        raise ValueError("DepthSplat materializer selected covariance is not symmetric")
    if bool((torch.linalg.eigvalsh(covariance) < -1e-6).any()):
        raise ValueError("DepthSplat materializer selected covariance is not PSD")
    expected_positions = plan.selection_mask.nonzero(as_tuple=False)
    expected_slots = (
        expected_positions[:, 0] * (height * width)
        + expected_positions[:, 1] * width
        + expected_positions[:, 2]
    ).to(device=packet.dense_slots.device, dtype=torch.int64)
    if not torch.equal(packet.dense_slots, expected_slots):
        raise ValueError("DepthSplat materializer packet does not match planned producer selection")
    trace = packet.source_trace
    required_trace = {
        "contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
        "source_bound": True,
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
        "source_rgb_keyword": "input_images",
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "head_forward_invocations": 1,
    }
    if not isinstance(trace, Mapping) or any(trace.get(key) != value for key, value in required_trace.items()):
        raise ValueError("DepthSplat materializer packet lacks native RGB/z-depth provenance")
    if trace.get("selection_mask_sha256") != _mask_sha256(plan.selection_mask):
        raise ValueError("DepthSplat materializer packet selection hash drifted")
    if (
        _require_sha256(trace.get("selected_descriptor_sha256"), label="selected descriptor")
        != _tensor_sha256(packet.raw_head_descriptors)
        or _require_sha256(trace.get("selected_rgb_sha256"), label="selected RGB")
        != _tensor_sha256(packet.source_rgb)
        or _require_sha256(
            trace.get("native_full_passthrough_mask_sha256"), label="native Full mask"
        )
        != _mask_sha256(plan.full_mask)
        or trace.get("native_full_passthrough_positions")
        != int(plan.full_mask.sum().item())
        or trace.get("source_view_count") != views
        or trace.get("source_image_shape") != [height, width]
    ):
        raise ValueError("DepthSplat materializer packet source geometry binding drifted")
    native_execution_sha256 = _require_sha256(
        trace.get("native_execution_sha256"), label="native execution"
    )
    if packed.source_trace.get("source_bound") is not True:
        raise ValueError("DepthSplat materializer packed attributes are not source-bound")
    if packed.source_trace_sha256 != canonical_json_sha256(dict(packed.source_trace)):
        raise ValueError("DepthSplat materializer packed trace digest drifted")
    attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=packed.dense_slots,
        means=packed.means,
        covariances=packed.covariances,
        harmonics=packed.harmonics,
        opacities=packed.opacities,
    )
    if (
        packed.attribute_binding_sha256 != attribute_binding
        or packed.source_trace.get("native_adapter_attribute_binding_sha256")
        != attribute_binding
    ):
        raise ValueError("DepthSplat materializer native Adapter attributes are unbound")
    full_positions = plan.full_mask.nonzero(as_tuple=False).to(device=packed.dense_slots.device)
    full_slots = (
        full_positions[:, 0] * (height * width)
        + full_positions[:, 1] * width
        + full_positions[:, 2]
    ).to(dtype=torch.int64)
    full_indices = torch.searchsorted(packed.dense_slots, full_slots)
    if bool((full_indices >= packed.dense_slots.numel()).any()) or not torch.equal(
        packed.dense_slots[full_indices], full_slots
    ):
        raise ValueError("DepthSplat materializer native Full slot is absent")
    full_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=full_slots,
        means=packed.means[full_indices],
        covariances=packed.covariances[full_indices],
        harmonics=packed.harmonics[full_indices],
        opacities=packed.opacities[full_indices],
    )
    if (
        packed.source_trace.get("native_full_adapter_attribute_execution_sha256")
        != native_execution_sha256
        or packed.source_trace.get("native_full_adapter_attribute_passthrough_mask_sha256")
        != _mask_sha256(plan.full_mask)
        or packed.source_trace.get("native_full_adapter_attribute_passthrough_count")
        != int(full_slots.numel())
        or packed.source_trace.get("native_full_adapter_attribute_binding_sha256")
        != full_attribute_binding
    ):
        raise ValueError("DepthSplat materializer native Full attributes are unbound")
    expected_packed_trace = dict(packet.source_trace)
    expected_packed_trace.update(
        {
            "native_adapter_attribute_binding_sha256": attribute_binding,
            "native_full_adapter_attribute_execution_sha256": native_execution_sha256,
            "native_full_adapter_attribute_passthrough_mask_sha256": _mask_sha256(
                plan.full_mask
            ),
            "native_full_adapter_attribute_binding_sha256": full_attribute_binding,
            "native_full_adapter_attribute_passthrough_count": int(full_slots.numel()),
            "selected_native_rgb_adapter_compact_count": count - int(full_slots.numel()),
            "selected_native_rgb_adapter_executed": count > int(full_slots.numel()),
        }
    )
    if dict(packed.source_trace) != expected_packed_trace:
        raise ValueError("DepthSplat materializer packet and Adapter traces diverged")
    slots = packet.dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(set(slots)) != len(slots) or any(slot < 0 or slot >= views * height * width for slot in slots):
        raise ValueError("DepthSplat materializer selected slots are invalid")
    if len(slots) > 1 and any(right <= left for left, right in zip(slots, slots[1:])):
        raise ValueError("DepthSplat materializer selected slots must be strictly ordered")
    return {int(slot): index for index, slot in enumerate(slots)}


def _validate_routing_inputs(
    routing_features: torch.Tensor,
    routing_z_depths: torch.Tensor,
    packet: DepthSplatSparseRawPacket,
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
    execution_profile: str,
) -> tuple[torch.Tensor, str]:
    if (
        not torch.is_tensor(routing_features)
        or routing_features.ndim != 5
        or routing_features.shape[0] != 1
        or routing_features.shape[1] != views
        or tuple(routing_features.shape[-2:]) != (height, width)
        or routing_features.shape[2] < 1
        or routing_features.dtype != torch.float32
        or routing_features.device != packet.raw_head_descriptors.device
        or not bool(torch.isfinite(routing_features).all())
    ):
        raise ValueError("DepthSplat materializer routing features are invalid")
    if (
        not torch.is_tensor(routing_z_depths)
        or routing_z_depths.shape != (1, views, height, width)
        or routing_z_depths.device != packet.raw_head_descriptors.device
        or routing_z_depths.dtype != torch.float32
        or not bool(torch.isfinite(routing_z_depths).all())
        or bool((routing_z_depths <= 0.0).any())
    ):
        raise ValueError("DepthSplat materializer routing z-depths are invalid")
    if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
        rebuilt = build_literal_paper_t4_probe_first_plan(
            routing_features,
            routing_z_depths,
            height=height,
            width=width,
            feature_threshold=float(plan.events["feature_threshold"]),
            depth_threshold=float(plan.events["depth_threshold"]),
        )
    elif execution_profile == DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE:
        rebuilt = build_incremental_probe_first_plan(
            routing_features,
            routing_z_depths,
            height=height,
            width=width,
            tile_size=4,
            feature_threshold=float(plan.events["feature_threshold"]),
            depth_threshold=float(plan.events["depth_threshold"]),
            decision_semantics=str(plan.events["decision_semantics"]),
            l1_anchor_semantics=str(plan.events["l1_anchor_semantics"]),
        )
    else:
        raise ValueError("DepthSplat materializer execution profile is invalid")
    if (
        not torch.equal(rebuilt.primary_mask, plan.primary_mask)
        or not torch.equal(rebuilt.secondary_mask, plan.secondary_mask)
        or not torch.equal(rebuilt.full_mask, plan.full_mask)
        or rebuilt.events.get("tile_trace_sha256") != plan.events.get("tile_trace_sha256")
    ):
        raise ValueError("DepthSplat materializer routing tensors do not reproduce the frozen plan")
    trace = packet.source_trace
    if _require_sha256(
        trace.get("routing_features_sha256"), label="routing features"
    ) != _tensor_sha256(routing_features):
        raise ValueError("DepthSplat materializer routing feature hash drifted")
    if _require_sha256(
        trace.get("routing_z_depths_sha256"), label="routing z-depth"
    ) != _tensor_sha256(routing_z_depths):
        raise ValueError("DepthSplat materializer routing z-depth hash drifted")
    positions = plan.selection_mask.nonzero(as_tuple=False)
    expected_depths = routing_z_depths[0, positions[:, 0], positions[:, 1], positions[:, 2]]
    if not torch.allclose(packet.depths, expected_depths, rtol=1e-5, atol=1e-5):
        raise ValueError("DepthSplat materializer selected packet z-depth drifted from router")
    feature_statistic = plan.events.get("feature_statistic")
    if not isinstance(feature_statistic, str):
        raise ValueError("DepthSplat materializer plan feature statistic is invalid")
    _scores, feature_norm = ProgressiveSAES.classify_tiles_by_features(
        routing_features,
        height,
        width,
        4,
        threshold=float(plan.events["feature_threshold"]),
        per_view=True,
        statistic=feature_statistic,
    )
    if (
        not torch.is_tensor(feature_norm)
        or feature_norm.shape != (views, routing_features.shape[2], height, width)
        or not bool(torch.isfinite(feature_norm).all())
    ):
        raise RuntimeError("DepthSplat materializer could not normalize routing features")
    if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
        # The literal route measures raw probe variance.  The bilateral
        # assignment must therefore use the same raw bilinear S1 map rather
        # than ``classify_tiles_by_features``' unit-normalized diagnostic map.
        raw_assignment_features = torch.nn.functional.interpolate(
            routing_features[0], size=(height, width), mode="bilinear", align_corners=False
        )
        if (
            raw_assignment_features.shape
            != (views, routing_features.shape[2], height, width)
            or not bool(torch.isfinite(raw_assignment_features).all())
        ):
            raise RuntimeError("DepthSplat materializer could not upsample raw S1 features")
        return raw_assignment_features, "raw-bilinear-s1-v1"
    return feature_norm, "unit-normalized-bilinear-s1-v1"


def _tile_feature_variance(record: Mapping[str, Any], feature_statistic: str) -> float:
    score = record.get("feature_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("DepthSplat materializer tile feature score is invalid")
    score = float(score)
    if not torch.isfinite(torch.tensor(score)) or score < 0.0:
        raise ValueError("DepthSplat materializer tile feature score is non-finite")
    return score * score if feature_statistic == "normalized-probe-vector-standard-deviation" else score


def _pixel_centres(
    positions: list[tuple[int, int]],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not positions:
        return torch.empty(0, 2, device=device, dtype=dtype)
    if any(row < 0 or row >= height or column < 0 or column >= width for row, column in positions):
        raise ValueError("DepthSplat materializer position is outside the image")
    rows = torch.tensor([row for row, _ in positions], device=device, dtype=dtype)
    columns = torch.tensor([column for _, column in positions], device=device, dtype=dtype)
    return torch.stack(((columns + 0.5) / width, (rows + 0.5) / height), dim=1)


def _source_image_grid(
    source_sample_image_grid: Callable[..., tuple[Any, Any]],
    *,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Capture the loaded source pixel-center grid once per preflight."""

    if not callable(source_sample_image_grid):
        raise TypeError("DepthSplat materializer requires source sample_image_grid")
    grid, _indices = source_sample_image_grid((height, width), device)
    if (
        not torch.is_tensor(grid)
        or grid.shape != (height, width, 2)
        or grid.device != device
        or not bool(torch.isfinite(grid).all())
    ):
        raise ValueError("DepthSplat source sample_image_grid output is invalid")
    return grid


def _coordinates_from_source_grid(
    raw_descriptors: torch.Tensor,
    positions: list[tuple[int, int]],
    source_grid: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Replay the loaded source's pixel-center plus sigmoid-offset expression."""

    if (
        raw_descriptors.ndim != 2
        or raw_descriptors.shape[0] != len(positions)
        or raw_descriptors.shape[1] < 3
        or source_grid.shape != (height, width, 2)
    ):
        raise ValueError("DepthSplat source-grid coordinate inputs are inconsistent")
    rows = torch.tensor([row for row, _ in positions], device=raw_descriptors.device)
    columns = torch.tensor([column for _, column in positions], device=raw_descriptors.device)
    if bool((rows < 0).any()) or bool((rows >= height).any()) or bool((columns < 0).any()) or bool((columns >= width).any()):
        raise ValueError("DepthSplat source-grid coordinate position is invalid")
    base = source_grid[rows, columns].to(dtype=raw_descriptors.dtype)
    pixel_size = torch.tensor((1.0 / width, 1.0 / height), device=base.device, dtype=base.dtype)
    return base + (raw_descriptors[:, 1:3].sigmoid() - 0.5) * pixel_size


def _spatial_weights(
    target_positions: list[tuple[int, int]],
    source_positions: list[tuple[int, int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not target_positions or not source_positions or len(set(source_positions)) != len(source_positions):
        raise ValueError("DepthSplat materializer spatial field is invalid")
    rows = [row for row, _ in source_positions]
    columns = [column for _, column in source_positions]
    top, bottom = min(rows), max(rows)
    left, right = min(columns), max(columns)
    corners = {(top, left), (top, right), (bottom, left), (bottom, right)}
    weights = torch.zeros((len(target_positions), len(source_positions)), device=device, dtype=dtype)
    if len(source_positions) == 4 and set(source_positions) == corners and top < bottom and left < right:
        source_to_index = {position: index for index, position in enumerate(source_positions)}
        for target_index, (row, column) in enumerate(target_positions):
            u = torch.as_tensor((column - left) / (right - left), device=device, dtype=dtype)
            v = torch.as_tensor((row - top) / (bottom - top), device=device, dtype=dtype)
            weights[target_index, source_to_index[(top, left)]] = (1.0 - u) * (1.0 - v)
            weights[target_index, source_to_index[(top, right)]] = u * (1.0 - v)
            weights[target_index, source_to_index[(bottom, left)]] = (1.0 - u) * v
            weights[target_index, source_to_index[(bottom, right)]] = u * v
    else:
        for target_index, (row, column) in enumerate(target_positions):
            squared = torch.tensor(
                [(row - source_row) ** 2 + (column - source_column) ** 2 for source_row, source_column in source_positions],
                device=device,
                dtype=dtype,
            )
            if bool((squared <= 0.0).any()):
                raise ValueError("DepthSplat materializer virtual target overlaps an anchor")
            inverse = squared.reciprocal()
            weights[target_index] = inverse / inverse.sum()
    if not bool(torch.isfinite(weights).all()) or not torch.allclose(
        weights.sum(dim=1), torch.ones(weights.shape[0], device=device, dtype=dtype), rtol=1e-5, atol=1e-5
    ):
        raise ValueError("DepthSplat materializer spatial field is not a partition of unity")
    return weights


def _selected_anchor_support_scale(values: torch.Tensor) -> torch.Tensor:
    """Use only retained-anchor variation to normalize a LOO residual."""

    if values.ndim < 2 or values.shape[0] < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("DepthSplat selected-anchor support is invalid")
    flattened = values.reshape(values.shape[0], -1)
    distances = torch.pdist(flattened)
    if distances.numel() == 0 or not bool(torch.isfinite(distances).all()):
        raise ValueError("DepthSplat selected-anchor support distances are invalid")
    magnitude = flattened.abs().amax().clamp_min(1.0)
    numerical_floor = torch.finfo(flattened.dtype).eps * 1024.0 * magnitude
    return torch.maximum(distances.median(), numerical_floor)


def _selected_anchor_opacity_logits(opacities: torch.Tensor) -> torch.Tensor:
    if (
        not bool(torch.isfinite(opacities).all())
        or bool((opacities < 0.0).any())
        or bool((opacities >= 1.0).any())
    ):
        raise ValueError("DepthSplat selected-anchor opacity is invalid")
    epsilon = torch.finfo(opacities.dtype).eps * 16.0
    bounded = opacities.clamp(min=epsilon, max=1.0 - epsilon)
    return torch.log(bounded) - torch.log1p(-bounded)


def _selected_anchor_attribute_loo_frozen_guard(
    value: Any | None,
    *,
    require_literal_t4_authenticated: bool = False,
    allow_serialized_literal_t4_projection: bool = False,
) -> dict[str, Any] | None:
    """Validate a guard without admitting a forged literal T=4 projection.

    Literal T=4 preflight accepts only the opaque capability issued by the
    calibration module after it has live-reloaded the frozen record.  A plain
    mapping is still accepted for the legacy development profile, and for the
    already-validated serialized evidence that a later final-route check
    replays; neither path can set a new literal runtime threshold.
    """

    if value is None:
        return None
    if require_literal_t4_authenticated and allow_serialized_literal_t4_projection:
        raise ValueError("DepthSplat LOO guard validation mode is inconsistent")
    if require_literal_t4_authenticated:
        from saes.depthsplat_literal_t4_acid_calibration import (
            verified_literal_t4_materializer_guard_projection,
        )

        value = verified_literal_t4_materializer_guard_projection(value)
    required = {
        "schema_version",
        "frozen_record_kind",
        "frozen_record_sha256",
        "threshold_value",
        "threshold_rule",
        "risk_metric",
        "materialization_profile",
        "route_plan_contract",
        "route_plan_config_sha256",
        "acid_binding_sha256",
        "application_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("DepthSplat selected-anchor LOO frozen guard is invalid")
    if (
        value.get("schema_version") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_GUARD_SCHEMA
        or value.get("risk_metric") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC
        or not isinstance(value.get("frozen_record_kind"), str)
        or not value["frozen_record_kind"]
        or not isinstance(value.get("materialization_profile"), str)
        or value["materialization_profile"]
        not in {
            DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
        }
        or not isinstance(value.get("route_plan_contract"), str)
        or not value["route_plan_contract"]
        or not isinstance(value.get("threshold_rule"), str)
        or not value["threshold_rule"]
        or isinstance(value.get("threshold_value"), bool)
        or not isinstance(value.get("threshold_value"), (int, float))
        or not torch.isfinite(torch.tensor(float(value["threshold_value"])))
        or float(value["threshold_value"]) < 0.0
    ):
        raise ValueError("DepthSplat selected-anchor LOO frozen guard changed")
    if (
        value["materialization_profile"]
        == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        and not require_literal_t4_authenticated
        and not allow_serialized_literal_t4_projection
    ):
        raise ValueError(
            "DepthSplat literal T=4 LOO requires an authenticated V16T4 guard"
        )
    for name in (
        "frozen_record_sha256",
        "route_plan_config_sha256",
        "acid_binding_sha256",
        "application_sha256",
    ):
        _require_sha256(value.get(name), label=f"selected-anchor LOO guard {name}")
    return {
        "schema_version": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_GUARD_SCHEMA,
        "frozen_record_kind": value["frozen_record_kind"],
        "frozen_record_sha256": value["frozen_record_sha256"],
        "threshold_value": float(value["threshold_value"]),
        "threshold_rule": value["threshold_rule"],
        "risk_metric": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
        "materialization_profile": value["materialization_profile"],
        "route_plan_contract": value["route_plan_contract"],
        "route_plan_config_sha256": value["route_plan_config_sha256"],
        "acid_binding_sha256": value["acid_binding_sha256"],
        "application_sha256": value["application_sha256"],
    }


def _selected_anchor_opacity_endpoint_count(
    *,
    packed: DepthSplatPackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    level: str,
    semantics: str,
    plan_record: Mapping[str, Any],
) -> int:
    """Read only selected anchor opacities to preserve a native Full fallback."""

    anchors = _route_anchor_positions(plan_record, level=level, semantics=semantics)
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=4,
        positions=anchors,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("DepthSplat selected-anchor LOO lacks a retained anchor")
    opacities = packed.opacities[[slot_to_index[slot] for slot in anchor_slots]]
    if (
        not bool(torch.isfinite(opacities).all())
        or bool((opacities < 0.0).any())
        or bool((opacities > 1.0).any())
    ):
        raise ValueError("DepthSplat selected-anchor opacity is invalid")
    return int((opacities >= 1.0).sum().item())


def depthsplat_selected_anchor_attribute_loo_certificate(
    *,
    packed: DepthSplatPackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    level: str,
    semantics: str,
    plan_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure selected-anchor RGB-SH/opacity replay risk without omitted labels.

    Each retained L0/L1 anchor is held out in turn and reconstructed from the
    remaining retained anchors using the same spatial field used to construct
    virtual attributes.  The label is therefore a native selected Adapter
    attribute, while a real omitted position is never opened or inspected.
    """

    if level not in {"L0", "L1"}:
        raise ValueError("DepthSplat selected-anchor LOO requires a compact level")
    anchors = _route_anchor_positions(plan_record, level=level, semantics=semantics)
    if len(anchors) < 4 or len(set(anchors)) != len(anchors):
        raise ValueError("DepthSplat selected-anchor LOO layout is invalid")
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=4,
        positions=anchors,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("DepthSplat selected-anchor LOO lacks a retained anchor")
    anchor_indices = [slot_to_index[slot] for slot in anchor_slots]
    harmonics = packed.harmonics[anchor_indices]
    opacities = packed.opacities[anchor_indices]
    if not bool(torch.isfinite(harmonics).all()):
        raise ValueError("DepthSplat selected-anchor harmonics are non-finite")
    logits = _selected_anchor_opacity_logits(opacities)
    global_positions = [
        (tile_y * 4 + row, tile_x * 4 + column) for row, column in anchors
    ]
    risks: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for held_offset, held_out in enumerate(anchors):
        source_offsets = [
            offset for offset in range(len(anchors)) if offset != held_offset
        ]
        source_positions = [global_positions[offset] for offset in source_offsets]
        target_position = [global_positions[held_offset]]
        spatial = _spatial_weights(
            target_position,
            source_positions,
            device=harmonics.device,
            dtype=harmonics.dtype,
        ).reshape(-1)
        source_harmonics = harmonics[source_offsets]
        source_logits = logits[source_offsets]
        predicted_harmonics = torch.einsum("n,ncd->cd", spatial, source_harmonics)
        predicted_logit = torch.dot(spatial, source_logits)
        harmonic_error = (
            (predicted_harmonics - harmonics[held_offset]).norm()
            / _selected_anchor_support_scale(source_harmonics)
        )
        opacity_error = (
            (predicted_logit - logits[held_offset]).abs()
            / _selected_anchor_support_scale(source_logits.reshape(-1, 1))
        )
        risk = torch.maximum(harmonic_error, opacity_error)
        if not bool(torch.isfinite(risk)) or bool(risk < 0.0):
            raise ValueError("DepthSplat selected-anchor LOO risk is invalid")
        risks.append(risk)
        records.append(
            {
                "held_out_local_position": [int(held_out[0]), int(held_out[1])],
                "harmonic_relative_error": float(harmonic_error.item()),
                "opacity_logit_relative_error": float(opacity_error.item()),
                "risk": float(risk.item()),
            }
        )
    values = torch.stack(risks).to(dtype=torch.float32)
    q75_risk = torch.quantile(values, 0.75)
    maximum_held_out_risk = values.max()
    if (
        not bool(torch.isfinite(q75_risk))
        or bool(q75_risk < 0.0)
        or not bool(torch.isfinite(maximum_held_out_risk))
        or bool(maximum_held_out_risk < 0.0)
    ):
        raise ValueError("DepthSplat selected-anchor LOO quantile is invalid")
    return {
        "checked": True,
        "scorable": True,
        "status": "scored",
        "reason": None,
        "certificate": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE,
        "policy": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY,
        "risk_metric": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
        "level": level,
        "anchor_count": len(anchors),
        "held_out_anchor_count": len(records),
        "held_out_anchor_records": records,
        "q75_risk": float(q75_risk.item()),
        "maximum_held_out_risk": float(maximum_held_out_risk.item()),
        "selected_anchor_native_attribute_label_reads": len(records),
        "selected_anchor_native_attribute_endpoint_reads": 0,
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_direct_deletion": False,
    }


def _selected_anchor_attribute_loo_unscorable_certificate(
    *,
    level: str,
    semantics: str,
    plan_record: Mapping[str, Any],
    endpoint_anchor_count: int,
) -> dict[str, Any]:
    """Record a native endpoint Full promotion without treating it as a failure.

    Exact opacity-one anchors cannot be replayed in logit space.  They are not
    omitted samples, so the correct source-faithful response is to retain the
    Full tile and record that this LOO observation was deliberately unscorable.
    """

    anchors = _route_anchor_positions(plan_record, level=level, semantics=semantics)
    if endpoint_anchor_count < 1 or endpoint_anchor_count > len(anchors):
        raise ValueError("DepthSplat selected-anchor endpoint evidence is invalid")
    return {
        "checked": False,
        "scorable": False,
        "status": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_ENDPOINT_UNSCORABLE,
        "reason": NATIVE_OPACITY_ENDPOINT_FULL_REASON,
        "certificate": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE,
        "policy": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY,
        "risk_metric": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
        "level": level,
        "anchor_count": len(anchors),
        "held_out_anchor_count": 0,
        "held_out_anchor_records": [],
        "q75_risk": None,
        "maximum_held_out_risk": None,
        "selected_anchor_native_attribute_label_reads": 0,
        "selected_anchor_native_attribute_endpoint_reads": endpoint_anchor_count,
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_direct_deletion": False,
    }


def _finite_selected_anchor_attribute_loo_summary(
    values: list[float],
) -> dict[str, float | int | None]:
    """Use a JSON-safe deterministic quantile summary for audit evidence."""

    if any(not isinstance(value, float) or not torch.isfinite(torch.tensor(value)) for value in values):
        raise ValueError("DepthSplat selected-anchor LOO aggregate contains an invalid risk")
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "maximum": None,
        }

    def quantile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(ordered),
        "minimum": quantile(0.0),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "maximum": quantile(1.0),
    }


def _selected_anchor_attribute_loo_aggregate(
    tile_trace: list[dict[str, Any]],
    *,
    mode: str,
    frozen_guard: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the serializable selected-only LOO aggregate from the tile trace."""

    if mode not in {"collect_only", "frozen_v16_guard"}:
        raise ValueError("DepthSplat selected-anchor LOO aggregate mode is invalid")
    if (mode == "frozen_v16_guard") != (frozen_guard is not None):
        raise ValueError("DepthSplat selected-anchor LOO aggregate guard mode drifted")
    records: list[dict[str, Any]] = []
    attempted_tiles = 0
    for entry in tile_trace:
        if not isinstance(entry, Mapping):
            raise ValueError("DepthSplat selected-anchor LOO tile trace is invalid")
        if entry.get("attempted") is not True:
            continue
        attempted_tiles += 1
        loo = entry.get("selected_anchor_attribute_loo")
        if not isinstance(loo, Mapping):
            raise ValueError("DepthSplat selected-anchor LOO collection is incomplete")
        record = {
            "view": entry.get("view"),
            "tile_y": entry.get("tile_y"),
            "tile_x": entry.get("tile_x"),
            "planned_route": entry.get("planned_route"),
            "accepted": entry.get("accepted"),
            "reason": entry.get("reason"),
            "selected_anchor_attribute_loo": dict(loo),
        }
        if (
            any(
                isinstance(record[name], bool) or not isinstance(record[name], int)
                for name in ("view", "tile_y", "tile_x")
            )
            or record["planned_route"] not in {"L0", "L1"}
            or not isinstance(record["accepted"], bool)
            or not isinstance(record["reason"], str)
        ):
            raise ValueError("DepthSplat selected-anchor LOO tile identity is invalid")
        records.append(record)
    if attempted_tiles != len(records):
        raise ValueError("DepthSplat selected-anchor LOO tile count drifted")

    scorable = [
        record["selected_anchor_attribute_loo"]
        for record in records
        if record["selected_anchor_attribute_loo"].get("scorable") is True
    ]
    unscorable = [
        record["selected_anchor_attribute_loo"]
        for record in records
        if record["selected_anchor_attribute_loo"].get("scorable") is False
    ]
    if len(scorable) + len(unscorable) != len(records):
        raise ValueError("DepthSplat selected-anchor LOO scoring state is invalid")
    maximum_risks: list[float] = []
    q75_risks: list[float] = []
    label_reads = 0
    endpoint_reads = 0
    for record in scorable:
        if (
            record.get("checked") is not True
            or record.get("status") != "scored"
            or record.get("certificate") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE
            or record.get("policy") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY
            or record.get("risk_metric") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC
            or record.get("source_nonprobe_s3_attribute_reads") != 0
            or record.get("nonzero_direct_deletion") is not False
            or not isinstance(record.get("held_out_anchor_records"), list)
            or record.get("held_out_anchor_count")
            != len(record["held_out_anchor_records"])
            or record.get("selected_anchor_native_attribute_label_reads")
            != record.get("held_out_anchor_count")
            or record.get("selected_anchor_native_attribute_endpoint_reads") != 0
        ):
            raise ValueError("DepthSplat selected-anchor LOO scored record is invalid")
        maximum = record.get("maximum_held_out_risk")
        q75 = record.get("q75_risk")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not torch.isfinite(torch.tensor(float(maximum)))
            or float(maximum) < 0.0
            or isinstance(q75, bool)
            or not isinstance(q75, (int, float))
            or not torch.isfinite(torch.tensor(float(q75)))
            or float(q75) < 0.0
            or float(q75) > float(maximum)
        ):
            raise ValueError("DepthSplat selected-anchor LOO risk is invalid")
        maximum_risks.append(float(maximum))
        q75_risks.append(float(q75))
        label_reads += int(record["selected_anchor_native_attribute_label_reads"])
    for record in unscorable:
        if (
            record.get("checked") is not False
            or record.get("status") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_ENDPOINT_UNSCORABLE
            or record.get("reason") != NATIVE_OPACITY_ENDPOINT_FULL_REASON
            or record.get("certificate") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE
            or record.get("policy") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY
            or record.get("risk_metric") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC
            or record.get("held_out_anchor_count") != 0
            or record.get("held_out_anchor_records") != []
            or record.get("q75_risk") is not None
            or record.get("maximum_held_out_risk") is not None
            or record.get("selected_anchor_native_attribute_label_reads") != 0
            or isinstance(record.get("selected_anchor_native_attribute_endpoint_reads"), bool)
            or not isinstance(record.get("selected_anchor_native_attribute_endpoint_reads"), int)
            or record["selected_anchor_native_attribute_endpoint_reads"] < 1
            or record.get("source_nonprobe_s3_attribute_reads") != 0
            or record.get("nonzero_direct_deletion") is not False
        ):
            raise ValueError("DepthSplat selected-anchor LOO endpoint record is invalid")
        endpoint_reads += int(record["selected_anchor_native_attribute_endpoint_reads"])
    if any(record["selected_anchor_attribute_loo"].get("source_nonprobe_s3_attribute_reads") != 0 for record in records):
        raise ValueError("DepthSplat selected-anchor LOO opened a nonprobe attribute")
    if any(
        record["selected_anchor_attribute_loo"].get("action") not in {
            "observed_only",
            "retain_compact",
            "promote_full",
            "promote_full_unscorable",
        }
        for record in records
    ):
        raise ValueError("DepthSplat selected-anchor LOO action is invalid")
    if mode == "collect_only" and any(
        record["selected_anchor_attribute_loo"].get("action")
        not in {"observed_only", "promote_full_unscorable"}
        for record in records
    ):
        raise ValueError("DepthSplat selected-anchor LOO collection action drifted")
    if mode == "frozen_v16_guard" and any(
        record["selected_anchor_attribute_loo"].get("action") == "observed_only"
        for record in records
    ):
        raise ValueError("DepthSplat selected-anchor LOO frozen guard was bypassed")

    return {
        "schema_version": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_AGGREGATE_SCHEMA,
        "certificate": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE,
        "policy": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY,
        "risk_metric": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
        "mode": mode,
        "frozen_guard": dict(frozen_guard) if frozen_guard is not None else None,
        "preflight_tile_trace_sha256": _canonical_sha256(tile_trace),
        "candidate_tile_count": attempted_tiles,
        "recorded_tile_count": len(records),
        "tile_records_sha256": _canonical_sha256(records),
        "scorable_tile_count": len(scorable),
        "unscorable_promoted_full_tile_count": len(unscorable),
        "accepted_compact_tile_count": sum(record["accepted"] for record in records),
        "guard_promoted_full_tile_count": sum(
            record["selected_anchor_attribute_loo"].get("action") == "promote_full"
            for record in records
        ),
        "maximum_held_out_risks": maximum_risks,
        "maximum_held_out_risk_summary": _finite_selected_anchor_attribute_loo_summary(
            maximum_risks
        ),
        "q75_risks": q75_risks,
        "q75_risk_summary": _finite_selected_anchor_attribute_loo_summary(q75_risks),
        "selected_anchor_native_attribute_label_reads": label_reads,
        "selected_anchor_native_attribute_endpoint_reads": endpoint_reads,
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_direct_deletion": False,
    }


def _validate_selected_anchor_attribute_loo_aggregate(
    *,
    events: Mapping[str, Any],
    tile_trace: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    """Rebuild the aggregate so a route cannot consume a tampered LOO trace."""

    execution_profile = events.get("execution_profile")
    frozen_guard = _selected_anchor_attribute_loo_frozen_guard(
        events.get("selected_anchor_attribute_loo_frozen_guard"),
        allow_serialized_literal_t4_projection=(
            execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        ),
    )
    collect_only = events.get("selected_anchor_attribute_loo_collect_only")
    guard_enabled = events.get("selected_anchor_attribute_loo_guard")
    aggregate = events.get("selected_anchor_attribute_loo_aggregate")
    if not isinstance(collect_only, bool) or not isinstance(guard_enabled, bool):
        raise ValueError("DepthSplat selected-anchor LOO event mode is invalid")
    if guard_enabled != (frozen_guard is not None) or (collect_only and guard_enabled):
        raise ValueError("DepthSplat selected-anchor LOO event guard drifted")
    if (
        frozen_guard is not None
        and frozen_guard["materialization_profile"] != execution_profile
    ):
        raise ValueError("DepthSplat selected-anchor LOO event profile drifted")
    if not collect_only and not guard_enabled:
        if aggregate is not None:
            raise ValueError("DepthSplat selected-anchor LOO aggregate was not requested")
        return None
    expected = _selected_anchor_attribute_loo_aggregate(
        [dict(record) for record in tile_trace],
        mode="frozen_v16_guard" if guard_enabled else "collect_only",
        frozen_guard=frozen_guard,
    )
    if (
        not isinstance(aggregate, Mapping)
        or dict(aggregate) != expected
        or events.get("selected_anchor_attribute_loo_aggregate_sha256")
        != _canonical_sha256(expected)
    ):
        raise ValueError("DepthSplat selected-anchor LOO aggregate changed")
    return expected


def _feature_continuity(
    feature_map: torch.Tensor,
    source_positions: list[tuple[int, int]],
    target_positions: list[tuple[int, int]],
    spatial_weights: torch.Tensor,
    *,
    maximum_relative_residual: float,
) -> dict[str, float | bool]:
    source = torch.stack([feature_map[:, row, column] for row, column in source_positions])
    target = torch.stack([feature_map[:, row, column] for row, column in target_positions])
    predicted = spatial_weights @ source
    residual = (predicted - target).square().mean(dim=1)
    diameter = (source.unsqueeze(0) - source.unsqueeze(1)).square().mean(dim=2).amax()
    denominator = diameter.clamp_min(torch.finfo(source.dtype).eps)
    relative = residual / denominator
    maximum = float(relative.max().item())
    return {
        "maximum_relative_residual": maximum,
        "source_feature_diameter": float(diameter.item()),
        "passed": bool(torch.isfinite(relative).all()) and maximum <= maximum_relative_residual,
    }


def _project_psd(covariance: torch.Tensor) -> torch.Tensor:
    symmetric = (covariance + covariance.mT) * 0.5
    values, vectors = torch.linalg.eigh(symmetric)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("DepthSplat materializer covariance spectrum is non-finite")
    projected = vectors @ torch.diag(values.clamp_min(1e-8)) @ vectors.mT
    return (projected + projected.mT) * 0.5


def depthsplat_z_depth_world_means(
    coordinates: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    z_depths: torch.Tensor,
    *,
    source_get_world_rays: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Lift image coordinates with the native source z-depth convention."""

    if not callable(source_get_world_rays):
        raise TypeError("DepthSplat materializer requires source get_world_rays")
    if (
        not torch.is_tensor(coordinates)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or extrinsics.shape != (coordinates.shape[0], 4, 4)
        or intrinsics.shape != (coordinates.shape[0], 3, 3)
        or z_depths.reshape(-1).shape[0] != coordinates.shape[0]
    ):
        raise ValueError("DepthSplat z-depth lift inputs are inconsistent")
    z_depths = z_depths.reshape(-1).to(device=coordinates.device, dtype=coordinates.dtype)
    if (
        not bool(torch.isfinite(coordinates).all())
        or not bool(torch.isfinite(extrinsics).all())
        or not bool(torch.isfinite(intrinsics).all())
        or not bool(torch.isfinite(z_depths).all())
        or bool((z_depths <= 0.0).any())
    ):
        raise ValueError("DepthSplat z-depth lift inputs are non-finite")
    origins, directions = source_get_world_rays(coordinates, extrinsics, intrinsics)
    if (
        not torch.is_tensor(origins)
        or not torch.is_tensor(directions)
        or origins.shape != (coordinates.shape[0], 3)
        or directions.shape != origins.shape
        or not bool(torch.isfinite(origins).all())
        or not bool(torch.isfinite(directions).all())
    ):
        raise ValueError("DepthSplat source get_world_rays output is invalid")
    means = origins + directions * z_depths.unsqueeze(1)
    if not bool(torch.isfinite(means).all()):
        raise ValueError("DepthSplat z-depth lift produced non-finite means")
    return means


def _anchor_geometry(
    *,
    packet: DepthSplatSparseRawPacket,
    packed: DepthSplatPackedGaussianAttributes,
    indices: list[int],
    source_positions: list[tuple[int, int]],
    view: int,
    height: int,
    width: int,
    source_grid: torch.Tensor,
    source_get_world_rays: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    expected_coordinates = _coordinates_from_source_grid(
        packet.raw_head_descriptors[indices],
        source_positions,
        source_grid,
        height=height,
        width=width,
    )
    if not torch.allclose(packet.coordinates[indices], expected_coordinates, rtol=2e-5, atol=2e-5):
        raise ValueError("DepthSplat materializer selected offsets drift from native geometry")
    expected_means = depthsplat_z_depth_world_means(
        packet.coordinates[indices],
        packet.extrinsics[indices],
        packet.intrinsics[indices],
        packet.depths[indices],
        source_get_world_rays=source_get_world_rays,
    )
    if not torch.allclose(packed.means[indices], expected_means, rtol=5e-5, atol=5e-5):
        raise ValueError("DepthSplat materializer selected means drift from native z-depth geometry")
    centres = _pixel_centres(
        source_positions,
        height=height,
        width=width,
        device=packet.coordinates.device,
        dtype=packet.coordinates.dtype,
    )
    offsets = packet.coordinates[indices] - centres
    limits = torch.tensor((0.5 / width, 0.5 / height), device=offsets.device, dtype=offsets.dtype)
    if not bool(torch.isfinite(offsets).all()) or bool((offsets.abs() > limits + 2e-5).any()):
        raise ValueError("DepthSplat materializer selected native offset is out of range")
    return offsets


def _coverage_closed_covariance(
    covariance: torch.Tensor,
    mean: torch.Tensor,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    *,
    maximum_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Close a merged 3D covariance over every virtual 2-sigma ellipsoid.

    Point-centre containment is insufficient: a virtual Gaussian's own
    covariance can extend outside the merged support even when its centre is
    close.  In the merged covariance metric, the sufficient containment bound
    is ``||delta|| + radius * sqrt(lambda_max(virtual_covariance)) <= radius``.
    """

    if (
        covariance.shape != (3, 3)
        or mean.shape != (3,)
        or virtual_means.ndim != 2
        or virtual_means.shape[1] != 3
        or virtual_covariances.shape != (virtual_means.shape[0], 3, 3)
    ):
        raise ValueError("DepthSplat materializer coverage support inputs are invalid")
    covariance = _project_psd(covariance)
    if virtual_means.numel() == 0:
        return covariance, {
            "maximum_containment_lhs_before_scale": 0.0,
            "maximum_containment_lhs_after_scale": 0.0,
            "maximum_whitened_virtual_covariance_eigenvalue": 0.0,
            "containment_radius": 2.0,
            "moment_covariance_scale": 1.0,
        }
    try:
        cholesky = torch.linalg.cholesky(covariance)
        deltas = virtual_means - mean.unsqueeze(0)
        whitened_deltas = torch.linalg.solve_triangular(
            cholesky, deltas.mT, upper=False
        ).mT
        containers = cholesky.unsqueeze(0).expand(virtual_means.shape[0], -1, -1)
        projected_virtual_covariances = torch.stack(
            [_project_psd(value) for value in virtual_covariances]
        )
        left = torch.linalg.solve_triangular(
            containers, projected_virtual_covariances, upper=False
        )
        whitened_covariances = torch.linalg.solve_triangular(
            containers, left.mT, upper=False
        ).mT
        eigenvalues = torch.linalg.eigvalsh(
            (whitened_covariances + whitened_covariances.mT) * 0.5
        )
    except RuntimeError as error:
        raise ValueError("DepthSplat materializer coverage containment failed") from error
    mahalanobis = whitened_deltas.square().sum(dim=1).clamp_min(0.0).sqrt()
    maximum_eigenvalues = eigenvalues.amax(dim=1).clamp_min(0.0)
    radius = 2.0
    lhs = mahalanobis + radius * maximum_eigenvalues.sqrt()
    if (
        not bool(torch.isfinite(lhs).all())
        or not bool(torch.isfinite(maximum_eigenvalues).all())
    ):
        raise ValueError("DepthSplat materializer coverage containment is non-finite")
    scale = max(1.0, float((lhs.max() / radius).square().item()))
    if scale > maximum_scale:
        raise ValueError("DepthSplat materializer coverage requires excessive covariance expansion")
    covariance = covariance * scale
    after_scale = lhs / scale**0.5
    if bool((after_scale > radius + 1e-5).any()):
        raise RuntimeError("DepthSplat materializer coverage closure did not contain support")
    return covariance, {
        "maximum_containment_lhs_before_scale": float(lhs.max().item()),
        "maximum_containment_lhs_after_scale": float(after_scale.max().item()),
        "maximum_whitened_virtual_covariance_eigenvalue": float(
            maximum_eigenvalues.max().item()
        ),
        "containment_radius": radius,
        "moment_covariance_scale": scale,
    }


def _route_anchor_positions(
    record: Mapping[str, Any], *, level: str, semantics: str
) -> list[tuple[int, int]]:
    if level == "L0":
        return compute_probe_positions(4)
    if level == "L1":
        return l1_local_positions_for_tile(record, tile_size=4, l1_anchor_semantics=semantics)
    if level == "Full":
        return _full_positions(4)
    raise ValueError("DepthSplat materializer route level is invalid")


def _update_binding(
    update_slots: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    harmonics: torch.Tensor,
    opacities: torch.Tensor,
) -> dict[str, str]:
    """Digest every materialized value before a later packet can consume it."""

    return {
        "slots_sha256": _tensor_sha256(update_slots),
        "means_sha256": _tensor_sha256(means),
        "covariances_sha256": _tensor_sha256(covariances),
        "harmonics_sha256": _tensor_sha256(harmonics),
        "opacities_sha256": _tensor_sha256(opacities),
    }


def _coverage_certificate_payload(
    *,
    update_slots: torch.Tensor,
    update_binding: Mapping[str, str],
    per_update: list[dict[str, Any]],
    tile_trace_sha256: str,
    maximum_covariance_scale: float,
) -> dict[str, Any]:
    """Bind shape-aware support checks to the exact ordered update packet."""

    if (
        not torch.is_tensor(update_slots)
        or update_slots.ndim != 1
        or update_slots.dtype != torch.int64
        or not isinstance(update_binding, Mapping)
        or not isinstance(tile_trace_sha256, str)
        or not isinstance(maximum_covariance_scale, float)
    ):
        raise ValueError("DepthSplat coverage certificate inputs are invalid")
    slots = update_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(per_update) != len(slots):
        raise ValueError("DepthSplat coverage certificate update count is incomplete")
    normalized_rows: list[dict[str, Any]] = []
    for update_index, (slot, row) in enumerate(zip(slots, per_update)):
        if not isinstance(row, Mapping):
            raise ValueError("DepthSplat coverage certificate row is invalid")
        normalized = dict(row)
        if (
            normalized.get("update_dense_slot") != int(slot)
            or normalized.get("update_index") != update_index
            or normalized.get("containment_radius") != 2.0
            or normalized.get("shape_aware_virtual_2sigma_support") is not True
        ):
            raise ValueError("DepthSplat coverage certificate slot binding changed")
        for key in (
            "maximum_containment_lhs_before_scale",
            "maximum_containment_lhs_after_scale",
            "maximum_whitened_virtual_covariance_eigenvalue",
            "moment_covariance_scale",
        ):
            value = normalized.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not torch.isfinite(torch.tensor(float(value)))
                or float(value) < 0.0
            ):
                raise ValueError("DepthSplat coverage certificate value is invalid")
        if (
            float(normalized["maximum_containment_lhs_after_scale"]) > 2.0 + 1e-5
            or float(normalized["moment_covariance_scale"])
            > maximum_covariance_scale + 1e-5
        ):
            raise ValueError("DepthSplat coverage certificate does not contain support")
        normalized_rows.append(normalized)
    return {
        "schema": DEPTHSPLAT_COVERAGE_CERTIFICATE,
        "geometry": "world-3d-whitened-ellipsoid-triangle-bound-v1",
        "containment_radius": 2.0,
        "after_scale_lhs_upper_bound": 2.0,
        "maximum_covariance_scale": maximum_covariance_scale,
        "shape_aware_virtual_2sigma_support": True,
        "tile_trace_sha256": tile_trace_sha256,
        "update_binding": dict(update_binding),
        "update_slots": [int(slot) for slot in slots],
        "per_update": normalized_rows,
    }


def _validate_coverage_certificate(
    preflight: DepthSplatCompactMaterializationPreflight,
) -> dict[str, Any]:
    """Reject a route if its shape-aware support certificate is incomplete."""

    events = preflight.events
    update_binding = _update_binding(
        preflight.update_dense_slots,
        preflight.means,
        preflight.covariances,
        preflight.harmonics,
        preflight.opacities,
    )
    payload = events.get("coverage_certificate_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("DepthSplat coverage certificate payload is missing")
    maximum_scale = events.get("maximum_coverage_covariance_scale")
    if (
        isinstance(maximum_scale, bool)
        or not isinstance(maximum_scale, (int, float))
        or float(maximum_scale) < 1.0
    ):
        raise ValueError("DepthSplat coverage certificate scale limit is invalid")
    expected = _coverage_certificate_payload(
        update_slots=preflight.update_dense_slots,
        update_binding=update_binding,
        per_update=list(payload.get("per_update", [])),
        tile_trace_sha256=str(events.get("tile_trace_sha256")),
        maximum_covariance_scale=float(maximum_scale),
    )
    if (
        events.get("coverage_certificate") != DEPTHSPLAT_COVERAGE_CERTIFICATE
        or dict(payload) != expected
        or events.get("coverage_certificate_sha256") != _canonical_sha256(expected)
    ):
        raise ValueError("DepthSplat coverage certificate binding changed")
    return expected


def _build_tile_updates(
    *,
    packet: DepthSplatSparseRawPacket,
    packed: DepthSplatPackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    feature_map: torch.Tensor,
    z_depth_map: torch.Tensor,
    record: Mapping[str, Any],
    level: str,
    semantics: str,
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    feature_statistic: str,
    source_grid: torch.Tensor,
    source_get_world_rays: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    maximum_feature_relative_residual: float,
    maximum_coverage_covariance_scale: float,
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, Any]]:
    anchors = _route_anchor_positions(record, level=level, semantics=semantics)
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=4,
        positions=anchors,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("DepthSplat materializer tile lacks a selected anchor")
    anchor_indices = [slot_to_index[slot] for slot in anchor_slots]
    if bool((packed.opacities[anchor_indices] >= 1.0).any()):
        raise ValueError(NATIVE_OPACITY_ENDPOINT_FULL_REASON)
    source_positions = [(tile_y * 4 + row, tile_x * 4 + column) for row, column in anchors]
    all_positions = _full_positions(4)
    target_local = [position for position in all_positions if position not in set(anchors)]
    if not target_local:
        return [], {
            "virtual_count": 0,
            "coverage": {
                "maximum_containment_lhs_before_scale": 0.0,
                "maximum_containment_lhs_after_scale": 0.0,
                "moment_covariance_scale_max": 1.0,
            },
        }
    target_positions = [(tile_y * 4 + row, tile_x * 4 + column) for row, column in target_local]
    target_z_depths = torch.stack(
        [z_depth_map[row, column] for row, column in target_positions]
    ).to(device=packet.depths.device, dtype=packet.depths.dtype)
    if not bool(torch.isfinite(target_z_depths).all()) or bool((target_z_depths <= 0.0).any()):
        raise ValueError("DepthSplat materializer virtual source z-depth is invalid")
    offsets = _anchor_geometry(
        packet=packet,
        packed=packed,
        indices=anchor_indices,
        source_positions=source_positions,
        view=view,
        height=height,
        width=width,
        source_grid=source_grid,
        source_get_world_rays=source_get_world_rays,
    )
    spatial = _spatial_weights(
        target_positions,
        source_positions,
        device=packed.means.device,
        dtype=packed.means.dtype,
    )
    continuity = _feature_continuity(
        feature_map,
        source_positions,
        target_positions,
        spatial,
        maximum_relative_residual=maximum_feature_relative_residual,
    )
    if continuity["passed"] is not True:
        raise ValueError("DepthSplat materializer feature continuity rejected tile")
    source_harmonics = packed.harmonics[anchor_indices]
    source_opacities = packed.opacities[anchor_indices]
    virtual_harmonics = (spatial @ source_harmonics.reshape(len(anchors), -1)).reshape(
        len(target_local), *source_harmonics.shape[1:]
    )
    logits = torch.logit(source_opacities.clamp(1e-6, 1.0 - 1e-6))
    virtual_opacities = (spatial @ logits).sigmoid()
    virtual_rgb = spatial @ packet.source_rgb[anchor_indices]
    if (
        not bool(torch.isfinite(virtual_harmonics).all())
        or not bool(torch.isfinite(virtual_opacities).all())
        or not bool(torch.isfinite(virtual_rgb).all())
        or bool((virtual_opacities < 0.0).any())
        or bool((virtual_opacities >= 1.0).any())
    ):
        raise ValueError("DepthSplat materializer RGB-SH virtual field is invalid")
    anchor_features = torch.stack([feature_map[:, row, column] for row, column in source_positions])
    coordinate_scale = 3.0
    feature_variance = _tile_feature_variance(record, feature_statistic)
    assignments: list[torch.Tensor] = []
    for local_row, local_column in target_local:
        target_feature = feature_map[:, tile_y * 4 + local_row, tile_x * 4 + local_column]
        spatial_distances = torch.tensor(
            [((local_row - row) / coordinate_scale) ** 2 + ((local_column - column) / coordinate_scale) ** 2 for row, column in anchors],
            device=packed.means.device,
            dtype=packed.means.dtype,
        )
        feature_distances = (anchor_features - target_feature.unsqueeze(0)).square().sum(dim=1)
        kwargs: dict[str, Any] = {}
        if level == "L1":
            primary_count = len(compute_probe_positions(4))
            if anchors[:primary_count] != compute_probe_positions(4):
                raise ValueError("DepthSplat materializer L1 anchors lost their primary prefix")
            anchor_depths = packet.depths[anchor_indices]
            kwargs = {
                "probe_depths": anchor_depths,
                "depth_reference_depths": anchor_depths[:primary_count],
                "beta_d": 1.0,
            }
        weights = paper_assignment_weights(
            spatial_distances,
            feature_distances.to(dtype=packed.means.dtype),
            feature_variance=feature_variance,
            beta_x=0.5,
            beta_f=0.1,
            level=level,
            **kwargs,
        )
        if not bool(torch.isfinite(weights).all()) or not torch.allclose(
            weights.sum(), torch.ones((), device=weights.device, dtype=weights.dtype), rtol=1e-5, atol=1e-5
        ):
            raise ValueError("DepthSplat materializer assignment is invalid")
        assignments.append(weights)
    assignment = torch.stack(assignments)
    target_centres = _pixel_centres(
        target_positions,
        height=height,
        width=width,
        device=packet.coordinates.device,
        dtype=packet.coordinates.dtype,
    )
    updates: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    coverage_scales: list[float] = []
    coverage_lhs: list[float] = []
    coverage_updates: list[dict[str, Any]] = []
    depth_order = torch.argsort(target_z_depths, stable=True)
    for anchor_offset, (slot, index) in enumerate(zip(anchor_slots, anchor_indices)):
        conditional_coordinates = target_centres + offsets[anchor_offset].unsqueeze(0)
        if (
            not bool(torch.isfinite(conditional_coordinates).all())
            or bool((conditional_coordinates < 0.0).any())
            or bool((conditional_coordinates > 1.0).any())
        ):
            raise ValueError("DepthSplat materializer transported offset leaves the native image")
        count = conditional_coordinates.shape[0]
        virtual_means = depthsplat_z_depth_world_means(
            conditional_coordinates,
            packet.extrinsics[index].unsqueeze(0).expand(count, -1, -1),
            packet.intrinsics[index].unsqueeze(0).expand(count, -1, -1),
            target_z_depths,
            source_get_world_rays=source_get_world_rays,
        )
        contribution_alpha = (
            assignment[:, anchor_offset] * virtual_opacities
        ).clamp(min=0.0, max=1.0 - 1e-6)
        base_opacity = packed.opacities[index]
        # Composite native anchor and virtual supports in a declared source S2
        # z-depth order. The incremental union masses sum exactly to the final
        # alpha and provide an order-defined, reproducible moment/SH measure.
        contributor_depths = torch.cat(
            (packet.depths[index].reshape(1), target_z_depths[depth_order])
        )
        contributor_alphas = torch.cat(
            (base_opacity.reshape(1), contribution_alpha[depth_order])
        )
        contributors = torch.cat(
            (
                packed.means[index].unsqueeze(0),
                virtual_means[depth_order],
            ),
            dim=0,
        )
        contributor_harmonics = torch.cat(
            (
                packed.harmonics[index].unsqueeze(0),
                virtual_harmonics[depth_order],
            ),
            dim=0,
        )
        ordered = torch.argsort(contributor_depths, stable=True)
        contributors = contributors[ordered]
        contributor_harmonics = contributor_harmonics[ordered]
        contributor_alphas = contributor_alphas[ordered]
        remaining_transmittance = torch.ones((), device=base_opacity.device, dtype=base_opacity.dtype)
        incremental_masses: list[torch.Tensor] = []
        for contribution in contributor_alphas:
            incremental = remaining_transmittance * contribution
            incremental_masses.append(incremental)
            remaining_transmittance = remaining_transmittance * (1.0 - contribution)
        weights = torch.stack(incremental_masses)
        masses = weights
        mass_sum = masses.sum()
        if not bool(torch.isfinite(mass_sum)) or bool(mass_sum <= torch.finfo(masses.dtype).eps):
            raise ValueError("DepthSplat materializer moment mass is invalid")
        merged_mean = (masses.unsqueeze(1) * contributors).sum(dim=0) / mass_sum
        base_covariance = packed.covariances[index]
        contributor_covariances = base_covariance.unsqueeze(0).expand(contributors.shape[0], -1, -1)
        deltas = contributors - merged_mean.unsqueeze(0)
        merged_covariance = (
            masses.reshape(-1, 1, 1)
            * (contributor_covariances + deltas.unsqueeze(2) @ deltas.unsqueeze(1))
        ).sum(dim=0) / mass_sum
        merged_covariance, coverage = _coverage_closed_covariance(
            merged_covariance,
            merged_mean,
            virtual_means,
            base_covariance.unsqueeze(0).expand(count, -1, -1),
            maximum_scale=maximum_coverage_covariance_scale,
        )
        merged_opacity = 1.0 - remaining_transmittance
        if not bool(torch.isfinite(merged_opacity)) or bool(merged_opacity < 0.0) or bool(merged_opacity >= 1.0):
            raise ValueError("DepthSplat materializer merged opacity is invalid")
        merged_harmonics = torch.einsum("n,ncd->cd", weights, contributor_harmonics) / mass_sum
        if not bool(torch.isfinite(merged_harmonics).all()):
            raise ValueError("DepthSplat materializer merged RGB-SH is non-finite")
        updates.append((slot, merged_mean, merged_covariance, merged_harmonics, merged_opacity))
        coverage_scales.append(float(coverage["moment_covariance_scale"]))
        coverage_lhs.append(float(coverage["maximum_containment_lhs_after_scale"]))
        coverage_updates.append(
            {
                "update_dense_slot": int(slot),
                "virtual_count": count,
                "shape_aware_virtual_2sigma_support": True,
                **coverage,
            }
        )
    return updates, {
        "virtual_count": len(target_local),
        "continuity": continuity,
        "coverage": {
            "certificate": DEPTHSPLAT_COVERAGE_CERTIFICATE,
            "shape_aware_virtual_2sigma_support": True,
            "maximum_containment_lhs_after_scale": max(coverage_lhs, default=0.0),
            "moment_covariance_scale_max": max(coverage_scales, default=1.0),
            "per_update": coverage_updates,
        },
        "virtual_rgb_range": [
            float(virtual_rgb.amin().item()),
            float(virtual_rgb.amax().item()),
        ],
        "virtual_z_depth_min": float(target_z_depths.min().item()),
        "virtual_z_depth_max": float(target_z_depths.max().item()),
        "opacity_compositing_order": "source-s2-z-depth-near-to-far-stable-v1",
    }


def _paper_support_without_covariance_expansion(
    covariance: torch.Tensor,
    mean: torch.Tensor,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
) -> dict[str, float]:
    """Validate literal 2-sigma support and promote Full instead of expanding.

    The development profile closes support by scaling covariance.  That is a
    useful diagnostic, but is not part of the literal selected-probe moment
    construction.  Here a support miss is deliberately a local Full fallback.
    """

    if (
        covariance.shape != (3, 3)
        or mean.shape != (3,)
        or virtual_means.ndim != 2
        or virtual_means.shape[1] != 3
        or virtual_covariances.shape != (virtual_means.shape[0], 3, 3)
    ):
        raise ValueError("DepthSplat formal paper support inputs are invalid")
    covariance = (covariance + covariance.mT) * 0.5
    if not bool(torch.isfinite(covariance).all()) or bool(
        (torch.linalg.eigvalsh(covariance) < -1e-6).any()
    ):
        raise ValueError("DepthSplat formal paper moment covariance is not PSD")
    if virtual_means.numel() == 0:
        return {
            "maximum_containment_lhs_before_scale": 0.0,
            "maximum_containment_lhs_after_scale": 0.0,
            "maximum_whitened_virtual_covariance_eigenvalue": 0.0,
            "containment_radius": 2.0,
            "moment_covariance_scale": 1.0,
        }
    try:
        cholesky = torch.linalg.cholesky(covariance)
        deltas = virtual_means - mean.unsqueeze(0)
        whitened_deltas = torch.linalg.solve_triangular(
            cholesky, deltas.mT, upper=False
        ).mT
        projected = (virtual_covariances + virtual_covariances.mT) * 0.5
        eigenvalues = torch.linalg.eigvalsh(projected)
        if bool((eigenvalues < -1e-6).any()):
            raise ValueError("DepthSplat formal paper virtual covariance is not PSD")
        left = torch.linalg.solve_triangular(
            cholesky.unsqueeze(0).expand(projected.shape[0], -1, -1),
            projected,
            upper=False,
        )
        whitened_covariances = torch.linalg.solve_triangular(
            cholesky.unsqueeze(0).expand(projected.shape[0], -1, -1),
            left.mT,
            upper=False,
        ).mT
        whitened_eigenvalues = torch.linalg.eigvalsh(
            (whitened_covariances + whitened_covariances.mT) * 0.5
        )
    except RuntimeError as error:
        raise ValueError("DepthSplat formal paper support containment failed") from error
    mahalanobis = whitened_deltas.square().sum(dim=1).clamp_min(0.0).sqrt()
    maximum_eigenvalues = whitened_eigenvalues.amax(dim=1).clamp_min(0.0)
    lhs = mahalanobis + 2.0 * maximum_eigenvalues.sqrt()
    if not bool(torch.isfinite(lhs).all()) or not bool(
        torch.isfinite(maximum_eigenvalues).all()
    ):
        raise ValueError("DepthSplat formal paper support is non-finite")
    if bool((lhs > 2.0 + 1e-5).any()):
        raise ValueError("DepthSplat formal paper support requires Full")
    return {
        "maximum_containment_lhs_before_scale": float(lhs.max().item()),
        "maximum_containment_lhs_after_scale": float(lhs.max().item()),
        "maximum_whitened_virtual_covariance_eigenvalue": float(
            maximum_eigenvalues.max().item()
        ),
        "containment_radius": 2.0,
        "moment_covariance_scale": 1.0,
    }


def _build_literal_paper_t4_selected_only_tile_updates(
    *,
    packet: DepthSplatSparseRawPacket,
    packed: DepthSplatPackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    raw_feature_map: torch.Tensor,
    record: Mapping[str, Any],
    level: str,
    semantics: str,
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    feature_statistic: str,
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Literal Kp=4 selected-probe moment materialization for the formal path."""

    if (
        semantics != PAPER_KP_ANCHOR_SEMANTICS
        or level not in {"L0", "L1"}
        or feature_statistic != "raw-probe-mean-channel-variance"
    ):
        raise ValueError("DepthSplat formal paper tile does not have the literal route")
    anchors = _route_anchor_positions(record, level=level, semantics=semantics)
    if anchors != compute_probe_positions(4):
        raise ValueError("DepthSplat formal paper tile does not retain Kp=4 corners")
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=4,
        positions=anchors,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("DepthSplat formal paper tile lacks a selected probe")
    anchor_indices = [slot_to_index[slot] for slot in anchor_slots]
    source_means = packed.means[anchor_indices]
    source_covariances = packed.covariances[anchor_indices]
    source_harmonics = packed.harmonics[anchor_indices]
    source_opacities = packed.opacities[anchor_indices]
    if bool((source_opacities >= 1.0).any()):
        raise ValueError(NATIVE_OPACITY_ENDPOINT_FULL_REASON)
    source_positions = [(tile_y * 4 + row, tile_x * 4 + column) for row, column in anchors]
    target_local = [
        position for position in _full_positions(4) if position not in set(anchors)
    ]
    if not target_local:
        return [], {
            "virtual_count": 0,
            "coverage": {
                "certificate": DEPTHSPLAT_COVERAGE_CERTIFICATE,
                "shape_aware_virtual_2sigma_support": True,
                "maximum_containment_lhs_after_scale": 0.0,
                "moment_covariance_scale_max": 1.0,
                "per_update": [],
            },
            "virtual_geometry_source": "selected-probe-only-none-v1",
            "virtual_attribute_source": "selected-probe-only-none-v1",
            "omitted_routing_z_depth_reads": 0,
            "opacity_compositing_order": "literal-weighted-average-no-alpha-union-v1",
        }
    target_positions = [
        (tile_y * 4 + row, tile_x * 4 + column) for row, column in target_local
    ]
    spatial = _spatial_weights(
        target_positions,
        source_positions,
        device=packed.means.device,
        dtype=packed.means.dtype,
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
    virtual_harmonics = (
        spatial @ source_harmonics.reshape(len(anchors), -1)
    ).reshape(len(target_local), *source_harmonics.shape[1:])
    virtual_opacities = spatial @ source_opacities
    if (
        not bool(torch.isfinite(virtual_means).all())
        or not bool(torch.isfinite(virtual_covariances).all())
        or not bool(torch.isfinite(virtual_harmonics).all())
        or not bool(torch.isfinite(virtual_opacities).all())
        or bool((virtual_opacities < 0.0).any())
        or bool((virtual_opacities >= 1.0).any())
        or bool((torch.linalg.eigvalsh(virtual_covariances) < -1e-6).any())
    ):
        raise ValueError("DepthSplat formal paper selected-only virtual field is invalid")
    anchor_features = torch.stack(
        [raw_feature_map[:, row, column] for row, column in source_positions]
    )
    feature_variance = _tile_feature_variance(record, feature_statistic)
    assignments: list[torch.Tensor] = []
    for local_row, local_column in target_local:
        target_feature = raw_feature_map[
            :, tile_y * 4 + local_row, tile_x * 4 + local_column
        ]
        spatial_distances = torch.tensor(
            [
                ((local_row - row) / 3.0) ** 2
                + ((local_column - column) / 3.0) ** 2
                for row, column in anchors
            ],
            device=packed.means.device,
            dtype=packed.means.dtype,
        )
        feature_distances = (anchor_features - target_feature.unsqueeze(0)).square().sum(dim=1)
        kwargs: dict[str, Any] = {}
        if level == "L1":
            kwargs = {
                "probe_depths": packet.depths[anchor_indices],
                "depth_reference_depths": packet.depths[anchor_indices],
                "beta_d": 1.0,
            }
        weights = paper_assignment_weights(
            spatial_distances,
            feature_distances.to(dtype=packed.means.dtype),
            feature_variance=feature_variance,
            beta_x=0.5,
            beta_f=0.1,
            level=level,
            **kwargs,
        )
        if not bool(torch.isfinite(weights).all()) or not torch.allclose(
            weights.sum(),
            torch.ones((), device=weights.device, dtype=weights.dtype),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError("DepthSplat formal paper assignment is invalid")
        assignments.append(weights)
    assignment = torch.stack(assignments)
    updates: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    coverage_updates: list[dict[str, Any]] = []
    coverage_lhs: list[float] = []
    for anchor_offset, (slot, index) in enumerate(zip(anchor_slots, anchor_indices)):
        masses = torch.cat(
            (
                torch.ones(1, device=packed.means.device, dtype=packed.means.dtype),
                assignment[:, anchor_offset],
            )
        )
        mass_sum = masses.sum()
        if not bool(torch.isfinite(mass_sum)) or bool(
            mass_sum <= torch.finfo(masses.dtype).eps
        ):
            raise ValueError("DepthSplat formal paper moment mass is invalid")
        contributors = torch.cat((source_means[anchor_offset].unsqueeze(0), virtual_means), dim=0)
        contributor_covariances = torch.cat(
            (source_covariances[anchor_offset].unsqueeze(0), virtual_covariances), dim=0
        )
        contributor_harmonics = torch.cat(
            (source_harmonics[anchor_offset].unsqueeze(0), virtual_harmonics), dim=0
        )
        contributor_opacities = torch.cat(
            (source_opacities[anchor_offset].reshape(1), virtual_opacities), dim=0
        )
        merged_mean = (masses.unsqueeze(1) * contributors).sum(dim=0) / mass_sum
        deltas = contributors - merged_mean.unsqueeze(0)
        merged_covariance = (
            masses.reshape(-1, 1, 1)
            * (contributor_covariances + deltas.unsqueeze(2) @ deltas.unsqueeze(1))
        ).sum(dim=0) / mass_sum
        merged_covariance = (merged_covariance + merged_covariance.mT) * 0.5
        merged_harmonics = torch.einsum(
            "n,ncd->cd", masses, contributor_harmonics
        ) / mass_sum
        merged_opacity = torch.dot(masses, contributor_opacities) / mass_sum
        if (
            not bool(torch.isfinite(merged_mean).all())
            or not bool(torch.isfinite(merged_covariance).all())
            or not bool(torch.isfinite(merged_harmonics).all())
            or not bool(torch.isfinite(merged_opacity))
            or bool((torch.linalg.eigvalsh(merged_covariance) < -1e-6).any())
            or bool(merged_opacity < source_opacities.min() - 1e-6)
            or bool(merged_opacity > source_opacities.max() + 1e-6)
            or bool((merged_harmonics < source_harmonics.amin(dim=0) - 1e-5).any())
            or bool((merged_harmonics > source_harmonics.amax(dim=0) + 1e-5).any())
        ):
            raise ValueError("DepthSplat formal paper literal attribute average is invalid")
        coverage = _paper_support_without_covariance_expansion(
            merged_covariance,
            merged_mean,
            virtual_means,
            virtual_covariances,
        )
        updates.append((slot, merged_mean, merged_covariance, merged_harmonics, merged_opacity))
        coverage_lhs.append(float(coverage["maximum_containment_lhs_after_scale"]))
        coverage_updates.append(
            {
                "update_dense_slot": int(slot),
                "virtual_count": len(target_local),
                "shape_aware_virtual_2sigma_support": True,
                **coverage,
            }
        )
    return updates, {
        "virtual_count": len(target_local),
        "coverage": {
            "certificate": DEPTHSPLAT_COVERAGE_CERTIFICATE,
            "shape_aware_virtual_2sigma_support": True,
            "maximum_containment_lhs_after_scale": max(coverage_lhs, default=0.0),
            "moment_covariance_scale_max": 1.0,
            "per_update": coverage_updates,
        },
        "virtual_geometry_source": "selected-probe-gaussian-spatial-moment-v1",
        "virtual_attribute_source": "selected-probe-gaussian-spatial-linear-v1",
        "omitted_routing_z_depth_reads": 0,
        "selected_probe_attribute_reads": len(anchor_indices),
        "opacity_compositing_order": "literal-weighted-average-no-alpha-union-v1",
        "assignment_feature_semantics": "raw-bilinear-s1-v1",
    }


def preflight_depthsplat_l0_l1_materialization(
    initial_packet: DepthSplatSparseRawPacket,
    initial_packed: DepthSplatPackedGaussianAttributes,
    plan: IncrementalProbeFirstPlan,
    routing_features: torch.Tensor,
    routing_z_depths: torch.Tensor,
    *,
    source_sample_image_grid: Callable[..., tuple[Any, Any]],
    source_get_world_rays: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    maximum_feature_relative_residual: float = DEPTHSPLAT_FEATURE_INTERPOLATION_MAX_RELATIVE_RESIDUAL,
    maximum_coverage_covariance_scale: float = DEPTHSPLAT_COMPACT_COVERAGE_MAX_COVARIANCE_SCALE,
    execution_profile: str = DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
    selected_anchor_attribute_loo_frozen_guard: Any | None = None,
    selected_anchor_attribute_loo_maximum_risk: float | None = None,
    collect_selected_anchor_attribute_loo_risk: bool = False,
) -> DepthSplatCompactMaterializationPreflight:
    """Construct all nonzero L0/L1 updates without reading skipped attributes.

    A runtime LOO threshold must arrive as a projection of a verified frozen
    ACID V16 record.  The legacy scalar is deliberately rejected so a DL3DV
    audit cannot quietly substitute an arbitrary value for that record.
    """

    if (
        not isinstance(maximum_feature_relative_residual, (int, float))
        or isinstance(maximum_feature_relative_residual, bool)
        or maximum_feature_relative_residual < 0.0
        or not isinstance(maximum_coverage_covariance_scale, (int, float))
        or isinstance(maximum_coverage_covariance_scale, bool)
        or maximum_coverage_covariance_scale < 1.0
        or not isinstance(collect_selected_anchor_attribute_loo_risk, bool)
        or execution_profile
        not in {
            DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
        }
    ):
        raise ValueError("DepthSplat materializer safety thresholds are invalid")
    if selected_anchor_attribute_loo_maximum_risk is not None:
        raise ValueError(
            "DepthSplat selected-anchor LOO requires a frozen V16 guard, not a scalar threshold"
        )
    frozen_loo_guard = _selected_anchor_attribute_loo_frozen_guard(
        selected_anchor_attribute_loo_frozen_guard,
        require_literal_t4_authenticated=(
            execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        ),
    )
    collect_loo = collect_selected_anchor_attribute_loo_risk or frozen_loo_guard is not None
    views, height, width, semantics = _require_plan(plan)
    if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
        if float(maximum_coverage_covariance_scale) != 1.0:
            raise ValueError(
                "DepthSplat formal paper profile forbids covariance expansion"
            )
        _validate_literal_paper_t4_plan(
            plan,
            views=views,
            height=height,
            width=width,
            semantics=semantics,
        )
    if frozen_loo_guard is not None:
        if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
            expected_route_config_sha256 = literal_paper_t4_route_config_sha256(
                plan.events
            )
            if (
                frozen_loo_guard["frozen_record_kind"]
                != DEPTHSPLAT_LITERAL_PAPER_T4_V16_RECORD_KIND
                or frozen_loo_guard["materialization_profile"]
                != DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
                or frozen_loo_guard["route_plan_contract"]
                != LITERAL_PAPER_T4_PLAN_CONTRACT
                or frozen_loo_guard["route_plan_contract"]
                != plan.events.get("contract_version")
                or frozen_loo_guard["route_plan_config_sha256"]
                != expected_route_config_sha256
                or plan.events.get("literal_paper_t4_route_config_sha256")
                != expected_route_config_sha256
            ):
                raise ValueError("DepthSplat formal V16L/T4 LOO guard binding changed")
        if frozen_loo_guard["materialization_profile"] != execution_profile:
            raise ValueError("DepthSplat selected-anchor LOO guard profile changed")
        if frozen_loo_guard["route_plan_contract"] != plan.events.get(
            "contract_version"
        ):
            raise ValueError("DepthSplat selected-anchor LOO guard route contract changed")
    slot_to_index = _validate_source(
        initial_packet, initial_packed, plan, views=views, height=height, width=width
    )
    assignment_features, assignment_feature_semantics = _validate_routing_inputs(
        routing_features,
        routing_z_depths,
        initial_packet,
        plan,
        views=views,
        height=height,
        width=width,
        execution_profile=execution_profile,
    )
    source_grid = (
        _source_image_grid(
            source_sample_image_grid,
            height=height,
            width=width,
            device=initial_packet.raw_head_descriptors.device,
        )
        if execution_profile == DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE
        else None
    )
    plan_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in plan.tile_trace
    }
    if len(plan_records) != views * (height // 4) * (width // 4):
        raise ValueError("DepthSplat materializer plan trace has duplicate tiles")
    updates: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    promote = torch.zeros_like(plan.selection_mask)
    trace: list[dict[str, Any]] = []
    rejection_reasons: dict[str, int] = {}
    feature_statistic = str(plan.events["feature_statistic"])
    for view in range(views):
        for tile_y in range(height // 4):
            for tile_x in range(width // 4):
                record = plan_records[(view, tile_y, tile_x)]
                level = record.get("pre_guard_route")
                if level not in {"L0", "L1", "Full"}:
                    raise ValueError("DepthSplat materializer plan route is invalid")
                entry: dict[str, Any] = {
                    "view": view,
                    "tile_y": tile_y,
                    "tile_x": tile_x,
                    "planned_route": level,
                    "attempted": level in {"L0", "L1"},
                    "accepted": level == "Full",
                    "source_nonprobe_s3_attribute_reads": 0,
                    "selected_anchor_attribute_loo": None,
                }
                if level == "Full":
                    entry["reason"] = "source_full_passthrough"
                    trace.append(entry)
                    continue
                try:
                    if collect_loo:
                        endpoint_anchor_count = _selected_anchor_opacity_endpoint_count(
                            packed=initial_packed,
                            slot_to_index=slot_to_index,
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            height=height,
                            width=width,
                            level=str(level),
                            semantics=semantics,
                            plan_record=record,
                        )
                        if endpoint_anchor_count:
                            loo = _selected_anchor_attribute_loo_unscorable_certificate(
                                level=str(level),
                                semantics=semantics,
                                plan_record=record,
                                endpoint_anchor_count=endpoint_anchor_count,
                            )
                            entry["selected_anchor_attribute_loo"] = {
                                **loo,
                                "maximum_allowed_risk": (
                                    frozen_loo_guard["threshold_value"]
                                    if frozen_loo_guard is not None
                                    else None
                                ),
                                "passed": None,
                                "action": "promote_full_unscorable",
                            }
                            raise ValueError(NATIVE_OPACITY_ENDPOINT_FULL_REASON)
                        loo = depthsplat_selected_anchor_attribute_loo_certificate(
                            packed=initial_packed,
                            slot_to_index=slot_to_index,
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            height=height,
                            width=width,
                            level=str(level),
                            semantics=semantics,
                            plan_record=record,
                        )
                        maximum_risk = (
                            frozen_loo_guard["threshold_value"]
                            if frozen_loo_guard is not None
                            else None
                        )
                        passed = (
                            None
                            if maximum_risk is None
                            else loo["maximum_held_out_risk"] <= float(maximum_risk)
                        )
                        loo = {
                            **loo,
                            "maximum_allowed_risk": maximum_risk,
                            "passed": passed,
                            "action": (
                                "observed_only"
                                if passed is None
                                else "retain_compact"
                                if passed
                                else "promote_full"
                            ),
                        }
                        entry["selected_anchor_attribute_loo"] = loo
                        if passed is False:
                            raise ValueError(
                                "DepthSplat selected-anchor LOO rejected tile"
                            )
                    if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
                        tile_updates, evidence = (
                            _build_literal_paper_t4_selected_only_tile_updates(
                                packet=initial_packet,
                                packed=initial_packed,
                                slot_to_index=slot_to_index,
                                raw_feature_map=assignment_features[view],
                                record=record,
                                level=level,
                                semantics=semantics,
                                view=view,
                                tile_y=tile_y,
                                tile_x=tile_x,
                                height=height,
                                width=width,
                                feature_statistic=feature_statistic,
                            )
                        )
                    else:
                        if source_grid is None:
                            raise RuntimeError("DepthSplat development source grid is missing")
                        tile_updates, evidence = _build_tile_updates(
                            packet=initial_packet,
                            packed=initial_packed,
                            slot_to_index=slot_to_index,
                            feature_map=assignment_features[view],
                            z_depth_map=routing_z_depths[0, view],
                            record=record,
                            level=level,
                            semantics=semantics,
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            height=height,
                            width=width,
                            feature_statistic=feature_statistic,
                            source_grid=source_grid,
                            source_get_world_rays=source_get_world_rays,
                            maximum_feature_relative_residual=float(maximum_feature_relative_residual),
                            maximum_coverage_covariance_scale=float(maximum_coverage_covariance_scale),
                        )
                except (RuntimeError, ValueError) as error:
                    reason = str(error) or type(error).__name__
                    _mark_tile(promote, view=view, tile_y=tile_y, tile_x=tile_x, tile_size=4)
                    entry.update({"accepted": False, "reason": reason})
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                else:
                    updates.extend(tile_updates)
                    entry.update({"accepted": True, "reason": "accepted", **evidence})
                trace.append(entry)
    ordered = sorted(updates, key=lambda value: value[0])
    if len({slot for slot, *_ in ordered}) != len(ordered):
        raise RuntimeError("DepthSplat materializer generated duplicate anchor updates")
    dtype = initial_packed.means.dtype
    device = initial_packed.means.device
    if ordered:
        update_slots = torch.tensor([slot for slot, *_ in ordered], device=device, dtype=torch.int64)
        means = torch.stack([mean for _, mean, _, _, _ in ordered])
        covariances = torch.stack([covariance for _, _, covariance, _, _ in ordered])
        harmonics = torch.stack([harmonic for _, _, _, harmonic, _ in ordered])
        opacities = torch.stack([opacity for _, _, _, _, opacity in ordered])
    else:
        update_slots = torch.empty(0, device=device, dtype=torch.int64)
        means = torch.empty(0, 3, device=device, dtype=dtype)
        covariances = torch.empty(0, 3, 3, device=device, dtype=dtype)
        harmonics = torch.empty(0, 3, initial_packed.harmonics.shape[2], device=device, dtype=initial_packed.harmonics.dtype)
        opacities = torch.empty(0, device=device, dtype=initial_packed.opacities.dtype)
    if (
        not bool(torch.isfinite(means).all())
        or not bool(torch.isfinite(covariances).all())
        or not bool(torch.isfinite(harmonics).all())
        or not bool(torch.isfinite(opacities).all())
        or bool((opacities < 0.0).any())
        or bool((opacities >= 1.0).any())
    ):
        raise ValueError("DepthSplat materializer preflight updates are invalid")
    coverage_records = [record.get("coverage") for record in trace if isinstance(record.get("coverage"), Mapping)]
    attempted_tiles = sum(bool(record["attempted"]) for record in trace)
    trace_sha256 = _canonical_sha256(trace)
    loo_aggregate = (
        _selected_anchor_attribute_loo_aggregate(
            trace,
            mode="frozen_v16_guard" if frozen_loo_guard is not None else "collect_only",
            frozen_guard=frozen_loo_guard,
        )
        if collect_loo
        else None
    )
    loo_aggregate_sha256 = (
        _canonical_sha256(loo_aggregate) if loo_aggregate is not None else None
    )
    initial_binding = {
        "plan_selection_mask_sha256": _mask_sha256(plan.selection_mask),
        "plan_tile_trace_sha256": str(plan.events["tile_trace_sha256"]),
        "packet_selection_mask_sha256": str(initial_packet.source_trace["selection_mask_sha256"]),
        "packet_selected_descriptor_sha256": _require_sha256(
            initial_packet.source_trace["selected_descriptor_sha256"],
            label="selected descriptor",
        ),
        "packet_selected_rgb_sha256": _require_sha256(
            initial_packet.source_trace["selected_rgb_sha256"], label="selected RGB"
        ),
        "native_execution_sha256": _require_sha256(
            initial_packet.source_trace["native_execution_sha256"], label="native execution"
        ),
        "packet_native_full_passthrough_mask_sha256": _require_sha256(
            initial_packet.source_trace["native_full_passthrough_mask_sha256"],
            label="initial native Full mask",
        ),
        "packed_source_trace_sha256": initial_packed.source_trace_sha256,
        "packed_native_attribute_binding_sha256": _require_sha256(
            initial_packed.attribute_binding_sha256, label="native Adapter attributes"
        ),
        "packed_native_full_attribute_binding_sha256": _require_sha256(
            initial_packed.source_trace[
                "native_full_adapter_attribute_binding_sha256"
            ],
            label="native Full Adapter attributes",
        ),
        "routing_features_sha256": _tensor_sha256(routing_features),
        "routing_z_depths_sha256": _tensor_sha256(routing_z_depths),
        "assignment_feature_map_sha256": _tensor_sha256(assignment_features),
        "assignment_feature_semantics": assignment_feature_semantics,
        "execution_profile": execution_profile,
    }
    update_binding = _update_binding(
        update_slots, means, covariances, harmonics, opacities
    )
    coverage_by_slot: dict[int, dict[str, Any]] = {}
    for tile_record in trace:
        coverage = tile_record.get("coverage")
        if not isinstance(coverage, Mapping):
            continue
        per_update = coverage.get("per_update")
        if not isinstance(per_update, list):
            raise ValueError("DepthSplat coverage tile record is incomplete")
        for row in per_update:
            if not isinstance(row, Mapping):
                raise ValueError("DepthSplat coverage tile row is invalid")
            slot = row.get("update_dense_slot")
            if isinstance(slot, bool) or not isinstance(slot, int) or slot in coverage_by_slot:
                raise ValueError("DepthSplat coverage tile slot is invalid")
            coverage_by_slot[slot] = dict(row)
    ordered_coverage_rows: list[dict[str, Any]] = []
    for update_index, (slot, *_values) in enumerate(ordered):
        row = coverage_by_slot.pop(int(slot), None)
        if row is None:
            raise ValueError("DepthSplat coverage is missing an accepted update slot")
        ordered_coverage_rows.append({**row, "update_index": update_index})
    if coverage_by_slot:
        raise ValueError("DepthSplat coverage has an unbound update slot")
    coverage_certificate_payload = _coverage_certificate_payload(
        update_slots=update_slots,
        update_binding=update_binding,
        per_update=ordered_coverage_rows,
        tile_trace_sha256=trace_sha256,
        maximum_covariance_scale=float(maximum_coverage_covariance_scale),
    )
    coverage_certificate_sha256 = _canonical_sha256(coverage_certificate_payload)
    materialization_session_sha256 = _canonical_sha256(
        {
            "schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
            "initial_binding": initial_binding,
            "update_binding": update_binding,
            "promote_full_mask_sha256": _mask_sha256(promote),
            "tile_trace_sha256": trace_sha256,
            "execution_profile": execution_profile,
            "assignment_feature_map_sha256": initial_binding[
                "assignment_feature_map_sha256"
            ],
            "assignment_feature_semantics": assignment_feature_semantics,
            "selected_anchor_attribute_loo_frozen_guard": frozen_loo_guard,
            "selected_anchor_attribute_loo_aggregate_sha256": loo_aggregate_sha256,
        }
    )
    events = {
        "schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
        "aggregation": DEPTHSPLAT_COMPACT_AGGREGATION,
        "coverage_certificate": DEPTHSPLAT_COVERAGE_CERTIFICATE,
        "coverage_certificate_payload": coverage_certificate_payload,
        "coverage_certificate_sha256": coverage_certificate_sha256,
        "maximum_coverage_covariance_scale": float(maximum_coverage_covariance_scale),
        "execution_profile": execution_profile,
        "formal_paper_selected_probe_only": (
            execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        ),
        "target_rgb_accessed": False,
        "target_rgb_accessed_before_commit": False,
        "target_camera_accessed_before_commit": False,
        "skipped_s3_attributes_accessed": False,
        "source_nonprobe_s3_attribute_reads": 0,
        "not_lossless_deletion": True,
        "nonzero_direct_deletion": False,
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "numerical_execution": "strict-fp32-native-attributes-v1",
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "initial_selection_mask_sha256": initial_binding["plan_selection_mask_sha256"],
        "initial_binding": initial_binding,
        "update_binding": update_binding,
        "materialization_session_sha256": materialization_session_sha256,
        "routing_features_sha256": initial_binding["routing_features_sha256"],
        "routing_z_depths_sha256": initial_binding["routing_z_depths_sha256"],
        "assignment_feature_map_sha256": initial_binding[
            "assignment_feature_map_sha256"
        ],
        "assignment_feature_semantics": assignment_feature_semantics,
        "source_pixel_grid_sha256": (
            _tensor_sha256(source_grid) if source_grid is not None else None
        ),
        "omitted_routing_z_depth_reads": (
            sum(int(record.get("omitted_routing_z_depth_reads", 0)) for record in trace)
            if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
            else None
        ),
        "l1_anchor_semantics": semantics,
        "attempted_tiles": attempted_tiles,
        "accepted_tiles": sum(bool(record["attempted"] and record["accepted"]) for record in trace),
        "promoted_full_tiles": sum(bool(record["attempted"] and not record["accepted"]) for record in trace),
        "update_anchor_count": int(update_slots.numel()),
        "coverage_checked_tiles": len(coverage_records),
        "coverage_max_containment_lhs_after_scale": max(
            (
                float(record["maximum_containment_lhs_after_scale"])
                for record in coverage_records
            ),
            default=0.0,
        ),
        "coverage_max_moment_covariance_scale": max((float(record["moment_covariance_scale_max"]) for record in coverage_records), default=0.0),
        "selected_anchor_attribute_loo_certificate": (
            DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE
        ),
        "selected_anchor_attribute_loo_policy": (
            DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY
        ),
        "selected_anchor_attribute_loo_risk_metric": (
            DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC
        ),
        "selected_anchor_attribute_loo_collect_only": (
            collect_loo and frozen_loo_guard is None
        ),
        "selected_anchor_attribute_loo_guard": frozen_loo_guard is not None,
        "selected_anchor_attribute_loo_frozen_guard": frozen_loo_guard,
        "selected_anchor_attribute_loo_aggregate": loo_aggregate,
        "selected_anchor_attribute_loo_aggregate_sha256": loo_aggregate_sha256,
        "promote_full_mask_sha256": _mask_sha256(promote),
        "tile_trace_sha256": trace_sha256,
        "rejection_reasons": rejection_reasons,
    }
    return DepthSplatCompactMaterializationPreflight(
        update_dense_slots=update_slots,
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        promote_full_mask=promote,
        tile_trace=tuple(trace),
        events=events,
    )


def resolve_depthsplat_compact_final_route(
    plan: IncrementalProbeFirstPlan,
    preflight: DepthSplatCompactMaterializationPreflight,
) -> DepthSplatCompactFinalRoute:
    """Resolve final L0/L1 output slots, discarding producer-only prefetches."""

    views, height, width, semantics = _require_plan(plan)
    if not isinstance(preflight, DepthSplatCompactMaterializationPreflight):
        raise TypeError("DepthSplat final route requires a materialization preflight")
    initial_binding = preflight.events.get("initial_binding")
    update_binding = preflight.events.get("update_binding")
    session = preflight.events.get("materialization_session_sha256")
    if (
        preflight.events.get("schema_version") != DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION
        or not isinstance(initial_binding, Mapping)
        or not isinstance(update_binding, Mapping)
        or not isinstance(session, str)
        or initial_binding.get("plan_selection_mask_sha256") != _mask_sha256(plan.selection_mask)
        or initial_binding.get("plan_tile_trace_sha256") != plan.events.get("tile_trace_sha256")
        or update_binding
        != _update_binding(
            preflight.update_dense_slots,
            preflight.means,
            preflight.covariances,
            preflight.harmonics,
            preflight.opacities,
        )
    ):
        raise ValueError("DepthSplat final route preflight binding changed")
    coverage_certificate = _validate_coverage_certificate(preflight)
    loo_aggregate = _validate_selected_anchor_attribute_loo_aggregate(
        events=preflight.events,
        tile_trace=preflight.tile_trace,
    )
    expected_session = _canonical_sha256(
        {
            "schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
            "initial_binding": dict(initial_binding),
            "update_binding": dict(update_binding),
            "promote_full_mask_sha256": _mask_sha256(preflight.promote_full_mask),
            "tile_trace_sha256": preflight.events.get("tile_trace_sha256"),
            "execution_profile": preflight.events.get("execution_profile"),
            "assignment_feature_map_sha256": preflight.events.get(
                "assignment_feature_map_sha256"
            ),
            "assignment_feature_semantics": preflight.events.get(
                "assignment_feature_semantics"
            ),
            "selected_anchor_attribute_loo_frozen_guard": preflight.events.get(
                "selected_anchor_attribute_loo_frozen_guard"
            ),
            "selected_anchor_attribute_loo_aggregate_sha256": preflight.events.get(
                "selected_anchor_attribute_loo_aggregate_sha256"
            ),
        }
    )
    if session != expected_session:
        raise ValueError("DepthSplat final route materialization session changed")
    if (
        preflight.promote_full_mask.shape != plan.selection_mask.shape
        or preflight.promote_full_mask.dtype != torch.bool
        or preflight.promote_full_mask.device != plan.selection_mask.device
    ):
        raise ValueError("DepthSplat final route promotion mask is invalid")
    preflight_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in preflight.tile_trace
    }
    if len(preflight_records) != len(plan.tile_trace):
        raise ValueError("DepthSplat final route preflight trace is incomplete")
    selected = torch.zeros_like(plan.selection_mask)
    full_passthrough = torch.zeros_like(plan.selection_mask)
    trace: list[dict[str, Any]] = []
    route_counts = {"L0": 0, "L1": 0, "Full": 0}
    for plan_record in plan.tile_trace:
        view = int(plan_record["view"])
        tile_y = int(plan_record["tile_y"])
        tile_x = int(plan_record["tile_x"])
        record = preflight_records[(view, tile_y, tile_x)]
        planned = str(plan_record["pre_guard_route"])
        promote_tile = preflight.promote_full_mask[
            view, tile_y * 4 : (tile_y + 1) * 4, tile_x * 4 : (tile_x + 1) * 4
        ]
        promoted = bool(promote_tile.all())
        if bool(promote_tile.any()) != promoted:
            raise ValueError("DepthSplat final route has a partial Full promotion")
        attempted = planned in {"L0", "L1"}
        if (
            record.get("planned_route") != planned
            or record.get("attempted") is not attempted
            or (attempted and promoted == (record.get("accepted") is True))
            or (not attempted and promoted)
        ):
            raise ValueError("DepthSplat final route preflight tile decision diverged")
        final = "Full" if promoted else planned
        positions = _route_anchor_positions(plan_record, level=final, semantics=semantics)
        _mark_tile(selected, view=view, tile_y=tile_y, tile_x=tile_x, tile_size=4, positions=positions)
        if final == "Full":
            _mark_tile(full_passthrough, view=view, tile_y=tile_y, tile_x=tile_x, tile_size=4)
        route_counts[final] += 1
        trace.append({
            "view": view,
            "tile_y": tile_y,
            "tile_x": tile_x,
            "planned_route": planned,
            "final_route": final,
            "compact_materialization": dict(record),
        })
    additional = selected & ~plan.selection_mask
    raw_request = plan.selection_mask | additional
    if bool((selected & ~raw_request).any()):
        raise RuntimeError("DepthSplat final route output exceeds native producer request")
    trace_sha256 = _canonical_sha256(trace)
    route_session_sha256 = _canonical_sha256(
        {
            "preflight_session_sha256": session,
            "selected_output_mask_sha256": _mask_sha256(selected),
            "additional_full_mask_sha256": _mask_sha256(additional),
            "raw_head_request_mask_sha256": _mask_sha256(raw_request),
            "full_passthrough_mask_sha256": _mask_sha256(full_passthrough),
            "tile_trace_sha256": trace_sha256,
        }
    )
    events = {
        "schema_version": "saes-depthsplat-compact-l0-l1-route-v1",
        "materializer_schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "target_rgb_accessed_before_commit": False,
        "target_camera_accessed_before_commit": False,
        "skipped_s3_attributes_accessed": False,
        "nonzero_direct_deletion": False,
        "not_lossless_deletion": True,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "initial_selection_mask_sha256": _mask_sha256(plan.selection_mask),
        "selected_output_mask_sha256": _mask_sha256(selected),
        "additional_full_mask_sha256": _mask_sha256(additional),
        "raw_head_request_mask_sha256": _mask_sha256(raw_request),
        "full_passthrough_mask_sha256": _mask_sha256(full_passthrough),
        "selected_output_descriptor_count": int(selected.sum().item()),
        "producer_only_prefetch_descriptor_count": int((raw_request & ~selected).sum().item()),
        "additional_full_descriptor_count": int(additional.sum().item()),
        "requires_incremental_full_dispatch": bool(additional.any()),
        "route_counts": route_counts,
        "l1_anchor_semantics": semantics,
        "preflight_trace_sha256": preflight.events.get("tile_trace_sha256"),
        "preflight_materialization_session_sha256": session,
        "coverage_certificate_sha256": preflight.events[
            "coverage_certificate_sha256"
        ],
        "coverage_certificate_geometry": coverage_certificate["geometry"],
        "selected_anchor_attribute_loo_aggregate_sha256": (
            _canonical_sha256(loo_aggregate) if loo_aggregate is not None else None
        ),
        "selected_anchor_attribute_loo_frozen_guard": preflight.events.get(
            "selected_anchor_attribute_loo_frozen_guard"
        ),
        "route_session_sha256": route_session_sha256,
        "tile_trace_sha256": trace_sha256,
    }
    return DepthSplatCompactFinalRoute(
        selected_output_mask=selected,
        additional_full_mask=additional,
        raw_head_request_mask=raw_request,
        full_passthrough_mask=full_passthrough,
        tile_trace=tuple(trace),
        events=events,
    )


def _validate_final_slots(
    packed: DepthSplatPackedGaussianAttributes,
    route: DepthSplatCompactFinalRoute,
) -> dict[int, int]:
    if not isinstance(packed, DepthSplatPackedGaussianAttributes):
        raise TypeError("DepthSplat final materialization requires native packed attributes")
    positions = route.selected_output_mask.nonzero(as_tuple=False)
    _, height, width = route.selected_output_mask.shape
    expected_slots = (
        positions[:, 0] * (height * width) + positions[:, 1] * width + positions[:, 2]
    ).to(device=packed.dense_slots.device, dtype=torch.int64)
    if not torch.equal(packed.dense_slots, expected_slots):
        raise ValueError("DepthSplat final packet does not match selected output mask")
    slots = packed.dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(set(slots)) != len(slots):
        raise ValueError("DepthSplat final packet has duplicate slots")
    return {int(slot): index for index, slot in enumerate(slots)}


def apply_depthsplat_compact_l0_l1_materialization(
    final_packed: DepthSplatPackedGaussianAttributes,
    preflight: DepthSplatCompactMaterializationPreflight,
    final_route: DepthSplatCompactFinalRoute,
) -> DepthSplatPackedGaussianAttributes:
    """Apply accepted anchor updates while preserving every Full slot bitwise."""

    if not isinstance(preflight, DepthSplatCompactMaterializationPreflight) or not isinstance(
        final_route, DepthSplatCompactFinalRoute
    ):
        raise TypeError("DepthSplat final materialization requires preflight and route")
    preflight_session = preflight.events.get("materialization_session_sha256")
    update_binding = preflight.events.get("update_binding")
    expected_update_binding = _update_binding(
        preflight.update_dense_slots,
        preflight.means,
        preflight.covariances,
        preflight.harmonics,
        preflight.opacities,
    )
    expected_route_session = _canonical_sha256(
        {
            "preflight_session_sha256": preflight_session,
            "selected_output_mask_sha256": _mask_sha256(final_route.selected_output_mask),
            "additional_full_mask_sha256": _mask_sha256(final_route.additional_full_mask),
            "raw_head_request_mask_sha256": _mask_sha256(final_route.raw_head_request_mask),
            "full_passthrough_mask_sha256": _mask_sha256(final_route.full_passthrough_mask),
            "tile_trace_sha256": final_route.events.get("tile_trace_sha256"),
        }
    )
    if (
        not isinstance(preflight_session, str)
        or update_binding != expected_update_binding
        or final_route.events.get("preflight_materialization_session_sha256")
        != preflight_session
        or final_route.events.get("preflight_trace_sha256")
        != preflight.events.get("tile_trace_sha256")
        or final_route.events.get("route_session_sha256") != expected_route_session
    ):
        raise ValueError("DepthSplat final packet route binding changed")
    coverage_certificate = _validate_coverage_certificate(preflight)
    loo_aggregate = _validate_selected_anchor_attribute_loo_aggregate(
        events=preflight.events,
        tile_trace=preflight.tile_trace,
    )
    if (
        final_route.events.get("coverage_certificate_sha256")
        != preflight.events.get("coverage_certificate_sha256")
        or final_route.events.get("coverage_certificate_geometry")
        != coverage_certificate["geometry"]
        or final_route.events.get("selected_anchor_attribute_loo_aggregate_sha256")
        != (_canonical_sha256(loo_aggregate) if loo_aggregate is not None else None)
        or final_route.events.get("selected_anchor_attribute_loo_frozen_guard")
        != preflight.events.get("selected_anchor_attribute_loo_frozen_guard")
    ):
        raise ValueError("DepthSplat final route evidence changed")
    final_trace = final_packed.source_trace
    final_native_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=final_packed.dense_slots,
        means=final_packed.means,
        covariances=final_packed.covariances,
        harmonics=final_packed.harmonics,
        opacities=final_packed.opacities,
    )
    initial_binding = preflight.events.get("initial_binding")
    if (
        not isinstance(final_trace, Mapping)
        or not isinstance(initial_binding, Mapping)
        or final_packed.source_trace_sha256 != canonical_json_sha256(dict(final_trace))
        or final_packed.attribute_binding_sha256 != final_native_attribute_binding
        or final_trace.get("native_adapter_attribute_binding_sha256")
        != final_native_attribute_binding
        or final_trace.get("native_execution_sha256")
        != initial_binding.get("native_execution_sha256")
        or final_trace.get("native_full_passthrough_mask_sha256")
        != _mask_sha256(final_route.full_passthrough_mask)
        or final_trace.get("native_full_passthrough_positions")
        != int(final_route.full_passthrough_mask.sum().item())
        or final_trace.get("native_full_adapter_attribute_execution_sha256")
        != initial_binding.get("native_execution_sha256")
        or final_trace.get("native_full_adapter_attribute_passthrough_mask_sha256")
        != _mask_sha256(final_route.full_passthrough_mask)
        or final_trace.get("native_full_adapter_attribute_passthrough_count")
        != int(final_route.full_passthrough_mask.sum().item())
        or final_trace.get("packet_selection_kind")
        != "depthsplat-final-selected-output-mask-v1"
        or final_trace.get("selection_mask_sha256")
        != _mask_sha256(final_route.selected_output_mask)
        or final_trace.get("producer_request_mask_sha256")
        != _mask_sha256(final_route.raw_head_request_mask)
    ):
        raise ValueError("DepthSplat final packet does not bind its native replay route")
    slot_to_index = _validate_final_slots(final_packed, final_route)
    final_full_positions = final_route.full_passthrough_mask.nonzero(as_tuple=False)
    _, final_height, final_width = final_route.full_passthrough_mask.shape
    final_full_slots_tensor = (
        final_full_positions[:, 0] * (final_height * final_width)
        + final_full_positions[:, 1] * final_width
        + final_full_positions[:, 2]
    ).to(device=final_packed.dense_slots.device, dtype=torch.int64)
    final_full_indices = torch.searchsorted(
        final_packed.dense_slots, final_full_slots_tensor
    )
    if bool((final_full_indices >= final_packed.dense_slots.numel()).any()) or not torch.equal(
        final_packed.dense_slots[final_full_indices], final_full_slots_tensor
    ):
        raise ValueError("DepthSplat final packet omits a Full passthrough slot")
    final_full_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=final_full_slots_tensor,
        means=final_packed.means[final_full_indices],
        covariances=final_packed.covariances[final_full_indices],
        harmonics=final_packed.harmonics[final_full_indices],
        opacities=final_packed.opacities[final_full_indices],
    )
    if (
        final_trace.get("native_full_adapter_attribute_binding_sha256")
        != final_full_attribute_binding
    ):
        raise ValueError("DepthSplat final Full attributes do not match their capture binding")
    count = int(preflight.update_dense_slots.numel())
    if (
        preflight.means.shape != (count, 3)
        or preflight.covariances.shape != (count, 3, 3)
        or preflight.harmonics.shape != (count, 3, final_packed.harmonics.shape[2])
        or preflight.opacities.shape != (count,)
    ):
        raise ValueError("DepthSplat materializer update tensor shapes are inconsistent")
    update_slots = preflight.update_dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if any(int(slot) not in slot_to_index for slot in update_slots):
        raise ValueError("DepthSplat materializer update does not belong to final packet")
    means = final_packed.means.clone()
    covariances = final_packed.covariances.clone()
    harmonics = final_packed.harmonics.clone()
    opacities = final_packed.opacities.clone()
    if count:
        indices = torch.tensor([slot_to_index[int(slot)] for slot in update_slots], device=means.device, dtype=torch.long)
        means[indices] = preflight.means.to(device=means.device, dtype=means.dtype)
        covariances[indices] = preflight.covariances.to(device=covariances.device, dtype=covariances.dtype)
        harmonics[indices] = preflight.harmonics.to(device=harmonics.device, dtype=harmonics.dtype)
        opacities[indices] = preflight.opacities.to(device=opacities.device, dtype=opacities.dtype)
    full_positions = final_route.full_passthrough_mask.nonzero(as_tuple=False)
    _, height, width = final_route.full_passthrough_mask.shape
    full_slots = (
        full_positions[:, 0] * (height * width) + full_positions[:, 1] * width + full_positions[:, 2]
    ).detach().to(device="cpu", dtype=torch.int64).tolist()
    full_indices = torch.tensor([slot_to_index[int(slot)] for slot in full_slots], device=means.device, dtype=torch.long)
    if full_indices.numel() and (
        not torch.equal(means[full_indices], final_packed.means[full_indices])
        or not torch.equal(covariances[full_indices], final_packed.covariances[full_indices])
        or not torch.equal(harmonics[full_indices], final_packed.harmonics[full_indices])
        or not torch.equal(opacities[full_indices], final_packed.opacities[full_indices])
    ):
        raise RuntimeError("DepthSplat materializer modified a Full passthrough attribute")
    if (
        not bool(torch.isfinite(means).all())
        or not bool(torch.isfinite(covariances).all())
        or not bool(torch.isfinite(harmonics).all())
        or not bool(torch.isfinite(opacities).all())
        or bool((opacities < 0.0).any())
        or bool((opacities > 1.0).any())
        or bool((torch.linalg.eigvalsh((covariances + covariances.mT) * 0.5) < -1e-6).any())
    ):
        raise ValueError("DepthSplat materialized attributes violate the native contract")
    source_trace = dict(final_packed.source_trace)
    source_trace.update(
        {
            "depthsplat_compact_materialization_schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
            "packet_selection_kind": "depthsplat-final-selected-output-mask-v1",
            "depthsplat_compact_preflight_trace_sha256": preflight.events["tile_trace_sha256"],
            "depthsplat_compact_final_route_trace_sha256": final_route.events["tile_trace_sha256"],
            "depthsplat_compact_materialization_session_sha256": preflight_session,
            "depthsplat_compact_route_session_sha256": expected_route_session,
            "depthsplat_compact_update_anchor_count": count,
            "depthsplat_compact_full_passthrough_count": int(full_indices.numel()),
            "depthsplat_compact_aggregation": DEPTHSPLAT_COMPACT_AGGREGATION,
            "depthsplat_compact_coverage_certificate": DEPTHSPLAT_COVERAGE_CERTIFICATE,
            "depthsplat_compact_coverage_certificate_sha256": preflight.events[
                "coverage_certificate_sha256"
            ],
            "depthsplat_compact_execution_profile": preflight.events[
                "execution_profile"
            ],
            "depthsplat_compact_assignment_feature_map_sha256": preflight.events[
                "assignment_feature_map_sha256"
            ],
            "depthsplat_compact_assignment_feature_semantics": preflight.events[
                "assignment_feature_semantics"
            ],
            "depthsplat_compact_omitted_routing_z_depth_reads": preflight.events[
                "omitted_routing_z_depth_reads"
            ],
            "depthsplat_compact_selected_anchor_attribute_loo_aggregate_sha256": (
                _canonical_sha256(loo_aggregate) if loo_aggregate is not None else None
            ),
            "depthsplat_compact_selected_anchor_attribute_loo_frozen_guard": preflight.events.get(
                "selected_anchor_attribute_loo_frozen_guard"
            ),
            "depthsplat_compact_skipped_s3_attributes_accessed": False,
            "depthsplat_compact_nonzero_direct_deletion": False,
            "depthsplat_compact_z_depth_geometry": True,
            "depthsplat_compact_source_rgb_sh_initialization": True,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        }
    )
    materialized_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=final_packed.dense_slots,
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )
    source_trace["depthsplat_compact_current_attribute_binding_sha256"] = (
        materialized_attribute_binding
    )
    return DepthSplatPackedGaussianAttributes(
        dense_slots=final_packed.dense_slots.clone(),
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        source_trace=source_trace,
        source_trace_sha256=canonical_json_sha256(source_trace),
        attribute_binding_sha256=materialized_attribute_binding,
    )


__all__ = [
    "DEPTHSPLAT_COMPACT_AGGREGATION",
    "DEPTHSPLAT_COMPACT_COVERAGE_MAX_COVARIANCE_SCALE",
    "DEPTHSPLAT_COVERAGE_CERTIFICATE",
    "DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION",
    "DepthSplatCompactFinalRoute",
    "DepthSplatCompactMaterializationPreflight",
    "NATIVE_OPACITY_ENDPOINT_FULL_REASON",
    "apply_depthsplat_compact_l0_l1_materialization",
    "depthsplat_z_depth_world_means",
    "preflight_depthsplat_l0_l1_materialization",
    "resolve_depthsplat_compact_final_route",
]
