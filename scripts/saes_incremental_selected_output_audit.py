#!/usr/bin/env python3
"""Audit selected raw-head replay and packed Adapter conversion without target data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.incremental_selected_output_execution import (  # noqa: E402
    incremental_selected_output_head_execution,
)
from saes.guarded_selected_route import (  # noqa: E402
    ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
    resolve_guarded_selected_route,
)
from saes.guard_policy import (  # noqa: E402
    CONTEXT_GUARD_MAX_CENTER_MAHALANOBIS,
    CONTEXT_GUARD_MAX_FOOTPRINT_RATIO,
    CONTEXT_GUARD_MAX_RELATIVE_DEPTH_SPAN,
    CONTEXT_GUARD_POLICY,
)
from saes.probe_first_schedule import (  # noqa: E402
    ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    BALANCED_L1_ANCHOR_SEMANTICS,
    LEGACY_L1_ANCHOR_SEMANTICS,
    build_incremental_probe_first_plan,
)
from saes.progressive_saes import DELETION_CERTIFICATE_SOURCE_KIND  # noqa: E402
from saes.selected_output_replay import FP32_ATOL, FP32_RTOL  # noqa: E402
from saes.sparse_gaussian_consumer import (  # noqa: E402
    PackedGaussianConsumer,
    PackedGaussianAttributes,
    SparseRawGaussianPacket,
)
from scripts.saes_selected_output_replay_audit import (  # noqa: E402
    strict_fp32_convolution_execution,
)


MODEL = "transplat"
DATASET = "dl3dv"
TILE_SIZE = 4
FEATURE_THRESHOLD = 0.20
DEPTH_THRESHOLD = 0.10
DECISION_SEMANTICS = "probe-normalized-std-first-hit"
AUDIT_SCHEMA_VERSION = "saes-incremental-selected-output-adapter-audit-v1"
GUARD_DISTRIBUTION_SCHEMA_VERSION = "saes-guarded-route-geometry-distribution-v1"
GUARD_DISTRIBUTION_FILENAME = "guard-distribution.json"
GUARD_DISTRIBUTION_THRESHOLDS = (2.0, 2.146, 2.448)


class _AdapterPreempted(RuntimeError):
    """Internal control flow that prevents the dense Adapter from consuming a map."""


class _PackedAdapterPreempted(RuntimeError):
    """Stop an encoder pass after its same-invocation packed Adapter diagnostic."""

    def __init__(self, captured: dict[str, Any]) -> None:
        super().__init__("incremental packed Adapter diagnostic complete")
        self.captured = captured


@dataclass(frozen=True)
class SelectedAdapterInputs:
    """Only the Adapter values belonging to selected source descriptors."""

    extrinsics: torch.Tensor
    intrinsics: torch.Tensor
    coordinates: torch.Tensor
    depths: torch.Tensor
    mapped_opacities: torch.Tensor
    raw_body: torch.Tensor
    image_shape: tuple[int, int]


def _stored_tensor(value: torch.Tensor, storage_device: torch.device | str | None) -> torch.Tensor:
    captured = value.detach()
    if storage_device is not None:
        captured = captured.to(storage_device)
    return captured.clone()


def _selected_positions(selection_mask: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if (
        not torch.is_tensor(selection_mask)
        or selection_mask.ndim != 3
        or selection_mask.dtype != torch.bool
        or selection_mask.shape[0] < 1
        or selection_mask.shape[1] < 1
        or selection_mask.shape[2] < 1
    ):
        raise ValueError("selected packet requires a nonempty [V,H,W] bool mask")
    positions = selection_mask.nonzero(as_tuple=False)
    if positions.numel() == 0:
        raise ValueError("selected packet requires at least one raw descriptor")
    return positions, int(selection_mask.shape[1]), int(selection_mask.shape[2])


def _selected_head_descriptors(
    raw_head: torch.Tensor,
    selection_mask: torch.Tensor,
    *,
    storage_device: torch.device | str | None = None,
) -> torch.Tensor:
    """Gather selected raw descriptors in canonical ``view,row,column`` order."""
    positions, height, width = _selected_positions(selection_mask)
    if (
        not torch.is_tensor(raw_head)
        or raw_head.ndim != 4
        or raw_head.shape[0] != selection_mask.shape[0]
        or tuple(raw_head.shape[-2:]) != (height, width)
    ):
        raise ValueError("raw head output does not match the selected [V,H,W] layout")
    positions = positions.to(raw_head.device)
    return _stored_tensor(
        raw_head.permute(0, 2, 3, 1)[
        positions[:, 0], positions[:, 1], positions[:, 2]
        ],
        storage_device,
    )


def _extract_selected_adapter_inputs(
    adapter_inputs: tuple[Any, ...],
    selection_mask: torch.Tensor,
    *,
    storage_device: torch.device | str | None = None,
) -> SelectedAdapterInputs:
    """Extract the selected subset from one native Adapter pre-hook invocation."""
    if len(adapter_inputs) != 7:
        raise ValueError("native GaussianAdapter did not receive seven positional inputs")
    (
        extrinsics,
        intrinsics,
        coordinates,
        depths,
        mapped_opacities,
        raw_body,
        image_shape,
    ) = adapter_inputs
    if not isinstance(image_shape, tuple) or len(image_shape) != 2:
        raise ValueError("native GaussianAdapter has an invalid image shape")
    height, width = image_shape
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("native GaussianAdapter image shape must be positive")
    positions, mask_height, mask_width = _selected_positions(selection_mask)
    if (height, width) != (mask_height, mask_width):
        raise ValueError("native GaussianAdapter image shape does not match selection")
    tensors = (
        extrinsics,
        intrinsics,
        coordinates,
        depths,
        mapped_opacities,
        raw_body,
    )
    if any(not torch.is_tensor(value) for value in tensors):
        raise ValueError("native GaussianAdapter inputs must be tensors")
    views = int(selection_mask.shape[0])
    pixels = height * width
    if (
        extrinsics.shape != (1, views, 1, 1, 1, 4, 4)
        or intrinsics.shape != (1, views, 1, 1, 1, 3, 3)
        or coordinates.shape != (1, views, pixels, 1, 1, 2)
        or depths.shape != (1, views, pixels, 1, 1)
        or mapped_opacities.shape != (1, views, pixels, 1, 1)
        or raw_body.ndim != 6
        or raw_body.shape[:5] != (1, views, pixels, 1, 1)
    ):
        raise ValueError("native GaussianAdapter inputs violate the B=1/S=1/P=1 contract")
    if any(value.device != raw_body.device for value in tensors):
        raise ValueError("native GaussianAdapter inputs must share one device")
    positions = positions.to(raw_body.device)
    view = positions[:, 0]
    pixel = positions[:, 1] * width + positions[:, 2]
    return SelectedAdapterInputs(
        extrinsics=_stored_tensor(extrinsics[0, view, 0, 0, 0], storage_device),
        intrinsics=_stored_tensor(intrinsics[0, view, 0, 0, 0], storage_device),
        coordinates=_stored_tensor(coordinates[0, view, pixel, 0, 0], storage_device),
        depths=_stored_tensor(depths[0, view, pixel, 0, 0], storage_device),
        mapped_opacities=_stored_tensor(
            mapped_opacities[0, view, pixel, 0, 0], storage_device
        ),
        raw_body=_stored_tensor(raw_body[0, view, pixel, 0, 0], storage_device),
        image_shape=(height, width),
    )


def _source_native_selected_coordinates_from_final_raw(
    final_raw_descriptors: torch.Tensor, selection_mask: torch.Tensor
) -> torch.Tensor:
    """Call TranSplat's shared offset geometry for the final selected packet."""
    positions, height, width = _selected_positions(selection_mask)
    if (
        not torch.is_tensor(final_raw_descriptors)
        or final_raw_descriptors.ndim != 2
        or final_raw_descriptors.shape[0] != positions.shape[0]
        or final_raw_descriptors.shape[1] < 2
    ):
        raise ValueError("final raw descriptors do not match the selected geometry layout")
    from transplat.src.model.encoder.encoder_trans import (
        gaussian_adapter_coordinates_from_raw_offsets,
    )

    pixels = (positions[:, 1] * width + positions[:, 2]).to(
        device=final_raw_descriptors.device, dtype=torch.int64
    )
    return gaussian_adapter_coordinates_from_raw_offsets(
        final_raw_descriptors[:, :2].sigmoid(),
        image_shape=(height, width),
        pixel_indices=pixels,
    )


def _require_source_trace_value(
    trace: Mapping[str, Any], name: str, expected_type: type[Any]
) -> Any:
    value = trace.get(name)
    if not isinstance(value, expected_type):
        raise ValueError(f"incremental head trace has no valid {name}")
    return value


def _validate_phase_plan_binding(
    plan_events: Mapping[str, Any], head_events: Mapping[str, Any]
) -> None:
    """Reject a scoped execution ledger that diverges from its route plan."""
    if not isinstance(plan_events, Mapping) or not isinstance(head_events, Mapping):
        raise TypeError("phase-plan binding requires mapping records")
    if head_events.get("head_forward_invocations") != 1:
        raise ValueError("phase-plan binding requires exactly one head invocation")
    phases = head_events.get("phases")
    if not isinstance(phases, list) or len(phases) != 3:
        raise ValueError("phase-plan binding requires primary, secondary, and full phases")
    expected = (
        ("primary", "primary_mask_sha256", "primary_head_final_positions"),
        ("secondary", "secondary_mask_sha256", "secondary_head_final_positions"),
        ("full", "full_mask_sha256", "full_head_final_positions"),
    )
    executed_total = 0
    for event, (phase_name, mask_key, count_key) in zip(phases, expected):
        if not isinstance(event, Mapping) or event.get("phase") != phase_name:
            raise ValueError(f"phase-plan binding has an invalid {phase_name} phase")
        if event.get("mask_sha256") != plan_events.get(mask_key):
            raise ValueError(f"phase-plan binding {phase_name} phase mask does not match plan")
        requested = event.get("head_final_positions_requested")
        reused = event.get("head_final_positions_reused")
        executed = event.get("head_final_positions_executed")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (requested, reused, executed)
        ):
            raise ValueError(f"phase-plan binding {phase_name} phase has invalid counters")
        if requested != plan_events.get(count_key):
            raise ValueError(f"phase-plan binding {phase_name} phase request count does not match plan")
        if requested != reused + executed:
            raise ValueError(f"phase-plan binding {phase_name} phase does not conserve work")
        executed_total += executed
    scheduled = plan_events.get("scheduled_head_positions")
    if (
        isinstance(scheduled, bool)
        or not isinstance(scheduled, int)
        or scheduled < 0
        or head_events.get("head_final_positions_executed") != scheduled
        or executed_total != scheduled
    ):
        raise ValueError("phase-plan binding does not cover the scheduled output union")


def _validate_appended_full_extension_binding(
    plan_events: Mapping[str, Any],
    initial_head_events: Mapping[str, Any],
    final_head_events: Mapping[str, Any],
    guard_events: Mapping[str, Any],
) -> None:
    """Bind a guard-requested Full extension to the original head session.

    The initial route remains the fixed three-phase plan.  A post-Adapter
    extension is valid only when it is a disjoint fourth phase on that exact
    source-bound head input and its final computed-mask digest covers the
    guard's requested union.  This proves no second context/head invocation
    was used to make the extension appear incremental.
    """
    _validate_phase_plan_binding(plan_events, initial_head_events)
    if not isinstance(final_head_events, Mapping) or not isinstance(guard_events, Mapping):
        raise TypeError("Full extension binding requires mapping records")
    if final_head_events.get("head_forward_invocations") != 1:
        raise ValueError("Full extension binding requires one scoped head invocation")
    for name in ("head_weight_sha256", "head_input_sha256"):
        initial = initial_head_events.get(name)
        final = final_head_events.get(name)
        if not isinstance(initial, str) or final != initial:
            raise ValueError(f"Full extension binding changed the source {name}")
    initial_phases = initial_head_events.get("phases")
    final_phases = final_head_events.get("phases")
    if (
        not isinstance(initial_phases, list)
        or not isinstance(final_phases, list)
        or len(initial_phases) != 3
        or len(final_phases) != 4
    ):
        raise ValueError("Full extension binding requires three initial phases and one extension")
    if final_phases[:3] != initial_phases:
        raise ValueError("Full extension binding changed the completed route plan phases")
    extension = final_phases[3]
    if not isinstance(extension, Mapping) or extension.get("phase") != "full_extension":
        raise ValueError("Full extension binding has no appended Full phase")
    extension_digest = extension.get("mask_sha256")
    guard_digest = guard_events.get("additional_full_mask_sha256")
    if not isinstance(extension_digest, str) or extension_digest != guard_digest:
        raise ValueError("Full extension phase mask does not match the guard request")
    requested = extension.get("head_final_positions_requested")
    reused = extension.get("head_final_positions_reused")
    executed = extension.get("head_final_positions_executed")
    expected_count = guard_events.get("additional_full_descriptor_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (requested, reused, executed, expected_count)
    ):
        raise ValueError("Full extension binding has invalid dispatch counters")
    if (
        requested != expected_count
        or requested != executed
        or reused != 0
        or final_head_events.get("full_extension_mask_sha256") != extension_digest
        or final_head_events.get("full_extension_positions_executed") != executed
        or final_head_events.get("full_extension_dispatched") is not True
    ):
        raise ValueError("Full extension binding does not conserve appended work")
    initial_executed = initial_head_events.get("head_final_positions_executed")
    final_executed = final_head_events.get("head_final_positions_executed")
    if (
        isinstance(initial_executed, bool)
        or isinstance(final_executed, bool)
        or not isinstance(initial_executed, int)
        or not isinstance(final_executed, int)
        or final_executed != initial_executed + executed
    ):
        raise ValueError("Full extension binding has an invalid final output count")
    if guard_events.get("requires_incremental_full_dispatch") is not True:
        raise ValueError("Full extension binding lacks a guard dispatch request")
    computed_digest = final_head_events.get("computed_mask_sha256")
    request_digest = guard_events.get("raw_head_request_mask_sha256")
    if not isinstance(computed_digest, str) or computed_digest != request_digest:
        raise ValueError("Full extension binding does not cover the guard request union")
    if final_head_events.get("full_tile_native_identity_verified") is not False:
        raise ValueError("Full extension binding cannot claim native Full identity")


def _build_sparse_packet(
    raw_head: torch.Tensor,
    selection_mask: torch.Tensor,
    selected_inputs: SelectedAdapterInputs,
    *,
    head_events: Mapping[str, Any],
    plan_events: Mapping[str, Any],
    coordinates_source: str = "native_adapter_inputs",
) -> SparseRawGaussianPacket:
    """Bind selected raw descriptors and same-pass Adapter side inputs together."""
    positions, height, width = _selected_positions(selection_mask)
    if raw_head.ndim == 4:
        raw_descriptors = _selected_head_descriptors(raw_head, selection_mask)
    elif raw_head.ndim == 2 and raw_head.shape[0] == positions.shape[0]:
        raw_descriptors = raw_head.detach().clone()
    else:
        raise ValueError("raw descriptors do not match the selected [V,H,W] layout")
    positions = positions.to(raw_descriptors.device)
    descriptor_count = int(positions.shape[0])
    if (
        selected_inputs.image_shape != (height, width)
        or selected_inputs.raw_body.shape[0] != descriptor_count
        or selected_inputs.depths.shape != (descriptor_count,)
        or selected_inputs.mapped_opacities.shape != (descriptor_count,)
        or raw_descriptors.shape[1] != selected_inputs.raw_body.shape[1] + 2
    ):
        raise ValueError("selected raw-head and Adapter inputs have incompatible shapes")
    side_tensors = (
        selected_inputs.raw_body,
        selected_inputs.depths,
        selected_inputs.mapped_opacities,
    )
    if any(value.device != raw_descriptors.device for value in side_tensors):
        raise ValueError("selected raw-head and Adapter inputs must share one device")
    if head_events.get("source_bound") is not True:
        raise ValueError("incremental head trace is not source-bound")
    if head_events.get("execution_scope") != "s3_raw_gaussian_head_only":
        raise ValueError("incremental head trace has an unsupported execution scope")
    if head_events.get("head_forward_invocations") != 1:
        raise ValueError(
            "packed Adapter audit requires one same-context raw-head invocation"
        )
    if coordinates_source not in {
        "native_adapter_inputs",
        "source_native_raw_offset_geometry",
        "source_native_raw_offset_geometry_after_extension",
    }:
        raise ValueError("packed packet has an unsupported coordinate source")
    executed = _require_source_trace_value(
        head_events, "head_final_positions_executed", int
    )
    if executed < descriptor_count:
        raise ValueError("incremental head trace executed too few selected descriptors")
    producer_schema = _require_source_trace_value(head_events, "schema_version", str)
    head_weight_sha256 = _require_source_trace_value(
        head_events, "head_weight_sha256", str
    )
    head_input_sha256 = _require_source_trace_value(
        head_events, "head_input_sha256", str
    )
    phase_trace_sha256 = _require_source_trace_value(
        head_events, "phase_trace_sha256", str
    )
    route_trace_sha256 = _require_source_trace_value(
        plan_events, "tile_trace_sha256", str
    )
    plan_contract = _require_source_trace_value(plan_events, "contract_version", str)
    route_primary_mask_sha256 = _require_source_trace_value(
        plan_events, "primary_mask_sha256", str
    )
    route_secondary_mask_sha256 = _require_source_trace_value(
        plan_events, "secondary_mask_sha256", str
    )
    route_full_mask_sha256 = _require_source_trace_value(
        plan_events, "full_mask_sha256", str
    )
    route_selection_mask_sha256 = _require_source_trace_value(
        plan_events, "selection_mask_sha256", str
    )

    view = positions[:, 0].to(dtype=torch.int64)
    pixel = (positions[:, 1] * width + positions[:, 2]).to(dtype=torch.int64)
    zero = torch.zeros_like(view)
    descriptor_keys = torch.stack((zero, view, pixel, zero), dim=1)
    primitive_keys = torch.stack((zero, view, pixel, zero, zero), dim=1)
    primitive_to_descriptor = torch.arange(
        descriptor_count, dtype=torch.int64, device=raw_descriptors.device
    )
    dense_slots = view * (height * width) + pixel
    source_trace = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_bound": True,
        "execution_scope": "s3_raw_gaussian_head_only",
        "producer_schema_version": producer_schema,
        "head_weight_sha256": head_weight_sha256,
        "head_input_sha256": head_input_sha256,
        "head_final_positions_executed": executed,
        "head_forward_invocations": 1,
        "phase_trace_sha256": phase_trace_sha256,
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
        "coordinates_source": coordinates_source,
        "route_plan_contract_version": plan_contract,
        "route_tile_trace_sha256": route_trace_sha256,
        "route_primary_mask_sha256": route_primary_mask_sha256,
        "route_secondary_mask_sha256": route_secondary_mask_sha256,
        "route_full_mask_sha256": route_full_mask_sha256,
        "route_selection_mask_sha256": route_selection_mask_sha256,
    }
    return SparseRawGaussianPacket(
        descriptor_keys=descriptor_keys,
        raw_descriptors=raw_descriptors,
        primitive_keys=primitive_keys,
        primitive_to_descriptor=primitive_to_descriptor,
        coordinates=selected_inputs.coordinates,
        depths=selected_inputs.depths,
        mapped_opacities=selected_inputs.mapped_opacities,
        dense_slots=dense_slots,
        source_trace=source_trace,
    )


def _packet_to_device(
    packet: SparseRawGaussianPacket, device: torch.device
) -> SparseRawGaussianPacket:
    return SparseRawGaussianPacket(
        descriptor_keys=packet.descriptor_keys.to(device),
        raw_descriptors=packet.raw_descriptors.to(device),
        primitive_keys=packet.primitive_keys.to(device),
        primitive_to_descriptor=packet.primitive_to_descriptor.to(device),
        coordinates=packet.coordinates.to(device),
        depths=packet.depths.to(device),
        mapped_opacities=packet.mapped_opacities.to(device),
        dense_slots=packet.dense_slots.to(device),
        source_trace=dict(packet.source_trace),
    )


def _adapter_inputs_to_device(
    inputs: SelectedAdapterInputs, device: torch.device
) -> SelectedAdapterInputs:
    return SelectedAdapterInputs(
        extrinsics=inputs.extrinsics.to(device),
        intrinsics=inputs.intrinsics.to(device),
        coordinates=inputs.coordinates.to(device),
        depths=inputs.depths.to(device),
        mapped_opacities=inputs.mapped_opacities.to(device),
        raw_body=inputs.raw_body.to(device),
        image_shape=inputs.image_shape,
    )


def _packed_attributes_to_device(
    packed: Any, device: torch.device
) -> PackedGaussianAttributes:
    if not isinstance(packed, PackedGaussianAttributes):
        raise TypeError("packed attribute transfer requires PackedGaussianAttributes")
    return PackedGaussianAttributes(
        batch_indices=packed.batch_indices.to(device),
        dense_slots=packed.dense_slots.to(device),
        means=packed.means.to(device),
        covariances=packed.covariances.to(device),
        harmonics=packed.harmonics.to(device),
        opacities=packed.opacities.to(device),
        source_trace=dict(packed.source_trace),
        source_trace_sha256=packed.source_trace_sha256,
    )


def _gaussians_on_cpu(gaussians: Any) -> Any:
    return type(gaussians)(
        means=_stored_tensor(gaussians.means, "cpu"),
        covariances=_stored_tensor(gaussians.covariances, "cpu"),
        harmonics=_stored_tensor(gaussians.harmonics, "cpu"),
        opacities=_stored_tensor(gaussians.opacities, "cpu"),
    )


def _release_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _canonical_json_sha256(value: Any) -> str:
    """Match the guarded route's canonical trace digest without retaining it."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"guard distribution has no valid {name}")
    return value


def _require_fixed_guard_route(
    plan_events: Mapping[str, Any], route_events: Mapping[str, Any]
) -> None:
    """Fail closed unless scalar aggregation is tied to the fixed guarded route."""
    if not isinstance(plan_events, Mapping) or not isinstance(route_events, Mapping):
        raise TypeError("guard distribution requires route-plan and guarded-route mappings")
    if plan_events.get("contract_version") != "saes-incremental-probe-first-plan-v1":
        raise ValueError("guard distribution route-plan contract is not fixed")
    if route_events.get("schema_version") != "saes-guarded-selected-route-v1":
        raise ValueError("guard distribution guarded-route schema is not fixed")
    expected_plan = {
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": DEPTH_THRESHOLD,
        "decision_semantics": DECISION_SEMANTICS,
    }
    for name, expected in expected_plan.items():
        if plan_events.get(name) != expected:
            raise ValueError(f"guard distribution route-plan {name} is not fixed")
    if plan_events.get("l0_anchor_count") != 4 or plan_events.get("l1_anchor_count") != 12:
        raise ValueError("guard distribution route-plan has the wrong anchor layout")
    if route_events.get("tile_size") != TILE_SIZE:
        raise ValueError("guard distribution guarded route has the wrong tile size")
    if route_events.get("materialization") != "representative":
        raise ValueError("guard distribution guarded route has the wrong materialization")
    if route_events.get("context_safety_guard") is not True:
        raise ValueError("guard distribution requires the context safety guard")
    if route_events.get("l0_anchor_count") != 4 or route_events.get("l1_anchor_count") != 12:
        raise ValueError("guard distribution guarded route has the wrong anchor layout")
    _require_sha256(
        plan_events.get("tile_trace_sha256"), name="route-plan tile trace SHA256"
    )
    _require_sha256(
        route_events.get("source_tile_trace_sha256"),
        name="guarded-route source trace SHA256",
    )
    _require_sha256(
        route_events.get("tile_trace_sha256"), name="guarded-route tile trace SHA256"
    )
    if route_events["source_tile_trace_sha256"] != plan_events["tile_trace_sha256"]:
        raise ValueError("guard distribution source trace does not match the route plan")


def _quantile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("guard distribution quantile requires finite values")
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _scalar_distribution(
    values: list[Any], *, thresholds: tuple[float, ...] = ()
) -> dict[str, Any]:
    """Aggregate a guard metric without serializing tile-level values."""
    finite_values: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("guard distribution encountered a non-scalar metric")
        scalar = float(value)
        if math.isfinite(scalar):
            finite_values.append(scalar)
    finite_values.sort()
    result: dict[str, Any] = {
        "count": len(values),
        "finite_count": len(finite_values),
        "nonfinite_count": len(values) - len(finite_values),
        "min": None,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "max": None,
    }
    if finite_values:
        result.update(
            {
                "min": finite_values[0],
                "p50": _quantile(finite_values, 0.50),
                "p90": _quantile(finite_values, 0.90),
                "p95": _quantile(finite_values, 0.95),
                "p99": _quantile(finite_values, 0.99),
                "max": finite_values[-1],
            }
        )
    if thresholds:
        result["cumulative_counts"] = {
            str(threshold): {
                "count": sum(value <= threshold for value in finite_values),
                "finite_fraction": (
                    sum(value <= threshold for value in finite_values)
                    / len(finite_values)
                    if finite_values
                    else 0.0
                ),
            }
            for threshold in thresholds
        }
    return result


def _guard_distribution_route_identity(
    plan_events: Mapping[str, Any], route_events: Mapping[str, Any]
) -> dict[str, Any]:
    identity = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "route_plan_contract_version": plan_events["contract_version"],
        "guarded_route_schema_version": route_events["schema_version"],
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": DEPTH_THRESHOLD,
        "decision_semantics": DECISION_SEMANTICS,
        "materialization": route_events["materialization"],
        "context_safety_guard": route_events["context_safety_guard"],
        "l0_anchor_count": route_events["l0_anchor_count"],
        "l1_anchor_count": route_events["l1_anchor_count"],
        "context_guard_policy": CONTEXT_GUARD_POLICY,
        "context_guard_max_footprint_ratio": CONTEXT_GUARD_MAX_FOOTPRINT_RATIO,
        "context_guard_max_relative_depth_span": (
            CONTEXT_GUARD_MAX_RELATIVE_DEPTH_SPAN
        ),
        "context_guard_max_center_mahalanobis": (
            CONTEXT_GUARD_MAX_CENTER_MAHALANOBIS
        ),
        "center_mahalanobis_report_thresholds": list(GUARD_DISTRIBUTION_THRESHOLDS),
    }
    return {**identity, "sha256": _canonical_json_sha256(identity)}


def _build_guard_distribution(
    *,
    guarded_route_trace: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    guarded_route_events: Mapping[str, Any],
    plan_events: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build scalar-only geometry summaries for the fixed target-free route."""
    _require_fixed_guard_route(plan_events, guarded_route_events)
    if not isinstance(input_identity, Mapping):
        raise TypeError("guard distribution requires a context-only input identity")
    _require_sha256(checkpoint_sha256, name="checkpoint SHA256")
    trace = list(guarded_route_trace)
    if not all(isinstance(record, Mapping) for record in trace):
        raise ValueError("guard distribution trace has an invalid tile record")
    trace_sha256 = _canonical_json_sha256(trace)
    if trace_sha256 != guarded_route_events["tile_trace_sha256"]:
        raise ValueError("guard distribution trace does not match the guarded route")
    route_counts = guarded_route_events.get("route_counts")
    if (
        not isinstance(route_counts, Mapping)
        or set(route_counts) != {"L0", "L1", "Full"}
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in route_counts.values()
        )
        or sum(route_counts.values()) != len(trace)
    ):
        raise ValueError("guard distribution route counts do not match the trace")

    metrics = {
        level: {
            "center_mahalanobis": [],
            "footprint_ratio": [],
            "relative_depth_span": [],
        }
        for level in ("L0", "L1")
    }
    for record in trace:
        checks = record.get("guard_checks")
        if not isinstance(checks, (list, tuple)):
            raise ValueError("guard distribution tile record has no guard checks")
        for check in checks:
            if not isinstance(check, Mapping):
                raise ValueError("guard distribution has an invalid guard check")
            level = check.get("level")
            if level not in metrics:
                raise ValueError("guard distribution has an unsupported guard level")
            context = check.get("context_safety")
            if not isinstance(context, Mapping):
                raise ValueError("guard distribution guard check lacks context scalars")
            metrics[level]["center_mahalanobis"].append(
                context.get("projected_center_mahalanobis_max")
            )
            metrics[level]["footprint_ratio"].append(
                context.get("coverage_footprint_ratio")
            )
            metrics[level]["relative_depth_span"].append(
                context.get("relative_depth_span")
            )

    route_identity = _guard_distribution_route_identity(plan_events, guarded_route_events)
    return {
        "schema_version": GUARD_DISTRIBUTION_SCHEMA_VERSION,
        "kind": "saes_target_free_guard_geometry_distribution",
        "status": "COMPLETED",
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "renderer_executed": False,
        "quality_metrics_computed": False,
        "scalar_only": True,
        "trace_binding": {
            "guarded_route_tile_trace_sha256": trace_sha256,
            "source_plan_tile_trace_sha256": plan_events["tile_trace_sha256"],
            "source_selection_mask_sha256": _require_sha256(
                guarded_route_events.get("source_selection_mask_sha256"),
                name="source selection mask SHA256",
            ),
            "raw_head_request_mask_sha256": _require_sha256(
                guarded_route_events.get("raw_head_request_mask_sha256"),
                name="raw-head request mask SHA256",
            ),
            "trace_record_count": len(trace),
            "input_identity_sha256": _canonical_json_sha256(dict(input_identity)),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "route_identity": route_identity,
        "levels": {
            level: {
                "context_guard_check_count": len(
                    metrics[level]["center_mahalanobis"]
                ),
                "center_mahalanobis": _scalar_distribution(
                    metrics[level]["center_mahalanobis"],
                    thresholds=GUARD_DISTRIBUTION_THRESHOLDS,
                ),
                "footprint_ratio": _scalar_distribution(
                    metrics[level]["footprint_ratio"]
                ),
                "relative_depth_span": _scalar_distribution(
                    metrics[level]["relative_depth_span"]
                ),
            }
            for level in ("L0", "L1")
        },
    }


def _write_guard_distribution(
    output_dir: Path, distribution: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist one hash-bound scalar-only guard-distribution artifact."""
    if not output_dir.is_dir():
        raise ValueError("guard distribution output directory does not exist")
    path = output_dir / GUARD_DISTRIBUTION_FILENAME
    if path.exists():
        raise FileExistsError("guard distribution destination already exists")
    payload = json.dumps(distribution, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    trace_binding = distribution.get("trace_binding")
    route_identity = distribution.get("route_identity")
    if not isinstance(trace_binding, Mapping) or not isinstance(route_identity, Mapping):
        raise ValueError("guard distribution has no required bindings")
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "schema_version": distribution.get("schema_version"),
        "guarded_route_tile_trace_sha256": trace_binding.get(
            "guarded_route_tile_trace_sha256"
        ),
        "route_identity_sha256": route_identity.get("sha256"),
        "scalar_only": True,
        "target_rgb_accessed": False,
    }


def _numeric_equivalence(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        raise TypeError("numeric equivalence requires tensors")
    if reference.shape != candidate.shape:
        return {
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "equivalent": False,
        }
    if reference.numel() == 0:
        raise ValueError("numeric equivalence requires a nonempty tensor")
    if reference.device != candidate.device:
        candidate = candidate.to(reference.device)
    delta = (reference - candidate).abs()
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "dtype": str(reference.dtype).replace("torch.", ""),
        "atol": FP32_ATOL,
        "rtol": FP32_RTOL,
        "maximum_absolute_delta": float(delta.max().item()),
        "mean_absolute_delta": float(delta.mean().item()),
        "equivalent": bool(
            torch.allclose(reference, candidate, atol=FP32_ATOL, rtol=FP32_RTOL)
        ),
    }


def _attribute_equivalence(
    packed: Any, dense_reference: Any, dense_slots: torch.Tensor
) -> dict[str, Any]:
    if dense_slots.ndim != 1 or dense_slots.dtype != torch.int64:
        raise ValueError("dense Adapter comparison requires canonical dense slots")
    attributes = {}
    for name in ("means", "covariances", "harmonics", "opacities"):
        candidate = getattr(packed, name)
        attributes[name] = _numeric_equivalence(
            getattr(dense_reference, name)[0, dense_slots.cpu()].to(candidate.device),
            candidate,
        )
    return {
        "attributes": attributes,
        "equivalent": all(value["equivalent"] for value in attributes.values()),
    }


class _CapturingAdapter:
    """Forward selected packet inputs to the native Adapter while retaining one call."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self.d_in = getattr(adapter, "d_in", None)
        self.inputs: tuple[Any, ...] | None = None

    def __call__(self, *args: Any) -> Any:
        if self.inputs is not None:
            raise RuntimeError("packed Adapter audit expected one conversion call")
        self.inputs = args
        return self._adapter(*args)


def _adapter_input_equivalence(
    adapter_inputs: tuple[Any, ...] | None, selected_inputs: SelectedAdapterInputs
) -> dict[str, Any]:
    if adapter_inputs is None or len(adapter_inputs) != 7:
        raise RuntimeError("packed Adapter conversion did not expose one input tuple")
    expected = (
        selected_inputs.extrinsics,
        selected_inputs.intrinsics,
        selected_inputs.coordinates,
        selected_inputs.depths,
        selected_inputs.mapped_opacities,
        selected_inputs.raw_body,
    )
    names = (
        "extrinsics",
        "intrinsics",
        "coordinates",
        "depths",
        "mapped_opacities",
        "raw_body",
    )
    comparisons = {
        name: _numeric_equivalence(reference, observed)
        for name, reference, observed in zip(names, expected, adapter_inputs[:6])
    }
    return {
        "image_shape_matches": adapter_inputs[6] == selected_inputs.image_shape,
        "inputs": comparisons,
        "equivalent": adapter_inputs[6] == selected_inputs.image_shape
        and all(value["equivalent"] for value in comparisons.values()),
    }


def _selected_adapter_inputs_equivalence(
    reference: SelectedAdapterInputs, candidate: SelectedAdapterInputs
) -> dict[str, Any]:
    """Compare dense and incremental Adapter inputs for one canonical selection."""
    if not isinstance(reference, SelectedAdapterInputs) or not isinstance(
        candidate, SelectedAdapterInputs
    ):
        raise TypeError("Adapter input comparison requires selected Adapter inputs")
    names = (
        "extrinsics",
        "intrinsics",
        "coordinates",
        "depths",
        "mapped_opacities",
        "raw_body",
    )
    comparisons = {
        name: _numeric_equivalence(getattr(reference, name), getattr(candidate, name))
        for name in names
    }
    return {
        "image_shape_matches": reference.image_shape == candidate.image_shape,
        "inputs": comparisons,
        "equivalent": reference.image_shape == candidate.image_shape
        and all(value["equivalent"] for value in comparisons.values()),
    }


def _capture_dense_reference(
    model: Any, context: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Run the dense context encoder once and capture its actual route inputs."""
    predictor = model.encoder.depth_predictor
    head = getattr(predictor, "to_gaussians", None)
    adapter = getattr(model.encoder, "gaussian_adapter", None)
    if head is None or adapter is None:
        raise RuntimeError("TranSplat encoder lacks the raw Gaussian head or Adapter")
    captured: dict[str, Any] = {}

    def capture_predictor(
        _module: Any, inputs: tuple[Any, ...], output: Any
    ) -> None:
        if (
            len(inputs) < 1
            or not torch.is_tensor(inputs[0])
            or inputs[0].ndim != 5
            or not isinstance(output, tuple)
            or len(output) != 3
            or not torch.is_tensor(output[0])
        ):
            raise RuntimeError("depth predictor did not expose B/V features and depths")
        captured["features"] = _stored_tensor(inputs[0], "cpu")
        captured["depths"] = _stored_tensor(output[0], "cpu")

    def capture_head(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if (
            len(inputs) != 1
            or not torch.is_tensor(inputs[0])
            or inputs[0].ndim != 4
            or not torch.is_tensor(output)
            or output.ndim != 4
        ):
            raise RuntimeError("raw Gaussian head did not expose [VB,C,H,W] tensors")
        captured["head_input"] = _stored_tensor(inputs[0], "cpu")
        captured["raw_head"] = _stored_tensor(output, "cpu")

    original_adapter_forward = adapter.forward
    had_instance_adapter_forward = "forward" in adapter.__dict__

    def capture_adapter(*inputs: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise RuntimeError("native GaussianAdapter unexpectedly received keyword inputs")
        if len(inputs) != 7 or "adapter_inputs" in captured:
            raise RuntimeError("dense encoder did not expose exactly one Adapter input tuple")
        captured["adapter_inputs"] = tuple(
            _stored_tensor(value, "cpu") if torch.is_tensor(value) else value
            for value in inputs
        )
        return original_adapter_forward(*inputs)

    predictor_handle = predictor.register_forward_hook(capture_predictor)
    head_handle = head.register_forward_hook(capture_head)
    adapter.forward = capture_adapter  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            dense_gaussians = _gaussians_on_cpu(
                model.encoder(context, 0, deterministic=True)
            )
    finally:
        if had_instance_adapter_forward:
            adapter.forward = original_adapter_forward  # type: ignore[method-assign]
        else:
            delattr(adapter, "forward")
        head_handle.remove()
        predictor_handle.remove()
    if set(captured) != {
        "features",
        "depths",
        "head_input",
        "raw_head",
        "adapter_inputs",
    }:
        raise RuntimeError("dense context pass did not expose all required raw-head inputs")
    return dense_gaussians, captured


def _capture_s1_s2_without_dense_adapter(
    model: Any, context: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Capture route inputs while preventing dense Adapter/Gaussian creation.

    TranSplat produces the raw head before exposing its final S2 depth tensor,
    so this planning pass still runs the source head.  It stops immediately at
    the Adapter boundary and retains neither raw descriptors nor dense
    Gaussian attributes.  Downstream records must therefore keep the global
    S2/S3 sparse-execution flag false.
    """
    predictor = model.encoder.depth_predictor
    adapter = getattr(model.encoder, "gaussian_adapter", None)
    if adapter is None:
        raise RuntimeError("TranSplat encoder lacks a GaussianAdapter")
    captured: dict[str, torch.Tensor] = {}

    def capture_predictor(
        _module: Any, inputs: tuple[Any, ...], output: Any
    ) -> None:
        if (
            len(inputs) < 1
            or not torch.is_tensor(inputs[0])
            or inputs[0].ndim != 5
            or not isinstance(output, tuple)
            or len(output) != 3
            or not torch.is_tensor(output[0])
        ):
            raise RuntimeError("planning pass did not expose B/V features and depths")
        if captured:
            raise RuntimeError("planning pass invoked its predictor more than once")
        captured["features"] = _stored_tensor(inputs[0], "cpu")
        captured["depths"] = _stored_tensor(output[0], "cpu")

    def preempt_adapter(*_inputs: Any, **_kwargs: Any) -> None:
        raise _AdapterPreempted()

    predictor_handle = predictor.register_forward_hook(capture_predictor)
    original_adapter_forward = adapter.forward
    had_instance_adapter_forward = "forward" in adapter.__dict__
    adapter.forward = preempt_adapter  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            try:
                model.encoder(context, 0, deterministic=True)
            except _AdapterPreempted:
                pass
    finally:
        if had_instance_adapter_forward:
            adapter.forward = original_adapter_forward  # type: ignore[method-assign]
        else:
            delattr(adapter, "forward")
        predictor_handle.remove()
    if set(captured) != {"features", "depths"}:
        raise RuntimeError(
            "planning pass did not stop at the dense Adapter boundary: "
            f"captured={sorted(captured)}"
        )
    return captured


def _capture_incremental_pre_adapter(
    model: Any,
    context: dict[str, Any],
    *,
    primary_mask: torch.Tensor,
    secondary_mask: torch.Tensor,
    full_mask: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a selected head pass and stop at the native Adapter entry boundary."""
    predictor = model.encoder.depth_predictor
    head = getattr(predictor, "to_gaussians", None)
    adapter = getattr(model.encoder, "gaussian_adapter", None)
    if head is None or adapter is None:
        raise RuntimeError("TranSplat encoder lacks raw-head or GaussianAdapter modules")
    captured: dict[str, Any] = {}

    def capture_head(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if (
            len(inputs) != 1
            or not torch.is_tensor(inputs[0])
            or inputs[0].ndim != 4
            or not torch.is_tensor(output)
            or output.ndim != 4
        ):
            raise RuntimeError("incremental raw head did not expose [VB,C,H,W] tensors")
        captured["head_input"] = _stored_tensor(inputs[0], "cpu")
        captured["raw_head"] = _selected_head_descriptors(
            output,
            primary_mask | secondary_mask | full_mask,
            storage_device="cpu",
        )

    def capture_adapter(*inputs: Any, **kwargs: Any) -> None:
        if kwargs:
            raise RuntimeError("native GaussianAdapter unexpectedly received keyword inputs")
        captured["adapter_inputs"] = _extract_selected_adapter_inputs(
            inputs,
            primary_mask | secondary_mask | full_mask,
            storage_device="cpu",
        )
        raise _AdapterPreempted()

    head_handle = head.register_forward_hook(capture_head)
    original_adapter_forward = adapter.forward
    had_instance_adapter_forward = "forward" in adapter.__dict__
    adapter.forward = capture_adapter  # type: ignore[method-assign]
    trace = None
    try:
        with incremental_selected_output_head_execution(
            head,
            primary_mask,
            secondary_mask,
            full_mask,
            tile_size=TILE_SIZE,
        ) as scoped_trace:
            trace = scoped_trace
            try:
                with torch.no_grad():
                    model.encoder(context, 0, deterministic=True)
            except _AdapterPreempted:
                pass
    finally:
        if had_instance_adapter_forward:
            adapter.forward = original_adapter_forward  # type: ignore[method-assign]
        else:
            delattr(adapter, "forward")
        head_handle.remove()
    expected_capture = {"head_input", "raw_head", "adapter_inputs"}
    if trace is None or set(captured) != expected_capture:
        raise RuntimeError(
            "incremental context pass did not stop at the Adapter boundary: "
            f"captured={sorted(captured)} trace_present={trace is not None}"
        )
    return trace.events, captured


def _capture_guarded_incremental_packed_adapter(
    model: Any,
    context: dict[str, Any],
    *,
    plan: Any,
    compact_nonzero_materialization: bool = False,
    compact_execution_policy: str | None = None,
    adaptive_l1_maximum_leave_one_out_residual: float | None = None,
    selected_anchor_v4_attribute_loo_maximum_risk: float | None = None,
    collect_selected_anchor_v4_attribute_loo_risk: bool = False,
) -> dict[str, Any]:
    """Run guard resolution and one optional Full extension in one encoder call.

    Route planning still precedes this diagnostic because TranSplat exposes S2
    depths only after its raw head has run.  Once the selected head call starts,
    however, S1/S2 capture, packed Adapter conversion, guard resolution, and a
    possible appended Full phase all stay within that one encoder invocation.
    The dense Adapter return is intentionally preempted so it cannot consume a
    dense raw-head map. The final packet contains only the guard-resolved
    selected outputs and can be consumed by the variable-length renderer path.
    """
    predictor = model.encoder.depth_predictor
    head = getattr(predictor, "to_gaussians", None)
    adapter = getattr(model.encoder, "gaussian_adapter", None)
    if head is None or adapter is None:
        raise RuntimeError("TranSplat encoder lacks raw-head or GaussianAdapter modules")
    if not isinstance(compact_nonzero_materialization, bool):
        raise ValueError("compact nonzero materialization must be boolean")
    if not isinstance(collect_selected_anchor_v4_attribute_loo_risk, bool):
        raise ValueError("compact V4 replay collection must be boolean")
    if (
        selected_anchor_v4_attribute_loo_maximum_risk is not None
        and compact_execution_policy
        != ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
    ):
        raise ValueError("compact V4 replay threshold requires its dedicated policy")
    if (
        not compact_nonzero_materialization
        and (
            compact_execution_policy is not None
            or adaptive_l1_maximum_leave_one_out_residual is not None
            or selected_anchor_v4_attribute_loo_maximum_risk is not None
            or collect_selected_anchor_v4_attribute_loo_risk
        )
    ):
        raise ValueError("compact policy arguments require compact materialization")
    captured: dict[str, Any] = {}

    def capture_predictor(
        _module: Any, inputs: tuple[Any, ...], output: Any
    ) -> None:
        if (
            len(inputs) < 1
            or not torch.is_tensor(inputs[0])
            or inputs[0].ndim != 5
            or not isinstance(output, tuple)
            or len(output) != 3
            or not torch.is_tensor(output[0])
        ):
            raise RuntimeError("incremental predictor did not expose B/V features and depths")
        if "features" in captured or "depths" in captured:
            raise RuntimeError("incremental encoder invoked its predictor more than once")
        captured["features"] = _stored_tensor(inputs[0], "cpu")
        captured["depths"] = _stored_tensor(output[0], "cpu")

    def capture_head_input(_module: Any, inputs: tuple[Any, ...]) -> None:
        if (
            len(inputs) != 1
            or not torch.is_tensor(inputs[0])
            or inputs[0].ndim != 4
        ):
            raise RuntimeError("incremental raw head did not receive [VB,C,H,W] input")
        if "head_input" in captured:
            raise RuntimeError("incremental encoder invoked its raw head more than once")
        captured["head_input"] = _stored_tensor(inputs[0], "cpu")

    predictor_handle = predictor.register_forward_hook(capture_predictor)
    head_handle = head.register_forward_pre_hook(capture_head_input)
    original_adapter_forward = adapter.forward
    had_instance_adapter_forward = "forward" in adapter.__dict__
    trace = None

    def packed_adapter(*inputs: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise RuntimeError("native GaussianAdapter unexpectedly received keyword inputs")
        if trace is None:
            raise RuntimeError("packed Adapter entered before the scoped head trace")
        if {"features", "depths", "head_input"} - set(captured):
            raise RuntimeError("packed Adapter lacks same-invocation S1/S2/head capture")
        initial_events = trace.initial_events
        initial_mask = trace.initial_selection_mask
        initial_inputs = _extract_selected_adapter_inputs(inputs, initial_mask)
        initial_packet = _build_sparse_packet(
            trace.live_values,
            initial_mask,
            initial_inputs,
            head_events=initial_events,
            plan_events=plan.events,
        )
        adapter_cameras = (
            inputs[0][:, :, 0, 0, 0],
            inputs[1][:, :, 0, 0, 0],
        )
        capturing_adapter = _CapturingAdapter(original_adapter_forward)
        capturing_adapter.d_in = getattr(adapter, "d_in", None)
        packed = PackedGaussianConsumer(capturing_adapter).convert(
            initial_packet,
            extrinsics=adapter_cameras[0],
            intrinsics=adapter_cameras[1],
            image_shape=inputs[6],
        )
        adapter_inputs = _adapter_input_equivalence(capturing_adapter.inputs, initial_inputs)
        cpu = torch.device("cpu")
        if compact_nonzero_materialization:
            from saes.guarded_selected_route import (
                ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY,
                ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
                ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY,
                PAPER_NONZERO_DEV_POLICY,
            )
            from saes.packed_l0_l1_materializer import (
                preflight_compact_l0_l1_materialization,
                resolve_compact_final_route,
            )
            if compact_execution_policy is not None:
                selected_compact_execution_policy = compact_execution_policy
            elif plan.events.get("l1_anchor_semantics") == LEGACY_L1_ANCHOR_SEMANTICS:
                selected_compact_execution_policy = ENGINEERING_L1_12_CONTINUITY_DEV_POLICY
            elif plan.events.get("l1_anchor_semantics") == BALANCED_L1_ANCHOR_SEMANTICS:
                selected_compact_execution_policy = (
                    ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY
                )
            elif plan.events.get("l1_anchor_semantics") == ADAPTIVE_L1_15_ANCHOR_SEMANTICS:
                selected_compact_execution_policy = ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY
            else:
                selected_compact_execution_policy = PAPER_NONZERO_DEV_POLICY
        else:
            selected_compact_execution_policy = None

        base_guarded_route = resolve_guarded_selected_route(
            plan,
            _packed_attributes_to_device(packed, cpu),
            features=captured["features"],
            depths=captured["depths"],
            context_extrinsics=adapter_cameras[0].detach().to(cpu),
            context_intrinsics=adapter_cameras[1].detach().to(cpu),
            source_opacities=(
                None if compact_nonzero_materialization else inputs[4].detach().to(cpu)
            ),
            source_opacity_certificate_kind=(
                None
                if compact_nonzero_materialization
                else DELETION_CERTIFICATE_SOURCE_KIND
            ),
            require_deletion_certificate=not compact_nonzero_materialization,
            execution_policy=selected_compact_execution_policy,
            adaptive_l1_maximum_leave_one_out_residual=(
                adaptive_l1_maximum_leave_one_out_residual
                if compact_nonzero_materialization
                else None
            ),
        )
        compact_preflight = None
        guarded_route: Any = base_guarded_route
        if compact_nonzero_materialization:
            compact_preflight = preflight_compact_l0_l1_materialization(
                _packet_to_device(initial_packet, cpu),
                _packed_attributes_to_device(packed, cpu),
                base_guarded_route,
                plan,
                captured["features"],
                adapter_cameras[0].detach().to(cpu),
                adapter_cameras[1].detach().to(cpu),
                selected_anchor_v4_attribute_loo_maximum_risk=(
                    selected_anchor_v4_attribute_loo_maximum_risk
                    if selected_compact_execution_policy
                    == ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
                    else None
                ),
                collect_selected_anchor_v4_attribute_loo_risk=(
                    collect_selected_anchor_v4_attribute_loo_risk
                ),
            )
            guarded_route = resolve_compact_final_route(
                base_guarded_route,
                plan,
                compact_preflight,
            )
        extension_event = None
        if guarded_route.events["requires_incremental_full_dispatch"]:
            extension_event = trace.append_full_extension(
                guarded_route.additional_full_mask.to(trace.live_values.device)
            )
        final_replay = trace.finalize()
        final_events = trace.events
        if extension_event is not None:
            _validate_appended_full_extension_binding(
                plan.events,
                initial_events,
                final_events,
                guarded_route.events,
            )
        final_output_mask = guarded_route.selected_output_mask.to(
            final_replay.values.device
        )
        final_inputs = _extract_selected_adapter_inputs(inputs, final_output_mask)
        stale_final_coordinates = _stored_tensor(final_inputs.coordinates, "cpu")
        # Poison every descriptor omitted from the final output, including
        # staged anchors used only to resolve a guard. This makes an accidental
        # dense read observable at the packet boundary.
        poisoned_final_values = final_replay.values.detach().clone()
        omitted_mask = ~final_output_mask
        omitted_count = int(omitted_mask.sum().item())
        if omitted_count:
            poisoned_final_values.permute(0, 2, 3, 1)[omitted_mask] = torch.nan
        final_raw_device = _selected_head_descriptors(
            poisoned_final_values, final_output_mask
        )
        final_raw = _stored_tensor(final_raw_device, "cpu")
        final_coordinates_rebuilt_from_source_geometry_after_extension = extension_event is not None
        final_inputs = SelectedAdapterInputs(
            extrinsics=final_inputs.extrinsics,
            intrinsics=final_inputs.intrinsics,
            coordinates=_source_native_selected_coordinates_from_final_raw(
                final_raw_device, final_output_mask
            ),
            depths=final_inputs.depths,
            mapped_opacities=final_inputs.mapped_opacities,
            raw_body=final_inputs.raw_body,
            image_shape=final_inputs.image_shape,
        )
        final_packet = _build_sparse_packet(
            final_raw_device,
            final_output_mask,
            final_inputs,
            head_events=final_events,
            plan_events=plan.events,
            coordinates_source=(
                "source_native_raw_offset_geometry_after_extension"
                if final_coordinates_rebuilt_from_source_geometry_after_extension
                else "source_native_raw_offset_geometry"
            ),
        )
        final_capturing_adapter = _CapturingAdapter(original_adapter_forward)
        final_capturing_adapter.d_in = getattr(adapter, "d_in", None)
        final_packed = PackedGaussianConsumer(final_capturing_adapter).convert(
            _packet_to_device(final_packet, inputs[5].device),
            extrinsics=adapter_cameras[0],
            intrinsics=adapter_cameras[1],
            image_shape=inputs[6],
        )
        if compact_nonzero_materialization:
            from saes.packed_l0_l1_materializer import apply_compact_l0_l1_materialization

            if compact_preflight is None:
                raise RuntimeError("compact materialization has no preflight")
            final_packed = apply_compact_l0_l1_materialization(
                final_packed,
                compact_preflight,
                guarded_route,
            )
        final_adapter_inputs = _adapter_input_equivalence(
            final_capturing_adapter.inputs, final_inputs
        )
        raise _PackedAdapterPreempted(
            {
                "features": captured["features"],
                "depths": captured["depths"],
                "head_input": captured["head_input"],
                "initial_head_events": initial_events,
                "final_head_events": final_events,
                "initial_packet": _packet_to_device(initial_packet, cpu),
                "initial_selected_inputs": _adapter_inputs_to_device(initial_inputs, cpu),
                "final_selected_inputs": _adapter_inputs_to_device(final_inputs, cpu),
                "stale_final_coordinates_before_source_geometry_rebuild": stale_final_coordinates,
                "final_raw_head": final_raw,
                "final_packet": _packet_to_device(final_packet, cpu),
                "final_packed": _packed_attributes_to_device(final_packed, cpu),
                "final_adapter_inputs": final_adapter_inputs,
                "final_coordinates_rebuilt_from_source_geometry_after_extension": (
                    final_coordinates_rebuilt_from_source_geometry_after_extension
                ),
                "omitted_raw_head_positions_poisoned": omitted_count,
                "packed": _packed_attributes_to_device(packed, cpu),
                "adapter_inputs": adapter_inputs,
                "guarded_route": guarded_route,
                "base_guarded_route": base_guarded_route,
                "compact_materialization_preflight": compact_preflight,
                "compact_nonzero_materialization": compact_nonzero_materialization,
                "extension_event": extension_event,
            }
        )

    adapter.forward = packed_adapter  # type: ignore[method-assign]
    try:
        with incremental_selected_output_head_execution(
            head,
            plan.primary_mask,
            plan.secondary_mask,
            plan.full_mask,
            tile_size=TILE_SIZE,
            defer_full_extension=True,
        ) as scoped_trace:
            trace = scoped_trace
            try:
                with torch.no_grad():
                    model.encoder(context, 0, deterministic=True)
            except _PackedAdapterPreempted as complete:
                captured = complete.captured
    finally:
        if had_instance_adapter_forward:
            adapter.forward = original_adapter_forward  # type: ignore[method-assign]
        else:
            delattr(adapter, "forward")
        head_handle.remove()
        predictor_handle.remove()
    required = {
        "features",
        "depths",
        "head_input",
        "initial_head_events",
        "final_head_events",
        "initial_packet",
        "initial_selected_inputs",
        "final_selected_inputs",
        "stale_final_coordinates_before_source_geometry_rebuild",
        "final_raw_head",
        "final_packet",
        "final_packed",
        "final_adapter_inputs",
        "final_coordinates_rebuilt_from_source_geometry_after_extension",
        "omitted_raw_head_positions_poisoned",
        "packed",
        "adapter_inputs",
        "guarded_route",
        "base_guarded_route",
        "compact_materialization_preflight",
        "compact_nonzero_materialization",
        "extension_event",
    }
    if trace is None or set(captured) != required:
        raise RuntimeError(
            "incremental context pass did not complete the guarded packed Adapter boundary: "
            f"captured={sorted(captured)} trace_present={trace is not None}"
        )
    return captured


def _load_context_only_encoder(
    input_root: Path, device: torch.device
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load only the frozen context-camera sidecar and an encoder-only TranSplat."""
    from data.context_only_audit_input import validate_context_only_audit_input
    from integration import create_model_loader, load_context_only_audit_data
    from scripts.ae_config import resolve_experiment
    from scripts.result_record import cached_sha256_file

    input_identity = validate_context_only_audit_input(input_root)
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    loader = create_model_loader(MODEL)
    bundle = loader.load_model(
        str(experiment.checkpoint),
        device=device,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        encoder_only=True,
    )
    if bundle.decoder is not None:
        raise RuntimeError("target-free audit unexpectedly constructed a decoder")
    data = load_context_only_audit_data(loader, bundle, input_root=input_root)
    if "target" in data.batch:
        raise RuntimeError("target-free audit loader returned a target mapping")
    model = bundle.model
    model.eval()
    context = {
        key: value.to(bundle.device) if torch.is_tensor(value) else value
        for key, value in data.batch["context"].items()
    }
    return (
        model,
        context,
        input_identity,
        {
            "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
            "environment_profile": experiment.environment_profile,
            "encoder_only": True,
            "decoder_constructed": False,
            "native_encoder_device": str(bundle.device),
        },
    )


def collect_incremental_selected_output_audit(
    *,
    input_root: Path,
    device: torch.device,
    guard_distribution_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one fixed context-only selected raw-head and Adapter diagnostic."""
    from scripts.result_record import source_identity

    model, context, input_identity, execution = _load_context_only_encoder(
        input_root, device
    )
    if context["image"].shape[0] != 1:
        raise RuntimeError("incremental selected-output audit requires batch size one")
    _, views, _, height, width = context["image"].shape
    with strict_fp32_convolution_execution() as numerical_execution:
        dense_reference, dense_capture = _capture_dense_reference(model, context)
        _release_cuda_cache(device)
        plan = build_incremental_probe_first_plan(
            dense_capture["features"],
            dense_capture["depths"],
            height=height,
            width=width,
            tile_size=TILE_SIZE,
            feature_threshold=FEATURE_THRESHOLD,
            depth_threshold=DEPTH_THRESHOLD,
            decision_semantics=DECISION_SEMANTICS,
        )
        if plan.selection_mask.shape != (views, height, width):
            raise RuntimeError("probe-first plan does not match the native B=1 view layout")
        incremental_capture = _capture_guarded_incremental_packed_adapter(
            model,
            context,
            plan=plan,
        )
        initial_head_events = incremental_capture["initial_head_events"]
        final_head_events = incremental_capture["final_head_events"]
        _validate_phase_plan_binding(plan.events, initial_head_events)
        _release_cuda_cache(device)
        dense_selected = _selected_head_descriptors(
            dense_capture["raw_head"], plan.selection_mask
        )
        guarded_route = incremental_capture["guarded_route"]
        final_mask = guarded_route.selected_output_mask
        dense_final = _selected_head_descriptors(dense_capture["raw_head"], final_mask)
        selected_inputs = incremental_capture["initial_selected_inputs"]
        final_selected_inputs = incremental_capture["final_selected_inputs"]
        assert isinstance(selected_inputs, SelectedAdapterInputs)
        assert isinstance(final_selected_inputs, SelectedAdapterInputs)
        packet = incremental_capture["initial_packet"]
        packed = incremental_capture["packed"]
        final_raw = incremental_capture["final_raw_head"]
        assert isinstance(packet, SparseRawGaussianPacket)
        assert isinstance(packed, PackedGaussianAttributes)
        assert torch.is_tensor(final_raw)
        body_binding = _numeric_equivalence(
            packet.raw_descriptors[:, 2:], selected_inputs.raw_body
        )
        final_body_binding = _numeric_equivalence(
            final_raw[:, 2:], final_selected_inputs.raw_body
        )
        dense_head_input = _numeric_equivalence(
            dense_capture["head_input"], incremental_capture["head_input"]
        )
        dense_features = _numeric_equivalence(
            dense_capture["features"], incremental_capture["features"]
        )
        dense_depths = _numeric_equivalence(
            dense_capture["depths"], incremental_capture["depths"]
        )
        selected_raw = _numeric_equivalence(dense_selected, packet.raw_descriptors)
        final_raw_equivalence = _numeric_equivalence(dense_final, final_raw)
        adapter_inputs = incremental_capture["adapter_inputs"]
        adapter_attributes = _attribute_equivalence(
            packed, dense_reference, packed.dense_slots
        )
    equivalent = all(
        value["equivalent"]
        for value in (
            body_binding,
            final_body_binding,
            dense_head_input,
            dense_features,
            dense_depths,
            selected_raw,
            final_raw_equivalence,
            adapter_inputs,
            adapter_attributes,
        )
    )
    record = {
        "schema_version": "1.0",
        "kind": "saes_incremental_selected_output_packed_adapter_audit",
        "status": "PASS" if equivalent else "FAIL",
        "paper_result_eligible": False,
        "model": MODEL,
        "dataset": DATASET,
        "scene": input_identity["scene"],
        "input_identity": input_identity,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_mapping_present": False,
        "execution_boundary": {
            "dense_context_planning_pass_executed": True,
            "incremental_context_adapter_boundary_pass_executed": True,
            "dense_native_adapter_reference_executed": True,
            "incremental_dense_native_adapter_executed": False,
            "packed_native_adapter_executed_in_same_encoder_invocation": True,
            "guarded_selected_route_resolved": True,
            "additional_full_dispatch_executed": incremental_capture["extension_event"]
            is not None,
            "appended_full_packed_adapter_executed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "timing_claim": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "scope": "s3_raw_gaussian_head_same_invocation_packed_adapter_and_guarded_full_extension_diagnostic_only",
        },
        "route_plan": plan.events,
        "raw_head": {
            "dense_head_input_equivalence": dense_head_input,
            "same_invocation_features_equivalence": dense_features,
            "same_invocation_depths_equivalence": dense_depths,
            "initial_selected_dense_vs_incremental_equivalence": selected_raw,
            "final_guard_request_dense_vs_incremental_equivalence": final_raw_equivalence,
            "initial_selected_raw_body_matches_native_adapter_input": body_binding,
            "final_guard_request_raw_body_matches_native_adapter_input": final_body_binding,
            "initial_incremental_execution": initial_head_events,
            "guarded_incremental_execution": final_head_events,
        },
        "packed_adapter": {
            "selected_descriptor_count": int(packet.descriptor_keys.shape[0]),
            "dense_slots_strictly_canonical": True,
            "packed_inputs_match_native_selected_inputs": adapter_inputs,
            "packed_attributes_vs_dense_reference": adapter_attributes,
            "scope": "initial_route_attributes_used_only_to_resolve_guard",
        },
        "guarded_selected_route": guarded_route.events,
        "numerical_execution": numerical_execution,
        "checkpoint_sha256": execution["checkpoint_sha256"],
        "execution": execution,
        "source": source_identity(),
    }
    if guard_distribution_output_dir is not None:
        distribution = _build_guard_distribution(
            guarded_route_trace=guarded_route.tile_trace,
            guarded_route_events=guarded_route.events,
            plan_events=plan.events,
            input_identity=input_identity,
            checkpoint_sha256=execution["checkpoint_sha256"],
        )
        record["guard_distribution"] = _write_guard_distribution(
            guard_distribution_output_dir, distribution
        )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--write-guard-distribution",
        action="store_true",
        help="Write hash-bound scalar-only context-guard geometry summaries",
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        record = collect_incremental_selected_output_audit(
            input_root=args.input_root,
            device=device,
            guard_distribution_output_dir=(
                args.output_dir if args.write_guard_distribution else None
            ),
        )
    except Exception as exc:
        record = {
            "schema_version": "1.0",
            "kind": "saes_incremental_selected_output_packed_adapter_audit",
            "status": "FAILED",
            "paper_result_eligible": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_mapping_present": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        exit_code = 2
    else:
        exit_code = 0
    from scripts.result_record import portable_command, write_result

    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
