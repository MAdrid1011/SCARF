"""Bounded same-weight S2/S3 streaming-cycle projection for DepthSplat.

This is an architectural projection, not a measurement from the dense PyTorch
encoder.  It uses a committed probe-first route and the existing weighted
operator-cycle breakdown to bound what a coefficient-compatible sparse
datapath could save.  The result intentionally keeps the conservative serial
schedule separate from the perfect-overlap upper bound.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


KIND = "depthsplat-same-weight-s2-s3-streaming-projection-v1"
DUAL_STREAM_KIND = "depthsplat-dual-stream-resource-schedule-v1"
S2_STAGE_NAMES = (
    "cost_volume",
    "unet_refinement",
    "depth_refinement",
    "depth_head",
    "softmax_regression",
)
S3_FEATURE_PREPARATION_STAGE = "upsampling"
S3_HEAD_STAGE = "gaussian_head"


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _ratio(numerator: int, denominator: int, *, label: str) -> float:
    if denominator <= 0 or numerator <= 0 or numerator > denominator:
        raise ValueError(f"{label} ratio is invalid")
    return numerator / denominator


def _scaled_cycles(value: int, fraction: float) -> int:
    return math.ceil(value * fraction)


def _positive_int(value: Any, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def simulate_depthsplat_dual_stream_schedule(
    *,
    dense_total_cycles: int,
    s2_probe_pipeline_cycles: int,
    s3_feature_preparation_cycles: int,
    s3_selected_head_cycles: int,
    saes_control_and_materialization_cycles: int,
    tile_count: int,
    target_saes_speedup: float = 1.26,
) -> dict[str, Any]:
    """Schedule a same-weight tiled S2/S3 streaming architecture.

    The submitted ``ScarfTop`` has one time-multiplexed MMCU and cannot use
    this schedule.  This model instead names the concrete architectural
    extension required for overlap: an independent S3 MMCU, a private S3
    weight bank, and a private S3 feature scratchpad.  It retains the source
    route's exact S2 and selected-head work fractions; only the issue schedule
    and storage topology differ.

    S3 feature tiles are emitted in raster order.  The SAES controller starts
    each tile as its feature tile arrives, and the S2 probe lane starts once
    that local decision is available.  The S3 head runs after the feature pass
    and the final S2 tile.  This is a finite tile schedule, not a fractional
    overlap knob or a wall-clock measurement.
    """

    dense_cycles = _positive_int(dense_total_cycles, label="dense total cycles")
    s2_cycles = _positive_int(
        s2_probe_pipeline_cycles, label="S2 probe-pipeline cycles"
    )
    feature_cycles = _positive_int(
        s3_feature_preparation_cycles, label="S3 feature-preparation cycles"
    )
    head_cycles = _positive_int(
        s3_selected_head_cycles, label="S3 selected-head cycles"
    )
    control_cycles = _nonnegative_int(
        saes_control_and_materialization_cycles,
        label="SAES control and materialization cycles",
    )
    tiles = _positive_int(tile_count, label="tile count")
    if (
        isinstance(target_saes_speedup, bool)
        or not isinstance(target_saes_speedup, (int, float))
        or not math.isfinite(float(target_saes_speedup))
        or float(target_saes_speedup) <= 0.0
    ):
        raise ValueError("target SAES speedup is invalid")

    # The macro stage counters need integral tile quanta.  Padding is visible
    # in the schedule rather than hidden in a fractional overlap assumption.
    feature_tile_cycles = _scaled_cycles(feature_cycles, 1.0 / tiles)
    s2_tile_cycles = _scaled_cycles(s2_cycles, 1.0 / tiles)
    control_tile_cycles = _scaled_cycles(control_cycles, 1.0 / tiles)
    head_tile_cycles = _scaled_cycles(head_cycles, 1.0 / tiles)

    feature_end = 0
    control_end = 0
    s2_end = 0
    for _ in range(tiles):
        feature_end += feature_tile_cycles
        control_end = max(control_end, feature_end) + control_tile_cycles
        s2_end = max(s2_end, control_end) + s2_tile_cycles

    # The S3 engine is deliberately phase-separated: feature preparation owns
    # it until the final feature tile completes, then selected-head work owns
    # it.  This avoids claiming a second, unmodeled S3 execution engine.
    head_start = max(feature_end, s2_end)
    head_end = head_start + tiles * head_tile_cycles
    scheduled_cycles = head_end
    target_cycles = dense_cycles / float(target_saes_speedup)
    serial_cycles = s2_cycles + feature_cycles + head_cycles + control_cycles
    overlap_cycles = max(0, serial_cycles - scheduled_cycles)

    return {
        "kind": DUAL_STREAM_KIND,
        "measured": False,
        "timing_class": "resource-bound-architectural-cycle-model",
        "quality_or_wall_clock_claim": False,
        "architecture": {
            "name": "depthsplat-coefficient-compatible-dual-stream-v1",
            "same_weights_as_source_route": True,
            "s2_compute": "existing 48x48 MMCU lane",
            "s3_compute": "independent 48x48 MMCU lane",
            "s3_weight_storage": "independent read bank with copied source weights",
            "s3_feature_storage": "independent scratchpad for feature-upsample stream",
            "tile_handoff": "existing dual-port TileBuffer S2-write/S3-read contract",
            "current_single_mmcu_rtl_compatible": False,
        },
        "source_dependency_contract": {
            "s3_feature_inputs_precede_final_s2_depth": True,
            "saes_decision_waits_for_local_s3_feature_tile": True,
            "s2_probe_tile_waits_for_local_saes_decision": True,
            "selected_s3_head_waits_for_s2_depth_and_s3_feature_completion": True,
            "feature_and_s2_work_are_tiled_in_raster_order": True,
        },
        "resources": {
            "s2_mmcu_lanes": 1,
            "s3_mmcu_lanes": 1,
            "saes_vector_control_lanes": 1,
            "s2_weight_read_banks": 1,
            "s3_weight_read_banks": 1,
            "s2_feature_scratchpads": 1,
            "s3_feature_scratchpads": 1,
            "tile_buffer_ports": {"s2_write": 1, "s3_read": 1},
        },
        "tile_schedule": {
            "tile_count": tiles,
            "feature_tile_cycles": feature_tile_cycles,
            "control_tile_cycles": control_tile_cycles,
            "s2_tile_cycles": s2_tile_cycles,
            "head_tile_cycles": head_tile_cycles,
            "s3_feature_complete_cycle": feature_end,
            "saes_control_complete_cycle": control_end,
            "s2_complete_cycle": s2_end,
            "s3_head_start_cycle": head_start,
            "s3_head_complete_cycle": head_end,
        },
        "latency": {
            "dense_baseline_cycles": dense_cycles,
            "no_overlap_sparse_cycles": serial_cycles,
            "dual_stream_sparse_cycles": scheduled_cycles,
            "realized_s2_s3_overlap_cycles": overlap_cycles,
            "dual_stream_speedup": dense_cycles / scheduled_cycles,
        },
        "target_speedup_feasibility": {
            "target_speedup": float(target_saes_speedup),
            "target_cycles": target_cycles,
            "target_reached": scheduled_cycles <= target_cycles,
            "cycle_margin_to_target": target_cycles - scheduled_cycles,
            "speedup_margin_to_target": dense_cycles / scheduled_cycles
            - float(target_saes_speedup),
        },
    }


def project_depthsplat_sparse_datapath_cycles(
    *,
    dense_cycle_breakdown: Mapping[str, Any],
    planned_s2_schedule: Mapping[str, Any],
    selected_output_head_schedule: Mapping[str, Any],
    saes_control_and_materialization_cycles: int,
    target_saes_speedup: float = 1.26,
) -> dict[str, Any]:
    """Project a source-bound probe-first S2/S3 schedule.

    The projected S2 branch evaluates only planned probe positions. The S3
    head branch follows the exact selected-output convolution closure already
    used by the route. Feature upsampling has no final-depth dependency in the
    pinned encoder source, so it forms an overlap window with S2. The function
    returns both no-overlap and perfect-overlap bounds rather than selecting an
    unmeasured overlap fraction.
    """

    if not isinstance(dense_cycle_breakdown, Mapping):
        raise TypeError("dense cycle breakdown must be a mapping")
    if not isinstance(planned_s2_schedule, Mapping):
        raise TypeError("planned S2 schedule must be a mapping")
    if not isinstance(selected_output_head_schedule, Mapping):
        raise TypeError("selected-head schedule must be a mapping")
    control_cycles = _nonnegative_int(
        saes_control_and_materialization_cycles,
        label="SAES control and materialization cycles",
    )
    stages = {
        name: _nonnegative_int(value, label=f"dense {name} cycles")
        for name, value in dense_cycle_breakdown.items()
        if name != "total"
    }
    required = set(S2_STAGE_NAMES) | {
        S3_FEATURE_PREPARATION_STAGE,
        S3_HEAD_STAGE,
    }
    missing = sorted(required.difference(stages))
    if missing:
        raise ValueError(f"dense cycle breakdown lacks {', '.join(missing)}")
    dense_total = sum(stages.values())
    if dense_total <= 0:
        raise ValueError("dense cycle breakdown has no work")

    s2_fraction = _ratio(
        _nonnegative_int(
            planned_s2_schedule.get("scheduled_s2_positions"),
            label="planned S2 positions",
        ),
        _nonnegative_int(
            planned_s2_schedule.get("dense_s2_positions"),
            label="dense S2 positions",
        ),
        label="S2",
    )
    s3_head_fraction = _ratio(
        _nonnegative_int(
            selected_output_head_schedule.get("scheduled_head_macs"),
            label="planned S3 head MACs",
        ),
        _nonnegative_int(
            selected_output_head_schedule.get("dense_head_macs"),
            label="dense S3 head MACs",
        ),
        label="S3 head",
    )

    projected_stages = dict(stages)
    for name in S2_STAGE_NAMES:
        projected_stages[name] = _scaled_cycles(stages[name], s2_fraction)
    projected_stages[S3_HEAD_STAGE] = _scaled_cycles(
        stages[S3_HEAD_STAGE], s3_head_fraction
    )

    projected_s2_cycles = sum(projected_stages[name] for name in S2_STAGE_NAMES)
    projected_s3_feature_cycles = projected_stages[S3_FEATURE_PREPARATION_STAGE]
    projected_s3_head_cycles = projected_stages[S3_HEAD_STAGE]
    fixed_cycles = sum(
        value
        for name, value in projected_stages.items()
        if name
        not in {*S2_STAGE_NAMES, S3_FEATURE_PREPARATION_STAGE, S3_HEAD_STAGE}
    )
    serial_projected_cycles = (
        fixed_cycles
        + projected_s2_cycles
        + projected_s3_feature_cycles
        + projected_s3_head_cycles
        + control_cycles
    )
    overlap_window_cycles = min(projected_s2_cycles, projected_s3_feature_cycles)
    perfect_overlap_cycles = serial_projected_cycles - overlap_window_cycles
    if perfect_overlap_cycles <= 0:
        raise RuntimeError("DepthSplat projected streaming latency is invalid")
    if (
        isinstance(target_saes_speedup, bool)
        or not isinstance(target_saes_speedup, (int, float))
        or not math.isfinite(float(target_saes_speedup))
        or float(target_saes_speedup) <= 0.0
    ):
        raise ValueError("target SAES speedup is invalid")
    target_cycles = dense_total / float(target_saes_speedup)
    required_overlap_cycles = max(0.0, serial_projected_cycles - target_cycles)
    required_overlap_fraction = (
        required_overlap_cycles / overlap_window_cycles
        if overlap_window_cycles
        else None
    )
    target_within_projection_bound = bool(
        target_cycles >= perfect_overlap_cycles
        and target_cycles <= serial_projected_cycles
    )

    return {
        "kind": KIND,
        "measured": False,
        "projection_scope": "same-weight-source-route-s2-s3-only",
        "quality_or_wall_clock_claim": False,
        "assumptions": {
            "s2_probe_positions_execute_before_nonprobe_positions": True,
            "s2_nonprobe_positions_are_not_issued_for_l0_l1_tiles": True,
            "s3_head_uses_selected_output_convolution_closure": True,
            "s3_feature_preparation_can_stream_after_s1_before_final_s2_depth": True,
            "s2_s3_feature_preparation_overlap_is_reported_as_a_bound": True,
            "s4_ggu_and_writeback_cycles_modeled": False,
        },
        "fractions": {
            "s2_executed_fraction": s2_fraction,
            "s2_planned_saving_rate": 1.0 - s2_fraction,
            "s3_head_executed_fraction": s3_head_fraction,
            "s3_head_planned_saving_rate": 1.0 - s3_head_fraction,
        },
        "dense_total_cycles": dense_total,
        "dense_stage_cycles": stages,
        "projected_stage_cycles": projected_stages,
        "saes_control_and_materialization_cycles": control_cycles,
        "streaming_schedule": {
            "fixed_cycles": fixed_cycles,
            "s2_probe_pipeline_cycles": projected_s2_cycles,
            "s3_feature_preparation_cycles": projected_s3_feature_cycles,
            "s3_selected_head_cycles": projected_s3_head_cycles,
            "overlap_window_cycles": overlap_window_cycles,
        },
        "latency_bounds": {
            "serial_projected_cycles": serial_projected_cycles,
            "perfect_s2_s3_feature_overlap_cycles": perfect_overlap_cycles,
            "serial_projected_speedup": dense_total / serial_projected_cycles,
            "perfect_overlap_speedup_upper_bound": dense_total
            / perfect_overlap_cycles,
        },
        "target_speedup_feasibility": {
            "target_speedup": float(target_saes_speedup),
            "target_cycles": target_cycles,
            "required_overlap_cycles": required_overlap_cycles,
            "required_overlap_fraction_of_window": required_overlap_fraction,
            "target_within_projection_bound": target_within_projection_bound,
        },
    }


__all__ = ("KIND", "project_depthsplat_sparse_datapath_cycles")
