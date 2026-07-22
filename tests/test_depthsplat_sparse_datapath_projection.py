from __future__ import annotations

import pytest

from saes.depthsplat_sparse_datapath_projection import (
    DUAL_STREAM_KIND,
    KIND,
    project_depthsplat_sparse_datapath_cycles,
    simulate_depthsplat_dual_stream_schedule,
)


def _breakdown() -> dict[str, int]:
    return {
        "cost_volume": 100,
        "unet_refinement": 100,
        "depth_refinement": 100,
        "depth_head": 100,
        "softmax_regression": 100,
        "upsampling": 200,
        "gaussian_head": 100,
        "feature_extraction": 50,
        "total": 850,
    }


def test_sparse_datapath_projection_reports_serial_and_overlap_bounds() -> None:
    result = project_depthsplat_sparse_datapath_cycles(
        dense_cycle_breakdown=_breakdown(),
        planned_s2_schedule={"dense_s2_positions": 100, "scheduled_s2_positions": 40},
        selected_output_head_schedule={"dense_head_macs": 100, "scheduled_head_macs": 25},
        saes_control_and_materialization_cycles=10,
    )

    assert result["kind"] == KIND
    assert result["measured"] is False
    assert result["quality_or_wall_clock_claim"] is False
    assert result["projected_stage_cycles"]["cost_volume"] == 40
    assert result["projected_stage_cycles"]["gaussian_head"] == 25
    assert result["latency_bounds"] == {
        "serial_projected_cycles": 485,
        "perfect_s2_s3_feature_overlap_cycles": 285,
        "serial_projected_speedup": pytest.approx(850 / 485),
        "perfect_overlap_speedup_upper_bound": pytest.approx(850 / 285),
    }


def test_sparse_datapath_projection_rejects_work_below_route_requirement() -> None:
    with pytest.raises(ValueError, match="S2 ratio"):
        project_depthsplat_sparse_datapath_cycles(
            dense_cycle_breakdown=_breakdown(),
            planned_s2_schedule={"dense_s2_positions": 100, "scheduled_s2_positions": 0},
            selected_output_head_schedule={"dense_head_macs": 100, "scheduled_head_macs": 25},
            saes_control_and_materialization_cycles=0,
        )


def test_dual_stream_schedule_models_tile_dependencies_and_resources() -> None:
    result = simulate_depthsplat_dual_stream_schedule(
        dense_total_cycles=850,
        s2_probe_pipeline_cycles=200,
        s3_feature_preparation_cycles=200,
        s3_selected_head_cycles=25,
        saes_control_and_materialization_cycles=10,
        tile_count=10,
    )

    assert result["kind"] == DUAL_STREAM_KIND
    assert result["measured"] is False
    assert result["architecture"]["same_weights_as_source_route"] is True
    assert result["architecture"]["current_single_mmcu_rtl_compatible"] is False
    assert result["resources"]["s2_mmcu_lanes"] == 1
    assert result["resources"]["s3_mmcu_lanes"] == 1
    assert result["tile_schedule"]["s3_feature_complete_cycle"] == 200
    assert result["tile_schedule"]["s2_complete_cycle"] == 221
    assert result["tile_schedule"]["s3_head_start_cycle"] == 221
    assert result["latency"] == {
        "dense_baseline_cycles": 850,
        "no_overlap_sparse_cycles": 435,
        "dual_stream_sparse_cycles": 251,
        "realized_s2_s3_overlap_cycles": 184,
        "dual_stream_speedup": pytest.approx(850 / 251),
    }
    assert result["target_speedup_feasibility"]["target_reached"] is True


def test_dual_stream_schedule_rejects_invalid_topology_inputs() -> None:
    with pytest.raises(ValueError, match="tile count must be positive"):
        simulate_depthsplat_dual_stream_schedule(
            dense_total_cycles=850,
            s2_probe_pipeline_cycles=200,
            s3_feature_preparation_cycles=200,
            s3_selected_head_cycles=25,
            saes_control_and_materialization_cycles=10,
            tile_count=0,
        )
