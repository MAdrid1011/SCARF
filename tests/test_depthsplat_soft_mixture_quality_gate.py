"""Focused identity checks for the direct DL3DV quality gate."""

from __future__ import annotations

import pytest
import torch

from scripts.saes_depthsplat_soft_mixture_sample0_quality_gate import (
    _collect_source_fsdr_trace,
    _candidate_work_guided_rate,
    _diagnostic_label,
    _fsdr_feature_buffer_transfer_cycles,
    _fsdr_scale_cache_sizes,
    _executed_saes_s2_s3_work,
    _psnr_only_quality_verdict,
    _require_quality_context_identity,
    _saes_fused_s2_schedule,
    _threshold_route_candidates,
    _validate_direct_conditional_plan,
    build_parser,
    direct_conditional_t4_profile,
)


class _Execution:
    def __init__(self) -> None:
        self.routing_features = torch.ones((1, 2, 3, 4, 4), dtype=torch.float32)
        self.routing_z_depths = torch.ones((1, 2, 4, 4), dtype=torch.float32)


def _exact_inputs() -> dict[str, object]:
    features = torch.ones((1, 2, 3, 4, 4), dtype=torch.float32)
    probabilities = torch.ones((1, 2, 8, 4, 4), dtype=torch.float32)
    return {
        "feature_source": "depthsplat-mono-dinov2-v1",
        "scales": (
            {
                "features": features,
                "probabilities": probabilities,
                "candidates": torch.linspace(1.0, 2.0, 8)
                .reshape(1, 1, 8, 1, 1)
                .expand(1, 2, 8, 4, 4),
            },
        ),
    }


def _identity(sample_index: int) -> dict[str, object]:
    digest = "a" * 64
    return {
        "source_sample_index": sample_index,
        "scene": "scene-1",
        "context_indices": [0, 9],
        "tree_sha256": digest,
        "manifest_sha256": digest,
        "audit_input_sha256": digest,
        "source_binding": {
            "canonical_index_sha256": digest,
            "canonical_sample_selection_sha256": digest,
            "canonical_selection_sha256": digest,
            "source_audit_input_sha256": digest,
            "source_audit_tree_sha256": digest,
            "source_sidecar_tree_sha256": digest,
        },
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
    }


def test_quality_gate_accepts_declared_nonzero_context_sample() -> None:
    identity = _identity(1)

    assert _require_quality_context_identity(identity, source_sample_index=1) == identity


def test_quality_gate_rejects_sample_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="fixed target-free context"):
        _require_quality_context_identity(_identity(1), source_sample_index=2)


def test_user_authorized_psnr_only_gate_keeps_secondary_metric_deltas_visible() -> None:
    verdict = _psnr_only_quality_verdict(
        {"psnr_db": 35.0, "ssim": 0.97, "lpips": 0.03},
        {"psnr_db": 34.51, "ssim": 0.95, "lpips": 0.08},
    )

    assert verdict["quality_gate"] == "user-authorized-psnr-only-v1"
    assert verdict["pass"] is True
    assert verdict["observed"]["psnr_loss_db"] == pytest.approx(0.49)
    assert verdict["secondary_metrics_reported_not_gated"] == {
        "ssim_loss": pytest.approx(0.02),
        "lpips_increase": pytest.approx(0.05),
    }


def test_direct_conditional_profile_binds_the_normalized_route() -> None:
    from saes.probe_first_schedule import (
        build_depthsplat_soft_mixture_normalized_t4_probe_first_plan,
    )

    features = torch.zeros((1, 1, 2, 4, 4), dtype=torch.float32)
    depths = torch.full((1, 1, 4, 4), 2.0, dtype=torch.float32)
    plan = build_depthsplat_soft_mixture_normalized_t4_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
    )

    assert len(_validate_direct_conditional_plan(plan, profile=direct_conditional_t4_profile())) == 64


def test_threshold_sweep_can_hold_one_isolated_route_constant() -> None:
    assert _threshold_route_candidates(
        (2.0, 2.25, 2.5), ("l0_to_l1",)
    ) == ((2.0, "l0_to_l1"), (2.25, "l0_to_l1"), (2.5, "l0_to_l1"))


def test_threshold_sweep_rejects_two_dimensional_isolated_scan() -> None:
    with pytest.raises(ValueError, match="requires one route mode"):
        _threshold_route_candidates((2.0, 2.25), ("l0_only", "l1_only"))


def test_multi_threshold_diagnostic_labels_are_distinct() -> None:
    assert _diagnostic_label(
        threshold=2.25,
        route_isolation="l0_to_l1",
        candidate_count=3,
        attribute=None,
    ) == "l0_to_l1_threshold_2_25"


def test_source_fsdr_trace_uses_only_captured_context_inputs() -> None:
    trace = _collect_source_fsdr_trace(_Execution())

    assert trace["paper_result_eligible"] is False
    assert trace["target_rgb_accessed"] is False
    assert trace["discrete_candidate_evidence"] is False
    assert trace["summary"]["total_pixels"] == 32
    assert trace["summary"]["full_depth_evaluations"] == 32 * 128
    assert (
        trace["summary"]["executed_depth_evaluations"]
        <= trace["summary"]["full_depth_evaluations"]
    )


def test_source_fsdr_trace_uses_exact_candidate_evidence_when_available() -> None:
    trace = _collect_source_fsdr_trace(_Execution(), exact_inputs=_exact_inputs())
    summary = trace["summary"]

    assert trace["mode"] == "source-exact-multiscale-cost-volume-candidates-v2"
    assert trace["discrete_candidate_evidence"] is True
    assert trace["feature_source"] == "depthsplat-mono-dinov2-v1"
    assert trace["parameters"]["guidance_policy"] == "paper-hamming-local-validity"
    assert trace["parameters"]["per_scale_cache_sizes"] == [32]
    assert summary["discrete_candidate_evidence"] is True
    assert summary["full_depth_evaluations"] == 32 * 8
    assert summary["guided_rate"] == pytest.approx(
        summary["guided"] / summary["total_pixels"]
    )
    assert summary["candidate_work_equivalent_guided_rate"] == pytest.approx(
        (1.0 - summary["executed_depth_evaluations"] / summary["full_depth_evaluations"])
        / 0.75
    )


def test_fsdr_splits_the_fixed_cam_budget_across_native_scales() -> None:
    assert _fsdr_scale_cache_sizes(1) == (32,)
    assert _fsdr_scale_cache_sizes(2) == (16, 16)
    assert _fsdr_scale_cache_sizes(3) == (11, 11, 10)

    with pytest.raises(ValueError, match="scale count"):
        _fsdr_scale_cache_sizes(0)


def test_fsdr_candidate_work_equivalent_guided_rate() -> None:
    assert _candidate_work_guided_rate(
        full_evaluations=160, executed_evaluations=100
    ) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="candidate-work"):
        _candidate_work_guided_rate(full_evaluations=0, executed_evaluations=0)


def test_source_fsdr_cli_does_not_require_a_saes_threshold() -> None:
    args = build_parser().parse_args(
        ["--source-fsdr-only", "--output-dir", "output"]
    )

    assert args.source_fsdr_only is True
    assert args.kernel_risk_thresholds is None


def test_fsdr_feature_buffer_transfer_cycles_follow_dual_port_fp16_rtl() -> None:
    assert _fsdr_feature_buffer_transfer_cycles(
        byte_count=1_409_286_144, label="baseline"
    ) == 352_321_536
    assert _fsdr_feature_buffer_transfer_cycles(
        byte_count=592_822_272, label="actual"
    ) == 148_205_568
    assert _fsdr_feature_buffer_transfer_cycles(byte_count=5, label="rounding") == 2


def test_fsdr_feature_buffer_transfer_cycles_reject_negative_bytes() -> None:
    with pytest.raises(RuntimeError, match="invalid"):
        _fsdr_feature_buffer_transfer_cycles(byte_count=-1, label="bytes")


def test_saes_fused_s2_schedule_counts_only_route_required_positions() -> None:
    schedule = _saes_fused_s2_schedule(
        l0_tiles=1,
        l1_tiles=1,
        full_tiles=1,
        primitives_per_pixel=1,
        tile_size=4,
    )

    assert schedule == {
        "dense_s2_positions": 48,
        "scheduled_s2_positions": 32,
        "skipped_s2_positions": 16,
        "s2_position_saving_rate": pytest.approx(1.0 / 3.0),
    }


def test_dense_capture_cannot_claim_route_planned_s2_s3_savings() -> None:
    planned = _saes_fused_s2_schedule(
        l0_tiles=1,
        l1_tiles=1,
        full_tiles=1,
        primitives_per_pixel=1,
        tile_size=4,
    )
    executed = _executed_saes_s2_s3_work(
        execution_events={"whole_pipeline_s2_s3_sparse_execution_verified": False},
        planned_s2_schedule=planned,
        selected_output_head_schedule={
            "dense_head_macs": 300,
            "scheduled_head_macs": 200,
        },
    )

    assert executed["execution_verified"] is False
    assert executed["executed_s2_positions"] == 48
    assert executed["executed_head_macs"] == 300
    assert executed["s2_position_saving_rate"] == 0.0
    assert executed["head_mac_saving_rate"] == 0.0


def test_verified_sparse_execution_requires_bound_noninflated_counters() -> None:
    planned = _saes_fused_s2_schedule(
        l0_tiles=1,
        l1_tiles=1,
        full_tiles=1,
        primitives_per_pixel=1,
        tile_size=4,
    )
    baseline = _executed_saes_s2_s3_work(
        execution_events={"whole_pipeline_s2_s3_sparse_execution_verified": False},
        planned_s2_schedule=planned,
        selected_output_head_schedule={
            "dense_head_macs": 300,
            "scheduled_head_macs": 200,
        },
    )
    verified = _executed_saes_s2_s3_work(
        execution_events={
            "whole_pipeline_s2_s3_sparse_execution_verified": True,
            "sparse_s2_s3_execution": {
                "contract_version": "depthsplat-probe-first-s2-s3-execution-v1",
                "planned_schedule_sha256": baseline["planned_schedule_sha256"],
                "executed_s2_positions": 32,
                "executed_head_macs": 200,
            },
        },
        planned_s2_schedule=planned,
        selected_output_head_schedule={
            "dense_head_macs": 300,
            "scheduled_head_macs": 200,
        },
    )
    assert verified["execution_verified"] is True
    assert verified["s2_position_saving_rate"] == pytest.approx(1.0 / 3.0)
    assert verified["head_mac_saving_rate"] == pytest.approx(1.0 / 3.0)
    with pytest.raises(RuntimeError, match="binding changed"):
        _executed_saes_s2_s3_work(
            execution_events={
                "whole_pipeline_s2_s3_sparse_execution_verified": True,
                "sparse_s2_s3_execution": {
                    "contract_version": "depthsplat-probe-first-s2-s3-execution-v1",
                    "planned_schedule_sha256": "0" * 64,
                    "executed_s2_positions": 32,
                    "executed_head_macs": 200,
                },
            },
            planned_s2_schedule=planned,
            selected_output_head_schedule={
                "dense_head_macs": 300,
                "scheduled_head_macs": 200,
            },
        )
    assert baseline["planned_schedule_sha256"] != "0" * 64
