import subprocess
import sys

import pytest


def _mixed_stats():
    return {
        "total_tiles_processed": 3,
        "level0_tiles": 1,
        "level1_tiles": 1,
        "full_tiles": 1,
        "level0_pixels": 12,
        "level1_pixels": 4,
        "l0_representatives": 4,
        "l1_lightweight_anchors": 12,
        "full_stage3_gaussians": 16,
    }


def _joint_calibrator_contract(*, model: str = "transplat"):
    return {
        "schema_version": "saes-joint-materialization-calibrator-v1",
        "asset_sha256": "a" * 64,
        "model": model,
        "calls": 16,
        "l0_calls": 4,
        "l1_calls": 12,
        "full_calls": 0,
        "selected_descriptor_reads": 16,
        "network_macs_per_call": 576,
        "weight_bytes": 1248,
        "selected_head": {
            "model": model,
            "contract_version": "saes-selected-output-replay-v1",
            # 16 calls at degree 2 cost 11,008 MACs including transforms, so
            # this explicitly leaves a valid <=5% selected-head budget.
            "dense_head_macs": 300_000,
            "replayed_head_macs": 0,
        },
    }


def test_event_ledger_charges_every_executed_l0_l1_operation():
    from saes.hardware_accounting import build_saes_event_ledger

    ledger = build_saes_event_ledger(
        _mixed_stats(), feature_dim=128, tile_size=4, sh_degree=4
    )

    assert ledger["timing_class"] == "analytic_no_overlap_not_rtl_cycle_equivalent"
    assert ledger["target_rgb_accessed"] is False
    assert ledger["events"] == {
        "total_tiles": 3,
        "l0_tiles": 1,
        "l1_tiles": 1,
        "full_tiles": 1,
        "l0_retained_anchors": 4,
        "l1_retained_anchors": 12,
        "full_stage3_gaussians": 16,
        "l0_nonanchors": 12,
        "l1_nonanchors": 4,
        "total_nonanchors": 16,
    }
    # T=4, K=4, L1=12, C=128 and W=64 make these fixed event counts
    # independently checkable without model outputs or target images.
    assert ledger["cycles"]["decision_total"] == 92
    assert ledger["cycles"]["assignment_total"] == 592
    assert ledger["cycles"]["moment_matching_total"] == 224
    assert ledger["cycles"]["storage_transfer"] == 377
    assert ledger["cycles"]["serialized_accounting_cycles"] == 1285
    assert ledger["traffic_bytes"]["charged_storage_total"] == 6019
    assert ledger["assumptions"]["rtl_cycle_equivalent"] is False


def test_event_ledger_charges_registered_probe_cross_check_control_work():
    from saes.hardware_accounting import build_saes_event_ledger

    stats = {
        "total_tiles_processed": 1,
        "level0_tiles": 0,
        "level1_tiles": 0,
        "full_tiles": 1,
        "materialization_guard_enabled": True,
        "l0_guard_checks": 1,
        "l1_guard_checks": 0,
        "l0_guard_rejections": 1,
        "l1_guard_rejections": 0,
        "guard_anchor_attribute_reads": 3 * 4,
        "guard_nonprobe_s3_attribute_reads": 0,
        "probe_cross_check_enabled": True,
        "probe_cross_check_l0_checks": 1,
        "probe_cross_check_l1_checks": 0,
        "probe_cross_check_l0_rejections": 1,
        "probe_cross_check_l1_rejections": 0,
    }
    ledger = build_saes_event_ledger(
        stats, feature_dim=128, tile_size=4, sh_degree=2
    )

    assert ledger["events"]["probe_cross_check_l0_checks"] == 1
    assert ledger["events"]["probe_cross_check_l0_rejections"] == 1
    assert ledger["cycles"]["probe_cross_check"] > 0
    assert "leave-one-out" in ledger["assumptions"]["probe_cross_check"]


def test_event_ledger_charges_adapter_offset_attribute_reconstruction():
    from saes.hardware_accounting import build_saes_event_ledger

    baseline = build_saes_event_ledger(
        _mixed_stats(), feature_dim=128, tile_size=4, sh_degree=2
    )
    stats = _mixed_stats()
    stats["adapter_offset_attribute_transport_uses"] = 12 * 4 + 4 * 12
    transport = build_saes_event_ledger(
        stats, feature_dim=128, tile_size=4, sh_degree=2
    )

    assert transport["events"]["adapter_offset_attribute_transport_pairs"] == 96
    assert transport["cycles"]["adapter_offset_attribute_reconstruction"] == 96
    assert transport["traffic_bytes"]["adapter_offset_attribute_transport_read"] == (
        96 * (54 + 2)
    )
    assert (
        transport["traffic_bytes"]["charged_storage_total"]
        - baseline["traffic_bytes"]["charged_storage_total"]
        == 96 * (54 + 2)
    )
    assert (
        transport["cycles"]["serialized_accounting_cycles"]
        - baseline["cycles"]["serialized_accounting_cycles"]
        == 96 + (96 * (54 + 2)) // 16
    )


def test_event_ledger_charges_joint_calibrator_and_requires_a_bound_head_trace():
    from saes.hardware_accounting import build_saes_event_ledger

    baseline = build_saes_event_ledger(
        _mixed_stats(), feature_dim=128, tile_size=4, sh_degree=2
    )
    stats = _mixed_stats()
    stats["joint_calibrator"] = _joint_calibrator_contract()
    ledger = build_saes_event_ledger(
        stats, feature_dim=128, tile_size=4, sh_degree=2
    )

    assert ledger["events"]["joint_calibrator_calls"] == 16
    assert ledger["events"]["joint_calibrator_l0_calls"] == 4
    assert ledger["events"]["joint_calibrator_l1_calls"] == 12
    assert ledger["events"]["joint_calibrator_full_calls"] == 0
    assert ledger["cycles"]["joint_calibrator"] == 172
    assert ledger["traffic_bytes"]["joint_calibrator_selected_descriptor_read"] == 16 * 92
    assert ledger["traffic_bytes"]["joint_calibrator_activation"] == 16 * (32 + 40) * 2
    assert ledger["traffic_bytes"]["joint_calibrator_weight"] == 1248
    assert ledger["traffic_bytes"]["charged_storage_total"] - baseline["traffic_bytes"]["charged_storage_total"] == 5024
    assert ledger["cycles"]["serialized_accounting_cycles"] > baseline["cycles"]["serialized_accounting_cycles"]


def test_event_ledger_rejects_joint_calibrator_without_full_budget_and_full_invariance():
    from saes.hardware_accounting import build_saes_event_ledger

    missing_head = _mixed_stats()
    missing_head["joint_calibrator"] = _joint_calibrator_contract()
    del missing_head["joint_calibrator"]["selected_head"]
    with pytest.raises(ValueError, match="selected-head event contract"):
        build_saes_event_ledger(
            missing_head, feature_dim=128, tile_size=4, sh_degree=2
        )

    full_call = _mixed_stats()
    full_call["joint_calibrator"] = _joint_calibrator_contract()
    full_call["joint_calibrator"]["full_calls"] = 1
    with pytest.raises(ValueError, match="Full slots"):
        build_saes_event_ledger(
            full_call, feature_dim=128, tile_size=4, sh_degree=2
        )

    over_budget = _mixed_stats()
    over_budget["joint_calibrator"] = _joint_calibrator_contract()
    over_budget["joint_calibrator"]["selected_head"]["dense_head_macs"] = 10_000
    with pytest.raises(ValueError, match="5%"):
        build_saes_event_ledger(
            over_budget, feature_dim=128, tile_size=4, sh_degree=2
        )

    depthsplat_contract = _mixed_stats()
    depthsplat_contract["joint_calibrator"] = _joint_calibrator_contract(
        model="depthsplat"
    )
    depthsplat_ledger = build_saes_event_ledger(
        depthsplat_contract, feature_dim=128, tile_size=4, sh_degree=2
    )
    assert depthsplat_ledger["events"]["joint_calibrator_calls"] == 16


def test_event_ledger_charges_assignment_consensus_virtual_outputs():
    from saes.hardware_accounting import build_saes_event_ledger

    baseline = build_saes_event_ledger(
        _mixed_stats(), feature_dim=128, tile_size=4, sh_degree=2
    )
    stats = _mixed_stats()
    stats.update(
        {
            "assignment_consensus_pseudo_outputs": 12 + 4,
            "assignment_consensus_anchor_pairs": 12 * 4 + 4 * 12,
            "assignment_consensus_offset_recoveries": 4 + 12,
            "assignment_consensus_target_lifts": (12 + 4) + (12 * 4 + 4 * 12),
        }
    )
    consensus = build_saes_event_ledger(
        stats, feature_dim=128, tile_size=4, sh_degree=2
    )

    assert consensus["events"]["assignment_consensus_pseudo_outputs"] == 16
    assert consensus["events"]["assignment_consensus_anchor_pairs"] == 96
    assert consensus["cycles"]["moment_matching_total"] == 0
    assert consensus["cycles"]["assignment_consensus_total"] > 0
    assert consensus["traffic_bytes"]["retained_descriptor_write"] == 0
    assert consensus["traffic_bytes"]["assignment_consensus_depth_read"] == 16 * 2
    assert consensus["traffic_bytes"]["assignment_consensus_virtual_output_write"] == (
        16 * consensus["inputs"]["descriptor_bytes"]
    )
    assert (
        consensus["cycles"]["serialized_accounting_cycles"]
        > baseline["cycles"]["serialized_accounting_cycles"]
    )


def test_event_ledger_rejects_attribute_transport_on_consensus_fallback():
    from saes.hardware_accounting import build_saes_event_ledger

    stats = _mixed_stats()
    stats["adapter_offset_attribute_transport_uses"] = 12 * 4 + 4 * 12
    stats["assignment_consensus_fallback_tiles"] = 1
    stats["assignment_consensus_l0_fallback_tiles"] = 1

    with pytest.raises(ValueError, match="must not reuse attribute transport"):
        build_saes_event_ledger(
            stats, feature_dim=128, tile_size=4, sh_degree=2
        )


def test_event_ledger_rejects_missing_consensus_success_counters():
    from saes.hardware_accounting import build_saes_event_ledger

    stats = _mixed_stats()
    stats["assignment_consensus_fallback_tiles"] = 1
    stats["assignment_consensus_l0_fallback_tiles"] = 1

    with pytest.raises(ValueError, match="pseudo_outputs is inconsistent"):
        build_saes_event_ledger(
            stats, feature_dim=128, tile_size=4, sh_degree=2
        )


def test_event_ledger_rejects_anchor_counts_that_do_not_match_the_route():
    from saes.hardware_accounting import build_saes_event_ledger

    stats = _mixed_stats()
    stats["l1_lightweight_anchors"] = 4
    with pytest.raises(ValueError, match="l1_lightweight_anchors"):
        build_saes_event_ledger(
            stats, feature_dim=128, tile_size=4, sh_degree=4
        )


def test_event_ledger_counts_full_path_without_sparse_merge_work():
    from saes.hardware_accounting import build_saes_event_ledger

    ledger = build_saes_event_ledger(
        {
            "total_tiles_processed": 1,
            "level0_tiles": 0,
            "level1_tiles": 0,
            "full_tiles": 1,
            "full_stage3_gaussians": 16,
        },
        feature_dim=128,
        tile_size=4,
        sh_degree=4,
    )

    assert ledger["events"]["total_nonanchors"] == 0
    assert ledger["cycles"]["assignment_total"] == 0
    assert ledger["cycles"]["moment_matching_total"] == 0
    assert ledger["cycles"]["controller_total"] == 3
    assert ledger["traffic_bytes"]["retained_descriptor_write"] == 0


def test_retained_descriptor_storage_layout_matches_the_rtl_128bit_contract():
    from saes.hardware_accounting import retained_descriptor_storage_layout

    degree2 = retained_descriptor_storage_layout(2)
    degree4 = retained_descriptor_storage_layout(4)

    assert degree2 == {
        "descriptor_layout_bytes": {
            "mean_fp32": 12,
            "covariance_upper_fp32": 24,
            "harmonics_fp16": 54,
            "opacity_fp16": 2,
        },
        "descriptor_bytes": 92,
        "storage_beat_bytes": 16,
        "storage_beats": 6,
    }
    assert degree4["descriptor_bytes"] == 188
    assert degree4["storage_beats"] == 12


def test_ablation_refuses_zero_cost_saes_and_charges_the_event_ledger():
    import scripts.demo as demo

    previous_config = demo.CONFIG
    demo.CONFIG = demo.SCARFConfig()
    try:
        tracker = demo.SavingsTracker()
        tracker.record_saes(total_pixels=48, saes_stats=_mixed_stats())
        with pytest.raises(ValueError, match="event ledger"):
            tracker.compute_ablation(
                feature_cycles=10_000,
                depth_cycles=10_000,
                ggu_cycles=1_000,
                dp_core_cycles=10_000,
                gauss_gen_cycles=10_000,
                cost_volume_cycles=3_000,
                s1_cnn_cycles=5_000,
            )

        tracker.record_saes(
            total_pixels=48,
            saes_stats=_mixed_stats(),
            feature_dim=128,
            tile_size=4,
            sh_degree=4,
        )
        ablation = tracker.compute_ablation(
            feature_cycles=10_000,
            depth_cycles=10_000,
            ggu_cycles=1_000,
            dp_core_cycles=10_000,
            gauss_gen_cycles=10_000,
            cost_volume_cycles=3_000,
            s1_cnn_cycles=5_000,
        )
        assert ablation["asic"]["saes_control"] == 0
        assert ablation["asic_fsdr"]["saes_control"] == 0
        assert ablation["asic_saes"]["saes_control"] == 1285
        assert ablation["asic_fsdr_saes"]["saes_control"] == 1285
        assert (
            ablation["_pipeline"]["saes_control_timing_class"]
            == "analytic_no_overlap_not_rtl_cycle_equivalent"
        )
        assert ablation["_pipeline"]["saes_total_ratio"] == pytest.approx(16 / 48)
        assert ablation["_pipeline"]["saes_s2_saving"] == 0.0
        assert ablation["_pipeline"]["saes_s3_saving"] == 0.0
        assert (
            ablation["_pipeline"]["saes_execution_dependency"]["status"]
            == "model_identity_missing"
        )
        assert ablation["asic_saes"]["dp_core"] == ablation["asic"]["dp_core"]
        assert ablation["asic_saes"]["gauss_gen"] == ablation["asic"]["gauss_gen"]
    finally:
        demo.CONFIG = previous_config


def test_model_execution_dependency_contracts_split_dense_and_selected_head_boundaries():
    from saes.execution_dependency import resolve_s2_s3_execution_contract

    for model in ("transplat", "mvsplat", "depthsplat"):
        contract = resolve_s2_s3_execution_contract(model)
        assert contract["s2_s3_sparse_execution_verified"] is False
        assert contract["status"].startswith("stage_specific_")
        assert set(contract["stages"]) == {
            "dense_shared_trunk",
            "s2_candidate_search",
            "selected_output_head",
            "s3_retained_descriptor",
            "s4_conversion",
        }
        assert contract["stages"]["dense_shared_trunk"]["savings_permitted"] is False
        assert contract["stages"]["s2_candidate_search"]["savings_permitted"] is False
        assert contract["stages"]["selected_output_head"]["savings_permitted"] is False

    transplat = resolve_s2_s3_execution_contract("transplat")
    assert transplat["stages"]["selected_output_head"] == {
        "execution": "same_weight_replay_available_audit_required",
        "savings_permitted": False,
        "head_structure": "Conv3x3->GELU->Conv3x3",
        "first_conv_closure": "dense_for_repeated_T4_corner_selection",
        "second_conv": "selected_outputs_only",
        "implementation": "saes.selected_output_replay",
        "audit_entrypoint": "scripts/saes_selected_output_replay_audit.py",
        "reason": (
            "the standalone primitive is not yet bound into a quality "
            "pipeline record with its actual selected-output event trace"
        ),
    }

    depthsplat = resolve_s2_s3_execution_contract("depthsplat")
    assert depthsplat["status"] == "stage_specific_dense_closure"
    assert depthsplat["stages"]["selected_output_head"] == {
        "execution": "same_weight_replicate_replay_available_audit_required",
        "savings_permitted": False,
        "head_structure": "Conv3x3->GELU->Conv3x3",
        "padding_mode": "replicate",
        "first_conv_closure": "dense_for_repeated_T4_corner_selection",
        "second_conv": "selected_outputs_only",
        "implementation": "saes.selected_output_replay",
        "audit_entrypoint": "scripts/acid_joint_runtime_control_evidence.py",
        "reason": (
            "the standalone final Gaussian-head replay is not yet bound "
            "into a source-bound runtime evidence record with its actual "
            "selected-output event trace"
        ),
    }


def test_execution_dependency_contract_imports_without_torch():
    code = """
import builtins

original_import = builtins.__import__

def import_without_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError('Torch must not be imported by schema helpers')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_torch
from saes.execution_dependency import resolve_s2_s3_execution_contract
assert resolve_s2_s3_execution_contract('mvsplat')['status'] == 'stage_specific_partial_replay'
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=__file__.rsplit("/tests/", 1)[0],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_model_execution_dependency_rejects_unknown_model_identity():
    from saes.execution_dependency import resolve_s2_s3_execution_contract

    with pytest.raises(ValueError, match="unsupported SAES execution-dependency"):
        resolve_s2_s3_execution_contract("unknown-model")


def test_unverified_saes_route_does_not_reduce_the_fsdr_denominator():
    import scripts.demo as demo

    previous_config = demo.CONFIG
    demo.CONFIG = demo.SCARFConfig()
    try:
        tracker = demo.SavingsTracker(model_type="transplat")
        tracker.record_saes(
            total_pixels=48,
            saes_stats=_mixed_stats(),
            feature_dim=128,
            tile_size=4,
            sh_degree=4,
        )
        for _ in range(48):
            tracker.record_fsdr_pixel("reuse", reused=True)
        ablation = tracker.compute_ablation(
            feature_cycles=10_000,
            depth_cycles=10_000,
            ggu_cycles=1_000,
            dp_core_cycles=10_000,
            gauss_gen_cycles=10_000,
            cost_volume_cycles=3_000,
            s1_cnn_cycles=5_000,
        )

        pipeline = ablation["_pipeline"]
        assert pipeline["saes_s2_saving"] == 0.0
        assert pipeline["fsdr_reuse_ratio"] == 1.0
        assert pipeline["fsdr_s2_saving"] > 0.0
        assert ablation["asic_fsdr_saes"]["dp_core"] == ablation["asic_fsdr"]["dp_core"]
    finally:
        demo.CONFIG = previous_config
