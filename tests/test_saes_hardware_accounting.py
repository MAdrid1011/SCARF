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
        "level1_pixels": 8,
        "l0_representatives": 4,
        "l1_lightweight_anchors": 8,
        "full_stage3_gaussians": 16,
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
        "l1_retained_anchors": 8,
        "full_stage3_gaussians": 16,
        "l0_nonanchors": 12,
        "l1_nonanchors": 8,
        "total_nonanchors": 20,
    }
    # T=4, K=4, 2K=8, C=128 and W=64 make these fixed event counts
    # independently checkable without model outputs or target images.
    assert ledger["cycles"]["decision_total"] == 92
    assert ledger["cycles"]["assignment_total"] == 692
    assert ledger["cycles"]["moment_matching_total"] == 248
    assert ledger["cycles"]["storage_transfer"] == 283
    assert ledger["cycles"]["serialized_accounting_cycles"] == 1315
    assert ledger["traffic_bytes"]["charged_storage_total"] == 4515
    assert ledger["assumptions"]["rtl_cycle_equivalent"] is False


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
        assert ablation["asic_saes"]["saes_control"] == 1315
        assert ablation["asic_fsdr_saes"]["saes_control"] == 1315
        assert (
            ablation["_pipeline"]["saes_control_timing_class"]
            == "analytic_no_overlap_not_rtl_cycle_equivalent"
        )
        assert ablation["_pipeline"]["saes_total_ratio"] == pytest.approx(20 / 48)
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


def test_model_execution_dependency_contracts_fail_closed_without_sparse_proof():
    from saes.execution_dependency import resolve_s2_s3_execution_contract

    expected_hashes = {
        "transplat": "8a73027989cfaafce2145b6370d2209450725eb33dfbd7c9afa0e5c5ead82679",
        "mvsplat": "ac5610772240887f7f5004c050fbc0ad08261337dd18b02f3430c7cf00174ea5",
        "depthsplat": "893747ecb7d3336f90b9f7afdf052cd3d946d8728b37ec5ebf558533a4befd76",
    }
    for model, expected_hash in expected_hashes.items():
        contract = resolve_s2_s3_execution_contract(model)
        assert contract["s2_s3_sparse_execution_verified"] is False
        assert contract["status"] == "dense_dependency_detected"
        assert contract["evidence"]["results_sha256"] == expected_hash


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
assert resolve_s2_s3_execution_contract('mvsplat')['status'] == 'dense_dependency_detected'
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
