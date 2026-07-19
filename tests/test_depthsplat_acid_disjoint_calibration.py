"""CPU contracts for native DepthSplat ACID 24/8 V15D/V16D records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _digest(number: int) -> str:
    return f"{number:064x}"


def _binding(module):
    train = [f"train-{index:02d}" for index in range(24)]
    holdout = [f"holdout-{index:02d}" for index in range(8)]

    def split(scenes, salt):
        return {
            "scenes": scenes,
            "scene_count": len(scenes),
            "scene_set_sha256": module.canonical_sha256(sorted(scenes)),
            "selection_sha256": _digest(salt),
            "sidecar_tree_sha256": _digest(salt + 1),
            "sidecar_manifest_sha256": _digest(salt + 2),
            "input_provenance_sha256": _digest(salt + 3),
        }

    return {
        "schema_version": "1.0",
        "kind": "saes-acid-context-only-calibration-binding",
        "dataset": "acid",
        "protocol_id": "acid-test-v1",
        "plan_sha256": _digest(10),
        "plan_file_sha256": _digest(11),
        "materialization_record_sha256": _digest(12),
        "materialization_tree_sha256": _digest(13),
        "materialization_manifest_sha256": _digest(14),
        "splits": {
            "calibration_train": split(train, 20),
            "calibration_holdout": split(holdout, 30),
        },
        "access": {
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "skipped_s3_attributes_accessed": False,
        },
    }


def _backend_identity():
    return {
        "schema_version": "depthsplat-backend-frozen-identity-v1",
        "model": "depthsplat",
        "dataset": "dl3dv",
        "experiment": "dl3dv",
        "environment_profile": "depthsplat",
        "gaussian_regressor_module": "encoder.gaussian_regressor",
        "gaussian_head_module": "encoder.gaussian_head",
        "gaussian_adapter_module": "encoder.gaussian_adapter",
        "decoder_module": "decoder",
        "raw_descriptor_layout": "opacity-logit-offset-xy-adapter-body-v1",
        "coordinate_semantics": "depthsplat-z-depth-pixel-center-plus-sigmoid-offset-rgb-sh-v1",
        "checkpoint": {
            "path": "depthsplat/checkpoints/dl3dv.ckpt",
            "sha256": _digest(40),
        },
        "evaluation_index": {
            "path": "depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json",
            "sha256": _digest(41),
            "source_index_sha256": _digest(42),
            "sample_selection_sha256": _digest(43),
            "sample_count": 140,
        },
        "runtime_source": {
            "repository_path": "depthsplat",
            "repository_commit": "a" * 40,
            "src_path": "depthsplat/src",
            "src_tree_sha256": _digest(44),
            "src_file_count": 1,
            "runtime_import_roots": [],
        },
        "simulator_source_files": [
            {"path": "saes/depthsplat_l0_l1_materializer.py", "sha256": _digest(45)}
        ],
        "source_files": {
            "adapter": {
                "path": "depthsplat/src/model/encoder/common/gaussian_adapter.py",
                "sha256": _digest(46),
            }
        },
        "submodule_git_head": "b" * 40,
    }


def _access():
    return {
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "skipped_s3_attributes_accessed": False,
    }


def _evidence(index: int):
    return {
        "native_execution": {
            "routing_features_sha256": _digest(1000 + index),
            "routing_z_depths_sha256": _digest(2000 + index),
            "native_dense_gaussian_regressor_executed": True,
            "selected_replicate_gaussian_head_executed": True,
            "selected_native_rgb_adapter_executed": True,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        },
        "packet": {
            "selected_descriptor_count": 2 + index,
            "selection_mask_sha256": _digest(3000 + index),
            "source_rgb_sha256": _digest(4000 + index),
            "source_trace_sha256": _digest(5000 + index),
            "native_execution_sha256": _digest(5100 + index),
            "initial_native_attribute_binding_sha256": _digest(5200 + index),
            "final_selected_adapter_source_trace_sha256": _digest(5300 + index),
            "final_selected_adapter_binding_sha256": _digest(5400 + index),
            "materialized_source_trace_sha256": _digest(5450 + index),
            "materialized_current_binding_sha256": _digest(5475 + index),
        },
        "materialization": {
            "plan_selection_mask_sha256": _digest(5500 + index),
            "plan_tile_trace_sha256": _digest(5600 + index),
            "l1_anchor_semantics": "engineering-lightweight-15-adaptive-center-s1-loo-v1",
            "preflight_tile_trace_sha256": _digest(6000 + index),
            "preflight_events_sha256": _digest(7000 + index),
            "route_tile_trace_sha256": _digest(8000 + index),
            "route_events_sha256": _digest(9000 + index),
            "selected_output_mask_sha256": _digest(10000 + index),
            "additional_full_mask_sha256": _digest(11000 + index),
            "raw_head_request_mask_sha256": _digest(12000 + index),
            "full_passthrough_mask_sha256": _digest(13000 + index),
            "materialized_coverage_certificate_sha256": _digest(13500 + index),
            "materialized_full_attribute_binding_sha256": _digest(13750 + index),
            "materialized_full_passthrough_count": 3,
            "materialized_update_anchor_count": 1,
            "selected_anchor_attribute_loo": {
                "certificate": "depthsplat-selected-anchor-loo-test-v1",
                "q75_risk": 0.25,
                "held_out_count": 1,
                "selected_label_reads": 1,
                "nonprobe_attribute_reads": 0,
                "held_out_records_sha256": _digest(14000 + index),
                "trace_sha256": _digest(14500 + index),
            },
            "route_counts": {"L0": 1, "L1": 1, "Full": 3},
        },
    }


def _records(module, binding, *, value_key: str):
    records = {}
    for split, offset in ((module.TRAIN_SPLIT, 0), (module.HOLDOUT_SPLIT, 100)):
        result = []
        for index, scene in enumerate(binding["splits"][split]["scenes"]):
            # Each scene has two values so both p50 and p25 rules are observable.
            result.append(
                {
                    "scene": scene,
                    value_key: [float(index + 1), float(index + 3)],
                    "evidence": _evidence(offset + index),
                    "access": _access(),
                }
            )
        records[split] = result
    return records


def test_collection_plan_binds_depthsplat_and_acid_but_is_explicitly_unrun():
    import saes.depthsplat_acid_disjoint_calibration as calibration

    binding = _binding(calibration)
    plan = calibration.build_collection_plan(
        binding=binding, backend_identity=_backend_identity()
    )

    assert plan["kind"] == calibration.COLLECTION_PLAN_KIND
    assert plan["status"] == calibration.NOT_RUN_STATUS
    assert plan["collection"]["gpu_collection_attempted"] is False
    assert plan["collection"]["gpu_collection_completed"] is False
    assert plan["collection"]["reason"] == "DEPTHSPLAT_GPU_COLLECTION_NOT_RUN"
    assert len(plan["acid_binding"]["splits"][calibration.TRAIN_SPLIT]["scenes"]) == 24
    assert len(plan["acid_binding"]["splits"][calibration.HOLDOUT_SPLIT]["scenes"]) == 8
    assert plan["application"]["backend_identity"]["checkpoint"]["path"] == (
        "depthsplat/checkpoints/dl3dv.ckpt"
    )
    assert plan["application"]["collector_source"]["contract"] == (
        calibration.COLLECTOR_SOURCE_CONTRACT
    )
    assert plan["sha256"] == calibration.canonical_sha256(
        {key: value for key, value in plan.items() if key != "sha256"}
    )


def test_v15d_freezes_from_complete_target_free_records_and_v16d_is_blocked():
    import saes.depthsplat_acid_disjoint_calibration as calibration

    binding = _binding(calibration)
    residuals = _records(calibration, binding, value_key="adaptive_l1_residuals")
    v15d = calibration.build_v15d_record(
        binding=binding,
        backend_identity=_backend_identity(),
        train_scene_records=residuals[calibration.TRAIN_SPLIT],
        holdout_scene_records=residuals[calibration.HOLDOUT_SPLIT],
    )
    v16d = calibration.build_v16d_record(
        binding=binding,
        backend_identity=_backend_identity(),
        v15d_record_sha256=v15d["sha256"],
    )

    assert v15d["status"] == calibration.FROZEN_STATUS
    assert v15d["signal"]["name"] == calibration.V15D_SIGNAL
    assert v15d["threshold"]["rule"] == calibration.V15D_THRESHOLD_RULE
    assert v16d["status"] == calibration.V16D_SCHEMA_GAP_STATUS
    assert v16d["freeze_blockers"] == list(calibration.V16D_FREEZE_BLOCKERS)
    assert v16d["base_v15d_sha256"] == v15d["sha256"]
    assert v16d["mechanism"]["nonzero_l0_l1_merge_required"] is True
    assert v16d["mechanism"]["source_rgb_sh_preserved"] is True
    assert v16d["mechanism"]["whole_pipeline_s2_s3_sparse_execution_verified"] is False
    assert "threshold" not in v16d


def test_scene_records_reject_target_access_and_metric_claims():
    import saes.depthsplat_acid_disjoint_calibration as calibration

    binding = _binding(calibration)
    records = _records(calibration, binding, value_key="adaptive_l1_residuals")
    records[calibration.TRAIN_SPLIT][0]["access"]["target_rgb_accessed"] = True
    with pytest.raises(ValueError, match="scene record"):
        calibration.build_v15d_record(
            binding=binding,
            backend_identity=_backend_identity(),
            train_scene_records=records[calibration.TRAIN_SPLIT],
            holdout_scene_records=records[calibration.HOLDOUT_SPLIT],
        )

    records = _records(calibration, binding, value_key="adaptive_l1_residuals")
    records[calibration.TRAIN_SPLIT][0]["evidence"]["native_execution"][
        "quality_metrics_computed"
    ] = True
    with pytest.raises(ValueError, match="target-free boundary"):
        calibration.build_v15d_record(
            binding=binding,
            backend_identity=_backend_identity(),
            train_scene_records=records[calibration.TRAIN_SPLIT],
            holdout_scene_records=records[calibration.HOLDOUT_SPLIT],
        )


def test_scene_records_require_an_actual_nonzero_l0_or_l1_merge():
    import saes.depthsplat_acid_disjoint_calibration as calibration

    binding = _binding(calibration)
    records = _records(calibration, binding, value_key="adaptive_l1_residuals")
    records[calibration.TRAIN_SPLIT][0]["evidence"]["materialization"]["route_counts"] = {
        "L0": 0,
        "L1": 0,
        "Full": 5,
    }
    with pytest.raises(ValueError, match="nonzero merge"):
        calibration.build_v15d_record(
            binding=binding,
            backend_identity=_backend_identity(),
            train_scene_records=records[calibration.TRAIN_SPLIT],
            holdout_scene_records=records[calibration.HOLDOUT_SPLIT],
        )


def test_scene_record_builder_binds_adaptive_plan_packet_and_materializer_traces():
    import saes.depthsplat_acid_disjoint_calibration as calibration

    plan_trace = [
        {
            "view": 0,
            "tile_y": 0,
            "tile_x": 0,
            "pre_guard_route": "L1",
            "depth_uniform": True,
            "adaptive_l1_leave_one_out_residual": 0.125,
        }
    ]
    plan_selection = _digest(20000)
    plan_events = {
        "contract_version": "saes-incremental-probe-first-plan-v1",
        "target_rgb_accessed": False,
        "gaussian_attributes_accessed": False,
        "tile_size": 4,
        "l1_anchor_semantics": calibration.V15D_L1_ANCHOR_SEMANTICS,
        "l1_anchor_selection": calibration.V15D_SIGNAL,
        "l1_anchor_selection_uses_s1_only": True,
        "selection_mask_sha256": plan_selection,
        "tile_trace_sha256": calibration.canonical_sha256(plan_trace),
    }
    routing_features = _digest(20001)
    routing_z_depths = _digest(20002)
    descriptor = _digest(20003)
    rgb = _digest(20004)
    native_execution_sha256 = _digest(20009)
    initial_attribute_binding = _digest(20010)
    final_selected_attribute_binding = _digest(20011)
    coverage_certificate_sha256 = _digest(20012)
    full_attribute_binding = _digest(20013)
    materialized_current_binding = _digest(20014)
    packet_trace = {
        "source_bound": True,
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
        "source_rgb_keyword": "input_images",
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "head_forward_invocations": 1,
        "selection_mask_sha256": plan_selection,
        "routing_features_sha256": routing_features,
        "routing_z_depths_sha256": routing_z_depths,
        "selected_descriptor_sha256": descriptor,
        "selected_rgb_sha256": rgb,
        "native_execution_sha256": native_execution_sha256,
    }
    packed_trace = {
        **packet_trace,
        "native_adapter_attribute_binding_sha256": initial_attribute_binding,
    }
    preflight_trace = [{"view": 0, "planned_route": "L1", "accepted": True}]
    preflight_events = {
        "schema_version": "saes-depthsplat-l0-l1-materializer-v1",
        "target_rgb_accessed": False,
        "target_rgb_accessed_before_commit": False,
        "target_camera_accessed_before_commit": False,
        "skipped_s3_attributes_accessed": False,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "nonzero_direct_deletion": False,
        "not_lossless_deletion": True,
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "source_nonprobe_s3_attribute_reads": 0,
        "l1_anchor_semantics": calibration.V15D_L1_ANCHOR_SEMANTICS,
        "selected_anchor_attribute_loo": {
            "certificate": "depthsplat-selected-anchor-loo-test-v1",
            "q75_risk": 0.125,
            "held_out_records": [{"anchor": [0, 0], "risk": 0.125}],
            "held_out_count": 1,
            "selected_label_reads": 1,
            "nonprobe_attribute_reads": 0,
        },
        "initial_binding": {
            "plan_selection_mask_sha256": plan_selection,
            "plan_tile_trace_sha256": calibration.canonical_sha256(plan_trace),
            "packet_selection_mask_sha256": plan_selection,
            "packet_selected_descriptor_sha256": descriptor,
            "packet_selected_rgb_sha256": rgb,
            "native_execution_sha256": native_execution_sha256,
            "packed_source_trace_sha256": calibration.canonical_sha256(packed_trace),
            "packed_native_attribute_binding_sha256": initial_attribute_binding,
            "routing_features_sha256": routing_features,
            "routing_z_depths_sha256": routing_z_depths,
        },
        "tile_trace_sha256": calibration.canonical_sha256(preflight_trace),
    }
    route_trace = [{"view": 0, "planned_route": "L1", "final_route": "L1"}]
    route_events = {
        "schema_version": "saes-depthsplat-compact-l0-l1-route-v1",
        "target_rgb_accessed": False,
        "target_rgb_accessed_before_commit": False,
        "target_camera_accessed_before_commit": False,
        "skipped_s3_attributes_accessed": False,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "nonzero_direct_deletion": False,
        "not_lossless_deletion": True,
        "l1_anchor_semantics": calibration.V15D_L1_ANCHOR_SEMANTICS,
        "preflight_trace_sha256": calibration.canonical_sha256(preflight_trace),
        "tile_trace_sha256": calibration.canonical_sha256(route_trace),
        "selected_output_mask_sha256": _digest(20005),
        "additional_full_mask_sha256": _digest(20006),
        "raw_head_request_mask_sha256": _digest(20007),
        "full_passthrough_mask_sha256": _digest(20008),
        "coverage_certificate_sha256": coverage_certificate_sha256,
        "route_counts": {"L0": 0, "L1": 1, "Full": 0},
    }
    final_selected_adapter_trace = {
        **packet_trace,
        "native_adapter_attribute_binding_sha256": final_selected_attribute_binding,
        "packet_selection_kind": "depthsplat-final-selected-output-mask-v1",
        "selection_mask_sha256": route_events["selected_output_mask_sha256"],
        "producer_request_mask_sha256": route_events["raw_head_request_mask_sha256"],
        "native_full_passthrough_mask_sha256": route_events[
            "full_passthrough_mask_sha256"
        ],
        "native_full_passthrough_positions": 1,
        "native_full_adapter_attribute_execution_sha256": native_execution_sha256,
        "native_full_adapter_attribute_passthrough_mask_sha256": route_events[
            "full_passthrough_mask_sha256"
        ],
        "native_full_adapter_attribute_passthrough_count": 1,
        "native_full_adapter_attribute_binding_sha256": full_attribute_binding,
    }
    materialized_trace = {
        **final_selected_adapter_trace,
        "depthsplat_compact_current_attribute_binding_sha256": materialized_current_binding,
        "depthsplat_compact_coverage_certificate": "depthsplat-selected-z-depth-2sigma-support-v1",
        "depthsplat_compact_coverage_certificate_sha256": coverage_certificate_sha256,
        "depthsplat_compact_full_passthrough_count": 1,
        "depthsplat_compact_update_anchor_count": 1,
    }
    native_execution = {
        "source_bound": True,
        "native_execution_sha256": native_execution_sha256,
        "gaussian_regressor": {"native_dense_executed": True},
        "gaussian_head": {"padding_mode": "replicate"},
        "adapter": {"source_rgb_keyword": "input_images", "z_depth_geometry": True},
        "routing": {
            "features_sha256": routing_features,
            "z_depth_sha256": routing_z_depths,
        },
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
    }
    selected_events = {
        "padding_mode": "replicate",
        "selected_final_output_positions": 3,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
    }

    record = calibration.build_v15d_scene_record(
        scene="scene-a",
        plan_events=plan_events,
        plan_tile_trace=plan_trace,
        native_execution_events=native_execution,
        selected_head_events=selected_events,
        selected_head_equivalence={"equivalent": True},
        packet_source_trace=packet_trace,
        packed_source_trace=packed_trace,
        final_selected_adapter_source_trace=final_selected_adapter_trace,
        final_selected_adapter_binding_sha256=final_selected_attribute_binding,
        materialized_source_trace=materialized_trace,
        materialized_current_binding_sha256=materialized_current_binding,
        selected_descriptor_count=3,
        preflight_events=preflight_events,
        preflight_tile_trace=preflight_trace,
        final_route_events=route_events,
        final_route_tile_trace=route_trace,
        access=_access(),
    )

    assert record["adaptive_l1_residuals"] == [0.125]
    assert record["evidence"]["packet"]["source_rgb_sha256"] == rgb
    assert record["evidence"]["materialization"]["plan_tile_trace_sha256"] == (
        plan_events["tile_trace_sha256"]
    )
    assert record["evidence"]["materialization"]["selected_anchor_attribute_loo"] == {
        "certificate": "depthsplat-selected-anchor-loo-test-v1",
        "q75_risk": 0.125,
        "held_out_count": 1,
        "selected_label_reads": 1,
        "nonprobe_attribute_reads": 0,
        "held_out_records_sha256": calibration.canonical_sha256(
            [{"anchor": [0, 0], "risk": 0.125}]
        ),
        "trace_sha256": calibration.canonical_sha256(
            preflight_events["selected_anchor_attribute_loo"]
        ),
    }

    missing_native_execution = dict(packet_trace)
    missing_native_execution.pop("native_execution_sha256")
    with pytest.raises(ValueError, match="native execution"):
        calibration.build_v15d_scene_record(
            scene="scene-a",
            plan_events=plan_events,
            plan_tile_trace=plan_trace,
            native_execution_events=native_execution,
            selected_head_events=selected_events,
            selected_head_equivalence={"equivalent": True},
            packet_source_trace=missing_native_execution,
            packed_source_trace={
                **missing_native_execution,
                "native_adapter_attribute_binding_sha256": initial_attribute_binding,
            },
            final_selected_adapter_source_trace=final_selected_adapter_trace,
            final_selected_adapter_binding_sha256=final_selected_attribute_binding,
            materialized_source_trace=materialized_trace,
            materialized_current_binding_sha256=materialized_current_binding,
            selected_descriptor_count=3,
            preflight_events=preflight_events,
            preflight_tile_trace=preflight_trace,
            final_route_events=route_events,
            final_route_tile_trace=route_trace,
            access=_access(),
        )

    missing_initial_attributes = dict(packed_trace)
    missing_initial_attributes.pop("native_adapter_attribute_binding_sha256")
    with pytest.raises(ValueError, match="initial native Adapter attributes"):
        calibration.build_v15d_scene_record(
            scene="scene-a",
            plan_events=plan_events,
            plan_tile_trace=plan_trace,
            native_execution_events=native_execution,
            selected_head_events=selected_events,
            selected_head_equivalence={"equivalent": True},
            packet_source_trace=packet_trace,
            packed_source_trace=missing_initial_attributes,
            final_selected_adapter_source_trace=final_selected_adapter_trace,
            final_selected_adapter_binding_sha256=final_selected_attribute_binding,
            materialized_source_trace=materialized_trace,
            materialized_current_binding_sha256=materialized_current_binding,
            selected_descriptor_count=3,
            preflight_events=preflight_events,
            preflight_tile_trace=preflight_trace,
            final_route_events=route_events,
            final_route_tile_trace=route_trace,
            access=_access(),
        )

    missing_final_attributes = dict(final_selected_adapter_trace)
    missing_final_attributes.pop("native_adapter_attribute_binding_sha256")
    with pytest.raises(ValueError, match="final selected Adapter binding"):
        calibration.build_v15d_scene_record(
            scene="scene-a",
            plan_events=plan_events,
            plan_tile_trace=plan_trace,
            native_execution_events=native_execution,
            selected_head_events=selected_events,
            selected_head_equivalence={"equivalent": True},
            packet_source_trace=packet_trace,
            packed_source_trace=packed_trace,
            final_selected_adapter_source_trace=missing_final_attributes,
            final_selected_adapter_binding_sha256=final_selected_attribute_binding,
            materialized_source_trace=materialized_trace,
            materialized_current_binding_sha256=materialized_current_binding,
            selected_descriptor_count=3,
            preflight_events=preflight_events,
            preflight_tile_trace=preflight_trace,
            final_route_events=route_events,
            final_route_tile_trace=route_trace,
            access=_access(),
        )

    missing_materialized_current = dict(materialized_trace)
    missing_materialized_current.pop(
        "depthsplat_compact_current_attribute_binding_sha256"
    )
    with pytest.raises(ValueError, match="materialized packet provenance"):
        calibration.build_v15d_scene_record(
            scene="scene-a",
            plan_events=plan_events,
            plan_tile_trace=plan_trace,
            native_execution_events=native_execution,
            selected_head_events=selected_events,
            selected_head_equivalence={"equivalent": True},
            packet_source_trace=packet_trace,
            packed_source_trace=packed_trace,
            final_selected_adapter_source_trace=final_selected_adapter_trace,
            final_selected_adapter_binding_sha256=final_selected_attribute_binding,
            materialized_source_trace=missing_materialized_current,
            materialized_current_binding_sha256=materialized_current_binding,
            selected_descriptor_count=3,
            preflight_events=preflight_events,
            preflight_tile_trace=preflight_trace,
            final_route_events=route_events,
            final_route_tile_trace=route_trace,
            access=_access(),
        )

    packet_trace.pop("routing_features_sha256")
    with pytest.raises(ValueError, match="routing feature binding"):
        calibration.build_v15d_scene_record(
            scene="scene-a",
            plan_events=plan_events,
            plan_tile_trace=plan_trace,
            native_execution_events=native_execution,
            selected_head_events=selected_events,
            selected_head_equivalence={"equivalent": True},
            packet_source_trace=packet_trace,
            packed_source_trace={
                **packet_trace,
                "native_adapter_attribute_binding_sha256": initial_attribute_binding,
            },
            final_selected_adapter_source_trace=final_selected_adapter_trace,
            final_selected_adapter_binding_sha256=final_selected_attribute_binding,
            materialized_source_trace=materialized_trace,
            materialized_current_binding_sha256=materialized_current_binding,
            selected_descriptor_count=3,
            preflight_events=preflight_events,
            preflight_tile_trace=preflight_trace,
            final_route_events=route_events,
            final_route_tile_trace=route_trace,
            access=_access(),
        )


def test_backend_identity_rejects_classic_or_wrong_checkpoint_binding():
    import saes.depthsplat_acid_disjoint_calibration as calibration

    binding = _binding(calibration)
    identity = _backend_identity()
    identity["model"] = "mvsplat"
    with pytest.raises(ValueError, match="native DL3DV route"):
        calibration.build_collection_plan(binding=binding, backend_identity=identity)

    identity = _backend_identity()
    identity["checkpoint"]["path"] = "depthsplat/checkpoints/re10k.ckpt"
    with pytest.raises(ValueError, match="application binding"):
        calibration.build_collection_plan(binding=binding, backend_identity=identity)


def test_frozen_load_rechecks_parent_and_live_bindings(tmp_path: Path, monkeypatch):
    import saes.depthsplat_acid_disjoint_calibration as calibration

    binding = _binding(calibration)
    backend = _backend_identity()
    residuals = _records(calibration, binding, value_key="adaptive_l1_residuals")
    v15d = calibration.build_v15d_record(
        binding=binding,
        backend_identity=backend,
        train_scene_records=residuals[calibration.TRAIN_SPLIT],
        holdout_scene_records=residuals[calibration.HOLDOUT_SPLIT],
    )
    v16d = calibration.build_v16d_record(
        binding=binding,
        backend_identity=backend,
        v15d_record_sha256=v15d["sha256"],
    )
    v15d_path = tmp_path / "v15d.json"
    v16d_path = tmp_path / "v16d.json"
    v15d_path.write_text(json.dumps(v15d), encoding="utf-8")
    v16d_path.write_text(json.dumps(v16d), encoding="utf-8")
    monkeypatch.setattr(calibration, "resolve_depthsplat_acid_binding", lambda **_kwargs: binding)
    monkeypatch.setattr(
        calibration,
        "_validate_live_application",
        lambda application, **_kwargs: application,
    )

    loaded = calibration.load_v16d_schema_gap_record(v16d_path, v15d_path=v15d_path)
    assert loaded["sha256"] == v16d["sha256"]

    v16d["base_v15d_sha256"] = _digest(999999)
    v16d["sha256"] = calibration.canonical_sha256(
        {key: value for key, value in v16d.items() if key != "sha256"}
    )
    v16d_path.write_text(json.dumps(v16d), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        calibration.load_v16d_schema_gap_record(v16d_path, v15d_path=v15d_path)


def test_native_module_does_not_import_classic_v16_or_backend_contract():
    import saes.depthsplat_acid_disjoint_calibration as calibration

    source = Path(calibration.__file__).read_text(encoding="utf-8")
    assert "classic_backend" not in source
    assert "build_v16_record" not in source
    assert "load_frozen_v16_threshold" not in source
