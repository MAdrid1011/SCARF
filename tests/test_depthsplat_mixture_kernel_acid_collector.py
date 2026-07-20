"""CPU orchestration contracts for v3 ACID kernel-risk collection."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def _digest(value: int) -> str:
    return f"{value:064x}"


def _access():
    return {
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "skipped_s3_attributes_accessed": False,
    }


def _binding(calibration):
    def split(prefix, count, salt):
        scenes = [f"{prefix}-{index:02d}" for index in range(count)]
        return {
            "scenes": scenes,
            "scene_count": count,
            "scene_set_sha256": calibration.canonical_sha256(sorted(scenes)),
            "selection_sha256": _digest(salt),
            "sidecar_tree_sha256": _digest(salt + 1),
            "sidecar_manifest_sha256": _digest(salt + 2),
            "input_provenance_sha256": _digest(salt + 3),
        }

    return {
        "schema_version": "1.0",
        "kind": "saes-acid-context-only-calibration-binding",
        "dataset": "acid",
        "protocol_id": "kernel-risk-collector-test-v1",
        "plan_sha256": _digest(1),
        "plan_file_sha256": _digest(2),
        "materialization_record_sha256": _digest(3),
        "materialization_tree_sha256": _digest(4),
        "materialization_manifest_sha256": _digest(5),
        "splits": {
            calibration.TRAIN_SPLIT: split("train", 24, 10),
            calibration.HOLDOUT_SPLIT: split("holdout", 8, 20),
        },
        "access": _access(),
    }


def _aggregate(calibration):
    return {
        "schema": calibration.DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_AGGREGATE_SCHEMA,
        "kind": calibration.KERNEL_CLOSURE_KIND,
        "policy": calibration.KERNEL_CLOSURE_POLICY,
    }


def _observation(calibration, *, split: str, scene: str, sample_index: int):
    risk = 0.1 + sample_index / 1000.0 + (1.0 if split == calibration.HOLDOUT_SPLIT else 0.0)
    source_only = {
        "source_camera_only": True,
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "omitted_s3_attributes_accessed": False,
        "boolean_owner_assignment_used": False,
        "projected_domain_guard_used": False,
        "covariance_expansion_used": False,
        "alpha_union_used": False,
    }
    closure = {
        "schema_version": calibration.KERNEL_CLOSURE_SCHEMA_VERSION,
        "kind": calibration.KERNEL_CLOSURE_KIND,
        "policy": calibration.KERNEL_CLOSURE_POLICY,
        "source_only": source_only,
        "tile_key": [0, 0, 0],
        "passed": True,
        "summary": {
            "input_valid": True,
            "reason": None,
            "strict_maximum_relative_risk": 2.0,
            "maximum_world_kernel_risk": risk,
            "maximum_source_kernel_risk": risk,
            "maximum_kernel_risk": risk,
            "maximum_source_log_depth_rms": 0.0,
            "per_output": [],
        },
        "binding": {"test": _digest(sample_index + 90)},
    }
    certificate = {
        "schema_version": calibration.SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION,
        "kind": calibration.SOFT_MIXTURE_CERTIFICATE_KIND,
        "policy": calibration.SOFT_MIXTURE_CERTIFICATE_POLICY,
        "passed": True,
    }
    trace = [
        {
            "view": 0,
            "tile_y": 0,
            "tile_x": 0,
            "accepted_level": "L0",
            "mixture_kernel_closure": closure,
            "mixture_kernel_closure_sha256": calibration.canonical_sha256(closure),
            "soft_mixture_certificate": certificate,
            "soft_mixture_certificate_sha256": calibration.canonical_sha256(certificate),
        }
    ]
    observations = calibration.kernel_risk_observations_from_preflight_trace(trace)
    profile = calibration.mixture_kernel_risk_v3_profile()
    aggregate = _aggregate(calibration)
    return {
        "scene": scene,
        "kernel_risk_observations": observations,
        "preflight_tile_trace": trace,
        "mixture_kernel_closure_aggregate": aggregate,
        "evidence": {
            "profile_sha256": calibration.mixture_kernel_risk_v3_profile_sha256(),
            "route_plan_config_sha256": profile["route_plan_config_sha256"],
            "native_execution_sha256": _digest(100),
            "initial_attribute_binding_sha256": _digest(101),
            "final_selected_attribute_binding_sha256": _digest(102),
            "selected_head_replay_fallback_summary": {},
            "materialized_attribute_binding_sha256": _digest(103),
            "full_passthrough_mask_sha256": _digest(104),
            "full_attribute_binding_sha256": _digest(105),
            "full_attributes_bitwise_native": True,
            "coverage_certificate": calibration.DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
            "coverage_certificate_sha256": _digest(106),
            "accepted_update_slots_sha256": _digest(107),
            # A calibration collection may be Full after strict abstention.
            "accepted_update_slot_count": 0,
            "mixture_kernel_closure_aggregate": aggregate,
            "mixture_kernel_closure_aggregate_sha256": calibration.canonical_sha256(aggregate),
            "mixture_kernel_closure_trace_sha256": calibration.canonical_sha256(observations),
            "source_nonprobe_s3_attribute_reads": 0,
            "nonzero_merge_applied": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        },
        "access": _access(),
    }


def test_private_collector_writes_32_traces_and_immediately_reloads(tmp_path, monkeypatch):
    import saes.depthsplat_mixture_kernel_acid_calibration as calibration
    import saes.depthsplat_mixture_kernel_acid_collector as collector

    monkeypatch.setattr(
        calibration,
        "_mixture_kernel_closure_aggregate_from_trace",
        lambda _trace: _aggregate(calibration),
    )
    binding = _binding(calibration)
    application = calibration.build_mixture_kernel_risk_application(
        backend_identity={"test_backend_identity_sha256": _digest(700)}
    )
    calls = []

    def worker(*, split, sample_index, scene, **kwargs):
        assert set(kwargs) == {"materialization_root", "plan_path"}
        calls.append((split, sample_index, scene))
        return _observation(
            calibration, split=split, scene=scene, sample_index=sample_index
        )

    def loader(path, **_kwargs):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        return {**record, "threshold_value": record["threshold"]["value"]}

    result = collector._collect_mixture_kernel_risk_calibration_with_dependencies(
        output_directory=tmp_path / "kernel-risk",
        plan_path=tmp_path / "plan.json",
        materialization_root=tmp_path / "materialization",
        scene_worker=worker,
        binding=binding,
        application=application,
        record_loader=loader,
    )

    output = tmp_path / "kernel-risk"
    assert result["record_path"] == output / calibration.KERNEL_RISK_RECORD_NAME
    assert result["scene_trace_count"] == 32
    assert len(calls) == 32
    assert calls[:2] == [
        (calibration.TRAIN_SPLIT, 0, "train-00"),
        (calibration.TRAIN_SPLIT, 1, "train-01"),
    ]
    record = json.loads(result["record_path"].read_text(encoding="utf-8"))
    assert record["threshold"]["value"] == pytest.approx(0.1)
    assert record["holdout_verification"]["threshold_updated"] is False
    assert len(list((output / "scenes").rglob("*.json"))) == 32
    with pytest.raises(FileExistsError, match="must be new"):
        collector._collect_mixture_kernel_risk_calibration_with_dependencies(
            output_directory=output,
            scene_worker=worker,
            binding=binding,
            application=application,
        )


def test_public_collector_has_no_injected_worker_and_requires_cuda(tmp_path):
    import saes.depthsplat_mixture_kernel_acid_collector as collector

    parameters = inspect.signature(collector.collect_mixture_kernel_risk_calibration).parameters
    assert set(parameters) == {"device", "output_directory", "plan_path", "materialization_root"}
    with pytest.raises(ValueError, match="requires a CUDA device"):
        collector.collect_mixture_kernel_risk_calibration(
            device=torch.device("cpu"), output_directory=tmp_path / "out"
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        collector.collect_mixture_kernel_risk_calibration(
            device=torch.device("cpu"),
            output_directory=tmp_path / "out",
            scene_worker=lambda **_kwargs: {},
        )


def test_collector_source_identity_covers_its_executable_worker_set():
    import saes.depthsplat_mixture_kernel_acid_calibration as calibration
    import saes.depthsplat_mixture_kernel_acid_collector as collector

    assert collector.REQUIRED_COLLECTOR_SOURCE_FILES == set(calibration._COLLECTOR_SOURCE_FILES)
