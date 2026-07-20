"""CPU contracts for the ACID 24/8 v3 mixture-kernel risk record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _digest(value: int) -> str:
    return f"{value:064x}"


def _access() -> dict[str, bool]:
    return {
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "skipped_s3_attributes_accessed": False,
    }


def _binding(module) -> dict:
    def split(prefix: str, count: int, salt: int) -> dict:
        scenes = [f"{prefix}-{index:02d}" for index in range(count)]
        return {
            "scenes": scenes,
            "scene_count": count,
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
        "protocol_id": "kernel-risk-test-v1",
        "plan_sha256": _digest(1),
        "plan_file_sha256": _digest(2),
        "materialization_record_sha256": _digest(3),
        "materialization_tree_sha256": _digest(4),
        "materialization_manifest_sha256": _digest(5),
        "splits": {
            module.TRAIN_SPLIT: split("train", 24, 10),
            module.HOLDOUT_SPLIT: split("holdout", 8, 20),
        },
        "access": _access(),
    }


def _fake_aggregate(module, _trace):
    return {
        "schema": module.DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_AGGREGATE_SCHEMA,
        "kind": module.KERNEL_CLOSURE_KIND,
        "policy": module.KERNEL_CLOSURE_POLICY,
    }


def _trace(module, *, risk: float, valid: bool = True, certificate_passed: bool = True):
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
        "schema_version": module.KERNEL_CLOSURE_SCHEMA_VERSION,
        "kind": module.KERNEL_CLOSURE_KIND,
        "policy": module.KERNEL_CLOSURE_POLICY,
        "source_only": source_only,
        "tile_key": [0, 0, 0],
        "passed": valid and risk <= 1.0,
        "summary": {
            "input_valid": valid,
            "reason": None if valid else "kernel-integral",
            "strict_maximum_relative_risk": 1.0,
            "maximum_world_kernel_risk": risk if valid else None,
            "maximum_source_kernel_risk": risk if valid else None,
            "maximum_kernel_risk": risk if valid else None,
            "maximum_source_log_depth_rms": 0.0 if valid else None,
            "per_output": [],
        },
        "binding": {"source": _digest(55)} if valid else None,
    }
    certificate = {
        "schema_version": module.SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION,
        "kind": module.SOFT_MIXTURE_CERTIFICATE_KIND,
        "policy": module.SOFT_MIXTURE_CERTIFICATE_POLICY,
        "passed": certificate_passed,
    }
    if valid:
        return [
            {
                "view": 0,
                "tile_y": 0,
                "tile_x": 0,
                "accepted_level": "L0",
                "mixture_kernel_closure": closure,
                "mixture_kernel_closure_sha256": module.canonical_sha256(closure),
                "soft_mixture_certificate": certificate,
                "soft_mixture_certificate_sha256": module.canonical_sha256(certificate),
            }
        ]
    return [
        {
            "view": 0,
            "tile_y": 0,
            "tile_x": 0,
            "accepted_level": None,
            "l0_mixture_kernel_closure_failure": closure,
            "l0_mixture_kernel_closure_failure_sha256": module.canonical_sha256(closure),
            "l0_soft_mixture_certificate_before_kernel_closure": certificate,
            "l0_soft_mixture_certificate_before_kernel_closure_sha256": module.canonical_sha256(certificate),
        }
    ]


def _scene_record(module, *, scene: str, split: str, risk: float, artifact_root: Path):
    trace = _trace(module, risk=risk)
    aggregate = _fake_aggregate(module, trace)
    artifact = module.build_kernel_risk_trace_artifact(
        scene=scene,
        split=split,
        preflight_tile_trace=trace,
        mixture_kernel_closure_aggregate=aggregate,
    )
    relative_path = f"scenes/{split}/{scene}.json"
    artifact_path = artifact_root / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    observations = artifact["kernel_risk_observations"]
    profile = module.mixture_kernel_risk_v3_profile()
    evidence = {
        "profile_sha256": module.mixture_kernel_risk_v3_profile_sha256(),
        "route_plan_config_sha256": profile["route_plan_config_sha256"],
        "native_execution_sha256": _digest(100),
        "initial_attribute_binding_sha256": _digest(101),
        "final_selected_attribute_binding_sha256": _digest(102),
        "selected_head_replay_fallback_summary": {},
        "materialized_attribute_binding_sha256": _digest(103),
        "full_passthrough_mask_sha256": _digest(104),
        "full_attribute_binding_sha256": _digest(105),
        "full_attributes_bitwise_native": True,
        "coverage_certificate": module.DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
        "coverage_certificate_sha256": _digest(106),
        "accepted_update_slots_sha256": _digest(107),
        "accepted_update_slot_count": 0,
        "mixture_kernel_closure_aggregate": aggregate,
        "mixture_kernel_closure_aggregate_sha256": module.canonical_sha256(aggregate),
        "mixture_kernel_closure_trace_sha256": module.canonical_sha256(observations),
        "trace_artifact": {
            "relative_path": relative_path,
            "sha256": artifact["sha256"],
            "preflight_tile_trace_sha256": module.canonical_sha256(trace),
            "kernel_risk_observations_sha256": module.canonical_sha256(observations),
        },
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_merge_applied": False,
        "renderer_executed": False,
        "quality_metrics_computed": False,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
    }
    return {
        "scene": scene,
        "kernel_risk_observations": observations,
        "evidence": evidence,
        "access": _access(),
    }


def _records(module, binding, *, train_offset: float, holdout_offset: float, root: Path):
    result = {}
    for split, offset in (
        (module.TRAIN_SPLIT, train_offset),
        (module.HOLDOUT_SPLIT, holdout_offset),
    ):
        result[split] = [
            _scene_record(
                module,
                scene=scene,
                split=split,
                risk=offset + 0.1 + index / 1000.0,
                artifact_root=root,
            )
            for index, scene in enumerate(binding["splits"][split]["scenes"])
        ]
    return result


def _application(module):
    return module.build_mixture_kernel_risk_application(
        backend_identity={"test_backend_identity_sha256": _digest(900)}
    )


def _record(module, binding, *, root: Path, train_offset=0.0, holdout_offset=1.0):
    records = _records(
        module,
        binding,
        train_offset=train_offset,
        holdout_offset=holdout_offset,
        root=root,
    )
    return module.build_kernel_risk_record(
        binding=binding,
        application=_application(module),
        profile=module.mixture_kernel_risk_v3_profile(),
        train_scene_records=records[module.TRAIN_SPLIT],
        holdout_scene_records=records[module.HOLDOUT_SPLIT],
    )


def _write_record(root: Path, record) -> Path:
    path = root / "kernel-risk-v3.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_positive_threshold_uses_train_minimum_per_scene_q25_only(tmp_path, monkeypatch):
    import saes.depthsplat_mixture_kernel_acid_calibration as calibration

    monkeypatch.setattr(
        calibration, "_mixture_kernel_closure_aggregate_from_trace", lambda trace: _fake_aggregate(calibration, trace)
    )
    binding = _binding(calibration)
    baseline = _record(calibration, binding, root=tmp_path)
    changed_holdout = _record(
        calibration, binding, root=tmp_path / "other", holdout_offset=7.0
    )

    expected = [0.1 + index / 1000.0 for index in range(24)]
    assert baseline["threshold"]["per_scene_q25"] == pytest.approx(expected)
    assert baseline["threshold"]["value"] == pytest.approx(min(expected))
    assert baseline["threshold"]["value"] > 0.0
    assert changed_holdout["threshold"] == baseline["threshold"]
    assert changed_holdout["holdout_verification"]["threshold_updated"] is False


def test_trace_requires_finite_sr_certified_candidate(monkeypatch):
    import saes.depthsplat_mixture_kernel_acid_calibration as calibration

    monkeypatch.setattr(
        calibration, "_mixture_kernel_closure_aggregate_from_trace", lambda trace: _fake_aggregate(calibration, trace)
    )
    with pytest.raises(ValueError, match="S/R certificate"):
        calibration.build_kernel_risk_trace_artifact(
            scene="scene",
            split=calibration.TRAIN_SPLIT,
            preflight_tile_trace=_trace(calibration, risk=0.1, certificate_passed=False),
            mixture_kernel_closure_aggregate=_fake_aggregate(calibration, []),
        )
    artifact = calibration.build_kernel_risk_trace_artifact(
        scene="scene",
        split=calibration.HOLDOUT_SPLIT,
        preflight_tile_trace=_trace(calibration, risk=0.1, valid=False),
        mixture_kernel_closure_aggregate=_fake_aggregate(calibration, []),
    )
    assert artifact["kernel_risk_observations"] == []
    with pytest.raises(ValueError, match="observations are invalid"):
        calibration._normalize_observations(
            [], label=calibration.TRAIN_SPLIT, require_nonempty=True
        )


def test_freeze_refuses_nonpositive_train_threshold(tmp_path, monkeypatch):
    import saes.depthsplat_mixture_kernel_acid_calibration as calibration

    monkeypatch.setattr(
        calibration, "_mixture_kernel_closure_aggregate_from_trace", lambda trace: _fake_aggregate(calibration, trace)
    )
    binding = _binding(calibration)
    records = _records(
        calibration,
        binding,
        train_offset=-0.1,
        holdout_offset=1.0,
        root=tmp_path,
    )
    with pytest.raises(ValueError, match="nonpositive"):
        calibration.build_kernel_risk_record(
            binding=binding,
            application=_application(calibration),
            profile=calibration.mixture_kernel_risk_v3_profile(),
            train_scene_records=records[calibration.TRAIN_SPLIT],
            holdout_scene_records=records[calibration.HOLDOUT_SPLIT],
        )


def test_freeze_refuses_a_train_scene_without_any_finite_candidate(tmp_path, monkeypatch):
    import saes.depthsplat_mixture_kernel_acid_calibration as calibration

    monkeypatch.setattr(
        calibration, "_mixture_kernel_closure_aggregate_from_trace", lambda trace: _fake_aggregate(calibration, trace)
    )
    binding = _binding(calibration)
    records = _records(
        calibration,
        binding,
        train_offset=0.0,
        holdout_offset=1.0,
        root=tmp_path,
    )
    records[calibration.TRAIN_SPLIT][0]["kernel_risk_observations"] = []
    with pytest.raises(ValueError, match="observations are invalid"):
        calibration.build_kernel_risk_record(
            binding=binding,
            application=_application(calibration),
            profile=calibration.mixture_kernel_risk_v3_profile(),
            train_scene_records=records[calibration.TRAIN_SPLIT],
            holdout_scene_records=records[calibration.HOLDOUT_SPLIT],
        )


def test_live_reload_rebuilds_traces_and_issues_opaque_guard(tmp_path, monkeypatch):
    import saes.depthsplat_mixture_kernel_acid_calibration as calibration

    monkeypatch.setattr(
        calibration, "_mixture_kernel_closure_aggregate_from_trace", lambda trace: _fake_aggregate(calibration, trace)
    )
    binding = _binding(calibration)
    record = _record(calibration, binding, root=tmp_path)
    path = _write_record(tmp_path, record)
    monkeypatch.setattr(calibration, "resolve_depthsplat_acid_binding", lambda **_kwargs: binding)
    monkeypatch.setattr(calibration, "_validate_live_application", lambda app, **_kwargs: app)

    loaded = calibration.load_frozen_kernel_risk_threshold(path)
    guard = calibration.to_materializer_guard(path)
    assert loaded["threshold_value"] == pytest.approx(record["threshold"]["value"])
    assert guard["frozen_record_sha256"] == record["sha256"]
    assert guard["threshold_value"] == pytest.approx(record["threshold"]["value"])
    assert guard["route_plan_config_sha256"] == record["profile"]["route_plan_config_sha256"]
    assert isinstance(guard, calibration.VerifiedMixtureKernelRiskGuard)
    assert calibration.verified_mixture_kernel_materializer_guard_projection(guard) == dict(guard)
    with pytest.raises(TypeError, match="authenticated"):
        calibration.verified_mixture_kernel_materializer_guard_projection(dict(guard))


def test_live_reload_rejects_rehashed_trace_tampering(tmp_path, monkeypatch):
    import saes.depthsplat_mixture_kernel_acid_calibration as calibration

    monkeypatch.setattr(
        calibration, "_mixture_kernel_closure_aggregate_from_trace", lambda trace: _fake_aggregate(calibration, trace)
    )
    binding = _binding(calibration)
    record = _record(calibration, binding, root=tmp_path)
    evidence = record["train_scene_records"][0]["evidence"]
    artifact_path = tmp_path / evidence["trace_artifact"]["relative_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["kernel_risk_observations"][0]["maximum_kernel_risk"] = 9.0
    artifact["sha256"] = calibration.canonical_sha256(
        {key: value for key, value in artifact.items() if key != "sha256"}
    )
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    evidence["trace_artifact"]["sha256"] = artifact["sha256"]
    record["sha256"] = calibration.canonical_sha256(
        {key: value for key, value in record.items() if key != "sha256"}
    )
    path = _write_record(tmp_path, record)
    monkeypatch.setattr(calibration, "resolve_depthsplat_acid_binding", lambda **_kwargs: binding)
    monkeypatch.setattr(calibration, "_validate_live_application", lambda app, **_kwargs: app)
    with pytest.raises(ValueError, match="linked"):
        calibration.load_frozen_kernel_risk_threshold(path)
