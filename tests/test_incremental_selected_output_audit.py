"""Unit coverage for the target-free incremental head/Adapter audit boundary."""

import copy
import hashlib
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


def _guard_distribution_fixture():
    from scripts.saes_incremental_selected_output_audit import _canonical_json_sha256

    def check(level, center, footprint, depth):
        return {
            "level": level,
            "anchor_count": 4 if level == "L0" else 12,
            "context_safety": {
                "passed": center <= 2.0,
                "coverage_footprint_ratio": footprint,
                "relative_depth_span": depth,
                "projected_center_mahalanobis_max": center,
                "center_overlap_passed": center <= 2.0,
                "reason": "accepted" if center <= 2.0 else "center_separation",
                "nonprobe_s3_attribute_reads": 0,
            },
        }

    trace = [
        {
            "view": 0,
            "tile_y": 0,
            "tile_x": 0,
            "pre_guard_route": "L0",
            "depth_uniform": True,
            "guard_checks": [check("L0", 1.9, 3.0, 0.10)],
            "final_route": "L0",
        },
        {
            "view": 0,
            "tile_y": 0,
            "tile_x": 1,
            "pre_guard_route": "L0",
            "depth_uniform": True,
            "guard_checks": [check("L0", 2.2, 4.0, 0.15)],
            "final_route": "Full",
        },
        {
            "view": 0,
            "tile_y": 1,
            "tile_x": 0,
            "pre_guard_route": "L1",
            "depth_uniform": True,
            "guard_checks": [check("L1", 2.146, 5.0, 0.20)],
            "final_route": "Full",
        },
        {
            "view": 0,
            "tile_y": 1,
            "tile_x": 1,
            "pre_guard_route": "L1",
            "depth_uniform": True,
            "guard_checks": [check("L1", 2.448, 6.0, 0.25)],
            "final_route": "Full",
        },
    ]
    plan_trace_sha256 = "a" * 64
    plan_events = {
        "contract_version": "saes-incremental-probe-first-plan-v1",
        "tile_size": 4,
        "feature_threshold": 0.2,
        "depth_threshold": 0.1,
        "decision_semantics": "paper-probe-normalized-feature-first-hit",
        "l0_anchor_count": 4,
        "l1_anchor_count": 15,
        "l1_anchor_semantics": "engineering-lightweight-15-adaptive-center-s1-loo-v1",
        "tile_trace_sha256": plan_trace_sha256,
    }
    route_events = {
        "schema_version": "saes-compact-l0-l1-route-v1",
        "tile_size": 4,
        "materialization": "representative",
        "context_safety_guard": False,
        "compact_materialization_enabled": True,
        "execution_policy": (
            "engineering-nonzero-l1-15-adaptive-absolute-residual-v4-attribute-loo-dev"
        ),
        "l0_anchor_count": 4,
        "l1_anchor_count": 15,
        "l1_anchor_semantics": "engineering-lightweight-15-adaptive-center-s1-loo-v1",
        "source_tile_trace_sha256": plan_trace_sha256,
        "tile_trace_sha256": _canonical_json_sha256(trace),
        "source_selection_mask_sha256": "b" * 64,
        "raw_head_request_mask_sha256": "c" * 64,
        "route_counts": {"L0": 1, "L1": 0, "Full": 3},
    }
    return trace, route_events, plan_events


def test_guard_distribution_is_scalar_hash_bound_and_threshold_partitioned(tmp_path):
    from scripts.saes_incremental_selected_output_audit import (
        _build_guard_distribution,
        _write_guard_distribution,
    )

    trace, route_events, plan_events = _guard_distribution_fixture()
    distribution = _build_guard_distribution(
        guarded_route_trace=trace,
        guarded_route_events=route_events,
        plan_events=plan_events,
        input_identity={"scene": "fixed-scene", "target_rgb_accessed": False},
        checkpoint_sha256="d" * 64,
    )

    assert distribution["paper_result_eligible"] is False
    assert distribution["target_rgb_accessed"] is False
    assert distribution["renderer_executed"] is False
    assert distribution["quality_metrics_computed"] is False
    assert distribution["scalar_only"] is True
    assert distribution["trace_binding"]["guarded_route_tile_trace_sha256"] == (
        route_events["tile_trace_sha256"]
    )
    assert len(distribution["route_identity"]["sha256"]) == 64

    l0 = distribution["levels"]["L0"]
    assert l0["context_guard_check_count"] == 2
    assert l0["center_mahalanobis"]["min"] == 1.9
    assert l0["center_mahalanobis"]["max"] == 2.2
    assert l0["center_mahalanobis"]["cumulative_counts"] == {
        "2.0": {"count": 1, "finite_fraction": 0.5},
        "2.146": {"count": 1, "finite_fraction": 0.5},
        "2.448": {"count": 2, "finite_fraction": 1.0},
    }
    l1 = distribution["levels"]["L1"]
    assert l1["center_mahalanobis"]["cumulative_counts"] == {
        "2.0": {"count": 0, "finite_fraction": 0.0},
        "2.146": {"count": 1, "finite_fraction": 0.5},
        "2.448": {"count": 2, "finite_fraction": 1.0},
    }
    encoded = json.dumps(distribution, allow_nan=False, sort_keys=True)
    for raw_trace_field in (
        '"guard_checks"',
        '"tile_x"',
        '"tile_y"',
        '"means"',
        '"covariances"',
        '"harmonics"',
        '"opacities"',
    ):
        assert raw_trace_field not in encoded

    artifact = _write_guard_distribution(tmp_path, distribution)
    path = tmp_path / artifact["path"]
    assert path.name == "guard-distribution.json"
    assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["trace_binding"] == distribution["trace_binding"]
    assert persisted["route_identity"] == distribution["route_identity"]


def test_guard_distribution_rejects_trace_drift_and_serializes_nonfinite_as_counts():
    from scripts.saes_incremental_selected_output_audit import (
        _build_guard_distribution,
        _canonical_json_sha256,
    )

    trace, route_events, plan_events = _guard_distribution_fixture()
    drifted = copy.deepcopy(trace)
    drifted[0]["guard_checks"][0]["context_safety"][
        "projected_center_mahalanobis_max"
    ] = 1.0
    with pytest.raises(ValueError, match="does not match the guarded route"):
        _build_guard_distribution(
            guarded_route_trace=drifted,
            guarded_route_events=route_events,
            plan_events=plan_events,
            input_identity={"scene": "fixed-scene"},
            checkpoint_sha256="d" * 64,
        )

    nonfinite = copy.deepcopy(trace)
    nonfinite[1]["guard_checks"][0]["context_safety"][
        "projected_center_mahalanobis_max"
    ] = float("inf")
    route_events = {**route_events, "tile_trace_sha256": _canonical_json_sha256(nonfinite)}
    distribution = _build_guard_distribution(
        guarded_route_trace=nonfinite,
        guarded_route_events=route_events,
        plan_events=plan_events,
        input_identity={"scene": "fixed-scene"},
        checkpoint_sha256="d" * 64,
    )
    center = distribution["levels"]["L0"]["center_mahalanobis"]
    assert center["count"] == 2
    assert center["finite_count"] == 1
    assert center["nonfinite_count"] == 1
    assert center["max"] == 1.9
    json.dumps(distribution, allow_nan=False, sort_keys=True)


def test_incremental_audit_cli_enables_distribution_only_with_explicit_flag(
    monkeypatch, tmp_path
):
    import scripts.result_record as result_record
    import scripts.saes_incremental_selected_output_audit as audit

    calls = []

    def fake_collect(**kwargs):
        calls.append(kwargs)
        return {
            "status": "PASS",
            "target_rgb_accessed": False,
            "guard_distribution": (
                {"path": "guard-distribution.json"}
                if kwargs["guard_distribution_output_dir"] is not None
                else None
            ),
        }

    def fake_write_result(record, path):
        path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(audit, "collect_incremental_selected_output_audit", fake_collect)
    monkeypatch.setattr(result_record, "write_result", fake_write_result)

    enabled = tmp_path / "enabled"
    assert (
        audit.main(
            [
                "--input-root",
                str(tmp_path / "input"),
                "--output-dir",
                str(enabled),
                "--device",
                "cpu",
                "--v15-calibration-record",
                str(tmp_path / "v15.json"),
                "--v16-calibration-record",
                str(tmp_path / "v16.json"),
                "--write-guard-distribution",
            ]
        )
        == 0
    )
    assert calls[-1]["guard_distribution_output_dir"] == enabled
    assert calls[-1]["v15_calibration_record"] == tmp_path / "v15.json"
    assert calls[-1]["v16_calibration_record"] == tmp_path / "v16.json"
    persisted = json.loads((enabled / "results.json").read_text(encoding="utf-8"))
    assert persisted["guard_distribution"] == {"path": "guard-distribution.json"}
    recorded_sha256 = persisted.pop("sha256")
    assert recorded_sha256 == audit._canonical_json_sha256(persisted)

    disabled = tmp_path / "disabled"
    assert (
        audit.main(
            [
                "--input-root",
                str(tmp_path / "input"),
                "--output-dir",
                str(disabled),
                "--device",
                "cpu",
                "--v15-calibration-record",
                str(tmp_path / "v15.json"),
                "--v16-calibration-record",
                str(tmp_path / "v16.json"),
            ]
        )
        == 0
    )
    assert calls[-1]["guard_distribution_output_dir"] is None


def _frozen_l1_calibration_record(*, checkpoint_sha256, acid_binding, threshold):
    return {
        "sha256": "a" * 64,
        "threshold_value": threshold,
        "application": {
            "model": "transplat",
            "dataset": "dl3dv",
            "checkpoint_sha256": checkpoint_sha256,
        },
        "acid_binding": acid_binding,
        "access": {
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "skipped_s3_attributes_accessed": False,
        },
    }


def test_audit_loads_the_verified_disjoint_pair_against_the_active_re10k_checkpoint(
    monkeypatch, tmp_path
):
    import saes.evaluation_disjoint_l1_calibration as calibration
    import scripts.saes_incremental_selected_output_audit as audit

    checkpoint = tmp_path / "re10k.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha256 = "d" * 64
    acid_binding = {"dataset": "acid", "binding": "same-24-8"}
    v15 = _frozen_l1_calibration_record(
        checkpoint_sha256=checkpoint_sha256, acid_binding=acid_binding, threshold=0.1
    )
    v16 = {
        **_frozen_l1_calibration_record(
            checkpoint_sha256=checkpoint_sha256, acid_binding=acid_binding, threshold=0.2
        ),
        "sha256": "b" * 64,
    }
    calls = []

    def load_v15(path, **kwargs):
        calls.append(("v15", Path(path), kwargs))
        return v15

    def load_v16(path, **kwargs):
        calls.append(("v16", Path(path), kwargs))
        return v16

    monkeypatch.setattr(calibration, "load_frozen_v15_threshold", load_v15)
    monkeypatch.setattr(calibration, "load_frozen_v16_threshold", load_v16)
    v15_path = tmp_path / "v15.json"
    v16_path = tmp_path / "v16.json"
    evidence = audit._load_evaluation_disjoint_l1_calibrations(
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        v15_calibration_record=v15_path,
        v16_calibration_record=v16_path,
        acid_calibration_plan=tmp_path / "acid-plan.json",
        acid_materialization_root=tmp_path / "acid-materialization",
    )

    assert calls[0][0] == "v15"
    assert calls[0][2]["checkpoint_path"] == checkpoint.resolve()
    assert calls[1][0] == "v16"
    assert calls[1][2]["checkpoint_path"] == checkpoint.resolve()
    assert calls[1][2]["v15_record_path"] == v15_path
    assert evidence == {
        "v15_calibration": {
            "sha256": "a" * 64,
            "threshold_value": 0.1,
            "acid_binding": acid_binding,
            "application": v15["application"],
        },
        "v16_calibration": {
            "sha256": "b" * 64,
            "threshold_value": 0.2,
            "acid_binding": acid_binding,
            "application": v16["application"],
        },
    }

    v16["application"] = {
        **v16["application"],
        "checkpoint_sha256": "e" * 64,
    }
    with pytest.raises(RuntimeError, match="active Re10K checkpoint"):
        audit._load_evaluation_disjoint_l1_calibrations(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            v15_calibration_record=v15_path,
            v16_calibration_record=v16_path,
        )


def test_audit_preserves_frozen_loader_rejections_for_legacy_parent_and_checkpoint(
    monkeypatch, tmp_path
):
    import saes.evaluation_disjoint_l1_calibration as calibration
    import scripts.saes_incremental_selected_output_audit as audit

    checkpoint = tmp_path / "re10k.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    kwargs = {
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": "d" * 64,
        "v15_calibration_record": tmp_path / "v15.json",
        "v16_calibration_record": tmp_path / "v16.json",
    }

    def reject_legacy(*_args, **_kwargs):
        raise ValueError("V15 record identity is invalid")

    monkeypatch.setattr(calibration, "load_frozen_v15_threshold", reject_legacy)
    with pytest.raises(ValueError, match="V15 record identity"):
        audit._load_evaluation_disjoint_l1_calibrations(**kwargs)

    acid_binding = {"dataset": "acid", "binding": "same-24-8"}
    valid_v15 = _frozen_l1_calibration_record(
        checkpoint_sha256="d" * 64, acid_binding=acid_binding, threshold=0.1
    )
    monkeypatch.setattr(
        calibration, "load_frozen_v15_threshold", lambda *_args, **_kwargs: valid_v15
    )

    def reject_parent(*_args, **_kwargs):
        raise ValueError("V16 record does not bind its verified V15 parent")

    monkeypatch.setattr(calibration, "load_frozen_v16_threshold", reject_parent)
    with pytest.raises(ValueError, match="verified V15 parent"):
        audit._load_evaluation_disjoint_l1_calibrations(**kwargs)

    def reject_checkpoint(*_args, **_kwargs):
        raise ValueError("V15 record application checkpoint changed")

    monkeypatch.setattr(calibration, "load_frozen_v15_threshold", reject_checkpoint)
    with pytest.raises(ValueError, match="application checkpoint changed"):
        audit._load_evaluation_disjoint_l1_calibrations(**kwargs)


def test_audit_verifies_frozen_calibrations_before_any_encoder_capture(monkeypatch, tmp_path):
    import scripts.saes_incremental_selected_output_audit as audit

    checkpoint = tmp_path / "re10k.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []
    execution = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": "d" * 64,
    }
    calibration = {
        "v15_calibration": {"threshold_value": 0.1},
        "v16_calibration": {"threshold_value": 0.2},
    }

    monkeypatch.setattr(
        audit,
        "_load_context_only_encoder",
        lambda *_args, **_kwargs: (
            object(),
            {"image": torch.zeros(1, 1, 1, 4, 4)},
            {"scene": "sample-zero", "context_indices": [1, 3], "source_sample_index": 0},
            execution,
        ),
    )

    def load_calibration(**kwargs):
        calls.append("calibration")
        assert kwargs["checkpoint_path"] == checkpoint
        assert kwargs["checkpoint_sha256"] == "d" * 64
        return calibration

    class CaptureStopped(RuntimeError):
        pass

    def stop_at_encoder(*_args, **_kwargs):
        calls.append("encoder")
        assert calls == ["calibration", "encoder"]
        raise CaptureStopped()

    monkeypatch.setattr(audit, "_load_evaluation_disjoint_l1_calibrations", load_calibration)
    monkeypatch.setattr(audit, "_capture_dense_reference", stop_at_encoder)
    with pytest.raises(CaptureStopped):
        audit.collect_incremental_selected_output_audit(
            input_root=tmp_path / "context-only",
            device=torch.device("cpu"),
            v15_calibration_record=tmp_path / "v15.json",
            v16_calibration_record=tmp_path / "v16.json",
        )
    assert calls == ["calibration", "encoder"]


def test_frozen_l1_15_packed_guard_binds_both_thresholds_and_final_trace():
    from types import SimpleNamespace

    from scripts.saes_incremental_selected_output_audit import (
        _require_frozen_l1_15_packed_guard,
    )

    v15 = {"threshold_value": 0.1}
    v16 = {"threshold_value": 0.2}
    plan = SimpleNamespace(
        events={
            "decision_semantics": "paper-probe-normalized-feature-first-hit",
            "l1_anchor_semantics": "engineering-lightweight-15-adaptive-center-s1-loo-v1",
            "l1_anchor_count": 15,
            "selection_mask_sha256": "c" * 64,
        }
    )
    policy = "engineering-nonzero-l1-15-adaptive-absolute-residual-v4-attribute-loo-dev"
    capture = {
        "compact_nonzero_materialization": True,
        "compact_materialization_preflight": SimpleNamespace(
            events={
                "execution_policy": policy,
                "selected_anchor_v4_attribute_loo_guard": True,
                "selected_anchor_v4_attribute_loo_maximum_risk": 0.2,
            }
        ),
        "guarded_route": SimpleNamespace(
            events={
                "execution_policy": policy,
                "adaptive_l1_absolute_residual_guard": True,
                "adaptive_l1_absolute_residual_maximum": 0.1,
                "l1_anchor_semantics": "engineering-lightweight-15-adaptive-center-s1-loo-v1",
                "l1_anchor_count": 15,
                "compact_materialization_enabled": True,
                "selected_output_mask_sha256": "e" * 64,
            }
        ),
        "final_packed": SimpleNamespace(
            source_trace={
                "compact_materialization_execution_policy": policy,
                "compact_materialization_selected_anchor_v4_attribute_loo_guard": True,
                "compact_materialization_selected_anchor_v4_attribute_loo_maximum_risk": 0.2,
                "route_selection_mask_sha256": "c" * 64,
            },
            source_trace_sha256="f" * 64,
        ),
    }

    assert _require_frozen_l1_15_packed_guard(
        plan=plan,
        capture=capture,
        v15_calibration=v15,
        v16_calibration=v16,
    ) == {
        "source_selection_mask_sha256": "c" * 64,
        "selected_output_mask_sha256": "e" * 64,
        "packed_source_trace_sha256": "f" * 64,
    }

    capture["compact_materialization_preflight"].events[
        "selected_anchor_v4_attribute_loo_maximum_risk"
    ] = 0.3
    with pytest.raises(RuntimeError, match="did not drive the packed guard"):
        _require_frozen_l1_15_packed_guard(
            plan=plan,
            capture=capture,
            v15_calibration=v15,
            v16_calibration=v16,
        )


def test_incremental_audit_rejects_phase_trace_that_does_not_match_the_plan():
    from scripts.saes_incremental_selected_output_audit import _validate_phase_plan_binding

    plan_events = {
        "primary_mask_sha256": "a" * 64,
        "secondary_mask_sha256": "b" * 64,
        "full_mask_sha256": "c" * 64,
        "selection_mask_sha256": "d" * 64,
        "primary_head_final_positions": 16,
        "secondary_head_final_positions": 16,
        "full_head_final_positions": 16,
        "scheduled_head_positions": 44,
    }
    head_events = {
        "head_forward_invocations": 1,
        "head_final_positions_executed": 44,
        "phases": [
            {
                "phase": "primary",
                "mask_sha256": "a" * 64,
                "head_final_positions_requested": 16,
                "head_final_positions_executed": 16,
                "head_final_positions_reused": 0,
            },
            {
                "phase": "secondary",
                "mask_sha256": "b" * 64,
                "head_final_positions_requested": 16,
                "head_final_positions_executed": 12,
                "head_final_positions_reused": 4,
            },
            {
                "phase": "full",
                "mask_sha256": "c" * 64,
                "head_final_positions_requested": 16,
                "head_final_positions_executed": 16,
                "head_final_positions_reused": 0,
            },
        ],
    }

    _validate_phase_plan_binding(plan_events, head_events)

    head_events["phases"][2]["mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="full phase mask"):
        _validate_phase_plan_binding(plan_events, head_events)


def test_incremental_audit_rejects_unbound_or_replayed_appended_full_extension():
    from scripts.saes_incremental_selected_output_audit import (
        _validate_appended_full_extension_binding,
        _validate_phase_plan_binding,
    )

    plan_events = {
        "primary_mask_sha256": "a" * 64,
        "secondary_mask_sha256": "b" * 64,
        "full_mask_sha256": "c" * 64,
        "selection_mask_sha256": "d" * 64,
        "primary_head_final_positions": 4,
        "secondary_head_final_positions": 8,
        "full_head_final_positions": 0,
        "scheduled_head_positions": 12,
    }
    initial = {
        "head_forward_invocations": 1,
        "head_weight_sha256": "e" * 64,
        "head_input_sha256": "f" * 64,
        "head_final_positions_executed": 12,
        "phases": [
            {
                "phase": "primary",
                "mask_sha256": "a" * 64,
                "head_final_positions_requested": 4,
                "head_final_positions_executed": 4,
                "head_final_positions_reused": 0,
            },
            {
                "phase": "secondary",
                "mask_sha256": "b" * 64,
                "head_final_positions_requested": 8,
                "head_final_positions_executed": 8,
                "head_final_positions_reused": 0,
            },
            {
                "phase": "full",
                "mask_sha256": "c" * 64,
                "head_final_positions_requested": 0,
                "head_final_positions_executed": 0,
                "head_final_positions_reused": 0,
            },
        ],
    }
    _validate_phase_plan_binding(plan_events, initial)
    extension = {
        "phase": "full_extension",
        "mask_sha256": "g" * 64,
        "head_final_positions_requested": 4,
        "head_final_positions_executed": 4,
        "head_final_positions_reused": 0,
    }
    final = {
        **initial,
        "head_final_positions_executed": 16,
        "computed_mask_sha256": "h" * 64,
        "full_extension_dispatched": True,
        "full_extension_positions_executed": 4,
        "full_extension_mask_sha256": "g" * 64,
        "full_tile_native_identity_verified": False,
        "phases": [*initial["phases"], extension],
    }
    guard = {
        "requires_incremental_full_dispatch": True,
        "additional_full_descriptor_count": 4,
        "additional_full_mask_sha256": "g" * 64,
        "raw_head_request_mask_sha256": "h" * 64,
    }

    _validate_appended_full_extension_binding(plan_events, initial, final, guard)

    bad_mask = {**final, "phases": [*final["phases"]]}
    bad_mask["phases"][-1] = {**extension, "mask_sha256": "0" * 64}
    with pytest.raises(ValueError, match="phase mask"):
        _validate_appended_full_extension_binding(plan_events, initial, bad_mask, guard)

    replayed = {**final, "phases": [*final["phases"]]}
    replayed["phases"][-1] = {**extension, "head_final_positions_reused": 1}
    with pytest.raises(ValueError, match="does not conserve"):
        _validate_appended_full_extension_binding(plan_events, initial, replayed, guard)

    rebound = {**final, "head_input_sha256": "1" * 64}
    with pytest.raises(ValueError, match="head_input_sha256"):
        _validate_appended_full_extension_binding(plan_events, initial, rebound, guard)

    duplicate_invocation = {**final, "head_forward_invocations": 2}
    with pytest.raises(ValueError, match="one scoped head invocation"):
        _validate_appended_full_extension_binding(
            plan_events, initial, duplicate_invocation, guard
        )


def test_incremental_audit_builds_a_canonical_selected_packet_only():
    from types import SimpleNamespace

    from saes.sparse_gaussian_consumer import PackedGaussianConsumer
    from scripts.saes_incremental_selected_output_audit import (
        _build_sparse_packet,
        _extract_selected_adapter_inputs,
    )

    selection = torch.zeros(2, 4, 4, dtype=torch.bool)
    selection[0, 0, 0] = True
    selection[0, 1, 3] = True
    selection[1, 3, 2] = True
    raw_head = torch.arange(2 * 5 * 4 * 4, dtype=torch.float32).reshape(2, 5, 4, 4)
    extrinsics = torch.eye(4).reshape(1, 1, 1, 1, 1, 4, 4).repeat(1, 2, 1, 1, 1, 1, 1)
    intrinsics = torch.eye(3).reshape(1, 1, 1, 1, 1, 3, 3).repeat(1, 2, 1, 1, 1, 1, 1)
    coordinates = torch.zeros(1, 2, 16, 1, 1, 2)
    depths = torch.arange(32, dtype=torch.float32).reshape(1, 2, 16, 1, 1) + 1.0
    opacities = depths / 100.0
    raw_body = torch.zeros(1, 2, 16, 1, 1, 3)
    for view in range(2):
        for pixel in range(16):
            raw_body[0, view, pixel, 0, 0] = raw_head[view, 2:, pixel // 4, pixel % 4]

    selected_inputs = _extract_selected_adapter_inputs(
        (extrinsics, intrinsics, coordinates, depths, opacities, raw_body, (4, 4)),
        selection,
    )
    packet = _build_sparse_packet(
        raw_head,
        selection,
        selected_inputs,
        head_events={
            "schema_version": "saes-incremental-head-execution-v1",
            "source_bound": True,
            "execution_scope": "s3_raw_gaussian_head_only",
            "head_weight_sha256": "a" * 64,
            "head_input_sha256": "b" * 64,
            "head_final_positions_executed": 3,
            "phase_trace_sha256": "c" * 64,
            "head_forward_invocations": 1,
        },
        plan_events={
            "contract_version": "saes-incremental-probe-first-plan-v1",
            "tile_trace_sha256": "d" * 64,
            "primary_mask_sha256": "e" * 64,
            "secondary_mask_sha256": "f" * 64,
            "full_mask_sha256": "0" * 64,
            "selection_mask_sha256": "1" * 64,
            "primary_head_final_positions": 3,
            "secondary_head_final_positions": 0,
            "full_head_final_positions": 0,
        },
    )

    assert packet.descriptor_keys.tolist() == [[0, 0, 0, 0], [0, 0, 7, 0], [0, 1, 14, 0]]
    assert packet.primitive_keys.tolist() == [
        [0, 0, 0, 0, 0],
        [0, 0, 7, 0, 0],
        [0, 1, 14, 0, 0],
    ]
    assert packet.dense_slots.tolist() == [0, 7, 30]
    torch.testing.assert_close(packet.raw_descriptors[:, 2:], selected_inputs.raw_body)
    assert packet.source_trace["source_bound"] is True
    assert packet.source_trace["route_tile_trace_sha256"] == "d" * 64
    assert packet.source_trace["route_primary_mask_sha256"] == "e" * 64
    assert packet.source_trace["route_secondary_mask_sha256"] == "f" * 64
    assert packet.source_trace["route_full_mask_sha256"] == "0" * 64
    assert packet.source_trace["route_selection_mask_sha256"] == "1" * 64
    assert packet.source_trace["head_forward_invocations"] == 1

    class Adapter:
        d_in = 3

        def __call__(
            self,
            _extrinsics,
            _intrinsics,
            coordinates,
            adapter_depths,
            adapter_opacities,
            adapter_raw_body,
            _image_shape,
        ):
            count = adapter_depths.numel()
            return SimpleNamespace(
                means=torch.cat((coordinates, adapter_depths.unsqueeze(-1)), dim=-1),
                covariances=torch.eye(3).expand(count, -1, -1).clone(),
                harmonics=adapter_raw_body.reshape(count, 3, 1),
                opacities=adapter_opacities,
            )

    packed = PackedGaussianConsumer(Adapter()).convert(
        packet,
        extrinsics=extrinsics[:, :, 0, 0, 0],
        intrinsics=intrinsics[:, :, 0, 0, 0],
        image_shape=(4, 4),
    )
    assert packed.dense_slots.tolist() == [0, 7, 30]

    selected_positions = selection.nonzero(as_tuple=False)
    selected_raw = raw_head.permute(0, 2, 3, 1)[
        selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2]
    ]
    packet_from_selected = _build_sparse_packet(
        selected_raw,
        selection,
        selected_inputs,
        head_events={
            "schema_version": "saes-incremental-head-execution-v1",
            "source_bound": True,
            "execution_scope": "s3_raw_gaussian_head_only",
            "head_weight_sha256": "a" * 64,
            "head_input_sha256": "b" * 64,
            "head_final_positions_executed": 3,
            "phase_trace_sha256": "c" * 64,
            "head_forward_invocations": 1,
        },
        plan_events={
            "contract_version": "saes-incremental-probe-first-plan-v1",
            "tile_trace_sha256": "d" * 64,
            "primary_mask_sha256": "e" * 64,
            "secondary_mask_sha256": "f" * 64,
            "full_mask_sha256": "0" * 64,
            "selection_mask_sha256": "1" * 64,
            "primary_head_final_positions": 3,
            "secondary_head_final_positions": 0,
            "full_head_final_positions": 0,
        },
    )
    torch.testing.assert_close(packet_from_selected.raw_descriptors, packet.raw_descriptors)


def test_selected_adapter_input_equivalence_isolates_coordinate_drift():
    from scripts.saes_incremental_selected_output_audit import (
        SelectedAdapterInputs,
        _selected_adapter_inputs_equivalence,
    )

    extrinsics = torch.eye(4).repeat(2, 1, 1)
    intrinsics = torch.eye(3).repeat(2, 1, 1)
    reference = SelectedAdapterInputs(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        coordinates=torch.tensor(((0.25, 0.75), (0.50, 0.50))),
        depths=torch.tensor((2.0, 3.0)),
        mapped_opacities=torch.tensor((0.2, 0.3)),
        raw_body=torch.ones(2, 3),
        image_shape=(2, 2),
    )
    candidate = SelectedAdapterInputs(
        extrinsics=extrinsics.clone(),
        intrinsics=intrinsics.clone(),
        coordinates=reference.coordinates + torch.tensor(((0.0, 0.0), (0.25, 0.0))),
        depths=reference.depths.clone(),
        mapped_opacities=reference.mapped_opacities.clone(),
        raw_body=reference.raw_body.clone(),
        image_shape=reference.image_shape,
    )

    report = _selected_adapter_inputs_equivalence(reference, candidate)

    assert report["equivalent"] is False
    assert report["inputs"]["coordinates"]["equivalent"] is False
    assert report["inputs"]["coordinates"]["maximum_absolute_delta"] == 0.25
    assert report["inputs"]["extrinsics"]["equivalent"] is True
    assert report["inputs"]["depths"]["equivalent"] is True


def test_dense_reference_captures_direct_adapter_inputs_and_restores_forward():
    from types import SimpleNamespace

    from scripts.saes_incremental_selected_output_audit import _capture_dense_reference

    class Adapter(nn.Module):
        def forward(
            self,
            _extrinsics,
            _intrinsics,
            coordinates,
            _depths,
            opacities,
            _raw_body,
            _image_shape,
        ):
            count = coordinates.shape[2]
            return SimpleNamespace(
                means=torch.zeros(1, count, 3),
                covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, count, 1, 1),
                harmonics=torch.zeros(1, count, 3, 1),
                opacities=opacities.reshape(1, count),
            )

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_gaussians = nn.Conv2d(1, 5, 1)

        def forward(self, features):
            raw = self.to_gaussians(features.reshape(1, 1, 2, 2))
            depths = torch.ones(1, 1, 4, 1, 1)
            densities = torch.full_like(depths, 0.2)
            return depths, densities, raw

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_predictor = Predictor()
            self.gaussian_adapter = Adapter()

        def forward(self, context, _global_step, deterministic=True):
            del deterministic
            depths, densities, raw = self.depth_predictor(context["features"])
            raw_body = raw[:, 2:].permute(0, 2, 3, 1).reshape(1, 1, 4, 1, 1, 3)
            extrinsics = torch.eye(4).reshape(1, 1, 1, 1, 1, 4, 4)
            intrinsics = torch.eye(3).reshape(1, 1, 1, 1, 1, 3, 3)
            coordinates = torch.zeros(1, 1, 4, 1, 1, 2)
            return self.gaussian_adapter.forward(
                extrinsics,
                intrinsics,
                coordinates,
                depths,
                densities,
                raw_body,
                (2, 2),
            )

    encoder = Encoder()
    model = SimpleNamespace(encoder=encoder)
    dense, captured = _capture_dense_reference(
        model, {"features": torch.ones(1, 1, 1, 2, 2)}
    )

    assert tuple(dense.means.shape) == (1, 4, 3)
    assert set(captured) == {
        "features",
        "depths",
        "head_input",
        "raw_head",
        "adapter_inputs",
    }
    assert len(captured["adapter_inputs"]) == 7
    assert tuple(captured["adapter_inputs"][2].shape) == (1, 1, 4, 1, 1, 2)
    assert encoder.gaussian_adapter.forward.__func__ is Adapter.forward


def test_planning_capture_stops_before_dense_adapter_or_gaussian_output():
    from types import SimpleNamespace

    from scripts.saes_incremental_selected_output_audit import (
        _capture_s1_s2_without_dense_adapter,
    )

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *_args):
            self.calls += 1
            raise AssertionError("planning pass must not execute the dense Adapter")

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_gaussians = nn.Conv2d(1, 5, 1)

        def forward(self, features):
            raw = self.to_gaussians(features.reshape(1, 1, 2, 2))
            depths = torch.ones(1, 1, 4, 1, 1)
            densities = torch.full_like(depths, 0.2)
            return depths, densities, raw

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_predictor = Predictor()
            self.gaussian_adapter = Adapter()

        def forward(self, context, _global_step, deterministic=True):
            del deterministic
            depths, densities, raw = self.depth_predictor(context["features"])
            raw_body = raw[:, 2:].permute(0, 2, 3, 1).reshape(1, 1, 4, 1, 1, 3)
            extrinsics = torch.eye(4).reshape(1, 1, 1, 1, 1, 4, 4)
            intrinsics = torch.eye(3).reshape(1, 1, 1, 1, 1, 3, 3)
            coordinates = torch.zeros(1, 1, 4, 1, 1, 2)
            return self.gaussian_adapter.forward(
                extrinsics,
                intrinsics,
                coordinates,
                depths,
                densities,
                raw_body,
                (2, 2),
            )

    encoder = Encoder()
    model = SimpleNamespace(encoder=encoder)
    captured = _capture_s1_s2_without_dense_adapter(
        model, {"features": torch.ones(1, 1, 1, 2, 2)}
    )

    assert set(captured) == {"features", "depths"}
    assert tuple(captured["features"].shape) == (1, 1, 1, 2, 2)
    assert tuple(captured["depths"].shape) == (1, 1, 4, 1, 1)
    assert encoder.gaussian_adapter.calls == 0
    assert encoder.gaussian_adapter.forward.__func__ is Adapter.forward


def test_incremental_audit_intercepts_a_direct_adapter_forward_call_and_restores_it():
    from types import SimpleNamespace

    from scripts.saes_incremental_selected_output_audit import (
        _capture_incremental_pre_adapter,
    )

    class Adapter(nn.Module):
        d_in = 3

        def forward(self, *_args):
            return "native-adapter-result"

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_gaussians = nn.Sequential(
                nn.Conv2d(3, 4, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(4, 5, 3, 1, 1),
            ).eval()

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_predictor = Predictor()
            self.gaussian_adapter = Adapter()

        def forward(self, context, _global_step, deterministic=True):
            del deterministic
            raw = self.depth_predictor.to_gaussians(context["head_input"])
            body = raw[:, 2:].permute(0, 2, 3, 1).reshape(1, 1, 16, 1, 1, 3)
            extrinsics = torch.eye(4).reshape(1, 1, 1, 1, 1, 4, 4)
            intrinsics = torch.eye(3).reshape(1, 1, 1, 1, 1, 3, 3)
            coordinates = torch.zeros(1, 1, 16, 1, 1, 2)
            depths = torch.ones(1, 1, 16, 1, 1)
            opacities = torch.full((1, 1, 16, 1, 1), 0.5)
            return self.gaussian_adapter.forward(
                extrinsics, intrinsics, coordinates, depths, opacities, body, (4, 4)
            )

    encoder = Encoder().eval()
    model = SimpleNamespace(encoder=encoder)
    primary = torch.zeros(1, 4, 4, dtype=torch.bool)
    primary[0, 0, 0] = True
    zero = torch.zeros_like(primary)

    events, captured = _capture_incremental_pre_adapter(
        model,
        {"head_input": torch.ones(1, 3, 4, 4)},
        primary_mask=primary,
        secondary_mask=zero,
        full_mask=zero,
    )

    assert events["head_forward_invocations"] == 1
    assert captured["raw_head"].shape == (1, 5)
    assert captured["adapter_inputs"].raw_body.shape == (1, 3)
    assert encoder.gaussian_adapter.forward() == "native-adapter-result"


def test_guarded_packed_adapter_executes_full_extension_in_one_encoder_invocation():
    from types import SimpleNamespace

    from saes.guarded_selected_route import (
        ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
    )
    from saes.probe_first_schedule import ADAPTIVE_L1_15_ANCHOR_SEMANTICS
    from saes.probe_first_schedule import build_incremental_probe_first_plan
    from scripts.saes_incremental_selected_output_audit import (
        EVALUATION_DISJOINT_L1_15_DECISION_SEMANTICS,
        _capture_guarded_incremental_packed_adapter,
    )

    class Adapter(nn.Module):
        d_in = 3

        def __init__(self, *, reject_l1: bool):
            super().__init__()
            self.calls = 0
            self.reject_l1 = reject_l1

        def forward(
            self,
            _extrinsics,
            _intrinsics,
            coordinates,
            adapter_depths,
            adapter_opacities,
            _raw_body,
            _image_shape,
        ):
            self.calls += 1
            count = adapter_depths.numel()
            harmonics = torch.ones(count, 3, 1, device=adapter_depths.device)
            if self.reject_l1:
                # This is the first non-primary boundary anchor, matching the
                # selected-only L1 guard rejection fixture.
                harmonics[1] *= -1.0
            return SimpleNamespace(
                means=torch.cat((coordinates, adapter_depths.unsqueeze(-1)), dim=-1),
                covariances=torch.eye(3, device=adapter_depths.device)
                .expand(count, -1, -1)
                .clone()
                * 0.25,
                harmonics=harmonics,
                opacities=adapter_opacities,
            )

    class Predictor(nn.Module):
        def __init__(self, *, source_opacity: float):
            super().__init__()
            self.source_opacity = source_opacity
            self.to_gaussians = nn.Sequential(
                nn.Conv2d(3, 4, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(4, 5, 3, 1, 1),
            ).eval()
            with torch.no_grad():
                for parameter in self.to_gaussians.parameters():
                    parameter.zero_()
                self.to_gaussians[2].bias.copy_(
                    torch.tensor((2.0, -2.0, 1.0, 2.0, 3.0))
                )

        def forward(self, features, *_args, **_kwargs):
            raw = self.to_gaussians(features[0])
            depths = torch.ones(1, 1, 16, 1, 1, device=raw.device)
            densities = torch.full_like(depths, self.source_opacity)
            return depths, densities, raw.permute(0, 2, 3, 1).reshape(1, 1, 16, 5)

    class Encoder(nn.Module):
        def __init__(self, *, source_opacity: float, reject_l1: bool):
            super().__init__()
            self.depth_predictor = Predictor(source_opacity=source_opacity)
            self.gaussian_adapter = Adapter(reject_l1=reject_l1)

        def forward(self, context, _global_step, deterministic=True):
            del deterministic
            depths, densities, raw = self.depth_predictor(context["features"])
            height = width = 4
            rows, columns = torch.meshgrid(
                torch.arange(height), torch.arange(width), indexing="ij"
            )
            base_coordinates = torch.stack(
                ((columns + 0.5) / width, (rows + 0.5) / height), dim=-1
            ).reshape(1, 1, 16, 1, 1, 2).to(raw)
            offsets = raw[..., :2].sigmoid().reshape(1, 1, 16, 1, 1, 2)
            coordinates = base_coordinates + (offsets - 0.5) * torch.tensor(
                (1 / width, 1 / height), dtype=raw.dtype, device=raw.device
            )
            extrinsics = torch.eye(4, device=raw.device).reshape(1, 1, 1, 1, 1, 4, 4)
            intrinsics = torch.eye(3, device=raw.device).reshape(1, 1, 1, 1, 1, 3, 3)
            return self.gaussian_adapter.forward(
                extrinsics,
                intrinsics,
                coordinates,
                depths,
                densities,
                raw[..., 2:].reshape(1, 1, 16, 1, 1, 3),
                (height, width),
            )

    features = torch.zeros(1, 1, 3, 4, 4)
    for (row, column), value in {
        (0, 0): (0.0, 0.0),
        (0, 3): (2.0, 0.0),
        (3, 0): (0.0, 2.0),
        (3, 3): (2.0, 2.0),
    }.items():
        features[0, 0, :2, row, column] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
    )
    assert int(plan.selection_mask.sum()) == 12
    encoder = Encoder(source_opacity=0.2, reject_l1=True).eval()
    model = SimpleNamespace(encoder=encoder)
    head_calls = []
    handle = encoder.depth_predictor.to_gaussians.register_forward_hook(
        lambda *_args: head_calls.append(True)
    )
    try:
        captured = _capture_guarded_incremental_packed_adapter(
            model,
            {"features": features},
            plan=plan,
        )
    finally:
        handle.remove()

    assert len(head_calls) == 1
    assert encoder.gaussian_adapter.calls == 2
    assert captured["initial_head_events"]["head_forward_invocations"] == 1
    assert captured["final_head_events"]["head_forward_invocations"] == 1
    assert captured["guarded_route"].events["requires_incremental_full_dispatch"]
    assert captured["extension_event"] is not None
    assert captured["extension_event"]["head_final_positions_executed"] == 4
    assert captured["final_head_events"]["head_final_positions_executed"] == 16
    assert len(captured["final_head_events"]["phases"]) == 4
    assert captured["final_head_events"]["phases"][-1]["phase"] == "full_extension"
    torch.testing.assert_close(
        captured["final_raw_head"][:, 2:],
        torch.tensor((1.0, 2.0, 3.0)).expand(16, -1),
    )
    torch.testing.assert_close(
        captured["final_selected_inputs"].raw_body,
        captured["final_raw_head"][:, 2:],
    )
    rows, columns = torch.meshgrid(
        torch.arange(4), torch.arange(4), indexing="ij"
    )
    base_coordinates = torch.stack(
        ((columns + 0.5) / 4, (rows + 0.5) / 4), dim=-1
    ).reshape(16, 2)
    final_offset = torch.tensor((2.0, -2.0)).sigmoid()
    expected_coordinates = base_coordinates + (final_offset - 0.5) * torch.tensor(
        (1 / 4, 1 / 4)
    )
    torch.testing.assert_close(
        captured["final_selected_inputs"].coordinates, expected_coordinates
    )
    torch.testing.assert_close(
        captured["final_packet"].coordinates, expected_coordinates
    )
    extension_slots = captured["guarded_route"].additional_full_mask.reshape(-1).nonzero(
        as_tuple=False
    ).squeeze(-1)
    assert extension_slots.numel() == 4
    assert not torch.allclose(
        captured["stale_final_coordinates_before_source_geometry_rebuild"][extension_slots],
        expected_coordinates[extension_slots],
    )
    assert captured["final_packet"].dense_slots.numel() == 16
    assert captured["final_packed"].dense_slots.numel() == 16
    assert captured["final_adapter_inputs"]["equivalent"] is True
    assert captured["final_coordinates_rebuilt_from_source_geometry_after_extension"] is True
    assert (
        captured["final_packet"].source_trace["coordinates_source"]
        == "source_native_raw_offset_geometry_after_extension"
    )
    assert captured["omitted_raw_head_positions_poisoned"] == 0
    assert encoder.gaussian_adapter.forward is not None

    # A certificate-approved L0 tile stages twelve descriptors to resolve the
    # guard but emits only its four primary anchors. The final packet must use
    # the output mask, not the producer's raw-head request union.
    selected_plan = build_incremental_probe_first_plan(
        torch.zeros_like(features),
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
    )
    assert int(selected_plan.selection_mask.sum()) == 12
    selected_encoder = Encoder(source_opacity=0.0, reject_l1=False).eval()
    selected_capture = _capture_guarded_incremental_packed_adapter(
        SimpleNamespace(encoder=selected_encoder),
        {"features": torch.zeros_like(features)},
        plan=selected_plan,
    )
    selected_route = selected_capture["guarded_route"]
    assert int(selected_route.raw_head_request_mask.sum()) == 12
    assert int(selected_route.selected_output_mask.sum()) == 4
    assert selected_capture["extension_event"] is None
    assert selected_capture["final_head_events"]["head_final_positions_executed"] == 12
    expected_slots = selected_route.selected_output_mask.reshape(-1).nonzero(
        as_tuple=False
    ).squeeze(-1)
    assert selected_capture["final_packet"].dense_slots.tolist() == expected_slots.tolist()
    assert selected_capture["final_packed"].dense_slots.tolist() == expected_slots.tolist()
    assert selected_capture["final_raw_head"].shape[0] == 4
    assert selected_capture["omitted_raw_head_positions_poisoned"] == 12

    # The frozen audit's deployment path uses an adaptive 15-anchor plan and
    # drives both the S1 residual and selected-anchor V4 replay guards before
    # emitting its compact packet.
    adaptive_plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics=EVALUATION_DISJOINT_L1_15_DECISION_SEMANTICS,
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )
    compact_capture = _capture_guarded_incremental_packed_adapter(
        SimpleNamespace(encoder=Encoder(source_opacity=0.2, reject_l1=False).eval()),
        {"features": features},
        plan=adaptive_plan,
        compact_nonzero_materialization=True,
        compact_execution_policy=(
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
        ),
        adaptive_l1_maximum_leave_one_out_residual=1.0,
        selected_anchor_v4_attribute_loo_maximum_risk=1.0,
    )
    compact_preflight = compact_capture["compact_materialization_preflight"]
    assert compact_capture["compact_nonzero_materialization"] is True
    assert compact_preflight is not None
    assert compact_preflight.events["execution_policy"] == (
        ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
    )
    assert compact_preflight.events["selected_anchor_v4_attribute_loo_guard"] is True
    assert compact_capture["guarded_route"].events["l1_anchor_count"] == 15
    assert compact_capture["final_packed"].source_trace[
        "compact_materialization_execution_policy"
    ] == ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
