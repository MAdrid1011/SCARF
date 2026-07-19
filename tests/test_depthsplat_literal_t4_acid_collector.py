"""CPU contracts for the literal T=4 ACID V16 collection boundary."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def _digest(number: int) -> str:
    return f"{number:064x}"


def _access() -> dict[str, bool]:
    return {
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "skipped_s3_attributes_accessed": False,
    }


def _binding(calibration) -> dict:
    def split(prefix: str, count: int, salt: int) -> dict:
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
        "protocol_id": "literal-t4-collector-test-v1",
        "plan_sha256": _digest(10),
        "plan_file_sha256": _digest(11),
        "materialization_record_sha256": _digest(12),
        "materialization_tree_sha256": _digest(13),
        "materialization_manifest_sha256": _digest(14),
        "splits": {
            calibration.TRAIN_SPLIT: split("train", 24, 20),
            calibration.HOLDOUT_SPLIT: split("holdout", 8, 30),
        },
        "access": _access(),
    }


def _loo_observation(calibration, risk: float) -> dict:
    corners = ([0, 0], [0, 3], [3, 0], [3, 3])
    return {
        "checked": True,
        "scorable": True,
        "status": "scored",
        "reason": None,
        "certificate": calibration.LITERAL_T4_LOO_CERTIFICATE,
        "policy": calibration.LITERAL_T4_LOO_POLICY,
        "risk_metric": calibration.LITERAL_T4_RISK_METRIC,
        "level": "L0",
        "anchor_count": 4,
        "held_out_anchor_count": 4,
        "held_out_anchor_records": [
            {
                "held_out_local_position": list(position),
                "harmonic_relative_error": risk,
                "opacity_logit_relative_error": risk,
                "risk": risk,
            }
            for position in corners
        ],
        "q75_risk": risk,
        "maximum_held_out_risk": risk,
        "selected_anchor_native_attribute_label_reads": 4,
        "selected_anchor_native_attribute_endpoint_reads": 0,
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_direct_deletion": False,
        "maximum_allowed_risk": None,
        "passed": None,
        "action": "observed_only",
    }


def _observation(calibration, *, split: str, scene: str, sample_index: int) -> dict:
    offset = 1.0 if split == calibration.HOLDOUT_SPLIT else 0.0
    risks = [offset + sample_index / 100.0 + value for value in (0.1, 0.2, 0.8)]
    trace = [
        {
            "view": 0,
            "tile_y": 0,
            "tile_x": index,
            "planned_route": "L0",
            "attempted": True,
            "accepted": True,
            "reason": "accepted",
            "source_nonprobe_s3_attribute_reads": 0,
            "selected_anchor_attribute_loo": _loo_observation(calibration, risk),
        }
        for index, risk in enumerate(risks)
    ]
    aggregate = calibration._aggregate_from_preflight_tile_trace(trace)
    number = sample_index + (100 if split == calibration.HOLDOUT_SPLIT else 0)
    return {
        "scene": scene,
        "maximum_held_out_risks": list(aggregate["maximum_held_out_risks"]),
        "preflight_tile_trace": trace,
        "selected_anchor_loo_aggregate": aggregate,
        "evidence": {
            "profile_sha256": calibration.literal_t4_profile_sha256(),
            "route_plan_config_sha256": calibration.literal_t4_profile()[
                "route_plan_config_sha256"
            ],
            "native_execution_sha256": _digest(1000 + number),
            "initial_attribute_binding_sha256": _digest(2000 + number),
            "final_selected_attribute_binding_sha256": _digest(3000 + number),
            "materialized_attribute_binding_sha256": _digest(4000 + number),
            "full_passthrough_mask_sha256": _digest(5000 + number),
            "full_attribute_binding_sha256": _digest(6000 + number),
            "full_attributes_bitwise_native": True,
            "coverage_certificate": calibration.LITERAL_T4_COVERAGE_CERTIFICATE,
            "coverage_certificate_sha256": _digest(7000 + number),
            "accepted_update_slots_sha256": _digest(8000 + number),
            "accepted_update_slot_count": 1,
            "source_nonprobe_s3_attribute_reads": 0,
            "nonzero_merge_applied": True,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        },
        "access": _access(),
    }


def _patch_live_calibration(calibration, collector, monkeypatch, binding):
    assert set(calibration._COLLECTOR_SOURCE_FILES) == collector.REQUIRED_COLLECTOR_SOURCE_FILES
    monkeypatch.setattr(
        calibration, "resolve_depthsplat_acid_binding", lambda **_kwargs: binding
    )
    monkeypatch.setattr(
        calibration, "_validate_live_application", lambda application, **_kwargs: application
    )


def test_stub_worker_writes_all_immutable_traces_then_live_reloads(
    tmp_path: Path, monkeypatch
):
    import saes.depthsplat_literal_t4_acid_calibration as calibration
    import saes.depthsplat_literal_t4_acid_collector as collector

    binding = _binding(calibration)
    _patch_live_calibration(calibration, collector, monkeypatch, binding)
    application = calibration.build_literal_t4_application(
        backend_identity={"test_backend_identity_sha256": _digest(900)}, root=collector.ROOT
    )
    calls: list[tuple[str, int, str]] = []

    def worker(*, split, sample_index, scene, **kwargs):
        assert "runtime" not in kwargs
        assert set(kwargs) == {"materialization_root", "plan_path"}
        calls.append((split, sample_index, scene))
        return _observation(
            calibration, split=split, scene=scene, sample_index=sample_index
        )

    result = collector._collect_literal_t4_v16_calibration_with_dependencies(
        output_directory=tmp_path / "literal-v16t4",
        plan_path=tmp_path / "plan.json",
        materialization_root=tmp_path / "materialization",
        scene_worker=worker,
        binding=binding,
        application=application,
    )

    output = tmp_path / "literal-v16t4"
    record_path = output / "v16t4.json"
    assert result["record_path"] == record_path
    assert result["scene_trace_count"] == 32
    assert len(calls) == 32
    assert calls[:2] == [
        (calibration.TRAIN_SPLIT, 0, "train-00"),
        (calibration.TRAIN_SPLIT, 1, "train-01"),
    ]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["threshold"]["value"] == pytest.approx(0.1)
    assert record["holdout_verification"]["threshold_updated"] is False
    assert result["record"]["sha256"] == record["sha256"]
    traces = sorted((output / "scenes").rglob("*.json"))
    assert len(traces) == 32
    trace = json.loads(traces[0].read_text(encoding="utf-8"))
    assert trace["selected_anchor_loo_aggregate"]["mode"] == "collect_only"
    assert trace["selected_anchor_loo_aggregate"]["frozen_guard"] is None
    with pytest.raises(FileExistsError, match="must be new"):
        collector._collect_literal_t4_v16_calibration_with_dependencies(
            output_directory=output,
            plan_path=tmp_path / "plan.json",
            materialization_root=tmp_path / "materialization",
            scene_worker=worker,
            binding=binding,
            application=application,
        )


def test_public_collector_seals_test_injections_and_requires_cuda(tmp_path: Path):
    import saes.depthsplat_literal_t4_acid_collector as collector

    parameters = inspect.signature(collector.collect_literal_t4_v16_calibration).parameters
    assert set(parameters) == {
        "device",
        "output_directory",
        "plan_path",
        "materialization_root",
    }

    output = tmp_path / "production-output"
    with pytest.raises(ValueError, match="requires a CUDA device"):
        collector.collect_literal_t4_v16_calibration(
            device=torch.device("cpu"),
            output_directory=output,
        )
    assert not output.exists()

    injected_dependencies = {
        "scene_worker": lambda **_kwargs: {},
        "binding": {},
        "application": {},
        "record_loader": lambda **_kwargs: {},
    }
    for name, value in injected_dependencies.items():
        with pytest.raises(TypeError, match=rf"unexpected keyword argument '{name}'"):
            collector.collect_literal_t4_v16_calibration(
                device=torch.device("cpu"),
                output_directory=output,
                **{name: value},
            )


def test_collector_refuses_a_record_identity_without_its_worker_files():
    import saes.depthsplat_literal_t4_acid_collector as collector

    application = {
        "collector_source": {
            "files": [
                {
                    "path": "saes/depthsplat_literal_t4_acid_calibration.py",
                    "sha256": _digest(1),
                }
            ]
        }
    }
    with pytest.raises(RuntimeError, match="identity is incomplete"):
        collector._require_complete_collector_source(application)


def test_frozen_collector_provenance_matches_the_executable_worker_set():
    import saes.depthsplat_literal_t4_acid_calibration as calibration
    import saes.depthsplat_literal_t4_acid_collector as collector

    assert set(calibration._COLLECTOR_SOURCE_FILES) == collector.REQUIRED_COLLECTOR_SOURCE_FILES
    assert len(calibration._COLLECTOR_SOURCE_FILES) == len(
        collector.REQUIRED_COLLECTOR_SOURCE_FILES
    )


def test_trace_path_is_scene_safe_and_index_stable():
    import saes.depthsplat_literal_t4_acid_collector as collector

    path = collector._trace_relative_path(
        split=collector.TRAIN_SPLIT, sample_index=7, scene="a/b/../scene"
    )
    assert path.startswith("scenes/calibration_train/007-")
    assert path.endswith(".json")
    assert "/../" not in path
