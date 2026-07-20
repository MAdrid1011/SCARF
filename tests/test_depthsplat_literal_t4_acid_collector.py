"""CPU contracts for the literal T=4 ACID V16 collection boundary."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

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


def _replay_events(*, fallback: bool) -> dict:
    dense_macs = 100
    candidate_macs = 10
    actual_macs = dense_macs if fallback else candidate_macs
    actual_positions = 0 if fallback else 2
    fallback_positions = 2 if fallback else 0
    row = {
        "batch_item": 0,
        "source_native_dense_head_capture": fallback,
        "selected_final_output_positions": 3,
        "source_native_full_passthrough_positions": 1,
        "selected_compact_requested_positions": 2,
        "compact_replay_candidate_positions": 2,
        "candidate_replay_macs": candidate_macs,
        "selected_compact_replay_positions": actual_positions,
        "compact_replay_candidate_finite": True,
        "compact_replay_candidate_maximum_absolute_delta": 1.5e-5 if fallback else 0.0,
        "compact_replay_candidate_mean_absolute_delta": 1.0e-5 if fallback else 0.0,
        "compact_replay_strict_equivalent": not fallback,
        "compact_replay_fallback_envelope_equivalent": True,
        "native_dense_fallback_applied": fallback,
        "native_dense_fallback_compact_positions": fallback_positions,
        "native_full_passthrough_bitwise": True,
        "dense_head_macs": dense_macs,
        "replayed_head_macs": actual_macs,
        "head_mac_saving": 1.0 - actual_macs / dense_macs,
    }
    return {
        "contract_version": "depthsplat-native-dense-regressor-selected-head-rgb-adapter-v1",
        "padding_mode": "replicate",
        "batch_size": 1,
        "head_cost_semantics": "logical-route-cost-excludes-fallback-validation-v1",
        "selected_final_output_positions": 3,
        "native_full_passthrough_positions": 1,
        "native_full_passthrough_bitwise": True,
        "dense_head_macs": dense_macs,
        "actual_head_macs": actual_macs,
        "head_mac_delta": dense_macs - actual_macs,
        "head_mac_saving": 1.0 - actual_macs / dense_macs,
        "selected_compact_requested_positions": 2,
        "compact_replay_candidate_positions": 2,
        "candidate_replay_macs": candidate_macs,
        "selected_compact_replay_positions": actual_positions,
        "compact_replay_strict_failure_view_count": int(fallback),
        "native_dense_fallback_envelope_atol": 2.0e-5,
        "native_dense_fallback_envelope_rtol": 1.0e-5,
        "native_dense_fallback_view_count": int(fallback),
        "native_dense_fallback_compact_positions": fallback_positions,
        "per_view": [row],
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
    }


def _replay_fallback_summary(calibration) -> dict:
    return calibration.build_selected_head_replay_fallback_summary(
        initial_events=_replay_events(fallback=False),
        final_events=_replay_events(fallback=True),
    )


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
            "selected_head_replay_fallback_summary": _replay_fallback_summary(
                calibration
            ),
            "materialized_attribute_binding_sha256": _digest(4000 + number),
            "full_passthrough_mask_sha256": _digest(5000 + number),
            "full_attribute_binding_sha256": _digest(6000 + number),
            "full_attributes_bitwise_native": True,
            "coverage_certificate": calibration.LITERAL_T4_MOMENT_CERTIFICATE,
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


def test_native_scene_observation_uses_no_grad_and_releases_before_cuda_cache(monkeypatch):
    import saes.depthsplat_literal_t4_acid_collector as collector

    class Tracked:
        pass

    class TrackedMapping(dict):
        pass

    references: dict[str, weakref.ReferenceType[object]] = {}
    execution_calls: list[tuple[str, bool]] = []
    cache_liveness: list[list[str]] = []

    def remember(name: str, value):
        assert name not in references
        references[name] = weakref.ref(value)
        return value

    def tracked(name: str, **attributes):
        value = Tracked()
        for key, item in attributes.items():
            setattr(value, key, item)
        return remember(name, value)

    def record(name: str) -> None:
        execution_calls.append((name, torch.is_grad_enabled()))

    def context_loader(**_kwargs):
        record("context_loader")
        raw_context = remember(
            "raw_context",
            TrackedMapping(
                image=None,
                extrinsics=None,
                intrinsics=None,
                index=None,
            ),
        )
        return tracked(
            "raw",
            context=raw_context,
            identity={
                "split": collector.TRAIN_SPLIT,
                "sample_index": 0,
                "scene": "scene-0",
                "target_rgb_accessed": False,
                "target_camera_metadata_accessed": False,
                "target_index_accessed": False,
                "teacher_artifact_accessed": False,
                "expected_results_accessed": False,
            },
        )

    def prepare(raw, **_kwargs):
        record("prepare")
        assert raw.context["image"] is None
        prepared_context = remember(
            "context",
            TrackedMapping(
                image=torch.zeros(1),
                extrinsics=torch.zeros(1),
                intrinsics=torch.zeros(1),
                index=torch.zeros(1),
                near=torch.zeros(1),
                far=torch.zeros(1),
            ),
        )
        return prepared_context, {
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
        }

    def capture(*_args, **_kwargs):
        record("capture")
        return tracked(
            "execution",
            routing_features=torch.zeros(1),
            routing_z_depths=torch.zeros(1),
            dense_raw_head=torch.zeros(1, 1, 4, 4),
            gaussian_head_input=torch.zeros(1),
            dense_gaussians=object(),
            events={"native_execution_sha256": _digest(1)},
        )

    def build_plan(*_args, **_kwargs):
        record("build_plan")
        selection_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        return tracked(
            "plan",
            events={"contract_version": collector.LITERAL_PAPER_T4_PLAN_CONTRACT},
            selection_mask=selection_mask,
            full_mask=selection_mask.clone(),
        )

    replay_count = 0

    def replay(*_args, **_kwargs):
        nonlocal replay_count
        record("replay")
        replay_count += 1
        return tracked(
            "initial_replay" if replay_count == 1 else "producer_replay",
            equivalence={"equivalent": True},
            events=_replay_events(fallback=replay_count == 2),
        )

    packet_count = 0

    def build_packet(*_args, **_kwargs):
        nonlocal packet_count
        record("build_packet")
        packet_count += 1
        return tracked(
            "initial_packet" if packet_count == 1 else "producer_packet"
        )

    class Consumer:
        convert_count = 0

        def __init__(self, _adapter):
            record("consumer")
            remember("consumer", self)

        def convert(self, *_args, **_kwargs):
            record("convert")
            type(self).convert_count += 1
            if type(self).convert_count == 1:
                return tracked(
                    "initial_packed",
                    attribute_binding_sha256=_digest(2),
                    source_trace={},
                )
            return tracked(
                "final_packed",
                attribute_binding_sha256=_digest(3),
                source_trace={
                    "native_full_adapter_attribute_binding_sha256": _digest(4)
                },
            )

    def compare_packed(*_args, **_kwargs):
        record("compare_packed")
        return {"equivalent": True}

    def source_geometry(*_args, **_kwargs):
        record("source_geometry")
        return (lambda *_args, **_kwargs: None, lambda *_args, **_kwargs: None)

    def preflight(*_args, **_kwargs):
        record("preflight")
        return tracked(
            "preflight",
            events={
                "execution_profile": collector.LITERAL_T4_MATERIALIZATION_PROFILE,
                "maximum_coverage_covariance_scale": 1.0,
                "selected_anchor_attribute_loo_collect_only": True,
                "selected_anchor_attribute_loo_frozen_guard": None,
                "selected_anchor_attribute_loo_aggregate": {
                    "maximum_held_out_risks": [0.125]
                },
                "coverage_certificate": collector.LITERAL_T4_MOMENT_CERTIFICATE,
                "coverage_certificate_sha256": _digest(5),
                "update_binding": {"slots_sha256": _digest(6)},
                "source_nonprobe_s3_attribute_reads": 0,
            },
            tile_trace=(
                {
                    "view": 0,
                    "tile_y": 0,
                    "tile_x": 0,
                    "selected_anchor_attribute_loo": None,
                },
            ),
            update_dense_slots=torch.tensor([0]),
        )

    def final_route(*_args, **_kwargs):
        record("final_route")
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        return tracked(
            "final_route",
            raw_head_request_mask=mask,
            full_passthrough_mask=mask.clone(),
            selected_output_mask=mask.clone(),
            events={"full_passthrough_mask_sha256": _digest(7)},
        )

    def subset_packet(*_args, **_kwargs):
        record("subset_packet")
        return tracked("final_packet")

    def compare_full(*_args, **_kwargs):
        record("compare_full")
        return {"bitwise_equivalent": True}

    def materialize(*_args, **_kwargs):
        record("materialize")
        return tracked("materialized", attribute_binding_sha256=_digest(8))

    @contextmanager
    def strict_fp32():
        record("strict_enter")
        try:
            yield
        finally:
            record("strict_exit")

    def empty_cache():
        gc.collect()
        cache_liveness.append(
            sorted(name for name, reference in references.items() if reference() is not None)
        )

    encoder = SimpleNamespace(gaussian_head=object(), gaussian_adapter=object())
    runtime = collector.LiteralT4CollectionRuntime(
        bundle=SimpleNamespace(
            config=SimpleNamespace(
                dataset=object(), model=SimpleNamespace(encoder=object())
            ),
            device=SimpleNamespace(type="cuda"),
            encoder=encoder,
        ),
        backend_identity={},
        checkpoint_sha256=_digest(9),
    )
    monkeypatch.setattr(collector, "prepare_acid_joint_model_context", prepare)
    monkeypatch.setattr(collector, "strict_fp32_convolution_execution", strict_fp32)
    monkeypatch.setattr(collector, "capture_depthsplat_native_execution", capture)
    monkeypatch.setattr(collector, "build_literal_paper_t4_probe_first_plan", build_plan)
    monkeypatch.setattr(collector, "replay_depthsplat_selected_head", replay)
    monkeypatch.setattr(collector, "build_depthsplat_sparse_raw_packet", build_packet)
    monkeypatch.setattr(collector, "DepthSplatPackedGaussianConsumer", Consumer)
    monkeypatch.setattr(collector, "compare_depthsplat_packed_to_dense", compare_packed)
    monkeypatch.setattr(collector, "_source_geometry_functions", source_geometry)
    monkeypatch.setattr(collector, "preflight_depthsplat_l0_l1_materialization", preflight)
    monkeypatch.setattr(collector, "resolve_depthsplat_compact_final_route", final_route)
    monkeypatch.setattr(collector, "subset_depthsplat_sparse_raw_packet", subset_packet)
    monkeypatch.setattr(
        collector, "compare_depthsplat_full_passthrough_to_dense_bitwise", compare_full
    )
    monkeypatch.setattr(collector, "apply_depthsplat_compact_l0_l1_materialization", materialize)
    monkeypatch.setattr(collector.torch.cuda, "empty_cache", empty_cache)

    observation = collector.collect_literal_t4_native_scene_observation(
        runtime=runtime,
        materialization_root=Path("materialization"),
        plan_path=Path("plan.json"),
        split=collector.TRAIN_SPLIT,
        sample_index=0,
        scene="scene-0",
        context_loader=context_loader,
    )

    assert observation["maximum_held_out_risks"] == [0.125]
    assert observation["preflight_tile_trace"][0]["view"] == 0
    assert execution_calls
    assert all(not grad_enabled for _, grad_enabled in execution_calls)
    assert cache_liveness == [[]]
    gc.collect()
    assert all(reference() is None for reference in references.values())
