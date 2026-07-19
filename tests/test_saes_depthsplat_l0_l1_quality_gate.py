"""CPU contracts for the fixed DepthSplat target-RGB quality gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _sha(character: str) -> str:
    return character * 64


def _identity(gate):
    context_indices = [0, 9]
    target_indices = [1, 3, 5, 7]
    selection = {
        "source_sample_index": 0,
        "scene": "scene-fixed",
        "context_indices": context_indices,
        "target_indices": target_indices,
    }
    return {
        "scene": "scene-fixed",
        "context_indices": context_indices,
        "source_binding": {
            "canonical_selection_sha256": gate.canonical_json_sha256(selection)
        },
    }


def _formal_audit(gate, identity, literal_guard, literal_profile, backend):
    expected_literal = gate._expected_literal_binding(
        literal_guard=literal_guard, literal_profile=literal_profile
    )
    boundary = {
        "encoder_only": True,
        "decoder_constructed": False,
        "renderer_executed": False,
        "target_view_rendered": False,
        "quality_metrics_computed": False,
        "timing_claim": False,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "native_dense_depth_predictor_executed": True,
        "native_dense_gaussian_regressor_executed": True,
        "initial_selected_native_rgb_adapter_executed": True,
        "final_selected_native_rgb_adapter_executed": True,
        "nonzero_l0_l1_materializer_executed": True,
    }
    return {
        "schema_version": gate.FORMAL_AUDIT_SCHEMA_VERSION,
        "kind": gate.FORMAL_AUDIT_KIND,
        "status": "PASS",
        "paper_result_eligible": False,
        "formal_target_free_audit": True,
        "formal_target_free_sidecar_used": True,
        "model": gate.MODEL,
        "dataset": gate.DATASET,
        "source_sample_index": gate.SOURCE_SAMPLE_INDEX,
        "scene": identity["scene"],
        "context_indices": identity["context_indices"],
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "execution_boundary": boundary,
        "context_only_input": {
            "identity": identity,
            "loaded_native_preprocessing": {"patch_size": 1},
        },
        "literal_t4_v16": expected_literal,
        "backend_identity": backend,
        "checkpoint_sha256": _sha("c"),
        "route_plan": {},
        "materialization_preflight": {},
        "materialization_preflight_tile_trace": [],
        "final_route": {},
        "native_execution": {},
        "initial_selected_head": {},
        "producer_selected_head": {},
        "initial_packet": {},
        "final_packet": {},
        "materialized_packet": {},
        "geometry_source": {},
    }


def test_quality_gate_parser_is_fixed_to_the_registered_route():
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    parser = gate.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "result"])
    args = parser.parse_args(
        [
            "--output-dir",
            "result",
            "--formal-audit",
            "audit/results.json",
            "--literal-t4-v16-record",
            "frozen/v16t4.json",
        ]
    )

    assert args.formal_audit.name == "results.json"
    assert args.literal_t4_v16_record.name == "v16t4.json"
    assert not hasattr(args, "sample_index")
    assert not hasattr(args, "model")
    assert not hasattr(args, "seed")


def test_formal_audit_must_remain_target_free_before_quality(monkeypatch):
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    identity = _identity(gate)
    literal_guard = {
        "frozen_record_sha256": _sha("a"),
        "frozen_record_kind": "literal-v16",
        "threshold_value": 0.25,
        "threshold_rule": "fixed",
        "risk_metric": "risk",
        "acid_binding_sha256": _sha("b"),
        "application_sha256": _sha("d"),
    }
    literal_profile = {"materialization_profile": "literal"}
    backend = {"model": "depthsplat"}
    audit = _formal_audit(gate, identity, literal_guard, literal_profile, backend)
    monkeypatch.setattr(gate, "_read_formal_audit", lambda _path: (audit, {}))
    monkeypatch.setattr(gate, "_require_formal_audit_runner", lambda _audit: None)

    accepted, _binding = gate._validate_formal_audit(
        "audit/results.json",
        input_identity=identity,
        literal_guard=literal_guard,
        literal_profile=literal_profile,
        backend_identity=backend,
        checkpoint_sha256=_sha("c"),
    )
    assert accepted is audit

    audit["target_rgb_accessed"] = True
    with pytest.raises(ValueError, match="target_rgb_accessed"):
        gate._validate_formal_audit(
            "audit/results.json",
            input_identity=identity,
            literal_guard=literal_guard,
            literal_profile=literal_profile,
            backend_identity=backend,
            checkpoint_sha256=_sha("c"),
        )


def test_native_target_batch_must_match_audited_selection():
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    identity = _identity(gate)
    target = {
        "index": torch.tensor([[1, 3, 5, 7]], dtype=torch.long),
        "image": torch.zeros(1, 4, 3, 2, 2),
    }
    batch = {
        "scene": ["scene-fixed"],
        "context": {"index": torch.tensor([[0, 9]], dtype=torch.long)},
        "target": target,
    }
    loader = SimpleNamespace(
        load_data=lambda *_args, **_kwargs: SimpleNamespace(batch=batch)
    )

    loaded_target, target_indices = gate._load_native_target_batch_after_packet_commit(
        loader,
        object(),
        scene="scene-fixed",
        context_indices=[0, 9],
        input_identity=identity,
    )
    assert loaded_target is target
    assert target_indices == [1, 3, 5, 7]

    drifted = {
        "scene": ["scene-fixed"],
        "context": {"index": torch.tensor([[0, 9]], dtype=torch.long)},
        "target": {
            "index": torch.tensor([[1, 3, 5, 8]], dtype=torch.long),
            "image": torch.zeros(1, 4, 3, 2, 2),
        },
    }
    loader = SimpleNamespace(
        load_data=lambda *_args, **_kwargs: SimpleNamespace(batch=drifted)
    )
    with pytest.raises(RuntimeError, match="selection differs"):
        gate._load_native_target_batch_after_packet_commit(
            loader,
            object(),
            scene="scene-fixed",
            context_indices=[0, 9],
            input_identity=identity,
        )


def test_quality_gate_requires_an_actual_nonzero_merge():
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    empty = SimpleNamespace(update_dense_slots=torch.empty(0, dtype=torch.long))
    with pytest.raises(RuntimeError, match="actual compact nonzero merge"):
        gate._require_nonzero_merge(empty)

    nonempty = SimpleNamespace(update_dense_slots=torch.tensor([7], dtype=torch.long))
    assert gate._require_nonzero_merge(nonempty) == 1
