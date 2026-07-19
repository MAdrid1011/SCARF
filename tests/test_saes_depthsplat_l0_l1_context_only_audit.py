"""CPU contracts for the formal DepthSplat context-only materialization audit."""

from __future__ import annotations

from copy import deepcopy

import pytest


torch = pytest.importorskip("torch")


def _sha(value: str) -> str:
    return value * 64


def _identity(*, sample_index: int = 0):
    return {
        "scene": "scene-fixed",
        "context_indices": [0, 9],
        "tree_sha256": _sha("a"),
        "manifest_sha256": _sha("b"),
        "audit_input_sha256": _sha("c"),
        "source_sample_index": sample_index,
        "source_binding": {
            "canonical_index_sha256": _sha("d"),
            "canonical_sample_selection_sha256": _sha("e"),
            "canonical_selection_sha256": _sha("f"),
            "source_audit_input_sha256": _sha("1"),
            "source_audit_tree_sha256": _sha("2"),
            "source_sidecar_tree_sha256": _sha("3"),
        },
        "sidecar": {"index_sha256": _sha("4"), "record_sha256": _sha("5")},
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
    }


def _context():
    return {
        "image": torch.zeros(1, 2, 3, 4, 8),
        "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
        "near": torch.full((1, 2), 0.1),
        "far": torch.full((1, 2), 100.0),
        "index": torch.tensor([[0, 9]], dtype=torch.long),
    }


def _batch(identity):
    return {
        "context": _context(),
        "scene": [identity["scene"]],
        "calibration": {
            **identity,
            "target_mapping_present": False,
            "native_preprocessing": {
                "crop_image_shape": [4, 8],
                "patch_size": 1,
                "prepared_image_shape": [4, 8],
            },
        },
    }


def test_formal_context_identity_accepts_only_source_sample_zero():
    from scripts.saes_depthsplat_l0_l1_context_only_audit import (
        _require_formal_context_identity,
    )

    assert _require_formal_context_identity(_identity())["source_sample_index"] == 0
    with pytest.raises(ValueError, match="sample zero"):
        _require_formal_context_identity(_identity(sample_index=1))


def test_formal_context_identity_rejects_target_access_metadata():
    from scripts.saes_depthsplat_l0_l1_context_only_audit import (
        _require_formal_context_identity,
    )

    identity = _identity()
    identity["target_camera_metadata_accessed"] = True
    with pytest.raises(ValueError, match="context-only"):
        _require_formal_context_identity(identity)


def test_loaded_context_batch_rejects_a_target_mapping_and_identity_drift():
    from scripts.saes_depthsplat_l0_l1_context_only_audit import (
        _require_loaded_context_batch,
    )

    identity = _identity()
    context, calibration = _require_loaded_context_batch(_batch(identity), identity)
    assert set(context) == {"image", "extrinsics", "intrinsics", "near", "far", "index"}
    assert calibration["target_mapping_present"] is False

    target_bearing = _batch(identity)
    target_bearing["target"] = {"image": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="target mapping"):
        _require_loaded_context_batch(target_bearing, identity)

    drifted = _batch(identity)
    drifted["calibration"] = deepcopy(drifted["calibration"])
    drifted["calibration"]["source_sample_index"] = 1
    with pytest.raises(RuntimeError, match="changed input identity"):
        _require_loaded_context_batch(drifted, identity)


def test_formal_audit_parser_exposes_no_sample_or_target_override():
    from scripts.saes_depthsplat_l0_l1_context_only_audit import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "result"])
    args = parser.parse_args(
        [
            "--output-dir",
            "result",
            "--literal-t4-v16-record",
            "frozen/v16t4.json",
        ]
    )

    assert not hasattr(args, "sample_index")
    assert not hasattr(args, "model")
    assert args.literal_t4_v16_record.name == "v16t4.json"


def test_literal_guard_is_live_loaded_before_native_audit_work(monkeypatch, tmp_path):
    import saes.depthsplat_literal_t4_acid_calibration as calibration
    import scripts.saes_depthsplat_l0_l1_context_only_audit as audit

    profile = audit.literal_t4_profile()
    guard = calibration._issue_verified_literal_t4_materializer_guard({
        "schema_version": audit.LITERAL_T4_GUARD_SCHEMA,
        "frozen_record_kind": audit.V16T4_KIND,
        "frozen_record_sha256": _sha("a"),
        "threshold_value": 0.25,
        "threshold_rule": "train-minimum-per-scene-q25",
        "risk_metric": "maximum-held-out-anchor-risk-v1",
        "materialization_profile": audit.LITERAL_T4_MATERIALIZATION_PROFILE,
        "route_plan_contract": audit.LITERAL_PAPER_T4_PLAN_CONTRACT,
        "route_plan_config_sha256": profile["route_plan_config_sha256"],
        "acid_binding_sha256": _sha("b"),
        "application_sha256": _sha("c"),
    })
    calls = []

    def fake_loader(path, *, root, plan_path, materialization_root):
        calls.append((path, root, plan_path, materialization_root))
        return guard

    monkeypatch.setattr(audit, "to_materializer_guard", fake_loader)
    loaded, loaded_profile = audit._load_literal_t4_v16_audit_guard(
        record_path=tmp_path / "v16t4.json",
        acid_plan_path=tmp_path / "plan.json",
        acid_materialization_root=tmp_path / "materialization",
    )

    assert loaded is guard
    assert loaded_profile == profile
    assert calls == [
        (
            tmp_path / "v16t4.json",
            audit.ROOT,
            tmp_path / "plan.json",
            tmp_path / "materialization",
        )
    ]


def test_literal_guard_rejects_an_adaptive_profile_before_model_loading(monkeypatch, tmp_path):
    import saes.depthsplat_literal_t4_acid_calibration as calibration
    import scripts.saes_depthsplat_l0_l1_context_only_audit as audit

    guard = calibration._issue_verified_literal_t4_materializer_guard({
        "schema_version": audit.LITERAL_T4_GUARD_SCHEMA,
        "frozen_record_kind": audit.V16T4_KIND,
        "frozen_record_sha256": _sha("a"),
        "threshold_value": 0.25,
        "threshold_rule": "train-minimum-per-scene-q25",
        "risk_metric": "maximum-held-out-anchor-risk-v1",
        "materialization_profile": "depthsplat-development-omitted-z-alpha-union-v1",
        "route_plan_contract": audit.LITERAL_PAPER_T4_PLAN_CONTRACT,
        "route_plan_config_sha256": audit.literal_t4_profile()[
            "route_plan_config_sha256"
        ],
        "acid_binding_sha256": _sha("b"),
        "application_sha256": _sha("c"),
    })
    monkeypatch.setattr(audit, "to_materializer_guard", lambda *_args, **_kwargs: guard)

    with pytest.raises(ValueError, match="literal T=4 V16 guard changed"):
        audit._load_literal_t4_v16_audit_guard(
            record_path=tmp_path / "v16t4.json",
            acid_plan_path=tmp_path / "plan.json",
            acid_materialization_root=tmp_path / "materialization",
        )


def test_literal_guard_rejects_a_plain_mapping_before_model_loading(monkeypatch, tmp_path):
    import scripts.saes_depthsplat_l0_l1_context_only_audit as audit

    guard = {
        "schema_version": audit.LITERAL_T4_GUARD_SCHEMA,
        "frozen_record_kind": audit.V16T4_KIND,
        "frozen_record_sha256": _sha("a"),
        "threshold_value": 0.25,
        "threshold_rule": "train-minimum-per-scene-q25",
        "risk_metric": "maximum-held-out-anchor-risk-v1",
        "materialization_profile": audit.LITERAL_T4_MATERIALIZATION_PROFILE,
        "route_plan_contract": audit.LITERAL_PAPER_T4_PLAN_CONTRACT,
        "route_plan_config_sha256": audit.literal_t4_profile()[
            "route_plan_config_sha256"
        ],
        "acid_binding_sha256": _sha("b"),
        "application_sha256": _sha("c"),
    }
    monkeypatch.setattr(audit, "to_materializer_guard", lambda *_args, **_kwargs: guard)

    with pytest.raises(ValueError, match="not authenticated"):
        audit._load_literal_t4_v16_audit_guard(
            record_path=tmp_path / "v16t4.json",
            acid_plan_path=tmp_path / "plan.json",
            acid_materialization_root=tmp_path / "materialization",
        )
