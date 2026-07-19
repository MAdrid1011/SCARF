"""Focused contracts for the independent literal DepthSplat V16T4 record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _digest(number: int) -> str:
    return f"{number:064x}"


def _access():
    return {
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "skipped_s3_attributes_accessed": False,
    }


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
        "protocol_id": "acid-literal-t4-test-v1",
        "plan_sha256": _digest(10),
        "plan_file_sha256": _digest(11),
        "materialization_record_sha256": _digest(12),
        "materialization_tree_sha256": _digest(13),
        "materialization_manifest_sha256": _digest(14),
        "splits": {
            "calibration_train": split(train, 20),
            "calibration_holdout": split(holdout, 30),
        },
        "access": _access(),
    }


def _application(module):
    return module.build_literal_t4_application(
        backend_identity={"test_backend_identity_sha256": _digest(40)}
    )


def _loo_observation(risk: float):
    positions = ([0, 0], [0, 3], [3, 0], [3, 3])
    return {
        "checked": True,
        "scorable": True,
        "status": "scored",
        "reason": None,
        "certificate": "depthsplat-selected-rgb-sh-opacity-all-anchor-loo-v1",
        "policy": "all-retained-l0-l1-anchors-selected-labels-only-v1",
        "risk_metric": "maximum-held-out-anchor-risk-v1",
        "level": "L0",
        "anchor_count": 4,
        "held_out_anchor_count": 4,
        "held_out_anchor_records": [
            {
                "held_out_local_position": list(position),
                "harmonic_relative_error": risk,
                "opacity_logit_relative_error": risk / 3.0,
                "risk": risk,
            }
            for position in positions
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


def _preflight_trace(risks):
    return [
        {
            "view": 0,
            "tile_y": 0,
            "tile_x": index,
            "planned_route": "L0",
            "attempted": True,
            "accepted": True,
            "reason": "accepted",
            "source_nonprobe_s3_attribute_reads": 0,
            "selected_anchor_attribute_loo": _loo_observation(risk),
        }
        for index, risk in enumerate(risks)
    ]


def _artifact(module, *, scene: str, split: str, risks):
    trace = _preflight_trace(risks)
    aggregate = module._aggregate_from_preflight_tile_trace(trace)
    return module.build_v16t4_trace_artifact(
        scene=scene,
        split=split,
        preflight_tile_trace=trace,
        selected_anchor_loo_aggregate=aggregate,
    )


def _artifact_reference(module, *, scene: str, split: str, artifact):
    aggregate = artifact["selected_anchor_loo_aggregate"]
    return {
        "relative_path": f"scenes/{split}/{scene}.json",
        "sha256": artifact["sha256"],
        "preflight_tile_trace_sha256": aggregate["preflight_tile_trace_sha256"],
        "tile_records_sha256": aggregate["tile_records_sha256"],
    }


def _write_artifact(root: Path, reference, artifact):
    path = root / reference["relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _evidence(module, number: int, *, aggregate, reference):
    return {
        "profile_sha256": module.literal_t4_profile_sha256(),
        "route_plan_config_sha256": module.literal_t4_profile()[
            "route_plan_config_sha256"
        ],
        "native_execution_sha256": _digest(1000 + number),
        "initial_attribute_binding_sha256": _digest(2000 + number),
        "final_selected_attribute_binding_sha256": _digest(3000 + number),
        "materialized_attribute_binding_sha256": _digest(4000 + number),
        "full_passthrough_mask_sha256": _digest(5000 + number),
        "full_attribute_binding_sha256": _digest(6000 + number),
        "full_attributes_bitwise_native": True,
        "coverage_certificate": module.LITERAL_T4_COVERAGE_CERTIFICATE,
        "coverage_certificate_sha256": _digest(7000 + number),
        "accepted_update_slots_sha256": _digest(8000 + number),
        "accepted_update_slot_count": 1,
        "selected_anchor_loo_aggregate": aggregate,
        "selected_anchor_loo_aggregate_sha256": module.canonical_sha256(aggregate),
        "selected_anchor_loo_trace_sha256": aggregate[
            "preflight_tile_trace_sha256"
        ],
        "trace_artifact": reference,
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_merge_applied": True,
        "renderer_executed": False,
        "quality_metrics_computed": False,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
    }


def _scene_record(module, *, scene: str, split: str, number: int, risks, artifact_root=None):
    artifact = _artifact(module, scene=scene, split=split, risks=risks)
    reference = _artifact_reference(
        module, scene=scene, split=split, artifact=artifact
    )
    if artifact_root is not None:
        _write_artifact(Path(artifact_root), reference, artifact)
    observed_risks = artifact["selected_anchor_loo_aggregate"][
        "maximum_held_out_risks"
    ]
    return {
        "scene": scene,
        "maximum_held_out_risks": observed_risks,
        "evidence": _evidence(
            module,
            number,
            aggregate=artifact["selected_anchor_loo_aggregate"],
            reference=reference,
        ),
        "access": _access(),
    }


def _records(module, binding, *, train_offset=0.0, holdout_offset=1.0, artifact_root=None):
    records = {}
    for split, offset, number_offset in (
        (module.TRAIN_SPLIT, train_offset, 0),
        (module.HOLDOUT_SPLIT, holdout_offset, 100),
    ):
        scene_records = []
        for index, scene in enumerate(binding["splits"][split]["scenes"]):
            values = [
                offset + 0.10 + index / 100.0,
                offset + 0.20 + index / 100.0,
                offset + 0.80 + index / 100.0,
            ]
            scene_records.append(
                _scene_record(
                    module,
                    scene=scene,
                    split=split,
                    number=number_offset + index,
                    risks=values,
                    artifact_root=artifact_root,
                )
            )
        records[split] = scene_records
    return records


def _record(module, binding=None, *, train_offset=0.0, holdout_offset=1.0, artifact_root=None):
    binding = _binding(module) if binding is None else binding
    records = _records(
        module,
        binding,
        train_offset=train_offset,
        holdout_offset=holdout_offset,
        artifact_root=artifact_root,
    )
    return module.build_v16t4_record(
        binding=binding,
        application=_application(module),
        profile=module.literal_t4_profile(),
        train_scene_records=records[module.TRAIN_SPLIT],
        holdout_scene_records=records[module.HOLDOUT_SPLIT],
    )


def _rehash(module, record):
    record["sha256"] = module.canonical_sha256(
        {key: value for key, value in record.items() if key != "sha256"}
    )


def _rehash_artifact(module, artifact_path: Path, evidence, artifact):
    artifact["sha256"] = module.canonical_sha256(
        {key: value for key, value in artifact.items() if key != "sha256"}
    )
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    evidence["trace_artifact"]["sha256"] = artifact["sha256"]


def _write_record(root: Path, record):
    path = root / "v16t4.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _patch_live(module, monkeypatch, binding):
    monkeypatch.setattr(module, "resolve_depthsplat_acid_binding", lambda **_kwargs: binding)
    monkeypatch.setattr(
        module, "_validate_live_application", lambda application, **_kwargs: application
    )


def _load(module, tmp_path: Path, monkeypatch, record, binding):
    path = _write_record(tmp_path, record)
    _patch_live(module, monkeypatch, binding)
    return module.load_frozen_v16t4_threshold(path)


def test_literal_profile_rejects_legacy_kind_and_adaptive_profile(tmp_path: Path, monkeypatch):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    records = _records(calibration, binding)
    legacy_profile = calibration.literal_t4_profile()
    legacy_profile["l1_anchor_semantics"] = "engineering-lightweight-15-adaptive-center-s1-loo-v1"
    with pytest.raises(ValueError, match="profile"):
        calibration.build_v16t4_record(
            binding=binding,
            application=_application(calibration),
            profile=legacy_profile,
            train_scene_records=records[calibration.TRAIN_SPLIT],
            holdout_scene_records=records[calibration.HOLDOUT_SPLIT],
        )

    record = _record(calibration, binding, artifact_root=tmp_path)
    record["kind"] = "depthsplat-nonzero-l0-l1-acid-disjoint-v16d-schema-gap"
    _rehash(calibration, record)
    with pytest.raises(ValueError, match="kind or schema"):
        _load(calibration, tmp_path, monkeypatch, record, binding)


def test_threshold_recomputes_from_train_scene_q25_only(tmp_path: Path, monkeypatch):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    expected_q25 = [
        values["maximum_held_out_risks"][0]
        for values in record["train_scene_records"]
    ]
    assert record["threshold"]["per_scene_q25"] == expected_q25
    assert record["threshold"]["value"] == min(expected_q25)

    replacement = _scene_record(
        calibration,
        scene=binding["splits"][calibration.TRAIN_SPLIT]["scenes"][0],
        split=calibration.TRAIN_SPLIT,
        number=0,
        risks=[9.0, 9.1, 9.2],
        artifact_root=tmp_path,
    )
    record["train_scene_records"][0] = replacement
    _rehash(calibration, record)
    with pytest.raises(ValueError, match="not reproducible"):
        _load(calibration, tmp_path, monkeypatch, record, binding)


def test_holdout_cannot_update_train_threshold():
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    baseline = _record(calibration, binding, holdout_offset=1.0)
    changed_holdout = _record(calibration, binding, holdout_offset=0.0)

    assert changed_holdout["threshold"] == baseline["threshold"]
    assert changed_holdout["holdout_verification"]["threshold_updated"] is False
    assert changed_holdout["holdout_verification"]["candidate_count"] == 24
    assert changed_holdout["holdout_verification"]["retained_count"] > 0


def test_loader_rejects_self_hash_and_profile_tampering(tmp_path: Path, monkeypatch):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    record["threshold"]["value"] = 123.0
    with pytest.raises(ValueError, match="SHA256"):
        _load(calibration, tmp_path, monkeypatch, record, binding)

    record = _record(calibration, binding, artifact_root=tmp_path)
    record["profile"]["depth_threshold"] = 0.5
    _rehash(calibration, record)
    with pytest.raises(ValueError, match="profile"):
        _load(calibration, tmp_path, monkeypatch, record, binding)


def test_loader_rejects_rehashed_trace_reference_drift(tmp_path: Path, monkeypatch):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    record["train_scene_records"][0]["evidence"]["trace_artifact"][
        "tile_records_sha256"
    ] = _digest(99999)
    _rehash(calibration, record)
    with pytest.raises(ValueError, match="LOO aggregate"):
        _load(calibration, tmp_path, monkeypatch, record, binding)


def test_loader_rejects_trace_payload_tampering_after_all_hashes_recomputed(
    tmp_path: Path, monkeypatch
):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    evidence = record["train_scene_records"][0]["evidence"]
    artifact_path = tmp_path / evidence["trace_artifact"]["relative_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["preflight_tile_trace"][0]["selected_anchor_attribute_loo"][
        "maximum_held_out_risk"
    ] = 5.0
    _rehash_artifact(calibration, artifact_path, evidence, artifact)
    _rehash(calibration, record)
    with pytest.raises(ValueError, match="trace selected-anchor risks changed"):
        _load(calibration, tmp_path, monkeypatch, record, binding)


@pytest.mark.parametrize(
    ("field", "match"),
    (
        ("risk", "not the maximum attribute error"),
        ("q75_risk", "selected-anchor risks changed"),
    ),
)
def test_loader_rejects_rehashed_99_risk_and_q75_tampering(
    tmp_path: Path, monkeypatch, field: str, match: str
):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    evidence = record["train_scene_records"][0]["evidence"]
    artifact_path = tmp_path / evidence["trace_artifact"]["relative_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    observation = artifact["preflight_tile_trace"][0]["selected_anchor_attribute_loo"]
    if field == "risk":
        observation["held_out_anchor_records"][0]["risk"] = 99.0
    else:
        observation[field] = 99.0
    _rehash_artifact(calibration, artifact_path, evidence, artifact)
    _rehash(calibration, record)
    with pytest.raises(ValueError, match=match):
        _load(calibration, tmp_path, monkeypatch, record, binding)


def test_loader_rejects_rehashed_noncorner_held_out_anchor(tmp_path: Path, monkeypatch):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    evidence = record["train_scene_records"][0]["evidence"]
    artifact_path = tmp_path / evidence["trace_artifact"]["relative_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["preflight_tile_trace"][0]["selected_anchor_attribute_loo"][
        "held_out_anchor_records"
    ][0]["held_out_local_position"] = [1, 1]
    _rehash_artifact(calibration, artifact_path, evidence, artifact)
    _rehash(calibration, record)
    with pytest.raises(ValueError, match="four corners"):
        _load(calibration, tmp_path, monkeypatch, record, binding)


def test_trace_q75_recomputes_with_materializer_float32_semantics():
    torch = pytest.importorskip("torch")
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    risks = [0.1, 0.2, 0.8, 1.3]
    observation = _loo_observation(risks[0])
    for held_out, risk in zip(observation["held_out_anchor_records"], risks):
        held_out["harmonic_relative_error"] = risk
        held_out["opacity_logit_relative_error"] = risk / 2.0
        held_out["risk"] = risk
    expected_q75 = torch.quantile(
        torch.tensor(risks, dtype=torch.float32), 0.75
    ).item()
    # Both producer values are deliberately distinct from the canonical
    # float32 values, but still within the accepted JSON serialization slack.
    observation["maximum_held_out_risk"] = 1.3000000000000003
    observation["q75_risk"] = 0.9250000119209291

    kind, maximum, q75, labels, endpoints = calibration._trace_loo_observation(
        observation, planned_route="L0"
    )
    assert kind == "scorable"
    assert maximum == torch.tensor(risks, dtype=torch.float32).max().item()
    assert q75 == expected_q75
    assert labels == 4
    assert endpoints == 0


def test_loader_rejects_trace_artifact_traversal_path(tmp_path: Path, monkeypatch):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    record["train_scene_records"][0]["evidence"]["trace_artifact"][
        "relative_path"
    ] = "../outside.json"
    _rehash(calibration, record)
    with pytest.raises(ValueError, match="safely relative"):
        _load(calibration, tmp_path, monkeypatch, record, binding)


def test_loader_rejects_final_trace_artifact_symlink(tmp_path: Path, monkeypatch):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    evidence = record["train_scene_records"][0]["evidence"]
    artifact_path = tmp_path / evidence["trace_artifact"]["relative_path"]
    target = tmp_path / "valid_trace_target.json"
    target.write_text(artifact_path.read_text(encoding="utf-8"), encoding="utf-8")
    artifact_path.unlink()
    artifact_path.symlink_to(target)
    with pytest.raises(ValueError, match="contains a symlink"):
        _load(calibration, tmp_path, monkeypatch, record, binding)


def test_loader_rejects_intermediate_trace_artifact_symlink(tmp_path: Path, monkeypatch):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    train_directory = tmp_path / "scenes" / calibration.TRAIN_SPLIT
    target = tmp_path / "real_train_artifacts"
    train_directory.rename(target)
    train_directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="contains a symlink"):
        _load(calibration, tmp_path, monkeypatch, record, binding)


def test_path_guard_runs_live_load_and_binds_full_identity(tmp_path: Path, monkeypatch):
    import saes.depthsplat_literal_t4_acid_calibration as calibration
    from saes.depthsplat_l0_l1_materializer import (
        _selected_anchor_attribute_loo_frozen_guard,
    )

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    path = _write_record(tmp_path, record)
    _patch_live(calibration, monkeypatch, binding)
    guard = calibration.to_materializer_guard(path)

    assert set(guard) == {
        "schema_version",
        "frozen_record_kind",
        "frozen_record_sha256",
        "threshold_value",
        "threshold_rule",
        "risk_metric",
        "materialization_profile",
        "route_plan_contract",
        "route_plan_config_sha256",
        "acid_binding_sha256",
        "application_sha256",
    }
    assert guard["frozen_record_kind"] == calibration.V16T4_KIND
    assert guard["frozen_record_sha256"] == record["sha256"]
    assert guard["route_plan_config_sha256"] == record["profile"]["route_plan_config_sha256"]
    assert guard["acid_binding_sha256"] == calibration.canonical_sha256(
        record["acid_binding"]
    )
    assert guard["application_sha256"] == calibration.canonical_sha256(
        record["application"]
    )
    assert isinstance(guard, calibration.VerifiedLiteralT4MaterializerGuard)
    assert _selected_anchor_attribute_loo_frozen_guard(
        guard,
        require_literal_t4_authenticated=True,
    ) == dict(guard)
    with pytest.raises(TypeError, match="issued by the verified loader"):
        calibration.VerifiedLiteralT4MaterializerGuard(dict(guard))
    with pytest.raises(TypeError, match="authenticated V16T4 guard"):
        calibration.verified_literal_t4_materializer_guard_projection(dict(guard))
    with pytest.raises(TypeError, match="record path"):
        calibration.to_materializer_guard(record)


def test_path_guard_binds_only_the_fixed_literal_0_2_0_1_route(
    tmp_path: Path, monkeypatch
):
    torch = pytest.importorskip("torch")
    import saes.depthsplat_literal_t4_acid_calibration as calibration
    from saes.probe_first_schedule import (
        build_literal_paper_t4_probe_first_plan,
        literal_paper_t4_route_config_sha256,
    )

    binding = _binding(calibration)
    record = _record(calibration, binding, artifact_root=tmp_path)
    path = _write_record(tmp_path, record)
    _patch_live(calibration, monkeypatch, binding)
    guard = calibration.to_materializer_guard(path)
    features = torch.zeros((1, 1, 3, 4, 4), dtype=torch.float32)
    z_depths = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    fixed = build_literal_paper_t4_probe_first_plan(
        features,
        z_depths,
        height=4,
        width=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
    )
    drifted = build_literal_paper_t4_probe_first_plan(
        features,
        z_depths,
        height=4,
        width=4,
        feature_threshold=0.1,
        depth_threshold=0.1,
    )
    assert guard["route_plan_contract"] == fixed.events["contract_version"]
    assert guard["route_plan_config_sha256"] == literal_paper_t4_route_config_sha256(
        fixed.events
    )
    assert guard["route_plan_config_sha256"] != literal_paper_t4_route_config_sha256(
        drifted.events
    )
