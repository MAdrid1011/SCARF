"""Tests for ACID-disjoint V15/V16 threshold records."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import saes.evaluation_disjoint_l1_calibration as calibration
from saes.incremental_selected_output_execution import (
    NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION,
    RAW_HEAD_EXECUTION_CONTRACT,
)


def _sha(character: str) -> str:
    return character * 64


def _native_dense_head_execution() -> dict:
    dense_positions = 32
    phases = []
    for index, phase in enumerate(("primary", "secondary", "full")):
        positions = dense_positions if index == 0 else 0
        phases.append(
            {
                "phase": phase,
                "mask_sha256": _sha("a" if index == 0 else "b" if index == 1 else "c"),
                "tile_trace_sha256": _sha(
                    "d" if index == 0 else "e" if index == 1 else "f"
                ),
                "first_conv_positions_executed": positions,
                "native_dense_first_conv_positions_executed": positions,
                "second_conv_positions_executed": positions,
                "native_dense_second_conv_positions_executed": positions,
            }
        )
    return {
        "schema_version": NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION,
        "raw_head_execution_contract": RAW_HEAD_EXECUTION_CONTRACT,
        "head_weight_sha256": _sha("0"),
        "head_input_sha256": _sha("1"),
        "phase_trace_sha256": _sha("2"),
        "tile_trace_sha256": _sha("3"),
        "execution_finalized": False,
        "dense_head_positions": dense_positions,
        "dense_head_macs": 4096,
        "actual_head_macs": 4096,
        "head_mac_delta": 0,
        "phases": phases,
    }


def _binding() -> dict:
    splits = {}
    for split, prefix, count, hashes in (
        (calibration.TRAIN_SPLIT, "train", 24, ("a", "b", "c", "d")),
        (calibration.HOLDOUT_SPLIT, "holdout", 8, ("e", "f", "0", "1")),
    ):
        scenes = [f"{prefix}-{index:02d}" for index in range(count)]
        splits[split] = {
            "scenes": scenes,
            "scene_count": count,
            "scene_set_sha256": calibration.canonical_sha256(sorted(scenes)),
            "selection_sha256": _sha(hashes[0]),
            "sidecar_tree_sha256": _sha(hashes[1]),
            "sidecar_manifest_sha256": _sha(hashes[2]),
            "input_provenance_sha256": _sha(hashes[3]),
        }
    return {
        "schema_version": "1.0",
        "kind": "saes-acid-context-only-calibration-binding",
        "dataset": "acid",
        "protocol_id": "saes-joint-materialization-calibration-acid-v1",
        "plan_sha256": _sha("c"),
        "plan_file_sha256": _sha("d"),
        "materialization_record_sha256": _sha("e"),
        "materialization_tree_sha256": _sha("f"),
        "materialization_manifest_sha256": _sha("0"),
        "splits": splits,
        "access": {
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "skipped_s3_attributes_accessed": False,
        },
    }


def _records(binding: dict, *, value_key: str, offset: float) -> tuple[list[dict], list[dict]]:
    access = binding["access"]

    def records(split: str, base: float) -> list[dict]:
        output = []
        for index, scene in enumerate(binding["splits"][split]["scenes"]):
            record = {
                "scene": scene,
                value_key: [base + index, base + index + 0.25, base + index + 0.5],
                "access": access,
            }
            if value_key == "risks":
                record["native_dense_head_execution"] = _native_dense_head_execution()
            output.append(record)
        return output

    return (
        records(calibration.TRAIN_SPLIT, offset),
        records(calibration.HOLDOUT_SPLIT, offset + 100.0),
    )


def _checkpoint(path: Path) -> str:
    path.write_bytes(b"re10k-checkpoint")
    return calibration.sha256_file(path)


def _mvsplat_backend_identity() -> dict:
    return {
        "schema_version": "classic-backend-frozen-identity-v1",
        "model": "mvsplat",
        "dataset": "dl3dv",
        "experiment": "re10k",
        "environment_profile": "classic",
        "raw_head_module": "encoder.depth_predictor.to_gaussians",
        "gaussian_adapter_module": "encoder.gaussian_adapter",
        "decoder_module": "decoder",
        "coordinate_semantics": "mvsplat-inline-pixel-center-plus-sigmoid-offset",
        "source_files": {
            "raw_head": {"path": "mvsplat/raw.py", "sha256": _sha("a")},
            "gaussian_adapter": {"path": "mvsplat/adapter.py", "sha256": _sha("b")},
            "decoder": {"path": "mvsplat/decoder.py", "sha256": _sha("c")},
            "coordinates": {"path": "mvsplat/coordinates.py", "sha256": _sha("d")},
        },
        "submodule_git_head": "e" * 40,
    }


def test_v15_uses_only_train_values_and_live_binding(tmp_path, monkeypatch):
    binding = _binding()
    train, holdout = _records(binding, value_key="residuals", offset=0.0)
    checkpoint = tmp_path / "re10k.ckpt"
    digest = _checkpoint(checkpoint)
    record = calibration.build_v15_record(
        binding=binding,
        application_checkpoint_sha256=digest,
        train_scene_records=train,
        holdout_scene_records=holdout,
    )
    assert record["threshold"]["value"] < 30.0
    assert record["holdout_verification"]["threshold_updated"] is False
    assert record["application"] == {
        "model": "transplat",
        "dataset": "dl3dv",
        "checkpoint_sha256": digest,
    }
    assert record["sha256"] == "f52689b0a33f467b638469be6ea5f36d65067ec3fb530df0d588ddd134d31b4c"
    path = tmp_path / "v15.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(calibration, "resolve_acid_binding", lambda **_kwargs: binding)

    loaded = calibration.load_frozen_v15_threshold(path, checkpoint_path=checkpoint)

    assert loaded["threshold_value"] == record["threshold"]["value"]
    assert len(loaded["train_scene_records"]) == 24
    assert len(loaded["holdout_scene_records"]) == 8


def test_application_model_is_frozen_and_fail_closed(tmp_path, monkeypatch):
    binding = _binding()
    train, holdout = _records(binding, value_key="residuals", offset=0.0)
    checkpoint = tmp_path / "re10k.ckpt"
    digest = _checkpoint(checkpoint)
    identity = _mvsplat_backend_identity()
    with pytest.raises(ValueError, match="requires a frozen classic backend identity"):
        calibration.build_v15_record(
            binding=binding,
            application_checkpoint_sha256=digest,
            application_model="mvsplat",
            train_scene_records=train,
            holdout_scene_records=holdout,
        )
    record = calibration.build_v15_record(
        binding=binding,
        application_checkpoint_sha256=digest,
        application_model="mvsplat",
        application_classic_backend_identity=identity,
        train_scene_records=train,
        holdout_scene_records=holdout,
    )
    assert record["application"] == {
        "model": "mvsplat",
        "dataset": "dl3dv",
        "checkpoint_sha256": digest,
        "classic_backend_identity": identity,
    }
    path = tmp_path / "mvsplat-v15.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(calibration, "resolve_acid_binding", lambda **_kwargs: binding)

    import saes.classic_backend as classic_backend

    contract = object()
    validated: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        classic_backend,
        "resolve_classic_backend_contract",
        lambda model, root: contract if model == "mvsplat" else None,
    )

    def validate_backend(actual_contract, actual_identity):
        validated.append((actual_contract, dict(actual_identity)))
        assert actual_contract is contract
        return identity

    monkeypatch.setattr(
        classic_backend,
        "validate_frozen_classic_backend_identity",
        validate_backend,
    )

    assert calibration.load_frozen_v15_threshold(
        path, checkpoint_path=checkpoint, application_model="mvsplat"
    )["sha256"] == record["sha256"]
    assert validated == [(contract, identity)]
    monkeypatch.setattr(
        classic_backend,
        "validate_frozen_classic_backend_identity",
        lambda *_args: (_ for _ in ()).throw(
            ValueError("frozen classic backend identity changed")
        ),
    )
    with pytest.raises(ValueError, match="classic backend identity changed"):
        calibration.load_frozen_v15_threshold(
            path, checkpoint_path=checkpoint, application_model="mvsplat"
        )
    with pytest.raises(ValueError, match="application checkpoint changed"):
        calibration.load_frozen_v15_threshold(path, checkpoint_path=checkpoint)
    with pytest.raises(ValueError, match="application model"):
        calibration.build_v15_record(
            binding=binding,
            application_checkpoint_sha256=digest,
            application_model="depthsplat",
            train_scene_records=train,
            holdout_scene_records=holdout,
        )


def test_records_reject_reordered_or_dl3dv_sample_records(tmp_path):
    binding = _binding()
    train, holdout = _records(binding, value_key="residuals", offset=0.0)
    checkpoint = _checkpoint(tmp_path / "re10k.ckpt")

    with pytest.raises(ValueError, match="scene order"):
        calibration.build_v15_record(
            binding=binding,
            application_checkpoint_sha256=checkpoint,
            train_scene_records=list(reversed(train)),
            holdout_scene_records=holdout,
        )

    poisoned = dict(train[0])
    poisoned["sample_index"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        calibration.build_v15_record(
            binding=binding,
            application_checkpoint_sha256=checkpoint,
            train_scene_records=[poisoned, *train[1:]],
            holdout_scene_records=holdout,
        )


def test_holdout_is_verification_only_and_threshold_schema_is_fail_closed(
    tmp_path, monkeypatch
):
    binding = _binding()
    train, holdout = _records(binding, value_key="residuals", offset=0.0)
    checkpoint_path = tmp_path / "re10k.ckpt"
    checkpoint_sha256 = _checkpoint(checkpoint_path)
    baseline = calibration.build_v15_record(
        binding=binding,
        application_checkpoint_sha256=checkpoint_sha256,
        train_scene_records=train,
        holdout_scene_records=holdout,
    )
    changed_holdout = copy.deepcopy(holdout)
    for record in changed_holdout:
        record["residuals"] = [value + 10_000.0 for value in record["residuals"]]
    verification_only = calibration.build_v15_record(
        binding=binding,
        application_checkpoint_sha256=checkpoint_sha256,
        train_scene_records=train,
        holdout_scene_records=changed_holdout,
    )
    assert verification_only["threshold"] == baseline["threshold"]
    assert verification_only["holdout_verification"] != baseline["holdout_verification"]

    tampered = copy.deepcopy(baseline)
    tampered["threshold"]["untrusted_extra"] = "ignored-by-legacy-loader"
    unsigned = dict(tampered)
    unsigned.pop("sha256")
    tampered["sha256"] = calibration.canonical_sha256(unsigned)
    path = tmp_path / "v15.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(calibration, "resolve_acid_binding", lambda **_kwargs: binding)
    with pytest.raises(ValueError, match="threshold has unexpected fields"):
        calibration.load_frozen_v15_threshold(path, checkpoint_path=checkpoint_path)


def test_v16_requires_the_verified_v15_parent_and_same_live_binding(tmp_path, monkeypatch):
    binding = _binding()
    checkpoint = tmp_path / "re10k.ckpt"
    digest = _checkpoint(checkpoint)
    v15_train, v15_holdout = _records(binding, value_key="residuals", offset=0.0)
    v15 = calibration.build_v15_record(
        binding=binding,
        application_checkpoint_sha256=digest,
        train_scene_records=v15_train,
        holdout_scene_records=v15_holdout,
    )
    v15_path = tmp_path / "v15.json"
    v15_path.write_text(json.dumps(v15), encoding="utf-8")
    v16_train, v16_holdout = _records(binding, value_key="risks", offset=0.1)
    v16 = calibration.build_v16_record(
        binding=binding,
        application_checkpoint_sha256=digest,
        v15_record_sha256=v15["sha256"],
        train_scene_records=v16_train,
        holdout_scene_records=v16_holdout,
    )
    v16_path = tmp_path / "v16.json"
    v16_path.write_text(json.dumps(v16), encoding="utf-8")
    monkeypatch.setattr(calibration, "resolve_acid_binding", lambda **_kwargs: binding)

    loaded = calibration.load_frozen_v16_threshold(
        v16_path, checkpoint_path=checkpoint, v15_record_path=v15_path
    )
    assert loaded["threshold_value"] == v16["threshold"]["value"]

    legacy = copy.deepcopy(v16)
    legacy.pop("raw_head_execution_contract")
    unsigned = dict(legacy)
    unsigned.pop("sha256")
    legacy["sha256"] = calibration.canonical_sha256(unsigned)
    v16_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected fields"):
        calibration.load_frozen_v16_threshold(
            v16_path, checkpoint_path=checkpoint, v15_record_path=v15_path
        )

    v16_path.write_text(json.dumps(v16), encoding="utf-8")

    v16["base_v15_sha256"] = _sha("1")
    v16_path.write_text(json.dumps(v16), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        calibration.load_frozen_v16_threshold(
            v16_path, checkpoint_path=checkpoint, v15_record_path=v15_path
        )

    changed_binding = _binding()
    changed_binding["plan_sha256"] = _sha("2")
    monkeypatch.setattr(calibration, "resolve_acid_binding", lambda **_kwargs: changed_binding)
    with pytest.raises(ValueError, match="ACID binding"):
        calibration.load_frozen_v15_threshold(v15_path, checkpoint_path=checkpoint)
