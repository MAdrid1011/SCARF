"""Focused wiring tests for the ACID-disjoint V15/V16 collector."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import saes.evaluation_disjoint_l1_calibration as records
import scripts.saes_acid_disjoint_l1_calibration as collector
from saes.incremental_selected_output_execution import (
    NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION,
    RAW_HEAD_EXECUTION_CONTRACT,
)


def _sha(character: str) -> str:
    return character * 64


def _native_dense_execution(*, finalized: bool) -> dict:
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
        "execution_finalized": finalized,
        "dense_head_positions": dense_positions,
        "dense_head_macs": 4096,
        "actual_head_macs": 4096,
        "head_mac_delta": 0,
        "phases": phases,
    }


def _binding() -> dict:
    splits = {}
    for split, prefix, count, hashes in (
        (records.TRAIN_SPLIT, "train", 24, ("a", "b", "c", "d")),
        (records.HOLDOUT_SPLIT, "holdout", 8, ("e", "f", "0", "1")),
    ):
        scenes = [f"{prefix}-{index:02d}" for index in range(count)]
        splits[split] = {
            "scenes": scenes,
            "scene_count": count,
            "scene_set_sha256": records.canonical_sha256(sorted(scenes)),
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


def test_collector_uses_acid_context_and_reloads_v15_before_v16(
    tmp_path: Path, monkeypatch
):
    binding = _binding()
    checkpoint = tmp_path / "re10k.ckpt"
    checkpoint.write_bytes(b"dl3dv-re10k")
    plan_path = tmp_path / "acid-plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    materialization = tmp_path / "materialization"
    materialization.mkdir()
    model_calls: list[dict] = []
    context_loads: list[tuple[str, int]] = []
    thresholds: list[float] = []

    class FakeLoader:
        def load_model(self, *args, **kwargs):
            model_calls.append({"args": args, "kwargs": kwargs})
            return SimpleNamespace(
                decoder=None,
                model=SimpleNamespace(encoder=object(), eval=lambda: None),
                config=SimpleNamespace(
                    dataset=object(), model=SimpleNamespace(encoder=object())
                ),
                device=torch.device("cpu"),
            )

    def fake_context_loader(*, split, sample_index, **_kwargs):
        context_loads.append((split, sample_index))
        scene = binding["splits"][split]["scenes"][sample_index]
        marker = sample_index + (100 if split == records.HOLDOUT_SPLIT else 0)
        return SimpleNamespace(
            context={
                "image": torch.zeros(1, 2, 3, 4, 4),
                "extrinsics": torch.eye(4).repeat(1, 2, 1, 1),
                "intrinsics": torch.eye(3).repeat(1, 2, 1, 1),
                "index": torch.tensor([[marker, marker]], dtype=torch.long),
            },
            identity={
                "scene": scene,
                "target_rgb_accessed": False,
                "target_camera_metadata_accessed": False,
                "target_index_accessed": False,
                "teacher_artifact_accessed": False,
                "expected_results_accessed": False,
            },
        )

    def fake_prepare(raw, **_kwargs):
        return (
            {
                **raw.context,
                "near": torch.ones(1, 2),
                "far": torch.full((1, 2), 1000.0),
            },
            {
                "target_mapping_present": False,
                "target_rgb_accessed": False,
                "target_camera_metadata_accessed": False,
                "target_index_accessed": False,
            },
        )

    def fake_plan(_model, context):
        marker = int(context["index"][0, 0].item())
        return SimpleNamespace(
            events={"selection_mask_sha256": _sha("a")},
            tile_trace=[
                {
                    "pre_guard_route": "L1",
                    "depth_uniform": True,
                    "adaptive_l1_leave_one_out_residual": float(marker),
                }
            ]
        )

    def fake_capture(_model, context, **kwargs):
        thresholds.append(float(kwargs["adaptive_l1_maximum_leave_one_out_residual"]))
        marker = int(context["index"][0, 0].item())
        return {
            "initial_head_events": {
                "marker": marker,
                "execution_finalized": False,
                "computed_mask_sha256": _sha("a"),
                "full_extension_dispatched": False,
            },
            "final_head_events": {
                "marker": marker,
                "execution_finalized": True,
                "computed_mask_sha256": _sha("a"),
                "full_extension_dispatched": False,
            },
            "compact_materialization_preflight": SimpleNamespace(
                events={
                    "target_rgb_accessed": False,
                    "skipped_s3_attributes_accessed": False,
                    "selected_anchor_v4_attribute_loo_collect_only": True,
                    "selected_anchor_v4_attribute_loo_guard": False,
                },
                tile_trace=(
                    {
                        "selected_anchor_v4_attribute_loo": {
                            "action": "observed_only",
                            "q75_risk": float(marker) + 0.25,
                        }
                    },
                ),
            )
        }

    monkeypatch.setattr(collector, "resolve_acid_binding", lambda **_kwargs: binding)
    monkeypatch.setattr(records, "resolve_acid_binding", lambda **_kwargs: binding)
    monkeypatch.setattr(
        collector,
        "resolve_experiment",
        lambda *_args: SimpleNamespace(
            model="transplat",
            dataset="dl3dv",
            experiment="re10k",
            checkpoint=checkpoint,
            hydra_overrides=(),
        ),
    )
    monkeypatch.setattr(collector, "create_model_loader", lambda _model: FakeLoader())
    monkeypatch.setattr(collector, "load_acid_joint_context", fake_context_loader)
    monkeypatch.setattr(collector, "prepare_acid_joint_model_context", fake_prepare)
    monkeypatch.setattr(collector, "_build_plan", fake_plan)
    monkeypatch.setattr(collector, "_capture_guarded_incremental_packed_adapter", fake_capture)
    def fake_native_dense_evidence(events):
        return _native_dense_execution(finalized=events["execution_finalized"])

    monkeypatch.setattr(
        collector, "native_dense_head_execution_evidence", fake_native_dense_evidence
    )
    monkeypatch.setattr(
        collector, "strict_fp32_convolution_execution", lambda: contextlib.nullcontext()
    )
    real_load_v15 = collector.load_frozen_v15_threshold
    v15_load_paths: list[Path] = []

    def checked_load_v15(path, **kwargs):
        assert Path(path).is_file()
        v15_load_paths.append(Path(path))
        return real_load_v15(path, **kwargs)

    monkeypatch.setattr(collector, "load_frozen_v15_threshold", checked_load_v15)

    v15, v16 = collector.collect_frozen_calibration_records(
        device=torch.device("cpu"),
        v15_output=tmp_path / "v15.json",
        v16_output=tmp_path / "v16.json",
        materialization_root=materialization,
        plan_path=plan_path,
    )

    assert model_calls == [
        {
            "args": (str(checkpoint),),
            "kwargs": {
                "experiment_name": "re10k",
                "hydra_overrides": (),
                "device": torch.device("cpu"),
                "encoder_only": True,
            },
        }
    ]
    assert context_loads == [
        *( (records.TRAIN_SPLIT, index) for index in range(24) ),
        *( (records.HOLDOUT_SPLIT, index) for index in range(8) ),
        *( (records.TRAIN_SPLIT, index) for index in range(24) ),
        *( (records.HOLDOUT_SPLIT, index) for index in range(8) ),
    ]
    assert v15["threshold"]["value"] == 12.0
    assert v15["holdout_verification"]["threshold_updated"] is False
    assert v16["base_v15_sha256"] == v15["sha256"]
    assert v16["threshold"]["value"] == 0.25
    assert v16["holdout_verification"]["threshold_updated"] is False
    assert v15_load_paths == [tmp_path / "v15.json"]
    assert thresholds == [v15["threshold_value"]] * 32
    assert json.loads((tmp_path / "v15.json").read_text(encoding="utf-8"))["sha256"] == v15[
        "sha256"
    ]
    assert json.loads((tmp_path / "v16.json").read_text(encoding="utf-8"))["sha256"] == v16[
        "sha256"
    ]


def test_collector_refuses_existing_frozen_outputs(tmp_path: Path):
    output = tmp_path / "v15.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must not already exist"):
        collector.collect_frozen_calibration_records(
            device=torch.device("cpu"),
            v15_output=output,
            v16_output=tmp_path / "v16.json",
        )


def test_collector_rejects_target_side_source_or_prepared_context():
    raw = SimpleNamespace(
        context={
            "image": torch.zeros(1, 2, 3, 4, 4),
            "extrinsics": torch.eye(4).repeat(1, 2, 1, 1),
            "intrinsics": torch.eye(3).repeat(1, 2, 1, 1),
            "index": torch.zeros(1, 2, dtype=torch.long),
        },
        identity={
            "target_rgb_accessed": True,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "teacher_artifact_accessed": False,
            "expected_results_accessed": False,
        },
    )
    preparation = {
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
    }

    with pytest.raises(RuntimeError, match="target_rgb_accessed"):
        collector._require_target_free_source(raw, preparation)

    with pytest.raises(RuntimeError, match="target-side field"):
        collector._require_prepared_context(
            {
                **raw.context,
                "near": torch.ones(1, 2),
                "far": torch.ones(1, 2),
                "target": {},
            }
        )


def _install_endpoint_smoke_mocks(
    monkeypatch,
    tmp_path: Path,
    *,
    skipped_s3_accessed: bool = False,
    risk_tile_count: int = 1,
):
    binding = _binding()
    checkpoint = tmp_path / "re10k.ckpt"
    checkpoint.write_bytes(b"dl3dv-re10k-endpoint")
    plan_path = tmp_path / "acid-plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    materialization = tmp_path / "materialization"
    materialization.mkdir()
    checkpoint_sha256 = collector.sha256_file(checkpoint)
    expected_application = {
        "model": "transplat",
        "dataset": "dl3dv",
        "checkpoint_sha256": checkpoint_sha256,
    }
    v15 = {
        "kind": records.V15_KIND,
        "sha256": "a" * 64,
        "threshold_value": 0.25,
        "application": expected_application,
        "acid_binding": binding,
        "access": binding["access"],
    }
    model = SimpleNamespace(encoder=object())
    bundle = SimpleNamespace(device=torch.device("cpu"))
    experiment = SimpleNamespace(checkpoint=checkpoint, experiment="re10k")
    raw = SimpleNamespace(
        identity={"scene": binding["splits"][records.HOLDOUT_SPLIT]["scenes"][0]}
    )
    context = {
        "image": torch.zeros(1, 2, 3, 4, 4),
        "extrinsics": torch.eye(4).repeat(1, 2, 1, 1),
        "intrinsics": torch.eye(3).repeat(1, 2, 1, 1),
        "index": torch.zeros(1, 2, dtype=torch.long),
        "near": torch.ones(1, 2),
        "far": torch.full((1, 2), 1000.0),
    }

    def fake_prepare_scene(**kwargs):
        assert kwargs["split"] == records.HOLDOUT_SPLIT
        assert kwargs["sample_index"] == 0
        assert kwargs["expected_scene"] == raw.identity["scene"]
        return raw, context

    plan = SimpleNamespace(
        events={
            "decision_semantics": collector.EVALUATION_DISJOINT_L1_15_DECISION_SEMANTICS,
            "l1_anchor_semantics": collector.ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
            "l1_anchor_count": 15,
            "tile_trace_sha256": "b" * 64,
        }
    )

    def fake_capture(_model, _context, **kwargs):
        assert kwargs["compact_nonzero_materialization"] is True
        assert (
            kwargs["compact_execution_policy"]
            == collector.ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY
        )
        assert kwargs["adaptive_l1_maximum_leave_one_out_residual"] == 0.25
        return {
            "compact_materialization_preflight": SimpleNamespace(
                events={
                    "target_rgb_accessed": False,
                    "skipped_s3_attributes_accessed": skipped_s3_accessed,
                    "selected_anchor_v4_attribute_loo_collect_only": True,
                    "selected_anchor_v4_attribute_loo_guard": False,
                    "native_opacity_endpoint_selected_count": 3,
                    "native_opacity_endpoint_promoted_full_tiles": 2,
                    "selected_anchor_v4_attribute_loo_checked_tiles": risk_tile_count,
                    "tile_trace_sha256": "c" * 64,
                },
                tile_trace=(
                    {
                        "selected_anchor_v4_attribute_loo": {
                            "action": "observed_only",
                            "q75_risk": 0.125,
                        }
                    },
                )
                if risk_tile_count
                else (),
            )
        }

    monkeypatch.setattr(collector, "resolve_acid_binding", lambda **_kwargs: binding)
    monkeypatch.setattr(
        collector,
        "_load_application_encoder",
        lambda _device: (model, bundle, experiment, checkpoint_sha256),
    )
    monkeypatch.setattr(
        collector,
        "load_frozen_v15_threshold",
        lambda path, **_kwargs: v15 if Path(path).name == "v15.json" else None,
    )
    monkeypatch.setattr(collector, "_prepare_scene", fake_prepare_scene)
    monkeypatch.setattr(collector, "_build_plan", lambda _model, _context: plan)
    monkeypatch.setattr(collector, "_capture_guarded_incremental_packed_adapter", fake_capture)
    monkeypatch.setattr(
        collector, "strict_fp32_convolution_execution", lambda: contextlib.nullcontext()
    )
    return binding, checkpoint, plan_path, materialization


def test_v16_endpoint_smoke_writes_verified_target_free_record(tmp_path: Path, monkeypatch):
    binding, checkpoint, plan_path, materialization = _install_endpoint_smoke_mocks(
        monkeypatch, tmp_path
    )
    output = tmp_path / "endpoint.json"
    v15_path = tmp_path / "v15.json"
    v15_path.write_text("{}", encoding="utf-8")

    record = collector.write_v16_endpoint_smoke(
        output=output,
        device=torch.device("cpu"),
        v15_record=v15_path,
        split=records.HOLDOUT_SPLIT,
        sample_index=0,
        materialization_root=materialization,
        plan_path=plan_path,
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == record
    assert record["source_context"] == {
        "split": records.HOLDOUT_SPLIT,
        "sample_index": 0,
        "scene": binding["splits"][records.HOLDOUT_SPLIT]["scenes"][0],
    }
    assert record["v15_identity"]["sha256"] == "a" * 64
    assert record["checkpoint_binding"]["checkpoint_path"] == str(checkpoint.resolve())
    assert record["access"] == binding["access"]
    assert record["preflight"] == {
        "native_opacity_endpoint_count": 3,
        "native_opacity_endpoint_promoted_tile_count": 2,
        "risk_tile_count": 1,
        "tile_trace_sha256": "c" * 64,
    }
    assert record["sha256"] == collector._with_smoke_sha256(record)["sha256"]


def test_v16_endpoint_smoke_fails_closed_on_skipped_s3_and_does_not_write(
    tmp_path: Path, monkeypatch
):
    _binding_value, _checkpoint, plan_path, materialization = _install_endpoint_smoke_mocks(
        monkeypatch, tmp_path, skipped_s3_accessed=True
    )
    output = tmp_path / "endpoint.json"
    v15_path = tmp_path / "v15.json"
    v15_path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="selected-only boundary"):
        collector.write_v16_endpoint_smoke(
            output=output,
            device=torch.device("cpu"),
            v15_record=v15_path,
            split=records.HOLDOUT_SPLIT,
            sample_index=0,
            materialization_root=materialization,
            plan_path=plan_path,
        )

    assert not output.exists()


def test_v16_endpoint_smoke_records_zero_v4_risk_tiles(tmp_path: Path, monkeypatch):
    _binding_value, _checkpoint, plan_path, materialization = _install_endpoint_smoke_mocks(
        monkeypatch, tmp_path, risk_tile_count=0
    )
    v15_path = tmp_path / "v15.json"
    v15_path.write_text("{}", encoding="utf-8")

    record = collector.collect_v16_endpoint_smoke(
        device=torch.device("cpu"),
        v15_record=v15_path,
        split=records.HOLDOUT_SPLIT,
        sample_index=0,
        materialization_root=materialization,
        plan_path=plan_path,
    )

    assert record["preflight"]["risk_tile_count"] == 0


def test_v16_endpoint_smoke_refuses_to_overwrite_before_collection(
    tmp_path: Path, monkeypatch
):
    output = tmp_path / "endpoint.json"
    output.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        collector,
        "collect_v16_endpoint_smoke",
        lambda **_kwargs: pytest.fail("existing output must prevent collection"),
    )

    with pytest.raises(FileExistsError, match="new file"):
        collector.write_v16_endpoint_smoke(
            output=output,
            device=torch.device("cpu"),
            v15_record=tmp_path / "v15.json",
            split=records.HOLDOUT_SPLIT,
            sample_index=0,
        )


def test_v16_endpoint_smoke_cli_routes_explicit_arguments(tmp_path: Path, monkeypatch):
    output = tmp_path / "endpoint.json"
    calls: list[dict] = []
    record = {
        "sha256": "a" * 64,
        "source_context": {"scene": "4fa73a829dde9435"},
    }

    def fake_write(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output"]).write_text(json.dumps(record), encoding="utf-8")
        return record

    monkeypatch.setattr(collector.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(collector, "write_v16_endpoint_smoke", fake_write)

    exit_code = collector.main(
        [
            "endpoint-smoke",
            "--output",
            str(output),
            "--v15-record",
            str(tmp_path / "v15.json"),
            "--split",
            records.HOLDOUT_SPLIT,
            "--sample-index",
            "0",
            "--materialization-root",
            str(tmp_path / "materialization"),
            "--plan",
            str(tmp_path / "plan.json"),
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "output": output,
            "device": torch.device("cuda"),
            "v15_record": tmp_path / "v15.json",
            "split": records.HOLDOUT_SPLIT,
            "sample_index": 0,
            "materialization_root": tmp_path / "materialization",
            "plan_path": tmp_path / "plan.json",
        }
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == record
