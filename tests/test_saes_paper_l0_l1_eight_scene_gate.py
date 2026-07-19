"""Contract coverage for the fixed eight-scene V16 quality driver."""

from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def _sha(character: str) -> str:
    return character * 64


def _native_dense_execution(*, finalized: bool) -> dict:
    from saes.incremental_selected_output_execution import (
        NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION,
        RAW_HEAD_EXECUTION_CONTRACT,
    )

    dense_positions = 32
    phases = []
    for index, phase in enumerate(("primary", "secondary", "full")):
        positions = dense_positions if index == 0 else 0
        phases.append(
            {
                "phase": phase,
                "mask_sha256": _sha("1" if index == 0 else "2" if index == 1 else "3"),
                "tile_trace_sha256": _sha(
                    "4" if index == 0 else "5" if index == 1 else "6"
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
        "head_weight_sha256": _sha("7"),
        "head_input_sha256": _sha("8"),
        "phase_trace_sha256": _sha("9"),
        "tile_trace_sha256": _sha("0"),
        "execution_finalized": finalized,
        "dense_head_positions": dense_positions,
        "dense_head_macs": 4096,
        "actual_head_macs": 4096,
        "head_mac_delta": 0,
        "phases": phases,
    }


def _metric(psnr: float, ssim: float, lpips: float) -> dict[str, float]:
    return {"psnr_db": psnr, "ssim": ssim, "lpips": lpips}


def _configure_expected(
    monkeypatch: pytest.MonkeyPatch, module, *, model_name: str = "transplat"
) -> dict[str, str]:
    identity = {
        "source": "a" * 64,
        "selection": "b" * 64,
        "checkpoint": "c" * 64,
        "v15": "d" * 64,
        "v16": "e" * 64,
        "mechanism": "f" * 64,
        "prepared_tree": "0" * 64,
        "raw_source_record": "1" * 64,
        "raw_revision": "fixture-revision",
        "raw_metadata": "2" * 64,
        "raw_filelist": "3" * 64,
        "raw_plans": "4" * 64,
    }
    monkeypatch.setattr(module, "EXPECTED_SOURCE_INDEX_SHA256", identity["source"])
    monkeypatch.setattr(module, "EXPECTED_SAMPLE_SELECTION_SHA256", identity["selection"])
    monkeypatch.setattr(module, "EXPECTED_CHECKPOINT_SHA256", identity["checkpoint"])
    monkeypatch.setattr(module, "EXPECTED_V15_SHA256", identity["v15"])
    monkeypatch.setattr(module, "EXPECTED_V16_SHA256", identity["v16"])
    monkeypatch.setattr(module, "EXPECTED_MECHANISM_SHA256", identity["mechanism"])
    monkeypatch.setattr(module, "EXPECTED_PREPARED_TREE_SHA256", identity["prepared_tree"])
    monkeypatch.setattr(module, "EXPECTED_RAW_SOURCE_RECORD_SHA256", identity["raw_source_record"])
    monkeypatch.setattr(module, "EXPECTED_RAW_SOURCE_REVISION", identity["raw_revision"])
    monkeypatch.setattr(module, "EXPECTED_RAW_BENCHMARK_METADATA_SHA256", identity["raw_metadata"])
    monkeypatch.setattr(module, "EXPECTED_RAW_FILELIST_SHA256", identity["raw_filelist"])
    monkeypatch.setattr(module, "EXPECTED_RAW_SCENE_SOURCE_PLANS_SHA256", identity["raw_plans"])
    if model_name == "mvsplat":
        monkeypatch.setattr(module, "EXPECTED_MVSPLAT_CHECKPOINT_SHA256", identity["checkpoint"])
        monkeypatch.setattr(module, "EXPECTED_MVSPLAT_V15_SHA256", identity["v15"])
        monkeypatch.setattr(module, "EXPECTED_MVSPLAT_V16_SHA256", identity["v16"])
        monkeypatch.setattr(
            module, "EXPECTED_MVSPLAT_MECHANISM_SHA256", identity["mechanism"]
        )
    return identity


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    module,
    tmp_path: Path,
    *,
    model_name: str = "transplat",
):
    expected_model = model_name
    identity = _configure_expected(monkeypatch, module, model_name=expected_model)
    calls: list[tuple[str, int]] = []
    model_calls: list[tuple[str, str]] = []
    backend_identity = {
        "model": "mvsplat",
        "coordinate_semantics": "mvsplat-inline-pixel-center-plus-sigmoid-offset",
        "source_files": {"coordinates": {"sha256": "a" * 64}},
    }
    selections = [
        {
            "sample_index": index,
            "execution_index": index,
            "scene": f"scene-{index}",
            "context_indices": [0, 9],
            "target_indices": [1, 3, 5, 7],
        }
        for index in range(8)
    ]
    context_by_index = {
        index: {
            "scene": f"scene-{index}",
            "context_indices": [0, 9],
            "source_sample_index": index,
            "source_binding": {
                "canonical_index_sha256": identity["source"],
                "canonical_sample_selection_sha256": identity["selection"],
                "canonical_selection_sha256": module._canonical_sha256(
                    {
                        "source_sample_index": index,
                        "scene": f"scene-{index}",
                        "context_indices": [0, 9],
                        "target_indices": [1, 3, 5, 7],
                    }
                ),
            },
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
        }
        for index in range(8)
    }

    def fixed_selections(*, model_name: str):
        if model_name != expected_model:
            pytest.fail("fixed selections received the wrong model")
        return type("Selection", (), {"sample_count": 8})(), object(), selections

    monkeypatch.setattr(module, "_fixed_selections", fixed_selections)

    def validate_prepared(_experiment, *, model_name: str):
        if model_name != expected_model:
            pytest.fail("prepared contract received the wrong model")
        return {"tree_sha256": module.EXPECTED_PREPARED_TREE_SHA256}

    monkeypatch.setattr(module, "_validate_prepared_contract", validate_prepared)

    def validate_frozen(**kwargs) -> None:
        if kwargs.get("model_name") != expected_model:
            pytest.fail("frozen calibration validation received the wrong model")
        model_calls.append(("calibration", kwargs["model_name"]))

    monkeypatch.setattr(module, "_validate_frozen_gate_calibrations", validate_frozen)

    def source_prepare(
        _raw_root: Path, *, output_dir: Path, model: str, sample_index: int
    ):
        if model != expected_model:
            pytest.fail("source preparation received the wrong model")
        model_calls.append(("source", model))
        calls.append(("source", sample_index))
        output_dir.mkdir(parents=True)
        record = {
            "model": model,
            "dataset": "dl3dv",
            "source_sample_index": sample_index,
            "target_rgb_included": False,
            "target_rgb_opened": False,
            "canonical_protocol": {
                "source_index_sha256": identity["source"],
                "sample_selection_sha256": identity["selection"],
                "dataset_tree_sha256": identity["prepared_tree"],
            },
            "selected_sample": {
                "scene": f"scene-{sample_index}",
                "context_indices": [0, 9],
                "target_indices": [1, 3, 5, 7],
            },
            "canonical_selection": {
                "source_sample_index": sample_index,
                "scene": f"scene-{sample_index}",
                "context_indices": [0, 9],
                "target_indices": [1, 3, 5, 7],
            },
            "source": {
                "revision": identity["raw_revision"],
                "source_record_sha256": identity["raw_source_record"],
                "benchmark_metadata_sha256": identity["raw_metadata"],
                "filelist_sha256": identity["raw_filelist"],
                "scene_source_plans_sha256": identity["raw_plans"],
            },
        }
        return record

    def context_prepare(_source_root: Path, *, output_root: Path, model: str):
        if model != expected_model:
            pytest.fail("context preparation received the wrong model")
        sample_index = int(_source_root.parent.name.rsplit("_", 1)[1])
        model_calls.append(("context", model))
        calls.append(("context", sample_index))
        output_root.mkdir(parents=True)
        return context_by_index[sample_index]

    def audit_collect(*, input_root: Path, **_kwargs):
        if _kwargs.get("model_name") != expected_model:
            pytest.fail("target-free audit received the wrong model")
        sample_index = int(input_root.parent.name.rsplit("_", 1)[1])
        model_calls.append(("audit", _kwargs["model_name"]))
        calls.append(("audit", sample_index))
        record = {
            "status": "PASS",
            "model": expected_model,
            "input_identity": context_by_index[sample_index],
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "access_evidence": {
                "target_mapping_present": False,
                "target_rgb_accessed": False,
                "target_camera_metadata_accessed": False,
                "target_index_accessed": False,
            },
            "sha256": "1" * 64,
            "raw_head": {
                "native_dense_head_execution": {
                    "initial": _native_dense_execution(finalized=False),
                    "guarded": _native_dense_execution(finalized=True),
                }
            },
        }
        if expected_model == "mvsplat":
            record["execution"] = {
                "model": "mvsplat",
                "checkpoint_sha256": identity["checkpoint"],
                "classic_backend_identity": backend_identity,
            }
            record["execution_boundary"] = {
                "backend_coordinate_semantics": backend_identity[
                    "coordinate_semantics"
                ]
            }
        return record

    def quality_collect(*, target_free_input_root: Path, sample_index: int, **_kwargs):
        from saes.incremental_selected_output_execution import RAW_HEAD_EXECUTION_CONTRACT

        if _kwargs.get("model_name") != expected_model:
            pytest.fail("quality pilot received the wrong model")
        model_calls.append(("quality", _kwargs["model_name"]))
        calls.append(("quality", sample_index))
        context = context_by_index[sample_index]
        baseline = _metric(35.0, 0.975, 0.03)
        compact = _metric(34.95, 0.974, 0.031)
        record = {
            "status": "PASS",
            "model": expected_model,
            "paper_result_eligible": False,
            "sample_index": sample_index,
            "execution_index": _kwargs["execution_index"],
            "native_sample_count": _kwargs["native_sample_count"],
            "scene": f"scene-{sample_index}",
            "context_indices": [0, 9],
            "target_indices": [1, 3, 5, 7],
            "checkpoint_sha256": identity["checkpoint"],
            "adaptive_l1_calibration": {"sha256": identity["v15"]},
            "adaptive_l1_v4_attribute_loo_calibration": {
                "sha256": identity["v16"],
                "raw_head_execution_contract": RAW_HEAD_EXECUTION_CONTRACT,
            },
            "paper_identity": {"sha256": identity["mechanism"]},
            "target_free_quality_gate": {
                "model": expected_model,
                "sample_index": sample_index,
                "context_input_identity": context,
                "source_selection_mask_sha256": "2" * 64,
                "selected_output_mask_sha256": "3" * 64,
                "packed_source_trace_sha256": "4" * 64,
            },
            "execution_boundary": {
                "whole_pipeline_s2_s3_sparse_execution_verified": False,
                "s2_s3_saving": 0.0,
                "timing_claim": False,
            },
            "raw_head_execution": {
                "initial": _native_dense_execution(finalized=False),
                "guarded": _native_dense_execution(finalized=True),
            },
            "compact_route": {"final_route": {"route_counts": {"L0": 0, "L1": 2, "Full": 6}}},
            "quality": {
                "baseline": baseline,
                "compact": compact,
                "verdict": {"pass": True},
                "views": [
                    {"baseline": baseline, "compact": compact, "target_index": target}
                    for target in (1, 3, 5, 7)
                ],
            },
        }
        if expected_model == "mvsplat":
            record["backend_binding"] = {
                "model": "mvsplat",
                "coordinate_semantics": backend_identity["coordinate_semantics"],
                "classic_backend_identity": backend_identity,
            }
            record["decoder_binding"] = {
                "model": "mvsplat",
                "decoder_module": {"path": "mvsplat/src/model/decoder/decoder.py"},
                "gaussians_module": {"path": "mvsplat/src/model/types.py"},
            }
        return record

    monkeypatch.setattr(module, "prepare_inputs", source_prepare)
    monkeypatch.setattr(module, "prepare_context_only_audit_input", context_prepare)

    def validate_context(root: Path, *, model: str):
        if model != expected_model:
            pytest.fail("context validation received the wrong model")
        model_calls.append(("validate_context", model))
        return context_by_index[int(Path(root).parent.name.rsplit("_", 1)[1])]

    monkeypatch.setattr(
        module,
        "validate_context_only_audit_input",
        validate_context,
    )
    monkeypatch.setattr(module, "collect_incremental_selected_output_audit", audit_collect)
    monkeypatch.setattr(module, "collect_paper_compact_packet_pilot", quality_collect)
    monkeypatch.setattr(module, "source_identity", lambda: {"fixture": True})
    monkeypatch.setattr(
        module,
        "_live_classic_backend_identity",
        lambda _experiment, *, model_name: (
            backend_identity if model_name == "mvsplat" else None
        ),
    )
    return calls, model_calls


def test_fixed_eight_scene_gate_runs_audit_before_quality_and_aggregates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from scripts import saes_paper_l0_l1_eight_scene_gate as gate

    calls, _model_calls = _install_fake_pipeline(monkeypatch, gate, tmp_path)
    output = tmp_path / "eight"
    record = gate.run_fixed_eight_scene_gate(
        output_dir=output,
        raw_root=tmp_path / "raw",
        device=torch.device("cpu"),
        v15_calibration_record=tmp_path / "v15.json",
        v16_calibration_record=tmp_path / "v16.json",
    )

    assert record["status"] == "PASS"
    assert record["paper_result_eligible"] is False
    assert record["quality"]["view_count"] == 32
    assert record["quality"]["route_totals"] == {"L0": 0, "L1": 16, "Full": 48}
    assert [entry["sample_index"] for entry in record["samples"]] == list(range(8))
    for sample_index in range(8):
        assert calls.index(("audit", sample_index)) < calls.index(("quality", sample_index))
    assert (output / "results.json").is_file()


def test_fixed_eight_scene_gate_forwards_mvsplat_identity_through_every_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from scripts import saes_paper_l0_l1_eight_scene_gate as gate

    calls, model_calls = _install_fake_pipeline(
        monkeypatch, gate, tmp_path, model_name="mvsplat"
    )
    record = gate.run_fixed_eight_scene_gate(
        output_dir=tmp_path / "mvsplat-eight",
        raw_root=tmp_path / "raw",
        device=torch.device("cpu"),
        v15_calibration_record=tmp_path / "v15.json",
        v16_calibration_record=tmp_path / "v16.json",
        model_name="mvsplat",
    )

    assert record["status"] == "PASS"
    assert record["model"] == "mvsplat"
    assert record["classic_backend_identity"]["model"] == "mvsplat"
    assert record["expected_identity"]["checkpoint_sha256"] == "c" * 64
    assert all(model == "mvsplat" for _stage, model in model_calls)
    assert model_calls.count(("calibration", "mvsplat")) == 1
    assert model_calls.count(("source", "mvsplat")) == 8
    assert model_calls.count(("context", "mvsplat")) == 8
    assert model_calls.count(("validate_context", "mvsplat")) == 8
    assert model_calls.count(("audit", "mvsplat")) == 8
    assert model_calls.count(("quality", "mvsplat")) == 8
    assert calls.index(("audit", 0)) < calls.index(("quality", 0))


def test_fixed_eight_scene_gate_fails_closed_when_quality_identity_drifts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from scripts import saes_paper_l0_l1_eight_scene_gate as gate

    _calls, _model_calls = _install_fake_pipeline(monkeypatch, gate, tmp_path)
    original = gate.collect_paper_compact_packet_pilot

    def drifted_quality(*args, **kwargs):
        record = original(*args, **kwargs)
        if kwargs["sample_index"] == 3:
            record["paper_identity"]["sha256"] = "0" * 64
        return record

    monkeypatch.setattr(gate, "collect_paper_compact_packet_pilot", drifted_quality)
    record = gate.run_fixed_eight_scene_gate(
        output_dir=tmp_path / "eight",
        raw_root=tmp_path / "raw",
        device=torch.device("cpu"),
        v15_calibration_record=tmp_path / "v15.json",
        v16_calibration_record=tmp_path / "v16.json",
    )

    assert record["status"] == "FAILED"
    assert record["sample_count_completed"] == 7
    assert record["failures"] == [
        {
            "sample_index": 3,
            "execution_index": 3,
            "type": "ValueError",
            "message": "quality mechanism identity changed",
        }
    ]
