"""CPU-only contracts for formal DepthSplat ACID sidecar input validation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


class _Tensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


def _binding():
    train = [f"train-{index:02d}" for index in range(24)]
    holdout = [f"holdout-{index:02d}" for index in range(8)]

    def split(scenes, offset):
        return {
            "scenes": scenes,
            "scene_count": len(scenes),
            "sidecar_tree_sha256": _digest(offset),
            "sidecar_manifest_sha256": _digest(offset + 1),
            "selection_sha256": _digest(offset + 2),
            "input_provenance_sha256": _digest(offset + 3),
        }

    return {
        "plan_sha256": _digest(1),
        "splits": {
            "calibration_train": split(train, 10),
            "calibration_holdout": split(holdout, 20),
        },
    }


def _collection_plan(module, binding):
    payload = {
        "schema_version": "1.0",
        "kind": "depthsplat-nonzero-l0-l1-acid-disjoint-collection-plan",
        "status": "PLANNED_CONTEXT_ONLY_GPU_NOT_RUN",
        "paper_result_eligible": False,
        "application": {
            "model": "depthsplat",
            "dataset": "dl3dv",
            "backend_identity": {"checkpoint": {"sha256": _digest(30)}},
            "collector_source": {"tree_sha256": _digest(31)},
        },
        "acid_binding": binding,
        "mechanism": {"id": "depthsplat-nonzero-l0-l1-moment-merge-v1"},
        "access": _access(),
        "collection": {"gpu_collection_attempted": False},
    }
    return {**payload, "sha256": module.canonical_sha256(payload)}


def _loader(binding, *, mutate=None):
    def load(*, split, sample_index, **_kwargs):
        scene = binding["splits"][split]["scenes"][sample_index]
        sidecar = {
            "tree_sha256": binding["splits"][split]["sidecar_tree_sha256"],
            "manifest_sha256": binding["splits"][split]["sidecar_manifest_sha256"],
            "selection_sha256": binding["splits"][split]["selection_sha256"],
            "input_provenance_sha256": binding["splits"][split][
                "input_provenance_sha256"
            ],
        }
        raw = SimpleNamespace(
            context={
                "image": _Tensor((1, 2, 3, 16, 24)),
                "extrinsics": _Tensor((1, 2, 4, 4)),
                "intrinsics": _Tensor((1, 2, 3, 3)),
                "index": _Tensor((1, 2)),
            },
            identity={
                "plan_sha256": binding["plan_sha256"],
                "split": split,
                "sample_index": sample_index,
                "scene": scene,
                "sidecar": sidecar,
                "target_rgb_accessed": False,
                "target_camera_metadata_accessed": False,
                "target_index_accessed": False,
                "teacher_artifact_accessed": False,
                "expected_results_accessed": False,
            },
        )
        if mutate is not None:
            mutate(raw, split, sample_index)
        return raw

    return load


def test_validates_every_acid_24_8_context_only_input_without_gpu_observations():
    import saes.depthsplat_acid_context_only_collector as collector

    binding = _binding()
    record = collector.validate_formal_acid_context_only_inputs(
        collection_plan=_collection_plan(collector, binding),
        plan_path="unused-plan.json",
        materialization_root="unused-root",
        context_loader=_loader(binding),
    )

    assert record["status"] == collector.INPUT_VALIDATED_STATUS
    assert len(record["validated_context_records"][collector.TRAIN_SPLIT]) == 24
    assert len(record["validated_context_records"][collector.HOLDOUT_SPLIT]) == 8
    assert record["native_risk_observations"]["collected"] is False
    assert record["freeze"]["allowed"] is False
    assert record["freeze"]["reason"] == "NO_NATIVE_RISK_OBSERVATIONS"
    assert record["gpu"]["gpu_collection_attempted"] is False
    assert record["gpu"]["encoder_executed"] is False
    assert "threshold" not in record
    assert len(record["collector_source"]["sha256"]) == 64


def test_rejects_target_bearing_context_input():
    import saes.depthsplat_acid_context_only_collector as collector

    binding = _binding()

    def add_target(raw, _split, _index):
        raw.context["target"] = _Tensor((1, 3, 16, 24))

    with pytest.raises(ValueError, match="target-bearing context"):
        collector.validate_formal_acid_context_only_inputs(
            collection_plan=_collection_plan(collector, binding),
            plan_path="unused-plan.json",
            materialization_root="unused-root",
            context_loader=_loader(binding, mutate=add_target),
        )


def test_rejects_target_access_flag_and_sidecar_hash_drift():
    import saes.depthsplat_acid_context_only_collector as collector

    binding = _binding()

    def touch_target(raw, _split, _index):
        raw.identity["target_rgb_accessed"] = True

    with pytest.raises(ValueError, match="target-free boundary"):
        collector.validate_formal_acid_context_only_inputs(
            collection_plan=_collection_plan(collector, binding),
            plan_path="unused-plan.json",
            materialization_root="unused-root",
            context_loader=_loader(binding, mutate=touch_target),
        )

    def drift_sidecar(raw, _split, _index):
        raw.identity["sidecar"]["tree_sha256"] = _digest(999)

    with pytest.raises(ValueError, match="sidecar identity changed"):
        collector.validate_formal_acid_context_only_inputs(
            collection_plan=_collection_plan(collector, binding),
            plan_path="unused-plan.json",
            materialization_root="unused-root",
            context_loader=_loader(binding, mutate=drift_sidecar),
        )


def test_input_validator_has_no_gpu_or_model_execution_dependency():
    import saes.depthsplat_acid_context_only_collector as collector

    source = Path(collector.__file__).read_text(encoding="utf-8")
    script = (
        Path(collector.__file__).parents[1]
        / "scripts"
        / "saes_depthsplat_acid_context_only_collector.py"
    ).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "load_model" not in source
    assert "cuda" not in source
    assert "load_model" not in script
    assert "--device" not in script
