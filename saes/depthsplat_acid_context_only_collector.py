"""CPU-only validation boundary for formal DepthSplat ACID collection.

This module does not load a DepthSplat checkpoint into a model, invoke an
encoder, materializer, renderer, quality metric, or CUDA.  It validates that
every planned ACID 24/8 item can be opened through the context-only sidecar
loader and records the exact source identity a later native-risk worker must
reuse.  Its output is intentionally not a V15D or V16D calibration record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from saes.depthsplat_acid_disjoint_calibration import (
    COLLECTION_PLAN_KIND,
    NOT_RUN_STATUS,
    canonical_sha256,
)


INPUT_VALIDATION_KIND = "depthsplat-acid-context-only-collector-input-validation"
INPUT_VALIDATED_STATUS = "INPUTS_VALIDATED_GPU_NOT_RUN"
TRAIN_SPLIT = "calibration_train"
HOLDOUT_SPLIT = "calibration_holdout"
SPLITS = (TRAIN_SPLIT, HOLDOUT_SPLIT)
_ACCESS = {
    "target_mapping_present": False,
    "target_rgb_accessed": False,
    "target_camera_metadata_accessed": False,
    "target_index_accessed": False,
    "skipped_s3_attributes_accessed": False,
}
_IDENTITY_FALSE_FIELDS = (
    "target_rgb_accessed",
    "target_camera_metadata_accessed",
    "target_index_accessed",
    "teacher_artifact_accessed",
    "expected_results_accessed",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_collection_plan(collection_plan: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "status",
        "paper_result_eligible",
        "application",
        "acid_binding",
        "mechanism",
        "access",
        "collection",
        "sha256",
    }
    if not isinstance(collection_plan, Mapping) or set(collection_plan) != required:
        raise ValueError("DepthSplat ACID collection plan has an invalid schema")
    payload = {key: value for key, value in collection_plan.items() if key != "sha256"}
    if collection_plan.get("sha256") != canonical_sha256(payload):
        raise ValueError("DepthSplat ACID collection plan SHA256 is invalid")
    if (
        collection_plan.get("kind") != COLLECTION_PLAN_KIND
        or collection_plan.get("status") != NOT_RUN_STATUS
        or collection_plan.get("paper_result_eligible") is not False
        or collection_plan.get("access") != _ACCESS
        or not isinstance(collection_plan.get("application"), Mapping)
        or not isinstance(collection_plan.get("acid_binding"), Mapping)
        or not isinstance(collection_plan.get("collection"), Mapping)
    ):
        raise ValueError("DepthSplat ACID collection plan is not a target-free unrun plan")
    binding = collection_plan["acid_binding"]
    splits = binding.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise ValueError("DepthSplat ACID collection plan lacks 24/8 splits")
    for split, expected_count in ((TRAIN_SPLIT, 24), (HOLDOUT_SPLIT, 8)):
        record = splits.get(split)
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("scenes"), list)
            or len(record["scenes"]) != expected_count
            or record.get("scene_count") != expected_count
            or len(set(record["scenes"])) != expected_count
            or any(not isinstance(scene, str) or not scene for scene in record["scenes"])
        ):
            raise ValueError(f"DepthSplat ACID collection plan {split} is invalid")
    return dict(collection_plan)


def _shape(value: Any, *, expected: tuple[int, ...], label: str) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"DepthSplat ACID {label} is not a tensor")
    actual = tuple(int(item) for item in shape)
    if actual != expected:
        raise ValueError(
            f"DepthSplat ACID {label} shape changed: expected {expected}, got {actual}"
        )
    return list(actual)


def _image_shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("DepthSplat ACID context image is not a tensor")
    actual = tuple(int(item) for item in shape)
    if len(actual) != 5 or actual[:3] != (1, 2, 3) or actual[-2] < 1 or actual[-1] < 1:
        raise ValueError("DepthSplat ACID context image shape changed")
    return list(actual)


def _validate_context_record(
    raw: Any,
    *,
    binding: Mapping[str, Any],
    split: str,
    sample_index: int,
    scene: str,
) -> dict[str, Any]:
    context = getattr(raw, "context", None)
    identity = getattr(raw, "identity", None)
    if not isinstance(context, Mapping) or set(context) != {
        "image",
        "extrinsics",
        "intrinsics",
        "index",
    }:
        raise ValueError("DepthSplat ACID collector received target-bearing context input")
    if not isinstance(identity, Mapping):
        raise ValueError("DepthSplat ACID collector source has no identity audit")
    if (
        identity.get("split") != split
        or identity.get("sample_index") != sample_index
        or identity.get("scene") != scene
        or identity.get("plan_sha256") != binding.get("plan_sha256")
        or any(identity.get(field) is not False for field in _IDENTITY_FALSE_FIELDS)
    ):
        raise ValueError("DepthSplat ACID collector source crossed its target-free boundary")
    sidecar = identity.get("sidecar")
    expected_sidecar = binding["splits"][split]
    if (
        not isinstance(sidecar, Mapping)
        or sidecar.get("tree_sha256") != expected_sidecar.get("sidecar_tree_sha256")
        or sidecar.get("manifest_sha256")
        != expected_sidecar.get("sidecar_manifest_sha256")
        or sidecar.get("selection_sha256") != expected_sidecar.get("selection_sha256")
        or sidecar.get("input_provenance_sha256")
        != expected_sidecar.get("input_provenance_sha256")
    ):
        raise ValueError("DepthSplat ACID collector sidecar identity changed")
    return {
        "scene": scene,
        "sample_index": sample_index,
        "context_shapes": {
            "image": _image_shape(context["image"]),
            "extrinsics": _shape(
                context["extrinsics"], expected=(1, 2, 4, 4), label="context extrinsics"
            ),
            "intrinsics": _shape(
                context["intrinsics"], expected=(1, 2, 3, 3), label="context intrinsics"
            ),
            "index": _shape(context["index"], expected=(1, 2), label="context index"),
        },
        "access": _ACCESS,
    }


def validate_formal_acid_context_only_inputs(
    *,
    collection_plan: Mapping[str, Any],
    plan_path: Path,
    materialization_root: Path,
    context_loader: Callable[..., Any],
) -> dict[str, Any]:
    """Validate all 24/8 raw sidecar inputs without creating observations.

    ``context_loader`` is injected for testability; the CLI binds it to
    :func:`integration.acid_joint_context.load_acid_joint_context`.  A caller
    receives no tensor payloads back, only safe shapes and immutable identity
    references, so this function cannot become a target-bearing cache.
    """

    plan = _validate_collection_plan(collection_plan)
    if not callable(context_loader):
        raise TypeError("DepthSplat ACID collector requires a context-only loader")
    binding = plan["acid_binding"]
    records: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        scenes = binding["splits"][split]["scenes"]
        split_records: list[dict[str, Any]] = []
        for sample_index, scene in enumerate(scenes):
            raw = context_loader(
                materialization_root=Path(materialization_root),
                split=split,
                sample_index=sample_index,
                plan_path=Path(plan_path),
            )
            split_records.append(
                _validate_context_record(
                    raw,
                    binding=binding,
                    split=split,
                    sample_index=sample_index,
                    scene=scene,
                )
            )
        records[split] = split_records
    payload = {
        "schema_version": "1.0",
        "kind": INPUT_VALIDATION_KIND,
        "status": INPUT_VALIDATED_STATUS,
        "paper_result_eligible": False,
        "application": dict(plan["application"]),
        "acid_binding": dict(binding),
        "collection_plan_sha256": plan["sha256"],
        "access": dict(_ACCESS),
        "validated_context_records": records,
        "native_risk_observations": {
            "collected": False,
            "reason": "CPU_ONLY_CONTEXT_INPUT_VALIDATION",
        },
        "freeze": {
            "v15d_threshold_frozen": False,
            "v16d_threshold_frozen": False,
            "allowed": False,
            "reason": "NO_NATIVE_RISK_OBSERVATIONS",
        },
        "gpu": {
            "gpu_collection_attempted": False,
            "gpu_collection_completed": False,
            "model_loaded": False,
            "encoder_executed": False,
            "materializer_executed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
        },
        "collector_source": {
            "path": "saes/depthsplat_acid_context_only_collector.py",
            "sha256": _sha256_file(Path(__file__)),
        },
    }
    return {**payload, "sha256": canonical_sha256(payload)}


__all__ = [
    "HOLDOUT_SPLIT",
    "INPUT_VALIDATED_STATUS",
    "INPUT_VALIDATION_KIND",
    "TRAIN_SPLIT",
    "validate_formal_acid_context_only_inputs",
]
