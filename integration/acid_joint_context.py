"""Load one verified ACID joint-calibration context without model dependencies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from data.plan_acid_joint_calibration import (
    CONTEXT_RECORD_KIND,
    HOLDOUT_SPLIT,
    SPLITS,
    TRAIN_SPLIT,
    validate_materialization,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_PATH = ROOT / "artifact" / "protocol" / "acid_joint_calibration_plan.json"
_RECORD_FIELDS = frozenset(
    (
        "schema_version",
        "kind",
        "key",
        "source_view_count",
        "context_indices",
        "context_cameras",
        "context_images",
    )
)


@dataclass(frozen=True)
class AcidJointContext:
    """Raw context tensors and provenance for one ACID calibration scene.

    ``context`` intentionally has no ``target`` mapping and no model-specific
    near/far bounds. A later, isolated per-model worker obtains those bounds
    from its frozen model configuration before it invokes an encoder.
    """

    context: dict[str, torch.Tensor]
    identity: dict[str, Any]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _split_scenes(plan: Mapping[str, Any], split: str) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"ACID joint split is invalid: {split}")
    partition = plan.get("partition")
    record = partition.get(split) if isinstance(partition, Mapping) else None
    scenes = record.get("scenes") if isinstance(record, Mapping) else None
    if (
        not isinstance(scenes, list)
        or not scenes
        or any(not isinstance(scene, str) or not scene for scene in scenes)
        or len(scenes) != len(set(scenes))
    ):
        raise ValueError(f"ACID joint plan has no valid {split} scenes")
    return list(scenes)


def _safe_chunk_path(test_root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("ACID joint sidecar index has a non-string record path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
        raise ValueError("ACID joint sidecar record path is unsafe")
    path = (Path(test_root).resolve() / pure.name).resolve()
    if path.parent != Path(test_root).resolve() or not path.is_file():
        raise FileNotFoundError("ACID joint sidecar record is unavailable")
    return path


def _load_context_record(sidecar_root: Path, scene: str) -> dict[str, Any]:
    test_root = Path(sidecar_root).resolve() / "test"
    index = _read_json(test_root / "index.json", "ACID joint sidecar index")
    if scene not in index:
        raise ValueError("ACID joint sidecar is missing its planned scene")
    chunk = torch.load(_safe_chunk_path(test_root, index[scene]), map_location="cpu")
    if not isinstance(chunk, list) or len(chunk) != 1 or not isinstance(chunk[0], dict):
        raise ValueError("ACID joint sidecar record is invalid")
    record = chunk[0]
    if (
        set(record) != _RECORD_FIELDS
        or record.get("schema_version") != "1.0"
        or record.get("kind") != CONTEXT_RECORD_KIND
        or record.get("key") != scene
    ):
        raise ValueError("ACID joint sidecar record has forbidden fields")
    indices = record.get("context_indices")
    if (
        not isinstance(indices, list)
        or len(indices) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indices)
        or len(indices) != len(set(indices))
    ):
        raise ValueError("ACID joint sidecar record has invalid context indices")
    source_view_count = record.get("source_view_count")
    if (
        isinstance(source_view_count, bool)
        or not isinstance(source_view_count, int)
        or source_view_count < 5
        or max(indices) >= source_view_count
    ):
        raise ValueError("ACID joint sidecar record has invalid source view metadata")
    cameras = record.get("context_cameras")
    images = record.get("context_images")
    if (
        not hasattr(cameras, "__len__")
        or len(cameras) != 2
        or not isinstance(images, list)
        or len(images) != 2
    ):
        raise ValueError("ACID joint sidecar record has invalid context payload")
    return record


def _decode_context_images(images: list[Any]) -> torch.Tensor:
    decoded = []
    for value in images:
        if torch.is_tensor(value):
            if value.dtype != torch.uint8:
                raise ValueError("ACID joint context image must be uint8 bytes")
            payload = value.detach().cpu().contiguous().numpy().tobytes()
        elif isinstance(value, (bytes, bytearray)):
            payload = bytes(value)
        else:
            raise ValueError("ACID joint context image has an unsupported type")
        with Image.open(BytesIO(payload)) as image:
            pixels = np.array(image.convert("RGB"), copy=True)
        decoded.append(torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0)
    if len(decoded) != 2:
        raise ValueError("ACID joint sidecar must contain exactly two context images")
    return torch.stack(decoded, dim=0)


def _context_camera_geometry(cameras: Any) -> tuple[torch.Tensor, torch.Tensor]:
    cameras = torch.as_tensor(cameras, dtype=torch.float32).clone()
    if (
        cameras.ndim != 2
        or tuple(cameras.shape) != (2, 18)
        or not bool(torch.isfinite(cameras).all())
    ):
        raise ValueError("ACID joint context cameras must have shape [2, 18]")
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(2, 1, 1)
    intrinsics[:, 0, 0] = cameras[:, 0]
    intrinsics[:, 1, 1] = cameras[:, 1]
    intrinsics[:, 0, 2] = cameras[:, 2]
    intrinsics[:, 1, 2] = cameras[:, 3]
    w2c = torch.eye(4, dtype=torch.float32).repeat(2, 1, 1)
    w2c[:, :3] = cameras[:, 6:].reshape(2, 3, 4)
    return torch.linalg.inv(w2c), intrinsics


def load_acid_joint_context(
    *,
    materialization_root: Path,
    split: str,
    sample_index: int,
    plan_path: Path = DEFAULT_PLAN_PATH,
) -> AcidJointContext:
    """Return one planned, validated ACID context-only training input.

    The materialization validator runs before the sidecar record is opened. It
    binds the plan and sidecar hashes, rechecks evaluation disjointness, and
    rejects target/teacher/paper-result artifacts. This loader adds only tensor
    decoding and camera conversion; it imports neither a model loader nor an
    upstream ``src`` package.
    """
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("ACID joint sample index must be nonnegative")
    plan = _read_json(plan_path, "ACID joint plan")
    materialization = validate_materialization(materialization_root, plan=plan)
    scenes = _split_scenes(plan, split)
    if sample_index >= len(scenes):
        raise IndexError(f"ACID joint sample_index={sample_index} is out of range")
    sidecars = materialization.get("sidecars")
    sidecar_identity = sidecars.get(split) if isinstance(sidecars, Mapping) else None
    if not isinstance(sidecar_identity, Mapping):
        raise ValueError("ACID joint materialization has no split sidecar identity")
    sidecar_root = Path(materialization_root).resolve() / "inputs" / split
    scene = scenes[sample_index]
    record = _load_context_record(sidecar_root, scene)
    images = _decode_context_images(record["context_images"])
    extrinsics, intrinsics = _context_camera_geometry(record["context_cameras"])
    context = {
        "image": images.unsqueeze(0),
        "extrinsics": extrinsics.unsqueeze(0),
        "intrinsics": intrinsics.unsqueeze(0),
        "index": torch.tensor(record["context_indices"], dtype=torch.long).unsqueeze(0),
    }
    identity = {
        "plan_sha256": plan["plan_sha256"],
        "split": split,
        "sample_index": sample_index,
        "scene": scene,
        "sidecar": dict(sidecar_identity),
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "teacher_artifact_accessed": False,
        "expected_results_accessed": False,
    }
    return AcidJointContext(context=context, identity=identity)


__all__ = [
    "AcidJointContext",
    "DEFAULT_PLAN_PATH",
    "HOLDOUT_SPLIT",
    "TRAIN_SPLIT",
    "load_acid_joint_context",
]
