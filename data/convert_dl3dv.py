#!/usr/bin/env python3
"""Convert the DL3DV benchmark into a deterministic chunked test dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image


DEFAULT_CHUNK_BYTES = 200_000_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _load_scene_index(path: Path) -> list[str]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("DL3DV scene index must be a JSON object")
    keys = sorted(source)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("DL3DV scene index must contain non-empty scene keys")
    return keys


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _frame_id(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"image has no numeric frame suffix: {path.name}") from exc


def _bytes_tensor(data: bytes) -> torch.Tensor:
    array = np.frombuffer(data, dtype=np.uint8).copy()
    return torch.from_numpy(array)


def _encode_image(
    path: Path,
    target_shape: tuple[int, int],
    allow_resize: bool,
    accepted_source_shapes: tuple[tuple[int, int], ...] | None,
) -> torch.Tensor:
    target_height, target_width = target_shape
    with Image.open(path) as image:
        source_width, source_height = image.size
        if (
            accepted_source_shapes is not None
            and (source_height, source_width) not in accepted_source_shapes
        ):
            expected = ", ".join(
                f"{height}x{width}" for height, width in accepted_source_shapes
            )
            raise ValueError(
                f"{path.name} is {source_height}x{source_width}; expected one of "
                f"{expected}"
            )
        if (source_height, source_width) == target_shape:
            return _bytes_tensor(path.read_bytes())
        if not allow_resize:
            raise ValueError(
                f"{path.name} is {source_height}x{source_width}; "
                f"expected {target_height}x{target_width}"
            )
        if source_width * target_height != source_height * target_width:
            raise ValueError(
                f"{path.name} aspect ratio cannot be resized without cropping: "
                f"{source_height}x{source_width} to {target_height}x{target_width}"
            )
        resized = image.convert("RGB").resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        encoded = io.BytesIO()
        resized.save(
            encoded,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
    return _bytes_tensor(encoded.getvalue())


def _load_metadata(path: Path) -> tuple[dict[str, Any], list[int]]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = ("h", "w", "fl_x", "fl_y", "cx", "cy", "frames")
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    height = float(metadata["h"])
    width = float(metadata["w"])
    if height <= 0 or width <= 0:
        raise ValueError(f"{path} has invalid image dimensions")

    blender_to_opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    intrinsics = [
        float(metadata["fl_x"]) / width,
        float(metadata["fl_y"]) / height,
        float(metadata["cx"]) / width,
        float(metadata["cy"]) / height,
        0.0,
        0.0,
    ]
    timestamps: list[int] = []
    cameras = []
    for frame in metadata["frames"]:
        frame_path = Path(frame["file_path"])
        timestamps.append(_frame_id(frame_path))
        transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"invalid transform shape for {frame_path}: {transform.shape}")
        opencv_c2w = transform @ blender_to_opencv
        camera = intrinsics + np.linalg.inv(opencv_c2w)[:3].reshape(-1).tolist()
        cameras.append(camera)
    if not cameras:
        raise ValueError(f"{path} contains no frames")
    return {
        "url": path.parent.parent.name,
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "cameras": torch.tensor(np.asarray(cameras), dtype=torch.float32),
    }, timestamps


def _load_scene(
    nerfstudio_dir: Path,
    image_subdir: str,
    target_shape: tuple[int, int],
    allow_resize: bool,
    accepted_source_shapes: tuple[tuple[int, int], ...] | None,
) -> dict[str, Any]:
    scene_key = nerfstudio_dir.parent.name
    image_dir = nerfstudio_dir / image_subdir
    metadata_path = nerfstudio_dir / "transforms.json"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing image directory {image_subdir}")
    if not metadata_path.is_file():
        raise FileNotFoundError("missing transforms.json")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    images = {_frame_id(path): path for path in image_paths}
    if len(images) != len(image_paths):
        raise ValueError("duplicate numeric frame suffix")
    example, timestamps = _load_metadata(metadata_path)
    missing = [timestamp for timestamp in timestamps if timestamp not in images]
    if missing:
        raise ValueError(f"missing image frames: {missing[:8]}")
    example["images"] = [
        _encode_image(
            images[timestamp],
            target_shape,
            allow_resize,
            accepted_source_shapes,
        )
        for timestamp in timestamps
    ]
    example["key"] = scene_key
    return example


def load_scene_image_sources(
    path: Path, source_key: str
) -> dict[str, tuple[str, tuple[int, int]]]:
    """Load one representation's exact source selection from the raw record."""

    record = json.loads(path.read_text(encoding="utf-8"))
    plans = record.get("scene_source_plans")
    if not isinstance(plans, dict) or not plans:
        raise ValueError("DL3DV source record has no scene source plans")
    if record.get("scene_source_plans_sha256") != _canonical_sha256(plans):
        raise ValueError("DL3DV source record scene source plan SHA256 mismatch")
    sources = {}
    for scene, plan in plans.items():
        if not isinstance(scene, str) or not scene or not isinstance(plan, dict):
            raise ValueError("DL3DV source plan has an invalid scene entry")
        item = plan.get(source_key)
        if not isinstance(item, dict):
            raise ValueError(f"DL3DV source plan has no {source_key} entry for {scene}")
        image_subdir = item.get("image_subdir")
        shape = item.get("source_image_shape")
        if (
            not isinstance(image_subdir, str)
            or not image_subdir
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError(f"DL3DV source plan has invalid {source_key} metadata for {scene}")
        sources[scene] = (image_subdir, (shape[0], shape[1]))
    return sources


def convert(
    input_dir: Path,
    output_dir: Path,
    *,
    image_subdir: str | None,
    target_shape: tuple[int, int],
    allow_resize: bool,
    schema: str,
    source: str,
    revision: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    scene_index: Path | None = None,
    accepted_source_shapes: tuple[tuple[int, int], ...] | None = None,
    scene_image_sources: Mapping[str, tuple[str, tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if image_subdir is None and scene_image_sources is None:
        raise ValueError("image_subdir or scene_image_sources is required")
    if accepted_source_shapes is not None:
        if not accepted_source_shapes or any(
            dimension <= 0
            for shape in accepted_source_shapes
            for dimension in shape
        ):
            raise ValueError("accepted_source_shapes dimensions must be positive")
        if len(set(accepted_source_shapes)) != len(accepted_source_shapes):
            raise ValueError("accepted_source_shapes must not contain duplicates")
    stage_dir = output_dir / "test"
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(input_dir.glob("*/nerfstudio"))
    if not scene_dirs:
        raise FileNotFoundError(f"no DL3DV nerfstudio scenes found in {input_dir}")
    selected_scene_keys = _load_scene_index(scene_index) if scene_index is not None else None
    if selected_scene_keys is not None:
        discovered = {path.parent.name: path for path in scene_dirs}
        missing = [key for key in selected_scene_keys if key not in discovered]
        if missing:
            raise FileNotFoundError(
                f"DL3DV source is missing {len(missing)} indexed scenes: {missing[:4]}"
            )
        scene_dirs = [discovered[key] for key in selected_scene_keys]
    scene_keys = [path.parent.name for path in scene_dirs]
    if scene_image_sources is not None:
        missing_sources = sorted(set(scene_keys) - set(scene_image_sources))
        extra_sources = sorted(set(scene_image_sources) - set(scene_keys))
        if missing_sources or extra_sources:
            raise ValueError(
                "scene image source plan does not match converted scenes: "
                f"missing={missing_sources[:3]}, extra={extra_sources[:3]}"
            )

    chunk: list[dict[str, Any]] = []
    current_bytes = 0
    chunk_count = 0
    converted_keys: list[str] = []
    skipped: list[dict[str, str]] = []

    def flush() -> None:
        nonlocal chunk, current_bytes, chunk_count
        if not chunk:
            return
        torch.save(chunk, stage_dir / f"{chunk_count:06d}.torch")
        chunk = []
        current_bytes = 0
        chunk_count += 1

    for nerfstudio_dir in scene_dirs:
        try:
            scene_key = nerfstudio_dir.parent.name
            effective_subdir = image_subdir
            effective_shapes = accepted_source_shapes
            if scene_image_sources is not None:
                effective_subdir, expected_shape = scene_image_sources[scene_key]
                effective_shapes = (expected_shape,)
            assert effective_subdir is not None
            example = _load_scene(
                nerfstudio_dir,
                effective_subdir,
                target_shape,
                allow_resize,
                effective_shapes,
            )
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
            if selected_scene_keys is not None:
                raise ValueError(
                    f"indexed DL3DV scene cannot be converted: "
                    f"{nerfstudio_dir.parent.name}: {exc}"
                ) from exc
            skipped.append({"scene": nerfstudio_dir.parent.name, "reason": str(exc)})
            continue
        encoded_bytes = sum(int(image.numel()) for image in example["images"])
        chunk.append(example)
        current_bytes += encoded_bytes
        converted_keys.append(example["key"])
        if current_bytes >= chunk_bytes:
            flush()
    flush()
    if not converted_keys:
        raise ValueError("no valid DL3DV scenes were converted")
    if selected_scene_keys is not None and converted_keys != selected_scene_keys:
        raise ValueError("converted DL3DV scenes do not match the fixed scene index")

    record = {
        "schema_version": "1.0",
        "dataset": "dl3dv",
        "representation": schema,
        "source": source,
        "revision": revision,
        "source_image_subdir": image_subdir,
        "accepted_source_image_shapes": [list(shape) for shape in accepted_source_shapes]
        if accepted_source_shapes is not None
        else None,
        "scene_image_sources": {
            scene: {
                "image_subdir": subdir,
                "source_image_shape": list(shape),
            }
            for scene, (subdir, shape) in sorted(scene_image_sources.items())
        }
        if scene_image_sources is not None
        else None,
        "scene_image_sources_sha256": _canonical_sha256(
            {
                scene: {
                    "image_subdir": subdir,
                    "source_image_shape": list(shape),
                }
                for scene, (subdir, shape) in sorted(scene_image_sources.items())
            }
        )
        if scene_image_sources is not None
        else None,
        "target_image_shape": list(target_shape),
        "resize": {
            "enabled": allow_resize,
            "method": "Pillow LANCZOS, JPEG quality 95, 4:4:4"
            if allow_resize
            else None,
        },
        "scene_count": len(converted_keys),
        "chunk_count": chunk_count,
        "scene_keys": converted_keys,
        "scene_index": str(scene_index) if scene_index is not None else None,
        "scene_index_sha256": _sha256_file(scene_index) if scene_index is not None else None,
        "skipped": skipped,
    }
    (output_dir / "conversion.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _source_shape(value: str) -> tuple[int, int]:
    try:
        height_text, width_text = value.lower().split("x", 1)
        height = int(height_text)
        width = int(width_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source shape must use HEIGHTxWIDTH") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("source shape dimensions must be positive")
    return height, width


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-subdir")
    parser.add_argument("--target-height", type=int, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--scene-index", type=Path)
    parser.add_argument("--source-shape", action="append", type=_source_shape)
    parser.add_argument("--scene-source-plan", type=Path)
    parser.add_argument("--scene-source-key")
    args = parser.parse_args()
    if (args.scene_source_plan is None) != (args.scene_source_key is None):
        parser.error("--scene-source-plan and --scene-source-key must be used together")
    if args.image_subdir is None and args.scene_source_plan is None:
        parser.error("--image-subdir or --scene-source-plan is required")
    if args.scene_source_plan is not None and args.source_shape:
        parser.error("--source-shape cannot be combined with --scene-source-plan")
    scene_image_sources = (
        load_scene_image_sources(args.scene_source_plan, args.scene_source_key)
        if args.scene_source_plan is not None
        else None
    )
    convert(
        args.input_dir,
        args.output_dir,
        image_subdir=args.image_subdir,
        target_shape=(args.target_height, args.target_width),
        allow_resize=args.resize,
        schema=args.schema,
        source=args.source,
        revision=args.revision,
        chunk_bytes=args.chunk_bytes,
        scene_index=args.scene_index,
        accepted_source_shapes=tuple(args.source_shape) if args.source_shape else None,
        scene_image_sources=scene_image_sources,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
