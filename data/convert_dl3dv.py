#!/usr/bin/env python3
"""Convert the DL3DV benchmark into a deterministic chunked test dataset."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_CHUNK_BYTES = 200_000_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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
) -> torch.Tensor:
    target_height, target_width = target_shape
    with Image.open(path) as image:
        source_width, source_height = image.size
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
        _encode_image(images[timestamp], target_shape, allow_resize)
        for timestamp in timestamps
    ]
    example["key"] = scene_key
    return example


def convert(
    input_dir: Path,
    output_dir: Path,
    *,
    image_subdir: str,
    target_shape: tuple[int, int],
    allow_resize: bool,
    schema: str,
    source: str,
    revision: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    stage_dir = output_dir / "test"
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(input_dir.glob("*/nerfstudio"))
    if not scene_dirs:
        raise FileNotFoundError(f"no DL3DV nerfstudio scenes found in {input_dir}")

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
            example = _load_scene(
                nerfstudio_dir, image_subdir, target_shape, allow_resize
            )
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
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

    record = {
        "schema_version": "1.0",
        "dataset": "dl3dv",
        "representation": schema,
        "source": source,
        "revision": revision,
        "source_image_subdir": image_subdir,
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
        "skipped": skipped,
    }
    (output_dir / "conversion.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-subdir", required=True)
    parser.add_argument("--target-height", type=int, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    args = parser.parse_args()
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
