#!/usr/bin/env python3
"""Prepare one selected DL3DV context-only sidecar for a target-free SAES audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_manifest import build as build_dataset_manifest
from data.convert_dl3dv import (
    IMAGE_SUFFIXES,
    _encode_image,
    _frame_id,
    _load_metadata,
    _sha256_file,
    load_scene_image_sources,
)
from scripts.calibration_inputs import materialize_target_free_inputs
from scripts.compile_protocol import canonicalize_index


SUPPORTED_MODELS = ("transplat", "mvsplat", "depthsplat")
DEFAULT_MODEL = "transplat"
SAMPLE_INDEX = 0
CONTEXT_COUNT = 2
TARGET_COUNT = 4
# The fixed protocols share view indices, but not their image representation.
# DepthSplat is evaluated at its native 270x480 resolution; classic backends
# retain the existing Re10K-compatible 360x640 preprocessing route.
MODEL_IMAGE_SPECS = {
    "transplat": ("re10k", (360, 640), True),
    "mvsplat": ("re10k", (360, 640), True),
    "depthsplat": ("native", (270, 480), False),
}
DEFAULT_RAW_ROOT = ROOT / "downloads" / "dl3dv-benchmark"
DEFAULT_PROTOCOL = ROOT / "artifact" / "evaluation_protocol.json"


class AuditInputError(ValueError):
    """Raised when the fixed target-free audit contract cannot be established."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditInputError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuditInputError(f"{label} must be a JSON object: {path}")
    return value


def _sample_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditInputError("DL3DV sample index must be a nonnegative integer")
    return value


def _model_name(value: Any) -> str:
    if value not in SUPPORTED_MODELS:
        raise AuditInputError(
            "target-free audit requires one of " + ", ".join(SUPPORTED_MODELS)
        )
    return str(value)


def _protocol_selection(
    protocol_path: Path, *, model: str, sample_index: int
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    model = _model_name(model)
    sample_index = _sample_index(sample_index)
    protocol = _load_json(protocol_path, "evaluation protocol")
    pairs = protocol.get("pairs")
    pair = f"{model}/dl3dv"
    contract = pairs.get(pair) if isinstance(pairs, Mapping) else None
    if not isinstance(contract, Mapping):
        raise AuditInputError(f"evaluation protocol has no {pair} contract")
    index_path = contract.get("index_path")
    index_sha256 = contract.get("source_index_sha256")
    if not isinstance(index_path, str) or not isinstance(index_sha256, str):
        raise AuditInputError("DL3DV protocol has no source-bound index")
    resolved_index = (ROOT / index_path).resolve()
    rows, summary = canonicalize_index(resolved_index, index_sha256)
    if len(rows) != contract.get("sample_count"):
        raise AuditInputError("DL3DV protocol sample count does not match its index")
    rows_by_source_index = {row["sample_index"]: row for row in rows}
    if sample_index not in rows_by_source_index:
        raise AuditInputError("DL3DV protocol has no requested sample index")
    row = rows_by_source_index[sample_index]
    context = row.get("context_indices")
    target = row.get("target_indices")
    if (
        not isinstance(context, list)
        or not isinstance(target, list)
        or len(context) != CONTEXT_COUNT
        or len(target) != TARGET_COUNT
    ):
        raise AuditInputError("fixed DL3DV sample does not have 2 context and 4 target views")
    if set(context) & set(target):
        raise AuditInputError("fixed DL3DV context and target views overlap")
    if summary["sample_selection_sha256"] != contract.get("sample_selection_sha256"):
        raise AuditInputError("DL3DV protocol selection hash does not match its contract")
    return dict(contract), resolved_index, row, summary


def _image_sources(
    raw_root: Path,
    *,
    scene: str,
    context_indices: list[int],
    source_key: str,
    target_shape: tuple[int, int],
    allow_resize: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[Any], list[dict[str, str]]]:
    source_path = raw_root / ".scarf-dl3dv-source.json"
    source = _load_json(source_path, "DL3DV benchmark source record")
    source_plans = load_scene_image_sources(source_path, source_key)
    try:
        image_subdir, source_shape = source_plans[scene]
    except KeyError as exc:
        raise AuditInputError("DL3DV source plan has no fixed audit scene") from exc

    nerfstudio = raw_root / scene / "nerfstudio"
    transforms = nerfstudio / "transforms.json"
    metadata, timestamps = _load_metadata(transforms)
    if max(context_indices) >= len(timestamps):
        raise AuditInputError("fixed DL3DV context index exceeds scene metadata")
    image_root = nerfstudio / image_subdir
    if not image_root.is_dir():
        raise AuditInputError("fixed DL3DV context image directory is missing")
    images = {
        _frame_id(path): path
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if len(images) != sum(
        1 for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ):
        raise AuditInputError("fixed DL3DV image directory has duplicate frame identifiers")

    serialized_images: list[Any] = [None] * len(timestamps)
    opened = [
        {
            "path": f"{scene}/nerfstudio/transforms.json",
            "role": "camera_geometry",
            "sha256": _sha256_file(transforms),
        }
    ]
    for index in context_indices:
        frame = timestamps[index]
        path = images.get(frame)
        if path is None:
            raise AuditInputError(f"fixed DL3DV context frame is missing: {frame}")
        serialized_images[index] = _encode_image(
            path,
            target_shape,
            allow_resize=allow_resize,
            accepted_source_shapes=(source_shape,),
        )
        opened.append(
            {
                "path": path.relative_to(raw_root).as_posix(),
                "role": "context_rgb",
                "sha256": _sha256_file(path),
            }
        )
    return source, metadata, serialized_images, opened


def prepare_inputs(
    raw_root: Path,
    *,
    output_dir: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    model: str = DEFAULT_MODEL,
    sample_index: int = SAMPLE_INDEX,
) -> dict[str, Any]:
    """Write one source-bound sidecar without decoding target RGB."""
    raw_root = Path(raw_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"target-free audit output already exists: {output_dir}")
    model = _model_name(model)
    sample_index = _sample_index(sample_index)
    source_key, target_shape, allow_resize = MODEL_IMAGE_SPECS[model]
    contract, index_path, row, selection_summary = _protocol_selection(
        protocol_path, model=model, sample_index=sample_index
    )
    scene = row["scene"]
    context_indices = list(row["context_indices"])
    target_indices = list(row["target_indices"])
    source, metadata, serialized_images, opened_source_files = _image_sources(
        raw_root,
        scene=scene,
        context_indices=context_indices,
        source_key=source_key,
        target_shape=target_shape,
        allow_resize=allow_resize,
    )
    source_record_path = raw_root / ".scarf-dl3dv-source.json"
    source_record_sha256 = _sha256_file(source_record_path)
    selection = {
        scene: {
            "context": context_indices,
            "target": target_indices,
        }
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = output_dir / "audit-selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selection_sha256 = _sha256_file(selection_path)
    _, audit_selection_summary = canonicalize_index(selection_path, selection_sha256)
    example = {
        "key": scene,
        "cameras": metadata["cameras"],
        "images": serialized_images,
    }
    sidecar = materialize_target_free_inputs(
        output_dir / "sidecar",
        dataset="dl3dv",
        examples={scene: example},
        index=selection,
        source={"prepared_source_sha256": source_record_sha256},
        selection_sha256=audit_selection_summary["sample_selection_sha256"],
    )
    record = {
        "schema_version": "1.0",
        "kind": "dl3dv_target_free_l1_primary_reference_audit_input",
        "status": "PASS",
        "paper_result_eligible": False,
        "model": model,
        "dataset": "dl3dv",
        "source_sample_index": sample_index,
        "selected_sample": {
            "scene": scene,
            "context_indices": context_indices,
            "target_indices": target_indices,
            "audit_selection_sha256": audit_selection_summary["sample_selection_sha256"],
            "audit_selection_file_sha256": selection_sha256,
        },
        "canonical_selection": {
            "source_sample_index": row["sample_index"],
            "scene": scene,
            "context_indices": context_indices,
            "target_indices": target_indices,
        },
        "canonical_protocol": {
            "pair": f"{model}/dl3dv",
            "index_path": contract["index_path"],
            "source_index_sha256": contract["source_index_sha256"],
            "sample_selection_sha256": selection_summary["sample_selection_sha256"],
            "dataset_tree_sha256": contract["dataset_tree_sha256"],
        },
        "source": {
            "revision": source.get("revision"),
            "benchmark_metadata_sha256": source.get("benchmark_metadata_sha256"),
            "filelist_sha256": source.get("filelist_sha256"),
            "scene_source_plans_sha256": source.get("scene_source_plans_sha256"),
            "source_record_sha256": source_record_sha256,
        },
        "target_rgb_included": False,
        "target_rgb_opened": False,
        "target_rgb_paths_passed_to_encoder": False,
        "opened_source_files": opened_source_files,
        "opened_source_file_count": len(opened_source_files),
        "sidecar": sidecar,
    }
    record_path = output_dir / "audit-input.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = build_dataset_manifest(
        output_dir,
        "dl3dv-target-free-audit",
        "official DL3DV benchmark context-only sidecar",
        f"protocol:{contract['source_index_sha256']};sample:{sample_index}",
    )
    manifest_path = output_dir / ".scarf-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **record,
        "output_tree_sha256": manifest["tree_sha256"],
        "output_manifest_sha256": _sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX)
    args = parser.parse_args()
    try:
        record = prepare_inputs(
            args.raw_root,
            output_dir=args.output_dir,
            model=args.model,
            sample_index=args.sample_index,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir / "audit-input.json")
    print(record["output_tree_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
