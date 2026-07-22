"""Build and validate target-free inputs for mechanism calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


INPUT_KIND = "scarf_target_free_calibration_input_v1"
RECORD_KIND = "scarf_target_free_calibration_record_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_chunk_path(test_root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("calibration input index has a non-string chunk path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe calibration input chunk path: {value}")
    path = (test_root / Path(*pure.parts)).resolve()
    if test_root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f"calibration input chunk is missing: {value}")
    return path


def _indices(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in value
        )
        or len(value) != len(set(value))
    ):
        raise ValueError(f"calibration record has invalid {label} indices")
    return list(value)


def build_target_free_record(
    example: Mapping[str, Any],
    *,
    scene: str,
    context_indices: list[int],
    target_indices: list[int],
) -> dict[str, Any]:
    """Keep only model inputs and target camera geometry for one scene."""
    if example.get("key") != scene:
        raise ValueError("calibration example scene does not match its index")
    images = example.get("images")
    cameras = example.get("cameras")
    if images is None or not hasattr(images, "__len__"):
        raise ValueError(f"calibration scene has no image sequence: {scene}")
    if cameras is None:
        raise ValueError(f"calibration scene has no camera geometry: {scene}")
    context = _indices(context_indices, "context")
    target = _indices(target_indices, "target")
    if max([*context, *target]) >= len(images):
        raise ValueError(f"calibration view index exceeds source scene: {scene}")
    return {
        "schema_version": "1.0",
        "kind": RECORD_KIND,
        "key": scene,
        "cameras": cameras,
        "context_indices": context,
        "target_indices": target,
        # Target RGB is deliberately omitted. The replay obtains target poses
        # from the camera tensor and decodes only these model-input views.
        "context_images": [images[index] for index in context],
    }


def materialize_target_free_inputs(
    output_root: Path,
    *,
    dataset: str,
    examples: Mapping[str, Mapping[str, Any]],
    index: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Any],
    selection_sha256: str,
) -> dict[str, Any]:
    """Write deterministic one-scene chunks that exclude every target RGB image."""
    import torch

    output_root = Path(output_root).resolve()
    test_root = output_root / "test"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"calibration input destination already exists: {output_root}")
    test_root.mkdir(parents=True, exist_ok=True)
    selected = list(index)
    if set(selected) != set(examples):
        missing = sorted(set(selected) - set(examples))
        extra = sorted(set(examples) - set(selected))
        raise ValueError(
            "calibration examples do not match the selected index: "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )

    input_index: dict[str, str] = {}
    for ordinal, scene in enumerate(selected):
        selection = index[scene]
        if not isinstance(selection, Mapping):
            raise ValueError(f"calibration selection is invalid: {scene}")
        record = build_target_free_record(
            examples[scene],
            scene=scene,
            context_indices=_indices(selection.get("context"), "context"),
            target_indices=_indices(selection.get("target"), "target"),
        )
        name = f"{ordinal:06d}.torch"
        torch.save([record], test_root / name)
        input_index[scene] = name

    (test_root / "index.json").write_text(
        json.dumps(input_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    opened_file_manifest = [
        {
            "path": "test/index.json",
            "role": "target_free_sidecar_index",
            "sha256": sha256_file(test_root / "index.json"),
        },
        *[
            {
                "path": f"test/{input_index[scene]}",
                "role": "target_free_sidecar_record",
                "sha256": sha256_file(test_root / input_index[scene]),
            }
            for scene in selected
        ],
    ]
    source_record = {
        "schema_version": "1.0",
        "kind": INPUT_KIND,
        "dataset": dataset,
        "target_rgb_included": False,
        "selection_sha256": selection_sha256,
        "selected_scene_count": len(selected),
        # This is the exact allowlist of serialized files from which the
        # calibration loader may decode model inputs. It names only target-free
        # sidecars, never upstream source images or target RGB payloads.
        "opened_file_manifest": opened_file_manifest,
        "opened_file_manifest_sha256": _canonical_sha256(opened_file_manifest),
        "source_dataset_tree_sha256": source.get("dataset_tree_sha256"),
        "source_dataset_manifest_sha256": source.get("dataset_manifest_sha256"),
        "source_prepared_provenance_sha256": source.get("prepared_source_sha256"),
    }
    source_path = output_root / ".scarf-calibration-input.json"
    source_path.write_text(
        json.dumps(source_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from data.build_manifest import build as build_dataset_manifest

    manifest = build_dataset_manifest(
        output_root,
        dataset,
        "target-free calibration sidecar",
        f"selection:{selection_sha256}",
    )
    manifest_path = output_root / ".scarf-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "calibration_input_root": f"inputs/{dataset}",
        "calibration_input_manifest_sha256": sha256_file(manifest_path),
        "calibration_input_tree_sha256": manifest["tree_sha256"],
        "calibration_input_provenance_sha256": sha256_file(source_path),
        "opened_file_manifest": opened_file_manifest,
        "opened_file_manifest_sha256": _canonical_sha256(opened_file_manifest),
        "target_rgb_included": False,
    }


def validate_target_free_input_root(root: Path, dataset: str) -> dict[str, Any]:
    """Verify that a calibration sidecar is complete and contains no target RGB."""
    from data.verify_prepared_dataset import verify_tree_manifest

    root = Path(root).resolve()
    source_path = root / ".scarf-calibration-input.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"calibration input provenance is missing: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("kind") != INPUT_KIND or source.get("dataset") != dataset:
        raise ValueError("calibration input provenance is invalid")
    if source.get("target_rgb_included") is not False:
        raise ValueError("calibration input contains target RGB")
    selection_sha256 = source.get("selection_sha256")
    if (
        not isinstance(selection_sha256, str)
        or len(selection_sha256) != 64
        or any(character not in "0123456789abcdef" for character in selection_sha256)
    ):
        raise ValueError("calibration input selection SHA256 is invalid")
    count = source.get("selected_scene_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("calibration input scene count is invalid")
    opened_file_manifest = source.get("opened_file_manifest")
    opened_file_manifest_sha256 = source.get("opened_file_manifest_sha256")
    if (
        not isinstance(opened_file_manifest, list)
        or len(opened_file_manifest) != count + 1
        or opened_file_manifest_sha256 != _canonical_sha256(opened_file_manifest)
    ):
        raise ValueError("calibration input opened-file manifest is invalid")
    expected_paths = {"test/index.json"}
    index = json.loads((root / "test" / "index.json").read_text(encoding="utf-8"))
    if not isinstance(index, dict) or len(index) != count:
        raise ValueError("calibration input index is invalid")
    expected_paths.update(f"test/{value}" for value in index.values())
    manifest_paths = set()
    for item in opened_file_manifest:
        if not isinstance(item, dict):
            raise ValueError("calibration input opened-file record is invalid")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or relative not in expected_paths
            or "target" in relative.lower()
            or item.get("role") not in {
                "target_free_sidecar_index",
                "target_free_sidecar_record",
            }
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
        ):
            raise ValueError("calibration input opened-file record is unsafe")
        path = root / relative
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError("calibration input opened-file manifest hash mismatch")
        manifest_paths.add(relative)
    if manifest_paths != expected_paths:
        raise ValueError("calibration input opened-file manifest is incomplete")
    tree = verify_tree_manifest(root, root / ".scarf-manifest.json")
    return {
        "representation": f"{dataset}-target-free-calibration-v1",
        "tree_sha256": tree["tree_sha256"],
        "manifest_sha256": tree["manifest_sha256"],
        "target_rgb_accessed": False,
        "target_rgb_included": False,
        "selected_scene_count": count,
        "selection_sha256": selection_sha256,
        "input_provenance_sha256": sha256_file(source_path),
        "opened_file_manifest": opened_file_manifest,
        "opened_file_manifest_sha256": opened_file_manifest_sha256,
    }


def calibration_scene_order(root: Path) -> list[str]:
    root = Path(root).resolve()
    test_root = root / "test"
    index = json.loads((test_root / "index.json").read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not index:
        raise ValueError("calibration input index is invalid")
    by_chunk: dict[Path, list[str]] = {}
    for scene, value in index.items():
        if not isinstance(scene, str) or not scene:
            raise ValueError("calibration input index has an invalid scene")
        by_chunk.setdefault(_safe_chunk_path(test_root, value), []).append(scene)
    return [scene for path in sorted(by_chunk) for scene in by_chunk[path]]


def load_target_free_record(root: Path, scene: str) -> dict[str, Any]:
    """Load one sidecar record and reject a format that carries target images."""
    import torch

    root = Path(root).resolve()
    test_root = root / "test"
    index = json.loads((test_root / "index.json").read_text(encoding="utf-8"))
    if scene not in index:
        raise ValueError(f"calibration input has no selected scene: {scene}")
    chunk = torch.load(_safe_chunk_path(test_root, index[scene]), map_location="cpu")
    if not isinstance(chunk, list) or len(chunk) != 1 or not isinstance(chunk[0], dict):
        raise ValueError(f"calibration input chunk is invalid: {scene}")
    record = chunk[0]
    if record.get("kind") != RECORD_KIND or record.get("key") != scene:
        raise ValueError(f"calibration input record is invalid: {scene}")
    if any(key in record for key in ("images", "target_images", "target_image", "target_rgb")):
        raise ValueError("calibration input record contains target RGB")
    context = _indices(record.get("context_indices"), "context")
    _indices(record.get("target_indices"), "target")
    context_images = record.get("context_images")
    if not isinstance(context_images, list) or len(context_images) != len(context):
        raise ValueError("calibration input record has invalid context images")
    if record.get("cameras") is None:
        raise ValueError("calibration input record has no camera geometry")
    return record
