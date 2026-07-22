#!/usr/bin/env python3
"""Compile DL3DV calibration archives into target-free model input sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.convert_dl3dv import IMAGE_SUFFIXES, _encode_image, _frame_id, _load_metadata
from data.download_dl3dv_calibration import sha256_file
from data.plan_dl3dv_calibration import canonical_sha256
from data.verify_prepared_dataset import verify_tree_manifest
from scripts.calibration_inputs import materialize_target_free_inputs
from scripts.compile_calibration import deterministic_views
from scripts.compile_protocol import canonicalize_index


NATIVE = "native"
RE10K = "re10k"
REPRESENTATIONS = (NATIVE, RE10K)
TRAIN_SPLIT = "calibration_train"
HOLDOUT_SPLIT = "calibration_holdout"
SPLITS = (TRAIN_SPLIT, HOLDOUT_SPLIT)
SPLIT_COUNTS = {TRAIN_SPLIT: 24, HOLDOUT_SPLIT: 8}
NATIVE_SHAPE = (270, 480)
RE10K_SHAPE = (360, 640)


class PreparationContractError(ValueError):
    """Raised when an extracted archive tree cannot be calibrated safely."""


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreparationContractError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationContractError(f"{label} must be a JSON object: {path}")
    return value


def _split_archives(plan: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise PreparationContractError(f"unknown DL3DV calibration split: {split}")
    selection = plan.get("selection")
    rows = selection.get(split) if isinstance(selection, Mapping) else None
    if not isinstance(rows, list) or len(rows) != SPLIT_COUNTS[split]:
        raise PreparationContractError(
            f"DL3DV plan has no {SPLIT_COUNTS[split]}-scene {split} selection"
        )
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PreparationContractError("DL3DV plan has an invalid calibration archive")
        scene, path, size, oid, object_id_kind = (
            row.get("scene"),
            row.get("path"),
            row.get("size"),
            row.get("oid"),
            row.get("object_id_kind"),
        )
        if (
            not isinstance(scene, str)
            or not scene
            or not isinstance(path, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(oid, str)
            or object_id_kind not in {"lfs_sha256", "git_blob"}
            or len(oid) != (64 if object_id_kind == "lfs_sha256" else 40)
            or any(char not in "0123456789abcdef" for char in oid)
        ):
            raise PreparationContractError("DL3DV plan has invalid calibration metadata")
        result.append(
            {
                "scene": scene,
                "path": path,
                "size": size,
                "oid": oid,
                "object_id_kind": object_id_kind,
            }
        )
    if len({row["scene"] for row in result}) != len(result):
        raise PreparationContractError("DL3DV plan repeats a calibration scene")
    return result


def _all_split_archives(plan: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    splits = {split: _split_archives(plan, split) for split in SPLITS}
    scenes = [record["scene"] for records in splits.values() for record in records]
    if len(scenes) != len(set(scenes)):
        raise PreparationContractError("DL3DV train and holdout selections overlap")
    return splits


def _validate_preparation(
    raw_root: Path, *, plan_path: Path, preparation_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    plan = _load_object(plan_path, "DL3DV download plan")
    if plan.get("kind") != "dl3dv_calibration_download_plan":
        raise PreparationContractError("DL3DV download plan kind is invalid")
    preparation = _load_object(preparation_path, "DL3DV archive preparation record")
    if preparation.get("kind") != "dl3dv_calibration_archive_preparation":
        raise PreparationContractError("DL3DV archive preparation record kind is invalid")
    if preparation.get("status") != "PREPARED":
        raise PreparationContractError("DL3DV calibration archives have not been prepared")
    if preparation.get("plan_sha256") != sha256_file(plan_path):
        raise PreparationContractError("DL3DV archive preparation does not bind this plan")
    if preparation.get("source") != plan.get("source"):
        raise PreparationContractError("DL3DV archive preparation source does not match plan")

    splits = _all_split_archives(plan)
    selected = [record for split in SPLITS for record in splits[split]]
    downloaded = preparation.get("downloaded_archives")
    if not isinstance(downloaded, list):
        raise PreparationContractError("DL3DV archive preparation has no downloaded archives")
    by_scene = {row.get("scene"): row for row in downloaded if isinstance(row, Mapping)}
    if len(by_scene) != len(downloaded):
        raise PreparationContractError("DL3DV archive preparation repeats an archive")
    if set(by_scene) != {record["scene"] for record in selected}:
        raise PreparationContractError(
            "DL3DV archive preparation does not contain exactly the frozen train and holdout archives"
        )
    for record in selected:
        downloaded_record = by_scene.get(record["scene"])
        if (
            not isinstance(downloaded_record, Mapping)
            or downloaded_record.get("path") != record["path"]
            or downloaded_record.get("size") != record["size"]
            or downloaded_record.get("source_object_id") != record["oid"]
            or downloaded_record.get("source_object_id_kind") != record["object_id_kind"]
            or not isinstance(downloaded_record.get("sha256"), str)
            or len(downloaded_record["sha256"]) != 64
        ):
            raise PreparationContractError(
                f"DL3DV archive preparation has no verified selected archive: {record['scene']}"
            )
    # Reject a scene-set mismatch before inspecting the prepared-tree manifest.
    # This keeps the failure at the calibration boundary even when an extra
    # scene makes the low-level manifest stale.
    _scene_dirs(raw_root, selected, require_exact=True)
    try:
        tree = verify_tree_manifest(raw_root, raw_root / ".scarf-manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PreparationContractError(
            "DL3DV prepared tree manifest is invalid"
        ) from exc
    prepared = preparation.get("prepared")
    if (
        not isinstance(prepared, Mapping)
        or prepared.get("tree_sha256") != tree["tree_sha256"]
    ):
        raise PreparationContractError("DL3DV prepared tree hash does not match its record")
    prepared_splits = prepared.get("split_scene_sets") if isinstance(prepared, Mapping) else None
    if not isinstance(prepared_splits, Mapping) or set(prepared_splits) != set(SPLITS):
        raise PreparationContractError("DL3DV prepared tree has no split binding")
    for split, records in splits.items():
        record = prepared_splits[split]
        scenes = sorted(item["scene"] for item in records)
        if (
            not isinstance(record, Mapping)
            or record.get("scene_count") != len(scenes)
            or record.get("scene_set_sha256") != canonical_sha256(scenes)
        ):
            raise PreparationContractError(
                f"DL3DV prepared tree split binding is invalid: {split}"
            )
    return plan, preparation, tree, splits


def _scene_dirs(
    raw_root: Path,
    selected: list[dict[str, Any]],
    *,
    require_exact: bool,
) -> dict[str, Path]:
    expected = {record["scene"] for record in selected}
    discovered = {
        path.parent.name: path
        for path in raw_root.glob("*/nerfstudio")
        if path.is_dir()
    }
    if not expected <= set(discovered) or (require_exact and set(discovered) != expected):
        missing = sorted(expected - set(discovered))
        extra = sorted(set(discovered) - expected)
        raise PreparationContractError(
            f"prepared DL3DV scene set does not match the fixed plan: "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )
    return discovered


def _native_image_paths(nerfstudio: Path, timestamps: list[int]) -> dict[int, Path]:
    """Select exactly one 270x480 tree that covers each camera frame."""
    choices: list[dict[int, Path]] = []
    for directory in sorted(nerfstudio.glob("images_*")):
        if not directory.is_dir():
            continue
        paths = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        by_timestamp = {_frame_id(path): path for path in paths}
        if len(by_timestamp) != len(paths) or any(item not in by_timestamp for item in timestamps):
            continue
        with Image.open(by_timestamp[timestamps[0]]) as image:
            shape = (image.height, image.width)
        if shape == NATIVE_SHAPE:
            choices.append(by_timestamp)
    if len(choices) != 1:
        raise PreparationContractError(
            f"{nerfstudio.parent.name} needs exactly one 270x480 image source"
        )
    return choices[0]


def _examples_and_index(raw_root: Path, selected: list[dict[str, Any]]) -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, list[int]]]
]:
    scene_dirs = _scene_dirs(raw_root, selected, require_exact=False)
    examples: dict[str, dict[str, Any]] = {}
    index: dict[str, dict[str, list[int]]] = {}
    for record in selected:
        scene = record["scene"]
        metadata, timestamps = _load_metadata(scene_dirs[scene] / "transforms.json")
        selection = deterministic_views(scene, len(timestamps))
        sources = _native_image_paths(scene_dirs[scene], timestamps)
        images: list[Any] = [None] * len(timestamps)
        for view in selection["context"]:
            images[view] = sources[timestamps[view]]
        examples[scene] = {
            "key": scene,
            "cameras": metadata["cameras"],
            # Targets intentionally remain None. The sidecar builder indexes
            # only contexts, and its output format has no target-image field.
            "images": images,
        }
        index[scene] = selection
    return examples, index


def _representation_examples(
    examples: Mapping[str, Mapping[str, Any]],
    index: Mapping[str, Mapping[str, list[int]]],
    representation: str,
) -> dict[str, dict[str, Any]]:
    if representation not in REPRESENTATIONS:
        raise PreparationContractError(f"unknown DL3DV representation: {representation}")
    target_shape = NATIVE_SHAPE if representation == NATIVE else RE10K_SHAPE
    allow_resize = representation == RE10K
    converted: dict[str, dict[str, Any]] = {}
    for scene, example in examples.items():
        copied = {"key": scene, "cameras": example["cameras"], "images": list(example["images"])}
        for view in index[scene]["context"]:
            source = copied["images"][view]
            if not isinstance(source, Path):
                raise PreparationContractError(f"DL3DV context image is missing: {scene}")
            copied["images"][view] = _encode_image(
                source,
                target_shape,
                allow_resize,
                (NATIVE_SHAPE,),
            )
        converted[scene] = copied
    return converted


def prepare_inputs(
    raw_root: Path,
    *,
    plan_path: Path,
    preparation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize native and Re10K target-free sidecars from 24 archives."""
    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"DL3DV calibration output directory is not empty: {output_dir}")
    plan, preparation, tree, splits = _validate_preparation(
        raw_root, plan_path=plan_path, preparation_path=preparation_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "dataset_tree_sha256": tree["tree_sha256"],
        "dataset_manifest_sha256": tree["manifest_sha256"],
        "prepared_source_sha256": sha256_file(preparation_path),
    }
    split_records: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        examples, index = _examples_and_index(raw_root, splits[split])
        representations: dict[str, dict[str, Any]] = {}
        for representation in REPRESENTATIONS:
            index_path = output_dir / f"{split}-{representation}.json"
            index_path.write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            index_sha256 = sha256_file(index_path)
            _, summary = canonicalize_index(index_path, index_sha256)
            sidecar = materialize_target_free_inputs(
                output_dir / "inputs" / split / representation,
                dataset="dl3dv",
                examples=_representation_examples(examples, index, representation),
                index=index,
                source=source,
                selection_sha256=summary["sample_selection_sha256"],
            )
            sidecar["calibration_input_root"] = f"inputs/{split}/{representation}"
            representations[representation] = {
                **sidecar,
                "index_file": index_path.name,
                "index_sha256": index_sha256,
                "selection_sha256": summary["sample_selection_sha256"],
                "sample_count": len(index),
                "target_shape": list(
                    NATIVE_SHAPE if representation == NATIVE else RE10K_SHAPE
                ),
            }
        split_records[split] = {
            "scene_count": len(index),
            "scene_set_sha256": canonical_sha256(sorted(index)),
            "representations": representations,
        }
    manifest = {
        "schema_version": "1.0",
        "kind": "dl3dv_calibration_protocol",
        "status": "PASS",
        "evaluation_disjoint": True,
        "source": plan["source"],
        "download_plan_sha256": sha256_file(plan_path),
        "archive_preparation_sha256": sha256_file(preparation_path),
        "calibration_scene_count": len(splits[TRAIN_SPLIT]),
        "holdout_scene_count": len(splits[HOLDOUT_SPLIT]),
        "datasets": {"dl3dv": {"splits": split_records}},
    }
    manifest["calibration_manifest_sha256"] = canonical_sha256(manifest)
    destination = output_dir / "manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preparation-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare_inputs(
            args.raw_root,
            plan_path=args.plan,
            preparation_path=args.preparation_record,
            output_dir=args.output_dir,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
