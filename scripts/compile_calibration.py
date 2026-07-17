#!/usr/bin/env python3
"""Compile evaluation-disjoint SCARF calibration indices from training splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibration_contract import (
    assert_evaluation_disjoint,
    canonical_sha256,
    hash_ranked_scene_names,
)
from scripts.calibration_inputs import materialize_target_free_inputs
from scripts.compile_protocol import canonicalize_index, sha256_file


DATASETS = ("re10k", "acid")
SELECTION_COUNT = 32
DEFAULT_CALIBRATION_ROOT = ROOT / "downloads" / "calibration" / "prepared"


def deterministic_views(scene: str, view_count: int) -> dict[str, list[int]]:
    """Select fixed cameras without reading images, predictions, or metrics."""
    if not isinstance(scene, str) or not scene:
        raise ValueError("calibration scene must be non-empty")
    if isinstance(view_count, bool) or not isinstance(view_count, int) or view_count < 5:
        raise ValueError(f"calibration scene {scene} has fewer than five views")
    context = [0, view_count - 1]
    available = [index for index in range(1, view_count - 1)]
    anchors = (0.25, 0.50, 0.75)
    target = []
    for fraction in anchors:
        preferred = round((view_count - 1) * fraction)
        selected = min(
            (index for index in available if index not in target),
            key=lambda index: (abs(index - preferred), index),
        )
        target.append(selected)
    return {"context": context, "target": target}


def compile_dataset_selection(
    *,
    dataset: str,
    scene_view_counts: Mapping[str, int],
    evaluation_rows: list[dict[str, Any]],
    count: int = SELECTION_COUNT,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported calibration dataset: {dataset}")
    evaluation_scenes = {row["scene"] for row in evaluation_rows}
    overlap = sorted(set(scene_view_counts) & evaluation_scenes)
    if overlap:
        raise ValueError(f"calibration/evaluation overlap: {overlap[0]}")
    selected_scenes = hash_ranked_scene_names(
        list(scene_view_counts), dataset=dataset, count=count
    )
    index = {
        scene: deterministic_views(scene, int(scene_view_counts[scene]))
        for scene in selected_scenes
    }
    rows = [
        {
            "scene": scene,
            "context_indices": entry["context"],
            "target_indices": entry["target"],
        }
        for scene, entry in index.items()
    ]
    assert_evaluation_disjoint(rows, evaluation_rows)
    return {
        "dataset": dataset,
        "sample_count": len(index),
        "index": index,
        "selection_sha256": canonical_sha256(rows),
    }


def _safe_chunk(split_root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("training index contains a non-string chunk path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe training chunk path: {value}")
    path = (split_root / Path(*pure.parts)).resolve()
    if split_root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f"training chunk is missing: {value}")
    return path


def _load_index(path: Path, dataset: str, label: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"{dataset} {label} is missing: {path}")
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or len(source) < SELECTION_COUNT:
        raise ValueError(f"{dataset} {label} has fewer than {SELECTION_COUNT} scenes")
    if any(
        not isinstance(scene, str)
        or not scene
        or not isinstance(chunk, str)
        or not chunk
        for scene, chunk in source.items()
    ):
        raise ValueError(f"{dataset} {label} contains an invalid scene or chunk path")
    return source


def _verify_prepared_source(
    dataset_root: Path,
    *,
    dataset: str,
    selected: list[str],
    selected_index: Mapping[str, str],
    full_index_sha256: str,
) -> str:
    """Bind a prepared subset to the exact full official training index."""
    source_path = dataset_root / ".scarf-calibration-source.json"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"{dataset} prepared calibration provenance is missing: {source_path}"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise ValueError(f"{dataset} calibration provenance is invalid")
    if source.get("dataset") != dataset:
        raise ValueError(f"{dataset} calibration provenance names a different dataset")
    if source.get("full_train_index_sha256") != full_index_sha256:
        raise ValueError(f"{dataset} calibration full-index hash does not match provenance")
    if source.get("selected_index_sha256") != canonical_sha256(dict(selected_index)):
        raise ValueError(f"{dataset} calibration selected-index hash does not match provenance")
    if source.get("selected_scenes") != selected:
        raise ValueError(f"{dataset} calibration selected scenes do not match provenance")
    if source.get("evaluation_disjoint") is not True:
        raise ValueError(f"{dataset} calibration provenance is not evaluation-disjoint")
    return sha256_file(source_path)


def _selected_examples(
    dataset_root: Path, dataset: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_root = dataset_root.resolve() / "train"
    index_path = split_root / "index.json"
    source = _load_index(index_path, dataset, "selected training index")
    full_index_path = split_root / "full-index.json"
    full_index = (
        _load_index(full_index_path, dataset, "full training index")
        if full_index_path.is_file()
        else source
    )
    selected = hash_ranked_scene_names(
        list(full_index), dataset=dataset, count=SELECTION_COUNT
    )
    prepared_source_sha256 = None
    if full_index_path.is_file():
        if set(source) != set(selected):
            raise ValueError(
                f"{dataset} selected training index is not the fixed 32-scene subset"
            )
        mismatch = next(
            (scene for scene in selected if source[scene] != full_index[scene]), None
        )
        if mismatch is not None:
            raise ValueError(
                f"{dataset} selected chunk does not match full training index: {mismatch}"
            )
        prepared_source_sha256 = _verify_prepared_source(
            dataset_root,
            dataset=dataset,
            selected=selected,
            selected_index=source,
            full_index_sha256=sha256_file(full_index_path),
        )

    import torch

    by_chunk: dict[Path, set[str]] = {}
    for scene in selected:
        by_chunk.setdefault(_safe_chunk(split_root, source[scene]), set()).add(scene)
    examples: dict[str, Any] = {}
    for path, required in sorted(by_chunk.items(), key=lambda item: str(item[0])):
        chunk = torch.load(path, map_location="cpu")
        if not isinstance(chunk, list):
            raise ValueError(f"training chunk is not a list: {path}")
        for example in chunk:
            if not isinstance(example, dict) or example.get("key") not in required:
                continue
            images = example.get("images")
            if images is None or not hasattr(images, "__len__"):
                raise ValueError(f"training scene has no image sequence: {example.get('key')}")
            examples[example["key"]] = example
    missing = sorted(set(selected) - set(examples))
    if missing:
        raise ValueError(f"selected training scene is absent from its chunk: {missing[0]}")

    manifest_path = dataset_root / ".scarf-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return examples, {
        "train_index_sha256": (
            sha256_file(full_index_path)
            if full_index_path.is_file()
            else sha256_file(index_path)
        ),
        "selected_train_index_sha256": sha256_file(index_path),
        "prepared_source_sha256": prepared_source_sha256,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_tree_sha256": manifest.get("tree_sha256"),
        "source": manifest.get("source"),
        "revision": manifest.get("revision"),
    }


def _materialize_subset(
    prepared_root: Path,
    dataset: str,
    examples: Mapping[str, Any],
    index: Mapping[str, Any],
    source: Mapping[str, Any],
    selection_sha256: str,
) -> dict[str, Any]:
    """Write a target-free execution sidecar for calibration replay."""
    return materialize_target_free_inputs(
        prepared_root,
        dataset=dataset,
        examples=examples,
        index=index,
        source=source,
        selection_sha256=selection_sha256,
    )


def _evaluation_rows(dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol_path = ROOT / "artifact/evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    records = [
        value
        for key, value in protocol.get("pairs", {}).items()
        if key.endswith(f"/{dataset}")
    ]
    if not records:
        raise ValueError(f"evaluation protocol has no {dataset} entry")
    identity = {
        (record.get("index_path"), record.get("source_index_sha256"))
        for record in records
    }
    if len(identity) != 1:
        raise ValueError(f"evaluation protocol has inconsistent {dataset} indices")
    relative, digest = identity.pop()
    path = (ROOT / relative).resolve()
    rows, summary = canonicalize_index(path, digest)
    return rows, {
        "index_path": relative,
        "source_index_sha256": digest,
        "sample_selection_sha256": summary["sample_selection_sha256"],
    }


def compile_manifest(
    output_dir: Path, calibration_root: Path = DEFAULT_CALIBRATION_ROOT
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    calibration_root = calibration_root.resolve()
    datasets: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        evaluation, evaluation_identity = _evaluation_rows(dataset)
        examples, source = _selected_examples(calibration_root / dataset, dataset)
        counts = {
            scene: len(example["images"])
            for scene, example in examples.items()
        }
        selection = compile_dataset_selection(
            dataset=dataset,
            scene_view_counts=counts,
            evaluation_rows=evaluation,
        )
        index_path = output_dir / f"{dataset}.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        selection_index = selection.pop("index")
        index_path.write_text(
            json.dumps(selection_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index_sha256 = sha256_file(index_path)
        _, index_summary = canonicalize_index(index_path, index_sha256)
        selection["selection_sha256"] = index_summary[
            "sample_selection_sha256"
        ]
        prepared = _materialize_subset(
            output_dir / "inputs" / dataset,
            dataset,
            examples,
            selection_index,
            source,
            selection["selection_sha256"],
        )
        datasets[dataset] = {
            **selection,
            **source,
            **prepared,
            "index_file": index_path.name,
            "index_sha256": index_sha256,
            "evaluation": evaluation_identity,
            "evaluation_disjoint": True,
        }
    manifest = {
        "schema_version": "1.0",
        "kind": "calibration_protocol",
        "status": "PASS",
        "selection_rule": "SHA256-ranked official training scenes",
        "selection_count_per_dataset": SELECTION_COUNT,
        "evaluation_disjoint": True,
        "input_layout": "prepared official training subset with full-index provenance",
        "datasets": datasets,
    }
    manifest["calibration_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    args = parser.parse_args()
    try:
        record = compile_manifest(args.output_dir, args.calibration_root)
        path = args.output_dir.resolve() / "manifest.json"
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
