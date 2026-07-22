#!/usr/bin/env python3
"""Plan a minimal, evaluation-disjoint DL3DV calibration download."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SOURCE_REPOSITORY = "DL3DV/DL3DV-ALL-480P"
SOURCE_TERMS_URL = f"https://huggingface.co/datasets/{SOURCE_REPOSITORY}"
CALIBRATION_TRAIN_COUNT = 24
CALIBRATION_HOLDOUT_COUNT = 8
SELECTION_DOMAIN = "SCARF-AE-dl3dv-calibration-v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scene_from_archive(record: Mapping[str, Any]) -> dict[str, Any]:
    path = record.get("path")
    pure = PurePosixPath(path) if isinstance(path, str) else None
    if (
        pure is None
        or pure.is_absolute()
        or ".." in pure.parts
        or len(pure.parts) != 2
        or pure.suffix != ".zip"
        or len(pure.stem) != 64
        or any(character not in "0123456789abcdef" for character in pure.stem)
    ):
        raise ValueError("DL3DV calibration source has an unsafe scene archive path")
    size = record.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("DL3DV calibration source archive has no positive size")
    oid = record.get("oid")
    object_id_kind = record.get("object_id_kind")
    allowed_lengths = {
        "lfs_sha256": 64,
        "git_blob": 40,
    }
    if (
        not isinstance(oid, str)
        or len(oid) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in oid)
    ):
        raise ValueError("DL3DV calibration source archive has no stable object id")
    if object_id_kind is None:
        # Backward-compatible fixtures and old saved trees with a 64-character
        # object id identify LFS bytes; a 40-character id is a Git blob id.
        object_id_kind = "lfs_sha256" if len(oid) == 64 else "git_blob"
    if object_id_kind not in allowed_lengths or len(oid) != allowed_lengths[object_id_kind]:
        raise ValueError("DL3DV calibration source archive has invalid object-id metadata")
    return {
        "scene": pure.stem,
        "path": pure.as_posix(),
        "size": size,
        # Gated API responses may expose either the LFS byte SHA256 or a Git
        # blob object id. Both bind the source tree; only the former can also
        # verify downloaded bytes before their local SHA256 is recorded.
        "oid": oid,
        "object_id_kind": object_id_kind,
    }


def _rank(scene: str, purpose: str) -> tuple[str, str]:
    material = f"{SELECTION_DOMAIN}\0{purpose}\0{scene}".encode("utf-8")
    return hashlib.sha256(material).hexdigest(), scene


def _select(
    candidates: Sequence[dict[str, Any]], *, purpose: str, count: int
) -> list[dict[str, Any]]:
    if count > len(candidates):
        raise ValueError(f"DL3DV has fewer than {count} eligible {purpose} scenes")
    by_scene = {item["scene"]: item for item in candidates}
    selected_scenes = sorted(by_scene, key=lambda scene: _rank(scene, purpose))[:count]
    return [by_scene[scene] for scene in selected_scenes]


def build_plan(
    tree: Sequence[Mapping[str, Any]],
    *,
    evaluation_scenes: Sequence[str],
    evaluation_index_sha256: str,
    revision: str,
) -> dict[str, Any]:
    """Select a 24+8 official archive set without reading images or metrics."""
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("DL3DV calibration source revision must be a 40-character commit")
    if (
        not isinstance(evaluation_index_sha256, str)
        or len(evaluation_index_sha256) != 64
        or any(character not in "0123456789abcdef" for character in evaluation_index_sha256)
    ):
        raise ValueError("DL3DV evaluation index SHA256 is invalid")
    evaluation = set(evaluation_scenes)
    if len(evaluation) != len(evaluation_scenes) or not evaluation:
        raise ValueError("DL3DV evaluation scenes must be unique and non-empty")
    if any(not isinstance(scene, str) or not scene for scene in evaluation):
        raise ValueError("DL3DV evaluation scene is invalid")

    archives = [_scene_from_archive(record) for record in tree]
    by_scene = {record["scene"]: record for record in archives}
    if len(by_scene) != len(archives):
        raise ValueError("DL3DV calibration source contains duplicate scene archives")
    eligible = [record for record in archives if record["scene"] not in evaluation]
    train = _select(eligible, purpose="train", count=CALIBRATION_TRAIN_COUNT)
    train_scenes = {record["scene"] for record in train}
    holdout = _select(
        [record for record in eligible if record["scene"] not in train_scenes],
        purpose="holdout",
        count=CALIBRATION_HOLDOUT_COUNT,
    )
    selected = [*train, *holdout]
    selected_scenes = {record["scene"] for record in selected}
    if selected_scenes & evaluation or len(selected_scenes) != len(selected):
        raise ValueError("DL3DV calibration selection is not scene-disjoint")
    canonical_tree = sorted(archives, key=lambda record: record["scene"])
    return {
        "schema_version": "1.0",
        "kind": "dl3dv_calibration_download_plan",
        "status": "PLANNED_AWAITING_UPSTREAM_ACCESS",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "terms_url": SOURCE_TERMS_URL,
            "revision": revision,
            "archive_tree_sha256": canonical_sha256(canonical_tree),
            "archive_scene_count": len(canonical_tree),
        },
        "evaluation": {
            "scene_count": len(evaluation),
            "index_sha256": evaluation_index_sha256,
        },
        "selection": {
            "domain": SELECTION_DOMAIN,
            "calibration_train": train,
            "calibration_holdout": holdout,
            "calibration_train_count": len(train),
            "calibration_holdout_count": len(holdout),
            "scene_disjoint": True,
            "selected_archives_sha256": canonical_sha256(selected),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--evaluation-index", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        tree = json.loads(args.tree.read_text(encoding="utf-8"))
        evaluation_index = json.loads(args.evaluation_index.read_text(encoding="utf-8"))
        if not isinstance(tree, list) or not isinstance(evaluation_index, dict):
            raise ValueError("DL3DV source tree or evaluation index is invalid")
        plan = build_plan(
            tree,
            evaluation_scenes=list(evaluation_index),
            evaluation_index_sha256=hashlib.sha256(
                args.evaluation_index.read_bytes()
            ).hexdigest(),
            revision=args.revision,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
