#!/usr/bin/env python3
"""Verify a prepared dataset tree and its canonical evaluation coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compile_protocol import canonicalize_index, validate_dataset_bounds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tree_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("prepared dataset manifest has no files")
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path
    }
    if set(actual_paths) != set(declared):
        missing = sorted(set(declared) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(declared))
        raise ValueError(
            f"prepared dataset file set mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    canonical = {}
    for relative, path in sorted(actual_paths.items()):
        record = declared.get(relative)
        if not isinstance(record, dict):
            raise ValueError(f"invalid prepared file record: {relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if record.get("size") != size or record.get("sha256") != digest:
            raise ValueError(f"prepared dataset hash mismatch: {relative}")
        canonical[relative] = {"size": size, "sha256": digest}
    tree_sha256 = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if tree_sha256 != manifest.get("tree_sha256"):
        raise ValueError("prepared dataset tree SHA256 mismatch")
    source = manifest.get("source")
    revision = manifest.get("revision")
    if not isinstance(source, str) or not source:
        raise ValueError("prepared dataset manifest has no source")
    if not isinstance(revision, str) or not revision:
        raise ValueError("prepared dataset manifest has no revision")
    return {
        "dataset": manifest.get("dataset"),
        "source": source,
        "revision": revision,
        "file_count": len(canonical),
        "tree_sha256": tree_sha256,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _safe_chunk_path(test_root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("dataset index has a non-string chunk path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe dataset chunk path: {value}")
    path = (test_root / Path(*pure.parts)).resolve()
    if test_root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"dataset chunk is missing: {value}")
    return path


def scene_view_counts(root: Path, scenes: set[str]) -> dict[str, int]:
    import torch

    test_root = Path(root).resolve() / "test"
    index = json.loads((test_root / "index.json").read_text(encoding="utf-8"))
    missing = sorted(scenes - set(index))
    if missing:
        raise ValueError(f"prepared dataset is missing protocol scene {missing[0]}")
    chunks: dict[Path, set[str]] = {}
    for scene in scenes:
        chunks.setdefault(_safe_chunk_path(test_root, index[scene]), set()).add(scene)
    counts = {}
    for path, required in sorted(chunks.items(), key=lambda item: str(item[0])):
        chunk = torch.load(path, map_location="cpu")
        if not isinstance(chunk, list):
            raise ValueError(f"dataset chunk is not a list: {path}")
        for example in chunk:
            if not isinstance(example, dict) or example.get("key") not in required:
                continue
            images = example.get("images")
            if images is None or not hasattr(images, "__len__"):
                raise ValueError(f"scene has no image sequence: {example.get('key')}")
            counts[example["key"]] = len(images)
    unresolved = sorted(scenes - set(counts))
    if unresolved:
        raise ValueError(f"protocol scene not found in its indexed chunk: {unresolved[0]}")
    return counts


def verify_prepared_dataset(
    root: Path,
    manifest_path: Path,
    index_path: Path,
    source_index_sha256: str,
    representation: str,
) -> dict[str, Any]:
    tree = verify_tree_manifest(root, manifest_path)
    rows, selection = canonicalize_index(index_path, source_index_sha256)
    counts = scene_view_counts(root, {row["scene"] for row in rows})
    validate_dataset_bounds(rows, counts)
    return {
        "schema_version": "1.0",
        "kind": "prepared_dataset_validation",
        "status": "PASS",
        "representation": representation,
        "tree": tree,
        "selection": selection,
        "minimum_view_count": min(counts.values()),
        "maximum_view_count": max(counts.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--evaluation-index", type=Path, required=True)
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest or args.root / ".scarf-manifest.json"
    try:
        record = verify_prepared_dataset(
            args.root,
            manifest,
            args.evaluation_index,
            args.source_index_sha256,
            args.representation,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
