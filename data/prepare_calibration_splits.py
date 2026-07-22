#!/usr/bin/env python3
"""Prepare only the deterministic training chunks needed for AE calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_manifest import build as build_dataset_manifest
from scripts.calibration_contract import (
    CALIBRATION_DOMAIN,
    assert_evaluation_disjoint,
    canonical_sha256,
    hash_ranked_scene_names,
)
from scripts.compile_protocol import canonicalize_index


DATASETS = ("re10k", "acid")
SELECTION_COUNT = 32
DEFAULT_CACHE = ROOT / "downloads" / "calibration"
DEFAULT_OUTPUT_ROOT = DEFAULT_CACHE / "prepared"
DEFAULT_MIRROR = "http://schadenfreude.csail.mit.edu:8000"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_chunk_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("training index chunk path must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
        raise ValueError(f"unsafe training index chunk path: {value}")
    return pure.name


def _load_evaluation_rows(dataset: str) -> list[dict[str, Any]]:
    protocol = json.loads(
        (ROOT / "artifact" / "evaluation_protocol.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    identities = {
        (record.get("index_path"), record.get("source_index_sha256"))
        for pair, record in protocol.get("pairs", {}).items()
        if pair.endswith(f"/{dataset}")
    }
    for relative, digest in identities:
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"evaluation protocol has no valid {dataset} index")
        index_rows, _ = canonicalize_index(ROOT / relative, digest)
        rows.extend(index_rows)
    return rows


def _extract_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    info = archive.getinfo(member)
    if info.is_dir():
        raise ValueError(f"archive member is a directory: {member}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with archive.open(info, "r") as source, temporary.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    temporary.replace(destination)


def _load_train_index(archive: zipfile.ZipFile, dataset: str) -> tuple[str, dict[str, str]]:
    member = f"{dataset}/train/index.json"
    try:
        with archive.open(member, "r") as stream:
            source = json.load(stream)
    except KeyError as exc:
        raise FileNotFoundError(f"archive has no training index: {member}") from exc
    if not isinstance(source, dict) or len(source) < SELECTION_COUNT:
        raise ValueError(f"{dataset} training index has fewer than {SELECTION_COUNT} scenes")
    normalized = {}
    for scene, chunk in source.items():
        if not isinstance(scene, str) or not scene:
            raise ValueError("training index has an invalid scene name")
        normalized[scene] = _safe_chunk_name(chunk)
    return member, normalized


def prepare_dataset(
    archive_path: Path,
    *,
    dataset: str,
    output_root: Path,
    source_url: str,
    evaluation_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract the immutable 32-scene calibration subset from one full archive."""
    if dataset not in DATASETS:
        raise ValueError(f"unsupported calibration dataset: {dataset}")
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"calibration archive is missing: {archive_path}")
    destination = (output_root / dataset).resolve()
    if destination.exists():
        raise FileExistsError(f"calibration destination already exists: {destination}")
    archive_sha256 = sha256_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        index_member, full_index = _load_train_index(archive, dataset)
        selected = hash_ranked_scene_names(
            list(full_index), dataset=dataset, count=SELECTION_COUNT
        )
        rows = [
            {"scene": scene, "context_indices": [0], "target_indices": [1]}
            for scene in selected
        ]
        # Scene identity is the disjointness boundary; temporary view values are
        # only used to reuse the shared scene-overlap validator.
        assert_evaluation_disjoint(rows, evaluation_rows or _load_evaluation_rows(dataset))
        selected_index = {scene: full_index[scene] for scene in selected}
        chunk_names = sorted(set(selected_index.values()))
        member_names = {info.filename for info in archive.infolist()}
        required = [index_member, *(f"{dataset}/train/{name}" for name in chunk_names)]
        missing = [member for member in required if member not in member_names]
        if missing:
            raise FileNotFoundError(f"archive is missing training chunk: {missing[0]}")

        train_root = destination / "train"
        _extract_member(archive, index_member, train_root / "full-index.json")
        for name in chunk_names:
            _extract_member(archive, f"{dataset}/train/{name}", train_root / name)

    (train_root / "index.json").write_text(
        json.dumps(selected_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    full_index_sha256 = sha256_file(train_root / "full-index.json")
    selection = {
        "schema_version": "1.0",
        "dataset": dataset,
        "archive": archive_path.name,
        "archive_sha256": archive_sha256,
        "source_url": source_url,
        "full_train_index_sha256": full_index_sha256,
        "selection_domain": CALIBRATION_DOMAIN,
        "selection_count": SELECTION_COUNT,
        "selected_scenes": selected,
        "selected_index_sha256": canonical_sha256(selected_index),
        "evaluation_disjoint": True,
    }
    (destination / ".scarf-calibration-source.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    revision = f"archive-sha256:{archive_sha256};full-index:{full_index_sha256}"
    manifest = build_dataset_manifest(destination, dataset, source_url, revision)
    (destination / ".scarf-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "dataset": dataset,
        "destination": str(destination),
        "archive_sha256": archive_sha256,
        "full_train_index_sha256": full_index_sha256,
        "selected_scene_count": len(selected),
        "selected_chunk_count": len(chunk_names),
        "prepared_tree_sha256": manifest["tree_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-url", default=DEFAULT_MIRROR)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    args = parser.parse_args()
    datasets = args.dataset or list(DATASETS)
    try:
        records = [
            prepare_dataset(
                args.cache / f"{dataset}.zip",
                dataset=dataset,
                output_root=args.output_root,
                source_url=f"{args.source_url.rstrip('/')}/{dataset}.zip",
            )
            for dataset in datasets
        ]
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
