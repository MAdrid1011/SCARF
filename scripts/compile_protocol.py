#!/usr/bin/env python3
"""Validate and compile the recovered SCARF evaluation-index contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "artifact/evaluation_protocol.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _view_indices(scene: str, label: str, value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{scene} has no {label} views")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in value):
        raise ValueError(f"{scene} has invalid {label} view indices")
    if len(set(value)) != len(value):
        raise ValueError(f"{scene} has duplicate {label} view indices")
    return list(value)


def canonicalize_index(
    path: Path, expected_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"source index SHA256 mismatch for {path}: {actual_sha256} != {expected_sha256}"
        )
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not source:
        raise ValueError(f"evaluation index must be a non-empty object: {path}")
    rows = []
    null_count = 0
    for source_index, (scene, entry) in enumerate(source.items()):
        if not isinstance(scene, str) or not scene:
            raise ValueError(f"evaluation index has an invalid scene key: {scene!r}")
        if entry is None:
            null_count += 1
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{scene} has an invalid index entry")
        context = _view_indices(scene, "context", entry.get("context"))
        target = _view_indices(scene, "target", entry.get("target"))
        rows.append(
            {
                # Preserve the upstream ordinal so null source entries do not
                # silently renumber the protocol's executable samples.
                "sample_index": source_index,
                "execution_index": len(rows),
                "scene": scene,
                "context_indices": context,
                "target_indices": target,
            }
        )
    # execution_index is an implementation detail used to address the filtered
    # dataloader. It is excluded from the stable source-selection identity.
    canonical_rows = [
        {key: value for key, value in row.items() if key != "execution_index"}
        for row in rows
    ]
    canonical = json.dumps(
        canonical_rows, sort_keys=True, separators=(",", ":")
    ).encode()
    return rows, {
        "source_entry_count": len(source),
        "null_entry_count": null_count,
        "sample_count": len(rows),
        "source_index_sha256": actual_sha256,
        "sample_selection_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def validate_dataset_bounds(
    rows: list[dict[str, Any]], scene_view_counts: Mapping[str, int]
) -> None:
    for row in rows:
        scene = row["scene"]
        count = scene_view_counts.get(scene)
        if count is None:
            raise ValueError(f"scene {scene} is missing from prepared dataset")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"scene {scene} has an invalid prepared view count")
        for index in [*row["context_indices"], *row["target_indices"]]:
            if index >= count:
                raise ValueError(
                    f"scene {scene} view index {index} exceeds prepared view count {count}"
                )


def _safe_protocol_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("protocol pair has no index_path")
    root = root.resolve()
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"unsafe protocol index path: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"protocol index is missing: {value}")
    return path


def compile_protocol(protocol_path: Path, root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "finalized":
        raise ValueError("evaluation protocol status must be finalized")
    pairs = protocol.get("pairs")
    if not isinstance(pairs, dict) or len(pairs) != 9:
        raise ValueError("evaluation protocol must contain all nine pairs")

    compiled_pairs: dict[str, dict[str, Any]] = {}
    datasets: dict[str, dict[str, Any]] = {}
    cache: dict[tuple[Path, str], tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for pair, record in sorted(pairs.items()):
        if not isinstance(record, dict) or "/" not in pair:
            raise ValueError(f"invalid protocol pair: {pair}")
        _, dataset = pair.split("/", 1)
        index_path = _safe_protocol_path(root, record.get("index_path"))
        expected_sha256 = record.get("source_index_sha256")
        if not isinstance(expected_sha256, str):
            raise ValueError(f"{pair} has no source index SHA256")
        key = (index_path, expected_sha256)
        if key not in cache:
            cache[key] = canonicalize_index(index_path, expected_sha256)
        _, summary = cache[key]
        for field, actual in summary.items():
            if record.get(field) != actual:
                raise ValueError(
                    f"{pair} {field} mismatch: {record.get(field)!r} != {actual!r}"
                )
        dataset_tree_sha256 = record.get("dataset_tree_sha256")
        if dataset_tree_sha256 is not None and (
            not isinstance(dataset_tree_sha256, str)
            or SHA256.fullmatch(dataset_tree_sha256) is None
        ):
            raise ValueError(f"{pair} has an invalid dataset tree SHA256")
        compiled_pairs[pair] = {
            "index_path": str(index_path.relative_to(root)),
            "dataset_representation": record.get("dataset_representation"),
            "dataset_tree_sha256": dataset_tree_sha256,
            **summary,
        }
        existing = datasets.get(dataset)
        if existing is not None and existing != summary:
            raise ValueError(f"{dataset} pairs do not share one selection contract")
        datasets[dataset] = dict(summary)
    return {
        "schema_version": "1.0",
        "kind": "compiled_evaluation_protocol",
        "status": "PASS",
        "protocol_path": str(protocol_path.relative_to(root)),
        "pairs": compiled_pairs,
        "datasets": datasets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        record = compile_protocol(args.protocol.resolve(), args.root.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
