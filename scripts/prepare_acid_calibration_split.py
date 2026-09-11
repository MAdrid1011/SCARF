#!/usr/bin/env python3
"""Materialize a deterministic target-free ACID calibration split.

The source tree contains real ACID training chunks.  This helper creates the
small sidecar tree consumed by ``--calibration-trace`` while retaining target
camera geometry and omitting every target RGB image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_manifest import build as build_dataset_manifest
from scripts.calibration_inputs import build_target_free_record
from scripts.calibration_contract import canonical_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(source_root: Path, output_root: Path, scenes: list[str]) -> Path:
    import torch

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source = _load(source_root / ".scarf-calibration-input.json")
    source_index = _load(source_root / "test" / "index.json")
    if not isinstance(source_index, dict):
        raise ValueError("source calibration index is invalid")
    if len(scenes) != len(set(scenes)) or any(scene not in source_index for scene in scenes):
        raise ValueError("requested split scenes are not present in the source sidecar")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"calibration split already exists: {output_root}")
    test_root = output_root / "test"
    test_root.mkdir(parents=True)
    output_index: dict[str, str] = {}
    evaluation_index: dict[str, dict[str, list[int]]] = {}
    rows = []
    for ordinal, scene in enumerate(scenes):
        record_path = source_root / "test" / source_index[scene]
        chunk = torch.load(record_path, map_location="cpu", weights_only=False)
        if not isinstance(chunk, list) or len(chunk) != 1 or not isinstance(chunk[0], dict):
            raise ValueError(f"invalid source sidecar record: {scene}")
        record = chunk[0]
        output_name = f"{ordinal:06d}.torch"
        shutil.copy2(record_path, test_root / output_name)
        output_index[scene] = output_name
        evaluation_index[scene] = {
            "context": list(record["context_indices"]),
            "target": list(record["target_indices"]),
        }
        rows.append({
            "scene": scene,
            "context_indices": list(record["context_indices"]),
            "target_indices": list(record["target_indices"]),
        })
    index_path = test_root / "index.json"
    index_path.write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "evaluation-index.json").write_text(
        json.dumps(evaluation_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    opened = [
        {
            "path": "test/index.json",
            "role": "target_free_sidecar_index",
            "sha256": _sha256(index_path),
        },
        *[
            {
                "path": f"test/{output_index[scene]}",
                "role": "target_free_sidecar_record",
                "sha256": _sha256(test_root / output_index[scene]),
            }
            for scene in scenes
        ],
    ]
    from scripts.compile_protocol import canonicalize_index
    evaluation_digest = _sha256(output_root / "evaluation-index.json")
    _, evaluation_summary = canonicalize_index(
        output_root / "evaluation-index.json", evaluation_digest
    )
    provenance = {
        "schema_version": "1.0",
        "kind": source["kind"],
        "dataset": source["dataset"],
        "target_rgb_included": False,
        "selection_sha256": evaluation_summary["sample_selection_sha256"],
        "selected_scene_count": len(scenes),
        "opened_file_manifest": opened,
        "opened_file_manifest_sha256": canonical_sha256(opened),
        "source_dataset_tree_sha256": source.get("source_dataset_tree_sha256"),
        "source_dataset_manifest_sha256": source.get("source_dataset_manifest_sha256"),
        "source_prepared_provenance_sha256": source.get("source_prepared_provenance_sha256"),
    }
    provenance_path = output_root / ".scarf-calibration-input.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = build_dataset_manifest(
        output_root,
        "acid",
        "target-free calibration sidecar",
        f"selection:{provenance['selection_sha256']}",
    )
    (output_root / ".scarf-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    args = parser.parse_args()
    try:
        print(materialize(args.source_root, args.output_root, args.scenes))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
