#!/usr/bin/env python3
"""Select one global SCARF mechanism tuple from isolated candidate records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibration_contract import PARAMETER_GRID, select_global_candidate
from scripts.mechanism_config import sha256_file as mechanism_sha256_file


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config(candidate_records: Path) -> dict:
    source = json.loads(candidate_records.read_text(encoding="utf-8"))
    candidates = source.get("candidates")
    if source.get("evaluation_disjoint") is not True:
        raise ValueError("calibration manifest is not evaluation-disjoint")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("calibration candidate record is empty")
    selected = select_global_candidate(candidates)
    manifest_sha256 = source.get("calibration_manifest_sha256")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError("calibration manifest SHA256 is missing")
    return {
        "schema_version": "1.0",
        "status": "calibrated",
        "global_configuration": True,
        "projection": {
            "kind": "random_hyperplane_lsh",
            "seed": 42,
            "rom": "artifact/lsh_projection.json",
            "rom_sha256": mechanism_sha256_file(ROOT / "artifact/lsh_projection.json"),
        },
        "fixed": {
            "fsdr_cache_size": 32,
            "fsdr_hamming_threshold": 3,
            "fsdr_guidance_policy": "paper-hamming-local-validity",
            "fsdr_contraction_ratio": 4,
            "saes_feature_threshold": 0.2,
            "saes_depth_threshold": 0.1,
            "saes_moment_geometry": "c2w-probe-depth-ray-v1",
            "saes_tile_size": 4,
        },
        "search_space": {key: list(values) for key, values in PARAMETER_GRID.items()},
        "selected": selected["parameters"],
        "calibration": {
            "manifest_sha256": manifest_sha256,
            "candidate_records_sha256": sha256_file(candidate_records),
            "evaluation_disjoint": True,
            "selection_rule": "quality constraints, maximum event work reduction",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-records", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config-output",
        type=Path,
        default=ROOT / "artifact/mechanism_config.json",
    )
    args = parser.parse_args()
    records = args.candidate_records or args.output_dir / "candidates.json"
    try:
        config = build_config(records)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        evidence = args.output_dir / "calibration-result.json"
        evidence.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
