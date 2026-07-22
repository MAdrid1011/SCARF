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

from scripts.calibration_contract import PARAMETER_GRID, canonical_parameters  # noqa: E402
from scripts.calibration_sweep import validate_split_candidate_records  # noqa: E402
from scripts.mechanism_config import FIXED, sha256_file as mechanism_sha256_file  # noqa: E402
from scripts.saes_execution_identity import build_saes_execution_identity  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config(candidate_records: Path) -> dict:
    candidate_records = candidate_records.resolve()
    source = json.loads(candidate_records.read_text(encoding="utf-8"))
    evidence = validate_split_candidate_records(source, candidate_records.parent)
    selected = evidence["selected_train_candidate"]
    holdout = evidence["validated_holdout_candidate"]
    manifest_sha256 = source.get("calibration_manifest_sha256")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError("calibration manifest SHA256 is missing")
    selected_parameters = canonical_parameters(selected["parameters"])
    if canonical_parameters(holdout["parameters"]) != selected_parameters:
        raise ValueError("holdout candidate does not match the selected train tuple")
    train = evidence["train"]
    holdout_split = evidence["holdout"]
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
        "fixed": dict(FIXED),
        "saes_execution_identity": build_saes_execution_identity(),
        "search_space": {key: list(values) for key, values in PARAMETER_GRID.items()},
        "selected": selected_parameters,
        "calibration": {
            "manifest_sha256": manifest_sha256,
            "candidate_records_sha256": sha256_file(candidate_records),
            "evaluation_disjoint": True,
            "protocol": "dl3dv_train_holdout_v1",
            "train_holdout_scene_disjoint": True,
            "train": {
                "selection_sha256": train["selection_sha256"],
                "scene_set_sha256": train["scene_set_sha256"],
                "pair_bindings_sha256": train["pair_bindings_sha256"],
                "trace_set_sha256": train["trace_set_sha256"],
                "candidate_set_sha256": train["candidate_set_sha256"],
                "selected_candidate_sha256": selected["candidate_sha256"],
            },
            "holdout": {
                "selection_sha256": holdout_split["selection_sha256"],
                "scene_set_sha256": holdout_split["scene_set_sha256"],
                "pair_bindings_sha256": holdout_split["pair_bindings_sha256"],
                "trace_set_sha256": holdout_split["trace_set_sha256"],
                "candidate_set_sha256": holdout_split["candidate_set_sha256"],
                "validated_parameters_sha256": holdout_split[
                    "validated_parameters_sha256"
                ],
                "validated_candidate_sha256": holdout["candidate_sha256"],
            },
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
