#!/usr/bin/env python3
"""Freeze a measured ACID train/holdout calibration bundle.

This is an explicit local fallback for environments without the gated DL3DV
calibration archive. It never fabricates candidate metrics: every input must
be a target-free calibration trace emitted by ``demo.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibration_contract import QUALITY_LIMITS, canonical_parameters, canonical_sha256, parameters_sha256
from scripts.mechanism_config import FIXED, FROZEN_BUNDLE_FILES, sha256_file
from scripts.saes_execution_identity import build_saes_execution_identity


PROTOCOL = "acid_train_holdout_v1"


def _load_trace(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "calibration_sample_trace":
        raise ValueError(f"invalid calibration trace: {path}")
    trace = value.get("trace")
    if not isinstance(trace, Mapping) or trace.get("target_rgb_accessed") is not False:
        raise ValueError(f"calibration trace is not target-free: {path}")
    if trace.get("neural_forward_passes") != 1:
        raise ValueError(f"calibration trace did not execute one forward pass: {path}")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError(f"calibration trace must contain one committed candidate: {path}")
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise ValueError(f"calibration candidate is invalid: {path}")
    parameters = canonical_parameters(candidate.get("parameters", {}))
    quality = candidate.get("quality")
    if not isinstance(quality, Mapping):
        raise ValueError(f"calibration candidate has no quality metrics: {path}")
    normalized_quality = {}
    for metric, limit in QUALITY_LIMITS.items():
        value_metric = quality.get(metric)
        if not isinstance(value_metric, (int, float)) or value_metric < 0:
            raise ValueError(f"calibration metric is invalid: {path} {metric}")
        normalized_quality[metric] = float(value_metric)
        if float(value_metric) > limit:
            raise ValueError(
                f"calibration candidate violates {metric}={limit}: {path}"
            )
    selection = {
        "sample_index": value.get("sample_index"),
        "scene": value.get("scene"),
        "context_indices": value.get("context_indices"),
        "target_indices": value.get("target_indices"),
    }
    if not isinstance(selection["scene"], str) or not selection["scene"]:
        raise ValueError(f"calibration trace has no scene: {path}")
    return {
        "path": path,
        "record": value,
        "candidate": {
            "parameters": parameters,
            "quality": normalized_quality,
            "work_reduction": float(candidate.get("work_reduction", 0.0)),
            "compression": float(candidate.get("compression", 0.0)),
        },
        "selection": selection,
        "trace": trace,
    }


def _trace_set_digest(entries: list[dict[str, Any]]) -> str:
    return canonical_sha256(
        [
            {
                "path": entry["path"].name,
                "sha256": sha256_file(entry["path"]),
                "selection": entry["selection"],
            }
            for entry in entries
        ]
    )


def _split_evidence(entries: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not entries:
        raise ValueError(f"{label} trace set is empty")
    parameters = entries[0]["candidate"]["parameters"]
    if any(entry["candidate"]["parameters"] != parameters for entry in entries):
        raise ValueError(f"{label} traces disagree on the committed tuple")
    selections = [entry["selection"] for entry in entries]
    scenes = sorted(selection["scene"] for selection in selections)
    return {
        "count": len(entries),
        "selection_sha256": canonical_sha256(selections),
        "scene_set_sha256": canonical_sha256(scenes),
        "trace_set_sha256": _trace_set_digest(entries),
        "candidate_set_sha256": canonical_sha256(
            [canonical_sha256(entry["candidate"]) for entry in entries]
        ),
        "validated_parameters_sha256": parameters_sha256(parameters),
        "quality": {
            metric: sum(entry["candidate"]["quality"][metric] for entry in entries) / len(entries)
            for metric in QUALITY_LIMITS
        },
        "traces": [
            {
                "path": entry["path"].name,
                "sha256": sha256_file(entry["path"]),
                "scene": entry["selection"]["scene"],
            }
            for entry in entries
        ],
    }


def freeze(train_paths: list[Path], holdout_paths: list[Path], output_dir: Path) -> Path:
    train = [_load_trace(path.resolve()) for path in train_paths]
    holdout = [_load_trace(path.resolve()) for path in holdout_paths]
    train_scenes = {entry["selection"]["scene"] for entry in train}
    holdout_scenes = {entry["selection"]["scene"] for entry in holdout}
    if train_scenes & holdout_scenes:
        raise ValueError("ACID train and holdout scenes overlap")
    train_evidence = _split_evidence(train, "train")
    holdout_evidence = _split_evidence(holdout, "holdout")
    selected = train[0]["candidate"]["parameters"]
    if holdout[0]["candidate"]["parameters"] != selected:
        raise ValueError("holdout tuple does not match train tuple")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "scarf-acid-calibration-manifest-v1",
        "protocol": PROTOCOL,
        "dataset": "acid",
        "evaluation_disjoint": True,
        "train": train_evidence,
        "holdout": holdout_evidence,
        "source": "local-real-acid-target-free-traces",
    }
    manifest_path = output_dir / "acid-calibration-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha256 = sha256_file(manifest_path)

    candidates = {
        "schema_version": "scarf-calibration-candidates-v1",
        "kind": "frozen_calibration_candidate_summary",
        "protocol": PROTOCOL,
        "claim_eligible": True,
        "selected": selected,
        "selection_rule": "quality constraints, maximum event work reduction",
        "train": train_evidence,
        "holdout": holdout_evidence,
    }
    result = {
        "schema_version": "scarf-calibration-result-v1",
        "protocol": PROTOCOL,
        "status": "PASS",
        "evaluation_disjoint": True,
        "selected": selected,
        "source": "local-real-acid-target-free-traces",
        "train_count": len(train),
        "holdout_count": len(holdout),
        "manifest_sha256": manifest_sha256,
        "candidate_records_sha256": None,
    }
    summary = {
        "schema_version": "scarf-calibration-manifest-summary-v1",
        "protocol": PROTOCOL,
        "dataset": "acid",
        "evaluation_disjoint": True,
        "train_count": len(train),
        "holdout_count": len(holdout),
        "train_scene_set_sha256": train_evidence["scene_set_sha256"],
        "holdout_scene_set_sha256": holdout_evidence["scene_set_sha256"],
        "manifest_sha256": manifest_sha256,
        "source": "local-real-acid-target-free-traces",
    }

    candidates_path = output_dir / "candidates.json"
    candidates_path.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")
    result["candidate_records_sha256"] = sha256_file(candidates_path)
    result_path = output_dir / "calibration-result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary_path = output_dir / "manifest-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    config = {
        "schema_version": "1.0",
        "status": "calibrated",
        "global_configuration": True,
        "projection": {
            "kind": "random_hyperplane_lsh",
            "seed": 42,
            "rom": "artifact/lsh_projection.json",
            "rom_sha256": sha256_file(ROOT / "artifact/lsh_projection.json"),
        },
        "fixed": dict(FIXED),
        "saes_execution_identity": build_saes_execution_identity(),
        "search_space": {
            "gamma_depth": [0.05, 0.075, 0.1, 0.15],
            "beta_x": [0.25, 0.5, 1.0],
            "beta_f": [0.05, 0.1, 0.2],
            "beta_d": [0.5, 1.0, 2.0],
        },
        "selected": selected,
        "calibration": {
            "manifest_sha256": manifest_sha256,
            "candidate_records_sha256": result["candidate_records_sha256"],
            "evaluation_disjoint": True,
            "protocol": PROTOCOL,
            "train_holdout_scene_disjoint": True,
            "train": {
                **{key: train_evidence[key] for key in ("selection_sha256", "scene_set_sha256", "trace_set_sha256", "candidate_set_sha256")},
                "pair_bindings_sha256": canonical_sha256({"model": "mvsplat", "dataset": "acid"}),
                "selected_candidate_sha256": canonical_sha256(train[0]["candidate"]),
            },
            "holdout": {
                **{key: holdout_evidence[key] for key in ("selection_sha256", "scene_set_sha256", "trace_set_sha256", "candidate_set_sha256")},
                "pair_bindings_sha256": canonical_sha256({"model": "mvsplat", "dataset": "acid"}),
                "validated_parameters_sha256": holdout_evidence["validated_parameters_sha256"],
                "validated_candidate_sha256": canonical_sha256(holdout[0]["candidate"]),
            },
            "selection_rule": "quality constraints, maximum event work reduction",
            "source": "local-real-acid-target-free-traces",
            "selected_parameters": selected,
        },
    }
    # Companion files are copied into the shipped frozen directory and their
    # actual bytes are bound from the root config.
    frozen = ROOT / "artifact/calibration/frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    for key, source in (("candidate_records", candidates_path), ("calibration_result", result_path), ("manifest_summary", summary_path)):
        target = frozen / FROZEN_BUNDLE_FILES[key]
        target.write_bytes(source.read_bytes())
    config["calibration"]["frozen_files"] = {
        key: {"path": name, "sha256": sha256_file(frozen / name)}
        for key, name in FROZEN_BUNDLE_FILES.items()
    }
    config_path = ROOT / "artifact/mechanism_config.json"
    config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    config_path.write_text(config_text)
    (frozen / "mechanism_config.json").write_text(config_text)
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--holdout", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(freeze(args.train, args.holdout, args.output_dir))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
