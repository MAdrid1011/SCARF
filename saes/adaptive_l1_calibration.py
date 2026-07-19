"""Target-free calibration records for adaptive L1 absolute-residual guards."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CALIBRATION_KIND = "saes-adaptive-l1-absolute-residual-calibration-v1"
CALIBRATION_SCHEMA_VERSION = "1.0"
DEFAULT_CALIBRATION_QUANTILE = 0.50


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def finite_quantile(values: Sequence[float], fraction: float) -> float:
    if not values or not isinstance(fraction, float) or not 0.0 <= fraction <= 1.0:
        raise ValueError("adaptive L1 calibration requires a nonempty valid quantile")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise ValueError("adaptive L1 calibration residuals must be finite and nonnegative")
    return ordered[round((len(ordered) - 1) * fraction)]


def residual_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("adaptive L1 calibration sample has no candidate residuals")
    return {
        "candidate_l1_tiles": len(values),
        "minimum": finite_quantile(values, 0.0),
        "p10": finite_quantile(values, 0.10),
        "p25": finite_quantile(values, 0.25),
        "p50": finite_quantile(values, 0.50),
        "p75": finite_quantile(values, 0.75),
        "p90": finite_quantile(values, 0.90),
        "p95": finite_quantile(values, 0.95),
        "p99": finite_quantile(values, 0.99),
        "maximum": finite_quantile(values, 1.0),
    }


def build_calibration_record(
    *,
    sample_records: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
    source_index_sha256: str,
    sample_selection_sha256: str,
    quantile: float = DEFAULT_CALIBRATION_QUANTILE,
) -> dict[str, Any]:
    """Build one self-hashed, target-free dev calibration record."""
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("adaptive L1 calibration checkpoint hash is invalid")
    hashes = (source_index_sha256, sample_selection_sha256)
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
        raise ValueError("adaptive L1 calibration selection hashes are invalid")
    if not isinstance(quantile, float) or quantile != DEFAULT_CALIBRATION_QUANTILE:
        raise ValueError("adaptive L1 calibration quantile must use the frozen median")
    indices: list[int] = []
    scenes: list[str] = []
    all_residuals: list[float] = []
    normalized_records: list[dict[str, Any]] = []
    for record in sample_records:
        if not isinstance(record, Mapping):
            raise ValueError("adaptive L1 calibration sample record is invalid")
        sample_index = record.get("sample_index")
        scene = record.get("scene")
        residuals = record.get("residuals")
        access = record.get("access")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index <= 0
            or not isinstance(scene, str)
            or not scene
            or not isinstance(residuals, Sequence)
            or isinstance(residuals, (str, bytes))
            or not isinstance(access, Mapping)
        ):
            raise ValueError("adaptive L1 calibration sample binding is invalid")
        if any(access.get(name) is not False for name in (
            "target_rgb_accessed",
            "target_camera_accessed",
            "skipped_s3_attributes_accessed",
        )):
            raise ValueError("adaptive L1 calibration must be target-free and selected-only")
        values = [float(value) for value in residuals]
        summary = residual_summary(values)
        indices.append(sample_index)
        scenes.append(scene)
        all_residuals.extend(values)
        normalized_records.append(
            {
                "sample_index": sample_index,
                "scene": scene,
                "residual_summary": summary,
                "access": {
                    "target_rgb_accessed": False,
                    "target_camera_accessed": False,
                    "skipped_s3_attributes_accessed": False,
                },
            }
        )
    if len(indices) < 2 or len(set(indices)) != len(indices) or len(set(scenes)) != len(scenes):
        raise ValueError("adaptive L1 calibration requires distinct non-evaluation scenes")
    threshold = finite_quantile(all_residuals, quantile)
    record: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": CALIBRATION_KIND,
        "status": "FROZEN_TARGET_FREE_DEV_CALIBRATION",
        "paper_result_eligible": False,
        "model": "transplat",
        "dataset": "dl3dv",
        "configuration": {
            "tile_size": 4,
            "feature_threshold": 0.20,
            "depth_threshold": 0.10,
            "decision_semantics": "probe-normalized-std-first-hit",
            "l1_anchor_semantics": "engineering-lightweight-15-adaptive-center-s1-loo-v1",
            "risk_signal": "s1-adaptive-center-leave-one-out-mean-square-residual",
            "quantile": quantile,
        },
        "source": {
            "checkpoint_sha256": checkpoint_sha256,
            "source_index_sha256": source_index_sha256,
            "sample_selection_sha256": sample_selection_sha256,
        },
        "access": {
            "target_rgb_accessed": False,
            "target_camera_accessed": False,
            "skipped_s3_attributes_accessed": False,
        },
        "calibration_samples": normalized_records,
        "aggregate_residual_summary": residual_summary(all_residuals),
        "threshold": {
            "kind": "maximum-absolute-s1-loo-residual",
            "value": threshold,
            "retained_fraction_on_calibration": quantile,
            "promote_when": "residual_gt_value",
        },
    }
    return {**record, "sha256": canonical_sha256(record)}


def load_frozen_threshold(
    path: Path,
    *,
    evaluation_sample_index: int,
    checkpoint_sha256: str,
    source_index_sha256: str,
    sample_selection_sha256: str,
) -> dict[str, Any]:
    """Load and validate the only runtime input allowed for the residual guard."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("adaptive L1 calibration record is unavailable") from error
    if not isinstance(record, dict):
        raise ValueError("adaptive L1 calibration record is invalid")
    recorded_sha256 = record.pop("sha256", None)
    if recorded_sha256 != canonical_sha256(record):
        raise ValueError("adaptive L1 calibration record hash is invalid")
    if (
        record.get("schema_version") != CALIBRATION_SCHEMA_VERSION
        or record.get("kind") != CALIBRATION_KIND
        or record.get("status") != "FROZEN_TARGET_FREE_DEV_CALIBRATION"
        or record.get("paper_result_eligible") is not False
        or record.get("model") != "transplat"
        or record.get("dataset") != "dl3dv"
    ):
        raise ValueError("adaptive L1 calibration record has an invalid identity")
    source = record.get("source")
    configuration = record.get("configuration")
    access = record.get("access")
    threshold = record.get("threshold")
    samples = record.get("calibration_samples")
    if (
        not isinstance(source, Mapping)
        or not isinstance(configuration, Mapping)
        or not isinstance(access, Mapping)
        or not isinstance(threshold, Mapping)
        or not isinstance(samples, list)
    ):
        raise ValueError("adaptive L1 calibration record is incomplete")
    if source != {
        "checkpoint_sha256": checkpoint_sha256,
        "source_index_sha256": source_index_sha256,
        "sample_selection_sha256": sample_selection_sha256,
    }:
        raise ValueError("adaptive L1 calibration source binding changed")
    if configuration.get("quantile") != DEFAULT_CALIBRATION_QUANTILE:
        raise ValueError("adaptive L1 calibration quantile changed")
    if any(access.get(name) is not False for name in (
        "target_rgb_accessed",
        "target_camera_accessed",
        "skipped_s3_attributes_accessed",
    )):
        raise ValueError("adaptive L1 calibration record is not target-free")
    indices = [sample.get("sample_index") for sample in samples if isinstance(sample, Mapping)]
    if evaluation_sample_index in indices or not indices or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in indices
    ):
        raise ValueError("adaptive L1 calibration overlaps its evaluation sample")
    value = threshold.get("value")
    if (
        threshold.get("kind") != "maximum-absolute-s1-loo-residual"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("adaptive L1 calibration threshold is invalid")
    return {**record, "sha256": recorded_sha256, "threshold_value": float(value)}
