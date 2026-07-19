"""Frozen target-free records for V4 selected-anchor attribute replay risk."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from saes.packed_l0_l1_materializer import (
    CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
    SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CERTIFICATE,
    SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CENTER_POLICY,
)
from saes.probe_first_schedule import ADAPTIVE_L1_15_ANCHOR_SEMANTICS


CALIBRATION_KIND = "saes-adaptive-l1-v4-attribute-loo-calibration-v1"
CALIBRATION_SCHEMA_VERSION = "1.0"
DEFAULT_CALIBRATION_QUANTILE = 0.25


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def finite_quantile(values: Sequence[float], fraction: float) -> float:
    if not values or not isinstance(fraction, float) or not 0.0 <= fraction <= 1.0:
        raise ValueError("V4 replay calibration requires a nonempty valid quantile")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise ValueError("V4 replay risks must be finite and nonnegative")
    return ordered[round((len(ordered) - 1) * fraction)]


def risk_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("V4 replay calibration sample has no L1 risks")
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


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"V4 replay calibration {label} hash is invalid")
    return value


def build_calibration_record(
    *,
    sample_records: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
    source_index_sha256: str,
    sample_selection_sha256: str,
    v15_calibration_sha256: str,
    quantile: float = DEFAULT_CALIBRATION_QUANTILE,
) -> dict[str, Any]:
    """Freeze a conservative minimum-per-scene V4 replay threshold."""
    checkpoint_sha256 = _require_sha256(checkpoint_sha256, "checkpoint")
    source_index_sha256 = _require_sha256(source_index_sha256, "source index")
    sample_selection_sha256 = _require_sha256(sample_selection_sha256, "selection")
    v15_calibration_sha256 = _require_sha256(v15_calibration_sha256, "V15 calibration")
    if quantile != DEFAULT_CALIBRATION_QUANTILE:
        raise ValueError("V4 replay calibration quantile must use the fixed q25")
    indices: list[int] = []
    scenes: list[str] = []
    normalized_records: list[dict[str, Any]] = []
    all_risks: list[float] = []
    per_scene_thresholds: list[float] = []
    for record in sample_records:
        if not isinstance(record, Mapping):
            raise ValueError("V4 replay calibration sample record is invalid")
        sample_index = record.get("sample_index")
        scene = record.get("scene")
        risks = record.get("risks")
        access = record.get("access")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index <= 0
            or not isinstance(scene, str)
            or not scene
            or not isinstance(risks, Sequence)
            or isinstance(risks, (str, bytes))
            or not isinstance(access, Mapping)
        ):
            raise ValueError("V4 replay calibration sample binding is invalid")
        if any(
            access.get(name) is not False
            for name in (
                "target_rgb_accessed",
                "target_camera_accessed",
                "skipped_s3_attributes_accessed",
            )
        ):
            raise ValueError("V4 replay calibration must be target-free and selected-only")
        values = [float(value) for value in risks]
        summary = risk_summary(values)
        scene_threshold = finite_quantile(values, quantile)
        indices.append(sample_index)
        scenes.append(scene)
        all_risks.extend(values)
        per_scene_thresholds.append(scene_threshold)
        normalized_records.append(
            {
                "sample_index": sample_index,
                "scene": scene,
                "risk_summary": summary,
                "q25_risk": scene_threshold,
                "access": {
                    "target_rgb_accessed": False,
                    "target_camera_accessed": False,
                    "skipped_s3_attributes_accessed": False,
                },
            }
        )
    if len(indices) < 2 or len(set(indices)) != len(indices) or len(set(scenes)) != len(scenes):
        raise ValueError("V4 replay calibration requires distinct non-evaluation scenes")
    threshold = min(per_scene_thresholds)
    record: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": CALIBRATION_KIND,
        "status": "FROZEN_TARGET_FREE_DEV_CALIBRATION",
        "paper_result_eligible": False,
        "model": "transplat",
        "dataset": "dl3dv",
        "configuration": {
            "tile_size": 4,
            "aggregation": CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
            "l1_anchor_semantics": ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
            "risk_signal": SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CERTIFICATE,
            "loo_center_policy": SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CENTER_POLICY,
            "replay_input_anchor_count": 14,
            "deployment_input_anchor_count": 15,
            "quantile": quantile,
            "threshold_rule": "minimum-per-scene-q25",
            "v15_filter_calibration_overlap": True,
        },
        "source": {
            "checkpoint_sha256": checkpoint_sha256,
            "source_index_sha256": source_index_sha256,
            "sample_selection_sha256": sample_selection_sha256,
            "v15_adaptive_l1_calibration_sha256": v15_calibration_sha256,
        },
        "access": {
            "target_rgb_accessed": False,
            "target_camera_accessed": False,
            "skipped_s3_attributes_accessed": False,
        },
        "calibration_samples": normalized_records,
        "aggregate_risk_summary": risk_summary(all_risks),
        "threshold": {
            "kind": "maximum-v4-selected-anchor-attribute-loo-q75-risk",
            "value": threshold,
            "calibration_quantile": quantile,
            "rule": "minimum-per-scene-q25",
            "per_scene_q25": per_scene_thresholds,
            "promote_when": "risk_gt_value",
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
    v15_calibration_sha256: str,
) -> dict[str, Any]:
    """Load the V16 threshold without allowing runtime calibration drift."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("V4 replay calibration record is unavailable") from error
    if not isinstance(record, dict):
        raise ValueError("V4 replay calibration record is invalid")
    recorded_sha256 = record.pop("sha256", None)
    if recorded_sha256 != canonical_sha256(record):
        raise ValueError("V4 replay calibration record hash is invalid")
    if (
        record.get("schema_version") != CALIBRATION_SCHEMA_VERSION
        or record.get("kind") != CALIBRATION_KIND
        or record.get("status") != "FROZEN_TARGET_FREE_DEV_CALIBRATION"
        or record.get("paper_result_eligible") is not False
        or record.get("model") != "transplat"
        or record.get("dataset") != "dl3dv"
    ):
        raise ValueError("V4 replay calibration record has an invalid identity")
    configuration = record.get("configuration")
    source = record.get("source")
    access = record.get("access")
    threshold = record.get("threshold")
    samples = record.get("calibration_samples")
    if not all(isinstance(value, Mapping) for value in (configuration, source, access, threshold)) or not isinstance(samples, list):
        raise ValueError("V4 replay calibration record is incomplete")
    expected_source = {
        "checkpoint_sha256": checkpoint_sha256,
        "source_index_sha256": source_index_sha256,
        "sample_selection_sha256": sample_selection_sha256,
        "v15_adaptive_l1_calibration_sha256": v15_calibration_sha256,
    }
    if source != expected_source:
        raise ValueError("V4 replay calibration source binding changed")
    if (
        configuration.get("tile_size") != 4
        or configuration.get("aggregation")
        != CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION
        or configuration.get("l1_anchor_semantics") != ADAPTIVE_L1_15_ANCHOR_SEMANTICS
        or configuration.get("risk_signal") != SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CERTIFICATE
        or configuration.get("loo_center_policy") != SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CENTER_POLICY
        or configuration.get("replay_input_anchor_count") != 14
        or configuration.get("deployment_input_anchor_count") != 15
        or configuration.get("quantile") != DEFAULT_CALIBRATION_QUANTILE
        or configuration.get("threshold_rule") != "minimum-per-scene-q25"
        or configuration.get("v15_filter_calibration_overlap") is not True
    ):
        raise ValueError("V4 replay calibration configuration changed")
    if any(
        access.get(name) is not False
        for name in (
            "target_rgb_accessed",
            "target_camera_accessed",
            "skipped_s3_attributes_accessed",
        )
    ):
        raise ValueError("V4 replay calibration record is not target-free")
    indices = [sample.get("sample_index") for sample in samples if isinstance(sample, Mapping)]
    if evaluation_sample_index in indices or not indices or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in indices
    ):
        raise ValueError("V4 replay calibration overlaps its evaluation sample")
    value = threshold.get("value")
    if (
        threshold.get("kind") != "maximum-v4-selected-anchor-attribute-loo-q75-risk"
        or threshold.get("rule") != "minimum-per-scene-q25"
        or threshold.get("calibration_quantile") != DEFAULT_CALIBRATION_QUANTILE
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("V4 replay calibration threshold is invalid")
    per_scene_q25 = threshold.get("per_scene_q25")
    if (
        not isinstance(per_scene_q25, list)
        or len(per_scene_q25) != len(samples)
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in per_scene_q25
        )
        or float(value) != min(float(item) for item in per_scene_q25)
    ):
        raise ValueError("V4 replay calibration per-scene threshold is invalid")
    return {**record, "sha256": recorded_sha256, "threshold_value": float(value)}
