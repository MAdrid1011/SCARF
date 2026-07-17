"""Pure, target-independent rules for SCARF mechanism calibration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


CALIBRATION_DOMAIN = "SCARF-AE-calibration-v1"
QUALITY_LIMITS = {
    "psnr_loss_db": 0.05,
    "ssim_loss": 0.003,
    "lpips_increase": 0.003,
}
PARAMETER_GRID = {
    "gamma_depth": (0.05, 0.075, 0.10, 0.15),
    "beta_x": (0.25, 0.50, 1.00),
    "beta_f": (0.05, 0.10, 0.20),
    "beta_d": (0.50, 1.00, 2.00),
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    scene = entry.get("scene")
    context = entry.get("context_indices")
    target = entry.get("target_indices")
    if not isinstance(scene, str) or not scene:
        raise ValueError("calibration entry has no scene")
    for label, values in (("context", context), ("target", target)):
        if (
            not isinstance(values, list)
            or not values
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"calibration entry has invalid {label} indices")
    return {
        "scene": scene,
        "context_indices": list(context),
        "target_indices": list(target),
    }


def hash_ranked_scene_names(
    scenes: Sequence[str], *, dataset: str, count: int
) -> list[str]:
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset must be non-empty")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("selection count must be positive")
    if any(not isinstance(scene, str) or not scene for scene in scenes):
        raise ValueError("calibration scene names must be non-empty strings")
    if len(scenes) != len(set(scenes)):
        raise ValueError("calibration entries contain duplicate scenes")
    if count > len(scenes):
        raise ValueError("selection count exceeds available scenes")

    def rank(scene: str) -> tuple[str, str]:
        material = f"{CALIBRATION_DOMAIN}\0{dataset}\0{scene}".encode("utf-8")
        return hashlib.sha256(material).hexdigest(), scene

    return sorted(scenes, key=rank)[:count]


def hash_ranked_selection(
    entries: Sequence[Mapping[str, Any]], *, dataset: str, count: int
) -> list[dict[str, Any]]:
    canonical = [_canonical_entry(entry) for entry in entries]
    by_scene = {entry["scene"]: entry for entry in canonical}
    selected = hash_ranked_scene_names(
        list(by_scene), dataset=dataset, count=count
    )
    return [dict(by_scene[scene]) for scene in selected]


def assert_evaluation_disjoint(
    calibration: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> None:
    calibration_scenes = {_canonical_entry(entry)["scene"] for entry in calibration}
    evaluation_scenes = {_canonical_entry(entry)["scene"] for entry in evaluation}
    overlap = sorted(calibration_scenes & evaluation_scenes)
    if overlap:
        raise ValueError(f"calibration/evaluation overlap: {overlap[0]}")


def validate_candidate(candidate: Mapping[str, Any]) -> None:
    if not isinstance(candidate, Mapping):
        raise ValueError("calibration candidate must be an object")
    if "pair_overrides" in candidate:
        raise ValueError("pair-specific calibration parameters are forbidden")
    parameters = candidate.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("calibration candidate has no parameters")
    if "projection_seed" in parameters:
        raise ValueError("projection seed search is forbidden")
    unknown = set(parameters) - set(PARAMETER_GRID)
    if unknown:
        raise ValueError(f"unknown calibration parameter: {sorted(unknown)[0]}")
    if parameters and set(parameters) != set(PARAMETER_GRID):
        raise ValueError("calibration candidate must contain the complete global tuple")
    for name, value in parameters.items():
        if value not in PARAMETER_GRID[name]:
            raise ValueError(f"calibration parameter is outside registered grid: {name}")


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _candidate_feasible(candidate: Mapping[str, Any]) -> bool:
    quality = candidate.get("quality")
    if not isinstance(quality, Mapping) or not quality:
        raise ValueError("calibration candidate has no pair quality records")
    for pair, metrics in quality.items():
        if not isinstance(pair, str) or "/" not in pair or not isinstance(metrics, Mapping):
            raise ValueError("calibration quality record is invalid")
        for metric, limit in QUALITY_LIMITS.items():
            if _finite_nonnegative(metrics.get(metric), f"{pair}.{metric}") > limit:
                return False
    return True


def select_global_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    feasible = []
    for candidate in candidates:
        validate_candidate(candidate)
        work = _finite_nonnegative(candidate.get("work_reduction"), "work_reduction")
        compression = _finite_nonnegative(candidate.get("compression"), "compression")
        if _candidate_feasible(candidate):
            parameters = candidate["parameters"]
            parameter_key = tuple(float(parameters[name]) for name in sorted(PARAMETER_GRID))
            feasible.append((-work, compression, parameter_key, dict(candidate)))
    if not feasible:
        raise ValueError("no calibration candidate satisfies the quality contract")
    feasible.sort(key=lambda item: item[:3])
    return feasible[0][3]
