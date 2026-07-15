#!/usr/bin/env python3
"""Validate one SCARF AE results.json record."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _get(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing required field: {path}")
        value = value[key]
    return value


def _positive(record: dict[str, Any], path: str) -> float:
    value = _get(record, path)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path} must be a positive finite number")
    return float(value)


def _sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate(record: dict[str, Any]) -> None:
    if _get(record, "schema_version") != "1.0":
        raise ValueError("schema_version must be 1.0")

    for path in (
        "provenance.git_commit",
        "provenance.git_dirty",
        "provenance.source_identity",
        "provenance.submodules.transplat",
        "provenance.submodules.mvsplat",
        "provenance.submodules.depthsplat",
        "provenance.command",
        "provenance.runtime_assets",
        "provenance.environment.profile",
        "provenance.environment.digest_sha256",
        "provenance.seed",
        "provenance.device.type",
        "provenance.dataset.name",
        "provenance.dataset.representation",
        "provenance.dataset.functional_fixture",
        "provenance.dataset.paper_result_eligible",
        "provenance.dataset.sha256",
        "provenance.dataset.tree_sha256",
        "provenance.checkpoint.path",
        "provenance.checkpoint.sha256",
        "provenance.checkpoint.load.matched_tensors",
        "provenance.checkpoint.load.matched_checkpoint_numel_fraction",
        "provenance.evaluation.kind",
        "quality.baseline.psnr_db",
        "quality.baseline.ssim",
        "quality.baseline.lpips",
        "quality.scarf.psnr_db",
        "quality.scarf.ssim",
        "quality.scarf.lpips",
        "quality.change.psnr_signed_pct",
        "quality.change.psnr_degradation_pct",
        "quality.change.psnr_absolute_pct",
        "performance.cycle_source",
        "performance.baseline_source",
        "validation.reproducible",
        "validation.reference_fallback_used",
    ):
        _get(record, path)

    evaluation = _get(record, "provenance.evaluation")
    if not isinstance(_get(record, "provenance.git_dirty"), bool):
        raise ValueError("provenance.git_dirty must be boolean")
    if _get(record, "provenance.source_identity") not in {"git", "release_manifest"}:
        raise ValueError("provenance.source_identity is invalid")
    runtime_assets = _get(record, "provenance.runtime_assets")
    if not isinstance(runtime_assets, dict) or not runtime_assets:
        raise ValueError("provenance.runtime_assets must be a non-empty object")
    command = _get(record, "provenance.command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) and value for value in command)
        or any(Path(value).is_absolute() for value in command)
    ):
        raise ValueError("provenance.command must be a non-empty portable command")
    environment_digest = _get(record, "provenance.environment.digest_sha256")
    if not _sha256_digest(environment_digest):
        raise ValueError("provenance.environment.digest_sha256 must be a SHA256 digest")
    representation = _get(record, "provenance.dataset.representation")
    functional_fixture = _get(record, "provenance.dataset.functional_fixture")
    paper_result_eligible = _get(record, "provenance.dataset.paper_result_eligible")
    if not isinstance(functional_fixture, bool) or not isinstance(
        paper_result_eligible, bool
    ):
        raise ValueError("dataset eligibility fields must be boolean")
    if paper_result_eligible is functional_fixture:
        raise ValueError("dataset paper-result eligibility is inconsistent")
    if functional_fixture and representation != "re10k-synthetic-functional-v1":
        raise ValueError("functional fixture representation is invalid")
    if _positive(record, "provenance.checkpoint.load.matched_tensors") <= 0:
        raise ValueError("checkpoint matched tensor count must be positive")
    matched_fraction = _get(
        record, "provenance.checkpoint.load.matched_checkpoint_numel_fraction"
    )
    if (
        not isinstance(matched_fraction, (int, float))
        or isinstance(matched_fraction, bool)
        or not math.isfinite(matched_fraction)
        or not 0 < matched_fraction <= 1
    ):
        raise ValueError("checkpoint matched numel fraction must be in (0, 1]")
    tree_sha256 = _get(record, "provenance.dataset.tree_sha256")
    if not isinstance(representation, str) or not representation:
        raise ValueError("provenance.dataset.representation must be non-empty")
    if (
        not isinstance(tree_sha256, str)
        or len(tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tree_sha256)
    ):
        raise ValueError("provenance.dataset.tree_sha256 must be a SHA256 digest")
    if evaluation["kind"] == "sample":
        index = _get(record, "provenance.evaluation.sample_index")
        execution_index = _get(record, "provenance.evaluation.execution_index")
        count = _get(record, "provenance.evaluation.candidate_count")
        if (
            not isinstance(index, int)
            or index < 0
            or not isinstance(execution_index, int)
            or not isinstance(count, int)
            or not 0 <= execution_index < count
        ):
            raise ValueError("invalid sample evaluation selection")
        scene = _get(record, "provenance.evaluation.scene")
        context = _get(record, "provenance.evaluation.context_indices")
        target = _get(record, "provenance.evaluation.target_indices")
        if not isinstance(scene, str) or not scene:
            raise ValueError("sample evaluation has no scene key")
        if not isinstance(context, list) or not context or not all(isinstance(v, int) for v in context):
            raise ValueError("sample evaluation has invalid context indices")
        if not isinstance(target, list) or not target or not all(isinstance(v, int) for v in target):
            raise ValueError("sample evaluation has invalid target indices")
        if _get(record, "provenance.evaluation.target_view_count") != len(target):
            raise ValueError("sample target view count does not match target indices")
        if _get(record, "provenance.evaluation.target_view_aggregation") != (
            "arithmetic mean over selected target views"
        ):
            raise ValueError("sample target view aggregation is invalid")
        views = _get(record, "quality.views")
        if not isinstance(views, list) or len(views) != len(target):
            raise ValueError("sample quality does not cover every target view")
        for expected_target, view in zip(target, views):
            if not isinstance(view, dict) or view.get("target_index") != expected_target:
                raise ValueError("sample quality target indices are inconsistent")
            for variant in ("baseline", "scarf"):
                metrics = view.get(variant)
                if not isinstance(metrics, dict):
                    raise ValueError(f"sample quality view has no {variant} metrics")
                for metric in ("psnr_db", "ssim", "lpips"):
                    value = metrics.get(metric)
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                    ):
                        raise ValueError(
                            f"sample quality view {variant}.{metric} must be finite"
                        )
    elif evaluation["kind"] == "dataset_aggregate":
        count = _get(record, "provenance.evaluation.sample_count")
        indices = _get(record, "provenance.evaluation.sample_indices")
        execution_indices = _get(record, "provenance.evaluation.execution_indices")
        if (
            not isinstance(count, int)
            or count <= 0
            or not isinstance(indices, list)
            or len(indices) != count
            or len(set(indices)) != count
            or execution_indices != list(range(count))
        ):
            raise ValueError("invalid dataset aggregate sample set")
        selections = _get(record, "provenance.evaluation.sample_selection")
        selection_hash = _get(record, "provenance.evaluation.sample_selection_sha256")
        if not isinstance(selections, list) or len(selections) != count:
            raise ValueError("dataset aggregate sample selection is incomplete")
        canonical = json.dumps(selections, sort_keys=True, separators=(",", ":")).encode()
        import hashlib

        if selection_hash != hashlib.sha256(canonical).hexdigest():
            raise ValueError("dataset aggregate sample selection hash mismatch")
    else:
        raise ValueError("provenance.evaluation.kind must be sample or dataset_aggregate")

    baseline = _positive(record, "performance.baseline_cycles")
    scarf = _positive(record, "performance.scarf_cycles")
    speedup = _positive(record, "performance.speedup")
    if not math.isclose(speedup, baseline / scarf, rel_tol=1e-6):
        raise ValueError("performance.speedup does not equal baseline_cycles/scarf_cycles")
    if _get(record, "validation.reference_fallback_used") is not False:
        raise ValueError("validation.reference_fallback_used must be false")
    if _get(record, "validation.reproducible") is not True:
        raise ValueError("validation.reproducible must be true")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        record = json.loads(args.result.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("result root must be an object")
        validate(record)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
