"""Build and validate target-free FSDR Table 2 evidence records."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


COUNT_KEYS = (
    "total_pixels",
    "cache_hits",
    "cache_misses",
    "guided_pixels",
    "guided_in_window",
    "guided_out_window",
    "guided_top1_covered",
    "guided_top1_missed",
    "depth_inconsistent",
    "hit_no_guide",
)


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA256 digest")
    return value


def _count(value: Any, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _validate_provenance(
    provenance: Mapping[str, Any], *, paper_result_eligible: bool
) -> None:
    commit = provenance.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("provenance.git_commit must be a Git commit")
    if not isinstance(provenance.get("git_dirty"), bool):
        raise ValueError("provenance.git_dirty must be boolean")
    if provenance.get("source_identity") not in {"git", "release_manifest"}:
        raise ValueError("provenance.source_identity is invalid")
    submodules = provenance.get("submodules")
    if not isinstance(submodules, dict) or set(submodules) != {
        "transplat",
        "mvsplat",
        "depthsplat",
    }:
        raise ValueError("provenance.submodules is incomplete")
    command = provenance.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(value, str) and value for value in command
    ):
        raise ValueError("provenance.command must be a non-empty string list")
    environment = provenance.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("provenance.environment is missing")
    _digest(environment.get("digest_sha256"), "environment digest")
    dataset = provenance.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("paper_result_eligible") is not paper_result_eligible
    ):
        expected = "paper-eligible" if paper_result_eligible else "non-claim"
        raise ValueError(f"FSDR evidence requires a {expected} dataset")
    if not isinstance(dataset.get("name"), str) or not dataset["name"]:
        raise ValueError("provenance.dataset.name is missing")
    _digest(dataset.get("tree_sha256"), "dataset tree")
    checkpoint = provenance.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("provenance.checkpoint is missing")
    _digest(checkpoint.get("sha256"), "checkpoint")
    load = checkpoint.get("load")
    if not isinstance(load, dict):
        raise ValueError("checkpoint load coverage is missing")
    _count(load.get("matched_tensors"), "matched_tensors", positive=True)
    fraction = load.get("matched_checkpoint_numel_fraction")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not math.isfinite(float(fraction))
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError("checkpoint matched fraction must be within (0, 1]")
    if not isinstance(provenance.get("runtime_assets"), dict) or not provenance[
        "runtime_assets"
    ]:
        raise ValueError("provenance.runtime_assets is missing")
    target_rgb = provenance.get("target_rgb")
    if not isinstance(target_rgb, dict):
        raise ValueError("provenance.target_rgb is missing")
    for field in (
        "loaded_by_native_dataloader",
        "removed_before_device_transfer_or_execution",
        "passed_to_model",
        "used_for_routing_or_metric",
    ):
        if not isinstance(target_rgb.get(field), bool):
            raise ValueError(f"provenance.target_rgb.{field} must be boolean")
    if target_rgb["removed_before_device_transfer_or_execution"] is not True:
        raise ValueError("FSDR target RGB must be removed before execution")
    if target_rgb["passed_to_model"] is not False:
        raise ValueError("FSDR evidence cannot pass target RGB to the model")
    if target_rgb["used_for_routing_or_metric"] is not False:
        raise ValueError("FSDR evidence cannot use target RGB")


def build_fsdr_sample_record(
    *,
    provenance: Mapping[str, Any],
    kind: str = "fsdr_sample",
    total_pixels: int,
    cache_hits: int,
    cache_misses: int,
    guided_pixels: int,
    guided_in_window: int,
    guided_out_window: int,
    guided_top1_covered: int,
    guided_top1_missed: int,
    depth_inconsistent: int,
    hit_no_guide: int,
    full_depth_candidates: int,
    narrowed_depth_candidates: int,
    candidate_domain: str,
    probability_source: str,
    evidence_height: int,
    evidence_width: int,
) -> dict[str, Any]:
    """Build one sample record from observed simulator counters."""
    if kind not in {"fsdr_sample", "fsdr_target_free_audit"}:
        raise ValueError("invalid FSDR sample evidence kind")
    counts = {
        key: _count(value, key, positive=key == "total_pixels")
        for key, value in {
            "total_pixels": total_pixels,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "guided_pixels": guided_pixels,
            "guided_in_window": guided_in_window,
            "guided_out_window": guided_out_window,
            "guided_top1_covered": guided_top1_covered,
            "guided_top1_missed": guided_top1_missed,
            "depth_inconsistent": depth_inconsistent,
            "hit_no_guide": hit_no_guide,
        }.items()
    }
    full = _count(full_depth_candidates, "full_depth_candidates", positive=True)
    narrowed = _count(
        narrowed_depth_candidates, "narrowed_depth_candidates", positive=True
    )
    if narrowed >= full:
        raise ValueError("narrowed depth candidates must be fewer than full candidates")
    if counts["cache_hits"] + counts["cache_misses"] != counts["total_pixels"]:
        raise ValueError("cache hit and miss counts do not cover total_pixels")
    if counts["guided_pixels"] > counts["cache_hits"]:
        raise ValueError("guided pixels cannot exceed cache hits")
    if (
        counts["guided_in_window"] + counts["guided_out_window"]
        != counts["guided_pixels"]
    ):
        raise ValueError("guided window counts do not cover guided pixels")
    if counts["guided_pixels"] <= 0:
        raise ValueError("guided_pixels must be positive; zero fallback is forbidden")
    if (
        counts["guided_top1_covered"] + counts["guided_top1_missed"]
        != counts["guided_pixels"]
    ):
        raise ValueError("discrete Top-1 counts do not cover guided pixels")
    if candidate_domain != "inverse_depth":
        raise ValueError("candidate_domain must be inverse_depth")
    if not isinstance(probability_source, str) or not probability_source:
        raise ValueError("probability_source is missing")
    evidence_height = _count(evidence_height, "evidence_height", positive=True)
    evidence_width = _count(evidence_width, "evidence_width", positive=True)

    record = {
        "schema_version": "1.0",
        "kind": kind,
        "provenance": copy.deepcopy(dict(provenance)),
        "fsdr": {
            "counts": counts,
            "config": {
                "full_depth_candidates": full,
                "narrowed_depth_candidates": narrowed,
                "candidate_domain": candidate_domain,
                "probability_source": probability_source,
                "evidence_height": evidence_height,
                "evidence_width": evidence_width,
            },
            "metrics": {
                "guided_rate": counts["guided_pixels"] / counts["total_pixels"],
                "top1_coverage": (
                    counts["guided_top1_covered"] / counts["guided_pixels"]
                ),
            },
            "derived": {
                "depth_evaluations_saved": counts["guided_pixels"]
                * (full - narrowed),
            },
        },
        "validation": {
            "reproducible": True,
            "reference_fallback_used": False,
            "target_image_used_for_routing": False,
            "discrete_candidate_evidence": True,
            "complete_sample": True,
            "paper_result_eligible": kind == "fsdr_sample",
        },
    }
    validate_fsdr_record(record)
    return record


def _canonical_selection(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": evaluation["sample_index"],
        "scene": evaluation["scene"],
        "context_indices": evaluation["context_indices"],
        "target_indices": evaluation["target_indices"],
    }


def validate_fsdr_record(record: Mapping[str, Any]) -> None:
    """Reject incomplete, synthetic, fallback, or internally inconsistent evidence."""
    if record.get("schema_version") != "1.0":
        raise ValueError("schema_version must be 1.0")
    kind = record.get("kind")
    if kind not in {
        "fsdr_sample",
        "fsdr_dataset_aggregate",
        "fsdr_target_free_audit",
    }:
        raise ValueError("invalid FSDR evidence kind")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("provenance is missing")
    paper_result_eligible = kind in {"fsdr_sample", "fsdr_dataset_aggregate"}
    _validate_provenance(
        provenance, paper_result_eligible=paper_result_eligible
    )
    evaluation = provenance.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("provenance.evaluation is missing")
    if kind in {"fsdr_sample", "fsdr_target_free_audit"}:
        if evaluation.get("kind") != "sample":
            raise ValueError("FSDR sample or audit has invalid evaluation kind")
        for key in ("sample_index", "execution_index", "candidate_count"):
            _count(evaluation.get(key), key, positive=key == "candidate_count")
        if not 0 <= evaluation["execution_index"] < evaluation["candidate_count"]:
            raise ValueError("sample execution index is out of range")
        _canonical_selection(evaluation)
    else:
        if evaluation.get("kind") != "dataset_aggregate":
            raise ValueError("FSDR aggregate has invalid evaluation kind")
        sample_count = _count(
            evaluation.get("sample_count"), "sample_count", positive=True
        )
        selections = evaluation.get("sample_selection")
        if not isinstance(selections, list) or len(selections) != sample_count:
            raise ValueError("aggregate sample selection is incomplete")
        canonical = json.dumps(
            selections, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if evaluation.get("sample_selection_sha256") != hashlib.sha256(
            canonical
        ).hexdigest():
            raise ValueError("aggregate sample selection hash mismatch")

    fsdr = record.get("fsdr")
    if not isinstance(fsdr, dict):
        raise ValueError("fsdr result is missing")
    counts = fsdr.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("fsdr counts are missing")
    checked = {
        key: _count(counts.get(key), key, positive=key == "total_pixels")
        for key in COUNT_KEYS
    }
    if checked["cache_hits"] + checked["cache_misses"] != checked["total_pixels"]:
        raise ValueError("cache counts do not cover total pixels")
    if checked["guided_pixels"] <= 0:
        raise ValueError("guided_pixels must be positive")
    if checked["guided_pixels"] > checked["cache_hits"]:
        raise ValueError("guided pixels exceed cache hits")
    if checked["guided_in_window"] + checked["guided_out_window"] != checked[
        "guided_pixels"
    ]:
        raise ValueError("guided window counts are inconsistent")
    if checked["guided_top1_covered"] + checked["guided_top1_missed"] != checked[
        "guided_pixels"
    ]:
        raise ValueError("discrete Top-1 counts are inconsistent")
    config = fsdr.get("config")
    if not isinstance(config, dict):
        raise ValueError("fsdr config is missing")
    full = _count(config.get("full_depth_candidates"), "full candidates", positive=True)
    narrowed = _count(
        config.get("narrowed_depth_candidates"), "narrowed candidates", positive=True
    )
    if narrowed >= full:
        raise ValueError("invalid narrowed candidate count")
    if config.get("candidate_domain") != "inverse_depth":
        raise ValueError("invalid candidate domain")
    if not isinstance(config.get("probability_source"), str) or not config[
        "probability_source"
    ]:
        raise ValueError("probability source is missing")
    _count(config.get("evidence_height"), "evidence height", positive=True)
    _count(config.get("evidence_width"), "evidence width", positive=True)
    metrics = fsdr.get("metrics")
    derived = fsdr.get("derived")
    if not isinstance(metrics, dict) or not isinstance(derived, dict):
        raise ValueError("fsdr metrics or derived counts are missing")
    expected_metrics = {
        "guided_rate": checked["guided_pixels"] / checked["total_pixels"],
        "top1_coverage": checked["guided_top1_covered"] / checked["guided_pixels"],
    }
    for key, expected in expected_metrics.items():
        actual = metrics.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"fsdr metric {key} does not match observed counts")
    saved = derived.get("depth_evaluations_saved")
    if saved != checked["guided_pixels"] * (full - narrowed):
        raise ValueError("depth evaluation savings do not match observed counts")
    validation = record.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("validation record is missing")
    if validation.get("reproducible") is not True:
        raise ValueError("FSDR evidence is not marked reproducible")
    if validation.get("reference_fallback_used") is not False:
        raise ValueError("reference fallback is forbidden")
    if validation.get("target_image_used_for_routing") is not False:
        raise ValueError("target-image routing is forbidden")
    if validation.get("discrete_candidate_evidence") is not True:
        raise ValueError("exact discrete candidate evidence is required")
    if validation.get("paper_result_eligible") is not paper_result_eligible:
        raise ValueError("FSDR evidence eligibility does not match its kind")


def aggregate_fsdr_records(
    records: Iterable[Mapping[str, Any]], *, expected_count: int
) -> dict[str, Any]:
    """Aggregate counters before computing rates, preserving pixel weighting."""
    expected_count = _count(expected_count, "expected_count", positive=True)
    ordered = sorted(
        (copy.deepcopy(dict(record)) for record in records),
        key=lambda record: record["provenance"]["evaluation"]["sample_index"],
    )
    if len(ordered) != expected_count:
        raise ValueError(f"expected {expected_count} samples, found {len(ordered)}")
    for record in ordered:
        validate_fsdr_record(record)
        if record["kind"] != "fsdr_sample":
            raise ValueError("aggregate inputs must be FSDR samples")

    first = ordered[0]
    invariant_paths = (
        ("git_commit",),
        ("git_dirty",),
        ("source_identity",),
        ("submodules",),
        ("model",),
        ("dataset",),
        ("checkpoint",),
        ("environment",),
        ("runtime_assets",),
        ("target_rgb",),
        ("seed",),
        ("device",),
    )
    for path in invariant_paths:
        expected: Any = first["provenance"]
        for key in path:
            expected = expected[key]
        for record in ordered[1:]:
            actual: Any = record["provenance"]
            for key in path:
                actual = actual[key]
            if actual != expected:
                raise ValueError(f"sample provenance mismatch: {'.'.join(path)}")
    config = first["fsdr"]["config"]
    if any(record["fsdr"]["config"] != config for record in ordered[1:]):
        raise ValueError("sample FSDR configs differ")

    counts = {
        key: sum(record["fsdr"]["counts"][key] for record in ordered)
        for key in COUNT_KEYS
    }
    selections = [
        _canonical_selection(record["provenance"]["evaluation"])
        for record in ordered
    ]
    indices = [selection["sample_index"] for selection in selections]
    execution_indices = [
        record["provenance"]["evaluation"]["execution_index"]
        for record in ordered
    ]
    if len(set(indices)) != expected_count:
        raise ValueError("sample indices are not unique")
    if sorted(execution_indices) != list(range(expected_count)):
        raise ValueError("sample execution indices are not contiguous")
    canonical = json.dumps(
        selections, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    provenance = copy.deepcopy(first["provenance"])
    provenance["command"] = ["python", "scripts/aggregate_fsdr.py"]
    provenance["evaluation"] = {
        "kind": "dataset_aggregate",
        "sample_count": expected_count,
        "sample_indices": sorted(indices),
        "execution_indices": sorted(execution_indices),
        "sample_selection": selections,
        "sample_selection_sha256": hashlib.sha256(canonical).hexdigest(),
        "aggregation": "sum counters, then derive pixel-weighted rates",
    }
    aggregate = build_fsdr_sample_record(
        provenance={
            **provenance,
            "evaluation": {
                **provenance["evaluation"],
                "kind": "sample",
                "sample_index": 0,
                "execution_index": 0,
                "candidate_count": 1,
                "scene": "aggregate-builder",
                "context_indices": [0],
                "target_indices": [0],
            },
        },
        total_pixels=counts["total_pixels"],
        cache_hits=counts["cache_hits"],
        cache_misses=counts["cache_misses"],
        guided_pixels=counts["guided_pixels"],
        guided_in_window=counts["guided_in_window"],
        guided_out_window=counts["guided_out_window"],
        guided_top1_covered=counts["guided_top1_covered"],
        guided_top1_missed=counts["guided_top1_missed"],
        depth_inconsistent=counts["depth_inconsistent"],
        hit_no_guide=counts["hit_no_guide"],
        full_depth_candidates=config["full_depth_candidates"],
        narrowed_depth_candidates=config["narrowed_depth_candidates"],
        candidate_domain=config["candidate_domain"],
        probability_source=config["probability_source"],
        evidence_height=config["evidence_height"],
        evidence_width=config["evidence_width"],
    )
    aggregate["kind"] = "fsdr_dataset_aggregate"
    aggregate["provenance"] = provenance
    aggregate["validation"]["complete_sample_set"] = True
    validate_fsdr_record(aggregate)
    return aggregate


def compare_fsdr_aggregate(
    record: Mapping[str, Any], expected_results: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare actual aggregate metrics with the immutable paper contract."""
    provenance = record.get("provenance", {})
    pair = f"{provenance.get('model')}/{provenance.get('dataset', {}).get('name')}"
    try:
        target = expected_results["mechanisms"][pair]
        tolerance = float(expected_results["mechanism_tolerance_absolute"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"expected FSDR contract is missing for {pair}") from error
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("mechanism tolerance must be finite and non-negative")
    checks = {}
    for metric in ("guided_rate", "top1_coverage"):
        actual = float(record["fsdr"]["metrics"][metric])
        expected = float(target[metric])
        delta = actual - expected
        checks[metric] = {
            "actual": actual,
            "expected": expected,
            "delta": delta,
            "absolute_tolerance": tolerance,
            "pass": abs(delta) <= tolerance,
        }
    return {
        "pair": pair,
        "status": "PASS" if all(check["pass"] for check in checks.values()) else "FAIL",
        "checks": checks,
    }


def write_json(record: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
