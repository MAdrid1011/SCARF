#!/usr/bin/env python3
"""Aggregate strict sample-level SCARF records into one dataset result."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.result_record import (
    EXECUTION_TRACE_SET_SCHEMA_VERSION,
    build_quality_record,
    execution_trace_performance_evidence_from_record,
    execution_trace_set_sha256,
    portable_command,
    write_result,
)
from scripts.execution_contract import execution_contract
from scripts.validate_result import reject_reference_only_record, validate


QUALITY_METRICS = ("psnr_db", "ssim", "lpips")
FSDR_EVENT_COUNTS = (
    "total_pixels",
    "cache_hits",
    "guided_pixels",
    "guided_top1_covered",
    "guided_top1_missed",
    "full_depth_evaluations",
    "executed_depth_evaluations",
    "feature_buffer_bytes_baseline",
    "feature_buffer_bytes_actual",
    "hamming_hits",
    "local_valid_hits",
    "local_invalid_fallbacks",
)
SAES_EVENT_COUNTS = (
    "total_tiles",
    "level0_tiles",
    "level1_tiles",
    "full_tiles",
    "baseline_gaussians",
    "actual_gaussians",
    "full_s2_evaluations",
    "executed_s2_evaluations",
    "l0_representatives",
    "l1_lightweight_anchors",
    "full_stage3_gaussians",
    "covariance_psd_violations",
)
SAES_EVENT_MAXIMA = (
    "assignment_weight_sum_error_max",
    "opacity_transmittance_error_max",
)
SAES_SPARSE_COUNTER_MAPS = (
    "deletion_certificate_rejection_reasons",
    "same_budget_dense_oracle_failure_reasons",
    "multicontext_tangent_fallback_reasons",
)
SAES_INVARIANT_FIELDS = (
    "saes_execution_identity",
    "route_sha256",
    "execution_dependency",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_tree(values: list[Any]) -> Any:
    if all(isinstance(value, dict) for value in values):
        keys = set(values[0])
        if any(set(value) != keys for value in values[1:]):
            raise ValueError("sample records have inconsistent nested fields")
        return {key: _mean_tree([value[key] for value in values]) for key in sorted(keys)}
    if all(isinstance(value, list) for value in values):
        if all(value == values[0] for value in values[1:]):
            return copy.deepcopy(values[0])
        return copy.deepcopy(values)
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("sample record contains a non-finite number")
        return statistics.fmean(float(value) for value in values)
    if all(value == values[0] for value in values[1:]):
        return copy.deepcopy(values[0])
    return copy.deepcopy(values)


def _normalize_sparse_saes_counter_maps(
    values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Zero-fill per-sample SAES reason counters before mean aggregation."""

    normalized = copy.deepcopy(values)
    for field in SAES_SPARSE_COUNTER_MAPS:
        reason_maps: list[dict[str, Any]] = []
        for value in normalized:
            saes = value.get("saes")
            if not isinstance(saes, dict) or field not in saes:
                reason_maps = []
                break
            reasons = saes[field]
            if not isinstance(reasons, dict):
                raise ValueError(f"fsdr_saes.saes.{field} must be an object")
            reason_maps.append(reasons)
        if not reason_maps:
            continue

        keys = set().union(*(reasons.keys() for reasons in reason_maps))
        for reasons in reason_maps:
            for key, count in reasons.items():
                if (
                    not isinstance(key, str)
                    or not isinstance(count, (int, float))
                    or isinstance(count, bool)
                    or not math.isfinite(float(count))
                    or count < 0
                ):
                    raise ValueError(
                        f"fsdr_saes.saes.{field} must contain nonnegative counters"
                    )
            for key in keys:
                reasons.setdefault(key, 0)
    return normalized


def _aggregate_fsdr_saes(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean measurements while preserving hash-bound SAES route provenance."""

    normalized = _normalize_sparse_saes_counter_maps(values)
    aggregate = _mean_tree(normalized)
    saes_values = [value.get("saes") for value in normalized]
    if not all(isinstance(value, dict) for value in saes_values):
        return aggregate

    for field in SAES_INVARIANT_FIELDS:
        field_values = [value.get(field) for value in saes_values]
        present = [value is not None for value in field_values]
        if not any(present):
            continue
        if not all(present) or any(
            value != field_values[0] for value in field_values[1:]
        ):
            raise ValueError(f"sample evidence mismatch: fsdr_saes.saes.{field}")
        aggregate["saes"][field] = copy.deepcopy(field_values[0])
    return aggregate


def _mean(records: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{'.'.join(path)} must be finite in every sample")
        values.append(float(value))
    return statistics.fmean(values)


def _dispersion(records: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float]:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(float(value))
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "population_stddev": statistics.pstdev(values),
    }


def _invariant(records: list[dict[str, Any]], path: tuple[str, ...]) -> Any:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(value)
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"sample evidence mismatch: {'.'.join(path)}")
    return copy.deepcopy(values[0])


def _sum_count(records: list[dict[str, Any]], path: tuple[str, ...]) -> int | float:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{'.'.join(path)} must be a nonnegative finite count")
        values.append(value)
    total = sum(values)
    return int(total) if all(isinstance(value, int) for value in values) else float(total)


def _max_count(records: list[dict[str, Any]], path: tuple[str, ...]) -> int | float:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{'.'.join(path)} must be finite and nonnegative")
        values.append(value)
    maximum = max(values)
    return int(maximum) if all(isinstance(value, int) for value in values) else float(maximum)


def _aggregate_v2_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in ("s1", "s2", "s3", "s4"):
        stages[stage] = {
            "cycles": _mean(records, ("performance", "stages", stage, "cycles")),
            "useful_mmcu_slots": _sum_count(
                records, ("performance", "stages", stage, "useful_mmcu_slots")
            ),
            "scheduled_mmcu_slots": _sum_count(
                records, ("performance", "stages", stage, "scheduled_mmcu_slots")
            ),
            "mmcu_slots_available": _invariant(
                records, ("performance", "stages", stage, "mmcu_slots_available")
            ),
            "source": _invariant(
                records, ("performance", "stages", stage, "source")
            ),
        }

    events: dict[str, Any] = {}
    schema_version = records[0]["schema_version"]
    v21 = schema_version == "2.1"
    for namespace, counts, flags in (
        (
            "fsdr",
            FSDR_EVENT_COUNTS,
            (
                "discrete_top1_available",
                "depth_evaluations_available",
                "feature_buffer_bytes_available",
            ),
        ),
        (
            "saes",
            SAES_EVENT_COUNTS,
            (
                "tile_path_available",
                "gaussian_counts_available",
                "s2_evaluations_available",
            ),
        ),
    ):
        if not v21:
            if namespace == "fsdr":
                counts = counts[:9]
            else:
                counts = counts[:8]
        events[namespace] = {
            field: _sum_count(records, ("events", namespace, field))
            for field in counts
        }
        events[namespace].update(
            {
                flag: _invariant(records, ("events", namespace, flag))
                for flag in flags
            }
        )
        events[namespace]["source"] = _invariant(
            records, ("events", namespace, "source")
        )
        if namespace == "saes" and v21:
            events[namespace].update(
                {
                    field: _max_count(records, ("events", namespace, field))
                    for field in SAES_EVENT_MAXIMA
                }
            )
            # The sparse-execution contract is model-scoped and must be
            # identical for every sample.  The measured saving fractions may
            # vary by sample, so preserve their dataset mean explicitly.
            events[namespace]["execution_dependency"] = _invariant(
                records, ("events", namespace, "execution_dependency")
            )
            events[namespace]["s2_s3_saving"] = {
                stage: _mean(records, ("events", namespace, "s2_s3_saving", stage))
                for stage in ("s2", "s3")
            }
    return {"stages": stages, "events": events}


def aggregate(paths: list[Path], expected_count: int) -> dict[str, Any]:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} sample results, found {len(paths)}")

    records = []
    evidence = []
    indices = []
    execution_indices = []
    for path in sorted(path.resolve() for path in paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        reject_reference_only_record(record)
        validate(record)
        evaluation = record["provenance"]["evaluation"]
        if evaluation.get("kind") != "sample":
            raise ValueError(f"input is not a sample result: {path}")
        index = evaluation.get("sample_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"invalid sample index in {path}")
        indices.append(index)
        execution_index = evaluation.get("execution_index", index)
        if (
            not isinstance(execution_index, int)
            or isinstance(execution_index, bool)
            or execution_index < 0
        ):
            raise ValueError(f"invalid execution index in {path}")
        execution_indices.append(execution_index)
        records.append(record)
        pair_root = path.parents[2]
        sample_evidence = {
            "path": str(path.relative_to(pair_root)),
            "sha256": sha256_file(path),
            "sample_index": index,
        }
        trace_digest = record["provenance"].get("execution_trace_sha256")
        if trace_digest is not None:
            sample_evidence["execution_trace_sha256"] = trace_digest
        orin_measurement = path.parent / "orin-evidence" / "measurement.json"
        if orin_measurement.is_file():
            sample_evidence["orin_measurement"] = {
                "path": str(orin_measurement.relative_to(pair_root)),
                "sha256": sha256_file(orin_measurement),
            }
        evidence.append(sample_evidence)

    if len(set(indices)) != expected_count:
        raise ValueError("sample source indices must be unique")
    if sorted(execution_indices) != list(range(expected_count)):
        raise ValueError("sample execution indices must be unique and contiguous from zero")

    trace_bound = [
        "execution_trace_sha256" in record["provenance"] for record in records
    ]
    if any(trace_bound) and not all(trace_bound):
        raise ValueError("sample provenance mismatch: execution trace binding")

    execution_contracts = [execution_contract(record) for record in records]
    if any(contract is not None for contract in execution_contracts):
        if any(contract is None for contract in execution_contracts):
            raise ValueError("sample provenance mismatch: execution_contract")
        if any(contract != execution_contracts[0] for contract in execution_contracts[1:]):
            raise ValueError("sample provenance mismatch: execution_contract")

    selections = [
        {
            "sample_index": record["provenance"]["evaluation"]["sample_index"],
            "scene": record["provenance"]["evaluation"]["scene"],
            "context_indices": record["provenance"]["evaluation"]["context_indices"],
            "target_indices": record["provenance"]["evaluation"]["target_indices"],
        }
        for record in records
    ]
    selections.sort(key=lambda item: item["sample_index"])
    selection_bytes = json.dumps(
        selections, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    trace_set: list[dict[str, Any]] | None = None
    trace_set_sha256: str | None = None
    if all(trace_bound):
        evidence_by_index = {
            item["sample_index"]: item for item in evidence
        }
        trace_set = [
            {
                "sample_index": record["provenance"]["evaluation"]["sample_index"],
                "execution_index": record["provenance"]["evaluation"][
                    "execution_index"
                ],
                "execution_trace_sha256": record["provenance"][
                    "execution_trace_sha256"
                ],
            }
            for record in records
        ]
        for entry in trace_set:
            orin_measurement = evidence_by_index[entry["sample_index"]].get(
                "orin_measurement"
            )
            if isinstance(orin_measurement, dict):
                entry["orin_measurement_sha256"] = orin_measurement["sha256"]
        trace_set.sort(key=lambda item: item["sample_index"])

    first = records[0]
    schema_version = first["schema_version"]
    if any(record["schema_version"] != schema_version for record in records[1:]):
        raise ValueError("sample provenance mismatch: schema_version")
    if schema_version in {"2.0", "2.1"} and any(
        record.get("evidence_class") != first.get("evidence_class")
        for record in records[1:]
    ):
        raise ValueError("sample provenance mismatch: evidence_class")
    invariants = (
        ("model", lambda item: item["provenance"]["model"]),
        ("dataset", lambda item: item["provenance"]["dataset"]),
        ("checkpoint", lambda item: item["provenance"]["checkpoint"]),
        ("git_commit", lambda item: item["provenance"]["git_commit"]),
        ("git_dirty", lambda item: item["provenance"]["git_dirty"]),
        ("source_identity", lambda item: item["provenance"]["source_identity"]),
        (
            "source_tree_sha256",
            lambda item: item["provenance"]["source_tree_sha256"],
        ),
        ("submodules", lambda item: item["provenance"]["submodules"]),
        ("runtime_assets", lambda item: item["provenance"]["runtime_assets"]),
        ("environment", lambda item: item["provenance"]["environment"]),
        ("cycle_source", lambda item: item["performance"]["cycle_source"]),
        ("baseline_source", lambda item: item["performance"]["baseline_source"]),
    )
    if schema_version == "2.1":
        invariants = (*invariants,
            (
                "mechanism_config_sha256",
                lambda item: item["provenance"]["mechanism_config_sha256"],
            ),
            (
                "calibration_provenance",
                lambda item: item["provenance"]["calibration_provenance"],
            ),
        )
    for label, getter in invariants:
        expected = getter(first)
        if any(getter(record) != expected for record in records[1:]):
            raise ValueError(f"sample provenance mismatch: {label}")

    baseline_quality = {
        metric: _mean(records, ("quality", "baseline", metric)) for metric in QUALITY_METRICS
    }
    scarf_quality = {
        metric: _mean(records, ("quality", "scarf", metric)) for metric in QUALITY_METRICS
    }
    baseline_cycles = _mean(records, ("performance", "baseline_cycles"))
    scarf_cycles = _mean(records, ("performance", "scarf_cycles"))
    components = {
        key: _mean(records, ("performance", "components", key))
        for key in first["performance"]["components"]
    }

    provenance = copy.deepcopy(first["provenance"])
    provenance.pop("execution_trace", None)
    provenance.pop("execution_trace_sha256", None)
    provenance["command"] = portable_command([sys.executable, *sys.argv])
    provenance["evaluation"] = {
        "kind": "dataset_aggregate",
        "sample_count": expected_count,
        "sample_indices": sorted(indices),
        "execution_indices": sorted(execution_indices),
        "sample_selection": selections,
        "sample_selection_sha256": selection_sha256,
        "aggregation": "arithmetic mean over deterministic sample indices",
        "sample_results": sorted(evidence, key=lambda item: item["sample_index"]),
    }
    if trace_set is not None:
        provenance["evaluation"].update(
            {
                "execution_trace_set_schema_version": EXECUTION_TRACE_SET_SCHEMA_VERSION,
                "execution_trace_set": trace_set,
            }
        )
    record = {
        "schema_version": schema_version,
        "provenance": provenance,
        "quality": build_quality_record(baseline_quality, scarf_quality),
        "performance": {
            "baseline_cycles": baseline_cycles,
            "scarf_cycles": scarf_cycles,
            "speedup": baseline_cycles / scarf_cycles,
            "baseline_source": first["performance"]["baseline_source"],
            "cycle_source": first["performance"]["cycle_source"] + ":sample_mean",
            "components": components,
        },
        "ablation": _mean_tree([item["ablation"] for item in records]),
        "fsdr_saes": _aggregate_fsdr_saes(
            [item["fsdr_saes"] for item in records]
        ),
        "hardware": copy.deepcopy(first["hardware"]),
        "validation": {
            "reproducible": True,
            "reference_fallback_used": False,
            "dataset_aggregate": True,
            "complete_sample_set": True,
            "dispersion": {
                f"{variant}.{metric}": _dispersion(records, ("quality", variant, metric))
                for variant in ("baseline", "scarf")
                for metric in QUALITY_METRICS
            },
        },
    }
    if schema_version in {"2.0", "2.1"}:
        evidence = _aggregate_v2_evidence(records)
        record["evidence_class"] = first["evidence_class"]
        record["performance"]["stages"] = evidence["stages"]
        record["events"] = evidence["events"]
        record["energy"] = _mean_tree([item["energy"] for item in records])
    if trace_set is not None:
        aggregate_performance_evidence = (
            execution_trace_performance_evidence_from_record(record)
        )
        trace_set_sha256 = execution_trace_set_sha256(
            trace_set, aggregate_performance_evidence
        )
        record["provenance"]["evaluation"].update(
            {
                "execution_trace_performance_evidence": aggregate_performance_evidence,
                "execution_trace_set_sha256": trace_set_sha256,
            }
        )
        record["quality"]["execution_trace_set_sha256"] = trace_set_sha256
        record["performance"]["execution_trace_set_sha256"] = trace_set_sha256
    validate(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = list(args.input_dir.glob("sample_*/results.json"))
        record = aggregate(paths, args.expected_count)
        write_result(record, args.output)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
