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

from scripts.result_record import build_quality_record, portable_command, write_result
from scripts.validate_result import validate


QUALITY_METRICS = ("psnr_db", "ssim", "lpips")


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

    first = records[0]
    invariants = (
        ("model", lambda item: item["provenance"]["model"]),
        ("dataset", lambda item: item["provenance"]["dataset"]),
        ("checkpoint", lambda item: item["provenance"]["checkpoint"]),
        ("git_commit", lambda item: item["provenance"]["git_commit"]),
        ("git_dirty", lambda item: item["provenance"]["git_dirty"]),
        ("source_identity", lambda item: item["provenance"]["source_identity"]),
        ("submodules", lambda item: item["provenance"]["submodules"]),
        ("runtime_assets", lambda item: item["provenance"]["runtime_assets"]),
        ("environment", lambda item: item["provenance"]["environment"]),
        ("cycle_source", lambda item: item["performance"]["cycle_source"]),
        ("baseline_source", lambda item: item["performance"]["baseline_source"]),
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
    record = {
        "schema_version": "1.0",
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
        "fsdr_saes": _mean_tree([item["fsdr_saes"] for item in records]),
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
