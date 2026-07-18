#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Non-interactive DeepScaleTool-compatible technology normalization."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


WORKBOOK_SHA256 = "561a3f8f5e91a3c496d6e0f4262c09412209f3bbc6d22b714cde323ebd958df8"
WORKBOOK_PATH = Path(__file__).resolve().parent / "vendor/DeepScaleTool.xlsm"
SUPPORTED_NODES = (130, 90, 65, 45, 40, 32, 28, 22, 14, 10, 7)

# Published workbook values, normalized to the reference node used by each
# metric. A factor is current[node] / target[node].
TABLES: dict[str, dict[int, float]] = {
    "area": dict(zip(SUPPORTED_NODES, (8.3, 3.94, 2.04, 1.0, 0.75, 0.49, 0.35, 0.22, 0.08, 0.03, 0.011))),
    "delay": dict(zip(SUPPORTED_NODES, (1.96, 1.31, 1.0, 0.81, 0.76, 0.70, 0.67, 0.62, 0.60, 0.57, 0.53))),
    "energy": dict(zip(SUPPORTED_NODES, (2.52, 1.51, 1.0, 0.63, 0.55, 0.44, 0.37, 0.30, 0.19, 0.15, 0.11))),
    "power": dict(zip(SUPPORTED_NODES, (1.28, 1.15, 1.0, 0.78, 0.73, 0.63, 0.56, 0.48, 0.32, 0.26, 0.21))),
    "throughput": dict(zip(SUPPORTED_NODES, (0.51, 0.76, 1.0, 1.23, 1.32, 1.43, 1.49, 1.61, 1.67, 1.75, 1.89))),
    "throughput_per_area": dict(zip(SUPPORTED_NODES, (0.06, 0.19, 0.49, 1.23, 1.76, 2.92, 4.25, 7.33, 20.83, 58.18, 188.59))),
}

FIELD_METRICS = {
    "area_mm2": "area",
    "cell_area_mm2": "area",
    "logic_area_mm2": "area",
    "sram_proxy_area_mm2": "area",
    "routing_filler_area_mm2": "area",
    "delay_ns": "delay",
    "critical_path_ns": "delay",
    "power_w": "power",
    "total_power_w": "power",
    "dynamic_power_w": "power",
    "static_power_w": "power",
    "energy_mj": "energy",
    "throughput_ips": "throughput",
    "max_frequency_mhz": "throughput",
    "throughput_per_area": "throughput_per_area",
}

MODEL_NOTES = {
    "area": "The cited TSMC comparison reports approximately 1% error.",
    "delay": "The cited TSMC comparison reports approximately 2.5% error.",
    "power": "The cited TSMC comparison reports approximately 5% error.",
    "energy": "No equivalent validated error bound is claimed for energy.",
    "throughput": "Derived from the DeepScaleTool throughput table.",
    "throughput_per_area": "Taken directly from the DeepScaleTool table.",
}

STANDARD_PROVENANCE_FIELDS = (
    "git_commit",
    "git_dirty",
    "source_identity",
    "source_tree_sha256",
    "submodules",
    "mechanism_config_sha256",
    "calibration_provenance",
)


def canonical_sha256(value: Any) -> str:
    """Return the stable JSON digest used to bind derived evidence."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _raw_provenance(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete standard provenance copied into a DeepScale result."""
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("raw ASAP7 PPA has no provenance object")
    missing = [field for field in STANDARD_PROVENANCE_FIELDS if field not in provenance]
    if missing:
        raise ValueError(
            "raw ASAP7 PPA provenance is missing " + ", ".join(missing)
        )
    if not _is_sha256(provenance["source_tree_sha256"]):
        raise ValueError("raw ASAP7 PPA has an invalid source_tree_sha256")
    if not _is_sha256(provenance["mechanism_config_sha256"]):
        raise ValueError("raw ASAP7 PPA has an invalid mechanism_config_sha256")
    if not isinstance(provenance["calibration_provenance"], Mapping):
        raise ValueError("raw ASAP7 PPA has invalid calibration provenance")
    return copy.deepcopy(dict(provenance))


def raw_asap7_input_binding(
    raw: Mapping[str, Any], raw_input_sha256: str
) -> dict[str, Any]:
    """Build the immutable identity recorded by a DeepScale derived PPA."""
    if not _is_sha256(raw_input_sha256):
        raise ValueError("raw ASAP7 input SHA256 must be a lowercase SHA256 digest")
    provenance = _raw_provenance(raw)
    return {
        "sha256": raw_input_sha256,
        "evidence_type": raw.get("evidence_type"),
        "physical_valid": raw.get("physical_valid"),
        "provenance_sha256": canonical_sha256(provenance),
        "source_tree_sha256": provenance["source_tree_sha256"],
        "mechanism_config_sha256": provenance["mechanism_config_sha256"],
        "calibration_provenance_sha256": canonical_sha256(
            provenance["calibration_provenance"]
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_workbook(path: Path = WORKBOOK_PATH) -> str:
    if not path.is_file():
        raise ValueError(f"pinned DeepScaleTool workbook is missing: {path}")
    actual = sha256_file(path)
    if actual != WORKBOOK_SHA256:
        raise ValueError(
            f"DeepScaleTool workbook SHA256 mismatch: expected {WORKBOOK_SHA256}, got {actual}"
        )
    return actual


def _metric_name(metric_or_field: str) -> str:
    metric = FIELD_METRICS.get(metric_or_field, metric_or_field)
    if metric not in TABLES:
        raise ValueError(f"unsupported metric: {metric_or_field}")
    return metric


def scaling_factor(metric: str, source_node: int, target_node: int) -> float:
    """Return DeepScaleTool's source/target factor for a metric."""
    metric = _metric_name(metric)
    for node in (source_node, target_node):
        if node not in SUPPORTED_NODES:
            raise ValueError(f"unsupported technology node: {node}")
    return TABLES[metric][source_node] / TABLES[metric][target_node]


def scale_value(metric: str, value: float, source_node: int, target_node: int) -> float:
    """Normalize a value using target = source / scaling_factor."""
    return float(value) / scaling_factor(metric, source_node, target_node)


def validate_scaled_record(
    scaled: Mapping[str, Any],
    raw: Mapping[str, Any],
    *,
    raw_input_sha256: str,
    source_node: int = 7,
    target_node: int = 28,
) -> None:
    """Validate that one estimate is derived from the supplied raw PPA record.

    The raw record's full provenance is intentionally copied, while the separate
    binding ties that copied record to the exact input bytes used for scaling.
    """
    if not isinstance(scaled, Mapping):
        raise ValueError("DeepScale result must be a JSON object")
    if scaled.get("raw") != raw:
        raise ValueError("DeepScale result does not preserve the supplied raw ASAP7 PPA")
    raw_provenance = _raw_provenance(raw)
    if scaled.get("provenance") != raw_provenance:
        raise ValueError("DeepScale result provenance does not match the raw ASAP7 PPA")
    if scaled.get("raw_asap7_input") != raw_asap7_input_binding(
        raw, raw_input_sha256
    ):
        raise ValueError("DeepScale result raw ASAP7 input binding does not match")
    scaling = scaled.get("scaling")
    if not isinstance(scaling, Mapping):
        raise ValueError("DeepScale result has no scaling metadata")
    if (
        scaling.get("tool") != "DeepScaleTool"
        or scaling.get("workbook_sha256") != WORKBOOK_SHA256
        or scaling.get("source_node_nm") != source_node
        or scaling.get("target_node_nm") != target_node
    ):
        raise ValueError("DeepScale result scaling metadata does not match the requested nodes")
    expected_type = (
        "28nm_equivalent_estimate"
        if target_node == 28
        else "technology_equivalent_estimate"
    )
    if scaled.get("evidence_type") != expected_type:
        raise ValueError("DeepScale result has an invalid evidence type")
    if scaled.get("validation") != {
        "raw_preserved": True,
        "foundry_measurement": False,
    }:
        raise ValueError("DeepScale result has an invalid validation record")


def scale_record(
    raw: dict[str, Any],
    source_node: int,
    target_node: int,
    *,
    raw_input_sha256: str,
) -> dict[str, Any]:
    verify_workbook()
    if raw.get("physical_valid") is not True:
        raise ValueError("physical_valid must be true before technology normalization")
    if source_node == 7 and raw.get("evidence_type") != "asap7_predictive_postroute":
        raise ValueError("7 nm DeepScale input must be an ASAP7 predictive PPA record")
    raw_provenance = _raw_provenance(raw)
    raw_binding = raw_asap7_input_binding(raw, raw_input_sha256)
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a non-empty object")

    scaled_metrics: dict[str, float] = {}
    factors: dict[str, float] = {}
    notes: dict[str, str] = {}
    for field, value in metrics.items():
        if field not in FIELD_METRICS:
            continue
        if not isinstance(value, (int, float)):
            raise ValueError(f"metric {field} must be numeric")
        metric = FIELD_METRICS[field]
        scaled_metrics[field] = scale_value(metric, value, source_node, target_node)
        factors[field] = scaling_factor(metric, source_node, target_node)
        notes[field] = MODEL_NOTES[metric]

    if not scaled_metrics:
        raise ValueError("metrics contains no scalable PPA fields")

    scaled_hierarchy: dict[str, dict[str, float | None]] = {}
    hierarchy = metrics.get("hierarchy")
    if hierarchy is not None:
        if not isinstance(hierarchy, dict) or not hierarchy:
            raise ValueError("metrics.hierarchy must be a non-empty object")
        for component, component_metrics in hierarchy.items():
            if not isinstance(component, str) or not component or not isinstance(
                component_metrics, dict
            ):
                raise ValueError("hierarchical PPA component is invalid")
            scaled_component: dict[str, float | None] = {}
            for field, value in component_metrics.items():
                if value is None:
                    scaled_component[field] = None
                    continue
                if field not in FIELD_METRICS or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"unsupported hierarchical PPA field: {component}.{field}"
                    )
                scaled_component[field] = scale_value(
                    FIELD_METRICS[field], value, source_node, target_node
                )
            scaled_hierarchy[component] = scaled_component

    scaled = {
        "schema_version": "1.0",
        "evidence_type": "28nm_equivalent_estimate" if target_node == 28 else "technology_equivalent_estimate",
        "source_process": f"ASAP7 predictive {source_node} nm" if source_node == 7 else f"{source_node} nm",
        "target_process": f"{target_node} nm equivalent",
        "raw": copy.deepcopy(raw),
        "raw_asap7_input": raw_binding,
        "provenance": raw_provenance,
        "scaled_metrics": scaled_metrics,
        "scaled_hierarchy": scaled_hierarchy,
        "scaling": {
            "tool": "DeepScaleTool",
            "workbook_sha256": WORKBOOK_SHA256,
            "source_node_nm": source_node,
            "target_node_nm": target_node,
            "formula": "target_value = source_value / factor",
            "factors": factors,
            "model_notes": notes,
            "cross_node_uncertainty": "Model error may accumulate across technology nodes.",
        },
        "validation": {"raw_preserved": True, "foundry_measurement": False},
    }
    validate_scaled_record(
        scaled,
        raw,
        raw_input_sha256=raw_input_sha256,
        source_node=source_node,
        target_node=target_node,
    )
    return scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-node", type=int, required=True)
    parser.add_argument("--target-node", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        raw_bytes = args.input.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
        scaled = scale_record(
            raw,
            args.source_node,
            args.target_node,
            raw_input_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(scaled, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
