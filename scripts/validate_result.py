#!/usr/bin/env python3
"""Validate one SCARF AE results.json record."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.execution_contract import (  # noqa: E402
    execution_contract,
    require_paper_execution_contract,
)
from scripts.result_record import (  # noqa: E402
    EXECUTION_TRACE_SCHEMA_VERSION,
    EXECUTION_TRACE_SET_SCHEMA_VERSION,
    execution_trace_inputs_from_record,
    execution_trace_performance_evidence_from_record,
    execution_trace_set_sha256,
    execution_trace_sha256,
)


AGGREGATE_EXECUTION_BINDING_FIELDS = (
    "mechanism_config_sha256",
    "checkpoint_sha256",
    "sample_selection_sha256",
    "execution_trace_set_sha256",
)
REFERENCE_ONLY_MARKER = "PAPER_REFERENCE_ONLY"


def _reference_only_marker_path(value: Any, path: str = "result") -> str | None:
    """Return the first JSON location carrying the reference-only sentinel."""
    if value == REFERENCE_ONLY_MARKER:
        return path
    if isinstance(value, dict):
        for key, child in value.items():
            marker_path = _reference_only_marker_path(child, f"{path}.{key}")
            if marker_path is not None:
                return marker_path
    elif isinstance(value, list):
        for index, child in enumerate(value):
            marker_path = _reference_only_marker_path(child, f"{path}[{index}]")
            if marker_path is not None:
                return marker_path
    return None


def reject_reference_only_record(record: Mapping[str, Any]) -> None:
    """Keep paper-reference artifacts out of every generated-evidence path."""
    marker_path = _reference_only_marker_path(record)
    if marker_path is not None:
        raise ValueError(
            "PAPER_REFERENCE_ONLY marker cannot be used as generated evidence: "
            f"{marker_path}"
        )


def aggregate_execution_binding_values(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the four bindings required to combine claim-tier aggregates."""
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    checkpoint = provenance.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        checkpoint = {}
    evaluation = provenance.get("evaluation")
    if not isinstance(evaluation, Mapping):
        evaluation = {}
    return {
        "mechanism_config_sha256": provenance.get("mechanism_config_sha256"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "sample_selection_sha256": evaluation.get("sample_selection_sha256"),
        "execution_trace_set_sha256": evaluation.get("execution_trace_set_sha256"),
    }


def aggregate_execution_binding_mismatches(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return missing or unequal claim-combination bindings for two aggregates."""
    first_values = aggregate_execution_binding_values(first)
    second_values = aggregate_execution_binding_values(second)
    return {
        field: {"first": first_values[field], "second": second_values[field]}
        for field in AGGREGATE_EXECUTION_BINDING_FIELDS
        if first_values[field] is None
        or second_values[field] is None
        or first_values[field] != second_values[field]
    }


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


def _finite_count(record: dict[str, Any], path: str) -> float:
    value = _get(record, path)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{path} must be a nonnegative finite count")
    return float(value)


def _validate_v2_evidence(record: dict[str, Any]) -> None:
    if not _sha256_digest(_get(record, "provenance.source_tree_sha256")):
        raise ValueError("provenance.source_tree_sha256 must be a SHA256 digest")
    evidence_class = _get(record, "evidence_class")
    if evidence_class not in {
        "independent_measurement",
        "deterministic_execution",
        "public_physical_proxy",
        "paper_comparison_target",
    }:
        raise ValueError("evidence_class is invalid")

    stages = _get(record, "performance.stages")
    if not isinstance(stages, dict) or set(stages) != {"s1", "s2", "s3", "s4"}:
        raise ValueError("performance.stages must contain exactly s1-s4")
    for stage in ("s1", "s2", "s3", "s4"):
        cycles = _finite_count(record, f"performance.stages.{stage}.cycles")
        useful = _finite_count(
            record, f"performance.stages.{stage}.useful_mmcu_slots"
        )
        scheduled = _finite_count(
            record, f"performance.stages.{stage}.scheduled_mmcu_slots"
        )
        available = _get(
            record, f"performance.stages.{stage}.mmcu_slots_available"
        )
        source = _get(record, f"performance.stages.{stage}.source")
        if cycles <= 0:
            raise ValueError(f"performance.stages.{stage}.cycles must be positive")
        if not isinstance(available, bool):
            raise ValueError(
                f"performance.stages.{stage}.mmcu_slots_available must be boolean"
            )
        if not isinstance(source, str) or not source:
            raise ValueError(f"performance.stages.{stage}.source must be non-empty")
        if not available and (useful != 0 or scheduled != 0):
            raise ValueError(
                f"performance.stages.{stage} unavailable MMCU evidence has counts"
            )
        if available and scheduled <= 0:
            raise ValueError(
                f"performance.stages.{stage} available MMCU evidence has no slots"
            )
        if useful > scheduled:
            raise ValueError(
                f"performance.stages.{stage} useful MMCU slots exceed scheduled slots"
            )

    for namespace in ("fsdr", "saes"):
        if not isinstance(_get(record, f"events.{namespace}"), dict):
            raise ValueError(f"events.{namespace} must be an object")

    fsdr_fields = (
        "total_pixels",
        "cache_hits",
        "guided_pixels",
        "guided_top1_covered",
        "guided_top1_missed",
        "full_depth_evaluations",
        "executed_depth_evaluations",
        "feature_buffer_bytes_baseline",
        "feature_buffer_bytes_actual",
    )
    fsdr = {
        field: _finite_count(record, f"events.fsdr.{field}")
        for field in fsdr_fields
    }
    if fsdr["cache_hits"] > fsdr["total_pixels"]:
        raise ValueError("events.fsdr cache hits exceed total pixels")
    if fsdr["guided_pixels"] > fsdr["cache_hits"]:
        raise ValueError("events.fsdr guided pixels exceed cache hits")
    discrete_available = _get(record, "events.fsdr").get(
        "discrete_top1_available", True
    )
    if not isinstance(discrete_available, bool):
        raise ValueError("events.fsdr.discrete_top1_available must be boolean")
    discrete_total = fsdr["guided_top1_covered"] + fsdr["guided_top1_missed"]
    if discrete_available and discrete_total != fsdr["guided_pixels"]:
        raise ValueError("events.fsdr discrete Top-1 counts do not cover guided pixels")
    if not discrete_available and discrete_total != 0:
        raise ValueError("events.fsdr unavailable Top-1 evidence has nonzero counts")
    for flag, fields in (
        (
            "depth_evaluations_available",
            ("full_depth_evaluations", "executed_depth_evaluations"),
        ),
        (
            "feature_buffer_bytes_available",
            ("feature_buffer_bytes_baseline", "feature_buffer_bytes_actual"),
        ),
    ):
        available = _get(record, "events.fsdr").get(flag)
        if not isinstance(available, bool):
            raise ValueError(f"events.fsdr.{flag} must be boolean")
        if not available and any(fsdr[field] != 0 for field in fields):
            raise ValueError(f"events.fsdr unavailable {flag} evidence has counts")
    if fsdr["executed_depth_evaluations"] > fsdr["full_depth_evaluations"]:
        raise ValueError("events.fsdr executed depth evaluations exceed full search")
    if fsdr["feature_buffer_bytes_actual"] > fsdr["feature_buffer_bytes_baseline"]:
        raise ValueError("events.fsdr actual feature-buffer bytes exceed baseline")

    saes_fields = (
        "total_tiles",
        "level0_tiles",
        "level1_tiles",
        "full_tiles",
        "baseline_gaussians",
        "actual_gaussians",
        "full_s2_evaluations",
        "executed_s2_evaluations",
    )
    saes = {
        field: _finite_count(record, f"events.saes.{field}")
        for field in saes_fields
    }
    if saes["level0_tiles"] + saes["level1_tiles"] + saes["full_tiles"] != saes[
        "total_tiles"
    ]:
        raise ValueError("events.saes tile-path counts do not cover total tiles")
    if saes["actual_gaussians"] > saes["baseline_gaussians"]:
        raise ValueError("events.saes actual Gaussians exceed baseline")
    if saes["executed_s2_evaluations"] > saes["full_s2_evaluations"]:
        raise ValueError("events.saes executed S2 evaluations exceed full path")
    for flag, fields in (
        (
            "tile_path_available",
            ("total_tiles", "level0_tiles", "level1_tiles", "full_tiles"),
        ),
        (
            "gaussian_counts_available",
            ("baseline_gaussians", "actual_gaussians"),
        ),
        (
            "s2_evaluations_available",
            ("full_s2_evaluations", "executed_s2_evaluations"),
        ),
    ):
        available = _get(record, "events.saes").get(flag)
        if not isinstance(available, bool):
            raise ValueError(f"events.saes.{flag} must be boolean")
        if not available and any(saes[field] != 0 for field in fields):
            raise ValueError(f"events.saes unavailable {flag} evidence has counts")

    for namespace in ("fsdr", "saes"):
        source = _get(record, f"events.{namespace}.source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"events.{namespace}.source must be non-empty")

    energy = _get(record, "energy")
    if not isinstance(energy, dict):
        raise ValueError("energy must be an object")
    energy_available = energy.get("available")
    energy_source = energy.get("source")
    if not isinstance(energy_available, bool):
        raise ValueError("energy.available must be boolean")
    if not isinstance(energy_source, str) or not energy_source:
        raise ValueError("energy.source must be non-empty")
    if energy_available:
        total = _positive(record, "energy.total_joules")
        components = _get(record, "energy.components_joules")
        if not isinstance(components, dict) or not components:
            raise ValueError("energy.components_joules must be a non-empty object")
        component_total = 0.0
        for name, value in components.items():
            if not isinstance(name, str) or not name:
                raise ValueError("energy component names must be non-empty")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("energy component values must be nonnegative")
            component_total += float(value)
        if not math.isclose(total, component_total, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError("energy.total_joules does not equal component sum")


def _validate_v21_evidence(record: dict[str, Any]) -> None:
    from saes.execution_dependency import resolve_s2_s3_execution_contract

    mechanism_digest = _get(record, "provenance.mechanism_config_sha256")
    if not _sha256_digest(mechanism_digest):
        raise ValueError("provenance.mechanism_config_sha256 must be a SHA256 digest")
    calibration = _get(record, "provenance.calibration_provenance")
    if not isinstance(calibration, dict):
        raise ValueError("provenance.calibration_provenance must be an object")
    if not _sha256_digest(calibration.get("manifest_sha256")):
        raise ValueError("calibration_provenance.manifest_sha256 must be a SHA256 digest")
    status = calibration.get("status")
    if status not in {"calibrated", "preregistered"}:
        raise ValueError("calibration_provenance.status is invalid")
    if status == "calibrated":
        if not _sha256_digest(calibration.get("candidate_records_sha256")):
            raise ValueError(
                "calibration_provenance.candidate_records_sha256 must be a SHA256 digest"
            )
        if calibration.get("evaluation_disjoint") is not True:
            raise ValueError("calibration provenance is not evaluation-disjoint")
        if calibration.get("protocol") not in {
            "dl3dv_train_holdout_v1",
            "acid_train_holdout_v1",
        }:
            raise ValueError("calibrated result has no supported train/holdout protocol")
        if calibration.get("train_holdout_scene_disjoint") is not True:
            raise ValueError("calibrated result train/holdout split is not disjoint")
        split_hashes = (
            "selection_sha256",
            "scene_set_sha256",
            "pair_bindings_sha256",
            "trace_set_sha256",
            "candidate_set_sha256",
        )
        split_records: dict[str, dict[str, Any]] = {}
        for split in ("train", "holdout"):
            split_record = calibration.get(split)
            if not isinstance(split_record, dict):
                raise ValueError(f"calibrated result has no {split} calibration evidence")
            for field in split_hashes:
                if not _sha256_digest(split_record.get(field)):
                    raise ValueError(
                        f"calibration_provenance.{split}.{field} must be a SHA256 digest"
                    )
            if split == "train":
                if not _sha256_digest(split_record.get("selected_candidate_sha256")):
                    raise ValueError(
                        "calibration_provenance.train.selected_candidate_sha256 "
                        "must be a SHA256 digest"
                    )
            else:
                for field in (
                    "validated_parameters_sha256",
                    "validated_candidate_sha256",
                ):
                    if not _sha256_digest(split_record.get(field)):
                        raise ValueError(
                            f"calibration_provenance.holdout.{field} must be a SHA256 digest"
                        )
            split_records[split] = split_record
        if (
            split_records["train"].get("selection_sha256")
            == split_records["holdout"].get("selection_sha256")
        ):
            raise ValueError("calibrated result reuses one train/holdout selection")
    else:
        if calibration.get("candidate_records_sha256") is not None:
            raise ValueError("preregistered calibration has candidate evidence")
        if calibration.get("evaluation_disjoint") is not False:
            raise ValueError("preregistered calibration has invalid disjoint status")
        if _get(record, "provenance.dataset.paper_result_eligible") is not False:
            raise ValueError("preregistered calibration cannot support paper evidence")
    if calibration.get("expected_results_accessed") is not False:
        raise ValueError("calibration provenance reports target-result access")
    if calibration.get("global_configuration") is not True:
        raise ValueError("calibration provenance is not one global configuration")

    hamming_hits = _finite_count(record, "events.fsdr.hamming_hits")
    local_valid = _finite_count(record, "events.fsdr.local_valid_hits")
    local_fallbacks = _finite_count(record, "events.fsdr.local_invalid_fallbacks")
    cache_hits = _finite_count(record, "events.fsdr.cache_hits")
    guided = _finite_count(record, "events.fsdr.guided_pixels")
    if hamming_hits != cache_hits:
        raise ValueError("events.fsdr Hamming hits do not match cache hits")
    if local_valid + local_fallbacks != hamming_hits:
        raise ValueError("events.fsdr local validity counts do not cover Hamming hits")
    if local_valid != guided:
        raise ValueError("events.fsdr local-valid hits do not match guided pixels")

    for field in (
        "l0_representatives",
        "l1_lightweight_anchors",
        "full_stage3_gaussians",
        "covariance_psd_violations",
    ):
        _finite_count(record, f"events.saes.{field}")
    for field in (
        "assignment_weight_sum_error_max",
        "opacity_transmittance_error_max",
    ):
        value = _get(record, f"events.saes.{field}")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"events.saes.{field} must be finite and nonnegative")
    if _get(record, "events.saes.covariance_psd_violations") != 0:
        raise ValueError("events.saes covariance PSD violations must be zero")
    model = _get(record, "provenance.model")
    if not isinstance(model, str) or not model:
        raise ValueError("provenance.model must be a non-empty string")
    expected_dependency = resolve_s2_s3_execution_contract(model)
    execution_dependency = _get(record, "events.saes.execution_dependency")
    if execution_dependency != expected_dependency:
        raise ValueError("SAES execution dependency does not match the model contract")
    if (
        _get(record, "provenance.dataset.paper_result_eligible") is True
        and expected_dependency["s2_s3_sparse_execution_verified"] is not True
    ):
        raise ValueError(
            "paper-result-eligible records require verified whole-pipeline "
            "SAES S2/S3 sparse execution"
        )
    saving = _get(record, "events.saes.s2_s3_saving")
    if not isinstance(saving, dict) or set(saving) != {"s2", "s3"}:
        raise ValueError("events.saes.s2_s3_saving must contain S2 and S3")
    for stage in ("s2", "s3"):
        value = saving[stage]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                f"events.saes.s2_s3_saving.{stage} must be a finite fraction"
            )
        if (
            expected_dependency["s2_s3_sparse_execution_verified"] is not True
            and float(value) != 0.0
        ):
            raise ValueError(
                "unverified SAES execution dependency cannot claim S2/S3 savings"
            )
    hardware_accounting = _get(record, "events.saes").get("hardware_accounting")
    if hardware_accounting is not None:
        if not isinstance(hardware_accounting, dict):
            raise ValueError("events.saes.hardware_accounting must be an object")
        if hardware_accounting.get("timing_class") != (
            "analytic_no_overlap_not_rtl_cycle_equivalent"
        ):
            raise ValueError("SAES hardware accounting timing class is invalid")
        if hardware_accounting.get("target_rgb_accessed") is not False:
            raise ValueError("SAES hardware accounting accessed target RGB")
        cycles = hardware_accounting.get("cycles")
        if not isinstance(cycles, dict):
            raise ValueError("SAES hardware accounting cycles must be an object")
        serialized_cycles = cycles.get("serialized_accounting_cycles")
        if (
            isinstance(serialized_cycles, bool)
            or not isinstance(serialized_cycles, (int, float))
            or not math.isfinite(serialized_cycles)
            or serialized_cycles < 0
            or int(serialized_cycles) != serialized_cycles
        ):
            raise ValueError(
                "SAES hardware accounting serialized cycles must be a nonnegative integer"
            )
        assumptions = hardware_accounting.get("assumptions")
        if not isinstance(assumptions, dict) or assumptions.get(
            "rtl_cycle_equivalent"
        ) is not False:
            raise ValueError(
                "SAES hardware accounting must not claim RTL-cycle equivalence"
            )


def _validate_execution_trace_binding(record: dict[str, Any]) -> None:
    """Verify optional-by-presence non-quality trace bindings."""
    provenance = _get(record, "provenance")
    quality = _get(record, "quality")
    performance = _get(record, "performance")
    evaluation = _get(record, "provenance.evaluation")
    if (
        not isinstance(provenance, dict)
        or not isinstance(quality, dict)
        or not isinstance(performance, dict)
        or not isinstance(evaluation, dict)
    ):
        raise ValueError("execution trace binding requires object result sections")

    kind = evaluation.get("kind")
    if kind == "sample":
        fields_present = any(
            field in provenance
            for field in ("execution_trace", "execution_trace_sha256")
        ) or any(
            field in quality for field in ("execution_trace_sha256",)
        ) or any(field in performance for field in ("execution_trace_sha256",))
        if not fields_present:
            return
        trace = provenance.get("execution_trace")
        digest = provenance.get("execution_trace_sha256")
        if (
            not isinstance(trace, dict)
            or set(trace) != {"schema_version", "inputs"}
            or trace.get("schema_version") != EXECUTION_TRACE_SCHEMA_VERSION
            or not isinstance(trace.get("inputs"), dict)
            or not _sha256_digest(digest)
        ):
            raise ValueError("sample execution trace binding is invalid")
        expected_inputs = execution_trace_inputs_from_record(record)
        if trace["inputs"] != expected_inputs:
            raise ValueError("sample execution trace inputs do not match the record")
        if digest != execution_trace_sha256(trace["inputs"]):
            raise ValueError("sample execution trace hash mismatch")
        if quality.get("execution_trace_sha256") != digest:
            raise ValueError("quality execution trace binding does not match provenance")
        if performance.get("execution_trace_sha256") != digest:
            raise ValueError(
                "performance execution trace binding does not match provenance"
            )
        return

    if kind != "dataset_aggregate":
        return
    fields_present = any(
        field in provenance
        for field in ("execution_trace", "execution_trace_sha256")
    ) or any(
        field in evaluation
        for field in (
            "execution_trace_set_schema_version",
            "execution_trace_set",
            "execution_trace_performance_evidence",
            "execution_trace_set_sha256",
        )
    ) or any(
        field in quality for field in ("execution_trace_set_sha256",)
    ) or any(field in performance for field in ("execution_trace_set_sha256",))
    if not fields_present:
        return
    if "execution_trace" in provenance or "execution_trace_sha256" in provenance:
        raise ValueError("dataset aggregate must not retain a sample execution trace")
    trace_set = evaluation.get("execution_trace_set")
    trace_performance_evidence = evaluation.get(
        "execution_trace_performance_evidence"
    )
    digest = evaluation.get("execution_trace_set_sha256")
    if (
        evaluation.get("execution_trace_set_schema_version")
        != EXECUTION_TRACE_SET_SCHEMA_VERSION
        or not isinstance(trace_set, list)
        or not isinstance(trace_performance_evidence, dict)
        or not _sha256_digest(digest)
    ):
        raise ValueError("dataset execution trace set binding is invalid")
    count = evaluation.get("sample_count")
    indices = evaluation.get("sample_indices")
    execution_indices = evaluation.get("execution_indices")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or not isinstance(indices, list)
        or not isinstance(execution_indices, list)
        or len(trace_set) != count
    ):
        raise ValueError("dataset execution trace set is incomplete")
    trace_indices: list[int] = []
    trace_execution_indices: list[int] = []
    trace_by_index: dict[int, str] = {}
    trace_measurements_by_index: dict[int, str] = {}
    for entry in trace_set:
        required_entry_fields = {
            "sample_index",
            "execution_index",
            "execution_trace_sha256",
        }
        optional_measurement_field = "orin_measurement_sha256"
        if (
            not isinstance(entry, dict)
            or set(entry) not in (
                required_entry_fields,
                required_entry_fields | {optional_measurement_field},
            )
        ):
            raise ValueError("dataset execution trace set entry is invalid")
        sample_index = entry["sample_index"]
        execution_index = entry["execution_index"]
        trace_digest = entry["execution_trace_sha256"]
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
            or isinstance(execution_index, bool)
            or not isinstance(execution_index, int)
            or execution_index < 0
            or not _sha256_digest(trace_digest)
        ):
            raise ValueError("dataset execution trace set entry has invalid values")
        trace_indices.append(sample_index)
        trace_execution_indices.append(execution_index)
        trace_by_index[sample_index] = trace_digest
        if optional_measurement_field in entry:
            measurement_digest = entry[optional_measurement_field]
            if not _sha256_digest(measurement_digest):
                raise ValueError("dataset Orin measurement trace hash is invalid")
            trace_measurements_by_index[sample_index] = measurement_digest
    if (
        len(trace_by_index) != count
        or trace_indices != sorted(trace_indices)
        or sorted(trace_indices) != indices
        or sorted(trace_execution_indices) != execution_indices
    ):
        raise ValueError("dataset execution trace set does not match sample selection")
    expected_performance_evidence = execution_trace_performance_evidence_from_record(
        record
    )
    if trace_performance_evidence != expected_performance_evidence:
        raise ValueError(
            "dataset execution trace performance evidence does not match the record"
        )
    if digest != execution_trace_set_sha256(trace_set, trace_performance_evidence):
        raise ValueError("dataset execution trace set hash mismatch")
    sample_results = evaluation.get("sample_results")
    if not isinstance(sample_results, list) or len(sample_results) != count:
        raise ValueError("dataset execution trace set has no complete sample evidence")
    evidence_by_index: dict[int, str] = {}
    evidence_measurements_by_index: dict[int, str] = {}
    for evidence in sample_results:
        if not isinstance(evidence, dict):
            raise ValueError("dataset sample evidence is invalid")
        sample_index = evidence.get("sample_index")
        trace_digest = evidence.get("execution_trace_sha256")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not _sha256_digest(trace_digest)
        ):
            raise ValueError("dataset sample trace evidence is invalid")
        evidence_by_index[sample_index] = trace_digest
        measurement = evidence.get("orin_measurement")
        if measurement is not None:
            if not isinstance(measurement, dict) or not _sha256_digest(
                measurement.get("sha256")
            ):
                raise ValueError("dataset Orin sample evidence is invalid")
            evidence_measurements_by_index[sample_index] = measurement["sha256"]
    if evidence_by_index != trace_by_index:
        raise ValueError("dataset sample trace evidence does not match trace set")
    if evidence_measurements_by_index != trace_measurements_by_index:
        raise ValueError("dataset Orin sample evidence does not match trace set")
    if quality.get("execution_trace_set_sha256") != digest:
        raise ValueError("quality execution trace set binding does not match provenance")
    if performance.get("execution_trace_set_sha256") != digest:
        raise ValueError(
            "performance execution trace set binding does not match provenance"
        )


def _validate_strict_saes_route_binding(record: dict[str, Any]) -> None:
    """Require strict records to carry the runtime route that produced them."""
    contract = execution_contract(record)
    if contract is None or contract["run_class"] not in {"claim", "functional"}:
        return
    provenance = _get(record, "provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("strict SAES route binding requires provenance")
    identity = provenance.get("saes_execution_identity")
    route_sha256 = provenance.get("saes_execution_route_sha256")
    from scripts.saes_execution_identity import validate_saes_execution_identity

    expected = validate_saes_execution_identity(identity)
    if route_sha256 != expected["route_sha256"]:
        raise ValueError("strict SAES route SHA256 does not match its identity")
    fsdr_saes = record.get("fsdr_saes")
    if not isinstance(fsdr_saes, Mapping) or not isinstance(
        fsdr_saes.get("saes"), Mapping
    ):
        raise ValueError("strict SAES route binding requires runtime SAES evidence")
    runtime = fsdr_saes["saes"]
    if runtime.get("saes_execution_identity") != expected:
        raise ValueError("runtime SAES execution identity does not match provenance")
    if runtime.get("route_sha256") != route_sha256:
        raise ValueError("runtime SAES route SHA256 does not match provenance")


def validate(record: dict[str, Any]) -> None:
    reject_reference_only_record(record)
    require_paper_execution_contract(record, surface="result validation")
    schema_version = _get(record, "schema_version")
    if schema_version not in {"1.0", "2.0", "2.1"}:
        raise ValueError("schema_version must be 1.0, 2.0, or 2.1")
    if schema_version in {"2.0", "2.1"}:
        _validate_v2_evidence(record)
    if schema_version == "2.1":
        _validate_v21_evidence(record)
        _validate_strict_saes_route_binding(record)

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
    if functional_fixture and paper_result_eligible:
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

    _validate_execution_trace_binding(record)
    contract = execution_contract(record)
    if contract is not None and contract["run_class"] == "claim":
        workflow = _get(record, "provenance").get("claim_workflow")
        if workflow not in {"quality", "mechanisms", "performance"}:
            raise ValueError(
                "claim result must declare provenance.claim_workflow"
            )
        if workflow != "quality":
            from scripts.result_record import claim_timing_from_record

            claim_timing_from_record(
                record, aggregate=evaluation["kind"] == "dataset_aggregate"
            )

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
