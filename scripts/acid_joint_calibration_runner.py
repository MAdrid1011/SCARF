#!/usr/bin/env python3
"""Train or validate the frozen author-side ACID joint calibrator.

The runner consumes only complete, hash-bound offline teacher caches. It never
opens raw ACID chunks, target RGB/cameras, an evaluation index, or expected
results. A runtime-control evidence record is required because a cache cannot
truthfully prove selected-head replay or its cost ledger by itself.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.acid_joint_training_contract import (
    ACCESS_AUDIT_KIND,
    CANDIDATE_PROVENANCE_KIND,
    CANDIDATE_PROVENANCE_SCHEMA_VERSION,
    CONTRACT_SCHEMA_VERSION,
    DEFAULT_TRAINING_CONTRACT_PATH,
    HOLDOUT_SPLIT,
    MODELS,
    RESULT_KIND,
    RUNTIME_PRECONDITIONS,
    RUNTIME_EVIDENCE_KIND,
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    TEACHER_CACHE_EXAMPLE_KIND,
    TEACHER_CACHE_KIND,
    TEACHER_CACHE_SCHEMA_VERSION,
    TRAIN_SPLIT,
    JointTrainingContractError,
    candidate_provenance_path_for_contract,
    candidate_verification_path_for_contract,
    sha256_file,
    validate_candidate_provenance,
    validate_candidate_verification,
    validate_live_teacher_fidelity_result,
    validate_live_training_contract,
    validate_teacher_fidelity_result,
)
from saes.joint_materialization_calibrator import (
    JointMaterializationCalibrator,
    calibrator_asset_manifest,
    load_calibrator_asset,
)
from saes.hardware_accounting import LEDGER_VERSION
from scripts.calibration_contract import canonical_sha256


CACHE_KIND = TEACHER_CACHE_KIND
CACHE_SCHEMA_VERSION = TEACHER_CACHE_SCHEMA_VERSION
EXAMPLE_KIND = TEACHER_CACHE_EXAMPLE_KIND
METRIC_COMPONENTS = ("mean", "covariance", "opacity", "sh")
SHA256 = set("0123456789abcdef")
_CACHE_CONTROL_REQUIRED = (
    "route_mask_unchanged_required",
    "retained_counts_unchanged_required",
    "full_passthrough_required",
    "selected_only_s3_required",
    "two_finite_skipped_s3_sentinels_required",
    "route_mask_unchanged",
    "retained_counts_unchanged",
    "operational_stats_unchanged",
    "tile_trace_unchanged",
    "ordinary_representative_output_unchanged",
    "full_passthrough",
    "selected_only_s3",
    "two_finite_skipped_s3_sentinels_passed",
)
_ROUTE_SUMMARY_FIELDS = (
    "level0_tiles",
    "level1_tiles",
    "full_tiles",
    "level0_pixels",
    "level1_pixels",
    "l0_representatives",
    "l1_lightweight_anchors",
    "full_stage3_gaussians",
)


class JointCalibrationRunError(RuntimeError):
    """Raised when a cache, runtime proof, or frozen run contract is invalid."""


def _canonical_contract_path() -> Path:
    return ROOT / "artifact" / "protocol" / DEFAULT_TRAINING_CONTRACT_PATH.name


def _require_canonical_path(path: Path, expected: Path, *, label: str) -> Path:
    actual = Path(path).resolve()
    expected = Path(expected).resolve()
    if actual != expected:
        raise JointCalibrationRunError(f"{label} must use the canonical frozen path")
    return actual


def _canonical_cache_root(contract: Mapping[str, Any], *, model: str, split: str) -> Path:
    root = _safe_path(ROOT, contract["teacher_objective"]["teacher_cache_root"], "teacher cache root")
    return root / model / split


def _canonical_runtime_evidence_path(
    contract: Mapping[str, Any], *, model: str, split: str
) -> Path:
    return _safe_path(
        ROOT,
        contract["runtime_evidence_records"][split][model],
        f"{model} runtime evidence",
    )


@dataclass(frozen=True)
class CacheExample:
    model: str
    descriptor: torch.Tensor
    compact: Mapping[str, torch.Tensor]
    teacher: Mapping[str, torch.Tensor]
    sh_degree: int


@dataclass(frozen=True)
class ModelCache:
    model: str
    split: str
    resolved_config_sha256: str
    prepared_patch_size: int
    examples: tuple[CacheExample, ...]
    controls: Mapping[str, bool]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointCalibrationRunError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise JointCalibrationRunError(f"{label} must be an object")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in SHA256 for char in value):
        raise JointCalibrationRunError(f"{label} must be a lowercase SHA256 digest")
    return value


def _safe_path(root: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise JointCalibrationRunError(f"{label} path is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise JointCalibrationRunError(f"{label} path is unsafe")
    resolved = (Path(root).resolve() / Path(*pure.parts)).resolve()
    if Path(root).resolve() not in resolved.parents and resolved != Path(root).resolve():
        raise JointCalibrationRunError(f"{label} path escapes its root")
    return resolved


def _parse_model_paths(values: Sequence[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise JointCalibrationRunError(f"{label} must use model=path")
        model, raw_path = value.split("=", 1)
        if model not in MODELS or not raw_path or model in result:
            raise JointCalibrationRunError(f"{label} has an invalid model binding")
        result[model] = Path(raw_path).resolve()
    if set(result) != set(MODELS):
        raise JointCalibrationRunError(f"{label} must bind exactly {', '.join(MODELS)}")
    return result


def _tensor(value: Any, *, shape: tuple[int, ...], label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape or not value.is_floating_point():
        raise JointCalibrationRunError(f"{label} has an invalid tensor shape")
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise JointCalibrationRunError(f"{label} is non-finite")
    return value


def _sh_degree(harmonics: torch.Tensor) -> int:
    if harmonics.ndim != 2 or harmonics.shape[0] != 3:
        raise JointCalibrationRunError("teacher-cache harmonics are invalid")
    root = math.isqrt(int(harmonics.shape[1]))
    if root * root != harmonics.shape[1] or not 1 <= root <= 5:
        raise JointCalibrationRunError("teacher-cache SH degree is unsupported")
    return root - 1


def _validate_example(value: Any, *, model: str) -> CacheExample:
    required = {"schema_version", "kind", "descriptor", "compact", "teacher", "level", "anchor_index"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointCalibrationRunError("teacher-cache example has an invalid schema")
    if value.get("schema_version") != CACHE_SCHEMA_VERSION or value.get("kind") != EXAMPLE_KIND:
        raise JointCalibrationRunError("teacher-cache example has an invalid identity")
    if value.get("level") not in {"L0", "L1"} or not isinstance(value.get("anchor_index"), int):
        raise JointCalibrationRunError("teacher-cache example has invalid route metadata")
    descriptor = _tensor(value.get("descriptor"), shape=(32,), label="teacher-cache descriptor")
    compact = value.get("compact")
    teacher = value.get("teacher")
    if not isinstance(compact, Mapping) or not isinstance(teacher, Mapping):
        raise JointCalibrationRunError("teacher-cache attributes are invalid")
    required_attributes = {"means", "covariances", "harmonics", "opacities"}
    if set(compact) != required_attributes or set(teacher) != required_attributes:
        raise JointCalibrationRunError("teacher-cache attribute families changed")
    compact_harmonics = compact["harmonics"]
    degree = _sh_degree(compact_harmonics)
    coefficient_count = (degree + 1) ** 2
    compact_values = {
        "means": _tensor(compact["means"], shape=(3,), label="compact means"),
        "covariances": _tensor(compact["covariances"], shape=(3, 3), label="compact covariances"),
        "harmonics": _tensor(compact_harmonics, shape=(3, coefficient_count), label="compact harmonics"),
        "opacities": _tensor(compact["opacities"].reshape(1), shape=(1,), label="compact opacities"),
    }
    teacher_values = {
        "means": _tensor(teacher["means"], shape=(3,), label="teacher means"),
        "covariances": _tensor(teacher["covariances"], shape=(3, 3), label="teacher covariances"),
        "harmonics": _tensor(teacher["harmonics"], shape=(3, coefficient_count), label="teacher harmonics"),
        "opacities": _tensor(teacher["opacities"].reshape(1), shape=(1,), label="teacher opacities"),
    }
    if not bool(torch.allclose(compact_values["covariances"], compact_values["covariances"].mT)):
        raise JointCalibrationRunError("compact covariance is not symmetric")
    if not bool(torch.allclose(teacher_values["covariances"], teacher_values["covariances"].mT)):
        raise JointCalibrationRunError("teacher covariance is not symmetric")
    for name, attributes in (("compact", compact_values), ("teacher", teacher_values)):
        if bool((attributes["opacities"] < 0.0).any()) or bool((attributes["opacities"] >= 1.0).any()):
            raise JointCalibrationRunError(f"{name} opacity is outside [0, 1)")
    return CacheExample(
        model=model,
        descriptor=descriptor,
        compact=compact_values,
        teacher=teacher_values,
        sh_degree=degree,
    )


def _expected_input_identity(contract: Mapping[str, Any], split: str) -> dict[str, Any]:
    sidecar = contract["context_only_inputs"]["splits"][split]
    return {
        key: sidecar[key]
        for key in (
            "tree_sha256",
            "manifest_sha256",
            "input_provenance_sha256",
            "selection_sha256",
            "scene_count",
        )
    }


def _cache_controls(record: Mapping[str, Any], *, example_count: int) -> Mapping[str, bool]:
    routing = record.get("routing")
    controls = routing.get("controls") if isinstance(routing, Mapping) else None
    if not isinstance(controls, Mapping):
        raise JointCalibrationRunError("teacher-cache record has no selected-only controls")
    required = {*_CACHE_CONTROL_REQUIRED, "full_slot_count", "sparse_packet_count"}
    if set(controls) != required:
        raise JointCalibrationRunError("teacher-cache record is missing selected-only controls")
    for key in _CACHE_CONTROL_REQUIRED:
        if controls.get(key) is not True:
            raise JointCalibrationRunError(f"teacher-cache selected-only control failed: {key}")
    for key in ("full_slot_count", "sparse_packet_count"):
        value = controls.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise JointCalibrationRunError(f"teacher-cache selected-only control is invalid: {key}")
    if controls["sparse_packet_count"] != example_count:
        raise JointCalibrationRunError("teacher-cache sparse packet count differs from examples")
    if not isinstance(routing, Mapping):
        raise JointCalibrationRunError("teacher-cache routing is invalid")
    if controls["full_slot_count"] != routing.get("full_passthrough_gaussians"):
        raise JointCalibrationRunError("teacher-cache Full passthrough count differs from controls")
    return {
        "route_mask_unchanged_required": True,
        "retained_counts_unchanged_required": True,
        "full_passthrough_required": True,
        "selected_only_s3_required": True,
        "two_finite_skipped_s3_sentinels_required": True,
    }


def _validate_source_identity(
    value: Any, *, contract: Mapping[str, Any], split: str
) -> None:
    expected = {
        "sidecar": contract["context_only_inputs"]["splits"][split],
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "expected_results_accessed": False,
    }
    if value != expected:
        raise JointCalibrationRunError("teacher-cache source identity crossed the context-only boundary")


def _validate_preparation(
    value: Any, *, contract: Mapping[str, Any], model: str
) -> None:
    required = {
        "source_image_shape",
        "prepared_image_shape",
        "context_view_count",
        "target_mapping_present",
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
        "baseline_normalized",
        "patch_size",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointCalibrationRunError("teacher-cache preparation has an invalid schema")
    expected = contract["preprocessing"]
    if (
        value.get("source_image_shape") != expected["source_image_shape"]
        or value.get("prepared_image_shape") != expected["target_image_shape"]
        or value.get("context_view_count") != 2
        or value.get("patch_size") != expected["per_model_patch_size"][model]
    ):
        raise JointCalibrationRunError("teacher-cache preparation differs from the frozen contract")
    for field in (
        "target_mapping_present",
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
    ):
        if value.get(field) is not False:
            raise JointCalibrationRunError(f"teacher-cache preparation crossed {field}")
    if not isinstance(value.get("baseline_normalized"), bool):
        raise JointCalibrationRunError("teacher-cache preparation baseline normalization is invalid")


def _validate_teacher_boundary(value: Any) -> None:
    expected = {
        "offline_teacher_only": True,
        "source": "dense_adaptor_output",
        "target": "assignment_aligned_dense_adaptor_nonprobe_aggregation",
        "selected_anchor_only": True,
        "skipped_s3_descriptors_accessed_offline_teacher_only": True,
        "skipped_s3_descriptors_persisted": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
    }
    if value != expected:
        raise JointCalibrationRunError("teacher-cache teacher boundary is invalid")


def _validate_routing(
    value: Any, *, contract: Mapping[str, Any], example_count: int
) -> None:
    required = {
        "mask_sha256",
        "route_summary",
        "level0_tiles",
        "level1_tiles",
        "full_tiles",
        "retained_representatives",
        "full_passthrough_gaussians",
        "cross_check_threshold",
        "execution_route_sha256",
        "controls",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointCalibrationRunError("teacher-cache routing has an invalid schema")
    _digest(value.get("mask_sha256"), "teacher-cache routing mask SHA256")
    if value.get("cross_check_threshold") != float(
        contract["saes_routing"]["parameters"]["cross_check_threshold"]
    ):
        raise JointCalibrationRunError("teacher-cache routing cross-check threshold changed")
    if (
        value.get("execution_route_sha256")
        != contract["saes_routing"]["execution_route_sha256"]
    ):
        raise JointCalibrationRunError("teacher-cache routing execution route changed")
    summary = value.get("route_summary")
    if not isinstance(summary, Mapping) or set(summary) != set(_ROUTE_SUMMARY_FIELDS):
        raise JointCalibrationRunError("teacher-cache route summary has an invalid schema")
    for field in _ROUTE_SUMMARY_FIELDS:
        item = summary.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise JointCalibrationRunError(f"teacher-cache route summary is invalid: {field}")
    if (
        value.get("level0_tiles") != summary["level0_tiles"]
        or value.get("level1_tiles") != summary["level1_tiles"]
        or value.get("full_tiles") != summary["full_tiles"]
        or value.get("retained_representatives") != example_count
        or example_count
        != summary["l0_representatives"] + summary["l1_lightweight_anchors"]
        or value.get("full_passthrough_gaussians") != summary["full_stage3_gaussians"]
    ):
        raise JointCalibrationRunError("teacher-cache routing counts are inconsistent")
    for field in ("retained_representatives", "full_passthrough_gaussians"):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise JointCalibrationRunError(f"teacher-cache routing is invalid: {field}")


def load_completed_cache(
    root: Path,
    *,
    contract: Mapping[str, Any],
    model: str,
    split: str,
) -> ModelCache:
    """Load one complete cache only after validating all bound identities."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, "teacher-cache manifest")
    required = {
        "schema_version", "kind", "status", "partial", "contract_sha256", "plan_sha256",
        "model", "split", "checkpoint_sha256", "config_source_sha256",
        "runtime_source", "environment_profile", "profile_interpreter", "resolved_config_sha256", "preprocessing_id", "prepared_patch_size",
        "saes_routing_sha256", "saes_execution_route_sha256", "cross_check_threshold", "input_identity", "scene_count", "record_count",
        "entries", "access_audit",
    }
    if set(manifest) != required:
        raise JointCalibrationRunError("teacher-cache manifest has unexpected fields")
    if manifest.get("cross_check_threshold") != float(
        contract["saes_routing"]["parameters"]["cross_check_threshold"]
    ):
        raise JointCalibrationRunError("teacher-cache cross-check threshold differs from the frozen contract")
    if (
        manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("kind") != CACHE_KIND
        or manifest.get("status") != "PASS_AUTHOR_SIDE_OFFLINE_TEACHER"
        or manifest.get("partial") is not False
        or manifest.get("contract_sha256") != contract["contract_sha256"]
        or manifest.get("plan_sha256") != contract["source_plan"]["plan_sha256"]
        or manifest.get("model") != model
        or manifest.get("split") != split
        or manifest.get("checkpoint_sha256") != contract["checkpoint_binding"]["models"][model]["checkpoint_sha256"]
        or manifest.get("config_source_sha256") != contract["model_config_binding"][model]["source_manifest_sha256"]
        or manifest.get("runtime_source") != contract["model_config_binding"][model]["runtime_source"]
        or manifest.get("environment_profile") != contract["model_config_binding"][model]["environment_profile"]
        or manifest.get("profile_interpreter") != contract["model_config_binding"][model]["profile_interpreter"]
        or manifest.get("resolved_config_sha256")
        != contract["model_config_binding"][model]["resolved_config_sha256"]
        or manifest.get("preprocessing_id") != contract["preprocessing"]["identifier"]
        or manifest.get("prepared_patch_size") != contract["preprocessing"]["per_model_patch_size"][model]
        or manifest.get("saes_routing_sha256") != contract["saes_routing"]["routing_sha256"]
        or manifest.get("saes_execution_route_sha256")
        != contract["saes_routing"]["execution_route_sha256"]
        or manifest.get("input_identity") != _expected_input_identity(contract, split)
    ):
        raise JointCalibrationRunError("teacher-cache manifest differs from the frozen contract")
    entries = manifest.get("entries")
    expected_scene_count = contract["context_only_inputs"]["splits"][split]["scene_count"]
    if not isinstance(entries, list) or len(entries) != expected_scene_count:
        raise JointCalibrationRunError("teacher-cache manifest does not cover its complete split")
    if manifest.get("scene_count") != expected_scene_count:
        raise JointCalibrationRunError("teacher-cache scene count is invalid")
    audit = manifest.get("access_audit")
    if not isinstance(audit, Mapping):
        raise JointCalibrationRunError("teacher-cache access audit is invalid")
    for field in (
        "target_rgb_accessed", "target_camera_metadata_accessed", "target_index_accessed",
        "expected_results_accessed", "evaluation_scene_accessed", "teacher_files_runtime_accessible",
        "descriptor_model_id_accessed", "descriptor_dataset_id_accessed", "optimizer_executed",
    ):
        if audit.get(field) is not False:
            raise JointCalibrationRunError(f"teacher-cache access audit failed: {field}")
    if audit.get("teacher_source") != "dense_adaptor_output":
        raise JointCalibrationRunError("teacher-cache used an unregistered teacher source")
    examples: list[CacheExample] = []
    controls: Mapping[str, bool] | None = None
    seen_indices: set[int] = set()
    for ordinal, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != {
            "sample_index", "scene", "path", "sha256", "byte_count", "record_count", "routing"
        }:
            raise JointCalibrationRunError("teacher-cache entry has an invalid schema")
        sample_index = entry.get("sample_index")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index != ordinal
            or sample_index in seen_indices
        ):
            raise JointCalibrationRunError("teacher-cache entry sample order is invalid")
        if not isinstance(entry.get("scene"), str) or not entry["scene"]:
            raise JointCalibrationRunError("teacher-cache entry scene is invalid")
        seen_indices.add(sample_index)
        path = _safe_path(root, entry.get("path"), "teacher-cache entry")
        if not path.is_file() or path.stat().st_size != entry.get("byte_count"):
            raise JointCalibrationRunError("teacher-cache entry is missing or truncated")
        if _sha256_file(path) != entry.get("sha256"):
            raise JointCalibrationRunError("teacher-cache entry SHA256 changed")
        try:
            record = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise JointCalibrationRunError("teacher-cache entry could not be loaded safely") from exc
        record_required = {
            "schema_version", "kind", "contract_sha256", "plan_sha256", "model", "split", "scene",
            "sample_index", "checkpoint_sha256", "config_source_sha256", "runtime_source", "resolved_config_sha256",
            "environment_profile", "profile_interpreter",
            "preprocessing_id", "prepared_patch_size", "saes_routing_sha256", "saes_execution_route_sha256", "input_identity",
            "source_identity", "preparation", "routing", "teacher", "examples",
        }
        if not isinstance(record, Mapping) or set(record) != record_required:
            raise JointCalibrationRunError("teacher-cache record has an invalid schema")
        for field, expected in (
            ("contract_sha256", contract["contract_sha256"]),
            ("plan_sha256", contract["source_plan"]["plan_sha256"]),
            ("model", model),
            ("split", split),
            ("checkpoint_sha256", contract["checkpoint_binding"]["models"][model]["checkpoint_sha256"]),
            ("config_source_sha256", contract["model_config_binding"][model]["source_manifest_sha256"]),
            ("runtime_source", contract["model_config_binding"][model]["runtime_source"]),
            ("environment_profile", contract["model_config_binding"][model]["environment_profile"]),
            ("profile_interpreter", contract["model_config_binding"][model]["profile_interpreter"]),
            ("resolved_config_sha256", contract["model_config_binding"][model]["resolved_config_sha256"]),
            ("preprocessing_id", contract["preprocessing"]["identifier"]),
            ("prepared_patch_size", contract["preprocessing"]["per_model_patch_size"][model]),
            ("saes_routing_sha256", contract["saes_routing"]["routing_sha256"]),
            (
                "saes_execution_route_sha256",
                contract["saes_routing"]["execution_route_sha256"],
            ),
            ("input_identity", _expected_input_identity(contract, split)),
            ("sample_index", sample_index),
        ):
            if record.get(field) != expected:
                raise JointCalibrationRunError(f"teacher-cache record identity changed: {field}")
        if record.get("scene") != entry["scene"]:
            raise JointCalibrationRunError("teacher-cache record scene differs from manifest")
        _validate_source_identity(record.get("source_identity"), contract=contract, split=split)
        _validate_preparation(record.get("preparation"), contract=contract, model=model)
        _validate_teacher_boundary(record.get("teacher"))
        row_examples = record.get("examples")
        if not isinstance(row_examples, list) or len(row_examples) != entry.get("record_count"):
            raise JointCalibrationRunError("teacher-cache record count is invalid")
        anchors = [item.get("anchor_index") if isinstance(item, Mapping) else None for item in row_examples]
        if (
            any(isinstance(anchor, bool) or not isinstance(anchor, int) or anchor < 0 for anchor in anchors)
            or anchors != sorted(anchors)
            or len(anchors) != len(set(anchors))
        ):
            raise JointCalibrationRunError("teacher-cache anchors are not in stable ascending order")
        _validate_routing(record.get("routing"), contract=contract, example_count=len(row_examples))
        if entry.get("routing") != record["routing"]:
            raise JointCalibrationRunError("teacher-cache manifest routing differs from scene record")
        local_controls = _cache_controls(record, example_count=len(row_examples))
        controls = local_controls if controls is None else controls
        if controls != local_controls:
            raise JointCalibrationRunError("teacher-cache selected-only controls differ by scene")
        examples.extend(_validate_example(item, model=model) for item in row_examples)
    if sorted(seen_indices) != list(range(expected_scene_count)):
        raise JointCalibrationRunError("teacher-cache does not enumerate the frozen split order")
    if manifest.get("record_count") != len(examples) or not examples or controls is None:
        raise JointCalibrationRunError("teacher-cache aggregate record count is invalid")
    return ModelCache(
        model=model,
        split=split,
        resolved_config_sha256=str(manifest["resolved_config_sha256"]),
        prepared_patch_size=int(manifest["prepared_patch_size"]),
        examples=tuple(examples),
        controls=controls,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch(examples: Sequence[CacheExample], start: int, batch_size: int, device: torch.device) -> tuple[dict[str, torch.Tensor], int, int]:
    if not examples or batch_size < 1:
        raise JointCalibrationRunError("teacher-cache batch is empty")
    indices = [(start + offset) % len(examples) for offset in range(batch_size)]
    selected = [examples[index] for index in indices]
    degrees = {item.sh_degree for item in selected}
    if len(degrees) != 1:
        raise JointCalibrationRunError("one model cache mixes unsupported SH degrees")
    degree = next(iter(degrees))

    def stack(family: str, name: str) -> torch.Tensor:
        return torch.stack([getattr(item, family)[name] for item in selected], dim=0).to(device)

    return {
        "descriptor": torch.stack([item.descriptor for item in selected], dim=0).to(device),
        "means": stack("compact", "means"),
        "covariances": stack("compact", "covariances"),
        "harmonics": stack("compact", "harmonics"),
        "opacities": stack("compact", "opacities"),
        "teacher_means": stack("teacher", "means"),
        "teacher_covariances": stack("teacher", "covariances"),
        "teacher_harmonics": stack("teacher", "harmonics"),
        "teacher_opacities": stack("teacher", "opacities"),
    }, (start + batch_size) % len(examples), degree


def _component_mse(prediction: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "mean": torch.mean((prediction["means"] - target["means"]).square()),
        "covariance": torch.mean((prediction["covariances"] - target["covariances"]).square()),
        "opacity": torch.mean((prediction["opacities"] - target["opacities"]).square()),
        "sh": torch.mean((prediction["harmonics"] - target["harmonics"]).square()),
    }


def _calibrated_batch(calibrator: JointMaterializationCalibrator, batch: Mapping[str, torch.Tensor], *, sh_degree: int) -> Mapping[str, torch.Tensor]:
    attributes = calibrator(
        batch["descriptor"], batch["means"], batch["covariances"], batch["harmonics"], batch["opacities"], sh_degree=sh_degree
    )
    return {
        "means": attributes.means,
        "covariances": attributes.covariances,
        "harmonics": attributes.harmonics,
        "opacities": attributes.opacities,
    }


def _teacher_targets(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "means": batch["teacher_means"],
        "covariances": batch["teacher_covariances"],
        "harmonics": batch["teacher_harmonics"],
        "opacities": batch["teacher_opacities"],
    }


def _identity_attributes(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "means": batch["means"],
        "covariances": batch["covariances"],
        "harmonics": batch["harmonics"],
        "opacities": batch["opacities"],
    }


def _learning_rate(recipe: Mapping[str, Any], update: int) -> float:
    total = int(recipe["total_updates"])
    warmup = int(recipe["warmup_updates"])
    base = float(recipe["learning_rate"])
    final = float(recipe["final_learning_rate"])
    if not 0 <= update < total:
        raise JointCalibrationRunError("optimizer update is outside the frozen schedule")
    if update < warmup:
        return base * float(update + 1) / float(warmup)
    phase = float(update - warmup) / float(total - warmup - 1)
    return final + (base - final) * (0.5 + 0.5 * math.cos(math.pi * phase))


def _global_identity_normalizers(
    caches: Mapping[str, ModelCache], *, floor: float
) -> dict[str, dict[str, float]]:
    """Compute the frozen per-model/component normalizers before optimization.

    The objective contract names a *per-model per-component identity MSE*, not
    a batch-local ratio.  Computing it once over each complete immutable cache
    prevents batch composition from changing the loss scale.
    """
    if not isinstance(floor, float) or not math.isfinite(floor) or floor <= 0.0:
        raise JointCalibrationRunError("identity normalizer floor is invalid")
    result: dict[str, dict[str, float]] = {}
    attribute_names = {
        "mean": "means",
        "covariance": "covariances",
        "opacity": "opacities",
        "sh": "harmonics",
    }
    for model in MODELS:
        cache = caches.get(model)
        if cache is None or not cache.examples:
            raise JointCalibrationRunError("complete model caches are required for normalizers")
        values: dict[str, float] = {}
        for component, attribute in attribute_names.items():
            squared_sum = 0.0
            element_count = 0
            for example in cache.examples:
                delta = example.compact[attribute].to(dtype=torch.float64) - example.teacher[attribute].to(dtype=torch.float64)
                squared_sum += float(delta.square().sum().item())
                element_count += delta.numel()
            if element_count <= 0:
                raise JointCalibrationRunError("teacher-cache identity normalizer is empty")
            values[component] = max(squared_sum / element_count, floor)
        result[model] = values
    return result


def train_shared_asset(caches: Mapping[str, ModelCache], *, contract: Mapping[str, Any], device: torch.device) -> JointMaterializationCalibrator:
    """Run exactly the registered global 12,000-step round-robin optimizer."""
    recipe = contract["optimization_recipe"]
    if recipe.get("resume_allowed") is not False or recipe.get("early_stopping_allowed") is not False or recipe.get("hyperparameter_search_allowed") is not False:
        raise JointCalibrationRunError("training contract does not forbid optimization drift")
    torch.manual_seed(int(recipe["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(recipe["seed"]))
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        calibrator = JointMaterializationCalibrator().to(device).train()
        optimizer = torch.optim.AdamW(
            calibrator.parameters(),
            lr=float(recipe["learning_rate"]),
            betas=tuple(float(value) for value in recipe["betas"]),
            eps=float(recipe["epsilon"]),
            weight_decay=float(recipe["weight_decay"]),
        )
        cursors = {model: 0 for model in MODELS}
        batch_size = int(recipe["batch_size"])
        normalizer_floor = float(contract["teacher_objective"]["loss"]["normalizer_floor"])
        normalizers = _global_identity_normalizers(caches, floor=normalizer_floor)
        for update in range(int(recipe["total_updates"])):
            model = MODELS[update % len(MODELS)]
            batch, cursors[model], degree = _batch(caches[model].examples, cursors[model], batch_size, device)
            target = _teacher_targets(batch)
            calibrated = _component_mse(_calibrated_batch(calibrator, batch, sh_degree=degree), target)
            loss = sum(
                calibrated[name] / normalizers[model][name]
                for name in METRIC_COMPONENTS
            ) / len(METRIC_COMPONENTS)
            if not bool(torch.isfinite(loss)):
                raise JointCalibrationRunError("frozen optimizer produced a non-finite loss")
            for group in optimizer.param_groups:
                group["lr"] = _learning_rate(recipe, update)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), float(recipe["gradient_clip_norm"]))
            optimizer.step()
        return calibrator.eval()
    finally:
        torch.use_deterministic_algorithms(previous_determinism)


def _metrics(calibrator: JointMaterializationCalibrator, cache: ModelCache, *, device: torch.device) -> dict[str, dict[str, float]]:
    sums = {name: {"identity": 0.0, "calibrated": 0.0, "count": 0} for name in METRIC_COMPONENTS}
    with torch.no_grad():
        for index in range(len(cache.examples)):
            batch, _, degree = _batch(cache.examples, index, 1, device)
            target = _teacher_targets(batch)
            baseline = _component_mse(_identity_attributes(batch), target)
            calibrated = _component_mse(_calibrated_batch(calibrator, batch, sh_degree=degree), target)
            elements = {
                "mean": int(target["means"].numel()),
                "covariance": int(target["covariances"].numel()),
                "opacity": int(target["opacities"].numel()),
                "sh": int(target["harmonics"].numel()),
            }
            for name in METRIC_COMPONENTS:
                sums[name]["identity"] += float(baseline[name].item()) * elements[name]
                sums[name]["calibrated"] += float(calibrated[name].item()) * elements[name]
                sums[name]["count"] += elements[name]
    return {
        name: {
            "identity_mse": values["identity"] / values["count"],
            "calibrated_mse": values["calibrated"] / values["count"],
        }
        for name, values in sums.items()
    }


def _runtime_evidence(
    path: Path,
    *,
    contract: Mapping[str, Any],
    model: str,
    split: str,
    asset: Mapping[str, Any],
    candidate_execution_verification: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    evidence = _read_json(path, f"{model} runtime-control evidence")
    required = {
        "schema_version", "kind", "contract_sha256", "plan_sha256", "model", "split",
        "checkpoint_sha256", "config_source_sha256", "runtime_source", "resolved_config_sha256",
        "environment_profile", "profile_interpreter",
        "preprocessing_id", "prepared_patch_size", "saes_routing_sha256", "saes_execution_route_sha256", "input_identity",
        "asset", "candidate_execution_verification", "execution_boundary", "selected_head", "ledger", "access_audit", "scene_records",
    }
    if set(evidence) != required or evidence.get("schema_version") != RUNTIME_EVIDENCE_SCHEMA_VERSION or evidence.get("kind") != RUNTIME_EVIDENCE_KIND:
        raise JointCalibrationRunError("runtime-control evidence has an invalid schema")
    if (
        evidence.get("contract_sha256") != contract["contract_sha256"]
        or evidence.get("plan_sha256") != contract["source_plan"]["plan_sha256"]
        or evidence.get("model") != model
        or evidence.get("split") != split
        or evidence.get("checkpoint_sha256") != contract["checkpoint_binding"]["models"][model]["checkpoint_sha256"]
        or evidence.get("config_source_sha256") != contract["model_config_binding"][model]["source_manifest_sha256"]
        or evidence.get("runtime_source") != contract["model_config_binding"][model]["runtime_source"]
        or evidence.get("environment_profile") != contract["model_config_binding"][model]["environment_profile"]
        or evidence.get("profile_interpreter") != contract["model_config_binding"][model]["profile_interpreter"]
        or evidence.get("resolved_config_sha256")
        != contract["model_config_binding"][model]["resolved_config_sha256"]
        or evidence.get("preprocessing_id") != contract["preprocessing"]["identifier"]
        or evidence.get("prepared_patch_size") != contract["preprocessing"]["per_model_patch_size"][model]
        or evidence.get("saes_routing_sha256") != contract["saes_routing"]["routing_sha256"]
        or evidence.get("saes_execution_route_sha256")
        != contract["saes_routing"]["execution_route_sha256"]
        or evidence.get("input_identity") != _expected_input_identity(contract, split)
        or evidence.get("asset") != asset
        or evidence.get("candidate_execution_verification")
        != candidate_execution_verification
    ):
        raise JointCalibrationRunError("runtime-control evidence differs from the frozen contract")
    head = evidence.get("selected_head")
    if (
        not isinstance(head, Mapping)
        or set(head) != {"model", "contract_version", "dense_head_macs", "replayed_head_macs"}
        or head.get("model") != model
        or head.get("contract_version") != "saes-selected-output-replay-v1"
    ):
        raise JointCalibrationRunError("runtime-control evidence has no model-bound selected-head replay")
    dense_macs = head.get("dense_head_macs")
    replayed_macs = head.get("replayed_head_macs")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (dense_macs, replayed_macs)) or dense_macs <= replayed_macs:
        raise JointCalibrationRunError("runtime-control evidence selected-head MACs are invalid")
    ledger = evidence.get("ledger")
    if not isinstance(ledger, Mapping) or ledger.get("ledger_version") != LEDGER_VERSION:
        raise JointCalibrationRunError("runtime-control evidence has no analytic SAES ledger")
    events = ledger.get("events")
    if not isinstance(events, Mapping):
        raise JointCalibrationRunError("runtime-control evidence ledger events are invalid")
    for field in ("joint_calibrator_calls", "joint_calibrator_selected_descriptor_reads", "joint_calibrator_skipped_head_macs"):
        if isinstance(events.get(field), bool) or not isinstance(events.get(field), int) or events[field] < 0:
            raise JointCalibrationRunError(f"runtime-control evidence ledger is missing {field}")
    if events["joint_calibrator_calls"] != events["joint_calibrator_selected_descriptor_reads"] or events["joint_calibrator_skipped_head_macs"] <= 0:
        raise JointCalibrationRunError("runtime-control evidence ledger does not charge the calibrator")
    for field in ("joint_calibrator_l0_calls", "joint_calibrator_l1_calls", "joint_calibrator_full_calls"):
        if isinstance(events.get(field), bool) or not isinstance(events.get(field), int) or events[field] < 0:
            raise JointCalibrationRunError(f"runtime-control evidence ledger is missing {field}")
    if (
        events["joint_calibrator_calls"]
        != events["joint_calibrator_l0_calls"]
        + events["joint_calibrator_l1_calls"]
        + events["joint_calibrator_full_calls"]
        or events["joint_calibrator_full_calls"] != 0
    ):
        raise JointCalibrationRunError("runtime-control evidence ledger has invalid Full accounting")
    expected_boundary = {
        "dense_route_prepass_for_validation": True,
        "selected_head_only": True,
        "s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "renderer_executed": False,
        "quality_metrics_computed": False,
    }
    if evidence.get("execution_boundary") != expected_boundary:
        raise JointCalibrationRunError("runtime-control evidence overstates its execution boundary")
    audit = evidence.get("access_audit")
    required_audit = {
        "target_rgb_accessed", "target_camera_metadata_accessed", "target_index_accessed",
        "expected_results_accessed", "evaluation_scene_accessed", "teacher_files_opened",
        "teacher_files_runtime_accessible", "runtime_teacher_path", "descriptor_model_id_accessed",
        "descriptor_dataset_id_accessed", "optimizer_executed", "asset_updated",
        "renderer_executed", "quality_metrics_computed",
    }
    false_fields = required_audit - {"runtime_teacher_path"}
    if (
        not isinstance(audit, Mapping)
        or set(audit) != required_audit
        or any(audit.get(field) is not False for field in false_fields)
        or audit.get("runtime_teacher_path") is not None
    ):
        raise JointCalibrationRunError("runtime-control evidence crossed an isolation boundary")
    records = evidence.get("scene_records")
    expected_count = contract["context_only_inputs"]["splits"][split]["scene_count"]
    record_fields = {
        "sample_index", "scene", "route_mask_sha256", "selected_head_sha256", "saes_stats_sha256"
    }
    if not isinstance(records, list) or len(records) != expected_count:
        raise JointCalibrationRunError("runtime-control evidence has incomplete scene coverage")
    for ordinal, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or set(record) != record_fields
            or record.get("sample_index") != ordinal
            or not isinstance(record.get("scene"), str)
            or not record["scene"]
        ):
            raise JointCalibrationRunError("runtime-control evidence scene order is invalid")
        for field in record_fields - {"sample_index", "scene"}:
            _digest(record.get(field), f"runtime-control evidence {field}")
    if len({record["scene"] for record in records}) != len(records):
        raise JointCalibrationRunError("runtime-control evidence has duplicate scenes")
    identity = {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "selected_head_sha256": canonical_sha256(head),
        "ledger_sha256": canonical_sha256(ledger),
        "asset_sha256": str(asset["sha256"]),
        "asset_state_sha256": str(asset["state_sha256"]),
    }
    return dict(evidence), identity


def _controls(cache: ModelCache, evidence: Mapping[str, Any]) -> dict[str, bool]:
    controls = {
        **cache.controls,
        "selected_head_event_contract_required": bool(evidence),
        "cost_ledger_required": bool(evidence),
        "all_three_models_required": True,
    }
    if set(controls) != set(RUNTIME_PRECONDITIONS) or any(value is not True for value in controls.values()):
        raise JointCalibrationRunError("run controls cannot satisfy the frozen teacher-fidelity gate")
    return controls


def _asset_identity(path: Path, *, record_path: str | None = None) -> dict[str, Any]:
    manifest = calibrator_asset_manifest(path)
    loaded = load_calibrator_asset(path, manifest)
    return {
        "asset_path": path.relative_to(ROOT).as_posix() if record_path is None else record_path,
        "state_sha256": loaded.runtime_state_sha256,
        **manifest,
    }


def _result(
    *,
    contract: Mapping[str, Any],
    stage: str,
    caches: Mapping[str, ModelCache],
    calibrator: JointMaterializationCalibrator,
    asset: Mapping[str, Any],
    evidence: Mapping[str, tuple[Mapping[str, Any], Mapping[str, str]]],
    device: torch.device,
    frozen_train_result_sha256: str | None,
    candidate_execution_verification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    models = {}
    for model in MODELS:
        cache = caches[model]
        runtime_evidence, runtime_identity = evidence[model]
        if runtime_evidence["resolved_config_sha256"] != cache.resolved_config_sha256:
            raise JointCalibrationRunError(
                f"runtime-control evidence {model} resolved config differs from cache"
            )
        models[model] = {
            "checkpoint_sha256": contract["checkpoint_binding"]["models"][model]["checkpoint_sha256"],
            "config_source_sha256": contract["model_config_binding"][model]["source_manifest_sha256"],
            "runtime_source": dict(contract["model_config_binding"][model]["runtime_source"]),
            "environment_profile": contract["model_config_binding"][model]["environment_profile"],
            "profile_interpreter": dict(contract["model_config_binding"][model]["profile_interpreter"]),
            "resolved_config_sha256": cache.resolved_config_sha256,
            "preprocessing_id": contract["preprocessing"]["identifier"],
            "prepared_patch_size": cache.prepared_patch_size,
            "saes_routing_sha256": contract["saes_routing"]["routing_sha256"],
            "saes_execution_route_sha256": contract["saes_routing"][
                "execution_route_sha256"
            ],
            "input_identity": _expected_input_identity(contract, stage),
            "metrics": _metrics(calibrator, cache, device=device),
            "controls": _controls(cache, runtime_evidence),
            "runtime_evidence": dict(runtime_identity),
        }
    audit = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": ACCESS_AUDIT_KIND,
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "stage": "train" if stage == TRAIN_SPLIT else "holdout",
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "expected_results_accessed": False,
        "evaluation_scene_accessed": False,
        "descriptor_model_id_accessed": False,
        "descriptor_dataset_id_accessed": False,
        "teacher_files_opened": True,
        "teacher_files_runtime_accessible": False,
        "runtime_teacher_path": None,
        "optimizer_executed": stage == TRAIN_SPLIT,
        "asset_updated": stage == TRAIN_SPLIT,
        "rerank_executed": False,
        "partition_reshuffled": False,
    }
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "stage": stage,
        "frozen_train_result_sha256": frozen_train_result_sha256,
        "asset": dict(asset),
        "candidate_execution_verification": candidate_execution_verification,
        "models": models,
        "access_audit": audit,
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"partial result already exists: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_asset_new(calibrator: JointMaterializationCalibrator, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen calibrator asset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"partial calibrator asset already exists: {temporary}")
    # Do not call Module.cpu(): training and metric evaluation may still use a
    # CUDA-resident calibrator after this immutable snapshot has been staged.
    state_dict = {
        name: value.detach().to(device="cpu").clone()
        for name, value in calibrator.state_dict().items()
    }
    torch.save({"state_dict": state_dict}, temporary)
    os.replace(temporary, path)


def _candidate_asset_path(asset_path: Path) -> Path:
    return asset_path.with_name(asset_path.name + ".candidate")


def _candidate_provenance_path(asset_path: Path) -> Path:
    candidate = _candidate_asset_path(asset_path)
    return candidate.with_name(candidate.name + ".provenance.json")


def _candidate_verification_path(asset_path: Path) -> Path:
    candidate = _candidate_asset_path(asset_path)
    return candidate.with_name(candidate.name + ".verification.json")


def _candidate_cache_manifest_identities(
    contract: Mapping[str, Any], *, split: str
) -> dict[str, dict[str, str]]:
    if split != TRAIN_SPLIT:
        raise JointCalibrationRunError("candidate provenance only permits the frozen train split")
    identities: dict[str, dict[str, str]] = {}
    for model in MODELS:
        manifest = _canonical_cache_root(contract, model=model, split=split) / "manifest.json"
        if not manifest.is_file():
            raise JointCalibrationRunError(f"{model} candidate cache manifest is unavailable")
        identities[model] = {
            "path": manifest.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(manifest),
        }
    return identities


def _candidate_provenance(
    path: Path,
    *,
    contract: Mapping[str, Any],
    candidate_asset: Mapping[str, Any],
    cache_manifests: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _safe_path(ROOT, candidate_provenance_path_for_contract(contract), "candidate provenance")
    _require_canonical_path(path, expected, label="candidate provenance")
    provenance = _read_json(path, "candidate provenance")
    identity = validate_candidate_provenance(
        provenance,
        contract=contract,
        candidate_asset=candidate_asset,
        cache_manifests=cache_manifests,
    )
    return {
        "path": expected.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(expected),
        "candidate_asset_sha256": identity["candidate_asset_sha256"],
        "candidate_asset_state_sha256": identity["candidate_asset_state_sha256"],
        "cache_manifests": identity["cache_manifests"],
        "local_hash_chain_is_not_cryptographic_proof": True,
    }


def _candidate_verification(
    path: Path | None,
    *,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if path is None:
        raise JointCalibrationRunError(
            "candidate runtime or promotion is BLOCKED_NO_VERIFIED_EXECUTION without a verifier record"
        )
    expected = _safe_path(ROOT, candidate_verification_path_for_contract(contract), "candidate verification")
    _require_canonical_path(path, expected, label="candidate verification")
    verification = _read_json(expected, "candidate verification")
    identity = validate_candidate_verification(
        verification,
        contract=contract,
        provenance_path=str(provenance["path"]),
        provenance_sha256=str(provenance["sha256"]),
        provenance=provenance,
    )
    return {
        "path": expected.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(expected),
        **identity,
    }


def _load_canonical_contract(contract_path: Path) -> dict[str, Any]:
    _require_canonical_path(
        contract_path, _canonical_contract_path(), label="joint training contract"
    )
    contract = _read_json(contract_path, "joint training contract")
    validate_live_training_contract(contract, repository_root=ROOT)
    return contract


def _load_canonical_caches(
    cache_roots: Mapping[str, Path], *, contract: Mapping[str, Any], split: str
) -> dict[str, ModelCache]:
    if set(cache_roots) != set(MODELS):
        raise JointCalibrationRunError("canonical execution requires all three model caches")
    caches: dict[str, ModelCache] = {}
    for model in MODELS:
        expected = _canonical_cache_root(contract, model=model, split=split)
        root = _require_canonical_path(
            cache_roots[model], expected, label=f"{model} {split} teacher cache"
        )
        caches[model] = load_completed_cache(
            root, contract=contract, model=model, split=split
        )
    return caches


def _load_canonical_runtime_evidence(
    runtime_evidence_paths: Mapping[str, Path],
    *,
    contract: Mapping[str, Any],
    split: str,
    asset: Mapping[str, Any],
    candidate_execution_verification: Mapping[str, Any] | None,
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, str]]]:
    if set(runtime_evidence_paths) != set(MODELS):
        raise JointCalibrationRunError("canonical execution requires all three runtime proofs")
    evidence: dict[str, tuple[Mapping[str, Any], Mapping[str, str]]] = {}
    for model in MODELS:
        expected = _canonical_runtime_evidence_path(contract, model=model, split=split)
        path = _require_canonical_path(
            runtime_evidence_paths[model], expected, label=f"{model} {split} runtime evidence"
        )
        evidence[model] = _runtime_evidence(
            path,
            contract=contract,
            model=model,
            split=split,
            asset=asset,
            candidate_execution_verification=candidate_execution_verification,
        )
    return evidence


def _load_asset(
    path: Path, *, asset: Mapping[str, Any], device: torch.device
) -> JointMaterializationCalibrator:
    try:
        manifest = {
            key: asset[key]
            for key in (
                "schema_version",
                "kind",
                "sha256",
                "byte_count",
                "descriptor_dim",
                "bottleneck_dim",
                "joint_output_dim",
            )
        }
        return load_calibrator_asset(path, manifest).to(device).eval()
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise JointCalibrationRunError("frozen calibrator asset cannot be loaded") from exc


def prepare_train_candidate(
    *,
    contract_path: Path,
    cache_roots: Mapping[str, Path],
    device: torch.device,
) -> dict[str, Any]:
    """Train exactly once into a noncanonical candidate asset.

    The candidate is deliberately not a runtime asset or a fidelity result. A
    source-bound context-only runtime proof must bind this exact candidate
    before ``finalize_train_candidate`` can promote it.
    """
    contract = _load_canonical_contract(contract_path)
    caches = _load_canonical_caches(cache_roots, contract=contract, split=TRAIN_SPLIT)
    asset_path = _safe_path(ROOT, contract["shared_asset"]["asset_path"], "calibrator asset")
    candidate_path = _candidate_asset_path(asset_path)
    provenance_path = _candidate_provenance_path(asset_path)
    verification_path = _candidate_verification_path(asset_path)
    result_path = _safe_path(ROOT, contract["result_records"][TRAIN_SPLIT], "train result")
    if (
        asset_path.exists()
        or result_path.exists()
        or candidate_path.exists()
        or provenance_path.exists()
        or verification_path.exists()
    ):
        raise FileExistsError(
            "train asset/result/candidate/provenance already exists; refusing an optimizer retry"
        )
    calibrator = train_shared_asset(caches, contract=contract, device=device)
    _write_asset_new(calibrator, candidate_path)
    asset = _asset_identity(
        candidate_path, record_path=contract["shared_asset"]["asset_path"]
    )
    cache_manifests = _candidate_cache_manifest_identities(contract, split=TRAIN_SPLIT)
    provenance = {
        "schema_version": CANDIDATE_PROVENANCE_SCHEMA_VERSION,
        "kind": CANDIDATE_PROVENANCE_KIND,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "candidate_asset_path": candidate_path.relative_to(ROOT).as_posix(),
        "candidate_asset_sha256": asset["sha256"],
        "candidate_asset_state_sha256": asset["state_sha256"],
        "candidate_asset_byte_count": asset["byte_count"],
        "cache_manifests": cache_manifests,
        "optimization_recipe_sha256": canonical_sha256(contract["optimization_recipe"]),
        "seed": contract["optimization_recipe"]["seed"],
        "total_updates": contract["optimization_recipe"]["total_updates"],
        "local_hash_chain_is_not_cryptographic_proof": True,
        "verification_status": "BLOCKED_NO_VERIFIED_EXECUTION",
    }
    validate_candidate_provenance(
        provenance,
        contract=contract,
        candidate_asset=asset,
        cache_manifests=cache_manifests,
    )
    _write_json_new(provenance_path, provenance)
    return {
        "status": "CANDIDATE_ASSET_READY_NONCLAIMING",
        "candidate_asset_path": candidate_path.relative_to(ROOT).as_posix(),
        "asset": asset,
        "candidate_provenance_path": provenance_path.relative_to(ROOT).as_posix(),
        "candidate_provenance_sha256": sha256_file(provenance_path),
        "verification_status": "BLOCKED_NO_VERIFIED_EXECUTION",
        "runtime_evidence_paths": dict(contract["runtime_evidence_records"][TRAIN_SPLIT]),
        "paper_result_eligible": False,
    }


def finalize_train_candidate(
    *,
    contract_path: Path,
    cache_roots: Mapping[str, Path],
    runtime_evidence_paths: Mapping[str, Path],
    device: torch.device,
    candidate_verification_path: Path | None = None,
) -> dict[str, Any]:
    """Gate a staged train asset before atomically promoting it to canonical."""
    contract = _load_canonical_contract(contract_path)
    caches = _load_canonical_caches(cache_roots, contract=contract, split=TRAIN_SPLIT)
    asset_path = _safe_path(ROOT, contract["shared_asset"]["asset_path"], "calibrator asset")
    candidate_path = _candidate_asset_path(asset_path)
    provenance_path = _candidate_provenance_path(asset_path)
    result_path = _safe_path(ROOT, contract["result_records"][TRAIN_SPLIT], "train result")
    if (
        asset_path.exists()
        or result_path.exists()
        or not candidate_path.is_file()
        or not provenance_path.is_file()
    ):
        raise JointCalibrationRunError("train finalization requires exactly one staged candidate asset")
    asset = _asset_identity(
        candidate_path, record_path=contract["shared_asset"]["asset_path"]
    )
    cache_manifests = _candidate_cache_manifest_identities(contract, split=TRAIN_SPLIT)
    provenance = _candidate_provenance(
        provenance_path,
        contract=contract,
        candidate_asset=asset,
        cache_manifests=cache_manifests,
    )
    verification = _candidate_verification(
        candidate_verification_path, contract=contract, provenance=provenance
    )
    candidate_execution_verification = {
        "provenance": provenance,
        "verification": verification,
    }
    calibrator = _load_asset(candidate_path, asset=asset, device=device)
    evidence = _load_canonical_runtime_evidence(
        runtime_evidence_paths,
        contract=contract,
        split=TRAIN_SPLIT,
        asset=asset,
        candidate_execution_verification=candidate_execution_verification,
    )
    result = _result(
        contract=contract,
        stage=TRAIN_SPLIT,
        caches=caches,
        calibrator=calibrator,
        asset=asset,
        evidence=evidence,
        device=device,
        frozen_train_result_sha256=None,
        candidate_execution_verification=candidate_execution_verification,
    )
    # The fidelity gate runs against the staged identity before anything reaches
    # the canonical asset path. A failed result leaves only the diagnostic
    # candidate outside the runtime namespace.
    validate_teacher_fidelity_result(result, contract=contract)
    os.replace(candidate_path, asset_path)
    _write_json_new(result_path, result)
    return validate_live_teacher_fidelity_result(
        result, result_path=result_path, contract=contract, repository_root=ROOT
    )


def run_holdout_stage(
    *,
    contract_path: Path,
    cache_roots: Mapping[str, Path],
    runtime_evidence_paths: Mapping[str, Path],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate holdout once with the immutable train asset and no optimizer."""
    contract = _load_canonical_contract(contract_path)
    caches = _load_canonical_caches(cache_roots, contract=contract, split=HOLDOUT_SPLIT)
    asset_path = _safe_path(ROOT, contract["shared_asset"]["asset_path"], "calibrator asset")
    train_path = _safe_path(ROOT, contract["result_records"][TRAIN_SPLIT], "frozen train result")
    result_path = _safe_path(ROOT, contract["result_records"][HOLDOUT_SPLIT], "holdout result")
    if result_path.exists():
        raise FileExistsError("holdout result already exists; refusing a replay")
    train_result = _read_json(train_path, "frozen train result")
    asset = train_result.get("asset") if isinstance(train_result, Mapping) else None
    if not isinstance(asset, Mapping):
        raise JointCalibrationRunError("frozen train result has no asset identity")
    calibrator = _load_asset(asset_path, asset=asset, device=device)
    if _asset_identity(asset_path) != asset:
        raise JointCalibrationRunError("canonical asset differs from frozen train result")
    evidence = _load_canonical_runtime_evidence(
        runtime_evidence_paths,
        contract=contract,
        split=HOLDOUT_SPLIT,
        asset=asset,
        candidate_execution_verification=None,
    )
    result = _result(
        contract=contract,
        stage=HOLDOUT_SPLIT,
        caches=caches,
        calibrator=calibrator,
        asset=asset,
        evidence=evidence,
        device=device,
        frozen_train_result_sha256=sha256_file(train_path),
        candidate_execution_verification=None,
    )
    validate_teacher_fidelity_result(result, contract=contract, frozen_train_result=train_result)
    _write_json_new(result_path, result)
    return validate_live_teacher_fidelity_result(
        result,
        result_path=result_path,
        contract=contract,
        repository_root=ROOT,
        frozen_train_result=train_result,
        frozen_train_result_path=train_path,
    )


def run_stage(
    *,
    contract_path: Path,
    cache_roots: Mapping[str, Path],
    runtime_evidence_paths: Mapping[str, Path],
    stage: str,
    device: torch.device,
) -> dict[str, Any]:
    """Compatibility entrypoint for the non-optimizing frozen holdout stage."""
    if stage != HOLDOUT_SPLIT:
        raise JointCalibrationRunError(
            "train execution is two-phase: prepare-train then finalize-train"
        )
    return run_holdout_stage(
        contract_path=contract_path,
        cache_roots=cache_roots,
        runtime_evidence_paths=runtime_evidence_paths,
        device=device,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-train")
    prepare.add_argument("--contract", type=Path, required=True)
    prepare.add_argument("--cache", action="append", required=True, metavar="MODEL=PATH")
    prepare.add_argument("--device", default="cuda")

    finalize = subparsers.add_parser("finalize-train")
    finalize.add_argument("--contract", type=Path, required=True)
    finalize.add_argument("--cache", action="append", required=True, metavar="MODEL=PATH")
    finalize.add_argument("--runtime-evidence", action="append", required=True, metavar="MODEL=PATH")
    finalize.add_argument("--candidate-verification", type=Path)
    finalize.add_argument("--device", default="cuda")

    holdout = subparsers.add_parser("holdout")
    holdout.add_argument("--contract", type=Path, required=True)
    holdout.add_argument("--cache", action="append", required=True, metavar="MODEL=PATH")
    holdout.add_argument("--runtime-evidence", action="append", required=True, metavar="MODEL=PATH")
    holdout.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        cache_roots = _parse_model_paths(args.cache, label="cache")
        if args.command == "prepare-train":
            result = prepare_train_candidate(
                contract_path=args.contract,
                cache_roots=cache_roots,
                device=torch.device(args.device),
            )
        else:
            runtime_evidence_paths = _parse_model_paths(
                args.runtime_evidence, label="runtime-evidence"
            )
            if args.command == "finalize-train":
                result = finalize_train_candidate(
                    contract_path=args.contract,
                    cache_roots=cache_roots,
                    runtime_evidence_paths=runtime_evidence_paths,
                    device=torch.device(args.device),
                    candidate_verification_path=args.candidate_verification,
                )
            else:
                result = run_holdout_stage(
                    contract_path=args.contract,
                    cache_roots=cache_roots,
                    runtime_evidence_paths=runtime_evidence_paths,
                    device=torch.device(args.device),
                )
    except (JointCalibrationRunError, JointTrainingContractError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
