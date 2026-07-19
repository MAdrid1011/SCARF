#!/usr/bin/env python3
"""Run one target-free normalized L0/L1 compact packet quality pilot.

The pilot is development-only. It uses the paper's T=4 four-probe L0 set and
an explicit adaptive 15-anchor L1 engineering closure, constructs non-probe
contributors from selected anchors, and keeps a failed preflight on native
Full. It does not claim sparse S2/S3 savings, timing, or paper-result
eligibility.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.probe_first_schedule import (
    ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    build_incremental_probe_first_plan,
)
from saes.evaluation_disjoint_l1_calibration import (
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
    load_frozen_v15_threshold,
    load_frozen_v16_threshold,
)
from saes.classic_backend import (
    freeze_classic_backend_identity,
    resolve_classic_backend_contract,
)
from saes.incremental_selected_output_execution import (
    RAW_HEAD_EXECUTION_CONTRACT,
    validate_native_dense_head_execution_evidence,
)
from saes.guarded_selected_route import (
    ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
)
from saes.packed_l0_l1_materializer import (
    CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
)
from saes.progressive_saes import PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS
from saes.sparse_gaussian_consumer import PackedGaussianAttributes
from scripts.saes_incremental_selected_output_audit import (
    DEPTH_THRESHOLD,
    FEATURE_THRESHOLD,
    TILE_SIZE,
    _attribute_equivalence,
    _capture_guarded_incremental_packed_adapter,
    _capture_s1_s2_without_dense_adapter,
    _packed_attributes_to_device,
    _release_cuda_cache,
)
from scripts.saes_selected_output_quality_gate import (
    _mean_metrics,
    _quality_verdict,
    _view_metrics,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution
from scripts.saes_sparse_consumer_render_audit import (
    _target_cameras,
    render_packed_gaussians,
)
from scripts.saes_sparse_packet_quality_pilot import (
    DATASET,
    MODEL,
    ROOT as PACKET_ROOT,
    SAMPLE_INDEX,
    SEED,
    _render_dense_baseline,
    _take_target_rgb_for_metrics,
)


PILOT_KIND = "saes_paper_normalized_l0_l1_15_anchor_compact_packet_quality_pilot"
DECISION_SEMANTICS = PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS
L1_ANCHOR_SEMANTICS = ADAPTIVE_L1_15_ANCHOR_SEMANTICS
COMPACT_AGGREGATION = CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION
TARGET_FREE_AUDIT_KIND = "saes_incremental_selected_output_packed_adapter_audit"
TARGET_FREE_AUDIT_STATUS = "PASS"
CLASSIC_MODELS = ("transplat", "mvsplat")


def _model_name(value: Any) -> str:
    if value not in CLASSIC_MODELS:
        raise ValueError("compact packet pilot requires a supported classic model")
    return str(value)


def _require_nonnegative_sample_index(sample_index: Any) -> int:
    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
    ):
        raise ValueError("sample_index must be a non-negative integer")
    return sample_index


def _require_positive_sample_count(sample_count: Any) -> int:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError("native_sample_count must be a positive integer")
    return sample_count


def _canonical_source_selection(
    *,
    index_path: Path,
    source_index_sha256: str,
    sample_index: int,
) -> dict[str, Any]:
    """Resolve one stable source ordinal without touching target-side data."""
    from scripts.compile_protocol import canonicalize_index

    rows, _summary = canonicalize_index(index_path, source_index_sha256)
    matches = [row for row in rows if row["sample_index"] == sample_index]
    if len(matches) != 1:
        raise ValueError("canonical evaluation index has no requested source sample")
    selection = matches[0]
    return {
        "source_sample_index": selection["sample_index"],
        "scene": selection["scene"],
        "context_indices": list(selection["context_indices"]),
        "target_indices": list(selection["target_indices"]),
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loaded_module_source_identity(
    module_name: str, *, expected_root: Path, label: str
) -> dict[str, str]:
    """Bind a runtime module to the selected classic-model source tree."""

    module = sys.modules.get(module_name)
    source = getattr(module, "__file__", None) if module is not None else None
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"compact packet pilot has no loaded {label} module source")
    source_path = Path(source).resolve()
    expected_root = Path(expected_root).resolve()
    if expected_root not in source_path.parents or not source_path.is_file():
        raise RuntimeError(
            f"compact packet pilot loaded {label} from a foreign source tree"
        )
    return {
        "module": module_name,
        "path": source_path.relative_to(ROOT).as_posix(),
        "sha256": _sha256_file(source_path),
    }


def _resolve_decoder_gaussians_type(
    decoder: Any, *, backend_contract: Any
) -> tuple[type[Any], dict[str, Any]]:
    """Resolve the actual decoder's Gaussian container without using global ``src``."""

    if not callable(getattr(decoder, "forward", None)):
        raise TypeError("compact packet pilot requires a native decoder with forward")
    decoder_type = type(decoder)
    decoder_module_name = decoder_type.__module__
    expected_root = Path(backend_contract.checkpoint).resolve().parent.parent
    decoder_module = sys.modules.get(decoder_module_name)
    gaussians_type = (
        getattr(decoder_module, "Gaussians", None)
        if decoder_module is not None
        else None
    )
    if not isinstance(gaussians_type, type):
        raise RuntimeError("compact packet pilot decoder module has no Gaussians type")
    fields = getattr(gaussians_type, "__dataclass_fields__", None)
    if not isinstance(fields, Mapping) or set(fields) != {
        "means",
        "covariances",
        "harmonics",
        "opacities",
    }:
        raise RuntimeError("compact packet pilot decoder Gaussians type changed")
    return gaussians_type, {
        "model": backend_contract.model,
        "decoder_class": f"{decoder_type.__module__}.{decoder_type.__qualname__}",
        "decoder_module": _loaded_module_source_identity(
            decoder_module_name, expected_root=expected_root, label="decoder"
        ),
        "gaussians_class": (
            f"{gaussians_type.__module__}.{gaussians_type.__qualname__}"
        ),
        "gaussians_module": _loaded_module_source_identity(
            gaussians_type.__module__, expected_root=expected_root, label="Gaussians"
        ),
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _read_target_free_quality_audit(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "target-free quality audit is unavailable or invalid"
        ) from error
    if not isinstance(record, dict):
        raise ValueError("target-free quality audit must be an object")
    recorded_sha256 = record.pop("sha256", None)
    _require_sha256(recorded_sha256, label="target-free quality audit SHA256")
    if recorded_sha256 != _canonical_sha256(record):
        raise ValueError("target-free quality audit SHA256 is invalid")
    return {**record, "sha256": recorded_sha256}, _sha256_file(path)


def _frozen_calibration_identity(
    calibration: Mapping[str, Any], *, require_native_dense_head_execution: bool = False
) -> dict[str, Any]:
    required = {"sha256", "threshold_value", "acid_binding", "application"}
    if not isinstance(calibration, Mapping) or not required.issubset(calibration):
        raise ValueError("frozen calibration is incomplete")
    _require_sha256(calibration["sha256"], label="frozen calibration SHA256")
    identity = {key: calibration[key] for key in sorted(required)}
    if require_native_dense_head_execution:
        if (
            calibration.get("raw_head_execution_contract")
            != RAW_HEAD_EXECUTION_CONTRACT
        ):
            raise ValueError("V16 calibration native dense head contract changed")
        identity["raw_head_execution_contract"] = RAW_HEAD_EXECUTION_CONTRACT
    return identity


def _target_free_audit_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _require_exact_audited_context_input_identity(
    audit: Mapping[str, Any], input_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless the route input is exactly the audited sidecar."""
    expected = audit.get("input_identity")
    if not isinstance(expected, Mapping) or dict(expected) != dict(input_identity):
        raise ValueError("target-free quality audit input identity changed")
    return dict(input_identity)


def _validate_target_free_context_input(
    audit: Mapping[str, Any], input_root: Path, *, model_name: str = MODEL
) -> dict[str, Any]:
    """Validate the context-only root before any native data loader can run."""
    from data.context_only_audit_input import validate_context_only_audit_input

    model_name = _model_name(model_name)
    return _require_exact_audited_context_input_identity(
        audit, validate_context_only_audit_input(input_root, model=model_name)
    )


def _validate_target_free_quality_audit(
    path: Path,
    *,
    scene: str,
    context_indices: list[int],
    checkpoint_sha256: str,
    v15_calibration: Mapping[str, Any],
    v16_calibration: Mapping[str, Any],
    source_selection_mask_sha256: str,
    selected_output_mask_sha256: str,
    packed_source_trace_sha256: str,
    sample_index: int = SAMPLE_INDEX,
    model_name: str = MODEL,
    classic_backend_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require a frozen target-free audit for the exact packet about to render."""
    sample_index = _require_nonnegative_sample_index(sample_index)
    model_name = _model_name(model_name)
    audit, file_sha256 = _read_target_free_quality_audit(path)
    if (
        audit.get("kind") != TARGET_FREE_AUDIT_KIND
        or audit.get("status") != TARGET_FREE_AUDIT_STATUS
        or audit.get("paper_result_eligible") is not False
        or audit.get("model") != model_name
        or audit.get("dataset") != DATASET
        or audit.get("scene") != scene
    ):
        raise ValueError("target-free quality audit identity changed")
    for field in (
        "target_mapping_present",
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
    ):
        if audit.get(field) is not False:
            raise ValueError(f"target-free quality audit crossed {field}")
    access_evidence = audit.get("access_evidence")
    if (
        not isinstance(access_evidence, Mapping)
        or access_evidence.get("target_mapping_present") is not False
        or access_evidence.get("target_rgb_accessed") is not False
        or access_evidence.get("target_camera_metadata_accessed") is not False
        or access_evidence.get("target_index_accessed") is not False
        or access_evidence.get("frozen_calibrations_target_free") is not True
        or access_evidence.get("frozen_calibrations_verified_before_encoder_execution")
        is not True
        or access_evidence.get("packed_guard_executed_before_target_access") is not True
    ):
        raise ValueError("target-free quality audit access evidence changed")
    if audit.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("target-free quality audit checkpoint changed")
    execution = audit.get("execution")
    execution_model = execution.get("model") if isinstance(execution, Mapping) else None
    if (
        not isinstance(execution, Mapping)
        or execution.get("checkpoint_sha256") != checkpoint_sha256
        or execution.get("encoder_only") is not True
        or execution.get("decoder_constructed") is not False
        or (model_name == "mvsplat" and execution_model != "mvsplat")
        or (model_name == "transplat" and execution_model not in (None, "transplat"))
    ):
        raise ValueError("target-free quality audit execution identity changed")
    if model_name == "mvsplat":
        if not isinstance(classic_backend_identity, Mapping) or execution.get(
            "classic_backend_identity"
        ) != dict(classic_backend_identity):
            raise ValueError(
                "target-free quality audit classic backend identity changed"
            )
    elif classic_backend_identity is not None:
        raise ValueError("TranSplat quality pilot must not carry a backend identity")
    boundary = audit.get("execution_boundary")
    expected_coordinate_semantics = (
        classic_backend_identity.get("coordinate_semantics")
        if isinstance(classic_backend_identity, Mapping)
        else None
    )
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("renderer_executed") is not False
        or boundary.get("quality_metrics_computed") is not False
        or boundary.get("compact_nonzero_materialization_enabled") is not True
        or boundary.get("frozen_l1_15_packed_guard_executed") is not True
        or boundary.get("frozen_calibrations_verified_before_encoder_execution")
        is not True
        or (
            model_name == "mvsplat"
            and boundary.get("backend_coordinate_semantics")
            != expected_coordinate_semantics
        )
    ):
        raise ValueError("target-free quality audit crossed the rendering boundary")
    input_identity = audit.get("input_identity")
    if (
        not isinstance(input_identity, Mapping)
        or input_identity.get("scene") != scene
        or input_identity.get("source_sample_index") != sample_index
        or input_identity.get("target_rgb_accessed") is not False
        or input_identity.get("target_camera_metadata_accessed") is not False
        or input_identity.get("context_indices") != context_indices
    ):
        raise ValueError("target-free quality audit input identity changed")
    expected_mechanism = {
        "decision_semantics": DECISION_SEMANTICS,
        "l1_anchor_semantics": L1_ANCHOR_SEMANTICS,
        "l1_anchor_count": 15,
        "execution_policy": ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
    }
    if audit.get("frozen_l1_15_mechanism") != expected_mechanism:
        raise ValueError("target-free quality audit mechanism changed")
    expected_route_plan = {
        "decision_semantics": DECISION_SEMANTICS,
        "l1_anchor_semantics": L1_ANCHOR_SEMANTICS,
        "l1_anchor_count": 15,
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": DEPTH_THRESHOLD,
    }
    route_plan = audit.get("route_plan")
    if not isinstance(route_plan, Mapping) or any(
        route_plan.get(key) != value for key, value in expected_route_plan.items()
    ):
        raise ValueError("target-free quality audit route plan mechanism changed")
    expected_v15 = _frozen_calibration_identity(v15_calibration)
    expected_v16 = _frozen_calibration_identity(
        v16_calibration, require_native_dense_head_execution=True
    )
    if audit.get("v15_calibration") != expected_v15:
        raise ValueError("target-free quality audit V15 calibration changed")
    if audit.get("v16_calibration") != expected_v16:
        raise ValueError("target-free quality audit V16 calibration changed")
    raw_head = audit.get("raw_head")
    if not isinstance(raw_head, Mapping):
        raise ValueError("target-free quality audit has no raw-head evidence")
    native_dense_execution = raw_head.get("native_dense_head_execution")
    if not isinstance(native_dense_execution, Mapping):
        raise ValueError("target-free quality audit has no native dense head evidence")
    initial_dense_execution = validate_native_dense_head_execution_evidence(
        native_dense_execution.get("initial"), expected_phase_count=3
    )
    guarded_dense_execution = validate_native_dense_head_execution_evidence(
        native_dense_execution.get("guarded")
    )
    if (
        initial_dense_execution["raw_head_execution_contract"]
        != expected_v16["raw_head_execution_contract"]
        or guarded_dense_execution["raw_head_execution_contract"]
        != expected_v16["raw_head_execution_contract"]
    ):
        raise ValueError("target-free quality audit native dense contract changed")
    packed_adapter = audit.get("packed_adapter")
    if not isinstance(packed_adapter, Mapping):
        raise ValueError("target-free quality audit has no packed source trace")
    final_source_trace = packed_adapter.get("final_packet_source_trace")
    if (
        not isinstance(final_source_trace, Mapping)
        or final_source_trace.get("raw_head_execution_contract")
        != expected_v16["raw_head_execution_contract"]
        or validate_native_dense_head_execution_evidence(
            final_source_trace.get("native_dense_head_execution")
        )
        != guarded_dense_execution
    ):
        raise ValueError("target-free quality audit packed native dense trace changed")
    route_binding = audit.get("route_binding")
    expected_route_binding = {
        "source_selection_mask_sha256": source_selection_mask_sha256,
        "selected_output_mask_sha256": selected_output_mask_sha256,
        "packed_source_trace_sha256": packed_source_trace_sha256,
    }
    if not isinstance(route_binding, Mapping) or any(
        route_binding.get(key) != value for key, value in expected_route_binding.items()
    ):
        raise ValueError("target-free quality audit route binding changed")
    for label, value in expected_route_binding.items():
        _require_sha256(value, label=f"target-free quality audit {label}")
    return {
        "path": _target_free_audit_path(path),
        "file_sha256": file_sha256,
        "record_sha256": audit["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "model": model_name,
        "sample_index": sample_index,
        "v15_calibration_sha256": expected_v15["sha256"],
        "v16_calibration_sha256": expected_v16["sha256"],
        **expected_route_binding,
    }


def _paper_identity(
    v15_calibration: Mapping[str, Any],
    v16_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": "saes-paper-normalized-l1-15-adaptive-packet-pilot-v9",
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": DEPTH_THRESHOLD,
        "decision_semantics": DECISION_SEMANTICS,
        "feature_scale_interpretation": (
            "unit-normalized-probe-vector-standard-deviation-engineering-closure"
        ),
        "l0_anchor_count": 4,
        "l1_anchor_count": 15,
        "l1_anchor_semantics": L1_ANCHOR_SEMANTICS,
        "l1_retention_interpretation": (
            "engineering-legacy12-plus-three-center-anchors-not-specified-by-paper"
        ),
        "nonzero_policy": ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
        "raw_head_execution_contract": RAW_HEAD_EXECUTION_CONTRACT,
        "adaptive_l1_absolute_residual_calibration_sha256": v15_calibration["sha256"],
        "adaptive_l1_absolute_residual_maximum": v15_calibration["threshold_value"],
        "aggregation": COMPACT_AGGREGATION,
        "coverage_certificate": (
            "selected-only-direct-geometry-spatial-attribute-l0-depth-continuity-intra-tile-2sigma-v1"
        ),
        "quality_tolerances": {
            "psnr_max_drop": 0.15,
            "ssim_max_drop": 0.005,
            "lpips_max_increase": 0.005,
        },
    }
    identity.update(
        {
            "selected_anchor_v4_attribute_loo_calibration_sha256": v16_calibration[
                "sha256"
            ],
            "selected_anchor_v4_attribute_loo_maximum_risk": v16_calibration[
                "threshold_value"
            ],
        }
    )
    return {**identity, "sha256": _canonical_sha256(identity)}


def _route_summary(capture: Mapping[str, Any]) -> dict[str, Any]:
    route = capture["guarded_route"]
    preflight = capture["compact_materialization_preflight"]
    if preflight is None:
        raise RuntimeError("compact packet pilot has no materialization preflight")

    def compact_events(events: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in events.items()
            if key != "full_passthrough_dense_slots"
        }

    return {
        "base_guard_route": compact_events(capture["base_guarded_route"].events),
        "final_route": compact_events(route.events),
        "preflight": preflight.events,
        "final_output_descriptor_count": int(route.selected_output_mask.sum().item()),
        "raw_head_request_descriptor_count": int(
            route.raw_head_request_mask.sum().item()
        ),
    }


def _route_diagnostics(plan: Any, capture: Mapping[str, Any]) -> dict[str, Any]:
    """Persist scalar route evidence without serializing tile-local attributes."""
    scores = sorted(float(record["feature_score"]) for record in plan.tile_trace)
    depth_uniform = sum(bool(record["depth_uniform"]) for record in plan.tile_trace)
    pre_guard = {"L0": 0, "L1": 0, "Full": 0}
    for record in plan.tile_trace:
        pre_guard[str(record["pre_guard_route"])] += 1
    base_route = capture["base_guarded_route"]
    guard_failures = {
        "materialization": 0,
        "probe_cross_check": 0,
        "context_safety": {},
    }
    for tile in base_route.tile_trace:
        for check in tile["guard_checks"]:
            if check["materialization"]["passed"] is not True:
                guard_failures["materialization"] += 1
            cross = check["probe_cross_check"]
            if cross["checked"] and cross["passed"] is not True:
                guard_failures["probe_cross_check"] += 1
            context = check["context_safety"]
            if context["passed"] is not True:
                reason = str(context["reason"])
                reasons = guard_failures["context_safety"]
                reasons[reason] = reasons.get(reason, 0) + 1

    def quantile(fraction: float) -> float:
        return scores[round((len(scores) - 1) * fraction)]

    return {
        "tile_count": len(scores),
        "feature_score": {
            "minimum": quantile(0.0),
            "p50": quantile(0.50),
            "p95": quantile(0.95),
            "maximum": quantile(1.0),
            "below_fixed_tau_f": sum(score < FEATURE_THRESHOLD for score in scores),
        },
        "depth_uniform_tile_count": depth_uniform,
        "pre_guard_route_counts": pre_guard,
        "guard_failures": guard_failures,
    }


def _posthoc_dense_assignment_mixture_reference(
    *,
    model: Any,
    dense_gaussians: Any,
    packet_color: torch.Tensor,
    final_packed: PackedGaussianAttributes,
    planning: Mapping[str, torch.Tensor],
    context: Mapping[str, Any],
    target_cameras: Mapping[str, torch.Tensor],
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Compare a committed packet to the legacy dense mixture without target RGB.

    Dense attributes are a posthoc reference only.  They are constructed after
    the compact packet decoder output has been fixed and cannot influence its
    route, assignments, or materialized attributes.
    """
    from saes.progressive_saes import apply_progressive_saes
    from scripts.saes_target_free_materialization_audit import _clone_gaussians

    reference = _clone_gaussians(dense_gaussians)
    height, width = image_shape
    modified, stats, _continue = apply_progressive_saes(
        reference,
        height,
        width,
        tile_size=TILE_SIZE,
        gpp=1,
        feature_var_threshold=FEATURE_THRESHOLD,
        depth_std_threshold=DEPTH_THRESHOLD,
        features=planning["features"].to(reference.means.device),
        depths=planning["depths"].to(reference.means.device),
        view_count=int(planning["features"].shape[1]),
        materialization="representative",
        decision_semantics=DECISION_SEMANTICS,
        beta_x=0.50,
        beta_f=0.10,
        beta_d=1.00,
        context_extrinsics=context["extrinsics"],
        context_intrinsics=context["intrinsics"],
        ray_depth_mode="euclidean",
        materialization_guard=False,
        context_safety_guard=False,
        require_deletion_certificate=False,
    )
    reference_color = _render_dense_baseline(
        model,
        reference,
        target=target_cameras,
        image_shape=image_shape,
    )
    delta = (packet_color - reference_color).abs()
    attribute_equivalence = _attribute_equivalence(
        final_packed, reference, final_packed.dense_slots
    )
    return {
        "posthoc_dense_s3_reference": True,
        "candidate_inputs_influenced_by_reference": False,
        "target_rgb_accessed": False,
        "dense_modified_count": int(modified.sum().item()),
        "dense_route_counts": {
            "L0": int(stats["level0_tiles"]),
            "L1": int(stats["level1_tiles"]),
            "Full": int(stats["full_tiles"]),
        },
        "selected_anchor_attribute_equivalence": attribute_equivalence,
        "packet_vs_dense_mixture_render": {
            "mean_absolute_delta": float(delta.mean().item()),
            "maximum_absolute_delta": float(delta.max().item()),
        },
    }


def _posthoc_dense_domain_coverage_reference(
    *,
    dense_gaussians: Any,
    final_packed: PackedGaussianAttributes,
    route: Any,
    context: Mapping[str, torch.Tensor],
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Audit committed compact tiles against dense S3 domains without targets.

    This runs only after the final packet is materialized. Dense descriptors
    provide a diagnostic reference and cannot influence routing, assignments,
    coverage closure, or the committed packet.
    """
    from saes.projected_domain_coverage_audit import (
        audit_projected_dense_domain_coverage,
    )

    height, width = image_shape
    dense_means = getattr(dense_gaussians, "means", None)
    dense_covariances = getattr(dense_gaussians, "covariances", None)
    dense_opacities = getattr(dense_gaussians, "opacities", None)
    if (
        not torch.is_tensor(dense_means)
        or not torch.is_tensor(dense_covariances)
        or not torch.is_tensor(dense_opacities)
        or dense_means.ndim != 3
        or dense_means.shape[0] != 1
        or dense_covariances.shape != (*dense_means.shape[:2], 3, 3)
        or dense_opacities.shape != dense_means.shape[:2]
    ):
        raise RuntimeError("compact packet pilot dense reference has an invalid layout")
    slot_to_index = {
        int(slot): index
        for index, slot in enumerate(final_packed.dense_slots.detach().cpu().tolist())
    }
    route_records = route.tile_trace
    primary = ((0, 0), (0, 3), (3, 0), (3, 3))
    l1_anchor_semantics = route.events.get("l1_anchor_semantics")
    if l1_anchor_semantics != ADAPTIVE_L1_15_ANCHOR_SEMANTICS:
        raise RuntimeError("compact packet pilot requires the adaptive L1-15 layout")
    aggregates = {
        "compact_tile_count": 0,
        "valid_tile_count": 0,
        "invalid_tile_count": 0,
        "dense_descriptor_count": 0,
        "candidate_descriptor_count": 0,
        "active_dense_descriptor_count": 0,
        "hole_count": 0,
        "dense_optical_mass": 0.0,
        "contained_optical_mass": 0.0,
        "reason_counts": {},
    }
    for record in route_records:
        if record.get("final_route") not in {"L0", "L1"}:
            continue
        view = int(record["view"])
        tile_y = int(record["tile_y"])
        tile_x = int(record["tile_x"])
        if record.get("final_route") == "L0":
            anchors = primary
        else:
            raw_anchors = record.get("retained_local_positions")
            if not isinstance(raw_anchors, list) or len(raw_anchors) != 15:
                raise RuntimeError(
                    "compact packet pilot has no bound adaptive L1 anchors"
                )
            try:
                anchors = tuple(
                    (int(position[0]), int(position[1])) for position in raw_anchors
                )
            except (IndexError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "compact packet pilot has invalid adaptive L1 anchors"
                ) from error
            if (
                len(set(anchors)) != 15
                or anchors[:4] != primary
                or any(
                    not 0 <= row < 4 or not 0 <= column < 4 for row, column in anchors
                )
            ):
                raise RuntimeError(
                    "compact packet pilot adaptive L1 anchors violate their contract"
                )
        anchor_slots = [
            view * height * width
            + (tile_y * 4 + local_y) * width
            + tile_x * 4
            + local_x
            for local_y, local_x in anchors
        ]
        nonprobe_slots = [
            view * height * width
            + (tile_y * 4 + local_y) * width
            + tile_x * 4
            + local_x
            for local_y in range(4)
            for local_x in range(4)
            if (local_y, local_x) not in set(anchors)
        ]
        if any(slot not in slot_to_index for slot in anchor_slots):
            raise RuntimeError("compact packet pilot lacks a retained compact anchor")
        candidate_indices = torch.tensor(
            [slot_to_index[slot] for slot in anchor_slots],
            device=final_packed.means.device,
            dtype=torch.long,
        )
        dense_indices = torch.tensor(
            nonprobe_slots,
            device=dense_means.device,
            dtype=torch.long,
        )
        audit = audit_projected_dense_domain_coverage(
            dense_means=dense_means[0, dense_indices],
            dense_covariances=dense_covariances[0, dense_indices],
            dense_opacities=dense_opacities[0, dense_indices],
            candidate_means=final_packed.means[candidate_indices],
            candidate_covariances=final_packed.covariances[candidate_indices],
            candidate_opacities=final_packed.opacities[candidate_indices],
            context_extrinsic=context["extrinsics"][0, view],
            context_intrinsic=context["intrinsics"][0, view],
        )
        aggregates["compact_tile_count"] += 1
        aggregates["dense_descriptor_count"] += audit.dense_descriptor_count
        aggregates["candidate_descriptor_count"] += audit.candidate_descriptor_count
        if not audit.valid:
            aggregates["invalid_tile_count"] += 1
            reason = audit.reason or "unknown"
            aggregates["reason_counts"][reason] = (
                aggregates["reason_counts"].get(reason, 0) + 1
            )
            continue
        aggregates["valid_tile_count"] += 1
        aggregates[
            "active_dense_descriptor_count"
        ] += audit.active_dense_descriptor_count
        aggregates["hole_count"] += int(audit.hole_count or 0)
        aggregates["dense_optical_mass"] += float(audit.dense_optical_mass or 0.0)
        aggregates["contained_optical_mass"] += float(
            audit.contained_optical_mass or 0.0
        )
    mass = aggregates["dense_optical_mass"]
    active = aggregates["active_dense_descriptor_count"]
    aggregates["mass_weighted_recall"] = (
        aggregates["contained_optical_mass"] / mass if mass > 0.0 else None
    )
    aggregates["count_recall"] = (
        (active - aggregates["hole_count"]) / active if active > 0 else None
    )
    return {
        "posthoc_dense_s3_reference": True,
        "candidate_inputs_influenced_by_reference": False,
        "target_rgb_accessed": False,
        "audit_kind": "actual-dense-nonprobe-domain-coverage-after-commit",
        "per_tile_records_persisted": False,
        "aggregate": aggregates,
    }


def _load_native_target_batch_after_packet_gate(
    loader: Any,
    model_bundle: Any,
    *,
    scene: str,
    context_indices: list[int],
    target_indices: list[int],
    execution_index: int = SAMPLE_INDEX,
    native_sample_count: int = 1,
) -> dict[str, Any]:
    """Open target-side data only after the audited compact packet is accepted."""
    execution_index = _require_nonnegative_sample_index(execution_index)
    native_sample_count = _require_positive_sample_count(native_sample_count)
    if execution_index >= native_sample_count:
        raise ValueError("execution_index must be smaller than native_sample_count")
    data = loader.load_data(
        model_bundle,
        dataset_name=DATASET,
        num_samples=native_sample_count,
        sample_index=execution_index,
    )
    native_batch = data.batch
    if native_batch.get("scene") != [scene]:
        raise RuntimeError(
            "native target batch scene does not match the audited context"
        )
    native_context = native_batch.pop("context", None)
    if not isinstance(native_context, Mapping) or not torch.is_tensor(
        native_context.get("index")
    ):
        raise RuntimeError("native target batch has no discardable context identity")
    native_context_indices = [
        int(value)
        for value in native_context["index"][0].detach().to(device="cpu").tolist()
    ]
    if native_context_indices != context_indices:
        raise RuntimeError(
            "native target batch context does not match the audited context"
        )
    target = native_batch.get("target")
    if not isinstance(target, dict):
        raise RuntimeError("compact packet pilot requires a native target mapping")
    native_target_indices = target.get("index")
    if not torch.is_tensor(native_target_indices):
        raise RuntimeError("native target batch has no target index identity")
    if [int(value) for value in native_target_indices[0].tolist()] != target_indices:
        raise RuntimeError(
            "native target batch target indices do not match the canonical selection"
        )
    return {"scene": [scene], "target": target}


def collect_paper_compact_packet_pilot(
    *,
    device: torch.device,
    posthoc_domain_audit_only: bool = False,
    v15_calibration_record: Path,
    v16_calibration_record: Path,
    target_free_audit_artifact: Path,
    target_free_input_root: Path,
    sample_index: int = SAMPLE_INDEX,
    execution_index: int | None = None,
    native_sample_count: int | None = None,
    acid_plan_path: Path = DEFAULT_PLAN_PATH,
    acid_materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
    model_name: str = MODEL,
) -> dict[str, Any]:
    """Run one packet pilot, optionally stopping at the target-free domain audit."""
    if not isinstance(posthoc_domain_audit_only, bool):
        raise TypeError("posthoc domain audit mode must be boolean")
    sample_index = _require_nonnegative_sample_index(sample_index)
    model_name = _model_name(model_name)
    from integration import create_model_loader, load_context_only_audit_data
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.result_record import cached_sha256_file, source_identity

    backend_contract = resolve_classic_backend_contract(model_name, ROOT)
    experiment = resolve_experiment(model_name, DATASET, PACKET_ROOT)
    selection = resolve_claim_selection(model_name, DATASET, PACKET_ROOT)
    if (
        experiment.model != backend_contract.model
        or experiment.dataset != backend_contract.dataset
        or experiment.experiment != backend_contract.experiment
        or experiment.checkpoint.resolve() != backend_contract.checkpoint
    ):
        raise RuntimeError("compact packet pilot classic backend identity changed")
    classic_backend_identity = (
        freeze_classic_backend_identity(backend_contract)
        if model_name == "mvsplat"
        else None
    )
    backend_binding = {
        "model": backend_contract.model,
        "raw_head_module": backend_contract.raw_head_module,
        "gaussian_adapter_module": backend_contract.gaussian_adapter_module,
        "decoder_module": backend_contract.decoder_module,
        "coordinate_semantics": backend_contract.coordinate_semantics,
    }
    if classic_backend_identity is not None:
        backend_binding["classic_backend_identity"] = classic_backend_identity
    execution_index = (
        sample_index
        if execution_index is None
        else _require_nonnegative_sample_index(execution_index)
    )
    native_sample_count = (
        selection.sample_count
        if native_sample_count is None
        else _require_positive_sample_count(native_sample_count)
    )
    if execution_index >= native_sample_count:
        raise ValueError("execution_index must be smaller than native_sample_count")
    checkpoint_sha256 = cached_sha256_file(experiment.checkpoint)
    v15_calibration = load_frozen_v15_threshold(
        v15_calibration_record,
        checkpoint_path=experiment.checkpoint,
        application_model=model_name,
        plan_path=acid_plan_path,
        materialization_root=acid_materialization_root,
    )
    v16_calibration = load_frozen_v16_threshold(
        v16_calibration_record,
        checkpoint_path=experiment.checkpoint,
        application_model=model_name,
        v15_record_path=v15_calibration_record,
        plan_path=acid_plan_path,
        materialization_root=acid_materialization_root,
    )
    if v15_calibration["sha256"] != v16_calibration["base_v15_sha256"]:
        raise RuntimeError("V16 calibration does not bind the loaded V15 calibration")
    paper_identity = _paper_identity(v15_calibration, v16_calibration)
    target_free_audit, target_free_audit_file_sha256 = _read_target_free_quality_audit(
        target_free_audit_artifact
    )
    context_input_identity = _validate_target_free_context_input(
        target_free_audit, target_free_input_root, model_name=model_name
    )
    loader = create_model_loader(model_name)
    bundle = loader.load_model(
        str(experiment.checkpoint),
        device=device,
        experiment_name=experiment.experiment,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        hydra_overrides=experiment.hydra_overrides,
    )
    if bundle.decoder is None:
        raise RuntimeError("compact packet pilot requires the native packet decoder")
    context_data = load_context_only_audit_data(
        loader,
        bundle,
        input_root=target_free_input_root,
        model_name=model_name,
    )
    if "target" in context_data.batch:
        raise RuntimeError(
            "compact packet pilot context route exposed a target mapping"
        )
    loaded_context_identity = context_data.batch.get("calibration")
    if not isinstance(loaded_context_identity, Mapping) or any(
        loaded_context_identity.get(key) != value
        for key, value in context_input_identity.items()
    ):
        raise RuntimeError("context-only loader changed the audited input identity")
    scene = str(context_data.batch["scene"][0])
    if scene != context_input_identity["scene"]:
        raise RuntimeError("context-only loader scene does not match the audited input")
    model = bundle.model
    loaded_device = bundle.device
    model.eval()
    decoder_gaussians_type, decoder_binding = _resolve_decoder_gaussians_type(
        model.decoder, backend_contract=backend_contract
    )

    context = {
        key: value.to(loaded_device) if torch.is_tensor(value) else value
        for key, value in context_data.batch["context"].items()
    }
    if context["image"].shape[0] != 1:
        raise RuntimeError("compact packet pilot requires batch size one")
    _, _views, _, height, width = context["image"].shape
    context_indices = [
        int(value) for value in context["index"][0].detach().to(device="cpu").tolist()
    ]

    with strict_fp32_convolution_execution() as numerical_execution:
        planning = _capture_s1_s2_without_dense_adapter(model, context)
        _release_cuda_cache(loaded_device)
        plan = build_incremental_probe_first_plan(
            planning["features"],
            planning["depths"],
            height=height,
            width=width,
            tile_size=TILE_SIZE,
            feature_threshold=FEATURE_THRESHOLD,
            depth_threshold=DEPTH_THRESHOLD,
            decision_semantics=DECISION_SEMANTICS,
            l1_anchor_semantics=L1_ANCHOR_SEMANTICS,
        )
        capture = _capture_guarded_incremental_packed_adapter(
            model,
            context,
            plan=plan,
            compact_nonzero_materialization=True,
            compact_execution_policy=ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
            adaptive_l1_maximum_leave_one_out_residual=v15_calibration[
                "threshold_value"
            ],
            selected_anchor_v4_attribute_loo_maximum_risk=v16_calibration[
                "threshold_value"
            ],
            require_native_dense_head_execution=True,
            model_name=model_name,
        )
        native_dense_head_execution = capture.get("native_dense_head_execution")
        if not isinstance(native_dense_head_execution, Mapping):
            raise RuntimeError(
                "quality capture did not retain native dense head evidence"
            )
        initial_native_dense_execution = validate_native_dense_head_execution_evidence(
            native_dense_head_execution.get("initial"), expected_phase_count=3
        )
        guarded_native_dense_execution = validate_native_dense_head_execution_evidence(
            native_dense_head_execution.get("guarded")
        )
        preflight = capture["compact_materialization_preflight"]
        if preflight is None:
            raise RuntimeError(
                "compact packet pilot did not retain materialization evidence"
            )
        final_packed = capture["final_packed"]
        if not isinstance(final_packed, PackedGaussianAttributes):
            raise RuntimeError("compact packet pilot did not produce packed attributes")
        final_source_trace = final_packed.source_trace
        if (
            not isinstance(final_source_trace, Mapping)
            or final_source_trace.get("raw_head_execution_contract")
            != RAW_HEAD_EXECUTION_CONTRACT
            or validate_native_dense_head_execution_evidence(
                final_source_trace.get("native_dense_head_execution")
            )
            != guarded_native_dense_execution
        ):
            raise RuntimeError("quality packet native dense head trace changed")
        compact_anchor_count = int(preflight.update_dense_slots.numel())
        route_summary = _route_summary(capture)
        route_diagnostics = _route_diagnostics(plan, capture)
        quality_gate = _validate_target_free_quality_audit(
            target_free_audit_artifact,
            scene=scene,
            context_indices=context_indices,
            checkpoint_sha256=checkpoint_sha256,
            v15_calibration=v15_calibration,
            v16_calibration=v16_calibration,
            source_selection_mask_sha256=_require_sha256(
                plan.events.get("selection_mask_sha256"),
                label="compact packet source selection mask",
            ),
            selected_output_mask_sha256=_require_sha256(
                capture["guarded_route"].events.get("selected_output_mask_sha256"),
                label="compact packet selected output mask",
            ),
            packed_source_trace_sha256=_require_sha256(
                final_packed.source_trace_sha256,
                label="compact packet source trace",
            ),
            sample_index=sample_index,
            model_name=model_name,
            classic_backend_identity=classic_backend_identity,
        )
        if (
            quality_gate["record_sha256"] != target_free_audit["sha256"]
            or quality_gate["file_sha256"] != target_free_audit_file_sha256
        ):
            raise RuntimeError(
                "target-free quality audit changed after input validation"
            )
        quality_gate = {
            **quality_gate,
            "context_input_identity": context_input_identity,
        }
        # The packet is committed before the canonical target-view metadata is
        # interpreted.  This preserves the target-free route boundary while
        # still rejecting a native target batch from a different selection.
        source_selection = _canonical_source_selection(
            index_path=selection.index_path,
            source_index_sha256=selection.source_index_sha256,
            sample_index=sample_index,
        )
        if (
            scene != source_selection["scene"]
            or context_indices != source_selection["context_indices"]
            or context_input_identity.get("source_sample_index")
            != source_selection["source_sample_index"]
        ):
            raise RuntimeError(
                "context-only input does not match the canonical source selection"
            )
        if compact_anchor_count == 0:
            return {
                "schema_version": "1.0",
                "kind": PILOT_KIND,
                "status": "NO_COMPACT_TILES",
                "paper_result_eligible": False,
                "model": model_name,
                "dataset": DATASET,
                "sample_index": sample_index,
                "execution_index": execution_index,
                "native_sample_count": native_sample_count,
                "scene": scene,
                "target_rgb_provenance": {
                    "loaded_by_native_dataloader": False,
                    "accessed_before_compact_commit": False,
                    "accessed_for_metrics": False,
                },
                "execution_boundary": {
                    "source_bound_packet_adapter_executed": True,
                    "compact_nonzero_materialization_enabled": True,
                    "nonzero_direct_deletion": False,
                    "whole_pipeline_s2_s3_sparse_execution_verified": False,
                    "timing_claim": False,
                },
                "paper_identity": paper_identity,
                "backend_binding": backend_binding,
                "decoder_binding": decoder_binding,
                "adaptive_l1_calibration": v15_calibration,
                "adaptive_l1_v4_attribute_loo_calibration": v16_calibration,
                "raw_head_execution": {
                    "initial": initial_native_dense_execution,
                    "guarded": guarded_native_dense_execution,
                },
                "target_free_quality_gate": quality_gate,
                "route_plan": plan.events,
                "compact_route": route_summary,
                "route_diagnostics": route_diagnostics,
                "numerical_execution": numerical_execution,
                "checkpoint_sha256": checkpoint_sha256,
                "source": source_identity(),
            }
        renderable = _packed_attributes_to_device(final_packed, loaded_device)
        if posthoc_domain_audit_only:
            with torch.no_grad():
                baseline_gaussians = model.encoder(context, 0, deterministic=True)
            if type(baseline_gaussians) is not decoder_gaussians_type:
                raise RuntimeError(
                    "native dense baseline Gaussians type does not match the active decoder"
                )
            posthoc_dense_reference = _posthoc_dense_domain_coverage_reference(
                dense_gaussians=baseline_gaussians,
                final_packed=renderable,
                route=capture["guarded_route"],
                context=context,
                image_shape=(height, width),
            )
            return {
                "schema_version": "1.0",
                "kind": PILOT_KIND,
                "status": "POSTHOC_DOMAIN_AUDIT_COMPLETE",
                "paper_result_eligible": False,
                "model": model_name,
                "dataset": DATASET,
                "sample_index": sample_index,
                "execution_index": execution_index,
                "native_sample_count": native_sample_count,
                "scene": scene,
                "target_rgb_provenance": {
                    "loaded_by_native_dataloader": False,
                    "accessed_before_compact_commit": False,
                    "accessed_for_metrics": False,
                },
                "execution_boundary": {
                    "source_bound_packet_adapter_executed": True,
                    "native_packet_decoder_executed": False,
                    "independent_dense_reference_executed_after_packet_commit": True,
                    "compact_nonzero_materialization_enabled": True,
                    "nonzero_direct_deletion": False,
                    "whole_pipeline_s2_s3_sparse_execution_verified": False,
                    "timing_claim": False,
                },
                "paper_identity": paper_identity,
                "backend_binding": backend_binding,
                "decoder_binding": decoder_binding,
                "adaptive_l1_calibration": v15_calibration,
                "adaptive_l1_v4_attribute_loo_calibration": v16_calibration,
                "raw_head_execution": {
                    "initial": initial_native_dense_execution,
                    "guarded": guarded_native_dense_execution,
                },
                "target_free_quality_gate": quality_gate,
                "route_plan": plan.events,
                "compact_route": route_summary,
                "route_diagnostics": route_diagnostics,
                "posthoc_dense_domain_coverage_reference": posthoc_dense_reference,
                "final_packet": {
                    "descriptor_count": int(final_packed.dense_slots.numel()),
                    "materialized_anchor_count": compact_anchor_count,
                    "source_trace_sha256": final_packed.source_trace_sha256,
                    "omitted_raw_head_positions_poisoned": capture[
                        "omitted_raw_head_positions_poisoned"
                    ],
                },
                "numerical_execution": numerical_execution,
                "checkpoint_sha256": checkpoint_sha256,
                "source": source_identity(),
            }
        # The compact route and packet are now committed. Only now may the
        # native loader construct target-side tensors; its context is discarded.
        target_batch = _load_native_target_batch_after_packet_gate(
            loader,
            bundle,
            scene=scene,
            context_indices=context_indices,
            target_indices=source_selection["target_indices"],
            execution_index=execution_index,
            native_sample_count=native_sample_count,
        )
        target_mapping = target_batch.get("target")
        if not isinstance(target_mapping, dict) or not torch.is_tensor(
            target_mapping.get("image")
        ):
            raise RuntimeError(
                "compact packet pilot requires native target RGB for metrics"
            )
        target_cameras = _target_cameras(target_batch, loaded_device)
        _packet_gaussians, packet_color = render_packed_gaussians(
            model.decoder,
            renderable,
            gaussians_type=decoder_gaussians_type,
            target=target_cameras,
            image_shape=(height, width),
            expected_descriptor_count=int(final_packed.dense_slots.numel()),
        )
        with torch.no_grad():
            baseline_gaussians = model.encoder(context, 0, deterministic=True)
        if type(baseline_gaussians) is not decoder_gaussians_type:
            raise RuntimeError(
                "native dense baseline Gaussians type does not match the active decoder"
            )
        posthoc_dense_reference = _posthoc_dense_domain_coverage_reference(
            dense_gaussians=baseline_gaussians,
            final_packed=renderable,
            route=capture["guarded_route"],
            context=context,
            image_shape=(height, width),
        )
        baseline_color = _render_dense_baseline(
            model,
            baseline_gaussians,
            target=target_cameras,
            image_shape=(height, width),
        )

    target_rgb = _take_target_rgb_for_metrics(target_batch, loaded_device)
    baseline_views = _view_metrics(baseline_color[0], target_rgb[0])
    compact_views = _view_metrics(packet_color[0], target_rgb[0])
    baseline_quality = _mean_metrics(baseline_views)
    compact_quality = _mean_metrics(compact_views)
    verdict = _quality_verdict(baseline_quality, compact_quality)
    return {
        "schema_version": "1.0",
        "kind": PILOT_KIND,
        "status": "PASS" if verdict["pass"] else "QUALITY_FAILED",
        "paper_result_eligible": False,
        "model": model_name,
        "dataset": DATASET,
        "sample_index": sample_index,
        "execution_index": execution_index,
        "native_sample_count": native_sample_count,
        "scene": scene,
        "context_indices": context_indices,
        "target_indices": [
            int(value) for value in target_batch["target"]["index"][0].tolist()
        ],
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": True,
            "accessed_before_compact_commit": False,
            "target_camera_metadata_accessed_before_compact_commit": False,
            "target_camera_metadata_accessed_after_compact_commit": True,
            "first_transfer_after_packet_and_baseline_outputs": True,
        },
        "execution_boundary": {
            "source_bound_packet_adapter_executed": True,
            "native_packet_decoder_executed": True,
            "independent_dense_baseline_executed_after_packet_decoder": True,
            "compact_nonzero_materialization_enabled": True,
            "nonzero_direct_deletion": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "s2_s3_saving": 0.0,
            "timing_claim": False,
        },
        "paper_identity": paper_identity,
        "backend_binding": backend_binding,
        "decoder_binding": decoder_binding,
        "adaptive_l1_calibration": v15_calibration,
        "adaptive_l1_v4_attribute_loo_calibration": v16_calibration,
        "raw_head_execution": {
            "initial": initial_native_dense_execution,
            "guarded": guarded_native_dense_execution,
        },
        "target_free_quality_gate": quality_gate,
        "route_plan": plan.events,
        "compact_route": route_summary,
        "route_diagnostics": route_diagnostics,
        "posthoc_dense_domain_coverage_reference": posthoc_dense_reference,
        "final_packet": {
            "descriptor_count": int(final_packed.dense_slots.numel()),
            "materialized_anchor_count": compact_anchor_count,
            "source_trace_sha256": final_packed.source_trace_sha256,
            "omitted_raw_head_positions_poisoned": capture[
                "omitted_raw_head_positions_poisoned"
            ],
        },
        "quality": {
            "baseline": baseline_quality,
            "compact": compact_quality,
            "verdict": verdict,
            "views": [
                {
                    "target_index": int(
                        target_batch["target"]["index"][0, index].item()
                    ),
                    "baseline": baseline_views[index],
                    "compact": compact_views[index],
                }
                for index in range(len(baseline_views))
            ],
        },
        "numerical_execution": numerical_execution,
        "checkpoint_sha256": checkpoint_sha256,
        "source": source_identity(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=CLASSIC_MODELS, default=MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--posthoc-domain-audit-only", action="store_true")
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX)
    parser.add_argument(
        "--execution-index",
        type=int,
        help="Prepared-dataloader ordinal; defaults to --sample-index for legacy sample-0 use",
    )
    parser.add_argument(
        "--native-sample-count",
        type=int,
        help="Prepared-dataloader length; defaults to the frozen claim selection count",
    )
    parser.add_argument("--v15-calibration-record", type=Path, required=True)
    parser.add_argument("--v16-calibration-record", type=Path, required=True)
    parser.add_argument("--target-free-audit-artifact", type=Path, required=True)
    parser.add_argument("--target-free-input-root", type=Path, required=True)
    parser.add_argument("--acid-plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument(
        "--acid-materialization-root",
        type=Path,
        default=DEFAULT_MATERIALIZATION_ROOT,
    )
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if args.execution_index is not None and args.execution_index < 0:
        parser.error("--execution-index must be non-negative")
    if args.native_sample_count is not None and args.native_sample_count <= 0:
        parser.error("--native-sample-count must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this local compact packet pilot requires CUDA")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from scripts.result_record import write_result

    try:
        record = collect_paper_compact_packet_pilot(
            device=device,
            posthoc_domain_audit_only=args.posthoc_domain_audit_only,
            v15_calibration_record=args.v15_calibration_record,
            v16_calibration_record=args.v16_calibration_record,
            target_free_audit_artifact=args.target_free_audit_artifact,
            target_free_input_root=args.target_free_input_root,
            sample_index=args.sample_index,
            execution_index=args.execution_index,
            native_sample_count=args.native_sample_count,
            acid_plan_path=args.acid_plan_path,
            acid_materialization_root=args.acid_materialization_root,
            model_name=args.model,
        )
        exit_code = (
            0 if record["status"] in {"PASS", "POSTHOC_DOMAIN_AUDIT_COMPLETE"} else 1
        )
    except Exception as exc:
        record = {
            "schema_version": "1.0",
            "kind": PILOT_KIND,
            "status": "FAILED",
            "paper_result_eligible": False,
            "model": args.model,
            "dataset": DATASET,
            "sample_index": args.sample_index,
            "execution_index": args.execution_index,
            "native_sample_count": args.native_sample_count,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
        exit_code = 1
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
