#!/usr/bin/env python3
"""Run one direct conditional SAES quality measurement on DL3DV.

The source route and materialized packet are committed before target frames
are opened. The resulting record reports route statistics and target-view
quality metrics from the same process.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration import create_model_loader, load_context_only_audit_data
from depth_predictor.depthsplat_predictor import DepthSplatDepthPredictorSim
from encoder.types import ENCODER_CYCLES
from fsdr import FSDRSimulator
from saes.depthsplat_backend import (
    canonical_json_sha256,
    freeze_depthsplat_backend_identity,
    resolve_depthsplat_backend_contract,
)
from saes.depthsplat_l0_l1_materializer import (
    DEPTHSPLAT_COMPACT_COVERAGE_MAX_COVARIANCE_SCALE,
    DEPTHSPLAT_COVERAGE_CERTIFICATE,
    DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
    DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
    DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
    apply_depthsplat_compact_l0_l1_materialization,
    preflight_depthsplat_l0_l1_materialization,
    resolve_depthsplat_compact_final_route,
)
from saes.depthsplat_mixture_kernel_guard import (
    DepthSplatTileKernelClosureMeasurementCache,
    KIND as DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND,
    POLICY as DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY,
    SCHEMA_VERSION as DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_SCHEMA_VERSION,
)
from saes.hardware_accounting import build_saes_event_ledger
from saes.depthsplat_mixture_kernel_acid_calibration import (
    KERNEL_RISK_GUARD_SCHEMA,
    KERNEL_RISK_METRIC,
    KERNEL_RISK_THRESHOLD_RULE,
    to_materializer_guard as load_mixture_kernel_risk_guard,
)
from saes.depthsplat_selected_output import (
    DepthSplatPackedGaussianConsumer,
    build_depthsplat_sparse_raw_packet,
    capture_depthsplat_native_execution,
    compare_depthsplat_full_passthrough_to_dense_bitwise,
    compare_depthsplat_packed_to_dense,
    estimate_depthsplat_selected_head_schedule,
    replay_depthsplat_selected_head,
    subset_depthsplat_sparse_raw_packet,
)
from saes.depthsplat_sparse_datapath_projection import (
    project_depthsplat_sparse_datapath_cycles,
)
from saes.probe_first_schedule import (
    BALANCED_L1_ANCHOR_SEMANTICS,
    DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
    DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_PLAN_CONTRACT,
    SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY,
    SOFT_MIXTURE_KERNEL_CLOSURE_T4_L0_SECONDARY_PREFETCH_POLICY,
    SOFT_MIXTURE_NORMALIZED_T4_L0_SECONDARY_PREFETCH_POLICY,
    build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan,
    build_depthsplat_soft_mixture_normalized_t4_probe_first_plan,
    soft_mixture_normalized_t4_route_config_sha256,
    soft_mixture_kernel_closure_t4_route_config_sha256,
)
from saes.progressive_saes import PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS
from scripts.ae_config import resolve_claim_selection, resolve_experiment
from scripts.depthsplat_execution import (
    align_depthsplat_plane_sweep_candidate_scales,
    extract_depthsplat_execution_tensors,
)
from scripts.fsdr_trace import (
    prepare_fsdr_candidate_frames,
    prepare_fsdr_frame,
    tile_probe_pixel_order,
)
from scripts.result_record import (
    cached_sha256_file,
    portable_command,
    source_identity,
    write_result,
)
from scripts.saes_depthsplat_coverage_enriched_t4_context_only_audit import (
    DATASET,
    DEPTH_THRESHOLD,
    FEATURE_THRESHOLD,
    MODEL,
    SOURCE_SAMPLE_INDEX,
    TILE_SIZE,
    _bind_formal_application,
    _require_bitwise_equivalent,
    _require_equivalent,
    _require_loaded_context_batch,
    _source_geometry_functions,
)
from scripts.saes_depthsplat_l0_l1_quality_gate import (
    DEFAULT_RAW_ROOT,
    _load_isolated_target_batch_after_packet_commit,
    _render_target_view,
    _take_target_rgb_for_metrics,
    _target_cameras,
)
from scripts.saes_selected_output_quality_gate import (
    _mean_metrics,
    _quality_verdict,
    _view_metrics,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution


QUALITY_GATE_KIND = "depthsplat-direct-conditional-t4-fused-dl3dv-quality-gate"
QUALITY_GATE_SCHEMA_VERSION = "4.0"
QUALITY_GATE_SEED = 0
SOFT_MIXTURE_MECHANISM = "direct-conditional-anchor-transport-v1"
KERNEL_CLOSURE_QUALITY_GATE_KIND = (
    "depthsplat-soft-mixture-kernel-closure-v3-fused-dl3dv-quality-gate"
)
KERNEL_CLOSURE_MECHANISM = "soft-mixture-kernel-closure-v3"
PSNR_ONLY_QUALITY_GATE_LIMIT_DB = 0.5
FSDR_CACHE_SIZE = 32
FSDR_HAMMING_THRESHOLD = 3
FSDR_NUM_DEPTH_CANDIDATES = 128
FSDR_DEPTH_CONSISTENCY_THRESHOLD = 0.10
# ``FeatureBuffer`` exposes two independent 16-bit data ports in the RTL.
# The source trace records FP16 feature-buffer bytes, so both ports provide
# four bytes of service per cycle when the plane sweep is transfer-bound.
FSDR_FEATURE_BUFFER_PORTS = 2
FSDR_FEATURE_BUFFER_PORT_BYTES = 2
FSDR_FEATURE_BUFFER_BYTES_PER_CYCLE = (
    FSDR_FEATURE_BUFFER_PORTS * FSDR_FEATURE_BUFFER_PORT_BYTES
)
# Candidate windows originate from a successful LSH/CAM lookup.  The local
# depth-validity check prevents a semantic hit at a discontinuity from routing
# a window away from the native full-search winner.
FSDR_GUIDANCE_POLICY = "paper-hamming-local-validity"
DEFAULT_INPUT_ROOT = (
    ROOT
    / "outputs"
    / "ae_dl3dv_repair_diagnostics"
    / "depthsplat_sample0_l0_l1_context_only_v2"
)
DEFAULT_SOURCE_AUDIT_ROOT = (
    ROOT
    / "outputs"
    / "ae_dl3dv_repair_diagnostics"
    / "depthsplat_sample0_l0_l1_source_target_free_v2"
)


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"DepthSplat soft-mixture runner has invalid {label} SHA256")
    return value


def _require_quality_context_identity(
    identity: Mapping[str, Any], *, source_sample_index: int
) -> dict[str, Any]:
    """Require one declared, target-free DL3DV context sample."""

    if (
        isinstance(source_sample_index, bool)
        or not isinstance(source_sample_index, int)
        or source_sample_index < 0
        or not isinstance(identity, Mapping)
        or identity.get("source_sample_index") != source_sample_index
        or not isinstance(identity.get("scene"), str)
        or not identity["scene"]
        or not isinstance(identity.get("context_indices"), list)
        or len(identity["context_indices"]) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in identity["context_indices"]
        )
        or len(set(identity["context_indices"])) != 2
        or identity.get("target_mapping_present") is not False
        or identity.get("target_rgb_accessed") is not False
        or identity.get("target_camera_metadata_accessed") is not False
        or identity.get("target_index_accessed") is not False
    ):
        raise ValueError("DepthSplat quality gate input is not fixed target-free context")
    source_binding = identity.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise ValueError("DepthSplat quality gate input has no source binding")
    for key in (
        "canonical_index_sha256",
        "canonical_sample_selection_sha256",
        "canonical_selection_sha256",
        "source_audit_input_sha256",
        "source_audit_tree_sha256",
        "source_sidecar_tree_sha256",
    ):
        _require_sha256(source_binding.get(key), f"context input {key}")
    for key in ("tree_sha256", "manifest_sha256", "audit_input_sha256"):
        _require_sha256(identity.get(key), f"context input {key}")
    return dict(identity)


def _require_live_tile_trace(
    tile_trace: Any, *, expected_sha256: Any, label: str
) -> str:
    if not isinstance(tile_trace, tuple) or not all(
        isinstance(record, Mapping) for record in tile_trace
    ):
        raise RuntimeError(f"DepthSplat soft-mixture {label} trace is invalid")
    expected = _require_sha256(expected_sha256, f"{label} trace")
    actual = canonical_json_sha256(tile_trace)
    if actual != expected:
        raise RuntimeError(f"DepthSplat soft-mixture {label} trace hash changed")
    return actual


def soft_mixture_kernel_closure_t4_profile(
    *,
    kernel_risk_guard: Mapping[str, Any] | None = None,
    direct_kernel_risk_threshold: float | None = None,
    depth_threshold: float = DEPTH_THRESHOLD,
) -> dict[str, Any]:
    """Return the source-only S/R replay configuration with kernel closure."""

    if kernel_risk_guard is not None and direct_kernel_risk_threshold is not None:
        raise ValueError("DepthSplat kernel-risk profile cannot mix threshold sources")
    if direct_kernel_risk_threshold is not None and (
        isinstance(direct_kernel_risk_threshold, bool)
        or not isinstance(direct_kernel_risk_threshold, (int, float))
        or not math.isfinite(float(direct_kernel_risk_threshold))
        or float(direct_kernel_risk_threshold) <= 0.0
    ):
        raise ValueError("DepthSplat direct kernel-risk threshold is invalid")
    if (
        isinstance(depth_threshold, bool)
        or not isinstance(depth_threshold, (int, float))
        or not math.isfinite(float(depth_threshold))
        or float(depth_threshold) < 0.0
    ):
        raise ValueError("DepthSplat depth routing threshold is invalid")

    profile: dict[str, Any] = {
        "materialization_profile": (
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
        ),
        "route_plan_contract": DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
        "contract_version": DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": float(depth_threshold),
        "decision_semantics": PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
        "feature_statistic": "normalized-probe-vector-standard-deviation",
        "assignment_feature_semantics": "unit-normalized-bilinear-s1-v1",
        "l1_anchor_semantics": BALANCED_L1_ANCHOR_SEMANTICS,
        "l0_anchor_count": 4,
        "l1_anchor_count": 12,
        "formal_paper_kp4": False,
        "depth_checked_after_l0_miss_only": False,
        "soft_mixture_normalized_l0_secondary_prefetch_policy": (
            SOFT_MIXTURE_NORMALIZED_T4_L0_SECONDARY_PREFETCH_POLICY
        ),
        "soft_mixture_kernel_closure_l0_secondary_prefetch_policy": (
            SOFT_MIXTURE_KERNEL_CLOSURE_T4_L0_SECONDARY_PREFETCH_POLICY
        ),
        "kernel_closure_guard_policy": SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY,
        "kernel_closure_evidence_policy": DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY,
        "maximum_coverage_covariance_scale": 1.0,
        "coverage_certificate": DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
        "support_containment_guard": False,
        "projected_domain_guard": False,
        "precalibration_only": (
            kernel_risk_guard is None and direct_kernel_risk_threshold is None
        ),
        "kernel_risk_frozen_guard": (
            dict(kernel_risk_guard) if kernel_risk_guard is not None else None
        ),
        "direct_kernel_risk_threshold": (
            float(direct_kernel_risk_threshold)
            if direct_kernel_risk_threshold is not None
            else None
        ),
    }
    profile["route_plan_config_sha256"] = (
        soft_mixture_kernel_closure_t4_route_config_sha256(profile)
    )
    return profile


def direct_conditional_t4_profile(
    *, route_isolation: str = "l0_l1"
) -> dict[str, Any]:
    """Return the normalized L0/L1 route with conditional anchor transport."""

    if route_isolation not in {"l0_l1", "l0_only"}:
        raise ValueError("DepthSplat direct conditional route is invalid")
    profile = {
        "materialization_profile": (
            DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE
        ),
        "contract_version": DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_PLAN_CONTRACT,
        "route_plan_contract": DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_PLAN_CONTRACT,
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": DEPTH_THRESHOLD,
        "decision_semantics": PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
        "feature_statistic": "normalized-probe-vector-standard-deviation",
        "assignment_feature_semantics": "unit-normalized-bilinear-s1-v1",
        "l1_anchor_semantics": BALANCED_L1_ANCHOR_SEMANTICS,
        "l0_anchor_count": 4,
        "l1_anchor_count": 12,
        "formal_paper_kp4": False,
        "depth_checked_after_l0_miss_only": False,
        "soft_mixture_normalized_l0_secondary_prefetch_policy": (
            SOFT_MIXTURE_NORMALIZED_T4_L0_SECONDARY_PREFETCH_POLICY
        ),
        "merge_semantics": (
            "coverage-closed-alpha-union-anchor-transport-v1"
        ),
        "assignment_transport": "bilateral-soft-moment-v1",
        "maximum_coverage_covariance_scale": (
            DEPTHSPLAT_COMPACT_COVERAGE_MAX_COVARIANCE_SCALE
        ),
        "route_isolation": route_isolation,
    }
    profile["route_plan_config_sha256"] = (
        soft_mixture_normalized_t4_route_config_sha256(profile)
    )
    return profile


def _validate_direct_conditional_plan(plan: Any, *, profile: Mapping[str, Any]) -> str:
    """Bind direct conditional materialization to the normalized paper route."""

    events = getattr(plan, "events", None)
    if not isinstance(events, Mapping):
        raise RuntimeError("DepthSplat direct conditional plan has no events")
    trace_sha256 = _require_live_tile_trace(
        getattr(plan, "tile_trace", None),
        expected_sha256=events.get("tile_trace_sha256"),
        label="direct conditional plan",
    )
    required = (
        ("contract_version", "route_plan_contract"),
        ("tile_size", "tile_size"),
        ("feature_threshold", "feature_threshold"),
        ("depth_threshold", "depth_threshold"),
        ("decision_semantics", "decision_semantics"),
        ("feature_statistic", "feature_statistic"),
        ("l1_anchor_semantics", "l1_anchor_semantics"),
        ("l0_anchor_count", "l0_anchor_count"),
        ("l1_anchor_count", "l1_anchor_count"),
        ("formal_paper_kp4", "formal_paper_kp4"),
        ("depth_checked_after_l0_miss_only", "depth_checked_after_l0_miss_only"),
        ("assignment_feature_semantics", "assignment_feature_semantics"),
    )
    if (
        profile.get("materialization_profile")
        != DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE
        or profile.get("route_isolation") not in {"l0_l1", "l0_only"}
        or any(events.get(event_key) != profile.get(profile_key) for event_key, profile_key in required)
        or events.get("soft_mixture_normalized_t4_route_config_sha256")
        != profile.get("route_plan_config_sha256")
        or events.get("target_rgb_accessed") is not False
        or events.get("gaussian_attributes_accessed") is not False
    ):
        raise RuntimeError("DepthSplat direct conditional plan changed")
    return trace_sha256


def _validate_plan(plan: Any, *, profile: Mapping[str, Any]) -> str:
    events = getattr(plan, "events", None)
    if not isinstance(events, Mapping):
        raise RuntimeError("DepthSplat soft-mixture plan has no events")
    trace_sha256 = _require_live_tile_trace(
        getattr(plan, "tile_trace", None),
        expected_sha256=events.get("tile_trace_sha256"),
        label="plan",
    )
    if (
        events.get("contract_version") != profile["route_plan_contract"]
        or events.get("tile_size") != profile["tile_size"]
        or events.get("feature_threshold") != profile["feature_threshold"]
        or events.get("depth_threshold") != profile["depth_threshold"]
        or events.get("decision_semantics") != profile["decision_semantics"]
        or events.get("feature_statistic") != profile["feature_statistic"]
        or events.get("l1_anchor_semantics") != profile["l1_anchor_semantics"]
        or events.get("l0_anchor_count") != profile["l0_anchor_count"]
        or events.get("l1_anchor_count") != profile["l1_anchor_count"]
        or events.get("formal_paper_kp4") != profile["formal_paper_kp4"]
        or events.get("depth_checked_after_l0_miss_only")
        != profile["depth_checked_after_l0_miss_only"]
        or events.get("assignment_feature_semantics")
        != profile["assignment_feature_semantics"]
        or events.get("soft_mixture_kernel_closure_t4_route_config_sha256")
        != profile["route_plan_config_sha256"]
        or events.get("soft_mixture_kernel_closure_l0_secondary_prefetch_policy")
        != profile["soft_mixture_kernel_closure_l0_secondary_prefetch_policy"]
        or events.get("kernel_closure_guard_policy")
        != profile["kernel_closure_guard_policy"]
        or events.get("target_rgb_accessed") is not False
        or events.get("gaussian_attributes_accessed") is not False
    ):
        raise RuntimeError("DepthSplat soft-mixture plan changed")
    return trace_sha256


def _soft_mixture_certificate_summary(preflight: Any) -> dict[str, Any]:
    """Return the sealed aggregate without copying tile certificates."""

    aggregate = preflight.events.get("soft_mixture_certificate_aggregate")
    aggregate_sha256 = preflight.events.get("soft_mixture_certificate_aggregate_sha256")
    required = {
        "schema",
        "certificate",
        "kind",
        "policy",
        "tile_certificate_attempt_count",
        "passed_tile_certificate_count",
        "failed_tile_certificate_count",
        "maximum_absolute_residual_by_field",
        "maximum_tolerance_by_field",
        "minimum_source_covariance_eigenvalue",
        "minimum_virtual_covariance_eigenvalue",
        "minimum_merged_covariance_eigenvalue",
        "source_only",
    }
    attempts = aggregate.get("tile_certificate_attempt_count") if isinstance(aggregate, Mapping) else None
    source_only = aggregate.get("source_only") if isinstance(aggregate, Mapping) else None
    source_only_valid = (
        attempts == 0 and source_only is None
    ) or (
        isinstance(source_only, Mapping)
        and source_only.get("target_rgb_accessed") is False
        and source_only.get("projected_domain_guard_used") is False
    )
    if (
        not isinstance(aggregate, Mapping)
        or set(aggregate) != required
        or aggregate.get("certificate") != DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE
        or aggregate.get("tile_certificate_attempt_count")
        != aggregate.get("passed_tile_certificate_count")
        + aggregate.get("failed_tile_certificate_count")
        or not source_only_valid
        or _require_sha256(aggregate_sha256, "soft-mixture aggregate")
        != canonical_json_sha256(dict(aggregate))
    ):
        raise RuntimeError("DepthSplat soft-mixture certificate aggregate changed")
    return {**dict(aggregate), "sha256": aggregate_sha256}


_KERNEL_CLOSURE_SOURCE_ONLY = {
    "source_camera_only": True,
    "target_mapping_present": False,
    "target_rgb_accessed": False,
    "target_camera_metadata_accessed": False,
    "target_index_accessed": False,
    "omitted_s3_attributes_accessed": False,
    "boolean_owner_assignment_used": False,
    "projected_domain_guard_used": False,
    "covariance_expansion_used": False,
    "alpha_union_used": False,
}
_KERNEL_CLOSURE_RISK_FIELDS = (
    "maximum_world_kernel_risk",
    "maximum_source_kernel_risk",
    "maximum_kernel_risk",
    "maximum_source_log_depth_rms",
)
_MIXTURE_KERNEL_CLOSURE_AGGREGATE_SCHEMA = (
    "depthsplat-mixture-kernel-closure-aggregate-v1"
)
_MIXTURE_KERNEL_CLOSURE_HISTOGRAM_UPPER_BOUNDS = (
    1.0e-6,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
)


def _nonnegative_finite_or_none(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise RuntimeError(f"DepthSplat mixture kernel-closure {label} is invalid")
    return float(value)


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"DepthSplat mixture kernel-closure {label} is invalid")
    return int(value)


def _quality_status(*, verdict: Mapping[str, Any], update_count: int) -> str:
    """Classify measured quality without treating an all-Full packet as compact success."""

    passed = verdict.get("pass") if isinstance(verdict, Mapping) else None
    if (
        not isinstance(passed, bool)
        or isinstance(update_count, bool)
        or not isinstance(update_count, int)
        or update_count < 0
    ):
        raise RuntimeError("DepthSplat soft-mixture quality status inputs are invalid")
    if not passed:
        return "QUALITY_FAILED"
    return "PASS" if update_count else "PASS_IDENTITY_FALLBACK"


def _psnr_only_quality_verdict(
    baseline: Mapping[str, float], sparse: Mapping[str, float]
) -> dict[str, Any]:
    """Apply the user-authorized PSNR gate while retaining all quality deltas."""

    full_verdict = _quality_verdict(dict(baseline), dict(sparse))
    observed = full_verdict.get("observed")
    if not isinstance(observed, Mapping):
        raise RuntimeError("DepthSplat PSNR-only quality metrics are invalid")
    psnr_loss_db = observed.get("psnr_loss_db")
    if (
        isinstance(psnr_loss_db, bool)
        or not isinstance(psnr_loss_db, (int, float))
        or not math.isfinite(float(psnr_loss_db))
    ):
        raise RuntimeError("DepthSplat PSNR-only quality loss is invalid")
    return {
        "quality_gate": "user-authorized-psnr-only-v1",
        "limits": {"psnr_loss_db": PSNR_ONLY_QUALITY_GATE_LIMIT_DB},
        "observed": dict(observed),
        "secondary_metrics_reported_not_gated": {
            "ssim_loss": observed.get("ssim_loss"),
            "lpips_increase": observed.get("lpips_increase"),
        },
        "pass": float(psnr_loss_db) <= PSNR_ONLY_QUALITY_GATE_LIMIT_DB,
    }


def _kernel_closure_rows(tile_trace: Any, *, policy: str) -> list[dict[str, Any]]:
    """Validate every source-only guard attempt retained in the route trace."""

    if not isinstance(tile_trace, tuple):
        raise RuntimeError("DepthSplat mixture kernel-closure trace is invalid")
    rows: list[dict[str, Any]] = []
    evidence_fields = (
        (
            "mixture_kernel_closure",
            "mixture_kernel_closure_sha256",
            "mixture_kernel_closure_passed",
        ),
        (
            "l0_mixture_kernel_closure_failure",
            "l0_mixture_kernel_closure_failure_sha256",
            None,
        ),
        (
            "l1_mixture_kernel_closure_failure",
            "l1_mixture_kernel_closure_failure_sha256",
            None,
        ),
    )
    for record in tile_trace:
        if not isinstance(record, Mapping):
            raise RuntimeError("DepthSplat mixture kernel-closure trace row is invalid")
        for evidence_key, sha256_key, passed_key in evidence_fields:
            evidence = record.get(evidence_key)
            evidence_sha256 = record.get(sha256_key)
            passed = (
                record.get(passed_key)
                if passed_key is not None
                else evidence.get("passed")
                if isinstance(evidence, Mapping)
                else None
            )
            if evidence is None and evidence_sha256 is None and passed is None:
                continue
            if (
                not isinstance(evidence, Mapping)
                or _require_sha256(evidence_sha256, evidence_key)
                != canonical_json_sha256(dict(evidence))
                or not isinstance(passed, bool)
                or evidence.get("passed") is not passed
                or evidence.get("schema_version")
                != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_SCHEMA_VERSION
                or evidence.get("kind") != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND
                or evidence.get("policy") != policy
                or evidence.get("policy")
                != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY
                or evidence.get("source_only") != _KERNEL_CLOSURE_SOURCE_ONLY
                or not isinstance(evidence.get("summary"), Mapping)
            ):
                raise RuntimeError("DepthSplat mixture kernel-closure evidence changed")
            summary = evidence["summary"]
            input_valid = summary.get("input_valid")
            threshold = _nonnegative_finite_or_none(
                summary.get("strict_maximum_relative_risk"),
                label="strict maximum relative risk",
            )
            reason = summary.get("reason")
            if (
                not isinstance(input_valid, bool)
                or threshold is None
                or (reason is not None and not isinstance(reason, str))
                or (input_valid and not isinstance(evidence.get("binding"), Mapping))
            ):
                raise RuntimeError(
                    "DepthSplat mixture kernel-closure summary changed"
                )
            risks = {
                field: _nonnegative_finite_or_none(summary.get(field), label=field)
                for field in _KERNEL_CLOSURE_RISK_FIELDS
            }
            if input_valid:
                if any(value is None for value in risks.values()):
                    raise RuntimeError("DepthSplat mixture kernel-closure risks changed")
                if (
                    risks["maximum_kernel_risk"]
                    != max(
                        risks["maximum_world_kernel_risk"],
                        risks["maximum_source_kernel_risk"],
                    )
                    or passed != (risks["maximum_kernel_risk"] <= threshold)
                ):
                    raise RuntimeError("DepthSplat mixture kernel-closure decision changed")
            elif passed:
                raise RuntimeError("DepthSplat invalid kernel-closure evidence passed")
            rows.append(dict(evidence))
    return rows


def _mixture_kernel_closure_summary(
    preflight: Any, *, profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and compact the source-only kernel-risk evidence for quality output."""

    aggregate = preflight.events.get("mixture_kernel_closure_aggregate")
    aggregate_sha256 = preflight.events.get("mixture_kernel_closure_aggregate_sha256")
    if not isinstance(aggregate, Mapping):
        raise RuntimeError("DepthSplat mixture kernel-closure aggregate is missing")
    aggregate_dict = dict(aggregate)
    required = {
        "schema",
        "kind",
        "policy",
        "source_only",
        "strict_maximum_relative_risk",
        "tile_kernel_closure_attempt_count",
        "passed_tile_kernel_closure_count",
        "failed_tile_kernel_closure_count",
        "maximum_world_kernel_risk",
        "maximum_source_kernel_risk",
        "maximum_kernel_risk",
        "maximum_source_log_depth_rms",
        "maximum_kernel_risk_distribution",
        "reason_counts",
    }
    if (
        set(aggregate_dict) != required
        or aggregate_dict.get("schema") != _MIXTURE_KERNEL_CLOSURE_AGGREGATE_SCHEMA
        or aggregate_dict.get("kind") != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND
        or aggregate_dict.get("policy") != profile["kernel_closure_evidence_policy"]
        or aggregate_dict.get("source_only")
        not in (_KERNEL_CLOSURE_SOURCE_ONLY, None)
        or _require_sha256(aggregate_sha256, "mixture kernel-closure aggregate")
        != canonical_json_sha256(aggregate_dict)
    ):
        raise RuntimeError("DepthSplat mixture kernel-closure aggregate changed")
    rows = _kernel_closure_rows(
        preflight.tile_trace, policy=profile["kernel_closure_evidence_policy"]
    )
    attempts = len(rows)
    passed = sum(row["passed"] is True for row in rows)
    failed = attempts - passed
    thresholds = {
        _nonnegative_finite_or_none(
            row["summary"].get("strict_maximum_relative_risk"),
            label="strict maximum relative risk",
        )
        for row in rows
    }
    if len(thresholds) > 1:
        raise RuntimeError("DepthSplat mixture kernel-closure risk limit drifted")
    expected_threshold = thresholds.pop() if thresholds else None
    values_by_field = {field: [] for field in _KERNEL_CLOSURE_RISK_FIELDS}
    finite_kernel_risks: list[float] = []
    unscorable = 0
    reasons = Counter()
    for row in rows:
        summary = row["summary"]
        if summary["input_valid"]:
            for field in _KERNEL_CLOSURE_RISK_FIELDS:
                value = _nonnegative_finite_or_none(summary.get(field), label=field)
                if value is None:
                    raise RuntimeError("DepthSplat mixture kernel-closure risk changed")
                values_by_field[field].append(value)
            finite_kernel_risks.append(values_by_field["maximum_kernel_risk"][-1])
        else:
            unscorable += 1
        reasons["accepted" if row["passed"] else summary["reason"] or "unscorable"] += 1
    histogram_counts: list[int] = []
    lower = float("-inf")
    for upper in _MIXTURE_KERNEL_CLOSURE_HISTOGRAM_UPPER_BOUNDS:
        histogram_counts.append(
            sum(lower < value <= upper for value in finite_kernel_risks)
        )
        lower = upper
    expected = {
        "schema": _MIXTURE_KERNEL_CLOSURE_AGGREGATE_SCHEMA,
        "kind": DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND,
        "policy": profile["kernel_closure_evidence_policy"],
        "source_only": _KERNEL_CLOSURE_SOURCE_ONLY if attempts else None,
        "strict_maximum_relative_risk": expected_threshold,
        "tile_kernel_closure_attempt_count": attempts,
        "passed_tile_kernel_closure_count": passed,
        "failed_tile_kernel_closure_count": failed,
        **{
            field: max(values_by_field[field]) if values_by_field[field] else None
            for field in _KERNEL_CLOSURE_RISK_FIELDS
        },
        "maximum_kernel_risk_distribution": {
            "finite_tile_risk_count": len(finite_kernel_risks),
            "unscorable_tile_count": unscorable,
            "minimum_kernel_risk": (
                min(finite_kernel_risks) if finite_kernel_risks else None
            ),
            "histogram_upper_bounds": list(
                _MIXTURE_KERNEL_CLOSURE_HISTOGRAM_UPPER_BOUNDS
            ),
            "histogram_counts": histogram_counts,
            "above_largest_histogram_bin_count": sum(
                value > _MIXTURE_KERNEL_CLOSURE_HISTOGRAM_UPPER_BOUNDS[-1]
                for value in finite_kernel_risks
            ),
        },
        "reason_counts": dict(sorted(reasons.items())),
    }
    if aggregate_dict != expected:
        raise RuntimeError("DepthSplat mixture kernel-closure aggregate changed")
    return {**aggregate_dict, "sha256": aggregate_sha256}


def _source_summary(
    *, plan: Any, preflight: Any, final_route: Any, profile: Mapping[str, Any]
) -> dict[str, Any]:
    planned = Counter()
    accepted = Counter()
    reasons = Counter()
    for record in plan.tile_trace:
        route = record.get("pre_guard_route")
        if route not in {"L0", "L1", "Full"}:
            raise RuntimeError("DepthSplat soft-mixture planned route is invalid")
        planned[route] += 1
    for record in preflight.tile_trace:
        if record.get("accepted") is True:
            level = record.get("accepted_level")
            if level not in {"L0", "L1", "Full"}:
                raise RuntimeError("DepthSplat soft-mixture accepted route is invalid")
            accepted[level] += 1
        elif record.get("attempted") is True:
            reason = record.get("reason")
            if not isinstance(reason, str) or not reason:
                raise RuntimeError("DepthSplat soft-mixture rejection lacks a reason")
            reasons[reason] += 1
    preflight_trace_sha256 = _require_live_tile_trace(
        preflight.tile_trace,
        expected_sha256=preflight.events.get("tile_trace_sha256"),
        label="preflight",
    )
    final_trace_sha256 = _require_live_tile_trace(
        final_route.tile_trace,
        expected_sha256=final_route.events.get("tile_trace_sha256"),
        label="final route",
    )
    certificate_payload = preflight.events.get("coverage_certificate_payload")
    initial_binding = preflight.events.get("initial_binding")
    certificate_summary = _soft_mixture_certificate_summary(preflight)
    kernel_closure = _mixture_kernel_closure_summary(preflight, profile=profile)
    frozen_kernel_guard = preflight.events.get("mixture_kernel_closure_frozen_guard")
    profile_kernel_guard = profile.get("kernel_risk_frozen_guard")
    direct_threshold = profile.get("direct_kernel_risk_threshold")
    calibrated = preflight.events.get("mixture_kernel_closure_calibrated_threshold")
    if direct_threshold is not None:
        if (
            isinstance(direct_threshold, bool)
            or not isinstance(direct_threshold, (int, float))
            or not math.isfinite(float(direct_threshold))
            or float(direct_threshold) <= 0.0
            or profile.get("precalibration_only") is not False
            or profile_kernel_guard is not None
            or frozen_kernel_guard is not None
            or calibrated is not False
            or kernel_closure["strict_maximum_relative_risk"]
            != float(direct_threshold)
        ):
            raise RuntimeError("DepthSplat direct kernel-risk threshold changed")
    elif (
        profile.get("precalibration_only") is not (profile_kernel_guard is None)
        or (profile_kernel_guard is None and frozen_kernel_guard is not None)
        or (profile_kernel_guard is not None and frozen_kernel_guard != profile_kernel_guard)
        or calibrated is not (frozen_kernel_guard is not None)
    ):
        raise RuntimeError("DepthSplat soft-mixture kernel-risk calibration binding changed")
    if frozen_kernel_guard is not None and (
        not isinstance(frozen_kernel_guard, Mapping)
        or frozen_kernel_guard.get("schema_version") != KERNEL_RISK_GUARD_SCHEMA
        or frozen_kernel_guard.get("risk_metric") != KERNEL_RISK_METRIC
        or frozen_kernel_guard.get("threshold_rule") != KERNEL_RISK_THRESHOLD_RULE
        or frozen_kernel_guard.get("threshold_value")
        != kernel_closure["strict_maximum_relative_risk"]
    ):
        raise RuntimeError("DepthSplat soft-mixture kernel-risk threshold changed")
    if _nonnegative_int(
        certificate_summary.get("passed_tile_certificate_count"),
        label="passed S/R certificate count",
    ) != _nonnegative_int(
        kernel_closure.get("tile_kernel_closure_attempt_count"),
        label="kernel-closure attempt count",
    ):
        raise RuntimeError(
            "DepthSplat soft-mixture kernel closures lack matching passed S/R certificates"
        )
    l0_to_l1_kernel_tiles = _nonnegative_int(
        preflight.events.get("mixture_kernel_closure_l0_to_l1_tile_count"),
        label="L0-to-L1 tile count",
    )
    kernel_full_promotions = _nonnegative_int(
        preflight.events.get("mixture_kernel_closure_full_promotion_tile_count"),
        label="Full-promotion tile count",
    )
    if (
        preflight.events.get("execution_profile") != profile["materialization_profile"]
        or preflight.events.get("maximum_coverage_covariance_scale") != 1.0
        or preflight.events.get("coverage_certificate") != profile["coverage_certificate"]
        or preflight.events.get("soft_mixture_guard") is not True
        or preflight.events.get("soft_mixture_projected_domain_guard_used") is not False
        or preflight.events.get("mixture_kernel_closure_guard_policy")
        != profile["kernel_closure_guard_policy"]
        or preflight.events.get("mixture_kernel_closure_strict_maximum_relative_risk")
        != kernel_closure["strict_maximum_relative_risk"]
        or preflight.events.get("target_rgb_accessed") is not False
        or preflight.events.get("target_camera_accessed_before_commit") is not False
        or preflight.events.get("skipped_s3_attributes_accessed") is not False
        or preflight.events.get("assignment_feature_semantics")
        != profile["assignment_feature_semantics"]
        or not isinstance(initial_binding, Mapping)
        or initial_binding.get("assignment_feature_semantics")
        != profile["assignment_feature_semantics"]
        or initial_binding.get("assignment_feature_map_sha256")
        != preflight.events.get("assignment_feature_map_sha256")
        or not isinstance(certificate_payload, Mapping)
        or certificate_payload.get("schema") != profile["coverage_certificate"]
        or certificate_payload.get("fixed_moment_covariance_scale") != 1.0
        or certificate_payload.get("support_containment_guard") is not False
        or certificate_payload.get("projected_support_guard") is not False
        or certificate_payload.get("tile_trace_sha256") != preflight_trace_sha256
        or final_route.events.get("preflight_trace_sha256") != preflight_trace_sha256
        or final_route.events.get("coverage_certificate_sha256")
        != preflight.events.get("coverage_certificate_sha256")
        or final_route.events.get("mixture_kernel_closure_aggregate_sha256")
        != kernel_closure["sha256"]
        or final_route.events.get("assignment_feature_semantics")
        != profile["assignment_feature_semantics"]
        or final_route.events.get("assignment_feature_map_sha256")
        != preflight.events.get("assignment_feature_map_sha256")
        or any(
            record.get("assignment_feature_semantics")
            != profile["assignment_feature_semantics"]
            for record in preflight.tile_trace
            if record.get("accepted_level") in {"L0", "L1"}
        )
        or any(
            profile.get("assignment_transport") is not None
            and record.get("assignment_transport")
            != profile["assignment_transport"]
            for record in preflight.tile_trace
            if record.get("accepted_level") in {"L0", "L1"}
        )
    ):
        raise RuntimeError("DepthSplat soft-mixture source contract changed")
    return {
        "profile": dict(profile),
        "route": {
            "planned": {level: int(planned[level]) for level in ("L0", "L1", "Full")},
            "accepted_before_final_full_promotion": {
                level: int(accepted[level]) for level in ("L0", "L1", "Full")
            },
            "final": dict(final_route.events["route_counts"]),
            "promoted_full_tiles": int(preflight.events["promoted_full_tiles"]),
            "l0_to_l1_retry_tiles": sum(
                record.get("planned_route") == "L0"
                and record.get("accepted_level") == "L1"
                for record in preflight.tile_trace
            ),
            "rejection_reasons": dict(sorted(reasons.items())),
        },
        "certificate": {
            "name": preflight.events["coverage_certificate"],
            "sha256": _require_sha256(
                preflight.events.get("coverage_certificate_sha256"), "certificate"
            ),
            "materialization_session_sha256": _require_sha256(
                preflight.events.get("materialization_session_sha256"),
                "materialization session",
            ),
            "route_session_sha256": _require_sha256(
                final_route.events.get("route_session_sha256"), "route session"
            ),
            "fixed_moment_covariance_scale": 1.0,
            "support_containment_guard": False,
            "projected_domain_guard": False,
            "summary": certificate_summary,
        },
        "kernel_closure": {
            "route_guard_policy": preflight.events[
                "mixture_kernel_closure_guard_policy"
            ],
            "evidence_policy": profile["kernel_closure_evidence_policy"],
            "aggregate_sha256": kernel_closure["sha256"],
            "strict_maximum_relative_risk": preflight.events[
                "mixture_kernel_closure_strict_maximum_relative_risk"
            ],
            "l0_to_l1_tile_count": l0_to_l1_kernel_tiles,
            "full_promotion_tile_count": kernel_full_promotions,
            "calibrated_threshold": calibrated,
            "direct_threshold": direct_threshold,
            "frozen_guard": frozen_kernel_guard,
            "summary": kernel_closure,
        },
        "trace_bindings": {
            "route_plan_tile_trace_sha256": _require_live_tile_trace(
                plan.tile_trace,
                expected_sha256=plan.events.get("tile_trace_sha256"),
                label="plan",
            ),
            "preflight_tile_trace_sha256": preflight_trace_sha256,
            "final_route_tile_trace_sha256": final_trace_sha256,
        },
        "assignment": {
            "semantics": preflight.events["assignment_feature_semantics"],
            "feature_map_sha256": _require_sha256(
                preflight.events.get("assignment_feature_map_sha256"),
                "assignment feature map",
            ),
        },
        "source_access": {
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "omitted_s3_attributes_accessed": False,
            "target_access_before_packet_commit": False,
        },
    }


def _direct_conditional_source_summary(
    *, plan: Any, preflight: Any, final_route: Any, profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize the source-only direct route without risk-gate telemetry."""

    planned = Counter()
    accepted = Counter()
    reasons = Counter()
    for record in plan.tile_trace:
        route = record.get("pre_guard_route")
        if route not in {"L0", "L1", "Full"}:
            raise RuntimeError("DepthSplat direct conditional planned route is invalid")
        planned[route] += 1
    for record in preflight.tile_trace:
        if record.get("accepted") is True:
            level = record.get("accepted_level")
            if level not in {"L0", "L1", "Full"}:
                raise RuntimeError("DepthSplat direct conditional accepted route is invalid")
            accepted[level] += 1
        elif record.get("attempted") is True:
            reason = record.get("reason")
            if not isinstance(reason, str) or not reason:
                raise RuntimeError("DepthSplat direct conditional rejection lacks a reason")
            reasons[reason] += 1
    preflight_trace_sha256 = _require_live_tile_trace(
        preflight.tile_trace,
        expected_sha256=preflight.events.get("tile_trace_sha256"),
        label="direct conditional preflight",
    )
    final_trace_sha256 = _require_live_tile_trace(
        final_route.tile_trace,
        expected_sha256=final_route.events.get("tile_trace_sha256"),
        label="direct conditional final route",
    )
    initial_binding = preflight.events.get("initial_binding")
    coverage_payload = preflight.events.get("coverage_certificate_payload")
    requested_coverage_scale = profile["maximum_coverage_covariance_scale"]
    observed_coverage_scale = preflight.events.get(
        "coverage_max_moment_covariance_scale"
    )
    observed_containment_lhs = preflight.events.get(
        "coverage_max_containment_lhs_after_scale"
    )
    if (
        preflight.events.get("execution_profile")
        != profile["materialization_profile"]
        or preflight.events.get("route_isolation") != profile["route_isolation"]
        or preflight.events.get("soft_mixture_certificate_aggregate") is not None
        or preflight.events.get("mixture_kernel_closure_aggregate") is not None
        or preflight.events.get("mixture_kernel_closure_frozen_guard") is not None
        or preflight.events.get("coverage_certificate")
        != DEPTHSPLAT_COVERAGE_CERTIFICATE
        or preflight.events.get("maximum_coverage_covariance_scale")
        != requested_coverage_scale
        or not isinstance(coverage_payload, Mapping)
        or coverage_payload.get("schema") != DEPTHSPLAT_COVERAGE_CERTIFICATE
        or coverage_payload.get("maximum_covariance_scale")
        != requested_coverage_scale
        or coverage_payload.get("shape_aware_virtual_2sigma_support") is not True
        or coverage_payload.get("tile_trace_sha256") != preflight_trace_sha256
        or isinstance(observed_coverage_scale, bool)
        or not isinstance(observed_coverage_scale, (int, float))
        or not math.isfinite(float(observed_coverage_scale))
        or not 1.0 <= float(observed_coverage_scale) <= float(requested_coverage_scale)
        or isinstance(observed_containment_lhs, bool)
        or not isinstance(observed_containment_lhs, (int, float))
        or not math.isfinite(float(observed_containment_lhs))
        or float(observed_containment_lhs) > 2.0 + 1e-5
        or preflight.events.get("assignment_feature_semantics")
        != profile["assignment_feature_semantics"]
        or not isinstance(initial_binding, Mapping)
        or initial_binding.get("assignment_feature_semantics")
        != profile["assignment_feature_semantics"]
        or initial_binding.get("assignment_feature_map_sha256")
        != preflight.events.get("assignment_feature_map_sha256")
        or final_route.events.get("assignment_feature_semantics")
        != profile["assignment_feature_semantics"]
        or final_route.events.get("assignment_feature_map_sha256")
        != preflight.events.get("assignment_feature_map_sha256")
        or any(
            record.get("assignment_feature_semantics")
            != profile["assignment_feature_semantics"]
            for record in preflight.tile_trace
            if record.get("accepted_level") in {"L0", "L1"}
        )
    ):
        raise RuntimeError("DepthSplat direct conditional source contract changed")
    return {
        "profile": dict(profile),
        "route": {
            "planned": {level: int(planned[level]) for level in ("L0", "L1", "Full")},
            "accepted_before_final_full_promotion": {
                level: int(accepted[level]) for level in ("L0", "L1", "Full")
            },
            "final": dict(final_route.events["route_counts"]),
            "promoted_full_tiles": int(preflight.events["promoted_full_tiles"]),
            "rejection_reasons": dict(sorted(reasons.items())),
        },
        "materialization": {
            "merge_semantics": profile["merge_semantics"],
            "assignment_transport": profile["assignment_transport"],
            "coverage_closed_support": True,
            "maximum_coverage_covariance_scale": float(requested_coverage_scale),
            "observed_maximum_covariance_scale": float(observed_coverage_scale),
            "maximum_containment_lhs_after_scale": float(observed_containment_lhs),
            "coverage_certificate": DEPTHSPLAT_COVERAGE_CERTIFICATE,
            "kernel_risk_guard_used": False,
            "soft_mixture_certificate_used": False,
        },
        "trace_bindings": {
            "route_plan_tile_trace_sha256": _require_live_tile_trace(
                plan.tile_trace,
                expected_sha256=plan.events.get("tile_trace_sha256"),
                label="direct conditional plan",
            ),
            "preflight_tile_trace_sha256": preflight_trace_sha256,
            "final_route_tile_trace_sha256": final_trace_sha256,
        },
        "assignment": {
            "semantics": preflight.events["assignment_feature_semantics"],
            "feature_map_sha256": _require_sha256(
                preflight.events.get("assignment_feature_map_sha256"),
                "direct conditional assignment feature map",
            ),
        },
        "source_access": {
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "omitted_s3_attributes_accessed": False,
            "target_access_before_packet_commit": False,
        },
    }


def _depthsplat_context_depth_bounds(
    context: Mapping[str, Any], *, view_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize native scalar or per-view bounds to [1,V] tensors."""

    bounds: list[torch.Tensor] = []
    for key in ("near", "far"):
        value = context.get(key)
        if not torch.is_tensor(value) or value.numel() not in {1, view_count}:
            raise RuntimeError("DepthSplat FSDR source bounds are invalid")
        if value.numel() == 1:
            value = value.reshape(1, 1).expand(1, view_count)
        else:
            value = value.reshape(1, view_count)
        if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
            raise RuntimeError("DepthSplat FSDR source bounds are invalid")
        bounds.append(value)
    return bounds[0], bounds[1]


def _capture_depthsplat_execution_with_fsdr_inputs(
    encoder: Any, context: Mapping[str, Any]
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Capture exact FSDR candidate evidence during the one native encoder pass."""

    predictor = getattr(encoder, "depth_predictor", None)
    if predictor is None or not callable(getattr(predictor, "register_forward_hook", None)):
        raise RuntimeError("DepthSplat encoder has no native depth predictor")
    captured: dict[str, Any] = {}

    def capture_predictor(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if "results" in captured or not isinstance(output, Mapping):
            raise RuntimeError("DepthSplat depth predictor boundary changed")
        captured["results"] = output

    predictor_module = importlib.import_module(type(predictor).__module__)
    native_warp = getattr(predictor_module, "warp_with_pose_depth_candidates", None)
    if not callable(native_warp):
        raise RuntimeError("DepthSplat native plane-sweep function is unavailable")
    captured_candidate_depths: list[torch.Tensor] = []

    def capture_warp(*args: Any, **kwargs: Any) -> torch.Tensor:
        candidate_depths = kwargs.get("depth", args[3] if len(args) > 3 else None)
        if not torch.is_tensor(candidate_depths):
            raise RuntimeError("DepthSplat plane-sweep candidates are unavailable")
        captured_candidate_depths.append(candidate_depths.detach().clone())
        return native_warp(*args, **kwargs)

    handle = predictor.register_forward_hook(capture_predictor)
    setattr(predictor_module, "warp_with_pose_depth_candidates", capture_warp)
    try:
        execution = capture_depthsplat_native_execution(
            encoder, context, source_root=ROOT / "depthsplat"
        )
    finally:
        setattr(predictor_module, "warp_with_pose_depth_candidates", native_warp)
        handle.remove()
    results = captured.get("results")
    if not isinstance(results, Mapping):
        raise RuntimeError("DepthSplat FSDR source capture is unavailable")
    if not torch.is_tensor(context.get("image")):
        raise RuntimeError("DepthSplat FSDR source context image is unavailable")
    batch, views, _channels, height, width = context["image"].shape
    if batch != 1:
        raise RuntimeError("DepthSplat FSDR source capture requires batch size one")
    tensors = extract_depthsplat_execution_tensors(
        results,
        batch_size=batch,
        view_count=views,
        image_height=height,
        image_width=width,
    )
    probability_scales = list(results["match_probs"])
    candidate_scales = align_depthsplat_plane_sweep_candidate_scales(
        probability_scales,
        captured_candidate_depths,
        batch_size=batch,
        view_count=views,
    )
    mono_features = tensors.mono_features.reshape(
        batch,
        views,
        tensors.mono_features.shape[1],
        tensors.mono_features.shape[2],
        tensors.mono_features.shape[3],
    )
    if mono_features.shape[:2] != (batch, views):
        raise RuntimeError("DepthSplat FSDR DINO feature views changed")
    return execution, {
        # FSDR hashes the semantic S1 representation, while every candidate
        # set below is captured from the matching call that consumed it.
        "feature_source": "depthsplat-mono-dinov2-v1",
        "scales": tuple(
            {
                "features": mono_features,
                "probabilities": probability.reshape(
                    batch,
                    views,
                    probability.shape[1],
                    probability.shape[2],
                    probability.shape[3],
                ),
                "candidates": candidates,
            }
            for probability, candidates in zip(probability_scales, candidate_scales)
        ),
    }


def _fsdr_scale_cache_sizes(scale_count: int) -> tuple[int, ...]:
    """Partition the fixed FSDR CAM budget across native candidate scales."""

    if (
        isinstance(scale_count, bool)
        or not isinstance(scale_count, int)
        or not 0 < scale_count <= FSDR_CACHE_SIZE
    ):
        raise ValueError("DepthSplat FSDR scale count is invalid")
    base, remainder = divmod(FSDR_CACHE_SIZE, scale_count)
    return tuple(base + int(index < remainder) for index in range(scale_count))


def _run_exact_fsdr_scale(
    scale_input: Mapping[str, Any], *, scale_index: int, cache_size: int
) -> tuple[FSDRSimulator, dict[str, Any]]:
    """Execute FSDR once for one native cost-volume scale."""
    features = scale_input.get("features")
    probabilities = scale_input.get("probabilities")
    candidates = scale_input.get("candidates")
    if (
        not torch.is_tensor(features)
        or not torch.is_tensor(probabilities)
        or not torch.is_tensor(candidates)
        or features.ndim != 5
        or probabilities.ndim != 5
        or candidates.shape != probabilities.shape
        or features.shape[:2] != probabilities.shape[:2]
    ):
        raise RuntimeError("DepthSplat FSDR scale inputs are invalid")
    candidate_count = int(probabilities.shape[2])
    frames = prepare_fsdr_candidate_frames(features, probabilities, candidates)
    if not frames:
        raise RuntimeError("DepthSplat FSDR scale inputs are empty")
    frame_height, frame_width = frames[0][4]
    if frame_height % TILE_SIZE or frame_width % TILE_SIZE:
        raise RuntimeError("DepthSplat FSDR scale is not tiled by four")
    simulator = FSDRSimulator(
        feature_dim=int(features.shape[2]),
        cache_size=cache_size,
        hamming_threshold=FSDR_HAMMING_THRESHOLD,
        num_depth_candidates=candidate_count,
        guidance_policy=FSDR_GUIDANCE_POLICY,
        depth_consistency_threshold=FSDR_DEPTH_CONSISTENCY_THRESHOLD,
        tile_size=TILE_SIZE,
    )
    schedule = tile_probe_pixel_order(
        height=frame_height, width=frame_width, tile_size=TILE_SIZE
    )
    pixels_per_frame = frame_height * frame_width
    for view_index, (
        frame_features,
        candidate_anchors,
        top1_indices,
        candidate_values,
        current_shape,
    ) in enumerate(frames):
        if current_shape != (frame_height, frame_width):
            raise RuntimeError("DepthSplat FSDR candidate shapes differ by view")
        simulator.begin_frame()
        simulator.process_discrete_frame(
            frame_features,
            candidate_anchors,
            top1_indices,
            candidate_values,
            width=frame_width,
            pixel_order=schedule,
            pixel_index_offset=view_index * pixels_per_frame,
        )
    summary = simulator.get_summary()
    expected_pixels = int(features.shape[1]) * pixels_per_frame
    if (
        summary.get("total_pixels") != expected_pixels
        or summary.get("full_depth_evaluations") != expected_pixels * candidate_count
        or summary.get("executed_depth_evaluations", 0)
        > summary["full_depth_evaluations"]
        or summary.get("discrete_candidate_evidence") is not True
    ):
        raise RuntimeError("DepthSplat FSDR scale accounting changed")
    return simulator, {
        "scale_index": scale_index,
        "frame_shape": [frame_height, frame_width],
        "cache_size": cache_size,
        "num_depth_candidates": candidate_count,
        "narrowed_depth_candidates": candidate_count // 4,
        "summary": summary,
    }


def _candidate_work_guided_rate(
    *, full_evaluations: int, executed_evaluations: int
) -> float:
    """Report the D/4-equivalent guidance implied by candidate-work savings."""

    if (
        isinstance(full_evaluations, bool)
        or not isinstance(full_evaluations, int)
        or full_evaluations <= 0
        or isinstance(executed_evaluations, bool)
        or not isinstance(executed_evaluations, int)
        or not 0 < executed_evaluations <= full_evaluations
    ):
        raise ValueError("DepthSplat FSDR candidate-work counters are invalid")
    saving_rate = 1.0 - executed_evaluations / full_evaluations
    guided_rate = saving_rate / 0.75
    if not 0.0 <= guided_rate <= 1.0 + 1e-8:
        raise ValueError("DepthSplat FSDR candidate-work equivalent is invalid")
    return min(guided_rate, 1.0)


def _aggregate_exact_fsdr_scale_runs(
    runs: Sequence[tuple[FSDRSimulator, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate independent native scales without mixing their cache domains."""
    if not runs:
        raise RuntimeError("DepthSplat FSDR has no executed scales")
    integer_fields = (
        "frames_started",
        "total_pixels",
        "cache_hits",
        "cache_misses",
        "guided",
        "guided_in_window",
        "guided_out_window",
        "guided_top1_covered",
        "guided_top1_missed",
        "discrete_top1_pixels",
        "depth_inconsistent",
        "hamming_hits",
        "local_valid_hits",
        "local_invalid_fallbacks",
        "hit_no_guide",
        "full_compute",
        "total_reuse",
        "total_validated",
        "full_depth_evaluations",
        "executed_depth_evaluations",
        "feature_buffer_bytes_baseline",
        "feature_buffer_bytes_actual",
    )
    summary = {field: sum(simulator.stats[field] for simulator, _ in runs) for field in integer_fields}
    total = summary["total_pixels"]
    if total <= 0:
        raise RuntimeError("DepthSplat FSDR scales contain no pixels")
    errors = [
        error for simulator, _ in runs for error in simulator.stats["depth_errors"]
    ]
    exact_guided = summary["guided_top1_covered"] + summary["guided_top1_missed"]
    guided_pixel_rate = summary["guided"] / total
    candidate_work_equivalent_guided_rate = _candidate_work_guided_rate(
        full_evaluations=summary["full_depth_evaluations"],
        executed_evaluations=summary["executed_depth_evaluations"],
    )
    summary.update(
        {
            "hit_rate": summary["cache_hits"] / total,
            # Table 2 defines Guided Rate as the fraction of pixels that take
            # the narrowed candidate path. Candidate-work savings are reported
            # separately because native scales use different candidate counts.
            "guided_rate": guided_pixel_rate,
            "guided_pixel_rate": guided_pixel_rate,
            "candidate_work_equivalent_guided_rate": (
                candidate_work_equivalent_guided_rate
            ),
            "reuse_rate": guided_pixel_rate,
            "validated_rate": guided_pixel_rate,
            "depth_evaluation_saving_rate": (
                1.0
                - summary["executed_depth_evaluations"]
                / summary["full_depth_evaluations"]
            ),
            "in_window_rate": summary["guided_in_window"] / max(summary["guided"], 1),
            "discrete_candidate_evidence": (
                summary["discrete_top1_pixels"] == total
                and exact_guided == summary["guided"]
            ),
            "top1_coverage": (
                summary["guided_top1_covered"] / exact_guided
                if exact_guided
                else None
            ),
            "hit_no_guide_rate": summary["hit_no_guide"] / total,
            "full_compute_rate": summary["full_compute"] / total,
            "depth_cache_size": sum(len(simulator.cache) for simulator, _ in runs),
            "reuse_data_size": sum(len(simulator.reuse_data) for simulator, _ in runs),
            "depth_error": {
                "count": len(errors),
                "mean": float(np.mean(errors)) if errors else 0.0,
                "max": float(np.max(errors)) if errors else 0.0,
                "p95": float(np.percentile(errors, 95)) if errors else 0.0,
            },
            "depth_evaluations_available": True,
            "feature_buffer_bytes_available": True,
            "feature_buffer_element_bytes": runs[0][0].feature_buffer_element_bytes,
            "guidance_policy": FSDR_GUIDANCE_POLICY,
            "depth_consistency_threshold": FSDR_DEPTH_CONSISTENCY_THRESHOLD,
            "local_depth_tile_size": TILE_SIZE,
        }
    )
    return summary


def _collect_source_fsdr_trace(
    execution: Any, *, exact_inputs: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Count the target-free FSDR path from the captured native route inputs.

    DepthSplat's selected-output capture exposes the source feature map and
    final z-depth used by SAES, but not its per-pixel cost-volume candidates.
    This is therefore an execution-bound development trace, not a Figure 11
    timing claim or discrete-top1 quality proof.
    """

    features = execution.routing_features
    depths = execution.routing_z_depths
    if (
        not torch.is_tensor(features)
        or not torch.is_tensor(depths)
        or features.ndim != 5
        or depths.ndim != 4
        or features.shape[:2] != depths.shape[:2]
        or features.shape[-2:] != depths.shape[-2:]
        or features.shape[0] != 1
        or features.shape[2] <= 0
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(depths).all())
    ):
        raise RuntimeError("DepthSplat FSDR source trace inputs are invalid")
    height, width = (int(value) for value in features.shape[-2:])
    if height % TILE_SIZE or width % TILE_SIZE:
        raise RuntimeError("DepthSplat FSDR source trace requires complete tiles")
    exact = exact_inputs is not None
    if exact:
        if not isinstance(exact_inputs, Mapping):
            raise RuntimeError("DepthSplat FSDR candidate inputs are invalid")
        scale_inputs = exact_inputs.get("scales")
        feature_source = exact_inputs.get(
            "feature_source", "captured-exact-cost-volume-feature-v1"
        )
        if (
            not isinstance(scale_inputs, (tuple, list))
            or not scale_inputs
            or not all(isinstance(item, Mapping) for item in scale_inputs)
            or not isinstance(feature_source, str)
            or not feature_source
        ):
            raise RuntimeError("DepthSplat FSDR candidate inputs are invalid")
        cache_sizes = _fsdr_scale_cache_sizes(len(scale_inputs))
        scale_runs = tuple(
            _run_exact_fsdr_scale(
                scale_input,
                scale_index=scale_index,
                cache_size=cache_sizes[scale_index],
            )
            for scale_index, scale_input in enumerate(scale_inputs)
        )
        summary = _aggregate_exact_fsdr_scale_runs(scale_runs)
        scale_summaries = [dict(scale_summary) for _, scale_summary in scale_runs]
        candidate_count = scale_summaries[0]["num_depth_candidates"]
        frame_height, frame_width = scale_summaries[0]["frame_shape"]
    else:
        candidate_count = FSDR_NUM_DEPTH_CANDIDATES
        fsdr_features = features
        feature_source = "saes-routing-feature-development-proxy-v1"
        frame_height, frame_width = height, width
        frames = []
    if not exact:
        simulator = FSDRSimulator(
            feature_dim=int(fsdr_features.shape[2]),
            cache_size=FSDR_CACHE_SIZE,
            hamming_threshold=FSDR_HAMMING_THRESHOLD,
            num_depth_candidates=candidate_count,
            guidance_policy=FSDR_GUIDANCE_POLICY,
            depth_consistency_threshold=FSDR_DEPTH_CONSISTENCY_THRESHOLD,
            tile_size=TILE_SIZE,
        )
        schedule = tile_probe_pixel_order(
            height=frame_height,
            width=frame_width,
            tile_size=TILE_SIZE,
        )
        for view_index in range(int(features.shape[1])):
            frame_features, frame_depths = prepare_fsdr_frame(
                features[:, view_index : view_index + 1],
                depths[:, view_index : view_index + 1],
                height=height,
                width=width,
            )
            simulator.begin_frame()
            simulator.process_frame(
                frame_features,
                frame_depths,
                width=width,
                pixel_order=schedule,
            )
        summary = simulator.get_summary()
        if (
            summary.get("total_pixels")
            != int(features.shape[1]) * frame_height * frame_width
            or summary.get("full_depth_evaluations")
            != summary["total_pixels"] * candidate_count
            or summary.get("executed_depth_evaluations", 0)
            > summary["full_depth_evaluations"]
            or summary.get("discrete_candidate_evidence") is not False
        ):
            raise RuntimeError("DepthSplat FSDR source trace accounting changed")
        scale_summaries = []
    return {
        "mode": (
            "source-exact-multiscale-cost-volume-candidates-v2"
            if exact
            else "source-feature-final-z-depth-development-proxy-v1"
        ),
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "discrete_candidate_evidence": exact,
        "context_view_count": int(features.shape[1]),
        "feature_source": feature_source,
        "frame_shape": [frame_height, frame_width],
        "scales": scale_summaries,
        "tile_schedule": "probe-first-row-major-v1",
        "parameters": {
            "cache_size": FSDR_CACHE_SIZE,
            "per_scale_cache_sizes": (
                list(cache_sizes) if exact else [FSDR_CACHE_SIZE]
            ),
            "hamming_threshold": FSDR_HAMMING_THRESHOLD,
            "guidance_policy": FSDR_GUIDANCE_POLICY,
            "num_depth_candidates": candidate_count,
            "narrowed_depth_candidates": candidate_count // 4,
            "depth_consistency_threshold": FSDR_DEPTH_CONSISTENCY_THRESHOLD,
        },
        "summary": summary,
    }


def _fsdr_cost_volume_cycles(
    *,
    scale_input: Mapping[str, Any],
    scale_summary: Mapping[str, Any],
    native_feature_channels: int,
) -> int:
    """Reprice one captured plane sweep using its executed FSDR candidates."""

    features = scale_input.get("features")
    probabilities = scale_input.get("probabilities")
    if (
        not torch.is_tensor(features)
        or not torch.is_tensor(probabilities)
        or features.ndim != 5
        or probabilities.ndim != 5
        or features.shape[:2] != probabilities.shape[:2]
    ):
        raise RuntimeError("DepthSplat FSDR cycle inputs are invalid")
    batch_size, views, _feature_channels, _feature_height, _feature_width = features.shape
    if isinstance(native_feature_channels, bool) or native_feature_channels <= 0:
        raise RuntimeError("DepthSplat native cost-volume channel count is invalid")
    feature_channels = int(native_feature_channels)
    depth_candidates = int(probabilities.shape[2])
    height, width = int(probabilities.shape[-2]), int(probabilities.shape[-1])
    source_pairs = batch_size * views * (views - 1)
    if source_pairs <= 0:
        raise RuntimeError("DepthSplat FSDR cycle inputs require multiple views")
    full_cycles, full_evaluations = DepthSplatDepthPredictorSim._plane_sweep_cycles(
        batch_size=batch_size,
        view_count=views,
        feature_channels=feature_channels,
        depth_candidates=depth_candidates,
        height=height,
        width=width,
    )
    expected_evaluations = _nonnegative_int(
        scale_summary.get("full_depth_evaluations"), label="full FSDR evaluations"
    )
    executed_evaluations = _nonnegative_int(
        scale_summary.get("executed_depth_evaluations"),
        label="executed FSDR evaluations",
    )
    if expected_evaluations != full_evaluations or executed_evaluations > full_evaluations:
        raise RuntimeError("DepthSplat FSDR cycle candidates changed")
    channel_batches = (feature_channels + 31) // 32
    bilinear_cycles_per_candidate = (
        ENCODER_CYCLES["bilinear_coord"]
        + ENCODER_CYCLES["bilinear_sample"]
        + ENCODER_CYCLES["bilinear_interpolate"]
        + 1
    ) * channel_batches
    backproject_cycles = source_pairs * 3 * 3 * height * width // 1024
    return int(
        backproject_cycles
        + executed_evaluations * 2 * 3 * 3 // 1024
        + executed_evaluations * bilinear_cycles_per_candidate
        + executed_evaluations * feature_channels // 1024
    )


def _fsdr_feature_buffer_transfer_cycles(*, byte_count: Any, label: str) -> int:
    """Serialize measured FP16 FeatureBuffer traffic through the RTL ports."""

    bytes_to_transfer = _nonnegative_int(byte_count, label=label)
    return math.ceil(bytes_to_transfer / FSDR_FEATURE_BUFFER_BYTES_PER_CYCLE)


def _saes_fused_s2_schedule(
    *,
    l0_tiles: int,
    l1_tiles: int,
    full_tiles: int,
    primitives_per_pixel: int,
    tile_size: int,
) -> dict[str, int | float]:
    """Describe the route's probe-first S2 request, not executed work."""

    l0_tiles = _nonnegative_int(l0_tiles, label="L0 tile count")
    l1_tiles = _nonnegative_int(l1_tiles, label="L1 tile count")
    full_tiles = _nonnegative_int(full_tiles, label="Full tile count")
    primitives_per_pixel = _nonnegative_int(
        primitives_per_pixel, label="primitives per pixel"
    )
    if primitives_per_pixel <= 0 or tile_size != TILE_SIZE:
        raise RuntimeError("DepthSplat fused SAES S2 schedule is invalid")
    pixels_per_tile = tile_size * tile_size
    dense_positions = (
        (l0_tiles + l1_tiles + full_tiles)
        * pixels_per_tile
        * primitives_per_pixel
    )
    scheduled_positions = (
        l0_tiles * 4 * primitives_per_pixel
        + l1_tiles * 12 * primitives_per_pixel
        + full_tiles * pixels_per_tile * primitives_per_pixel
    )
    if scheduled_positions <= 0 or scheduled_positions > dense_positions:
        raise RuntimeError("DepthSplat fused SAES S2 schedule exceeds dense work")
    return {
        "dense_s2_positions": dense_positions,
        "scheduled_s2_positions": scheduled_positions,
        "skipped_s2_positions": dense_positions - scheduled_positions,
        "s2_position_saving_rate": 1.0 - scheduled_positions / dense_positions,
    }


def _executed_saes_s2_s3_work(
    *,
    execution_events: Mapping[str, Any],
    planned_s2_schedule: Mapping[str, Any],
    selected_output_head_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind cycle attribution to source work that actually ran.

    A route plan is not execution evidence.  The dense capture used by the
    quality path intentionally has no SAES S2/S3 saving, even though it can
    describe the positions a future probe-first producer would request.
    """

    if not isinstance(execution_events, Mapping):
        raise RuntimeError("DepthSplat execution events are invalid")
    dense_s2_positions = _nonnegative_int(
        planned_s2_schedule.get("dense_s2_positions"),
        label="dense S2 positions",
    )
    planned_s2_positions = _nonnegative_int(
        planned_s2_schedule.get("scheduled_s2_positions"),
        label="planned S2 positions",
    )
    dense_head_macs = _nonnegative_int(
        selected_output_head_schedule.get("dense_head_macs"),
        label="dense selected-head MACs",
    )
    planned_head_macs = _nonnegative_int(
        selected_output_head_schedule.get("scheduled_head_macs"),
        label="planned selected-head MACs",
    )
    if (
        dense_s2_positions <= 0
        or planned_s2_positions <= 0
        or planned_s2_positions > dense_s2_positions
        or dense_head_macs <= 0
        or planned_head_macs <= 0
        or planned_head_macs > dense_head_macs
    ):
        raise RuntimeError("DepthSplat planned S2/S3 work is invalid")

    planned_schedule_sha256 = canonical_json_sha256(
        {
            "s2": {
                "dense_positions": dense_s2_positions,
                "planned_positions": planned_s2_positions,
            },
            "s3": {
                "dense_head_macs": dense_head_macs,
                "planned_head_macs": planned_head_macs,
            },
        }
    )
    verified = execution_events.get("whole_pipeline_s2_s3_sparse_execution_verified")
    if verified is False:
        return {
            "execution_verified": False,
            "reason": "native_dense_capture",
            "planned_schedule_sha256": planned_schedule_sha256,
            "dense_s2_positions": dense_s2_positions,
            "planned_s2_positions": planned_s2_positions,
            "executed_s2_positions": dense_s2_positions,
            "dense_head_macs": dense_head_macs,
            "planned_head_macs": planned_head_macs,
            "executed_head_macs": dense_head_macs,
            "s2_position_saving_rate": 0.0,
            "head_mac_saving_rate": 0.0,
        }
    if verified is not True:
        raise RuntimeError("DepthSplat S2/S3 execution-verification flag is invalid")
    sparse = execution_events.get("sparse_s2_s3_execution")
    if not isinstance(sparse, Mapping):
        raise RuntimeError("DepthSplat verified sparse execution has no counters")
    if (
        sparse.get("contract_version")
        != "depthsplat-probe-first-s2-s3-execution-v1"
        or sparse.get("planned_schedule_sha256") != planned_schedule_sha256
    ):
        raise RuntimeError("DepthSplat sparse execution binding changed")
    executed_s2_positions = _nonnegative_int(
        sparse.get("executed_s2_positions"), label="executed S2 positions"
    )
    executed_head_macs = _nonnegative_int(
        sparse.get("executed_head_macs"), label="executed selected-head MACs"
    )
    if (
        executed_s2_positions < planned_s2_positions
        or executed_s2_positions > dense_s2_positions
        or executed_head_macs < planned_head_macs
        or executed_head_macs > dense_head_macs
    ):
        raise RuntimeError("DepthSplat sparse execution counters are invalid")
    return {
        "execution_verified": True,
        "reason": "source-bound-sparse-executor",
        "planned_schedule_sha256": planned_schedule_sha256,
        "dense_s2_positions": dense_s2_positions,
        "planned_s2_positions": planned_s2_positions,
        "executed_s2_positions": executed_s2_positions,
        "dense_head_macs": dense_head_macs,
        "planned_head_macs": planned_head_macs,
        "executed_head_macs": executed_head_macs,
        "s2_position_saving_rate": 1.0 - executed_s2_positions / dense_s2_positions,
        "head_mac_saving_rate": 1.0 - executed_head_macs / dense_head_macs,
    }


def _mechanism_cycle_summary(
    *,
    encoder: Any,
    context: Mapping[str, Any],
    execution: Any,
    fsdr_execution_inputs: Mapping[str, Any],
    fsdr_source_trace: Mapping[str, Any],
    final_route: Any,
    materialized: Any,
    selected_output_head_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive Figure 11 and Tables 2-3 inputs from one committed source trace.

    FSDR cycles use captured candidate work. SAES cycles use only counters
    from a verified sparse producer; a dense quality capture remains dense
    regardless of the route's probe-first request.
    """

    scales = fsdr_execution_inputs.get("scales")
    source_scales = fsdr_source_trace.get("scales")
    if (
        not isinstance(scales, tuple)
        or not isinstance(source_scales, list)
        or len(scales) != len(source_scales)
        or not scales
        or not torch.is_tensor(execution.routing_features)
        or not torch.is_tensor(context.get("image"))
    ):
        raise RuntimeError("DepthSplat cycle trace is incomplete")
    native_scales: list[tuple[int, int, int]] = []
    fsdr_cost_volume_cycles = 0
    native_feature_channels = int(
        getattr(encoder.depth_predictor, "feature_channels", 0)
    )
    for scale_input, scale_record in zip(scales, source_scales, strict=True):
        probabilities = scale_input.get("probabilities")
        summary = scale_record.get("summary") if isinstance(scale_record, Mapping) else None
        if not torch.is_tensor(probabilities) or not isinstance(summary, Mapping):
            raise RuntimeError("DepthSplat FSDR scale trace is invalid")
        native_scales.append(
            (
                int(probabilities.shape[2]),
                int(probabilities.shape[-2]),
                int(probabilities.shape[-1]),
            )
        )
        fsdr_cost_volume_cycles += _fsdr_cost_volume_cycles(
            scale_input=scale_input,
            scale_summary=summary,
            native_feature_channels=native_feature_channels,
        )

    # Reuse the native DepthSplat model's existing cycle formula with exactly
    # the captured probability-volume shapes rather than a synthetic scale.
    simulator = DepthSplatDepthPredictorSim(
        device=torch.device("cpu"), use_hw_computation=False
    )
    simulator._depth_predictor = encoder.depth_predictor
    simulator._native_cost_volume_scales = native_scales
    simulator._estimate_pipeline_cycles(execution.routing_features, context["image"])
    no_opt_breakdown = simulator.get_cycle_breakdown().to_dict()
    if no_opt_breakdown["cost_volume"] <= 0 or fsdr_cost_volume_cycles <= 0:
        raise RuntimeError("DepthSplat cycle model has no cost-volume work")
    if fsdr_cost_volume_cycles > no_opt_breakdown["cost_volume"]:
        raise RuntimeError("DepthSplat FSDR increased the native cost volume")

    route_counts = final_route.events.get("route_counts")
    if not isinstance(route_counts, Mapping):
        raise RuntimeError("DepthSplat final route has no counters")
    l0_tiles = _nonnegative_int(route_counts.get("L0"), label="L0 tile count")
    l1_tiles = _nonnegative_int(route_counts.get("L1"), label="L1 tile count")
    full_tiles = _nonnegative_int(route_counts.get("Full"), label="Full tile count")
    views, _channels, height, width = execution.dense_raw_head.shape
    if height % TILE_SIZE or width % TILE_SIZE:
        raise RuntimeError("DepthSplat cycle route is not tile aligned")
    total_tiles = l0_tiles + l1_tiles + full_tiles
    if total_tiles != views * (height // TILE_SIZE) * (width // TILE_SIZE):
        raise RuntimeError("DepthSplat final route does not cover all tiles")
    dense_count = int(execution.dense_gaussians.means.shape[1])
    slots = views * height * width
    primitives_per_pixel, remainder = divmod(dense_count, slots)
    if remainder or primitives_per_pixel <= 0:
        raise RuntimeError("DepthSplat Gaussian layout is not pixel aligned")
    planned_saes_s2 = _saes_fused_s2_schedule(
        l0_tiles=l0_tiles,
        l1_tiles=l1_tiles,
        full_tiles=full_tiles,
        primitives_per_pixel=primitives_per_pixel,
        tile_size=TILE_SIZE,
    )
    if planned_saes_s2["dense_s2_positions"] != dense_count:
        raise RuntimeError("DepthSplat fused SAES S2 schedule changed")
    harmonic_coefficients = int(execution.dense_gaussians.harmonics.shape[-1])
    sh_degree = math.isqrt(harmonic_coefficients) - 1
    if (sh_degree + 1) ** 2 != harmonic_coefficients:
        raise RuntimeError("DepthSplat SH layout is not square")
    saes_stats = {
        "total_tiles_processed": total_tiles,
        "level0_tiles": l0_tiles,
        "level1_tiles": l1_tiles,
        "full_tiles": full_tiles,
        "l0_representatives": l0_tiles * 4 * primitives_per_pixel,
        "l1_lightweight_anchors": l1_tiles * 12 * primitives_per_pixel,
        "full_stage3_gaussians": full_tiles * TILE_SIZE * TILE_SIZE * primitives_per_pixel,
    }
    saes_ledger = build_saes_event_ledger(
        saes_stats,
        feature_dim=int(execution.routing_features.shape[2]),
        tile_size=TILE_SIZE,
        sh_degree=sh_degree,
        primitives_per_pixel=primitives_per_pixel,
    )
    saes_overhead = int(saes_ledger["cycles"]["serialized_accounting_cycles"])
    if saes_overhead <= 0:
        raise RuntimeError("DepthSplat SAES ledger has no executed work")
    executed_saes_work = _executed_saes_s2_s3_work(
        execution_events=execution.events,
        planned_s2_schedule=planned_saes_s2,
        selected_output_head_schedule=selected_output_head_schedule,
    )

    fsdr_summary = fsdr_source_trace.get("summary")
    if not isinstance(fsdr_summary, Mapping):
        raise RuntimeError("DepthSplat FSDR summary is invalid")
    feature_buffer_bytes_baseline = _nonnegative_int(
        fsdr_summary.get("feature_buffer_bytes_baseline"),
        label="baseline FeatureBuffer bytes",
    )
    feature_buffer_bytes_actual = _nonnegative_int(
        fsdr_summary.get("feature_buffer_bytes_actual"),
        label="actual FeatureBuffer bytes",
    )
    if (
        feature_buffer_bytes_baseline <= 0
        or feature_buffer_bytes_actual > feature_buffer_bytes_baseline
    ):
        raise RuntimeError("DepthSplat FSDR FeatureBuffer counters are invalid")
    feature_buffer_cycles_baseline = _fsdr_feature_buffer_transfer_cycles(
        byte_count=feature_buffer_bytes_baseline,
        label="baseline FeatureBuffer bytes",
    )
    feature_buffer_cycles_actual = _fsdr_feature_buffer_transfer_cycles(
        byte_count=feature_buffer_bytes_actual,
        label="actual FeatureBuffer bytes",
    )
    saes_s2_fraction = float(executed_saes_work["executed_s2_positions"]) / dense_count
    saes_head_fraction = (
        float(executed_saes_work["executed_head_macs"])
        / float(executed_saes_work["dense_head_macs"])
    )
    saes_feature_buffer_cycles = math.ceil(
        feature_buffer_cycles_baseline * saes_s2_fraction
    )
    combined_feature_buffer_cycles = math.ceil(
        feature_buffer_cycles_actual * saes_s2_fraction
    )

    def variant(
        breakdown: Mapping[str, int],
        *,
        feature_buffer_transfer_cycles: int,
        saes_control_cycles: int = 0,
    ) -> dict[str, Any]:
        stages = {name: int(value) for name, value in breakdown.items() if name != "total"}
        total = (
            sum(stages.values())
            + feature_buffer_transfer_cycles
            + saes_control_cycles
        )
        return {
            "total_cycles": total,
            "depth_predictor_breakdown": {**stages, "total": sum(stages.values())},
            "fsdr_feature_buffer_transfer_cycles": feature_buffer_transfer_cycles,
            "saes_control_and_materialization_cycles": saes_control_cycles,
        }

    fsdr_breakdown = dict(no_opt_breakdown)
    fsdr_breakdown["cost_volume"] = fsdr_cost_volume_cycles
    saes_breakdown = dict(no_opt_breakdown)
    saes_breakdown["cost_volume"] = math.ceil(
        no_opt_breakdown["cost_volume"] * saes_s2_fraction
    )
    saes_breakdown["gaussian_head"] = math.ceil(
        no_opt_breakdown["gaussian_head"] * saes_head_fraction
    )
    combined_breakdown = dict(fsdr_breakdown)
    combined_breakdown["cost_volume"] = math.ceil(
        fsdr_cost_volume_cycles * saes_s2_fraction
    )
    combined_breakdown["gaussian_head"] = saes_breakdown["gaussian_head"]
    no_opt = variant(
        no_opt_breakdown,
        feature_buffer_transfer_cycles=feature_buffer_cycles_baseline,
    )
    fsdr_only = variant(
        fsdr_breakdown,
        feature_buffer_transfer_cycles=feature_buffer_cycles_actual,
    )
    saes_only = variant(
        saes_breakdown,
        feature_buffer_transfer_cycles=saes_feature_buffer_cycles,
        saes_control_cycles=saes_overhead,
    )
    combined = variant(
        combined_breakdown,
        feature_buffer_transfer_cycles=combined_feature_buffer_cycles,
        saes_control_cycles=saes_overhead,
    )
    projected_sparse_datapath = project_depthsplat_sparse_datapath_cycles(
        dense_cycle_breakdown=no_opt_breakdown,
        planned_s2_schedule=planned_saes_s2,
        selected_output_head_schedule=selected_output_head_schedule,
        saes_control_and_materialization_cycles=saes_overhead,
    )
    final_count = int(materialized.dense_slots.numel())
    return {
        "kind": "depthsplat-native-source-trace-cycle-summary-v3",
        "timing_class": (
            "cycle_model_with_verified_s2_s3_execution_and_"
            "dual_port_feature_buffer_transfer"
        ),
        "figure11_eligible": False,
        "figure11_ineligibility": (
            "single-scene development record; Figure 11 requires the fixed "
            "DL3DV aggregate and a verified sparse S2/S3 producer"
        ),
        "variants": {
            "no_opt": no_opt,
            "fsdr_only": fsdr_only,
            "saes_only": saes_only,
            "combined": combined,
        },
        "speedup": {
            "fsdr_only": no_opt["total_cycles"] / fsdr_only["total_cycles"],
            "saes_only": no_opt["total_cycles"] / saes_only["total_cycles"],
            "combined": no_opt["total_cycles"] / combined["total_cycles"],
        },
        "feature_buffer_cycle_model": {
            "rtl_module": "FeatureBuffer",
            "data_width_bits_per_port": FSDR_FEATURE_BUFFER_PORT_BYTES * 8,
            "independent_read_ports": FSDR_FEATURE_BUFFER_PORTS,
            "service_bytes_per_cycle": FSDR_FEATURE_BUFFER_BYTES_PER_CYCLE,
            "baseline_transfer_cycles": feature_buffer_cycles_baseline,
            "actual_transfer_cycles": feature_buffer_cycles_actual,
        },
        "saes_s2_s3_execution": {
            "planned_probe_first_s2_schedule": planned_saes_s2,
            "executed_work": executed_saes_work,
            "cost_volume_cycles_baseline": no_opt_breakdown["cost_volume"],
            "cost_volume_cycles_saes": saes_breakdown["cost_volume"],
            "cost_volume_cycles_combined": combined_breakdown["cost_volume"],
            "feature_buffer_transfer_cycles_baseline": feature_buffer_cycles_baseline,
            "feature_buffer_transfer_cycles_saes": saes_feature_buffer_cycles,
            "feature_buffer_transfer_cycles_combined": combined_feature_buffer_cycles,
        },
        "projected_sparse_datapath_cycles": projected_sparse_datapath,
        "table_2_fsdr": {
            "guided_rate": fsdr_summary["guided_rate"],
            "top1_coverage": fsdr_summary["top1_coverage"],
            "full_depth_evaluations": fsdr_summary["full_depth_evaluations"],
            "executed_depth_evaluations": fsdr_summary["executed_depth_evaluations"],
            "depth_evaluation_saving_rate": 1.0
            - fsdr_summary["executed_depth_evaluations"]
            / fsdr_summary["full_depth_evaluations"],
            "feature_buffer_bytes_baseline": fsdr_summary["feature_buffer_bytes_baseline"],
            "feature_buffer_bytes_actual": fsdr_summary["feature_buffer_bytes_actual"],
            "feature_buffer_reduction_rate": 1.0
            - fsdr_summary["feature_buffer_bytes_actual"]
            / fsdr_summary["feature_buffer_bytes_baseline"],
            "per_scale": source_scales,
        },
        "table_3_saes": {
            "route_counts": {"L0": l0_tiles, "L1": l1_tiles, "Full": full_tiles},
            "dense_gaussian_count": dense_count,
            "final_gaussian_count": final_count,
            "gaussian_saving_rate": (dense_count - final_count) / dense_count,
            "event_ledger": saes_ledger,
            "selected_output_head_schedule": dict(selected_output_head_schedule),
            "dense_s2_s3_execution_verified": bool(
                executed_saes_work["execution_verified"]
            ),
        },
    }


def _build_committed_packet(
    *,
    encoder: Any,
    context: Mapping[str, Any],
    profile: Mapping[str, Any],
    kernel_risk_guard: Any | None = None,
    direct_kernel_risk_threshold: float | None = None,
    execution: Any | None = None,
    fsdr_execution_inputs: Mapping[str, torch.Tensor] | None = None,
    fsdr_source_trace: Mapping[str, Any] | None = None,
    mixture_kernel_closure_measurement_cache: Any | None = None,
    route_isolation: str = "l0_l1",
) -> dict[str, Any]:
    """Run the complete source phase and return only a sealed packet state."""

    if execution is None:
        execution, fsdr_execution_inputs = _capture_depthsplat_execution_with_fsdr_inputs(
            encoder, context
        )
    if execution.routing_features is None or execution.routing_z_depths is None:
        raise RuntimeError("DepthSplat soft-mixture has no source routing tensors")
    _views, _channels, height, width = execution.dense_raw_head.shape
    if height % TILE_SIZE or width % TILE_SIZE:
        raise RuntimeError("DepthSplat soft-mixture image is not tiled by four")
    direct_conditional = (
        profile.get("materialization_profile")
        == DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE
    )
    if direct_conditional:
        if route_isolation != profile.get("route_isolation"):
            raise ValueError("DepthSplat direct conditional route changed")
        plan = build_depthsplat_soft_mixture_normalized_t4_probe_first_plan(
            execution.routing_features,
            execution.routing_z_depths,
            height=height,
            width=width,
            feature_threshold=float(profile["feature_threshold"]),
            depth_threshold=float(profile["depth_threshold"]),
        )
        _validate_direct_conditional_plan(plan, profile=profile)
    else:
        plan = build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan(
            execution.routing_features,
            execution.routing_z_depths,
            height=height,
            width=width,
            feature_threshold=float(profile["feature_threshold"]),
            depth_threshold=float(profile["depth_threshold"]),
        )
        _validate_plan(plan, profile=profile)
    initial_replay = replay_depthsplat_selected_head(
        encoder.gaussian_head,
        execution.gaussian_head_input,
        execution.dense_raw_head,
        plan.selection_mask,
        native_full_mask=plan.full_mask,
    )
    initial_equivalence = _require_equivalent(
        initial_replay.equivalence, "DepthSplat soft-mixture initial selected replay"
    )
    initial_packet = build_depthsplat_sparse_raw_packet(execution, initial_replay)
    consumer = DepthSplatPackedGaussianConsumer(encoder.gaussian_adapter)
    initial_packed = consumer.convert(
        initial_packet,
        image_shape=(height, width),
        native_execution=execution,
        native_full_mask=plan.full_mask,
    )
    initial_adapter_equivalence = _require_equivalent(
        compare_depthsplat_packed_to_dense(initial_packed, execution.dense_gaussians),
        "DepthSplat soft-mixture initial Adapter packet",
    )
    initial_full_passthrough = _require_bitwise_equivalent(
        compare_depthsplat_full_passthrough_to_dense_bitwise(
            initial_packed, execution.dense_gaussians, plan.full_mask
        ),
        "DepthSplat soft-mixture initial Full packet",
    )
    sample_image_grid, get_world_rays, geometry_source = _source_geometry_functions(encoder)
    preflight_started = time.perf_counter()
    preflight = preflight_depthsplat_l0_l1_materialization(
        initial_packet,
        initial_packed,
        plan,
        execution.routing_features,
        execution.routing_z_depths,
        source_sample_image_grid=sample_image_grid,
        source_get_world_rays=get_world_rays,
        maximum_coverage_covariance_scale=profile[
            "maximum_coverage_covariance_scale"
        ],
        execution_profile=profile["materialization_profile"],
        mixture_kernel_closure_frozen_guard=kernel_risk_guard,
        mixture_kernel_closure_maximum_relative_risk=direct_kernel_risk_threshold,
        mixture_kernel_closure_measurement_cache=(
            mixture_kernel_closure_measurement_cache
        ),
        route_isolation=route_isolation,
    )
    preflight_seconds = time.perf_counter() - preflight_started
    final_route = resolve_depthsplat_compact_final_route(plan, preflight)
    source_summary = (
        _direct_conditional_source_summary(
            plan=plan, preflight=preflight, final_route=final_route, profile=profile
        )
        if direct_conditional
        else _source_summary(
            plan=plan, preflight=preflight, final_route=final_route, profile=profile
        )
    )
    source_summary["fsdr"] = dict(
        fsdr_source_trace
        if fsdr_source_trace is not None
        else _collect_source_fsdr_trace(
            execution, exact_inputs=fsdr_execution_inputs
        )
    )
    update_count = int(preflight.update_dense_slots.numel())
    producer_replay = replay_depthsplat_selected_head(
        encoder.gaussian_head,
        execution.gaussian_head_input,
        execution.dense_raw_head,
        final_route.raw_head_request_mask,
        native_full_mask=final_route.full_passthrough_mask,
    )
    producer_equivalence = _require_equivalent(
        producer_replay.equivalence, "DepthSplat soft-mixture final selected replay"
    )
    producer_packet = build_depthsplat_sparse_raw_packet(execution, producer_replay)
    final_packet = subset_depthsplat_sparse_raw_packet(
        producer_packet, final_route.selected_output_mask
    )
    final_packed = consumer.convert(
        final_packet,
        image_shape=(height, width),
        native_execution=execution,
        native_full_mask=final_route.full_passthrough_mask,
    )
    final_adapter_equivalence = _require_equivalent(
        compare_depthsplat_packed_to_dense(final_packed, execution.dense_gaussians),
        "DepthSplat soft-mixture final Adapter packet",
    )
    final_full_passthrough = _require_bitwise_equivalent(
        compare_depthsplat_full_passthrough_to_dense_bitwise(
            final_packed, execution.dense_gaussians, final_route.full_passthrough_mask
        ),
        "DepthSplat soft-mixture final Full packet",
    )
    materialization_started = time.perf_counter()
    materialized = apply_depthsplat_compact_l0_l1_materialization(
        final_packed, preflight, final_route
    )
    materialization_seconds = time.perf_counter() - materialization_started
    materialized_full_passthrough = _require_bitwise_equivalent(
        compare_depthsplat_full_passthrough_to_dense_bitwise(
            materialized, execution.dense_gaussians, final_route.full_passthrough_mask
        ),
        "DepthSplat soft-mixture materialized Full packet",
    )
    if int(materialized.dense_slots.numel()) != int(final_route.selected_output_mask.sum()):
        raise RuntimeError("DepthSplat soft-mixture materialized packet route drifted")
    selected_output_head_schedule = estimate_depthsplat_selected_head_schedule(
        encoder.gaussian_head,
        final_route.raw_head_request_mask,
        emitted_output_mask=final_route.selected_output_mask,
        native_full_mask=final_route.full_passthrough_mask,
    )
    mechanism_cycles = _mechanism_cycle_summary(
        encoder=encoder,
        context=context,
        execution=execution,
        fsdr_execution_inputs=fsdr_execution_inputs,
        fsdr_source_trace=source_summary["fsdr"],
        final_route=final_route,
        materialized=materialized,
        selected_output_head_schedule=selected_output_head_schedule,
    )
    return {
        "execution": execution,
        "height": height,
        "width": width,
        "source_summary": source_summary,
        "packet_committed": True,
        "nonzero_merge_update_slot_count": update_count,
        "materialized": materialized,
        "final_route": final_route,
        "route_isolation": route_isolation,
        "timing": {
            "guard_preflight_wall_seconds": preflight_seconds,
            "materialization_wall_seconds": materialization_seconds,
        },
        "packet_summary": {
            "compact_route_present": update_count > 0,
            "initial_descriptor_count": int(initial_packed.dense_slots.numel()),
            "producer_descriptor_count": int(producer_packet.dense_slots.numel()),
            "final_descriptor_count": int(final_packed.dense_slots.numel()),
            "materialized_descriptor_count": int(materialized.dense_slots.numel()),
            "producer_only_prefetch_descriptor_count": int(
                final_route.events["producer_only_prefetch_descriptor_count"]
            ),
            "initial_selected_head_equivalence": initial_equivalence,
            "producer_selected_head_equivalence": producer_equivalence,
            "initial_adapter_equivalence": initial_adapter_equivalence,
            "final_adapter_equivalence": final_adapter_equivalence,
            "initial_full_passthrough_bitwise": initial_full_passthrough,
            "final_full_passthrough_bitwise": final_full_passthrough,
            "materialized_full_passthrough_bitwise": materialized_full_passthrough,
            "final_attribute_binding_sha256": final_packed.attribute_binding_sha256,
            "materialized_attribute_binding_sha256": materialized.attribute_binding_sha256,
        },
        "geometry_source": geometry_source,
        "mechanism_cycles": mechanism_cycles,
    }


def collect_depthsplat_soft_mixture_sample0_quality_gate(
    *,
    input_root: Path,
    source_audit_root: Path,
    kernel_risk_record: Path | None,
    direct_kernel_risk_threshold: float | None,
    device: torch.device,
    source_sample_index: int = SOURCE_SAMPLE_INDEX,
    route_isolation: str = "l0_l1",
) -> dict[str, Any]:
    """Execute source telemetry and real target quality in a single process."""

    from data.context_only_audit_input import validate_context_only_audit_input

    if kernel_risk_record is not None or direct_kernel_risk_threshold is not None:
        raise ValueError(
            "DepthSplat direct conditional quality does not accept a kernel-risk threshold"
        )
    profile = direct_conditional_t4_profile(route_isolation=route_isolation)
    input_identity = _require_quality_context_identity(
        validate_context_only_audit_input(input_root, model=MODEL),
        source_sample_index=source_sample_index,
    )
    backend_contract = resolve_depthsplat_backend_contract(ROOT)
    backend_identity = freeze_depthsplat_backend_identity(backend_contract)
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    selection = resolve_claim_selection(MODEL, DATASET, ROOT)
    checkpoint_sha256 = cached_sha256_file(experiment.checkpoint)
    loader = create_model_loader(MODEL)
    bundle = loader.load_model(
        str(experiment.checkpoint),
        device=device,
        experiment_name=experiment.experiment,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        hydra_overrides=experiment.hydra_overrides,
    )
    if bundle.decoder is None:
        raise RuntimeError("DepthSplat soft-mixture runner requires the native decoder")
    context_data = load_context_only_audit_data(
        loader, bundle, input_root=Path(input_root), model_name=MODEL
    )
    context_cpu, loaded_calibration = _require_loaded_context_batch(
        context_data.batch, input_identity
    )
    context = {
        key: value.to(bundle.device) if torch.is_tensor(value) else value
        for key, value in context_cpu.items()
    }
    bundle.model.eval()

    with strict_fp32_convolution_execution() as numerical_execution:
        fsdr_source_trace: dict[str, Any] | None = None
        committed = _build_committed_packet(
            encoder=bundle.encoder,
            context=context,
            profile=profile,
            fsdr_source_trace=fsdr_source_trace,
            route_isolation=route_isolation,
        )
        source_summary = committed["source_summary"]
        common = {
            "schema_version": QUALITY_GATE_SCHEMA_VERSION,
            "kind": QUALITY_GATE_KIND,
            "paper_result_eligible": False,
            "model": MODEL,
            "dataset": DATASET,
            "source_sample_index": input_identity["source_sample_index"],
            "scene": input_identity["scene"],
            "context_indices": list(input_identity["context_indices"]),
            "mechanism": SOFT_MIXTURE_MECHANISM,
            "route_policy": {
                "mode": f"normalized_{route_isolation}_direct_conditional",
                "feature_threshold": FEATURE_THRESHOLD,
                "depth_threshold": DEPTH_THRESHOLD,
                "tile_size": TILE_SIZE,
            },
            "source_phase": source_summary,
            "context_only_input": {
                "identity": input_identity,
                "loaded_native_preprocessing": loaded_calibration["native_preprocessing"],
            },
            "backend_identity": backend_identity,
            "checkpoint_sha256": checkpoint_sha256,
            "numerical_execution": numerical_execution,
        }
        if committed["packet_committed"] is not True:
            return {
                **common,
                "status": "NO_COMPACT_ROUTE",
                "nonzero_merge_applied": False,
                "nonzero_merge_update_slot_count": 0,
                "target_rgb_provenance": {
                    "target_access_occurred": False,
                    "reason": "no-nonzero-compact-update-before-target-access",
                },
                "execution_boundary": {
                    "source_route_executed": True,
                    "packet_committed_before_target_access": False,
                    "target_mapping_opened": False,
                    "target_view_rendered": False,
                    "quality_metrics_computed": False,
                    "whole_pipeline_s2_s3_sparse_execution_verified": False,
                    "timing_claim": False,
                },
                "packet": committed["initial"],
            }

        # This is the first target-side operation after the source packet commits.
        target_mapping, target_indices, isolated_target_provenance = (
            _load_isolated_target_batch_after_packet_commit(
                loader,
                bundle,
                scene=input_identity["scene"],
                context_indices=list(input_identity["context_indices"]),
                input_identity=input_identity,
                native_preprocessing=loaded_calibration["native_preprocessing"],
                source_audit_root=source_audit_root,
                raw_root=DEFAULT_RAW_ROOT,
            )
        )
        target_cameras = _target_cameras(target_mapping, bundle.device)
        materialized_gaussians = committed["materialized"].as_single_batch(
            type(committed["execution"].dense_gaussians)
        )
        materialized_color = _render_target_view(
            bundle.decoder,
            materialized_gaussians,
            target=target_cameras,
            image_shape=(committed["height"], committed["width"]),
        )
        baseline_color = _render_target_view(
            bundle.decoder,
            committed["execution"].dense_gaussians,
            target=target_cameras,
            image_shape=(committed["height"], committed["width"]),
        )
        target_rgb = _take_target_rgb_for_metrics(target_mapping, bundle.device)

    if baseline_color.shape != materialized_color.shape or baseline_color.shape != target_rgb.shape:
        raise RuntimeError("DepthSplat soft-mixture render and target RGB shapes differ")
    baseline_views = _view_metrics(baseline_color[0], target_rgb[0])
    materialized_views = _view_metrics(materialized_color[0], target_rgb[0])
    baseline_quality = _mean_metrics(baseline_views)
    materialized_quality = _mean_metrics(materialized_views)
    verdict = _psnr_only_quality_verdict(baseline_quality, materialized_quality)
    update_count = committed["nonzero_merge_update_slot_count"]
    dense_gaussian_count = int(committed["execution"].dense_gaussians.means.shape[1])
    final_gaussian_count = committed["packet_summary"]["materialized_descriptor_count"]
    return {
        **common,
        "status": _quality_status(verdict=verdict, update_count=update_count),
        "nonzero_merge_applied": update_count > 0,
        "nonzero_merge_update_slot_count": update_count,
        "compression": {
            "dense_gaussian_count": dense_gaussian_count,
            "final_gaussian_count": final_gaussian_count,
            "gaussian_saving_rate": (
                (dense_gaussian_count - final_gaussian_count) / dense_gaussian_count
            ),
        },
        "packet": committed["packet_summary"],
        "mechanism_cycles": committed["mechanism_cycles"],
        "geometry_source": committed["geometry_source"],
        "target_rgb_provenance": {
            "target_access_occurred": True,
            "target_mapping_constructed_after_source_packet_commit": True,
            "target_camera_metadata_accessed_before_packet_commit": False,
            "target_index_accessed_before_packet_commit": False,
            "target_rgb_passed_to_encoder_or_route": False,
            "isolated_dl3dv_raw_reader": True,
            **isolated_target_provenance,
        },
        "execution_boundary": {
            "source_route_executed": True,
            "packet_committed_before_target_access": True,
            "target_mapping_opened": True,
            "native_packet_decoder_executed": True,
            "independent_dense_baseline_executed": True,
            "quality_metrics_computed": True,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
            "timing_claim": False,
        },
        "quality": {
            "baseline": baseline_quality,
            "materialized": materialized_quality,
            "verdict": verdict,
            "views": [
                {
                    "target_index": target_indices[index],
                    "baseline": baseline_views[index],
                    "materialized": materialized_views[index],
                }
                for index in range(len(baseline_views))
            ],
        },
    }


def _load_quality_gate_runtime(
    *, input_root: Path, device: torch.device, source_sample_index: int
) -> dict[str, Any]:
    """Load the target-free source runtime shared by a direct threshold sweep."""

    from data.context_only_audit_input import validate_context_only_audit_input

    runtime_started = time.perf_counter()
    input_identity = _require_quality_context_identity(
        validate_context_only_audit_input(input_root, model=MODEL),
        source_sample_index=source_sample_index,
    )
    backend_contract = resolve_depthsplat_backend_contract(ROOT)
    backend_identity = freeze_depthsplat_backend_identity(backend_contract)
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    selection = resolve_claim_selection(MODEL, DATASET, ROOT)
    _bind_formal_application(
        input_identity=input_identity,
        backend_contract=backend_contract,
        experiment=experiment,
        selection=selection,
    )
    checkpoint_sha256 = cached_sha256_file(experiment.checkpoint)
    loader = create_model_loader(MODEL)
    model_load_started = time.perf_counter()
    bundle = loader.load_model(
        str(experiment.checkpoint),
        device=device,
        experiment_name=experiment.experiment,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        hydra_overrides=experiment.hydra_overrides,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    if bundle.decoder is None:
        raise RuntimeError("DepthSplat soft-mixture runner requires the native decoder")
    context_data = load_context_only_audit_data(
        loader, bundle, input_root=Path(input_root), model_name=MODEL
    )
    context_cpu, loaded_calibration = _require_loaded_context_batch(
        context_data.batch, input_identity
    )
    context = {
        key: value.to(bundle.device) if torch.is_tensor(value) else value
        for key, value in context_cpu.items()
    }
    bundle.model.eval()
    return {
        "input_identity": input_identity,
        "backend_identity": backend_identity,
        "checkpoint_sha256": checkpoint_sha256,
        "loader": loader,
        "bundle": bundle,
        "context": context,
        "loaded_calibration": loaded_calibration,
        "timing": {
            "model_load_wall_seconds": model_load_seconds,
            "runtime_setup_wall_seconds": time.perf_counter() - runtime_started,
        },
    }


def collect_depthsplat_source_fsdr_development_trace(
    *,
    input_root: Path,
    device: torch.device,
    source_sample_index: int,
) -> dict[str, Any]:
    """Collect the source-only FSDR development inputs for one DL3DV sample."""

    runtime = _load_quality_gate_runtime(
        input_root=input_root,
        device=device,
        source_sample_index=source_sample_index,
    )
    bundle = runtime["bundle"]
    with strict_fp32_convolution_execution() as numerical_execution:
        encoder_started = time.perf_counter()
        execution, fsdr_execution_inputs = _capture_depthsplat_execution_with_fsdr_inputs(
            bundle.encoder, runtime["context"]
        )
        encoder_seconds = time.perf_counter() - encoder_started
        fsdr_started = time.perf_counter()
        fsdr_trace = _collect_source_fsdr_trace(
            execution, exact_inputs=fsdr_execution_inputs
        )
        fsdr_seconds = time.perf_counter() - fsdr_started
    identity = runtime["input_identity"]
    return {
        "schema_version": QUALITY_GATE_SCHEMA_VERSION,
        "kind": "depthsplat-fsdr-source-development-trace-v1",
        "paper_result_eligible": False,
        "model": MODEL,
        "dataset": DATASET,
        "source_sample_index": identity["source_sample_index"],
        "scene": identity["scene"],
        "context_indices": list(identity["context_indices"]),
        "source_phase": {"fsdr": fsdr_trace},
        "context_only_input": {
            "identity": identity,
            "loaded_native_preprocessing": runtime["loaded_calibration"][
                "native_preprocessing"
            ],
        },
        "backend_identity": runtime["backend_identity"],
        "checkpoint_sha256": runtime["checkpoint_sha256"],
        "numerical_execution": numerical_execution,
        "timing": {
            **runtime["timing"],
            "encoder_wall_seconds": encoder_seconds,
            "fsdr_source_trace_wall_seconds": fsdr_seconds,
        },
        "target_rgb_provenance": {
            "target_access_occurred": False,
            "target_rgb_passed_to_encoder_or_route": False,
        },
        "execution_boundary": {
            "source_route_executed": True,
            "target_mapping_opened": False,
            "target_view_rendered": False,
            "quality_metrics_computed": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
            "timing_claim": False,
        },
        "status": "PASS",
    }


def _direct_kernel_risk_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize a finite, non-duplicated direct DL3DV threshold set."""

    thresholds: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError("DepthSplat direct kernel-risk threshold is invalid")
        threshold = float(value)
        if threshold in thresholds:
            raise ValueError("DepthSplat direct kernel-risk thresholds must be unique")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("DepthSplat direct kernel-risk threshold set is empty")
    return tuple(thresholds)


def _route_isolation_modes(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ("l0_l1",)
    modes = tuple(values)
    if (
        not modes
        or any(
            mode not in {"l0_l1", "l0_only", "l1_only", "l0_to_l1"}
            for mode in modes
        )
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("DepthSplat route-isolation mode is invalid")
    return modes


def _attribute_attribution_names(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    names = tuple(values)
    if (
        not names
        or any(name not in {"mean", "covariance", "opacity", "sh"} for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("DepthSplat attribute attribution is invalid")
    return names


def _threshold_route_candidates(
    thresholds: Sequence[float],
    route_isolation_modes: Sequence[str] | None,
) -> tuple[tuple[float, str], ...]:
    """Expand only the unambiguous threshold/route sweep shapes."""

    modes = _route_isolation_modes(route_isolation_modes)
    if route_isolation_modes is None:
        return tuple((threshold, "l0_l1") for threshold in thresholds)
    if len(thresholds) > 1 and len(modes) > 1:
        raise ValueError(
            "DepthSplat multi-threshold route isolation requires one route mode"
        )
    if len(modes) == 1:
        return tuple((threshold, modes[0]) for threshold in thresholds)
    return tuple((thresholds[0], mode) for mode in modes)


def _diagnostic_label(
    *,
    threshold: float,
    route_isolation: str,
    candidate_count: int,
    attribute: str | None,
) -> str:
    """Keep multi-threshold diagnostic images from overwriting one another."""

    label = route_isolation
    if candidate_count > 1:
        label = f"{label}_threshold_{threshold:g}".replace(".", "_")
    return label if attribute is None else f"{label}_restore_{attribute}"


def _restore_dense_attribute_for_diagnosis(
    materialized: Any, dense_gaussians: Any, *, attribute: str
) -> Any:
    """Restore one dense attribute for target-side diagnosis only."""

    field = {
        "mean": "means",
        "covariance": "covariances",
        "opacity": "opacities",
        "sh": "harmonics",
    }.get(attribute)
    if field is None:
        raise ValueError("DepthSplat diagnostic attribute is invalid")
    values = getattr(dense_gaussians, field)[0].index_select(
        0, materialized.dense_slots
    )
    return replace(materialized, **{field: values})


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    value = image.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(value * 255.0).astype(np.uint8), mode="RGB").save(path)


def _write_route_isolation_visuals(
    *,
    output_dir: Path,
    label: str,
    baseline_color: torch.Tensor,
    materialized_color: torch.Tensor,
    context: Mapping[str, Any],
    final_route: Any,
) -> dict[str, Any]:
    """Write target error and source route overlays after target-side metrics."""

    root = output_dir / label
    root.mkdir(parents=True, exist_ok=True)
    baseline_paths: list[str] = []
    materialized_paths: list[str] = []
    heatmap_paths: list[str] = []
    heatmap_vmax: list[float] = []
    for index in range(baseline_color.shape[0]):
        baseline_path = root / f"target_{index:02d}_baseline.png"
        materialized_path = root / f"target_{index:02d}_materialized.png"
        heatmap_path = root / f"target_{index:02d}_rgb_error_heatmap.png"
        _save_rgb(baseline_path, baseline_color[index])
        _save_rgb(materialized_path, materialized_color[index])
        error = (baseline_color[index] - materialized_color[index]).abs().mean(dim=0)
        vmax = max(float(torch.quantile(error.flatten(), 0.99).item()), 1.0e-6)
        normalized = (error / vmax).clamp(0.0, 1.0).detach().cpu().numpy()
        heatmap = np.stack(
            (normalized, np.sqrt(normalized), 0.15 * (1.0 - normalized)), axis=-1
        )
        Image.fromarray(np.rint(heatmap * 255.0).astype(np.uint8), mode="RGB").save(
            heatmap_path
        )
        baseline_paths.append(baseline_path.relative_to(output_dir.parent).as_posix())
        materialized_paths.append(
            materialized_path.relative_to(output_dir.parent).as_posix()
        )
        heatmap_paths.append(heatmap_path.relative_to(output_dir.parent).as_posix())
        heatmap_vmax.append(vmax)

    route_masks = final_route.selected_output_mask
    source_images = context["image"][0]
    overlay_paths: list[str] = []
    colors = {
        "L0": torch.tensor((0.10, 0.75, 0.25), device=source_images.device),
        "L1": torch.tensor((0.15, 0.45, 0.95), device=source_images.device),
        "Full": torch.tensor((0.90, 0.20, 0.16), device=source_images.device),
    }
    for view in range(route_masks.shape[0]):
        height, width = route_masks.shape[-2:]
        source = torch.nn.functional.interpolate(
            source_images[view].unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
        overlay = source * 0.55
        for record in final_route.tile_trace:
            if int(record["view"]) != view:
                continue
            tile_y, tile_x = int(record["tile_y"]), int(record["tile_x"])
            overlay[:, tile_y * 4 : (tile_y + 1) * 4, tile_x * 4 : (tile_x + 1) * 4] += (
                colors[record["final_route"]].reshape(3, 1, 1) * 0.45
            )
        overlay_path = root / f"source_{view:02d}_route_mask_overlay.png"
        _save_rgb(overlay_path, overlay)
        overlay_paths.append(overlay_path.relative_to(output_dir.parent).as_posix())
    return {
        "baseline": baseline_paths,
        "materialized": materialized_paths,
        "rgb_error_heatmap": heatmap_paths,
        "rgb_error_heatmap_vmax": heatmap_vmax,
        "route_mask_overlay": overlay_paths,
    }


def collect_depthsplat_soft_mixture_sample0_direct_threshold_sweep(
    *,
    input_root: Path,
    source_audit_root: Path,
    direct_kernel_risk_thresholds: Sequence[float],
    device: torch.device,
    source_sample_index: int = SOURCE_SAMPLE_INDEX,
    route_isolation_modes: Sequence[str] | None = None,
    attribute_attributions: Sequence[str] | None = None,
    diagnostic_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate direct DL3DV thresholds with one source and target runtime."""

    thresholds = _direct_kernel_risk_thresholds(direct_kernel_risk_thresholds)
    attributions = _attribute_attribution_names(attribute_attributions)
    candidates = _threshold_route_candidates(thresholds, route_isolation_modes)
    if attributions and (
        len(candidates) != 1 or candidates[0][1] not in {"l0_only", "l1_only"}
    ):
        raise ValueError(
            "DepthSplat attribute attribution requires one isolated compact route"
        )
    runtime = _load_quality_gate_runtime(
        input_root=input_root,
        device=device,
        source_sample_index=source_sample_index,
    )
    input_identity = runtime["input_identity"]
    bundle = runtime["bundle"]
    measurement_cache = DepthSplatTileKernelClosureMeasurementCache()
    threshold_runs: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []

    with strict_fp32_convolution_execution() as numerical_execution:
        encoder_started = time.perf_counter()
        execution, fsdr_execution_inputs = _capture_depthsplat_execution_with_fsdr_inputs(
            bundle.encoder, runtime["context"]
        )
        encoder_seconds = time.perf_counter() - encoder_started
        fsdr_source_trace = _collect_source_fsdr_trace(
            execution, exact_inputs=fsdr_execution_inputs
        )
        for threshold, route_isolation in candidates:
            profile = soft_mixture_kernel_closure_t4_profile(
                direct_kernel_risk_threshold=threshold,
            )
            committed = _build_committed_packet(
                encoder=bundle.encoder,
                context=runtime["context"],
                profile=profile,
                direct_kernel_risk_threshold=threshold,
                execution=execution,
                fsdr_execution_inputs=fsdr_execution_inputs,
                mixture_kernel_closure_measurement_cache=measurement_cache,
                fsdr_source_trace=fsdr_source_trace,
                route_isolation=route_isolation,
            )
            if committed["packet_committed"] is not True:
                raise RuntimeError(
                    "DepthSplat threshold sweep cannot open target before every packet commits"
                )
            threshold_runs.append((threshold, route_isolation, profile, committed))

        # All source-only routes are committed before the first target-side read.
        target_mapping, target_indices, isolated_target_provenance = (
            _load_isolated_target_batch_after_packet_commit(
                runtime["loader"],
                bundle,
                scene=input_identity["scene"],
                context_indices=list(input_identity["context_indices"]),
                input_identity=input_identity,
                native_preprocessing=runtime["loaded_calibration"]["native_preprocessing"],
                source_audit_root=source_audit_root,
                raw_root=DEFAULT_RAW_ROOT,
            )
        )
        target_cameras = _target_cameras(target_mapping, bundle.device)
        baseline_render_started = time.perf_counter()
        baseline_color = _render_target_view(
            bundle.decoder,
            execution.dense_gaussians,
            target=target_cameras,
            image_shape=(threshold_runs[0][3]["height"], threshold_runs[0][3]["width"]),
        )
        baseline_render_seconds = time.perf_counter() - baseline_render_started
        target_rgb = _take_target_rgb_for_metrics(target_mapping, bundle.device)
        render_runs: list[tuple[float, str, dict[str, Any], dict[str, Any], str | None, Any]] = []
        for threshold, route_isolation, profile, committed in threshold_runs:
            render_runs.append(
                (threshold, route_isolation, profile, committed, None, committed["materialized"])
            )
            for attribute in attributions:
                render_runs.append(
                    (
                        threshold,
                        route_isolation,
                        profile,
                        committed,
                        attribute,
                        _restore_dense_attribute_for_diagnosis(
                            committed["materialized"],
                            execution.dense_gaussians,
                            attribute=attribute,
                        ),
                    )
                )
        materialized_colors: list[torch.Tensor] = []
        materialized_render_seconds: list[float] = []
        for _threshold, _route_isolation, _profile, committed, _attribute, materialized in render_runs:
            materialized_gaussians = materialized.as_single_batch(
                type(execution.dense_gaussians)
            )
            materialized_render_started = time.perf_counter()
            materialized_color = _render_target_view(
                bundle.decoder,
                materialized_gaussians,
                target=target_cameras,
                image_shape=(committed["height"], committed["width"]),
            )
            materialized_render_seconds.append(
                time.perf_counter() - materialized_render_started
            )
            if (
                baseline_color.shape != materialized_color.shape
                or baseline_color.shape != target_rgb.shape
            ):
                raise RuntimeError(
                    "DepthSplat soft-mixture render and target RGB shapes differ"
                )
            materialized_colors.append(materialized_color)

    baseline_metric_started = time.perf_counter()
    baseline_views = _view_metrics(baseline_color[0], target_rgb[0])
    baseline_quality = _mean_metrics(baseline_views)
    baseline_metric_seconds = time.perf_counter() - baseline_metric_started
    target_provenance = {
        "target_access_occurred": True,
        "target_mapping_constructed_after_source_packet_commit": True,
        "target_camera_metadata_accessed_before_packet_commit": False,
        "target_index_accessed_before_packet_commit": False,
        "target_rgb_passed_to_encoder_or_route": False,
        "isolated_dl3dv_raw_reader": True,
        **isolated_target_provenance,
    }
    results: list[dict[str, Any]] = []
    for ordinal, ((threshold, route_isolation, profile, committed, attribute, _materialized), materialized_color) in enumerate(
        zip(render_runs, materialized_colors, strict=True)
    ):
        metric_started = time.perf_counter()
        materialized_views = _view_metrics(materialized_color[0], target_rgb[0])
        materialized_quality = _mean_metrics(materialized_views)
        verdict = _psnr_only_quality_verdict(baseline_quality, materialized_quality)
        metric_seconds = time.perf_counter() - metric_started
        update_count = committed["nonzero_merge_update_slot_count"]
        dense_gaussian_count = int(execution.dense_gaussians.means.shape[1])
        final_gaussian_count = committed["packet_summary"][
            "materialized_descriptor_count"
        ]
        visuals = (
            _write_route_isolation_visuals(
                output_dir=diagnostic_dir,
                label=_diagnostic_label(
                    threshold=threshold,
                    route_isolation=route_isolation,
                    candidate_count=len(candidates),
                    attribute=attribute,
                ),
                baseline_color=baseline_color[0],
                materialized_color=materialized_color[0],
                context=runtime["context"],
                final_route=committed["final_route"],
            )
            if diagnostic_dir is not None
            else None
        )
        results.append(
            {
                "schema_version": QUALITY_GATE_SCHEMA_VERSION,
                "kind": KERNEL_CLOSURE_QUALITY_GATE_KIND,
                "paper_result_eligible": False,
                "model": MODEL,
                "dataset": DATASET,
                "source_sample_index": input_identity["source_sample_index"],
                "scene": input_identity["scene"],
                "context_indices": list(input_identity["context_indices"]),
                "mechanism": KERNEL_CLOSURE_MECHANISM,
                "route_isolation": route_isolation,
                "attribute_attribution": attribute,
                "offline_diagnostic_only": attribute is not None,
                "kernel_risk_threshold_source": {
                    "mode": "direct_dl3dv_development",
                    "record_path": None,
                    "threshold_value": threshold,
                    "frozen_guard": None,
                },
                "source_phase": committed["source_summary"],
                "context_only_input": {
                    "identity": input_identity,
                    "loaded_native_preprocessing": runtime["loaded_calibration"][
                        "native_preprocessing"
                    ],
                },
                "backend_identity": runtime["backend_identity"],
                "checkpoint_sha256": runtime["checkpoint_sha256"],
                "numerical_execution": numerical_execution,
                "threshold_sweep": {
                    "ordinal": ordinal,
                    "threshold_count": len(thresholds),
                    "candidate_count": len(render_runs),
                    "shared_execution": {
                        "model_loaded_once": True,
                        "context_loaded_once": True,
                        "native_encoder_capture_once": True,
                        "target_mapping_loaded_once": True,
                        "dense_baseline_render_count": 1,
                        "all_source_packets_committed_before_target_access": True,
                    },
                },
                "status": _quality_status(verdict=verdict, update_count=update_count),
                "nonzero_merge_applied": update_count > 0,
                "nonzero_merge_update_slot_count": update_count,
                "compression": {
                    "dense_gaussian_count": dense_gaussian_count,
                    "final_gaussian_count": final_gaussian_count,
                    "gaussian_saving_rate": (
                        (dense_gaussian_count - final_gaussian_count)
                        / dense_gaussian_count
                    ),
                },
                "packet": committed["packet_summary"],
                "mechanism_cycles": committed["mechanism_cycles"],
                "geometry_source": committed["geometry_source"],
                "timing": {
                    **runtime["timing"],
                    "encoder_wall_seconds": encoder_seconds,
                    **committed["timing"],
                    "baseline_render_wall_seconds": baseline_render_seconds,
                    "baseline_metric_wall_seconds": baseline_metric_seconds,
                    "materialized_render_wall_seconds": materialized_render_seconds[
                        ordinal
                    ],
                    "metric_wall_seconds": metric_seconds,
                },
                "diagnostics": visuals,
                "target_rgb_provenance": target_provenance,
                "execution_boundary": {
                    "source_route_and_certificate_executed": True,
                    "packet_committed_before_target_access": True,
                    "all_threshold_source_packets_committed_before_target_access": True,
                    "target_mapping_opened": True,
                    "native_packet_decoder_executed": True,
                    "independent_dense_baseline_executed": True,
                    "quality_metrics_computed": True,
                    "whole_pipeline_s2_s3_sparse_execution_verified": False,
                    "global_s2_s3_savings_claimed": False,
                    "timing_claim": False,
                },
                "quality": {
                    "baseline": baseline_quality,
                    "materialized": materialized_quality,
                    "verdict": verdict,
                    "views": [
                        {
                            "target_index": target_indices[index],
                            "baseline": baseline_views[index],
                            "materialized": materialized_views[index],
                        }
                        for index in range(len(baseline_views))
                    ],
                },
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--source-audit-root", type=Path, default=DEFAULT_SOURCE_AUDIT_ROOT)
    parser.add_argument("--source-sample-index", type=int, default=SOURCE_SAMPLE_INDEX)
    parser.add_argument(
        "--source-fsdr-only",
        action="store_true",
        help="collect target-free FSDR development inputs without SAES routing",
    )
    parser.add_argument(
        "--kernel-risk-threshold",
        type=float,
        action="append",
        dest="kernel_risk_thresholds",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--route-isolation",
        choices=("l0_l1", "l0_only", "l1_only", "l0_to_l1"),
        action="append",
        dest="route_isolation_modes",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--attribute-attribution",
        choices=("mean", "covariance", "opacity", "sh"),
        action="append",
        dest="attribute_attributions",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("DepthSplat soft-mixture quality gate requires CUDA")
    random.seed(QUALITY_GATE_SEED)
    np.random.seed(QUALITY_GATE_SEED)
    torch.manual_seed(QUALITY_GATE_SEED)
    torch.cuda.manual_seed_all(QUALITY_GATE_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    route_isolation_modes = args.route_isolation_modes
    attribute_attributions = args.attribute_attributions
    try:
        if args.source_sample_index < 0:
            raise ValueError("DepthSplat source sample index is invalid")
        if args.source_fsdr_only:
            if args.kernel_risk_thresholds is not None:
                raise ValueError("FSDR source trace does not accept a SAES threshold")
            if route_isolation_modes is not None or attribute_attributions is not None:
                raise ValueError("FSDR source trace does not accept SAES diagnostics")
            record = collect_depthsplat_source_fsdr_development_trace(
                input_root=args.input_root,
                device=device,
                source_sample_index=args.source_sample_index,
            )
            exit_code = 0
        elif (
            args.kernel_risk_thresholds is None
            and attribute_attributions is None
            and (
                route_isolation_modes is None
                or route_isolation_modes in (["l0_l1"], ["l0_only"])
            )
        ):
            route_isolation = (
                route_isolation_modes[0]
                if route_isolation_modes is not None
                else "l0_l1"
            )
            record = collect_depthsplat_soft_mixture_sample0_quality_gate(
                input_root=args.input_root,
                source_audit_root=args.source_audit_root,
                kernel_risk_record=None,
                direct_kernel_risk_threshold=None,
                device=device,
                source_sample_index=args.source_sample_index,
                route_isolation=route_isolation,
            )
            exit_code = (
                0
                if record["status"] in {"PASS", "PASS_IDENTITY_FALLBACK"}
                else 1
            )
        elif (
            args.kernel_risk_thresholds == [2.0]
            and route_isolation_modes == ["l0_to_l1"]
            and attribute_attributions is None
        ):
            records = collect_depthsplat_soft_mixture_sample0_direct_threshold_sweep(
                input_root=args.input_root,
                source_audit_root=args.source_audit_root,
                direct_kernel_risk_thresholds=(2.0,),
                device=device,
                source_sample_index=args.source_sample_index,
                route_isolation_modes=("l0_to_l1",),
            )
            if len(records) != 1:
                raise RuntimeError("DepthSplat registered kernel route returned multiple records")
            record = records[0]
            exit_code = (
                0
                if record["status"] in {"PASS", "PASS_IDENTITY_FALLBACK"}
                else 1
            )
        else:
            raise ValueError(
                "DepthSplat quality gate supports one direct route or the registered "
                "kernel-closure threshold=2.0 l0_to_l1 route"
            )
    except Exception as error:
        record = {
            "schema_version": QUALITY_GATE_SCHEMA_VERSION,
            "kind": QUALITY_GATE_KIND,
            "status": "FAILED",
            "paper_result_eligible": False,
            "model": MODEL,
            "dataset": DATASET,
            "source_sample_index": args.source_sample_index,
            "mechanism": SOFT_MIXTURE_MECHANISM,
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        exit_code = 1
    record["runner"] = {
        "path": Path(__file__).relative_to(ROOT).as_posix(),
        "sha256": cached_sha256_file(Path(__file__)),
    }
    record["source"] = source_identity()
    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
