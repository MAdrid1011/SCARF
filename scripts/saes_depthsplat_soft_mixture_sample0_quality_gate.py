#!/usr/bin/env python3
"""Run one fused DL3DV sample-0 quality gate for kernel-closed soft mixture.

This runner deliberately has no standalone-audit input.  It first constructs
and seals the source-only S/R route, certificate, and materialized packet;
only a committed packet may open the isolated target frames.
The resulting record contains compact source accounting together with the
target-view quality metrics from the same process.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration import create_model_loader, load_context_only_audit_data
from saes.depthsplat_backend import (
    canonical_json_sha256,
    freeze_depthsplat_backend_identity,
    resolve_depthsplat_backend_contract,
)
from saes.depthsplat_l0_l1_materializer import (
    DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
    DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
    apply_depthsplat_compact_l0_l1_materialization,
    preflight_depthsplat_l0_l1_materialization,
    resolve_depthsplat_compact_final_route,
)
from saes.depthsplat_mixture_kernel_guard import (
    KIND as DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND,
    POLICY as DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY,
    SCHEMA_VERSION as DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_SCHEMA_VERSION,
)
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
    replay_depthsplat_selected_head,
    subset_depthsplat_sparse_raw_packet,
)
from saes.probe_first_schedule import (
    BALANCED_L1_ANCHOR_SEMANTICS,
    DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
    SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY,
    SOFT_MIXTURE_KERNEL_CLOSURE_T4_L0_SECONDARY_PREFETCH_POLICY,
    build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan,
    soft_mixture_kernel_closure_t4_route_config_sha256,
)
from saes.progressive_saes import PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS
from scripts.ae_config import resolve_claim_selection, resolve_experiment
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
    _require_formal_context_identity,
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


QUALITY_GATE_KIND = "depthsplat-soft-mixture-kernel-closure-v3-fused-sample0-quality-gate"
QUALITY_GATE_SCHEMA_VERSION = "3.0"
QUALITY_GATE_SEED = 0
SOFT_MIXTURE_MECHANISM = "soft-mixture-kernel-closure-v3"
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

    profile: dict[str, Any] = {
        "materialization_profile": (
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
        ),
        "route_plan_contract": DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
        "contract_version": DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
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


def _build_committed_packet(
    *,
    encoder: Any,
    context: Mapping[str, Any],
    profile: Mapping[str, Any],
    kernel_risk_guard: Any | None = None,
    direct_kernel_risk_threshold: float | None = None,
) -> dict[str, Any]:
    """Run the complete source phase and return only a sealed packet state."""

    execution = capture_depthsplat_native_execution(
        encoder, context, source_root=ROOT / "depthsplat"
    )
    if execution.routing_features is None or execution.routing_z_depths is None:
        raise RuntimeError("DepthSplat soft-mixture has no source routing tensors")
    _views, _channels, height, width = execution.dense_raw_head.shape
    if height % TILE_SIZE or width % TILE_SIZE:
        raise RuntimeError("DepthSplat soft-mixture image is not tiled by four")
    plan = build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan(
        execution.routing_features,
        execution.routing_z_depths,
        height=height,
        width=width,
        feature_threshold=FEATURE_THRESHOLD,
        depth_threshold=DEPTH_THRESHOLD,
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
    preflight = preflight_depthsplat_l0_l1_materialization(
        initial_packet,
        initial_packed,
        plan,
        execution.routing_features,
        execution.routing_z_depths,
        source_sample_image_grid=sample_image_grid,
        source_get_world_rays=get_world_rays,
        maximum_coverage_covariance_scale=1.0,
        execution_profile=profile["materialization_profile"],
        mixture_kernel_closure_frozen_guard=kernel_risk_guard,
        mixture_kernel_closure_maximum_relative_risk=direct_kernel_risk_threshold,
    )
    final_route = resolve_depthsplat_compact_final_route(plan, preflight)
    source_summary = _source_summary(
        plan=plan, preflight=preflight, final_route=final_route, profile=profile
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
    materialized = apply_depthsplat_compact_l0_l1_materialization(
        final_packed, preflight, final_route
    )
    materialized_full_passthrough = _require_bitwise_equivalent(
        compare_depthsplat_full_passthrough_to_dense_bitwise(
            materialized, execution.dense_gaussians, final_route.full_passthrough_mask
        ),
        "DepthSplat soft-mixture materialized Full packet",
    )
    if int(materialized.dense_slots.numel()) != int(final_route.selected_output_mask.sum()):
        raise RuntimeError("DepthSplat soft-mixture materialized packet route drifted")
    return {
        "execution": execution,
        "height": height,
        "width": width,
        "source_summary": source_summary,
        "packet_committed": True,
        "nonzero_merge_update_slot_count": update_count,
        "materialized": materialized,
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
    }


def collect_depthsplat_soft_mixture_sample0_quality_gate(
    *,
    input_root: Path,
    source_audit_root: Path,
    kernel_risk_record: Path | None,
    direct_kernel_risk_threshold: float | None,
    device: torch.device,
) -> dict[str, Any]:
    """Execute source telemetry and real target quality in a single process."""

    from data.context_only_audit_input import validate_context_only_audit_input

    if (kernel_risk_record is None) == (direct_kernel_risk_threshold is None):
        raise ValueError("DepthSplat quality gate requires exactly one kernel-risk source")
    kernel_risk_guard = (
        load_mixture_kernel_risk_guard(Path(kernel_risk_record), root=ROOT)
        if kernel_risk_record is not None
        else None
    )
    profile = soft_mixture_kernel_closure_t4_profile(
        kernel_risk_guard=kernel_risk_guard,
        direct_kernel_risk_threshold=direct_kernel_risk_threshold,
    )
    input_identity = _require_formal_context_identity(
        validate_context_only_audit_input(input_root, model=MODEL)
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
        committed = _build_committed_packet(
            encoder=bundle.encoder,
            context=context,
            profile=profile,
            kernel_risk_guard=kernel_risk_guard,
            direct_kernel_risk_threshold=direct_kernel_risk_threshold,
        )
        source_summary = committed["source_summary"]
        common = {
            "schema_version": QUALITY_GATE_SCHEMA_VERSION,
            "kind": QUALITY_GATE_KIND,
            "paper_result_eligible": False,
            "model": MODEL,
            "dataset": DATASET,
            "source_sample_index": SOURCE_SAMPLE_INDEX,
            "scene": input_identity["scene"],
            "context_indices": list(input_identity["context_indices"]),
            "mechanism": SOFT_MIXTURE_MECHANISM,
            "kernel_risk_threshold_source": {
                "mode": (
                    "frozen_acid_record"
                    if kernel_risk_guard is not None
                    else "direct_dl3dv_development"
                ),
                "record_path": (
                    str(Path(kernel_risk_record))
                    if kernel_risk_record is not None
                    else None
                ),
                "threshold_value": (
                    dict(kernel_risk_guard)["threshold_value"]
                    if kernel_risk_guard is not None
                    else float(direct_kernel_risk_threshold)
                ),
                "frozen_guard": (
                    dict(kernel_risk_guard) if kernel_risk_guard is not None else None
                ),
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
                    "source_route_and_certificate_executed": True,
                    "packet_committed_before_target_access": False,
                    "target_mapping_opened": False,
                    "target_view_rendered": False,
                    "quality_metrics_computed": False,
                    "whole_pipeline_s2_s3_sparse_execution_verified": False,
                    "timing_claim": False,
                },
                "packet": committed["initial"],
            }

        # This is the first target-side operation. All source route, packet,
        # and certificate bindings above have already been committed.
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
    verdict = _quality_verdict(baseline_quality, materialized_quality)
    update_count = committed["nonzero_merge_update_slot_count"]
    return {
        **common,
        "status": _quality_status(verdict=verdict, update_count=update_count),
        "nonzero_merge_applied": update_count > 0,
        "nonzero_merge_update_slot_count": update_count,
        "packet": committed["packet_summary"],
        "geometry_source": committed["geometry_source"],
        "target_rgb_provenance": {
            "target_access_occurred": True,
            "target_mapping_constructed_after_source_packet_commit": True,
            "target_camera_metadata_accessed_before_packet_commit": False,
            "target_index_accessed_before_packet_commit": False,
            "target_rgb_passed_to_encoder_or_route": False,
            "isolated_sample0_raw_reader": True,
            **isolated_target_provenance,
        },
        "execution_boundary": {
            "source_route_and_certificate_executed": True,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--source-audit-root", type=Path, default=DEFAULT_SOURCE_AUDIT_ROOT)
    parser.add_argument(
        "--kernel-risk-threshold",
        type=float,
        default=1.0,
        help="direct DL3DV development threshold for maximum kernel risk",
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
    try:
        record = collect_depthsplat_soft_mixture_sample0_quality_gate(
            input_root=args.input_root,
            source_audit_root=args.source_audit_root,
            kernel_risk_record=None,
            direct_kernel_risk_threshold=args.kernel_risk_threshold,
            device=device,
        )
        exit_code = (
            0
            if record["status"] in {"PASS", "PASS_IDENTITY_FALLBACK"}
            else 1
        )
    except Exception as error:
        record = {
            "schema_version": QUALITY_GATE_SCHEMA_VERSION,
            "kind": QUALITY_GATE_KIND,
            "status": "FAILED",
            "paper_result_eligible": False,
            "model": MODEL,
            "dataset": DATASET,
            "source_sample_index": SOURCE_SAMPLE_INDEX,
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
