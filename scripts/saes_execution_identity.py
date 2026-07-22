"""Build the immutable SAES route identity shared by config producers and readers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from saes.guard_policy import (
    CONTEXT_GUARD_MAX_CENTER_MAHALANOBIS,
    CONTEXT_GUARD_MAX_FOOTPRINT_RATIO,
    CONTEXT_GUARD_MAX_RELATIVE_DEPTH_SPAN,
    CONTEXT_GUARD_POLICY,
    MATERIALIZATION_GUARD_MAX_OPACITY_DISTANCE,
    MATERIALIZATION_GUARD_MIN_COVARIANCE_COSINE,
    MATERIALIZATION_GUARD_MIN_HARMONIC_COSINE,
    MATERIALIZATION_GUARD_POLICY,
)
from saes.probe_layout import compute_lightweight_positions, compute_probe_positions


SCHEMA_VERSION = "saes-execution-identity-v1"
TILE_SIZE = 4
FEATURE_THRESHOLD = 0.2
DEPTH_THRESHOLD = 0.1
CROSS_CHECK_THRESHOLD = 0.015
DECISION_SEMANTICS = "probe-normalized-std-first-hit"
DEPTH_ROUTING_SEMANTICS = "metric-depth-standard-deviation"
L1_DEPTH_REFERENCE = "primary-routing-probes-v1"
MOMENT_GEOMETRY = "c2w-probe-depth-ray-v1"
MATERIALIZATION = "representative"
MATERIALIZATION_GUARD = True
CONTEXT_SAFETY_GUARD = True


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _positions(values: list[tuple[int, int]]) -> list[list[int]]:
    return [[row, column] for row, column in values]


def build_saes_execution_identity() -> dict[str, Any]:
    """Return a fresh, hash-bound identity for the fixed SAES route.

    The returned mapping is recreated on every call so callers cannot mutate a
    shared global identity. The layout comes from the pure probe-layout source
    of truth, which keeps config validation coupled to the executable route.
    """

    l0_positions = _positions(compute_probe_positions(TILE_SIZE))
    l1_positions = _positions(compute_lightweight_positions(TILE_SIZE))
    if len(l0_positions) != 4 or len(l1_positions) != 12:
        raise RuntimeError(
            "registered T=4 SAES layout must contain four L0 and twelve L1 anchors"
        )
    if l1_positions[: len(l0_positions)] != l0_positions:
        raise RuntimeError(
            "registered T=4 L1 layout must preserve the L0 anchor prefix"
        )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": DEPTH_THRESHOLD,
        "cross_check_threshold": CROSS_CHECK_THRESHOLD,
        "decision_semantics": DECISION_SEMANTICS,
        "depth_routing_semantics": DEPTH_ROUTING_SEMANTICS,
        "route_order": "feature_L0_then_primary_probe_depth_L1_then_Full",
        "l0_anchor_layout": "t4-corners-v1",
        "l0_anchor_positions": l0_positions,
        "l0_anchor_count": len(l0_positions),
        "l1_anchor_layout": "t4-corners-plus-boundary-v1",
        "l1_anchor_positions": l1_positions,
        "l1_anchor_count": len(l1_positions),
        "l1_depth_reference": L1_DEPTH_REFERENCE,
        "materialization": MATERIALIZATION,
        "materialization_guard": MATERIALIZATION_GUARD,
        "materialization_guard_policy": MATERIALIZATION_GUARD_POLICY,
        "materialization_guard_min_covariance_cosine": (
            MATERIALIZATION_GUARD_MIN_COVARIANCE_COSINE
        ),
        "materialization_guard_min_harmonic_cosine": (
            MATERIALIZATION_GUARD_MIN_HARMONIC_COSINE
        ),
        "materialization_guard_max_opacity_distance": (
            MATERIALIZATION_GUARD_MAX_OPACITY_DISTANCE
        ),
        "context_safety_guard": CONTEXT_SAFETY_GUARD,
        "context_guard_policy": CONTEXT_GUARD_POLICY,
        "context_guard_max_footprint_ratio": CONTEXT_GUARD_MAX_FOOTPRINT_RATIO,
        "context_guard_max_relative_depth_span": (
            CONTEXT_GUARD_MAX_RELATIVE_DEPTH_SPAN
        ),
        "context_guard_max_center_mahalanobis": (
            CONTEXT_GUARD_MAX_CENTER_MAHALANOBIS
        ),
        "moment_geometry": MOMENT_GEOMETRY,
        "full_tile_mode": "dense_passthrough",
    }
    return {**identity, "route_sha256": _canonical_sha256(identity)}


def validate_saes_execution_identity(value: Any) -> dict[str, Any]:
    """Fail closed unless a config binds the exact registered SAES route."""

    if not isinstance(value, Mapping):
        raise ValueError("mechanism config SAES execution identity is missing")
    expected = build_saes_execution_identity()
    if dict(value) != expected:
        raise ValueError(
            "mechanism config SAES execution identity does not match the "
            "registered 12-anchor guard-on representative route"
        )
    return expected
