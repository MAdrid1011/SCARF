"""DepthSplat-native nonzero L0/L1 Gaussian materialization.

DepthSplat has a different producer and geometry contract from TranSplat and
MVSplat.  Its raw descriptor starts with an opacity logit, its Adapter derives
the SH DC term from selected context RGB, and its depth is camera *z-depth*.
This module therefore intentionally does not consume the classic packet or
coordinate helpers.  It uses a target-free probe plan only to choose anchors,
constructs skipped-domain virtual Gaussians from selected native attributes
and the already source-bound S2 z-depth map, then absorbs them into retained
anchors by first/second moment matching.

The materializer is fail-closed.  Any provenance, RGB/SH, z-depth, PSD,
continuity, or opacity inconsistency promotes the whole tile to its
source-native Full path before the final packet is committed.  The development
profile additionally closes virtual 2-sigma support by covariance expansion;
the literal paper profile deliberately does not add that non-paper route guard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from saes.depthsplat_backend import canonical_json_sha256
from saes.depthsplat_selected_output import (
    DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
    DepthSplatPackedGaussianAttributes,
    DepthSplatSparseRawPacket,
    depthsplat_attribute_binding_sha256,
)
from saes.depthsplat_owner_coverage import (
    AUDIT_KIND as DEPTHSPLAT_OWNER_COVERAGE_AUDIT_KIND,
    AUDIT_SCHEMA_VERSION as DEPTHSPLAT_OWNER_COVERAGE_AUDIT_SCHEMA_VERSION,
    OWNER_ASSIGNMENT_POLICY as DEPTHSPLAT_OWNER_ASSIGNMENT_POLICY,
    audit_depthsplat_owner_coverage,
)
from saes.depthsplat_support_basis_coverage import (
    AUDIT_KIND as DEPTHSPLAT_SUPPORT_BASIS_AUDIT_KIND,
    AUDIT_SCHEMA_VERSION as DEPTHSPLAT_SUPPORT_BASIS_AUDIT_SCHEMA_VERSION,
    SUPPORT_BASIS_POLICY as DEPTHSPLAT_SUPPORT_BASIS_POLICY,
    audit_depthsplat_tile_support_basis,
)
from saes.depthsplat_soft_mixture_certificate import (
    KIND as DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_KIND,
    POLICY as DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_POLICY,
    SCHEMA_VERSION as DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION,
    certify_depthsplat_tile_soft_mixture,
)
from saes.depthsplat_mixture_kernel_guard import (
    KIND as DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND,
    POLICY as DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY,
    SCHEMA_VERSION as DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_SCHEMA_VERSION,
    assess_depthsplat_tile_kernel_closure,
)
from saes.probe_first_schedule import (
    ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    BALANCED_L1_ANCHOR_SEMANTICS,
    COVERAGE_ENRICHED_T4_L0_SECONDARY_PREFETCH_POLICY,
    DEPTHSPLAT_COVERAGE_ENRICHED_T4_PLAN_CONTRACT,
    DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
    DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_PLAN_CONTRACT,
    DEPTHSPLAT_SOFT_MIXTURE_T4_PLAN_CONTRACT,
    DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT,
    LEGACY_L1_ANCHOR_SEMANTICS,
    LITERAL_PAPER_T4_PLAN_CONTRACT,
    PAPER_KP_ANCHOR_SEMANTICS,
    IncrementalProbeFirstPlan,
    build_depthsplat_coverage_enriched_t4_probe_first_plan,
    build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan,
    build_depthsplat_soft_mixture_normalized_t4_probe_first_plan,
    build_depthsplat_soft_mixture_t4_probe_first_plan,
    build_literal_paper_t4_probe_first_plan,
    build_incremental_probe_first_plan,
    coverage_enriched_t4_route_config_sha256,
    literal_paper_t4_route_config_sha256,
    l1_local_positions_for_tile,
    build_depthsplat_support_basis_t4_probe_first_plan,
    support_basis_t4_route_config_sha256,
    soft_mixture_t4_route_config_sha256,
    soft_mixture_normalized_t4_route_config_sha256,
    soft_mixture_kernel_closure_t4_route_config_sha256,
    SUPPORT_BASIS_T4_L0_SECONDARY_PREFETCH_POLICY,
    SOFT_MIXTURE_NORMALIZED_T4_L0_SECONDARY_PREFETCH_POLICY,
    SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY,
    SOFT_MIXTURE_KERNEL_CLOSURE_T4_L0_SECONDARY_PREFETCH_POLICY,
    SOFT_MIXTURE_T4_L0_SECONDARY_PREFETCH_POLICY,
)
from saes.probe_layout import compute_probe_positions
from saes.progressive_saes import (
    PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
    ProgressiveSAES,
    paper_assignment_weights,
)


DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION = "saes-depthsplat-l0-l1-materializer-v1"
DEPTHSPLAT_COMPACT_AGGREGATION = (
    "depthsplat-selected-rgb-sh-z-depth-conditional-moment-merge-v1"
)
DEPTHSPLAT_COVERAGE_CERTIFICATE = (
    "depthsplat-selected-z-depth-3d-2sigma-ellipsoid-support-v2"
)
DEPTHSPLAT_LITERAL_PAPER_T4_MOMENT_CERTIFICATE = (
    "depthsplat-literal-paper-t4-finite-psd-moment-merge-fixed-scale-v1"
)
DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE = (
    "depthsplat-coverage-enriched-t4-owner-anchor-projected-2sigma-fixed-scale-v2"
)
DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE = (
    "depthsplat-support-basis-t4-composed-soft-ledger-projected-2sigma-fixed-scale-v1"
)
DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE = (
    "depthsplat-soft-mixture-t4-source-only-moment-replay-fixed-scale-v1"
)
DEPTHSPLAT_COMPACT_COVERAGE_MAX_COVARIANCE_SCALE = 16.0
DEPTHSPLAT_FEATURE_INTERPOLATION_MAX_RELATIVE_RESIDUAL = 1.0
DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE = (
    "depthsplat-development-omitted-z-alpha-union-v1"
)
DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE = (
    "depthsplat-literal-paper-t4-selected-probe-moment-v1"
)
DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE = (
    "depthsplat-coverage-enriched-t4-balanced-l1-owner-support-v1"
)
DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE = (
    "depthsplat-support-basis-t4-balanced-l1-cooperative-v1"
)
DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE = (
    "depthsplat-soft-mixture-t4-balanced-l1-source-only-moment-replay-v1"
)
DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE = (
    "depthsplat-soft-mixture-normalized-t4-balanced-l1-source-only-moment-replay-v2"
)
DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE = (
    "depthsplat-soft-mixture-kernel-closure-t4-balanced-l1-source-only-v3"
)
DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE = (
    "depthsplat-direct-conditional-t4-balanced-l1-source-only-v1"
)
DEPTHSPLAT_LITERAL_PAPER_T4_DECISION_SEMANTICS = (
    "paper-probe-feature-variance-first-hit"
)
DEPTHSPLAT_LITERAL_PAPER_T4_V16_RECORD_KIND = (
    "depthsplat-nonzero-l0-l1-acid-disjoint-v16l-t4"
)
NATIVE_OPACITY_ENDPOINT_FULL_REASON = "native_opacity_endpoint_requires_full"
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE = (
    "depthsplat-selected-rgb-sh-opacity-all-anchor-loo-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY = (
    "all-retained-l0-l1-anchors-selected-labels-only-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_AGGREGATE_SCHEMA = (
    "depthsplat-selected-anchor-attribute-loo-aggregate-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_GUARD_SCHEMA = (
    "depthsplat-selected-anchor-attribute-loo-frozen-v16-guard-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC = (
    "maximum-held-out-anchor-risk-v1"
)
DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_ENDPOINT_UNSCORABLE = (
    "native-opacity-endpoint-promoted-full-v1"
)
DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_AGGREGATE_SCHEMA = (
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


class _DepthSplatTileRejection(ValueError):
    """An expected source-only guard decision that promotes one tile to Full."""


_SOFT_MIXTURE_EXECUTION_PROFILES = frozenset(
    (
        DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
    )
)


def _is_soft_mixture_profile(execution_profile: Any) -> bool:
    return execution_profile in _SOFT_MIXTURE_EXECUTION_PROFILES


def _is_kernel_closure_profile(execution_profile: Any) -> bool:
    return (
        execution_profile
        == DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
    )


def _is_direct_conditional_profile(execution_profile: Any) -> bool:
    return execution_profile == DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE


@dataclass(frozen=True)
class DepthSplatCompactMaterializationPreflight:
    """Selected-only updates and Full promotions for one probe plan."""

    update_dense_slots: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor
    promote_full_mask: torch.Tensor
    tile_trace: tuple[dict[str, Any], ...]
    events: dict[str, Any]


@dataclass(frozen=True)
class DepthSplatCompactFinalRoute:
    """Final decoder output and producer request masks after preflight."""

    selected_output_mask: torch.Tensor
    additional_full_mask: torch.Tensor
    raw_head_request_mask: torch.Tensor
    full_passthrough_mask: torch.Tensor
    tile_trace: tuple[dict[str, Any], ...]
    events: dict[str, Any]


def _tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("DepthSplat materializer tensor digest requires a tensor")
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _mask_sha256(mask: torch.Tensor) -> str:
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise ValueError("DepthSplat materializer mask must be boolean")
    return _tensor_sha256(mask.to(dtype=torch.uint8))


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("DepthSplat materializer trace must be JSON-serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def _require_bound_tile_trace(
    tile_trace: Any, *, expected_sha256: Any, label: str
) -> str:
    """Rehash a live trace before it can influence a routing decision."""

    if not isinstance(tile_trace, tuple) or not all(
        isinstance(record, Mapping) for record in tile_trace
    ):
        raise ValueError(f"DepthSplat {label} is invalid")
    expected = _require_sha256(expected_sha256, label=label)
    actual = _canonical_sha256(tile_trace)
    if actual != expected:
        raise ValueError(f"DepthSplat {label} hash changed")
    return actual


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"DepthSplat materializer has no valid {label} hash")
    return value


def _full_positions(tile_size: int) -> list[tuple[int, int]]:
    return [(row, column) for row in range(tile_size) for column in range(tile_size)]


def _tile_slots(
    *,
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    tile_size: int,
    positions: list[tuple[int, int]],
) -> list[int]:
    slots: list[int] = []
    for row, column in positions:
        if not 0 <= row < tile_size or not 0 <= column < tile_size:
            raise ValueError("DepthSplat materializer local anchor is invalid")
        absolute_row = tile_y * tile_size + row
        absolute_column = tile_x * tile_size + column
        if not 0 <= absolute_row < height or not 0 <= absolute_column < width:
            raise ValueError("DepthSplat materializer anchor falls outside its image")
        slots.append(view * (height * width) + absolute_row * width + absolute_column)
    return slots


def _mark_tile(
    mask: torch.Tensor,
    *,
    view: int,
    tile_y: int,
    tile_x: int,
    tile_size: int,
    positions: list[tuple[int, int]] | None = None,
) -> None:
    for row, column in positions if positions is not None else _full_positions(tile_size):
        mask[view, tile_y * tile_size + row, tile_x * tile_size + column] = True


def _require_plan(plan: IncrementalProbeFirstPlan) -> tuple[int, int, int, str]:
    if not isinstance(plan, IncrementalProbeFirstPlan):
        raise TypeError("DepthSplat materializer requires an incremental probe plan")
    events = plan.events
    if events.get("contract_version") not in {
        "saes-incremental-probe-first-plan-v1",
        LITERAL_PAPER_T4_PLAN_CONTRACT,
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_PLAN_CONTRACT,
        DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT,
        DEPTHSPLAT_SOFT_MIXTURE_T4_PLAN_CONTRACT,
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_PLAN_CONTRACT,
        DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
    }:
        raise ValueError("DepthSplat materializer plan contract changed")
    tile_size = events.get("tile_size")
    if tile_size != 4:
        raise ValueError("DepthSplat materializer supports exactly T=4")
    masks = (plan.primary_mask, plan.secondary_mask, plan.full_mask, plan.selection_mask)
    if any(not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 3 for mask in masks):
        raise ValueError("DepthSplat materializer plan masks are invalid")
    if not torch.equal(plan.selection_mask, plan.primary_mask | plan.secondary_mask | plan.full_mask):
        raise ValueError("DepthSplat materializer plan mask union is inconsistent")
    views, height, width = plan.selection_mask.shape
    if min(views, height, width) < 1 or height % tile_size or width % tile_size:
        raise ValueError("DepthSplat materializer plan geometry is invalid")
    semantics = events.get("l1_anchor_semantics", LEGACY_L1_ANCHOR_SEMANTICS)
    if semantics not in {
        PAPER_KP_ANCHOR_SEMANTICS,
        LEGACY_L1_ANCHOR_SEMANTICS,
        BALANCED_L1_ANCHOR_SEMANTICS,
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    }:
        raise ValueError("DepthSplat materializer plan L1 semantics are invalid")
    if len(plan.tile_trace) != views * (height // tile_size) * (width // tile_size):
        raise ValueError("DepthSplat materializer plan tile trace is incomplete")
    _require_bound_tile_trace(
        plan.tile_trace,
        expected_sha256=events.get("tile_trace_sha256"),
        label="plan tile trace",
    )
    return views, height, width, str(semantics)


def _validate_execution_profile_plan_binding(
    plan: IncrementalProbeFirstPlan, *, execution_profile: str
) -> None:
    """Keep each materializer mechanism tied to its own probe-plan contract."""

    expected_profiles = {
        "saes-incremental-probe-first-plan-v1": DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
        LITERAL_PAPER_T4_PLAN_CONTRACT: DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_PLAN_CONTRACT: (
            DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE
        ),
        DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT: (
            DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE
        ),
        DEPTHSPLAT_SOFT_MIXTURE_T4_PLAN_CONTRACT: (
            DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE
        ),
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_PLAN_CONTRACT: {
            DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
        },
        DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT: (
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
        ),
    }.get(plan.events.get("contract_version"))
    if expected_profiles is None:
        raise ValueError("DepthSplat materializer execution profile does not match plan contract")
    if isinstance(expected_profiles, str):
        expected_profiles = {expected_profiles}
    if execution_profile not in expected_profiles:
        raise ValueError("DepthSplat materializer execution profile does not match plan contract")


def _validate_literal_paper_t4_plan(
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
    semantics: str,
) -> None:
    """Reject any engineering routing behavior from the formal T=4 profile."""

    events = plan.events
    if (
        events.get("contract_version") != LITERAL_PAPER_T4_PLAN_CONTRACT
        or events.get("formal_paper_kp4") is not True
        or events.get("decision_semantics")
        != DEPTHSPLAT_LITERAL_PAPER_T4_DECISION_SEMANTICS
        or events.get("feature_statistic") != "raw-probe-mean-channel-variance"
        or semantics != PAPER_KP_ANCHOR_SEMANTICS
        or events.get("l0_anchor_count") != 4
        or events.get("l1_anchor_count") != 4
        or events.get("l1_anchor_selection") != "fixed-layout-v1"
        or events.get("l1_anchor_selection_uses_s1_only") is not False
        or events.get("depth_checked_after_l0_miss_only") is not True
        or int(plan.secondary_mask.sum().item()) != 0
        or not torch.equal(plan.selection_mask, plan.primary_mask | plan.full_mask)
    ):
        raise ValueError("DepthSplat formal paper T=4 routing contract changed")
    corners = [list(position) for position in compute_probe_positions(4)]
    expected_tiles = views * (height // 4) * (width // 4)
    if len(plan.tile_trace) != expected_tiles:
        raise ValueError("DepthSplat formal paper T=4 tile trace is incomplete")
    for record in plan.tile_trace:
        if not isinstance(record, Mapping):
            raise ValueError("DepthSplat formal paper T=4 tile record is invalid")
        route = record.get("pre_guard_route")
        if (
            route not in {"L0", "L1", "Full"}
            or record.get("primary_local_positions") != corners
            or record.get("secondary_local_positions") != []
            or record.get("depth_checked_after_l0_miss_only") is not True
            or "adaptive_l1_omitted_local_position" in record
            or "adaptive_l1_leave_one_out_residual" in record
        ):
            raise ValueError("DepthSplat formal paper T=4 route is not literal")
        if route == "L0" and record.get("depth_uniform") is not None:
            raise ValueError("DepthSplat formal L0 route inspected depth before a miss")
        if route in {"L1", "Full"} and not isinstance(record.get("depth_uniform"), bool):
            raise ValueError("DepthSplat formal L1/Full route lacks conditional depth evidence")


def _validate_coverage_enriched_t4_plan(
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
    semantics: str,
) -> None:
    """Validate the opt-in L0-to-balanced-L1 prefetch route.

    This is intentionally separate from the literal paper contract.  It uses
    the same raw four-corner feature decision but reads source depth probes to
    make an L1 packet available only for a later fail-closed coverage retry.
    """

    events = plan.events
    if (
        events.get("contract_version")
        != DEPTHSPLAT_COVERAGE_ENRICHED_T4_PLAN_CONTRACT
        or events.get("formal_paper_kp4") is not False
        or events.get("decision_semantics")
        != DEPTHSPLAT_LITERAL_PAPER_T4_DECISION_SEMANTICS
        or events.get("feature_statistic") != "raw-probe-mean-channel-variance"
        or semantics != BALANCED_L1_ANCHOR_SEMANTICS
        or events.get("l0_anchor_count") != 4
        or events.get("l1_anchor_count") != 12
        or events.get("depth_checked_after_l0_miss_only") is not False
        or events.get("coverage_enriched_l0_secondary_prefetch_policy")
        != COVERAGE_ENRICHED_T4_L0_SECONDARY_PREFETCH_POLICY
        or events.get("coverage_enriched_t4_route_config_sha256")
        != coverage_enriched_t4_route_config_sha256(events)
    ):
        raise ValueError("DepthSplat coverage-enriched T=4 routing contract changed")
    corners = [list(position) for position in compute_probe_positions(4)]
    balanced = [
        list(position)
        for position in l1_local_positions_for_tile(
            {}, tile_size=4, l1_anchor_semantics=BALANCED_L1_ANCHOR_SEMANTICS
        )
    ]
    secondary = balanced[len(corners) :]
    if len(plan.tile_trace) != views * (height // 4) * (width // 4):
        raise ValueError("DepthSplat coverage-enriched T=4 tile trace is incomplete")
    for record in plan.tile_trace:
        route = record.get("pre_guard_route")
        depth_uniform = record.get("depth_uniform")
        if (
            route not in {"L0", "L1", "Full"}
            or record.get("primary_local_positions") != corners
            or not isinstance(depth_uniform, bool)
        ):
            raise ValueError("DepthSplat coverage-enriched T=4 tile trace is invalid")
        expected_secondary = (
            secondary if route == "L1" or (route == "L0" and depth_uniform) else []
        )
        if record.get("secondary_local_positions") != expected_secondary:
            raise ValueError("DepthSplat coverage-enriched L1 prefetch changed")


def _validate_support_basis_t4_plan(
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
    semantics: str,
) -> None:
    """Validate the separate cooperative support-basis L0-to-L1 route."""

    events = plan.events
    if (
        events.get("contract_version") != DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT
        or events.get("formal_paper_kp4") is not False
        or events.get("decision_semantics")
        != DEPTHSPLAT_LITERAL_PAPER_T4_DECISION_SEMANTICS
        or events.get("feature_statistic") != "raw-probe-mean-channel-variance"
        or semantics != BALANCED_L1_ANCHOR_SEMANTICS
        or events.get("l0_anchor_count") != 4
        or events.get("l1_anchor_count") != 12
        or events.get("depth_checked_after_l0_miss_only") is not False
        or events.get("support_basis_l0_secondary_prefetch_policy")
        != SUPPORT_BASIS_T4_L0_SECONDARY_PREFETCH_POLICY
        or events.get("support_basis_t4_route_config_sha256")
        != support_basis_t4_route_config_sha256(events)
    ):
        raise ValueError("DepthSplat support-basis T=4 routing contract changed")
    corners = [list(position) for position in compute_probe_positions(4)]
    balanced = [
        list(position)
        for position in l1_local_positions_for_tile(
            {}, tile_size=4, l1_anchor_semantics=BALANCED_L1_ANCHOR_SEMANTICS
        )
    ]
    secondary = balanced[len(corners) :]
    if len(plan.tile_trace) != views * (height // 4) * (width // 4):
        raise ValueError("DepthSplat support-basis T=4 tile trace is incomplete")
    for record in plan.tile_trace:
        route = record.get("pre_guard_route")
        depth_uniform = record.get("depth_uniform")
        if (
            route not in {"L0", "L1", "Full"}
            or record.get("primary_local_positions") != corners
            or not isinstance(depth_uniform, bool)
        ):
            raise ValueError("DepthSplat support-basis T=4 tile trace is invalid")
        expected_secondary = (
            secondary if route == "L1" or (route == "L0" and depth_uniform) else []
        )
        if record.get("secondary_local_positions") != expected_secondary:
            raise ValueError("DepthSplat support-basis L1 prefetch changed")


def _validate_soft_mixture_t4_plan(
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
    semantics: str,
) -> None:
    """Validate the separate source-only soft-mixture L0-to-L1 route."""

    events = plan.events
    if (
        events.get("contract_version") != DEPTHSPLAT_SOFT_MIXTURE_T4_PLAN_CONTRACT
        or events.get("formal_paper_kp4") is not False
        or events.get("decision_semantics")
        != DEPTHSPLAT_LITERAL_PAPER_T4_DECISION_SEMANTICS
        or events.get("feature_statistic") != "raw-probe-mean-channel-variance"
        or semantics != BALANCED_L1_ANCHOR_SEMANTICS
        or events.get("l0_anchor_count") != 4
        or events.get("l1_anchor_count") != 12
        or events.get("depth_checked_after_l0_miss_only") is not False
        or events.get("soft_mixture_l0_secondary_prefetch_policy")
        != SOFT_MIXTURE_T4_L0_SECONDARY_PREFETCH_POLICY
        or events.get("soft_mixture_t4_route_config_sha256")
        != soft_mixture_t4_route_config_sha256(events)
    ):
        raise ValueError("DepthSplat soft-mixture T=4 routing contract changed")
    corners = [list(position) for position in compute_probe_positions(4)]
    balanced = [
        list(position)
        for position in l1_local_positions_for_tile(
            {}, tile_size=4, l1_anchor_semantics=BALANCED_L1_ANCHOR_SEMANTICS
        )
    ]
    secondary = balanced[len(corners) :]
    if len(plan.tile_trace) != views * (height // 4) * (width // 4):
        raise ValueError("DepthSplat soft-mixture T=4 tile trace is incomplete")
    for record in plan.tile_trace:
        route = record.get("pre_guard_route")
        depth_uniform = record.get("depth_uniform")
        if (
            route not in {"L0", "L1", "Full"}
            or record.get("primary_local_positions") != corners
            or not isinstance(depth_uniform, bool)
        ):
            raise ValueError("DepthSplat soft-mixture T=4 tile trace is invalid")
        expected_secondary = (
            secondary if route == "L1" or (route == "L0" and depth_uniform) else []
        )
        if record.get("secondary_local_positions") != expected_secondary:
            raise ValueError("DepthSplat soft-mixture L1 prefetch changed")


def _validate_soft_mixture_normalized_t4_plan(
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
    semantics: str,
) -> None:
    """Validate the scale-invariant paper-feature soft-mixture route."""

    events = plan.events
    if (
        events.get("contract_version")
        != DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_PLAN_CONTRACT
        or events.get("formal_paper_kp4") is not False
        or events.get("decision_semantics")
        != PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS
        or events.get("feature_statistic")
        != "normalized-probe-vector-standard-deviation"
        or semantics != BALANCED_L1_ANCHOR_SEMANTICS
        or events.get("l0_anchor_count") != 4
        or events.get("l1_anchor_count") != 12
        or events.get("depth_checked_after_l0_miss_only") is not False
        or events.get("assignment_feature_semantics")
        != "unit-normalized-bilinear-s1-v1"
        or events.get("soft_mixture_normalized_l0_secondary_prefetch_policy")
        != SOFT_MIXTURE_NORMALIZED_T4_L0_SECONDARY_PREFETCH_POLICY
        or events.get("soft_mixture_normalized_t4_route_config_sha256")
        != soft_mixture_normalized_t4_route_config_sha256(events)
    ):
        raise ValueError("DepthSplat soft-mixture normalized T=4 routing contract changed")
    corners = [list(position) for position in compute_probe_positions(4)]
    balanced = [
        list(position)
        for position in l1_local_positions_for_tile(
            {}, tile_size=4, l1_anchor_semantics=BALANCED_L1_ANCHOR_SEMANTICS
        )
    ]
    secondary = balanced[len(corners) :]
    if len(plan.tile_trace) != views * (height // 4) * (width // 4):
        raise ValueError("DepthSplat soft-mixture normalized T=4 tile trace is incomplete")
    for record in plan.tile_trace:
        route = record.get("pre_guard_route")
        depth_uniform = record.get("depth_uniform")
        if (
            route not in {"L0", "L1", "Full"}
            or record.get("primary_local_positions") != corners
            or not isinstance(depth_uniform, bool)
        ):
            raise ValueError("DepthSplat soft-mixture normalized T=4 tile trace is invalid")
        expected_secondary = (
            secondary if route == "L1" or (route == "L0" and depth_uniform) else []
        )
        if record.get("secondary_local_positions") != expected_secondary:
            raise ValueError("DepthSplat soft-mixture normalized L1 prefetch changed")


def _validate_soft_mixture_kernel_closure_t4_plan(
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
    semantics: str,
) -> None:
    """Validate the normalized route and its strict closure-guard binding."""

    events = plan.events
    if (
        events.get("contract_version")
        != DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT
        or events.get("formal_paper_kp4") is not False
        or events.get("decision_semantics")
        != PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS
        or events.get("feature_statistic")
        != "normalized-probe-vector-standard-deviation"
        or events.get("assignment_feature_semantics")
        != "unit-normalized-bilinear-s1-v1"
        or semantics != BALANCED_L1_ANCHOR_SEMANTICS
        or events.get("l0_anchor_count") != 4
        or events.get("l1_anchor_count") != 12
        or events.get("depth_checked_after_l0_miss_only") is not False
        or events.get("soft_mixture_kernel_closure_l0_secondary_prefetch_policy")
        != SOFT_MIXTURE_KERNEL_CLOSURE_T4_L0_SECONDARY_PREFETCH_POLICY
        or events.get("kernel_closure_guard_policy")
        != SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY
        or events.get("soft_mixture_kernel_closure_t4_route_config_sha256")
        != soft_mixture_kernel_closure_t4_route_config_sha256(events)
    ):
        raise ValueError("DepthSplat soft-mixture kernel-closure routing contract changed")
    corners = [list(position) for position in compute_probe_positions(4)]
    balanced = [
        list(position)
        for position in l1_local_positions_for_tile(
            {}, tile_size=4, l1_anchor_semantics=BALANCED_L1_ANCHOR_SEMANTICS
        )
    ]
    secondary = balanced[len(corners) :]
    if len(plan.tile_trace) != views * (height // 4) * (width // 4):
        raise ValueError("DepthSplat soft-mixture kernel-closure tile trace is incomplete")
    for record in plan.tile_trace:
        route = record.get("pre_guard_route")
        depth_uniform = record.get("depth_uniform")
        if (
            route not in {"L0", "L1", "Full"}
            or record.get("primary_local_positions") != corners
            or not isinstance(depth_uniform, bool)
        ):
            raise ValueError("DepthSplat soft-mixture kernel-closure tile trace is invalid")
        expected_secondary = (
            secondary if route == "L1" or (route == "L0" and depth_uniform) else []
        )
        if record.get("secondary_local_positions") != expected_secondary:
            raise ValueError("DepthSplat soft-mixture kernel-closure L1 prefetch changed")


def _validate_source(
    packet: DepthSplatSparseRawPacket,
    packed: DepthSplatPackedGaussianAttributes,
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
) -> dict[int, int]:
    if not isinstance(packet, DepthSplatSparseRawPacket) or not isinstance(
        packed, DepthSplatPackedGaussianAttributes
    ):
        raise TypeError("DepthSplat materializer requires native selected packet and attributes")
    count = int(packet.dense_slots.numel())
    if (
        count < 1
        or packet.descriptor_keys.shape != (count, 4)
        or packet.descriptor_keys.dtype != torch.int64
        or packet.raw_head_descriptors.ndim != 2
        or packet.raw_head_descriptors.shape[0] != count
        or packet.raw_head_descriptors.shape[1] < 4
        or packet.extrinsics.shape != (count, 4, 4)
        or packet.intrinsics.shape != (count, 3, 3)
        or packet.coordinates.shape != (count, 2)
        or packet.depths.shape != (count,)
        or packet.mapped_opacities.shape != (count,)
        or packet.source_rgb.shape != (count, 3)
        or packet.dense_slots.shape != (count,)
        or packet.dense_slots.dtype != torch.int64
        or packed.dense_slots.shape != (count,)
        or not torch.equal(packet.dense_slots, packed.dense_slots)
        or packed.means.shape != (count, 3)
        or packed.covariances.shape != (count, 3, 3)
        or packed.harmonics.ndim != 3
        or packed.harmonics.shape[:2] != (count, 3)
        or packed.opacities.shape != (count,)
    ):
        raise ValueError("DepthSplat materializer selected packet tensors are inconsistent")
    values = (
        packet.raw_head_descriptors,
        packet.extrinsics,
        packet.intrinsics,
        packet.coordinates,
        packet.depths,
        packet.mapped_opacities,
        packet.source_rgb,
        packed.means,
        packed.covariances,
        packed.harmonics,
        packed.opacities,
    )
    if any(value.device != packet.raw_head_descriptors.device for value in values):
        raise ValueError("DepthSplat materializer inputs must share one device")
    if any(value.dtype != torch.float32 for value in values):
        raise ValueError("DepthSplat materializer requires strict float32 native attributes")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("DepthSplat materializer selected inputs must be finite")
    if not bool((packet.depths > 0.0).all()):
        raise ValueError("DepthSplat materializer requires positive selected z-depths")
    if not torch.allclose(
        packet.mapped_opacities,
        packet.raw_head_descriptors[:, 0].sigmoid(),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("DepthSplat materializer packet opacity logit drifted")
    if not bool((packed.opacities >= 0.0).all()) or not bool((packed.opacities <= 1.0).all()):
        raise ValueError("DepthSplat materializer selected opacities are invalid")
    covariance = (packed.covariances + packed.covariances.mT) * 0.5
    if float((packed.covariances - packed.covariances.mT).abs().amax().item()) > 1e-4:
        raise ValueError("DepthSplat materializer selected covariance is not symmetric")
    if bool((torch.linalg.eigvalsh(covariance) < -1e-6).any()):
        raise ValueError("DepthSplat materializer selected covariance is not PSD")
    expected_positions = plan.selection_mask.nonzero(as_tuple=False)
    expected_slots = (
        expected_positions[:, 0] * (height * width)
        + expected_positions[:, 1] * width
        + expected_positions[:, 2]
    ).to(device=packet.dense_slots.device, dtype=torch.int64)
    if not torch.equal(packet.dense_slots, expected_slots):
        raise ValueError("DepthSplat materializer packet does not match planned producer selection")
    trace = packet.source_trace
    required_trace = {
        "contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
        "source_bound": True,
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
        "source_rgb_keyword": "input_images",
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "head_forward_invocations": 1,
    }
    if not isinstance(trace, Mapping) or any(trace.get(key) != value for key, value in required_trace.items()):
        raise ValueError("DepthSplat materializer packet lacks native RGB/z-depth provenance")
    if trace.get("selection_mask_sha256") != _mask_sha256(plan.selection_mask):
        raise ValueError("DepthSplat materializer packet selection hash drifted")
    if (
        _require_sha256(trace.get("selected_descriptor_sha256"), label="selected descriptor")
        != _tensor_sha256(packet.raw_head_descriptors)
        or _require_sha256(trace.get("selected_rgb_sha256"), label="selected RGB")
        != _tensor_sha256(packet.source_rgb)
        or _require_sha256(
            trace.get("native_full_passthrough_mask_sha256"), label="native Full mask"
        )
        != _mask_sha256(plan.full_mask)
        or trace.get("native_full_passthrough_positions")
        != int(plan.full_mask.sum().item())
        or trace.get("source_view_count") != views
        or trace.get("source_image_shape") != [height, width]
    ):
        raise ValueError("DepthSplat materializer packet source geometry binding drifted")
    native_execution_sha256 = _require_sha256(
        trace.get("native_execution_sha256"), label="native execution"
    )
    if packed.source_trace.get("source_bound") is not True:
        raise ValueError("DepthSplat materializer packed attributes are not source-bound")
    if packed.source_trace_sha256 != canonical_json_sha256(dict(packed.source_trace)):
        raise ValueError("DepthSplat materializer packed trace digest drifted")
    attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=packed.dense_slots,
        means=packed.means,
        covariances=packed.covariances,
        harmonics=packed.harmonics,
        opacities=packed.opacities,
    )
    if (
        packed.attribute_binding_sha256 != attribute_binding
        or packed.source_trace.get("native_adapter_attribute_binding_sha256")
        != attribute_binding
    ):
        raise ValueError("DepthSplat materializer native Adapter attributes are unbound")
    full_positions = plan.full_mask.nonzero(as_tuple=False).to(device=packed.dense_slots.device)
    full_slots = (
        full_positions[:, 0] * (height * width)
        + full_positions[:, 1] * width
        + full_positions[:, 2]
    ).to(dtype=torch.int64)
    full_indices = torch.searchsorted(packed.dense_slots, full_slots)
    if bool((full_indices >= packed.dense_slots.numel()).any()) or not torch.equal(
        packed.dense_slots[full_indices], full_slots
    ):
        raise ValueError("DepthSplat materializer native Full slot is absent")
    full_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=full_slots,
        means=packed.means[full_indices],
        covariances=packed.covariances[full_indices],
        harmonics=packed.harmonics[full_indices],
        opacities=packed.opacities[full_indices],
    )
    if (
        packed.source_trace.get("native_full_adapter_attribute_execution_sha256")
        != native_execution_sha256
        or packed.source_trace.get("native_full_adapter_attribute_passthrough_mask_sha256")
        != _mask_sha256(plan.full_mask)
        or packed.source_trace.get("native_full_adapter_attribute_passthrough_count")
        != int(full_slots.numel())
        or packed.source_trace.get("native_full_adapter_attribute_binding_sha256")
        != full_attribute_binding
    ):
        raise ValueError("DepthSplat materializer native Full attributes are unbound")
    expected_packed_trace = dict(packet.source_trace)
    expected_packed_trace.update(
        {
            "native_adapter_attribute_binding_sha256": attribute_binding,
            "native_full_adapter_attribute_execution_sha256": native_execution_sha256,
            "native_full_adapter_attribute_passthrough_mask_sha256": _mask_sha256(
                plan.full_mask
            ),
            "native_full_adapter_attribute_binding_sha256": full_attribute_binding,
            "native_full_adapter_attribute_passthrough_count": int(full_slots.numel()),
            "selected_native_rgb_adapter_compact_count": count - int(full_slots.numel()),
            "selected_native_rgb_adapter_executed": count > int(full_slots.numel()),
        }
    )
    if dict(packed.source_trace) != expected_packed_trace:
        raise ValueError("DepthSplat materializer packet and Adapter traces diverged")
    slots = packet.dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(set(slots)) != len(slots) or any(slot < 0 or slot >= views * height * width for slot in slots):
        raise ValueError("DepthSplat materializer selected slots are invalid")
    if len(slots) > 1 and any(right <= left for left, right in zip(slots, slots[1:])):
        raise ValueError("DepthSplat materializer selected slots must be strictly ordered")
    return {int(slot): index for index, slot in enumerate(slots)}


def _validate_routing_inputs(
    routing_features: torch.Tensor,
    routing_z_depths: torch.Tensor,
    packet: DepthSplatSparseRawPacket,
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
    execution_profile: str,
) -> tuple[torch.Tensor, str]:
    if (
        not torch.is_tensor(routing_features)
        or routing_features.ndim != 5
        or routing_features.shape[0] != 1
        or routing_features.shape[1] != views
        or tuple(routing_features.shape[-2:]) != (height, width)
        or routing_features.shape[2] < 1
        or routing_features.dtype != torch.float32
        or routing_features.device != packet.raw_head_descriptors.device
        or not bool(torch.isfinite(routing_features).all())
    ):
        raise ValueError("DepthSplat materializer routing features are invalid")
    if (
        not torch.is_tensor(routing_z_depths)
        or routing_z_depths.shape != (1, views, height, width)
        or routing_z_depths.device != packet.raw_head_descriptors.device
        or routing_z_depths.dtype != torch.float32
        or not bool(torch.isfinite(routing_z_depths).all())
        or bool((routing_z_depths <= 0.0).any())
    ):
        raise ValueError("DepthSplat materializer routing z-depths are invalid")
    if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
        rebuilt = build_literal_paper_t4_probe_first_plan(
            routing_features,
            routing_z_depths,
            height=height,
            width=width,
            feature_threshold=float(plan.events["feature_threshold"]),
            depth_threshold=float(plan.events["depth_threshold"]),
        )
    elif execution_profile == DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE:
        rebuilt = build_depthsplat_coverage_enriched_t4_probe_first_plan(
            routing_features,
            routing_z_depths,
            height=height,
            width=width,
            feature_threshold=float(plan.events["feature_threshold"]),
            depth_threshold=float(plan.events["depth_threshold"]),
        )
    elif execution_profile == DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE:
        rebuilt = build_depthsplat_support_basis_t4_probe_first_plan(
            routing_features,
            routing_z_depths,
            height=height,
            width=width,
            feature_threshold=float(plan.events["feature_threshold"]),
            depth_threshold=float(plan.events["depth_threshold"]),
        )
    elif execution_profile == DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE:
        rebuilt = build_depthsplat_soft_mixture_t4_probe_first_plan(
            routing_features,
            routing_z_depths,
            height=height,
            width=width,
            feature_threshold=float(plan.events["feature_threshold"]),
            depth_threshold=float(plan.events["depth_threshold"]),
        )
    elif execution_profile in {
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
    }:
        rebuilt = build_depthsplat_soft_mixture_normalized_t4_probe_first_plan(
            routing_features,
            routing_z_depths,
            height=height,
            width=width,
            feature_threshold=float(plan.events["feature_threshold"]),
            depth_threshold=float(plan.events["depth_threshold"]),
        )
    elif execution_profile == DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE:
        rebuilt = build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan(
            routing_features,
            routing_z_depths,
            height=height,
            width=width,
            feature_threshold=float(plan.events["feature_threshold"]),
            depth_threshold=float(plan.events["depth_threshold"]),
        )
    elif execution_profile == DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE:
        rebuilt = build_incremental_probe_first_plan(
            routing_features,
            routing_z_depths,
            height=height,
            width=width,
            tile_size=4,
            feature_threshold=float(plan.events["feature_threshold"]),
            depth_threshold=float(plan.events["depth_threshold"]),
            decision_semantics=str(plan.events["decision_semantics"]),
            l1_anchor_semantics=str(plan.events["l1_anchor_semantics"]),
        )
    else:
        raise ValueError("DepthSplat materializer execution profile is invalid")
    if (
        not torch.equal(rebuilt.primary_mask, plan.primary_mask)
        or not torch.equal(rebuilt.secondary_mask, plan.secondary_mask)
        or not torch.equal(rebuilt.full_mask, plan.full_mask)
        or rebuilt.events.get("tile_trace_sha256") != plan.events.get("tile_trace_sha256")
    ):
        raise ValueError("DepthSplat materializer routing tensors do not reproduce the frozen plan")
    trace = packet.source_trace
    if _require_sha256(
        trace.get("routing_features_sha256"), label="routing features"
    ) != _tensor_sha256(routing_features):
        raise ValueError("DepthSplat materializer routing feature hash drifted")
    if _require_sha256(
        trace.get("routing_z_depths_sha256"), label="routing z-depth"
    ) != _tensor_sha256(routing_z_depths):
        raise ValueError("DepthSplat materializer routing z-depth hash drifted")
    positions = plan.selection_mask.nonzero(as_tuple=False)
    expected_depths = routing_z_depths[0, positions[:, 0], positions[:, 1], positions[:, 2]]
    if not torch.allclose(packet.depths, expected_depths, rtol=1e-5, atol=1e-5):
        raise ValueError("DepthSplat materializer selected packet z-depth drifted from router")
    feature_statistic = plan.events.get("feature_statistic")
    if not isinstance(feature_statistic, str):
        raise ValueError("DepthSplat materializer plan feature statistic is invalid")
    _scores, feature_norm = ProgressiveSAES.classify_tiles_by_features(
        routing_features,
        height,
        width,
        4,
        threshold=float(plan.events["feature_threshold"]),
        per_view=True,
        statistic=feature_statistic,
    )
    if (
        not torch.is_tensor(feature_norm)
        or feature_norm.shape != (views, routing_features.shape[2], height, width)
        or not bool(torch.isfinite(feature_norm).all())
    ):
        raise RuntimeError("DepthSplat materializer could not normalize routing features")
    if execution_profile in {
        DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
    }:
        # The literal/raw profiles measure raw probe variance. Their bilateral
        # assignments therefore use raw bilinear S1; the normalized soft
        # mixture profile intentionally falls through to ``feature_norm``.
        raw_assignment_features = torch.nn.functional.interpolate(
            routing_features[0], size=(height, width), mode="bilinear", align_corners=False
        )
        if (
            raw_assignment_features.shape
            != (views, routing_features.shape[2], height, width)
            or not bool(torch.isfinite(raw_assignment_features).all())
        ):
            raise RuntimeError("DepthSplat materializer could not upsample raw S1 features")
        return raw_assignment_features, "raw-bilinear-s1-v1"
    if (
        execution_profile
        in {
            DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
        }
        and plan.events.get("assignment_feature_semantics")
        != "unit-normalized-bilinear-s1-v1"
    ):
        raise ValueError("DepthSplat normalized soft-mixture assignment semantics changed")
    return feature_norm, "unit-normalized-bilinear-s1-v1"


def _tile_feature_variance(record: Mapping[str, Any], feature_statistic: str) -> float:
    score = record.get("feature_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("DepthSplat materializer tile feature score is invalid")
    score = float(score)
    if not torch.isfinite(torch.tensor(score)) or score < 0.0:
        raise ValueError("DepthSplat materializer tile feature score is non-finite")
    return score * score if feature_statistic == "normalized-probe-vector-standard-deviation" else score


def _pixel_centres(
    positions: list[tuple[int, int]],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not positions:
        return torch.empty(0, 2, device=device, dtype=dtype)
    if any(row < 0 or row >= height or column < 0 or column >= width for row, column in positions):
        raise ValueError("DepthSplat materializer position is outside the image")
    rows = torch.tensor([row for row, _ in positions], device=device, dtype=dtype)
    columns = torch.tensor([column for _, column in positions], device=device, dtype=dtype)
    return torch.stack(((columns + 0.5) / width, (rows + 0.5) / height), dim=1)


def _source_image_grid(
    source_sample_image_grid: Callable[..., tuple[Any, Any]],
    *,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Capture the loaded source pixel-center grid once per preflight."""

    if not callable(source_sample_image_grid):
        raise TypeError("DepthSplat materializer requires source sample_image_grid")
    grid, _indices = source_sample_image_grid((height, width), device)
    if (
        not torch.is_tensor(grid)
        or grid.shape != (height, width, 2)
        or grid.device != device
        or not bool(torch.isfinite(grid).all())
    ):
        raise ValueError("DepthSplat source sample_image_grid output is invalid")
    return grid


def _coordinates_from_source_grid(
    raw_descriptors: torch.Tensor,
    positions: list[tuple[int, int]],
    source_grid: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Replay the loaded source's pixel-center plus sigmoid-offset expression."""

    if (
        raw_descriptors.ndim != 2
        or raw_descriptors.shape[0] != len(positions)
        or raw_descriptors.shape[1] < 3
        or source_grid.shape != (height, width, 2)
    ):
        raise ValueError("DepthSplat source-grid coordinate inputs are inconsistent")
    rows = torch.tensor([row for row, _ in positions], device=raw_descriptors.device)
    columns = torch.tensor([column for _, column in positions], device=raw_descriptors.device)
    if bool((rows < 0).any()) or bool((rows >= height).any()) or bool((columns < 0).any()) or bool((columns >= width).any()):
        raise ValueError("DepthSplat source-grid coordinate position is invalid")
    base = source_grid[rows, columns].to(dtype=raw_descriptors.dtype)
    pixel_size = torch.tensor((1.0 / width, 1.0 / height), device=base.device, dtype=base.dtype)
    return base + (raw_descriptors[:, 1:3].sigmoid() - 0.5) * pixel_size


def _spatial_weights(
    target_positions: list[tuple[int, int]],
    source_positions: list[tuple[int, int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not target_positions or not source_positions or len(set(source_positions)) != len(source_positions):
        raise ValueError("DepthSplat materializer spatial field is invalid")
    rows = [row for row, _ in source_positions]
    columns = [column for _, column in source_positions]
    top, bottom = min(rows), max(rows)
    left, right = min(columns), max(columns)
    corners = {(top, left), (top, right), (bottom, left), (bottom, right)}
    weights = torch.zeros((len(target_positions), len(source_positions)), device=device, dtype=dtype)
    if len(source_positions) == 4 and set(source_positions) == corners and top < bottom and left < right:
        source_to_index = {position: index for index, position in enumerate(source_positions)}
        for target_index, (row, column) in enumerate(target_positions):
            u = torch.as_tensor((column - left) / (right - left), device=device, dtype=dtype)
            v = torch.as_tensor((row - top) / (bottom - top), device=device, dtype=dtype)
            weights[target_index, source_to_index[(top, left)]] = (1.0 - u) * (1.0 - v)
            weights[target_index, source_to_index[(top, right)]] = u * (1.0 - v)
            weights[target_index, source_to_index[(bottom, left)]] = (1.0 - u) * v
            weights[target_index, source_to_index[(bottom, right)]] = u * v
    else:
        for target_index, (row, column) in enumerate(target_positions):
            squared = torch.tensor(
                [(row - source_row) ** 2 + (column - source_column) ** 2 for source_row, source_column in source_positions],
                device=device,
                dtype=dtype,
            )
            if bool((squared <= 0.0).any()):
                raise ValueError("DepthSplat materializer virtual target overlaps an anchor")
            inverse = squared.reciprocal()
            weights[target_index] = inverse / inverse.sum()
    if not bool(torch.isfinite(weights).all()) or not torch.allclose(
        weights.sum(dim=1), torch.ones(weights.shape[0], device=device, dtype=dtype), rtol=1e-5, atol=1e-5
    ):
        raise ValueError("DepthSplat materializer spatial field is not a partition of unity")
    return weights


def _selected_anchor_support_scale(values: torch.Tensor) -> torch.Tensor:
    """Use only retained-anchor variation to normalize a LOO residual."""

    if values.ndim < 2 or values.shape[0] < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("DepthSplat selected-anchor support is invalid")
    flattened = values.reshape(values.shape[0], -1)
    distances = torch.pdist(flattened)
    if distances.numel() == 0 or not bool(torch.isfinite(distances).all()):
        raise ValueError("DepthSplat selected-anchor support distances are invalid")
    magnitude = flattened.abs().amax().clamp_min(1.0)
    numerical_floor = torch.finfo(flattened.dtype).eps * 1024.0 * magnitude
    return torch.maximum(distances.median(), numerical_floor)


def _selected_anchor_opacity_logits(opacities: torch.Tensor) -> torch.Tensor:
    if (
        not bool(torch.isfinite(opacities).all())
        or bool((opacities < 0.0).any())
        or bool((opacities >= 1.0).any())
    ):
        raise ValueError("DepthSplat selected-anchor opacity is invalid")
    epsilon = torch.finfo(opacities.dtype).eps * 16.0
    bounded = opacities.clamp(min=epsilon, max=1.0 - epsilon)
    return torch.log(bounded) - torch.log1p(-bounded)


def _selected_anchor_attribute_loo_frozen_guard(
    value: Any | None,
    *,
    require_literal_t4_authenticated: bool = False,
    allow_serialized_literal_t4_projection: bool = False,
) -> dict[str, Any] | None:
    """Validate a guard without admitting a forged literal T=4 projection.

    Literal T=4 preflight accepts only the opaque capability issued by the
    calibration module after it has live-reloaded the frozen record.  A plain
    mapping is still accepted for the legacy development profile, and for the
    already-validated serialized evidence that a later final-route check
    replays; neither path can set a new literal runtime threshold.
    """

    if value is None:
        return None
    if require_literal_t4_authenticated and allow_serialized_literal_t4_projection:
        raise ValueError("DepthSplat LOO guard validation mode is inconsistent")
    if require_literal_t4_authenticated:
        from saes.depthsplat_literal_t4_acid_calibration import (
            verified_literal_t4_materializer_guard_projection,
        )

        value = verified_literal_t4_materializer_guard_projection(value)
    required = {
        "schema_version",
        "frozen_record_kind",
        "frozen_record_sha256",
        "threshold_value",
        "threshold_rule",
        "risk_metric",
        "materialization_profile",
        "route_plan_contract",
        "route_plan_config_sha256",
        "acid_binding_sha256",
        "application_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("DepthSplat selected-anchor LOO frozen guard is invalid")
    if (
        value.get("schema_version") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_GUARD_SCHEMA
        or value.get("risk_metric") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC
        or not isinstance(value.get("frozen_record_kind"), str)
        or not value["frozen_record_kind"]
        or not isinstance(value.get("materialization_profile"), str)
        or value["materialization_profile"]
        not in {
            DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
        }
        or not isinstance(value.get("route_plan_contract"), str)
        or not value["route_plan_contract"]
        or not isinstance(value.get("threshold_rule"), str)
        or not value["threshold_rule"]
        or isinstance(value.get("threshold_value"), bool)
        or not isinstance(value.get("threshold_value"), (int, float))
        or not torch.isfinite(torch.tensor(float(value["threshold_value"])))
        or float(value["threshold_value"]) < 0.0
    ):
        raise ValueError("DepthSplat selected-anchor LOO frozen guard changed")
    if (
        value["materialization_profile"]
        == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        and not require_literal_t4_authenticated
        and not allow_serialized_literal_t4_projection
    ):
        raise ValueError(
            "DepthSplat literal T=4 LOO requires an authenticated V16T4 guard"
        )
    for name in (
        "frozen_record_sha256",
        "route_plan_config_sha256",
        "acid_binding_sha256",
        "application_sha256",
    ):
        _require_sha256(value.get(name), label=f"selected-anchor LOO guard {name}")
    return {
        "schema_version": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_GUARD_SCHEMA,
        "frozen_record_kind": value["frozen_record_kind"],
        "frozen_record_sha256": value["frozen_record_sha256"],
        "threshold_value": float(value["threshold_value"]),
        "threshold_rule": value["threshold_rule"],
        "risk_metric": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
        "materialization_profile": value["materialization_profile"],
        "route_plan_contract": value["route_plan_contract"],
        "route_plan_config_sha256": value["route_plan_config_sha256"],
        "acid_binding_sha256": value["acid_binding_sha256"],
        "application_sha256": value["application_sha256"],
    }


def _mixture_kernel_closure_frozen_guard(
    value: Any | None,
    *,
    require_authenticated: bool = False,
    allow_serialized_projection: bool = False,
) -> dict[str, Any] | None:
    """Validate the ACID-issued kernel-risk capability for the v3 profile.

    A positive closure threshold may enter initial materialization only through
    the opaque capability produced by the live ACID 24/8 record loader.  The
    serialized projection is accepted solely when replaying an already sealed
    preflight/final-route trace.
    """

    if value is None:
        return None
    if require_authenticated and allow_serialized_projection:
        raise ValueError("DepthSplat kernel-risk guard validation mode is inconsistent")
    from saes.depthsplat_mixture_kernel_acid_calibration import (
        KERNEL_RISK_GUARD_SCHEMA,
        KERNEL_RISK_KIND,
        KERNEL_RISK_METRIC,
        KERNEL_RISK_THRESHOLD_RULE,
        verified_mixture_kernel_risk_guard_projection,
    )

    if require_authenticated:
        value = verified_mixture_kernel_risk_guard_projection(value)
    required = {
        "schema_version",
        "frozen_record_kind",
        "frozen_record_sha256",
        "threshold_value",
        "threshold_rule",
        "risk_metric",
        "materialization_profile",
        "route_plan_contract",
        "route_plan_config_sha256",
        "kernel_closure_schema_version",
        "kernel_closure_kind",
        "kernel_closure_policy",
        "acid_binding_sha256",
        "application_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("DepthSplat kernel-risk frozen guard is invalid")
    threshold = value.get("threshold_value")
    if (
        value.get("schema_version") != KERNEL_RISK_GUARD_SCHEMA
        or value.get("frozen_record_kind") != KERNEL_RISK_KIND
        or value.get("threshold_rule") != KERNEL_RISK_THRESHOLD_RULE
        or value.get("risk_metric") != KERNEL_RISK_METRIC
        or value.get("materialization_profile")
        != DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
        or value.get("route_plan_contract")
        != DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT
        or value.get("kernel_closure_schema_version")
        != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_SCHEMA_VERSION
        or value.get("kernel_closure_kind") != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND
        or value.get("kernel_closure_policy")
        != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not torch.isfinite(torch.tensor(float(threshold)))
        or float(threshold) <= 0.0
    ):
        raise ValueError("DepthSplat kernel-risk frozen guard changed")
    for name in (
        "frozen_record_sha256",
        "route_plan_config_sha256",
        "acid_binding_sha256",
        "application_sha256",
    ):
        _require_sha256(value.get(name), label=f"kernel-risk guard {name}")
    return {
        "schema_version": KERNEL_RISK_GUARD_SCHEMA,
        "frozen_record_kind": KERNEL_RISK_KIND,
        "frozen_record_sha256": value["frozen_record_sha256"],
        "threshold_value": float(threshold),
        "threshold_rule": KERNEL_RISK_THRESHOLD_RULE,
        "risk_metric": KERNEL_RISK_METRIC,
        "materialization_profile": (
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
        ),
        "route_plan_contract": DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
        "route_plan_config_sha256": value["route_plan_config_sha256"],
        "kernel_closure_schema_version": DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_SCHEMA_VERSION,
        "kernel_closure_kind": DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND,
        "kernel_closure_policy": DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY,
        "acid_binding_sha256": value["acid_binding_sha256"],
        "application_sha256": value["application_sha256"],
    }


def _selected_anchor_opacity_endpoint_count(
    *,
    packed: DepthSplatPackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    level: str,
    semantics: str,
    plan_record: Mapping[str, Any],
) -> int:
    """Read only selected anchor opacities to preserve a native Full fallback."""

    anchors = _route_anchor_positions(plan_record, level=level, semantics=semantics)
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=4,
        positions=anchors,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("DepthSplat selected-anchor LOO lacks a retained anchor")
    opacities = packed.opacities[[slot_to_index[slot] for slot in anchor_slots]]
    if (
        not bool(torch.isfinite(opacities).all())
        or bool((opacities < 0.0).any())
        or bool((opacities > 1.0).any())
    ):
        raise ValueError("DepthSplat selected-anchor opacity is invalid")
    return int((opacities >= 1.0).sum().item())


def depthsplat_selected_anchor_attribute_loo_certificate(
    *,
    packed: DepthSplatPackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    level: str,
    semantics: str,
    plan_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure selected-anchor RGB-SH/opacity replay risk without omitted labels.

    Each retained L0/L1 anchor is held out in turn and reconstructed from the
    remaining retained anchors using the same spatial field used to construct
    virtual attributes.  The label is therefore a native selected Adapter
    attribute, while a real omitted position is never opened or inspected.
    """

    if level not in {"L0", "L1"}:
        raise ValueError("DepthSplat selected-anchor LOO requires a compact level")
    anchors = _route_anchor_positions(plan_record, level=level, semantics=semantics)
    if len(anchors) < 4 or len(set(anchors)) != len(anchors):
        raise ValueError("DepthSplat selected-anchor LOO layout is invalid")
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=4,
        positions=anchors,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("DepthSplat selected-anchor LOO lacks a retained anchor")
    anchor_indices = [slot_to_index[slot] for slot in anchor_slots]
    harmonics = packed.harmonics[anchor_indices]
    opacities = packed.opacities[anchor_indices]
    if not bool(torch.isfinite(harmonics).all()):
        raise ValueError("DepthSplat selected-anchor harmonics are non-finite")
    logits = _selected_anchor_opacity_logits(opacities)
    global_positions = [
        (tile_y * 4 + row, tile_x * 4 + column) for row, column in anchors
    ]
    risks: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for held_offset, held_out in enumerate(anchors):
        source_offsets = [
            offset for offset in range(len(anchors)) if offset != held_offset
        ]
        source_positions = [global_positions[offset] for offset in source_offsets]
        target_position = [global_positions[held_offset]]
        spatial = _spatial_weights(
            target_position,
            source_positions,
            device=harmonics.device,
            dtype=harmonics.dtype,
        ).reshape(-1)
        source_harmonics = harmonics[source_offsets]
        source_logits = logits[source_offsets]
        predicted_harmonics = torch.einsum("n,ncd->cd", spatial, source_harmonics)
        predicted_logit = torch.dot(spatial, source_logits)
        harmonic_error = (
            (predicted_harmonics - harmonics[held_offset]).norm()
            / _selected_anchor_support_scale(source_harmonics)
        )
        opacity_error = (
            (predicted_logit - logits[held_offset]).abs()
            / _selected_anchor_support_scale(source_logits.reshape(-1, 1))
        )
        risk = torch.maximum(harmonic_error, opacity_error)
        if not bool(torch.isfinite(risk)) or bool(risk < 0.0):
            raise ValueError("DepthSplat selected-anchor LOO risk is invalid")
        risks.append(risk)
        records.append(
            {
                "held_out_local_position": [int(held_out[0]), int(held_out[1])],
                "harmonic_relative_error": float(harmonic_error.item()),
                "opacity_logit_relative_error": float(opacity_error.item()),
                "risk": float(risk.item()),
            }
        )
    values = torch.stack(risks).to(dtype=torch.float32)
    q75_risk = torch.quantile(values, 0.75)
    maximum_held_out_risk = values.max()
    if (
        not bool(torch.isfinite(q75_risk))
        or bool(q75_risk < 0.0)
        or not bool(torch.isfinite(maximum_held_out_risk))
        or bool(maximum_held_out_risk < 0.0)
    ):
        raise ValueError("DepthSplat selected-anchor LOO quantile is invalid")
    return {
        "checked": True,
        "scorable": True,
        "status": "scored",
        "reason": None,
        "certificate": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE,
        "policy": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY,
        "risk_metric": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
        "level": level,
        "anchor_count": len(anchors),
        "held_out_anchor_count": len(records),
        "held_out_anchor_records": records,
        "q75_risk": float(q75_risk.item()),
        "maximum_held_out_risk": float(maximum_held_out_risk.item()),
        "selected_anchor_native_attribute_label_reads": len(records),
        "selected_anchor_native_attribute_endpoint_reads": 0,
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_direct_deletion": False,
    }


def _selected_anchor_attribute_loo_unscorable_certificate(
    *,
    level: str,
    semantics: str,
    plan_record: Mapping[str, Any],
    endpoint_anchor_count: int,
) -> dict[str, Any]:
    """Record a native endpoint Full promotion without treating it as a failure.

    Exact opacity-one anchors cannot be replayed in logit space.  They are not
    omitted samples, so the correct source-faithful response is to retain the
    Full tile and record that this LOO observation was deliberately unscorable.
    """

    anchors = _route_anchor_positions(plan_record, level=level, semantics=semantics)
    if endpoint_anchor_count < 1 or endpoint_anchor_count > len(anchors):
        raise ValueError("DepthSplat selected-anchor endpoint evidence is invalid")
    return {
        "checked": False,
        "scorable": False,
        "status": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_ENDPOINT_UNSCORABLE,
        "reason": NATIVE_OPACITY_ENDPOINT_FULL_REASON,
        "certificate": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE,
        "policy": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY,
        "risk_metric": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
        "level": level,
        "anchor_count": len(anchors),
        "held_out_anchor_count": 0,
        "held_out_anchor_records": [],
        "q75_risk": None,
        "maximum_held_out_risk": None,
        "selected_anchor_native_attribute_label_reads": 0,
        "selected_anchor_native_attribute_endpoint_reads": endpoint_anchor_count,
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_direct_deletion": False,
    }


def _finite_selected_anchor_attribute_loo_summary(
    values: list[float],
) -> dict[str, float | int | None]:
    """Use a JSON-safe deterministic quantile summary for audit evidence."""

    if any(not isinstance(value, float) or not torch.isfinite(torch.tensor(value)) for value in values):
        raise ValueError("DepthSplat selected-anchor LOO aggregate contains an invalid risk")
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "maximum": None,
        }

    def quantile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(ordered),
        "minimum": quantile(0.0),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "maximum": quantile(1.0),
    }


def _selected_anchor_attribute_loo_aggregate(
    tile_trace: list[dict[str, Any]],
    *,
    mode: str,
    frozen_guard: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the serializable selected-only LOO aggregate from the tile trace."""

    if mode not in {"collect_only", "frozen_v16_guard"}:
        raise ValueError("DepthSplat selected-anchor LOO aggregate mode is invalid")
    if (mode == "frozen_v16_guard") != (frozen_guard is not None):
        raise ValueError("DepthSplat selected-anchor LOO aggregate guard mode drifted")
    records: list[dict[str, Any]] = []
    attempted_tiles = 0
    for entry in tile_trace:
        if not isinstance(entry, Mapping):
            raise ValueError("DepthSplat selected-anchor LOO tile trace is invalid")
        if entry.get("attempted") is not True:
            continue
        attempted_tiles += 1
        loo = entry.get("selected_anchor_attribute_loo")
        if not isinstance(loo, Mapping):
            raise ValueError("DepthSplat selected-anchor LOO collection is incomplete")
        record = {
            "view": entry.get("view"),
            "tile_y": entry.get("tile_y"),
            "tile_x": entry.get("tile_x"),
            "planned_route": entry.get("planned_route"),
            "accepted": entry.get("accepted"),
            "reason": entry.get("reason"),
            "selected_anchor_attribute_loo": dict(loo),
        }
        if (
            any(
                isinstance(record[name], bool) or not isinstance(record[name], int)
                for name in ("view", "tile_y", "tile_x")
            )
            or record["planned_route"] not in {"L0", "L1"}
            or not isinstance(record["accepted"], bool)
            or not isinstance(record["reason"], str)
        ):
            raise ValueError("DepthSplat selected-anchor LOO tile identity is invalid")
        records.append(record)
    if attempted_tiles != len(records):
        raise ValueError("DepthSplat selected-anchor LOO tile count drifted")

    scorable = [
        record["selected_anchor_attribute_loo"]
        for record in records
        if record["selected_anchor_attribute_loo"].get("scorable") is True
    ]
    unscorable = [
        record["selected_anchor_attribute_loo"]
        for record in records
        if record["selected_anchor_attribute_loo"].get("scorable") is False
    ]
    if len(scorable) + len(unscorable) != len(records):
        raise ValueError("DepthSplat selected-anchor LOO scoring state is invalid")
    maximum_risks: list[float] = []
    q75_risks: list[float] = []
    label_reads = 0
    endpoint_reads = 0
    for record in scorable:
        if (
            record.get("checked") is not True
            or record.get("status") != "scored"
            or record.get("certificate") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE
            or record.get("policy") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY
            or record.get("risk_metric") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC
            or record.get("source_nonprobe_s3_attribute_reads") != 0
            or record.get("nonzero_direct_deletion") is not False
            or not isinstance(record.get("held_out_anchor_records"), list)
            or record.get("held_out_anchor_count")
            != len(record["held_out_anchor_records"])
            or record.get("selected_anchor_native_attribute_label_reads")
            != record.get("held_out_anchor_count")
            or record.get("selected_anchor_native_attribute_endpoint_reads") != 0
        ):
            raise ValueError("DepthSplat selected-anchor LOO scored record is invalid")
        maximum = record.get("maximum_held_out_risk")
        q75 = record.get("q75_risk")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not torch.isfinite(torch.tensor(float(maximum)))
            or float(maximum) < 0.0
            or isinstance(q75, bool)
            or not isinstance(q75, (int, float))
            or not torch.isfinite(torch.tensor(float(q75)))
            or float(q75) < 0.0
            or float(q75) > float(maximum)
        ):
            raise ValueError("DepthSplat selected-anchor LOO risk is invalid")
        maximum_risks.append(float(maximum))
        q75_risks.append(float(q75))
        label_reads += int(record["selected_anchor_native_attribute_label_reads"])
    for record in unscorable:
        if (
            record.get("checked") is not False
            or record.get("status") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_ENDPOINT_UNSCORABLE
            or record.get("reason") != NATIVE_OPACITY_ENDPOINT_FULL_REASON
            or record.get("certificate") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE
            or record.get("policy") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY
            or record.get("risk_metric") != DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC
            or record.get("held_out_anchor_count") != 0
            or record.get("held_out_anchor_records") != []
            or record.get("q75_risk") is not None
            or record.get("maximum_held_out_risk") is not None
            or record.get("selected_anchor_native_attribute_label_reads") != 0
            or isinstance(record.get("selected_anchor_native_attribute_endpoint_reads"), bool)
            or not isinstance(record.get("selected_anchor_native_attribute_endpoint_reads"), int)
            or record["selected_anchor_native_attribute_endpoint_reads"] < 1
            or record.get("source_nonprobe_s3_attribute_reads") != 0
            or record.get("nonzero_direct_deletion") is not False
        ):
            raise ValueError("DepthSplat selected-anchor LOO endpoint record is invalid")
        endpoint_reads += int(record["selected_anchor_native_attribute_endpoint_reads"])
    if any(record["selected_anchor_attribute_loo"].get("source_nonprobe_s3_attribute_reads") != 0 for record in records):
        raise ValueError("DepthSplat selected-anchor LOO opened a nonprobe attribute")
    if any(
        record["selected_anchor_attribute_loo"].get("action") not in {
            "observed_only",
            "retain_compact",
            "promote_full",
            "promote_full_unscorable",
        }
        for record in records
    ):
        raise ValueError("DepthSplat selected-anchor LOO action is invalid")
    if mode == "collect_only" and any(
        record["selected_anchor_attribute_loo"].get("action")
        not in {"observed_only", "promote_full_unscorable"}
        for record in records
    ):
        raise ValueError("DepthSplat selected-anchor LOO collection action drifted")
    if mode == "frozen_v16_guard" and any(
        record["selected_anchor_attribute_loo"].get("action") == "observed_only"
        for record in records
    ):
        raise ValueError("DepthSplat selected-anchor LOO frozen guard was bypassed")

    return {
        "schema_version": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_AGGREGATE_SCHEMA,
        "certificate": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE,
        "policy": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY,
        "risk_metric": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
        "mode": mode,
        "frozen_guard": dict(frozen_guard) if frozen_guard is not None else None,
        "preflight_tile_trace_sha256": _canonical_sha256(tile_trace),
        "candidate_tile_count": attempted_tiles,
        "recorded_tile_count": len(records),
        "tile_records_sha256": _canonical_sha256(records),
        "scorable_tile_count": len(scorable),
        "unscorable_promoted_full_tile_count": len(unscorable),
        "accepted_compact_tile_count": sum(record["accepted"] for record in records),
        "guard_promoted_full_tile_count": sum(
            record["selected_anchor_attribute_loo"].get("action") == "promote_full"
            for record in records
        ),
        "maximum_held_out_risks": maximum_risks,
        "maximum_held_out_risk_summary": _finite_selected_anchor_attribute_loo_summary(
            maximum_risks
        ),
        "q75_risks": q75_risks,
        "q75_risk_summary": _finite_selected_anchor_attribute_loo_summary(q75_risks),
        "selected_anchor_native_attribute_label_reads": label_reads,
        "selected_anchor_native_attribute_endpoint_reads": endpoint_reads,
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_direct_deletion": False,
    }


def _validate_selected_anchor_attribute_loo_aggregate(
    *,
    events: Mapping[str, Any],
    tile_trace: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    """Rebuild the aggregate so a route cannot consume a tampered LOO trace."""

    execution_profile = events.get("execution_profile")
    frozen_guard = _selected_anchor_attribute_loo_frozen_guard(
        events.get("selected_anchor_attribute_loo_frozen_guard"),
        allow_serialized_literal_t4_projection=(
            execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        ),
    )
    collect_only = events.get("selected_anchor_attribute_loo_collect_only")
    guard_enabled = events.get("selected_anchor_attribute_loo_guard")
    aggregate = events.get("selected_anchor_attribute_loo_aggregate")
    if not isinstance(collect_only, bool) or not isinstance(guard_enabled, bool):
        raise ValueError("DepthSplat selected-anchor LOO event mode is invalid")
    if guard_enabled != (frozen_guard is not None) or (collect_only and guard_enabled):
        raise ValueError("DepthSplat selected-anchor LOO event guard drifted")
    if (
        frozen_guard is not None
        and frozen_guard["materialization_profile"] != execution_profile
    ):
        raise ValueError("DepthSplat selected-anchor LOO event profile drifted")
    if not collect_only and not guard_enabled:
        if aggregate is not None:
            raise ValueError("DepthSplat selected-anchor LOO aggregate was not requested")
        return None
    expected = _selected_anchor_attribute_loo_aggregate(
        [dict(record) for record in tile_trace],
        mode="frozen_v16_guard" if guard_enabled else "collect_only",
        frozen_guard=frozen_guard,
    )
    if (
        not isinstance(aggregate, Mapping)
        or dict(aggregate) != expected
        or events.get("selected_anchor_attribute_loo_aggregate_sha256")
        != _canonical_sha256(expected)
    ):
        raise ValueError("DepthSplat selected-anchor LOO aggregate changed")
    return expected


def _feature_continuity(
    feature_map: torch.Tensor,
    source_positions: list[tuple[int, int]],
    target_positions: list[tuple[int, int]],
    spatial_weights: torch.Tensor,
    *,
    maximum_relative_residual: float,
) -> dict[str, float | bool]:
    source = torch.stack([feature_map[:, row, column] for row, column in source_positions])
    target = torch.stack([feature_map[:, row, column] for row, column in target_positions])
    predicted = spatial_weights @ source
    residual = (predicted - target).square().mean(dim=1)
    diameter = (source.unsqueeze(0) - source.unsqueeze(1)).square().mean(dim=2).amax()
    denominator = diameter.clamp_min(torch.finfo(source.dtype).eps)
    relative = residual / denominator
    maximum = float(relative.max().item())
    return {
        "maximum_relative_residual": maximum,
        "source_feature_diameter": float(diameter.item()),
        "passed": bool(torch.isfinite(relative).all()) and maximum <= maximum_relative_residual,
    }


def _project_psd(covariance: torch.Tensor) -> torch.Tensor:
    symmetric = (covariance + covariance.mT) * 0.5
    values, vectors = torch.linalg.eigh(symmetric)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("DepthSplat materializer covariance spectrum is non-finite")
    projected = vectors @ torch.diag(values.clamp_min(1e-8)) @ vectors.mT
    return (projected + projected.mT) * 0.5


def depthsplat_z_depth_world_means(
    coordinates: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    z_depths: torch.Tensor,
    *,
    source_get_world_rays: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Lift image coordinates with the native source z-depth convention."""

    if not callable(source_get_world_rays):
        raise TypeError("DepthSplat materializer requires source get_world_rays")
    if (
        not torch.is_tensor(coordinates)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or extrinsics.shape != (coordinates.shape[0], 4, 4)
        or intrinsics.shape != (coordinates.shape[0], 3, 3)
        or z_depths.reshape(-1).shape[0] != coordinates.shape[0]
    ):
        raise ValueError("DepthSplat z-depth lift inputs are inconsistent")
    z_depths = z_depths.reshape(-1).to(device=coordinates.device, dtype=coordinates.dtype)
    if (
        not bool(torch.isfinite(coordinates).all())
        or not bool(torch.isfinite(extrinsics).all())
        or not bool(torch.isfinite(intrinsics).all())
        or not bool(torch.isfinite(z_depths).all())
        or bool((z_depths <= 0.0).any())
    ):
        raise ValueError("DepthSplat z-depth lift inputs are non-finite")
    origins, directions = source_get_world_rays(coordinates, extrinsics, intrinsics)
    if (
        not torch.is_tensor(origins)
        or not torch.is_tensor(directions)
        or origins.shape != (coordinates.shape[0], 3)
        or directions.shape != origins.shape
        or not bool(torch.isfinite(origins).all())
        or not bool(torch.isfinite(directions).all())
    ):
        raise ValueError("DepthSplat source get_world_rays output is invalid")
    means = origins + directions * z_depths.unsqueeze(1)
    if not bool(torch.isfinite(means).all()):
        raise ValueError("DepthSplat z-depth lift produced non-finite means")
    return means


def _anchor_geometry(
    *,
    packet: DepthSplatSparseRawPacket,
    packed: DepthSplatPackedGaussianAttributes,
    indices: list[int],
    source_positions: list[tuple[int, int]],
    view: int,
    height: int,
    width: int,
    source_grid: torch.Tensor,
    source_get_world_rays: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    expected_coordinates = _coordinates_from_source_grid(
        packet.raw_head_descriptors[indices],
        source_positions,
        source_grid,
        height=height,
        width=width,
    )
    if not torch.allclose(packet.coordinates[indices], expected_coordinates, rtol=2e-5, atol=2e-5):
        raise ValueError("DepthSplat materializer selected offsets drift from native geometry")
    expected_means = depthsplat_z_depth_world_means(
        packet.coordinates[indices],
        packet.extrinsics[indices],
        packet.intrinsics[indices],
        packet.depths[indices],
        source_get_world_rays=source_get_world_rays,
    )
    if not torch.allclose(packed.means[indices], expected_means, rtol=5e-5, atol=5e-5):
        raise ValueError("DepthSplat materializer selected means drift from native z-depth geometry")
    centres = _pixel_centres(
        source_positions,
        height=height,
        width=width,
        device=packet.coordinates.device,
        dtype=packet.coordinates.dtype,
    )
    offsets = packet.coordinates[indices] - centres
    limits = torch.tensor((0.5 / width, 0.5 / height), device=offsets.device, dtype=offsets.dtype)
    if not bool(torch.isfinite(offsets).all()) or bool((offsets.abs() > limits + 2e-5).any()):
        raise ValueError("DepthSplat materializer selected native offset is out of range")
    return offsets


def _coverage_closed_covariance(
    covariance: torch.Tensor,
    mean: torch.Tensor,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    *,
    maximum_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Close a merged 3D covariance over every virtual 2-sigma ellipsoid.

    Point-centre containment is insufficient: a virtual Gaussian's own
    covariance can extend outside the merged support even when its centre is
    close.  In the merged covariance metric, the sufficient containment bound
    is ``||delta|| + radius * sqrt(lambda_max(virtual_covariance)) <= radius``.
    """

    if (
        covariance.shape != (3, 3)
        or mean.shape != (3,)
        or virtual_means.ndim != 2
        or virtual_means.shape[1] != 3
        or virtual_covariances.shape != (virtual_means.shape[0], 3, 3)
    ):
        raise ValueError("DepthSplat materializer coverage support inputs are invalid")
    covariance = _project_psd(covariance)
    if virtual_means.numel() == 0:
        return covariance, {
            "maximum_containment_lhs_before_scale": 0.0,
            "maximum_containment_lhs_after_scale": 0.0,
            "maximum_whitened_virtual_covariance_eigenvalue": 0.0,
            "containment_radius": 2.0,
            "moment_covariance_scale": 1.0,
        }
    try:
        cholesky = torch.linalg.cholesky(covariance)
        deltas = virtual_means - mean.unsqueeze(0)
        whitened_deltas = torch.linalg.solve_triangular(
            cholesky, deltas.mT, upper=False
        ).mT
        containers = cholesky.unsqueeze(0).expand(virtual_means.shape[0], -1, -1)
        projected_virtual_covariances = torch.stack(
            [_project_psd(value) for value in virtual_covariances]
        )
        left = torch.linalg.solve_triangular(
            containers, projected_virtual_covariances, upper=False
        )
        whitened_covariances = torch.linalg.solve_triangular(
            containers, left.mT, upper=False
        ).mT
        eigenvalues = torch.linalg.eigvalsh(
            (whitened_covariances + whitened_covariances.mT) * 0.5
        )
    except RuntimeError as error:
        raise ValueError("DepthSplat materializer coverage containment failed") from error
    mahalanobis = whitened_deltas.square().sum(dim=1).clamp_min(0.0).sqrt()
    maximum_eigenvalues = eigenvalues.amax(dim=1).clamp_min(0.0)
    radius = 2.0
    lhs = mahalanobis + radius * maximum_eigenvalues.sqrt()
    if (
        not bool(torch.isfinite(lhs).all())
        or not bool(torch.isfinite(maximum_eigenvalues).all())
    ):
        raise ValueError("DepthSplat materializer coverage containment is non-finite")
    scale = max(1.0, float((lhs.max() / radius).square().item()))
    if scale > maximum_scale:
        raise ValueError("DepthSplat materializer coverage requires excessive covariance expansion")
    covariance = covariance * scale
    after_scale = lhs / scale**0.5
    if bool((after_scale > radius + 1e-5).any()):
        raise RuntimeError("DepthSplat materializer coverage closure did not contain support")
    return covariance, {
        "maximum_containment_lhs_before_scale": float(lhs.max().item()),
        "maximum_containment_lhs_after_scale": float(after_scale.max().item()),
        "maximum_whitened_virtual_covariance_eigenvalue": float(
            maximum_eigenvalues.max().item()
        ),
        "containment_radius": radius,
        "moment_covariance_scale": scale,
    }


def _route_anchor_positions(
    record: Mapping[str, Any], *, level: str, semantics: str
) -> list[tuple[int, int]]:
    if level == "L0":
        return compute_probe_positions(4)
    if level == "L1":
        return l1_local_positions_for_tile(record, tile_size=4, l1_anchor_semantics=semantics)
    if level == "Full":
        return _full_positions(4)
    raise ValueError("DepthSplat materializer route level is invalid")


def _update_binding(
    update_slots: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    harmonics: torch.Tensor,
    opacities: torch.Tensor,
) -> dict[str, str]:
    """Digest every materialized value before a later packet can consume it."""

    return {
        "slots_sha256": _tensor_sha256(update_slots),
        "means_sha256": _tensor_sha256(means),
        "covariances_sha256": _tensor_sha256(covariances),
        "harmonics_sha256": _tensor_sha256(harmonics),
        "opacities_sha256": _tensor_sha256(opacities),
    }


def _coverage_certificate_payload(
    *,
    update_slots: torch.Tensor,
    update_binding: Mapping[str, str],
    per_update: list[dict[str, Any]],
    tile_trace_sha256: str,
    maximum_covariance_scale: float,
) -> dict[str, Any]:
    """Bind shape-aware support checks to the exact ordered update packet."""

    if (
        not torch.is_tensor(update_slots)
        or update_slots.ndim != 1
        or update_slots.dtype != torch.int64
        or not isinstance(update_binding, Mapping)
        or not isinstance(tile_trace_sha256, str)
        or not isinstance(maximum_covariance_scale, float)
    ):
        raise ValueError("DepthSplat coverage certificate inputs are invalid")
    slots = update_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(per_update) != len(slots):
        raise ValueError("DepthSplat coverage certificate update count is incomplete")
    normalized_rows: list[dict[str, Any]] = []
    for update_index, (slot, row) in enumerate(zip(slots, per_update)):
        if not isinstance(row, Mapping):
            raise ValueError("DepthSplat coverage certificate row is invalid")
        normalized = dict(row)
        if (
            normalized.get("update_dense_slot") != int(slot)
            or normalized.get("update_index") != update_index
            or normalized.get("containment_radius") != 2.0
            or normalized.get("shape_aware_virtual_2sigma_support") is not True
        ):
            raise ValueError("DepthSplat coverage certificate slot binding changed")
        for key in (
            "maximum_containment_lhs_before_scale",
            "maximum_containment_lhs_after_scale",
            "maximum_whitened_virtual_covariance_eigenvalue",
            "moment_covariance_scale",
        ):
            value = normalized.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not torch.isfinite(torch.tensor(float(value)))
                or float(value) < 0.0
            ):
                raise ValueError("DepthSplat coverage certificate value is invalid")
        if (
            float(normalized["maximum_containment_lhs_after_scale"]) > 2.0 + 1e-5
            or float(normalized["moment_covariance_scale"])
            > maximum_covariance_scale + 1e-5
        ):
            raise ValueError("DepthSplat coverage certificate does not contain support")
        normalized_rows.append(normalized)
    return {
        "schema": DEPTHSPLAT_COVERAGE_CERTIFICATE,
        "geometry": "world-3d-whitened-ellipsoid-triangle-bound-v1",
        "containment_radius": 2.0,
        "after_scale_lhs_upper_bound": 2.0,
        "maximum_covariance_scale": maximum_covariance_scale,
        "shape_aware_virtual_2sigma_support": True,
        "tile_trace_sha256": tile_trace_sha256,
        "update_binding": dict(update_binding),
        "update_slots": [int(slot) for slot in slots],
        "per_update": normalized_rows,
    }


def _literal_moment_merge_certificate_payload(
    *,
    update_slots: torch.Tensor,
    update_binding: Mapping[str, str],
    update_means: torch.Tensor,
    update_covariances: torch.Tensor,
    per_update: list[dict[str, Any]],
    tile_trace_sha256: str,
) -> dict[str, Any]:
    """Bind literal finite-PSD, fixed-scale moment results to their packet."""

    if (
        not torch.is_tensor(update_slots)
        or update_slots.ndim != 1
        or update_slots.dtype != torch.int64
        or not isinstance(update_binding, Mapping)
        or not torch.is_tensor(update_means)
        or update_means.shape != (int(update_slots.numel()), 3)
        or not torch.is_tensor(update_covariances)
        or update_covariances.shape != (int(update_slots.numel()), 3, 3)
        or not isinstance(tile_trace_sha256, str)
    ):
        raise ValueError("DepthSplat literal moment certificate inputs are invalid")
    if (
        not bool(torch.isfinite(update_means).all())
        or not bool(torch.isfinite(update_covariances).all())
    ):
        raise ValueError("DepthSplat literal moment certificate is non-finite")
    symmetric_covariances = (update_covariances + update_covariances.mT) * 0.5
    if bool((torch.linalg.eigvalsh(symmetric_covariances) < -1e-6).any()):
        raise ValueError("DepthSplat literal moment certificate is not PSD")
    slots = update_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(per_update) != len(slots):
        raise ValueError("DepthSplat literal moment certificate update count is incomplete")
    normalized_rows: list[dict[str, Any]] = []
    for update_index, (slot, row) in enumerate(zip(slots, per_update)):
        if not isinstance(row, Mapping):
            raise ValueError("DepthSplat literal moment certificate row is invalid")
        virtual_count = row.get("virtual_count")
        if (
            row.get("update_dense_slot") != int(slot)
            or row.get("update_index") != update_index
            or isinstance(virtual_count, bool)
            or not isinstance(virtual_count, int)
            or virtual_count < 0
            or row.get("finite_psd_moment_merge") is not True
            or row.get("moment_covariance_scale") != 1.0
            or row.get("support_containment_guard") is not False
        ):
            raise ValueError("DepthSplat literal moment certificate slot binding changed")
        normalized_rows.append(
            {
                "update_dense_slot": int(slot),
                "update_index": update_index,
                "virtual_count": virtual_count,
                "finite_psd_moment_merge": True,
                "moment_covariance_scale": 1.0,
                "support_containment_guard": False,
            }
        )
    return {
        "schema": DEPTHSPLAT_LITERAL_PAPER_T4_MOMENT_CERTIFICATE,
        "geometry": "gaussian-parameter-space-first-second-moment-v1",
        "finite_psd_moment_merge": True,
        "fixed_moment_covariance_scale": 1.0,
        "support_containment_guard": False,
        "tile_trace_sha256": tile_trace_sha256,
        "update_binding": dict(update_binding),
        "update_slots": [int(slot) for slot in slots],
        "per_update": normalized_rows,
    }


def _coverage_enriched_moment_certificate_payload(
    *,
    update_slots: torch.Tensor,
    update_binding: Mapping[str, str],
    update_means: torch.Tensor,
    update_covariances: torch.Tensor,
    per_update: list[dict[str, Any]],
    tile_trace_sha256: str,
) -> dict[str, Any]:
    """Bind fixed-scale owner-anchor projected support to accepted updates."""

    if (
        not torch.is_tensor(update_slots)
        or update_slots.ndim != 1
        or update_slots.dtype != torch.int64
        or not isinstance(update_binding, Mapping)
        or not torch.is_tensor(update_means)
        or update_means.shape != (int(update_slots.numel()), 3)
        or not torch.is_tensor(update_covariances)
        or update_covariances.shape != (int(update_slots.numel()), 3, 3)
        or not isinstance(tile_trace_sha256, str)
    ):
        raise ValueError("DepthSplat coverage-enriched certificate inputs are invalid")
    if (
        not bool(torch.isfinite(update_means).all())
        or not bool(torch.isfinite(update_covariances).all())
        or bool(
            (
                torch.linalg.eigvalsh(
                    (update_covariances + update_covariances.mT) * 0.5
                )
                < -1e-6
            ).any()
        )
    ):
        raise ValueError("DepthSplat coverage-enriched certificate is not finite PSD")
    slots = update_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(per_update) != len(slots):
        raise ValueError("DepthSplat coverage-enriched certificate update count is incomplete")
    normalized_rows: list[dict[str, Any]] = []
    for update_index, (slot, row) in enumerate(zip(slots, per_update)):
        if not isinstance(row, Mapping):
            raise ValueError("DepthSplat coverage-enriched certificate row is invalid")
        virtual_count = row.get("virtual_count")
        owner = row.get("owner_support")
        if (
            row.get("update_dense_slot") != int(slot)
            or row.get("update_index") != update_index
            or isinstance(virtual_count, bool)
            or not isinstance(virtual_count, int)
            or virtual_count < 0
            or row.get("finite_psd_moment_merge") is not True
            or row.get("moment_covariance_scale") != 1.0
            or row.get("support_containment_guard") is not True
            or row.get("owner_support_schema_version")
            != DEPTHSPLAT_OWNER_COVERAGE_AUDIT_SCHEMA_VERSION
            or row.get("owner_support_kind") != DEPTHSPLAT_OWNER_COVERAGE_AUDIT_KIND
            or row.get("owner_assignment_policy")
            != DEPTHSPLAT_OWNER_ASSIGNMENT_POLICY
        ):
            raise ValueError("DepthSplat coverage-enriched certificate owner binding changed")
        assignment_sha256 = _require_sha256(
            row.get("virtual_owner_indices_sha256"),
            label="coverage-enriched virtual owner assignment",
        )
        assignment_counts_sha256 = _require_sha256(
            row.get("owner_assignment_counts_sha256"),
            label="coverage-enriched owner assignment counts",
        )
        if isinstance(owner, Mapping):
            owner_index = owner.get("owner_index")
            owned_virtual_count = owner.get("assigned_virtual_count")
            owner_passed = owner.get("passed")
            coverage = owner.get("coverage")
            if (
                isinstance(owner_index, bool)
                or not isinstance(owner_index, int)
                or owner_index < 0
                or isinstance(owned_virtual_count, bool)
                or not isinstance(owned_virtual_count, int)
                or owned_virtual_count < 0
                or owner.get("source_anchor_count") != 1
                or owner.get("source_anchor_support_included") is not True
                or owner.get("source_anchor_support_passed") is not True
                or owner_passed is not True
                or not isinstance(coverage, Mapping)
            ):
                raise ValueError("DepthSplat coverage-enriched owner evidence changed")
            if coverage.get("valid") is True:
                if coverage.get("hole_count") != 0:
                    raise ValueError("DepthSplat coverage-enriched owner support is incomplete")
            elif not (
                coverage.get("valid") is False
                and coverage.get("reason") == "zero-dense-optical-mass"
            ):
                raise ValueError("DepthSplat coverage-enriched owner support is incomplete")
        else:
            owner_index = row.get("owner_index")
            owned_virtual_count = row.get("owned_virtual_count")
            if (
                isinstance(owner_index, bool)
                or not isinstance(owner_index, int)
                or owner_index < 0
                or isinstance(owned_virtual_count, bool)
                or not isinstance(owned_virtual_count, int)
                or owned_virtual_count < 0
                or row.get("source_anchor_count") != 1
                or row.get("source_anchor_support_included") is not True
                or row.get("source_anchor_support_passed") is not True
                or row.get("owner_support_passed") is not True
            ):
                raise ValueError("DepthSplat coverage-enriched owner certificate changed")
        normalized_rows.append(
            {
                "update_dense_slot": int(slot),
                "update_index": update_index,
                "virtual_count": virtual_count,
                "finite_psd_moment_merge": True,
                "moment_covariance_scale": 1.0,
                "support_containment_guard": True,
                "owner_index": int(owner_index),
                "owned_virtual_count": int(owned_virtual_count),
                "source_anchor_count": 1,
                "source_anchor_support_included": True,
                "source_anchor_support_passed": True,
                "owner_support_schema_version": DEPTHSPLAT_OWNER_COVERAGE_AUDIT_SCHEMA_VERSION,
                "owner_support_kind": DEPTHSPLAT_OWNER_COVERAGE_AUDIT_KIND,
                "owner_assignment_policy": DEPTHSPLAT_OWNER_ASSIGNMENT_POLICY,
                "virtual_owner_indices_sha256": assignment_sha256,
                "owner_assignment_counts_sha256": assignment_counts_sha256,
                "owner_support_passed": True,
            }
        )
    return {
        "schema": DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE,
        "geometry": "source-camera-owner-anchor-and-virtual-projected-2sigma-v1",
        "finite_psd_moment_merge": True,
        "fixed_moment_covariance_scale": 1.0,
        "support_containment_guard": True,
        "tile_trace_sha256": tile_trace_sha256,
        "update_binding": dict(update_binding),
        "update_slots": [int(slot) for slot in slots],
        "per_update": normalized_rows,
    }


def _support_basis_moment_certificate_payload(
    *,
    update_slots: torch.Tensor,
    update_binding: Mapping[str, str],
    update_means: torch.Tensor,
    update_covariances: torch.Tensor,
    per_update: list[dict[str, Any]],
    tile_trace: Any,
    tile_trace_sha256: str,
) -> dict[str, Any]:
    """Bind same-tile composed-ledger support evidence to accepted updates."""

    if (
        not torch.is_tensor(update_slots)
        or update_slots.ndim != 1
        or update_slots.dtype != torch.int64
        or not isinstance(update_binding, Mapping)
        or not torch.is_tensor(update_means)
        or update_means.shape != (int(update_slots.numel()), 3)
        or not torch.is_tensor(update_covariances)
        or update_covariances.shape != (int(update_slots.numel()), 3, 3)
        or not isinstance(tile_trace_sha256, str)
        or not isinstance(tile_trace, (list, tuple))
    ):
        raise ValueError("DepthSplat support-basis certificate inputs are invalid")
    if (
        not bool(torch.isfinite(update_means).all())
        or not bool(torch.isfinite(update_covariances).all())
        or bool(
            (
                torch.linalg.eigvalsh(
                    (update_covariances + update_covariances.mT) * 0.5
                )
                < -1e-6
            ).any()
        )
    ):
        raise ValueError("DepthSplat support-basis certificate is not finite PSD")
    slots = update_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(per_update) != len(slots):
        raise ValueError("DepthSplat support-basis certificate update count is incomplete")

    tile_support_by_slot: dict[int, tuple[Mapping[str, Any], str]] = {}
    for tile_record in tile_trace:
        if not isinstance(tile_record, Mapping):
            raise ValueError("DepthSplat support-basis trace record is invalid")
        coverage = tile_record.get("coverage")
        if not isinstance(coverage, Mapping):
            continue
        support_basis = coverage.get("support_basis")
        support_basis_sha256 = coverage.get("support_basis_sha256")
        coverage_rows = coverage.get("per_update")
        if support_basis is None and support_basis_sha256 is None:
            continue
        if (
            not isinstance(support_basis, Mapping)
            or not isinstance(coverage_rows, list)
            or not isinstance(support_basis_sha256, str)
            or support_basis_sha256 != _canonical_sha256(support_basis)
        ):
            raise ValueError("DepthSplat support-basis trace evidence is invalid")
        for row in coverage_rows:
            if not isinstance(row, Mapping):
                raise ValueError("DepthSplat support-basis trace row is invalid")
            slot = row.get("update_dense_slot")
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot in tile_support_by_slot
            ):
                raise ValueError("DepthSplat support-basis trace slot is invalid")
            tile_support_by_slot[slot] = (support_basis, support_basis_sha256)

    normalized_rows: list[dict[str, Any]] = []
    for update_index, (slot, row) in enumerate(zip(slots, per_update)):
        if not isinstance(row, Mapping):
            raise ValueError("DepthSplat support-basis certificate row is invalid")
        evidence = tile_support_by_slot.pop(int(slot), None)
        if evidence is None:
            raise ValueError("DepthSplat support-basis evidence is missing an accepted update")
        support_basis, support_basis_sha256 = evidence
        virtual_count = row.get("virtual_count")
        binding = support_basis.get("binding")
        source_only = support_basis.get("source_only")
        summary = support_basis.get("summary")
        if (
            row.get("update_dense_slot") != int(slot)
            or row.get("update_index") != update_index
            or isinstance(virtual_count, bool)
            or not isinstance(virtual_count, int)
            or virtual_count < 0
            or row.get("finite_psd_moment_merge") is not True
            or row.get("moment_covariance_scale") != 1.0
            or row.get("support_containment_guard") is not True
            or row.get("support_basis_schema_version")
            != DEPTHSPLAT_SUPPORT_BASIS_AUDIT_SCHEMA_VERSION
            or row.get("support_basis_kind") != DEPTHSPLAT_SUPPORT_BASIS_AUDIT_KIND
            or row.get("support_basis_policy") != DEPTHSPLAT_SUPPORT_BASIS_POLICY
            or row.get("support_basis_sha256") != support_basis_sha256
            or row.get("support_basis_passed") is not True
            or support_basis.get("schema_version")
            != DEPTHSPLAT_SUPPORT_BASIS_AUDIT_SCHEMA_VERSION
            or support_basis.get("kind") != DEPTHSPLAT_SUPPORT_BASIS_AUDIT_KIND
            or support_basis.get("support_basis_policy")
            != DEPTHSPLAT_SUPPORT_BASIS_POLICY
            or support_basis.get("passed") is not True
            or not isinstance(binding, Mapping)
            or not isinstance(source_only, Mapping)
            or not isinstance(summary, Mapping)
            or source_only.get("source_camera_only") is not True
            or source_only.get("target_mapping_present") is not False
            or source_only.get("target_rgb_accessed") is not False
            or source_only.get("target_camera_metadata_accessed") is not False
            or source_only.get("target_index_accessed") is not False
            or source_only.get("same_tile_candidate_basis_only") is not True
            or source_only.get("fixed_covariance_scale") is not True
            or summary.get("hole_count") != 0
            or summary.get("all_active_same_tile_supports_contained") is not True
        ):
            raise ValueError("DepthSplat support-basis certificate binding changed")
        for key in (
            "anchor_dense_slots_sha256",
            "virtual_origin_slots_sha256",
            "virtual_means_sha256",
            "virtual_covariances_sha256",
            "virtual_opacities_sha256",
            "virtual_source_spatial_weights_sha256",
            "bilateral_assignment_weights_sha256",
            "virtual_candidate_basis_mask_sha256",
            "anchor_candidate_basis_mask_sha256",
            "source_to_output_ledger_sha256",
            "anchor_source_means_sha256",
            "anchor_source_covariances_sha256",
            "anchor_source_opacities_sha256",
            "merged_means_sha256",
            "merged_covariances_sha256",
            "merged_opacities_sha256",
            "context_extrinsics_sha256",
            "context_intrinsics_sha256",
        ):
            _require_sha256(binding.get(key), label=f"support-basis {key}")
        normalized_rows.append(
            {
                "update_dense_slot": int(slot),
                "update_index": update_index,
                "virtual_count": virtual_count,
                "finite_psd_moment_merge": True,
                "moment_covariance_scale": 1.0,
                "support_containment_guard": True,
                "support_basis_schema_version": DEPTHSPLAT_SUPPORT_BASIS_AUDIT_SCHEMA_VERSION,
                "support_basis_kind": DEPTHSPLAT_SUPPORT_BASIS_AUDIT_KIND,
                "support_basis_policy": DEPTHSPLAT_SUPPORT_BASIS_POLICY,
                "support_basis_sha256": support_basis_sha256,
                "support_basis_passed": True,
            }
        )
    if tile_support_by_slot:
        raise ValueError("DepthSplat support-basis trace has an unbound update")
    return {
        "schema": DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE,
        "geometry": "source-camera-composed-soft-ledger-same-tile-projected-2sigma-v1",
        "finite_psd_moment_merge": True,
        "fixed_moment_covariance_scale": 1.0,
        "support_containment_guard": True,
        "tile_trace_sha256": tile_trace_sha256,
        "update_binding": dict(update_binding),
        "update_slots": [int(slot) for slot in slots],
        "per_update": normalized_rows,
    }


def _soft_mixture_moment_certificate_payload(
    *,
    update_slots: torch.Tensor,
    update_binding: Mapping[str, str],
    update_means: torch.Tensor,
    update_covariances: torch.Tensor,
    per_update: list[dict[str, Any]],
    tile_trace: Any,
    tile_trace_sha256: str,
) -> dict[str, Any]:
    """Bind source-only soft S/R replay certificates to accepted updates."""

    if (
        not torch.is_tensor(update_slots)
        or update_slots.ndim != 1
        or update_slots.dtype != torch.int64
        or not isinstance(update_binding, Mapping)
        or not torch.is_tensor(update_means)
        or update_means.shape != (int(update_slots.numel()), 3)
        or not torch.is_tensor(update_covariances)
        or update_covariances.shape != (int(update_slots.numel()), 3, 3)
        or not isinstance(tile_trace_sha256, str)
        or not isinstance(tile_trace, (list, tuple))
    ):
        raise ValueError("DepthSplat soft-mixture certificate inputs are invalid")
    if (
        not bool(torch.isfinite(update_means).all())
        or not bool(torch.isfinite(update_covariances).all())
        or bool(
            (
                torch.linalg.eigvalsh(
                    (update_covariances + update_covariances.mT) * 0.5
                )
                < -1e-6
            ).any()
        )
    ):
        raise ValueError("DepthSplat soft-mixture certificate is not finite PSD")
    slots = update_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(per_update) != len(slots):
        raise ValueError("DepthSplat soft-mixture certificate update count is incomplete")

    evidence_by_slot: dict[int, tuple[Mapping[str, Any], str]] = {}
    for tile_record in tile_trace:
        if not isinstance(tile_record, Mapping):
            raise ValueError("DepthSplat soft-mixture trace record is invalid")
        coverage = tile_record.get("coverage")
        certificate = tile_record.get("soft_mixture_certificate")
        certificate_sha256 = tile_record.get("soft_mixture_certificate_sha256")
        certificate_passed = tile_record.get("soft_mixture_certificate_passed")
        if certificate is None and certificate_sha256 is None and certificate_passed is None:
            continue
        if (
            not isinstance(coverage, Mapping)
            or not isinstance(certificate, Mapping)
            or not isinstance(certificate_sha256, str)
            or certificate_sha256 != _canonical_sha256(certificate)
            or certificate_passed is not True
            or certificate.get("schema_version")
            != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION
            or certificate.get("kind") != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_KIND
            or certificate.get("policy") != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_POLICY
            or certificate.get("passed") is not True
            or coverage.get("projected_support_guard") is not False
            or coverage.get("support_containment_guard") is not False
            or not isinstance(coverage.get("per_update"), list)
        ):
            raise ValueError("DepthSplat soft-mixture trace evidence is invalid")
        for row in coverage["per_update"]:
            if not isinstance(row, Mapping):
                raise ValueError("DepthSplat soft-mixture trace row is invalid")
            slot = row.get("update_dense_slot")
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot in evidence_by_slot
            ):
                raise ValueError("DepthSplat soft-mixture trace slot is invalid")
            evidence_by_slot[slot] = (certificate, certificate_sha256)

    normalized_rows: list[dict[str, Any]] = []
    for update_index, (slot, row) in enumerate(zip(slots, per_update)):
        if not isinstance(row, Mapping):
            raise ValueError("DepthSplat soft-mixture certificate row is invalid")
        evidence = evidence_by_slot.pop(int(slot), None)
        if evidence is None:
            raise ValueError("DepthSplat soft-mixture evidence is missing an accepted update")
        certificate, certificate_sha256 = evidence
        summary = certificate.get("summary")
        source_only = certificate.get("source_only")
        binding = certificate.get("binding")
        virtual_count = row.get("virtual_count")
        if (
            row.get("update_dense_slot") != int(slot)
            or row.get("update_index") != update_index
            or isinstance(virtual_count, bool)
            or not isinstance(virtual_count, int)
            or virtual_count < 1
            or row.get("finite_psd_moment_merge") is not True
            or row.get("moment_covariance_scale") != 1.0
            or row.get("support_containment_guard") is not False
            or row.get("projected_support_guard") is not False
            or row.get("soft_mixture_certificate_schema_version")
            != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION
            or row.get("soft_mixture_certificate_kind")
            != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_KIND
            or row.get("soft_mixture_certificate_policy")
            != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_POLICY
            or row.get("soft_mixture_certificate_sha256") != certificate_sha256
            or row.get("soft_mixture_certificate_passed") is not True
            or not isinstance(summary, Mapping)
            or not isinstance(source_only, Mapping)
            or not isinstance(binding, Mapping)
            or summary.get("input_valid") is not True
            or summary.get("reason") is not None
            or source_only.get("source_camera_only") is not True
            or source_only.get("target_mapping_present") is not False
            or source_only.get("target_rgb_accessed") is not False
            or source_only.get("target_camera_metadata_accessed") is not False
            or source_only.get("target_index_accessed") is not False
            or source_only.get("omitted_s3_attributes_accessed") is not False
            or source_only.get("input_covariances_mutated") is not False
            or source_only.get("fixed_covariance_scale") is not True
            or source_only.get("boolean_owner_assignment_used") is not False
            or source_only.get("projected_domain_guard_used") is not False
        ):
            raise ValueError("DepthSplat soft-mixture certificate binding changed")
        for key in (
            "tile_key_sha256",
            "anchor_dense_slots_sha256",
            "virtual_origin_slots_sha256",
            "spatial_weights_sha256",
            "bilateral_assignment_weights_sha256",
            "anchor_source_means_sha256",
            "anchor_source_covariances_sha256",
            "anchor_source_harmonics_sha256",
            "anchor_source_opacities_sha256",
            "virtual_means_sha256",
            "virtual_covariances_sha256",
            "virtual_harmonics_sha256",
            "virtual_opacities_sha256",
            "merged_means_sha256",
            "merged_covariances_sha256",
            "merged_harmonics_sha256",
            "merged_opacities_sha256",
            "context_extrinsics_sha256",
            "context_intrinsics_sha256",
        ):
            _require_sha256(binding.get(key), label=f"soft-mixture {key}")
        normalized_rows.append(
            {
                "update_dense_slot": int(slot),
                "update_index": update_index,
                "virtual_count": virtual_count,
                "finite_psd_moment_merge": True,
                "moment_covariance_scale": 1.0,
                "support_containment_guard": False,
                "projected_support_guard": False,
                "soft_mixture_certificate_schema_version": (
                    DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION
                ),
                "soft_mixture_certificate_kind": DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_KIND,
                "soft_mixture_certificate_policy": (
                    DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_POLICY
                ),
                "soft_mixture_certificate_sha256": certificate_sha256,
                "soft_mixture_certificate_passed": True,
            }
        )
    if evidence_by_slot:
        raise ValueError("DepthSplat soft-mixture trace has an unbound update")
    return {
        "schema": DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
        "geometry": "source-only-spatial-s-bilateral-r-first-second-moment-replay-v1",
        "finite_psd_moment_merge": True,
        "fixed_moment_covariance_scale": 1.0,
        "support_containment_guard": False,
        "projected_support_guard": False,
        "tile_trace_sha256": tile_trace_sha256,
        "update_binding": dict(update_binding),
        "update_slots": [int(slot) for slot in slots],
        "per_update": normalized_rows,
    }


def _ordered_coverage_rows_from_trace(
    *, tile_trace: Any, update_slots: torch.Tensor
) -> list[dict[str, Any]]:
    """Bind each committed update to the coverage evidence in its live trace."""

    if not isinstance(tile_trace, (list, tuple)) or not torch.is_tensor(update_slots):
        raise ValueError("DepthSplat coverage trace inputs are invalid")
    coverage_by_slot: dict[int, dict[str, Any]] = {}
    for tile_record in tile_trace:
        if not isinstance(tile_record, Mapping):
            raise ValueError("DepthSplat coverage tile trace is invalid")
        coverage = tile_record.get("coverage")
        if not isinstance(coverage, Mapping):
            continue
        per_update = coverage.get("per_update")
        if not isinstance(per_update, list):
            raise ValueError("DepthSplat coverage tile record is incomplete")
        for row in per_update:
            if not isinstance(row, Mapping):
                raise ValueError("DepthSplat coverage tile row is invalid")
            slot = row.get("update_dense_slot")
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot in coverage_by_slot
            ):
                raise ValueError("DepthSplat coverage tile slot is invalid")
            coverage_by_slot[slot] = dict(row)
    ordered_rows: list[dict[str, Any]] = []
    for update_index, slot in enumerate(
        update_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    ):
        row = coverage_by_slot.pop(int(slot), None)
        if row is None:
            raise ValueError("DepthSplat coverage is missing an accepted update slot")
        ordered_rows.append({**row, "update_index": update_index})
    if coverage_by_slot:
        raise ValueError("DepthSplat coverage has an unbound update slot")
    return ordered_rows


def _validate_coverage_certificate(
    preflight: DepthSplatCompactMaterializationPreflight,
) -> dict[str, Any]:
    """Revalidate the profile-specific materialization certificate."""

    events = preflight.events
    trace_sha256 = _require_bound_tile_trace(
        preflight.tile_trace,
        expected_sha256=events.get("tile_trace_sha256"),
        label="preflight tile trace",
    )
    update_binding = _update_binding(
        preflight.update_dense_slots,
        preflight.means,
        preflight.covariances,
        preflight.harmonics,
        preflight.opacities,
    )
    trace_rows = _ordered_coverage_rows_from_trace(
        tile_trace=preflight.tile_trace,
        update_slots=preflight.update_dense_slots,
    )
    payload = events.get("coverage_certificate_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("DepthSplat coverage certificate payload is missing")
    profile = events.get("execution_profile")
    maximum_scale = events.get("maximum_coverage_covariance_scale")
    if profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
        if (
            isinstance(maximum_scale, bool)
            or not isinstance(maximum_scale, (int, float))
            or float(maximum_scale) != 1.0
        ):
            raise ValueError("DepthSplat literal moment certificate scale changed")
        expected_certificate = DEPTHSPLAT_LITERAL_PAPER_T4_MOMENT_CERTIFICATE
        expected = _literal_moment_merge_certificate_payload(
            update_slots=preflight.update_dense_slots,
            update_binding=update_binding,
            update_means=preflight.means,
            update_covariances=preflight.covariances,
            per_update=trace_rows,
            tile_trace_sha256=trace_sha256,
        )
    elif profile == DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE:
        if (
            isinstance(maximum_scale, bool)
            or not isinstance(maximum_scale, (int, float))
            or float(maximum_scale) != 1.0
        ):
            raise ValueError("DepthSplat coverage-enriched certificate scale changed")
        expected_certificate = DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE
        expected = _coverage_enriched_moment_certificate_payload(
            update_slots=preflight.update_dense_slots,
            update_binding=update_binding,
            update_means=preflight.means,
            update_covariances=preflight.covariances,
            per_update=trace_rows,
            tile_trace_sha256=trace_sha256,
        )
    elif profile == DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE:
        if (
            isinstance(maximum_scale, bool)
            or not isinstance(maximum_scale, (int, float))
            or float(maximum_scale) != 1.0
        ):
            raise ValueError("DepthSplat support-basis certificate scale changed")
        expected_certificate = DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE
        expected = _support_basis_moment_certificate_payload(
            update_slots=preflight.update_dense_slots,
            update_binding=update_binding,
            update_means=preflight.means,
            update_covariances=preflight.covariances,
            per_update=trace_rows,
            tile_trace=preflight.tile_trace,
            tile_trace_sha256=trace_sha256,
        )
    elif _is_soft_mixture_profile(profile):
        if (
            isinstance(maximum_scale, bool)
            or not isinstance(maximum_scale, (int, float))
            or float(maximum_scale) != 1.0
        ):
            raise ValueError("DepthSplat soft-mixture certificate scale changed")
        expected_certificate = DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE
        expected = _soft_mixture_moment_certificate_payload(
            update_slots=preflight.update_dense_slots,
            update_binding=update_binding,
            update_means=preflight.means,
            update_covariances=preflight.covariances,
            per_update=trace_rows,
            tile_trace=preflight.tile_trace,
            tile_trace_sha256=trace_sha256,
        )
    elif profile in {
        DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
    }:
        if (
            isinstance(maximum_scale, bool)
            or not isinstance(maximum_scale, (int, float))
            or float(maximum_scale) < 1.0
        ):
            raise ValueError("DepthSplat coverage certificate scale limit is invalid")
        expected_certificate = DEPTHSPLAT_COVERAGE_CERTIFICATE
        expected = _coverage_certificate_payload(
            update_slots=preflight.update_dense_slots,
            update_binding=update_binding,
            per_update=trace_rows,
            tile_trace_sha256=trace_sha256,
            maximum_covariance_scale=float(maximum_scale),
        )
    else:
        raise ValueError("DepthSplat materialization certificate profile is invalid")
    if (
        events.get("coverage_certificate") != expected_certificate
        or dict(payload) != expected
        or events.get("coverage_certificate_sha256") != _canonical_sha256(expected)
    ):
        raise ValueError("DepthSplat materialization certificate binding changed")
    return expected


def _build_tile_updates(
    *,
    packet: DepthSplatSparseRawPacket,
    packed: DepthSplatPackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    feature_map: torch.Tensor,
    z_depth_map: torch.Tensor,
    record: Mapping[str, Any],
    level: str,
    semantics: str,
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    feature_statistic: str,
    source_grid: torch.Tensor,
    source_get_world_rays: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    maximum_feature_relative_residual: float,
    maximum_coverage_covariance_scale: float,
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, Any]]:
    anchors = _route_anchor_positions(record, level=level, semantics=semantics)
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=4,
        positions=anchors,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("DepthSplat materializer tile lacks a selected anchor")
    anchor_indices = [slot_to_index[slot] for slot in anchor_slots]
    if bool((packed.opacities[anchor_indices] >= 1.0).any()):
        raise _DepthSplatTileRejection(NATIVE_OPACITY_ENDPOINT_FULL_REASON)
    source_positions = [(tile_y * 4 + row, tile_x * 4 + column) for row, column in anchors]
    all_positions = _full_positions(4)
    target_local = [position for position in all_positions if position not in set(anchors)]
    if not target_local:
        return [], {
            "virtual_count": 0,
            "coverage": {
                "maximum_containment_lhs_before_scale": 0.0,
                "maximum_containment_lhs_after_scale": 0.0,
                "moment_covariance_scale_max": 1.0,
            },
        }
    target_positions = [(tile_y * 4 + row, tile_x * 4 + column) for row, column in target_local]
    target_z_depths = torch.stack(
        [z_depth_map[row, column] for row, column in target_positions]
    ).to(device=packet.depths.device, dtype=packet.depths.dtype)
    if not bool(torch.isfinite(target_z_depths).all()) or bool((target_z_depths <= 0.0).any()):
        raise ValueError("DepthSplat materializer virtual source z-depth is invalid")
    offsets = _anchor_geometry(
        packet=packet,
        packed=packed,
        indices=anchor_indices,
        source_positions=source_positions,
        view=view,
        height=height,
        width=width,
        source_grid=source_grid,
        source_get_world_rays=source_get_world_rays,
    )
    spatial = _spatial_weights(
        target_positions,
        source_positions,
        device=packed.means.device,
        dtype=packed.means.dtype,
    )
    continuity = _feature_continuity(
        feature_map,
        source_positions,
        target_positions,
        spatial,
        maximum_relative_residual=maximum_feature_relative_residual,
    )
    if continuity["passed"] is not True:
        raise ValueError("DepthSplat materializer feature continuity rejected tile")
    source_harmonics = packed.harmonics[anchor_indices]
    source_opacities = packed.opacities[anchor_indices]
    virtual_harmonics = (spatial @ source_harmonics.reshape(len(anchors), -1)).reshape(
        len(target_local), *source_harmonics.shape[1:]
    )
    logits = torch.logit(source_opacities.clamp(1e-6, 1.0 - 1e-6))
    virtual_opacities = (spatial @ logits).sigmoid()
    virtual_rgb = spatial @ packet.source_rgb[anchor_indices]
    if (
        not bool(torch.isfinite(virtual_harmonics).all())
        or not bool(torch.isfinite(virtual_opacities).all())
        or not bool(torch.isfinite(virtual_rgb).all())
        or bool((virtual_opacities < 0.0).any())
        or bool((virtual_opacities >= 1.0).any())
    ):
        raise ValueError("DepthSplat materializer RGB-SH virtual field is invalid")
    anchor_features = torch.stack([feature_map[:, row, column] for row, column in source_positions])
    coordinate_scale = 3.0
    feature_variance = _tile_feature_variance(record, feature_statistic)
    assignments: list[torch.Tensor] = []
    for local_row, local_column in target_local:
        target_feature = feature_map[:, tile_y * 4 + local_row, tile_x * 4 + local_column]
        spatial_distances = torch.tensor(
            [((local_row - row) / coordinate_scale) ** 2 + ((local_column - column) / coordinate_scale) ** 2 for row, column in anchors],
            device=packed.means.device,
            dtype=packed.means.dtype,
        )
        feature_distances = (anchor_features - target_feature.unsqueeze(0)).square().sum(dim=1)
        kwargs: dict[str, Any] = {}
        if level == "L1":
            primary_count = len(compute_probe_positions(4))
            if anchors[:primary_count] != compute_probe_positions(4):
                raise ValueError("DepthSplat materializer L1 anchors lost their primary prefix")
            anchor_depths = packet.depths[anchor_indices]
            kwargs = {
                "probe_depths": anchor_depths,
                "depth_reference_depths": anchor_depths[:primary_count],
                "beta_d": 1.0,
            }
        weights = paper_assignment_weights(
            spatial_distances,
            feature_distances.to(dtype=packed.means.dtype),
            feature_variance=feature_variance,
            beta_x=0.5,
            beta_f=0.1,
            level=level,
            **kwargs,
        )
        if not bool(torch.isfinite(weights).all()) or not torch.allclose(
            weights.sum(), torch.ones((), device=weights.device, dtype=weights.dtype), rtol=1e-5, atol=1e-5
        ):
            raise ValueError("DepthSplat materializer assignment is invalid")
        assignments.append(weights)
    assignment = torch.stack(assignments)
    target_centres = _pixel_centres(
        target_positions,
        height=height,
        width=width,
        device=packet.coordinates.device,
        dtype=packet.coordinates.dtype,
    )
    updates: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    coverage_scales: list[float] = []
    coverage_lhs: list[float] = []
    coverage_updates: list[dict[str, Any]] = []
    depth_order = torch.argsort(target_z_depths, stable=True)
    for anchor_offset, (slot, index) in enumerate(zip(anchor_slots, anchor_indices)):
        conditional_coordinates = target_centres + offsets[anchor_offset].unsqueeze(0)
        if (
            not bool(torch.isfinite(conditional_coordinates).all())
            or bool((conditional_coordinates < 0.0).any())
            or bool((conditional_coordinates > 1.0).any())
        ):
            raise ValueError("DepthSplat materializer transported offset leaves the native image")
        count = conditional_coordinates.shape[0]
        virtual_means = depthsplat_z_depth_world_means(
            conditional_coordinates,
            packet.extrinsics[index].unsqueeze(0).expand(count, -1, -1),
            packet.intrinsics[index].unsqueeze(0).expand(count, -1, -1),
            target_z_depths,
            source_get_world_rays=source_get_world_rays,
        )
        contribution_alpha = (
            assignment[:, anchor_offset] * virtual_opacities
        ).clamp(min=0.0, max=1.0 - 1e-6)
        base_opacity = packed.opacities[index]
        # Composite native anchor and virtual supports in a declared source S2
        # z-depth order. The incremental union masses sum exactly to the final
        # alpha and provide an order-defined, reproducible moment/SH measure.
        contributor_depths = torch.cat(
            (
                packet.depths[index].reshape(1),
                target_z_depths[depth_order],
            )
        )
        contributor_alphas = torch.cat(
            (
                base_opacity.reshape(1),
                contribution_alpha[depth_order],
            )
        )
        contributors = torch.cat(
            (
                packed.means[index].unsqueeze(0),
                virtual_means[depth_order],
            ),
            dim=0,
        )
        contributor_harmonics = torch.cat(
            (
                packed.harmonics[index].unsqueeze(0),
                virtual_harmonics[depth_order],
            ),
            dim=0,
        )
        ordered = torch.argsort(contributor_depths, stable=True)
        contributors = contributors[ordered]
        contributor_harmonics = contributor_harmonics[ordered]
        contributor_alphas = contributor_alphas[ordered]
        remaining_transmittance = torch.ones((), device=base_opacity.device, dtype=base_opacity.dtype)
        incremental_masses: list[torch.Tensor] = []
        for contribution in contributor_alphas:
            incremental = remaining_transmittance * contribution
            incremental_masses.append(incremental)
            remaining_transmittance = remaining_transmittance * (1.0 - contribution)
        weights = torch.stack(incremental_masses)
        masses = weights
        mass_sum = masses.sum()
        if not bool(torch.isfinite(mass_sum)) or bool(mass_sum <= torch.finfo(masses.dtype).eps):
            raise ValueError("DepthSplat materializer moment mass is invalid")
        merged_mean = (masses.unsqueeze(1) * contributors).sum(dim=0) / mass_sum
        base_covariance = packed.covariances[index]
        contributor_covariances = base_covariance.unsqueeze(0).expand(contributors.shape[0], -1, -1)
        deltas = contributors - merged_mean.unsqueeze(0)
        merged_covariance = (
            masses.reshape(-1, 1, 1)
            * (contributor_covariances + deltas.unsqueeze(2) @ deltas.unsqueeze(1))
        ).sum(dim=0) / mass_sum
        merged_covariance, coverage = _coverage_closed_covariance(
            merged_covariance,
            merged_mean,
            virtual_means,
            base_covariance.unsqueeze(0).expand(count, -1, -1),
            maximum_scale=maximum_coverage_covariance_scale,
        )
        merged_opacity = 1.0 - remaining_transmittance
        if not bool(torch.isfinite(merged_opacity)) or bool(merged_opacity < 0.0) or bool(merged_opacity >= 1.0):
            raise ValueError("DepthSplat materializer merged opacity is invalid")
        merged_harmonics = torch.einsum("n,ncd->cd", weights, contributor_harmonics) / mass_sum
        if not bool(torch.isfinite(merged_harmonics).all()):
            raise ValueError("DepthSplat materializer merged RGB-SH is non-finite")
        updates.append((slot, merged_mean, merged_covariance, merged_harmonics, merged_opacity))
        coverage_scales.append(float(coverage["moment_covariance_scale"]))
        coverage_lhs.append(float(coverage["maximum_containment_lhs_after_scale"]))
        coverage_updates.append(
            {
                "update_dense_slot": int(slot),
                "virtual_count": count,
                "shape_aware_virtual_2sigma_support": True,
                **coverage,
            }
        )
    return updates, {
        "virtual_count": len(target_local),
        "continuity": continuity,
        "coverage": {
            "certificate": DEPTHSPLAT_COVERAGE_CERTIFICATE,
            "shape_aware_virtual_2sigma_support": True,
            "maximum_containment_lhs_after_scale": max(coverage_lhs, default=0.0),
            "moment_covariance_scale_max": max(coverage_scales, default=1.0),
            "per_update": coverage_updates,
        },
        "virtual_rgb_range": [
            float(virtual_rgb.amin().item()),
            float(virtual_rgb.amax().item()),
        ],
        "virtual_z_depth_min": float(target_z_depths.min().item()),
        "virtual_z_depth_max": float(target_z_depths.max().item()),
        "opacity_compositing_order": "source-s2-z-depth-near-to-far-stable-v1",
    }


def _literal_finite_psd_moment_merge(
    covariance: torch.Tensor,
    mean: torch.Tensor,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
) -> dict[str, Any]:
    """Validate a literal fixed-scale first/second-moment merge.

    Section 3 defines bilateral probe assignment followed by first/second
    moment matching.  It does not impose a separate 2-sigma containment test
    between each virtual Gaussian and its retained representative.  The
    literal route therefore accepts every finite PSD moment result unchanged
    and leaves support-closure only to the development profile.
    """

    if (
        covariance.shape != (3, 3)
        or mean.shape != (3,)
        or virtual_means.ndim != 2
        or virtual_means.shape[1] != 3
        or virtual_covariances.shape != (virtual_means.shape[0], 3, 3)
    ):
        raise ValueError("DepthSplat formal paper moment inputs are invalid")
    covariance = (covariance + covariance.mT) * 0.5
    projected_virtual_covariances = (
        virtual_covariances + virtual_covariances.mT
    ) * 0.5
    if (
        not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(covariance).all())
        or not bool(torch.isfinite(virtual_means).all())
        or not bool(torch.isfinite(projected_virtual_covariances).all())
    ):
        raise ValueError("DepthSplat formal paper moment merge is non-finite")
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance)
    virtual_eigenvalues = (
        torch.linalg.eigvalsh(projected_virtual_covariances)
        if virtual_means.numel()
        else torch.empty(0, device=covariance.device, dtype=covariance.dtype)
    )
    if bool((covariance_eigenvalues < -1e-6).any()) or bool(
        (virtual_eigenvalues < -1e-6).any()
    ):
        raise ValueError("DepthSplat formal paper moment covariance is not PSD")
    return {
        "finite_psd_moment_merge": True,
        "moment_covariance_scale": 1.0,
        "support_containment_guard": False,
        "minimum_merged_covariance_eigenvalue": float(
            covariance_eigenvalues.min().item()
        ),
        "minimum_virtual_covariance_eigenvalue": float(
            virtual_eigenvalues.min().item()
            if virtual_eigenvalues.numel()
            else 0.0
        ),
    }


def _validate_coverage_enriched_owner_support(
    owner_support: Any,
    *,
    owner_count: int,
    virtual_count: int,
    virtual_owner_indices: torch.Tensor,
) -> dict[str, Any]:
    """Reject incomplete owner evidence before a compact tile can survive."""

    if not isinstance(owner_support, Mapping):
        raise ValueError("DepthSplat coverage-enriched owner evidence is invalid")
    source_only = owner_support.get("source_only")
    ownership = owner_support.get("ownership")
    summary = owner_support.get("summary")
    records = owner_support.get("owners")
    expected_counts = [
        int((virtual_owner_indices == owner_index).sum().item())
        for owner_index in range(owner_count)
    ]
    if (
        owner_support.get("schema_version")
        != DEPTHSPLAT_OWNER_COVERAGE_AUDIT_SCHEMA_VERSION
        or owner_support.get("kind") != DEPTHSPLAT_OWNER_COVERAGE_AUDIT_KIND
        or owner_support.get("sigma") != 2.0
        or not isinstance(source_only, Mapping)
        or source_only.get("source_camera_only") is not True
        or source_only.get("target_mapping_present") is not False
        or source_only.get("target_rgb_accessed") is not False
        or source_only.get("target_camera_metadata_accessed") is not False
        or source_only.get("target_index_accessed") is not False
        or source_only.get("input_covariances_mutated") is not False
        or source_only.get("owner_source_anchor_support_included") is not True
        or not isinstance(ownership, Mapping)
        or ownership.get("policy") != DEPTHSPLAT_OWNER_ASSIGNMENT_POLICY
        or _require_sha256(
            ownership.get("virtual_owner_indices_sha256"),
            label="coverage-enriched virtual owner assignment",
        )
        != _tensor_sha256(virtual_owner_indices)
        or ownership.get("owner_assignment_counts") != expected_counts
        or not isinstance(summary, Mapping)
        or summary.get("input_valid") is not True
        or summary.get("owner_count") != owner_count
        or summary.get("source_anchor_count") != owner_count
        or summary.get("virtual_primitive_count") != virtual_count
        or summary.get("assigned_owner_count") != sum(count > 0 for count in expected_counts)
        or summary.get("empty_owner_count") != sum(count == 0 for count in expected_counts)
        or summary.get("checked_owner_count") != owner_count
        or summary.get("invalid_owner_assignment_count") != 0
        or not isinstance(records, list)
        or len(records) != owner_count
    ):
        raise ValueError("DepthSplat coverage-enriched owner evidence binding changed")

    normalized_records: list[dict[str, Any]] = []
    failed_owner_count = 0
    aggregate_hole_count = 0
    for owner_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("DepthSplat coverage-enriched owner record is invalid")
        coverage = record.get("coverage")
        passed = record.get("passed")
        audit_valid = record.get("audit_valid")
        if (
            record.get("owner_index") != owner_index
            or record.get("source_anchor_count") != 1
            or record.get("source_anchor_support_included") is not True
            or record.get("source_anchor_support_passed") is not passed
            or record.get("assigned_virtual_count") != expected_counts[owner_index]
            or record.get("dense_descriptor_count") != expected_counts[owner_index] + 1
            or record.get("checked") is not True
            or not isinstance(passed, bool)
            or not isinstance(audit_valid, bool)
            or not isinstance(coverage, Mapping)
            or coverage.get("dense_descriptor_count") != expected_counts[owner_index] + 1
        ):
            raise ValueError("DepthSplat coverage-enriched owner record binding changed")
        if audit_valid:
            hole_count = coverage.get("hole_count")
            if (
                coverage.get("valid") is not True
                or isinstance(hole_count, bool)
                or not isinstance(hole_count, int)
                or hole_count < 0
                or record.get("active_dense_descriptor_count")
                != coverage.get("active_dense_descriptor_count")
                or passed != (hole_count == 0)
            ):
                raise ValueError("DepthSplat coverage-enriched owner coverage changed")
            aggregate_hole_count += hole_count
        else:
            if (
                coverage.get("valid") is not False
                or coverage.get("reason") != "zero-dense-optical-mass"
                or passed is not True
                or record.get("active_dense_descriptor_count") != 0
            ):
                raise ValueError("DepthSplat coverage-enriched inactive owner changed")
        if not passed:
            failed_owner_count += 1
        normalized_records.append(dict(record))

    passed = owner_support.get("passed")
    if (
        not isinstance(passed, bool)
        or summary.get("failed_owner_count") != failed_owner_count
        or summary.get("hole_count") != aggregate_hole_count
        or summary.get("all_active_owner_anchor_and_virtual_2sigma_supports_contained")
        is not passed
        or passed != (failed_owner_count == 0 and aggregate_hole_count == 0)
    ):
        raise ValueError("DepthSplat coverage-enriched owner summary changed")
    return {
        **dict(owner_support),
        "owners": normalized_records,
        "ownership": dict(ownership),
        "summary": dict(summary),
        "source_only": dict(source_only),
    }


def _validate_support_basis_evidence(
    support_basis: Any,
    *,
    anchor_slots: list[int],
    virtual_origin_slots: list[int],
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_opacities: torch.Tensor,
    virtual_source_spatial_weights: torch.Tensor,
    bilateral_assignment_weights: torch.Tensor,
    anchor_source_means: torch.Tensor,
    anchor_source_covariances: torch.Tensor,
    anchor_source_opacities: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_opacities: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
) -> dict[str, Any]:
    """Validate a local soft-ledger certificate before it can retain a tile."""

    anchor_count = len(anchor_slots)
    virtual_count = len(virtual_origin_slots)
    if not isinstance(support_basis, Mapping):
        raise ValueError("DepthSplat support-basis evidence is invalid")
    source_only = support_basis.get("source_only")
    summary = support_basis.get("summary")
    binding = support_basis.get("binding")
    anchors = support_basis.get("anchors")
    virtuals = support_basis.get("virtuals")
    expected_anchor_slots = torch.tensor(
        anchor_slots,
        device=virtual_source_spatial_weights.device,
        dtype=torch.int64,
    )
    expected_virtual_slots = torch.tensor(
        virtual_origin_slots,
        device=virtual_source_spatial_weights.device,
        dtype=torch.int64,
    )
    virtual_masks = bilateral_assignment_weights > 0.0
    source_to_output = (
        virtual_source_spatial_weights.mT @ bilateral_assignment_weights
    )
    anchor_masks = (source_to_output > 0.0) | torch.eye(
        anchor_count,
        device=virtual_source_spatial_weights.device,
        dtype=torch.bool,
    )
    expected_binding = {
        "anchor_dense_slots_sha256": _tensor_sha256(expected_anchor_slots),
        "virtual_origin_slots_sha256": _tensor_sha256(expected_virtual_slots),
        "virtual_means_sha256": _tensor_sha256(virtual_means),
        "virtual_covariances_sha256": _tensor_sha256(virtual_covariances),
        "virtual_opacities_sha256": _tensor_sha256(virtual_opacities),
        "virtual_source_spatial_weights_sha256": _tensor_sha256(
            virtual_source_spatial_weights
        ),
        "bilateral_assignment_weights_sha256": _tensor_sha256(
            bilateral_assignment_weights
        ),
        "virtual_candidate_basis_mask_sha256": _tensor_sha256(
            virtual_masks.to(dtype=torch.uint8)
        ),
        "anchor_candidate_basis_mask_sha256": _tensor_sha256(
            anchor_masks.to(dtype=torch.uint8)
        ),
        "source_to_output_ledger_sha256": _tensor_sha256(source_to_output),
        "anchor_source_means_sha256": _tensor_sha256(anchor_source_means),
        "anchor_source_covariances_sha256": _tensor_sha256(
            anchor_source_covariances
        ),
        "anchor_source_opacities_sha256": _tensor_sha256(anchor_source_opacities),
        "merged_means_sha256": _tensor_sha256(merged_means),
        "merged_covariances_sha256": _tensor_sha256(merged_covariances),
        "merged_opacities_sha256": _tensor_sha256(merged_opacities),
        "context_extrinsics_sha256": _tensor_sha256(context_extrinsics),
        "context_intrinsics_sha256": _tensor_sha256(context_intrinsics),
    }
    if (
        support_basis.get("schema_version")
        != DEPTHSPLAT_SUPPORT_BASIS_AUDIT_SCHEMA_VERSION
        or support_basis.get("kind") != DEPTHSPLAT_SUPPORT_BASIS_AUDIT_KIND
        or support_basis.get("support_basis_policy")
        != DEPTHSPLAT_SUPPORT_BASIS_POLICY
        or support_basis.get("sigma") != 2.0
        or not isinstance(source_only, Mapping)
        or source_only.get("source_camera_only") is not True
        or source_only.get("target_mapping_present") is not False
        or source_only.get("target_rgb_accessed") is not False
        or source_only.get("target_camera_metadata_accessed") is not False
        or source_only.get("target_index_accessed") is not False
        or source_only.get("input_covariances_mutated") is not False
        or source_only.get("same_tile_candidate_basis_only") is not True
        or source_only.get("fixed_covariance_scale") is not True
        or not isinstance(binding, Mapping)
        or any(binding.get(key) != value for key, value in expected_binding.items())
        or not isinstance(summary, Mapping)
        or summary.get("input_valid") is not True
        or summary.get("anchor_count") != anchor_count
        or summary.get("source_anchor_count") != anchor_count
        or summary.get("candidate_anchor_count") != anchor_count
        or summary.get("virtual_primitive_count") != virtual_count
        or summary.get("checked_source_anchor_count") != anchor_count
        or summary.get("checked_virtual_count") != virtual_count
        or not isinstance(anchors, list)
        or not isinstance(virtuals, list)
        or len(anchors) != anchor_count
        or len(virtuals) != virtual_count
    ):
        raise ValueError("DepthSplat support-basis evidence binding changed")

    failed_anchor_count = 0
    failed_virtual_count = 0
    active_anchor_count = 0
    active_virtual_count = 0
    hole_count = 0

    def validate_records(
        records: list[Any],
        *,
        source_kind: str,
        slots: list[int],
        masks: torch.Tensor,
    ) -> tuple[int, int, int]:
        failed = 0
        active = 0
        holes = 0
        for source_index, record in enumerate(records):
            expected_candidates = [
                int(index)
                for index in masks[source_index]
                .nonzero(as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            ]
            if (
                not isinstance(record, Mapping)
                or record.get("source_kind") != source_kind
                or record.get("source_index") != source_index
                or record.get("source_slot") != int(slots[source_index])
                or record.get("basis_candidate_anchor_indices") != expected_candidates
                or record.get("checked") is not True
                or not isinstance(record.get("passed"), bool)
                or record.get("active") not in {True, False, None}
            ):
                raise ValueError("DepthSplat support-basis record binding changed")
            coverage = record.get("coverage")
            passed = record["passed"]
            if record.get("active") is True:
                if (
                    not isinstance(coverage, Mapping)
                    or coverage.get("valid") is not True
                    or not isinstance(coverage.get("hole_count"), int)
                    or coverage["hole_count"] < 0
                    or passed != (coverage["hole_count"] == 0)
                ):
                    raise ValueError("DepthSplat support-basis active coverage changed")
                active += 1
                holes += int(coverage["hole_count"])
            elif record.get("active") is False:
                if (
                    not isinstance(coverage, Mapping)
                    or coverage.get("valid") is not False
                    or coverage.get("reason") != "zero-dense-optical-mass"
                    or passed is not True
                ):
                    raise ValueError("DepthSplat support-basis inactive coverage changed")
            elif passed:
                raise ValueError("DepthSplat support-basis inactive record passed")
            failed += int(not passed)
        return failed, active, holes

    failed_anchor_count, active_anchor_count, anchor_holes = validate_records(
        anchors,
        source_kind="anchor",
        slots=anchor_slots,
        masks=anchor_masks,
    )
    failed_virtual_count, active_virtual_count, virtual_holes = validate_records(
        virtuals,
        source_kind="virtual",
        slots=virtual_origin_slots,
        masks=virtual_masks,
    )
    hole_count = anchor_holes + virtual_holes
    passed = support_basis.get("passed")
    if (
        not isinstance(passed, bool)
        or summary.get("failed_source_anchor_count") != failed_anchor_count
        or summary.get("failed_virtual_count") != failed_virtual_count
        or summary.get("active_source_anchor_count") != active_anchor_count
        or summary.get("active_virtual_primitive_count") != active_virtual_count
        or summary.get("active_dense_primitive_count")
        != active_anchor_count + active_virtual_count
        or summary.get("hole_count") != hole_count
        or summary.get("all_active_same_tile_supports_contained") is not passed
        or passed != (
            failed_anchor_count == 0
            and failed_virtual_count == 0
            and hole_count == 0
        )
    ):
        raise ValueError("DepthSplat support-basis summary changed")
    return {
        **dict(support_basis),
        "source_only": dict(source_only),
        "summary": dict(summary),
        "binding": dict(binding),
        "anchors": [dict(record) for record in anchors],
        "virtuals": [dict(record) for record in virtuals],
    }


_SOFT_MIXTURE_RESIDUAL_FIELDS = (
    "virtual_means",
    "virtual_covariances",
    "virtual_harmonics",
    "virtual_opacities",
    "merged_means",
    "merged_covariances",
    "merged_harmonics",
    "merged_opacities",
)


def _validate_soft_mixture_evidence(
    soft_mixture: Any,
    *,
    tile_key: tuple[int, int, int],
    anchor_dense_slots: torch.Tensor,
    virtual_origin_slots: torch.Tensor,
    spatial_weights: torch.Tensor,
    bilateral_assignment_weights: torch.Tensor,
    anchor_source_means: torch.Tensor,
    anchor_source_covariances: torch.Tensor,
    anchor_source_harmonics: torch.Tensor,
    anchor_source_opacities: torch.Tensor,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_harmonics: torch.Tensor,
    virtual_opacities: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_harmonics: torch.Tensor,
    merged_opacities: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    virtual_mean_geometry: str = "selected-anchor-spatial-moment-v1",
    virtual_mean_source: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Rebuild a soft S/R certificate from live source-only tile tensors."""

    expected = certify_depthsplat_tile_soft_mixture(
        tile_key=tile_key,
        anchor_dense_slots=anchor_dense_slots,
        virtual_origin_slots=virtual_origin_slots,
        spatial_weights=spatial_weights,
        bilateral_assignment_weights=bilateral_assignment_weights,
        anchor_source_means=anchor_source_means,
        anchor_source_covariances=anchor_source_covariances,
        anchor_source_harmonics=anchor_source_harmonics,
        anchor_source_opacities=anchor_source_opacities,
        virtual_means=virtual_means,
        virtual_covariances=virtual_covariances,
        virtual_harmonics=virtual_harmonics,
        virtual_opacities=virtual_opacities,
        merged_means=merged_means,
        merged_covariances=merged_covariances,
        merged_harmonics=merged_harmonics,
        merged_opacities=merged_opacities,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        virtual_mean_geometry=virtual_mean_geometry,
        virtual_mean_source=virtual_mean_source,
    )
    if not isinstance(soft_mixture, Mapping) or dict(soft_mixture) != expected:
        raise ValueError("DepthSplat soft-mixture certificate binding changed")
    source_only = expected.get("source_only")
    summary = expected.get("summary")
    binding = expected.get("binding")
    if (
        expected.get("schema_version") != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION
        or expected.get("kind") != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_KIND
        or expected.get("policy") != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_POLICY
        or expected.get("tile_key") != list(tile_key)
        or not isinstance(source_only, Mapping)
        or source_only.get("source_camera_only") is not True
        or source_only.get("target_mapping_present") is not False
        or source_only.get("target_rgb_accessed") is not False
        or source_only.get("target_camera_metadata_accessed") is not False
        or source_only.get("target_index_accessed") is not False
        or source_only.get("omitted_s3_attributes_accessed") is not False
        or source_only.get("input_covariances_mutated") is not False
        or source_only.get("fixed_covariance_scale") is not True
        or source_only.get("boolean_owner_assignment_used") is not False
        or source_only.get("projected_domain_guard_used") is not False
        or not isinstance(summary, Mapping)
        or not isinstance(binding, Mapping)
    ):
        raise ValueError("DepthSplat soft-mixture source-only contract changed")
    for key in (
        "tile_key_sha256",
        "anchor_dense_slots_sha256",
        "virtual_origin_slots_sha256",
        "spatial_weights_sha256",
        "bilateral_assignment_weights_sha256",
        "anchor_source_means_sha256",
        "anchor_source_covariances_sha256",
        "anchor_source_harmonics_sha256",
        "anchor_source_opacities_sha256",
        "virtual_means_sha256",
        "virtual_covariances_sha256",
        "virtual_harmonics_sha256",
        "virtual_opacities_sha256",
        "merged_means_sha256",
        "merged_covariances_sha256",
        "merged_harmonics_sha256",
        "merged_opacities_sha256",
        "context_extrinsics_sha256",
        "context_intrinsics_sha256",
    ):
        _require_sha256(binding.get(key), label=f"soft-mixture {key}")
    return expected


def _validate_mixture_kernel_closure_evidence(
    kernel_closure: Any,
    *,
    tile_key: tuple[int, int, int],
    anchor_dense_slots: torch.Tensor,
    virtual_origin_slots: torch.Tensor,
    bilateral_assignment_weights: torch.Tensor,
    anchor_source_means: torch.Tensor,
    anchor_source_covariances: torch.Tensor,
    anchor_source_opacities: torch.Tensor,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    virtual_opacities: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    merged_opacities: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    strict_maximum_relative_risk: float | None = None,
    measurement_cache: Any | None = None,
) -> dict[str, Any]:
    """Rebuild the strict source-only closure guard from live tile tensors."""

    expected = assess_depthsplat_tile_kernel_closure(
        tile_key=tile_key,
        anchor_dense_slots=anchor_dense_slots,
        virtual_origin_slots=virtual_origin_slots,
        bilateral_assignment_weights=bilateral_assignment_weights,
        anchor_source_means=anchor_source_means,
        anchor_source_covariances=anchor_source_covariances,
        anchor_source_opacities=anchor_source_opacities,
        virtual_means=virtual_means,
        virtual_covariances=virtual_covariances,
        virtual_opacities=virtual_opacities,
        merged_means=merged_means,
        merged_covariances=merged_covariances,
        merged_opacities=merged_opacities,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        strict_maximum_relative_risk=strict_maximum_relative_risk,
        measurement_cache=measurement_cache,
    )
    if not isinstance(kernel_closure, Mapping) or dict(kernel_closure) != expected:
        raise ValueError("DepthSplat mixture kernel-closure binding changed")
    source_only = expected.get("source_only")
    summary = expected.get("summary")
    binding = expected.get("binding")
    if (
        expected.get("schema_version") != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_SCHEMA_VERSION
        or expected.get("kind") != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND
        or expected.get("policy") != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY
        or expected.get("tile_key") != list(tile_key)
        or not isinstance(expected.get("passed"), bool)
        or not isinstance(source_only, Mapping)
        or source_only.get("source_camera_only") is not True
        or source_only.get("target_mapping_present") is not False
        or source_only.get("target_rgb_accessed") is not False
        or source_only.get("target_camera_metadata_accessed") is not False
        or source_only.get("target_index_accessed") is not False
        or source_only.get("omitted_s3_attributes_accessed") is not False
        or source_only.get("boolean_owner_assignment_used") is not False
        or source_only.get("projected_domain_guard_used") is not False
        or source_only.get("covariance_expansion_used") is not False
        or source_only.get("alpha_union_used") is not False
        or not isinstance(summary, Mapping)
    ):
        raise ValueError("DepthSplat mixture kernel-closure source-only contract changed")
    input_valid = summary.get("input_valid")
    if not isinstance(input_valid, bool):
        raise ValueError("DepthSplat mixture kernel-closure validity changed")
    if not input_valid:
        if expected.get("passed") is not False or binding is not None:
            raise ValueError("DepthSplat invalid mixture kernel-closure changed")
        return expected
    if not isinstance(binding, Mapping):
        raise ValueError("DepthSplat mixture kernel-closure binding is missing")
    for key in (
        "tile_key_sha256",
        "anchor_dense_slots_sha256",
        "virtual_origin_slots_sha256",
        "bilateral_assignment_weights_sha256",
        "anchor_source_means_sha256",
        "anchor_source_covariances_sha256",
        "anchor_source_opacities_sha256",
        "virtual_means_sha256",
        "virtual_covariances_sha256",
        "virtual_opacities_sha256",
        "merged_means_sha256",
        "merged_covariances_sha256",
        "merged_opacities_sha256",
        "context_extrinsics_sha256",
        "context_intrinsics_sha256",
    ):
        _require_sha256(binding.get(key), label=f"mixture kernel-closure {key}")
    return expected


def _soft_mixture_certificate_aggregate_from_trace(
    tile_trace: Any,
) -> dict[str, Any]:
    """Summarize replay certificates without retaining a target-side audit."""

    if not isinstance(tile_trace, (list, tuple)):
        raise ValueError("DepthSplat soft-mixture aggregate trace is invalid")
    maximum_absolute = {name: 0.0 for name in _SOFT_MIXTURE_RESIDUAL_FIELDS}
    maximum_tolerance = {name: 0.0 for name in _SOFT_MIXTURE_RESIDUAL_FIELDS}
    minima: dict[str, float | None] = {
        "minimum_source_covariance_eigenvalue": None,
        "minimum_virtual_covariance_eigenvalue": None,
        "minimum_merged_covariance_eigenvalue": None,
    }
    source_only: dict[str, bool] | None = None
    attempts = 0
    passed = 0
    for record in tile_trace:
        if not isinstance(record, Mapping):
            raise ValueError("DepthSplat soft-mixture aggregate tile is invalid")
        seen_certificate_sha256: set[str] = set()
        for certificate_key, sha256_key, passed_key in (
            (
                "soft_mixture_certificate",
                "soft_mixture_certificate_sha256",
                "soft_mixture_certificate_passed",
            ),
            (
                "l0_soft_mixture_failure",
                "l0_soft_mixture_failure_sha256",
                None,
            ),
            (
                "l1_soft_mixture_failure",
                "l1_soft_mixture_failure_sha256",
                None,
            ),
            (
                "l0_soft_mixture_certificate_before_kernel_closure",
                "l0_soft_mixture_certificate_before_kernel_closure_sha256",
                None,
            ),
            (
                "l1_soft_mixture_certificate_before_kernel_closure",
                "l1_soft_mixture_certificate_before_kernel_closure_sha256",
                None,
            ),
        ):
            certificate = record.get(certificate_key)
            certificate_sha256 = record.get(sha256_key)
            certificate_passed = (
                record.get(passed_key)
                if passed_key is not None
                else certificate.get("passed")
                if isinstance(certificate, Mapping)
                else None
            )
            if (
                certificate is None
                and certificate_sha256 is None
                and certificate_passed is None
            ):
                continue
            if (
                not isinstance(certificate, Mapping)
                or not isinstance(certificate_sha256, str)
                or certificate_sha256 != _canonical_sha256(certificate)
                or not isinstance(certificate_passed, bool)
                or certificate.get("schema_version")
                != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION
                or certificate.get("kind") != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_KIND
                or certificate.get("policy") != DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_POLICY
                or certificate.get("passed") is not certificate_passed
                or not isinstance(certificate.get("source_only"), Mapping)
                or not isinstance(certificate.get("summary"), Mapping)
            ):
                raise ValueError("DepthSplat soft-mixture aggregate certificate changed")
            current_source_only = dict(certificate["source_only"])
            if source_only is None:
                source_only = current_source_only
            elif source_only != current_source_only:
                raise ValueError(
                    "DepthSplat soft-mixture aggregate source-only flags drifted"
                )
            summary = certificate["summary"]
            residuals = summary.get("residuals")
            if not isinstance(residuals, Mapping):
                raise ValueError("DepthSplat soft-mixture aggregate residuals are invalid")
            for name in _SOFT_MIXTURE_RESIDUAL_FIELDS:
                residual = residuals.get(name)
                if (
                    not isinstance(residual, Mapping)
                    or isinstance(residual.get("maximum_absolute"), bool)
                    or not isinstance(residual.get("maximum_absolute"), (int, float))
                    or isinstance(residual.get("tolerance"), bool)
                    or not isinstance(residual.get("tolerance"), (int, float))
                ):
                    raise ValueError("DepthSplat soft-mixture aggregate residual changed")
                maximum_absolute[name] = max(
                    maximum_absolute[name], float(residual["maximum_absolute"])
                )
                maximum_tolerance[name] = max(
                    maximum_tolerance[name], float(residual["tolerance"])
                )
            for name in minima:
                value = summary.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        "DepthSplat soft-mixture aggregate covariance summary changed"
                    )
                minimum = float(value)
                minima[name] = (
                    minimum if minima[name] is None else min(minima[name], minimum)
                )
            # A passed kernel attempt retains its S/R certificate both before
            # closure and as accepted tile evidence. They are one attempt.
            if certificate_sha256 in seen_certificate_sha256:
                continue
            seen_certificate_sha256.add(certificate_sha256)
            attempts += 1
            passed += int(certificate_passed)
    return {
        "schema": "depthsplat-soft-mixture-tile-certificate-aggregate-v1",
        "certificate": DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
        "kind": DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_KIND,
        "policy": DEPTHSPLAT_SOFT_MIXTURE_CERTIFICATE_POLICY,
        "source_only": source_only,
        "tile_certificate_attempt_count": attempts,
        "passed_tile_certificate_count": passed,
        "failed_tile_certificate_count": attempts - passed,
        "maximum_absolute_residual_by_field": maximum_absolute,
        "maximum_tolerance_by_field": maximum_tolerance,
        **minima,
    }


def _validate_soft_mixture_certificate_aggregate(
    *, events: Mapping[str, Any], tile_trace: Any
) -> dict[str, Any]:
    """Require the compact soft-mixture summary to rebuild from the trace."""

    expected = _soft_mixture_certificate_aggregate_from_trace(tile_trace)
    aggregate = events.get("soft_mixture_certificate_aggregate")
    if (
        not isinstance(aggregate, Mapping)
        or dict(aggregate) != expected
        or events.get("soft_mixture_certificate_aggregate_sha256")
        != _canonical_sha256(expected)
    ):
        raise ValueError("DepthSplat soft-mixture aggregate binding changed")
    return expected


def _mixture_kernel_closure_aggregate_from_trace(tile_trace: Any) -> dict[str, Any]:
    """Summarize strict kernel-closure decisions from the committed tile trace."""

    if not isinstance(tile_trace, (list, tuple)):
        raise ValueError("DepthSplat mixture kernel-closure aggregate trace is invalid")
    source_only: dict[str, bool] | None = None
    strict_risk: float | None = None
    maximum_world: float | None = None
    maximum_source: float | None = None
    maximum_kernel: float | None = None
    maximum_log_depth_rms: float | None = None
    finite_risks: list[float] = []
    reasons: dict[str, int] = {}
    attempts = 0
    passed = 0
    unscorable = 0

    for record in tile_trace:
        if not isinstance(record, Mapping):
            raise ValueError("DepthSplat mixture kernel-closure aggregate tile is invalid")
        for closure_key, sha256_key, passed_key in (
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
        ):
            closure = record.get(closure_key)
            closure_sha256 = record.get(sha256_key)
            closure_passed = (
                record.get(passed_key)
                if passed_key is not None
                else closure.get("passed")
                if isinstance(closure, Mapping)
                else None
            )
            if closure is None and closure_sha256 is None and closure_passed is None:
                continue
            if (
                not isinstance(closure, Mapping)
                or not isinstance(closure_sha256, str)
                or closure_sha256 != _canonical_sha256(closure)
                or not isinstance(closure_passed, bool)
                or closure.get("schema_version")
                != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_SCHEMA_VERSION
                or closure.get("kind") != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND
                or closure.get("policy") != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY
                or closure.get("passed") is not closure_passed
                or not isinstance(closure.get("source_only"), Mapping)
                or not isinstance(closure.get("summary"), Mapping)
            ):
                raise ValueError("DepthSplat mixture kernel-closure aggregate evidence changed")
            current_source_only = dict(closure["source_only"])
            if (
                current_source_only.get("source_camera_only") is not True
                or current_source_only.get("target_mapping_present") is not False
                or current_source_only.get("target_rgb_accessed") is not False
                or current_source_only.get("target_camera_metadata_accessed") is not False
                or current_source_only.get("target_index_accessed") is not False
                or current_source_only.get("omitted_s3_attributes_accessed") is not False
                or current_source_only.get("boolean_owner_assignment_used") is not False
                or current_source_only.get("projected_domain_guard_used") is not False
                or current_source_only.get("covariance_expansion_used") is not False
                or current_source_only.get("alpha_union_used") is not False
            ):
                raise ValueError("DepthSplat mixture kernel-closure source-only flags changed")
            if source_only is None:
                source_only = current_source_only
            elif source_only != current_source_only:
                raise ValueError("DepthSplat mixture kernel-closure source-only flags drifted")
            summary = closure["summary"]
            input_valid = summary.get("input_valid")
            current_strict_risk = summary.get("strict_maximum_relative_risk")
            reason = summary.get("reason")
            if (
                not isinstance(input_valid, bool)
                or isinstance(current_strict_risk, bool)
                or not isinstance(current_strict_risk, (int, float))
                or not torch.isfinite(torch.tensor(float(current_strict_risk)))
                or float(current_strict_risk) < 0.0
                or reason is not None and not isinstance(reason, str)
            ):
                raise ValueError("DepthSplat mixture kernel-closure summary changed")
            binding = closure.get("binding")
            if input_valid and not isinstance(binding, Mapping):
                raise ValueError("DepthSplat mixture kernel-closure binding is missing")
            if not input_valid and binding is not None:
                raise ValueError("DepthSplat invalid kernel-closure binding changed")
            current_strict_risk = float(current_strict_risk)
            if strict_risk is None:
                strict_risk = current_strict_risk
            elif strict_risk != current_strict_risk:
                raise ValueError("DepthSplat mixture kernel-closure threshold drifted")
            if input_valid:
                values = {
                    "maximum_world_kernel_risk": summary.get("maximum_world_kernel_risk"),
                    "maximum_source_kernel_risk": summary.get("maximum_source_kernel_risk"),
                    "maximum_kernel_risk": summary.get("maximum_kernel_risk"),
                    "maximum_source_log_depth_rms": summary.get("maximum_source_log_depth_rms"),
                }
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not torch.isfinite(torch.tensor(float(value)))
                    or float(value) < 0.0
                    for value in values.values()
                ):
                    raise ValueError("DepthSplat mixture kernel-closure risks changed")
                world = float(values["maximum_world_kernel_risk"])
                source = float(values["maximum_source_kernel_risk"])
                kernel = float(values["maximum_kernel_risk"])
                depth_rms = float(values["maximum_source_log_depth_rms"])
                if kernel != max(world, source) or closure_passed != (kernel <= current_strict_risk):
                    raise ValueError("DepthSplat mixture kernel-closure decision changed")
                maximum_world = world if maximum_world is None else max(maximum_world, world)
                maximum_source = source if maximum_source is None else max(maximum_source, source)
                maximum_kernel = kernel if maximum_kernel is None else max(maximum_kernel, kernel)
                maximum_log_depth_rms = (
                    depth_rms
                    if maximum_log_depth_rms is None
                    else max(maximum_log_depth_rms, depth_rms)
                )
                finite_risks.append(kernel)
            elif closure_passed:
                raise ValueError("DepthSplat invalid kernel-closure evidence passed")
            else:
                unscorable += 1
            attempts += 1
            passed += int(closure_passed)
            reason_name = "accepted" if closure_passed else str(reason or "unscorable")
            reasons[reason_name] = reasons.get(reason_name, 0) + 1

    histogram_counts: list[int] = []
    lower = float("-inf")
    for upper in _MIXTURE_KERNEL_CLOSURE_HISTOGRAM_UPPER_BOUNDS:
        histogram_counts.append(sum(lower < value <= upper for value in finite_risks))
        lower = upper
    return {
        "schema": DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_AGGREGATE_SCHEMA,
        "kind": DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_KIND,
        "policy": DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY,
        "source_only": source_only,
        "strict_maximum_relative_risk": strict_risk,
        "tile_kernel_closure_attempt_count": attempts,
        "passed_tile_kernel_closure_count": passed,
        "failed_tile_kernel_closure_count": attempts - passed,
        "maximum_world_kernel_risk": maximum_world,
        "maximum_source_kernel_risk": maximum_source,
        "maximum_kernel_risk": maximum_kernel,
        "maximum_source_log_depth_rms": maximum_log_depth_rms,
        "maximum_kernel_risk_distribution": {
            "finite_tile_risk_count": len(finite_risks),
            "unscorable_tile_count": unscorable,
            "minimum_kernel_risk": min(finite_risks) if finite_risks else None,
            "histogram_upper_bounds": list(_MIXTURE_KERNEL_CLOSURE_HISTOGRAM_UPPER_BOUNDS),
            "histogram_counts": histogram_counts,
            "above_largest_histogram_bin_count": sum(
                value > _MIXTURE_KERNEL_CLOSURE_HISTOGRAM_UPPER_BOUNDS[-1]
                for value in finite_risks
            ),
        },
        "reason_counts": dict(sorted(reasons.items())),
    }


def _validate_mixture_kernel_closure_aggregate(
    *, events: Mapping[str, Any], tile_trace: Any
) -> dict[str, Any]:
    """Require the strict closure telemetry to rebuild from the tile trace."""

    expected = _mixture_kernel_closure_aggregate_from_trace(tile_trace)
    aggregate = events.get("mixture_kernel_closure_aggregate")
    if (
        not isinstance(aggregate, Mapping)
        or dict(aggregate) != expected
        or events.get("mixture_kernel_closure_aggregate_sha256")
        != _canonical_sha256(expected)
        or events.get("mixture_kernel_closure_guard_policy")
        != SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY
        or events.get("mixture_kernel_closure_evidence_policy")
        != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY
        or events.get("mixture_kernel_closure_strict_maximum_relative_risk")
        != expected["strict_maximum_relative_risk"]
    ):
        raise ValueError("DepthSplat mixture kernel-closure aggregate binding changed")
    return expected


def _build_fixed_scale_selected_only_tile_updates(
    *,
    packet: DepthSplatSparseRawPacket,
    packed: DepthSplatPackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    raw_feature_map: torch.Tensor,
    record: Mapping[str, Any],
    level: str,
    semantics: str,
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    feature_statistic: str,
    assignment_feature_semantics: str,
    coverage_enriched: bool,
    support_basis_profile: bool = False,
    soft_mixture_profile: bool = False,
    kernel_closure_profile: bool = False,
    conditional_anchor_transport: bool = False,
    kernel_closure_maximum_relative_risk: float | None = None,
    kernel_closure_measurement_cache: Any | None = None,
    routing_z_depth_map: torch.Tensor | None = None,
    source_get_world_rays: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Build selected-only fixed-scale moment updates for one compact tile."""

    if (
        level not in {"L0", "L1"}
        or feature_statistic
        not in {
            "raw-probe-mean-channel-variance",
            "normalized-probe-vector-standard-deviation",
        }
        or assignment_feature_semantics
        not in {"raw-bilinear-s1-v1", "unit-normalized-bilinear-s1-v1"}
        or (kernel_closure_profile and not soft_mixture_profile)
        or (
            conditional_anchor_transport
            and (
                coverage_enriched
                or support_basis_profile
                or soft_mixture_profile
                or kernel_closure_profile
            )
        )
        or (
            kernel_closure_maximum_relative_risk is not None
            and (
                isinstance(kernel_closure_maximum_relative_risk, bool)
                or not isinstance(kernel_closure_maximum_relative_risk, (int, float))
                or not torch.isfinite(
                    torch.tensor(float(kernel_closure_maximum_relative_risk))
                )
                or float(kernel_closure_maximum_relative_risk) <= 0.0
            )
        )
        or sum(
            bool(value)
            for value in (
                coverage_enriched,
                support_basis_profile,
                soft_mixture_profile,
            )
        ) > 1
    ):
        raise ValueError("DepthSplat fixed-scale tile does not have the required route")
    anchors = _route_anchor_positions(record, level=level, semantics=semantics)
    engineering_profile = (
        coverage_enriched
        or support_basis_profile
        or soft_mixture_profile
        or conditional_anchor_transport
    )
    projected_support_guard = coverage_enriched or support_basis_profile
    source_label = "selected-anchor" if engineering_profile else "selected-probe"
    if not engineering_profile and (
        semantics != PAPER_KP_ANCHOR_SEMANTICS
        or anchors != compute_probe_positions(4)
    ):
        raise ValueError("DepthSplat formal paper tile does not retain Kp=4 corners")
    if engineering_profile:
        expected = (
            compute_probe_positions(4)
            if level == "L0"
            else l1_local_positions_for_tile(
                record,
                tile_size=4,
                l1_anchor_semantics=BALANCED_L1_ANCHOR_SEMANTICS,
            )
        )
        if semantics != BALANCED_L1_ANCHOR_SEMANTICS or anchors != expected:
            raise ValueError("DepthSplat engineering tile anchor layout changed")
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=4,
        positions=anchors,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("DepthSplat fixed-scale tile lacks a selected anchor")
    anchor_indices = [slot_to_index[slot] for slot in anchor_slots]
    source_means = packed.means[anchor_indices]
    source_covariances = packed.covariances[anchor_indices]
    source_harmonics = packed.harmonics[anchor_indices]
    source_opacities = packed.opacities[anchor_indices]
    if bool((source_opacities >= 1.0).any()):
        raise _DepthSplatTileRejection(NATIVE_OPACITY_ENDPOINT_FULL_REASON)
    source_positions = [(tile_y * 4 + row, tile_x * 4 + column) for row, column in anchors]
    target_local = [
        position for position in _full_positions(4) if position not in set(anchors)
    ]
    if not target_local:
        return [], {
            "virtual_count": 0,
            "coverage": {
                "certificate": (
                    DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE
                    if support_basis_profile
                    else DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE
                    if soft_mixture_profile
                    else DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE
                    if coverage_enriched
                    else DEPTHSPLAT_LITERAL_PAPER_T4_MOMENT_CERTIFICATE
                ),
                "finite_psd_moment_merge": True,
                "support_containment_guard": projected_support_guard,
                "projected_support_guard": projected_support_guard,
                "moment_covariance_scale_max": 1.0,
                "per_update": [],
            },
            "virtual_geometry_source": f"{source_label}-only-none-v1",
            "virtual_attribute_source": f"{source_label}-only-none-v1",
            "omitted_routing_z_depth_reads": 0,
            **{
                (
                    "selected_anchor_attribute_reads"
                    if engineering_profile
                    else "selected_probe_attribute_reads"
                ): len(anchor_indices)
            },
            "opacity_compositing_order": "literal-weighted-average-no-alpha-union-v1",
        }
    target_positions = [
        (tile_y * 4 + row, tile_x * 4 + column) for row, column in target_local
    ]
    spatial = _spatial_weights(
        target_positions,
        source_positions,
        device=packed.means.device,
        dtype=packed.means.dtype,
    )
    virtual_mean_geometry = "selected-anchor-spatial-moment-v1"
    virtual_mean_source: torch.Tensor | None = None
    conditional_virtual_means: list[torch.Tensor] | None = None
    if conditional_anchor_transport:
        if (
            not torch.is_tensor(routing_z_depth_map)
            or routing_z_depth_map.shape != (height, width)
            or not bool(torch.isfinite(routing_z_depth_map).all())
            or not callable(source_get_world_rays)
        ):
            raise ValueError("DepthSplat conditional transport inputs are invalid")
        target_centres = _pixel_centres(
            target_positions,
            height=height,
            width=width,
            device=packed.means.device,
            dtype=packed.means.dtype,
        )
        anchor_centres = _pixel_centres(
            source_positions,
            height=height,
            width=width,
            device=packed.means.device,
            dtype=packed.means.dtype,
        )
        anchor_offsets = packet.coordinates[anchor_indices] - anchor_centres
        offset_limit = torch.tensor(
            (0.5 / width, 0.5 / height),
            device=anchor_offsets.device,
            dtype=anchor_offsets.dtype,
        )
        if (
            not bool(torch.isfinite(anchor_offsets).all())
            or bool((anchor_offsets.abs() > offset_limit + 2e-5).any())
        ):
            raise ValueError("DepthSplat conditional native offset transport is invalid")
        virtual_z_depths = torch.stack(
            [routing_z_depth_map[row, column] for row, column in target_positions]
        ).to(device=packed.means.device, dtype=packed.means.dtype)
        conditional_virtual_means = []
        for anchor_offset, index in enumerate(anchor_indices):
            coordinates = target_centres + anchor_offsets[anchor_offset].unsqueeze(0)
            if (
                not bool(torch.isfinite(coordinates).all())
                or bool((coordinates < 0.0).any())
                or bool((coordinates > 1.0).any())
            ):
                raise ValueError(
                    "DepthSplat conditional transported coordinate is invalid"
                )
            conditional_virtual_means.append(
                depthsplat_z_depth_world_means(
                    coordinates,
                    packet.extrinsics[index].unsqueeze(0).expand(
                        len(target_local), -1, -1
                    ),
                    packet.intrinsics[index].unsqueeze(0).expand(
                        len(target_local), -1, -1
                    ),
                    virtual_z_depths,
                    source_get_world_rays=source_get_world_rays,
                )
            )
        virtual_means = conditional_virtual_means[0]
        virtual_mean_geometry = (
            "per-receiving-anchor-source-z-depth-offset-transport-v1"
        )
    elif kernel_closure_profile and level == "L1":
        if (
            not torch.is_tensor(routing_z_depth_map)
            or routing_z_depth_map.shape != (height, width)
            or not bool(torch.isfinite(routing_z_depth_map).all())
            or not callable(source_get_world_rays)
        ):
            raise ValueError("DepthSplat L1 source z-depth mean inputs are invalid")
        virtual_coordinates = _pixel_centres(
            target_positions,
            height=height,
            width=width,
            device=packed.means.device,
            dtype=packed.means.dtype,
        )
        anchor_centres = _pixel_centres(
            source_positions,
            height=height,
            width=width,
            device=packed.means.device,
            dtype=packed.means.dtype,
        )
        anchor_offsets = packet.coordinates[anchor_indices] - anchor_centres
        offset_limit = torch.tensor(
            (0.5 / width, 0.5 / height),
            device=anchor_offsets.device,
            dtype=anchor_offsets.dtype,
        )
        if (
            not bool(torch.isfinite(anchor_offsets).all())
            or bool((anchor_offsets.abs() > offset_limit + 2e-5).any())
        ):
            raise ValueError("DepthSplat L1 native offset transport is invalid")
        virtual_coordinates = virtual_coordinates + spatial @ anchor_offsets
        if (
            not bool(torch.isfinite(virtual_coordinates).all())
            or bool((virtual_coordinates < 0.0).any())
            or bool((virtual_coordinates > 1.0).any())
        ):
            raise ValueError("DepthSplat L1 transported virtual coordinate is invalid")
        virtual_z_depths = torch.stack(
            [routing_z_depth_map[row, column] for row, column in target_positions]
        ).to(device=packed.means.device, dtype=packed.means.dtype)
        context_extrinsic = packet.extrinsics[anchor_indices[0]].unsqueeze(0)
        context_intrinsic = packet.intrinsics[anchor_indices[0]].unsqueeze(0)
        virtual_mean_source = depthsplat_z_depth_world_means(
            virtual_coordinates,
            context_extrinsic.expand(len(target_local), -1, -1),
            context_intrinsic.expand(len(target_local), -1, -1),
            virtual_z_depths,
            source_get_world_rays=source_get_world_rays,
        )
        virtual_means = virtual_mean_source
        virtual_mean_geometry = "source-z-depth-camera-ray-offset-transport-v2"
    else:
        virtual_means = spatial @ source_means
    mean_deltas = source_means.unsqueeze(0) - virtual_means.unsqueeze(1)
    virtual_covariances = (
        spatial.reshape(spatial.shape[0], spatial.shape[1], 1, 1)
        * (
            source_covariances.unsqueeze(0)
            + mean_deltas.unsqueeze(3) @ mean_deltas.unsqueeze(2)
        )
    ).sum(dim=1)
    virtual_covariances = (virtual_covariances + virtual_covariances.mT) * 0.5
    virtual_harmonics = (
        spatial @ source_harmonics.reshape(len(anchors), -1)
    ).reshape(len(target_local), *source_harmonics.shape[1:])
    virtual_opacities = spatial @ source_opacities
    if (
        not bool(torch.isfinite(virtual_means).all())
        or not bool(torch.isfinite(virtual_covariances).all())
        or not bool(torch.isfinite(virtual_harmonics).all())
        or not bool(torch.isfinite(virtual_opacities).all())
        or bool((virtual_opacities < 0.0).any())
        or bool((virtual_opacities >= 1.0).any())
        or bool((torch.linalg.eigvalsh(virtual_covariances) < -1e-6).any())
    ):
        raise ValueError("DepthSplat fixed-scale selected-only virtual field is invalid")
    anchor_features = torch.stack(
        [raw_feature_map[:, row, column] for row, column in source_positions]
    )
    feature_variance = _tile_feature_variance(record, feature_statistic)
    assignments: list[torch.Tensor] = []
    for local_row, local_column in target_local:
        target_feature = raw_feature_map[
            :, tile_y * 4 + local_row, tile_x * 4 + local_column
        ]
        spatial_distances = torch.tensor(
            [
                ((local_row - row) / 3.0) ** 2
                + ((local_column - column) / 3.0) ** 2
                for row, column in anchors
            ],
            device=packed.means.device,
            dtype=packed.means.dtype,
        )
        feature_distances = (anchor_features - target_feature.unsqueeze(0)).square().sum(dim=1)
        kwargs: dict[str, Any] = {}
        if level == "L1":
            kwargs = {
                "probe_depths": packet.depths[anchor_indices],
                "depth_reference_depths": packet.depths[
                    anchor_indices[: len(compute_probe_positions(4))]
                ],
                "beta_d": 1.0,
            }
        weights = paper_assignment_weights(
            spatial_distances,
            feature_distances.to(dtype=packed.means.dtype),
            feature_variance=feature_variance,
            beta_x=0.5,
            beta_f=0.1,
            level=level,
            **kwargs,
        )
        if not bool(torch.isfinite(weights).all()) or not torch.allclose(
            weights.sum(),
            torch.ones((), device=weights.device, dtype=weights.dtype),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError("DepthSplat fixed-scale assignment is invalid")
        assignments.append(weights)
    assignment = torch.stack(assignments)
    assignment_transport = "bilateral-soft-moment-v1"
    updates: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    coverage_updates: list[dict[str, Any]] = []
    minimum_merged_eigenvalues: list[float] = []
    minimum_virtual_eigenvalues: list[float] = []
    for anchor_offset, (slot, index) in enumerate(zip(anchor_slots, anchor_indices)):
        masses = torch.cat(
            (
                torch.ones(1, device=packed.means.device, dtype=packed.means.dtype),
                assignment[:, anchor_offset],
            )
        )
        mass_sum = masses.sum()
        if not bool(torch.isfinite(mass_sum)) or bool(
            mass_sum <= torch.finfo(masses.dtype).eps
        ):
            raise ValueError("DepthSplat fixed-scale moment mass is invalid")
        if conditional_anchor_transport:
            if conditional_virtual_means is None:
                raise RuntimeError("DepthSplat conditional transport state is missing")
            update_virtual_means = conditional_virtual_means[anchor_offset]
            update_virtual_covariances = source_covariances[anchor_offset].unsqueeze(
                0
            ).expand(len(target_local), -1, -1)
            contributors = torch.cat(
                (source_means[anchor_offset].unsqueeze(0), update_virtual_means),
                dim=0,
            )
            contributor_covariances = torch.cat(
                (
                    source_covariances[anchor_offset].unsqueeze(0),
                    update_virtual_covariances,
                ),
                dim=0,
            )
            contributor_harmonics = torch.cat(
                (
                    source_harmonics[anchor_offset].unsqueeze(0),
                    source_harmonics[anchor_offset]
                    .unsqueeze(0)
                    .expand(len(target_local), -1, -1),
                ),
                dim=0,
            )
            contributor_opacities = torch.cat(
                (
                    source_opacities[anchor_offset].reshape(1),
                    source_opacities[anchor_offset]
                    .expand(len(target_local)),
                ),
                dim=0,
            )
        else:
            update_virtual_means = virtual_means
            update_virtual_covariances = virtual_covariances
            contributors = torch.cat(
                (source_means[anchor_offset].unsqueeze(0), virtual_means), dim=0
            )
            contributor_covariances = torch.cat(
                (
                    source_covariances[anchor_offset].unsqueeze(0),
                    virtual_covariances,
                ),
                dim=0,
            )
            contributor_harmonics = torch.cat(
                (source_harmonics[anchor_offset].unsqueeze(0), virtual_harmonics),
                dim=0,
            )
            contributor_opacities = torch.cat(
                (source_opacities[anchor_offset].reshape(1), virtual_opacities),
                dim=0,
            )
        merged_mean = (masses.unsqueeze(1) * contributors).sum(dim=0) / mass_sum
        deltas = contributors - merged_mean.unsqueeze(0)
        merged_covariance = (
            masses.reshape(-1, 1, 1)
            * (contributor_covariances + deltas.unsqueeze(2) @ deltas.unsqueeze(1))
        ).sum(dim=0) / mass_sum
        merged_covariance = (merged_covariance + merged_covariance.mT) * 0.5
        merged_harmonics = torch.einsum(
            "n,ncd->cd", masses, contributor_harmonics
        ) / mass_sum
        merged_opacity = torch.dot(masses, contributor_opacities) / mass_sum
        if (
            not bool(torch.isfinite(merged_mean).all())
            or not bool(torch.isfinite(merged_covariance).all())
            or not bool(torch.isfinite(merged_harmonics).all())
            or not bool(torch.isfinite(merged_opacity))
            or bool((torch.linalg.eigvalsh(merged_covariance) < -1e-6).any())
            or bool(merged_opacity < source_opacities.min() - 1e-6)
            or bool(merged_opacity > source_opacities.max() + 1e-6)
            or bool((merged_harmonics < source_harmonics.amin(dim=0) - 1e-5).any())
            or bool((merged_harmonics > source_harmonics.amax(dim=0) + 1e-5).any())
        ):
            raise ValueError("DepthSplat fixed-scale attribute average is invalid")
        coverage = _literal_finite_psd_moment_merge(
            merged_covariance,
            merged_mean,
            update_virtual_means,
            update_virtual_covariances,
        )
        updates.append((slot, merged_mean, merged_covariance, merged_harmonics, merged_opacity))
        minimum_merged_eigenvalues.append(
            float(coverage["minimum_merged_covariance_eigenvalue"])
        )
        minimum_virtual_eigenvalues.append(
            float(coverage["minimum_virtual_covariance_eigenvalue"])
        )
        coverage_updates.append(
            {
                "update_dense_slot": int(slot),
                "virtual_count": len(target_local),
                **coverage,
                "support_containment_guard": projected_support_guard,
                "projected_support_guard": projected_support_guard,
            }
        )
    owner_support: dict[str, Any] | None = None
    tile_support_basis: dict[str, Any] | None = None
    support_basis_sha256: str | None = None
    tile_soft_mixture: dict[str, Any] | None = None
    soft_mixture_sha256: str | None = None
    tile_kernel_closure: dict[str, Any] | None = None
    kernel_closure_sha256: str | None = None
    if coverage_enriched:
        virtual_owner_indices = assignment.argmax(dim=1).to(dtype=torch.int64)
        owner_support = audit_depthsplat_owner_coverage(
            virtual_means=virtual_means,
            virtual_covariances=virtual_covariances,
            virtual_opacities=virtual_opacities,
            virtual_owner_indices=virtual_owner_indices,
            owner_source_means=source_means,
            owner_source_covariances=source_covariances,
            owner_source_opacities=source_opacities,
            merged_means=torch.stack([item[1] for item in updates]),
            merged_covariances=torch.stack([item[2] for item in updates]),
            merged_opacities=torch.stack([item[4] for item in updates]),
            owner_source_extrinsics=packet.extrinsics[anchor_indices],
            owner_source_intrinsics=packet.intrinsics[anchor_indices],
        )
        owner_support = _validate_coverage_enriched_owner_support(
            owner_support,
            owner_count=len(updates),
            virtual_count=len(target_local),
            virtual_owner_indices=virtual_owner_indices,
        )
        owner_records = owner_support["owners"]
        if len(owner_records) != len(coverage_updates):
            raise RuntimeError("DepthSplat coverage-enriched owner evidence is incomplete")
        ownership = owner_support["ownership"]
        coverage_updates = [
            {
                **row,
                "owner_support": dict(owner_records[index]),
                "owner_support_schema_version": owner_support["schema_version"],
                "owner_support_kind": owner_support["kind"],
                "owner_assignment_policy": ownership["policy"],
                "virtual_owner_indices_sha256": ownership[
                    "virtual_owner_indices_sha256"
                ],
                "owner_assignment_counts_sha256": _canonical_sha256(
                    ownership["owner_assignment_counts"]
                ),
            }
            for index, row in enumerate(coverage_updates)
        ]
    elif support_basis_profile:
        virtual_origin_slots = _tile_slots(
            view=view,
            tile_y=tile_y,
            tile_x=tile_x,
            height=height,
            width=width,
            tile_size=4,
            positions=target_local,
        )
        tile_support_basis = audit_depthsplat_tile_support_basis(
            virtual_means=virtual_means,
            virtual_covariances=virtual_covariances,
            virtual_opacities=virtual_opacities,
            virtual_origin_slots=torch.tensor(
                virtual_origin_slots,
                device=packed.means.device,
                dtype=torch.int64,
            ),
            virtual_source_spatial_weights=spatial,
            bilateral_assignment_weights=assignment,
            anchor_source_means=source_means,
            anchor_source_covariances=source_covariances,
            anchor_source_opacities=source_opacities,
            anchor_dense_slots=torch.tensor(
                anchor_slots,
                device=packed.means.device,
                dtype=torch.int64,
            ),
            merged_means=torch.stack([item[1] for item in updates]),
            merged_covariances=torch.stack([item[2] for item in updates]),
            merged_opacities=torch.stack([item[4] for item in updates]),
            context_extrinsics=packet.extrinsics[anchor_indices],
            context_intrinsics=packet.intrinsics[anchor_indices],
        )
        tile_support_basis = _validate_support_basis_evidence(
            tile_support_basis,
            anchor_slots=anchor_slots,
            virtual_origin_slots=virtual_origin_slots,
            virtual_means=virtual_means,
            virtual_covariances=virtual_covariances,
            virtual_opacities=virtual_opacities,
            virtual_source_spatial_weights=spatial,
            bilateral_assignment_weights=assignment,
            anchor_source_means=source_means,
            anchor_source_covariances=source_covariances,
            anchor_source_opacities=source_opacities,
            merged_means=torch.stack([item[1] for item in updates]),
            merged_covariances=torch.stack([item[2] for item in updates]),
            merged_opacities=torch.stack([item[4] for item in updates]),
            context_extrinsics=packet.extrinsics[anchor_indices],
            context_intrinsics=packet.intrinsics[anchor_indices],
        )
        support_basis_sha256 = _canonical_sha256(tile_support_basis)
        coverage_updates = [
            {
                **row,
                "support_basis_schema_version": tile_support_basis["schema_version"],
                "support_basis_kind": tile_support_basis["kind"],
                "support_basis_policy": tile_support_basis["support_basis_policy"],
                "support_basis_sha256": support_basis_sha256,
                "support_basis_passed": tile_support_basis["passed"],
            }
            for row in coverage_updates
        ]
    elif soft_mixture_profile:
        virtual_origin_slots = _tile_slots(
            view=view,
            tile_y=tile_y,
            tile_x=tile_x,
            height=height,
            width=width,
            tile_size=4,
            positions=target_local,
        )
        tile_soft_mixture = certify_depthsplat_tile_soft_mixture(
            tile_key=(view, tile_y, tile_x),
            anchor_dense_slots=torch.tensor(
                anchor_slots, device=packed.means.device, dtype=torch.int64
            ),
            virtual_origin_slots=torch.tensor(
                virtual_origin_slots, device=packed.means.device, dtype=torch.int64
            ),
            spatial_weights=spatial,
            bilateral_assignment_weights=assignment,
            anchor_source_means=source_means,
            anchor_source_covariances=source_covariances,
            anchor_source_harmonics=source_harmonics,
            anchor_source_opacities=source_opacities,
            virtual_means=virtual_means,
            virtual_covariances=virtual_covariances,
            virtual_harmonics=virtual_harmonics,
            virtual_opacities=virtual_opacities,
            merged_means=torch.stack([item[1] for item in updates]),
            merged_covariances=torch.stack([item[2] for item in updates]),
            merged_harmonics=torch.stack([item[3] for item in updates]),
            merged_opacities=torch.stack([item[4] for item in updates]),
            context_extrinsics=packet.extrinsics[anchor_indices],
            context_intrinsics=packet.intrinsics[anchor_indices],
            virtual_mean_geometry=virtual_mean_geometry,
            virtual_mean_source=virtual_mean_source,
        )
        tile_soft_mixture = _validate_soft_mixture_evidence(
            tile_soft_mixture,
            tile_key=(view, tile_y, tile_x),
            anchor_dense_slots=torch.tensor(
                anchor_slots, device=packed.means.device, dtype=torch.int64
            ),
            virtual_origin_slots=torch.tensor(
                virtual_origin_slots, device=packed.means.device, dtype=torch.int64
            ),
            spatial_weights=spatial,
            bilateral_assignment_weights=assignment,
            anchor_source_means=source_means,
            anchor_source_covariances=source_covariances,
            anchor_source_harmonics=source_harmonics,
            anchor_source_opacities=source_opacities,
            virtual_means=virtual_means,
            virtual_covariances=virtual_covariances,
            virtual_harmonics=virtual_harmonics,
            virtual_opacities=virtual_opacities,
            merged_means=torch.stack([item[1] for item in updates]),
            merged_covariances=torch.stack([item[2] for item in updates]),
            merged_harmonics=torch.stack([item[3] for item in updates]),
            merged_opacities=torch.stack([item[4] for item in updates]),
            context_extrinsics=packet.extrinsics[anchor_indices],
            context_intrinsics=packet.intrinsics[anchor_indices],
            virtual_mean_geometry=virtual_mean_geometry,
            virtual_mean_source=virtual_mean_source,
        )
        soft_mixture_sha256 = _canonical_sha256(tile_soft_mixture)
        if kernel_closure_profile and tile_soft_mixture["passed"] is True:
            tile_kernel_closure = assess_depthsplat_tile_kernel_closure(
                tile_key=(view, tile_y, tile_x),
                anchor_dense_slots=torch.tensor(
                    anchor_slots, device=packed.means.device, dtype=torch.int64
                ),
                virtual_origin_slots=torch.tensor(
                    virtual_origin_slots, device=packed.means.device, dtype=torch.int64
                ),
                bilateral_assignment_weights=assignment,
                anchor_source_means=source_means,
                anchor_source_covariances=source_covariances,
                anchor_source_opacities=source_opacities,
                virtual_means=virtual_means,
                virtual_covariances=virtual_covariances,
                virtual_opacities=virtual_opacities,
                merged_means=torch.stack([item[1] for item in updates]),
                merged_covariances=torch.stack([item[2] for item in updates]),
                merged_opacities=torch.stack([item[4] for item in updates]),
                context_extrinsics=packet.extrinsics[anchor_indices],
                context_intrinsics=packet.intrinsics[anchor_indices],
                strict_maximum_relative_risk=kernel_closure_maximum_relative_risk,
                measurement_cache=kernel_closure_measurement_cache,
            )
            tile_kernel_closure = _validate_mixture_kernel_closure_evidence(
                tile_kernel_closure,
                tile_key=(view, tile_y, tile_x),
                anchor_dense_slots=torch.tensor(
                    anchor_slots, device=packed.means.device, dtype=torch.int64
                ),
                virtual_origin_slots=torch.tensor(
                    virtual_origin_slots, device=packed.means.device, dtype=torch.int64
                ),
                bilateral_assignment_weights=assignment,
                anchor_source_means=source_means,
                anchor_source_covariances=source_covariances,
                anchor_source_opacities=source_opacities,
                virtual_means=virtual_means,
                virtual_covariances=virtual_covariances,
                virtual_opacities=virtual_opacities,
                merged_means=torch.stack([item[1] for item in updates]),
                merged_covariances=torch.stack([item[2] for item in updates]),
                merged_opacities=torch.stack([item[4] for item in updates]),
                context_extrinsics=packet.extrinsics[anchor_indices],
                context_intrinsics=packet.intrinsics[anchor_indices],
                strict_maximum_relative_risk=kernel_closure_maximum_relative_risk,
                measurement_cache=kernel_closure_measurement_cache,
            )
            kernel_closure_sha256 = _canonical_sha256(tile_kernel_closure)
        coverage_updates = [
            {
                **row,
                "soft_mixture_certificate_schema_version": tile_soft_mixture[
                    "schema_version"
                ],
                "soft_mixture_certificate_kind": tile_soft_mixture["kind"],
                "soft_mixture_certificate_policy": tile_soft_mixture["policy"],
                "soft_mixture_certificate_sha256": soft_mixture_sha256,
                "soft_mixture_certificate_passed": tile_soft_mixture["passed"],
                "projected_support_guard": False,
            }
            for row in coverage_updates
        ]
    return updates, {
        "virtual_count": len(target_local),
        "virtual_mean_geometry": virtual_mean_geometry,
        "coverage": {
            "certificate": (
                DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE
                if support_basis_profile
                else DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE
                if soft_mixture_profile
                else DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE
                if coverage_enriched
                else DEPTHSPLAT_LITERAL_PAPER_T4_MOMENT_CERTIFICATE
            ),
            "finite_psd_moment_merge": True,
            "support_containment_guard": projected_support_guard,
            "projected_support_guard": projected_support_guard,
            "moment_covariance_scale_max": 1.0,
            "minimum_merged_covariance_eigenvalue": min(
                minimum_merged_eigenvalues, default=0.0
            ),
            "minimum_virtual_covariance_eigenvalue": min(
                minimum_virtual_eigenvalues, default=0.0
            ),
            "per_update": coverage_updates,
            "owner_support": owner_support,
            "support_basis": tile_support_basis,
            "support_basis_sha256": support_basis_sha256,
        },
        "virtual_geometry_source": (
            virtual_mean_geometry
            if conditional_anchor_transport
            else f"{source_label}-gaussian-spatial-moment-v1"
        ),
        "virtual_attribute_source": (
            "per-receiving-anchor-source-attribute-transport-v1"
            if conditional_anchor_transport
            else f"{source_label}-gaussian-spatial-linear-v1"
        ),
        "omitted_routing_z_depth_reads": 0,
        **{
            (
                "selected_anchor_attribute_reads"
                if engineering_profile
                else "selected_probe_attribute_reads"
            ): len(anchor_indices)
        },
        "opacity_compositing_order": "literal-weighted-average-no-alpha-union-v1",
        "assignment_feature_semantics": assignment_feature_semantics,
        "assignment_transport": assignment_transport,
        "soft_mixture_certificate": tile_soft_mixture,
        "soft_mixture_certificate_sha256": soft_mixture_sha256,
        "soft_mixture_certificate_passed": (
            tile_soft_mixture["passed"] if tile_soft_mixture is not None else None
        ),
        "mixture_kernel_closure": tile_kernel_closure,
        "mixture_kernel_closure_sha256": kernel_closure_sha256,
        "mixture_kernel_closure_passed": (
            tile_kernel_closure["passed"] if tile_kernel_closure is not None else None
        ),
    }


def preflight_depthsplat_l0_l1_materialization(
    initial_packet: DepthSplatSparseRawPacket,
    initial_packed: DepthSplatPackedGaussianAttributes,
    plan: IncrementalProbeFirstPlan,
    routing_features: torch.Tensor,
    routing_z_depths: torch.Tensor,
    *,
    source_sample_image_grid: Callable[..., tuple[Any, Any]],
    source_get_world_rays: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    maximum_feature_relative_residual: float = DEPTHSPLAT_FEATURE_INTERPOLATION_MAX_RELATIVE_RESIDUAL,
    maximum_coverage_covariance_scale: float = DEPTHSPLAT_COMPACT_COVERAGE_MAX_COVARIANCE_SCALE,
    execution_profile: str = DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
    selected_anchor_attribute_loo_frozen_guard: Any | None = None,
    selected_anchor_attribute_loo_maximum_risk: float | None = None,
    collect_selected_anchor_attribute_loo_risk: bool = False,
    mixture_kernel_closure_frozen_guard: Any | None = None,
    mixture_kernel_closure_maximum_relative_risk: float | None = None,
    mixture_kernel_closure_measurement_cache: Any | None = None,
    route_isolation: str = "l0_l1",
) -> DepthSplatCompactMaterializationPreflight:
    """Construct all nonzero L0/L1 updates without reading skipped attributes.

    A runtime LOO threshold must arrive as a projection of a verified frozen
    ACID V16 record.  The legacy scalar is deliberately rejected so a DL3DV
    audit cannot quietly substitute an arbitrary value for that record.
    """

    if (
        not isinstance(maximum_feature_relative_residual, (int, float))
        or isinstance(maximum_feature_relative_residual, bool)
        or maximum_feature_relative_residual < 0.0
        or not isinstance(maximum_coverage_covariance_scale, (int, float))
        or isinstance(maximum_coverage_covariance_scale, bool)
        or maximum_coverage_covariance_scale < 1.0
        or not isinstance(collect_selected_anchor_attribute_loo_risk, bool)
        or execution_profile
        not in {
            DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
        }
        or route_isolation not in {"l0_l1", "l0_only", "l1_only", "l0_to_l1"}
    ):
        raise ValueError("DepthSplat materializer safety thresholds are invalid")
    if selected_anchor_attribute_loo_maximum_risk is not None:
        raise ValueError(
            "DepthSplat selected-anchor LOO requires a frozen V16 guard, not a scalar threshold"
        )
    frozen_loo_guard = _selected_anchor_attribute_loo_frozen_guard(
        selected_anchor_attribute_loo_frozen_guard,
        require_literal_t4_authenticated=(
            execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        ),
    )
    frozen_kernel_guard = _mixture_kernel_closure_frozen_guard(
        mixture_kernel_closure_frozen_guard,
        require_authenticated=(
            execution_profile
            == DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
            and mixture_kernel_closure_frozen_guard is not None
        ),
    )
    if (
        frozen_kernel_guard is not None
        and execution_profile
        != DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
    ):
        raise ValueError("DepthSplat kernel-risk guard requires the v3 profile")
    if mixture_kernel_closure_maximum_relative_risk is not None:
        if (
            frozen_kernel_guard is not None
            or execution_profile
            != DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
            or isinstance(mixture_kernel_closure_maximum_relative_risk, bool)
            or not isinstance(
                mixture_kernel_closure_maximum_relative_risk, (int, float)
            )
            or not torch.isfinite(
                torch.tensor(float(mixture_kernel_closure_maximum_relative_risk))
            )
            or float(mixture_kernel_closure_maximum_relative_risk) <= 0.0
        ):
            raise ValueError("DepthSplat direct kernel-risk threshold is invalid")
    collect_loo = collect_selected_anchor_attribute_loo_risk or frozen_loo_guard is not None
    views, height, width, semantics = _require_plan(plan)
    _validate_execution_profile_plan_binding(
        plan, execution_profile=execution_profile
    )
    if execution_profile in {
        DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
    }:
        if float(maximum_coverage_covariance_scale) != 1.0:
            raise ValueError(
                "DepthSplat fixed-scale profile forbids covariance expansion"
            )
    if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
        _validate_literal_paper_t4_plan(
            plan,
            views=views,
            height=height,
            width=width,
            semantics=semantics,
        )
    elif execution_profile == DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE:
        _validate_coverage_enriched_t4_plan(
            plan,
            views=views,
            height=height,
            width=width,
            semantics=semantics,
        )
    elif execution_profile == DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE:
        _validate_support_basis_t4_plan(
            plan,
            views=views,
            height=height,
            width=width,
            semantics=semantics,
        )
    elif execution_profile == DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE:
        _validate_soft_mixture_t4_plan(
            plan,
            views=views,
            height=height,
            width=width,
            semantics=semantics,
        )
    elif execution_profile in {
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
    }:
        _validate_soft_mixture_normalized_t4_plan(
            plan,
            views=views,
            height=height,
            width=width,
            semantics=semantics,
        )
    elif execution_profile == DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE:
        _validate_soft_mixture_kernel_closure_t4_plan(
            plan,
            views=views,
            height=height,
            width=width,
            semantics=semantics,
        )
        if frozen_kernel_guard is not None:
            expected_route_config_sha256 = soft_mixture_kernel_closure_t4_route_config_sha256(
                plan.events
            )
            if (
                frozen_kernel_guard["materialization_profile"] != execution_profile
                or frozen_kernel_guard["route_plan_contract"]
                != plan.events.get("contract_version")
                or frozen_kernel_guard["route_plan_config_sha256"]
                != expected_route_config_sha256
                or plan.events.get("soft_mixture_kernel_closure_t4_route_config_sha256")
                != expected_route_config_sha256
            ):
                raise ValueError("DepthSplat kernel-risk guard route binding changed")
    if frozen_loo_guard is not None:
        if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
            expected_route_config_sha256 = literal_paper_t4_route_config_sha256(
                plan.events
            )
            if (
                frozen_loo_guard["frozen_record_kind"]
                != DEPTHSPLAT_LITERAL_PAPER_T4_V16_RECORD_KIND
                or frozen_loo_guard["materialization_profile"]
                != DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
                or frozen_loo_guard["route_plan_contract"]
                != LITERAL_PAPER_T4_PLAN_CONTRACT
                or frozen_loo_guard["route_plan_contract"]
                != plan.events.get("contract_version")
                or frozen_loo_guard["route_plan_config_sha256"]
                != expected_route_config_sha256
                or plan.events.get("literal_paper_t4_route_config_sha256")
                != expected_route_config_sha256
            ):
                raise ValueError("DepthSplat formal V16L/T4 LOO guard binding changed")
        if frozen_loo_guard["materialization_profile"] != execution_profile:
            raise ValueError("DepthSplat selected-anchor LOO guard profile changed")
        if frozen_loo_guard["route_plan_contract"] != plan.events.get(
            "contract_version"
        ):
            raise ValueError("DepthSplat selected-anchor LOO guard route contract changed")
    slot_to_index = _validate_source(
        initial_packet, initial_packed, plan, views=views, height=height, width=width
    )
    assignment_features, assignment_feature_semantics = _validate_routing_inputs(
        routing_features,
        routing_z_depths,
        initial_packet,
        plan,
        views=views,
        height=height,
        width=width,
        execution_profile=execution_profile,
    )
    source_grid = (
        _source_image_grid(
            source_sample_image_grid,
            height=height,
            width=width,
            device=initial_packet.raw_head_descriptors.device,
        )
        if execution_profile
        in {
            DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
        }
        else None
    )
    plan_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in plan.tile_trace
    }
    if len(plan_records) != views * (height // 4) * (width // 4):
        raise ValueError("DepthSplat materializer plan trace has duplicate tiles")
    updates: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    promote = torch.zeros_like(plan.selection_mask)
    trace: list[dict[str, Any]] = []
    rejection_reasons: dict[str, int] = {}
    feature_statistic = str(plan.events["feature_statistic"])

    if (
        execution_profile
        in {
            DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
        }
        and collect_loo
    ):
        raise ValueError(
            "DepthSplat coverage-enriched profile requires its own frozen attribute guard"
        )

    def build_compact_tile(
        *,
        record: Mapping[str, Any],
        level: str,
        view: int,
        tile_y: int,
        tile_x: int,
    ) -> tuple[
        list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        dict[str, Any],
    ]:
        if execution_profile in {
            DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
        }:
            return _build_fixed_scale_selected_only_tile_updates(
                packet=initial_packet,
                packed=initial_packed,
                slot_to_index=slot_to_index,
                raw_feature_map=assignment_features[view],
                record=record,
                level=level,
                semantics=semantics,
                view=view,
                tile_y=tile_y,
                tile_x=tile_x,
                height=height,
                width=width,
                feature_statistic=feature_statistic,
                assignment_feature_semantics=assignment_feature_semantics,
                coverage_enriched=(
                    execution_profile
                    == DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE
                ),
                support_basis_profile=(
                    execution_profile
                    == DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE
                ),
                soft_mixture_profile=_is_soft_mixture_profile(execution_profile),
                kernel_closure_profile=_is_kernel_closure_profile(execution_profile),
                conditional_anchor_transport=_is_direct_conditional_profile(
                    execution_profile
                ),
                kernel_closure_maximum_relative_risk=(
                    frozen_kernel_guard["threshold_value"]
                    if frozen_kernel_guard is not None
                    else mixture_kernel_closure_maximum_relative_risk
                ),
                kernel_closure_measurement_cache=(
                    mixture_kernel_closure_measurement_cache
                ),
                routing_z_depth_map=routing_z_depths[0, view],
                source_get_world_rays=source_get_world_rays,
            )
        if source_grid is None:
            raise RuntimeError("DepthSplat development source grid is missing")
        tile_updates, evidence = _build_tile_updates(
            packet=initial_packet,
            packed=initial_packed,
            slot_to_index=slot_to_index,
            feature_map=assignment_features[view],
            z_depth_map=routing_z_depths[0, view],
            record=record,
            level=level,
            semantics=semantics,
            view=view,
            tile_y=tile_y,
            tile_x=tile_x,
            height=height,
            width=width,
            feature_statistic=feature_statistic,
            source_grid=source_grid,
            source_get_world_rays=source_get_world_rays,
            maximum_feature_relative_residual=float(maximum_feature_relative_residual),
            maximum_coverage_covariance_scale=float(maximum_coverage_covariance_scale),
        )
        if _is_direct_conditional_profile(execution_profile):
            evidence = {
                **evidence,
                "assignment_feature_semantics": assignment_feature_semantics,
                "assignment_transport": "bilateral-soft-moment-v1",
            }
        return tile_updates, evidence

    def l1_prefetch_is_available(
        *, record: Mapping[str, Any], view: int, tile_y: int, tile_x: int
    ) -> bool:
        positions = _route_anchor_positions(record, level="L1", semantics=semantics)
        slots = _tile_slots(
            view=view,
            tile_y=tile_y,
            tile_x=tile_x,
            height=height,
            width=width,
            tile_size=4,
            positions=positions,
        )
        return all(slot in slot_to_index for slot in slots)

    def soft_mixture_candidate_failure(
        evidence: Mapping[str, Any], *, level: str, entry: dict[str, Any]
    ) -> str | None:
        """Record the first source-only compact guard that rejected one level."""

        prefix = level.lower()
        certificate = evidence.get("soft_mixture_certificate")
        certificate_sha256 = evidence.get("soft_mixture_certificate_sha256")
        certificate_passed = evidence.get("soft_mixture_certificate_passed")
        if (
            not isinstance(certificate, Mapping)
            or not isinstance(certificate_sha256, str)
            or certificate_sha256 != _canonical_sha256(certificate)
            or not isinstance(certificate_passed, bool)
            or certificate.get("passed") is not certificate_passed
        ):
            raise ValueError("DepthSplat soft-mixture certificate trace binding changed")
        if certificate_passed is False:
            entry[f"{prefix}_soft_mixture_failure"] = certificate
            entry[f"{prefix}_soft_mixture_failure_sha256"] = certificate_sha256
            return "soft-mixture replay"
        if not _is_kernel_closure_profile(execution_profile):
            return None
        entry[f"{prefix}_soft_mixture_certificate_before_kernel_closure"] = certificate
        entry[f"{prefix}_soft_mixture_certificate_before_kernel_closure_sha256"] = (
            certificate_sha256
        )
        closure = evidence.get("mixture_kernel_closure")
        closure_sha256 = evidence.get("mixture_kernel_closure_sha256")
        closure_passed = evidence.get("mixture_kernel_closure_passed")
        if (
            not isinstance(closure, Mapping)
            or not isinstance(closure_sha256, str)
            or closure_sha256 != _canonical_sha256(closure)
            or not isinstance(closure_passed, bool)
            or closure.get("passed") is not closure_passed
        ):
            raise ValueError("DepthSplat mixture kernel-closure trace binding changed")
        if closure_passed is False:
            entry[f"{prefix}_mixture_kernel_closure_failure"] = closure
            entry[f"{prefix}_mixture_kernel_closure_failure_sha256"] = closure_sha256
            return "mixture kernel-closure"
        return None

    for view in range(views):
        for tile_y in range(height // 4):
            for tile_x in range(width // 4):
                record = plan_records[(view, tile_y, tile_x)]
                level = record.get("pre_guard_route")
                if level not in {"L0", "L1", "Full"}:
                    raise ValueError("DepthSplat materializer plan route is invalid")
                entry: dict[str, Any] = {
                    "view": view,
                    "tile_y": tile_y,
                    "tile_x": tile_x,
                    "planned_route": level,
                    "attempted": level in {"L0", "L1"},
                    "accepted": level == "Full",
                    "accepted_level": "Full" if level == "Full" else None,
                    "source_nonprobe_s3_attribute_reads": 0,
                    "selected_anchor_attribute_loo": None,
                }
                if level == "Full":
                    entry["reason"] = "source_full_passthrough"
                    trace.append(entry)
                    continue
                if (
                    route_isolation == "l1_only" and level == "L0"
                ) or (route_isolation == "l0_only" and level == "L1"):
                    reason = f"route-isolation-{level.lower()}-forced-full"
                    _mark_tile(
                        promote, view=view, tile_y=tile_y, tile_x=tile_x, tile_size=4
                    )
                    entry.update({"accepted": False, "reason": reason})
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    trace.append(entry)
                    continue
                candidate_level = level
                if route_isolation == "l0_to_l1" and level == "L0":
                    if (
                        record.get("depth_uniform") is not True
                        or not l1_prefetch_is_available(
                            record=record,
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                        )
                    ):
                        reason = "route-isolation-l0-to-l1-prefetch-unavailable-forced-full"
                        _mark_tile(
                            promote,
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            tile_size=4,
                        )
                        entry.update({"accepted": False, "reason": reason})
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                        trace.append(entry)
                        continue
                    candidate_level = "L1"
                    entry["l0_to_l1_coordinated"] = True
                try:
                    if collect_loo:
                        endpoint_anchor_count = _selected_anchor_opacity_endpoint_count(
                            packed=initial_packed,
                            slot_to_index=slot_to_index,
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            height=height,
                            width=width,
                            level=str(candidate_level),
                            semantics=semantics,
                            plan_record=record,
                        )
                        if endpoint_anchor_count:
                            loo = _selected_anchor_attribute_loo_unscorable_certificate(
                                level=str(candidate_level),
                                semantics=semantics,
                                plan_record=record,
                                endpoint_anchor_count=endpoint_anchor_count,
                            )
                            entry["selected_anchor_attribute_loo"] = {
                                **loo,
                                "maximum_allowed_risk": (
                                    frozen_loo_guard["threshold_value"]
                                    if frozen_loo_guard is not None
                                    else None
                                ),
                                "passed": None,
                                "action": "promote_full_unscorable",
                            }
                            raise _DepthSplatTileRejection(
                                NATIVE_OPACITY_ENDPOINT_FULL_REASON
                            )
                        loo = depthsplat_selected_anchor_attribute_loo_certificate(
                            packed=initial_packed,
                            slot_to_index=slot_to_index,
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            height=height,
                            width=width,
                            level=str(candidate_level),
                            semantics=semantics,
                            plan_record=record,
                        )
                        maximum_risk = (
                            frozen_loo_guard["threshold_value"]
                            if frozen_loo_guard is not None
                            else None
                        )
                        passed = (
                            None
                            if maximum_risk is None
                            else loo["maximum_held_out_risk"] <= float(maximum_risk)
                        )
                        loo = {
                            **loo,
                            "maximum_allowed_risk": maximum_risk,
                            "passed": passed,
                            "action": (
                                "observed_only"
                                if passed is None
                                else "retain_compact"
                                if passed
                                else "promote_full"
                            ),
                        }
                        entry["selected_anchor_attribute_loo"] = loo
                        if passed is False:
                            raise _DepthSplatTileRejection(
                                "DepthSplat selected-anchor LOO rejected tile"
                            )
                    tile_updates, evidence = build_compact_tile(
                        record=record,
                        level=str(candidate_level),
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                    )
                    accepted_level = str(candidate_level)
                    if _is_soft_mixture_profile(execution_profile):
                        failure = soft_mixture_candidate_failure(
                            evidence, level=str(candidate_level), entry=entry
                        )
                        if failure is not None:
                            can_enrich = (
                                route_isolation != "l0_only"
                                and
                                candidate_level == "L0"
                                and record.get("depth_uniform") is True
                                and l1_prefetch_is_available(
                                    record=record,
                                    view=view,
                                    tile_y=tile_y,
                                    tile_x=tile_x,
                                )
                            )
                            if not can_enrich:
                                raise _DepthSplatTileRejection(
                                    f"DepthSplat {failure} rejected tile"
                                )
                            l1_updates, l1_evidence = build_compact_tile(
                                record=record,
                                level="L1",
                                view=view,
                                tile_y=tile_y,
                                tile_x=tile_x,
                            )
                            l1_failure = soft_mixture_candidate_failure(
                                l1_evidence, level="L1", entry=entry
                            )
                            if l1_failure is not None:
                                raise _DepthSplatTileRejection(
                                    f"DepthSplat L1 {l1_failure} rejected tile"
                                )
                            tile_updates = l1_updates
                            evidence = l1_evidence
                            accepted_level = "L1"
                            entry["l1_enrichment_prefetched"] = True
                    elif execution_profile in {
                        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
                        DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
                        DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
                    }:
                        support_key = (
                            "owner_support"
                            if execution_profile
                            == DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE
                            else "support_basis"
                        )
                        support_label = (
                            "owner support"
                            if support_key == "owner_support"
                            else "support basis"
                        )
                        profile_label = (
                            "DepthSplat coverage-enriched"
                            if support_key == "owner_support"
                            else "DepthSplat support-basis"
                        )
                        support = evidence.get("coverage", {}).get(support_key)
                        if (
                            not isinstance(support, Mapping)
                            or support.get("passed") is not True
                        ):
                            entry[f"l0_{support_key}_failure"] = support
                            can_enrich = (
                                level == "L0"
                                and record.get("depth_uniform") is True
                                and l1_prefetch_is_available(
                                    record=record,
                                    view=view,
                                    tile_y=tile_y,
                                    tile_x=tile_x,
                                )
                            )
                            if not can_enrich:
                                raise _DepthSplatTileRejection(
                                    f"{profile_label} {support_label} rejected tile"
                                )
                            l1_updates, l1_evidence = build_compact_tile(
                                record=record,
                                level="L1",
                                view=view,
                                tile_y=tile_y,
                                tile_x=tile_x,
                            )
                            l1_support = l1_evidence.get("coverage", {}).get(
                                support_key
                            )
                            if (
                                not isinstance(l1_support, Mapping)
                                or l1_support.get("passed") is not True
                            ):
                                entry[f"l1_{support_key}_failure"] = l1_support
                                raise _DepthSplatTileRejection(
                                    f"{profile_label} L1 {support_label} rejected tile"
                                )
                            tile_updates = l1_updates
                            evidence = l1_evidence
                            accepted_level = "L1"
                            entry["l1_enrichment_prefetched"] = True
                except _DepthSplatTileRejection as error:
                    reason = str(error) or type(error).__name__
                    _mark_tile(promote, view=view, tile_y=tile_y, tile_x=tile_x, tile_size=4)
                    entry.update({"accepted": False, "reason": reason})
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                except (RuntimeError, ValueError) as error:
                    if _is_kernel_closure_profile(execution_profile):
                        raise
                    reason = str(error) or type(error).__name__
                    _mark_tile(promote, view=view, tile_y=tile_y, tile_x=tile_x, tile_size=4)
                    entry.update({"accepted": False, "reason": reason})
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                else:
                    updates.extend(tile_updates)
                    entry.update(
                        {
                            "accepted": True,
                            "accepted_level": accepted_level,
                            "reason": (
                                "accepted_l1_enrichment"
                                if accepted_level != level
                                else "accepted"
                            ),
                            **evidence,
                        }
                    )
                trace.append(entry)
    ordered = sorted(updates, key=lambda value: value[0])
    if len({slot for slot, *_ in ordered}) != len(ordered):
        raise RuntimeError("DepthSplat materializer generated duplicate anchor updates")
    dtype = initial_packed.means.dtype
    device = initial_packed.means.device
    if ordered:
        update_slots = torch.tensor([slot for slot, *_ in ordered], device=device, dtype=torch.int64)
        means = torch.stack([mean for _, mean, _, _, _ in ordered])
        covariances = torch.stack([covariance for _, _, covariance, _, _ in ordered])
        harmonics = torch.stack([harmonic for _, _, _, harmonic, _ in ordered])
        opacities = torch.stack([opacity for _, _, _, _, opacity in ordered])
    else:
        update_slots = torch.empty(0, device=device, dtype=torch.int64)
        means = torch.empty(0, 3, device=device, dtype=dtype)
        covariances = torch.empty(0, 3, 3, device=device, dtype=dtype)
        harmonics = torch.empty(0, 3, initial_packed.harmonics.shape[2], device=device, dtype=initial_packed.harmonics.dtype)
        opacities = torch.empty(0, device=device, dtype=initial_packed.opacities.dtype)
    if (
        not bool(torch.isfinite(means).all())
        or not bool(torch.isfinite(covariances).all())
        or not bool(torch.isfinite(harmonics).all())
        or not bool(torch.isfinite(opacities).all())
        or bool((opacities < 0.0).any())
        or bool((opacities >= 1.0).any())
    ):
        raise ValueError("DepthSplat materializer preflight updates are invalid")
    coverage_records = [record.get("coverage") for record in trace if isinstance(record.get("coverage"), Mapping)]
    attempted_tiles = sum(bool(record["attempted"]) for record in trace)
    trace_sha256 = _canonical_sha256(trace)
    loo_aggregate = (
        _selected_anchor_attribute_loo_aggregate(
            trace,
            mode="frozen_v16_guard" if frozen_loo_guard is not None else "collect_only",
            frozen_guard=frozen_loo_guard,
        )
        if collect_loo
        else None
    )
    loo_aggregate_sha256 = (
        _canonical_sha256(loo_aggregate) if loo_aggregate is not None else None
    )
    initial_binding = {
        "plan_selection_mask_sha256": _mask_sha256(plan.selection_mask),
        "plan_tile_trace_sha256": str(plan.events["tile_trace_sha256"]),
        "packet_selection_mask_sha256": str(initial_packet.source_trace["selection_mask_sha256"]),
        "packet_selected_descriptor_sha256": _require_sha256(
            initial_packet.source_trace["selected_descriptor_sha256"],
            label="selected descriptor",
        ),
        "packet_selected_rgb_sha256": _require_sha256(
            initial_packet.source_trace["selected_rgb_sha256"], label="selected RGB"
        ),
        "native_execution_sha256": _require_sha256(
            initial_packet.source_trace["native_execution_sha256"], label="native execution"
        ),
        "packet_native_full_passthrough_mask_sha256": _require_sha256(
            initial_packet.source_trace["native_full_passthrough_mask_sha256"],
            label="initial native Full mask",
        ),
        "packed_source_trace_sha256": initial_packed.source_trace_sha256,
        "packed_native_attribute_binding_sha256": _require_sha256(
            initial_packed.attribute_binding_sha256, label="native Adapter attributes"
        ),
        "packed_native_full_attribute_binding_sha256": _require_sha256(
            initial_packed.source_trace[
                "native_full_adapter_attribute_binding_sha256"
            ],
            label="native Full Adapter attributes",
        ),
        "routing_features_sha256": _tensor_sha256(routing_features),
        "routing_z_depths_sha256": _tensor_sha256(routing_z_depths),
        "assignment_feature_map_sha256": _tensor_sha256(assignment_features),
        "assignment_feature_semantics": assignment_feature_semantics,
        "execution_profile": execution_profile,
        "route_isolation": route_isolation,
    }
    update_binding = _update_binding(
        update_slots, means, covariances, harmonics, opacities
    )
    ordered_coverage_rows = _ordered_coverage_rows_from_trace(
        tile_trace=trace, update_slots=update_slots
    )
    if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
        coverage_certificate = DEPTHSPLAT_LITERAL_PAPER_T4_MOMENT_CERTIFICATE
        coverage_certificate_payload = _literal_moment_merge_certificate_payload(
            update_slots=update_slots,
            update_binding=update_binding,
            update_means=means,
            update_covariances=covariances,
            per_update=ordered_coverage_rows,
            tile_trace_sha256=trace_sha256,
        )
    elif execution_profile == DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE:
        coverage_certificate = DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE
        coverage_certificate_payload = _coverage_enriched_moment_certificate_payload(
            update_slots=update_slots,
            update_binding=update_binding,
            update_means=means,
            update_covariances=covariances,
            per_update=ordered_coverage_rows,
            tile_trace_sha256=trace_sha256,
        )
    elif execution_profile == DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE:
        coverage_certificate = DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE
        coverage_certificate_payload = _support_basis_moment_certificate_payload(
            update_slots=update_slots,
            update_binding=update_binding,
            update_means=means,
            update_covariances=covariances,
            per_update=ordered_coverage_rows,
            tile_trace=trace,
            tile_trace_sha256=trace_sha256,
        )
    elif _is_soft_mixture_profile(execution_profile):
        coverage_certificate = DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE
        coverage_certificate_payload = _soft_mixture_moment_certificate_payload(
            update_slots=update_slots,
            update_binding=update_binding,
            update_means=means,
            update_covariances=covariances,
            per_update=ordered_coverage_rows,
            tile_trace=trace,
            tile_trace_sha256=trace_sha256,
        )
    else:
        coverage_certificate = DEPTHSPLAT_COVERAGE_CERTIFICATE
        coverage_certificate_payload = _coverage_certificate_payload(
            update_slots=update_slots,
            update_binding=update_binding,
            per_update=ordered_coverage_rows,
            tile_trace_sha256=trace_sha256,
            maximum_covariance_scale=float(maximum_coverage_covariance_scale),
        )
    coverage_certificate_sha256 = _canonical_sha256(coverage_certificate_payload)
    soft_mixture_certificate_aggregate = (
        _soft_mixture_certificate_aggregate_from_trace(trace)
        if _is_soft_mixture_profile(execution_profile)
        else None
    )
    soft_mixture_certificate_aggregate_sha256 = (
        _canonical_sha256(soft_mixture_certificate_aggregate)
        if soft_mixture_certificate_aggregate is not None
        else None
    )
    mixture_kernel_closure_aggregate = (
        _mixture_kernel_closure_aggregate_from_trace(trace)
        if _is_kernel_closure_profile(execution_profile)
        else None
    )
    mixture_kernel_closure_aggregate_sha256 = (
        _canonical_sha256(mixture_kernel_closure_aggregate)
        if mixture_kernel_closure_aggregate is not None
        else None
    )
    mixture_kernel_closure_l0_to_l1_tile_count = sum(
        record.get("planned_route") == "L0"
        and record.get("accepted_level") == "L1"
        and isinstance(record.get("l0_mixture_kernel_closure_failure"), Mapping)
        for record in trace
    )
    mixture_kernel_closure_full_promotion_tile_count = sum(
        bool(record.get("attempted"))
        and record.get("accepted") is False
        and any(
            isinstance(record.get(key), Mapping)
            for key in (
                "l0_mixture_kernel_closure_failure",
                "l1_mixture_kernel_closure_failure",
            )
        )
        for record in trace
    )
    materialization_session_sha256 = _canonical_sha256(
        {
            "schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
            "initial_binding": initial_binding,
            "update_binding": update_binding,
            "promote_full_mask_sha256": _mask_sha256(promote),
            "tile_trace_sha256": trace_sha256,
            "execution_profile": execution_profile,
            "assignment_feature_map_sha256": initial_binding[
                "assignment_feature_map_sha256"
            ],
            "assignment_feature_semantics": assignment_feature_semantics,
            "selected_anchor_attribute_loo_frozen_guard": frozen_loo_guard,
            "selected_anchor_attribute_loo_aggregate_sha256": loo_aggregate_sha256,
            "coverage_certificate_sha256": coverage_certificate_sha256,
            "soft_mixture_certificate_aggregate_sha256": (
                soft_mixture_certificate_aggregate_sha256
            ),
            "mixture_kernel_closure_aggregate_sha256": (
                mixture_kernel_closure_aggregate_sha256
            ),
            "mixture_kernel_closure_frozen_guard": frozen_kernel_guard,
        }
    )
    events = {
        "schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
        "aggregation": DEPTHSPLAT_COMPACT_AGGREGATION,
        "coverage_certificate": coverage_certificate,
        "coverage_certificate_payload": coverage_certificate_payload,
        "coverage_certificate_sha256": coverage_certificate_sha256,
        "soft_mixture_certificate_aggregate": soft_mixture_certificate_aggregate,
        "soft_mixture_certificate_aggregate_sha256": (
            soft_mixture_certificate_aggregate_sha256
        ),
        "mixture_kernel_closure_aggregate": mixture_kernel_closure_aggregate,
        "mixture_kernel_closure_aggregate_sha256": (
            mixture_kernel_closure_aggregate_sha256
        ),
        "mixture_kernel_closure_guard_policy": (
            SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY
            if _is_kernel_closure_profile(execution_profile)
            else None
        ),
        "mixture_kernel_closure_evidence_policy": (
            DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_POLICY
            if _is_kernel_closure_profile(execution_profile)
            else None
        ),
        "mixture_kernel_closure_strict_maximum_relative_risk": (
            mixture_kernel_closure_aggregate["strict_maximum_relative_risk"]
            if mixture_kernel_closure_aggregate is not None
            else None
        ),
        "mixture_kernel_closure_frozen_guard": (
            frozen_kernel_guard
            if _is_kernel_closure_profile(execution_profile)
            else None
        ),
        "mixture_kernel_closure_calibrated_threshold": (
            frozen_kernel_guard is not None
            if _is_kernel_closure_profile(execution_profile)
            else None
        ),
        "maximum_coverage_covariance_scale": float(maximum_coverage_covariance_scale),
        "execution_profile": execution_profile,
        "route_isolation": route_isolation,
        "formal_paper_selected_probe_only": (
            execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        ),
        "target_rgb_accessed": False,
        "target_rgb_accessed_before_commit": False,
        "target_camera_accessed_before_commit": False,
        "skipped_s3_attributes_accessed": False,
        "source_nonprobe_s3_attribute_reads": 0,
        "not_lossless_deletion": True,
        "nonzero_direct_deletion": False,
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "numerical_execution": "strict-fp32-native-attributes-v1",
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "initial_selection_mask_sha256": initial_binding["plan_selection_mask_sha256"],
        "initial_binding": initial_binding,
        "update_binding": update_binding,
        "materialization_session_sha256": materialization_session_sha256,
        "routing_features_sha256": initial_binding["routing_features_sha256"],
        "routing_z_depths_sha256": initial_binding["routing_z_depths_sha256"],
        "assignment_feature_map_sha256": initial_binding[
            "assignment_feature_map_sha256"
        ],
        "assignment_feature_semantics": assignment_feature_semantics,
        "source_pixel_grid_sha256": (
            _tensor_sha256(source_grid) if source_grid is not None else None
        ),
        "omitted_routing_z_depth_reads": (
            sum(int(record.get("omitted_routing_z_depth_reads", 0)) for record in trace)
            if execution_profile
            in {
                DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
            }
            else None
        ),
        "l1_anchor_semantics": semantics,
        "attempted_tiles": attempted_tiles,
        "accepted_tiles": sum(bool(record["attempted"] and record["accepted"]) for record in trace),
        "promoted_full_tiles": sum(bool(record["attempted"] and not record["accepted"]) for record in trace),
        "update_anchor_count": int(update_slots.numel()),
        "coverage_checked_tiles": len(coverage_records),
        "coverage_max_containment_lhs_after_scale": (
            None
            if execution_profile
            in {
                DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
            }
            else max(
                (
                    float(record["maximum_containment_lhs_after_scale"])
                    for record in coverage_records
                ),
                default=0.0,
            )
        ),
        "coverage_max_moment_covariance_scale": max(
            (float(record["moment_covariance_scale_max"]) for record in coverage_records),
            default=0.0,
        ),
        "literal_finite_psd_moment_merge": (
            execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
        ),
        "literal_support_containment_guard": (
            False
            if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE
            else None
        ),
        "coverage_enriched_owner_support_guard": (
            True
            if execution_profile
            == DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE
            else None
        ),
        "support_basis_guard": (
            True
            if execution_profile == DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE
            else None
        ),
        "soft_mixture_guard": (
            True
            if _is_soft_mixture_profile(execution_profile)
            else None
        ),
        "soft_mixture_projected_domain_guard_used": (
            False
            if _is_soft_mixture_profile(execution_profile)
            else None
        ),
        "coverage_enriched_l0_to_l1_tile_count": sum(
            record.get("accepted_level") == "L1"
            and record.get("planned_route") == "L0"
            and execution_profile
            == DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE
            for record in trace
        ),
        "support_basis_l0_to_l1_tile_count": sum(
            record.get("accepted_level") == "L1"
            and record.get("planned_route") == "L0"
            and execution_profile == DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE
            for record in trace
        ),
        "soft_mixture_l0_to_l1_tile_count": sum(
            record.get("accepted_level") == "L1"
            and record.get("planned_route") == "L0"
            and _is_soft_mixture_profile(execution_profile)
            for record in trace
        ),
        "mixture_kernel_closure_l0_to_l1_tile_count": (
            mixture_kernel_closure_l0_to_l1_tile_count
            if _is_kernel_closure_profile(execution_profile)
            else None
        ),
        "mixture_kernel_closure_full_promotion_tile_count": (
            mixture_kernel_closure_full_promotion_tile_count
            if _is_kernel_closure_profile(execution_profile)
            else None
        ),
        "selected_anchor_attribute_loo_certificate": (
            DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE
        ),
        "selected_anchor_attribute_loo_policy": (
            DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY
        ),
        "selected_anchor_attribute_loo_risk_metric": (
            DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC
        ),
        "selected_anchor_attribute_loo_collect_only": (
            collect_loo and frozen_loo_guard is None
        ),
        "selected_anchor_attribute_loo_guard": frozen_loo_guard is not None,
        "selected_anchor_attribute_loo_frozen_guard": frozen_loo_guard,
        "selected_anchor_attribute_loo_aggregate": loo_aggregate,
        "selected_anchor_attribute_loo_aggregate_sha256": loo_aggregate_sha256,
        "promote_full_mask_sha256": _mask_sha256(promote),
        "tile_trace_sha256": trace_sha256,
        "rejection_reasons": rejection_reasons,
    }
    return DepthSplatCompactMaterializationPreflight(
        update_dense_slots=update_slots,
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        promote_full_mask=promote,
        tile_trace=tuple(trace),
        events=events,
    )


def resolve_depthsplat_compact_final_route(
    plan: IncrementalProbeFirstPlan,
    preflight: DepthSplatCompactMaterializationPreflight,
) -> DepthSplatCompactFinalRoute:
    """Resolve final L0/L1 output slots, discarding producer-only prefetches."""

    views, height, width, semantics = _require_plan(plan)
    plan_trace_sha256 = _require_bound_tile_trace(
        plan.tile_trace,
        expected_sha256=plan.events.get("tile_trace_sha256"),
        label="plan tile trace",
    )
    if not isinstance(preflight, DepthSplatCompactMaterializationPreflight):
        raise TypeError("DepthSplat final route requires a materialization preflight")
    preflight_trace_sha256 = _require_bound_tile_trace(
        preflight.tile_trace,
        expected_sha256=preflight.events.get("tile_trace_sha256"),
        label="preflight tile trace",
    )
    execution_profile = preflight.events.get("execution_profile")
    if execution_profile not in {
        DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
        }:
        raise ValueError("DepthSplat final route materialization profile is invalid")
    _validate_execution_profile_plan_binding(
        plan, execution_profile=execution_profile
    )
    if execution_profile == DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE:
        _validate_literal_paper_t4_plan(
            plan, views=views, height=height, width=width, semantics=semantics
        )
    elif execution_profile == DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE:
        _validate_coverage_enriched_t4_plan(
            plan, views=views, height=height, width=width, semantics=semantics
        )
    elif execution_profile == DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE:
        _validate_support_basis_t4_plan(
            plan, views=views, height=height, width=width, semantics=semantics
        )
    elif execution_profile == DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE:
        _validate_soft_mixture_t4_plan(
            plan, views=views, height=height, width=width, semantics=semantics
        )
    elif execution_profile in {
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
    }:
        _validate_soft_mixture_normalized_t4_plan(
            plan, views=views, height=height, width=width, semantics=semantics
        )
    elif execution_profile == DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE:
        _validate_soft_mixture_kernel_closure_t4_plan(
            plan, views=views, height=height, width=width, semantics=semantics
        )
    initial_binding = preflight.events.get("initial_binding")
    update_binding = preflight.events.get("update_binding")
    session = preflight.events.get("materialization_session_sha256")
    if (
        preflight.events.get("schema_version") != DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION
        or not isinstance(initial_binding, Mapping)
        or not isinstance(update_binding, Mapping)
        or not isinstance(session, str)
        or initial_binding.get("plan_selection_mask_sha256") != _mask_sha256(plan.selection_mask)
        or initial_binding.get("plan_tile_trace_sha256") != plan_trace_sha256
        or initial_binding.get("execution_profile") != execution_profile
        or initial_binding.get("assignment_feature_map_sha256")
        != preflight.events.get("assignment_feature_map_sha256")
        or initial_binding.get("assignment_feature_semantics")
        != preflight.events.get("assignment_feature_semantics")
        or (
            execution_profile
            in {
                DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
                DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
            }
            and preflight.events.get("assignment_feature_semantics")
            != "unit-normalized-bilinear-s1-v1"
        )
        or update_binding
        != _update_binding(
            preflight.update_dense_slots,
            preflight.means,
            preflight.covariances,
            preflight.harmonics,
            preflight.opacities,
        )
    ):
        raise ValueError("DepthSplat final route preflight binding changed")
    coverage_certificate = _validate_coverage_certificate(preflight)
    soft_mixture_aggregate = (
        _validate_soft_mixture_certificate_aggregate(
            events=preflight.events, tile_trace=preflight.tile_trace
        )
        if _is_soft_mixture_profile(execution_profile)
        else None
    )
    mixture_kernel_closure_aggregate = (
        _validate_mixture_kernel_closure_aggregate(
            events=preflight.events, tile_trace=preflight.tile_trace
        )
        if _is_kernel_closure_profile(execution_profile)
        else None
    )
    frozen_kernel_guard = (
        _mixture_kernel_closure_frozen_guard(
            preflight.events.get("mixture_kernel_closure_frozen_guard"),
            allow_serialized_projection=True,
        )
        if _is_kernel_closure_profile(execution_profile)
        else None
    )
    if (
        preflight.events.get("mixture_kernel_closure_calibrated_threshold")
        != (frozen_kernel_guard is not None if _is_kernel_closure_profile(execution_profile) else None)
    ):
        raise ValueError("DepthSplat final route kernel-risk guard changed")
    coverage_certificate_sha256 = _require_sha256(
        preflight.events.get("coverage_certificate_sha256"),
        label="coverage certificate",
    )
    loo_aggregate = _validate_selected_anchor_attribute_loo_aggregate(
        events=preflight.events,
        tile_trace=preflight.tile_trace,
    )
    expected_session = _canonical_sha256(
        {
            "schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
            "initial_binding": dict(initial_binding),
            "update_binding": dict(update_binding),
            "promote_full_mask_sha256": _mask_sha256(preflight.promote_full_mask),
            "tile_trace_sha256": preflight_trace_sha256,
            "execution_profile": execution_profile,
            "assignment_feature_map_sha256": preflight.events.get(
                "assignment_feature_map_sha256"
            ),
            "assignment_feature_semantics": preflight.events.get(
                "assignment_feature_semantics"
            ),
            "selected_anchor_attribute_loo_frozen_guard": preflight.events.get(
                "selected_anchor_attribute_loo_frozen_guard"
            ),
            "selected_anchor_attribute_loo_aggregate_sha256": preflight.events.get(
                "selected_anchor_attribute_loo_aggregate_sha256"
            ),
            "coverage_certificate_sha256": coverage_certificate_sha256,
            "soft_mixture_certificate_aggregate_sha256": (
                _canonical_sha256(soft_mixture_aggregate)
                if soft_mixture_aggregate is not None
                else None
            ),
            "mixture_kernel_closure_aggregate_sha256": (
                _canonical_sha256(mixture_kernel_closure_aggregate)
                if mixture_kernel_closure_aggregate is not None
                else None
            ),
            "mixture_kernel_closure_frozen_guard": frozen_kernel_guard,
        }
    )
    if session != expected_session:
        raise ValueError("DepthSplat final route materialization session changed")
    if (
        preflight.promote_full_mask.shape != plan.selection_mask.shape
        or preflight.promote_full_mask.dtype != torch.bool
        or preflight.promote_full_mask.device != plan.selection_mask.device
    ):
        raise ValueError("DepthSplat final route promotion mask is invalid")
    preflight_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in preflight.tile_trace
    }
    if len(preflight_records) != len(plan.tile_trace):
        raise ValueError("DepthSplat final route preflight trace is incomplete")
    if (
        execution_profile
        in {
            DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
        }
        and any(
            record.get("assignment_feature_semantics")
            != "unit-normalized-bilinear-s1-v1"
            for record in preflight.tile_trace
            if record.get("accepted_level") in {"L0", "L1"}
        )
    ):
        raise ValueError("DepthSplat final route normalized assignment trace changed")
    update_slots = {
        int(slot)
        for slot in preflight.update_dense_slots.detach().to(
            device="cpu", dtype=torch.int64
        ).tolist()
    }
    if len(update_slots) != int(preflight.update_dense_slots.numel()):
        raise ValueError("DepthSplat final route preflight update slots are duplicated")
    selected = torch.zeros_like(plan.selection_mask)
    full_passthrough = torch.zeros_like(plan.selection_mask)
    trace: list[dict[str, Any]] = []
    route_counts = {"L0": 0, "L1": 0, "Full": 0}
    for plan_record in plan.tile_trace:
        view = int(plan_record["view"])
        tile_y = int(plan_record["tile_y"])
        tile_x = int(plan_record["tile_x"])
        record = preflight_records[(view, tile_y, tile_x)]
        planned = str(plan_record["pre_guard_route"])
        promote_tile = preflight.promote_full_mask[
            view, tile_y * 4 : (tile_y + 1) * 4, tile_x * 4 : (tile_x + 1) * 4
        ]
        promoted = bool(promote_tile.all())
        if bool(promote_tile.any()) != promoted:
            raise ValueError("DepthSplat final route has a partial Full promotion")
        attempted = planned in {"L0", "L1"}
        if (
            record.get("planned_route") != planned
            or record.get("attempted") is not attempted
            or (attempted and promoted == (record.get("accepted") is True))
            or (not attempted and promoted)
        ):
            raise ValueError("DepthSplat final route preflight tile decision diverged")
        accepted_level = record.get("accepted_level")
        if planned == "Full":
            if (
                record.get("accepted") is not True
                or accepted_level != "Full"
                or record.get("reason") != "source_full_passthrough"
            ):
                raise ValueError("DepthSplat final route native Full decision changed")
            final = "Full"
        elif promoted:
            final = "Full"
        else:
            if accepted_level not in {"L0", "L1"}:
                raise ValueError("DepthSplat final route has no accepted compact level")
            if (
                accepted_level != planned
                and not (
                    preflight.events.get("execution_profile")
                    in {
                        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
                        DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
                        DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
                        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
                        DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
                    }
                    and planned == "L0"
                    and accepted_level == "L1"
                )
            ):
                raise ValueError("DepthSplat final route compact level changed")
            final = str(accepted_level)
        positions = _route_anchor_positions(plan_record, level=final, semantics=semantics)
        if final == "L1" and planned == "L0":
            prefetched = plan.selection_mask[
                view,
                tile_y * 4 : (tile_y + 1) * 4,
                tile_x * 4 : (tile_x + 1) * 4,
            ]
            expected_prefetch = torch.zeros_like(prefetched)
            for row, column in positions:
                expected_prefetch[row, column] = True
            if bool((expected_prefetch & ~prefetched).any()):
                raise ValueError("DepthSplat final route L1 enrichment was not prefetched")
        tile_slots = set(
            _tile_slots(
                view=view,
                tile_y=tile_y,
                tile_x=tile_x,
                height=height,
                width=width,
                tile_size=4,
                positions=_full_positions(4),
            )
        )
        actual_update_slots = update_slots & tile_slots
        expected_update_slots = (
            set(
                _tile_slots(
                    view=view,
                    tile_y=tile_y,
                    tile_x=tile_x,
                    height=height,
                    width=width,
                    tile_size=4,
                    positions=positions,
                )
            )
            if final in {"L0", "L1"}
            else set()
        )
        if actual_update_slots != expected_update_slots:
            raise ValueError("DepthSplat final route compact updates do not match accepted anchors")
        _mark_tile(selected, view=view, tile_y=tile_y, tile_x=tile_x, tile_size=4, positions=positions)
        if final == "Full":
            _mark_tile(full_passthrough, view=view, tile_y=tile_y, tile_x=tile_x, tile_size=4)
        route_counts[final] += 1
        trace.append({
            "view": view,
            "tile_y": tile_y,
            "tile_x": tile_x,
            "planned_route": planned,
            "final_route": final,
            "compact_materialization": dict(record),
        })
    additional = selected & ~plan.selection_mask
    raw_request = plan.selection_mask | additional
    if bool((selected & ~raw_request).any()):
        raise RuntimeError("DepthSplat final route output exceeds native producer request")
    trace_sha256 = _canonical_sha256(trace)
    route_session_sha256 = _canonical_sha256(
        {
            "preflight_session_sha256": session,
            "selected_output_mask_sha256": _mask_sha256(selected),
            "additional_full_mask_sha256": _mask_sha256(additional),
            "raw_head_request_mask_sha256": _mask_sha256(raw_request),
            "full_passthrough_mask_sha256": _mask_sha256(full_passthrough),
            "tile_trace_sha256": trace_sha256,
            "assignment_feature_map_sha256": preflight.events.get(
                "assignment_feature_map_sha256"
            ),
            "assignment_feature_semantics": preflight.events.get(
                "assignment_feature_semantics"
            ),
            "mixture_kernel_closure_aggregate_sha256": (
                _canonical_sha256(mixture_kernel_closure_aggregate)
                if mixture_kernel_closure_aggregate is not None
                else None
            ),
            "mixture_kernel_closure_frozen_guard": frozen_kernel_guard,
        }
    )
    events = {
        "schema_version": "saes-depthsplat-compact-l0-l1-route-v1",
        "materializer_schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "target_rgb_accessed_before_commit": False,
        "target_camera_accessed_before_commit": False,
        "skipped_s3_attributes_accessed": False,
        "nonzero_direct_deletion": False,
        "not_lossless_deletion": True,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "initial_selection_mask_sha256": _mask_sha256(plan.selection_mask),
        "selected_output_mask_sha256": _mask_sha256(selected),
        "additional_full_mask_sha256": _mask_sha256(additional),
        "raw_head_request_mask_sha256": _mask_sha256(raw_request),
        "full_passthrough_mask_sha256": _mask_sha256(full_passthrough),
        "selected_output_descriptor_count": int(selected.sum().item()),
        "producer_only_prefetch_descriptor_count": int((raw_request & ~selected).sum().item()),
        "additional_full_descriptor_count": int(additional.sum().item()),
        "requires_incremental_full_dispatch": bool(additional.any()),
        "route_counts": route_counts,
        "l1_anchor_semantics": semantics,
        "assignment_feature_map_sha256": preflight.events.get(
            "assignment_feature_map_sha256"
        ),
        "assignment_feature_semantics": preflight.events.get(
            "assignment_feature_semantics"
        ),
        "preflight_trace_sha256": preflight.events.get("tile_trace_sha256"),
        "preflight_materialization_session_sha256": session,
        "coverage_certificate_sha256": preflight.events[
            "coverage_certificate_sha256"
        ],
        "coverage_certificate_geometry": coverage_certificate["geometry"],
        "soft_mixture_certificate_aggregate_sha256": (
            _canonical_sha256(soft_mixture_aggregate)
            if soft_mixture_aggregate is not None
            else None
        ),
        "soft_mixture_projected_domain_guard_used": (
            False
            if _is_soft_mixture_profile(execution_profile)
            else None
        ),
        "mixture_kernel_closure_aggregate_sha256": (
            _canonical_sha256(mixture_kernel_closure_aggregate)
            if mixture_kernel_closure_aggregate is not None
            else None
        ),
        "mixture_kernel_closure_guard_policy": preflight.events.get(
            "mixture_kernel_closure_guard_policy"
        ),
        "mixture_kernel_closure_evidence_policy": preflight.events.get(
            "mixture_kernel_closure_evidence_policy"
        ),
        "mixture_kernel_closure_strict_maximum_relative_risk": preflight.events.get(
            "mixture_kernel_closure_strict_maximum_relative_risk"
        ),
        "mixture_kernel_closure_frozen_guard": frozen_kernel_guard,
        "mixture_kernel_closure_calibrated_threshold": (
            frozen_kernel_guard is not None
            if _is_kernel_closure_profile(execution_profile)
            else None
        ),
        "mixture_kernel_closure_l0_to_l1_tile_count": preflight.events.get(
            "mixture_kernel_closure_l0_to_l1_tile_count"
        ),
        "mixture_kernel_closure_full_promotion_tile_count": preflight.events.get(
            "mixture_kernel_closure_full_promotion_tile_count"
        ),
        "selected_anchor_attribute_loo_aggregate_sha256": (
            _canonical_sha256(loo_aggregate) if loo_aggregate is not None else None
        ),
        "selected_anchor_attribute_loo_frozen_guard": preflight.events.get(
            "selected_anchor_attribute_loo_frozen_guard"
        ),
        "route_session_sha256": route_session_sha256,
        "tile_trace_sha256": trace_sha256,
    }
    return DepthSplatCompactFinalRoute(
        selected_output_mask=selected,
        additional_full_mask=additional,
        raw_head_request_mask=raw_request,
        full_passthrough_mask=full_passthrough,
        tile_trace=tuple(trace),
        events=events,
    )


def _validate_final_slots(
    packed: DepthSplatPackedGaussianAttributes,
    route: DepthSplatCompactFinalRoute,
) -> dict[int, int]:
    if not isinstance(packed, DepthSplatPackedGaussianAttributes):
        raise TypeError("DepthSplat final materialization requires native packed attributes")
    positions = route.selected_output_mask.nonzero(as_tuple=False)
    _, height, width = route.selected_output_mask.shape
    expected_slots = (
        positions[:, 0] * (height * width) + positions[:, 1] * width + positions[:, 2]
    ).to(device=packed.dense_slots.device, dtype=torch.int64)
    if not torch.equal(packed.dense_slots, expected_slots):
        raise ValueError("DepthSplat final packet does not match selected output mask")
    slots = packed.dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(set(slots)) != len(slots):
        raise ValueError("DepthSplat final packet has duplicate slots")
    return {int(slot): index for index, slot in enumerate(slots)}


def apply_depthsplat_compact_l0_l1_materialization(
    final_packed: DepthSplatPackedGaussianAttributes,
    preflight: DepthSplatCompactMaterializationPreflight,
    final_route: DepthSplatCompactFinalRoute,
) -> DepthSplatPackedGaussianAttributes:
    """Apply accepted anchor updates while preserving every Full slot bitwise."""

    if not isinstance(preflight, DepthSplatCompactMaterializationPreflight) or not isinstance(
        final_route, DepthSplatCompactFinalRoute
    ):
        raise TypeError("DepthSplat final materialization requires preflight and route")
    preflight_trace_sha256 = _require_bound_tile_trace(
        preflight.tile_trace,
        expected_sha256=preflight.events.get("tile_trace_sha256"),
        label="preflight tile trace",
    )
    final_route_trace_sha256 = _require_bound_tile_trace(
        final_route.tile_trace,
        expected_sha256=final_route.events.get("tile_trace_sha256"),
        label="final route tile trace",
    )
    preflight_session = preflight.events.get("materialization_session_sha256")
    update_binding = preflight.events.get("update_binding")
    kernel_closure_profile = _is_kernel_closure_profile(
        preflight.events.get("execution_profile")
    )
    frozen_kernel_guard = (
        _mixture_kernel_closure_frozen_guard(
            preflight.events.get("mixture_kernel_closure_frozen_guard"),
            allow_serialized_projection=True,
        )
        if kernel_closure_profile
        else None
    )
    if (
        preflight.events.get("mixture_kernel_closure_calibrated_threshold")
        != (frozen_kernel_guard is not None if kernel_closure_profile else None)
    ):
        raise ValueError("DepthSplat final packet kernel-risk guard changed")
    expected_update_binding = _update_binding(
        preflight.update_dense_slots,
        preflight.means,
        preflight.covariances,
        preflight.harmonics,
        preflight.opacities,
    )
    expected_route_session = _canonical_sha256(
        {
            "preflight_session_sha256": preflight_session,
            "selected_output_mask_sha256": _mask_sha256(final_route.selected_output_mask),
            "additional_full_mask_sha256": _mask_sha256(final_route.additional_full_mask),
            "raw_head_request_mask_sha256": _mask_sha256(final_route.raw_head_request_mask),
            "full_passthrough_mask_sha256": _mask_sha256(final_route.full_passthrough_mask),
            "tile_trace_sha256": final_route_trace_sha256,
            "assignment_feature_map_sha256": preflight.events.get(
                "assignment_feature_map_sha256"
            ),
            "assignment_feature_semantics": preflight.events.get(
                "assignment_feature_semantics"
            ),
            "mixture_kernel_closure_aggregate_sha256": preflight.events.get(
                "mixture_kernel_closure_aggregate_sha256"
            ),
            "mixture_kernel_closure_frozen_guard": frozen_kernel_guard,
        }
    )
    if (
        not isinstance(preflight_session, str)
        or update_binding != expected_update_binding
        or final_route.events.get("preflight_materialization_session_sha256")
        != preflight_session
        or final_route.events.get("preflight_trace_sha256")
        != preflight_trace_sha256
        or final_route.events.get("route_session_sha256") != expected_route_session
    ):
        raise ValueError("DepthSplat final packet route binding changed")
    coverage_certificate = _validate_coverage_certificate(preflight)
    soft_mixture_aggregate = (
        _validate_soft_mixture_certificate_aggregate(
            events=preflight.events, tile_trace=preflight.tile_trace
        )
        if _is_soft_mixture_profile(preflight.events.get("execution_profile"))
        else None
    )
    mixture_kernel_closure_aggregate = (
        _validate_mixture_kernel_closure_aggregate(
            events=preflight.events, tile_trace=preflight.tile_trace
        )
        if kernel_closure_profile
        else None
    )
    loo_aggregate = _validate_selected_anchor_attribute_loo_aggregate(
        events=preflight.events,
        tile_trace=preflight.tile_trace,
    )
    initial_binding = preflight.events.get("initial_binding")
    if (
        not isinstance(initial_binding, Mapping)
        or initial_binding.get("assignment_feature_map_sha256")
        != preflight.events.get("assignment_feature_map_sha256")
        or initial_binding.get("assignment_feature_semantics")
        != preflight.events.get("assignment_feature_semantics")
        or (
            preflight.events.get("execution_profile")
            in {
            DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
            }
            and preflight.events.get("assignment_feature_semantics")
            != "unit-normalized-bilinear-s1-v1"
        )
    ):
        raise ValueError("DepthSplat final packet assignment feature binding changed")
    coverage_certificate_sha256 = _require_sha256(
        preflight.events.get("coverage_certificate_sha256"),
        label="coverage certificate",
    )
    expected_preflight_session = _canonical_sha256(
        {
            "schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
            "initial_binding": dict(initial_binding)
            if isinstance(initial_binding, Mapping)
            else initial_binding,
            "update_binding": dict(update_binding)
            if isinstance(update_binding, Mapping)
            else update_binding,
            "promote_full_mask_sha256": _mask_sha256(preflight.promote_full_mask),
            "tile_trace_sha256": preflight_trace_sha256,
            "execution_profile": preflight.events.get("execution_profile"),
            "assignment_feature_map_sha256": preflight.events.get(
                "assignment_feature_map_sha256"
            ),
            "assignment_feature_semantics": preflight.events.get(
                "assignment_feature_semantics"
            ),
            "selected_anchor_attribute_loo_frozen_guard": preflight.events.get(
                "selected_anchor_attribute_loo_frozen_guard"
            ),
            "selected_anchor_attribute_loo_aggregate_sha256": preflight.events.get(
                "selected_anchor_attribute_loo_aggregate_sha256"
            ),
            "coverage_certificate_sha256": coverage_certificate_sha256,
            "soft_mixture_certificate_aggregate_sha256": (
                _canonical_sha256(soft_mixture_aggregate)
                if soft_mixture_aggregate is not None
                else None
            ),
            "mixture_kernel_closure_aggregate_sha256": (
                _canonical_sha256(mixture_kernel_closure_aggregate)
                if mixture_kernel_closure_aggregate is not None
                else None
            ),
            "mixture_kernel_closure_frozen_guard": frozen_kernel_guard,
        }
    )
    if preflight_session != expected_preflight_session:
        raise ValueError("DepthSplat final packet preflight session changed")
    if (
        final_route.events.get("coverage_certificate_sha256")
        != preflight.events.get("coverage_certificate_sha256")
        or final_route.events.get("coverage_certificate_geometry")
        != coverage_certificate["geometry"]
        or final_route.events.get("soft_mixture_certificate_aggregate_sha256")
        != (
            _canonical_sha256(soft_mixture_aggregate)
            if soft_mixture_aggregate is not None
            else None
        )
        or final_route.events.get("soft_mixture_projected_domain_guard_used")
        != (
            False
            if soft_mixture_aggregate is not None
            else None
        )
        or final_route.events.get("mixture_kernel_closure_aggregate_sha256")
        != (
            _canonical_sha256(mixture_kernel_closure_aggregate)
            if mixture_kernel_closure_aggregate is not None
            else None
        )
        or final_route.events.get("mixture_kernel_closure_guard_policy")
        != preflight.events.get("mixture_kernel_closure_guard_policy")
        or final_route.events.get("mixture_kernel_closure_evidence_policy")
        != preflight.events.get("mixture_kernel_closure_evidence_policy")
        or final_route.events.get("mixture_kernel_closure_strict_maximum_relative_risk")
        != preflight.events.get("mixture_kernel_closure_strict_maximum_relative_risk")
        or final_route.events.get("mixture_kernel_closure_frozen_guard")
        != frozen_kernel_guard
        or final_route.events.get("mixture_kernel_closure_calibrated_threshold")
        != (frozen_kernel_guard is not None if kernel_closure_profile else None)
        or final_route.events.get("mixture_kernel_closure_l0_to_l1_tile_count")
        != preflight.events.get("mixture_kernel_closure_l0_to_l1_tile_count")
        or final_route.events.get("mixture_kernel_closure_full_promotion_tile_count")
        != preflight.events.get("mixture_kernel_closure_full_promotion_tile_count")
        or final_route.events.get("selected_anchor_attribute_loo_aggregate_sha256")
        != (_canonical_sha256(loo_aggregate) if loo_aggregate is not None else None)
        or final_route.events.get("selected_anchor_attribute_loo_frozen_guard")
        != preflight.events.get("selected_anchor_attribute_loo_frozen_guard")
    ):
        raise ValueError("DepthSplat final route evidence changed")
    final_trace = final_packed.source_trace
    final_native_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=final_packed.dense_slots,
        means=final_packed.means,
        covariances=final_packed.covariances,
        harmonics=final_packed.harmonics,
        opacities=final_packed.opacities,
    )
    initial_binding = preflight.events.get("initial_binding")
    if (
        not isinstance(final_trace, Mapping)
        or not isinstance(initial_binding, Mapping)
        or final_packed.source_trace_sha256 != canonical_json_sha256(dict(final_trace))
        or final_packed.attribute_binding_sha256 != final_native_attribute_binding
        or final_trace.get("native_adapter_attribute_binding_sha256")
        != final_native_attribute_binding
        or final_trace.get("native_execution_sha256")
        != initial_binding.get("native_execution_sha256")
        or final_trace.get("native_full_passthrough_mask_sha256")
        != _mask_sha256(final_route.full_passthrough_mask)
        or final_trace.get("native_full_passthrough_positions")
        != int(final_route.full_passthrough_mask.sum().item())
        or final_trace.get("native_full_adapter_attribute_execution_sha256")
        != initial_binding.get("native_execution_sha256")
        or final_trace.get("native_full_adapter_attribute_passthrough_mask_sha256")
        != _mask_sha256(final_route.full_passthrough_mask)
        or final_trace.get("native_full_adapter_attribute_passthrough_count")
        != int(final_route.full_passthrough_mask.sum().item())
        or final_trace.get("packet_selection_kind")
        != "depthsplat-final-selected-output-mask-v1"
        or final_trace.get("selection_mask_sha256")
        != _mask_sha256(final_route.selected_output_mask)
        or final_trace.get("producer_request_mask_sha256")
        != _mask_sha256(final_route.raw_head_request_mask)
    ):
        raise ValueError("DepthSplat final packet does not bind its native replay route")
    slot_to_index = _validate_final_slots(final_packed, final_route)
    final_full_positions = final_route.full_passthrough_mask.nonzero(as_tuple=False)
    _, final_height, final_width = final_route.full_passthrough_mask.shape
    final_full_slots_tensor = (
        final_full_positions[:, 0] * (final_height * final_width)
        + final_full_positions[:, 1] * final_width
        + final_full_positions[:, 2]
    ).to(device=final_packed.dense_slots.device, dtype=torch.int64)
    final_full_indices = torch.searchsorted(
        final_packed.dense_slots, final_full_slots_tensor
    )
    if bool((final_full_indices >= final_packed.dense_slots.numel()).any()) or not torch.equal(
        final_packed.dense_slots[final_full_indices], final_full_slots_tensor
    ):
        raise ValueError("DepthSplat final packet omits a Full passthrough slot")
    final_full_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=final_full_slots_tensor,
        means=final_packed.means[final_full_indices],
        covariances=final_packed.covariances[final_full_indices],
        harmonics=final_packed.harmonics[final_full_indices],
        opacities=final_packed.opacities[final_full_indices],
    )
    if (
        final_trace.get("native_full_adapter_attribute_binding_sha256")
        != final_full_attribute_binding
    ):
        raise ValueError("DepthSplat final Full attributes do not match their capture binding")
    count = int(preflight.update_dense_slots.numel())
    if (
        preflight.means.shape != (count, 3)
        or preflight.covariances.shape != (count, 3, 3)
        or preflight.harmonics.shape != (count, 3, final_packed.harmonics.shape[2])
        or preflight.opacities.shape != (count,)
    ):
        raise ValueError("DepthSplat materializer update tensor shapes are inconsistent")
    update_slots = preflight.update_dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if any(int(slot) not in slot_to_index for slot in update_slots):
        raise ValueError("DepthSplat materializer update does not belong to final packet")
    means = final_packed.means.clone()
    covariances = final_packed.covariances.clone()
    harmonics = final_packed.harmonics.clone()
    opacities = final_packed.opacities.clone()
    if count:
        indices = torch.tensor([slot_to_index[int(slot)] for slot in update_slots], device=means.device, dtype=torch.long)
        means[indices] = preflight.means.to(device=means.device, dtype=means.dtype)
        covariances[indices] = preflight.covariances.to(device=covariances.device, dtype=covariances.dtype)
        harmonics[indices] = preflight.harmonics.to(device=harmonics.device, dtype=harmonics.dtype)
        opacities[indices] = preflight.opacities.to(device=opacities.device, dtype=opacities.dtype)
    full_positions = final_route.full_passthrough_mask.nonzero(as_tuple=False)
    _, height, width = final_route.full_passthrough_mask.shape
    full_slots = (
        full_positions[:, 0] * (height * width) + full_positions[:, 1] * width + full_positions[:, 2]
    ).detach().to(device="cpu", dtype=torch.int64).tolist()
    full_indices = torch.tensor([slot_to_index[int(slot)] for slot in full_slots], device=means.device, dtype=torch.long)
    if full_indices.numel() and (
        not torch.equal(means[full_indices], final_packed.means[full_indices])
        or not torch.equal(covariances[full_indices], final_packed.covariances[full_indices])
        or not torch.equal(harmonics[full_indices], final_packed.harmonics[full_indices])
        or not torch.equal(opacities[full_indices], final_packed.opacities[full_indices])
    ):
        raise RuntimeError("DepthSplat materializer modified a Full passthrough attribute")
    if (
        not bool(torch.isfinite(means).all())
        or not bool(torch.isfinite(covariances).all())
        or not bool(torch.isfinite(harmonics).all())
        or not bool(torch.isfinite(opacities).all())
        or bool((opacities < 0.0).any())
        or bool((opacities > 1.0).any())
        or bool((torch.linalg.eigvalsh((covariances + covariances.mT) * 0.5) < -1e-6).any())
    ):
        raise ValueError("DepthSplat materialized attributes violate the native contract")
    source_trace = dict(final_packed.source_trace)
    source_trace.update(
        {
            "depthsplat_compact_materialization_schema_version": DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION,
            "packet_selection_kind": "depthsplat-final-selected-output-mask-v1",
            "depthsplat_compact_preflight_trace_sha256": preflight.events["tile_trace_sha256"],
            "depthsplat_compact_final_route_trace_sha256": final_route.events["tile_trace_sha256"],
            "depthsplat_compact_materialization_session_sha256": preflight_session,
            "depthsplat_compact_route_session_sha256": expected_route_session,
            "depthsplat_compact_update_anchor_count": count,
            "depthsplat_compact_full_passthrough_count": int(full_indices.numel()),
            "depthsplat_compact_aggregation": DEPTHSPLAT_COMPACT_AGGREGATION,
            "depthsplat_compact_coverage_certificate": preflight.events[
                "coverage_certificate"
            ],
            "depthsplat_compact_coverage_certificate_sha256": preflight.events[
                "coverage_certificate_sha256"
            ],
            "depthsplat_compact_soft_mixture_certificate_aggregate_sha256": (
                _canonical_sha256(soft_mixture_aggregate)
                if soft_mixture_aggregate is not None
                else None
            ),
            "depthsplat_compact_soft_mixture_projected_domain_guard_used": (
                False if soft_mixture_aggregate is not None else None
            ),
            "depthsplat_compact_mixture_kernel_closure_aggregate_sha256": (
                _canonical_sha256(mixture_kernel_closure_aggregate)
                if mixture_kernel_closure_aggregate is not None
                else None
            ),
            "depthsplat_compact_mixture_kernel_closure_guard_policy": preflight.events.get(
                "mixture_kernel_closure_guard_policy"
            ),
            "depthsplat_compact_mixture_kernel_closure_evidence_policy": preflight.events.get(
                "mixture_kernel_closure_evidence_policy"
            ),
            "depthsplat_compact_mixture_kernel_closure_strict_maximum_relative_risk": preflight.events.get(
                "mixture_kernel_closure_strict_maximum_relative_risk"
            ),
            "depthsplat_compact_execution_profile": preflight.events[
                "execution_profile"
            ],
            "depthsplat_compact_assignment_feature_map_sha256": preflight.events[
                "assignment_feature_map_sha256"
            ],
            "depthsplat_compact_assignment_feature_semantics": preflight.events[
                "assignment_feature_semantics"
            ],
            "depthsplat_compact_omitted_routing_z_depth_reads": preflight.events[
                "omitted_routing_z_depth_reads"
            ],
            "depthsplat_compact_selected_anchor_attribute_loo_aggregate_sha256": (
                _canonical_sha256(loo_aggregate) if loo_aggregate is not None else None
            ),
            "depthsplat_compact_selected_anchor_attribute_loo_frozen_guard": preflight.events.get(
                "selected_anchor_attribute_loo_frozen_guard"
            ),
            "depthsplat_compact_skipped_s3_attributes_accessed": False,
            "depthsplat_compact_nonzero_direct_deletion": False,
            "depthsplat_compact_z_depth_geometry": True,
            "depthsplat_compact_source_rgb_sh_initialization": True,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        }
    )
    materialized_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=final_packed.dense_slots,
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )
    source_trace["depthsplat_compact_current_attribute_binding_sha256"] = (
        materialized_attribute_binding
    )
    return DepthSplatPackedGaussianAttributes(
        dense_slots=final_packed.dense_slots.clone(),
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        source_trace=source_trace,
        source_trace_sha256=canonical_json_sha256(source_trace),
        attribute_binding_sha256=materialized_attribute_binding,
    )


__all__ = [
    "DEPTHSPLAT_COMPACT_AGGREGATION",
    "DEPTHSPLAT_COMPACT_COVERAGE_MAX_COVARIANCE_SCALE",
    "DEPTHSPLAT_COVERAGE_CERTIFICATE",
    "DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE",
    "DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE",
    "DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE",
    "DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE",
    "DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE",
    "DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE",
    "DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE",
    "DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE",
    "DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE",
    "DEPTHSPLAT_LITERAL_PAPER_T4_MOMENT_CERTIFICATE",
    "DEPTHSPLAT_MATERIALIZER_SCHEMA_VERSION",
    "DepthSplatCompactFinalRoute",
    "DepthSplatCompactMaterializationPreflight",
    "NATIVE_OPACITY_ENDPOINT_FULL_REASON",
    "apply_depthsplat_compact_l0_l1_materialization",
    "depthsplat_z_depth_world_means",
    "preflight_depthsplat_l0_l1_materialization",
    "resolve_depthsplat_compact_final_route",
]
