"""Target-free conservative head schedules for probe-first SAES execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from saes.progressive_saes import (
    PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
    ProgressiveSAES,
)
from saes.probe_layout import (
    compute_balanced_lightweight_positions,
    compute_lightweight_positions,
    compute_probe_positions,
)


PAPER_KP_ANCHOR_SEMANTICS = "paper-kp-v1"
LEGACY_L1_ANCHOR_SEMANTICS = "legacy-lightweight-12-dev"
BALANCED_L1_ANCHOR_SEMANTICS = "engineering-lightweight-12-balanced-v1"
ADAPTIVE_L1_15_ANCHOR_SEMANTICS = (
    "engineering-lightweight-15-adaptive-center-s1-loo-v1"
)
ADAPTIVE_L1_15_SELECTION_SEMANTICS = (
    "s1-inverse-distance-leave-one-out-minimum-mean-square-residual-v1"
)
ADAPTIVE_L1_15_MAX_BEST_TO_SECOND_RESIDUAL_RATIO = 0.5


@dataclass(frozen=True)
class ProbeFirstSchedule:
    """Per-view raw-head positions required before materialization guards run."""

    selection_mask: torch.Tensor
    events: dict[str, Any]


@dataclass(frozen=True)
class IncrementalProbeFirstPlan:
    """Ordered raw-head requests needed by a conservative probe-first route.

    The three masks are logical requests rather than disjoint work sets.  An
    initially-Full tile, for example, receives its primary probes before the
    Full request so an incremental producer can prove cache reuse.  Consumers
    must use ``selection_mask`` when they need the union of all requested
    positions.
    """

    primary_mask: torch.Tensor
    secondary_mask: torch.Tensor
    full_mask: torch.Tensor
    selection_mask: torch.Tensor
    tile_trace: tuple[dict[str, Any], ...]
    events: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mask_sha256(mask: torch.Tensor) -> str:
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise ValueError("probe-first mask digest requires a bool tensor")
    value = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _full_local_positions(tile_size: int) -> list[tuple[int, int]]:
    return [(row, column) for row in range(tile_size) for column in range(tile_size)]


def _l1_anchor_count(l1_anchor_semantics: str, *, tile_size: int) -> int:
    if l1_anchor_semantics == PAPER_KP_ANCHOR_SEMANTICS:
        return len(compute_probe_positions(tile_size))
    if l1_anchor_semantics == LEGACY_L1_ANCHOR_SEMANTICS:
        return len(compute_lightweight_positions(tile_size))
    if l1_anchor_semantics == BALANCED_L1_ANCHOR_SEMANTICS:
        return len(compute_balanced_lightweight_positions(tile_size))
    if l1_anchor_semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS:
        if tile_size != 4:
            raise ValueError("adaptive L1-15 is defined only for T=4")
        return 15
    raise ValueError("unsupported L1 anchor semantics")


def _position_from_trace(value: Any, *, tile_size: int, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{label} must be a two-integer local position")
    row, column = int(value[0]), int(value[1])
    if not 0 <= row < tile_size or not 0 <= column < tile_size:
        raise ValueError(f"{label} falls outside its tile")
    return row, column


def l1_local_positions_for_tile(
    record: Mapping[str, Any],
    *,
    tile_size: int,
    l1_anchor_semantics: str,
) -> list[tuple[int, int]]:
    """Reconstruct one tile's declared L1 anchors from its bound plan trace."""
    if not isinstance(record, Mapping):
        raise TypeError("L1 anchor reconstruction requires a tile trace record")
    primary = compute_probe_positions(tile_size)
    if l1_anchor_semantics == PAPER_KP_ANCHOR_SEMANTICS:
        positions = primary
    elif l1_anchor_semantics == LEGACY_L1_ANCHOR_SEMANTICS:
        positions = compute_lightweight_positions(tile_size)
    elif l1_anchor_semantics == BALANCED_L1_ANCHOR_SEMANTICS:
        positions = compute_balanced_lightweight_positions(tile_size)
    elif l1_anchor_semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS:
        if tile_size != 4:
            raise ValueError("adaptive L1-15 is defined only for T=4")
        omitted = _position_from_trace(
            record.get("adaptive_l1_omitted_local_position"),
            tile_size=tile_size,
            label="adaptive L1 omission",
        )
        center_positions = ((1, 1), (1, 2), (2, 1), (2, 2))
        if omitted not in center_positions:
            raise ValueError("adaptive L1-15 must omit one center position")
        positions = [
            *compute_lightweight_positions(tile_size),
            *(position for position in center_positions if position != omitted),
        ]
    else:
        raise ValueError("unsupported L1 anchor semantics")
    if (
        positions[: len(primary)] != primary
        or len(positions)
        != _l1_anchor_count(l1_anchor_semantics, tile_size=tile_size)
        or len(set(positions)) != len(positions)
    ):
        raise RuntimeError("L1 anchor layout does not preserve the primary prefix")
    declared = record.get("l1_anchor_local_positions")
    if declared is not None:
        if not isinstance(declared, (list, tuple)):
            raise ValueError("tile trace L1 anchors must be a sequence")
        parsed = [
            _position_from_trace(value, tile_size=tile_size, label="tile trace L1 anchor")
            for value in declared
        ]
        if parsed != positions:
            raise ValueError("tile trace L1 anchors do not match its declared semantics")
    return positions


def _adaptive_l1_omissions(
    features: torch.Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[tuple[int, int], ...],
]:
    """Choose one S1-predictable non-primary omission for every T=4 tile."""
    if (
        features.ndim != 4
        or tile_size != 4
        or not torch.is_floating_point(features)
        or features.shape[-2:] != (height, width)
    ):
        raise ValueError("adaptive L1-15 requires aligned floating T=4 S1 features")
    full_positions = _full_local_positions(tile_size)
    candidates = ((1, 1), (1, 2), (2, 1), (2, 2))
    flat_index = {position: index for index, position in enumerate(full_positions)}
    candidate_indices = torch.tensor(
        [flat_index[position] for position in candidates],
        device=features.device,
        dtype=torch.long,
    )
    working_dtype = (
        torch.float32
        if features.dtype in {torch.float16, torch.bfloat16}
        else features.dtype
    )
    weights = torch.zeros(
        len(candidates),
        len(full_positions),
        device=features.device,
        dtype=working_dtype,
    )
    for candidate_index, position in enumerate(candidates):
        source_indices = [
            index for index, source in enumerate(full_positions) if source != position
        ]
        inverse_squared_distances = torch.tensor(
            [
                1.0
                / float(
                    (position[0] - full_positions[index][0]) ** 2
                    + (position[1] - full_positions[index][1]) ** 2
                )
                for index in source_indices
            ],
            device=features.device,
            dtype=working_dtype,
        )
        weights[candidate_index, source_indices] = (
            inverse_squared_distances / inverse_squared_distances.sum()
        )
    views, channels = int(features.shape[0]), int(features.shape[1])
    tiles_y, tiles_x = height // tile_size, width // tile_size
    with torch.no_grad():
        tile_features = (
            features.detach()
            .to(dtype=working_dtype)
            .reshape(views, channels, tiles_y, tile_size, tiles_x, tile_size)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(views, tiles_y, tiles_x, tile_size * tile_size, channels)
        )
        predicted = torch.einsum("pl,vyxlc->vyxpc", weights, tile_features)
        targets = tile_features[..., candidate_indices, :]
        residuals = (predicted - targets).square().mean(dim=-1)
        if not bool(torch.isfinite(residuals).all()):
            raise ValueError("adaptive L1-15 S1 residuals are non-finite")
        smallest = residuals.topk(k=2, dim=-1, largest=False, sorted=True).values
        best_residuals, second_residuals = smallest.unbind(dim=-1)
        omissions = residuals.argmin(dim=-1)
        residual_ratios = torch.where(
            second_residuals > torch.finfo(residuals.dtype).eps,
            best_residuals / second_residuals,
            torch.ones_like(best_residuals),
        )
        if not bool(torch.isfinite(residual_ratios).all()):
            raise ValueError("adaptive L1-15 S1 residual ratios are non-finite")
    return (
        omissions.detach().to(device="cpu", dtype=torch.long),
        best_residuals.detach().to(device="cpu", dtype=torch.float64),
        residual_ratios.detach().to(device="cpu", dtype=torch.float64),
        candidates,
    )


def _feature_statistic(decision_semantics: str) -> str:
    mapping = {
        "current": "normalized-probe-total-variance",
        "probe-vector-first-hit": "raw-probe-vector-variance",
        "probe-channel-variance-first-hit": "raw-probe-mean-channel-variance",
        "paper-probe-feature-variance-first-hit": "raw-probe-mean-channel-variance",
        "probe-normalized-std-first-hit": "normalized-probe-vector-standard-deviation",
        PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS: (
            "normalized-probe-vector-standard-deviation"
        ),
    }
    try:
        return mapping[decision_semantics]
    except KeyError as exc:
        raise ValueError(f"unsupported SAES decision semantics: {decision_semantics}") from exc


def _validate_probe_first_inputs(
    features: torch.Tensor,
    depths: torch.Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
    feature_threshold: float,
    depth_threshold: float,
    decision_semantics: str,
) -> tuple[ProgressiveSAES, dict[tuple[int, int, int], float], torch.Tensor]:
    """Validate the target-free S1/S2 route inputs and construct its router."""

    if (
        not torch.is_tensor(features)
        or not torch.is_tensor(depths)
        or features.ndim != 5
        or depths.ndim not in (4, 5)
        or features.shape[0] != 1
        or depths.shape[:2] != features.shape[:2]
        or height <= 0
        or width <= 0
        or tile_size <= 0
        or height % tile_size
        or width % tile_size
    ):
        raise ValueError("probe-first schedule requires B=1 aligned tiled S1/S2 tensors")
    views = int(features.shape[1])
    router = ProgressiveSAES(
        height,
        width,
        initial_tile_size=tile_size,
        feature_var_threshold=feature_threshold,
        depth_std_threshold=depth_threshold,
        view_count=views,
        decision_semantics=decision_semantics,
        materialization_guard=False,
    )
    tile_scores, feature_norm = ProgressiveSAES.classify_tiles_by_features(
        features,
        height,
        width,
        tile_size,
        threshold=feature_threshold,
        per_view=True,
        statistic=_feature_statistic(decision_semantics),
    )
    if (
        not torch.is_tensor(feature_norm)
        or feature_norm.shape != (views, features.shape[2], height, width)
        or feature_norm.device != features.device
        or not bool(torch.isfinite(feature_norm).all())
    ):
        raise ValueError("probe-first schedule could not construct normalized S1 features")
    return router, tile_scores, feature_norm


def _phase_positions(
    router: ProgressiveSAES,
    *,
    feature_score: float,
    feature_threshold: float,
    depth_passes: bool,
    tile_size: int,
    l1_positions: list[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Return one tile's route and phase-local logical requests."""
    primary = list(router.probe_positions)
    primary_set = set(primary)
    lightweight_extra = [
        position
        for position in l1_positions
        if position not in primary_set
    ]
    if (
        len(primary) != len(primary_set)
        or list(l1_positions[: len(primary)]) != primary
    ):
        raise RuntimeError("probe layout is not a primary-prefix lightweight layout")

    if feature_score < feature_threshold:
        # A uniform-depth L0 candidate may later be promoted to L1 by a
        # materialization guard, so it has its extra anchors ready.  A
        # nonuniform L0 candidate remains primary-only until a later Full
        # promotion explicitly requests the missing positions.
        return (
            "L0",
            primary,
            lightweight_extra if depth_passes else [],
            [],
        )
    if depth_passes:
        return "L1", primary, lightweight_extra, []
    return (
        "Full",
        primary,
        [],
        [
            (local_y, local_x)
            for local_y in range(tile_size)
            for local_x in range(tile_size)
        ],
    )


def _mark_positions(
    mask: torch.Tensor,
    *,
    view: int,
    origin_y: int,
    origin_x: int,
    positions: list[tuple[int, int]],
) -> None:
    for local_y, local_x in positions:
        mask[view, origin_y + local_y, origin_x + local_x] = True


def build_incremental_probe_first_plan(
    features: torch.Tensor,
    depths: torch.Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
    feature_threshold: float,
    depth_threshold: float,
    decision_semantics: str,
    l1_anchor_semantics: str = LEGACY_L1_ANCHOR_SEMANTICS,
) -> IncrementalProbeFirstPlan:
    """Build canonical primary, secondary, and Full raw-head request masks.

    This is a target-free route plan: it consumes only S1 features and S2
    depth probes.  It deliberately does not resolve any materialization guard
    that needs raw Gaussian attributes; future guard promotions must append a
    new Full request and preserve the prior phase ledger.
    """
    router, tile_scores, normalized_features = _validate_probe_first_inputs(
        features,
        depths,
        height=height,
        width=width,
        tile_size=tile_size,
        feature_threshold=feature_threshold,
        depth_threshold=depth_threshold,
        decision_semantics=decision_semantics,
    )
    feature_statistic = _feature_statistic(decision_semantics)
    views = int(features.shape[1])
    l1_anchor_count = _l1_anchor_count(
        l1_anchor_semantics, tile_size=tile_size
    )
    if l1_anchor_semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS:
        (
            adaptive_omissions,
            adaptive_residuals,
            adaptive_residual_ratios,
            adaptive_candidates,
        ) = _adaptive_l1_omissions(
            normalized_features,
            height=height,
            width=width,
            tile_size=tile_size,
        )
        static_l1_positions: list[tuple[int, int]] | None = None
    else:
        static_l1_positions = l1_local_positions_for_tile(
            {},
            tile_size=tile_size,
            l1_anchor_semantics=l1_anchor_semantics,
        )
        adaptive_omissions = None
        adaptive_residuals = None
        adaptive_residual_ratios = None
        adaptive_candidates = ()
    primary_mask = torch.zeros(
        (views, height, width), dtype=torch.bool, device=features.device
    )
    secondary_mask = torch.zeros_like(primary_mask)
    full_mask = torch.zeros_like(primary_mask)
    counts = {
        "potential_l0_tiles": 0,
        "potential_l1_tiles": 0,
        "potential_full_tiles": 0,
        "l0_l1_fallback_anchor_tiles": 0,
        "l0_full_fallback_anchor_tiles": 0,
    }
    tile_trace: list[dict[str, Any]] = []
    for view in range(views):
        for tile_y in range(height // tile_size):
            for tile_x in range(width // tile_size):
                depth_passes = router.check_depth_uniformity(
                    depths,
                    tile_y,
                    tile_x,
                    tile_size,
                    height,
                    width,
                    depth_threshold,
                    probe_positions=router.probe_positions,
                    view_index=view,
                    relative=False,
                )
                feature_score = tile_scores[(view, tile_y, tile_x)]
                if feature_score < feature_threshold:
                    counts["potential_l0_tiles"] += 1
                    if depth_passes:
                        counts["l0_l1_fallback_anchor_tiles"] += 1
                    else:
                        counts["l0_full_fallback_anchor_tiles"] += 1
                elif depth_passes:
                    counts["potential_l1_tiles"] += 1
                else:
                    counts["potential_full_tiles"] += 1
                adaptive_omitted_position: tuple[int, int] | None = None
                adaptive_residual: float | None = None
                adaptive_residual_ratio: float | None = None
                if static_l1_positions is None:
                    if (
                        adaptive_omissions is None
                        or adaptive_residuals is None
                        or adaptive_residual_ratios is None
                    ):
                        raise RuntimeError("adaptive L1-15 selection is unavailable")
                    omitted_index = int(adaptive_omissions[view, tile_y, tile_x].item())
                    if not 0 <= omitted_index < len(adaptive_candidates):
                        raise RuntimeError("adaptive L1-15 selected an invalid omission")
                    adaptive_omitted_position = adaptive_candidates[omitted_index]
                    adaptive_residual = float(
                        adaptive_residuals[view, tile_y, tile_x].item()
                    )
                    adaptive_residual_ratio = float(
                        adaptive_residual_ratios[view, tile_y, tile_x].item()
                    )
                    l1_positions = l1_local_positions_for_tile(
                        {
                            "adaptive_l1_omitted_local_position": list(
                                adaptive_omitted_position
                            )
                        },
                        tile_size=tile_size,
                        l1_anchor_semantics=l1_anchor_semantics,
                    )
                else:
                    l1_positions = static_l1_positions
                pre_guard_route, primary, secondary, full = _phase_positions(
                    router,
                    feature_score=feature_score,
                    feature_threshold=feature_threshold,
                    depth_passes=depth_passes,
                    tile_size=tile_size,
                    l1_positions=l1_positions,
                )
                origin_y = tile_y * tile_size
                origin_x = tile_x * tile_size
                _mark_positions(
                    primary_mask,
                    view=view,
                    origin_y=origin_y,
                    origin_x=origin_x,
                    positions=primary,
                )
                _mark_positions(
                    secondary_mask,
                    view=view,
                    origin_y=origin_y,
                    origin_x=origin_x,
                    positions=secondary,
                )
                _mark_positions(
                    full_mask,
                    view=view,
                    origin_y=origin_y,
                    origin_x=origin_x,
                    positions=full,
                )
                record: dict[str, Any] = {
                    "view": view,
                    "tile_y": tile_y,
                    "tile_x": tile_x,
                    "feature_score": feature_score,
                    "depth_uniform": bool(depth_passes),
                    "pre_guard_route": pre_guard_route,
                    "primary_local_positions": [list(position) for position in primary],
                    "secondary_local_positions": [list(position) for position in secondary],
                    "full_local_positions": [list(position) for position in full],
                }
                if adaptive_omitted_position is not None:
                    record["l1_anchor_local_positions"] = [
                        list(position) for position in l1_positions
                    ]
                    record["adaptive_l1_omitted_local_position"] = list(
                        adaptive_omitted_position
                    )
                    record["adaptive_l1_leave_one_out_residual"] = adaptive_residual
                    record["adaptive_l1_best_to_second_residual_ratio"] = (
                        adaptive_residual_ratio
                    )
                tile_trace.append(record)
    selection_mask = primary_mask | secondary_mask | full_mask
    total_tiles = views * (height // tile_size) * (width // tile_size)
    events = {
        "contract_version": "saes-incremental-probe-first-plan-v1",
        "target_rgb_accessed": False,
        "gaussian_attributes_accessed": False,
        "view_count": views,
        "tile_size": tile_size,
        "feature_threshold": feature_threshold,
        # This names the scalar compared directly with ``feature_threshold``.
        # It makes the threshold's coordinate system part of the route ledger.
        "feature_statistic": feature_statistic,
        "depth_threshold": depth_threshold,
        "decision_semantics": decision_semantics,
        "l1_anchor_semantics": l1_anchor_semantics,
        "l1_anchor_selection": (
            ADAPTIVE_L1_15_SELECTION_SEMANTICS
            if l1_anchor_semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS
            else "fixed-layout-v1"
        ),
        "l1_anchor_selection_uses_s1_only": (
            l1_anchor_semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS
        ),
        "l0_anchor_count": len(router.probe_positions),
        "l1_anchor_count": l1_anchor_count,
        "total_tiles": total_tiles,
        **counts,
        "primary_head_final_positions": int(primary_mask.sum().item()),
        "secondary_head_final_positions": int(secondary_mask.sum().item()),
        "full_head_final_positions": int(full_mask.sum().item()),
        "scheduled_head_positions": int(selection_mask.sum().item()),
        "dense_head_positions": views * height * width,
        "primary_mask_sha256": _mask_sha256(primary_mask),
        "secondary_mask_sha256": _mask_sha256(secondary_mask),
        "full_mask_sha256": _mask_sha256(full_mask),
        "selection_mask_sha256": _mask_sha256(selection_mask),
        "tile_trace_records": len(tile_trace),
        "tile_trace_sha256": _canonical_sha256(tile_trace),
    }
    return IncrementalProbeFirstPlan(
        primary_mask=primary_mask,
        secondary_mask=secondary_mask,
        full_mask=full_mask,
        selection_mask=selection_mask,
        tile_trace=tuple(tile_trace),
        events=events,
    )


def build_conservative_probe_first_schedule(
    features: torch.Tensor,
    depths: torch.Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
    feature_threshold: float,
    depth_threshold: float,
    decision_semantics: str,
    l1_anchor_semantics: str = LEGACY_L1_ANCHOR_SEMANTICS,
) -> ProbeFirstSchedule:
    """Return the legacy union mask from an incremental probe-first plan."""
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=height,
        width=width,
        tile_size=tile_size,
        feature_threshold=feature_threshold,
        depth_threshold=depth_threshold,
        decision_semantics=decision_semantics,
        l1_anchor_semantics=l1_anchor_semantics,
    )
    events = dict(plan.events)
    events["contract_version"] = "saes-probe-first-head-schedule-v1"
    return ProbeFirstSchedule(selection_mask=plan.selection_mask, events=events)


def retained_mask_from_saes_modified(
    modified_mask: torch.Tensor, *, views: int, height: int, width: int
) -> torch.Tensor:
    """Map a one-primitive SAES modified mask to raw-head `[view,H,W]` order."""
    if (
        modified_mask.ndim != 1
        or modified_mask.dtype != torch.bool
        or modified_mask.numel() != views * height * width
    ):
        raise ValueError("SAES modified mask does not match one-primitive head layout")
    return (~modified_mask).reshape(views, height, width)
