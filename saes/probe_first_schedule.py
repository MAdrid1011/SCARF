"""Target-free conservative head schedules for probe-first SAES execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from saes.progressive_saes import ProgressiveSAES


@dataclass(frozen=True)
class ProbeFirstSchedule:
    """Per-view raw-head positions required before materialization guards run."""

    selection_mask: torch.Tensor
    events: dict[str, Any]


def _feature_statistic(decision_semantics: str) -> str:
    mapping = {
        "current": "normalized-probe-total-variance",
        "probe-vector-first-hit": "raw-probe-vector-variance",
        "probe-channel-variance-first-hit": "raw-probe-mean-channel-variance",
        "probe-normalized-std-first-hit": "normalized-probe-vector-standard-deviation",
    }
    try:
        return mapping[decision_semantics]
    except KeyError as exc:
        raise ValueError(f"unsupported SAES decision semantics: {decision_semantics}") from exc


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
) -> ProbeFirstSchedule:
    """Schedule only positions needed to resolve L0/L1 guard outcomes.

    The schedule is derived before any raw Gaussian head output exists. Full
    tiles retain every position. L1 candidates retain their fixed 2K anchors.
    An L0 candidate retains 2K anchors only when its existing probe-depth test
    could legitimately promote a failed L0 guard to L1; otherwise it retains
    only K and a later guard rejection becomes Full.
    """
    if (
        features.ndim != 5
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
    tile_scores, _ = ProgressiveSAES.classify_tiles_by_features(
        features,
        height,
        width,
        tile_size,
        threshold=feature_threshold,
        per_view=True,
        statistic=_feature_statistic(decision_semantics),
    )
    mask = torch.zeros((views, height, width), dtype=torch.bool, device=features.device)
    counts = {
        "potential_l0_tiles": 0,
        "potential_l1_tiles": 0,
        "potential_full_tiles": 0,
        "l0_l1_fallback_anchor_tiles": 0,
        "l0_full_fallback_anchor_tiles": 0,
    }
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
                if tile_scores[(view, tile_y, tile_x)] < feature_threshold:
                    counts["potential_l0_tiles"] += 1
                    if depth_passes:
                        positions = router.lightweight_positions
                        counts["l0_l1_fallback_anchor_tiles"] += 1
                    else:
                        positions = router.probe_positions
                        counts["l0_full_fallback_anchor_tiles"] += 1
                elif depth_passes:
                    counts["potential_l1_tiles"] += 1
                    positions = router.lightweight_positions
                else:
                    counts["potential_full_tiles"] += 1
                    positions = [
                        (local_y, local_x)
                        for local_y in range(tile_size)
                        for local_x in range(tile_size)
                    ]
                origin_y = tile_y * tile_size
                origin_x = tile_x * tile_size
                for local_y, local_x in positions:
                    mask[view, origin_y + local_y, origin_x + local_x] = True
    total_tiles = views * (height // tile_size) * (width // tile_size)
    events = {
        "contract_version": "saes-probe-first-head-schedule-v1",
        "target_rgb_accessed": False,
        "gaussian_attributes_accessed": False,
        "view_count": views,
        "tile_size": tile_size,
        "feature_threshold": feature_threshold,
        "depth_threshold": depth_threshold,
        "decision_semantics": decision_semantics,
        "l0_anchor_count": len(router.probe_positions),
        "l1_anchor_count": len(router.lightweight_positions),
        "total_tiles": total_tiles,
        **counts,
        "scheduled_head_positions": int(mask.sum().item()),
        "dense_head_positions": views * height * width,
    }
    return ProbeFirstSchedule(selection_mask=mask, events=events)


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
