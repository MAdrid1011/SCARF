"""Prepare view-aligned feature/depth frames for FSDR simulation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from saes.progressive_saes import ProgressiveSAES


def depthsplat_global_candidate_tensors(
    match_probs: list[torch.Tensor],
    *,
    near: torch.Tensor,
    far: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DepthSplat's executed first-scale full-search evidence."""
    if not match_probs:
        raise ValueError("DepthSplat did not return full-search probabilities")
    probabilities_bv = match_probs[0]
    if probabilities_bv.dim() != 4:
        raise ValueError("DepthSplat probability volume must have shape [BV,D,H,W]")
    if near.dim() != 2 or far.shape != near.shape:
        raise ValueError("DepthSplat near/far bounds must have shape [B,V]")
    batch, views = near.shape
    if probabilities_bv.shape[0] != batch * views:
        raise ValueError("DepthSplat probability views do not match near/far bounds")
    candidate_count = probabilities_bv.shape[1]
    near_bv = near.to(probabilities_bv).clamp_min(1e-8)
    far_bv = far.to(probabilities_bv).clamp_min(1e-8)
    interpolation = torch.linspace(
        0.0,
        1.0,
        candidate_count,
        device=probabilities_bv.device,
        dtype=probabilities_bv.dtype,
    ).reshape(1, 1, candidate_count)
    minimum = (1.0 / far_bv).unsqueeze(-1)
    maximum = (1.0 / near_bv).unsqueeze(-1)
    candidates = (minimum + interpolation * (maximum - minimum))[..., None, None]
    probabilities = probabilities_bv.reshape(
        batch,
        views,
        candidate_count,
        *probabilities_bv.shape[-2:],
    )
    return probabilities, candidates


def tile_probe_pixel_order(
    *, height: int, width: int, tile_size: int
) -> list[int]:
    """Return the paper's per-tile probe-first, then row-major schedule."""
    if min(height, width, tile_size) <= 0:
        raise ValueError("FSDR schedule dimensions must be positive")
    if height % tile_size or width % tile_size:
        raise ValueError("FSDR schedule requires complete tiles")
    probes = ProgressiveSAES.compute_probe_positions(tile_size)
    probe_set = set(probes)
    order: list[int] = []
    for tile_y in range(0, height, tile_size):
        for tile_x in range(0, width, tile_size):
            positions = [
                *probes,
                *[
                    (local_y, local_x)
                    for local_y in range(tile_size)
                    for local_x in range(tile_size)
                    if (local_y, local_x) not in probe_set
                ],
            ]
            order.extend(
                (tile_y + local_y) * width + tile_x + local_x
                for local_y, local_x in positions
            )
    return order


def prepare_fsdr_frame(
    features: torch.Tensor,
    depths: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened feature and depth tensors from context view zero."""
    if features.dim() != 5 or features.shape[0] != 1 or features.shape[1] < 1:
        raise ValueError(f"unsupported FSDR feature shape: {tuple(features.shape)}")
    if height <= 0 or width <= 0:
        raise ValueError("FSDR frame dimensions must be positive")
    feature_map = F.interpolate(
        features[0, 0].unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    feature_frame = feature_map.permute(1, 2, 0).reshape(
        height * width, feature_map.shape[0]
    )
    if depths.dim() == 5:
        depth_frame = depths[0, 0, : height * width, 0, 0]
    elif depths.dim() == 4:
        depth_frame = depths[0, 0, :height, :width].reshape(-1)
    else:
        raise ValueError(f"unsupported FSDR depth shape: {tuple(depths.shape)}")
    if depth_frame.numel() != height * width:
        raise ValueError("FSDR depth frame does not cover the selected feature view")
    return feature_frame, depth_frame


def prepare_fsdr_candidate_frame(
    features: torch.Tensor,
    depth_probs: torch.Tensor,
    depth_candidates: torch.Tensor,
    *,
    view_index: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
]:
    """Align one feature view with its exact full-search candidate evidence."""
    if features.dim() != 5 or features.shape[0] != 1:
        raise ValueError(f"unsupported FSDR feature shape: {tuple(features.shape)}")
    if depth_probs.dim() != 5 or depth_probs.shape[0] != 1:
        raise ValueError(
            f"full-search probabilities must have shape [1,V,D,H,W], got {tuple(depth_probs.shape)}"
        )
    if depth_candidates.dim() == 3:
        depth_candidates = depth_candidates[..., None, None]
    if depth_candidates.dim() != 5 or depth_candidates.shape[0] != 1:
        raise ValueError(
            "depth candidates must have shape [1,V,D], [1,V,D,1,1], or [1,V,D,H,W]"
        )
    views = depth_probs.shape[1]
    if features.shape[1] != views or depth_candidates.shape[1] != views:
        raise ValueError("feature, probability, and candidate view counts differ")
    if not 0 <= view_index < views:
        raise ValueError("FSDR evidence view index is out of range")
    candidate_count = depth_probs.shape[2]
    if depth_candidates.shape[2] != candidate_count:
        raise ValueError("probability and candidate count differ")
    height, width = depth_probs.shape[-2:]
    candidate_spatial = depth_candidates.shape[-2:]
    if candidate_spatial not in {(1, 1), (height, width)}:
        raise ValueError("candidate tensor does not match probability resolution")
    probabilities = depth_probs[0, view_index]
    candidates = depth_candidates[0, view_index]
    if candidate_spatial == (1, 1):
        candidates = candidates.expand(-1, height, width)
    if not torch.isfinite(probabilities).all() or not torch.isfinite(candidates).all():
        raise ValueError("candidate evidence contains non-finite values")
    if (probabilities < 0).any():
        raise ValueError("full-search probabilities must be non-negative")
    probability_mass = probabilities.sum(dim=0)
    if (probability_mass <= 0).any():
        raise ValueError("full-search probability mass must be positive")

    feature_map = F.interpolate(
        features[0, view_index].unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    feature_frame = feature_map.permute(1, 2, 0).reshape(
        height * width, feature_map.shape[0]
    )
    candidate_frame = candidates.permute(1, 2, 0).reshape(
        height * width, candidate_count
    )
    top1 = probabilities.argmax(dim=0).reshape(-1)
    anchors = (
        (probabilities * candidates).sum(dim=0) / probability_mass
    ).reshape(-1)
    return feature_frame, anchors, top1, candidate_frame, (height, width)
