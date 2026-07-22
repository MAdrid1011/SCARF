"""Bind DepthSplat's executed cost-volume tensors to SCARF evidence paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange


@dataclass(frozen=True)
class DepthSplatExecutionTensors:
    """Reference tensors produced by one pinned MultiViewUniMatch execution."""

    depths: torch.Tensor
    densities: torch.Tensor
    matching_features: torch.Tensor
    matching_feature_scales: tuple[torch.Tensor, ...]
    mono_features: torch.Tensor
    final_depth: torch.Tensor
    final_match_probability: torch.Tensor


def _tensor_list(results: Mapping[str, Any], key: str) -> list[torch.Tensor]:
    value = results.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"DepthSplat results have no non-empty {key}")
    if not all(torch.is_tensor(item) for item in value):
        raise ValueError(f"DepthSplat {key} must contain tensors")
    return list(value)


def extract_depthsplat_execution_tensors(
    results: Mapping[str, Any],
    *,
    batch_size: int,
    view_count: int,
    image_height: int,
    image_width: int,
) -> DepthSplatExecutionTensors:
    """Extract the exact upstream tensors used by GGU, SAES, and FSDR.

    Each multi-view feature scale is the feature map consumed by the matching
    stage with the same index. ``matching_features`` remains the first
    full-search scale for the existing FSDR path; ``matching_feature_scales``
    records every executed cost-volume input. The final depth/probability pair
    is the one consumed by the Gaussian regressor and Gaussian head.
    """
    if min(batch_size, view_count, image_height, image_width) <= 0:
        raise ValueError("DepthSplat execution dimensions must be positive")
    rows = batch_size * view_count
    depth_preds = _tensor_list(results, "depth_preds")
    match_probs = _tensor_list(results, "match_probs")
    features_mv = _tensor_list(results, "features_mv")
    mono_features = _tensor_list(results, "features_mono_intermediate")
    if len(features_mv) != len(match_probs):
        raise ValueError(
            "DepthSplat matching feature scales must match probability scales"
        )

    final_depth = depth_preds[-1]
    if final_depth.shape != (batch_size, view_count, image_height, image_width):
        raise ValueError(
            "DepthSplat final depth must have shape [B,V,H,W], got "
            f"{tuple(final_depth.shape)}"
        )
    final_match_probability = match_probs[-1]
    if (
        final_match_probability.dim() != 4
        or final_match_probability.shape[0] != rows
        or final_match_probability.shape[1] <= 0
    ):
        raise ValueError("DepthSplat final probability has invalid [BV,D,H,W] shape")
    match_probability_max = final_match_probability.max(dim=1, keepdim=True).values
    if match_probability_max.shape[-2:] != (image_height, image_width):
        match_probability_max = F.interpolate(
            match_probability_max,
            size=(image_height, image_width),
            mode="nearest",
        )

    matching_feature_scales: list[torch.Tensor] = []
    for scale_index, (matching_features_bv, probability) in enumerate(
        zip(features_mv, match_probs)
    ):
        if (
            matching_features_bv.dim() != 4
            or matching_features_bv.shape[0] != rows
            or matching_features_bv.shape[1] <= 0
            or matching_features_bv.shape[-2:] != probability.shape[-2:]
        ):
            raise ValueError(
                "DepthSplat matching feature scale "
                f"{scale_index} must align exactly with its probability volume"
            )
        matching_feature_scales.append(
            rearrange(
                matching_features_bv,
                "(b v) c h w -> b v c h w",
                b=batch_size,
                v=view_count,
            )
        )

    mono = mono_features[-1]
    if mono.dim() != 4 or mono.shape[0] != rows or mono.shape[1] <= 0:
        raise ValueError("DepthSplat mono feature has invalid [BV,C,H,W] shape")

    return DepthSplatExecutionTensors(
        depths=rearrange(final_depth, "b v h w -> b v (h w) () ()"),
        densities=rearrange(
            match_probability_max,
            "(b v) c h w -> b v (c h w) () ()",
            b=batch_size,
            v=view_count,
        ),
        matching_features=matching_feature_scales[0],
        matching_feature_scales=tuple(matching_feature_scales),
        mono_features=mono,
        final_depth=final_depth,
        final_match_probability=final_match_probability,
    )


def align_depthsplat_plane_sweep_candidate_scales(
    probability_scales: Sequence[torch.Tensor],
    plane_sweep_depth_scales: Sequence[torch.Tensor],
    *,
    batch_size: int,
    view_count: int,
) -> tuple[torch.Tensor, ...]:
    """Align captured native plane-sweep depths with probability volumes.

    The native warp receives one candidate tensor for each reference/source
    pair.  For the fixed two-view DL3DV protocol, that pair axis is exactly the
    batch-by-view probability axis, so the captured metric-depth candidates can
    be used directly for discrete FSDR window accounting at every scale.
    """
    if batch_size <= 0 or view_count != 2:
        raise ValueError("plane-sweep candidate capture requires the two-view protocol")
    if len(probability_scales) != len(plane_sweep_depth_scales) or not probability_scales:
        raise ValueError("plane-sweep probability and candidate scales must match")

    aligned: list[torch.Tensor] = []
    rows = batch_size * view_count
    for scale_index, (probabilities, candidate_depths) in enumerate(
        zip(probability_scales, plane_sweep_depth_scales)
    ):
        if (
            probabilities.dim() != 4
            or candidate_depths.dim() != 4
            or probabilities.shape != candidate_depths.shape
            or probabilities.shape[0] != rows
            or probabilities.shape[1] <= 0
            or not bool(torch.isfinite(candidate_depths).all())
            or bool((candidate_depths <= 0).any())
        ):
            raise ValueError(
                f"plane-sweep candidate scale {scale_index} is not aligned with native probabilities"
            )
        aligned.append(
            rearrange(
                candidate_depths,
                "(b v) d h w -> b v d h w",
                b=batch_size,
                v=view_count,
            )
        )
    return tuple(aligned)
