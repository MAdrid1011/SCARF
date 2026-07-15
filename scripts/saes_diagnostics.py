"""Non-claim diagnostics for SAES decision statistics and sparse coverage."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from saes.progressive_saes import ProgressiveSAES


def declared_level0_rate(
    expected_results: str | Path,
    model: str,
    dataset: str,
) -> float:
    """Load the paper-declared L0 rate used as a diagnostic boundary."""
    with Path(expected_results).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    pair = f"{model}/{dataset}"
    try:
        value = float(payload["mechanisms"][pair]["level0_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing declared L0 rate for {pair}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid declared L0 rate for {pair}: {value}")
    return value


def representative_indices(
    modified_mask: torch.Tensor,
    *,
    view_count: int,
    height: int,
    width: int,
    tile_size: int,
    primitives_per_pixel: int,
) -> torch.Tensor:
    """Recover retained probe indices for every tile modified by sparse SAES."""
    expected = view_count * height * width * primitives_per_pixel
    flat = modified_mask.reshape(-1)
    if flat.numel() != expected:
        raise ValueError(
            f"modified mask has {flat.numel()} entries, expected {expected}"
        )
    probes = ProgressiveSAES.compute_probe_positions(tile_size)
    retained: list[int] = []
    for view in range(view_count):
        for tile_y in range(0, height - tile_size + 1, tile_size):
            for tile_x in range(0, width - tile_size + 1, tile_size):
                tile_indices = [
                    (
                        (
                            view * height * width
                            + (tile_y + local_y) * width
                            + tile_x
                            + local_x
                        )
                        * primitives_per_pixel
                        + slot
                    )
                    for local_y in range(tile_size)
                    for local_x in range(tile_size)
                    for slot in range(primitives_per_pixel)
                ]
                tile_tensor = torch.tensor(
                    tile_indices, device=flat.device, dtype=torch.long
                )
                if not bool(flat[tile_tensor].any().item()):
                    continue
                for local_y, local_x in probes:
                    for slot in range(primitives_per_pixel):
                        index = (
                            (
                                view * height * width
                                + (tile_y + local_y) * width
                                + tile_x
                                + local_x
                            )
                            * primitives_per_pixel
                            + slot
                        )
                        if bool(flat[index].item()):
                            raise ValueError("SAES marked a retained probe as removed")
                        retained.append(index)
    return torch.tensor(retained, device=flat.device, dtype=torch.long)


def build_coverage_variant(
    gaussians: Any,
    representatives: torch.Tensor,
    *,
    covariance_scale: float,
    opacity_scale: float,
) -> Any:
    """Clone a sparse Gaussian set and adjust only representative coverage."""
    if not math.isfinite(covariance_scale) or covariance_scale <= 0:
        raise ValueError("covariance_scale must be finite and positive")
    if not math.isfinite(opacity_scale) or opacity_scale <= 0:
        raise ValueError("opacity_scale must be finite and positive")
    means = gaussians.means.clone()
    covariances = gaussians.covariances.clone()
    harmonics = gaussians.harmonics.clone()
    opacities = gaussians.opacities.clone()
    representatives = representatives.to(covariances.device)
    if representatives.numel():
        covariances[0, representatives] *= covariance_scale
        alpha = opacities[0, representatives].clamp(0.0, 1.0 - 1e-6)
        opacities[0, representatives] = 1.0 - torch.pow(
            1.0 - alpha, opacity_scale
        )
    return type(gaussians)(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )


def build_component_variant(
    sparse: Any,
    original: Any,
    representatives: torch.Tensor,
    *,
    restore: tuple[str, ...],
) -> Any:
    """Restore selected representative attributes for attribution only."""
    allowed = {"means", "covariances", "harmonics", "opacities"}
    unknown = set(restore) - allowed
    if unknown:
        raise ValueError(f"unsupported Gaussian attributes: {sorted(unknown)}")
    values = {
        name: getattr(sparse, name).clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    representatives = representatives.to(values["means"].device)
    for name in restore:
        source = getattr(original, name)
        values[name][0, representatives] = source[0, representatives]
    return type(sparse)(**values)


def probe_feature_variances(
    features: torch.Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
) -> dict[tuple[int, int, int], float]:
    """Return raw probe-vector variance for every view and complete tile."""
    if features.dim() != 5 or features.shape[0] != 1:
        raise ValueError(f"unsupported feature shape: {tuple(features.shape)}")
    if height < tile_size or width < tile_size:
        raise ValueError("image dimensions must contain at least one complete tile")
    feature_up = F.interpolate(
        features[0], size=(height, width), mode="bilinear", align_corners=False
    ).float()
    probes = ProgressiveSAES.compute_probe_positions(tile_size)
    scores: dict[tuple[int, int, int], float] = {}
    for view in range(feature_up.shape[0]):
        for tile_y in range(0, height - tile_size + 1, tile_size):
            for tile_x in range(0, width - tile_size + 1, tile_size):
                probe_features = torch.stack(
                    [
                        feature_up[
                            view, :, tile_y + local_y, tile_x + local_x
                        ]
                        for local_y, local_x in probes
                    ]
                )
                variance = (
                    (probe_features - probe_features.mean(dim=0))
                    .square()
                    .sum(dim=1)
                    .mean()
                )
                scores[(view, tile_y // tile_size, tile_x // tile_size)] = float(
                    variance.item()
                )
    return scores


def probe_cross_check_errors(
    gaussians: Any,
    *,
    view_count: int,
    height: int,
    width: int,
    tile_size: int,
    primitives_per_pixel: int,
) -> dict[tuple[int, int, int], float]:
    """Return target-free leave-one-out probe errors for every complete tile."""
    scorer = ProgressiveSAES(
        height,
        width,
        initial_tile_size=tile_size,
        view_count=view_count,
        primitives_per_pixel=primitives_per_pixel,
    )
    scores: dict[tuple[int, int, int], float] = {}
    for view in range(view_count):
        for tile_y in range(height // tile_size):
            for tile_x in range(width // tile_size):
                probe_indices = [
                    (
                        (
                            view * height * width
                            + (tile_y * tile_size + local_y) * width
                            + tile_x * tile_size
                            + local_x
                        )
                        * primitives_per_pixel
                    )
                    for local_y, local_x in scorer.probe_positions
                ]
                scores[(view, tile_y, tile_x)] = scorer.probe_cross_check_error(
                    gaussians, probe_indices
                )
    return scores


def build_ranked_tile_subset_variant(
    original: Any,
    sparse: Any,
    modified_mask: torch.Tensor,
    tile_scores: dict[tuple[int, int, int], float],
    *,
    view_count: int,
    height: int,
    width: int,
    tile_size: int,
    primitives_per_pixel: int,
    target_fraction: float,
    ranking_statistic: str,
) -> tuple[Any, dict[str, Any]]:
    """Apply sparse results only to the lowest-score eligible tile subset."""
    if not math.isfinite(target_fraction) or not 0.0 <= target_fraction <= 1.0:
        raise ValueError("target_fraction must be finite and within [0, 1]")
    if not ranking_statistic:
        raise ValueError("ranking_statistic must not be empty")
    if min(view_count, height, width, tile_size, primitives_per_pixel) < 1:
        raise ValueError("layout dimensions must be positive")
    gaussian_count = view_count * height * width * primitives_per_pixel
    flat_modified = modified_mask.reshape(-1)
    if flat_modified.numel() != gaussian_count:
        raise ValueError(
            f"modified mask has {flat_modified.numel()} entries, "
            f"expected {gaussian_count}"
        )

    tiles_high = height // tile_size
    tiles_wide = width // tile_size
    total_tiles = view_count * tiles_high * tiles_wide
    eligible: list[tuple[float, int, int, int, list[int]]] = []
    for view in range(view_count):
        for tile_y in range(tiles_high):
            for tile_x in range(tiles_wide):
                key = (view, tile_y, tile_x)
                if key not in tile_scores or not math.isfinite(tile_scores[key]):
                    raise ValueError(f"missing finite score for tile {key}")
                indices = [
                    (
                        (
                            view * height * width
                            + (tile_y * tile_size + local_y) * width
                            + tile_x * tile_size
                            + local_x
                        )
                        * primitives_per_pixel
                        + slot
                    )
                    for local_y in range(tile_size)
                    for local_x in range(tile_size)
                    for slot in range(primitives_per_pixel)
                ]
                index_tensor = torch.tensor(
                    indices, device=flat_modified.device, dtype=torch.long
                )
                if bool(flat_modified[index_tensor].any().item()):
                    eligible.append(
                        (float(tile_scores[key]), view, tile_y, tile_x, indices)
                    )

    selected_tile_count = int(round(target_fraction * total_tiles))
    if selected_tile_count > len(eligible):
        raise ValueError(
            f"target selects {selected_tile_count} tiles, but only "
            f"{len(eligible)} sparse tiles are eligible"
        )
    eligible.sort(key=lambda item: item[:4])
    selected = eligible[:selected_tile_count]
    selected_indices = [index for item in selected for index in item[4]]

    values = {
        name: getattr(original, name).clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    if selected_indices:
        indices = torch.tensor(
            selected_indices, device=values["means"].device, dtype=torch.long
        )
        for name, destination in values.items():
            source = getattr(sparse, name).to(destination.device)
            destination[0, indices] = source[0, indices]

    return type(original)(**values), {
        "target_fraction": float(target_fraction),
        "total_tile_count": total_tiles,
        "eligible_tile_count": len(eligible),
        "selected_tile_count": selected_tile_count,
        "selected_fraction": selected_tile_count / total_tiles,
        "ranking_statistic": ranking_statistic,
        "selected_tiles": [
            {
                "view": view,
                "tile_y": tile_y,
                "tile_x": tile_x,
                "score": score,
            }
            for score, view, tile_y, tile_x, _ in selected
        ],
    }


def _summary(values: list[float], threshold: float) -> dict[str, Any]:
    if not values:
        raise ValueError("diagnostic statistic has no values")
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = {
        str(percentile): float(
            torch.quantile(tensor, percentile / 100.0).item()
        )
        for percentile in (0, 1, 2, 5, 10, 15, 20, 30, 50, 75, 90, 95, 99, 100)
    }
    below = tensor < threshold
    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "threshold": float(threshold),
        "below_threshold": int(below.sum().item()),
        "below_threshold_rate": float(below.double().mean().item()),
        "percentiles": quantiles,
    }


def _depth_value(
    depths: torch.Tensor, view: int, y: int, x: int, width: int
) -> torch.Tensor:
    if depths.dim() == 5:
        return depths[0, view, y * width + x].reshape(-1)[0]
    if depths.dim() == 4:
        return depths[0, view, y, x]
    raise ValueError(f"unsupported depth shape: {tuple(depths.shape)}")


def normalize_inverse_depth(
    depths: torch.Tensor,
    *,
    near: torch.Tensor,
    far: torch.Tensor,
) -> torch.Tensor:
    """Map metric depths to the upstream models' inverse-depth candidate coordinate."""
    if depths.dim() not in (4, 5) or depths.shape[:2] != near.shape:
        raise ValueError("depths and near bounds must share [batch, view] dimensions")
    if far.shape != near.shape:
        raise ValueError("near and far bounds must have identical shapes")
    if not torch.isfinite(depths).all() or not torch.isfinite(near).all() or not torch.isfinite(far).all():
        raise ValueError("depth values and bounds must be finite")
    if (depths <= 0).any() or (near <= 0).any() or (far <= near).any():
        raise ValueError("inverse-depth normalization requires 0 < near < far")

    trailing = (1,) * (depths.dim() - 2)
    near_inverse = near.to(depths).reciprocal().reshape(*near.shape, *trailing)
    far_inverse = far.to(depths).reciprocal().reshape(*far.shape, *trailing)
    return (depths.reciprocal() - far_inverse) / (near_inverse - far_inverse)


def _first_hit_summary(
    feature_values: list[float],
    depth_values: list[float],
    *,
    feature_threshold: float,
    depth_threshold: float,
) -> dict[str, Any]:
    if len(feature_values) != len(depth_values):
        raise ValueError("aligned feature and depth statistics must have equal lengths")
    l0_count = sum(value < feature_threshold for value in feature_values)
    l1_count = sum(
        feature >= feature_threshold and depth < depth_threshold
        for feature, depth in zip(feature_values, depth_values)
    )
    count = len(feature_values)
    full_count = count - l0_count - l1_count
    return {
        "count": count,
        "l0_count": l0_count,
        "l1_count": l1_count,
        "full_count": full_count,
        "l0_rate": l0_count / count if count else 0.0,
        "l1_rate": l1_count / count if count else 0.0,
        "full_rate": full_count / count if count else 0.0,
    }


def decision_statistics(
    features: torch.Tensor,
    depths: torch.Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
    feature_threshold: float,
    depth_threshold: float,
    near: torch.Tensor | None = None,
    far: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare plausible scalar interpretations of the paper's probe statistics."""
    if features.dim() != 5:
        raise ValueError(f"unsupported feature shape: {tuple(features.shape)}")
    feature_up = F.interpolate(
        features[0], size=(height, width), mode="bilinear", align_corners=False
    ).float()
    normalized = feature_up / feature_up.norm(dim=1, keepdim=True).clamp_min(1e-8)
    probes = ProgressiveSAES.compute_probe_positions(tile_size)

    raw_vector_variance: list[float] = []
    unit_vector_variance: list[float] = []
    current_channel_std: list[float] = []
    absolute_depth_std: list[float] = []
    relative_depth_std: list[float] = []
    if (near is None) != (far is None):
        raise ValueError("near and far bounds must be provided together")
    candidate_depths = (
        normalize_inverse_depth(depths, near=near, far=far)
        if near is not None and far is not None
        else None
    )
    candidate_coordinate_std: list[float] = []

    for view in range(feature_up.shape[0]):
        for tile_y in range(0, height - tile_size + 1, tile_size):
            for tile_x in range(0, width - tile_size + 1, tile_size):
                raw_probe = torch.stack(
                    [
                        feature_up[view, :, tile_y + local_y, tile_x + local_x]
                        for local_y, local_x in probes
                    ]
                )
                unit_probe = torch.stack(
                    [
                        normalized[view, :, tile_y + local_y, tile_x + local_x]
                        for local_y, local_x in probes
                    ]
                )
                raw_vector_variance.append(
                    float(
                        (raw_probe - raw_probe.mean(dim=0))
                        .square()
                        .sum(dim=1)
                        .mean()
                        .item()
                    )
                )
                unit_vector_variance.append(
                    float(
                        (unit_probe - unit_probe.mean(dim=0))
                        .square()
                        .sum(dim=1)
                        .mean()
                        .item()
                    )
                )
                tile = normalized[
                    view,
                    :,
                    tile_y : tile_y + tile_size,
                    tile_x : tile_x + tile_size,
                ].reshape(normalized.shape[1], -1)
                current_channel_std.append(float(tile.std(dim=1).mean().item()))

                probe_depths = torch.stack(
                    [
                        _depth_value(
                            depths,
                            view,
                            tile_y + local_y,
                            tile_x + local_x,
                            width,
                        ).float()
                        for local_y, local_x in probes
                    ]
                )
                absolute = probe_depths.std(unbiased=False)
                relative = absolute / probe_depths.mean().abs().clamp_min(1e-8)
                absolute_depth_std.append(float(absolute.item()))
                relative_depth_std.append(float(relative.item()))
                if candidate_depths is not None:
                    candidate_probe_depths = torch.stack(
                        [
                            _depth_value(
                                candidate_depths,
                                view,
                                tile_y + local_y,
                                tile_x + local_x,
                                width,
                            ).float()
                            for local_y, local_x in probes
                        ]
                    )
                    candidate_coordinate_std.append(
                        float(candidate_probe_depths.std(unbiased=False).item())
                    )

    feature_statistics = {
        "raw_vector_variance": raw_vector_variance,
        "unit_vector_variance": unit_vector_variance,
        "current_channel_std": current_channel_std,
    }
    depth_statistics = {
        "absolute_std": absolute_depth_std,
        "relative_std": relative_depth_std,
    }
    if candidate_depths is not None:
        depth_statistics["candidate_coordinate_std"] = candidate_coordinate_std

    return {
        "schema_version": "1.0",
        "kind": "saes_decision_statistics",
        "paper_result_eligible": False,
        "tile_count": len(raw_vector_variance),
        "view_count": int(feature_up.shape[0]),
        "probe_count": len(probes),
        "feature": {
            name: _summary(values, feature_threshold)
            for name, values in feature_statistics.items()
        },
        "depth": {
            name: _summary(values, depth_threshold)
            for name, values in depth_statistics.items()
        },
        "first_hit": {
            feature_name: {
                depth_name: _first_hit_summary(
                    feature_values,
                    depth_values,
                    feature_threshold=feature_threshold,
                    depth_threshold=depth_threshold,
                )
                for depth_name, depth_values in depth_statistics.items()
            }
            for feature_name, feature_values in feature_statistics.items()
        },
    }
