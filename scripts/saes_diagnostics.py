"""Non-claim diagnostics for SAES decision statistics and sparse coverage."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from saes.progressive_saes import ProgressiveSAES, paper_assignment_weights


def representative_indices(
    modified_mask: torch.Tensor,
    *,
    view_count: int,
    height: int,
    width: int,
    tile_size: int,
    primitives_per_pixel: int,
) -> torch.Tensor:
    """Recover every retained output index from sparse-SAEs' modified mask.

    L0 retains K probes while the declared L1 engineering path retains 2K
    native anchors.  The mask is the source of truth for which outputs survived
    in each early tile; assuming that every early tile has the L0 probe layout
    silently drops valid L1 anchors from downstream diagnostics.
    """
    expected = view_count * height * width * primitives_per_pixel
    flat = modified_mask.reshape(-1)
    if flat.numel() != expected:
        raise ValueError(
            f"modified mask has {flat.numel()} entries, expected {expected}"
        )
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
                retained_tile = tile_tensor[~flat[tile_tensor]]
                if retained_tile.numel() == 0:
                    raise ValueError("sparse SAES tile has no retained outputs")
                retained.extend(retained_tile.cpu().tolist())
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


def _audit_summary(values: list[float]) -> dict[str, float | int]:
    """Summarize one target-free attribute-error series."""
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "maximum": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "maximum": float(tensor.max().item()),
    }


def materialization_attribute_audit(
    original: Any,
    sparse: Any,
    *,
    features: torch.Tensor,
    depths: torch.Tensor,
    height: int,
    width: int,
    tile_size: int,
    feature_threshold: float,
    depth_threshold: float,
    view_count: int,
    decision_semantics: str,
    materialization: str = "representative",
) -> dict[str, Any]:
    """Compare sparse representatives to full Stage-3 attributes post hoc.

    This is a diagnostic oracle only.  It receives the already-materialized
    full tensor after the probe-only path has made its decisions, and never
    returns a routing signal or a parameter choice.  Its purpose is to locate
    which Gaussian-parameter family makes a sparse tile diverge before any
    image-quality evaluation is considered.
    """
    required = ("means", "covariances", "harmonics", "opacities")
    if any(not hasattr(value, name) for value in (original, sparse) for name in required):
        raise ValueError("materialization audit requires complete Gaussian attributes")
    if features.dim() != 5 or depths.dim() not in (4, 5):
        raise ValueError("materialization audit requires S1 features and S2 depths")
    if min(height, width, tile_size, view_count) <= 0:
        raise ValueError("materialization audit dimensions must be positive")
    if materialization not in (
        "representative",
        "transmittance-diagnostic",
        "virtual-reconstruction-diagnostic",
        "l1-primary-depth-reference-diagnostic",
        "conditional-anchor-transport-diagnostic",
    ):
        raise ValueError(f"unsupported audited SAES materialization: {materialization}")
    virtual_reconstruction = materialization in {
        "virtual-reconstruction-diagnostic",
    }
    position_count = view_count * height * width
    gaussian_count = int(original.means.shape[1])
    if (
        sparse.means.shape != original.means.shape
        or gaussian_count % position_count
        or original.covariances.shape[1] != gaussian_count
        or original.harmonics.shape[1] != gaussian_count
        or original.opacities.shape[1] != gaussian_count
    ):
        raise ValueError("materialization audit received incompatible Gaussian layout")
    primitives_per_pixel = gaussian_count // position_count
    statistic = {
        "probe-vector-first-hit": "raw-probe-vector-variance",
        "probe-channel-variance-first-hit": "raw-probe-mean-channel-variance",
        "probe-normalized-std-first-hit": "normalized-probe-vector-standard-deviation",
        "current": "normalized-probe-total-variance",
    }.get(decision_semantics)
    if statistic is None:
        raise ValueError(f"unsupported SAES decision semantics: {decision_semantics}")
    variances, normalized_features = ProgressiveSAES.classify_tiles_by_features(
        features,
        height,
        width,
        tile_size,
        per_view=True,
        statistic=statistic,
    )
    scorer = ProgressiveSAES(
        height,
        width,
        initial_tile_size=tile_size,
        feature_var_threshold=feature_threshold,
        depth_std_threshold=depth_threshold,
        view_count=view_count,
        primitives_per_pixel=primitives_per_pixel,
        decision_semantics=decision_semantics,
    )
    level_errors: dict[str, dict[str, list[float] | int]] = {
        level: {
            "tiles": 0,
            "mean_relative": [],
            "covariance_relative": [],
            "harmonic_relative": [],
            "opacity_absolute": [],
            "anchor_mean_update": [],
            "anchor_covariance_update": [],
            "anchor_harmonic_update": [],
            "anchor_opacity_update": [],
            "optical_depth_relative_error": [],
            "optical_depth_signed_error": [],
            "virtual_mean_relative": [],
            "virtual_covariance_relative": [],
            "virtual_harmonic_relative": [],
            "virtual_opacity_absolute": [],
        }
        for level in ("L0", "L1")
    }
    # These oracle comparisons are deliberately post-hoc.  They use the same
    # target-free routing and assignment signals as the sparse path, then
    # substitute the withheld full Stage-3 attributes only to expose where
    # selected-anchor materialization loses fidelity. L1 is kept separate
    # because it executes 2K selected outputs after its K primary probes route
    # the tile, whereas L0 executes K outputs.
    full_oracle_errors: dict[str, dict[str, list[float]]] = {
        level: {
            "mean_relative": [],
            "covariance_relative": [],
            "harmonic_relative": [],
            "opacity_absolute": [],
        }
        for level in ("L0", "L1")
    }
    # Per-representative oracle errors do not compare alternative legal L1
    # groupings fairly: a path may preserve an extra native anchor instead of
    # asking it to absorb a skipped position. This aggregate is still strictly
    # post-hoc, but compares the complete retained tile mixture against the
    # complete full-Stage-3 tile under the assignment masses used by that path.
    tile_mixture_errors: dict[str, dict[str, list[float]]] = {
        level: {
            "mean_relative": [],
            "covariance_relative": [],
            "harmonic_relative": [],
            "opacity_average_absolute": [],
            "optical_depth_relative": [],
        }
        for level in ("L0", "L1")
    }
    l1_oracle_errors_by_anchor_kind: dict[str, dict[str, list[float]]] = {
        kind: {
            "mean_relative": [],
            "covariance_relative": [],
            "harmonic_relative": [],
            "opacity_absolute": [],
        }
        for kind in ("primary_probe", "selected_lightweight_anchor")
    }
    # This is a post-hoc S2 oracle only. Neither reconstruction statistic feeds
    # routing, selection, or a run-time parameter choice.
    depth_reconstruction_errors: dict[str, dict[str, list[float]]] = {
        level: {
            "assignment_weighted_relative": [],
            "corner_bilinear_relative": [],
        }
        for level in ("L0", "L1")
    }

    def flat_index(view: int, local_y: int, local_x: int, slot: int) -> int:
        return (
            (view * height * width + local_y * width + local_x)
            * primitives_per_pixel
            + slot
        )

    def selected_anchor_depth(
        view: int, row: int, column: int, slot: int
    ) -> torch.Tensor:
        """Read one selected S2 anchor depth in the flattened SAES layout."""
        if depths.dim() == 5:
            values = depths[0, view, row * width + column].reshape(-1)
            return values[min(slot, values.numel() - 1)].to(
                device=original.means.device, dtype=original.means.dtype
            )
        return depths[0, view, row, column].reshape(()).to(
            device=original.means.device, dtype=original.means.dtype
        )

    def l1_selected_anchor_depths(
        *,
        view: int,
        tile_y: int,
        tile_x: int,
        slot: int,
        primary_positions: list[tuple[int, int]],
        retained_positions: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Read exactly the 2K S2 positions selected after an L1 decision.

        L1 classification is still based only on the K primary probes. Once it
        succeeds, the declared engineering path executes S2/S3 at deterministic
        2K anchors. This audit mirrors that boundary and never reads S2 values
        at the remaining skipped positions.
        """
        primary_count = len(primary_positions)
        if retained_positions[:primary_count] != primary_positions:
            raise ValueError("L1 retained positions must preserve the primary probes")
        return torch.stack(
            [
                selected_anchor_depth(view, tile_y + row, tile_x + column, slot)
                for row, column in retained_positions
            ]
        )

    tiles_h = height // tile_size
    tiles_w = width // tile_size
    for view in range(view_count):
        for tile_row in range(tiles_h):
            for tile_column in range(tiles_w):
                feature_variance = variances[(view, tile_row, tile_column)]
                assignment_feature_variance = scorer._assignment_feature_variance(
                    feature_variance
                )
                if feature_variance < feature_threshold:
                    level = "L0"
                    retained_positions = scorer.probe_positions
                elif scorer.check_depth_uniformity(
                    depths,
                    tile_row,
                    tile_column,
                    tile_size,
                    height,
                    width,
                    depth_threshold,
                    probe_positions=scorer.probe_positions,
                    view_index=view,
                    relative=False,
                ):
                    level = "L1"
                    retained_positions = scorer.lightweight_positions
                else:
                    continue
                errors = level_errors[level]
                errors["tiles"] = int(errors["tiles"]) + 1
                retained = set(retained_positions)
                tile_y = tile_row * tile_size
                tile_x = tile_column * tile_size
                for local_y, local_x in retained_positions:
                    for slot in range(primitives_per_pixel):
                        index = flat_index(view, tile_y + local_y, tile_x + local_x, slot)
                        for key, field in (
                            ("anchor_mean_update", "means"),
                            ("anchor_covariance_update", "covariances"),
                            ("anchor_harmonic_update", "harmonics"),
                            ("anchor_opacity_update", "opacities"),
                        ):
                            before = getattr(original, field)[0, index].float()
                            after = getattr(sparse, field)[0, index].float()
                            denominator = before.norm().clamp_min(1e-8)
                            errors[key].append(float((after - before).norm().div(denominator).item()))

                # Alpha compositing is linear in optical depth
                # tau=-log(1-alpha), not in alpha itself.  This post-hoc check
                # does not feed routing or materialization; it only reveals
                # whether a sparse tile preserves the aggregate opacity mass
                # of the full Stage-3 reference.
                for slot in range(primitives_per_pixel):
                    full_indices = torch.tensor(
                        [
                            flat_index(view, tile_y + local_y, tile_x + local_x, slot)
                            for local_y in range(tile_size)
                            for local_x in range(tile_size)
                        ],
                        device=original.opacities.device,
                        dtype=torch.long,
                    )
                    output_positions = (
                        [
                            (local_y, local_x)
                            for local_y in range(tile_size)
                            for local_x in range(tile_size)
                        ]
                        if virtual_reconstruction
                        else retained_positions
                    )
                    output_indices = torch.tensor(
                        [
                            flat_index(view, tile_y + local_y, tile_x + local_x, slot)
                            for local_y, local_x in output_positions
                        ],
                        device=original.opacities.device,
                        dtype=torch.long,
                    )
                    full_alpha = original.opacities[0, full_indices].float().reshape(-1)
                    sparse_alpha = sparse.opacities[0, output_indices].float().reshape(-1)
                    full_optical_depth = -torch.log1p(
                        -full_alpha.clamp(0.0, 1.0 - 1e-6)
                    ).sum()
                    sparse_optical_depth = -torch.log1p(
                        -sparse_alpha.clamp(0.0, 1.0 - 1e-6)
                    ).sum()
                    signed_error = (
                        (sparse_optical_depth - full_optical_depth)
                        / full_optical_depth.abs().clamp_min(1e-8)
                    )
                    errors["optical_depth_signed_error"].append(
                        float(signed_error.item())
                    )
                    errors["optical_depth_relative_error"].append(
                        float(signed_error.abs().item())
                    )
                for local_y in range(tile_size):
                    for local_x in range(tile_size):
                        if (local_y, local_x) in retained:
                            continue
                        for slot in range(primitives_per_pixel):
                            actual_index = flat_index(view, tile_y + local_y, tile_x + local_x, slot)
                            if virtual_reconstruction:
                                reconstructed_index = actual_index
                                error_keys = (
                                    "virtual_mean_relative",
                                    "virtual_covariance_relative",
                                    "virtual_harmonic_relative",
                                    "virtual_opacity_absolute",
                                )
                            else:
                                nearest_y, nearest_x = min(
                                    retained_positions,
                                    key=lambda point: (
                                        (local_y - point[0]) ** 2
                                        + (local_x - point[1]) ** 2,
                                        point,
                                    ),
                                )
                                reconstructed_index = flat_index(
                                    view, tile_y + nearest_y, tile_x + nearest_x, slot
                                )
                                error_keys = (
                                    "mean_relative",
                                    "covariance_relative",
                                    "harmonic_relative",
                                    "opacity_absolute",
                                )
                            actual_mean = original.means[0, actual_index].float()
                            representative_mean = sparse.means[0, reconstructed_index].float()
                            actual_covariance = original.covariances[0, actual_index].float()
                            representative_covariance = sparse.covariances[0, reconstructed_index].float()
                            actual_harmonic = original.harmonics[0, actual_index].float()
                            representative_harmonic = sparse.harmonics[0, reconstructed_index].float()
                            errors[error_keys[0]].append(
                                float(
                                    (representative_mean - actual_mean).norm()
                                    .div(actual_mean.norm().clamp_min(1e-8))
                                    .item()
                                )
                            )
                            errors[error_keys[1]].append(
                                float(
                                    (representative_covariance - actual_covariance).norm()
                                    .div(actual_covariance.norm().clamp_min(1e-8))
                                    .item()
                                )
                            )
                            errors[error_keys[2]].append(
                                float(
                                    (representative_harmonic - actual_harmonic).norm()
                                    .div(actual_harmonic.norm().clamp_min(1e-8))
                                    .item()
                                )
                            )
                            errors[error_keys[3]].append(
                                float(
                                    (sparse.opacities[0, reconstructed_index].float()
                                    - original.opacities[0, actual_index].float())
                                    .abs()
                                    .item()
                                )
                            )

                # Build the exact full-Stage-3 counterpart only after routing
                # has completed. L1 first routes on primary probes, then uses
                # only its selected 2K native anchors; the remaining positions
                # stay unavailable to both the mechanism and this audit's
                # assignment calculation.
                probe_positions = retained_positions
                non_probe_positions = [
                    (local_y, local_x)
                    for local_y in range(tile_size)
                    for local_x in range(tile_size)
                    if (local_y, local_x) not in retained
                ]
                view_features = normalized_features[view]
                coordinate_scale = max(tile_size - 1, 1)
                for slot in range(primitives_per_pixel):
                    l1_anchor_depths = (
                        l1_selected_anchor_depths(
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            slot=slot,
                            primary_positions=scorer.probe_positions,
                            retained_positions=probe_positions,
                        )
                        if level == "L1"
                        else None
                    )
                    anchor_depths = (
                        l1_anchor_depths
                        if l1_anchor_depths is not None
                        else torch.stack(
                            [
                                selected_anchor_depth(
                                    view,
                                    tile_y + local_y,
                                    tile_x + local_x,
                                    slot,
                                )
                                for local_y, local_x in probe_positions
                            ]
                        )
                    )
                    # The L1 reliability reference is always the K routing
                    # probes. Extra 2K anchors are an execution expansion and
                    # do not enter the paper's mean/std normalization.
                    depth_reference_depths = (
                        l1_anchor_depths[: len(scorer.probe_positions)]
                        if level == "L1"
                        else None
                    )
                    assignment_rows = []
                    for local_y, local_x in non_probe_positions:
                        gy, gx = tile_y + local_y, tile_x + local_x
                        feature_i = view_features[:, gy, gx]
                        spatial_distances = []
                        feature_distances = []
                        for probe_y, probe_x in probe_positions:
                            probe_feature = view_features[
                                :, tile_y + probe_y, tile_x + probe_x
                            ]
                            spatial_distances.append(
                                ((local_y - probe_y) / coordinate_scale) ** 2
                                + ((local_x - probe_x) / coordinate_scale) ** 2
                            )
                            feature_distances.append(
                                float((feature_i - probe_feature).square().sum().item())
                            )
                        assignment_rows.append(
                            paper_assignment_weights(
                                torch.tensor(
                                    spatial_distances,
                                    device=original.means.device,
                                    dtype=original.means.dtype,
                                ),
                                torch.tensor(
                                    feature_distances,
                                    device=original.means.device,
                                    dtype=original.means.dtype,
                                ),
                                feature_variance=assignment_feature_variance,
                                beta_x=scorer.beta_x,
                                beta_f=scorer.beta_f,
                                level=level,
                                probe_depths=l1_anchor_depths,
                                depth_reference_depths=depth_reference_depths,
                                beta_d=scorer.beta_d if level == "L1" else None,
                            )
                        )
                    assignments = torch.stack(assignment_rows, dim=0)
                    assignment_depths = assignments @ anchor_depths
                    if tile_size > 1:
                        corner_positions = (
                            (0, 0),
                            (0, tile_size - 1),
                            (tile_size - 1, 0),
                            (tile_size - 1, tile_size - 1),
                        )
                        corner_depths = {
                            position: selected_anchor_depth(
                                view,
                                tile_y + position[0],
                                tile_x + position[1],
                                slot,
                            )
                            for position in corner_positions
                        }
                        for row, (local_y, local_x) in enumerate(non_probe_positions):
                            actual_depth = selected_anchor_depth(
                                view, tile_y + local_y, tile_x + local_x, slot
                            )
                            denominator = actual_depth.abs().clamp_min(1e-8)
                            depth_reconstruction_errors[level][
                                "assignment_weighted_relative"
                            ].append(
                                float(
                                    (assignment_depths[row] - actual_depth)
                                    .abs()
                                    .div(denominator)
                                    .item()
                                )
                            )
                            row_fraction = local_y / (tile_size - 1)
                            column_fraction = local_x / (tile_size - 1)
                            top = (
                                corner_depths[(0, 0)] * (1.0 - column_fraction)
                                + corner_depths[(0, tile_size - 1)] * column_fraction
                            )
                            bottom = (
                                corner_depths[(tile_size - 1, 0)]
                                * (1.0 - column_fraction)
                                + corner_depths[(tile_size - 1, tile_size - 1)]
                                * column_fraction
                            )
                            bilinear_depth = (
                                top * (1.0 - row_fraction)
                                + bottom * row_fraction
                            )
                            depth_reconstruction_errors[level][
                                "corner_bilinear_relative"
                            ].append(
                                float(
                                    (bilinear_depth - actual_depth)
                                    .abs()
                                    .div(denominator)
                                    .item()
                                )
                            )
                    probe_indices = torch.tensor(
                        [
                            flat_index(view, tile_y + local_y, tile_x + local_x, slot)
                            for local_y, local_x in probe_positions
                        ],
                        device=original.means.device,
                        dtype=torch.long,
                    )
                    non_probe_indices = torch.tensor(
                        [
                            flat_index(view, tile_y + local_y, tile_x + local_x, slot)
                            for local_y, local_x in non_probe_positions
                        ],
                        device=original.means.device,
                        dtype=torch.long,
                    )
                    # This comparison is intentionally after the sparse path
                    # has made all routing and aggregation decisions. Each
                    # retained output receives its own anchor mass plus the
                    # soft-assignment mass of the skipped positions it
                    # absorbs. Native L1 residual anchors retain unit mass.
                    if virtual_reconstruction:
                        output_positions = [
                            (local_y, local_x)
                            for local_y in range(tile_size)
                            for local_x in range(tile_size)
                        ]
                        output_masses = torch.ones(
                            len(output_positions),
                            device=original.means.device,
                            dtype=original.means.dtype,
                        )
                    else:
                        output_positions = retained_positions
                        output_masses = torch.ones(
                            len(output_positions),
                            device=original.means.device,
                            dtype=original.means.dtype,
                        )
                        output_masses[: len(probe_positions)] += assignments.sum(
                            dim=0
                        )
                    output_indices = torch.tensor(
                        [
                            flat_index(view, tile_y + local_y, tile_x + local_x, slot)
                            for local_y, local_x in output_positions
                        ],
                        device=original.means.device,
                        dtype=torch.long,
                    )
                    full_tile_indices = torch.tensor(
                        [
                            flat_index(view, tile_y + local_y, tile_x + local_x, slot)
                            for local_y in range(tile_size)
                            for local_x in range(tile_size)
                        ],
                        device=original.means.device,
                        dtype=torch.long,
                    )
                    full_weights = torch.full(
                        (full_tile_indices.numel(),),
                        1.0 / full_tile_indices.numel(),
                        device=original.means.device,
                        dtype=original.means.dtype,
                    )
                    output_weights = output_masses / output_masses.sum().clamp_min(1e-8)
                    full_means = original.means[0, full_tile_indices].float()
                    output_means = sparse.means[0, output_indices].float()
                    full_covariances = original.covariances[0, full_tile_indices].float()
                    output_covariances = sparse.covariances[0, output_indices].float()
                    full_harmonics = original.harmonics[0, full_tile_indices].float()
                    output_harmonics = sparse.harmonics[0, output_indices].float()
                    full_opacities = original.opacities[0, full_tile_indices].float().reshape(-1)
                    output_opacities = sparse.opacities[0, output_indices].float().reshape(-1)
                    full_weights_f = full_weights.to(full_means)
                    output_weights_f = output_weights.to(output_means)
                    full_mean = torch.einsum("n,ni->i", full_weights_f, full_means)
                    output_mean = torch.einsum(
                        "n,ni->i", output_weights_f, output_means
                    )
                    full_centered = full_means - full_mean
                    output_centered = output_means - output_mean
                    full_covariance = torch.einsum(
                        "n,nij->ij",
                        full_weights_f,
                        full_covariances
                        + torch.einsum("ni,nj->nij", full_centered, full_centered),
                    )
                    output_covariance = torch.einsum(
                        "n,nij->ij",
                        output_weights_f,
                        output_covariances
                        + torch.einsum("ni,nj->nij", output_centered, output_centered),
                    )
                    full_harmonic = torch.einsum(
                        "n,n...->...", full_weights_f, full_harmonics
                    )
                    output_harmonic = torch.einsum(
                        "n,n...->...", output_weights_f, output_harmonics
                    )
                    full_opacity_average = torch.einsum(
                        "n,n->", full_weights_f, full_opacities
                    )
                    output_opacity_average = torch.einsum(
                        "n,n->", output_weights_f, output_opacities
                    )
                    full_optical_depth = -torch.log1p(
                        -full_opacities.clamp(0.0, 1.0 - 1e-6)
                    ).sum()
                    output_optical_depth = -torch.log1p(
                        -output_opacities.clamp(0.0, 1.0 - 1e-6)
                    ).sum()
                    tile_mixture_errors[level]["mean_relative"].append(
                        float(
                            (output_mean - full_mean)
                            .norm()
                            .div(full_mean.norm().clamp_min(1e-8))
                            .item()
                        )
                    )
                    tile_mixture_errors[level]["covariance_relative"].append(
                        float(
                            (output_covariance - full_covariance)
                            .norm()
                            .div(full_covariance.norm().clamp_min(1e-8))
                            .item()
                        )
                    )
                    tile_mixture_errors[level]["harmonic_relative"].append(
                        float(
                            (output_harmonic - full_harmonic)
                            .norm()
                            .div(full_harmonic.norm().clamp_min(1e-8))
                            .item()
                        )
                    )
                    tile_mixture_errors[level]["opacity_average_absolute"].append(
                        float((output_opacity_average - full_opacity_average).abs().item())
                    )
                    tile_mixture_errors[level]["optical_depth_relative"].append(
                        float(
                            (output_optical_depth - full_optical_depth)
                            .abs()
                            .div(full_optical_depth.abs().clamp_min(1e-8))
                            .item()
                        )
                    )
                    for probe_offset, representative_index in enumerate(probe_indices):
                        l1_anchor_kind = (
                            "primary_probe"
                            if probe_offset < len(scorer.probe_positions)
                            else "selected_lightweight_anchor"
                        ) if level == "L1" else None
                        weights = torch.cat(
                            (
                                torch.ones(1, device=original.means.device, dtype=original.means.dtype),
                                assignments[:, probe_offset],
                            )
                        )
                        weights = weights / weights.sum().clamp_min(1e-8)
                        contributors = torch.cat(
                            (representative_index.reshape(1), non_probe_indices)
                        )
                        oracle_means = original.means[0, contributors].float()
                        oracle_covariances = original.covariances[0, contributors].float()
                        oracle_harmonics = original.harmonics[0, contributors].float()
                        oracle_opacities = original.opacities[0, contributors].float()
                        oracle_weights = weights.to(oracle_means)
                        oracle_mean = torch.einsum("n,ni->i", oracle_weights, oracle_means)
                        centered = oracle_means - oracle_mean
                        oracle_covariance = torch.einsum(
                            "n,nij->ij",
                            oracle_weights,
                            oracle_covariances
                            + torch.einsum("ni,nj->nij", centered, centered),
                        )
                        oracle_harmonic = torch.einsum(
                            "n,n...->...", oracle_weights, oracle_harmonics
                        )
                        oracle_opacity = torch.einsum(
                            "n,n->", oracle_weights, oracle_opacities.reshape(-1)
                        )
                        sparse_mean = sparse.means[0, representative_index].float()
                        sparse_covariance = sparse.covariances[0, representative_index].float()
                        sparse_harmonic = sparse.harmonics[0, representative_index].float()
                        sparse_opacity = sparse.opacities[0, representative_index].float()
                        mean_relative = float(
                            (sparse_mean - oracle_mean).norm()
                            .div(oracle_mean.norm().clamp_min(1e-8))
                            .item()
                        )
                        covariance_relative = float(
                            (sparse_covariance - oracle_covariance).norm()
                            .div(oracle_covariance.norm().clamp_min(1e-8))
                            .item()
                        )
                        harmonic_relative = float(
                            (sparse_harmonic - oracle_harmonic).norm()
                            .div(oracle_harmonic.norm().clamp_min(1e-8))
                            .item()
                        )
                        opacity_absolute = float((sparse_opacity - oracle_opacity).abs().item())
                        oracle_metric_values = {
                            "mean_relative": mean_relative,
                            "covariance_relative": covariance_relative,
                            "harmonic_relative": harmonic_relative,
                            "opacity_absolute": opacity_absolute,
                        }
                        for metric, value in oracle_metric_values.items():
                            full_oracle_errors[level][metric].append(value)
                            if l1_anchor_kind is not None:
                                l1_oracle_errors_by_anchor_kind[l1_anchor_kind][metric].append(
                                    value
                                )

    levels = {}
    for level, errors in level_errors.items():
        level_record = {
            "tiles": int(errors["tiles"]),
            "non_probe_attribute_error": {
                key: _audit_summary(errors[key])
                for key in (
                    "mean_relative",
                    "covariance_relative",
                    "harmonic_relative",
                    "opacity_absolute",
                )
            },
            "retained_anchor_update": {
                key: _audit_summary(errors[key])
                for key in (
                    "anchor_mean_update",
                    "anchor_covariance_update",
                    "anchor_harmonic_update",
                    "anchor_opacity_update",
                )
            },
            "optical_depth_conservation": {
                "relative_error": _audit_summary(
                    errors["optical_depth_relative_error"]
                ),
                "signed_error": _audit_summary(
                    errors["optical_depth_signed_error"]
                ),
            },
        }
        if virtual_reconstruction:
            level_record["virtual_non_probe_reconstruction_error"] = {
                output_key: _audit_summary(errors[source_key])
                for output_key, source_key in (
                    ("mean_relative", "virtual_mean_relative"),
                    ("covariance_relative", "virtual_covariance_relative"),
                    ("harmonic_relative", "virtual_harmonic_relative"),
                    ("opacity_absolute", "virtual_opacity_absolute"),
                )
            }
        levels[level] = level_record

    return {
        "schema_version": "1.0",
        "kind": "saes_probe_materialization_attribute_audit",
        "paper_result_eligible": False,
        "routing_signal_used": False,
        "full_stage3_reference_use": "posthoc-diagnostic-only",
        "materialization": materialization,
        "non_probe_comparison": (
            "direct reconstructed non-probe descriptor versus posthoc full Stage-3"
            if virtual_reconstruction
            else "nearest retained descriptor versus posthoc full Stage-3"
        ),
        "full_oracle_representative_error_applicable": not virtual_reconstruction,
        "full_oracle_l1_anchor_semantics": (
            "posthoc full Stage-3 attributes at all 2K retained positions; "
            "L1 assignment depths are native S2 values at only those selected anchors"
        ),
        "feature_statistic": statistic,
        "levels": levels,
        "full_oracle_representative_error_l0": {
            key: _audit_summary(values)
            for key, values in full_oracle_errors["L0"].items()
        },
        "full_oracle_representative_error_l1": {
            key: _audit_summary(values)
            for key, values in full_oracle_errors["L1"].items()
        },
        "full_oracle_representative_error_l1_by_anchor_kind": {
            kind: {
                metric: _audit_summary(values)
                for metric, values in errors.items()
            }
            for kind, errors in l1_oracle_errors_by_anchor_kind.items()
        },
        "full_oracle_tile_mixture_error": {
            level: {
                metric: _audit_summary(values)
                for metric, values in errors.items()
            }
            for level, errors in tile_mixture_errors.items()
        },
        "posthoc_nonprobe_s2_depth_reconstruction": {
            "reference_use": "posthoc full S2 non-probe depth only",
            "assignment_weighted_relative_error": {
                level: _audit_summary(errors["assignment_weighted_relative"])
                for level, errors in depth_reconstruction_errors.items()
            },
            "corner_bilinear_relative_error": {
                level: _audit_summary(errors["corner_bilinear_relative"])
                for level, errors in depth_reconstruction_errors.items()
            },
        },
    }


def raw_s3_probe_interpolation_audit(
    raw_descriptors: torch.Tensor,
    depths: torch.Tensor,
    opacities: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    features: torch.Tensor,
    height: int,
    width: int,
    tile_size: int,
    feature_threshold: float,
    depth_threshold: float,
    decision_semantics: str,
) -> dict[str, Any]:
    """Audit raw S3 per-position interpolation after routing, before S4.

    The raw tensors are complete only because this is a post-hoc oracle.  The
    reconstruction side of the comparison reads selected anchor descriptors,
    S1 features, and selected S2 depths only; withheld raw descriptors are
    consulted solely to report error.  It is therefore safe for diagnosing the
    fidelity of direct raw-descriptor interpolation, but it does not execute a
    retained-anchor first/second-moment merge or emit a runtime candidate. It
    cannot drive a route, parameter choice, or conclusion about the distinct
    post-GGU primitive-moment implementation.
    """
    if (
        raw_descriptors.ndim != 5
        or depths.ndim != 5
        or opacities.ndim != 5
        or coordinates.ndim != 5
        or features.ndim != 5
    ):
        raise ValueError("raw S3 audit expects five-dimensional tensors")
    batches, view_count, pixel_count, surface_count, descriptor_dim = (
        raw_descriptors.shape
    )
    if batches != 1 or pixel_count != height * width or descriptor_dim <= 7:
        raise ValueError("raw S3 descriptor layout is incompatible with the image")
    if coordinates.shape != (1, view_count, pixel_count, surface_count, 2):
        raise ValueError("raw S3 coordinates do not align with descriptor layout")
    if depths.shape[:4] != (1, view_count, pixel_count, surface_count):
        raise ValueError("raw S3 depths do not align with descriptor layout")
    if opacities.shape != depths.shape or features.shape[:2] != (1, view_count):
        raise ValueError("raw S3 auxiliary tensors do not align with descriptors")
    if min(height, width, tile_size, view_count, surface_count) <= 0:
        raise ValueError("raw S3 audit dimensions must be positive")
    if height % tile_size or width % tile_size:
        raise ValueError("raw S3 audit requires dimensions divisible by tile size")

    primitives_per_pixel = surface_count * depths.shape[-1]
    statistic = {
        "probe-vector-first-hit": "raw-probe-vector-variance",
        "probe-channel-variance-first-hit": "raw-probe-mean-channel-variance",
        "probe-normalized-std-first-hit": "normalized-probe-vector-standard-deviation",
        "current": "normalized-probe-total-variance",
    }.get(decision_semantics)
    if statistic is None:
        raise ValueError(f"unsupported SAES decision semantics: {decision_semantics}")
    variances, normalized_features = ProgressiveSAES.classify_tiles_by_features(
        features,
        height,
        width,
        tile_size,
        per_view=True,
        statistic=statistic,
    )
    scorer = ProgressiveSAES(
        height,
        width,
        initial_tile_size=tile_size,
        feature_var_threshold=feature_threshold,
        depth_std_threshold=depth_threshold,
        view_count=view_count,
        primitives_per_pixel=primitives_per_pixel,
        decision_semantics=decision_semantics,
    )
    metric_names = (
        "coordinate_relative",
        "depth_relative",
        "raw_scale_relative",
        "quaternion_sign_invariant",
        "harmonic_relative",
        "opacity_absolute",
    )
    errors: dict[str, dict[str, dict[str, list[float]]]] = {
        level: {
            method: {metric: [] for metric in metric_names}
            for method in ("assignment_interpolation", "nearest_anchor")
        }
        for level in ("L0", "L1")
    }
    tile_counts = {"L0": 0, "L1": 0}
    assignment_weight_sum_error_max = 0.0
    coordinate_scale = max(tile_size - 1, 1)

    def pixel_index(row: int, column: int) -> int:
        return row * width + column

    def normalized_quaternion(value: torch.Tensor) -> torch.Tensor:
        return value / value.norm().clamp_min(1e-8)

    def relative_error(predicted: torch.Tensor, actual: torch.Tensor) -> float:
        return float(
            (predicted - actual)
            .norm()
            .div(actual.norm().clamp_min(1e-8))
            .item()
        )

    for view in range(view_count):
        for tile_row in range(height // tile_size):
            for tile_column in range(width // tile_size):
                feature_variance = variances[(view, tile_row, tile_column)]
                if feature_variance < feature_threshold:
                    level = "L0"
                    retained_positions = scorer.probe_positions
                elif scorer.check_depth_uniformity(
                    depths,
                    tile_row,
                    tile_column,
                    tile_size,
                    height,
                    width,
                    depth_threshold,
                    probe_positions=scorer.probe_positions,
                    view_index=view,
                    relative=False,
                ):
                    level = "L1"
                    retained_positions = scorer.lightweight_positions
                else:
                    continue
                tile_counts[level] += 1
                tile_y = tile_row * tile_size
                tile_x = tile_column * tile_size
                retained = set(retained_positions)
                skipped_positions = [
                    (local_y, local_x)
                    for local_y in range(tile_size)
                    for local_x in range(tile_size)
                    if (local_y, local_x) not in retained
                ]
                assignment_feature_variance = scorer._assignment_feature_variance(
                    feature_variance
                )
                view_features = normalized_features[view]
                for slot in range(primitives_per_pixel):
                    surface = slot // depths.shape[-1]
                    gaussian_slot = slot % depths.shape[-1]
                    anchor_pixels = [
                        pixel_index(tile_y + local_y, tile_x + local_x)
                        for local_y, local_x in retained_positions
                    ]
                    anchor_raw = raw_descriptors[0, view, anchor_pixels, surface]
                    anchor_coordinates = coordinates[0, view, anchor_pixels, surface]
                    anchor_depths = depths[0, view, anchor_pixels, surface, gaussian_slot]
                    anchor_opacities = opacities[
                        0, view, anchor_pixels, surface, gaussian_slot
                    ]
                    anchor_quaternions = torch.stack(
                        [normalized_quaternion(value) for value in anchor_raw[:, 3:7]],
                        dim=0,
                    )
                    reference_quaternion = anchor_quaternions[0]
                    signs = torch.where(
                        (anchor_quaternions @ reference_quaternion).unsqueeze(-1)
                        < 0.0,
                        -torch.ones_like(anchor_quaternions[:, :1]),
                        torch.ones_like(anchor_quaternions[:, :1]),
                    )
                    aligned_quaternions = anchor_quaternions * signs
                    for local_y, local_x in skipped_positions:
                        global_y = tile_y + local_y
                        global_x = tile_x + local_x
                        feature_i = view_features[:, global_y, global_x]
                        spatial_distances = []
                        feature_distances = []
                        for probe_y, probe_x in retained_positions:
                            probe_feature = view_features[
                                :, tile_y + probe_y, tile_x + probe_x
                            ]
                            spatial_distances.append(
                                ((local_y - probe_y) / coordinate_scale) ** 2
                                + ((local_x - probe_x) / coordinate_scale) ** 2
                            )
                            feature_distances.append(
                                float((feature_i - probe_feature).square().sum().item())
                            )
                        assignment = paper_assignment_weights(
                            torch.tensor(
                                spatial_distances,
                                device=raw_descriptors.device,
                                dtype=raw_descriptors.dtype,
                            ),
                            torch.tensor(
                                feature_distances,
                                device=raw_descriptors.device,
                                dtype=raw_descriptors.dtype,
                            ),
                            feature_variance=assignment_feature_variance,
                            beta_x=scorer.beta_x,
                            beta_f=scorer.beta_f,
                            level=level,
                            probe_depths=anchor_depths if level == "L1" else None,
                            beta_d=scorer.beta_d if level == "L1" else None,
                        )
                        assignment_weight_sum_error_max = max(
                            assignment_weight_sum_error_max,
                            float((assignment.sum() - 1.0).abs().item()),
                        )
                        actual_pixel = pixel_index(global_y, global_x)
                        actual_raw = raw_descriptors[0, view, actual_pixel, surface]
                        actual_coordinate = coordinates[
                            0, view, actual_pixel, surface
                        ]
                        actual_depth = depths[
                            0, view, actual_pixel, surface, gaussian_slot
                        ]
                        actual_opacity = opacities[
                            0, view, actual_pixel, surface, gaussian_slot
                        ]
                        actual_quaternion = normalized_quaternion(actual_raw[3:7])
                        predicted = {
                            "coordinate_relative": assignment @ anchor_coordinates,
                            "depth_relative": torch.dot(assignment, anchor_depths),
                            "raw_scale_relative": assignment @ anchor_raw[:, :3],
                            "quaternion_sign_invariant": normalized_quaternion(
                                assignment @ aligned_quaternions
                            ),
                            "harmonic_relative": assignment @ anchor_raw[:, 7:],
                            "opacity_absolute": torch.dot(assignment, anchor_opacities),
                        }
                        nearest_offset = min(
                            range(len(retained_positions)),
                            key=lambda index: (
                                (local_y - retained_positions[index][0]) ** 2
                                + (local_x - retained_positions[index][1]) ** 2,
                                retained_positions[index],
                            ),
                        )
                        nearest = {
                            "coordinate_relative": anchor_coordinates[nearest_offset],
                            "depth_relative": anchor_depths[nearest_offset],
                            "raw_scale_relative": anchor_raw[nearest_offset, :3],
                            "quaternion_sign_invariant": anchor_quaternions[nearest_offset],
                            "harmonic_relative": anchor_raw[nearest_offset, 7:],
                            "opacity_absolute": anchor_opacities[nearest_offset],
                        }
                        actual = {
                            "coordinate_relative": actual_coordinate,
                            "depth_relative": actual_depth,
                            "raw_scale_relative": actual_raw[:3],
                            "quaternion_sign_invariant": actual_quaternion,
                            "harmonic_relative": actual_raw[7:],
                            "opacity_absolute": actual_opacity,
                        }
                        for method, values in (
                            ("assignment_interpolation", predicted),
                            ("nearest_anchor", nearest),
                        ):
                            for metric, estimate in values.items():
                                if metric == "quaternion_sign_invariant":
                                    error = float(
                                        (1.0 - estimate.dot(actual[metric]).abs())
                                        .clamp_min(0.0)
                                        .item()
                                    )
                                elif metric == "opacity_absolute":
                                    error = float(
                                        (estimate - actual[metric]).abs().item()
                                    )
                                elif metric == "depth_relative":
                                    error = float(
                                        (estimate - actual[metric])
                                        .abs()
                                        .div(actual[metric].abs().clamp_min(1e-8))
                                        .item()
                                    )
                                else:
                                    error = relative_error(estimate, actual[metric])
                                errors[level][method][metric].append(error)

    return {
        "schema_version": "1.0",
        "kind": "saes_s3_raw_descriptor_audit",
        "paper_result_eligible": False,
        "routing_signal_used": False,
        "full_s3_reference_use": "posthoc-diagnostic-only",
        "aggregate_moment_matching_executed": False,
        "retained_descriptor_output_emitted": False,
        "audit_scope": (
            "per-skipped-position raw-S3 interpolation oracle; not a "
            "retained-anchor moment-matching implementation"
        ),
        "target_rgb_accessed": False,
        "feature_statistic": statistic,
        "feature_grid": {
            "input_height": int(features.shape[-2]),
            "input_width": int(features.shape[-1]),
            "routing_height": height,
            "routing_width": width,
            "resampling": "bilinear-align_corners-false-via-ProgressiveSAES",
        },
        "raw_descriptor_layout": {
            "views": view_count,
            "pixels_per_view": pixel_count,
            "surfaces": surface_count,
            "gaussians_per_surface": depths.shape[-1],
            "descriptor_dimension": descriptor_dim,
        },
        "tile_counts": tile_counts,
        "assignment_weight_sum_error_max": assignment_weight_sum_error_max,
        "errors": {
            level: {
                method: {
                    metric: _audit_summary(values)
                    for metric, values in metric_values.items()
                }
                for method, metric_values in methods.items()
            }
            for level, methods in errors.items()
        },
    }
