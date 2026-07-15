#!/usr/bin/env python3
"""Replay the paper sensitivity grid from one model-resident sample trace."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from fsdr import FSDRSimulator
from saes import apply_progressive_saes
from scripts.result_record import build_quality_record


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


def _mean_metrics(per_view: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: sum(item[metric] for item in per_view) / len(per_view)
        for metric in ("psnr_db", "ssim", "lpips")
    }


def replay_sample(
    *,
    studies: dict[str, dict[str, Any]],
    config: Any,
    gaussians: Any,
    features: torch.Tensor,
    depths: torch.Tensor,
    target_images: torch.Tensor,
    render: Callable[[Any], torch.Tensor],
    gaussian_type: type,
    savings_tracker_type: type,
    cycle_counter_type: type,
    image_height: int,
    image_width: int,
    feature_cycles: int,
    depth_cycles: int,
    dp_core_cycles: int,
    cost_volume_cycles: int,
    gauss_gen_cycles: int,
    s1_cnn_cycles: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not torch.is_tensor(features) or not torch.is_tensor(depths):
        raise ValueError("sensitivity replay requires feature and depth tensors")
    if min(
        feature_cycles,
        depth_cycles,
        dp_core_cycles,
        cost_volume_cycles,
        gauss_gen_cycles,
        s1_cnn_cycles,
    ) <= 0:
        raise ValueError("sensitivity replay requires complete positive cycle evidence")

    means = gaussians.means.detach()
    covariances = gaussians.covariances.detach()
    harmonics = gaussians.harmonics.detach()
    opacities = gaussians.opacities.detach()
    gaussian_count = means.shape[1]
    ggu_counter = cycle_counter_type(ggu_pe_count=config.ggu_pe_count)
    ggu_counter.add_ggu(gaussian_count, sh_degree=config.sh_degree)
    ggu_cycles = ggu_counter.get_summary()["ggu_cycles"]

    from lpips import LPIPS
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(
        target_images.device
    )
    lpips_metric = LPIPS(net="vgg").to(target_images.device).eval()

    def view_metrics(images: torch.Tensor) -> list[dict[str, float]]:
        rows = []
        for image, target in zip(images, target_images):
            image_batch = image.unsqueeze(0)
            target_batch = target.unsqueeze(0)
            mse = F.mse_loss(image_batch, target_batch)
            psnr = -10 * torch.log10(mse).item()
            ssim = ssim_metric(image_batch, target_batch).item()
            with torch.no_grad():
                lpips = lpips_metric(
                    image_batch, target_batch, normalize=True
                ).reshape(-1).mean().item()
            rows.append({"psnr_db": psnr, "ssim": ssim, "lpips": lpips})
        return rows

    original = gaussian_type(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )
    baseline_metrics = _mean_metrics(view_metrics(render(original)))
    defaults = {
        "fsdr_cache_size": config.fsdr_cache_size,
        "fsdr_hamming_threshold": config.fsdr_hamming_threshold,
        "saes_feature_variance": config.feature_var_threshold,
        "saes_depth_variance": config.depth_std_threshold,
        "saes_tile_size": config.tile_size,
    }
    features_up = F.interpolate(
        features[0],
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    ).mean(dim=0)
    frame_features = features_up.permute(1, 2, 0).reshape(
        image_height * image_width, features_up.shape[0]
    )
    if depths.dim() == 5:
        frame_depths = depths[0, 0, : image_height * image_width, 0, 0]
    elif depths.dim() == 4:
        frame_depths = depths[0, 0, :image_height, :image_width].reshape(-1)
    else:
        raise ValueError("unsupported sensitivity depth tensor")
    replays = []
    for study, study_config in studies.items():
        for value in study_config["values"]:
            parameters = dict(defaults)
            parameters[study] = value
            trial_gaussians = gaussian_type(
                means=means.clone(),
                covariances=covariances.clone(),
                harmonics=harmonics.clone(),
                opacities=opacities.clone(),
            )
            modified_mask, saes_stats, _ = apply_progressive_saes(
                trial_gaussians,
                image_height,
                image_width,
                int(parameters["saes_tile_size"]),
                gpp=1,
                feature_var_threshold=float(parameters["saes_feature_variance"]),
                depth_std_threshold=float(parameters["saes_depth_variance"]),
                features=features,
                depths=depths,
                cross_check_threshold=config.saes_cross_check,
            )
            fsdr = FSDRSimulator(
                feature_dim=config.feature_dim,
                cache_size=int(parameters["fsdr_cache_size"]),
                hamming_threshold=int(parameters["fsdr_hamming_threshold"]),
                reuse_hamming=int(parameters["fsdr_hamming_threshold"]),
                reuse_spatial=config.fsdr_reuse_spatial,
                reuse_confidence=config.fsdr_reuse_confidence,
                num_depth_candidates=config.num_depth_candidates,
            )
            savings = savings_tracker_type()
            savings.record_saes(
                total_pixels=image_height * image_width, saes_stats=saes_stats
            )
            paths = fsdr.process_frame(frame_features, frame_depths, image_width)
            for pixel_index, path in enumerate(paths):
                savings.record_fsdr_pixel(
                    path, reused=(pixel_index in fsdr.reuse_data)
                )

            combo_means = trial_gaussians.means.clone()
            for pixel_index, reuse in fsdr.reuse_data.items():
                if not modified_mask[pixel_index]:
                    ratio = reuse["depth_ratio"]
                    if not math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
                        combo_means[0, pixel_index] = means[0, pixel_index] * ratio
            combined = gaussian_type(
                means=combo_means,
                covariances=trial_gaussians.covariances,
                harmonics=trial_gaussians.harmonics,
                opacities=trial_gaussians.opacities,
            )
            scarf_metrics = _mean_metrics(view_metrics(render(combined)))
            quality = build_quality_record(baseline_metrics, scarf_metrics)
            ablation = savings.compute_ablation(
                feature_cycles=feature_cycles,
                depth_cycles=depth_cycles,
                ggu_cycles=ggu_cycles,
                dp_core_cycles=dp_core_cycles,
                gauss_gen_cycles=gauss_gen_cycles,
                cost_volume_cycles=cost_volume_cycles,
                s1_cnn_cycles=s1_cnn_cycles,
                fsdr_narrowing_ratio=(
                    1.0
                    - config.fsdr_narrowed_candidates
                    / config.num_depth_candidates
                ),
            )
            baseline_cycles = ablation["asic"]["eff_total"]
            scarf_cycles = ablation["asic_fsdr_saes"]["eff_total"]
            replays.append(
                {
                    "study": study,
                    "value": value,
                    "is_default": value == study_config["default"],
                    "quality": quality,
                    "performance": {
                        "baseline_cycles": baseline_cycles,
                        "scarf_cycles": scarf_cycles,
                        "speedup": baseline_cycles / scarf_cycles,
                        "baseline_source": "scarf_no_optimization_cycles",
                        "cycle_source": "sensitivity_trace_replay",
                    },
                    "fsdr_saes": {
                        "fsdr": fsdr.get_summary(),
                        "saes": saes_stats,
                    },
                }
            )
    trace = {
        "neural_forward_passes": 1,
        "feature_tensor": tensor_record(features),
        "depth_tensor": tensor_record(depths),
        "gaussian_means": tensor_record(means),
        "cycle_breakdown": {
            "feature": feature_cycles,
            "depth": depth_cycles,
            "dp_core": dp_core_cycles,
            "cost_volume": cost_volume_cycles,
            "gaussian": gauss_gen_cycles,
            "ggu": ggu_cycles,
        },
        "replay_count": len(replays),
    }
    return replays, trace
