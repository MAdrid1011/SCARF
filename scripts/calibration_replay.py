"""Replay the registered mechanism grid without target images or paper values."""

from __future__ import annotations

import hashlib
import itertools
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from fsdr import FSDRSimulator
from saes import apply_progressive_saes
from scripts.calibration_contract import PARAMETER_GRID


FIDELITY_PSNR_FLOOR_DB = 40.0


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


def registered_candidates() -> list[dict[str, float]]:
    names = tuple(PARAMETER_GRID)
    return [
        {name: float(value) for name, value in zip(names, values)}
        for values in itertools.product(*(PARAMETER_GRID[name] for name in names))
    ]


def replay_calibration_sample(
    *,
    config: Any,
    gaussians: Any,
    features: torch.Tensor,
    depths: torch.Tensor,
    render: Callable[[Any], torch.Tensor],
    gaussian_type: type,
    image_height: int,
    image_width: int,
    context_extrinsics: torch.Tensor | None = None,
    context_intrinsics: torch.Tensor | None = None,
    ray_depth_mode: str = "euclidean",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not torch.is_tensor(features) or not torch.is_tensor(depths):
        raise ValueError("calibration replay requires feature and depth tensors")
    means = gaussians.means.detach()
    covariances = gaussians.covariances.detach()
    harmonics = gaussians.harmonics.detach()
    opacities = gaussians.opacities.detach()
    baseline = gaussian_type(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )
    baseline_images = render(baseline).detach()

    from lpips import LPIPS
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(
        baseline_images.device
    )
    lpips_metric = LPIPS(net="vgg").to(baseline_images.device).eval()
    view_features = F.interpolate(
        features[0],
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )
    view_count = int(view_features.shape[0])
    pixels_per_view = image_height * image_width
    primitives_per_pixel = means.shape[1] // (view_count * pixels_per_view)
    if primitives_per_pixel <= 0 or (
        primitives_per_pixel * view_count * pixels_per_view != means.shape[1]
    ):
        raise ValueError("calibration Gaussian layout does not match the context views")
    candidates = []
    for parameters in registered_candidates():
        trial = gaussian_type(
            means=means.clone(),
            covariances=covariances.clone(),
            harmonics=harmonics.clone(),
            opacities=opacities.clone(),
        )
        modified, saes, _ = apply_progressive_saes(
            trial,
            image_height,
            image_width,
            int(config.tile_size),
            feature_var_threshold=float(config.feature_var_threshold),
            depth_std_threshold=float(config.depth_std_threshold),
            features=features,
            depths=depths,
            view_count=int(features.shape[1]),
            beta_x=parameters["beta_x"],
            beta_f=parameters["beta_f"],
            beta_d=parameters["beta_d"],
            num_depth_candidates=int(config.num_depth_candidates),
            context_extrinsics=context_extrinsics,
            context_intrinsics=context_intrinsics,
            ray_depth_mode=ray_depth_mode,
        )
        fsdr = FSDRSimulator(
            feature_dim=int(view_features.shape[1]),
            cache_size=int(config.fsdr_cache_size),
            hamming_threshold=int(config.fsdr_hamming_threshold),
            num_depth_candidates=int(config.num_depth_candidates),
            seed=int(config.fsdr_seed),
            guidance_policy="paper-hamming-local-validity",
            depth_consistency_threshold=parameters["gamma_depth"],
            tile_size=int(config.tile_size),
        )
        for view in range(view_features.shape[0]):
            fsdr.begin_frame()
            frame = view_features[view].permute(1, 2, 0).reshape(
                image_height * image_width, view_features.shape[1]
            )
            if depths.dim() == 5:
                frame_depths = depths[0, view, : image_height * image_width, 0, 0]
            elif depths.dim() == 4:
                frame_depths = depths[0, view, :image_height, :image_width].reshape(-1)
            else:
                raise ValueError("unsupported calibration depth tensor")
            pixel_offset = view * pixels_per_view
            paths = fsdr.process_frame(frame, frame_depths, image_width)
            for local_index, path in enumerate(paths):
                reuse = fsdr.reuse_data.get(local_index)
                if reuse is None:
                    continue
                ratio = float(reuse["depth_ratio"])
                if math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
                    continue
                for slot in range(primitives_per_pixel):
                    global_index = (
                        (pixel_offset + local_index) * primitives_per_pixel + slot
                    )
                    if not bool(modified[global_index].item()):
                        trial.means[0, global_index] = (
                            means[0, global_index] * ratio
                        )

        approximate_images = render(trial)
        mse = F.mse_loss(approximate_images, baseline_images).clamp_min(1e-12)
        fidelity_psnr = float((-10.0 * torch.log10(mse)).item())
        ssim = float(ssim_metric(approximate_images, baseline_images).item())
        with torch.no_grad():
            lpips = float(
                lpips_metric(approximate_images, baseline_images, normalize=True)
                .reshape(-1)
                .mean()
                .item()
            )
        fsdr_summary = fsdr.get_summary()
        full_work = (
            int(fsdr_summary["full_depth_evaluations"])
            + int(saes["full_s2_evaluations"])
            + int(means.shape[1])
        )
        actual_work = (
            int(fsdr_summary["executed_depth_evaluations"])
            + int(saes["executed_s2_evaluations"])
            + int(saes["effective_gaussians"])
        )
        candidates.append(
            {
                "parameters": parameters,
                "quality": {
                    "psnr_loss_db": max(0.0, FIDELITY_PSNR_FLOOR_DB - fidelity_psnr),
                    "ssim_loss": max(0.0, 1.0 - ssim),
                    "lpips_increase": max(0.0, lpips),
                },
                "fidelity": {
                    "psnr_db": fidelity_psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                    "psnr_floor_db": FIDELITY_PSNR_FLOOR_DB,
                },
                "work_reduction": 1.0 - actual_work / max(full_work, 1),
                "compression": 1.0 - int(saes["effective_gaussians"]) / means.shape[1],
                "events": {"fsdr": fsdr_summary, "saes": saes},
            }
        )
    trace = {
        "neural_forward_passes": 1,
        "target_rgb_accessed": False,
        "feature_tensor": _tensor_record(features),
        "depth_tensor": _tensor_record(depths),
        "gaussian_means": _tensor_record(means),
        "camera_geometry": (
            {
                "extrinsics": _tensor_record(context_extrinsics),
                "intrinsics": _tensor_record(context_intrinsics),
                "ray_depth_mode": ray_depth_mode,
            }
            if context_extrinsics is not None and context_intrinsics is not None
            else None
        ),
        "candidate_count": len(candidates),
    }
    return candidates, trace
