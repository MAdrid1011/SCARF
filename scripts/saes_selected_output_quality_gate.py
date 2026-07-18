#!/usr/bin/env python3
"""Run one fixed DL3DV quality pilot through the real selected-output head path."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.probe_first_schedule import (
    build_conservative_probe_first_schedule,
    retained_mask_from_saes_modified,
)
from saes.progressive_saes import apply_progressive_saes
from saes.selected_output_execution import selected_output_head_execution
from scripts.saes_dependency_audit import _context_on_device
from scripts.saes_selected_output_replay_audit import (
    _classic_raw_head,
    strict_fp32_convolution_execution,
)
from scripts.saes_target_free_materialization_audit import (
    _capture_encoder_execution,
    _clone_gaussians,
)


QUALITY_LIMITS = {
    "psnr_loss_db": 0.15,
    "ssim_loss": 0.005,
    "lpips_increase": 0.005,
}
DECISION_SEMANTICS = "probe-normalized-std-first-hit"
DEPTH_ROUTING_SEMANTICS = "metric-depth-standard-deviation"
# This entrypoint is deliberately bound to the single post-audit diagnostic
# registered in PLAN.md. It must not become a free-form quality retry surface.
MATERIALIZATION = "conditional-adapter-offset-transport-diagnostic"
SEMANTIC_EQ_ATOL = 1.0e-5
SEMANTIC_EQ_RTOL = 1.0e-5


def _sha256_mask(mask: torch.Tensor) -> str:
    return hashlib.sha256(
        mask.detach().to(device="cpu", dtype=torch.uint8).numpy().tobytes()
    ).hexdigest()


def _render(model: Any, gaussians: Any, target: dict[str, Any], image_shape: tuple[int, int]) -> torch.Tensor:
    with torch.no_grad():
        output = model.decoder.forward(
            gaussians,
            target["extrinsics"],
            target["intrinsics"],
            target["near"],
            target["far"],
            image_shape,
            depth_mode=None,
        )
    return output.color[0]


def _view_metrics(images: torch.Tensor, references: torch.Tensor) -> list[dict[str, float]]:
    from lpips import LPIPS
    from torchmetrics.functional.image import structural_similarity_index_measure

    metric = LPIPS(net="vgg").to(images.device).eval()
    values = []
    with torch.no_grad():
        for image, reference in zip(images, references):
            mse = (image - reference).square().mean()
            values.append(
                {
                    "psnr_db": float((-10.0 * torch.log10(mse)).item()),
                    "ssim": float(
                        structural_similarity_index_measure(
                            image.unsqueeze(0), reference.unsqueeze(0), data_range=1.0
                        ).item()
                    ),
                    "lpips": float(
                        metric(image.unsqueeze(0), reference.unsqueeze(0), normalize=True)
                        .reshape(-1)
                        .mean()
                        .item()
                    ),
                }
            )
    return values


def _mean_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("quality gate has no target views")
    return {
        key: sum(value[key] for value in values) / len(values)
        for key in ("psnr_db", "ssim", "lpips")
    }


def _quality_verdict(
    baseline: dict[str, float], sparse: dict[str, float]
) -> dict[str, Any]:
    delta = {
        "psnr_loss_db": baseline["psnr_db"] - sparse["psnr_db"],
        "ssim_loss": baseline["ssim"] - sparse["ssim"],
        "lpips_increase": sparse["lpips"] - baseline["lpips"],
    }
    return {
        "limits": dict(QUALITY_LIMITS),
        "observed": delta,
        "pass": all(delta[key] <= QUALITY_LIMITS[key] for key in QUALITY_LIMITS),
    }


def _sparse_encoder_pass(
    model: Any,
    context: dict[str, Any],
    selection_mask: torch.Tensor,
) -> tuple[Any, dict[str, Any]]:
    head = _classic_raw_head(model, "transplat")
    with selected_output_head_execution(head, selection_mask) as trace:
        with torch.no_grad():
            gaussians = model.encoder(context, False, deterministic=True)
    return gaussians, trace.events


def _target_camera_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move only camera metadata needed by the decoder before the quality phase."""
    target = batch.get("target")
    if not isinstance(target, dict):
        raise RuntimeError("selected-output quality pilot batch has no target mapping")
    required = ("extrinsics", "intrinsics", "near", "far")
    if any(key not in target or not torch.is_tensor(target[key]) for key in required):
        raise RuntimeError("selected-output quality pilot target camera metadata is incomplete")
    return {key: target[key].to(device) for key in required}


def _take_target_rgb_for_metrics(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Remove and transfer target RGB only after sparse execution is committed."""
    target = batch.get("target")
    if not isinstance(target, dict):
        raise RuntimeError("selected-output quality pilot batch has no target mapping")
    images = target.pop("image", None)
    if not torch.is_tensor(images):
        raise RuntimeError("selected-output quality pilot requires native target RGB")
    return images.to(device)


def _active_attribute_equivalence(
    dense_reference: Any, sparse_candidate: Any, retained: torch.Tensor
) -> dict[str, Any]:
    """Check selected-head materialization against a dense-SAes control only."""
    if retained.ndim != 1 or retained.dtype != torch.bool or not bool(retained.any()):
        raise ValueError("semantic equivalence requires a nonempty retained descriptor mask")
    deltas = {}
    equivalent = True
    for name in ("means", "covariances", "harmonics", "opacities"):
        reference = getattr(dense_reference, name)[0, retained]
        candidate = getattr(sparse_candidate, name)[0, retained]
        if reference.shape != candidate.shape:
            raise RuntimeError(f"semantic equivalence shape mismatch for {name}")
        delta = (reference - candidate).abs()
        deltas[name] = {
            "maximum_absolute_delta": float(delta.max().item()),
            "mean_absolute_delta": float(delta.mean().item()),
        }
        equivalent = equivalent and bool(
            torch.allclose(reference, candidate, rtol=SEMANTIC_EQ_RTOL, atol=SEMANTIC_EQ_ATOL)
        )
    return {
        "retained_descriptor_count": int(retained.sum().item()),
        "atol": SEMANTIC_EQ_ATOL,
        "rtol": SEMANTIC_EQ_RTOL,
        "per_attribute": deltas,
        "equivalent": equivalent,
    }


def _image_equivalence(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Record a target-free decoder equivalence check for the fixed sparse output."""
    if reference.shape != candidate.shape:
        raise RuntimeError("semantic render equivalence image shapes differ")
    delta = (reference - candidate).abs()
    return {
        "atol": SEMANTIC_EQ_ATOL,
        "rtol": SEMANTIC_EQ_RTOL,
        "maximum_absolute_delta": float(delta.max().item()),
        "mean_absolute_delta": float(delta.mean().item()),
        "equivalent": bool(
            torch.allclose(reference, candidate, rtol=SEMANTIC_EQ_RTOL, atol=SEMANTIC_EQ_ATOL)
        ),
    }


def _retain_renderable_gaussians(gaussians: Any, retained: torch.Tensor) -> Any:
    """Drop SAES-removed zero-opacity descriptors before native S4 rendering."""
    if (
        retained.ndim != 1
        or retained.dtype != torch.bool
        or retained.numel() != gaussians.means.shape[1]
        or not bool(retained.any())
    ):
        raise ValueError("renderable Gaussian selection must match a nonempty flattened set")
    return type(gaussians)(
        means=gaussians.means[:, retained].contiguous(),
        covariances=gaussians.covariances[:, retained].contiguous(),
        harmonics=gaussians.harmonics[:, retained].contiguous(),
        opacities=gaussians.opacities[:, retained].contiguous(),
    )


def _apply_fixed_saes(
    gaussians: Any,
    *,
    features: torch.Tensor,
    depths: torch.Tensor,
    context: dict[str, Any],
    height: int,
    width: int,
    views: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    modified, stats, _ = apply_progressive_saes(
        gaussians,
        height,
        width,
        tile_size=4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        view_count=views,
        materialization=MATERIALIZATION,
        decision_semantics=DECISION_SEMANTICS,
        depth_routing_semantics=DEPTH_ROUTING_SEMANTICS,
        context_extrinsics=context["extrinsics"],
        context_intrinsics=context["intrinsics"],
    )
    return modified, stats


def collect_quality_pilot(*, device: torch.device) -> dict[str, Any]:
    """Execute the pre-registered sample-0 selected-output quality pilot."""
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    experiment = resolve_experiment("transplat", "dl3dv", ROOT)
    selection = resolve_claim_selection("transplat", "dl3dv", ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        "transplat",
        dataset_name="dl3dv",
        checkpoint_path=experiment.checkpoint,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=1,
        sample_index=0,
    )
    model.eval()
    native_target_rgb_loaded = "image" in batch["target"]
    target = _target_camera_inputs(batch, loaded_device)
    context = _context_on_device(batch, loaded_device)
    _, views, _, height, width = context["image"].shape
    if context["image"].shape[0] != 1:
        raise RuntimeError("selected-output quality pilot requires batch size one")

    # The following route construction occurs before target images are read for
    # rendering or quality. It consumes only the actual S1/S2 tensors captured
    # from the original dense encoder execution.
    with strict_fp32_convolution_execution() as numerical_execution:
        baseline_gaussians, features, depths = _capture_encoder_execution(model, context)
        primitives_per_pixel = baseline_gaussians.means.shape[1] // (views * height * width)
        if primitives_per_pixel != 1:
            raise RuntimeError("selected-output quality pilot requires one Gaussian per pixel")
        schedule = build_conservative_probe_first_schedule(
            features,
            depths,
            height=height,
            width=width,
            tile_size=4,
            feature_threshold=0.2,
            depth_threshold=0.1,
            decision_semantics=DECISION_SEMANTICS,
        )
        dense_saes_gaussians = _clone_gaussians(baseline_gaussians)
        dense_modified, dense_stats = _apply_fixed_saes(
            dense_saes_gaussians,
            features=features,
            depths=depths,
            context=context,
            height=height,
            width=width,
            views=views,
        )
        provisional_gaussians, provisional_head_events = _sparse_encoder_pass(
            model, context, schedule.selection_mask
        )
        provisional_modified, provisional_stats = _apply_fixed_saes(
            provisional_gaussians,
            features=features,
            depths=depths,
            context=context,
            height=height,
            width=width,
            views=views,
        )
        final_selection = retained_mask_from_saes_modified(
            provisional_modified, views=views, height=height, width=width
        )
        sparse_gaussians, final_head_events = _sparse_encoder_pass(
            model, context, final_selection
        )
        final_modified, final_stats = _apply_fixed_saes(
            sparse_gaussians,
            features=features,
            depths=depths,
            context=context,
            height=height,
            width=width,
            views=views,
        )
    if not torch.equal(provisional_modified, final_modified):
        raise RuntimeError("guard-resolved SAES mask changed between selected-output passes")
    if not torch.equal(final_selection, retained_mask_from_saes_modified(
        final_modified, views=views, height=height, width=width
    )):
        raise RuntimeError("final selected-output mask does not cover every retained descriptor")
    if not torch.equal(dense_modified, final_modified):
        raise RuntimeError("selected-head execution changed the dense SAES routing mask")

    retained = ~final_modified
    attribute_equivalence = _active_attribute_equivalence(
        dense_saes_gaussians, sparse_gaussians, retained
    )
    if not attribute_equivalence["equivalent"]:
        raise RuntimeError("selected-head execution changed retained SAES attributes")
    dense_saes_renderable = _retain_renderable_gaussians(dense_saes_gaussians, retained)
    sparse_renderable = _retain_renderable_gaussians(sparse_gaussians, retained)
    dense_saes_images = _render(model, dense_saes_renderable, target, (height, width))
    dense_saes_repeat_images = _render(
        model, dense_saes_renderable, target, (height, width)
    )
    sparse_images = _render(model, sparse_renderable, target, (height, width))
    dense_repeat_equivalence = _image_equivalence(
        dense_saes_images, dense_saes_repeat_images
    )
    render_equivalence = _image_equivalence(dense_saes_images, sparse_images)

    # The output path is now fixed. Only this final section may read target RGB.
    target_images = _take_target_rgb_for_metrics(batch, loaded_device)
    baseline_images = _render(model, baseline_gaussians, target, (height, width))
    baseline_views = _view_metrics(baseline_images, target_images[0])
    sparse_views = _view_metrics(sparse_images, target_images[0])
    baseline_quality = _mean_metrics(baseline_views)
    sparse_quality = _mean_metrics(sparse_views)
    verdict = _quality_verdict(baseline_quality, sparse_quality)

    quality_views = [
        {
            "target_index": int(batch["target"]["index"][0, index].item()),
            "baseline": baseline_views[index],
            "sparse": sparse_views[index],
        }
        for index in range(len(baseline_views))
    ]
    return {
        "schema_version": "1.0",
        "kind": "saes_selected_output_quality_pilot",
        "paper_result_eligible": False,
        "model": "transplat",
        "dataset": "dl3dv",
        "sample_index": 0,
        "scene": str(batch["scene"][0]),
        "context_indices": [int(value) for value in batch["context"]["index"][0].tolist()],
        "target_indices": [int(value) for value in batch["target"]["index"][0].tolist()],
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": native_target_rgb_loaded,
            "not_accessed_or_transferred_before_sparse_mask_committed": True,
            "passed_to_encoder_or_route": False,
            "removed_from_native_batch_for_metrics": True,
            "first_read_after_sparse_mask_committed": True,
        },
        "execution_boundary": {
            "dense_control_s1_s2_executed": True,
            "provisional_selected_head_executed": True,
            "final_selected_head_executed": True,
            "native_gaussian_adapter_executed": True,
            "native_decoder_executed": True,
            "quality_metrics_computed": True,
            "s2_s3_sparse_execution_verified": False,
            "reason_global_s2_s3_unverified": "shared S1/S2/refinement trunk remains dense",
        },
        "routing": {
            "feature_threshold": 0.2,
            "depth_threshold": 0.1,
            "decision_semantics": DECISION_SEMANTICS,
            "depth_routing_semantics": DEPTH_ROUTING_SEMANTICS,
            "materialization": MATERIALIZATION,
            "schedule": schedule.events,
            "provisional_mask_sha256": _sha256_mask(schedule.selection_mask),
            "final_mask_sha256": _sha256_mask(final_selection),
            "dense_control_mask_sha256": _sha256_mask(dense_modified),
            "provisional_saes_stats": provisional_stats,
            "final_saes_stats": final_stats,
            "dense_control_saes_stats": dense_stats,
        },
        "head_execution": {
            "provisional_guard_pass": provisional_head_events,
            "final_output_pass": final_head_events,
            "actual_head_macs": (
                provisional_head_events["actual_head_macs"]
                + final_head_events["actual_head_macs"]
            ),
            "dense_reference_head_macs": final_head_events["dense_head_macs"],
            "signed_head_mac_delta": (
                final_head_events["dense_head_macs"]
                - provisional_head_events["actual_head_macs"]
                - final_head_events["actual_head_macs"]
            ),
            "upstream_s2_saving": 0.0,
        },
        "selected_head_semantic_equivalence": {
            "route_mask_equivalent": True,
            "retained_attributes": attribute_equivalence,
            "decoder_render": render_equivalence,
            "dense_repeat_render": dense_repeat_equivalence,
            "strict_decoder_fp32_equivalent": render_equivalence["equivalent"],
            "quality_diagnostic_is_nonclaiming_if_strict_decoder_check_fails": (
                not render_equivalence["equivalent"]
            ),
            "removed_zero_opacity_descriptors_before_decoder": int(
                final_modified.sum().item()
            ),
            "target_rgb_accessed": False,
        },
        "quality": {
            "baseline": baseline_quality,
            "sparse": sparse_quality,
            "views": quality_views,
            "verdict": verdict,
        },
        "numerical_execution": numerical_execution,
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "source": source_identity(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this fixed quality pilot requires an available CUDA device")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        record = collect_quality_pilot(device=device)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    record["command"] = [
        "python",
        str(Path(__file__).relative_to(ROOT)),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / "results.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
