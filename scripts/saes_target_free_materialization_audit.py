#!/usr/bin/env python3
"""Run a context-only SAES materialization audit without hardware simulation.

The full demo also constructs cycle-simulation state that is irrelevant to a
Gaussian-attribute audit.  This entrypoint instead captures the original
encoder's S1 feature tensor and S2 depth output, executes the unchanged
GaussianAdapter once, and checks that changing skipped raw descriptors cannot
change the retained SAES output.  It stops before any target-side work,
renderer, metric, or GGU execution.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes
from scripts.saes_dependency_audit import _context_on_device, remove_target_rgb


def _clone_gaussians(gaussians: Any) -> Any:
    """Clone the four source attributes without reading model internals."""
    return type(gaussians)(
        means=gaussians.means.detach().clone(),
        covariances=gaussians.covariances.detach().clone(),
        harmonics=gaussians.harmonics.detach().clone(),
        opacities=gaussians.opacities.detach().clone(),
    )


def _gaussians_on_cpu(gaussians: Any) -> Any:
    """Move captured native outputs to CPU for fixed-function audit work."""
    return type(gaussians)(
        means=gaussians.means.detach().cpu().clone(),
        covariances=gaussians.covariances.detach().cpu().clone(),
        harmonics=gaussians.harmonics.detach().cpu().clone(),
        opacities=gaussians.opacities.detach().cpu().clone(),
    )


def _capture_encoder_execution(
    model: Any, context: dict[str, Any]
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Return the native adapter output plus the actual S1/S2 tensors it consumed."""
    predictor = model.encoder.depth_predictor
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 5:
            raise RuntimeError("depth predictor did not receive [B,V,C,H,W] features")
        if not isinstance(output, tuple) or len(output) < 1 or not torch.is_tensor(output[0]):
            raise RuntimeError("depth predictor did not emit a depth tensor")
        captured["features"] = inputs[0].detach().clone()
        captured["depths"] = output[0].detach().clone()

    handle = predictor.register_forward_hook(capture)
    try:
        with torch.no_grad():
            gaussians = model.encoder(context, False, deterministic=True)
    finally:
        handle.remove()
    if set(captured) != {"features", "depths"}:
        raise RuntimeError("encoder execution did not expose one S1/S2 capture")
    return gaussians, captured["features"], captured["depths"]


def _poison_skipped_descriptors(gaussians: Any, skipped: torch.Tensor) -> None:
    """Change only descriptors marked skipped by the first target-free pass."""
    gaussians.means[0, skipped] = 1.0e4
    gaussians.covariances[0, skipped] = -1.0e4
    gaussians.harmonics[0, skipped] = 1.0e4
    gaussians.opacities[0, skipped] = 0.99


def _max_attribute_delta(reference: Any, candidate: Any, indices: torch.Tensor) -> dict[str, float]:
    return {
        name: float(
            (getattr(reference, name)[0, indices] - getattr(candidate, name)[0, indices])
            .abs()
            .max()
            .item()
        )
        if indices.numel()
        else 0.0
        for name in ("means", "covariances", "harmonics", "opacities")
    }


def _mask_sha256(mask: torch.Tensor) -> str:
    return hashlib.sha256(
        mask.detach().to(device="cpu", dtype=torch.uint8).numpy().tobytes()
    ).hexdigest()


def _routing_statistic_summary(
    features: torch.Tensor, *, height: int, width: int
) -> dict[str, dict[str, float | int]]:
    """Record fixed paper/current probe-statistic distributions without routing by them."""
    summary: dict[str, dict[str, float | int]] = {}
    for statistic in (
        "raw-probe-vector-variance",
        "raw-probe-mean-channel-variance",
        "normalized-probe-total-variance",
        "normalized-probe-vector-standard-deviation",
    ):
        values, _ = ProgressiveSAES.classify_tiles_by_features(
            features, height, width, 4, per_view=True, statistic=statistic
        )
        series = torch.tensor(list(values.values()), dtype=torch.float64)
        summary[statistic] = {
            "count": int(series.numel()),
            "rate_below_tau_f_0_2": float((series < 0.2).double().mean().item()),
            "minimum": float(series.min().item()),
            "p50": float(torch.quantile(series, 0.50).item()),
            "p95": float(torch.quantile(series, 0.95).item()),
            "maximum": float(series.max().item()),
        }
    return summary


def _l0_range_envelope_summary(
    source: Any,
    materialized: Any,
    mask: torch.Tensor,
    *,
    height: int,
    width: int,
    views: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Measure, but do not enforce, source-envelope excursions for all-L0 output."""
    if stats["level1_tiles"] or stats["full_tiles"]:
        return {"applicable": False, "reason": "audit route is not all-L0"}
    source_determinants = torch.linalg.det(source.covariances[0])
    output_determinants = torch.linalg.det(materialized.covariances[0])
    source_opacities = source.opacities[0]
    output_opacities = materialized.opacities[0]
    determinant_violations = 0
    opacity_violations = 0
    for view in range(views):
        for tile_y in range(0, height, 4):
            for tile_x in range(0, width, 4):
                anchors = torch.tensor(
                    [
                        view * height * width + (tile_y + row) * width + tile_x + column
                        for row, column in ((0, 0), (0, 3), (3, 0), (3, 3))
                    ],
                    dtype=torch.long,
                )
                if bool(mask[anchors].any()):
                    raise RuntimeError("all-L0 range audit found a skipped primary probe")
                if bool((output_determinants[anchors] > source_determinants[anchors].max()).any()):
                    determinant_violations += 1
                if bool(
                    (
                        (output_opacities[anchors] < source_opacities[anchors].min())
                        | (output_opacities[anchors] > source_opacities[anchors].max())
                    ).any()
                ):
                    opacity_violations += 1
    tile_count = views * (height // 4) * (width // 4)
    return {
        "applicable": True,
        "tile_count": tile_count,
        "determinant_above_selected_anchor_max_tiles": determinant_violations,
        "determinant_above_selected_anchor_max_rate": determinant_violations / tile_count,
        "opacity_outside_selected_anchor_range_tiles": opacity_violations,
        "opacity_outside_selected_anchor_range_rate": opacity_violations / tile_count,
    }


def collect_materialization_audit(
    *, model_name: str, sample_index: int, device: torch.device
) -> dict[str, Any]:
    """Run the predeclared target-free DL3DV sample-0 materialization gate."""
    if model_name not in {"transplat", "mvsplat"}:
        raise ValueError("the focused materialization audit currently supports transplat/mvsplat")
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    experiment = resolve_experiment(model_name, "dl3dv", ROOT)
    selection = resolve_claim_selection(model_name, "dl3dv", ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        model_name,
        dataset_name="dl3dv",
        checkpoint_path=experiment.checkpoint,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=sample_index + 1,
        sample_index=sample_index,
    )
    model.eval()
    native_target_rgb_loaded = remove_target_rgb(batch)
    context = _context_on_device(batch, loaded_device)
    source_gaussians, features, depths = _capture_encoder_execution(model, context)
    _, views, _, height, width = context["image"].shape
    # The encoder path above is the real CUDA execution.  SAES attribute
    # merging is deterministic fixed-function work, so move its captured
    # inputs to CPU before the two poison-invariance passes.  This avoids
    # allocating simulator-unrelated CUDA scratch buffers on a shared host.
    native_encoder_device = str(source_gaussians.means.device)
    source_gaussians = _gaussians_on_cpu(source_gaussians)
    features = features.detach().cpu()
    depths = depths.detach().cpu()
    audit_extrinsics = context["extrinsics"].detach().cpu()
    audit_intrinsics = context["intrinsics"].detach().cpu()
    del model, context
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    materialized = _clone_gaussians(source_gaussians)
    mask, stats, _ = apply_progressive_saes(
        materialized,
        height,
        width,
        tile_size=4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        view_count=views,
        materialization="conditional-optical-mass-diagnostic",
        context_extrinsics=audit_extrinsics,
        context_intrinsics=audit_intrinsics,
    )
    routing_statistics = _routing_statistic_summary(
        features, height=height, width=width
    )
    range_envelope = _l0_range_envelope_summary(
        source_gaussians,
        materialized,
        mask,
        height=height,
        width=width,
        views=views,
        stats=stats,
    )
    skipped = mask.nonzero(as_tuple=False).flatten()
    retained = (~mask).nonzero(as_tuple=False).flatten()
    poisoned = _clone_gaussians(source_gaussians)
    _poison_skipped_descriptors(poisoned, skipped)
    poisoned_mask, poisoned_stats, _ = apply_progressive_saes(
        poisoned,
        height,
        width,
        tile_size=4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        view_count=views,
        materialization="conditional-optical-mass-diagnostic",
        context_extrinsics=audit_extrinsics,
        context_intrinsics=audit_intrinsics,
    )
    if not torch.equal(mask, poisoned_mask):
        raise RuntimeError("poisoned descriptor audit changed the target-free route")
    retained_delta = _max_attribute_delta(materialized, poisoned, retained)
    if any(value != 0.0 for value in retained_delta.values()):
        raise RuntimeError("retained attributes depend on skipped raw descriptors")
    if stats != poisoned_stats:
        raise RuntimeError("poisoned descriptor audit changed SAES event statistics")

    return {
        "schema_version": "1.0",
        "kind": "saes_conditional_optical_mass_target_free_audit",
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "native_dataloader_loaded_target_rgb": native_target_rgb_loaded,
        "model": model_name,
        "dataset": "dl3dv",
        "sample_index": sample_index,
        "scene": str(batch["scene"][0]),
        "context_indices": [int(value) for value in batch["context"]["index"][0].tolist()],
        "execution_boundary": {
            "completed": ("S1", "S2", "S3", "S4", "SAES_attribute_audit"),
            "ggu_executed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "hardware_cycle_simulator_executed": False,
        },
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": native_target_rgb_loaded,
            "removed_before_context_device_transfer": True,
            "passed_to_model": False,
            "used_for_routing_or_metric": False,
        },
        "native_execution": {
            "native_encoder_device": native_encoder_device,
            "saes_attribute_audit_device": "cpu",
            "features_shape": list(features.shape),
            "depths_shape": list(depths.shape),
            "gaussian_count": int(source_gaussians.means.shape[1]),
        },
        "selection": {
            "skipped_descriptor_count": int(skipped.numel()),
            "retained_descriptor_count": int(retained.numel()),
            "skipped_mask_sha256": _mask_sha256(mask),
        },
        "saes_stats": stats,
        "routing_statistic_diagnostics": routing_statistics,
        "l0_range_envelope_diagnostic": range_envelope,
        "skipped_descriptor_poison_audit": {
            "poison_value": 1.0e4,
            "route_identical": True,
            "event_statistics_identical": True,
            "retained_maximum_absolute_delta": retained_delta,
            "skipped_stage3_attributes_read": False,
        },
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "source": source_identity(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("transplat", "mvsplat"), default="transplat")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_index != 0:
        parser.error("the materialization audit is predeclared for DL3DV sample index 0")
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    record = collect_materialization_audit(
        model_name=args.model, sample_index=args.sample_index, device=device
    )
    from scripts.result_record import portable_command, write_result

    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")


if __name__ == "__main__":
    main()
