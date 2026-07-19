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
from typing import TYPE_CHECKING, Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    import torch


# This module is also imported by command-surface checks in an environment
# without Torch.  Keep its parser and fixed audit contract available there, and
# load the tensor/model stack only when a runtime helper is actually used.
torch: Any | None = None
ProgressiveSAES: Any | None = None
apply_progressive_saes: Any | None = None
_context_on_device: Any | None = None
remove_target_rgb: Any | None = None
_runtime_dependencies_loaded = False


def _load_torch() -> Any:
    """Load Torch only for a tensor-backed audit operation."""

    global torch

    if torch is not None:
        return torch
    try:
        import torch as torch_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAES target-free materialization audit requires a Torch installation"
        ) from exc
    torch = torch_module
    return torch_module


def _load_runtime_dependencies() -> None:
    """Load SAES and context helpers required by a real materialization audit."""

    global ProgressiveSAES
    global _context_on_device
    global _runtime_dependencies_loaded
    global apply_progressive_saes
    global remove_target_rgb

    if _runtime_dependencies_loaded:
        return
    _load_torch()
    from saes.progressive_saes import (
        ProgressiveSAES as progressive_saes,
        apply_progressive_saes as progressive_saes_apply,
    )
    from scripts.saes_dependency_audit import (
        _context_on_device as context_on_device,
        remove_target_rgb as remove_target,
    )

    ProgressiveSAES = progressive_saes
    apply_progressive_saes = progressive_saes_apply
    _context_on_device = context_on_device
    remove_target_rgb = remove_target
    _runtime_dependencies_loaded = True


MATERIALIZATION_CHOICES = (
    "conditional-anchor-transport-diagnostic",
    "conditional-adapter-offset-transport-diagnostic",
    "conditional-optical-mass-diagnostic",
    "conditional-projected-optical-mass-diagnostic",
)

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
    torch_module = _load_torch()
    predictor = model.encoder.depth_predictor
    captured: dict[str, Any] = {}

    def capture(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if (
            not inputs
            or not torch_module.is_tensor(inputs[0])
            or inputs[0].ndim != 5
        ):
            raise RuntimeError("depth predictor did not receive [B,V,C,H,W] features")
        if (
            not isinstance(output, tuple)
            or len(output) < 1
            or not torch_module.is_tensor(output[0])
        ):
            raise RuntimeError("depth predictor did not emit a depth tensor")
        captured["features"] = inputs[0].detach().clone()
        captured["depths"] = output[0].detach().clone()

    handle = predictor.register_forward_hook(capture)
    try:
        with torch_module.no_grad():
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
    torch_module = _load_torch()
    return hashlib.sha256(
        mask.detach().to(device="cpu", dtype=torch_module.uint8).numpy().tobytes()
    ).hexdigest()


def _routing_statistic_summary(
    features: torch.Tensor, *, height: int, width: int
) -> dict[str, dict[str, float | int]]:
    """Record fixed paper/current probe-statistic distributions without routing by them."""
    _load_runtime_dependencies()
    torch_module = _load_torch()
    if ProgressiveSAES is None:
        raise RuntimeError("SAES runtime dependency failed to load")
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
        series = torch_module.tensor(list(values.values()), dtype=torch_module.float64)
        summary[statistic] = {
            "count": int(series.numel()),
            "rate_below_tau_f_0_2": float((series < 0.2).double().mean().item()),
            "minimum": float(series.min().item()),
            "p50": float(torch_module.quantile(series, 0.50).item()),
            "p95": float(torch_module.quantile(series, 0.95).item()),
            "maximum": float(series.max().item()),
        }
    return summary


def _pre_fallback_routing_summary(
    features: torch.Tensor,
    depths: torch.Tensor,
    *,
    near: torch.Tensor,
    far: torch.Tensor,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Compare fixed-unit L1 interpretations before materialization fallback.

    This is an observational routing ledger, not a calibration sweep. Every
    entry uses the paper's fixed 0.20/0.10 thresholds and the same normalized
    probe-vector L0 statistic. It does not inspect raw skipped descriptors or
    run a renderer.
    """
    _load_runtime_dependencies()
    if ProgressiveSAES is None:
        raise RuntimeError("SAES runtime dependency failed to load")
    feature_scores, _ = ProgressiveSAES.classify_tiles_by_features(
        features,
        height,
        width,
        4,
        per_view=True,
        statistic="normalized-probe-vector-standard-deviation",
    )
    candidate_depths = ProgressiveSAES.inverse_depth_candidate_coordinate(
        depths, near=near, far=far
    )
    coordinates = {
        "metric-depth-standard-deviation": (depths, False),
        "relative-metric-depth-standard-deviation": (depths, True),
        "inverse-depth-candidate-coordinate-standard-deviation": (
            candidate_depths,
            False,
        ),
    }
    result: dict[str, Any] = {}
    views = int(features.shape[1])
    for name, (routing_depths, relative) in coordinates.items():
        scorer = ProgressiveSAES(
            height,
            width,
            initial_tile_size=4,
            feature_var_threshold=0.2,
            depth_std_threshold=0.1,
            view_count=views,
            decision_semantics="probe-normalized-std-first-hit",
        )
        counts = {"L0": 0, "L1": 0, "Full": 0}
        for view in range(views):
            for tile_y in range(height // 4):
                for tile_x in range(width // 4):
                    if feature_scores[(view, tile_y, tile_x)] < 0.2:
                        counts["L0"] += 1
                    elif scorer.check_depth_uniformity(
                        routing_depths,
                        tile_y,
                        tile_x,
                        4,
                        height,
                        width,
                        0.1,
                        probe_positions=scorer.probe_positions,
                        view_index=view,
                        relative=relative,
                    ):
                        counts["L1"] += 1
                    else:
                        counts["Full"] += 1
        total = sum(counts.values())
        result[name] = {
            "diagnostic_only": True,
            "tile_count": total,
            "l0_count": counts["L0"],
            "l1_count": counts["L1"],
            "full_count": counts["Full"],
            "l0_rate": counts["L0"] / total,
            "l1_rate": counts["L1"] / total,
            "full_rate": counts["Full"] / total,
        }
    return result


def _finite_summary(values: torch.Tensor) -> dict[str, float | int]:
    """Summarize a nonempty finite tensor without exposing individual values."""
    torch_module = _load_torch()
    flattened = values.detach().reshape(-1).double()
    if flattened.numel() == 0 or not bool(torch_module.isfinite(flattened).all()):
        raise ValueError("attribute diagnostic requires nonempty finite values")
    return {
        "count": int(flattened.numel()),
        "minimum": float(flattened.min().item()),
        "p50": float(torch_module.quantile(flattened, 0.50).item()),
        "p95": float(torch_module.quantile(flattened, 0.95).item()),
        "maximum": float(flattened.max().item()),
        "mean": float(flattened.mean().item()),
    }


def _retained_anchor_attribute_diagnostic(
    source: Any, materialized: Any, retained: torch.Tensor
) -> dict[str, Any]:
    """Profile only changed retained anchors after target-free materialization."""
    torch_module = _load_torch()
    attributes = ("means", "covariances", "harmonics", "opacities")
    retained = retained.to(source.means.device)
    changed = torch_module.zeros(
        retained.numel(), dtype=torch_module.bool, device=retained.device
    )
    for name in attributes:
        delta = (
            getattr(materialized, name)[0, retained]
            - getattr(source, name)[0, retained]
        )
        changed |= delta.reshape(delta.shape[0], -1).abs().amax(dim=1) > 0.0
    changed_indices = retained[changed]
    if changed_indices.numel() == 0:
        return {
            "changed_retained_anchor_count": 0,
            "source_or_skipped_descriptor_access": False,
        }

    source_means = source.means[0, changed_indices]
    output_means = materialized.means[0, changed_indices]
    source_covariances = source.covariances[0, changed_indices]
    output_covariances = materialized.covariances[0, changed_indices]
    source_alpha = source.opacities[0, changed_indices].reshape(-1)
    output_alpha = materialized.opacities[0, changed_indices].reshape(-1)
    source_harmonics = source.harmonics[0, changed_indices]
    output_harmonics = materialized.harmonics[0, changed_indices]
    eps = torch_module.finfo(source_means.dtype).eps
    determinant_floor = torch_module.finfo(source_means.dtype).tiny
    source_covariances_symmetric = (
        source_covariances + source_covariances.mT
    ) * 0.5
    output_covariances_symmetric = (
        output_covariances + output_covariances.mT
    ) * 0.5
    source_det = torch_module.linalg.det(source_covariances_symmetric)
    output_det = torch_module.linalg.det(output_covariances_symmetric)
    return {
        "changed_retained_anchor_count": int(changed_indices.numel()),
        "source_or_skipped_descriptor_access": False,
        "mean_displacement_l2": _finite_summary(
            (output_means - source_means).norm(dim=1)
        ),
        "covariance_determinant_ratio": _finite_summary(
            output_det / source_det.clamp_min(determinant_floor)
        ),
        "source_covariance_min_eigenvalue": _finite_summary(
            torch_module.linalg.eigvalsh(source_covariances_symmetric)[:, 0]
        ),
        "output_covariance_min_eigenvalue": _finite_summary(
            torch_module.linalg.eigvalsh(output_covariances_symmetric)[:, 0]
        ),
        "covariance_increment_min_eigenvalue": _finite_summary(
            torch_module.linalg.eigvalsh(
                output_covariances_symmetric - source_covariances_symmetric
            )[:, 0]
        ),
        "source_covariance_asymmetry_frobenius": _finite_summary(
            (source_covariances - source_covariances.mT)
            .reshape(changed_indices.numel(), -1)
            .norm(dim=1)
        ),
        "source_opacity": _finite_summary(source_alpha),
        "output_opacity": _finite_summary(output_alpha),
        "opacity_ratio": _finite_summary(output_alpha / source_alpha.clamp_min(eps)),
        "harmonic_relative_change": _finite_summary(
            (output_harmonics - source_harmonics)
            .reshape(changed_indices.numel(), -1)
            .norm(dim=1)
            / source_harmonics.reshape(changed_indices.numel(), -1)
            .norm(dim=1)
            .clamp_min(eps)
        ),
    }


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
    torch_module = _load_torch()
    if stats["level1_tiles"] or stats["full_tiles"]:
        return {"applicable": False, "reason": "audit route is not all-L0"}
    source_determinants = torch_module.linalg.det(source.covariances[0])
    output_determinants = torch_module.linalg.det(materialized.covariances[0])
    source_opacities = source.opacities[0]
    output_opacities = materialized.opacities[0]
    determinant_violations = 0
    opacity_violations = 0
    for view in range(views):
        for tile_y in range(0, height, 4):
            for tile_x in range(0, width, 4):
                anchors = torch_module.tensor(
                    [
                        view * height * width + (tile_y + row) * width + tile_x + column
                        for row, column in ((0, 0), (0, 3), (3, 0), (3, 3))
                    ],
                    dtype=torch_module.long,
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
    *,
    model_name: str,
    sample_index: int,
    device: torch.device,
    decision_semantics: str = "current",
    depth_routing_semantics: str = "metric-depth-standard-deviation",
    materialization: str = "conditional-optical-mass-diagnostic",
) -> dict[str, Any]:
    """Run the predeclared target-free DL3DV sample-0 materialization gate."""
    if model_name not in {"transplat", "mvsplat"}:
        raise ValueError("the focused materialization audit currently supports transplat/mvsplat")
    _load_runtime_dependencies()
    torch_module = _load_torch()
    if (
        apply_progressive_saes is None
        or _context_on_device is None
        or remove_target_rgb is None
    ):
        raise RuntimeError("SAES materialization audit runtime dependency failed to load")
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
    audit_near = context["near"].detach().cpu()
    audit_far = context["far"].detach().cpu()
    del model, context
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()

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
        materialization=materialization,
        decision_semantics=decision_semantics,
        context_extrinsics=audit_extrinsics,
        context_intrinsics=audit_intrinsics,
        depth_routing_semantics=depth_routing_semantics,
        depth_near=audit_near,
        depth_far=audit_far,
    )
    routing_statistics = _routing_statistic_summary(
        features, height=height, width=width
    )
    pre_fallback_routing = _pre_fallback_routing_summary(
        features,
        depths,
        near=audit_near,
        far=audit_far,
        height=height,
        width=width,
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
    retained_anchor_attributes = _retained_anchor_attribute_diagnostic(
        source_gaussians, materialized, retained
    )
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
        materialization=materialization,
        decision_semantics=decision_semantics,
        context_extrinsics=audit_extrinsics,
        context_intrinsics=audit_intrinsics,
        depth_routing_semantics=depth_routing_semantics,
        depth_near=audit_near,
        depth_far=audit_far,
    )
    if not torch_module.equal(mask, poisoned_mask):
        raise RuntimeError("poisoned descriptor audit changed the target-free route")
    retained_delta = _max_attribute_delta(materialized, poisoned, retained)
    if any(value != 0.0 for value in retained_delta.values()):
        raise RuntimeError("retained attributes depend on skipped raw descriptors")
    if stats != poisoned_stats:
        raise RuntimeError("poisoned descriptor audit changed SAES event statistics")

    return {
        "schema_version": "1.0",
        "kind": "saes_target_free_materialization_audit",
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
        "routing_contract": {
            "materialization": materialization,
            "feature_decision_semantics": decision_semantics,
            "depth_routing_semantics": depth_routing_semantics,
            "feature_threshold": 0.2,
            "depth_threshold": 0.1,
            "global_thresholds_unchanged": True,
            "nonzero_sparse_work": bool(stats["zeroed_gaussians"] > 0),
        },
        "saes_stats": stats,
        "routing_statistic_diagnostics": routing_statistics,
        "pre_fallback_routing_diagnostic": pre_fallback_routing,
        "l0_range_envelope_diagnostic": range_envelope,
        "retained_anchor_attribute_diagnostic": retained_anchor_attributes,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("transplat", "mvsplat"), default="transplat")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--decision-semantics",
        choices=("current", "probe-normalized-std-first-hit"),
        default="current",
    )
    parser.add_argument(
        "--depth-routing-semantics",
        choices=(
            "metric-depth-standard-deviation",
            "inverse-depth-candidate-coordinate-standard-deviation",
        ),
        default="metric-depth-standard-deviation",
    )
    parser.add_argument(
        "--materialization",
        choices=MATERIALIZATION_CHOICES,
        default="conditional-optical-mass-diagnostic",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_index != 0:
        parser.error("the materialization audit is predeclared for DL3DV sample index 0")
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    try:
        _load_runtime_dependencies()
    except RuntimeError as exc:
        parser.error(str(exc))
    torch_module = _load_torch()
    device = torch_module.device(args.device)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch_module.manual_seed(args.seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(args.seed)

    record = collect_materialization_audit(
        model_name=args.model,
        sample_index=args.sample_index,
        device=device,
        decision_semantics=args.decision_semantics,
        depth_routing_semantics=args.depth_routing_semantics,
        materialization=args.materialization,
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
