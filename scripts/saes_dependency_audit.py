#!/usr/bin/env python3
"""Audit whether dense non-probe S2/S3 activations affect retained probes."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _RawHeadCaptured(RuntimeError):
    """Internal control flow used to terminate before S4 or rendering."""


def build_probe_mask(*, height: int, width: int, tile_size: int) -> torch.Tensor:
    """Return the corner-probe mask for a tiled full-resolution activation map."""
    if min(height, width, tile_size) <= 0:
        raise ValueError("height, width, and tile_size must be positive")
    if height % tile_size or width % tile_size:
        raise ValueError("activation dimensions must be divisible by tile_size")
    mask = torch.zeros((height, width), dtype=torch.bool)
    for tile_y in range(0, height, tile_size):
        for tile_x in range(0, width, tile_size):
            mask[tile_y, tile_x] = True
            mask[tile_y, tile_x + tile_size - 1] = True
            mask[tile_y + tile_size - 1, tile_x] = True
            mask[tile_y + tile_size - 1, tile_x + tile_size - 1] = True
    return mask


def summarize_probe_dependency(
    baseline: torch.Tensor, perturbed: torch.Tensor, probe_mask: torch.Tensor
) -> dict[str, int | float | bool]:
    """Summarize raw-head differences at retained probes only."""
    if (
        baseline.ndim != 4
        or baseline.shape != perturbed.shape
        or probe_mask.shape != baseline.shape[-2:]
        or probe_mask.dtype != torch.bool
    ):
        raise ValueError("raw-head tensors and probe mask are incompatible")
    delta = (perturbed - baseline).abs()[..., probe_mask.to(baseline.device)]
    reference = baseline[..., probe_mask.to(baseline.device)]
    changed = delta > 0
    reference_norm = reference.norm().clamp_min(1e-8)
    return {
        "probe_value_count": int(delta.numel()),
        "probe_changed_value_count": int(changed.sum().item()),
        "maximum_absolute_delta": float(delta.max().item()),
        "mean_absolute_delta": float(delta.mean().item()),
        "relative_l2_delta": float(delta.norm().div(reference_norm).item()),
        "dependency_detected": bool(changed.any().item()),
        "nonprobe_delta_not_reported": True,
    }


def _context_on_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch["context"].items()
    }


def remove_target_rgb(batch: dict[str, Any]) -> bool:
    """Remove native target RGB before a context-only dependency execution."""
    target = batch.get("target")
    if not isinstance(target, dict):
        raise RuntimeError("dependency audit batch has no target mapping")
    loaded = torch.is_tensor(target.get("image"))
    target.pop("image", None)
    return loaded


def _raw_head_modules(
    model: Any, model_name: str
) -> tuple[Any, Any, str]:
    """Return the native raw head and its full-resolution dense input stage."""
    if model_name in {"transplat", "mvsplat"}:
        predictor = model.encoder.depth_predictor
        raw_head = getattr(predictor, "to_gaussians", None)
        perturb_module = getattr(predictor, "refine_unet", None)
        if raw_head is None or perturb_module is None:
            raise RuntimeError(
                f"{model_name} predictor lacks refine_unet or to_gaussians"
            )
        module_name = (
            "DepthPredictorTrans.refine_unet"
            if model_name == "transplat"
            else "DepthPredictorMultiView.refine_unet"
        )
        return raw_head, perturb_module, module_name
    if model_name == "depthsplat":
        encoder = model.encoder
        raw_head = getattr(encoder, "gaussian_head", None)
        perturb_module = getattr(encoder, "gaussian_regressor", None)
        if raw_head is None or perturb_module is None:
            raise RuntimeError(
                "DepthSplat encoder lacks gaussian_regressor or gaussian_head"
            )
        return raw_head, perturb_module, "EncoderDepthSplat.gaussian_regressor"
    raise ValueError(f"unsupported dependency-audit model: {model_name}")


def _capture_raw_head(
    model: Any,
    context: dict[str, Any],
    probe_mask: torch.Tensor,
    *,
    model_name: str,
    perturb_nonprobes: bool,
    perturb_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Capture raw head output with an optional fixed refinement perturbation."""
    raw_head, perturb_module, _ = _raw_head_modules(model, model_name)
    captured: list[torch.Tensor] = []

    def capture_and_stop(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(output) or output.ndim != 4:
            raise RuntimeError("raw Gaussian head did not emit [VB,C,H,W]")
        captured.append(output.detach().clone())
        raise _RawHeadCaptured()

    def perturb_dense_input(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 4:
            raise RuntimeError("dense raw-head input is not one [VB,C,H,W] tensor")
        activation = inputs[0]
        if activation.shape[-2:] != probe_mask.shape:
            raise RuntimeError(
                "dense raw-head input does not align with configured full-resolution tiles"
            )
        if perturb_mask is None:
            keep = probe_mask.to(device=activation.device)
        else:
            if (
                perturb_mask.shape != probe_mask.shape
                or perturb_mask.dtype != torch.bool
                or bool((perturb_mask & probe_mask).any().item())
            ):
                raise RuntimeError(
                    "refinement perturbation must select only non-probe positions"
                )
            keep = ~perturb_mask.to(device=activation.device)
        keep = keep.view(1, 1, *probe_mask.shape)
        return (torch.where(keep, activation, torch.zeros_like(activation)),)

    capture_handle = raw_head.register_forward_hook(capture_and_stop)
    perturb_handle = (
        perturb_module.register_forward_pre_hook(perturb_dense_input)
        if perturb_nonprobes
        else None
    )
    try:
        with torch.no_grad():
            model.encoder(context, False, deterministic=True)
    except _RawHeadCaptured:
        pass
    finally:
        capture_handle.remove()
        if perturb_handle is not None:
            perturb_handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"expected exactly one raw Gaussian-head capture, observed {len(captured)}"
        )
    return captured[0]


def _capture_transplat_raw_head(
    model: Any,
    context: dict[str, Any],
    probe_mask: torch.Tensor,
    *,
    perturb_nonprobes: bool,
    perturb_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility boundary for the fixed TranSplat locality audit."""
    return _capture_raw_head(
        model,
        context,
        probe_mask,
        model_name="transplat",
        perturb_nonprobes=perturb_nonprobes,
        perturb_mask=perturb_mask,
    )


def collect_dependency_audit(
    *, model_name: str, sample_index: int, device: torch.device
) -> dict[str, Any]:
    """Run one fixed context-only DL3DV dependency diagnostic."""
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
    _, _, _, height, width = context["image"].shape
    probe_mask = build_probe_mask(height=height, width=width, tile_size=4)
    raw_head, perturb_module, perturb_module_name = _raw_head_modules(
        model, model_name
    )
    del raw_head, perturb_module
    baseline = _capture_raw_head(
        model,
        context,
        probe_mask,
        model_name=model_name,
        perturb_nonprobes=False,
    )
    perturbed = _capture_raw_head(
        model,
        context,
        probe_mask,
        model_name=model_name,
        perturb_nonprobes=True,
    )
    if baseline.shape != perturbed.shape:
        raise RuntimeError("baseline and perturbed raw head have different shapes")
    return {
        "schema_version": "1.0",
        "kind": "saes_s2s3_dense_dependency_audit",
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "native_dataloader_loaded_target_rgb": native_target_rgb_loaded,
        "model": model_name,
        "dataset": "dl3dv",
        "sample_index": sample_index,
        "scene": str(batch["scene"][0]),
        "context_indices": [
            int(value) for value in batch["context"]["index"][0].tolist()
        ],
        "probe_pattern": "four-corners-per-4x4-full-resolution-tile",
        "perturbation": {
            "module": perturb_module_name,
            "input": "full-resolution [VB,C,H,W] refinement activation",
            "nonprobe_action": "set to zero",
            "probe_action": "preserve exactly",
            "routing_signal_used": False,
        },
        "execution_boundary": {
            "completed": ("S1", "S2", "S3_raw_gaussian_head"),
            "ggu_s4_executed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "raw_head_capture_terminates_execution": True,
        },
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": native_target_rgb_loaded,
            "removed_before_context_device_transfer": True,
            "passed_to_model": False,
            "used_for_routing_or_metric": False,
        },
        "raw_head_shape": list(baseline.shape),
        "probe_dependency": summarize_probe_dependency(
            baseline, perturbed, probe_mask
        ),
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "source": source_identity(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("transplat", "mvsplat", "depthsplat"), default="transplat"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
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

    record = collect_dependency_audit(
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
