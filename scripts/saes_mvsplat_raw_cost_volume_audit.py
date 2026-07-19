#!/usr/bin/env python3
"""Audit MVSplat's selected native raw cost-volume primitive.

The audit captures the native pre-refinement correlation input from one
target-free context pass, recomputes it through primary -> secondary -> Full
selected queries, and compares the result to the captured native raw volume.
It is diagnostic-only: the current refinement and Gaussian paths stay dense.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _RawCostVolumeCaptured(RuntimeError):
    """Terminate the native encoder immediately after raw-CV construction."""


def build_fixed_native_cv_phase_masks(
    *, batch: int, height: int, width: int, tile_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a fixed primary/L1-anchor/Full trace on the native CV grid.

    This is intentionally a coverage trace, not an SAES route.  It makes the
    primary and secondary requests explicit, then asks Full for the complete
    native grid so the producer can prove exact raw-CV recovery without an
    unvalidated mapping from the 256x256 Gaussian grid to the CV grid.
    """

    from saes.probe_layout import compute_lightweight_positions, compute_probe_positions

    if min(batch, height, width, tile_size) <= 0 or height % tile_size or width % tile_size:
        raise ValueError("native cost-volume dimensions must be tiled exactly")
    primary_positions = compute_probe_positions(tile_size)
    lightweight_positions = compute_lightweight_positions(tile_size)
    primary_set = set(primary_positions)
    if len(primary_set) != len(primary_positions) or lightweight_positions[: len(primary_positions)] != primary_positions:
        raise RuntimeError("declared L1 anchors do not retain the primary prefix")
    secondary_positions = [
        position for position in lightweight_positions if position not in primary_set
    ]
    primary = torch.zeros((batch, height, width), dtype=torch.bool, device=device)
    secondary = torch.zeros_like(primary)
    for origin_y in range(0, height, tile_size):
        for origin_x in range(0, width, tile_size):
            for local_y, local_x in primary_positions:
                primary[:, origin_y + local_y, origin_x + local_x] = True
            for local_y, local_x in secondary_positions:
                secondary[:, origin_y + local_y, origin_x + local_x] = True
    if bool((primary & secondary).any()):
        raise RuntimeError("native CV primary and secondary masks overlap")
    return primary, secondary, torch.ones_like(primary)


def _capture_native_raw_cost_volume(
    model: Any, context: dict[str, Any]
) -> dict[str, Any]:
    """Capture the exact native raw-CV input and every preceding warp binding."""

    predictor = getattr(model.encoder, "depth_predictor", None)
    if predictor is None:
        raise RuntimeError("MVSplat encoder has no depth predictor")
    refinement = getattr(predictor, "corr_refine_net", None)
    if refinement is None:
        raise RuntimeError("MVSplat predictor has no native corr_refine_net")
    module = importlib.import_module(type(predictor).__module__)
    original_warp = getattr(module, "warp_with_pose_depth_candidates", None)
    if not callable(original_warp):
        raise RuntimeError("MVSplat native warp helper is unavailable")
    captured: dict[str, Any] = {"warps": []}

    def capture_warp(
        source_features: torch.Tensor,
        feature_pixel_intrinsics: torch.Tensor,
        relative_reference_to_source_pose: torch.Tensor,
        depth: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if (
            source_features.ndim != 4
            or feature_pixel_intrinsics.shape != (source_features.shape[0], 3, 3)
            or relative_reference_to_source_pose.shape != (source_features.shape[0], 4, 4)
            or depth.ndim != 4
            or depth.shape[0] != source_features.shape[0]
            or depth.shape[2:] != source_features.shape[2:]
        ):
            raise RuntimeError("MVSplat warp binding has an unexpected shape")
        captured["warps"].append(
            {
                "source_features": source_features.detach().clone(),
                "feature_pixel_intrinsics": feature_pixel_intrinsics.detach().clone(),
                "relative_reference_to_source_pose": relative_reference_to_source_pose.detach().clone(),
                "metric_depth": depth.detach().clone(),
            }
        )
        return original_warp(
            source_features,
            feature_pixel_intrinsics,
            relative_reference_to_source_pose,
            depth,
            *args,
            **kwargs,
        )

    def capture_raw_input(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 4:
            raise RuntimeError("MVSplat corr_refine_net received no [VB,D+C,h,w] input")
        if "raw_input" in captured:
            raise RuntimeError("MVSplat corr_refine_net was invoked more than once")
        captured["raw_input"] = inputs[0].detach().clone()
        raise _RawCostVolumeCaptured()

    setattr(module, "warp_with_pose_depth_candidates", capture_warp)
    handle = refinement.register_forward_pre_hook(capture_raw_input)
    try:
        with torch.no_grad():
            model.encoder(context, False, deterministic=True)
    except _RawCostVolumeCaptured:
        pass
    finally:
        handle.remove()
        setattr(module, "warp_with_pose_depth_candidates", original_warp)
    if "raw_input" not in captured or not captured["warps"]:
        raise RuntimeError("native MVSplat execution did not expose its raw cost volume")
    return captured


def _bind_native_capture(captured: dict[str, Any]) -> dict[str, Any]:
    """Validate one native capture and prepare exact selected-query inputs."""

    raw_input = captured.get("raw_input")
    warps = captured.get("warps")
    if not torch.is_tensor(raw_input) or raw_input.ndim != 4 or not isinstance(warps, list):
        raise RuntimeError("native raw-CV capture is malformed")
    first = warps[0]
    if not isinstance(first, dict):
        raise RuntimeError("native MVSplat warp capture is malformed")
    source = first.get("source_features")
    feature_pixel_intrinsics = first.get("feature_pixel_intrinsics")
    metric_depth = first.get("metric_depth")
    if (
        not torch.is_tensor(source)
        or not torch.is_tensor(feature_pixel_intrinsics)
        or not torch.is_tensor(metric_depth)
        or source.ndim != 4
        or metric_depth.shape[:1] != source.shape[:1]
        or metric_depth.shape[2:] != source.shape[2:]
    ):
        raise RuntimeError("first MVSplat warp lacks source-bound tensors")
    depth_count = int(metric_depth.shape[1])
    if raw_input.shape[1] <= depth_count:
        raise RuntimeError("MVSplat raw input has no concatenated reference features")
    canonical_depth = metric_depth[:, :, :1, :1]
    if not torch.equal(metric_depth, canonical_depth.expand_as(metric_depth)):
        raise RuntimeError("MVSplat metric depth candidates are not spatially constant")
    source_features = []
    relative_reference_to_source_poses = []
    for record in warps:
        if not isinstance(record, dict):
            raise RuntimeError("MVSplat warp capture contains a malformed record")
        record_source = record.get("source_features")
        record_intrinsics = record.get("feature_pixel_intrinsics")
        record_pose = record.get("relative_reference_to_source_pose")
        record_depth = record.get("metric_depth")
        if (
            not torch.is_tensor(record_source)
            or not torch.is_tensor(record_intrinsics)
            or not torch.is_tensor(record_pose)
            or not torch.is_tensor(record_depth)
            or record_source.shape != source.shape
            or not torch.equal(record_intrinsics, feature_pixel_intrinsics)
            or not torch.equal(record_depth, metric_depth)
        ):
            raise RuntimeError("MVSplat source views do not share one native CV binding")
        source_features.append(record_source)
        relative_reference_to_source_poses.append(record_pose)
    return {
        "reference_features": raw_input[:, depth_count:],
        "source_features": tuple(source_features),
        "feature_pixel_intrinsics": feature_pixel_intrinsics,
        "relative_reference_to_source_poses": tuple(relative_reference_to_source_poses),
        "inverse_depth_candidates": canonical_depth.reciprocal(),
        "native_raw_cost_volume": raw_input[:, :depth_count],
        "native_raw_input": raw_input,
    }


def _validate_phase_ledger(
    ledger: dict[str, Any],
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Reject missing, repeated, or unbound primary/secondary/Full work."""

    phases = ledger.get("phases")
    if not isinstance(phases, list) or len(phases) != 3:
        raise RuntimeError("selected raw-CV ledger has no three-phase trace")
    previous = torch.zeros_like(masks[0])
    for expected_name, mask, phase in zip(("primary", "secondary", "full"), masks, phases):
        if not isinstance(phase, dict) or phase.get("phase") != expected_name:
            raise RuntimeError("selected raw-CV phase order differs from the fixed trace")
        requested = int(mask.sum().item())
        executed = int(phase.get("raw_candidate_positions_executed", -1))
        reused = int(phase.get("raw_candidate_positions_reused", -1))
        if requested != int(phase.get("raw_candidate_positions_requested", -1)):
            raise RuntimeError("selected raw-CV phase request count is unbound")
        if executed + reused != requested:
            raise RuntimeError("selected raw-CV phase does not conserve requested work")
        if executed != int((mask & ~previous).sum().item()):
            raise RuntimeError("selected raw-CV phase replayed or omitted native positions")
        previous |= mask
    if int(ledger.get("raw_candidate_positions_executed", -1)) != int(previous.sum().item()):
        raise RuntimeError("selected raw-CV total execution count is inconsistent")


def collect_raw_cost_volume_audit(
    *, sample_index: int, device: torch.device
) -> dict[str, Any]:
    """Run the fixed context-only MVSplat raw-CV equivalence audit."""

    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity
    from saes.mvsplat_raw_cost_volume import MVSplatSelectedRawCostVolumeProducer

    experiment = resolve_experiment("mvsplat", "dl3dv", ROOT)
    selection = resolve_claim_selection("mvsplat", "dl3dv", ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        "mvsplat",
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
    target = batch.pop("target", None)
    native_target_rgb_loaded = isinstance(target, dict) and "image" in target
    context = {
        key: value.to(loaded_device) if torch.is_tensor(value) else value
        for key, value in batch["context"].items()
    }
    captured = _capture_native_raw_cost_volume(model, context)
    binding = _bind_native_capture(captured)
    native = binding["native_raw_cost_volume"]
    primary, secondary, full = build_fixed_native_cv_phase_masks(
        batch=native.shape[0],
        height=native.shape[2],
        width=native.shape[3],
        tile_size=4,
        device=loaded_device,
    )
    producer = MVSplatSelectedRawCostVolumeProducer(
        reference_features=binding["reference_features"],
        source_features=binding["source_features"],
        feature_pixel_intrinsics=binding["feature_pixel_intrinsics"],
        relative_reference_to_source_poses=binding[
            "relative_reference_to_source_poses"
        ],
        inverse_depth_candidates=binding["inverse_depth_candidates"],
        tile_size=4,
    )
    producer.execute("primary", primary)
    producer.execute("secondary", secondary)
    producer.execute("full", full)
    replay = producer.finalize(full)
    _validate_phase_ledger(replay.events, (primary, secondary, full))
    comparison = producer.verify_against_dense(native, mask=full, rtol=1.0e-5, atol=1.0e-5)
    status = "PASS" if comparison["equivalent"] else "FAIL"
    return {
        "schema_version": "1.0",
        "kind": "saes_mvsplat_selected_raw_cost_volume_audit",
        "status": status,
        "paper_result_eligible": False,
        "model": "mvsplat",
        "dataset": "dl3dv",
        "sample_index": sample_index,
        "scene": str(batch["scene"][0]),
        "context_indices": [int(value) for value in batch["context"]["index"][0].tolist()],
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "native_dataloader_loaded_target_rgb": native_target_rgb_loaded,
        "target_mapping_present_after_sanitization": "target" in batch,
        "execution_boundary": {
            "dense_s1_executed": True,
            "native_dense_raw_cost_volume_reference_executed": True,
            "selected_raw_cost_volume_phases_executed": True,
            "corr_refine_net_executed": False,
            "depth_head_executed": False,
            "full_resolution_refinement_executed": False,
            "gaussian_head_executed": False,
            "adapter_executed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "s2_s3_savings_claimed": False,
            "scope": "mvsplat_pre_refinement_raw_correlation_only",
        },
        "fixed_phase_trace": {
            "kind": "native_cv_coverage_trace_not_saes_route",
            "tile_size": 4,
            "primary_positions": int(primary.sum().item()),
            "secondary_positions": int(secondary.sum().item()),
            "full_logical_positions": int(full.sum().item()),
            "native_grid_shape": list(native.shape[-2:]),
            "head_grid_mapping_attempted": False,
        },
        "native_reference": {
            "corr_refine_input_source": "DepthPredictorMultiView.corr_refine_net pre-hook",
            "native_corr_refine_input_shape": list(binding["native_raw_input"].shape),
            "native_raw_cost_volume_shape": list(native.shape),
            "source_warp_count": len(binding["source_features"]),
            "source_view_layout": "view-major (v b)",
        },
        "selected_raw_cost_volume": replay.events,
        "native_raw_cost_volume_equivalence": comparison,
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "source": source_identity(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_index != 0:
        parser.error("the raw-CV audit is fixed to DL3DV sample index 0")
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
    record = collect_raw_cost_volume_audit(sample_index=args.sample_index, device=device)
    from scripts.result_record import portable_command, write_result

    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")


if __name__ == "__main__":
    main()
