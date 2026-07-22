#!/usr/bin/env python3
"""Run the local DL3DV sample-0 Full-route packet fidelity pilot.

This is intentionally narrower than the default simulator: it proves that a
source-bound packed Gaussian route can reach the native decoder under the
frozen SAES identity when that route fails closed to Full. It does not claim a
nonzero deletion, a representative materialization, or whole-pipeline sparse
S2/S3 execution.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.probe_first_schedule import build_incremental_probe_first_plan
from saes.sparse_gaussian_consumer import PackedGaussianAttributes
from scripts.saes_execution_identity import build_saes_execution_identity
from scripts.saes_incremental_selected_output_audit import (
    DECISION_SEMANTICS,
    DEPTH_THRESHOLD,
    FEATURE_THRESHOLD,
    TILE_SIZE,
    _attribute_equivalence,
    _capture_guarded_incremental_packed_adapter,
    _capture_s1_s2_without_dense_adapter,
    _packed_attributes_to_device,
    _release_cuda_cache,
)
from scripts.saes_selected_output_quality_gate import (
    _mean_metrics,
    _quality_verdict,
    _view_metrics,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution
from scripts.saes_sparse_consumer_render_audit import (
    _target_cameras,
    render_packed_gaussians,
)


MODEL = "transplat"
DATASET = "dl3dv"
SAMPLE_INDEX = 0
SEED = 0
PILOT_KIND = "saes_sparse_packet_full_route_quality_pilot"


def _take_target_rgb_for_metrics(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    """Move target RGB only after both decoder outputs have been fixed."""
    target = batch.get("target")
    if not isinstance(target, dict):
        raise RuntimeError("packet quality pilot has no mutable target mapping")
    images = target.pop("image", None)
    if not torch.is_tensor(images):
        raise RuntimeError("packet quality pilot requires native target RGB")
    return images.to(device)


def _render_dense_baseline(
    model: Any,
    gaussians: Any,
    *,
    target: Mapping[str, torch.Tensor],
    image_shape: tuple[int, int],
) -> torch.Tensor:
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
    color = getattr(output, "color", None)
    if not torch.is_tensor(color) or not bool(torch.isfinite(color).all()):
        raise RuntimeError("native dense baseline decoder did not emit finite color")
    return color


def _require_sample_zero_full_route(
    capture: Mapping[str, Any], *, views: int, height: int, width: int
) -> int:
    """Accept only the known frozen Full-route identity for this fidelity pilot."""
    guarded_route = capture.get("guarded_route")
    if guarded_route is None or not hasattr(guarded_route, "events"):
        raise RuntimeError("packet quality pilot has no guarded route")
    events = guarded_route.events
    expected_tiles = views * (height // TILE_SIZE) * (width // TILE_SIZE)
    expected_descriptors = views * height * width
    if events.get("route_counts") != {"L0": 0, "L1": 0, "Full": expected_tiles}:
        raise RuntimeError("sample-0 packet route does not match the frozen Full fidelity route")
    if events.get("deletion_certificate_required") is not True:
        raise RuntimeError("packet route did not enforce the frozen deletion certificate")
    final_mask = guarded_route.raw_head_request_mask
    if (
        not torch.is_tensor(final_mask)
        or final_mask.dtype != torch.bool
        or int(final_mask.sum().item()) != expected_descriptors
        or not bool(final_mask.all())
    ):
        raise RuntimeError("sample-0 packet route did not request every Full descriptor")
    final_packet = capture.get("final_packet")
    final_packed = capture.get("final_packed")
    if (
        final_packet is None
        or not isinstance(final_packed, PackedGaussianAttributes)
        or final_packet.dense_slots.numel() != expected_descriptors
        or final_packed.dense_slots.numel() != expected_descriptors
    ):
        raise RuntimeError("sample-0 packet output does not cover the Full route")
    return expected_descriptors


def collect_packet_quality_pilot(*, device: torch.device) -> dict[str, Any]:
    """Compare Full-route packed decoding to the native dense baseline once."""
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    selection = resolve_claim_selection(MODEL, DATASET, ROOT)
    identity = build_saes_execution_identity()
    model, batch, _cfg, loaded_device = load_model_and_data(
        MODEL,
        dataset_name=DATASET,
        checkpoint_path=experiment.checkpoint,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=1,
        sample_index=SAMPLE_INDEX,
    )
    model.eval()
    from src.model.types import Gaussians

    context = {
        key: value.to(loaded_device) if torch.is_tensor(value) else value
        for key, value in batch["context"].items()
    }
    if context["image"].shape[0] != 1:
        raise RuntimeError("packet quality pilot requires batch size one")
    _, views, _, height, width = context["image"].shape

    with strict_fp32_convolution_execution() as numerical_execution:
        planning_inputs = _capture_s1_s2_without_dense_adapter(model, context)
        _release_cuda_cache(loaded_device)
        plan = build_incremental_probe_first_plan(
            planning_inputs["features"],
            planning_inputs["depths"],
            height=height,
            width=width,
            tile_size=TILE_SIZE,
            feature_threshold=FEATURE_THRESHOLD,
            depth_threshold=DEPTH_THRESHOLD,
            decision_semantics=DECISION_SEMANTICS,
        )
        capture = _capture_guarded_incremental_packed_adapter(model, context, plan=plan)
        expected_descriptor_count = _require_sample_zero_full_route(
            capture, views=views, height=height, width=width
        )
        final_packed = capture["final_packed"]
        if not isinstance(final_packed, PackedGaussianAttributes):
            raise RuntimeError("packet capture did not produce packed attributes")
        renderable_packed = _packed_attributes_to_device(final_packed, loaded_device)
        # Packet construction is complete before any target-side metadata is
        # read for rendering or metric computation.
        target_mapping = batch.get("target")
        if not isinstance(target_mapping, dict) or not torch.is_tensor(target_mapping.get("image")):
            raise RuntimeError("packet quality pilot requires native target RGB for metrics")
        target_cameras = _target_cameras(batch, loaded_device)
        _packet_gaussians, packet_color = render_packed_gaussians(
            model.decoder,
            renderable_packed,
            gaussians_type=Gaussians,
            target=target_cameras,
            image_shape=(height, width),
            expected_descriptor_count=expected_descriptor_count,
        )

        # The baseline is intentionally independent of route construction and
        # is executed only after the compact decoder output is committed.
        with torch.no_grad():
            baseline_gaussians = model.encoder(context, 0, deterministic=True)
        baseline_color = _render_dense_baseline(
            model,
            baseline_gaussians,
            target=target_cameras,
            image_shape=(height, width),
        )
        adapter_equivalence = _attribute_equivalence(
            final_packed, baseline_gaussians, final_packed.dense_slots
        )
        if not adapter_equivalence["equivalent"]:
            raise RuntimeError("Full-route packed attributes drift from the native baseline")

    # Both source outputs are fixed. This is the first RGB tensor transfer.
    target_rgb = _take_target_rgb_for_metrics(batch, loaded_device)
    baseline_views = _view_metrics(baseline_color[0], target_rgb[0])
    packet_views = _view_metrics(packet_color[0], target_rgb[0])
    baseline_quality = _mean_metrics(baseline_views)
    packet_quality = _mean_metrics(packet_views)
    verdict = _quality_verdict(baseline_quality, packet_quality)
    if not verdict["pass"]:
        raise RuntimeError("Full-route packed decoder missed the fixed quality tolerance")

    guarded_route = capture["guarded_route"]
    return {
        "schema_version": "1.0",
        "kind": PILOT_KIND,
        "status": "PASS",
        "paper_result_eligible": False,
        "model": MODEL,
        "dataset": DATASET,
        "sample_index": SAMPLE_INDEX,
        "scene": str(batch["scene"][0]),
        "context_indices": [int(value) for value in batch["context"]["index"][0].tolist()],
        "target_indices": [int(value) for value in batch["target"]["index"][0].tolist()],
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": True,
            "not_accessed_or_transferred_before_packet_and_baseline_outputs": True,
            "target_camera_metadata_accessed_before_packet_commit": False,
            "target_camera_metadata_accessed_after_packet_commit": True,
            "passed_to_encoder_or_route": False,
            "first_transfer_after_route_and_decoder_commit": True,
        },
        "execution_boundary": {
            "planning_dense_raw_head_executed": True,
            "planning_dense_adapter_executed": False,
            "planning_dense_gaussians_materialized": False,
            "source_bound_packet_adapter_executed": True,
            "native_packet_decoder_executed": True,
            "independent_dense_baseline_executed_after_packet_decoder": True,
            "quality_metrics_computed": True,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "s2_s3_saving": 0.0,
            "timing_claim": False,
        },
        "fidelity_scope": {
            "route": "sample-0 frozen Full identity only",
            "packet_is_strictly_compact": False,
            "nonzero_deletion_verified": False,
            "representative_materialization_verified": False,
            "reason": "all tiles fail closed to Full before a deletion can be certified",
        },
        "saes_execution_identity": identity,
        "route_plan": plan.events,
        "guarded_selected_route": guarded_route.events,
        "final_packet": {
            "descriptor_count": expected_descriptor_count,
            "source_trace_sha256": final_packed.source_trace_sha256,
            "omitted_raw_head_positions_poisoned": capture[
                "omitted_raw_head_positions_poisoned"
            ],
            "native_adapter_attribute_equivalence": adapter_equivalence,
            "decoder_gaussian_count": expected_descriptor_count,
        },
        "quality": {
            "baseline": baseline_quality,
            "packet": packet_quality,
            "verdict": verdict,
            "views": [
                {
                    "target_index": int(batch["target"]["index"][0, index].item()),
                    "baseline": baseline_views[index],
                    "packet": packet_views[index],
                }
                for index in range(len(baseline_views))
            ],
        },
        "numerical_execution": numerical_execution,
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "source": source_identity(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this local packet quality pilot requires CUDA")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from scripts.result_record import write_result

    try:
        record = collect_packet_quality_pilot(device=device)
        exit_code = 0
    except Exception as exc:
        record = {
            "schema_version": "1.0",
            "kind": PILOT_KIND,
            "status": "FAILED",
            "paper_result_eligible": False,
            "model": MODEL,
            "dataset": DATASET,
            "sample_index": SAMPLE_INDEX,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
        exit_code = 1
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
