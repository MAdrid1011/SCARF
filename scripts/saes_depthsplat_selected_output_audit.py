#!/usr/bin/env python3
"""Audit DepthSplat's source-bound selected head and RGB Adapter boundary.

This is a target-free prerequisite for the DepthSplat L0/L1 simulator.  It
runs the native dense regressor, replays only selected replicate-padded final
head outputs, then rebatches their source RGB and z-depth side inputs through
the original GaussianAdapter.  It is neither a merge-quality result nor an
S2/S3 saving claim.
"""

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

from saes.depthsplat_backend import (  # noqa: E402
    freeze_depthsplat_backend_identity,
    resolve_depthsplat_backend_contract,
)
from saes.depthsplat_selected_output import (  # noqa: E402
    DepthSplatPackedGaussianConsumer,
    build_depthsplat_sparse_raw_packet,
    capture_depthsplat_native_execution,
    compare_depthsplat_packed_to_dense,
    replay_depthsplat_selected_head,
)
from scripts.saes_dependency_audit import (  # noqa: E402
    _context_on_device,
    build_probe_mask,
    remove_target_rgb,
)
from scripts.saes_selected_output_replay_audit import (  # noqa: E402
    strict_fp32_convolution_execution,
)


AUDIT_KIND = "depthsplat-selected-output-rgb-adapter-audit"
AUDIT_SCHEMA_VERSION = "1.0"
_FIXED_SAMPLE_INDEX = 0


def _selection_mask(*, views: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Build the predeclared four-corners-per-T=4 source head selection."""

    probe = build_probe_mask(height=height, width=width, tile_size=4).to(device)
    return probe.unsqueeze(0).expand(views, -1, -1).clone()


def collect_depthsplat_selected_output_audit(
    *, sample_index: int, device: torch.device
) -> dict[str, Any]:
    """Run exactly one target-free native DepthSplat/DL3DV audit endpoint."""

    if sample_index != _FIXED_SAMPLE_INDEX:
        raise ValueError("DepthSplat selected-output audit is predeclared for DL3DV sample index 0")
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import source_identity

    backend_contract = resolve_depthsplat_backend_contract(ROOT)
    backend_identity = freeze_depthsplat_backend_identity(backend_contract)
    experiment = resolve_experiment("depthsplat", "dl3dv", ROOT)
    selection = resolve_claim_selection("depthsplat", "dl3dv", ROOT)
    if (
        experiment.checkpoint.resolve() != backend_contract.checkpoint
        or selection.index_path.resolve() != backend_contract.evaluation_index
    ):
        raise RuntimeError("DepthSplat audit application identity differs from its frozen backend")
    model, batch, _cfg, loaded_device = load_model_and_data(
        "depthsplat",
        dataset_name="dl3dv",
        checkpoint_path=experiment.checkpoint,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=1,
        sample_index=sample_index,
    )
    model.eval()
    native_target_rgb_loaded = remove_target_rgb(batch)
    context = _context_on_device(batch, loaded_device)
    with strict_fp32_convolution_execution() as numerical_execution:
        execution = capture_depthsplat_native_execution(
            model.encoder, context, source_root=ROOT / "depthsplat"
        )
        views, _, height, width = execution.dense_raw_head.shape
        selection_mask = _selection_mask(
            views=views, height=height, width=width, device=loaded_device
        )
        replay = replay_depthsplat_selected_head(
            model.encoder.gaussian_head,
            execution.gaussian_head_input,
            execution.dense_raw_head,
            selection_mask,
        )
        packet = build_depthsplat_sparse_raw_packet(execution, replay)
        packed = DepthSplatPackedGaussianConsumer(
            model.encoder.gaussian_adapter
        ).convert(packet, image_shape=(height, width))
        adapter_equivalence = compare_depthsplat_packed_to_dense(
            packed, execution.dense_gaussians
        )
    if replay.equivalence["equivalent"] is not True:
        raise RuntimeError("DepthSplat selected gaussian_head replay is not equivalent")
    if adapter_equivalence["equivalent"] is not True:
        raise RuntimeError("DepthSplat selected RGB Adapter packet is not equivalent")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "paper_result_eligible": False,
        "model": "depthsplat",
        "dataset": "dl3dv",
        "sample_index": sample_index,
        "scene": str(batch["scene"][0]),
        "context_indices": [
            int(value) for value in batch["context"]["index"][0].tolist()
        ],
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": native_target_rgb_loaded,
            "removed_before_context_device_transfer": True,
            "passed_to_encoder_or_route": False,
            "used_for_adapter_side_inputs": False,
            "used_for_metric": False,
        },
        "selection": {
            "pattern": "four-corners-per-4x4-full-resolution-tile",
            "mask_sha256": packet.source_trace["selection_mask_sha256"],
            "selected_descriptor_count": int(packet.dense_slots.numel()),
            "raw_head_shape": list(execution.dense_raw_head.shape),
        },
        "execution_boundary": {
            "dense_depth_predictor_executed": True,
            "native_dense_gaussian_regressor_executed": True,
            "selected_replicate_gaussian_head_executed": True,
            "selected_native_rgb_adapter_executed": True,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        },
        "native_execution": execution.events,
        "selected_head": {
            "equivalence": replay.equivalence,
            "events": replay.events,
        },
        "packet": {
            "descriptor_count": int(packet.dense_slots.numel()),
            "source_trace": dict(packet.source_trace),
            "source_trace_sha256": packed.source_trace_sha256,
        },
        "adapter_equivalence": adapter_equivalence,
        "backend_identity": backend_identity,
        "source": source_identity(),
        "numerical_execution": numerical_execution,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=_FIXED_SAMPLE_INDEX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("DepthSplat selected-output audit requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        record = collect_depthsplat_selected_output_audit(
            sample_index=args.sample_index, device=device
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    from scripts.result_record import portable_command, write_result

    args.output_dir.mkdir(parents=True, exist_ok=False)
    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
