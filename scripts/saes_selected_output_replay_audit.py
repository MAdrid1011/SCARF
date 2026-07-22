#!/usr/bin/env python3
"""Audit same-weight selected Gaussian-head replay without target RGB or rendering."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.selected_output_replay import (
    compare_selected_outputs,
    replay_two_conv_selected_outputs,
)
from scripts.saes_dependency_audit import (
    _context_on_device,
    build_probe_mask,
    remove_target_rgb,
)


class _RawHeadCaptured(RuntimeError):
    """Internal sentinel used to stop execution directly after the raw head."""


@contextmanager
def strict_fp32_convolution_execution() -> dict[str, bool]:
    """Disable TF32 only while comparing two algebraically identical heads.

    The dense CUDNN kernel and the selected-output patch kernel otherwise use
    different TF32 accumulation paths on Ampere GPUs.  This audit is a
    numerical equivalence proof rather than a throughput measurement, so it
    executes both sides in the same fixed FP32 mode and restores the caller's
    process-wide settings on exit.
    """
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield {
            "cuda_matmul_tf32_enabled": False,
            "cudnn_tf32_enabled": False,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def _classic_raw_head(model: Any, model_name: str) -> Any:
    if model_name not in {"transplat", "mvsplat"}:
        raise ValueError("selected-output replay is currently defined for transplat/mvsplat")
    predictor = model.encoder.depth_predictor
    head = getattr(predictor, "to_gaussians", None)
    if head is None:
        raise RuntimeError(f"{model_name} predictor lacks to_gaussians")
    return head


def capture_raw_head_io(
    model: Any, context: dict[str, Any], *, model_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture head input/output and terminate before adapter or renderer work."""
    head = _classic_raw_head(model, model_name)
    captured: dict[str, torch.Tensor] = {}

    def capture_and_stop(
        _module: Any, inputs: tuple[Any, ...], output: Any
    ) -> None:
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 4:
            raise RuntimeError("raw Gaussian head did not receive one [VB,C,H,W] input")
        if not torch.is_tensor(output) or output.ndim != 4:
            raise RuntimeError("raw Gaussian head did not emit [VB,C,H,W]")
        captured["input"] = inputs[0].detach().clone()
        captured["output"] = output.detach().clone()
        raise _RawHeadCaptured()

    handle = head.register_forward_hook(capture_and_stop)
    try:
        with torch.no_grad():
            model.encoder(context, False, deterministic=True)
    except _RawHeadCaptured:
        pass
    finally:
        handle.remove()
    if set(captured) != {"input", "output"}:
        raise RuntimeError("raw Gaussian head capture did not complete exactly once")
    return captured["input"], captured["output"]


def _coordinates_sha256(coordinates: torch.Tensor) -> str:
    payload = coordinates.detach().to(device="cpu", dtype=torch.int64).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def collect_selected_output_replay_audit(
    *, model_name: str, sample_index: int, device: torch.device
) -> dict[str, Any]:
    """Run one target-free DL3DV selected-output equivalence audit."""
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
    with strict_fp32_convolution_execution() as numerical_execution:
        head_input, dense_head_output = capture_raw_head_io(
            model, context, model_name=model_name
        )
        _, _, height, width = head_input.shape
        selection_mask = build_probe_mask(height=height, width=width, tile_size=4)
        replay = replay_two_conv_selected_outputs(
            _classic_raw_head(model, model_name), head_input, selection_mask
        )
        equivalence = compare_selected_outputs(dense_head_output, replay)
    if not equivalence["equivalent"]:
        raise RuntimeError(
            "same-weight selected-output replay is not numerically equivalent: "
            f"max_abs={equivalence['maximum_absolute_delta']}"
        )
    return {
        "schema_version": "1.0",
        "kind": "saes_selected_output_replay_audit",
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "native_dataloader_loaded_target_rgb": native_target_rgb_loaded,
        "model": model_name,
        "dataset": "dl3dv",
        "sample_index": sample_index,
        "scene": str(batch["scene"][0]),
        "context_indices": [int(value) for value in batch["context"]["index"][0].tolist()],
        "selection": {
            "pattern": "four-corners-per-4x4-full-resolution-tile",
            "coordinates_sha256": _coordinates_sha256(replay.coordinates),
            "selected_output_positions": int(replay.coordinates.shape[0]),
        },
        "execution_boundary": {
            "completed": ("S1", "S2_dense", "S3_dense_closure", "S3_selected_output_replay"),
            "gaussian_adapter_executed": False,
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
        "head_input_shape": list(head_input.shape),
        "dense_head_output_shape": list(dense_head_output.shape),
        "equivalence": equivalence,
        "numerical_execution": numerical_execution,
        "execution_events": replay.events,
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
        parser.error("the selected-output audit is predeclared for DL3DV sample index 0")
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

    record = collect_selected_output_replay_audit(
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
