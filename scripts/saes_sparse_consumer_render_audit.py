#!/usr/bin/env python3
"""Render one source-bound SAES selected packet without target RGB metrics."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.probe_first_schedule import build_incremental_probe_first_plan
from saes.sparse_gaussian_consumer import PackedGaussianAttributes
from scripts.saes_incremental_selected_output_audit import (
    DECISION_SEMANTICS,
    DEPTH_THRESHOLD,
    FEATURE_THRESHOLD,
    TILE_SIZE,
    _attribute_equivalence,
    _capture_dense_reference,
    _capture_guarded_incremental_packed_adapter,
    _extract_selected_adapter_inputs,
    _numeric_equivalence,
    _packed_attributes_to_device,
    _release_cuda_cache,
    _selected_adapter_inputs_equivalence,
    _selected_head_descriptors,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution


MODEL = "transplat"
DATASET = "dl3dv"
AUDIT_KIND = "saes_sparse_consumer_renderer_audit"


class _SparseConsumerAttributeDrift(RuntimeError):
    """Carry fail-closed dense-versus-incremental diagnostics to the result file."""

    def __init__(self, message: str, *, diagnosis: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnosis = diagnosis


def _target_cameras(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    target = batch.get("target")
    if not isinstance(target, Mapping):
        raise RuntimeError("sparse consumer render audit has no target camera mapping")
    required = ("extrinsics", "intrinsics", "near", "far")
    if any(not torch.is_tensor(target.get(name)) for name in required):
        raise RuntimeError("sparse consumer render audit has incomplete target cameras")
    return {name: target[name].to(device) for name in required}


def render_packed_gaussians(
    decoder: Any,
    packed: PackedGaussianAttributes,
    *,
    gaussians_type: type[Any],
    target: Mapping[str, torch.Tensor],
    image_shape: tuple[int, int],
    expected_descriptor_count: int,
) -> tuple[Any, torch.Tensor]:
    """Render only the final packed attributes and reject shape drift."""

    if not isinstance(packed, PackedGaussianAttributes):
        raise TypeError("renderer audit requires packed Gaussian attributes")
    if (
        isinstance(expected_descriptor_count, bool)
        or not isinstance(expected_descriptor_count, int)
        or expected_descriptor_count <= 0
        or packed.dense_slots.numel() != expected_descriptor_count
    ):
        raise ValueError("packed descriptor count does not match the final route")
    gaussians = packed.as_single_batch(gaussians_type)
    for name in ("means", "covariances", "harmonics", "opacities"):
        value = getattr(gaussians, name, None)
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"packed renderer received invalid {name}")
    if gaussians.means.shape[1] != expected_descriptor_count:
        raise RuntimeError("decoder Gaussian batch does not match the final route")
    with torch.no_grad():
        output = decoder.forward(
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
        raise RuntimeError("packed renderer did not emit a finite color tensor")
    return gaussians, color


def _worst_descriptor_delta(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    dense_slots: torch.Tensor,
    *,
    height: int,
    width: int,
) -> dict[str, Any] | None:
    """Locate the largest per-descriptor difference in canonical decoder order."""
    if (
        reference.shape != candidate.shape
        or reference.ndim < 1
        or reference.shape[0] != dense_slots.numel()
        or reference.shape[0] == 0
    ):
        return None
    if candidate.device != reference.device:
        candidate = candidate.to(reference.device)
    per_descriptor = (reference - candidate).abs().reshape(reference.shape[0], -1).amax(dim=1)
    descriptor_index = int(per_descriptor.argmax().item())
    dense_slot = int(dense_slots[descriptor_index].item())
    pixels_per_view = height * width
    view = dense_slot // pixels_per_view
    pixel = dense_slot % pixels_per_view
    return {
        "descriptor_index": descriptor_index,
        "dense_slot": dense_slot,
        "view": view,
        "row": pixel // width,
        "column": pixel % width,
        "maximum_absolute_delta": float(per_descriptor[descriptor_index].item()),
    }


def _final_packet_equivalence_gate(
    *,
    packet_adapter_inputs: Mapping[str, Any],
    geometry_inputs: Mapping[str, Mapping[str, Any]],
    raw_offset_equivalence: Mapping[str, Any],
    adapter_equivalence: Mapping[str, Any],
) -> bool:
    """Gate only values the packed Adapter and decoder actually consume."""
    return (
        packet_adapter_inputs.get("equivalent") is True
        and all(comparison.get("equivalent") is True for comparison in geometry_inputs.values())
        and raw_offset_equivalence.get("equivalent") is True
        and adapter_equivalence.get("equivalent") is True
    )


def collect_sparse_consumer_render_audit(
    *, device: torch.device, sample_index: int
) -> dict[str, Any]:
    """Run one local packet-to-renderer diagnostic after route commitment."""

    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("sample_index must be a nonnegative integer")
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    selection = resolve_claim_selection(MODEL, DATASET, ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        MODEL,
        dataset_name=DATASET,
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
    target_mapping = batch.get("target")
    if not isinstance(target_mapping, dict) or "image" not in target_mapping:
        raise RuntimeError("sparse consumer render audit requires native target RGB removal")
    target = _target_cameras(batch, loaded_device)
    del target_mapping["image"]
    context = {
        key: value.to(loaded_device) if torch.is_tensor(value) else value
        for key, value in batch["context"].items()
    }
    if context["image"].shape[0] != 1:
        raise RuntimeError("sparse consumer render audit requires batch size one")
    _, views, _, height, width = context["image"].shape

    with strict_fp32_convolution_execution() as numerical_execution:
        dense_reference, dense_capture = _capture_dense_reference(model, context)
        _release_cuda_cache(loaded_device)
        plan = build_incremental_probe_first_plan(
            dense_capture["features"],
            dense_capture["depths"],
            height=height,
            width=width,
            tile_size=TILE_SIZE,
            feature_threshold=FEATURE_THRESHOLD,
            depth_threshold=DEPTH_THRESHOLD,
            decision_semantics=DECISION_SEMANTICS,
        )
        capture = _capture_guarded_incremental_packed_adapter(model, context, plan=plan)
        final_packet = capture["final_packet"]
        final_packed = capture["final_packed"]
        if not isinstance(final_packed, PackedGaussianAttributes):
            raise RuntimeError("incremental audit did not return final packed attributes")
        expected_descriptor_count = int(
            capture["guarded_route"].selected_output_mask.sum().item()
        )
        if final_packet.dense_slots.numel() != expected_descriptor_count:
            raise RuntimeError("final raw packet does not cover the selected output")
        if final_packed.dense_slots.numel() != expected_descriptor_count:
            raise RuntimeError("final packed attributes do not cover the selected output")
        final_mask = capture["guarded_route"].selected_output_mask
        dense_final_inputs = _extract_selected_adapter_inputs(
            dense_capture["adapter_inputs"], final_mask
        )
        dense_vs_incremental_inputs = _selected_adapter_inputs_equivalence(
            dense_final_inputs, capture["final_selected_inputs"]
        )
        dense_final_raw = _selected_head_descriptors(
            dense_capture["raw_head"], final_mask
        )
        raw_descriptor_equivalence = _numeric_equivalence(
            dense_final_raw, final_packet.raw_descriptors
        )
        raw_offset_equivalence = _numeric_equivalence(
            dense_final_raw[:, :2], final_packet.raw_descriptors[:, :2]
        )
        adapter_equivalence = _attribute_equivalence(
            final_packed, dense_reference, final_packed.dense_slots
        )
        geometry_inputs = {
            "camera_rotation": _numeric_equivalence(
                dense_final_inputs.extrinsics[..., :3, :3],
                capture["final_selected_inputs"].extrinsics[..., :3, :3],
            ),
            "camera_translation": _numeric_equivalence(
                dense_final_inputs.extrinsics[..., :3, 3],
                capture["final_selected_inputs"].extrinsics[..., :3, 3],
            ),
            "intrinsics": dense_vs_incremental_inputs["inputs"]["intrinsics"],
            "coordinates": dense_vs_incremental_inputs["inputs"]["coordinates"],
            "depths": dense_vs_incremental_inputs["inputs"]["depths"],
        }
        geometry_inputs_equivalent = all(
            comparison["equivalent"] for comparison in geometry_inputs.values()
        )
        diagnosis = {
            "dense_vs_incremental_final_adapter_inputs": dense_vs_incremental_inputs,
            "dense_vs_incremental_geometry_inputs": geometry_inputs,
            "dense_vs_incremental_geometry_inputs_match": geometry_inputs_equivalent,
            "dense_vs_incremental_final_raw_descriptor": raw_descriptor_equivalence,
            "dense_vs_incremental_final_raw_offset_xy": raw_offset_equivalence,
            "packed_attributes_vs_dense_reference": adapter_equivalence,
            "worst_mean_descriptor": _worst_descriptor_delta(
                dense_reference.means[0, final_packed.dense_slots.cpu()].to(
                    final_packed.means.device
                ),
                final_packed.means,
                final_packed.dense_slots,
                height=height,
                width=width,
            ),
        }
        gate_passed = _final_packet_equivalence_gate(
            packet_adapter_inputs=capture["final_adapter_inputs"],
            geometry_inputs=geometry_inputs,
            raw_offset_equivalence=raw_offset_equivalence,
            adapter_equivalence=adapter_equivalence,
        )
        if not gate_passed:
            maxima = {
                name: values["maximum_absolute_delta"]
                for name, values in adapter_equivalence["attributes"].items()
            }
            raise _SparseConsumerAttributeDrift(
                "final packed Adapter/geometry gate failed against dense reference: "
                f"{maxima}",
                diagnosis=diagnosis,
            )
        renderable_packed = _packed_attributes_to_device(final_packed, loaded_device)
        _gaussians, rendered_color = render_packed_gaussians(
            model.decoder,
            renderable_packed,
            gaussians_type=type(dense_reference),
            target=target,
            image_shape=(height, width),
            expected_descriptor_count=expected_descriptor_count,
        )

    guarded_route = capture["guarded_route"]
    return {
        "schema_version": "1.0",
        "kind": AUDIT_KIND,
        "status": "PASS",
        "paper_result_eligible": False,
        "model": MODEL,
        "dataset": DATASET,
        "sample_index": sample_index,
        "scene": str(batch["scene"][0]),
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": True,
        "target_mapping_present": True,
        "execution_boundary": {
            "dense_context_planning_pass_executed": True,
            "incremental_context_adapter_boundary_pass_executed": True,
            "packed_final_adapter_executed_in_same_encoder_invocation": True,
            "guarded_selected_route_resolved": True,
            "additional_full_dispatch_executed": capture["extension_event"] is not None,
            "renderer_executed": True,
            "quality_metrics_computed": False,
            "timing_claim": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "scope": "source_bound_selected_packet_to_native_adapter_to_renderer_diagnostic_only",
        },
        "route_plan": plan.events,
        "guarded_selected_route": guarded_route.events,
        "final_packet": {
            "descriptor_count": expected_descriptor_count,
            "dense_slots_strictly_canonical": True,
            "source_trace_sha256": final_packed.source_trace_sha256,
            "omitted_raw_head_positions_poisoned": capture[
                "omitted_raw_head_positions_poisoned"
            ],
            "packed_attributes_vs_dense_reference": adapter_equivalence,
            "final_adapter_inputs_match_selected_packet": capture[
                "final_adapter_inputs"
            ],
            "final_coordinates_rebuilt_from_source_geometry_after_extension": capture[
                "final_coordinates_rebuilt_from_source_geometry_after_extension"
            ],
            "dense_vs_incremental_geometry_inputs_match": geometry_inputs_equivalent,
            "dense_vs_incremental_final_adapter_inputs": dense_vs_incremental_inputs,
            "dense_vs_incremental_final_raw_descriptor": raw_descriptor_equivalence,
            "dense_vs_incremental_final_raw_offset_xy": raw_offset_equivalence,
        },
        "render": {
            "color_shape": list(rendered_color.shape),
            "finite": True,
            "decoder_gaussian_count": expected_descriptor_count,
        },
        "numerical_execution": numerical_execution,
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "source": source_identity(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this local renderer audit requires CUDA")
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        record = collect_sparse_consumer_render_audit(
            device=device, sample_index=args.sample_index
        )
    except Exception as exc:
        record = {
            "schema_version": "1.0",
            "kind": AUDIT_KIND,
            "status": "FAILED",
            "paper_result_eligible": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, _SparseConsumerAttributeDrift):
            record["diagnosis"] = exc.diagnosis
        exit_code = 2
    else:
        exit_code = 0
    from scripts.result_record import portable_command, write_result

    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
