#!/usr/bin/env python3
"""Run a source-bound DepthSplat L0/L1 materializer decoder smoke.

This entrypoint is deliberately a development diagnostic rather than a
formal target-free or quality gate.  The native DL3DV loader currently builds
a target mapping for DepthSplat, so the mapping's RGB is removed and the
entire mapping is discarded before the context is moved to the device.  The
remaining execution uses only context images and context cameras, including
for the native decoder shape smoke.  A formal context-only sidecar must
replace this input boundary before a quality or paper-facing claim is made.
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

from saes.depthsplat_backend import (  # noqa: E402
    freeze_depthsplat_backend_identity,
    resolve_depthsplat_backend_contract,
)
from saes.depthsplat_l0_l1_materializer import (  # noqa: E402
    apply_depthsplat_compact_l0_l1_materialization,
    preflight_depthsplat_l0_l1_materialization,
    resolve_depthsplat_compact_final_route,
)
from saes.depthsplat_selected_output import (  # noqa: E402
    DepthSplatPackedGaussianConsumer,
    build_depthsplat_sparse_raw_packet,
    compare_depthsplat_full_passthrough_to_dense_bitwise,
    capture_depthsplat_native_execution,
    compare_depthsplat_packed_to_dense,
    replay_depthsplat_selected_head,
    subset_depthsplat_sparse_raw_packet,
)
from saes.probe_first_schedule import (  # noqa: E402
    LEGACY_L1_ANCHOR_SEMANTICS,
    build_incremental_probe_first_plan,
)
from saes.progressive_saes import (  # noqa: E402
    PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
)
from scripts.saes_dependency_audit import remove_target_rgb  # noqa: E402
from scripts.saes_selected_output_replay_audit import (  # noqa: E402
    strict_fp32_convolution_execution,
)


MODEL = "depthsplat"
DATASET = "dl3dv"
SAMPLE_INDEX = 0
SEED = 0
TILE_SIZE = 4
FEATURE_THRESHOLD = 0.20
DEPTH_THRESHOLD = 0.10
DECISION_SEMANTICS = PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS
L1_ANCHOR_SEMANTICS = LEGACY_L1_ANCHOR_SEMANTICS
AUDIT_KIND = "depthsplat-source-bound-l0-l1-development-materializer-smoke"
AUDIT_SCHEMA_VERSION = "1.0"


def _discard_target_before_context_transfer(
    batch: dict[str, Any], device: torch.device
) -> tuple[dict[str, Any], bool]:
    """Keep only context tensors after removing native-loader target RGB.

    ``remove_target_rgb`` is the one permitted interaction with the native
    target mapping: it drops the RGB tensor without transferring or inspecting
    its pixels.  The mapping is then discarded as a whole, so target camera
    metadata and target indices cannot reach the route, Adapter, or decoder.
    """

    if not isinstance(batch, dict):
        raise TypeError("DepthSplat smoke requires a mutable native batch")
    native_target_rgb_loaded = remove_target_rgb(batch)
    batch.pop("target", None)
    source_context = batch.get("context")
    if not isinstance(source_context, Mapping):
        raise RuntimeError("DepthSplat smoke batch has no context mapping")
    required = ("image", "extrinsics", "intrinsics", "near", "far", "index")
    if any(not torch.is_tensor(source_context.get(name)) for name in required):
        raise RuntimeError("DepthSplat smoke context is incomplete")
    context = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in source_context.items()
    }
    if "target" in batch:
        raise RuntimeError("DepthSplat smoke retained a target mapping")
    return context, native_target_rgb_loaded


def _source_geometry_functions(encoder: Any) -> tuple[Any, Any, dict[str, str]]:
    """Resolve geometry only from the loaded DepthSplat ``src`` namespace."""

    module_name = type(encoder).__module__
    marker = ".model."
    if marker not in module_name:
        raise RuntimeError("DepthSplat encoder has an unexpected source module name")
    geometry_module_name = module_name.split(marker, 1)[0] + ".geometry.projection"
    module = sys.modules.get(geometry_module_name)
    source = getattr(module, "__file__", None)
    source_root = (ROOT / "depthsplat" / "src").resolve()
    if not isinstance(source, str):
        raise RuntimeError("DepthSplat source geometry module was not loaded")
    source_path = Path(source).resolve()
    if source_root not in source_path.parents:
        raise RuntimeError("DepthSplat geometry module was loaded from a foreign source tree")
    sample_image_grid = getattr(module, "sample_image_grid", None)
    get_world_rays = getattr(module, "get_world_rays", None)
    if not callable(sample_image_grid) or not callable(get_world_rays):
        raise RuntimeError("DepthSplat source geometry module lacks required functions")
    return sample_image_grid, get_world_rays, {
        "module": geometry_module_name,
        "path": source_path.relative_to(ROOT).as_posix(),
    }


def _context_decoder_inputs(context: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return only context cameras for a target-free decoder shape smoke."""

    required = ("extrinsics", "intrinsics", "near", "far")
    if any(not torch.is_tensor(context.get(name)) for name in required):
        raise RuntimeError("DepthSplat context decoder inputs are incomplete")
    inputs = {name: context[name] for name in required}
    if inputs["extrinsics"].ndim != 4 or inputs["intrinsics"].ndim != 4:
        raise RuntimeError("DepthSplat context decoder camera shapes are invalid")
    if inputs["near"].ndim != 2 or inputs["far"].ndim != 2:
        raise RuntimeError("DepthSplat context decoder bounds are invalid")
    return inputs


def _require_equivalent(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise RuntimeError(f"{label} returned an invalid equivalence report")
    if report.get("equivalent") is not True:
        raise RuntimeError(
            f"{label} is not equivalent to source-native attributes: {dict(report)}"
        )
    return dict(report)


def _require_bitwise_equivalent(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise RuntimeError(f"{label} returned an invalid bitwise report")
    if report.get("bitwise_equivalent") is not True:
        raise RuntimeError(f"{label} is not bitwise equivalent: {dict(report)}")
    return dict(report)


def _render_context_decoder_smoke(
    decoder: Any,
    packed: Any,
    *,
    gaussians_type: type[Any],
    context: Mapping[str, Any],
    image_shape: tuple[int, int],
) -> tuple[Any, torch.Tensor]:
    """Invoke the native decoder only after the materialized packet commits."""

    if not callable(getattr(decoder, "forward", None)):
        raise RuntimeError("DepthSplat smoke has no native decoder")
    gaussians = packed.as_single_batch(gaussians_type)
    for name in ("means", "covariances", "harmonics", "opacities"):
        value = getattr(gaussians, name, None)
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"DepthSplat materialized decoder input is invalid: {name}")
    camera = _context_decoder_inputs(context)
    with torch.no_grad():
        output = decoder.forward(
            gaussians,
            camera["extrinsics"],
            camera["intrinsics"],
            camera["near"],
            camera["far"],
            image_shape,
            depth_mode=None,
        )
    color = getattr(output, "color", None)
    if not torch.is_tensor(color) or not bool(torch.isfinite(color).all()):
        raise RuntimeError("DepthSplat native decoder did not emit finite context-camera color")
    expected = (
        context["image"].shape[0],
        context["image"].shape[1],
        3,
        image_shape[0],
        image_shape[1],
    )
    if tuple(color.shape) != expected:
        raise RuntimeError(
            "DepthSplat native decoder context-camera shape changed: "
            f"expected {expected}, got {tuple(color.shape)}"
        )
    return gaussians, color


def collect_depthsplat_l0_l1_execution_audit(
    *, sample_index: int, device: torch.device
) -> dict[str, Any]:
    """Execute a target-discarded sample-0 L0/L1 packet-to-decoder smoke."""

    if sample_index != SAMPLE_INDEX:
        raise ValueError("DepthSplat materializer smoke is predeclared for DL3DV sample index 0")
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    backend_contract = resolve_depthsplat_backend_contract(ROOT)
    backend_identity = freeze_depthsplat_backend_identity(backend_contract)
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    selection = resolve_claim_selection(MODEL, DATASET, ROOT)
    if (
        experiment.checkpoint.resolve() != backend_contract.checkpoint
        or selection.index_path.resolve() != backend_contract.evaluation_index
    ):
        raise RuntimeError("DepthSplat smoke application identity differs from its backend")
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
        sample_index=sample_index,
    )
    model.eval()
    context, native_target_rgb_loaded = _discard_target_before_context_transfer(
        batch, loaded_device
    )
    scene = str(batch["scene"][0])
    context_indices = [
        int(value) for value in context["index"][0].detach().to(device="cpu").tolist()
    ]
    del batch

    with strict_fp32_convolution_execution() as numerical_execution:
        execution = capture_depthsplat_native_execution(
            model.encoder, context, source_root=ROOT / "depthsplat"
        )
        if execution.routing_features is None or execution.routing_z_depths is None:
            raise RuntimeError("DepthSplat native execution did not retain target-free routing tensors")
        _views, _channels, height, width = execution.dense_raw_head.shape
        if height % TILE_SIZE or width % TILE_SIZE:
            raise RuntimeError("DepthSplat native image shape is not divisible by the L0/L1 tile size")
        plan = build_incremental_probe_first_plan(
            execution.routing_features,
            execution.routing_z_depths,
            height=height,
            width=width,
            tile_size=TILE_SIZE,
            feature_threshold=FEATURE_THRESHOLD,
            depth_threshold=DEPTH_THRESHOLD,
            decision_semantics=DECISION_SEMANTICS,
            l1_anchor_semantics=L1_ANCHOR_SEMANTICS,
        )
        initial_replay = replay_depthsplat_selected_head(
            model.encoder.gaussian_head,
            execution.gaussian_head_input,
            execution.dense_raw_head,
            plan.selection_mask,
            native_full_mask=plan.full_mask,
        )
        if initial_replay.equivalence.get("equivalent") is not True:
            raise RuntimeError("DepthSplat initial selected raw-head replay is not equivalent")
        initial_packet = build_depthsplat_sparse_raw_packet(execution, initial_replay)
        consumer = DepthSplatPackedGaussianConsumer(model.encoder.gaussian_adapter)
        initial_packed = consumer.convert(
            initial_packet,
            image_shape=(height, width),
            native_execution=execution,
            native_full_mask=plan.full_mask,
        )
        initial_adapter_equivalence = _require_equivalent(
            compare_depthsplat_packed_to_dense(initial_packed, execution.dense_gaussians),
            "DepthSplat initial selected Adapter packet",
        )
        initial_full_passthrough = _require_bitwise_equivalent(
            compare_depthsplat_full_passthrough_to_dense_bitwise(
                initial_packed, execution.dense_gaussians, plan.full_mask
            ),
            "DepthSplat initial Full attributes",
        )
        sample_image_grid, get_world_rays, geometry_identity = _source_geometry_functions(
            model.encoder
        )
        preflight = preflight_depthsplat_l0_l1_materialization(
            initial_packet,
            initial_packed,
            plan,
            execution.routing_features,
            execution.routing_z_depths,
            source_sample_image_grid=sample_image_grid,
            source_get_world_rays=get_world_rays,
        )
        final_route = resolve_depthsplat_compact_final_route(plan, preflight)
        producer_replay = replay_depthsplat_selected_head(
            model.encoder.gaussian_head,
            execution.gaussian_head_input,
            execution.dense_raw_head,
            final_route.raw_head_request_mask,
            native_full_mask=final_route.full_passthrough_mask,
        )
        if producer_replay.equivalence.get("equivalent") is not True:
            raise RuntimeError("DepthSplat final raw-head replay is not equivalent")
        producer_packet = build_depthsplat_sparse_raw_packet(execution, producer_replay)
        final_packet = subset_depthsplat_sparse_raw_packet(
            producer_packet, final_route.selected_output_mask
        )
        final_packed = consumer.convert(
            final_packet,
            image_shape=(height, width),
            native_execution=execution,
            native_full_mask=final_route.full_passthrough_mask,
        )
        final_adapter_equivalence = _require_equivalent(
            compare_depthsplat_packed_to_dense(final_packed, execution.dense_gaussians),
            "DepthSplat final selected Adapter packet",
        )
        final_full_passthrough = _require_bitwise_equivalent(
            compare_depthsplat_full_passthrough_to_dense_bitwise(
                final_packed,
                execution.dense_gaussians,
                final_route.full_passthrough_mask,
            ),
            "DepthSplat final Full attributes",
        )
        materialized = apply_depthsplat_compact_l0_l1_materialization(
            final_packed, preflight, final_route
        )
        materialized_full_passthrough = _require_bitwise_equivalent(
            compare_depthsplat_full_passthrough_to_dense_bitwise(
                materialized,
                execution.dense_gaussians,
                final_route.full_passthrough_mask,
            ),
            "DepthSplat materialized Full attributes",
        )
        decoder_gaussians, rendered_color = _render_context_decoder_smoke(
            model.decoder,
            materialized,
            gaussians_type=type(execution.dense_gaussians),
            context=context,
            image_shape=(height, width),
        )

    compact_updates = int(preflight.update_dense_slots.numel())
    final_descriptor_count = int(materialized.dense_slots.numel())
    if final_descriptor_count != int(final_route.selected_output_mask.sum().item()):
        raise RuntimeError("DepthSplat final materialized packet does not match its route")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "status": "PASS",
        "paper_result_eligible": False,
        "formal_target_free_sidecar_available": False,
        "formal_target_free_sidecar_used": False,
        "model": MODEL,
        "dataset": DATASET,
        "sample_index": sample_index,
        "scene": scene,
        "context_indices": context_indices,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": native_target_rgb_loaded,
            "pixel_values_accessed": False,
            "removed_before_context_device_transfer": True,
            "target_mapping_discarded_before_context_device_transfer": True,
            "target_camera_metadata_read": False,
            "target_index_read": False,
            "formal_context_only_sidecar_used": False,
        },
        "development_scope": {
            "name": "source-bound-target-discarded-materializer-smoke",
            "formal_target_free_audit": False,
            "quality_metrics_computed": False,
            "target_view_rendered": False,
            "context_camera_decoder_smoke_only": True,
            "nonzero_merge_anchor_count": compact_updates,
            "nonzero_merge_executed": compact_updates > 0,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
            "timing_claim": False,
        },
        "execution_boundary": {
            "native_dataloader_target_mapping_constructed": True,
            "native_dense_depth_predictor_executed": True,
            "native_dense_gaussian_regressor_executed": True,
            "source_selected_raw_head_replays_executed": 2,
            "initial_selected_native_rgb_adapter_executed": True,
            "final_selected_native_rgb_adapter_executed": True,
            "nonzero_l0_l1_materializer_executed": True,
            "native_decoder_context_camera_smoke_executed": True,
            "native_decoder_target_camera_executed": False,
            "renderer_executed": True,
            "quality_metrics_computed": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        },
        "route_plan": plan.events,
        "materialization_preflight": preflight.events,
        "final_route": final_route.events,
        "native_execution": execution.events,
        "initial_selected_head": {
            "equivalence": initial_replay.equivalence,
            "events": initial_replay.events,
        },
        "producer_selected_head": {
            "equivalence": producer_replay.equivalence,
            "events": producer_replay.events,
        },
        "geometry_source": geometry_identity,
        "initial_packet": {
            "descriptor_count": int(initial_packed.dense_slots.numel()),
            "source_trace_sha256": initial_packed.source_trace_sha256,
            "attribute_binding_sha256": initial_packed.attribute_binding_sha256,
            "native_adapter_equivalence": initial_adapter_equivalence,
            "full_passthrough_bitwise": initial_full_passthrough,
        },
        "final_packet": {
            "producer_descriptor_count": int(producer_packet.dense_slots.numel()),
            "descriptor_count": final_descriptor_count,
            "source_trace_sha256": final_packed.source_trace_sha256,
            "attribute_binding_sha256": final_packed.attribute_binding_sha256,
            "native_adapter_equivalence_before_materialization": final_adapter_equivalence,
            "full_passthrough_bitwise_before_materialization": final_full_passthrough,
            "producer_only_prefetch_descriptor_count": final_route.events[
                "producer_only_prefetch_descriptor_count"
            ],
            "skipped_s3_attributes_accessed": False,
            "nonzero_direct_deletion": False,
        },
        "materialized_packet": {
            "descriptor_count": final_descriptor_count,
            "source_trace_sha256": materialized.source_trace_sha256,
            "attribute_binding_sha256": materialized.attribute_binding_sha256,
            "full_passthrough_bitwise": materialized_full_passthrough,
        },
        "decoder_smoke": {
            "camera_source": "context-only",
            "color_shape": list(rendered_color.shape),
            "finite": True,
            "decoder_gaussian_count": int(decoder_gaussians.means.shape[1]),
        },
        "backend_identity": backend_identity,
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "runner": {
            "path": Path(__file__).relative_to(ROOT).as_posix(),
            "sha256": cached_sha256_file(Path(__file__)),
        },
        "source": source_identity(),
        "numerical_execution": numerical_execution,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("DepthSplat materializer smoke requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from scripts.result_record import portable_command, write_result

    try:
        record = collect_depthsplat_l0_l1_execution_audit(
            sample_index=args.sample_index, device=device
        )
        exit_code = 0
    except Exception as exc:
        record = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "kind": AUDIT_KIND,
            "status": "FAILED",
            "paper_result_eligible": False,
            "formal_target_free_sidecar_available": False,
            "formal_target_free_sidecar_used": False,
            "model": MODEL,
            "dataset": DATASET,
            "sample_index": args.sample_index,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
        exit_code = 1
    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
