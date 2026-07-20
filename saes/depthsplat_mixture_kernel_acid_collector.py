"""CUDA-only, target-free ACID 24/8 collection for the DepthSplat v3 risk guard.

The collector uses the strict numerical closure default only to expose every
finite candidate risk safely.  A positive threshold is never chosen here: it
is frozen later from the train-only traces by the sibling calibration module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from integration.acid_joint_context import load_acid_joint_context
from integration.acid_joint_model_context import prepare_acid_joint_model_context
from saes.depthsplat_acid_disjoint_calibration import (
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
    canonical_sha256,
    resolve_depthsplat_acid_binding,
)
from saes.depthsplat_l0_l1_materializer import (
    DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
    apply_depthsplat_compact_l0_l1_materialization,
    preflight_depthsplat_l0_l1_materialization,
    resolve_depthsplat_compact_final_route,
)
from saes.depthsplat_literal_t4_acid_calibration import (
    build_selected_head_replay_fallback_summary,
)
from saes.depthsplat_literal_t4_acid_collector import (
    LiteralT4CollectionRuntime,
    _copy_scalar_trace_evidence,
    _require_bitwise_equivalent,
    _require_equivalent,
    _require_prepared_context,
    _require_target_free_context,
    _source_geometry_functions,
    load_literal_t4_collection_runtime,
)
from saes.depthsplat_mixture_kernel_acid_calibration import (
    HOLDOUT_SPLIT,
    KERNEL_RISK_RECORD_NAME,
    TRAIN_SPLIT,
    _COLLECTOR_SOURCE_FILES,
    build_kernel_risk_record,
    build_kernel_risk_trace_artifact,
    build_mixture_kernel_risk_application,
    kernel_risk_observations_from_preflight_trace,
    load_frozen_kernel_risk_threshold,
    mixture_kernel_risk_v3_profile,
)
from saes.depthsplat_selected_output import (
    DepthSplatPackedGaussianConsumer,
    build_depthsplat_sparse_raw_packet,
    capture_depthsplat_native_execution,
    compare_depthsplat_full_passthrough_to_dense_bitwise,
    compare_depthsplat_packed_to_dense,
    replay_depthsplat_selected_head,
    subset_depthsplat_sparse_raw_packet,
)
from saes.probe_first_schedule import (
    DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
    build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution


ROOT = Path(__file__).resolve().parents[1]
MODEL = "depthsplat"
DATASET = "dl3dv"
TILE_SIZE = 4
FEATURE_THRESHOLD = 0.20
DEPTH_THRESHOLD = 0.10
TRACE_DIRECTORY_NAME = "scenes"
REQUIRED_COLLECTOR_SOURCE_FILES = frozenset(_COLLECTOR_SOURCE_FILES)
_ACCESS = {
    "target_mapping_present": False,
    "target_rgb_accessed": False,
    "target_camera_metadata_accessed": False,
    "target_index_accessed": False,
    "skipped_s3_attributes_accessed": False,
}


def _require_cuda_device(device: torch.device) -> torch.device:
    if not isinstance(device, torch.device) or device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("DepthSplat kernel-risk ACID collection requires a CUDA device")
    return device


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"kernel-risk collector has invalid {label} SHA256")
    return value


def _require_complete_collector_source(application: Mapping[str, Any]) -> None:
    source = application.get("collector_source") if isinstance(application, Mapping) else None
    files = source.get("files") if isinstance(source, Mapping) else None
    paths = {
        entry.get("path")
        for entry in files
        if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
    } if isinstance(files, list) else set()
    if paths != REQUIRED_COLLECTOR_SOURCE_FILES:
        raise RuntimeError("kernel-risk collector identity is incomplete")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
    except BaseException:
        try:
            Path(path).unlink(missing_ok=True)
        finally:
            raise


def _trace_relative_path(*, split: str, sample_index: int, scene: str) -> str:
    if split not in {TRAIN_SPLIT, HOLDOUT_SPLIT} or sample_index < 0 or not scene:
        raise ValueError("kernel-risk trace identity is invalid")
    digest = hashlib.sha256(scene.encode("utf-8")).hexdigest()[:16]
    return f"{TRACE_DIRECTORY_NAME}/{split}/{sample_index:03d}-{digest}.json"


def load_mixture_kernel_risk_collection_runtime(
    *, device: torch.device
) -> LiteralT4CollectionRuntime:
    """Reuse the native encoder-only DepthSplat runtime from literal T=4."""

    return load_literal_t4_collection_runtime(device=_require_cuda_device(device))


def collect_mixture_kernel_risk_native_scene_observation(
    *,
    runtime: LiteralT4CollectionRuntime,
    materialization_root: Path,
    plan_path: Path,
    split: str,
    sample_index: int,
    scene: str,
    context_loader: Callable[..., Any] = load_acid_joint_context,
) -> dict[str, Any]:
    """Collect one v3 source-only risk trace; unscorable candidates stay Full."""

    if not isinstance(runtime, LiteralT4CollectionRuntime):
        raise TypeError("kernel-risk collector requires a native runtime")
    raw: Any | None = None
    context: Mapping[str, Any] | None = None
    preparation: Mapping[str, Any] | None = None
    execution: Any | None = None
    plan: Any | None = None
    initial_replay: Any | None = None
    initial_packet: Any | None = None
    initial_packed: Any | None = None
    consumer: Any | None = None
    preflight: Any | None = None
    final_route: Any | None = None
    producer_replay: Any | None = None
    producer_packet: Any | None = None
    final_packet: Any | None = None
    final_packed: Any | None = None
    materialized: Any | None = None
    sample_image_grid: Any | None = None
    get_world_rays: Any | None = None
    aggregate: Mapping[str, Any] | None = None
    update_binding: Mapping[str, Any] | None = None
    full_binding: Any | None = None
    full_bitwise: Mapping[str, Any] | None = None
    trace: list[dict[str, Any]] | None = None
    observation: dict[str, Any] | None = None
    try:
        with torch.no_grad():
            raw = context_loader(
                materialization_root=Path(materialization_root),
                split=split,
                sample_index=sample_index,
                plan_path=Path(plan_path),
            )
            context, preparation = prepare_acid_joint_model_context(
                raw,
                dataset_cfg=runtime.bundle.config.dataset,
                encoder_cfg=runtime.bundle.config.model.encoder,
                device=runtime.bundle.device,
            )
            _require_target_free_context(
                raw,
                preparation,
                split=split,
                sample_index=sample_index,
                expected_scene=scene,
            )
            _require_prepared_context(context)
            with strict_fp32_convolution_execution():
                execution = capture_depthsplat_native_execution(
                    runtime.bundle.encoder, context, source_root=ROOT / "depthsplat"
                )
                if execution.routing_features is None or execution.routing_z_depths is None:
                    raise RuntimeError("kernel-risk collector has no routing tensors")
                _views, _channels, height, width = execution.dense_raw_head.shape
                if height % TILE_SIZE or width % TILE_SIZE:
                    raise RuntimeError("kernel-risk collector image shape is not divisible by four")
                plan = build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan(
                    execution.routing_features,
                    execution.routing_z_depths,
                    height=height,
                    width=width,
                    feature_threshold=FEATURE_THRESHOLD,
                    depth_threshold=DEPTH_THRESHOLD,
                )
                if (
                    plan.events.get("contract_version")
                    != DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT
                ):
                    raise RuntimeError("kernel-risk collector route plan changed")
                initial_replay = replay_depthsplat_selected_head(
                    runtime.bundle.encoder.gaussian_head,
                    execution.gaussian_head_input,
                    execution.dense_raw_head,
                    plan.selection_mask,
                    native_full_mask=plan.full_mask,
                )
                _require_equivalent(initial_replay.equivalence, "initial selected head")
                initial_packet = build_depthsplat_sparse_raw_packet(execution, initial_replay)
                consumer = DepthSplatPackedGaussianConsumer(runtime.bundle.encoder.gaussian_adapter)
                initial_packed = consumer.convert(
                    initial_packet,
                    image_shape=(height, width),
                    native_execution=execution,
                    native_full_mask=plan.full_mask,
                )
                _require_equivalent(
                    compare_depthsplat_packed_to_dense(initial_packed, execution.dense_gaussians),
                    "initial Adapter packet",
                )
                sample_image_grid, get_world_rays = _source_geometry_functions(runtime.bundle.encoder)
                # No positive threshold is supplied here. The strict numerical default
                # promotes unsafe/unscorable candidates Full while preserving their
                # finite S/R-certified risk observations in the trace.
                preflight = preflight_depthsplat_l0_l1_materialization(
                    initial_packet,
                    initial_packed,
                    plan,
                    execution.routing_features,
                    execution.routing_z_depths,
                    source_sample_image_grid=sample_image_grid,
                    source_get_world_rays=get_world_rays,
                    maximum_coverage_covariance_scale=1.0,
                    execution_profile=DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
                )
                profile = mixture_kernel_risk_v3_profile()
                if (
                    preflight.events.get("execution_profile")
                    != profile["materialization_profile"]
                    or preflight.events.get("maximum_coverage_covariance_scale") != 1.0
                    or preflight.events.get("target_rgb_accessed") is not False
                    or preflight.events.get("target_camera_accessed_before_commit") is not False
                    or preflight.events.get("mixture_kernel_closure_guard_policy")
                    != profile["kernel_closure_guard_policy"]
                ):
                    raise RuntimeError("kernel-risk collector preflight profile changed")
                aggregate = preflight.events.get("mixture_kernel_closure_aggregate")
                if not isinstance(aggregate, Mapping):
                    raise RuntimeError("kernel-risk collector has no closure aggregate")
                final_route = resolve_depthsplat_compact_final_route(plan, preflight)
                producer_replay = replay_depthsplat_selected_head(
                    runtime.bundle.encoder.gaussian_head,
                    execution.gaussian_head_input,
                    execution.dense_raw_head,
                    final_route.raw_head_request_mask,
                    native_full_mask=final_route.full_passthrough_mask,
                )
                _require_equivalent(producer_replay.equivalence, "final selected head")
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
                _require_equivalent(
                    compare_depthsplat_packed_to_dense(final_packed, execution.dense_gaussians),
                    "final Adapter packet",
                )
                full_bitwise = _require_bitwise_equivalent(
                    compare_depthsplat_full_passthrough_to_dense_bitwise(
                        final_packed,
                        execution.dense_gaussians,
                        final_route.full_passthrough_mask,
                    ),
                    "final Full attributes",
                )
                materialized = apply_depthsplat_compact_l0_l1_materialization(
                    final_packed, preflight, final_route
                )
                _require_bitwise_equivalent(
                    compare_depthsplat_full_passthrough_to_dense_bitwise(
                        materialized,
                        execution.dense_gaussians,
                        final_route.full_passthrough_mask,
                    ),
                    "materialized Full attributes",
                )

            update_binding = preflight.events.get("update_binding")
            if not isinstance(update_binding, Mapping):
                raise RuntimeError("kernel-risk collector preflight has no update binding")
            full_binding = final_packed.source_trace.get("native_full_adapter_attribute_binding_sha256")
            trace = [
                _copy_scalar_trace_evidence(entry, label="preflight tile trace")
                for entry in preflight.tile_trace
            ]
            aggregate_copy = _copy_scalar_trace_evidence(
                aggregate, label="mixture kernel-closure aggregate"
            )
            if not isinstance(aggregate_copy, Mapping):
                raise RuntimeError("kernel-risk collector aggregate copy changed")
            observations = kernel_risk_observations_from_preflight_trace(trace)
            if split == TRAIN_SPLIT and not observations:
                raise RuntimeError("kernel-risk collector scene has no finite S/R-certified candidate")
            profile = mixture_kernel_risk_v3_profile()
            observation = {
                "scene": scene,
                "kernel_risk_observations": observations,
                "preflight_tile_trace": trace,
                "mixture_kernel_closure_aggregate": dict(aggregate_copy),
                "evidence": {
                    "profile_sha256": canonical_sha256(profile),
                    "route_plan_config_sha256": profile["route_plan_config_sha256"],
                    "native_execution_sha256": _require_sha256(
                        execution.events.get("native_execution_sha256"), "native execution"
                    ),
                    "initial_attribute_binding_sha256": _require_sha256(
                        initial_packed.attribute_binding_sha256, "initial Adapter binding"
                    ),
                    "final_selected_attribute_binding_sha256": _require_sha256(
                        final_packed.attribute_binding_sha256, "final selected Adapter binding"
                    ),
                    "selected_head_replay_fallback_summary": build_selected_head_replay_fallback_summary(
                        initial_events=initial_replay.events,
                        final_events=producer_replay.events,
                    ),
                    "materialized_attribute_binding_sha256": _require_sha256(
                        materialized.attribute_binding_sha256, "materialized Adapter binding"
                    ),
                    "full_passthrough_mask_sha256": _require_sha256(
                        final_route.events.get("full_passthrough_mask_sha256"),
                        "Full passthrough mask",
                    ),
                    "full_attribute_binding_sha256": _require_sha256(
                        full_binding, "Full attribute binding"
                    ),
                    "full_attributes_bitwise_native": full_bitwise["bitwise_equivalent"],
                    "coverage_certificate": preflight.events.get("coverage_certificate"),
                    "coverage_certificate_sha256": _require_sha256(
                        preflight.events.get("coverage_certificate_sha256"),
                        "coverage certificate",
                    ),
                    "accepted_update_slots_sha256": _require_sha256(
                        update_binding.get("slots_sha256"), "accepted update slots"
                    ),
                    "accepted_update_slot_count": int(preflight.update_dense_slots.numel()),
                    "mixture_kernel_closure_aggregate": dict(aggregate_copy),
                    "mixture_kernel_closure_aggregate_sha256": canonical_sha256(aggregate_copy),
                    "mixture_kernel_closure_trace_sha256": canonical_sha256(observations),
                    "source_nonprobe_s3_attribute_reads": preflight.events.get(
                        "source_nonprobe_s3_attribute_reads"
                    ),
                    "nonzero_merge_applied": int(preflight.update_dense_slots.numel()) > 0,
                    "renderer_executed": False,
                    "quality_metrics_computed": False,
                    "whole_pipeline_s2_s3_sparse_execution_verified": False,
                    "global_s2_s3_savings_claimed": False,
                },
                "access": dict(_ACCESS),
            }
    finally:
        materialized = None
        final_packed = None
        final_packet = None
        producer_packet = None
        producer_replay = None
        final_route = None
        preflight = None
        initial_packed = None
        initial_packet = None
        initial_replay = None
        consumer = None
        plan = None
        execution = None
        sample_image_grid = None
        get_world_rays = None
        aggregate = None
        update_binding = None
        full_binding = None
        full_bitwise = None
        trace = None
        preparation = None
        context = None
        raw = None
        if runtime.bundle.device.type == "cuda":
            torch.cuda.empty_cache()
    if observation is None:
        raise RuntimeError("kernel-risk collector did not produce scene evidence")
    return observation


def _scene_record_from_observation(
    *,
    observation: Mapping[str, Any],
    split: str,
    sample_index: int,
    expected_scene: str,
    output_directory: Path,
) -> dict[str, Any]:
    required = {
        "scene",
        "kernel_risk_observations",
        "preflight_tile_trace",
        "mixture_kernel_closure_aggregate",
        "evidence",
        "access",
    }
    if not isinstance(observation, Mapping) or set(observation) != required:
        raise ValueError("kernel-risk collector scene observation has an invalid schema")
    if observation.get("scene") != expected_scene or observation.get("access") != _ACCESS:
        raise ValueError("kernel-risk collector scene identity changed")
    profile = mixture_kernel_risk_v3_profile()
    evidence = observation.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("kernel-risk collector scene evidence is invalid")
    required_evidence = {
        "profile_sha256",
        "route_plan_config_sha256",
        "native_execution_sha256",
        "initial_attribute_binding_sha256",
        "final_selected_attribute_binding_sha256",
        "selected_head_replay_fallback_summary",
        "materialized_attribute_binding_sha256",
        "full_passthrough_mask_sha256",
        "full_attribute_binding_sha256",
        "full_attributes_bitwise_native",
        "coverage_certificate",
        "coverage_certificate_sha256",
        "accepted_update_slots_sha256",
        "accepted_update_slot_count",
        "mixture_kernel_closure_aggregate",
        "mixture_kernel_closure_aggregate_sha256",
        "mixture_kernel_closure_trace_sha256",
        "source_nonprobe_s3_attribute_reads",
        "nonzero_merge_applied",
        "renderer_executed",
        "quality_metrics_computed",
        "whole_pipeline_s2_s3_sparse_execution_verified",
        "global_s2_s3_savings_claimed",
    }
    if set(evidence) != required_evidence:
        raise ValueError("kernel-risk collector scene evidence schema changed")
    artifact = build_kernel_risk_trace_artifact(
        scene=expected_scene,
        split=split,
        preflight_tile_trace=observation["preflight_tile_trace"],
        mixture_kernel_closure_aggregate=observation["mixture_kernel_closure_aggregate"],
    )
    observations = artifact["kernel_risk_observations"]
    if observation.get("kernel_risk_observations") != observations:
        raise ValueError("kernel-risk collector observation extraction changed")
    relative_path = _trace_relative_path(
        split=split, sample_index=sample_index, scene=expected_scene
    )
    _write_new_json(output_directory / relative_path, artifact)
    aggregate = artifact["mixture_kernel_closure_aggregate"]
    final_evidence = {
        **dict(evidence),
        "mixture_kernel_closure_aggregate": aggregate,
        "mixture_kernel_closure_aggregate_sha256": canonical_sha256(aggregate),
        "mixture_kernel_closure_trace_sha256": canonical_sha256(observations),
        "trace_artifact": {
            "relative_path": relative_path,
            "sha256": artifact["sha256"],
            "preflight_tile_trace_sha256": canonical_sha256(
                artifact["preflight_tile_trace"]
            ),
            "kernel_risk_observations_sha256": canonical_sha256(observations),
        },
    }
    return {
        "scene": expected_scene,
        "kernel_risk_observations": observations,
        "evidence": final_evidence,
        "access": dict(_ACCESS),
    }


def _collect_mixture_kernel_risk_calibration_with_dependencies(
    *,
    output_directory: Path,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
    scene_worker: Callable[..., Mapping[str, Any]],
    binding: Mapping[str, Any],
    application: Mapping[str, Any],
    runtime: LiteralT4CollectionRuntime | None = None,
    record_loader: Callable[..., Mapping[str, Any]] = load_frozen_kernel_risk_threshold,
) -> dict[str, Any]:
    """Private dependency-injected orchestration used by CPU contract tests."""

    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"kernel-risk calibration output directory must be new: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    _require_complete_collector_source(application)
    if not callable(scene_worker) or not callable(record_loader):
        raise TypeError("kernel-risk collector requires callable worker and record loader")
    records: dict[str, list[dict[str, Any]]] = {TRAIN_SPLIT: [], HOLDOUT_SPLIT: []}
    for split in (TRAIN_SPLIT, HOLDOUT_SPLIT):
        split_binding = binding.get("splits", {}).get(split) if isinstance(binding, Mapping) else None
        scenes = split_binding.get("scenes") if isinstance(split_binding, Mapping) else None
        if not isinstance(scenes, list):
            raise ValueError(f"kernel-risk collector has no {split} scene list")
        for sample_index, scene in enumerate(scenes):
            kwargs: dict[str, Any] = {
                "materialization_root": Path(materialization_root),
                "plan_path": Path(plan_path),
                "split": split,
                "sample_index": sample_index,
                "scene": scene,
            }
            if runtime is not None:
                kwargs["runtime"] = runtime
            records[split].append(
                _scene_record_from_observation(
                    observation=scene_worker(**kwargs),
                    split=split,
                    sample_index=sample_index,
                    expected_scene=scene,
                    output_directory=destination,
                )
            )
    record = build_kernel_risk_record(
        binding=binding,
        application=application,
        profile=mixture_kernel_risk_v3_profile(),
        train_scene_records=records[TRAIN_SPLIT],
        holdout_scene_records=records[HOLDOUT_SPLIT],
    )
    record_path = destination / KERNEL_RISK_RECORD_NAME
    _write_new_json(record_path, record)
    loaded = record_loader(
        record_path,
        root=ROOT,
        plan_path=Path(plan_path),
        materialization_root=Path(materialization_root),
    )
    if (
        not isinstance(loaded, Mapping)
        or loaded.get("sha256") != record["sha256"]
        or loaded.get("profile_sha256") != canonical_sha256(mixture_kernel_risk_v3_profile())
    ):
        raise RuntimeError("kernel-risk collector live reload changed its frozen record")
    return {
        "record_path": record_path,
        "record": dict(loaded),
        "scene_trace_count": sum(len(items) for items in records.values()),
    }


def collect_mixture_kernel_risk_calibration(
    *,
    device: torch.device,
    output_directory: Path,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Run the production encoder-only ACID 24/8 v3 risk collection."""

    device = _require_cuda_device(device)
    runtime = load_mixture_kernel_risk_collection_runtime(device=device)
    binding = resolve_depthsplat_acid_binding(
        plan_path=Path(plan_path), materialization_root=Path(materialization_root)
    )
    application = build_mixture_kernel_risk_application(
        backend_identity=runtime.backend_identity, root=ROOT
    )
    return _collect_mixture_kernel_risk_calibration_with_dependencies(
        output_directory=output_directory,
        plan_path=plan_path,
        materialization_root=materialization_root,
        scene_worker=collect_mixture_kernel_risk_native_scene_observation,
        runtime=runtime,
        binding=binding,
        application=application,
    )


__all__ = [
    "DATASET",
    "DEPTH_THRESHOLD",
    "FEATURE_THRESHOLD",
    "KERNEL_RISK_RECORD_NAME",
    "MODEL",
    "REQUIRED_COLLECTOR_SOURCE_FILES",
    "TILE_SIZE",
    "collect_mixture_kernel_risk_calibration",
    "collect_mixture_kernel_risk_native_scene_observation",
    "load_mixture_kernel_risk_collection_runtime",
]
