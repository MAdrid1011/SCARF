"""Native, target-free ACID collection for the literal DepthSplat T=4 V16 guard.

The frozen-record schema lives in :mod:`saes.depthsplat_literal_t4_acid_calibration`.
This module owns the CUDA-only observation path: it opens an ACID context-only
sidecar, executes the encoder and selected native Adapter path, persists each
scene's complete preflight trace, then freezes and immediately revalidates the
record.  It never constructs a decoder, renderer, target mapping, quality
metric, or timing result.

The private orchestration helper accepts injected dependencies so its file
layout, record ordering, and live-reload boundary can be covered on CPU.  The
public collector and CLI always take the CUDA-native path.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from integration.acid_joint_context import load_acid_joint_context
from integration.acid_joint_model_context import prepare_acid_joint_model_context
from integration.model_loader import create_model_loader
from saes.depthsplat_acid_disjoint_calibration import (
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
    canonical_sha256,
    resolve_depthsplat_acid_binding,
)
from saes.depthsplat_backend import (
    freeze_depthsplat_backend_identity,
    resolve_depthsplat_backend_contract,
)
from saes.depthsplat_l0_l1_materializer import (
    DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
    apply_depthsplat_compact_l0_l1_materialization,
    preflight_depthsplat_l0_l1_materialization,
    resolve_depthsplat_compact_final_route,
)
from saes.depthsplat_literal_t4_acid_calibration import (
    HOLDOUT_SPLIT,
    LITERAL_T4_MOMENT_CERTIFICATE,
    LITERAL_T4_MATERIALIZATION_PROFILE,
    LITERAL_T4_PROFILE_ID,
    LITERAL_T4_PROFILE_SCHEMA,
    LITERAL_T4_RISK_METRIC,
    TRAIN_SPLIT,
    build_literal_t4_application,
    build_selected_head_replay_fallback_summary,
    build_v16t4_record,
    build_v16t4_trace_artifact,
    literal_t4_profile,
    literal_t4_profile_sha256,
    load_frozen_v16t4_threshold,
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
    LITERAL_PAPER_T4_PLAN_CONTRACT,
    build_literal_paper_t4_probe_first_plan,
)
from scripts.ae_config import resolve_claim_selection, resolve_experiment
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution


ROOT = Path(__file__).resolve().parents[1]
MODEL = "depthsplat"
DATASET = "dl3dv"
TILE_SIZE = 4
FEATURE_THRESHOLD = 0.20
DEPTH_THRESHOLD = 0.10
V16T4_RECORD_NAME = "v16t4.json"
TRACE_DIRECTORY_NAME = "scenes"

# Keep this list synchronized with the frozen record's collector-source
# identity.  The explicit check below prevents a collection from claiming a
# source identity that omits the executable worker or its direct boundaries.
REQUIRED_COLLECTOR_SOURCE_FILES = frozenset(
    {
        "saes/depthsplat_literal_t4_acid_calibration.py",
        "saes/depthsplat_literal_t4_acid_collector.py",
        "scripts/saes_depthsplat_literal_t4_acid_v16_calibration.py",
        "saes/depthsplat_acid_disjoint_calibration.py",
        "saes/depthsplat_backend.py",
        "saes/depthsplat_l0_l1_materializer.py",
        "saes/depthsplat_selected_output.py",
        "saes/probe_first_schedule.py",
        "saes/probe_layout.py",
        "saes/progressive_saes.py",
        "saes/selected_output_replay.py",
        "integration/model_loader.py",
        "integration/acid_joint_context.py",
        "integration/acid_joint_model_context.py",
        "scripts/ae_config.py",
        "scripts/saes_selected_output_replay_audit.py",
    }
)

_ACCESS = {
    "target_mapping_present": False,
    "target_rgb_accessed": False,
    "target_camera_metadata_accessed": False,
    "target_index_accessed": False,
    "skipped_s3_attributes_accessed": False,
}
_SCENE_BASE_EVIDENCE_KEYS = {
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
    "source_nonprobe_s3_attribute_reads",
    "nonzero_merge_applied",
    "renderer_executed",
    "quality_metrics_computed",
    "whole_pipeline_s2_s3_sparse_execution_verified",
    "global_s2_s3_savings_claimed",
}


@dataclass(frozen=True)
class LiteralT4CollectionRuntime:
    """One encoder-only DepthSplat runtime reused across the ACID 24/8 pass."""

    bundle: Any
    backend_identity: Mapping[str, Any]
    checkpoint_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"literal T=4 collector has no valid {label} SHA256")
    return value


def _require_cuda_device(device: torch.device) -> torch.device:
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise ValueError("literal T=4 ACID collection requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("literal T=4 ACID collection requires available CUDA")
    return device


def _require_complete_collector_source(application: Mapping[str, Any]) -> None:
    """Reject a collection if its record would omit its active worker code."""

    source = application.get("collector_source") if isinstance(application, Mapping) else None
    files = source.get("files") if isinstance(source, Mapping) else None
    paths = {
        entry.get("path")
        for entry in files
        if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
    } if isinstance(files, list) else set()
    missing = sorted(REQUIRED_COLLECTOR_SOURCE_FILES - paths)
    if missing:
        raise RuntimeError(
            "literal T=4 collector source identity is incomplete: " + ", ".join(missing)
        )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a JSON artifact once, leaving an interrupted artifact visible."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite literal T=4 collection artifact: {destination}"
        ) from error


def _trace_relative_path(*, split: str, sample_index: int, scene: str) -> str:
    if split not in (TRAIN_SPLIT, HOLDOUT_SPLIT):
        raise ValueError("literal T=4 trace split is invalid")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("literal T=4 trace sample index is invalid")
    if not isinstance(scene, str) or not scene:
        raise ValueError("literal T=4 trace scene is invalid")
    digest = hashlib.sha256(scene.encode("utf-8")).hexdigest()[:16]
    return f"{TRACE_DIRECTORY_NAME}/{split}/{sample_index:03d}-{digest}.json"


def _require_target_free_context(
    raw: Any,
    preparation: Mapping[str, Any],
    *,
    split: str,
    sample_index: int,
    expected_scene: str,
) -> None:
    context = getattr(raw, "context", None)
    identity = getattr(raw, "identity", None)
    if (
        not isinstance(context, Mapping)
        or set(context) != {"image", "extrinsics", "intrinsics", "index"}
        or not isinstance(identity, Mapping)
        or identity.get("split") != split
        or identity.get("sample_index") != sample_index
        or identity.get("scene") != expected_scene
        or any(
            identity.get(key) is not False
            for key in (
                "target_rgb_accessed",
                "target_camera_metadata_accessed",
                "target_index_accessed",
                "teacher_artifact_accessed",
                "expected_results_accessed",
            )
        )
        or not isinstance(preparation, Mapping)
        or any(
            preparation.get(key) is not False
            for key in (
                "target_mapping_present",
                "target_rgb_accessed",
                "target_camera_metadata_accessed",
                "target_index_accessed",
            )
        )
    ):
        raise RuntimeError("literal T=4 collector crossed its target-free context boundary")


def _require_prepared_context(context: Mapping[str, Any]) -> None:
    if set(context) != {"image", "extrinsics", "intrinsics", "index", "near", "far"}:
        raise RuntimeError("literal T=4 collector context schema changed")
    if any(not torch.is_tensor(value) for value in context.values()):
        raise RuntimeError("literal T=4 collector context contains a non-tensor")


def _source_geometry_functions(encoder: Any) -> tuple[Any, Any]:
    """Resolve geometry from the loaded native DepthSplat namespace only."""

    module_name = type(encoder).__module__
    if ".model." not in module_name:
        raise RuntimeError("literal T=4 collector encoder module changed")
    geometry_module_name = module_name.split(".model.", 1)[0] + ".geometry.projection"
    module = sys.modules.get(geometry_module_name)
    source = getattr(module, "__file__", None)
    source_root = (ROOT / "depthsplat" / "src").resolve()
    source_path = Path(source).resolve() if isinstance(source, str) else None
    sample_image_grid = getattr(module, "sample_image_grid", None)
    get_world_rays = getattr(module, "get_world_rays", None)
    if (
        source_path is None
        or source_root not in source_path.parents
        or not callable(sample_image_grid)
        or not callable(get_world_rays)
    ):
        raise RuntimeError("literal T=4 collector native geometry source changed")
    return sample_image_grid, get_world_rays


def _require_equivalent(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(report, Mapping) or report.get("equivalent") is not True:
        raise RuntimeError(f"literal T=4 collector {label} is not native equivalent")
    return dict(report)


def _require_bitwise_equivalent(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(report, Mapping) or report.get("bitwise_equivalent") is not True:
        raise RuntimeError(f"literal T=4 collector {label} is not bitwise native")
    return dict(report)


def _copy_scalar_trace_evidence(value: Any, *, label: str) -> Any:
    """Detach persisted trace evidence from live native tensor packets."""

    if torch.is_tensor(value):
        raise RuntimeError(f"literal T=4 collector {label} retained a tensor")
    if isinstance(value, Mapping):
        return {
            key: _copy_scalar_trace_evidence(item, label=label)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_scalar_trace_evidence(item, label=label) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_scalar_trace_evidence(item, label=label) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RuntimeError(f"literal T=4 collector {label} is not scalar trace evidence")


def load_literal_t4_collection_runtime(*, device: torch.device) -> LiteralT4CollectionRuntime:
    """Load exactly the encoder required by the literal ACID observer."""

    device = _require_cuda_device(device)
    backend_contract = resolve_depthsplat_backend_contract(ROOT)
    backend_identity = freeze_depthsplat_backend_identity(backend_contract)
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    selection = resolve_claim_selection(MODEL, DATASET, ROOT)
    if (
        experiment.model != MODEL
        or experiment.dataset != DATASET
        or experiment.experiment != backend_contract.experiment
        or Path(experiment.checkpoint).resolve() != Path(backend_contract.checkpoint).resolve()
        or Path(selection.index_path).resolve() != Path(backend_contract.evaluation_index).resolve()
    ):
        raise RuntimeError("literal T=4 collector application identity changed")
    checkpoint = Path(experiment.checkpoint)
    checkpoint_sha256 = _sha256_file(checkpoint)
    loader = create_model_loader(MODEL)
    bundle = loader.load_model(
        str(checkpoint),
        device=device,
        experiment_name=experiment.experiment,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        hydra_overrides=experiment.hydra_overrides,
        encoder_only=True,
    )
    if getattr(bundle, "decoder", None) is not None:
        raise RuntimeError("literal T=4 collector unexpectedly constructed a decoder")
    if _sha256_file(checkpoint) != checkpoint_sha256:
        raise RuntimeError("literal T=4 collector checkpoint changed during loading")
    if getattr(bundle, "encoder", None) is None or getattr(bundle, "model", None) is None:
        raise RuntimeError("literal T=4 collector loader returned an incomplete bundle")
    bundle.model.eval()
    return LiteralT4CollectionRuntime(
        bundle=bundle,
        backend_identity=backend_identity,
        checkpoint_sha256=checkpoint_sha256,
    )


def collect_literal_t4_native_scene_observation(
    *,
    runtime: LiteralT4CollectionRuntime,
    materialization_root: Path,
    plan_path: Path,
    split: str,
    sample_index: int,
    scene: str,
    context_loader: Callable[..., Any] = load_acid_joint_context,
) -> dict[str, Any]:
    """Collect one source-only literal T=4 scene observation on the GPU."""

    if not isinstance(runtime, LiteralT4CollectionRuntime):
        raise TypeError("literal T=4 collector requires a native runtime")
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
        # The native encoder retains a custom checkpoint/autograd path in its
        # source tree. ``no_grad`` prevents graph retention without turning
        # its captured tensors into inference tensors.
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
                    raise RuntimeError("literal T=4 collector has no routing tensors")
                _views, _channels, height, width = execution.dense_raw_head.shape
                if height % TILE_SIZE or width % TILE_SIZE:
                    raise RuntimeError("literal T=4 collector image shape is not divisible by four")
                plan = build_literal_paper_t4_probe_first_plan(
                    execution.routing_features,
                    execution.routing_z_depths,
                    height=height,
                    width=width,
                    feature_threshold=FEATURE_THRESHOLD,
                    depth_threshold=DEPTH_THRESHOLD,
                )
                if plan.events.get("contract_version") != LITERAL_PAPER_T4_PLAN_CONTRACT:
                    raise RuntimeError("literal T=4 collector route plan changed")
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
                preflight = preflight_depthsplat_l0_l1_materialization(
                    initial_packet,
                    initial_packed,
                    plan,
                    execution.routing_features,
                    execution.routing_z_depths,
                    source_sample_image_grid=sample_image_grid,
                    source_get_world_rays=get_world_rays,
                    maximum_coverage_covariance_scale=1.0,
                    execution_profile=DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
                    collect_selected_anchor_attribute_loo_risk=True,
                )
                if (
                    preflight.events.get("execution_profile")
                    != LITERAL_T4_MATERIALIZATION_PROFILE
                    or preflight.events.get("maximum_coverage_covariance_scale") != 1.0
                    or preflight.events.get("selected_anchor_attribute_loo_collect_only")
                    is not True
                    or preflight.events.get("selected_anchor_attribute_loo_frozen_guard") is not None
                    or int(preflight.update_dense_slots.numel()) < 1
                ):
                    raise RuntimeError("literal T=4 collector preflight profile changed")
                aggregate = preflight.events.get("selected_anchor_attribute_loo_aggregate")
                if not isinstance(aggregate, Mapping):
                    raise RuntimeError("literal T=4 collector has no LOO aggregate")
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
                raise RuntimeError("literal T=4 collector preflight has no update binding")
            full_binding = final_packed.source_trace.get(
                "native_full_adapter_attribute_binding_sha256"
            )
            trace = [
                _copy_scalar_trace_evidence(entry, label="preflight tile trace")
                for entry in preflight.tile_trace
            ]
            frozen_aggregate = _copy_scalar_trace_evidence(
                aggregate, label="selected-anchor LOO aggregate"
            )
            if not isinstance(frozen_aggregate, Mapping):
                raise RuntimeError("literal T=4 collector LOO aggregate copy changed")
            observation = {
                "scene": scene,
                "maximum_held_out_risks": list(
                    frozen_aggregate.get("maximum_held_out_risks", [])
                ),
                "preflight_tile_trace": trace,
                "selected_anchor_loo_aggregate": dict(frozen_aggregate),
                "evidence": {
                    "profile_sha256": literal_t4_profile_sha256(),
                    "route_plan_config_sha256": literal_t4_profile()["route_plan_config_sha256"],
                    "native_execution_sha256": _require_sha256(
                        execution.events.get("native_execution_sha256"), "native execution"
                    ),
                    "initial_attribute_binding_sha256": _require_sha256(
                        initial_packed.attribute_binding_sha256, "initial Adapter binding"
                    ),
                    "final_selected_attribute_binding_sha256": _require_sha256(
                        final_packed.attribute_binding_sha256, "final selected Adapter binding"
                    ),
                    "selected_head_replay_fallback_summary": (
                        build_selected_head_replay_fallback_summary(
                            initial_events=initial_replay.events,
                            final_events=producer_replay.events,
                        )
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
        # The returned observation contains only copied scalar/trace evidence.
        # Drop every scene-sized packet before returning blocks to CUDA's cache.
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
        raise RuntimeError("literal T=4 collector did not produce scene evidence")
    return observation


def _scene_record_from_observation(
    *,
    observation: Mapping[str, Any],
    split: str,
    sample_index: int,
    expected_scene: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Persist an immutable trace and construct one frozen-record scene row."""

    required = {
        "scene",
        "maximum_held_out_risks",
        "preflight_tile_trace",
        "selected_anchor_loo_aggregate",
        "evidence",
        "access",
    }
    if not isinstance(observation, Mapping) or set(observation) != required:
        raise ValueError("literal T=4 collector scene observation has an invalid schema")
    if observation.get("scene") != expected_scene or observation.get("access") != _ACCESS:
        raise ValueError("literal T=4 collector scene observation identity changed")
    base_evidence = observation.get("evidence")
    if not isinstance(base_evidence, Mapping) or set(base_evidence) != _SCENE_BASE_EVIDENCE_KEYS:
        raise ValueError("literal T=4 collector scene base evidence has an invalid schema")
    if (
        base_evidence.get("profile_sha256") != literal_t4_profile_sha256()
        or base_evidence.get("route_plan_config_sha256")
        != literal_t4_profile()["route_plan_config_sha256"]
        or base_evidence.get("coverage_certificate") != LITERAL_T4_MOMENT_CERTIFICATE
        or base_evidence.get("accepted_update_slot_count", 0) < 1
        or base_evidence.get("nonzero_merge_applied") is not True
        or base_evidence.get("renderer_executed") is not False
        or base_evidence.get("quality_metrics_computed") is not False
        or base_evidence.get("whole_pipeline_s2_s3_sparse_execution_verified") is not False
        or base_evidence.get("global_s2_s3_savings_claimed") is not False
        or base_evidence.get("source_nonprobe_s3_attribute_reads") != 0
    ):
        raise ValueError("literal T=4 collector scene evidence crossed its source-only contract")
    artifact = build_v16t4_trace_artifact(
        scene=expected_scene,
        split=split,
        preflight_tile_trace=observation["preflight_tile_trace"],
        selected_anchor_loo_aggregate=observation["selected_anchor_loo_aggregate"],
    )
    relative_path = _trace_relative_path(
        split=split, sample_index=sample_index, scene=expected_scene
    )
    _write_new_json(output_directory / relative_path, artifact)
    aggregate = artifact["selected_anchor_loo_aggregate"]
    evidence = {
        **dict(base_evidence),
        "selected_anchor_loo_aggregate": aggregate,
        "selected_anchor_loo_aggregate_sha256": canonical_sha256(aggregate),
        "selected_anchor_loo_trace_sha256": aggregate["preflight_tile_trace_sha256"],
        "trace_artifact": {
            "relative_path": relative_path,
            "sha256": artifact["sha256"],
            "preflight_tile_trace_sha256": aggregate["preflight_tile_trace_sha256"],
            "tile_records_sha256": aggregate["tile_records_sha256"],
        },
    }
    return {
        "scene": expected_scene,
        "maximum_held_out_risks": list(observation["maximum_held_out_risks"]),
        "evidence": evidence,
        "access": dict(_ACCESS),
    }


def _collect_literal_t4_v16_calibration_with_dependencies(
    *,
    output_directory: Path,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
    scene_worker: Callable[..., Mapping[str, Any]],
    binding: Mapping[str, Any],
    application: Mapping[str, Any],
    runtime: LiteralT4CollectionRuntime | None = None,
    record_loader: Callable[..., Mapping[str, Any]] = load_frozen_v16t4_threshold,
) -> dict[str, Any]:
    """Private collection orchestration with explicit testable dependencies.

    Production callers must use :func:`collect_literal_t4_v16_calibration`.
    This helper exists so CPU contract tests can exercise artifact layout and
    reload behavior without presenting a substitute worker as a production
    collection path.
    """

    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(
            f"literal T=4 calibration output directory must be new: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    live_binding = dict(binding)
    application = dict(application)
    _require_complete_collector_source(application)
    if not callable(scene_worker) or not callable(record_loader):
        raise TypeError("literal T=4 collector requires callable worker and record loader")

    records: dict[str, list[dict[str, Any]]] = {TRAIN_SPLIT: [], HOLDOUT_SPLIT: []}
    for split in (TRAIN_SPLIT, HOLDOUT_SPLIT):
        split_binding = live_binding.get("splits", {}).get(split)
        scenes = split_binding.get("scenes") if isinstance(split_binding, Mapping) else None
        if not isinstance(scenes, list):
            raise ValueError(f"literal T=4 collector has no {split} scene list")
        for sample_index, scene in enumerate(scenes):
            kwargs = {
                "materialization_root": Path(materialization_root),
                "plan_path": Path(plan_path),
                "split": split,
                "sample_index": sample_index,
                "scene": scene,
            }
            if runtime is not None:
                kwargs["runtime"] = runtime
            observation = scene_worker(**kwargs)
            records[split].append(
                _scene_record_from_observation(
                    observation=observation,
                    split=split,
                    sample_index=sample_index,
                    expected_scene=scene,
                    output_directory=destination,
                )
            )

    record = build_v16t4_record(
        binding=live_binding,
        application=application,
        profile=literal_t4_profile(),
        train_scene_records=records[TRAIN_SPLIT],
        holdout_scene_records=records[HOLDOUT_SPLIT],
    )
    record_path = destination / V16T4_RECORD_NAME
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
        or loaded.get("profile_sha256") != literal_t4_profile_sha256()
    ):
        raise RuntimeError("literal T=4 collector live reload changed its frozen record")
    return {
        "record_path": record_path,
        "record": dict(loaded),
        "scene_trace_count": sum(len(values) for values in records.values()),
    }


def collect_literal_t4_v16_calibration(
    *,
    device: torch.device,
    output_directory: Path,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Run the production CUDA-native ACID 24/8 literal T=4 collection.

    This public entrypoint deliberately owns all production dependencies.  It
    always loads the source-bound encoder on CUDA, resolves the live ACID
    binding, and uses the native scene observer; test workers and synthetic
    identities are confined to the private orchestration helper above.
    """

    device = _require_cuda_device(device)
    runtime = load_literal_t4_collection_runtime(device=device)
    live_binding = resolve_depthsplat_acid_binding(
        plan_path=Path(plan_path), materialization_root=Path(materialization_root)
    )
    application = build_literal_t4_application(
        backend_identity=runtime.backend_identity,
        root=ROOT,
    )
    return _collect_literal_t4_v16_calibration_with_dependencies(
        output_directory=output_directory,
        plan_path=plan_path,
        materialization_root=materialization_root,
        scene_worker=collect_literal_t4_native_scene_observation,
        runtime=runtime,
        binding=live_binding,
        application=application,
    )


__all__ = [
    "DATASET",
    "FEATURE_THRESHOLD",
    "HOLDOUT_SPLIT",
    "LITERAL_T4_MATERIALIZATION_PROFILE",
    "LITERAL_T4_PROFILE_ID",
    "LITERAL_T4_PROFILE_SCHEMA",
    "LITERAL_T4_RISK_METRIC",
    "LiteralT4CollectionRuntime",
    "MODEL",
    "REQUIRED_COLLECTOR_SOURCE_FILES",
    "TILE_SIZE",
    "TRAIN_SPLIT",
    "V16T4_RECORD_NAME",
    "collect_literal_t4_native_scene_observation",
    "collect_literal_t4_v16_calibration",
    "load_literal_t4_collection_runtime",
]
