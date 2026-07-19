#!/usr/bin/env python3
"""Audit one formal target-free DepthSplat DL3DV L0/L1 packet path.

The input is a context-camera-only sidecar for the fixed DL3DV source sample
zero.  This command loads an encoder-only DepthSplat model, materializes the
source-bound selected packet and nonzero L0/L1 merge, and stops before any
decoder, target view, quality metric, timing, or S2/S3 saving claim.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import random
import re
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
from saes.depthsplat_acid_disjoint_calibration import (  # noqa: E402
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
)
from saes.depthsplat_literal_t4_acid_calibration import (  # noqa: E402
    LITERAL_T4_GUARD_SCHEMA,
    LITERAL_T4_MATERIALIZATION_PROFILE,
    V16T4_KIND,
    VerifiedLiteralT4MaterializerGuard,
    literal_t4_profile,
    literal_t4_profile_sha256,
    to_materializer_guard,
    verified_literal_t4_materializer_guard_projection,
)
from saes.depthsplat_l0_l1_materializer import (  # noqa: E402
    apply_depthsplat_compact_l0_l1_materialization,
    preflight_depthsplat_l0_l1_materialization,
    resolve_depthsplat_compact_final_route,
)
from saes.depthsplat_selected_output import (  # noqa: E402
    DepthSplatPackedGaussianConsumer,
    build_depthsplat_sparse_raw_packet,
    capture_depthsplat_native_execution,
    compare_depthsplat_full_passthrough_to_dense_bitwise,
    compare_depthsplat_packed_to_dense,
    replay_depthsplat_selected_head,
    subset_depthsplat_sparse_raw_packet,
)
from saes.probe_first_schedule import (  # noqa: E402
    LITERAL_PAPER_T4_PLAN_CONTRACT,
    build_literal_paper_t4_probe_first_plan,
)
from scripts.saes_selected_output_replay_audit import (  # noqa: E402
    strict_fp32_convolution_execution,
)


MODEL = "depthsplat"
DATASET = "dl3dv"
SOURCE_SAMPLE_INDEX = 0
SEED = 0
TILE_SIZE = 4
FEATURE_THRESHOLD = 0.20
DEPTH_THRESHOLD = 0.10
AUDIT_KIND = "depthsplat-formal-context-only-l0-l1-materialization-audit"
AUDIT_SCHEMA_VERSION = "1.0"
DEFAULT_INPUT_ROOT = (
    ROOT
    / "outputs"
    / "ae_dl3dv_repair_diagnostics"
    / "depthsplat_sample0_l0_l1_context_only_v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"DepthSplat formal audit has an invalid {label} SHA256")
    return value


def _load_literal_t4_v16_audit_guard(
    *,
    record_path: Path,
    acid_plan_path: Path,
    acid_materialization_root: Path,
) -> tuple[VerifiedLiteralT4MaterializerGuard, dict[str, Any]]:
    """Live-load the only V16 guard accepted before native model work."""

    guard = to_materializer_guard(
        Path(record_path),
        root=ROOT,
        plan_path=Path(acid_plan_path),
        materialization_root=Path(acid_materialization_root),
    )
    if not isinstance(guard, VerifiedLiteralT4MaterializerGuard):
        raise ValueError(
            "DepthSplat formal audit literal T=4 V16 guard is not authenticated"
        )
    try:
        projection = verified_literal_t4_materializer_guard_projection(guard)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "DepthSplat formal audit literal T=4 V16 guard is not authenticated"
        ) from error
    profile = literal_t4_profile()
    required = {
        "schema_version",
        "frozen_record_kind",
        "frozen_record_sha256",
        "threshold_value",
        "threshold_rule",
        "risk_metric",
        "materialization_profile",
        "route_plan_contract",
        "route_plan_config_sha256",
        "acid_binding_sha256",
        "application_sha256",
    }
    if (
        set(projection) != required
        or projection.get("schema_version") != LITERAL_T4_GUARD_SCHEMA
        or projection.get("frozen_record_kind") != V16T4_KIND
        or projection.get("materialization_profile")
        != LITERAL_T4_MATERIALIZATION_PROFILE
        or projection.get("route_plan_contract") != LITERAL_PAPER_T4_PLAN_CONTRACT
        or projection.get("route_plan_config_sha256")
        != profile["route_plan_config_sha256"]
    ):
        raise ValueError("DepthSplat formal audit literal T=4 V16 guard changed")
    for key in (
        "frozen_record_sha256",
        "route_plan_config_sha256",
        "acid_binding_sha256",
        "application_sha256",
    ):
        _require_sha256(projection.get(key), f"literal T=4 guard {key}")
    threshold = projection.get("threshold_value")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
        or float(threshold) < 0.0
    ):
        raise ValueError("DepthSplat formal audit literal T=4 guard threshold changed")
    return guard, profile


def _require_formal_context_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the predeclared DepthSplat source-sample-zero sidecar."""

    if not isinstance(identity, Mapping):
        raise TypeError("DepthSplat formal audit context identity must be a mapping")
    if (
        identity.get("source_sample_index") != SOURCE_SAMPLE_INDEX
        or not isinstance(identity.get("scene"), str)
        or not identity["scene"]
        or not isinstance(identity.get("context_indices"), list)
        or len(identity["context_indices"]) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in identity["context_indices"]
        )
        or len(set(identity["context_indices"])) != 2
        or identity.get("target_rgb_accessed") is not False
        or identity.get("target_camera_metadata_accessed") is not False
    ):
        raise ValueError("DepthSplat formal audit input is not fixed context-only sample zero")
    source_binding = identity.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise ValueError("DepthSplat formal audit input has no source binding")
    for key in (
        "canonical_index_sha256",
        "canonical_sample_selection_sha256",
        "canonical_selection_sha256",
        "source_audit_input_sha256",
        "source_audit_tree_sha256",
        "source_sidecar_tree_sha256",
    ):
        _require_sha256(source_binding.get(key), f"context input {key}")
    for key in ("tree_sha256", "manifest_sha256", "audit_input_sha256"):
        _require_sha256(identity.get(key), f"context input {key}")
    return dict(identity)


def _bind_formal_application(
    *,
    input_identity: Mapping[str, Any],
    backend_contract: Any,
    experiment: Any,
    selection: Any,
) -> None:
    """Bind one context-only input to the live source/checkpoint contract."""

    if (
        getattr(backend_contract, "model", None) != MODEL
        or getattr(backend_contract, "dataset", None) != DATASET
        or getattr(experiment, "model", None) != MODEL
        or getattr(experiment, "dataset", None) != DATASET
        or getattr(experiment, "experiment", None) != backend_contract.experiment
        or Path(experiment.checkpoint).resolve()
        != Path(backend_contract.checkpoint).resolve()
        or Path(selection.index_path).resolve()
        != Path(backend_contract.evaluation_index).resolve()
    ):
        raise RuntimeError("DepthSplat formal audit application identity changed")
    binding = input_identity["source_binding"]
    if (
        binding["canonical_index_sha256"] != selection.source_index_sha256
        or binding["canonical_sample_selection_sha256"]
        != selection.sample_selection_sha256
    ):
        raise RuntimeError("DepthSplat formal audit sidecar selection changed")


def _require_loaded_context_batch(
    batch: Mapping[str, Any], input_identity: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject a loader result that exposes any target-side payload."""

    if not isinstance(batch, Mapping) or "target" in batch:
        raise RuntimeError("DepthSplat formal audit loader returned a target mapping")
    context = batch.get("context")
    required_context = ("image", "extrinsics", "intrinsics", "near", "far", "index")
    if not isinstance(context, Mapping) or any(
        not torch.is_tensor(context.get(name)) for name in required_context
    ):
        raise RuntimeError("DepthSplat formal audit loader returned an incomplete context")
    scene = batch.get("scene")
    if (
        not isinstance(scene, list)
        or scene != [input_identity["scene"]]
        or context["index"].shape != (1, len(input_identity["context_indices"]))
        or [int(value) for value in context["index"][0].tolist()]
        != input_identity["context_indices"]
    ):
        raise RuntimeError("DepthSplat formal audit loader changed the fixed context")
    calibration = batch.get("calibration")
    if not isinstance(calibration, Mapping):
        raise RuntimeError("DepthSplat formal audit loader has no input identity")
    for key, value in input_identity.items():
        if calibration.get(key) != value:
            raise RuntimeError("DepthSplat formal audit loader changed input identity")
    if (
        calibration.get("target_mapping_present") is not False
        or calibration.get("target_rgb_accessed") is not False
        or calibration.get("target_camera_metadata_accessed") is not False
        or not isinstance(calibration.get("native_preprocessing"), Mapping)
    ):
        raise RuntimeError("DepthSplat formal audit loader crossed the target-free boundary")
    return dict(context), dict(calibration)


def _source_geometry_functions(encoder: Any) -> tuple[Any, Any, dict[str, str]]:
    """Resolve geometry only from the loaded native DepthSplat namespace."""

    module_name = type(encoder).__module__
    if ".model." not in module_name:
        raise RuntimeError("DepthSplat formal audit encoder module changed")
    geometry_module_name = module_name.split(".model.", 1)[0] + ".geometry.projection"
    module = sys.modules.get(geometry_module_name)
    source = getattr(module, "__file__", None)
    source_root = (ROOT / "depthsplat" / "src").resolve()
    if not isinstance(source, str):
        raise RuntimeError("DepthSplat formal audit geometry module was not loaded")
    source_path = Path(source).resolve()
    if source_root not in source_path.parents:
        raise RuntimeError("DepthSplat formal audit geometry source is foreign")
    sample_image_grid = getattr(module, "sample_image_grid", None)
    get_world_rays = getattr(module, "get_world_rays", None)
    if not callable(sample_image_grid) or not callable(get_world_rays):
        raise RuntimeError("DepthSplat formal audit source geometry is incomplete")
    return sample_image_grid, get_world_rays, {
        "module": geometry_module_name,
        "path": source_path.relative_to(ROOT).as_posix(),
    }


def _require_equivalent(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise RuntimeError(f"{label} returned an invalid equivalence report")
    if report.get("equivalent") is not True:
        raise RuntimeError(f"{label} is not source-native equivalent: {dict(report)}")
    return dict(report)


def _require_bitwise_equivalent(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise RuntimeError(f"{label} returned an invalid bitwise report")
    if report.get("bitwise_equivalent") is not True:
        raise RuntimeError(f"{label} is not bitwise equivalent: {dict(report)}")
    return dict(report)


def collect_depthsplat_l0_l1_context_only_audit(
    *,
    input_root: Path,
    device: torch.device,
    literal_t4_v16_record: Path,
    literal_t4_acid_plan_path: Path = DEFAULT_PLAN_PATH,
    literal_t4_acid_materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Run one formal target-free packet/materialization audit without rendering."""

    from data.context_only_audit_input import validate_context_only_audit_input
    from integration import create_model_loader, load_context_only_audit_data
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.result_record import cached_sha256_file, source_identity

    # The record's path loader rechecks ACID, backend, collector-source, and
    # per-scene trace identities.  It must complete before model construction.
    literal_guard, literal_profile = _load_literal_t4_v16_audit_guard(
        record_path=literal_t4_v16_record,
        acid_plan_path=literal_t4_acid_plan_path,
        acid_materialization_root=literal_t4_acid_materialization_root,
    )
    literal_guard_projection = literal_guard.as_dict()
    input_identity = _require_formal_context_identity(
        validate_context_only_audit_input(input_root, model=MODEL)
    )
    backend_contract = resolve_depthsplat_backend_contract(ROOT)
    backend_identity = freeze_depthsplat_backend_identity(backend_contract)
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    selection = resolve_claim_selection(MODEL, DATASET, ROOT)
    _bind_formal_application(
        input_identity=input_identity,
        backend_contract=backend_contract,
        experiment=experiment,
        selection=selection,
    )
    loader = create_model_loader(MODEL)
    bundle = loader.load_model(
        str(experiment.checkpoint),
        device=device,
        experiment_name=experiment.experiment,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        hydra_overrides=experiment.hydra_overrides,
        encoder_only=True,
    )
    if bundle.decoder is not None:
        raise RuntimeError("DepthSplat formal audit unexpectedly constructed a decoder")
    data = load_context_only_audit_data(
        loader, bundle, input_root=Path(input_root), model_name=MODEL
    )
    context_cpu, loaded_calibration = _require_loaded_context_batch(
        data.batch, input_identity
    )
    context = {
        key: value.to(bundle.device) if torch.is_tensor(value) else value
        for key, value in context_cpu.items()
    }
    bundle.model.eval()

    with strict_fp32_convolution_execution() as numerical_execution:
        execution = capture_depthsplat_native_execution(
            bundle.encoder, context, source_root=ROOT / "depthsplat"
        )
        if execution.routing_features is None or execution.routing_z_depths is None:
            raise RuntimeError("DepthSplat formal audit has no target-free routing tensors")
        _views, _channels, height, width = execution.dense_raw_head.shape
        if height % TILE_SIZE or width % TILE_SIZE:
            raise RuntimeError("DepthSplat formal audit image shape is not tiled by four")
        plan = build_literal_paper_t4_probe_first_plan(
            execution.routing_features,
            execution.routing_z_depths,
            height=height,
            width=width,
            feature_threshold=FEATURE_THRESHOLD,
            depth_threshold=DEPTH_THRESHOLD,
        )
        if (
            plan.events.get("contract_version") != LITERAL_PAPER_T4_PLAN_CONTRACT
            or plan.events.get("literal_paper_t4_route_config_sha256")
            != literal_guard["route_plan_config_sha256"]
            or plan.events.get("l0_anchor_count") != 4
            or plan.events.get("l1_anchor_count") != 4
            or int(plan.secondary_mask.sum().item()) != 0
        ):
            raise RuntimeError("DepthSplat formal audit literal T=4 plan changed")
        initial_replay = replay_depthsplat_selected_head(
            bundle.encoder.gaussian_head,
            execution.gaussian_head_input,
            execution.dense_raw_head,
            plan.selection_mask,
            native_full_mask=plan.full_mask,
        )
        if initial_replay.equivalence.get("equivalent") is not True:
            raise RuntimeError("DepthSplat formal audit initial selected replay drifted")
        initial_packet = build_depthsplat_sparse_raw_packet(execution, initial_replay)
        consumer = DepthSplatPackedGaussianConsumer(bundle.encoder.gaussian_adapter)
        initial_packed = consumer.convert(
            initial_packet,
            image_shape=(height, width),
            native_execution=execution,
            native_full_mask=plan.full_mask,
        )
        initial_adapter_equivalence = _require_equivalent(
            compare_depthsplat_packed_to_dense(initial_packed, execution.dense_gaussians),
            "DepthSplat formal audit initial Adapter packet",
        )
        initial_full_passthrough = _require_bitwise_equivalent(
            compare_depthsplat_full_passthrough_to_dense_bitwise(
                initial_packed, execution.dense_gaussians, plan.full_mask
            ),
            "DepthSplat formal audit initial Full attributes",
        )
        sample_image_grid, get_world_rays, geometry_source = _source_geometry_functions(
            bundle.encoder
        )
        preflight = preflight_depthsplat_l0_l1_materialization(
            initial_packet,
            initial_packed,
            plan,
            execution.routing_features,
            execution.routing_z_depths,
            source_sample_image_grid=sample_image_grid,
            source_get_world_rays=get_world_rays,
            maximum_coverage_covariance_scale=1.0,
            execution_profile=LITERAL_T4_MATERIALIZATION_PROFILE,
            selected_anchor_attribute_loo_frozen_guard=literal_guard,
        )
        if (
            preflight.events.get("execution_profile")
            != literal_profile["materialization_profile"]
            or preflight.events.get("maximum_coverage_covariance_scale") != 1.0
            or preflight.events.get("selected_anchor_attribute_loo_frozen_guard")
            != literal_guard_projection
            or preflight.events.get("selected_anchor_attribute_loo_collect_only")
            is not False
            or preflight.events.get("selected_anchor_attribute_loo_guard") is not True
        ):
            raise RuntimeError("DepthSplat formal audit literal V16 preflight changed")
        final_route = resolve_depthsplat_compact_final_route(plan, preflight)
        producer_replay = replay_depthsplat_selected_head(
            bundle.encoder.gaussian_head,
            execution.gaussian_head_input,
            execution.dense_raw_head,
            final_route.raw_head_request_mask,
            native_full_mask=final_route.full_passthrough_mask,
        )
        if producer_replay.equivalence.get("equivalent") is not True:
            raise RuntimeError("DepthSplat formal audit final selected replay drifted")
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
            "DepthSplat formal audit final Adapter packet",
        )
        final_full_passthrough = _require_bitwise_equivalent(
            compare_depthsplat_full_passthrough_to_dense_bitwise(
                final_packed,
                execution.dense_gaussians,
                final_route.full_passthrough_mask,
            ),
            "DepthSplat formal audit final Full attributes",
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
            "DepthSplat formal audit materialized Full attributes",
        )

    descriptor_count = int(materialized.dense_slots.numel())
    if descriptor_count != int(final_route.selected_output_mask.sum().item()):
        raise RuntimeError("DepthSplat formal audit materialized packet route drifted")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "status": "PASS",
        "paper_result_eligible": False,
        "formal_target_free_audit": True,
        "formal_target_free_sidecar_used": True,
        "model": MODEL,
        "dataset": DATASET,
        "source_sample_index": SOURCE_SAMPLE_INDEX,
        "scene": input_identity["scene"],
        "context_indices": list(input_identity["context_indices"]),
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "execution_boundary": {
            "encoder_only": True,
            "decoder_constructed": False,
            "renderer_executed": False,
            "target_view_rendered": False,
            "quality_metrics_computed": False,
            "timing_claim": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
            "native_dense_depth_predictor_executed": True,
            "native_dense_gaussian_regressor_executed": True,
            "initial_selected_native_rgb_adapter_executed": True,
            "final_selected_native_rgb_adapter_executed": True,
            "nonzero_l0_l1_materializer_executed": True,
        },
        "context_only_input": {
            "identity": input_identity,
            "loaded_native_preprocessing": loaded_calibration["native_preprocessing"],
        },
        "literal_t4_v16": {
            "frozen_record_sha256": literal_guard["frozen_record_sha256"],
            "frozen_record_kind": literal_guard["frozen_record_kind"],
            "profile": literal_profile,
            "profile_sha256": literal_t4_profile_sha256(),
            "threshold_value": literal_guard["threshold_value"],
            "threshold_rule": literal_guard["threshold_rule"],
            "risk_metric": literal_guard["risk_metric"],
            "acid_binding_sha256": literal_guard["acid_binding_sha256"],
            "application_sha256": literal_guard["application_sha256"],
            "guard": literal_guard_projection,
        },
        "route_plan": plan.events,
        "materialization_preflight": preflight.events,
        "materialization_preflight_tile_trace": [
            dict(record) for record in preflight.tile_trace
        ],
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
        "initial_packet": {
            "descriptor_count": int(initial_packed.dense_slots.numel()),
            "source_trace_sha256": initial_packed.source_trace_sha256,
            "attribute_binding_sha256": initial_packed.attribute_binding_sha256,
            "native_adapter_equivalence": initial_adapter_equivalence,
            "full_passthrough_bitwise": initial_full_passthrough,
        },
        "final_packet": {
            "producer_descriptor_count": int(producer_packet.dense_slots.numel()),
            "descriptor_count": descriptor_count,
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
            "descriptor_count": descriptor_count,
            "source_trace_sha256": materialized.source_trace_sha256,
            "attribute_binding_sha256": materialized.attribute_binding_sha256,
            "full_passthrough_bitwise": materialized_full_passthrough,
        },
        "geometry_source": geometry_source,
        "backend_identity": backend_identity,
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "runner": {
            "path": Path(__file__).relative_to(ROOT).as_posix(),
            "sha256": cached_sha256_file(Path(__file__)),
        },
        "source": source_identity(),
        "numerical_execution": numerical_execution,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--literal-t4-v16-record", type=Path, required=True)
    parser.add_argument(
        "--literal-t4-acid-plan-path", type=Path, default=DEFAULT_PLAN_PATH
    )
    parser.add_argument(
        "--literal-t4-acid-materialization-root",
        type=Path,
        default=DEFAULT_MATERIALIZATION_ROOT,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("DepthSplat formal context-only audit requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from scripts.result_record import portable_command, write_result

    try:
        record = collect_depthsplat_l0_l1_context_only_audit(
            input_root=args.input_root,
            device=device,
            literal_t4_v16_record=args.literal_t4_v16_record,
            literal_t4_acid_plan_path=args.literal_t4_acid_plan_path,
            literal_t4_acid_materialization_root=(
                args.literal_t4_acid_materialization_root
            ),
        )
        exit_code = 0
    except Exception as exc:
        record = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "kind": AUDIT_KIND,
            "status": "FAILED",
            "paper_result_eligible": False,
            "formal_target_free_audit": True,
            "formal_target_free_sidecar_used": True,
            "model": MODEL,
            "dataset": DATASET,
            "source_sample_index": SOURCE_SAMPLE_INDEX,
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
