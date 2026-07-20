#!/usr/bin/env python3
"""Audit literal T=4 virtual-support coverage from a context-only sidecar.

This is a diagnostic-only encoder path.  It reconstructs the current frozen
V16T4 preflight from source contexts, then checks each merged anchor against
the twelve selected-only virtual Gaussians in its source camera.  It never
constructs a decoder, opens target metadata or RGB, renders, or computes a
quality metric.
"""

# ruff: noqa: E402

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
from saes.depthsplat_l0_l1_materializer import (  # noqa: E402
    preflight_depthsplat_l0_l1_materialization,
)
from saes.depthsplat_literal_t4_virtual_coverage_audit import (  # noqa: E402
    AUDIT_KIND,
    AUDIT_SCHEMA_VERSION,
    audit_literal_t4_virtual_anchor_coverage,
)
from saes.depthsplat_selected_output import (  # noqa: E402
    DepthSplatPackedGaussianConsumer,
    build_depthsplat_sparse_raw_packet,
    capture_depthsplat_native_execution,
    compare_depthsplat_packed_to_dense,
    replay_depthsplat_selected_head,
)
from saes.probe_first_schedule import (  # noqa: E402
    LITERAL_PAPER_T4_PLAN_CONTRACT,
    build_literal_paper_t4_probe_first_plan,
)
from scripts.saes_depthsplat_l0_l1_context_only_audit import (  # noqa: E402
    DATASET,
    DEPTH_THRESHOLD,
    FEATURE_THRESHOLD,
    MODEL,
    SEED,
    SOURCE_SAMPLE_INDEX,
    TILE_SIZE,
    _bind_formal_application,
    _load_literal_t4_v16_audit_guard,
    _require_equivalent,
    _require_formal_context_identity,
    _require_loaded_context_batch,
    _source_geometry_functions,
)
from scripts.saes_selected_output_replay_audit import (  # noqa: E402
    strict_fp32_convolution_execution,
)


DEFAULT_INPUT_ROOT = (
    ROOT
    / "outputs"
    / "ae_dl3dv_repair_diagnostics"
    / "depthsplat_sample0_l0_l1_context_only_v2"
)


def collect_literal_t4_virtual_coverage_audit(
    *,
    input_root: Path,
    device: torch.device,
    literal_t4_v16_record: Path,
    literal_t4_acid_plan_path: Path,
    literal_t4_acid_materialization_root: Path,
) -> dict[str, Any]:
    """Collect one target-free, source-camera virtual-support observation."""

    from data.context_only_audit_input import validate_context_only_audit_input
    from integration import create_model_loader, load_context_only_audit_data
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.result_record import cached_sha256_file, source_identity

    # Authenticate the current fixed-scale V16T4 application before native work.
    literal_guard, literal_profile = _load_literal_t4_v16_audit_guard(
        record_path=literal_t4_v16_record,
        acid_plan_path=literal_t4_acid_plan_path,
        acid_materialization_root=literal_t4_acid_materialization_root,
    )
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
        raise RuntimeError("literal virtual coverage audit unexpectedly constructed a decoder")
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
            raise RuntimeError("literal virtual coverage audit has no source routing tensors")
        views, _channels, height, width = execution.dense_raw_head.shape
        if height % TILE_SIZE or width % TILE_SIZE:
            raise RuntimeError("literal virtual coverage audit image shape is not tiled by four")
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
            or views != len(input_identity["context_indices"])
        ):
            raise RuntimeError("literal virtual coverage audit route plan changed")
        initial_replay = replay_depthsplat_selected_head(
            bundle.encoder.gaussian_head,
            execution.gaussian_head_input,
            execution.dense_raw_head,
            plan.selection_mask,
            native_full_mask=plan.full_mask,
        )
        _require_equivalent(
            initial_replay.equivalence,
            "literal virtual coverage initial selected replay",
        )
        initial_packet = build_depthsplat_sparse_raw_packet(execution, initial_replay)
        consumer = DepthSplatPackedGaussianConsumer(bundle.encoder.gaussian_adapter)
        initial_packed = consumer.convert(
            initial_packet,
            image_shape=(height, width),
            native_execution=execution,
            native_full_mask=plan.full_mask,
        )
        _require_equivalent(
            compare_depthsplat_packed_to_dense(initial_packed, execution.dense_gaussians),
            "literal virtual coverage initial Adapter packet",
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
            execution_profile=literal_profile["materialization_profile"],
            selected_anchor_attribute_loo_frozen_guard=literal_guard,
        )
        if (
            preflight.events.get("execution_profile")
            != literal_profile["materialization_profile"]
            or preflight.events.get("maximum_coverage_covariance_scale") != 1.0
            or preflight.events.get("literal_support_containment_guard") is not False
            or preflight.events.get("selected_anchor_attribute_loo_frozen_guard")
            != literal_guard.as_dict()
        ):
            raise RuntimeError("literal virtual coverage audit preflight changed")
        coverage = audit_literal_t4_virtual_anchor_coverage(
            packet=initial_packet,
            packed=initial_packed,
            plan=plan,
            routing_features=execution.routing_features,
            routing_z_depths=execution.routing_z_depths,
            preflight=preflight,
        )

    support_complete = coverage["summary"]["continuity"][
        "all_active_virtual_2sigma_supports_contained"
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "status": "PASS",
        "paper_result_eligible": False,
        "diagnostic_verdict": (
            "no-fixed-scale-2sigma-support-hole"
            if support_complete
            else "fixed-scale-2sigma-support-incomplete"
        ),
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
            "native_dense_depth_predictor_executed": True,
            "native_dense_gaussian_regressor_executed": True,
            "selected_native_rgb_adapter_executed": True,
            "nonzero_l0_l1_preflight_executed": True,
        },
        "context_only_input": {
            "identity": input_identity,
            "loaded_native_preprocessing": loaded_calibration["native_preprocessing"],
        },
        "literal_t4_v16": {
            "frozen_record_sha256": literal_guard["frozen_record_sha256"],
            "frozen_record_kind": literal_guard["frozen_record_kind"],
            "threshold_value": literal_guard["threshold_value"],
            "threshold_rule": literal_guard["threshold_rule"],
            "risk_metric": literal_guard["risk_metric"],
            "materialization_profile": literal_profile["materialization_profile"],
            "route_plan_config_sha256": literal_guard["route_plan_config_sha256"],
            "guard": literal_guard.as_dict(),
        },
        "preflight_binding": {
            "update_anchor_count": int(preflight.update_dense_slots.numel()),
            "update_binding": dict(preflight.events["update_binding"]),
            "tile_trace_sha256": preflight.events["tile_trace_sha256"],
            "coverage_certificate": preflight.events["coverage_certificate"],
            "coverage_certificate_sha256": preflight.events[
                "coverage_certificate_sha256"
            ],
        },
        "coverage": coverage,
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
    from saes.depthsplat_acid_disjoint_calibration import (
        DEFAULT_MATERIALIZATION_ROOT,
        DEFAULT_PLAN_PATH,
    )

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
        parser.error("literal virtual coverage audit requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from scripts.result_record import portable_command, write_result

    try:
        record = collect_literal_t4_virtual_coverage_audit(
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
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
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
