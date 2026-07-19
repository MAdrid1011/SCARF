#!/usr/bin/env python3
# ruff: noqa: E402
"""Diagnose source-only SAES risk signals against a posthoc dense-S3 oracle.

This development-only audit fixes the route before opening dense non-probe
attributes.  Its output cannot authorize a runtime deletion, tune a threshold,
or make a paper claim.  It exists only to determine whether a later, separately
validated risk-gate experiment is worth attempting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.progressive_saes import apply_progressive_saes
from scripts.calibration_inputs import sha256_file
from scripts.saes_dependency_audit import _context_on_device
from scripts.saes_diagnostics import guard_partition_full_s3_attribute_audit
from scripts.saes_l1_primary_reference_attribute_audit import (
    DATASET,
    DECISION_SEMANTICS,
    DEPTH_ROUTING_SEMANTICS,
    DEPTH_THRESHOLD,
    FEATURE_THRESHOLD,
    MODEL,
    SAMPLE_INDEX,
    SEED,
    TILE_SIZE,
    _load_audit_input,
)
from scripts.saes_l1_primary_reference_guard_partition_audit import (
    TRACE_FIELDS,
    _assert_mask_matches_trace,
    _assert_numerical_invariants,
    _partition_trace,
    _validate_trace,
)
from scripts.saes_target_free_materialization_audit import (
    _capture_encoder_execution,
    _clone_gaussians,
    _gaussians_on_cpu,
    _mask_sha256,
    _max_attribute_delta,
    _poison_skipped_descriptors,
)


AUDIT_KIND = "saes_deletion_source_scalar_dense_s3_risk_audit"
MATERIALIZATION = "representative"
FIXED_INPUT_ROOT = (
    ROOT
    / "outputs/ae_dl3dv_repair_diagnostics/"
    "transplat_sample0_l1_primary_reference_target_free_input_v1"
)
SOURCE_FIELDS = (
    "feature_variance",
    "feature_candidate",
    "depth_candidate",
    "routing_level",
    "guard_accepted",
    "anchor_count",
    "covariance_cosine_minimum",
    "harmonic_cosine_minimum",
    "opacity_distance_maximum",
    "probe_cross_check_checked",
    "probe_cross_check_error_max",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_scalars_from_trace(
    trace: list[dict[str, Any]],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Keep only source-bound tile scalars from the committed guard trace."""
    scalars: dict[tuple[int, int, int], dict[str, Any]] = {}
    for tile in trace:
        key = (tile["view_index"], tile["tile_row"], tile["tile_column"])
        route = tile["routing_level_before_materialization"]
        if key in scalars:
            raise RuntimeError("guard trace has duplicate tile source scalars")
        source: dict[str, Any] = {
            "feature_variance": float(tile["feature_variance"]),
            "feature_candidate": bool(tile["feature_candidate"]),
            "depth_candidate": tile["depth_candidate"],
            "routing_level": route,
            "guard_accepted": False,
            "anchor_count": 0,
            "covariance_cosine_minimum": None,
            "harmonic_cosine_minimum": None,
            "opacity_distance_maximum": None,
            "probe_cross_check_checked": False,
            "probe_cross_check_error_max": None,
        }
        if route in {"L0", "L1"}:
            accepted = next(
                (
                    guard
                    for guard in reversed(tile["guard_checks"])
                    if guard["level"] == route and guard["passed"]
                ),
                None,
            )
            if accepted is None:
                raise RuntimeError("sparse route has no accepted source guard")
            cross_check = accepted.get("probe_cross_check")
            source.update(
                {
                    "guard_accepted": True,
                    "anchor_count": int(accepted["anchor_count"]),
                    "covariance_cosine_minimum": float(
                        accepted["covariance_cosine_minimum"]
                    ),
                    "harmonic_cosine_minimum": float(
                        accepted["harmonic_cosine_minimum"]
                    ),
                    "opacity_distance_maximum": float(
                        accepted["opacity_distance_maximum"]
                    ),
                    "probe_cross_check_checked": bool(
                        cross_check and cross_check["checked"]
                    ),
                    "probe_cross_check_error_max": (
                        float(cross_check["error_max"])
                        if cross_check
                        and cross_check["checked"]
                        and cross_check["error_max"] is not None
                        else None
                    ),
                }
            )
        if tuple(source) != SOURCE_FIELDS:
            raise RuntimeError("source scalar record has an unstable schema")
        scalars[key] = source
    return scalars


def _scalar_trace_only(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop post-route annotations before reusing the fixed scalar validator."""
    scalar_trace = []
    for record in trace:
        if not TRACE_FIELDS <= set(record):
            raise RuntimeError("SAES trace is missing a required source scalar")
        scalar_trace.append({field: record[field] for field in TRACE_FIELDS})
    return scalar_trace


def _scalar_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only source fields; dense-S3 oracle values stay separate."""
    selected = [record for record in records if record["oracle_applicable"]]
    result: dict[str, Any] = {
        "sparse_candidate_tiles": len(selected),
        "full_tiles": len(records) - len(selected),
        "fields": {},
    }
    for field in (
        "feature_variance",
        "covariance_cosine_minimum",
        "harmonic_cosine_minimum",
        "opacity_distance_maximum",
        "probe_cross_check_error_max",
    ):
        values = [
            record["source_scalars"][field]
            for record in selected
            if record["source_scalars"][field] is not None
        ]
        if not values:
            result["fields"][field] = {"count": 0}
            continue
        tensor = torch.tensor(values, dtype=torch.float64)
        result["fields"][field] = {
            "count": int(tensor.numel()),
            "minimum": float(tensor.min().item()),
            "p50": float(torch.quantile(tensor, 0.50).item()),
            "p95": float(torch.quantile(tensor, 0.95).item()),
            "maximum": float(tensor.max().item()),
        }
    return result


def _apply_representative_clone(
    gaussians: Any,
    *,
    height: int,
    width: int,
    view_count: int,
    features: torch.Tensor,
    depths: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    tile_trace: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    return apply_progressive_saes(
        gaussians,
        height,
        width,
        tile_size=TILE_SIZE,
        feature_var_threshold=FEATURE_THRESHOLD,
        depth_std_threshold=DEPTH_THRESHOLD,
        features=features,
        depths=depths,
        view_count=view_count,
        materialization=MATERIALIZATION,
        decision_semantics=DECISION_SEMANTICS,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        depth_routing_semantics=DEPTH_ROUTING_SEMANTICS,
        depth_near=near,
        depth_far=far,
        materialization_guard=True,
        require_deletion_certificate=False,
        tile_trace=tile_trace,
    )[:2]


def collect_deletion_risk_audit(*, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect one fixed target-free representative deletion-risk diagnostic."""
    from scripts.ae_config import resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    input_root = FIXED_INPUT_ROOT.resolve()
    input_record, sidecar_identity, input_tree = _load_audit_input(input_root)
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        MODEL,
        dataset_name=DATASET,
        checkpoint_path=experiment.checkpoint,
        dataset_root=input_root / "sidecar",
        evaluation_index=input_root / "audit-selection.json",
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=1,
        sample_index=SAMPLE_INDEX,
        calibration_target_free=True,
        encoder_only=True,
    )
    if "image" in batch.get("target", {}):
        raise RuntimeError("deletion risk audit received target RGB before encoding")
    if batch.get("scene") != [input_record["selected_sample"]["scene"]]:
        raise RuntimeError("deletion risk audit scene does not match the fixed sidecar")
    model.eval()
    context = _context_on_device(batch, loaded_device)
    source_gaussians, features, depths = _capture_encoder_execution(model, context)
    _, view_count, _, height, width = context["image"].shape
    native_encoder_device = str(source_gaussians.means.device)
    source_gaussians = _gaussians_on_cpu(source_gaussians)
    features = features.detach().cpu()
    depths = depths.detach().cpu()
    extrinsics = context["extrinsics"].detach().cpu()
    intrinsics = context["intrinsics"].detach().cpu()
    near = context["near"].detach().cpu()
    far = context["far"].detach().cpu()
    del model, context
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    materialized = _clone_gaussians(source_gaussians)
    route_trace: list[dict[str, Any]] = []
    mask, stats = _apply_representative_clone(
        materialized,
        height=height,
        width=width,
        view_count=view_count,
        features=features,
        depths=depths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        near=near,
        far=far,
        tile_trace=route_trace,
    )
    scalar_route_trace = _scalar_trace_only(route_trace)
    trace_summary = _validate_trace(scalar_route_trace, stats, guard_enabled=True)
    _assert_numerical_invariants(stats, label="representative")
    _assert_mask_matches_trace(
        mask,
        scalar_route_trace,
        height=height,
        width=width,
        view_count=view_count,
    )
    skipped = mask.nonzero(as_tuple=False).flatten()
    retained = (~mask).nonzero(as_tuple=False).flatten()
    if skipped.numel() == 0:
        raise RuntimeError("deletion risk audit produced no sparse candidate tiles")

    poisoned = _clone_gaussians(source_gaussians)
    _poison_skipped_descriptors(poisoned, skipped)
    poison_trace: list[dict[str, Any]] = []
    poisoned_mask, poisoned_stats = _apply_representative_clone(
        poisoned,
        height=height,
        width=width,
        view_count=view_count,
        features=features,
        depths=depths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        near=near,
        far=far,
        tile_trace=poison_trace,
    )
    scalar_poison_trace = _scalar_trace_only(poison_trace)
    poison_summary = _validate_trace(
        scalar_poison_trace, poisoned_stats, guard_enabled=True
    )
    _assert_numerical_invariants(poisoned_stats, label="poisoned")
    if not torch.equal(mask, poisoned_mask):
        raise RuntimeError("poisoned skipped descriptors changed the source route")
    if stats != poisoned_stats or scalar_route_trace != scalar_poison_trace:
        raise RuntimeError("poisoned skipped descriptors changed source-only controls")
    retained_delta = _max_attribute_delta(materialized, poisoned, retained)
    if any(value != 0.0 for value in retained_delta.values()):
        raise RuntimeError("retained representative outputs depend on skipped S3")

    partition_by_tile, partition_counts = _partition_trace(scalar_route_trace)
    source_scalars = _source_scalars_from_trace(scalar_route_trace)
    oracle = guard_partition_full_s3_attribute_audit(
        source_gaussians,
        materialized,
        features=features,
        depths=depths,
        height=height,
        width=width,
        tile_size=TILE_SIZE,
        feature_threshold=FEATURE_THRESHOLD,
        depth_threshold=DEPTH_THRESHOLD,
        view_count=view_count,
        decision_semantics=DECISION_SEMANTICS,
        materialization=MATERIALIZATION,
        effective_mask=mask,
        partition_by_tile=partition_by_tile,
        source_scalars_by_tile=source_scalars,
        emit_per_tile_records=True,
    )
    tile_records = oracle.pop("per_tile_records")
    if not isinstance(tile_records, list) or len(tile_records) != len(route_trace):
        raise RuntimeError("dense-S3 oracle did not emit a complete tile record set")
    if any(record["source_scalars"] != source_scalars[(record["view_index"], record["tile_row"], record["tile_column"])] for record in tile_records):
        raise RuntimeError("dense-S3 oracle changed a source-only scalar record")

    source_trace = [
        {
            "view_index": key[0],
            "tile_row": key[1],
            "tile_column": key[2],
            "source_scalars": scalars,
        }
        for key, scalars in sorted(source_scalars.items())
    ]
    source = source_identity()
    result = {
        "schema_version": "1.0",
        "kind": AUDIT_KIND,
        "status": "COMPLETED",
        "paper_result_eligible": False,
        "quality_retry_authorized": False,
        "runtime_route_modified": False,
        "oracle_threshold_selection_used": False,
        "expected_results_accessed": False,
        "target_rgb_accessed": False,
        "model": MODEL,
        "dataset": DATASET,
        "sample_index": SAMPLE_INDEX,
        "scene": input_record["selected_sample"]["scene"],
        "context_indices": input_record["selected_sample"]["context_indices"],
        "target_indices": input_record["selected_sample"]["target_indices"],
        "execution_boundary": {
            "completed": ("S1", "S2", "S3", "S4", "SAES_risk_oracle"),
            "decoder_executed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "hardware_cycle_simulator_executed": False,
        },
        "target_rgb_provenance": {
            "native_dataloader_loaded_target_rgb": False,
            "target_image_present_before_encoder": False,
            "target_rgb_in_sidecar": False,
            "target_rgb_passed_to_model": False,
        },
        "fixed_contract": {
            "input_root": str(FIXED_INPUT_ROOT.relative_to(ROOT)),
            "seed": SEED,
            "tile_size": TILE_SIZE,
            "feature_threshold": FEATURE_THRESHOLD,
            "depth_threshold": DEPTH_THRESHOLD,
            "feature_decision_semantics": DECISION_SEMANTICS,
            "depth_routing_semantics": DEPTH_ROUTING_SEMANTICS,
            "materialization": MATERIALIZATION,
            "materialization_guard": True,
            "deletion_certificate_required": False,
            "source_fields": SOURCE_FIELDS,
            "oracle_use": "posthoc-dense-S3-diagnostic-only",
        },
        "input_provenance": {
            "audit_input_sha256": sha256_file(input_root / "audit-input.json"),
            "audit_input_tree_sha256": input_tree["tree_sha256"],
            "audit_input_manifest_sha256": input_tree["manifest_sha256"],
            "sidecar": sidecar_identity,
            "opened_source_files": input_record["opened_source_files"],
        },
        "native_execution": {
            "native_encoder_device": native_encoder_device,
            "saes_audit_device": "cpu",
            "features_shape": list(features.shape),
            "depths_shape": list(depths.shape),
            "gaussian_count": int(source_gaussians.means.shape[1]),
        },
        "selection": {
            "skipped_descriptor_count": int(skipped.numel()),
            "retained_descriptor_count": int(retained.numel()),
            "sparse_mask_sha256": _mask_sha256(mask),
        },
        "source_only_route": {
            "saes_stats": stats,
            "scalar_trace": trace_summary,
            "scalar_trace_sha256": _canonical_sha256(source_trace),
            "scalar_summary": _scalar_summary(tile_records),
        },
        "skipped_descriptor_poison_audit": {
            "route_identical": True,
            "event_statistics_identical": True,
            "scalar_trace_identical": True,
            "retained_maximum_absolute_delta": retained_delta,
            "skipped_stage3_attributes_read": False,
            "poison_trace_sha256": poison_summary["canonical_scalar_trace_sha256"],
        },
        "guard_partitions": {
            "partition_counts": partition_counts,
            "posthoc_dense_s3_oracle": oracle,
        },
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "source": source,
    }
    return result, source_trace, tile_records


def _write_json(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    result, source_trace, tile_records = collect_deletion_risk_audit(device=device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_trace_path = args.output_dir / "source-scalar-trace.json"
    tile_records_path = args.output_dir / "tile-source-oracle-records.json"
    result["trace_artifacts"] = {
        "source_scalar_trace": {
            "path": source_trace_path.name,
            "sha256": _write_json(source_trace_path, source_trace),
            "tile_count": len(source_trace),
            "contains_raw_attributes": False,
        },
        "tile_source_oracle_records": {
            "path": tile_records_path.name,
            "sha256": _write_json(tile_records_path, tile_records),
            "tile_count": len(tile_records),
            "dense_s3_posthoc_only": True,
        },
    }
    from scripts.result_record import portable_command, write_result

    result["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    write_result(result, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
