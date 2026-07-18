#!/usr/bin/env python3
"""Diagnose fixed SAES guard partitions without target RGB or rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.probe_layout import compute_lightweight_positions, compute_probe_positions
from saes.progressive_saes import apply_progressive_saes
from scripts.calibration_inputs import sha256_file
from scripts.saes_dependency_audit import _context_on_device
from scripts.saes_diagnostics import guard_partition_full_s3_attribute_audit
from scripts.saes_l1_primary_reference_attribute_audit import (
    ATTRIBUTE_TRANSPORT_MATERIALIZATION,
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
from scripts.saes_target_free_materialization_audit import (
    _capture_encoder_execution,
    _clone_gaussians,
    _gaussians_on_cpu,
    _mask_sha256,
    _max_attribute_delta,
    _poison_skipped_descriptors,
)


MATERIALIZATION = ATTRIBUTE_TRANSPORT_MATERIALIZATION
AUDIT_KIND = "saes_l1_primary_reference_guard_partition_oracle_audit"
FIXED_INPUT_ROOT = (
    ROOT
    / "outputs/ae_dl3dv_repair_diagnostics/"
    "transplat_sample0_l1_primary_reference_target_free_input_v1"
)
PARTITIONS = (
    "guard_accepted",
    "l0_rejected_l1_accepted",
    "rejected_to_full",
    "noncandidate_full",
)
TRACE_FIELDS = frozenset(
    (
        "view_index",
        "tile_row",
        "tile_column",
        "feature_variance",
        "feature_candidate",
        "depth_candidate",
        "guard_enabled",
        "guard_checks",
        "routing_level_before_materialization",
    )
)
GUARD_TRACE_FIELDS = frozenset(
    (
        "level",
        "anchor_count",
        "passed",
        "covariance_cosine_minimum",
        "harmonic_cosine_minimum",
        "opacity_distance_maximum",
        "nonprobe_s3_attribute_reads",
    )
)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_finite_scalar(value: Any) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _trace_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (record["view_index"], record["tile_row"], record["tile_column"])


def _validate_trace(
    trace: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    guard_enabled: bool,
) -> dict[str, Any]:
    """Verify scalar-only trace provenance and its event-counter agreement."""
    if not isinstance(trace, list):
        raise RuntimeError("SAES guard trace must be a list")
    if stats.get("materialization_guard_enabled") is not guard_enabled:
        raise RuntimeError("SAES guard provenance does not match the audit clone")
    if len(trace) != stats.get("total_tiles_processed"):
        raise RuntimeError("SAES trace count does not match processed tile count")

    routes = {"L0": 0, "L1": 0, "Full": 0}
    checks = {"L0": 0, "L1": 0}
    rejected = {"L0": 0, "L1": 0}
    anchor_descriptors = 0
    nonprobe_s3_reads = 0
    keys = set()
    for record in trace:
        if not isinstance(record, dict) or set(record) != TRACE_FIELDS:
            raise RuntimeError("SAES trace exposed an unsupported field")
        for field in ("view_index", "tile_row", "tile_column"):
            if not isinstance(record[field], int) or isinstance(record[field], bool):
                raise RuntimeError("SAES trace has a non-integer tile coordinate")
            if record[field] < 0:
                raise RuntimeError("SAES trace has a negative tile coordinate")
        if not _is_finite_scalar(record["feature_variance"]):
            raise RuntimeError("SAES trace has a non-finite feature scalar")
        if not isinstance(record["feature_candidate"], bool):
            raise RuntimeError("SAES trace has an invalid feature candidate")
        if record["depth_candidate"] is not None and not isinstance(
            record["depth_candidate"], bool
        ):
            raise RuntimeError("SAES trace has an invalid depth candidate")
        if record["guard_enabled"] is not guard_enabled:
            raise RuntimeError("SAES trace guard flag does not match the clone")
        if record["routing_level_before_materialization"] not in routes:
            raise RuntimeError("SAES trace has an unsupported route")
        if not isinstance(record["guard_checks"], list):
            raise RuntimeError("SAES trace guard checks must be a list")
        key = _trace_key(record)
        if key in keys:
            raise RuntimeError("SAES trace contains duplicate tiles")
        keys.add(key)
        routes[record["routing_level_before_materialization"]] += 1

        for guard in record["guard_checks"]:
            if not isinstance(guard, dict) or set(guard) != GUARD_TRACE_FIELDS:
                raise RuntimeError("SAES trace exposed an unsupported guard field")
            if guard["level"] not in checks:
                raise RuntimeError("SAES trace has an unsupported guard level")
            if not isinstance(guard["anchor_count"], int) or isinstance(
                guard["anchor_count"], bool
            ):
                raise RuntimeError("SAES trace has an invalid guard anchor count")
            if guard["anchor_count"] <= 0 or not isinstance(guard["passed"], bool):
                raise RuntimeError("SAES trace has invalid guard provenance")
            for field in (
                "covariance_cosine_minimum",
                "harmonic_cosine_minimum",
                "opacity_distance_maximum",
            ):
                if not _is_finite_scalar(guard[field]):
                    raise RuntimeError("SAES trace has a non-finite guard scalar")
            if not isinstance(guard["nonprobe_s3_attribute_reads"], int) or isinstance(
                guard["nonprobe_s3_attribute_reads"], bool
            ):
                raise RuntimeError("SAES trace has an invalid S3-read counter")
            if guard["nonprobe_s3_attribute_reads"] < 0:
                raise RuntimeError("SAES trace has a negative S3-read counter")
            checks[guard["level"]] += 1
            rejected[guard["level"]] += int(not guard["passed"])
            anchor_descriptors += guard["anchor_count"]
            nonprobe_s3_reads += guard["nonprobe_s3_attribute_reads"]

    expected_routes = {
        "L0": stats.get("level0_tiles"),
        "L1": stats.get("level1_tiles"),
        "Full": stats.get("full_tiles"),
    }
    if routes != expected_routes or sum(routes.values()) != stats.get(
        "total_tiles_processed"
    ):
        raise RuntimeError("SAES trace routes disagree with runtime statistics")

    if guard_enabled:
        expected = {
            "L0": stats.get("l0_guard_checks"),
            "L1": stats.get("l1_guard_checks"),
        }
        expected_rejected = {
            "L0": stats.get("l0_guard_rejections"),
            "L1": stats.get("l1_guard_rejections"),
        }
        if checks != expected or rejected != expected_rejected:
            raise RuntimeError("SAES trace guard counters disagree with runtime statistics")
        if stats.get("l1_guard_attempts_after_l0_rejection") != rejected["L0"]:
            raise RuntimeError("SAES trace L0 rejection count disagrees with L1 attempts")
        if stats.get("guard_anchor_attribute_reads") != 3 * anchor_descriptors:
            raise RuntimeError("SAES trace guard anchor reads disagree with runtime statistics")
        if stats.get("guard_nonprobe_s3_attribute_reads") != nonprobe_s3_reads:
            raise RuntimeError("SAES trace S3 reads disagree with runtime statistics")
    else:
        disabled_counters = (
            "l0_guard_checks",
            "l1_guard_checks",
            "l0_guard_rejections",
            "l1_guard_rejections",
            "l1_guard_attempts_after_l0_rejection",
            "guard_anchor_attribute_reads",
            "guard_nonprobe_s3_attribute_reads",
        )
        if (
            any(checks.values())
            or any(rejected.values())
            or anchor_descriptors
            or any(stats.get(key) != 0 for key in disabled_counters)
        ):
            raise RuntimeError("disabled SAES guard emitted guard work")

    if nonprobe_s3_reads != 0:
        raise RuntimeError("SAES guard read a non-probe S3 attribute")
    return {
        "canonical_scalar_trace_sha256": _canonical_json_sha256(trace),
        "tile_count": len(trace),
        "pre_materialization_routes": routes,
        "guard_checks": checks,
        "guard_rejections": rejected,
        "guard_anchor_descriptors": anchor_descriptors,
        "guard_nonprobe_s3_attribute_reads": nonprobe_s3_reads,
    }


def _assert_shadow_candidates_match(
    guarded_trace: list[dict[str, Any]], shadow_trace: list[dict[str, Any]]
) -> None:
    if len(guarded_trace) != len(shadow_trace):
        raise RuntimeError("guarded and shadow traces have different tile counts")
    for guarded, shadow in zip(guarded_trace, shadow_trace):
        if _trace_key(guarded) != _trace_key(shadow):
            raise RuntimeError("guarded and shadow traces have different tile order")
        for field in ("feature_variance", "feature_candidate"):
            if guarded[field] != shadow[field]:
                raise RuntimeError("guarded and shadow traces changed an S1 candidate")
        guarded_depth = guarded["depth_candidate"]
        shadow_depth = shadow["depth_candidate"]
        if guarded_depth is not None and shadow_depth is not None:
            if guarded_depth != shadow_depth:
                raise RuntimeError("guarded and shadow traces changed an S2 candidate")
        elif shadow_depth is not None:
            raise RuntimeError("shadow evaluated depth when guarded routing did not")
        elif guarded_depth is not None and not (
            shadow["feature_candidate"]
            and shadow["routing_level_before_materialization"] == "L0"
        ):
            raise RuntimeError("shadow omitted a depth candidate without first-hit short-circuit")
        if shadow["guard_enabled"] is not False or shadow["guard_checks"]:
            raise RuntimeError("unguarded shadow unexpectedly executed a guard")


def _partition_trace(
    trace: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int, int], str], dict[str, int]]:
    labels: dict[tuple[int, int, int], str] = {}
    counts = {name: 0 for name in PARTITIONS}
    for record in trace:
        route = record["routing_level_before_materialization"]
        checks = record["guard_checks"]
        l0_rejected = any(
            guard["level"] == "L0" and not guard["passed"] for guard in checks
        )
        l1_accepted = any(
            guard["level"] == "L1" and guard["passed"] for guard in checks
        )
        any_rejected = any(not guard["passed"] for guard in checks)
        if l0_rejected and l1_accepted and route == "L1":
            label = "l0_rejected_l1_accepted"
        elif route in {"L0", "L1"} and checks and not any_rejected:
            label = "guard_accepted"
        elif route == "Full" and any_rejected:
            label = "rejected_to_full"
        elif route == "Full" and not checks:
            label = "noncandidate_full"
        else:
            raise RuntimeError("SAES trace has an unsupported guard partition")
        key = _trace_key(record)
        labels[key] = label
        counts[label] += 1
    return labels, counts


def _shadow_partition_labels(
    partition_by_tile: dict[tuple[int, int, int], str]
) -> dict[tuple[int, int, int], str]:
    """Keep canonical guard labels disjoint in the non-claim shadow audit."""
    if set(partition_by_tile.values()) - set(PARTITIONS):
        raise RuntimeError("shadow audit received an unsupported canonical partition")
    return dict(partition_by_tile)


def _mask_route(
    mask: torch.Tensor,
    *,
    view: int,
    tile_row: int,
    tile_column: int,
    height: int,
    width: int,
    view_count: int,
) -> str:
    gaussian_count = int(mask.numel())
    pixels = view_count * height * width
    if gaussian_count % pixels:
        raise RuntimeError("SAES mask does not match the fixed Gaussian layout")
    primitives_per_pixel = gaussian_count // pixels
    tile_y = tile_row * TILE_SIZE
    tile_x = tile_column * TILE_SIZE
    skipped_per_slot = []
    for slot in range(primitives_per_pixel):
        indices = torch.tensor(
            [
                ((view * height * width + (tile_y + local_y) * width + tile_x + local_x)
                * primitives_per_pixel)
                + slot
                for local_y in range(TILE_SIZE)
                for local_x in range(TILE_SIZE)
            ],
            device=mask.device,
            dtype=torch.long,
        )
        skipped_per_slot.append(int(mask[indices].sum().item()))
    if len(set(skipped_per_slot)) != 1:
        raise RuntimeError("SAES mask has inconsistent primitive slots")
    skipped = skipped_per_slot[0]
    if skipped == 0:
        return "Full"
    if skipped == TILE_SIZE * TILE_SIZE - len(compute_probe_positions(TILE_SIZE)):
        return "L0"
    if skipped == TILE_SIZE * TILE_SIZE - len(compute_lightweight_positions(TILE_SIZE)):
        return "L1"
    raise RuntimeError("SAES mask has an unsupported tile route")


def _assert_mask_matches_trace(
    mask: torch.Tensor,
    trace: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    view_count: int,
) -> None:
    for record in trace:
        actual = _mask_route(
            mask,
            view=record["view_index"],
            tile_row=record["tile_row"],
            tile_column=record["tile_column"],
            height=height,
            width=width,
            view_count=view_count,
        )
        if actual != record["routing_level_before_materialization"]:
            raise RuntimeError("SAES mask diverged from its pre-materialization trace")


def _assert_numerical_invariants(stats: dict[str, Any], *, label: str) -> None:
    if stats.get("covariance_psd_violations"):
        raise RuntimeError(f"{label} SAES execution violated covariance PSD")
    if stats.get("guard_nonprobe_s3_attribute_reads"):
        raise RuntimeError(f"{label} SAES guard read a skipped S3 descriptor")
    if stats.get("opacity_transmittance_error_max"):
        raise RuntimeError(f"{label} SAES execution violated opacity transmittance")
    if stats.get("assignment_weight_sum_error_max", 0.0) > 1.0e-5:
        raise RuntimeError(f"{label} SAES execution has non-normalized assignments")
    if stats.get("adapter_offset_transport_fallback_tiles"):
        raise RuntimeError(f"{label} SAES execution fell back after adapter transport")


def _run_saes_clone(
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
    materialization_guard: bool,
    tile_trace: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    mask, stats, _ = apply_progressive_saes(
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
        materialization_guard=materialization_guard,
        tile_trace=tile_trace,
    )
    return mask, stats


def collect_guard_partition_audit(
    *, device: torch.device, input_root: Path | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the single fixed target-free guard-partition diagnostic."""
    fixed_input_root = FIXED_INPUT_ROOT.resolve()
    if input_root is not None and input_root.resolve() != fixed_input_root:
        raise RuntimeError("guard partition audit only accepts its fixed target-free input")
    input_root = fixed_input_root
    from scripts.ae_config import resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    source = source_identity()
    if source.get("source") == "git" and source.get("git_dirty") is not False:
        raise RuntimeError("guard partition audit requires a clean source tree before execution")
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
        raise RuntimeError("guard partition audit received target RGB before encoding")
    if batch.get("scene") != [input_record["selected_sample"]["scene"]]:
        raise RuntimeError("guard partition audit scene does not match the fixed sidecar")
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

    guarded = _clone_gaussians(source_gaussians)
    guarded_trace: list[dict[str, Any]] = []
    guarded_mask, guarded_stats = _run_saes_clone(
        guarded,
        height=height,
        width=width,
        view_count=view_count,
        features=features,
        depths=depths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        near=near,
        far=far,
        materialization_guard=True,
        tile_trace=guarded_trace,
    )
    guarded_trace_summary = _validate_trace(
        guarded_trace, guarded_stats, guard_enabled=True
    )
    _assert_numerical_invariants(guarded_stats, label="guarded")
    _assert_mask_matches_trace(
        guarded_mask,
        guarded_trace,
        height=height,
        width=width,
        view_count=view_count,
    )
    skipped = guarded_mask.nonzero(as_tuple=False).flatten()
    retained = (~guarded_mask).nonzero(as_tuple=False).flatten()
    if skipped.numel() == 0:
        raise RuntimeError("fixed guard partition audit produced no sparse work")

    shadow = _clone_gaussians(source_gaussians)
    shadow_trace: list[dict[str, Any]] = []
    shadow_mask, shadow_stats = _run_saes_clone(
        shadow,
        height=height,
        width=width,
        view_count=view_count,
        features=features,
        depths=depths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        near=near,
        far=far,
        materialization_guard=False,
        tile_trace=shadow_trace,
    )
    shadow_trace_summary = _validate_trace(
        shadow_trace, shadow_stats, guard_enabled=False
    )
    _assert_numerical_invariants(shadow_stats, label="shadow")
    _assert_shadow_candidates_match(guarded_trace, shadow_trace)
    _assert_mask_matches_trace(
        shadow_mask,
        shadow_trace,
        height=height,
        width=width,
        view_count=view_count,
    )

    poisoned = _clone_gaussians(source_gaussians)
    _poison_skipped_descriptors(poisoned, skipped)
    poison_trace: list[dict[str, Any]] = []
    poisoned_mask, poisoned_stats = _run_saes_clone(
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
        materialization_guard=True,
        tile_trace=poison_trace,
    )
    poison_trace_summary = _validate_trace(poison_trace, poisoned_stats, guard_enabled=True)
    _assert_numerical_invariants(poisoned_stats, label="poisoned")
    if not torch.equal(guarded_mask, poisoned_mask):
        raise RuntimeError("poisoned descriptors changed the guarded sparse mask")
    if guarded_stats != poisoned_stats:
        raise RuntimeError("poisoned descriptors changed guarded event statistics")
    if guarded_trace != poison_trace:
        raise RuntimeError("poisoned descriptors changed the scalar guard trace")
    retained_delta = _max_attribute_delta(guarded, poisoned, retained)
    if any(value != 0.0 for value in retained_delta.values()):
        raise RuntimeError("guarded retained descriptors depend on skipped raw S3 attributes")

    partition_by_tile, partition_counts = _partition_trace(guarded_trace)
    canonical_oracle = guard_partition_full_s3_attribute_audit(
        source_gaussians,
        guarded,
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
        effective_mask=guarded_mask,
        partition_by_tile=partition_by_tile,
    )
    shadow_oracle = guard_partition_full_s3_attribute_audit(
        source_gaussians,
        shadow,
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
        effective_mask=shadow_mask,
        partition_by_tile=_shadow_partition_labels(partition_by_tile),
    )

    return (
        {
            "schema_version": "1.0",
            "kind": AUDIT_KIND,
            "status": "COMPLETED",
            "paper_result_eligible": False,
            "quality_retry_authorized": False,
            "expected_results_accessed": False,
            "target_rgb_accessed": False,
            "model": MODEL,
            "dataset": DATASET,
            "sample_index": SAMPLE_INDEX,
            "scene": input_record["selected_sample"]["scene"],
            "context_indices": input_record["selected_sample"]["context_indices"],
            "target_indices": input_record["selected_sample"]["target_indices"],
            "execution_boundary": {
                "completed": ("S1", "S2", "S3", "S4", "SAES_guard_partition_audit"),
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
                "input_root_is_fixed": True,
                "seed": SEED,
                "tile_size": TILE_SIZE,
                "feature_threshold": FEATURE_THRESHOLD,
                "depth_threshold": DEPTH_THRESHOLD,
                "feature_decision_semantics": DECISION_SEMANTICS,
                "depth_routing_semantics": DEPTH_ROUTING_SEMANTICS,
                "materialization": MATERIALIZATION,
                "materialization_guard": "canonical-true-shadow-false-poisoned-true",
                "l1_depth_reference": "primary-probes",
                "l1_anchor_layout": "2K-native-selected-anchors",
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
                "guarded_skipped_mask_sha256": _mask_sha256(guarded_mask),
                "shadow_skipped_mask_sha256": _mask_sha256(shadow_mask),
            },
            "guarded_execution": {
                "saes_stats": guarded_stats,
                "scalar_trace": guarded_trace_summary,
            },
            "unguarded_shadow": {
                "claim_path": False,
                "saes_stats": shadow_stats,
                "scalar_trace": shadow_trace_summary,
            },
            "skipped_descriptor_poison_audit": {
                "poison_value": 1.0e4,
                "route_identical": True,
                "event_statistics_identical": True,
                "scalar_trace_identical": True,
                "retained_maximum_absolute_delta": retained_delta,
                "skipped_stage3_attributes_read": False,
                "poison_trace_sha256": poison_trace_summary["canonical_scalar_trace_sha256"],
            },
            "guard_partitions": {
                "partition_counts": partition_counts,
                "partition_semantics": {
                    "guard_accepted": "sparse L0/L1 output with no failed guard",
                    "l0_rejected_l1_accepted": (
                        "reported even when zero; structurally unreachable for the "
                        "current nested L1 anchors and identical guard predicate"
                    ),
                    "rejected_to_full": (
                        "canonical Full passthrough is not an oracle sample; "
                        "the shadow's same label is diagnostic-only"
                    ),
                    "noncandidate_full": "Full without a guard failure",
                },
                "canonical_full_s3_attribute_oracle": canonical_oracle,
                "shadow_counterfactual": {
                    "claim_path": False,
                    "shadow_only": True,
                    "partition_labels": "preserved from the canonical guarded trace",
                    "rejected_to_full_interpretation": (
                        "shadow sparse routes are diagnostic-only and cannot authorize "
                        "a guard or quality change"
                    ),
                    "full_s3_attribute_oracle": shadow_oracle,
                },
            },
            "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
            "source": source,
        },
        guarded_trace,
        shadow_trace,
    )


def _write_scalar_trace(path: Path, trace: list[dict[str, Any]]) -> str:
    path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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

    record, guarded_trace, shadow_trace = collect_guard_partition_audit(device=device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    guarded_trace_path = args.output_dir / "guarded-scalar-trace.json"
    shadow_trace_path = args.output_dir / "shadow-scalar-trace.json"
    record["trace_artifacts"] = {
        "guarded_scalar_trace": {
            "path": guarded_trace_path.name,
            "sha256": _write_scalar_trace(guarded_trace_path, guarded_trace),
            "tile_count": len(guarded_trace),
        },
        "shadow_scalar_trace": {
            "path": shadow_trace_path.name,
            "sha256": _write_scalar_trace(shadow_trace_path, shadow_trace),
            "tile_count": len(shadow_trace),
        },
        "contains_raw_anchor_or_nonprobe_attributes": False,
    }
    from scripts.result_record import portable_command, write_result

    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
