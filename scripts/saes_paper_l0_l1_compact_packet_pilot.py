#!/usr/bin/env python3
"""Run one target-free normalized L0/L1 compact packet quality pilot.

The pilot is development-only. It uses the paper's T=4 four-probe L0 set and
an explicit adaptive 15-anchor L1 engineering closure, constructs non-probe
contributors from selected anchors, and keeps a failed preflight on native
Full. It does not claim sparse S2/S3 savings, timing, or paper-result
eligibility.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.probe_first_schedule import (
    ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    build_incremental_probe_first_plan,
)
from saes.adaptive_l1_calibration import load_frozen_threshold
from saes.adaptive_l1_v4_attribute_calibration import (
    load_frozen_threshold as load_v4_attribute_loo_threshold,
)
from saes.guarded_selected_route import (
    ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
    ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
)
from saes.packed_l0_l1_materializer import (
    CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
)
from saes.progressive_saes import PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS
from saes.sparse_gaussian_consumer import PackedGaussianAttributes
from scripts.saes_incremental_selected_output_audit import (
    DEPTH_THRESHOLD,
    FEATURE_THRESHOLD,
    TILE_SIZE,
    _attribute_equivalence,
    _capture_guarded_incremental_packed_adapter,
    _capture_s1_s2_without_dense_adapter,
    _packed_attributes_to_device,
    _release_cuda_cache,
)
from scripts.saes_selected_output_quality_gate import (
    _mean_metrics,
    _quality_verdict,
    _view_metrics,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution
from scripts.saes_sparse_consumer_render_audit import (
    _target_cameras,
    render_packed_gaussians,
)
from scripts.saes_sparse_packet_quality_pilot import (
    DATASET,
    MODEL,
    ROOT as PACKET_ROOT,
    SAMPLE_INDEX,
    SEED,
    _render_dense_baseline,
    _take_target_rgb_for_metrics,
)


PILOT_KIND = "saes_paper_normalized_l0_l1_15_anchor_compact_packet_quality_pilot"
DECISION_SEMANTICS = PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS
L1_ANCHOR_SEMANTICS = ADAPTIVE_L1_15_ANCHOR_SEMANTICS
COMPACT_AGGREGATION = CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION
DEFAULT_CALIBRATION_RECORD = (
    ROOT
    / "outputs/ae_dl3dv_local_repro_v5"
    / "saes_l1_absolute_residual_calibration_v15.json"
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _paper_identity(
    calibration: Mapping[str, Any],
    v4_attribute_loo_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    v4_loo_enabled = v4_attribute_loo_calibration is not None
    identity = {
        "schema_version": "saes-paper-normalized-l1-15-adaptive-packet-pilot-v8",
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": DEPTH_THRESHOLD,
        "decision_semantics": DECISION_SEMANTICS,
        "feature_scale_interpretation": (
            "unit-normalized-probe-vector-standard-deviation-engineering-closure"
        ),
        "l0_anchor_count": 4,
        "l1_anchor_count": 15,
        "l1_anchor_semantics": L1_ANCHOR_SEMANTICS,
        "l1_retention_interpretation": (
            "engineering-legacy12-plus-three-center-anchors-not-specified-by-paper"
        ),
        "nonzero_policy": (
            ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
            if v4_loo_enabled
            else ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY
        ),
        "adaptive_l1_absolute_residual_calibration_sha256": calibration["sha256"],
        "adaptive_l1_absolute_residual_maximum": calibration["threshold_value"],
        "aggregation": COMPACT_AGGREGATION,
        "coverage_certificate": (
            "selected-only-direct-geometry-spatial-attribute-l0-depth-continuity-intra-tile-2sigma-v1"
        ),
        "quality_tolerances": {
            "psnr_max_drop": 0.15,
            "ssim_max_drop": 0.005,
            "lpips_max_increase": 0.005,
        },
    }
    if v4_loo_enabled:
        identity.update(
            {
                "selected_anchor_v4_attribute_loo_calibration_sha256": v4_attribute_loo_calibration[
                    "sha256"
                ],
                "selected_anchor_v4_attribute_loo_maximum_risk": v4_attribute_loo_calibration[
                    "threshold_value"
                ],
            }
        )
    return {**identity, "sha256": _canonical_sha256(identity)}


def _route_summary(capture: Mapping[str, Any]) -> dict[str, Any]:
    route = capture["guarded_route"]
    preflight = capture["compact_materialization_preflight"]
    if preflight is None:
        raise RuntimeError("compact packet pilot has no materialization preflight")
    def compact_events(events: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in events.items()
            if key != "full_passthrough_dense_slots"
        }

    return {
        "base_guard_route": compact_events(capture["base_guarded_route"].events),
        "final_route": compact_events(route.events),
        "preflight": preflight.events,
        "final_output_descriptor_count": int(route.selected_output_mask.sum().item()),
        "raw_head_request_descriptor_count": int(route.raw_head_request_mask.sum().item()),
    }


def _route_diagnostics(plan: Any, capture: Mapping[str, Any]) -> dict[str, Any]:
    """Persist scalar route evidence without serializing tile-local attributes."""
    scores = sorted(float(record["feature_score"]) for record in plan.tile_trace)
    depth_uniform = sum(bool(record["depth_uniform"]) for record in plan.tile_trace)
    pre_guard = {"L0": 0, "L1": 0, "Full": 0}
    for record in plan.tile_trace:
        pre_guard[str(record["pre_guard_route"])] += 1
    base_route = capture["base_guarded_route"]
    guard_failures = {
        "materialization": 0,
        "probe_cross_check": 0,
        "context_safety": {},
    }
    for tile in base_route.tile_trace:
        for check in tile["guard_checks"]:
            if check["materialization"]["passed"] is not True:
                guard_failures["materialization"] += 1
            cross = check["probe_cross_check"]
            if cross["checked"] and cross["passed"] is not True:
                guard_failures["probe_cross_check"] += 1
            context = check["context_safety"]
            if context["passed"] is not True:
                reason = str(context["reason"])
                reasons = guard_failures["context_safety"]
                reasons[reason] = reasons.get(reason, 0) + 1

    def quantile(fraction: float) -> float:
        return scores[round((len(scores) - 1) * fraction)]

    return {
        "tile_count": len(scores),
        "feature_score": {
            "minimum": quantile(0.0),
            "p50": quantile(0.50),
            "p95": quantile(0.95),
            "maximum": quantile(1.0),
            "below_fixed_tau_f": sum(score < FEATURE_THRESHOLD for score in scores),
        },
        "depth_uniform_tile_count": depth_uniform,
        "pre_guard_route_counts": pre_guard,
        "guard_failures": guard_failures,
    }


def _posthoc_dense_assignment_mixture_reference(
    *,
    model: Any,
    dense_gaussians: Any,
    packet_color: torch.Tensor,
    final_packed: PackedGaussianAttributes,
    planning: Mapping[str, torch.Tensor],
    context: Mapping[str, Any],
    target_cameras: Mapping[str, torch.Tensor],
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Compare a committed packet to the legacy dense mixture without target RGB.

    Dense attributes are a posthoc reference only.  They are constructed after
    the compact packet decoder output has been fixed and cannot influence its
    route, assignments, or materialized attributes.
    """
    from saes.progressive_saes import apply_progressive_saes
    from scripts.saes_target_free_materialization_audit import _clone_gaussians

    reference = _clone_gaussians(dense_gaussians)
    height, width = image_shape
    modified, stats, _continue = apply_progressive_saes(
        reference,
        height,
        width,
        tile_size=TILE_SIZE,
        gpp=1,
        feature_var_threshold=FEATURE_THRESHOLD,
        depth_std_threshold=DEPTH_THRESHOLD,
        features=planning["features"].to(reference.means.device),
        depths=planning["depths"].to(reference.means.device),
        view_count=int(planning["features"].shape[1]),
        materialization="representative",
        decision_semantics=DECISION_SEMANTICS,
        beta_x=0.50,
        beta_f=0.10,
        beta_d=1.00,
        context_extrinsics=context["extrinsics"],
        context_intrinsics=context["intrinsics"],
        ray_depth_mode="euclidean",
        materialization_guard=False,
        context_safety_guard=False,
        require_deletion_certificate=False,
    )
    reference_color = _render_dense_baseline(
        model,
        reference,
        target=target_cameras,
        image_shape=image_shape,
    )
    delta = (packet_color - reference_color).abs()
    attribute_equivalence = _attribute_equivalence(
        final_packed, reference, final_packed.dense_slots
    )
    return {
        "posthoc_dense_s3_reference": True,
        "candidate_inputs_influenced_by_reference": False,
        "target_rgb_accessed": False,
        "dense_modified_count": int(modified.sum().item()),
        "dense_route_counts": {
            "L0": int(stats["level0_tiles"]),
            "L1": int(stats["level1_tiles"]),
            "Full": int(stats["full_tiles"]),
        },
        "selected_anchor_attribute_equivalence": attribute_equivalence,
        "packet_vs_dense_mixture_render": {
            "mean_absolute_delta": float(delta.mean().item()),
            "maximum_absolute_delta": float(delta.max().item()),
        },
    }


def _posthoc_dense_domain_coverage_reference(
    *,
    dense_gaussians: Any,
    final_packed: PackedGaussianAttributes,
    route: Any,
    context: Mapping[str, torch.Tensor],
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Audit committed compact tiles against dense S3 domains without targets.

    This runs only after the final packet is materialized. Dense descriptors
    provide a diagnostic reference and cannot influence routing, assignments,
    coverage closure, or the committed packet.
    """
    from saes.projected_domain_coverage_audit import audit_projected_dense_domain_coverage

    height, width = image_shape
    dense_means = getattr(dense_gaussians, "means", None)
    dense_covariances = getattr(dense_gaussians, "covariances", None)
    dense_opacities = getattr(dense_gaussians, "opacities", None)
    if (
        not torch.is_tensor(dense_means)
        or not torch.is_tensor(dense_covariances)
        or not torch.is_tensor(dense_opacities)
        or dense_means.ndim != 3
        or dense_means.shape[0] != 1
        or dense_covariances.shape != (*dense_means.shape[:2], 3, 3)
        or dense_opacities.shape != dense_means.shape[:2]
    ):
        raise RuntimeError("compact packet pilot dense reference has an invalid layout")
    slot_to_index = {
        int(slot): index
        for index, slot in enumerate(final_packed.dense_slots.detach().cpu().tolist())
    }
    route_records = route.tile_trace
    primary = ((0, 0), (0, 3), (3, 0), (3, 3))
    l1_anchor_semantics = route.events.get("l1_anchor_semantics")
    if l1_anchor_semantics != ADAPTIVE_L1_15_ANCHOR_SEMANTICS:
        raise RuntimeError("compact packet pilot requires the adaptive L1-15 layout")
    aggregates = {
        "compact_tile_count": 0,
        "valid_tile_count": 0,
        "invalid_tile_count": 0,
        "dense_descriptor_count": 0,
        "candidate_descriptor_count": 0,
        "active_dense_descriptor_count": 0,
        "hole_count": 0,
        "dense_optical_mass": 0.0,
        "contained_optical_mass": 0.0,
        "reason_counts": {},
    }
    for record in route_records:
        if record.get("final_route") not in {"L0", "L1"}:
            continue
        view = int(record["view"])
        tile_y = int(record["tile_y"])
        tile_x = int(record["tile_x"])
        if record.get("final_route") == "L0":
            anchors = primary
        else:
            raw_anchors = record.get("retained_local_positions")
            if not isinstance(raw_anchors, list) or len(raw_anchors) != 15:
                raise RuntimeError("compact packet pilot has no bound adaptive L1 anchors")
            try:
                anchors = tuple((int(position[0]), int(position[1])) for position in raw_anchors)
            except (IndexError, TypeError, ValueError) as error:
                raise RuntimeError("compact packet pilot has invalid adaptive L1 anchors") from error
            if (
                len(set(anchors)) != 15
                or anchors[:4] != primary
                or any(not 0 <= row < 4 or not 0 <= column < 4 for row, column in anchors)
            ):
                raise RuntimeError("compact packet pilot adaptive L1 anchors violate their contract")
        anchor_slots = [
            view * height * width
            + (tile_y * 4 + local_y) * width
            + tile_x * 4
            + local_x
            for local_y, local_x in anchors
        ]
        nonprobe_slots = [
            view * height * width + (tile_y * 4 + local_y) * width + tile_x * 4 + local_x
            for local_y in range(4)
            for local_x in range(4)
            if (local_y, local_x) not in set(anchors)
        ]
        if any(slot not in slot_to_index for slot in anchor_slots):
            raise RuntimeError("compact packet pilot lacks a retained compact anchor")
        candidate_indices = torch.tensor(
            [slot_to_index[slot] for slot in anchor_slots],
            device=final_packed.means.device,
            dtype=torch.long,
        )
        dense_indices = torch.tensor(
            nonprobe_slots,
            device=dense_means.device,
            dtype=torch.long,
        )
        audit = audit_projected_dense_domain_coverage(
            dense_means=dense_means[0, dense_indices],
            dense_covariances=dense_covariances[0, dense_indices],
            dense_opacities=dense_opacities[0, dense_indices],
            candidate_means=final_packed.means[candidate_indices],
            candidate_covariances=final_packed.covariances[candidate_indices],
            candidate_opacities=final_packed.opacities[candidate_indices],
            context_extrinsic=context["extrinsics"][0, view],
            context_intrinsic=context["intrinsics"][0, view],
        )
        aggregates["compact_tile_count"] += 1
        aggregates["dense_descriptor_count"] += audit.dense_descriptor_count
        aggregates["candidate_descriptor_count"] += audit.candidate_descriptor_count
        if not audit.valid:
            aggregates["invalid_tile_count"] += 1
            reason = audit.reason or "unknown"
            aggregates["reason_counts"][reason] = (
                aggregates["reason_counts"].get(reason, 0) + 1
            )
            continue
        aggregates["valid_tile_count"] += 1
        aggregates["active_dense_descriptor_count"] += audit.active_dense_descriptor_count
        aggregates["hole_count"] += int(audit.hole_count or 0)
        aggregates["dense_optical_mass"] += float(audit.dense_optical_mass or 0.0)
        aggregates["contained_optical_mass"] += float(audit.contained_optical_mass or 0.0)
    mass = aggregates["dense_optical_mass"]
    active = aggregates["active_dense_descriptor_count"]
    aggregates["mass_weighted_recall"] = (
        aggregates["contained_optical_mass"] / mass if mass > 0.0 else None
    )
    aggregates["count_recall"] = (
        (active - aggregates["hole_count"]) / active if active > 0 else None
    )
    return {
        "posthoc_dense_s3_reference": True,
        "candidate_inputs_influenced_by_reference": False,
        "target_rgb_accessed": False,
        "audit_kind": "actual-dense-nonprobe-domain-coverage-after-commit",
        "per_tile_records_persisted": False,
        "aggregate": aggregates,
    }


def collect_paper_compact_packet_pilot(
    *,
    device: torch.device,
    posthoc_domain_audit_only: bool = False,
    calibration_record: Path = DEFAULT_CALIBRATION_RECORD,
    v4_attribute_loo_calibration_record: Path | None = None,
) -> dict[str, Any]:
    """Run one packet pilot, optionally stopping at the target-free domain audit."""
    if not isinstance(posthoc_domain_audit_only, bool):
        raise TypeError("posthoc domain audit mode must be boolean")
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    experiment = resolve_experiment(MODEL, DATASET, PACKET_ROOT)
    selection = resolve_claim_selection(MODEL, DATASET, PACKET_ROOT)
    checkpoint_sha256 = cached_sha256_file(experiment.checkpoint)
    calibration = load_frozen_threshold(
        calibration_record,
        evaluation_sample_index=SAMPLE_INDEX,
        checkpoint_sha256=checkpoint_sha256,
        source_index_sha256=selection.source_index_sha256,
        sample_selection_sha256=selection.sample_selection_sha256,
    )
    v4_attribute_loo_calibration = (
        load_v4_attribute_loo_threshold(
            v4_attribute_loo_calibration_record,
            evaluation_sample_index=SAMPLE_INDEX,
            checkpoint_sha256=checkpoint_sha256,
            source_index_sha256=selection.source_index_sha256,
            sample_selection_sha256=selection.sample_selection_sha256,
            v15_calibration_sha256=calibration["sha256"],
        )
        if v4_attribute_loo_calibration_record is not None
        else None
    )
    paper_identity = _paper_identity(calibration, v4_attribute_loo_calibration)
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
        sample_index=SAMPLE_INDEX,
    )
    model.eval()
    from src.model.types import Gaussians

    context = {
        key: value.to(loaded_device) if torch.is_tensor(value) else value
        for key, value in batch["context"].items()
    }
    if context["image"].shape[0] != 1:
        raise RuntimeError("compact packet pilot requires batch size one")
    _, _views, _, height, width = context["image"].shape

    with strict_fp32_convolution_execution() as numerical_execution:
        planning = _capture_s1_s2_without_dense_adapter(model, context)
        _release_cuda_cache(loaded_device)
        plan = build_incremental_probe_first_plan(
            planning["features"],
            planning["depths"],
            height=height,
            width=width,
            tile_size=TILE_SIZE,
            feature_threshold=FEATURE_THRESHOLD,
            depth_threshold=DEPTH_THRESHOLD,
            decision_semantics=DECISION_SEMANTICS,
            l1_anchor_semantics=L1_ANCHOR_SEMANTICS,
        )
        capture = _capture_guarded_incremental_packed_adapter(
            model,
            context,
            plan=plan,
            compact_nonzero_materialization=True,
            compact_execution_policy=(
                ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
                if v4_attribute_loo_calibration is not None
                else ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY
            ),
            adaptive_l1_maximum_leave_one_out_residual=calibration["threshold_value"],
            selected_anchor_v4_attribute_loo_maximum_risk=(
                v4_attribute_loo_calibration["threshold_value"]
                if v4_attribute_loo_calibration is not None
                else None
            ),
        )
        preflight = capture["compact_materialization_preflight"]
        if preflight is None:
            raise RuntimeError("compact packet pilot did not retain materialization evidence")
        final_packed = capture["final_packed"]
        if not isinstance(final_packed, PackedGaussianAttributes):
            raise RuntimeError("compact packet pilot did not produce packed attributes")
        compact_anchor_count = int(preflight.update_dense_slots.numel())
        route_summary = _route_summary(capture)
        route_diagnostics = _route_diagnostics(plan, capture)
        if compact_anchor_count == 0:
            return {
                "schema_version": "1.0",
                "kind": PILOT_KIND,
                "status": "NO_COMPACT_TILES",
                "paper_result_eligible": False,
                "model": MODEL,
                "dataset": DATASET,
                "sample_index": SAMPLE_INDEX,
                "scene": str(batch["scene"][0]),
                "target_rgb_provenance": {
                    "loaded_by_native_dataloader": True,
                    "accessed_before_compact_commit": False,
                    "accessed_for_metrics": False,
                },
                "execution_boundary": {
                    "source_bound_packet_adapter_executed": True,
                    "compact_nonzero_materialization_enabled": True,
                    "nonzero_direct_deletion": False,
                    "whole_pipeline_s2_s3_sparse_execution_verified": False,
                    "timing_claim": False,
                },
                "paper_identity": paper_identity,
                "adaptive_l1_calibration": calibration,
                "adaptive_l1_v4_attribute_loo_calibration": v4_attribute_loo_calibration,
                "route_plan": plan.events,
                "compact_route": route_summary,
                "route_diagnostics": route_diagnostics,
                "numerical_execution": numerical_execution,
                "checkpoint_sha256": checkpoint_sha256,
                "source": source_identity(),
            }
        renderable = _packed_attributes_to_device(final_packed, loaded_device)
        if posthoc_domain_audit_only:
            with torch.no_grad():
                baseline_gaussians = model.encoder(context, 0, deterministic=True)
            posthoc_dense_reference = _posthoc_dense_domain_coverage_reference(
                dense_gaussians=baseline_gaussians,
                final_packed=renderable,
                route=capture["guarded_route"],
                context=context,
                image_shape=(height, width),
            )
            return {
                "schema_version": "1.0",
                "kind": PILOT_KIND,
                "status": "POSTHOC_DOMAIN_AUDIT_COMPLETE",
                "paper_result_eligible": False,
                "model": MODEL,
                "dataset": DATASET,
                "sample_index": SAMPLE_INDEX,
                "scene": str(batch["scene"][0]),
                "target_rgb_provenance": {
                    "loaded_by_native_dataloader": True,
                    "accessed_before_compact_commit": False,
                    "accessed_for_metrics": False,
                },
                "execution_boundary": {
                    "source_bound_packet_adapter_executed": True,
                    "native_packet_decoder_executed": False,
                    "independent_dense_reference_executed_after_packet_commit": True,
                    "compact_nonzero_materialization_enabled": True,
                    "nonzero_direct_deletion": False,
                    "whole_pipeline_s2_s3_sparse_execution_verified": False,
                    "timing_claim": False,
                },
                "paper_identity": paper_identity,
                "adaptive_l1_calibration": calibration,
                "adaptive_l1_v4_attribute_loo_calibration": v4_attribute_loo_calibration,
                "route_plan": plan.events,
                "compact_route": route_summary,
                "route_diagnostics": route_diagnostics,
                "posthoc_dense_domain_coverage_reference": posthoc_dense_reference,
                "final_packet": {
                    "descriptor_count": int(final_packed.dense_slots.numel()),
                    "materialized_anchor_count": compact_anchor_count,
                    "source_trace_sha256": final_packed.source_trace_sha256,
                    "omitted_raw_head_positions_poisoned": capture[
                        "omitted_raw_head_positions_poisoned"
                    ],
                },
                "numerical_execution": numerical_execution,
                "checkpoint_sha256": checkpoint_sha256,
                "source": source_identity(),
            }
        # The compact route and packet are now committed. Target metadata is
        # used solely by the later renderer/metric phase.
        target_mapping = batch.get("target")
        if not isinstance(target_mapping, dict) or not torch.is_tensor(target_mapping.get("image")):
            raise RuntimeError("compact packet pilot requires native target RGB for metrics")
        target_cameras = _target_cameras(batch, loaded_device)
        _packet_gaussians, packet_color = render_packed_gaussians(
            model.decoder,
            renderable,
            gaussians_type=Gaussians,
            target=target_cameras,
            image_shape=(height, width),
            expected_descriptor_count=int(final_packed.dense_slots.numel()),
        )
        with torch.no_grad():
            baseline_gaussians = model.encoder(context, 0, deterministic=True)
        posthoc_dense_reference = _posthoc_dense_domain_coverage_reference(
            dense_gaussians=baseline_gaussians,
            final_packed=renderable,
            route=capture["guarded_route"],
            context=context,
            image_shape=(height, width),
        )
        baseline_color = _render_dense_baseline(
            model,
            baseline_gaussians,
            target=target_cameras,
            image_shape=(height, width),
        )

    target_rgb = _take_target_rgb_for_metrics(batch, loaded_device)
    baseline_views = _view_metrics(baseline_color[0], target_rgb[0])
    compact_views = _view_metrics(packet_color[0], target_rgb[0])
    baseline_quality = _mean_metrics(baseline_views)
    compact_quality = _mean_metrics(compact_views)
    verdict = _quality_verdict(baseline_quality, compact_quality)
    return {
        "schema_version": "1.0",
        "kind": PILOT_KIND,
        "status": "PASS" if verdict["pass"] else "QUALITY_FAILED",
        "paper_result_eligible": False,
        "model": MODEL,
        "dataset": DATASET,
        "sample_index": SAMPLE_INDEX,
        "scene": str(batch["scene"][0]),
        "context_indices": [int(value) for value in batch["context"]["index"][0].tolist()],
        "target_indices": [int(value) for value in batch["target"]["index"][0].tolist()],
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": True,
            "accessed_before_compact_commit": False,
            "target_camera_metadata_accessed_before_compact_commit": False,
            "target_camera_metadata_accessed_after_compact_commit": True,
            "first_transfer_after_packet_and_baseline_outputs": True,
        },
        "execution_boundary": {
            "source_bound_packet_adapter_executed": True,
            "native_packet_decoder_executed": True,
            "independent_dense_baseline_executed_after_packet_decoder": True,
            "compact_nonzero_materialization_enabled": True,
            "nonzero_direct_deletion": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "s2_s3_saving": 0.0,
            "timing_claim": False,
        },
        "paper_identity": paper_identity,
        "adaptive_l1_calibration": calibration,
        "adaptive_l1_v4_attribute_loo_calibration": v4_attribute_loo_calibration,
        "route_plan": plan.events,
        "compact_route": route_summary,
        "route_diagnostics": route_diagnostics,
        "posthoc_dense_domain_coverage_reference": posthoc_dense_reference,
        "final_packet": {
            "descriptor_count": int(final_packed.dense_slots.numel()),
            "materialized_anchor_count": compact_anchor_count,
            "source_trace_sha256": final_packed.source_trace_sha256,
            "omitted_raw_head_positions_poisoned": capture[
                "omitted_raw_head_positions_poisoned"
            ],
        },
        "quality": {
            "baseline": baseline_quality,
            "compact": compact_quality,
            "verdict": verdict,
            "views": [
                {
                    "target_index": int(batch["target"]["index"][0, index].item()),
                    "baseline": baseline_views[index],
                    "compact": compact_views[index],
                }
                for index in range(len(baseline_views))
            ],
        },
        "numerical_execution": numerical_execution,
        "checkpoint_sha256": checkpoint_sha256,
        "source": source_identity(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--posthoc-domain-audit-only", action="store_true")
    parser.add_argument("--calibration-record", type=Path, default=DEFAULT_CALIBRATION_RECORD)
    parser.add_argument("--v4-attribute-loo-calibration-record", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this local compact packet pilot requires CUDA")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from scripts.result_record import write_result

    try:
        record = collect_paper_compact_packet_pilot(
            device=device,
            posthoc_domain_audit_only=args.posthoc_domain_audit_only,
            calibration_record=args.calibration_record,
            v4_attribute_loo_calibration_record=args.v4_attribute_loo_calibration_record,
        )
        exit_code = 0 if record["status"] in {"PASS", "POSTHOC_DOMAIN_AUDIT_COMPLETE"} else 1
    except Exception as exc:
        record = {
            "schema_version": "1.0",
            "kind": PILOT_KIND,
            "status": "FAILED",
            "paper_result_eligible": False,
            "model": MODEL,
            "dataset": DATASET,
            "sample_index": SAMPLE_INDEX,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
        exit_code = 1
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
