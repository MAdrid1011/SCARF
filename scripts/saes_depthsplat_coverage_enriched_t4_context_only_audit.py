#!/usr/bin/env python3
"""Audit the target-free coverage-enriched DepthSplat T=4 route.

The command consumes only the fixed DL3DV sample-zero context sidecar.  It
loads the encoder, verifies selected-head and Adapter provenance, builds the
coverage-enriched L0/L1/Full route, and records the requested fixed-scale
structural certificate. It deliberately stops before applying updates,
constructing a decoder, opening a target view, rendering, or scoring quality.

This is a pre-calibration mechanism audit, not a replacement for the frozen
literal V16T4 evaluation route.  In particular, it never loads that route's
attribute guard because the coverage-enriched profile has a different planner
and materialization contract.
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
    canonical_json_sha256,
    freeze_depthsplat_backend_identity,
    resolve_depthsplat_backend_contract,
)
from saes.depthsplat_l0_l1_materializer import (  # noqa: E402
    DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
    DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE,
    DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
    DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE,
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
    BALANCED_L1_ANCHOR_SEMANTICS,
    COVERAGE_ENRICHED_T4_L0_SECONDARY_PREFETCH_POLICY,
    DEPTHSPLAT_COVERAGE_ENRICHED_T4_PLAN_CONTRACT,
    DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT,
    build_depthsplat_coverage_enriched_t4_probe_first_plan,
    build_depthsplat_support_basis_t4_probe_first_plan,
    coverage_enriched_t4_route_config_sha256,
    support_basis_t4_route_config_sha256,
    SUPPORT_BASIS_T4_L0_SECONDARY_PREFETCH_POLICY,
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
AUDIT_KIND = "depthsplat-coverage-enriched-t4-context-only-precalibration-audit"
AUDIT_SCHEMA_VERSION = "1.0"
AUDIT_STATUS = "PASS_PRECALIBRATION_ONLY"
OWNER_SUPPORT_MECHANISM = "owner-support-v1"
SUPPORT_BASIS_MECHANISM = "support-basis-v1"
DEFAULT_INPUT_ROOT = (
    ROOT
    / "outputs"
    / "ae_dl3dv_repair_diagnostics"
    / "depthsplat_sample0_l0_l1_context_only_v2"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"DepthSplat coverage-enriched audit has an invalid {label} SHA256")
    return value


def _require_live_tile_trace(
    tile_trace: Any, *, expected_sha256: Any, label: str
) -> str:
    """Rehash the live route trace before recording its routing evidence."""

    if not isinstance(tile_trace, tuple) or not all(
        isinstance(record, Mapping) for record in tile_trace
    ):
        raise RuntimeError(f"DepthSplat coverage-enriched {label} trace is invalid")
    expected = _require_sha256(expected_sha256, f"{label} trace")
    actual = canonical_json_sha256(tile_trace)
    if actual != expected:
        raise RuntimeError(f"DepthSplat coverage-enriched {label} trace hash changed")
    return actual


def coverage_enriched_t4_profile() -> dict[str, Any]:
    """Return the immutable, pre-calibration coverage-enriched configuration."""

    profile: dict[str, Any] = {
        "materialization_profile": (
            DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE
        ),
        "route_plan_contract": DEPTHSPLAT_COVERAGE_ENRICHED_T4_PLAN_CONTRACT,
        "contract_version": DEPTHSPLAT_COVERAGE_ENRICHED_T4_PLAN_CONTRACT,
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": DEPTH_THRESHOLD,
        "decision_semantics": "paper-probe-feature-variance-first-hit",
        "feature_statistic": "raw-probe-mean-channel-variance",
        "l1_anchor_semantics": BALANCED_L1_ANCHOR_SEMANTICS,
        "l0_anchor_count": 4,
        "l1_anchor_count": 12,
        "formal_paper_kp4": False,
        "depth_checked_after_l0_miss_only": False,
        "coverage_enriched_l0_secondary_prefetch_policy": (
            COVERAGE_ENRICHED_T4_L0_SECONDARY_PREFETCH_POLICY
        ),
        "maximum_coverage_covariance_scale": 1.0,
        "coverage_certificate": DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE,
        "owner_support_guard": True,
        "attribute_loo_guard": None,
        "precalibration_only": True,
    }
    profile["route_plan_config_sha256"] = coverage_enriched_t4_route_config_sha256(
        profile
    )
    return profile


def support_basis_t4_profile() -> dict[str, Any]:
    """Return the immutable source-only cooperative support-basis profile."""

    profile: dict[str, Any] = {
        "materialization_profile": DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
        "route_plan_contract": DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT,
        "contract_version": DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT,
        "tile_size": TILE_SIZE,
        "feature_threshold": FEATURE_THRESHOLD,
        "depth_threshold": DEPTH_THRESHOLD,
        "decision_semantics": "paper-probe-feature-variance-first-hit",
        "feature_statistic": "raw-probe-mean-channel-variance",
        "l1_anchor_semantics": BALANCED_L1_ANCHOR_SEMANTICS,
        "l0_anchor_count": 4,
        "l1_anchor_count": 12,
        "formal_paper_kp4": False,
        "depth_checked_after_l0_miss_only": False,
        "support_basis_l0_secondary_prefetch_policy": (
            SUPPORT_BASIS_T4_L0_SECONDARY_PREFETCH_POLICY
        ),
        "maximum_coverage_covariance_scale": 1.0,
        "coverage_certificate": DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE,
        "support_basis_guard": True,
        "attribute_loo_guard": None,
        "precalibration_only": True,
    }
    profile["route_plan_config_sha256"] = support_basis_t4_route_config_sha256(profile)
    return profile


def _profile_for_mechanism(mechanism: str) -> dict[str, Any]:
    if mechanism == OWNER_SUPPORT_MECHANISM:
        return coverage_enriched_t4_profile()
    if mechanism == SUPPORT_BASIS_MECHANISM:
        return support_basis_t4_profile()
    raise ValueError("DepthSplat coverage audit mechanism is invalid")


def _require_formal_context_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the predeclared source-sample-zero context sidecar."""

    if not isinstance(identity, Mapping):
        raise TypeError("DepthSplat coverage-enriched context identity must be a mapping")
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
        or identity.get("target_mapping_present") is not False
        or identity.get("target_rgb_accessed") is not False
        or identity.get("target_camera_metadata_accessed") is not False
        or identity.get("target_index_accessed") is not False
    ):
        raise ValueError(
            "DepthSplat coverage-enriched audit input is not fixed context-only sample zero"
        )
    source_binding = identity.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise ValueError("DepthSplat coverage-enriched audit input has no source binding")
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
    """Bind the source-only sidecar to the live model/checkpoint contract."""

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
        raise RuntimeError("DepthSplat coverage-enriched audit application identity changed")
    binding = input_identity["source_binding"]
    if (
        binding["canonical_index_sha256"] != selection.source_index_sha256
        or binding["canonical_sample_selection_sha256"]
        != selection.sample_selection_sha256
    ):
        raise RuntimeError("DepthSplat coverage-enriched audit sidecar selection changed")


def _require_loaded_context_batch(
    batch: Mapping[str, Any], input_identity: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject a loader result that exposes target-side payloads or identity drift."""

    if not isinstance(batch, Mapping) or "target" in batch:
        raise RuntimeError(
            "DepthSplat coverage-enriched audit loader returned a target mapping"
        )
    context = batch.get("context")
    required_context = ("image", "extrinsics", "intrinsics", "near", "far", "index")
    if not isinstance(context, Mapping) or any(
        not torch.is_tensor(context.get(name)) for name in required_context
    ):
        raise RuntimeError(
            "DepthSplat coverage-enriched audit loader returned an incomplete context"
        )
    scene = batch.get("scene")
    if (
        not isinstance(scene, list)
        or scene != [input_identity["scene"]]
        or context["index"].shape != (1, len(input_identity["context_indices"]))
        or [int(value) for value in context["index"][0].tolist()]
        != input_identity["context_indices"]
    ):
        raise RuntimeError("DepthSplat coverage-enriched audit loader changed the fixed context")
    calibration = batch.get("calibration")
    if not isinstance(calibration, Mapping):
        raise RuntimeError("DepthSplat coverage-enriched audit loader has no input identity")
    for key, value in input_identity.items():
        if calibration.get(key) != value:
            raise RuntimeError("DepthSplat coverage-enriched audit loader changed input identity")
    if (
        calibration.get("target_mapping_present") is not False
        or calibration.get("target_rgb_accessed") is not False
        or calibration.get("target_camera_metadata_accessed") is not False
        or calibration.get("target_index_accessed") is not False
        or not isinstance(calibration.get("native_preprocessing"), Mapping)
    ):
        raise RuntimeError(
            "DepthSplat coverage-enriched audit loader crossed the target-free boundary"
        )
    return dict(context), dict(calibration)


def _source_geometry_functions(encoder: Any) -> tuple[Any, Any, dict[str, str]]:
    """Resolve geometry only from the loaded native DepthSplat namespace."""

    module_name = type(encoder).__module__
    if ".model." not in module_name:
        raise RuntimeError("DepthSplat coverage-enriched audit encoder module changed")
    geometry_module_name = module_name.split(".model.", 1)[0] + ".geometry.projection"
    module = sys.modules.get(geometry_module_name)
    source = getattr(module, "__file__", None)
    source_root = (ROOT / "depthsplat" / "src").resolve()
    if not isinstance(source, str):
        raise RuntimeError("DepthSplat coverage-enriched audit geometry module was not loaded")
    source_path = Path(source).resolve()
    if source_root not in source_path.parents:
        raise RuntimeError("DepthSplat coverage-enriched audit geometry source is foreign")
    sample_image_grid = getattr(module, "sample_image_grid", None)
    get_world_rays = getattr(module, "get_world_rays", None)
    if not callable(sample_image_grid) or not callable(get_world_rays):
        raise RuntimeError("DepthSplat coverage-enriched audit source geometry is incomplete")
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


def _require_coverage_enriched_plan(
    plan: Any, *, profile: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], str]:
    """Freeze the profile-specific planner settings before packet production."""

    events = getattr(plan, "events", None)
    profile = dict(coverage_enriched_t4_profile() if profile is None else profile)
    route_config_key = (
        "support_basis_t4_route_config_sha256"
        if profile["route_plan_contract"] == DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT
        else "coverage_enriched_t4_route_config_sha256"
    )
    if not isinstance(events, Mapping):
        raise RuntimeError("DepthSplat coverage-enriched audit returned no route events")
    trace_sha256 = _require_live_tile_trace(
        getattr(plan, "tile_trace", None),
        expected_sha256=events.get("tile_trace_sha256"),
        label="plan",
    )
    if (
        events.get("contract_version") != profile["route_plan_contract"]
        or events.get("tile_size") != profile["tile_size"]
        or events.get("feature_threshold") != profile["feature_threshold"]
        or events.get("depth_threshold") != profile["depth_threshold"]
        or events.get("l0_anchor_count") != profile["l0_anchor_count"]
        or events.get("l1_anchor_count") != profile["l1_anchor_count"]
        or events.get(route_config_key)
        != profile["route_plan_config_sha256"]
        or events.get("target_rgb_accessed") is not False
        or events.get("gaussian_attributes_accessed") is not False
    ):
        raise RuntimeError("DepthSplat coverage-enriched T=4 plan changed")
    return profile, trace_sha256


def _preflight_route_counts(
    *,
    plan: Any,
    preflight: Any,
    final_route: Any,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize the L0 -> L1 -> Full decisions without hiding promotions."""

    planned = {"L0": 0, "L1": 0, "Full": 0}
    accepted = {"L0": 0, "L1": 0, "Full": 0}
    rejected_reasons: dict[str, int] = {}
    for record in plan.tile_trace:
        route = record.get("pre_guard_route")
        if route not in planned:
            raise RuntimeError("DepthSplat coverage-enriched plan route is invalid")
        planned[route] += 1
    for record in preflight.tile_trace:
        accepted_level = record.get("accepted_level")
        if record.get("accepted") is True:
            if accepted_level not in accepted:
                raise RuntimeError("DepthSplat coverage-enriched accepted route is invalid")
            accepted[accepted_level] += 1
        elif record.get("attempted") is True:
            reason = record.get("reason")
            if not isinstance(reason, str) or not reason:
                raise RuntimeError("DepthSplat coverage-enriched rejection has no reason")
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
    profile = coverage_enriched_t4_profile() if profile is None else profile
    support_basis = (
        profile["route_plan_contract"] == DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT
    )
    l0_to_l1_key = (
        "support_basis_l0_to_l1_tile_count"
        if support_basis
        else "coverage_enriched_l0_to_l1_tile_count"
    )
    summary = {
        "planned": planned,
        "accepted_before_final_full_promotion": accepted,
        "final": dict(final_route.events["route_counts"]),
        "promoted_full": int(preflight.events["promoted_full_tiles"]),
        "rejection_reasons": rejected_reasons,
    }
    summary[
        "l0_to_l1_support_basis" if support_basis else "l0_to_l1_enriched"
    ] = int(preflight.events[l0_to_l1_key])
    return summary


def _require_coverage_enriched_preflight(
    *, profile: Mapping[str, Any], preflight: Any
) -> str:
    """Reject drift into literal or development materialization semantics."""

    events = preflight.events
    trace_sha256 = _require_live_tile_trace(
        getattr(preflight, "tile_trace", None),
        expected_sha256=events.get("tile_trace_sha256"),
        label="preflight",
    )
    certificate_payload = events.get("coverage_certificate_payload")
    support_basis = (
        profile["route_plan_contract"] == DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT
    )
    if (
        events.get("execution_profile") != profile["materialization_profile"]
        or events.get("maximum_coverage_covariance_scale")
        != profile["maximum_coverage_covariance_scale"]
        or events.get("coverage_certificate") != profile["coverage_certificate"]
        or (
            events.get("support_basis_guard") is not True
            if support_basis
            else events.get("coverage_enriched_owner_support_guard") is not True
        )
        or events.get("formal_paper_selected_probe_only") is not False
        or events.get("selected_anchor_attribute_loo_collect_only") is not False
        or events.get("selected_anchor_attribute_loo_guard") is not False
        or events.get("selected_anchor_attribute_loo_frozen_guard") is not None
        or events.get("selected_anchor_attribute_loo_aggregate") is not None
        or events.get("target_rgb_accessed") is not False
        or events.get("target_camera_accessed_before_commit") is not False
        or events.get("skipped_s3_attributes_accessed") is not False
        or not isinstance(certificate_payload, Mapping)
        or certificate_payload.get("schema") != profile["coverage_certificate"]
        or certificate_payload.get("fixed_moment_covariance_scale") != 1.0
        or certificate_payload.get("support_containment_guard") is not True
        or certificate_payload.get("tile_trace_sha256") != trace_sha256
    ):
        raise RuntimeError("DepthSplat coverage-enriched preflight changed")
    per_update = certificate_payload.get("per_update")
    if not isinstance(per_update, list):
        raise RuntimeError("DepthSplat coverage-enriched owner certificate is incomplete")
    for row in per_update:
        if not isinstance(row, Mapping):
            raise RuntimeError("DepthSplat coverage-enriched certificate row is invalid")
        if support_basis:
            if (
                row.get("support_basis_schema_version")
                != "depthsplat-tile-support-basis-coverage-audit-v1"
                or row.get("support_basis_kind")
                != "depthsplat-source-only-same-tile-support-basis-coverage"
                or row.get("support_basis_policy")
                != "same-tile-composed-soft-ledger-support-v1"
                or row.get("support_basis_passed") is not True
                or not isinstance(row.get("support_basis_sha256"), str)
            ):
                raise RuntimeError("DepthSplat support-basis certificate changed")
        elif (
            row.get("source_anchor_count") != 1
            or row.get("source_anchor_support_included") is not True
            or row.get("source_anchor_support_passed") is not True
            or row.get("owner_support_passed") is not True
        ):
            raise RuntimeError("DepthSplat coverage-enriched owner certificate changed")
    _require_sha256(events.get("coverage_certificate_sha256"), "preflight certificate")
    _require_sha256(events.get("materialization_session_sha256"), "preflight session")
    return trace_sha256


def _require_coverage_enriched_final_route(
    *, preflight: Any, preflight_trace_sha256: str, final_route: Any
) -> str:
    """Verify the final L0/L1/Full decision still binds live preflight evidence."""

    events = getattr(final_route, "events", None)
    if not isinstance(events, Mapping):
        raise RuntimeError("DepthSplat coverage-enriched audit returned no final route")
    trace_sha256 = _require_live_tile_trace(
        getattr(final_route, "tile_trace", None),
        expected_sha256=events.get("tile_trace_sha256"),
        label="final route",
    )
    route_counts = events.get("route_counts")
    if (
        events.get("preflight_trace_sha256") != preflight_trace_sha256
        or events.get("coverage_certificate_sha256")
        != preflight.events.get("coverage_certificate_sha256")
        or not isinstance(route_counts, Mapping)
        or set(route_counts) != {"L0", "L1", "Full"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in route_counts.values()
        )
    ):
        raise RuntimeError("DepthSplat coverage-enriched final route changed")
    return trace_sha256


def collect_depthsplat_coverage_enriched_t4_context_only_audit(
    *,
    input_root: Path,
    device: torch.device,
    mechanism: str = OWNER_SUPPORT_MECHANISM,
) -> dict[str, Any]:
    """Run one source-only, pre-calibration coverage-enriched route audit."""

    from data.context_only_audit_input import validate_context_only_audit_input
    from integration import create_model_loader, load_context_only_audit_data
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.result_record import cached_sha256_file, source_identity

    requested_profile = _profile_for_mechanism(mechanism)
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
        raise RuntimeError("DepthSplat coverage-enriched audit unexpectedly constructed a decoder")
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
            raise RuntimeError(
                "DepthSplat coverage-enriched audit has no target-free routing tensors"
            )
        _views, _channels, height, width = execution.dense_raw_head.shape
        if height % TILE_SIZE or width % TILE_SIZE:
            raise RuntimeError("DepthSplat coverage-enriched audit image shape is not tiled by four")
        if mechanism == OWNER_SUPPORT_MECHANISM:
            plan = build_depthsplat_coverage_enriched_t4_probe_first_plan(
                execution.routing_features,
                execution.routing_z_depths,
                height=height,
                width=width,
                feature_threshold=FEATURE_THRESHOLD,
                depth_threshold=DEPTH_THRESHOLD,
            )
        elif mechanism == SUPPORT_BASIS_MECHANISM:
            plan = build_depthsplat_support_basis_t4_probe_first_plan(
                execution.routing_features,
                execution.routing_z_depths,
                height=height,
                width=width,
                feature_threshold=FEATURE_THRESHOLD,
                depth_threshold=DEPTH_THRESHOLD,
            )
        else:
            raise RuntimeError("DepthSplat coverage audit mechanism changed")
        profile, plan_trace_sha256 = _require_coverage_enriched_plan(
            plan, profile=requested_profile
        )
        initial_replay = replay_depthsplat_selected_head(
            bundle.encoder.gaussian_head,
            execution.gaussian_head_input,
            execution.dense_raw_head,
            plan.selection_mask,
            native_full_mask=plan.full_mask,
        )
        if initial_replay.equivalence.get("equivalent") is not True:
            raise RuntimeError("DepthSplat coverage-enriched initial selected replay drifted")
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
            "DepthSplat coverage-enriched initial Adapter packet",
        )
        initial_full_passthrough = _require_bitwise_equivalent(
            compare_depthsplat_full_passthrough_to_dense_bitwise(
                initial_packed, execution.dense_gaussians, plan.full_mask
            ),
            "DepthSplat coverage-enriched initial Full attributes",
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
            execution_profile=profile["materialization_profile"],
        )
        preflight_trace_sha256 = _require_coverage_enriched_preflight(
            profile=profile, preflight=preflight
        )
        final_route = resolve_depthsplat_compact_final_route(plan, preflight)
        final_route_trace_sha256 = _require_coverage_enriched_final_route(
            preflight=preflight,
            preflight_trace_sha256=preflight_trace_sha256,
            final_route=final_route,
        )
        producer_replay = replay_depthsplat_selected_head(
            bundle.encoder.gaussian_head,
            execution.gaussian_head_input,
            execution.dense_raw_head,
            final_route.raw_head_request_mask,
            native_full_mask=final_route.full_passthrough_mask,
        )
        if producer_replay.equivalence.get("equivalent") is not True:
            raise RuntimeError("DepthSplat coverage-enriched final selected replay drifted")
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
            "DepthSplat coverage-enriched final Adapter packet",
        )
        final_full_passthrough = _require_bitwise_equivalent(
            compare_depthsplat_full_passthrough_to_dense_bitwise(
                final_packed,
                execution.dense_gaussians,
                final_route.full_passthrough_mask,
            ),
            "DepthSplat coverage-enriched final Full attributes",
        )

    descriptor_count = int(final_packed.dense_slots.numel())
    if descriptor_count != int(final_route.selected_output_mask.sum().item()):
        raise RuntimeError("DepthSplat coverage-enriched final packet route drifted")
    route_counts = _preflight_route_counts(
        plan=plan, preflight=preflight, final_route=final_route, profile=profile
    )
    access_flags = {
        "context_only_sidecar_used": True,
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "native_dl3dv_chunk_loader_used": False,
        "nonselected_dl3dv_sample_data_opened": False,
        "literal_v16t4_guard_loaded": False,
        "literal_v16t4_guard_required": False,
        "decoder_constructed": False,
        "renderer_executed": False,
        "target_view_rendered": False,
        "quality_metrics_computed": False,
        "materialization_applied": False,
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "status": AUDIT_STATUS,
        "paper_result_eligible": False,
        "precalibration_only": True,
        "formal_target_free_audit": True,
        "formal_target_free_sidecar_used": True,
        "model": MODEL,
        "dataset": DATASET,
        "source_sample_index": SOURCE_SAMPLE_INDEX,
        "scene": input_identity["scene"],
        "context_indices": list(input_identity["context_indices"]),
        "mechanism": mechanism,
        "profile": profile,
        "access_flags": access_flags,
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
            "coverage_enriched_preflight_executed": (
                mechanism == OWNER_SUPPORT_MECHANISM
            ),
            "support_basis_preflight_executed": mechanism == SUPPORT_BASIS_MECHANISM,
            "coverage_enriched_materialization_applied": False,
            "support_basis_materialization_applied": False,
        },
        "context_only_input": {
            "identity": input_identity,
            "loaded_native_preprocessing": loaded_calibration["native_preprocessing"],
        },
        "route_plan": plan.events,
        "route_plan_tile_trace": [dict(record) for record in plan.tile_trace],
        "route_counts": route_counts,
        "trace_bindings": {
            "route_plan_tile_trace_sha256": plan_trace_sha256,
            "preflight_tile_trace_sha256": preflight_trace_sha256,
            "final_route_tile_trace_sha256": final_route_trace_sha256,
        },
        "preflight_certificate": {
            "name": preflight.events["coverage_certificate"],
            "sha256": preflight.events["coverage_certificate_sha256"],
            "payload": preflight.events["coverage_certificate_payload"],
            "materialization_session_sha256": preflight.events[
                "materialization_session_sha256"
            ],
            "tile_trace_sha256": preflight_trace_sha256,
            "owner_support_guard": preflight.events[
                "coverage_enriched_owner_support_guard"
            ],
            "support_basis_guard": preflight.events.get("support_basis_guard"),
        },
        "materialization_preflight": preflight.events,
        "materialization_preflight_tile_trace": [
            dict(record) for record in preflight.tile_trace
        ],
        "final_route": final_route.events,
        "final_route_tile_trace": [dict(record) for record in final_route.tile_trace],
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--mechanism",
        choices=(OWNER_SUPPORT_MECHANISM, SUPPORT_BASIS_MECHANISM),
        default=OWNER_SUPPORT_MECHANISM,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("DepthSplat coverage-enriched context-only audit requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from scripts.result_record import portable_command, write_result

    try:
        record = collect_depthsplat_coverage_enriched_t4_context_only_audit(
            input_root=args.input_root, device=device, mechanism=args.mechanism
        )
        exit_code = 0
    except Exception as exc:
        record = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "kind": AUDIT_KIND,
            "status": "FAILED_PRECALIBRATION_ONLY",
            "paper_result_eligible": False,
            "precalibration_only": True,
            "formal_target_free_audit": True,
            "formal_target_free_sidecar_used": True,
            "model": MODEL,
            "dataset": DATASET,
            "source_sample_index": SOURCE_SAMPLE_INDEX,
            "mechanism": args.mechanism,
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
