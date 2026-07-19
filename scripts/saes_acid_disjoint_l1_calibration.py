#!/usr/bin/env python3
"""Freeze ACID-disjoint, target-free V15/V16 L1 calibration records.

The calibration source is the verified ACID 24/8 context-only materialization.
The application model is nevertheless TranSplat's DL3DV/Re10K checkpoint and
configuration.  No ACID target tensor, native ACID dataloader, target camera,
or DL3DV evaluation sample is opened by this collector.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.acid_joint_context import load_acid_joint_context
from integration.acid_joint_model_context import prepare_acid_joint_model_context
from integration.model_loader import create_model_loader
from saes.evaluation_disjoint_l1_calibration import (
    CLASSIC_APPLICATION_MODELS,
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
    HOLDOUT_SPLIT,
    TRAIN_SPLIT,
    build_v15_record,
    build_v16_record,
    load_frozen_v15_threshold,
    load_frozen_v16_threshold,
    resolve_acid_binding,
    sha256_file,
)
from saes.guarded_selected_route import (
    ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
)
from saes.incremental_selected_output_execution import (
    native_dense_head_execution_evidence,
)
from saes.probe_first_schedule import (
    ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    build_incremental_probe_first_plan,
)
from scripts.ae_config import resolve_experiment
from scripts.saes_incremental_selected_output_audit import (
    DEPTH_THRESHOLD,
    EVALUATION_DISJOINT_L1_15_DECISION_SEMANTICS,
    FEATURE_THRESHOLD,
    TILE_SIZE,
    _capture_guarded_incremental_packed_adapter,
    _capture_s1_s2_without_dense_adapter,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution


MODEL = "transplat"
DATASET = "dl3dv"
APPLICATION_EXPERIMENT = "re10k"
ENDPOINT_SMOKE_SCHEMA_VERSION = "1.0"
ENDPOINT_SMOKE_KIND = "saes-acid-disjoint-v16-endpoint-smoke-v1"
_RAW_CONTEXT_FIELDS = {"image", "extrinsics", "intrinsics", "index"}
_PREPARED_CONTEXT_FIELDS = {
    "image",
    "extrinsics",
    "intrinsics",
    "index",
    "near",
    "far",
}
_ACCESS = {
    "target_mapping_present": False,
    "target_rgb_accessed": False,
    "target_camera_metadata_accessed": False,
    "target_index_accessed": False,
    "skipped_s3_attributes_accessed": False,
}


def _context_access() -> dict[str, bool]:
    return dict(_ACCESS)


def _application_model_name(value: Any) -> str:
    if value not in CLASSIC_APPLICATION_MODELS:
        raise ValueError("ACID calibration requires a supported classic application model")
    return str(value)


def _release_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _emit_progress(*, phase: str, split: str, completed: int, total: int) -> None:
    """Emit bounded machine-readable collection progress without metric claims."""

    if completed == 1 or completed == total or completed % 4 == 0:
        print(
            "__DS_PROGRESS__ "
            + json.dumps(
                {
                    "phase": phase,
                    "split": split,
                    "completed_scenes": completed,
                    "total_scenes": total,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _require_target_free_source(raw: Any, preparation: Mapping[str, Any]) -> None:
    """Reject a context record or preprocessing audit that crossed the boundary."""

    context = getattr(raw, "context", None)
    identity = getattr(raw, "identity", None)
    if not isinstance(context, Mapping) or set(context) != _RAW_CONTEXT_FIELDS:
        raise RuntimeError("ACID calibration source did not contain only raw context tensors")
    if not isinstance(identity, Mapping):
        raise RuntimeError("ACID calibration source has no identity audit")
    for field in (
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
        "teacher_artifact_accessed",
        "expected_results_accessed",
    ):
        if identity.get(field) is not False:
            raise RuntimeError(f"ACID calibration source crossed its boundary: {field}")
    for field in (
        "target_mapping_present",
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
    ):
        if preparation.get(field) is not False:
            raise RuntimeError(f"ACID model preparation crossed its boundary: {field}")


def _require_prepared_context(context: Mapping[str, Any]) -> tuple[int, int]:
    if set(context) != _PREPARED_CONTEXT_FIELDS:
        raise RuntimeError("ACID model context contains a target-side field")
    image = context.get("image")
    if not torch.is_tensor(image) or image.ndim != 5 or image.shape[0] != 1:
        raise RuntimeError("ACID calibration requires one finite context batch")
    height, width = (int(value) for value in image.shape[-2:])
    if height <= 0 or width <= 0:
        raise RuntimeError("ACID calibration context image has an invalid shape")
    return height, width


def _build_plan(model: Any, context: Mapping[str, Any]) -> Any:
    height, width = _require_prepared_context(context)
    planning = _capture_s1_s2_without_dense_adapter(model, dict(context))
    return build_incremental_probe_first_plan(
        planning["features"],
        planning["depths"],
        height=height,
        width=width,
        tile_size=TILE_SIZE,
        feature_threshold=FEATURE_THRESHOLD,
        depth_threshold=DEPTH_THRESHOLD,
        decision_semantics=EVALUATION_DISJOINT_L1_15_DECISION_SEMANTICS,
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )


def _candidate_l1_residuals(plan: Any) -> list[float]:
    """Return every tile for which the V15 L1 residual guard is applicable."""

    values: list[float] = []
    for record in plan.tile_trace:
        if not isinstance(record, Mapping):
            raise RuntimeError("ACID residual planning record is invalid")
        is_candidate = record.get("pre_guard_route") == "L1" or (
            record.get("pre_guard_route") == "L0"
            and record.get("depth_uniform") is True
        )
        if not is_candidate:
            continue
        residual = record.get("adaptive_l1_leave_one_out_residual")
        if (
            isinstance(residual, bool)
            or not isinstance(residual, (int, float))
            or not math.isfinite(float(residual))
            or float(residual) < 0.0
        ):
            raise RuntimeError("ACID residual planning candidate is invalid")
        values.append(float(residual))
    if not values:
        raise RuntimeError("ACID calibration scene has no V15 residual candidates")
    return values


def _observed_v16_risks(
    capture: Mapping[str, Any], *, require_nonempty: bool = True
) -> list[float]:
    """Extract selected-anchor risks observed after a fixed V15 route decision."""

    preflight = capture.get("compact_materialization_preflight")
    events = getattr(preflight, "events", None)
    tile_trace = getattr(preflight, "tile_trace", None)
    if not isinstance(events, Mapping) or not isinstance(tile_trace, Sequence):
        raise RuntimeError("ACID V16 calibration has no compact preflight evidence")
    if (
        events.get("target_rgb_accessed") is not False
        or events.get("skipped_s3_attributes_accessed") is not False
        or events.get("selected_anchor_v4_attribute_loo_collect_only") is not True
        or events.get("selected_anchor_v4_attribute_loo_guard") is not False
    ):
        raise RuntimeError("ACID V16 calibration crossed its selected-only boundary")
    values: list[float] = []
    for tile in tile_trace:
        if not isinstance(tile, Mapping):
            raise RuntimeError("ACID V16 preflight tile record is invalid")
        replay = tile.get("selected_anchor_v4_attribute_loo")
        if replay is None:
            continue
        if not isinstance(replay, Mapping) or replay.get("action") != "observed_only":
            raise RuntimeError("ACID V16 replay record is invalid")
        risk = replay.get("q75_risk")
        if (
            isinstance(risk, bool)
            or not isinstance(risk, (int, float))
            or not math.isfinite(float(risk))
            or float(risk) < 0.0
        ):
            raise RuntimeError("ACID V16 replay risk is invalid")
        values.append(float(risk))
    if not values and require_nonempty:
        raise RuntimeError("ACID calibration scene has no V15-retained V16 risk tiles")
    return values


def _native_dense_v16_execution_evidence(
    capture: Mapping[str, Any], *, plan: Any
) -> dict[str, Any]:
    """Bind ACID V16 risks to the actual native-dense raw-head capture."""
    if not isinstance(capture, Mapping):
        raise TypeError("ACID V16 capture must be a mapping")
    initial_events = capture.get("initial_head_events")
    final_events = capture.get("final_head_events")
    initial = native_dense_head_execution_evidence(initial_events)
    final = native_dense_head_execution_evidence(final_events)
    if (
        initial["execution_finalized"] is not False
        or final["execution_finalized"] is not True
    ):
        raise RuntimeError("ACID V16 capture changed its initial/final head boundary")
    plan_events = getattr(plan, "events", None)
    if not isinstance(plan_events, Mapping):
        raise RuntimeError("ACID V16 capture has no route-plan events")
    selection_digest = plan_events.get("selection_mask_sha256")
    if (
        not isinstance(selection_digest, str)
        or initial_events.get("computed_mask_sha256") != selection_digest
        or initial_events.get("full_extension_dispatched") is not False
    ):
        raise RuntimeError("ACID V16 capture does not bind the native dense route plan")
    if len(final["phases"]) == 3:
        if (
            final != {**initial, "execution_finalized": True}
            or final_events.get("computed_mask_sha256") != selection_digest
            or final_events.get("full_extension_dispatched") is not False
        ):
            raise RuntimeError("ACID V16 capture changed after the native dense initial route")
        return initial
    guarded_route = capture.get("guarded_route")
    guard_events = getattr(guarded_route, "events", None)
    extension_event = capture.get("extension_event")
    extension = final["phases"][-1]
    if (
        len(final["phases"]) != 4
        or final["phases"][:3] != initial["phases"]
        or not isinstance(guard_events, Mapping)
        or not isinstance(extension_event, Mapping)
        or final_events.get("full_extension_dispatched") is not True
        or final_events.get("computed_mask_sha256")
        != guard_events.get("raw_head_request_mask_sha256")
        or extension.get("mask_sha256")
        != guard_events.get("additional_full_mask_sha256")
    ):
        raise RuntimeError("ACID V16 capture has an unbound native dense Full extension")
    return initial


def _load_application_encoder(
    device: torch.device, *, model_name: str = MODEL
) -> tuple[Any, Any, Any, str]:
    """Load only the DL3DV application encoder, never the ACID checkpoint."""

    model_name = _application_model_name(model_name)
    experiment = resolve_experiment(model_name, DATASET, ROOT)
    if (
        experiment.model != model_name
        or experiment.dataset != DATASET
        or experiment.experiment != APPLICATION_EXPERIMENT
        or experiment.checkpoint.name != "re10k.ckpt"
    ):
        raise RuntimeError("ACID calibration must use the DL3DV/Re10K classic mapping")
    checkpoint_path = Path(experiment.checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    loader = create_model_loader(model_name)
    bundle = loader.load_model(
        str(checkpoint_path),
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        encoder_only=True,
    )
    if getattr(bundle, "decoder", None) is not None:
        raise RuntimeError("ACID calibration unexpectedly constructed a decoder")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("DL3DV application checkpoint changed during loading")
    model = getattr(bundle, "model", None)
    config = getattr(bundle, "config", None)
    if model is None or getattr(model, "encoder", None) is None or config is None:
        raise RuntimeError("ACID calibration loader returned an incomplete encoder bundle")
    model.eval()
    return model, bundle, experiment, checkpoint_sha256


def _prepare_scene(
    *,
    bundle: Any,
    materialization_root: Path,
    plan_path: Path,
    split: str,
    sample_index: int,
    expected_scene: str,
) -> tuple[Any, dict[str, Any]]:
    raw = load_acid_joint_context(
        materialization_root=materialization_root,
        split=split,
        sample_index=sample_index,
        plan_path=plan_path,
    )
    if getattr(raw, "identity", {}).get("scene") != expected_scene:
        raise RuntimeError("ACID calibration scene order no longer matches its frozen binding")
    context, preparation = prepare_acid_joint_model_context(
        raw,
        dataset_cfg=bundle.config.dataset,
        encoder_cfg=bundle.config.model.encoder,
        device=bundle.device,
    )
    _require_target_free_source(raw, preparation)
    _require_prepared_context(context)
    return raw, context


def _collect_v15_split(
    *,
    model: Any,
    bundle: Any,
    binding: Mapping[str, Any],
    materialization_root: Path,
    plan_path: Path,
    split: str,
) -> list[dict[str, Any]]:
    scenes = binding["splits"][split]["scenes"]
    records: list[dict[str, Any]] = []
    for sample_index, expected_scene in enumerate(scenes):
        raw, context = _prepare_scene(
            bundle=bundle,
            materialization_root=materialization_root,
            plan_path=plan_path,
            split=split,
            sample_index=sample_index,
            expected_scene=expected_scene,
        )
        try:
            with strict_fp32_convolution_execution():
                plan = _build_plan(model, context)
                residuals = _candidate_l1_residuals(plan)
            records.append(
                {
                    "scene": str(raw.identity["scene"]),
                    "residuals": residuals,
                    "access": _context_access(),
                }
            )
            _emit_progress(
                phase="v15_residuals",
                split=split,
                completed=sample_index + 1,
                total=len(scenes),
            )
        finally:
            del context
            _release_cuda_cache(bundle.device)
    return records


def _collect_v16_split(
    *,
    model: Any,
    bundle: Any,
    binding: Mapping[str, Any],
    materialization_root: Path,
    plan_path: Path,
    split: str,
    v15_threshold: float,
    model_name: str = MODEL,
) -> list[dict[str, Any]]:
    if not math.isfinite(float(v15_threshold)) or float(v15_threshold) < 0.0:
        raise ValueError("verified V15 threshold is invalid")
    scenes = binding["splits"][split]["scenes"]
    records: list[dict[str, Any]] = []
    for sample_index, expected_scene in enumerate(scenes):
        raw, context = _prepare_scene(
            bundle=bundle,
            materialization_root=materialization_root,
            plan_path=plan_path,
            split=split,
            sample_index=sample_index,
            expected_scene=expected_scene,
        )
        try:
            with strict_fp32_convolution_execution():
                plan = _build_plan(model, context)
                capture = _capture_guarded_incremental_packed_adapter(
                    model,
                    context,
                    plan=plan,
                    compact_nonzero_materialization=True,
                    compact_execution_policy=(
                        ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY
                    ),
                    adaptive_l1_maximum_leave_one_out_residual=float(v15_threshold),
                    collect_selected_anchor_v4_attribute_loo_risk=True,
                    require_native_dense_head_execution=True,
                    model_name=model_name,
                )
                execution = _native_dense_v16_execution_evidence(capture, plan=plan)
                risks = _observed_v16_risks(capture)
            records.append(
                {
                    "scene": str(raw.identity["scene"]),
                    "risks": risks,
                    "access": _context_access(),
                    "native_dense_head_execution": execution,
                }
            )
            _emit_progress(
                phase="v16_risks",
                split=split,
                completed=sample_index + 1,
                total=len(scenes),
            )
        finally:
            del context
            _release_cuda_cache(bundle.device)
    return records


def _require_fixed_holdout(record: Mapping[str, Any], *, label: str) -> None:
    verification = record.get("holdout_verification")
    if (
        not isinstance(verification, Mapping)
        or verification.get("threshold_updated") is not False
        or not isinstance(verification.get("candidate_count"), int)
        or verification["candidate_count"] <= 0
    ):
        raise RuntimeError(f"{label} holdout verification did not preserve its fixed threshold")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen calibration record: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"calibration partial output already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        # Keep the partial file for an interrupted collection diagnosis.
        raise


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} must be a lowercase SHA256")
    return value


def _require_nonnegative_count(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a nonnegative integer")
    return value


def _with_smoke_sha256(record: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a smoke record to canonical content excluding its self-digest."""

    payload = dict(record)
    payload.pop("sha256", None)
    return {**payload, "sha256": _canonical_sha256(payload)}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_endpoint_plan(plan: Any) -> dict[str, Any]:
    """Keep the smoke aligned with the frozen paper-normalized L1-15 route."""

    events = getattr(plan, "events", None)
    if (
        not isinstance(events, Mapping)
        or events.get("decision_semantics")
        != EVALUATION_DISJOINT_L1_15_DECISION_SEMANTICS
        or events.get("l1_anchor_semantics") != ADAPTIVE_L1_15_ANCHOR_SEMANTICS
        or events.get("l1_anchor_count") != 15
    ):
        raise RuntimeError("V16 endpoint smoke did not use the frozen adaptive L1-15 plan")
    return {
        "decision_semantics": EVALUATION_DISJOINT_L1_15_DECISION_SEMANTICS,
        "l1_anchor_semantics": ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
        "l1_anchor_count": 15,
        "tile_trace_sha256": _require_sha256(
            events.get("tile_trace_sha256"), label="V16 endpoint plan trace"
        ),
    }


def _endpoint_preflight_summary(
    capture: Mapping[str, Any], *, risks: Sequence[float]
) -> dict[str, Any]:
    """Extract only scalar target-free endpoint evidence from V16 preflight."""

    preflight = capture.get("compact_materialization_preflight")
    events = getattr(preflight, "events", None)
    if not isinstance(events, Mapping):
        raise RuntimeError("V16 endpoint smoke has no compact preflight events")
    # `_observed_v16_risks` already verifies the selected-only access boundary.
    # Repeat the two direct access checks here so this persisted endpoint cannot
    # be produced if a later caller bypasses that collector helper.
    if (
        events.get("target_rgb_accessed") is not False
        or events.get("skipped_s3_attributes_accessed") is not False
    ):
        raise RuntimeError("V16 endpoint smoke crossed its target or skipped-S3 boundary")
    endpoint_count = _require_nonnegative_count(
        events.get("native_opacity_endpoint_selected_count"),
        label="V16 endpoint selected count",
    )
    promoted_tile_count = _require_nonnegative_count(
        events.get("native_opacity_endpoint_promoted_full_tiles"),
        label="V16 endpoint promoted tile count",
    )
    risk_tile_count = _require_nonnegative_count(
        events.get("selected_anchor_v4_attribute_loo_checked_tiles"),
        label="V16 endpoint risk tile count",
    )
    if risk_tile_count != len(risks):
        raise RuntimeError("V16 endpoint smoke risk count does not match its preflight")
    return {
        "native_opacity_endpoint_count": endpoint_count,
        "native_opacity_endpoint_promoted_tile_count": promoted_tile_count,
        "risk_tile_count": risk_tile_count,
        "tile_trace_sha256": _require_sha256(
            events.get("tile_trace_sha256"), label="V16 endpoint preflight trace"
        ),
    }


def _v15_endpoint_identity(
    v15: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    expected_application = {
        "model": MODEL,
        "dataset": DATASET,
        "checkpoint_sha256": _require_sha256(
            checkpoint_sha256, label="active Re10K checkpoint"
        ),
    }
    if (
        not isinstance(v15, Mapping)
        or v15.get("application") != expected_application
        or v15.get("acid_binding") != binding
        or v15.get("access") != _ACCESS
    ):
        raise RuntimeError("V16 endpoint smoke V15 record does not match its live binding")
    threshold = v15.get("threshold_value")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) < 0.0
    ):
        raise RuntimeError("V16 endpoint smoke V15 threshold is invalid")
    return {
        "kind": v15.get("kind"),
        "sha256": _require_sha256(v15.get("sha256"), label="verified V15 record"),
        "threshold_value": float(threshold),
        "application": expected_application,
        "acid_binding_sha256": _canonical_sha256(dict(binding)),
    }


def collect_v16_endpoint_smoke(
    *,
    device: torch.device,
    v15_record: Path,
    split: str,
    sample_index: int,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
    plan_path: Path = DEFAULT_PLAN_PATH,
) -> dict[str, Any]:
    """Run one verified target-free V16 preflight endpoint on an ACID scene."""

    materialization_root = Path(materialization_root).resolve()
    plan_path = Path(plan_path).resolve()
    if split not in {TRAIN_SPLIT, HOLDOUT_SPLIT}:
        raise ValueError("V16 endpoint smoke split is invalid")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("V16 endpoint smoke sample index is invalid")
    binding = resolve_acid_binding(
        plan_path=plan_path, materialization_root=materialization_root
    )
    scenes = binding["splits"][split]["scenes"]
    if sample_index >= len(scenes):
        raise IndexError("V16 endpoint smoke sample index is out of range")
    model, bundle, experiment, checkpoint_sha256 = _load_application_encoder(device)
    try:
        verified_v15 = load_frozen_v15_threshold(
            Path(v15_record),
            checkpoint_path=experiment.checkpoint,
            plan_path=plan_path,
            materialization_root=materialization_root,
        )
        v15_identity = _v15_endpoint_identity(
            verified_v15,
            binding=binding,
            checkpoint_sha256=checkpoint_sha256,
        )
        expected_scene = scenes[sample_index]
        raw, context = _prepare_scene(
            bundle=bundle,
            materialization_root=materialization_root,
            plan_path=plan_path,
            split=split,
            sample_index=sample_index,
            expected_scene=expected_scene,
        )
        with strict_fp32_convolution_execution():
            plan = _build_plan(model, context)
            route_plan = _require_endpoint_plan(plan)
            capture = _capture_guarded_incremental_packed_adapter(
                model,
                context,
                plan=plan,
                compact_nonzero_materialization=True,
                compact_execution_policy=(
                    ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY
                ),
                adaptive_l1_maximum_leave_one_out_residual=v15_identity[
                    "threshold_value"
                ],
                collect_selected_anchor_v4_attribute_loo_risk=True,
            )
            risks = _observed_v16_risks(capture, require_nonempty=False)
        preflight = _endpoint_preflight_summary(capture, risks=risks)
        if raw.identity.get("scene") != expected_scene:
            raise RuntimeError("V16 endpoint smoke source scene changed during collection")
        return _with_smoke_sha256(
            {
                "schema_version": ENDPOINT_SMOKE_SCHEMA_VERSION,
                "kind": ENDPOINT_SMOKE_KIND,
                "status": "PASS_TARGET_FREE_V16_ENDPOINT",
                "paper_result_eligible": False,
                "source_context": {
                    "split": split,
                    "sample_index": sample_index,
                    "scene": expected_scene,
                },
                "v15_identity": v15_identity,
                "acid_binding": dict(binding),
                "checkpoint_binding": {
                    "application": {
                        "model": MODEL,
                        "dataset": DATASET,
                        "checkpoint_sha256": checkpoint_sha256,
                    },
                    "checkpoint_path": str(Path(experiment.checkpoint).resolve()),
                    "experiment": experiment.experiment,
                    "encoder_only": True,
                    "decoder_constructed": False,
                },
                "access": _context_access(),
                "route_plan": route_plan,
                "preflight": preflight,
            }
        )
    finally:
        del model, bundle
        _release_cuda_cache(device)


def write_v16_endpoint_smoke(
    *, output: Path, **kwargs: Any
) -> dict[str, Any]:
    """Collect and persist one endpoint record without replacing frozen evidence."""

    destination = Path(output)
    partial = destination.with_name(f".{destination.name}.partial")
    if destination.exists() or partial.exists():
        raise FileExistsError("V16 endpoint smoke output must be a new file")
    record = collect_v16_endpoint_smoke(**kwargs)
    _write_new_json(destination, record)
    return record


def collect_frozen_calibration_records(
    *,
    device: torch.device,
    v15_output: Path,
    v16_output: Path,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
    plan_path: Path = DEFAULT_PLAN_PATH,
    model_name: str = MODEL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect ACID 24/8 records and freeze V15 then V16 under one application.

    V15 uses only train residuals.  Its holdout residuals are recorded solely
    for fixed-threshold verification.  The V15 JSON is written and loaded
    again before V16 is collected, so V16 risks cannot use an unfrozen parent.
    V16 then uses only train risks and verifies its holdout without updating.
    """

    v15_output = Path(v15_output)
    v16_output = Path(v16_output)
    materialization_root = Path(materialization_root).resolve()
    plan_path = Path(plan_path).resolve()
    if v15_output.resolve() == v16_output.resolve():
        raise ValueError("V15 and V16 calibration outputs must be different files")
    if v15_output.exists() or v16_output.exists():
        raise FileExistsError("frozen calibration outputs must not already exist")
    model_name = _application_model_name(model_name)
    binding = resolve_acid_binding(
        plan_path=plan_path, materialization_root=materialization_root
    )
    model, bundle, experiment, checkpoint_sha256 = _load_application_encoder(
        device, model_name=model_name
    )
    try:
        v15_train = _collect_v15_split(
            model=model,
            bundle=bundle,
            binding=binding,
            materialization_root=materialization_root,
            plan_path=plan_path,
            split=TRAIN_SPLIT,
        )
        v15_holdout = _collect_v15_split(
            model=model,
            bundle=bundle,
            binding=binding,
            materialization_root=materialization_root,
            plan_path=plan_path,
            split=HOLDOUT_SPLIT,
        )
        v15 = build_v15_record(
            binding=binding,
            application_checkpoint_sha256=checkpoint_sha256,
            application_model=model_name,
            train_scene_records=v15_train,
            holdout_scene_records=v15_holdout,
        )
        _require_fixed_holdout(v15, label="V15")
        _write_new_json(v15_output, v15)
        verified_v15 = load_frozen_v15_threshold(
            v15_output,
            checkpoint_path=experiment.checkpoint,
            application_model=model_name,
            plan_path=plan_path,
            materialization_root=materialization_root,
        )
        _require_fixed_holdout(verified_v15, label="V15")

        v16_train = _collect_v16_split(
            model=model,
            bundle=bundle,
            binding=binding,
            materialization_root=materialization_root,
            plan_path=plan_path,
            split=TRAIN_SPLIT,
            v15_threshold=verified_v15["threshold_value"],
            model_name=model_name,
        )
        v16_holdout = _collect_v16_split(
            model=model,
            bundle=bundle,
            binding=binding,
            materialization_root=materialization_root,
            plan_path=plan_path,
            split=HOLDOUT_SPLIT,
            v15_threshold=verified_v15["threshold_value"],
            model_name=model_name,
        )
        v16 = build_v16_record(
            binding=binding,
            application_checkpoint_sha256=checkpoint_sha256,
            application_model=model_name,
            v15_record_sha256=verified_v15["sha256"],
            train_scene_records=v16_train,
            holdout_scene_records=v16_holdout,
        )
        _require_fixed_holdout(v16, label="V16")
        _write_new_json(v16_output, v16)
        verified_v16 = load_frozen_v16_threshold(
            v16_output,
            checkpoint_path=experiment.checkpoint,
            application_model=model_name,
            v15_record_path=v15_output,
            plan_path=plan_path,
            materialization_root=materialization_root,
        )
        _require_fixed_holdout(verified_v16, label="V16")
        return verified_v15, verified_v16
    finally:
        del model, bundle
        _release_cuda_cache(device)


def calibration_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model", choices=tuple(sorted(CLASSIC_APPLICATION_MODELS)), default=MODEL
    )
    parser.add_argument("--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("ACID-disjoint V15/V16 calibration requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        v15, v16 = collect_frozen_calibration_records(
            device=device,
            v15_output=args.output_dir / "v15.json",
            v16_output=args.output_dir / "v16.json",
            materialization_root=args.materialization_root,
            plan_path=args.plan,
            model_name=args.model,
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "v15": str(args.output_dir / "v15.json"),
                "v15_sha256": v15["sha256"],
                "v16": str(args.output_dir / "v16.json"),
                "v16_sha256": v16["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def endpoint_smoke_main(argv: Sequence[str] | None = None) -> int:
    """CLI for one explicit ACID target-free V16 preflight endpoint."""

    parser = argparse.ArgumentParser(description=collect_v16_endpoint_smoke.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v15-record", type=Path, required=True)
    parser.add_argument("--split", choices=(TRAIN_SPLIT, HOLDOUT_SPLIT), required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_name(f".{args.output.name}.partial").exists():
        parser.error("--output must be a new file")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("ACID target-free V16 endpoint smoke requires CUDA")
    try:
        record = write_v16_endpoint_smoke(
            output=args.output,
            device=device,
            v15_record=args.v15_record,
            split=args.split,
            sample_index=args.sample_index,
            materialization_root=args.materialization_root,
            plan_path=args.plan,
        )
    except (OSError, RuntimeError, ValueError, TypeError, IndexError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": record["sha256"],
                "scene": record["source_context"]["scene"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "endpoint-smoke":
        return endpoint_smoke_main(values[1:])
    return calibration_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
