#!/usr/bin/env python3
"""Run the fixed DepthSplat DL3DV sample-0 target-RGB quality gate.

The gate first reloads the frozen literal T=4 V16 record and a successful
formal context-only audit. It reconstructs the exact packet from the audited
sidecar before it reads the isolated sample-zero target frames. Target cameras
are used only for the two committed decoder renders, and target RGB is
transferred only afterwards for PSNR, SSIM, and LPIPS measurement.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
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

from integration import create_model_loader, load_context_only_audit_data
from saes.depthsplat_acid_disjoint_calibration import (
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
)
from saes.depthsplat_backend import (
    canonical_json_sha256,
    freeze_depthsplat_backend_identity,
    resolve_depthsplat_backend_contract,
)
from saes.depthsplat_l0_l1_materializer import (
    apply_depthsplat_compact_l0_l1_materialization,
    preflight_depthsplat_l0_l1_materialization,
    resolve_depthsplat_compact_final_route,
)
from saes.depthsplat_literal_t4_acid_calibration import (
    LITERAL_T4_MATERIALIZATION_PROFILE,
    literal_t4_profile_sha256,
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
from scripts.result_record import (
    cached_sha256_file,
    portable_command,
    sha256_file,
    source_identity,
    write_result,
)
from scripts.saes_depthsplat_l0_l1_context_only_audit import (
    AUDIT_KIND as FORMAL_AUDIT_KIND,
    AUDIT_SCHEMA_VERSION as FORMAL_AUDIT_SCHEMA_VERSION,
    DATASET,
    DEFAULT_INPUT_ROOT,
    DEPTH_THRESHOLD,
    FEATURE_THRESHOLD,
    MODEL,
    SOURCE_SAMPLE_INDEX,
    TILE_SIZE,
    _bind_formal_application,
    _load_literal_t4_v16_audit_guard,
    _require_bitwise_equivalent,
    _require_equivalent,
    _require_formal_context_identity,
    _require_loaded_context_batch,
    _source_geometry_functions,
)
from scripts.saes_selected_output_quality_gate import (
    _mean_metrics,
    _quality_verdict,
    _view_metrics,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution


QUALITY_GATE_KIND = "depthsplat-formal-l0-l1-target-rgb-quality-gate"
QUALITY_GATE_SCHEMA_VERSION = "1.0"
QUALITY_GATE_SEED = 0
DEFAULT_SOURCE_AUDIT_ROOT = (
    ROOT
    / "outputs"
    / "ae_dl3dv_repair_diagnostics"
    / "depthsplat_sample0_l0_l1_source_target_free_v1"
)
DEFAULT_RAW_ROOT = ROOT / "downloads" / "dl3dv-benchmark"
NATIVE_TARGET_IMAGE_SHAPE = (270, 480)
ISOLATED_TARGET_READER = "depthsplat-isolated-sample0-raw-target-reader-v1"


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"DepthSplat quality gate has an invalid {label} SHA256")
    return value


def _portable_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"DepthSplat quality gate {label} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"DepthSplat quality gate {label} must be an object")
    return value


def _require_indices(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in value
        )
        or len(value) != len(set(value))
    ):
        raise ValueError(f"DepthSplat quality gate has invalid {label}")
    return list(value)


def _read_formal_audit(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one immutable formal-audit result and bind its file and content."""

    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("formal DepthSplat target-free audit is unavailable or invalid") from error
    if not isinstance(raw, dict):
        raise ValueError("formal DepthSplat target-free audit must be an object")

    record = dict(raw)
    embedded_sha256 = record.pop("sha256", None)
    if embedded_sha256 is not None:
        _require_sha256(embedded_sha256, "formal audit embedded record")
        if embedded_sha256 != canonical_json_sha256(record):
            raise ValueError("formal DepthSplat target-free audit embedded SHA256 is invalid")
    return record, {
        "path": _portable_path(path),
        "file_sha256": sha256_file(path),
        "record_sha256": canonical_json_sha256(record),
        "embedded_record_sha256": embedded_sha256,
    }


def _require_formal_audit_runner(audit: Mapping[str, Any]) -> None:
    runner = audit.get("runner")
    expected = "scripts/saes_depthsplat_l0_l1_context_only_audit.py"
    if (
        not isinstance(runner, Mapping)
        or runner.get("path") != expected
        or not isinstance(runner.get("sha256"), str)
    ):
        raise ValueError("formal DepthSplat target-free audit runner identity changed")
    runner_path = (ROOT / expected).resolve()
    if cached_sha256_file(runner_path) != runner["sha256"]:
        raise ValueError("formal DepthSplat target-free audit runner source changed")


def _expected_literal_binding(
    *, literal_guard: Mapping[str, Any], literal_profile: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "frozen_record_sha256": literal_guard["frozen_record_sha256"],
        "frozen_record_kind": literal_guard["frozen_record_kind"],
        "profile": dict(literal_profile),
        "profile_sha256": literal_t4_profile_sha256(),
        "threshold_value": literal_guard["threshold_value"],
        "threshold_rule": literal_guard["threshold_rule"],
        "risk_metric": literal_guard["risk_metric"],
        "acid_binding_sha256": literal_guard["acid_binding_sha256"],
        "application_sha256": literal_guard["application_sha256"],
        "guard": dict(literal_guard),
    }


def _validate_formal_audit(
    path: Path,
    *,
    input_identity: Mapping[str, Any],
    literal_guard: Mapping[str, Any],
    literal_profile: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the exact target-free proof before target data can be opened."""

    audit, binding = _read_formal_audit(path)
    if (
        audit.get("schema_version") != FORMAL_AUDIT_SCHEMA_VERSION
        or audit.get("kind") != FORMAL_AUDIT_KIND
        or audit.get("status") != "PASS"
        or audit.get("paper_result_eligible") is not False
        or audit.get("formal_target_free_audit") is not True
        or audit.get("formal_target_free_sidecar_used") is not True
        or audit.get("model") != MODEL
        or audit.get("dataset") != DATASET
        or audit.get("source_sample_index") != SOURCE_SAMPLE_INDEX
        or audit.get("scene") != input_identity["scene"]
        or audit.get("context_indices") != input_identity["context_indices"]
    ):
        raise ValueError("formal DepthSplat target-free audit identity changed")
    for field in (
        "target_mapping_present",
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
    ):
        if audit.get(field) is not False:
            raise ValueError(f"formal DepthSplat target-free audit crossed {field}")

    boundary = audit.get("execution_boundary")
    expected_boundary = {
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
    }
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not value for key, value in expected_boundary.items()
    ):
        raise ValueError("formal DepthSplat target-free audit execution boundary changed")

    context_only_input = audit.get("context_only_input")
    if (
        not isinstance(context_only_input, Mapping)
        or context_only_input.get("identity") != dict(input_identity)
        or not isinstance(context_only_input.get("loaded_native_preprocessing"), Mapping)
    ):
        raise ValueError("formal DepthSplat target-free audit context input changed")
    if audit.get("literal_t4_v16") != _expected_literal_binding(
        literal_guard=literal_guard, literal_profile=literal_profile
    ):
        raise ValueError("formal DepthSplat target-free audit frozen V16T4 binding changed")
    if (
        audit.get("backend_identity") != dict(backend_identity)
        or audit.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("formal DepthSplat target-free audit backend binding changed")
    _require_formal_audit_runner(audit)

    required_traces = (
        "route_plan",
        "materialization_preflight",
        "materialization_preflight_tile_trace",
        "final_route",
        "native_execution",
        "initial_selected_head",
        "producer_selected_head",
        "initial_packet",
        "final_packet",
        "materialized_packet",
        "geometry_source",
    )
    if any(field not in audit for field in required_traces):
        raise ValueError("formal DepthSplat target-free audit has incomplete packet traces")
    return audit, binding


def _materialization_evidence(
    *,
    execution: Any,
    plan: Any,
    preflight: Any,
    final_route: Any,
    initial_replay: Any,
    producer_replay: Any,
    producer_packet: Any,
    initial_packed: Any,
    final_packed: Any,
    materialized: Any,
    initial_adapter_equivalence: Mapping[str, Any],
    initial_full_passthrough: Mapping[str, Any],
    final_adapter_equivalence: Mapping[str, Any],
    final_full_passthrough: Mapping[str, Any],
    materialized_full_passthrough: Mapping[str, Any],
    geometry_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Match the formal audit's serializable packet/provenance payload."""

    return {
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
            "native_adapter_equivalence": dict(initial_adapter_equivalence),
            "full_passthrough_bitwise": dict(initial_full_passthrough),
        },
        "final_packet": {
            "producer_descriptor_count": int(producer_packet.dense_slots.numel()),
            "descriptor_count": int(final_packed.dense_slots.numel()),
            "source_trace_sha256": final_packed.source_trace_sha256,
            "attribute_binding_sha256": final_packed.attribute_binding_sha256,
            "native_adapter_equivalence_before_materialization": dict(
                final_adapter_equivalence
            ),
            "full_passthrough_bitwise_before_materialization": dict(
                final_full_passthrough
            ),
            "producer_only_prefetch_descriptor_count": final_route.events[
                "producer_only_prefetch_descriptor_count"
            ],
            "skipped_s3_attributes_accessed": False,
            "nonzero_direct_deletion": False,
        },
        "materialized_packet": {
            "descriptor_count": int(materialized.dense_slots.numel()),
            "source_trace_sha256": materialized.source_trace_sha256,
            "attribute_binding_sha256": materialized.attribute_binding_sha256,
            "full_passthrough_bitwise": dict(materialized_full_passthrough),
        },
        "geometry_source": dict(geometry_source),
    }


def _require_nonzero_merge(preflight: Any) -> int:
    """Reject an all-Full route before the quality phase can open targets."""

    update_slots = getattr(preflight, "update_dense_slots", None)
    if not torch.is_tensor(update_slots):
        raise RuntimeError("DepthSplat quality gate preflight has no merge updates")
    count = int(update_slots.numel())
    if count <= 0:
        raise RuntimeError("DepthSplat quality gate requires an actual compact nonzero merge")
    return count


def _rebuild_materialized_packet(
    *,
    encoder: Any,
    context: Mapping[str, Any],
    literal_guard: Any,
    literal_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the audited literal T=4 packet without target-side inputs."""

    execution = capture_depthsplat_native_execution(
        encoder, dict(context), source_root=ROOT / "depthsplat"
    )
    if execution.routing_features is None or execution.routing_z_depths is None:
        raise RuntimeError("DepthSplat quality gate has no target-free routing tensors")
    _views, _channels, height, width = execution.dense_raw_head.shape
    if height % TILE_SIZE or width % TILE_SIZE:
        raise RuntimeError("DepthSplat quality gate image shape is not tiled by four")
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
        raise RuntimeError("DepthSplat quality gate literal T=4 plan changed")

    initial_replay = replay_depthsplat_selected_head(
        encoder.gaussian_head,
        execution.gaussian_head_input,
        execution.dense_raw_head,
        plan.selection_mask,
        native_full_mask=plan.full_mask,
    )
    if initial_replay.equivalence.get("equivalent") is not True:
        raise RuntimeError("DepthSplat quality gate initial selected replay drifted")
    initial_packet = build_depthsplat_sparse_raw_packet(execution, initial_replay)
    consumer = DepthSplatPackedGaussianConsumer(encoder.gaussian_adapter)
    initial_packed = consumer.convert(
        initial_packet,
        image_shape=(height, width),
        native_execution=execution,
        native_full_mask=plan.full_mask,
    )
    initial_adapter_equivalence = _require_equivalent(
        compare_depthsplat_packed_to_dense(initial_packed, execution.dense_gaussians),
        "DepthSplat quality gate initial Adapter packet",
    )
    initial_full_passthrough = _require_bitwise_equivalent(
        compare_depthsplat_full_passthrough_to_dense_bitwise(
            initial_packed, execution.dense_gaussians, plan.full_mask
        ),
        "DepthSplat quality gate initial Full attributes",
    )
    sample_image_grid, get_world_rays, geometry_source = _source_geometry_functions(
        encoder
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
        != literal_guard.as_dict()
        or preflight.events.get("selected_anchor_attribute_loo_collect_only")
        is not False
        or preflight.events.get("selected_anchor_attribute_loo_guard") is not True
    ):
        raise RuntimeError("DepthSplat quality gate literal V16 preflight changed")
    nonzero_merge_update_slot_count = _require_nonzero_merge(preflight)
    final_route = resolve_depthsplat_compact_final_route(plan, preflight)
    producer_replay = replay_depthsplat_selected_head(
        encoder.gaussian_head,
        execution.gaussian_head_input,
        execution.dense_raw_head,
        final_route.raw_head_request_mask,
        native_full_mask=final_route.full_passthrough_mask,
    )
    if producer_replay.equivalence.get("equivalent") is not True:
        raise RuntimeError("DepthSplat quality gate final selected replay drifted")
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
        "DepthSplat quality gate final Adapter packet",
    )
    final_full_passthrough = _require_bitwise_equivalent(
        compare_depthsplat_full_passthrough_to_dense_bitwise(
            final_packed, execution.dense_gaussians, final_route.full_passthrough_mask
        ),
        "DepthSplat quality gate final Full attributes",
    )
    materialized = apply_depthsplat_compact_l0_l1_materialization(
        final_packed, preflight, final_route
    )
    materialized_full_passthrough = _require_bitwise_equivalent(
        compare_depthsplat_full_passthrough_to_dense_bitwise(
            materialized, execution.dense_gaussians, final_route.full_passthrough_mask
        ),
        "DepthSplat quality gate materialized Full attributes",
    )
    if int(materialized.dense_slots.numel()) != int(
        final_route.selected_output_mask.sum().item()
    ):
        raise RuntimeError("DepthSplat quality gate final materialized packet route drifted")
    evidence = _materialization_evidence(
        execution=execution,
        plan=plan,
        preflight=preflight,
        final_route=final_route,
        initial_replay=initial_replay,
        producer_replay=producer_replay,
        producer_packet=producer_packet,
        initial_packed=initial_packed,
        final_packed=final_packed,
        materialized=materialized,
        initial_adapter_equivalence=initial_adapter_equivalence,
        initial_full_passthrough=initial_full_passthrough,
        final_adapter_equivalence=final_adapter_equivalence,
        final_full_passthrough=final_full_passthrough,
        materialized_full_passthrough=materialized_full_passthrough,
        geometry_source=geometry_source,
    )
    return {
        "execution": execution,
        "materialized": materialized,
        "height": height,
        "width": width,
        "nonzero_merge_update_slot_count": nonzero_merge_update_slot_count,
        "evidence": evidence,
    }


def _require_rebuilt_trace_matches_audit(
    audit: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    for field, value in evidence.items():
        if audit.get(field) != value:
            raise RuntimeError(
                f"DepthSplat quality gate reconstructed {field} differs from the formal audit"
            )


def _load_isolated_source_target_record(
    *,
    source_audit_root: Path,
    scene: str,
    context_indices: list[int],
    input_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load only the target-free selection and camera sidecar bound to context."""

    from scripts.calibration_inputs import (
        load_target_free_record,
        validate_target_free_input_root,
    )
    from data.verify_prepared_dataset import verify_tree_manifest

    source_audit_root = Path(source_audit_root).resolve()
    source_audit_path = source_audit_root / "audit-input.json"
    source_binding = input_identity.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise RuntimeError("DepthSplat quality gate context input has no source binding")
    expected_audit_sha256 = _require_sha256(
        source_binding.get("source_audit_input_sha256"), "source audit input"
    )
    expected_audit_tree_sha256 = _require_sha256(
        source_binding.get("source_audit_tree_sha256"), "source audit tree"
    )
    expected_sidecar_tree_sha256 = _require_sha256(
        source_binding.get("source_sidecar_tree_sha256"), "source audit sidecar tree"
    )
    if sha256_file(source_audit_path) != expected_audit_sha256:
        raise RuntimeError("DepthSplat isolated target source audit differs from context audit")
    source_tree = verify_tree_manifest(
        source_audit_root, source_audit_root / ".scarf-manifest.json"
    )
    if source_tree.get("tree_sha256") != expected_audit_tree_sha256:
        raise RuntimeError("DepthSplat isolated target source tree differs from context audit")

    source_audit = _read_json_object(source_audit_path, "source target-free audit")
    source_sample_index = input_identity.get(
        "source_sample_index", SOURCE_SAMPLE_INDEX
    )
    if (
        isinstance(source_sample_index, bool)
        or not isinstance(source_sample_index, int)
        or source_sample_index < 0
    ):
        raise RuntimeError("DepthSplat quality gate context input has an invalid sample index")
    expected_selection = {
        "source_sample_index": source_sample_index,
        "scene": scene,
        "context_indices": list(context_indices),
    }
    selected = source_audit.get("selected_sample")
    canonical_selection = source_audit.get("canonical_selection")
    canonical_protocol = source_audit.get("canonical_protocol")
    if (
        source_audit.get("schema_version") != "1.0"
        or source_audit.get("kind")
        != "dl3dv_target_free_l1_primary_reference_audit_input"
        or source_audit.get("status") != "PASS"
        or source_audit.get("paper_result_eligible") is not False
        or source_audit.get("model") != MODEL
        or source_audit.get("dataset") != DATASET
        or source_audit.get("source_sample_index") != source_sample_index
        or source_audit.get("target_rgb_included") is not False
        or source_audit.get("target_rgb_opened") is not False
        or source_audit.get("target_rgb_paths_passed_to_encoder") is not False
        or not isinstance(selected, Mapping)
        or not isinstance(canonical_selection, Mapping)
        or not isinstance(canonical_protocol, Mapping)
    ):
        raise RuntimeError("DepthSplat isolated target source audit is not target-free")

    target_indices = _require_indices(selected.get("target_indices"), "target indices")
    selection = {**expected_selection, "target_indices": target_indices}
    if (
        dict(canonical_selection) != selection
        or selected.get("scene") != scene
        or selected.get("context_indices") != list(context_indices)
        or canonical_json_sha256(selection)
        != source_binding.get("canonical_selection_sha256")
        or canonical_protocol.get("source_index_sha256")
        != source_binding.get("canonical_index_sha256")
        or canonical_protocol.get("sample_selection_sha256")
        != source_binding.get("canonical_sample_selection_sha256")
    ):
        raise RuntimeError("DepthSplat isolated target selection differs from context audit")

    source = source_audit.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError("DepthSplat isolated target source has no raw provenance")
    for key in (
        "benchmark_metadata_sha256",
        "filelist_sha256",
        "scene_source_plans_sha256",
        "source_record_sha256",
    ):
        _require_sha256(source.get(key), f"isolated target source {key}")
    if not isinstance(source.get("revision"), str) or not source["revision"]:
        raise RuntimeError("DepthSplat isolated target source has an invalid revision")

    sidecar_identity = validate_target_free_input_root(source_audit_root / "sidecar", DATASET)
    if (
        sidecar_identity.get("target_rgb_accessed") is not False
        or sidecar_identity.get("target_rgb_included") is not False
        or sidecar_identity.get("selected_scene_count") != 1
        or sidecar_identity.get("selection_sha256")
        != selected.get("audit_selection_sha256")
        or sidecar_identity.get("tree_sha256") != expected_sidecar_tree_sha256
    ):
        raise RuntimeError("DepthSplat isolated target sidecar is not target-free")
    source_record = load_target_free_record(source_audit_root / "sidecar", scene)
    if (
        source_record.get("key") != scene
        or source_record.get("context_indices") != list(context_indices)
        or source_record.get("target_indices") != target_indices
        or not torch.is_tensor(source_record.get("cameras"))
    ):
        raise RuntimeError("DepthSplat isolated target camera record differs from audit")
    return source_record, dict(source), dict(sidecar_identity), {
        "path": _portable_path(source_audit_path),
        "sha256": expected_audit_sha256,
    }


def _load_isolated_sample0_target_data(
    *,
    raw_root: Path,
    scene: str,
    target_indices: list[int],
    source_record: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[torch.Tensor, list[bytes], list[dict[str, str]], dict[str, Any]]:
    """Read only the fixed sample-zero target frames from the raw benchmark."""

    from data.convert_dl3dv import (
        IMAGE_SUFFIXES,
        _frame_id,
        _load_metadata,
        load_scene_image_sources,
    )

    raw_root = Path(raw_root).resolve()
    raw_source_path = raw_root / ".scarf-dl3dv-source.json"
    expected_source_sha256 = _require_sha256(
        source.get("source_record_sha256"), "isolated target raw source"
    )
    if sha256_file(raw_source_path) != expected_source_sha256:
        raise RuntimeError("DepthSplat isolated target raw source differs from audit")
    raw_source = _read_json_object(raw_source_path, "raw target source")
    if (
        raw_source.get("revision") != source.get("revision")
        or raw_source.get("benchmark_metadata_sha256")
        != source.get("benchmark_metadata_sha256")
        or raw_source.get("filelist_sha256") != source.get("filelist_sha256")
        or raw_source.get("scene_source_plans_sha256")
        != source.get("scene_source_plans_sha256")
    ):
        raise RuntimeError("DepthSplat isolated target raw provenance differs from audit")
    native_sources = load_scene_image_sources(raw_source_path, "native")
    try:
        image_subdir, source_shape = native_sources[scene]
    except KeyError as error:
        raise RuntimeError("DepthSplat isolated target source has no audited scene") from error
    if source_shape != NATIVE_TARGET_IMAGE_SHAPE:
        raise RuntimeError("DepthSplat isolated target source has the wrong native shape")

    scene_root = raw_root / scene / "nerfstudio"
    transforms_path = scene_root / "transforms.json"
    metadata, timestamps = _load_metadata(transforms_path)
    cameras = source_record["cameras"]
    if (
        not torch.is_tensor(cameras)
        or not torch.equal(cameras.detach().cpu(), metadata["cameras"].cpu())
        or max(target_indices) >= len(timestamps)
    ):
        raise RuntimeError("DepthSplat isolated target camera geometry differs from source")

    image_root = scene_root / image_subdir
    images: dict[int, Path] = {}
    for path in image_root.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            frame = _frame_id(path)
            if frame in images:
                raise RuntimeError("DepthSplat isolated target source has duplicate frames")
            images[frame] = path

    opened = [
        {
            "path": _portable_path(raw_source_path),
            "role": "source_plan",
            "sha256": expected_source_sha256,
        },
        {
            "path": _portable_path(transforms_path),
            "role": "target_camera_geometry",
            "sha256": sha256_file(transforms_path),
        },
    ]
    target_images: list[bytes] = []
    for index in target_indices:
        image_path = images.get(timestamps[index])
        if image_path is None:
            raise RuntimeError("DepthSplat isolated target source is missing a target frame")
        target_images.append(image_path.read_bytes())
        opened.append(
            {
                "path": _portable_path(image_path),
                "role": "target_rgb",
                "sha256": sha256_file(image_path),
            }
        )
    return cameras, target_images, opened, {
        "path": _portable_path(raw_source_path),
        "sha256": expected_source_sha256,
        "revision": source["revision"],
        "native_image_subdir": image_subdir,
        "native_image_shape": list(source_shape),
    }


def _load_isolated_target_batch_after_packet_commit(
    loader: Any,
    bundle: Any,
    *,
    scene: str,
    context_indices: list[int],
    input_identity: Mapping[str, Any],
    native_preprocessing: Mapping[str, Any],
    source_audit_root: Path = DEFAULT_SOURCE_AUDIT_ROOT,
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    """Build the audited target mapping without opening a native DL3DV chunk."""

    from integration.model_loader import (
        _calibration_camera_geometry,
        _decode_calibration_context_images,
    )

    source_record, source, sidecar_identity, source_audit_identity = (
        _load_isolated_source_target_record(
            source_audit_root=source_audit_root,
            scene=scene,
            context_indices=context_indices,
            input_identity=input_identity,
        )
    )
    target_indices = _require_indices(source_record["target_indices"], "target indices")
    cameras, target_image_bytes, opened_source_files, raw_source_identity = (
        _load_isolated_sample0_target_data(
            raw_root=raw_root,
            scene=scene,
            target_indices=target_indices,
            source_record=source_record,
            source=source,
        )
    )
    target_images = _decode_calibration_context_images(target_image_bytes)
    if tuple(target_images.shape) != (
        len(target_indices),
        3,
        *NATIVE_TARGET_IMAGE_SHAPE,
    ):
        raise RuntimeError("DepthSplat isolated target RGB has the wrong native shape")
    extrinsics, intrinsics = _calibration_camera_geometry(cameras)
    if max([*context_indices, *target_indices]) >= extrinsics.shape[0]:
        raise RuntimeError("DepthSplat isolated target camera index exceeds source geometry")

    loader._setup_imports()
    try:
        from src.dataset.shims.crop_shim import apply_crop_shim_to_views
        from src.dataset.shims.patch_shim import apply_patch_shim_to_views

        dataset_cfg = bundle.config.dataset
        context_extrinsics = extrinsics[context_indices]
        scale: torch.Tensor | float = 1.0
        if len(context_indices) == 2 and bool(
            getattr(dataset_cfg, "make_baseline_1", False)
        ):
            scale = (context_extrinsics[0, :3, 3] - context_extrinsics[1, :3, 3]).norm()
            if float(scale) < float(getattr(dataset_cfg, "baseline_epsilon", 0.0)):
                raise RuntimeError("DepthSplat isolated target has an insufficient baseline")
            extrinsics = extrinsics.clone()
            extrinsics[:, :3, 3] /= scale
        near_value = float(getattr(dataset_cfg, "near", -1.0))
        far_value = float(getattr(dataset_cfg, "far", -1.0))
        near_value = 0.1 if near_value == -1.0 else near_value
        far_value = 1000.0 if far_value == -1.0 else far_value
        nf_scale: torch.Tensor | float = (
            scale if bool(getattr(dataset_cfg, "baseline_scale_bounds", True)) else 1.0
        )
        target = {
            "extrinsics": extrinsics[target_indices].unsqueeze(0),
            "intrinsics": intrinsics[target_indices].unsqueeze(0),
            "image": target_images.unsqueeze(0),
            "near": torch.full((1, len(target_indices)), near_value) / nf_scale,
            "far": torch.full((1, len(target_indices)), far_value) / nf_scale,
            "index": torch.tensor(target_indices, dtype=torch.long).unsqueeze(0),
        }
        image_shape = tuple(bundle.config.dataset.image_shape)
        if tuple(target["image"].shape[-2:]) != image_shape:
            target = apply_crop_shim_to_views(target, image_shape)
        patch_size = int(getattr(bundle.encoder.cfg, "shim_patch_size", 1)) * int(
            getattr(bundle.encoder.cfg, "downscale_factor", 1)
        )
        if patch_size < 1:
            raise RuntimeError("DepthSplat isolated target has an invalid patch size")
        target = apply_patch_shim_to_views(target, patch_size)
    finally:
        loader._restore_cwd()

    expected_shape = native_preprocessing.get("prepared_image_shape")
    if (
        not isinstance(expected_shape, list)
        or len(expected_shape) != 2
        or [int(value) for value in target["image"].shape[-2:]] != expected_shape
    ):
        raise RuntimeError("DepthSplat isolated target preprocessing differs from context audit")
    return target, target_indices, {
        "reader": ISOLATED_TARGET_READER,
        "source_audit_input": source_audit_identity,
        "source_sidecar": {
            "tree_sha256": sidecar_identity["tree_sha256"],
            "manifest_sha256": sidecar_identity["manifest_sha256"],
            "input_provenance_sha256": sidecar_identity["input_provenance_sha256"],
            "selection_sha256": sidecar_identity["selection_sha256"],
        },
        "raw_source": raw_source_identity,
        "opened_source_files": opened_source_files,
        "native_dl3dv_chunk_loader_used": False,
        "data_module_used": False,
        "nonselected_dl3dv_sample_data_opened": False,
    }


def _target_cameras(target: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    required = ("extrinsics", "intrinsics", "near", "far")
    if any(not torch.is_tensor(target.get(name)) for name in required):
        raise RuntimeError("DepthSplat native target cameras are incomplete")
    return {name: target[name].to(device) for name in required}


def _render_target_view(
    decoder: Any,
    gaussians: Any,
    *,
    target: Mapping[str, torch.Tensor],
    image_shape: tuple[int, int],
) -> torch.Tensor:
    if not callable(getattr(decoder, "forward", None)):
        raise RuntimeError("DepthSplat quality gate has no native decoder")
    with torch.no_grad():
        output = decoder.forward(
            gaussians,
            target["extrinsics"],
            target["intrinsics"],
            target["near"],
            target["far"],
            image_shape,
            depth_mode=None,
        )
    color = getattr(output, "color", None)
    if not torch.is_tensor(color) or color.ndim != 5 or not bool(torch.isfinite(color).all()):
        raise RuntimeError("DepthSplat native target decoder did not emit finite color")
    return color


def _take_target_rgb_for_metrics(target: dict[str, Any], device: torch.device) -> torch.Tensor:
    image = target.pop("image", None)
    if not torch.is_tensor(image) or image.ndim != 5:
        raise RuntimeError("DepthSplat quality gate target RGB is invalid")
    return image.to(device)


def collect_depthsplat_l0_l1_quality_gate(
    *,
    input_root: Path,
    formal_audit_path: Path,
    literal_t4_v16_record: Path,
    device: torch.device,
    literal_t4_acid_plan_path: Path = DEFAULT_PLAN_PATH,
    literal_t4_acid_materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
    source_audit_root: Path = DEFAULT_SOURCE_AUDIT_ROOT,
) -> dict[str, Any]:
    """Run the fixed quality gate after all target-free invariants pass."""

    from data.context_only_audit_input import validate_context_only_audit_input
    from scripts.ae_config import resolve_claim_selection, resolve_experiment

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
    checkpoint_sha256 = cached_sha256_file(experiment.checkpoint)
    formal_audit, formal_audit_binding = _validate_formal_audit(
        formal_audit_path,
        input_identity=input_identity,
        literal_guard=literal_guard_projection,
        literal_profile=literal_profile,
        backend_identity=backend_identity,
        checkpoint_sha256=checkpoint_sha256,
    )

    loader = create_model_loader(MODEL)
    bundle = loader.load_model(
        str(experiment.checkpoint),
        device=device,
        experiment_name=experiment.experiment,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        hydra_overrides=experiment.hydra_overrides,
    )
    if bundle.decoder is None:
        raise RuntimeError("DepthSplat quality gate requires the native decoder")
    context_data = load_context_only_audit_data(
        loader, bundle, input_root=Path(input_root), model_name=MODEL
    )
    context_cpu, loaded_calibration = _require_loaded_context_batch(
        context_data.batch, input_identity
    )
    context = {
        key: value.to(bundle.device) if torch.is_tensor(value) else value
        for key, value in context_cpu.items()
    }
    bundle.model.eval()

    with strict_fp32_convolution_execution() as numerical_execution:
        rebuilt = _rebuild_materialized_packet(
            encoder=bundle.encoder,
            context=context,
            literal_guard=literal_guard,
            literal_profile=literal_profile,
        )
        _require_rebuilt_trace_matches_audit(formal_audit, rebuilt["evidence"])

        # Packet and audit bindings are complete. This is the first target-side
        # operation in the gate; it bypasses the chunked native test loader so
        # sample-one and sample-two payloads cannot enter this process.
        target_mapping, target_indices, isolated_target_provenance = (
            _load_isolated_target_batch_after_packet_commit(
                loader,
                bundle,
                scene=input_identity["scene"],
                context_indices=list(input_identity["context_indices"]),
                input_identity=input_identity,
                native_preprocessing=loaded_calibration["native_preprocessing"],
                source_audit_root=source_audit_root,
            )
        )
        target_cameras = _target_cameras(target_mapping, bundle.device)
        materialized_gaussians = rebuilt["materialized"].as_single_batch(
            type(rebuilt["execution"].dense_gaussians)
        )
        materialized_color = _render_target_view(
            bundle.decoder,
            materialized_gaussians,
            target=target_cameras,
            image_shape=(rebuilt["height"], rebuilt["width"]),
        )
        baseline_color = _render_target_view(
            bundle.decoder,
            rebuilt["execution"].dense_gaussians,
            target=target_cameras,
            image_shape=(rebuilt["height"], rebuilt["width"]),
        )
        target_rgb = _take_target_rgb_for_metrics(target_mapping, bundle.device)

    if baseline_color.shape != materialized_color.shape or baseline_color.shape != target_rgb.shape:
        raise RuntimeError("DepthSplat quality gate render and target RGB shapes differ")
    baseline_views = _view_metrics(baseline_color[0], target_rgb[0])
    materialized_views = _view_metrics(materialized_color[0], target_rgb[0])
    baseline_quality = _mean_metrics(baseline_views)
    materialized_quality = _mean_metrics(materialized_views)
    verdict = _quality_verdict(baseline_quality, materialized_quality)

    return {
        "schema_version": QUALITY_GATE_SCHEMA_VERSION,
        "kind": QUALITY_GATE_KIND,
        "status": "PASS" if verdict["pass"] else "QUALITY_FAILED",
        "paper_result_eligible": False,
        "model": MODEL,
        "dataset": DATASET,
        "source_sample_index": SOURCE_SAMPLE_INDEX,
        "scene": input_identity["scene"],
        "context_indices": list(input_identity["context_indices"]),
        "target_indices": target_indices,
        "formal_target_free_audit": formal_audit_binding,
        "audit_replay": {
            "reconstructed_context_only_packet_matches_formal_audit": True,
            "verified_trace_fields": sorted(rebuilt["evidence"]),
            "nonzero_merge_applied": True,
            "nonzero_merge_update_slot_count": rebuilt[
                "nonzero_merge_update_slot_count"
            ],
        },
        "context_only_input": {
            "identity": input_identity,
            "loaded_native_preprocessing": loaded_calibration["native_preprocessing"],
        },
        "literal_t4_v16": _expected_literal_binding(
            literal_guard=literal_guard_projection, literal_profile=literal_profile
        ),
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": False,
            "target_mapping_constructed_after_formal_audit_and_packet_commit": True,
            "target_camera_metadata_accessed_before_packet_commit": False,
            "target_camera_metadata_accessed_after_packet_commit": True,
            "target_index_accessed_before_packet_commit": False,
            "target_index_accessed_after_packet_commit": True,
            "target_rgb_passed_to_encoder_or_route": False,
            "first_target_rgb_transfer_after_packet_and_baseline_outputs": True,
            "isolated_sample0_raw_reader": True,
            **isolated_target_provenance,
        },
        "execution_boundary": {
            "formal_target_free_audit_revalidated_before_target_mapping": True,
            "formal_context_only_sidecar_used": True,
            "source_bound_packet_adapter_executed": True,
            "nonzero_l0_l1_materializer_executed": True,
            "native_packet_decoder_executed": True,
            "independent_dense_baseline_executed_after_packet_decoder": True,
            "quality_metrics_computed": True,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
            "timing_claim": False,
        },
        "nonzero_merge_applied": True,
        "nonzero_merge_update_slot_count": rebuilt[
            "nonzero_merge_update_slot_count"
        ],
        **rebuilt["evidence"],
        "quality": {
            "baseline": baseline_quality,
            "materialized": materialized_quality,
            "verdict": verdict,
            "views": [
                {
                    "target_index": target_indices[index],
                    "baseline": baseline_views[index],
                    "materialized": materialized_views[index],
                }
                for index in range(len(baseline_views))
            ],
        },
        "backend_identity": backend_identity,
        "checkpoint_sha256": checkpoint_sha256,
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
    parser.add_argument("--formal-audit", type=Path, required=True)
    parser.add_argument("--literal-t4-v16-record", type=Path, required=True)
    parser.add_argument("--source-audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("DepthSplat formal quality gate requires CUDA")
    random.seed(QUALITY_GATE_SEED)
    np.random.seed(QUALITY_GATE_SEED)
    torch.manual_seed(QUALITY_GATE_SEED)
    torch.cuda.manual_seed_all(QUALITY_GATE_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        record = collect_depthsplat_l0_l1_quality_gate(
            input_root=args.input_root,
            formal_audit_path=args.formal_audit,
            literal_t4_v16_record=args.literal_t4_v16_record,
            device=device,
            source_audit_root=args.source_audit_root,
        )
        exit_code = 0 if record["status"] == "PASS" else 1
    except Exception as error:
        record = {
            "schema_version": QUALITY_GATE_SCHEMA_VERSION,
            "kind": QUALITY_GATE_KIND,
            "status": "FAILED",
            "paper_result_eligible": False,
            "model": MODEL,
            "dataset": DATASET,
            "source_sample_index": SOURCE_SAMPLE_INDEX,
            "failure": {"type": type(error).__name__, "message": str(error)},
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
