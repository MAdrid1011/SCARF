"""DepthSplat-only ACID-disjoint V15D/V16D calibration records.

The existing ``evaluation_disjoint_l1_calibration`` module is deliberately
classic-only: its V16 evidence is a classic raw-head execution trace.  This
module does not import it or a classic backend.  It binds the same validated
ACID 24/8 context-only sidecars to the native DepthSplat DL3DV identity and
to the target-free, nonzero L0/L1 merge evidence emitted by the DepthSplat
materializer.

It has two deliberately separate states:

* ``PLANNED_CONTEXT_ONLY_GPU_NOT_RUN`` is a collection plan.  It proves input
  identity but contains no observed thresholds and is not usable at runtime.
* ``FROZEN_EVALUATION_DISJOINT_TARGET_FREE`` is currently available only for
  V15D's source-bound route residual.  V16D has an explicit schema gap and
  cannot be frozen until the native attribute binding and after-scale coverage
  certificate exist.

The latter remains paper-ineligible until a later DL3DV target-free audit and
quality gate consume it.  No target RGB, target camera/index, renderer, or
quality metric is represented in a collection observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPLIT = "calibration_train"
HOLDOUT_SPLIT = "calibration_holdout"
SPLITS = (TRAIN_SPLIT, HOLDOUT_SPLIT)

SCHEMA_VERSION = "1.0"
V15D_KIND = "depthsplat-nonzero-l0-l1-acid-disjoint-v15d"
V16D_KIND = "depthsplat-nonzero-l0-l1-acid-disjoint-v16d-schema-gap"
COLLECTION_PLAN_KIND = "depthsplat-nonzero-l0-l1-acid-disjoint-collection-plan"
FROZEN_STATUS = "FROZEN_EVALUATION_DISJOINT_TARGET_FREE"
NOT_RUN_STATUS = "PLANNED_CONTEXT_ONLY_GPU_NOT_RUN"
V16D_SCHEMA_GAP_STATUS = "BLOCKED_NATIVE_ATTRIBUTE_BINDING_AND_COVERAGE_CERTIFICATE"

MECHANISM_ID = "depthsplat-nonzero-l0-l1-moment-merge-v1"
V15D_SIGNAL = "s1-inverse-distance-leave-one-out-minimum-mean-square-residual-v1"
V15D_THRESHOLD_RULE = "train-global-p50"
V15D_L1_ANCHOR_SEMANTICS = "engineering-lightweight-15-adaptive-center-s1-loo-v1"
V16D_FREEZE_BLOCKERS = (
    "native_dense_attribute_full_slot_provenance_binding_missing",
    "two_sigma_virtual_support_containment_certificate_missing",
    "accepted_l0_l1_update_slot_set_binding_missing",
    "coverage_geometry_certificate_schema_binding_missing",
    "depthsplat_selected_anchor_risk_contract_missing",
)
COLLECTOR_SOURCE_CONTRACT = "depthsplat-acid-disjoint-v15d-collector-source-v1"
_COLLECTOR_SOURCE_FILES = (
    "saes/depthsplat_acid_disjoint_calibration.py",
    "scripts/saes_depthsplat_acid_disjoint_l1_calibration.py",
    "saes/depthsplat_acid_context_only_collector.py",
    "scripts/saes_depthsplat_acid_context_only_collector.py",
    "integration/acid_joint_context.py",
    "integration/acid_joint_model_context.py",
    "data/plan_acid_joint_calibration.py",
)

DEFAULT_PLAN_PATH = ROOT / "artifact" / "protocol" / "acid_joint_calibration_plan.json"
DEFAULT_MATERIALIZATION_ROOT = (
    ROOT / "outputs" / "calibration" / "acid_joint_calibration_v1_context_only"
)

ACCESS_KEYS = (
    "target_mapping_present",
    "target_rgb_accessed",
    "target_camera_metadata_accessed",
    "target_index_accessed",
    "skipped_s3_attributes_accessed",
)
_SHA256_CHARS = frozenset("0123456789abcdef")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-serializable record in the on-disk canonical form."""

    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("DepthSplat calibration payload is not JSON-serializable") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"DepthSplat calibration cannot read {Path(path).name}") from error
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _finite_quantile(values: Sequence[float], fraction: float) -> float:
    if not values or not isinstance(fraction, float) or not 0.0 <= fraction <= 1.0:
        raise ValueError("DepthSplat calibration requires a nonempty valid quantile")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise ValueError("DepthSplat calibration values must be finite and nonnegative")
    return ordered[round((len(ordered) - 1) * fraction)]


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("DepthSplat calibration summary requires values")
    return {
        "candidate_count": len(values),
        "minimum": _finite_quantile(values, 0.0),
        "p10": _finite_quantile(values, 0.10),
        "p25": _finite_quantile(values, 0.25),
        "p50": _finite_quantile(values, 0.50),
        "p75": _finite_quantile(values, 0.75),
        "p90": _finite_quantile(values, 0.90),
        "p95": _finite_quantile(values, 0.95),
        "p99": _finite_quantile(values, 0.99),
        "maximum": _finite_quantile(values, 1.0),
    }


def _access_record() -> dict[str, bool]:
    return {key: False for key in ACCESS_KEYS}


def _validate_acid_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the reusable ACID 24/8 context-only binding surface."""

    required = {
        "schema_version",
        "kind",
        "dataset",
        "protocol_id",
        "plan_sha256",
        "plan_file_sha256",
        "materialization_record_sha256",
        "materialization_tree_sha256",
        "materialization_manifest_sha256",
        "splits",
        "access",
    }
    if not isinstance(binding, Mapping) or set(binding) != required:
        raise ValueError("DepthSplat ACID binding has an invalid schema")
    if (
        binding.get("schema_version") != SCHEMA_VERSION
        or binding.get("kind") != "saes-acid-context-only-calibration-binding"
        or binding.get("dataset") != "acid"
        or not isinstance(binding.get("protocol_id"), str)
        or not binding["protocol_id"]
        or binding.get("access") != _access_record()
    ):
        raise ValueError("DepthSplat ACID binding is not target-free context-only")
    for field in (
        "plan_sha256",
        "plan_file_sha256",
        "materialization_record_sha256",
        "materialization_tree_sha256",
        "materialization_manifest_sha256",
    ):
        _require_sha256(binding.get(field), f"DepthSplat ACID {field}")

    raw_splits = binding.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(SPLITS):
        raise ValueError("DepthSplat ACID binding splits are invalid")
    normalized_splits: dict[str, dict[str, Any]] = {}
    for split, expected_count in ((TRAIN_SPLIT, 24), (HOLDOUT_SPLIT, 8)):
        value = raw_splits.get(split)
        required_split = {
            "scenes",
            "scene_count",
            "scene_set_sha256",
            "selection_sha256",
            "sidecar_tree_sha256",
            "sidecar_manifest_sha256",
            "input_provenance_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required_split:
            raise ValueError(f"DepthSplat ACID {split} binding is invalid")
        scenes = value.get("scenes")
        if (
            not isinstance(scenes, list)
            or len(scenes) != expected_count
            or value.get("scene_count") != expected_count
            or any(not isinstance(scene, str) or not scene for scene in scenes)
            or len(set(scenes)) != len(scenes)
            or canonical_sha256(sorted(scenes)) != value.get("scene_set_sha256")
        ):
            raise ValueError(f"DepthSplat ACID {split} scenes are invalid")
        for field in required_split - {"scenes", "scene_count"}:
            _require_sha256(value.get(field), f"DepthSplat ACID {split} {field}")
        normalized_splits[split] = dict(value)
    if set(normalized_splits[TRAIN_SPLIT]["scenes"]) & set(
        normalized_splits[HOLDOUT_SPLIT]["scenes"]
    ):
        raise ValueError("DepthSplat ACID train and holdout scenes overlap")
    return {**dict(binding), "splits": normalized_splits, "access": _access_record()}


def resolve_depthsplat_acid_binding(
    *,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Load the live validated ACID 24/8 context-only sidecar identity."""

    # Keep the sidecar binding local to the native DepthSplat route.  In
    # particular, importing the classic V16 record implementation here would
    # accidentally couple this route to a foreign raw-head evidence contract.
    from data.plan_acid_joint_calibration import validate_materialization

    plan_path = Path(plan_path).resolve()
    materialization_root = Path(materialization_root).resolve()
    plan = _read_json(plan_path, "DepthSplat ACID calibration plan")
    materialization = validate_materialization(materialization_root, plan=plan)
    partition = plan.get("partition")
    if not isinstance(partition, Mapping):
        raise ValueError("DepthSplat ACID plan has no partition")
    materialization_record = materialization_root / "materialization.json"
    if not materialization_record.is_file():
        raise ValueError("DepthSplat ACID materialization record is missing")
    splits: dict[str, Any] = {}
    for split, expected_count in ((TRAIN_SPLIT, 24), (HOLDOUT_SPLIT, 8)):
        partition_record = partition.get(split)
        sidecar = materialization.get("sidecars", {}).get(split)
        if not isinstance(partition_record, Mapping) or not isinstance(sidecar, Mapping):
            raise ValueError(f"DepthSplat ACID binding has no {split} identity")
        scenes = partition_record.get("scenes")
        if (
            not isinstance(scenes, list)
            or len(scenes) != expected_count
            or any(not isinstance(scene, str) or not scene for scene in scenes)
            or len(set(scenes)) != len(scenes)
            or partition_record.get("scene_count") != expected_count
            or sidecar.get("scene_count") != expected_count
        ):
            raise ValueError(f"DepthSplat ACID {split} identity is invalid")
        splits[split] = {
            "scenes": list(scenes),
            "scene_count": expected_count,
            "scene_set_sha256": _require_sha256(
                partition_record.get("scene_set_sha256"),
                f"DepthSplat ACID {split} scene set",
            ),
            "selection_sha256": _require_sha256(
                sidecar.get("selection_sha256"), f"DepthSplat ACID {split} selection"
            ),
            "sidecar_tree_sha256": _require_sha256(
                sidecar.get("tree_sha256"), f"DepthSplat ACID {split} sidecar tree"
            ),
            "sidecar_manifest_sha256": _require_sha256(
                sidecar.get("manifest_sha256"),
                f"DepthSplat ACID {split} sidecar manifest",
            ),
            "input_provenance_sha256": _require_sha256(
                sidecar.get("input_provenance_sha256"),
                f"DepthSplat ACID {split} input provenance",
            ),
        }
    if set(splits[TRAIN_SPLIT]["scenes"]) & set(splits[HOLDOUT_SPLIT]["scenes"]):
        raise ValueError("DepthSplat ACID train and holdout scenes overlap")
    if plan.get("dataset") != "acid" or plan.get("paper_result_eligible") is not False:
        raise ValueError("DepthSplat ACID plan has the wrong scope")
    return _validate_acid_binding(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "saes-acid-context-only-calibration-binding",
            "dataset": "acid",
            "protocol_id": plan.get("protocol_id"),
            "plan_sha256": _require_sha256(plan.get("plan_sha256"), "DepthSplat ACID plan"),
            "plan_file_sha256": _sha256_file(plan_path),
            "materialization_record_sha256": _sha256_file(materialization_record),
            "materialization_tree_sha256": _require_sha256(
                materialization.get("tree_sha256"), "DepthSplat ACID materialization tree"
            ),
            "materialization_manifest_sha256": _require_sha256(
                materialization.get("manifest_sha256"),
                "DepthSplat ACID materialization manifest",
            ),
            "splits": splits,
            "access": _access_record(),
        }
    )


def _validate_backend_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the stable native identity shape without importing classic code."""

    required = {
        "schema_version",
        "model",
        "dataset",
        "experiment",
        "environment_profile",
        "gaussian_regressor_module",
        "gaussian_head_module",
        "gaussian_adapter_module",
        "decoder_module",
        "raw_descriptor_layout",
        "coordinate_semantics",
        "checkpoint",
        "evaluation_index",
        "runtime_source",
        "simulator_source_files",
        "source_files",
        "submodule_git_head",
    }
    if not isinstance(identity, Mapping) or set(identity) != required:
        raise ValueError("DepthSplat backend identity has an invalid schema")
    if (
        identity.get("schema_version") != "depthsplat-backend-frozen-identity-v1"
        or identity.get("model") != "depthsplat"
        or identity.get("dataset") != "dl3dv"
        or identity.get("experiment") != "dl3dv"
        or identity.get("environment_profile") != "depthsplat"
        or identity.get("raw_descriptor_layout")
        != "opacity-logit-offset-xy-adapter-body-v1"
        or identity.get("coordinate_semantics")
        != "depthsplat-z-depth-pixel-center-plus-sigmoid-offset-rgb-sh-v1"
    ):
        raise ValueError("DepthSplat backend identity is not the native DL3DV route")
    checkpoint = identity.get("checkpoint")
    evaluation_index = identity.get("evaluation_index")
    runtime_source = identity.get("runtime_source")
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"path", "sha256"}
        or checkpoint.get("path") != "depthsplat/checkpoints/dl3dv.ckpt"
        or not isinstance(evaluation_index, Mapping)
        or not isinstance(runtime_source, Mapping)
    ):
        raise ValueError("DepthSplat backend identity application binding is invalid")
    _require_sha256(checkpoint.get("sha256"), "DepthSplat checkpoint")
    _require_sha256(evaluation_index.get("sha256"), "DepthSplat evaluation index")
    _require_sha256(
        evaluation_index.get("source_index_sha256"), "DepthSplat source index"
    )
    _require_sha256(
        evaluation_index.get("sample_selection_sha256"), "DepthSplat sample selection"
    )
    _require_sha256(runtime_source.get("src_tree_sha256"), "DepthSplat source tree")
    if (
        isinstance(evaluation_index.get("sample_count"), bool)
        or not isinstance(evaluation_index.get("sample_count"), int)
        or evaluation_index["sample_count"] < 1
        or not isinstance(identity.get("simulator_source_files"), list)
        or not identity["simulator_source_files"]
        or not isinstance(identity.get("source_files"), Mapping)
        or not identity["source_files"]
        or not isinstance(identity.get("submodule_git_head"), str)
        or len(identity["submodule_git_head"]) != 40
    ):
        raise ValueError("DepthSplat backend identity source binding is invalid")
    for source in identity["simulator_source_files"]:
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
            raise ValueError("DepthSplat simulator source identity is invalid")
        _require_sha256(source.get("sha256"), "DepthSplat simulator source")
    for source in identity["source_files"].values():
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
            raise ValueError("DepthSplat native source identity is invalid")
        _require_sha256(source.get("sha256"), "DepthSplat native source")
    return dict(identity)


def _collector_source_identity(root: Path = ROOT) -> dict[str, Any]:
    """Bind the collector and context-only sidecar interpreter itself."""

    root = Path(root).resolve()
    files: list[dict[str, str]] = []
    for relative_path in _COLLECTOR_SOURCE_FILES:
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError("DepthSplat collector source file is invalid")
        files.append({"path": relative_path, "sha256": _sha256_file(path)})
    return {
        "contract": COLLECTOR_SOURCE_CONTRACT,
        "files": files,
        "tree_sha256": canonical_sha256(files),
    }


def _application(
    backend_identity: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    return {
        "model": "depthsplat",
        "dataset": "dl3dv",
        "backend_identity": _validate_backend_identity(backend_identity),
        "collector_source": _collector_source_identity(root),
    }


def _route_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"L0", "L1", "Full"}:
        raise ValueError("DepthSplat route counts are invalid")
    normalized: dict[str, int] = {}
    for name in ("L0", "L1", "Full"):
        count = value.get(name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("DepthSplat route count is invalid")
        normalized[name] = count
    if normalized["L0"] + normalized["L1"] < 1:
        raise ValueError("DepthSplat calibration needs at least one nonzero merge route")
    return normalized


def _normalize_evidence(value: Any) -> dict[str, Any]:
    """Keep scene evidence compact and reject any metric or target-side field."""

    required = {"native_execution", "packet", "materialization"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("DepthSplat calibration evidence has an invalid schema")
    native = value.get("native_execution")
    expected_native = {
        "routing_features_sha256",
        "routing_z_depths_sha256",
        "native_dense_gaussian_regressor_executed",
        "selected_replicate_gaussian_head_executed",
        "selected_native_rgb_adapter_executed",
        "renderer_executed",
        "quality_metrics_computed",
        "whole_pipeline_s2_s3_sparse_execution_verified",
        "global_s2_s3_savings_claimed",
    }
    if not isinstance(native, Mapping) or set(native) != expected_native:
        raise ValueError("DepthSplat native execution evidence is invalid")
    for field in ("routing_features_sha256", "routing_z_depths_sha256"):
        _require_sha256(native.get(field), f"DepthSplat {field}")
    if native != {
        "routing_features_sha256": native["routing_features_sha256"],
        "routing_z_depths_sha256": native["routing_z_depths_sha256"],
        "native_dense_gaussian_regressor_executed": True,
        "selected_replicate_gaussian_head_executed": True,
        "selected_native_rgb_adapter_executed": True,
        "renderer_executed": False,
        "quality_metrics_computed": False,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
    }:
        raise ValueError("DepthSplat calibration execution crossed its target-free boundary")

    packet = value.get("packet")
    expected_packet = {
        "selected_descriptor_count",
        "selection_mask_sha256",
        "source_rgb_sha256",
        "source_trace_sha256",
        "native_execution_sha256",
        "initial_native_attribute_binding_sha256",
        "final_selected_adapter_source_trace_sha256",
        "final_selected_adapter_binding_sha256",
        "materialized_source_trace_sha256",
        "materialized_current_binding_sha256",
    }
    if not isinstance(packet, Mapping) or set(packet) != expected_packet:
        raise ValueError("DepthSplat calibration packet evidence is invalid")
    count = packet.get("selected_descriptor_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("DepthSplat packet has no selected descriptors")
    for field in expected_packet - {"selected_descriptor_count"}:
        _require_sha256(packet.get(field), f"DepthSplat packet {field}")

    materialization = value.get("materialization")
    expected_materialization = {
        "plan_selection_mask_sha256",
        "plan_tile_trace_sha256",
        "l1_anchor_semantics",
        "preflight_tile_trace_sha256",
        "preflight_events_sha256",
        "route_tile_trace_sha256",
        "route_events_sha256",
        "selected_output_mask_sha256",
        "additional_full_mask_sha256",
        "raw_head_request_mask_sha256",
        "full_passthrough_mask_sha256",
        "materialized_coverage_certificate_sha256",
        "materialized_full_attribute_binding_sha256",
        "materialized_full_passthrough_count",
        "materialized_update_anchor_count",
        "selected_anchor_attribute_loo",
        "route_counts",
    }
    if not isinstance(materialization, Mapping) or set(materialization) != expected_materialization:
        raise ValueError("DepthSplat materialization evidence is invalid")
    if materialization.get("l1_anchor_semantics") != V15D_L1_ANCHOR_SEMANTICS:
        raise ValueError("DepthSplat V15D evidence does not use adaptive L1 anchors")
    for field in expected_materialization - {
        "route_counts",
        "l1_anchor_semantics",
        "materialized_full_passthrough_count",
        "materialized_update_anchor_count",
        "selected_anchor_attribute_loo",
    }:
        _require_sha256(materialization.get(field), f"DepthSplat {field}")
    for field in (
        "materialized_full_passthrough_count",
        "materialized_update_anchor_count",
    ):
        count = materialization.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"DepthSplat {field} is invalid")
    if materialization["materialized_update_anchor_count"] < 1:
        raise ValueError("DepthSplat calibration requires a nonzero materialized merge")
    selected_anchor_attribute_loo = _normalize_selected_anchor_attribute_loo(
        materialization["selected_anchor_attribute_loo"]
    )
    return {
        "native_execution": dict(native),
        "packet": dict(packet),
        "materialization": {
            **{
                field: materialization[field]
                for field in expected_materialization
                - {"route_counts", "selected_anchor_attribute_loo"}
            },
            "selected_anchor_attribute_loo": selected_anchor_attribute_loo,
            "route_counts": _route_counts(materialization["route_counts"]),
        },
    }


def _normalize_scene_records(
    records: Sequence[Mapping[str, Any]],
    *,
    binding: Mapping[str, Any],
    split: str,
    value_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"DepthSplat {split} scene records are invalid")
    expected_scenes = list(binding["splits"][split]["scenes"])
    if len(records) != len(expected_scenes):
        raise ValueError(f"DepthSplat {split} scene count changed")
    normalized: list[dict[str, Any]] = []
    for position, (scene, record) in enumerate(zip(expected_scenes, records)):
        if (
            not isinstance(record, Mapping)
            or set(record) != {"scene", value_key, "evidence", "access"}
            or record.get("scene") != scene
            or record.get("access") != _access_record()
        ):
            raise ValueError(f"DepthSplat {split} scene record is invalid at {position}")
        raw_values = record.get(value_key)
        if (
            not isinstance(raw_values, Sequence)
            or isinstance(raw_values, (str, bytes))
            or not raw_values
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in raw_values
            )
        ):
            raise ValueError(f"DepthSplat {split} calibration values are invalid")
        values = [float(item) for item in raw_values]
        if not all(math.isfinite(item) and item >= 0.0 for item in values):
            raise ValueError(f"DepthSplat {split} calibration values are not finite")
        normalized.append(
            {
                "scene": scene,
                value_key: values,
                "evidence": _normalize_evidence(record.get("evidence")),
                "access": _access_record(),
            }
        )
    return normalized


def _require_false_fields(value: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    if any(value.get(field) is not False for field in fields):
        raise ValueError(f"DepthSplat {label} crossed its target-free boundary")


def _normalize_selected_anchor_attribute_loo(value: Any) -> dict[str, Any]:
    """Compact a selected-only RGB-SH/opacity LOO trace for later V16D audit.

    This is evidence collection only.  V15D continues to threshold its
    source-side adaptive route residual, while V16D remains blocker-only until
    every other native/full/coverage prerequisite is available.
    """

    raw_required = {
        "certificate",
        "q75_risk",
        "held_out_records",
        "held_out_count",
        "selected_label_reads",
        "nonprobe_attribute_reads",
    }
    normalized_required = {
        "certificate",
        "q75_risk",
        "held_out_count",
        "selected_label_reads",
        "nonprobe_attribute_reads",
        "held_out_records_sha256",
        "trace_sha256",
    }
    if not isinstance(value, Mapping) or set(value) not in (
        raw_required,
        normalized_required,
    ):
        raise ValueError("DepthSplat selected-anchor attribute LOO trace is invalid")
    certificate = value.get("certificate")
    risk = value.get("q75_risk")
    count = value.get("held_out_count")
    label_reads = value.get("selected_label_reads")
    nonprobe_reads = value.get("nonprobe_attribute_reads")
    if (
        not isinstance(certificate, str)
        or not certificate
        or isinstance(risk, bool)
        or not isinstance(risk, (int, float))
        or not math.isfinite(float(risk))
        or float(risk) < 0.0
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or isinstance(label_reads, bool)
        or not isinstance(label_reads, int)
        or label_reads != count
        or nonprobe_reads != 0
    ):
        raise ValueError("DepthSplat selected-anchor attribute LOO access is invalid")
    if set(value) == normalized_required:
        for field in ("held_out_records_sha256", "trace_sha256"):
            _require_sha256(value.get(field), f"DepthSplat selected-anchor LOO {field}")
        return {
            "certificate": certificate,
            "q75_risk": float(risk),
            "held_out_count": count,
            "selected_label_reads": label_reads,
            "nonprobe_attribute_reads": 0,
            "held_out_records_sha256": value["held_out_records_sha256"],
            "trace_sha256": value["trace_sha256"],
        }

    records = value.get("held_out_records")
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
        or any(not isinstance(record, Mapping) for record in records)
        or count != len(records)
    ):
        raise ValueError("DepthSplat selected-anchor attribute LOO access is invalid")
    try:
        held_out_records_sha256 = canonical_sha256(list(records))
        trace_sha256 = canonical_sha256(dict(value))
    except ValueError as error:
        raise ValueError("DepthSplat selected-anchor attribute LOO is not serializable") from error
    return {
        "certificate": certificate,
        "q75_risk": float(risk),
        "held_out_count": count,
        "selected_label_reads": label_reads,
        "nonprobe_attribute_reads": 0,
        "held_out_records_sha256": held_out_records_sha256,
        "trace_sha256": trace_sha256,
    }


def build_v15d_scene_record(
    *,
    scene: str,
    plan_events: Mapping[str, Any],
    plan_tile_trace: Sequence[Mapping[str, Any]],
    native_execution_events: Mapping[str, Any],
    selected_head_events: Mapping[str, Any],
    selected_head_equivalence: Mapping[str, Any],
    packet_source_trace: Mapping[str, Any],
    packed_source_trace: Mapping[str, Any],
    final_selected_adapter_source_trace: Mapping[str, Any],
    final_selected_adapter_binding_sha256: str,
    materialized_source_trace: Mapping[str, Any],
    materialized_current_binding_sha256: str,
    selected_descriptor_count: int,
    preflight_events: Mapping[str, Any],
    preflight_tile_trace: Sequence[Mapping[str, Any]],
    final_route_events: Mapping[str, Any],
    final_route_tile_trace: Sequence[Mapping[str, Any]],
    access: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one target-free V15D scene record from native trace objects.

    The formal ACID worker supplies these fields after it has loaded one
    context-only sidecar scene.  This helper deliberately accepts plain trace
    mappings rather than model objects, so it cannot accidentally open a
    target mapping or invoke a renderer while serializing calibration data.
    """

    if not isinstance(scene, str) or not scene:
        raise ValueError("DepthSplat V15D scene name is invalid")
    if access != _access_record():
        raise ValueError("DepthSplat V15D scene is not target-free")
    if (
        not isinstance(plan_events, Mapping)
        or not isinstance(plan_tile_trace, Sequence)
        or isinstance(plan_tile_trace, (str, bytes))
        or plan_events.get("contract_version")
        != "saes-incremental-probe-first-plan-v1"
        or plan_events.get("target_rgb_accessed") is not False
        or plan_events.get("gaussian_attributes_accessed") is not False
        or plan_events.get("tile_size") != 4
        or plan_events.get("l1_anchor_semantics") != V15D_L1_ANCHOR_SEMANTICS
        or plan_events.get("l1_anchor_selection") != V15D_SIGNAL
        or plan_events.get("l1_anchor_selection_uses_s1_only") is not True
    ):
        raise ValueError("DepthSplat V15D plan does not have the adaptive target-free contract")
    plan_trace_sha256 = canonical_sha256(list(plan_tile_trace))
    if plan_events.get("tile_trace_sha256") != plan_trace_sha256:
        raise ValueError("DepthSplat V15D plan trace digest changed")
    plan_selection_sha256 = _require_sha256(
        plan_events.get("selection_mask_sha256"), "DepthSplat V15D plan selection"
    )

    residuals: list[float] = []
    for record in plan_tile_trace:
        if not isinstance(record, Mapping):
            raise ValueError("DepthSplat V15D plan tile record is invalid")
        is_candidate = record.get("pre_guard_route") == "L1" or (
            record.get("pre_guard_route") == "L0"
            and record.get("depth_uniform") is True
        )
        if not is_candidate:
            continue
        value = record.get("adaptive_l1_leave_one_out_residual")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("DepthSplat V15D plan candidate residual is invalid")
        residuals.append(float(value))
    if not residuals:
        raise ValueError("DepthSplat V15D scene has no adaptive L1 residual candidates")

    if not isinstance(native_execution_events, Mapping):
        raise ValueError("DepthSplat V15D native execution evidence is invalid")
    regressor = native_execution_events.get("gaussian_regressor")
    head = native_execution_events.get("gaussian_head")
    adapter = native_execution_events.get("adapter")
    routing = native_execution_events.get("routing")
    if (
        native_execution_events.get("source_bound") is not True
        or not isinstance(regressor, Mapping)
        or regressor.get("native_dense_executed") is not True
        or not isinstance(head, Mapping)
        or head.get("padding_mode") != "replicate"
        or not isinstance(adapter, Mapping)
        or adapter.get("source_rgb_keyword") != "input_images"
        or adapter.get("z_depth_geometry") is not True
        or not isinstance(routing, Mapping)
        or native_execution_events.get("whole_pipeline_s2_s3_sparse_execution_verified")
        is not False
        or native_execution_events.get("global_s2_s3_savings_claimed") is not False
    ):
        raise ValueError("DepthSplat V15D native execution contract changed")
    routing_features_sha256 = _require_sha256(
        routing.get("features_sha256"), "DepthSplat V15D routing features"
    )
    routing_z_depths_sha256 = _require_sha256(
        routing.get("z_depth_sha256"), "DepthSplat V15D routing z-depths"
    )
    native_execution_sha256 = _require_sha256(
        native_execution_events.get("native_execution_sha256"),
        "DepthSplat V15D native execution",
    )
    if (
        not isinstance(selected_head_events, Mapping)
        or selected_head_events.get("padding_mode") != "replicate"
        or selected_head_events.get("selected_final_output_positions") != selected_descriptor_count
        or selected_head_events.get("whole_pipeline_s2_s3_sparse_execution_verified")
        is not False
        or selected_head_events.get("global_s2_s3_savings_claimed") is not False
        or not isinstance(selected_head_equivalence, Mapping)
        or selected_head_equivalence.get("equivalent") is not True
    ):
        raise ValueError("DepthSplat V15D selected-head replay is not source-equivalent")
    if isinstance(selected_descriptor_count, bool) or not isinstance(selected_descriptor_count, int) or selected_descriptor_count < 1:
        raise ValueError("DepthSplat V15D selected descriptor count is invalid")

    required_packet_trace = {
        "source_bound": True,
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
        "source_rgb_keyword": "input_images",
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "head_forward_invocations": 1,
    }
    if (
        not isinstance(packet_source_trace, Mapping)
        or any(packet_source_trace.get(key) != value for key, value in required_packet_trace.items())
        or not isinstance(packed_source_trace, Mapping)
    ):
        raise ValueError("DepthSplat V15D packet is not source-bound")
    if packet_source_trace.get("selection_mask_sha256") != plan_selection_sha256:
        raise ValueError("DepthSplat V15D packet selection changed")
    if packet_source_trace.get("routing_features_sha256") != routing_features_sha256:
        raise ValueError("DepthSplat V15D packet routing feature binding changed")
    if packet_source_trace.get("routing_z_depths_sha256") != routing_z_depths_sha256:
        raise ValueError("DepthSplat V15D packet routing z-depth binding changed")
    selected_descriptor_sha256 = _require_sha256(
        packet_source_trace.get("selected_descriptor_sha256"),
        "DepthSplat V15D selected descriptor",
    )
    selected_rgb_sha256 = _require_sha256(
        packet_source_trace.get("selected_rgb_sha256"), "DepthSplat V15D selected RGB"
    )
    packet_native_execution_sha256 = _require_sha256(
        packet_source_trace.get("native_execution_sha256"),
        "DepthSplat V15D native execution",
    )
    if packet_native_execution_sha256 != native_execution_sha256:
        raise ValueError("DepthSplat V15D packet native execution binding changed")
    initial_native_attribute_binding_sha256 = _require_sha256(
        packed_source_trace.get("native_adapter_attribute_binding_sha256"),
        "DepthSplat V15D initial native Adapter attributes",
    )
    expected_packed_source_trace = dict(packet_source_trace)
    expected_packed_source_trace["native_adapter_attribute_binding_sha256"] = (
        initial_native_attribute_binding_sha256
    )
    if dict(packed_source_trace) != expected_packed_source_trace:
        raise ValueError("DepthSplat V15D initial native Adapter trace changed")
    packed_source_trace_sha256 = canonical_sha256(dict(packed_source_trace))

    final_selected_adapter_binding_sha256 = _require_sha256(
        final_selected_adapter_binding_sha256,
        "DepthSplat V15D final selected native Adapter attributes",
    )
    if not isinstance(final_selected_adapter_source_trace, Mapping):
        raise ValueError("DepthSplat V15D final selected Adapter trace is invalid")
    materialized_current_binding_sha256 = _require_sha256(
        materialized_current_binding_sha256,
        "DepthSplat V15D materialized current attributes",
    )
    if not isinstance(materialized_source_trace, Mapping):
        raise ValueError("DepthSplat V15D materialized packet trace is invalid")

    if (
        not isinstance(preflight_events, Mapping)
        or not isinstance(preflight_tile_trace, Sequence)
        or isinstance(preflight_tile_trace, (str, bytes))
        or preflight_events.get("schema_version")
        != "saes-depthsplat-l0-l1-materializer-v1"
    ):
        raise ValueError("DepthSplat V15D preflight evidence is invalid")
    _require_false_fields(
        preflight_events,
        (
            "target_rgb_accessed",
            "target_rgb_accessed_before_commit",
            "target_camera_accessed_before_commit",
            "skipped_s3_attributes_accessed",
            "whole_pipeline_s2_s3_sparse_execution_verified",
            "global_s2_s3_savings_claimed",
        ),
        "V15D preflight",
    )
    if (
        preflight_events.get("nonzero_direct_deletion") is not False
        or preflight_events.get("not_lossless_deletion") is not True
        or preflight_events.get("source_rgb_sh_initialization") is not True
        or preflight_events.get("z_depth_geometry") is not True
        or preflight_events.get("source_nonprobe_s3_attribute_reads") != 0
        or preflight_events.get("l1_anchor_semantics") != V15D_L1_ANCHOR_SEMANTICS
    ):
        raise ValueError("DepthSplat V15D preflight merge contract changed")
    selected_anchor_attribute_loo = _normalize_selected_anchor_attribute_loo(
        preflight_events.get("selected_anchor_attribute_loo")
    )
    initial_binding = preflight_events.get("initial_binding")
    if not isinstance(initial_binding, Mapping) or initial_binding != {
        "plan_selection_mask_sha256": plan_selection_sha256,
        "plan_tile_trace_sha256": plan_trace_sha256,
        "packet_selection_mask_sha256": plan_selection_sha256,
        "packet_selected_descriptor_sha256": selected_descriptor_sha256,
        "packet_selected_rgb_sha256": selected_rgb_sha256,
        "native_execution_sha256": native_execution_sha256,
        "packed_source_trace_sha256": packed_source_trace_sha256,
        "packed_native_attribute_binding_sha256": initial_native_attribute_binding_sha256,
        "routing_features_sha256": routing_features_sha256,
        "routing_z_depths_sha256": routing_z_depths_sha256,
    }:
        raise ValueError("DepthSplat V15D preflight initial binding changed")
    preflight_trace_sha256 = canonical_sha256(list(preflight_tile_trace))
    if preflight_events.get("tile_trace_sha256") != preflight_trace_sha256:
        raise ValueError("DepthSplat V15D preflight trace digest changed")

    if (
        not isinstance(final_route_events, Mapping)
        or not isinstance(final_route_tile_trace, Sequence)
        or isinstance(final_route_tile_trace, (str, bytes))
        or final_route_events.get("schema_version")
        != "saes-depthsplat-compact-l0-l1-route-v1"
    ):
        raise ValueError("DepthSplat V15D final route evidence is invalid")
    _require_false_fields(
        final_route_events,
        (
            "target_rgb_accessed",
            "target_rgb_accessed_before_commit",
            "target_camera_accessed_before_commit",
            "skipped_s3_attributes_accessed",
            "whole_pipeline_s2_s3_sparse_execution_verified",
            "global_s2_s3_savings_claimed",
        ),
        "V15D final route",
    )
    if (
        final_route_events.get("nonzero_direct_deletion") is not False
        or final_route_events.get("not_lossless_deletion") is not True
        or final_route_events.get("l1_anchor_semantics") != V15D_L1_ANCHOR_SEMANTICS
        or final_route_events.get("preflight_trace_sha256") != preflight_trace_sha256
    ):
        raise ValueError("DepthSplat V15D final route merge contract changed")
    final_route_trace_sha256 = canonical_sha256(list(final_route_tile_trace))
    if final_route_events.get("tile_trace_sha256") != final_route_trace_sha256:
        raise ValueError("DepthSplat V15D final route trace digest changed")
    full_passthrough_mask_sha256 = _require_sha256(
        final_route_events.get("full_passthrough_mask_sha256"),
        "DepthSplat V15D Full passthrough mask",
    )
    route_coverage_certificate_sha256 = _require_sha256(
        final_route_events.get("coverage_certificate_sha256"),
        "DepthSplat V15D route coverage certificate",
    )
    if (
        final_selected_adapter_source_trace.get("native_execution_sha256")
        != native_execution_sha256
        or final_selected_adapter_source_trace.get(
            "native_adapter_attribute_binding_sha256"
        )
        != final_selected_adapter_binding_sha256
        or final_selected_adapter_source_trace.get("packet_selection_kind")
        != "depthsplat-final-selected-output-mask-v1"
        or final_selected_adapter_source_trace.get("selection_mask_sha256")
        != final_route_events.get("selected_output_mask_sha256")
        or final_selected_adapter_source_trace.get("producer_request_mask_sha256")
        != final_route_events.get("raw_head_request_mask_sha256")
        or final_selected_adapter_source_trace.get(
            "native_full_passthrough_mask_sha256"
        )
        != full_passthrough_mask_sha256
        or final_selected_adapter_source_trace.get(
            "native_full_adapter_attribute_execution_sha256"
        )
        != native_execution_sha256
        or final_selected_adapter_source_trace.get(
            "native_full_adapter_attribute_passthrough_mask_sha256"
        )
        != full_passthrough_mask_sha256
    ):
        raise ValueError("DepthSplat V15D final selected Adapter binding changed")
    full_passthrough_count = final_selected_adapter_source_trace.get(
        "native_full_passthrough_positions"
    )
    if (
        isinstance(full_passthrough_count, bool)
        or not isinstance(full_passthrough_count, int)
        or full_passthrough_count < 0
        or final_selected_adapter_source_trace.get(
            "native_full_adapter_attribute_passthrough_count"
        )
        != full_passthrough_count
    ):
        raise ValueError("DepthSplat V15D final selected Full provenance changed")
    final_full_attribute_binding_sha256 = _require_sha256(
        final_selected_adapter_source_trace.get(
            "native_full_adapter_attribute_binding_sha256"
        ),
        "DepthSplat V15D final selected Full attributes",
    )
    final_selected_adapter_source_trace_sha256 = canonical_sha256(
        dict(final_selected_adapter_source_trace)
    )
    if (
        materialized_source_trace.get("native_execution_sha256")
        != native_execution_sha256
        or materialized_source_trace.get("native_adapter_attribute_binding_sha256")
        != final_selected_adapter_binding_sha256
        or materialized_source_trace.get(
            "depthsplat_compact_current_attribute_binding_sha256"
        )
        != materialized_current_binding_sha256
        or materialized_source_trace.get("depthsplat_compact_coverage_certificate_sha256")
        != route_coverage_certificate_sha256
        or not isinstance(
            materialized_source_trace.get("depthsplat_compact_coverage_certificate"), str
        )
        or not materialized_source_trace["depthsplat_compact_coverage_certificate"]
        or materialized_source_trace.get("native_full_passthrough_mask_sha256")
        != full_passthrough_mask_sha256
        or materialized_source_trace.get(
            "native_full_adapter_attribute_execution_sha256"
        )
        != native_execution_sha256
        or materialized_source_trace.get(
            "native_full_adapter_attribute_passthrough_mask_sha256"
        )
        != full_passthrough_mask_sha256
        or materialized_source_trace.get(
            "native_full_adapter_attribute_passthrough_count"
        )
        != full_passthrough_count
        or materialized_source_trace.get(
            "native_full_adapter_attribute_binding_sha256"
        )
        != final_full_attribute_binding_sha256
        or materialized_source_trace.get("depthsplat_compact_full_passthrough_count")
        != full_passthrough_count
    ):
        raise ValueError("DepthSplat V15D materialized packet provenance changed")
    materialized_update_anchor_count = materialized_source_trace.get(
        "depthsplat_compact_update_anchor_count"
    )
    if (
        isinstance(materialized_update_anchor_count, bool)
        or not isinstance(materialized_update_anchor_count, int)
        or materialized_update_anchor_count < 1
    ):
        raise ValueError("DepthSplat V15D materialized packet has no nonzero merge")
    materialized_source_trace_sha256 = canonical_sha256(dict(materialized_source_trace))
    materialization = {
        "plan_selection_mask_sha256": plan_selection_sha256,
        "plan_tile_trace_sha256": plan_trace_sha256,
        "l1_anchor_semantics": V15D_L1_ANCHOR_SEMANTICS,
        "preflight_tile_trace_sha256": preflight_trace_sha256,
        "preflight_events_sha256": canonical_sha256(dict(preflight_events)),
        "route_tile_trace_sha256": final_route_trace_sha256,
        "route_events_sha256": canonical_sha256(dict(final_route_events)),
        "selected_output_mask_sha256": _require_sha256(
            final_route_events.get("selected_output_mask_sha256"),
            "DepthSplat V15D selected output mask",
        ),
        "additional_full_mask_sha256": _require_sha256(
            final_route_events.get("additional_full_mask_sha256"),
            "DepthSplat V15D additional Full mask",
        ),
        "raw_head_request_mask_sha256": _require_sha256(
            final_route_events.get("raw_head_request_mask_sha256"),
            "DepthSplat V15D raw request mask",
        ),
        "full_passthrough_mask_sha256": full_passthrough_mask_sha256,
        "materialized_coverage_certificate_sha256": route_coverage_certificate_sha256,
        "materialized_full_attribute_binding_sha256": final_full_attribute_binding_sha256,
        "materialized_full_passthrough_count": full_passthrough_count,
        "materialized_update_anchor_count": materialized_update_anchor_count,
        "selected_anchor_attribute_loo": selected_anchor_attribute_loo,
        "route_counts": _route_counts(final_route_events.get("route_counts")),
    }
    evidence = {
        "native_execution": {
            "routing_features_sha256": routing_features_sha256,
            "routing_z_depths_sha256": routing_z_depths_sha256,
            "native_dense_gaussian_regressor_executed": True,
            "selected_replicate_gaussian_head_executed": True,
            "selected_native_rgb_adapter_executed": True,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        },
        "packet": {
            "selected_descriptor_count": selected_descriptor_count,
            "selection_mask_sha256": plan_selection_sha256,
            "source_rgb_sha256": selected_rgb_sha256,
            "source_trace_sha256": packed_source_trace_sha256,
            "native_execution_sha256": native_execution_sha256,
            "initial_native_attribute_binding_sha256": initial_native_attribute_binding_sha256,
            "final_selected_adapter_source_trace_sha256": final_selected_adapter_source_trace_sha256,
            "final_selected_adapter_binding_sha256": final_selected_adapter_binding_sha256,
            "materialized_source_trace_sha256": materialized_source_trace_sha256,
            "materialized_current_binding_sha256": materialized_current_binding_sha256,
        },
        "materialization": materialization,
    }
    # Run the same strict compact-schema validation used when freezing.
    normalized_evidence = _normalize_evidence(evidence)
    return {
        "scene": scene,
        "adaptive_l1_residuals": residuals,
        "evidence": normalized_evidence,
        "access": _access_record(),
    }


def _split_verification(
    records: Sequence[Mapping[str, Any]], *, value_key: str, threshold: float
) -> dict[str, Any]:
    values = [value for record in records for value in record[value_key]]
    retained = sum(value <= threshold for value in values)
    return {
        "threshold_value": threshold,
        "candidate_count": len(values),
        "retained_count": retained,
        "promoted_count": len(values) - retained,
        "summary": _summary(values),
        "threshold_updated": False,
    }


def _mechanism_record() -> dict[str, Any]:
    return {
        "id": MECHANISM_ID,
        "nonzero_l0_l1_merge_required": True,
        "full_tiles_preserve_native_attributes": True,
        "virtual_gaussians_moment_matched_into_retained_anchors": True,
        "source_rgb_sh_preserved": True,
        "coordinate_semantics": "native-depthsplat-z-depth",
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
    }


def _base_record(
    *,
    kind: str,
    binding: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
    train_scene_records: Sequence[Mapping[str, Any]],
    holdout_scene_records: Sequence[Mapping[str, Any]],
    value_key: str,
    threshold: float,
    threshold_rule: str,
    signal: str,
) -> dict[str, Any]:
    validated_binding = _validate_acid_binding(binding)
    application = _application(backend_identity)
    train = _normalize_scene_records(
        train_scene_records,
        binding=validated_binding,
        split=TRAIN_SPLIT,
        value_key=value_key,
    )
    holdout = _normalize_scene_records(
        holdout_scene_records,
        binding=validated_binding,
        split=HOLDOUT_SPLIT,
        value_key=value_key,
    )
    train_values = [value for record in train for value in record[value_key]]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "status": FROZEN_STATUS,
        "paper_result_eligible": False,
        "application": application,
        "acid_binding": validated_binding,
        "mechanism": _mechanism_record(),
        "signal": {
            "name": signal,
            "source": "target-free-depthsplat-probe-plan-tile-trace",
            "smaller_is_safer": True,
        },
        "split_policy": {
            "train_split": TRAIN_SPLIT,
            "holdout_split": HOLDOUT_SPLIT,
            "threshold_source": "train_only",
            "holdout_threshold_update_allowed": False,
        },
        "access": _access_record(),
        "train_scene_records": train,
        "holdout_scene_records": holdout,
        "train_summary": _summary(train_values),
        "holdout_verification": _split_verification(
            holdout, value_key=value_key, threshold=threshold
        ),
        "threshold": {
            "value": threshold,
            "promote_when": f"{value_key}_gt_value",
            "rule": threshold_rule,
        },
    }


def build_v15d_record(
    *,
    binding: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
    train_scene_records: Sequence[Mapping[str, Any]],
    holdout_scene_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze V15D's route residual from the 24 ACID train scenes only.

    This is the same target-free probe-plan signal used to decide whether an
    adaptive L1 anchor is trustworthy.  It is intentionally not a post-merge
    continuity diagnostic: that diagnostic is a materializer safety guard and
    has no established threshold-to-quality interpretation.
    """

    validated_binding = _validate_acid_binding(binding)
    train = _normalize_scene_records(
        train_scene_records,
        binding=validated_binding,
        split=TRAIN_SPLIT,
        value_key="adaptive_l1_residuals",
    )
    threshold = _finite_quantile(
        [value for record in train for value in record["adaptive_l1_residuals"]], 0.50
    )
    record = _base_record(
        kind=V15D_KIND,
        binding=validated_binding,
        backend_identity=backend_identity,
        train_scene_records=train,
        holdout_scene_records=holdout_scene_records,
        value_key="adaptive_l1_residuals",
        threshold=threshold,
        threshold_rule=V15D_THRESHOLD_RULE,
        signal=V15D_SIGNAL,
    )
    return {**record, "sha256": canonical_sha256(record)}


def build_v16d_record(
    *,
    binding: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
    v15d_record_sha256: str,
) -> dict[str, Any]:
    """Record why V16D cannot yet freeze a threshold.

    ``maximum_relative_residual`` is only an interpolation guard, and the
    current containment lhs is measured before covariance expansion.  The
    packed trace also omits a digest of final means/covariances/SH/opacities.
    A record that converts any of those diagnostics into a threshold would
    look frozen without proving the native attribute path it controls.
    """

    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": V16D_KIND,
        "status": V16D_SCHEMA_GAP_STATUS,
        "paper_result_eligible": False,
        "application": _application(backend_identity),
        "acid_binding": _validate_acid_binding(binding),
        "mechanism": _mechanism_record(),
        "access": _access_record(),
        "base_v15d_sha256": _require_sha256(v15d_record_sha256, "V15D record"),
        "freeze_blockers": list(V16D_FREEZE_BLOCKERS),
        "diagnostic_only": {
            "continuity_maximum_relative_residual": "not-a-calibration-risk",
            "coverage_maximum_containment_lhs": "pre-expansion-not-comparable",
            "coverage_moment_covariance_scale_max": "record-only-until-risk-contract",
            "coverage_center_only_lhs": "not-a-two-sigma-support-certificate",
        },
        "required_future_evidence": {
            "native_full_slot_and_attribute_provenance": [
                "native_execution_sha256",
                "native_dense_attribute_full_slot_provenance_sha256",
                "initial_adapter_attribute_binding_sha256",
                "final_adapter_attribute_binding_sha256",
                "materialized_attribute_binding_sha256",
            ],
            "coverage_certificate": {
                "geometry_certificate_schema_sha256": "must_bind-z-depth-and-2sigma-shape-semantics",
                "two_sigma_virtual_support_lhs_per_anchor": "must_cover-every-virtual-covariance-shape",
                "two_sigma_virtual_support_lhs_per_tile": "must_be_lte_4",
                "moment_covariance_scale_per_anchor": "must_be_bound_to_final_update",
                "moment_covariance_scale_per_tile": "must_be_bound_to_final_update",
            },
            "accepted_update_slots": {
                "accepted_l0_l1_update_slots_sha256": "must_equal-all-and-only-accepted-compact-updates",
                "accepted_l0_l1_update_slot_count": "must_equal-bound-slot-set-count",
                "final_materialized_update_slots_sha256": "must_equal-accepted-slot-set",
            },
            "selected_anchor_risk": "target-free-native-attribute-loo-contract",
        },
    }
    return {**record, "sha256": canonical_sha256(record)}


def build_collection_plan(
    *, binding: Mapping[str, Any], backend_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a collection contract without claiming that a GPU run happened."""

    validated_binding = _validate_acid_binding(binding)
    application = _application(backend_identity)
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": COLLECTION_PLAN_KIND,
        "status": NOT_RUN_STATUS,
        "paper_result_eligible": False,
        "application": application,
        "acid_binding": validated_binding,
        "mechanism": _mechanism_record(),
        "access": _access_record(),
        "collection": {
            "gpu_collection_attempted": False,
            "gpu_collection_completed": False,
            "reason": "DEPTHSPLAT_GPU_COLLECTION_NOT_RUN",
            "expected_scenes": {
                TRAIN_SPLIT: list(validated_binding["splits"][TRAIN_SPLIT]["scenes"]),
                HOLDOUT_SPLIT: list(
                    validated_binding["splits"][HOLDOUT_SPLIT]["scenes"]
                ),
            },
            "v15d_scene_record_schema": {
                "value_key": "adaptive_l1_residuals",
                "signal": V15D_SIGNAL,
                "threshold_rule": V15D_THRESHOLD_RULE,
            },
            "v16d_freeze": {
                "status": V16D_SCHEMA_GAP_STATUS,
                "freeze_blockers": list(V16D_FREEZE_BLOCKERS),
                "diagnostics_may_be_recorded_but_not_thresholded": [
                    "continuity_maximum_relative_residual",
                "coverage_maximum_containment_lhs",
                "coverage_moment_covariance_scale_max",
                "coverage_center_only_lhs",
                ],
            },
            "required_evidence": {
                "native_execution": [
                    "routing_features_sha256",
                    "routing_z_depths_sha256",
                    "native_dense_gaussian_regressor_executed",
                    "selected_replicate_gaussian_head_executed",
                    "selected_native_rgb_adapter_executed",
                ],
                "packet": [
                    "selected_descriptor_count",
                    "selection_mask_sha256",
                    "source_rgb_sha256",
                    "source_trace_sha256",
                    "native_execution_sha256",
                    "initial_native_attribute_binding_sha256",
                    "final_selected_adapter_source_trace_sha256",
                    "final_selected_adapter_binding_sha256",
                    "materialized_source_trace_sha256",
                    "materialized_current_binding_sha256",
                ],
                "materialization": [
                    "plan_selection_mask_sha256",
                    "plan_tile_trace_sha256",
                    "l1_anchor_semantics",
                    "preflight_tile_trace_sha256",
                    "preflight_events_sha256",
                    "route_tile_trace_sha256",
                    "route_events_sha256",
                    "selected_output_mask_sha256",
                    "additional_full_mask_sha256",
                    "raw_head_request_mask_sha256",
                    "full_passthrough_mask_sha256",
                    "materialized_coverage_certificate_sha256",
                    "materialized_full_attribute_binding_sha256",
                    "materialized_full_passthrough_count",
                    "materialized_update_anchor_count",
                    "route_counts",
                ],
            },
        },
    }
    return {**record, "sha256": canonical_sha256(record)}


def _read_record(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{kind} record is unavailable or invalid") from error
    if not isinstance(record, dict):
        raise ValueError(f"{kind} record must be an object")
    recorded_sha256 = record.pop("sha256", None)
    if recorded_sha256 != canonical_sha256(record):
        raise ValueError(f"{kind} record SHA256 is invalid")
    if record.get("kind") != kind or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{kind} record identity is invalid")
    return {**record, "sha256": recorded_sha256}


def _validate_live_application(
    application: Any, *, root: Path
) -> dict[str, Any]:
    if (
        not isinstance(application, Mapping)
        or set(application)
        != {"model", "dataset", "backend_identity", "collector_source"}
        or application.get("model") != "depthsplat"
        or application.get("dataset") != "dl3dv"
        or not isinstance(application.get("backend_identity"), Mapping)
    ):
        raise ValueError("DepthSplat calibration application is invalid")
    from saes.depthsplat_backend import (
        resolve_depthsplat_backend_contract,
        validate_frozen_depthsplat_backend_identity,
    )

    contract = resolve_depthsplat_backend_contract(Path(root))
    identity = validate_frozen_depthsplat_backend_identity(
        contract, application["backend_identity"]
    )
    expected = _application(identity, root=Path(root))
    if application.get("collector_source") != expected["collector_source"]:
        raise ValueError("DepthSplat calibration collector source changed")
    return expected


def _validate_loaded_record(
    record: Mapping[str, Any],
    *,
    kind: str,
    root: Path,
    plan_path: Path,
    materialization_root: Path,
) -> dict[str, Any]:
    if kind != V15D_KIND:
        raise ValueError("only V15D has a frozen threshold contract")
    required = {
        "schema_version",
        "kind",
        "status",
        "paper_result_eligible",
        "application",
        "acid_binding",
        "mechanism",
        "signal",
        "split_policy",
        "access",
        "train_scene_records",
        "holdout_scene_records",
        "train_summary",
        "holdout_verification",
        "threshold",
        "sha256",
    }
    if set(record) != required:
        raise ValueError(f"{kind} record has unexpected fields")
    if (
        record.get("status") != FROZEN_STATUS
        or record.get("paper_result_eligible") is not False
        or record.get("access") != _access_record()
        or record.get("mechanism") != _mechanism_record()
    ):
        raise ValueError(f"{kind} record is not a valid frozen target-free calibration")
    expected_application = _validate_live_application(record.get("application"), root=root)
    if record.get("application") != expected_application:
        raise ValueError(f"{kind} DepthSplat application identity changed")
    live_binding = resolve_depthsplat_acid_binding(
        plan_path=plan_path, materialization_root=materialization_root
    )
    if record.get("acid_binding") != live_binding:
        raise ValueError(f"{kind} ACID 24/8 binding changed")
    if record.get("split_policy") != {
        "train_split": TRAIN_SPLIT,
        "holdout_split": HOLDOUT_SPLIT,
        "threshold_source": "train_only",
        "holdout_threshold_update_allowed": False,
    }:
        raise ValueError(f"{kind} split policy changed")
    value_key = "adaptive_l1_residuals"
    expected_signal = V15D_SIGNAL
    expected_rule = V15D_THRESHOLD_RULE
    if record.get("signal") != {
        "name": expected_signal,
        "source": "target-free-depthsplat-probe-plan-tile-trace",
        "smaller_is_safer": True,
    }:
        raise ValueError(f"{kind} signal contract changed")
    train = _normalize_scene_records(
        record.get("train_scene_records"),
        binding=live_binding,
        split=TRAIN_SPLIT,
        value_key=value_key,
    )
    holdout = _normalize_scene_records(
        record.get("holdout_scene_records"),
        binding=live_binding,
        split=HOLDOUT_SPLIT,
        value_key=value_key,
    )
    train_values = [value for scene in train for value in scene[value_key]]
    expected_threshold = _finite_quantile(train_values, 0.50)
    threshold = record.get("threshold")
    expected_threshold_record: dict[str, Any] = {
        "value": expected_threshold,
        "promote_when": f"{value_key}_gt_value",
        "rule": expected_rule,
    }
    if threshold != expected_threshold_record:
        raise ValueError(f"{kind} threshold is not reproducible from ACID train")
    if record.get("train_summary") != _summary(train_values):
        raise ValueError(f"{kind} train summary changed")
    if record.get("holdout_verification") != _split_verification(
        holdout, value_key=value_key, threshold=expected_threshold
    ):
        raise ValueError(f"{kind} holdout verification changed")
    return dict(record)


def load_frozen_v15d_threshold(
    path: Path,
    *,
    root: Path = ROOT,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Load V15D only when live DepthSplat and ACID identities still match."""

    record = _read_record(Path(path), kind=V15D_KIND)
    return _validate_loaded_record(
        record,
        kind=V15D_KIND,
        root=Path(root),
        plan_path=Path(plan_path),
        materialization_root=Path(materialization_root),
    )


def load_v16d_schema_gap_record(
    path: Path,
    *,
    v15d_path: Path,
    root: Path = ROOT,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Load the V16D blocker record; it never exposes a usable threshold."""

    v15d = load_frozen_v15d_threshold(
        Path(v15d_path),
        root=Path(root),
        plan_path=Path(plan_path),
        materialization_root=Path(materialization_root),
    )
    record = _read_record(Path(path), kind=V16D_KIND)
    required = {
        "schema_version",
        "kind",
        "status",
        "paper_result_eligible",
        "application",
        "acid_binding",
        "mechanism",
        "access",
        "base_v15d_sha256",
        "freeze_blockers",
        "diagnostic_only",
        "required_future_evidence",
        "sha256",
    }
    if set(record) != required:
        raise ValueError("V16D schema-gap record has unexpected fields")
    if (
        record.get("status") != V16D_SCHEMA_GAP_STATUS
        or record.get("paper_result_eligible") is not False
        or record.get("mechanism") != _mechanism_record()
        or record.get("access") != _access_record()
        or record.get("freeze_blockers") != list(V16D_FREEZE_BLOCKERS)
        or record.get("diagnostic_only")
        != {
            "continuity_maximum_relative_residual": "not-a-calibration-risk",
            "coverage_maximum_containment_lhs": "pre-expansion-not-comparable",
            "coverage_moment_covariance_scale_max": "record-only-until-risk-contract",
            "coverage_center_only_lhs": "not-a-two-sigma-support-certificate",
        }
        or record.get("required_future_evidence")
        != {
            "native_full_slot_and_attribute_provenance": [
                "native_execution_sha256",
                "native_dense_attribute_full_slot_provenance_sha256",
                "initial_adapter_attribute_binding_sha256",
                "final_adapter_attribute_binding_sha256",
                "materialized_attribute_binding_sha256",
            ],
            "coverage_certificate": {
                "geometry_certificate_schema_sha256": "must_bind-z-depth-and-2sigma-shape-semantics",
                "two_sigma_virtual_support_lhs_per_anchor": "must_cover-every-virtual-covariance-shape",
                "two_sigma_virtual_support_lhs_per_tile": "must_be_lte_4",
                "moment_covariance_scale_per_anchor": "must_be_bound_to_final_update",
                "moment_covariance_scale_per_tile": "must_be_bound_to_final_update",
            },
            "accepted_update_slots": {
                "accepted_l0_l1_update_slots_sha256": "must_equal-all-and-only-accepted-compact-updates",
                "accepted_l0_l1_update_slot_count": "must_equal-bound-slot-set-count",
                "final_materialized_update_slots_sha256": "must_equal-accepted-slot-set",
            },
            "selected_anchor_risk": "target-free-native-attribute-loo-contract",
        }
    ):
        raise ValueError("V16D schema-gap record changed")
    expected_application = _validate_live_application(record.get("application"), root=Path(root))
    if record.get("application") != expected_application:
        raise ValueError("V16D DepthSplat application identity changed")
    live_binding = resolve_depthsplat_acid_binding(
        plan_path=Path(plan_path), materialization_root=Path(materialization_root)
    )
    if record.get("acid_binding") != live_binding:
        raise ValueError("V16D ACID 24/8 binding changed")
    if record.get("base_v15d_sha256") != v15d["sha256"]:
        raise ValueError("V16D record does not bind the supplied V15D record")
    if record["application"] != v15d["application"]:
        raise ValueError("V15D/V16D DepthSplat applications differ")
    if record["acid_binding"] != v15d["acid_binding"]:
        raise ValueError("V15D/V16D ACID bindings differ")
    return dict(record)


__all__ = [
    "COLLECTION_PLAN_KIND",
    "COLLECTOR_SOURCE_CONTRACT",
    "DEFAULT_MATERIALIZATION_ROOT",
    "DEFAULT_PLAN_PATH",
    "FROZEN_STATUS",
    "HOLDOUT_SPLIT",
    "MECHANISM_ID",
    "NOT_RUN_STATUS",
    "TRAIN_SPLIT",
    "V15D_KIND",
    "V15D_L1_ANCHOR_SEMANTICS",
    "V15D_SIGNAL",
    "V15D_THRESHOLD_RULE",
    "V16D_KIND",
    "V16D_FREEZE_BLOCKERS",
    "V16D_SCHEMA_GAP_STATUS",
    "build_collection_plan",
    "build_v15d_scene_record",
    "build_v15d_record",
    "build_v16d_record",
    "canonical_sha256",
    "load_frozen_v15d_threshold",
    "load_v16d_schema_gap_record",
    "resolve_depthsplat_acid_binding",
]
