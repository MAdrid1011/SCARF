"""Frozen ACID-disjoint V15/V16 calibration records for DL3DV application."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from saes.incremental_selected_output_execution import RAW_HEAD_EXECUTION_CONTRACT


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPLIT = "calibration_train"
HOLDOUT_SPLIT = "calibration_holdout"
SPLITS = (TRAIN_SPLIT, HOLDOUT_SPLIT)
V15_KIND = "saes-adaptive-l1-acid-disjoint-v15"
V16_KIND = "saes-adaptive-l1-acid-disjoint-v16"
SCHEMA_VERSION = "1.0"
STATUS = "FROZEN_EVALUATION_DISJOINT_TARGET_FREE"
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


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _finite_quantile(values: Sequence[float], fraction: float) -> float:
    if not values or not isinstance(fraction, float) or not 0.0 <= fraction <= 1.0:
        raise ValueError("calibration quantile requires nonempty finite values")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise ValueError("calibration values must be finite and nonnegative")
    return ordered[round((len(ordered) - 1) * fraction)]


def _summary(values: Sequence[float]) -> dict[str, float | int]:
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


def resolve_acid_binding(
    *,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Resolve the live, validated ACID 24/8 context-only input identity."""
    from data.plan_acid_joint_calibration import validate_materialization

    plan_path = Path(plan_path).resolve()
    materialization_root = Path(materialization_root).resolve()
    plan = _read_json(plan_path, "ACID calibration plan")
    materialization = validate_materialization(materialization_root, plan=plan)
    partition = plan.get("partition")
    if not isinstance(partition, Mapping):
        raise ValueError("ACID calibration plan has no partition")
    materialization_record = materialization_root / "materialization.json"
    if not materialization_record.is_file():
        raise ValueError("ACID calibration materialization record is missing")
    splits: dict[str, Any] = {}
    for split, expected_count in ((TRAIN_SPLIT, 24), (HOLDOUT_SPLIT, 8)):
        partition_record = partition.get(split)
        sidecar = materialization.get("sidecars", {}).get(split)
        if not isinstance(partition_record, Mapping) or not isinstance(sidecar, Mapping):
            raise ValueError(f"ACID calibration binding has no {split} identity")
        scenes = partition_record.get("scenes")
        if (
            not isinstance(scenes, list)
            or len(scenes) != expected_count
            or any(not isinstance(scene, str) or not scene for scene in scenes)
            or len(set(scenes)) != len(scenes)
            or partition_record.get("scene_count") != expected_count
            or sidecar.get("scene_count") != expected_count
        ):
            raise ValueError(f"ACID calibration {split} identity is invalid")
        splits[split] = {
            "scenes": list(scenes),
            "scene_count": expected_count,
            "scene_set_sha256": _require_sha256(
                partition_record.get("scene_set_sha256"), f"ACID {split} scene set"
            ),
            "selection_sha256": _require_sha256(
                sidecar.get("selection_sha256"), f"ACID {split} selection"
            ),
            "sidecar_tree_sha256": _require_sha256(
                sidecar.get("tree_sha256"), f"ACID {split} sidecar tree"
            ),
            "sidecar_manifest_sha256": _require_sha256(
                sidecar.get("manifest_sha256"), f"ACID {split} sidecar manifest"
            ),
            "input_provenance_sha256": _require_sha256(
                sidecar.get("input_provenance_sha256"),
                f"ACID {split} input provenance",
            ),
        }
    if set(splits[TRAIN_SPLIT]["scenes"]) & set(splits[HOLDOUT_SPLIT]["scenes"]):
        raise ValueError("ACID calibration train and holdout scenes overlap")
    if plan.get("dataset") != "acid" or plan.get("paper_result_eligible") is not False:
        raise ValueError("ACID calibration plan does not have the required scope")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "saes-acid-context-only-calibration-binding",
        "dataset": "acid",
        "protocol_id": plan.get("protocol_id"),
        "plan_sha256": _require_sha256(plan.get("plan_sha256"), "ACID plan"),
        "plan_file_sha256": sha256_file(plan_path),
        "materialization_record_sha256": sha256_file(materialization_record),
        "materialization_tree_sha256": _require_sha256(
            materialization.get("tree_sha256"), "ACID materialization tree"
        ),
        "materialization_manifest_sha256": _require_sha256(
            materialization.get("manifest_sha256"), "ACID materialization manifest"
        ),
        "splits": splits,
        "access": _access_record(),
    }


def _validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
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
        raise ValueError("ACID calibration binding has an invalid schema")
    if (
        binding.get("schema_version") != SCHEMA_VERSION
        or binding.get("kind") != "saes-acid-context-only-calibration-binding"
        or binding.get("dataset") != "acid"
        or not isinstance(binding.get("protocol_id"), str)
        or not binding["protocol_id"]
    ):
        raise ValueError("ACID calibration binding identity changed")
    for field in (
        "plan_sha256",
        "plan_file_sha256",
        "materialization_record_sha256",
        "materialization_tree_sha256",
        "materialization_manifest_sha256",
    ):
        _require_sha256(binding.get(field), f"ACID calibration {field}")
    if binding.get("access") != _access_record():
        raise ValueError("ACID calibration binding is not context-only")
    raw_splits = binding.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(SPLITS):
        raise ValueError("ACID calibration binding splits are invalid")
    normalized_splits: dict[str, Any] = {}
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
            raise ValueError(f"ACID calibration {split} binding is invalid")
        scenes = value.get("scenes")
        if (
            not isinstance(scenes, list)
            or len(scenes) != expected_count
            or value.get("scene_count") != expected_count
            or any(not isinstance(scene, str) or not scene for scene in scenes)
            or len(set(scenes)) != len(scenes)
        ):
            raise ValueError(f"ACID calibration {split} scenes are invalid")
        if canonical_sha256(sorted(scenes)) != value.get("scene_set_sha256"):
            raise ValueError(f"ACID calibration {split} scene digest changed")
        for field in required_split - {"scenes", "scene_count"}:
            _require_sha256(value.get(field), f"ACID calibration {split} {field}")
        normalized_splits[split] = dict(value)
    if set(normalized_splits[TRAIN_SPLIT]["scenes"]) & set(
        normalized_splits[HOLDOUT_SPLIT]["scenes"]
    ):
        raise ValueError("ACID calibration binding train and holdout overlap")
    return dict(binding)


def _application(checkpoint_sha256: str) -> dict[str, Any]:
    return {
        "model": "transplat",
        "dataset": "dl3dv",
        "checkpoint_sha256": _require_sha256(
            checkpoint_sha256, "DL3DV application checkpoint"
        ),
    }


def _normalize_scene_records(
    records: Sequence[Mapping[str, Any]],
    *,
    binding: Mapping[str, Any],
    split: str,
    value_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"{split} calibration records are invalid")
    expected_scenes = list(binding["splits"][split]["scenes"])
    if len(records) != len(expected_scenes):
        raise ValueError(f"{split} calibration record count changed")
    normalized: list[dict[str, Any]] = []
    for position, (scene, record) in enumerate(zip(expected_scenes, records)):
        if not isinstance(record, Mapping) or set(record) != {"scene", value_key, "access"}:
            raise ValueError(f"{split} calibration scene record has unexpected fields")
        if record.get("scene") != scene:
            raise ValueError(f"{split} calibration scene order changed at {position}")
        raw_values = record.get(value_key)
        if (
            not isinstance(raw_values, Sequence)
            or isinstance(raw_values, (str, bytes))
            or not raw_values
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_values
            )
        ):
            raise ValueError(f"{split} calibration values are invalid")
        values = [float(value) for value in raw_values]
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"{split} calibration values must be finite and nonnegative")
        if record.get("access") != _access_record():
            raise ValueError(f"{split} calibration crossed the context-only boundary")
        normalized.append(
            {
                "scene": scene,
                value_key: values,
                "access": _access_record(),
            }
        )
    return normalized


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


def _base_record(
    *,
    kind: str,
    binding: Mapping[str, Any],
    application_checkpoint_sha256: str,
    train_records: Sequence[Mapping[str, Any]],
    holdout_records: Sequence[Mapping[str, Any]],
    value_key: str,
    threshold: float,
    threshold_rule: str,
) -> dict[str, Any]:
    validated_binding = _validate_binding(binding)
    train = _normalize_scene_records(
        train_records, binding=validated_binding, split=TRAIN_SPLIT, value_key=value_key
    )
    holdout = _normalize_scene_records(
        holdout_records,
        binding=validated_binding,
        split=HOLDOUT_SPLIT,
        value_key=value_key,
    )
    train_values = [value for record in train for value in record[value_key]]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "status": STATUS,
        "paper_result_eligible": False,
        "application": _application(application_checkpoint_sha256),
        "acid_binding": validated_binding,
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


def build_v15_record(
    *,
    binding: Mapping[str, Any],
    application_checkpoint_sha256: str,
    train_scene_records: Sequence[Mapping[str, Any]],
    holdout_scene_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze V15's median residual threshold from ACID train scenes only."""
    validated_binding = _validate_binding(binding)
    train = _normalize_scene_records(
        train_scene_records,
        binding=validated_binding,
        split=TRAIN_SPLIT,
        value_key="residuals",
    )
    threshold = _finite_quantile(
        [value for record in train for value in record["residuals"]], 0.50
    )
    record = _base_record(
        kind=V15_KIND,
        binding=validated_binding,
        application_checkpoint_sha256=application_checkpoint_sha256,
        train_records=train,
        holdout_records=holdout_scene_records,
        value_key="residuals",
        threshold=threshold,
        threshold_rule="train-global-p50",
    )
    return {**record, "sha256": canonical_sha256(record)}


def build_v16_record(
    *,
    binding: Mapping[str, Any],
    application_checkpoint_sha256: str,
    v15_record_sha256: str,
    train_scene_records: Sequence[Mapping[str, Any]],
    holdout_scene_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze V16's minimum-per-scene q25 risk from ACID train scenes only."""
    validated_binding = _validate_binding(binding)
    train = _normalize_scene_records(
        train_scene_records,
        binding=validated_binding,
        split=TRAIN_SPLIT,
        value_key="risks",
    )
    per_scene_q25 = [
        _finite_quantile(record["risks"], 0.25)
        for record in train
    ]
    threshold = min(per_scene_q25)
    record = _base_record(
        kind=V16_KIND,
        binding=validated_binding,
        application_checkpoint_sha256=application_checkpoint_sha256,
        train_records=train,
        holdout_records=holdout_scene_records,
        value_key="risks",
        threshold=threshold,
        threshold_rule="train-minimum-per-scene-q25",
    )
    record["base_v15_sha256"] = _require_sha256(v15_record_sha256, "V15 record")
    record["raw_head_execution_contract"] = RAW_HEAD_EXECUTION_CONTRACT
    record["threshold"]["per_scene_q25"] = per_scene_q25
    return {**record, "sha256": canonical_sha256(record)}


def _load_record(path: Path, *, kind: str) -> dict[str, Any]:
    record = _read_json(path, f"{kind} record")
    recorded_sha256 = record.pop("sha256", None)
    if recorded_sha256 != canonical_sha256(record):
        raise ValueError(f"{kind} record SHA256 is invalid")
    if record.get("kind") != kind or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{kind} record identity is invalid")
    return {**record, "sha256": recorded_sha256}


def _validate_loaded_record(
    record: Mapping[str, Any],
    *,
    kind: str,
    checkpoint_path: Path,
    plan_path: Path,
    materialization_root: Path,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "status",
        "paper_result_eligible",
        "application",
        "acid_binding",
        "split_policy",
        "access",
        "train_scene_records",
        "holdout_scene_records",
        "train_summary",
        "holdout_verification",
        "threshold",
        "sha256",
    }
    if kind == V16_KIND:
        required |= {"base_v15_sha256", "raw_head_execution_contract"}
    if set(record) != required:
        raise ValueError(f"{kind} record has unexpected fields")
    if record.get("status") != STATUS or record.get("paper_result_eligible") is not False:
        raise ValueError(f"{kind} record is not a frozen disjoint calibration")
    if record.get("access") != _access_record():
        raise ValueError(f"{kind} record is not target-free")
    if (
        kind == V16_KIND
        and record.get("raw_head_execution_contract") != RAW_HEAD_EXECUTION_CONTRACT
    ):
        raise ValueError("V16 record raw-head execution contract changed")
    expected_application = _application(sha256_file(Path(checkpoint_path)))
    if record.get("application") != expected_application:
        raise ValueError(f"{kind} record application checkpoint changed")
    live_binding = resolve_acid_binding(
        plan_path=plan_path, materialization_root=materialization_root
    )
    if record.get("acid_binding") != live_binding:
        raise ValueError(f"{kind} record ACID binding changed")
    if record.get("split_policy") != {
        "train_split": TRAIN_SPLIT,
        "holdout_split": HOLDOUT_SPLIT,
        "threshold_source": "train_only",
        "holdout_threshold_update_allowed": False,
    }:
        raise ValueError(f"{kind} record split policy changed")
    value_key = "residuals" if kind == V15_KIND else "risks"
    train = _normalize_scene_records(
        record["train_scene_records"],
        binding=live_binding,
        split=TRAIN_SPLIT,
        value_key=value_key,
    )
    holdout = _normalize_scene_records(
        record["holdout_scene_records"],
        binding=live_binding,
        split=HOLDOUT_SPLIT,
        value_key=value_key,
    )
    values = [value for scene in train for value in scene[value_key]]
    threshold = record.get("threshold")
    if not isinstance(threshold, Mapping):
        raise ValueError(f"{kind} record threshold is invalid")
    expected_threshold_fields = {"value", "promote_when", "rule"}
    if kind == V16_KIND:
        expected_threshold_fields.add("per_scene_q25")
    if set(threshold) != expected_threshold_fields:
        raise ValueError(f"{kind} record threshold has unexpected fields")
    expected_rule = (
        "train-global-p50" if kind == V15_KIND else "train-minimum-per-scene-q25"
    )
    expected_threshold = (
        _finite_quantile(values, 0.50)
        if kind == V15_KIND
        else min(_finite_quantile(scene[value_key], 0.25) for scene in train)
    )
    if (
        threshold.get("rule") != expected_rule
        or threshold.get("promote_when") != f"{value_key}_gt_value"
        or threshold.get("value") != expected_threshold
    ):
        raise ValueError(f"{kind} record threshold does not match train-only values")
    if kind == V16_KIND:
        expected_q25 = [_finite_quantile(scene[value_key], 0.25) for scene in train]
        if threshold.get("per_scene_q25") != expected_q25:
            raise ValueError("V16 record per-scene q25 values changed")
    if record.get("train_summary") != _summary(values):
        raise ValueError(f"{kind} record train summary changed")
    if record.get("holdout_verification") != _split_verification(
        holdout, value_key=value_key, threshold=expected_threshold
    ):
        raise ValueError(f"{kind} record holdout verification changed")
    return {**record, "threshold_value": float(expected_threshold)}


def load_frozen_v15_threshold(
    path: Path,
    *,
    checkpoint_path: Path,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Load V15 only when its ACID source and DL3DV application still match."""
    return _validate_loaded_record(
        _load_record(path, kind=V15_KIND),
        kind=V15_KIND,
        checkpoint_path=checkpoint_path,
        plan_path=plan_path,
        materialization_root=materialization_root,
    )


def load_frozen_v16_threshold(
    path: Path,
    *,
    checkpoint_path: Path,
    v15_record_path: Path,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Load V16 only with its exact verified V15 parent and live ACID binding."""
    v15 = load_frozen_v15_threshold(
        v15_record_path,
        checkpoint_path=checkpoint_path,
        plan_path=plan_path,
        materialization_root=materialization_root,
    )
    record = _validate_loaded_record(
        _load_record(path, kind=V16_KIND),
        kind=V16_KIND,
        checkpoint_path=checkpoint_path,
        plan_path=plan_path,
        materialization_root=materialization_root,
    )
    if record.get("base_v15_sha256") != v15["sha256"]:
        raise ValueError("V16 record does not bind its verified V15 parent")
    if record["acid_binding"] != v15["acid_binding"]:
        raise ValueError("V15 and V16 ACID bindings differ")
    return record


__all__ = [
    "DEFAULT_MATERIALIZATION_ROOT",
    "DEFAULT_PLAN_PATH",
    "HOLDOUT_SPLIT",
    "TRAIN_SPLIT",
    "V15_KIND",
    "V16_KIND",
    "build_v15_record",
    "build_v16_record",
    "canonical_sha256",
    "load_frozen_v15_threshold",
    "load_frozen_v16_threshold",
    "resolve_acid_binding",
    "sha256_file",
]
