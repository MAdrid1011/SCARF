#!/usr/bin/env python3
"""Freeze and validate the ACID-only joint-calibration input contract.

This protocol is deliberately separate from the legacy mechanism calibration
grid.  It consumes the already prepared, evaluation-disjoint ACID subset only
on the author side, partitions its fixed 32 scenes into 24 training and eight
holdout scenes, and materializes context-camera-only sidecars.  It is not
paper-result evidence and cannot change ``artifact/mechanism_config.json``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_manifest import build as build_dataset_manifest
from data.verify_prepared_dataset import verify_tree_manifest
from scripts.calibration_contract import CALIBRATION_DOMAIN, canonical_sha256, hash_ranked_scene_names
from scripts.compile_calibration import deterministic_views
from scripts.compile_protocol import canonicalize_index


DATASET = "acid"
MODELS = ("transplat", "mvsplat", "depthsplat")
TRAIN_SPLIT = "calibration_train"
HOLDOUT_SPLIT = "calibration_holdout"
SPLITS = (TRAIN_SPLIT, HOLDOUT_SPLIT)
SPLIT_COUNTS = {TRAIN_SPLIT: 24, HOLDOUT_SPLIT: 8}
SELECTION_COUNT = sum(SPLIT_COUNTS.values())
PROTOCOL_ID = "saes-joint-materialization-calibration-acid-v1"
PARTITION_DOMAIN = "SCARF-AE-acid-joint-calibration-v1"
PLAN_KIND = "acid_joint_calibration_protocol"
PLAN_STATUS = "PLANNED_AUTHOR_SIDE_PREREQUISITE"
CONTEXT_INPUT_KIND = "scarf_acid_joint_context_input_v1"
CONTEXT_RECORD_KIND = "scarf_acid_joint_context_record_v1"
MATERIALIZATION_KIND = "acid_joint_calibration_sidecar_materialization"
ACCESS_AUDIT_KIND = "acid_joint_calibration_access_audit_v1"
CONTEXT_VIEW_RULE = "deterministic first-last context views via scripts.compile_calibration.deterministic_views"
SHA256 = re.compile(r"[0-9a-f]{64}")

_PLAN_SOURCE_FIELDS = frozenset(
    (
        "archive_sha256",
        "full_train_index_sha256",
        "selected_index_sha256",
        "source_provenance_sha256",
        "prepared_manifest_sha256",
        "prepared_tree_sha256",
        "prepared_tree_revalidation_required",
    )
)
_CONTEXT_RECORD_FIELDS = frozenset(
    (
        "schema_version",
        "kind",
        "key",
        "source_view_count",
        "context_indices",
        "context_cameras",
        "context_images",
    )
)
_CONTEXT_INPUT_FIELDS = frozenset(
    (
        "schema_version",
        "kind",
        "status",
        "dataset",
        "split",
        "plan_sha256",
        "scene_count",
        "scene_set_sha256",
        "context_view_rule",
        "selection_sha256",
        "source_binding",
        "opened_file_manifest",
        "opened_file_manifest_sha256",
        "target_rgb_included",
        "target_camera_metadata_included",
        "target_index_included",
        "expected_results_included",
        "teacher_artifact_included",
    )
)
_MATERIALIZATION_FIELDS = frozenset(
    (
        "schema_version",
        "kind",
        "status",
        "plan_sha256",
        "prepared_tree_sha256",
        "prepared_manifest_sha256",
        "sidecars",
        "paper_result_eligible",
        "mechanism_config_write_allowed",
    )
)


class JointCalibrationContractError(ValueError):
    """Raised when ACID joint-calibration provenance or isolation is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointCalibrationContractError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise JointCalibrationContractError(f"{label} must be an object")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise JointCalibrationContractError(f"{label} must be a SHA256 digest")
    return value


def _scenes(value: Any, label: str, *, count: int | None = None) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(scene, str) or not scene for scene in value)
        or len(value) != len(set(value))
        or (count is not None and len(value) != count)
    ):
        suffix = f" with exactly {count} scenes" if count is not None else ""
        raise JointCalibrationContractError(f"{label} must be unique non-empty scenes{suffix}")
    return list(value)


def _indices(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in value
        )
        or len(value) != len(set(value))
    ):
        raise JointCalibrationContractError(f"{label} has invalid context indices")
    return list(value)


def _safe_relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise JointCalibrationContractError(f"{label} path is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise JointCalibrationContractError(f"{label} path is unsafe")
    path = (Path(root).resolve() / Path(*pure.parts)).resolve()
    if Path(root).resolve() not in path.parents and path != Path(root).resolve():
        raise JointCalibrationContractError(f"{label} path escapes its root")
    return path


def _safe_chunk_path(train_root: Path, value: Any) -> Path:
    pure = PurePosixPath(value) if isinstance(value, str) else None
    if pure is None or len(pure.parts) != 1:
        raise JointCalibrationContractError("prepared ACID chunk path must name one training file")
    path = _safe_relative_path(train_root, value, "prepared ACID chunk")
    if not path.is_file():
        raise FileNotFoundError(f"prepared ACID chunk is missing: {value}")
    return path


def _partition_rank(scene: str, split: str) -> tuple[str, str]:
    material = f"{PARTITION_DOMAIN}\0{split}\0{scene}".encode("utf-8")
    return hashlib.sha256(material).hexdigest(), scene


def _partition_record(scenes: list[str]) -> dict[str, Any]:
    return {
        "scene_count": len(scenes),
        "scenes": scenes,
        "scene_set_sha256": canonical_sha256(sorted(scenes)),
        "selection_sha256": canonical_sha256(scenes),
    }


def deterministic_partition(
    selected_scenes: Sequence[str], *, evaluation_scenes: Sequence[str]
) -> dict[str, Any]:
    """Split the fixed ACID selection independently of source-file ordering."""
    selected = _scenes(list(selected_scenes), "prepared ACID selection", count=SELECTION_COUNT)
    evaluation = _scenes(list(evaluation_scenes), "ACID evaluation selection")
    overlap = sorted(set(selected) & set(evaluation))
    if overlap:
        raise JointCalibrationContractError(
            f"ACID calibration/evaluation overlap: {overlap[0]}"
        )
    train = sorted(selected, key=lambda scene: _partition_rank(scene, TRAIN_SPLIT))[
        : SPLIT_COUNTS[TRAIN_SPLIT]
    ]
    train_set = set(train)
    holdout = sorted(
        (scene for scene in selected if scene not in train_set),
        key=lambda scene: _partition_rank(scene, HOLDOUT_SPLIT),
    )[: SPLIT_COUNTS[HOLDOUT_SPLIT]]
    if len(holdout) != SPLIT_COUNTS[HOLDOUT_SPLIT] or train_set & set(holdout):
        raise JointCalibrationContractError("ACID train/holdout partition is invalid")
    return {
        "domain": PARTITION_DOMAIN,
        "selected_scene_count": len(selected),
        "selected_scene_set_sha256": canonical_sha256(sorted(selected)),
        TRAIN_SPLIT: _partition_record(train),
        HOLDOUT_SPLIT: _partition_record(holdout),
        "train_holdout_scene_disjoint": True,
        "evaluation_disjoint": True,
    }


def _load_prepared_selection(prepared_root: Path) -> dict[str, Any]:
    """Bind planning to the immutable 32-scene ACID prepared tree metadata."""
    prepared_root = Path(prepared_root).resolve()
    source_path = prepared_root / ".scarf-calibration-source.json"
    manifest_path = prepared_root / ".scarf-manifest.json"
    train_root = prepared_root / "train"
    source = _read_json(source_path, "ACID calibration source provenance")
    manifest = _read_json(manifest_path, "ACID prepared-tree manifest")
    selected_index = _read_json(train_root / "index.json", "ACID selected training index")
    full_index = _read_json(train_root / "full-index.json", "ACID full training index")

    if (
        source.get("dataset") != DATASET
        or source.get("selection_domain") != CALIBRATION_DOMAIN
        or source.get("selection_count") != SELECTION_COUNT
        or source.get("evaluation_disjoint") is not True
    ):
        raise JointCalibrationContractError("ACID prepared source has the wrong selection contract")
    for field in ("archive_sha256", "full_train_index_sha256", "selected_index_sha256"):
        _sha256(source.get(field), f"ACID source {field}")
    if manifest.get("dataset") != DATASET:
        raise JointCalibrationContractError("ACID prepared-tree manifest has the wrong dataset")
    _sha256(manifest.get("tree_sha256"), "ACID prepared-tree hash")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise JointCalibrationContractError("ACID prepared-tree manifest has no file inventory")

    for relative, path in (
        (".scarf-calibration-source.json", source_path),
        ("train/index.json", train_root / "index.json"),
        ("train/full-index.json", train_root / "full-index.json"),
    ):
        record = files.get(relative)
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != sha256_file(path)
            or record.get("size") != path.stat().st_size
        ):
            raise JointCalibrationContractError(
                f"ACID prepared-tree manifest does not bind {relative}"
            )
    if sha256_file(train_root / "full-index.json") != source["full_train_index_sha256"]:
        raise JointCalibrationContractError("ACID full training index hash does not match provenance")
    if not all(
        isinstance(scene, str)
        and scene
        and isinstance(chunk, str)
        and chunk
        for scene, chunk in full_index.items()
    ):
        raise JointCalibrationContractError("ACID full training index is invalid")
    expected_selected = hash_ranked_scene_names(
        list(full_index), dataset=DATASET, count=SELECTION_COUNT
    )
    selected_scenes = _scenes(
        source.get("selected_scenes"), "ACID source selected scenes", count=SELECTION_COUNT
    )
    if selected_scenes != expected_selected:
        raise JointCalibrationContractError("ACID source selected scenes are not the fixed hash-ranked set")
    if set(selected_index) != set(expected_selected):
        raise JointCalibrationContractError("ACID selected training index does not contain the fixed scene set")
    if canonical_sha256(selected_index) != source["selected_index_sha256"]:
        raise JointCalibrationContractError("ACID selected training index hash does not match provenance")
    for scene in expected_selected:
        if selected_index.get(scene) != full_index.get(scene):
            raise JointCalibrationContractError(
                f"ACID selected chunk does not match full training index: {scene}"
            )
        relative = selected_index[scene]
        chunk_path = _safe_chunk_path(train_root, relative)
        manifest_record = files.get(f"train/{relative}")
        if (
            not isinstance(manifest_record, Mapping)
            or manifest_record.get("sha256") != sha256_file(chunk_path)
            or manifest_record.get("size") != chunk_path.stat().st_size
        ):
            raise JointCalibrationContractError(
                f"ACID prepared-tree manifest does not bind selected chunk: {relative}"
            )
    return {
        "root": prepared_root,
        "source": source,
        "manifest": manifest,
        "source_path": source_path,
        "manifest_path": manifest_path,
        "selected_index": selected_index,
        "selected_scenes": expected_selected,
    }


def load_acid_evaluation_contract(
    evaluation_protocol: Path = ROOT / "artifact" / "evaluation_protocol.json",
    *,
    repository_root: Path = ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the one finalized ACID evaluation identity shared by all models."""
    repository_root = Path(repository_root).resolve()
    protocol_path = Path(evaluation_protocol).resolve()
    protocol = _read_json(protocol_path, "evaluation protocol")
    if protocol.get("status") != "finalized" or not isinstance(protocol.get("pairs"), Mapping):
        raise JointCalibrationContractError("evaluation protocol is not finalized")
    pair_records: dict[str, Mapping[str, Any]] = {}
    for model in MODELS:
        pair = f"{model}/{DATASET}"
        record = protocol["pairs"].get(pair)
        if not isinstance(record, Mapping):
            raise JointCalibrationContractError(f"evaluation protocol has no {pair} record")
        pair_records[pair] = record
    identity = {
        (record.get("index_path"), record.get("source_index_sha256"))
        for record in pair_records.values()
    }
    if len(identity) != 1:
        raise JointCalibrationContractError("ACID evaluation pairs do not share one index identity")
    relative, source_digest = identity.pop()
    index_path = _safe_relative_path(repository_root, relative, "ACID evaluation index")
    if not index_path.is_file():
        raise FileNotFoundError(f"ACID evaluation index is missing: {relative}")
    _sha256(source_digest, "ACID evaluation source index hash")
    rows, summary = canonicalize_index(index_path, source_digest)
    for pair, record in pair_records.items():
        for field, value in summary.items():
            if record.get(field) != value:
                raise JointCalibrationContractError(
                    f"{pair} evaluation {field} does not match the source index"
                )
    return rows, {
        "index_path": str(index_path.relative_to(repository_root)),
        "source_index_sha256": source_digest,
        "sample_selection_sha256": summary["sample_selection_sha256"],
        "sample_count": summary["sample_count"],
        "source_entry_count": summary["source_entry_count"],
        "null_entry_count": summary["null_entry_count"],
        "model_pairs": [f"{model}/{DATASET}" for model in MODELS],
        "evaluation_scene_set_sha256": canonical_sha256(
            sorted(row["scene"] for row in rows)
        ),
    }


def _sidecar_requirements() -> dict[str, Any]:
    return {
        "required": True,
        "materialization_status": "REQUIRED",
        "input_kind": CONTEXT_INPUT_KIND,
        "record_kind": CONTEXT_RECORD_KIND,
        "context_view_rule": CONTEXT_VIEW_RULE,
        "allowed_record_fields": sorted(_CONTEXT_RECORD_FIELDS),
        "target_rgb_included": False,
        "target_camera_metadata_included": False,
        "target_index_included": False,
        "expected_results_included": False,
        "teacher_artifact_included": False,
        "opened_file_allowlist_required": True,
    }


def _checkpoint_requirements() -> dict[str, Any]:
    return {
        "required_before_optimization": True,
        "status": "UNBOUND",
        "models": list(MODELS),
        "required_fields_per_model": ["checkpoint_path", "checkpoint_sha256"],
        "shared_asset_model_independent": True,
    }


def _teacher_runtime_isolation() -> dict[str, Any]:
    return {
        "offline_teacher_only": True,
        "teacher_storage_scope": "author_training_only",
        "teacher_files_runtime_accessible": False,
        "teacher_sources": ["dense_adaptor_output", "dense_context_render"],
        "teacher_forbidden_inputs": [
            "target_rgb",
            "target_camera_metadata",
            "target_indices",
            "expected_results.json",
            "evaluation_scenes",
        ],
        "descriptor_allowed_inputs": [
            "probe_attributes",
            "context_feature_statistics",
            "context_depth_statistics",
            "assignment_statistics",
            "context_camera_geometry",
        ],
        "descriptor_forbidden_inputs": [
            "model_id",
            "dataset_id",
            "target_rgb",
            "target_camera_metadata",
            "target_indices",
            "skipped_s3_descriptors",
            "teacher_files",
            "expected_results.json",
            "evaluation_scenes",
        ],
        "runtime_teacher_path": None,
    }


def _holdout_policy() -> dict[str, Any]:
    return {
        "optimizer_allowed": False,
        "asset_update_allowed": False,
        "rerank_allowed": False,
        "partition_reshuffle_allowed": False,
        "threshold_or_budget_change_allowed": False,
        "frozen_train_asset_required": True,
    }


def build_plan(
    prepared_root: Path,
    *,
    evaluation_rows: Sequence[Mapping[str, Any]],
    evaluation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a hash-bound ACID 24/8 plan from prevalidated identities."""
    prepared = _load_prepared_selection(prepared_root)
    evaluation_scenes = _scenes(
        [row.get("scene") for row in evaluation_rows], "ACID evaluation selection"
    )
    partition = deterministic_partition(
        prepared["selected_scenes"], evaluation_scenes=evaluation_scenes
    )
    required_evaluation = {
        "index_path",
        "source_index_sha256",
        "sample_selection_sha256",
        "sample_count",
        "source_entry_count",
        "null_entry_count",
        "model_pairs",
        "evaluation_scene_set_sha256",
    }
    if not isinstance(evaluation_contract, Mapping) or set(evaluation_contract) != required_evaluation:
        raise JointCalibrationContractError("ACID evaluation contract has unexpected fields")
    if evaluation_contract.get("evaluation_scene_set_sha256") != canonical_sha256(
        sorted(evaluation_scenes)
    ):
        raise JointCalibrationContractError("ACID evaluation scene set hash is invalid")
    source = prepared["source"]
    manifest = prepared["manifest"]
    plan: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": PLAN_KIND,
        "status": PLAN_STATUS,
        "protocol_id": PROTOCOL_ID,
        "dataset": DATASET,
        "model_scope": list(MODELS),
        "author_side_prerequisite": True,
        "paper_result_eligible": False,
        "mechanism_config_write_allowed": False,
        "dl3dv_quality_gate_authorized": False,
        "source": {
            "archive_sha256": source["archive_sha256"],
            "full_train_index_sha256": source["full_train_index_sha256"],
            "selected_index_sha256": source["selected_index_sha256"],
            "source_provenance_sha256": sha256_file(prepared["source_path"]),
            "prepared_manifest_sha256": sha256_file(prepared["manifest_path"]),
            "prepared_tree_sha256": manifest["tree_sha256"],
            "prepared_tree_revalidation_required": True,
        },
        "evaluation": dict(evaluation_contract),
        "partition": partition,
        "context_only_sidecars": _sidecar_requirements(),
        "checkpoint_binding": _checkpoint_requirements(),
        "teacher_runtime_isolation": _teacher_runtime_isolation(),
        "holdout_policy": _holdout_policy(),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    validate_plan(plan, evaluation_rows=evaluation_rows)
    return plan


def plan_acid_joint_calibration(
    prepared_root: Path,
    *,
    evaluation_protocol: Path = ROOT / "artifact" / "evaluation_protocol.json",
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Build the only allowed ACID joint-calibration planning record."""
    rows, evaluation = load_acid_evaluation_contract(
        evaluation_protocol, repository_root=repository_root
    )
    return build_plan(
        prepared_root, evaluation_rows=rows, evaluation_contract=evaluation
    )


def _validate_partition(
    partition: Any, *, evaluation_scenes: Sequence[str] | None
) -> dict[str, Any]:
    if not isinstance(partition, Mapping):
        raise JointCalibrationContractError("ACID joint plan has no partition")
    expected_fields = {
        "domain",
        "selected_scene_count",
        "selected_scene_set_sha256",
        TRAIN_SPLIT,
        HOLDOUT_SPLIT,
        "train_holdout_scene_disjoint",
        "evaluation_disjoint",
    }
    if set(partition) != expected_fields or partition.get("domain") != PARTITION_DOMAIN:
        raise JointCalibrationContractError("ACID joint partition has unexpected fields")
    if partition.get("selected_scene_count") != SELECTION_COUNT:
        raise JointCalibrationContractError("ACID joint partition has the wrong selected count")
    split_scenes: dict[str, list[str]] = {}
    for split in SPLITS:
        record = partition.get(split)
        if not isinstance(record, Mapping) or set(record) != {
            "scene_count",
            "scenes",
            "scene_set_sha256",
            "selection_sha256",
        }:
            raise JointCalibrationContractError(f"ACID {split} partition is invalid")
        scenes = _scenes(record.get("scenes"), f"ACID {split} partition", count=SPLIT_COUNTS[split])
        if (
            record.get("scene_count") != SPLIT_COUNTS[split]
            or record.get("scene_set_sha256") != canonical_sha256(sorted(scenes))
            or record.get("selection_sha256") != canonical_sha256(scenes)
        ):
            raise JointCalibrationContractError(f"ACID {split} partition hashes are invalid")
        split_scenes[split] = scenes
    union = [*split_scenes[TRAIN_SPLIT], *split_scenes[HOLDOUT_SPLIT]]
    if (
        len(set(union)) != SELECTION_COUNT
        or partition.get("selected_scene_set_sha256") != canonical_sha256(sorted(union))
        or partition.get("train_holdout_scene_disjoint") is not True
        or partition.get("evaluation_disjoint") is not True
    ):
        raise JointCalibrationContractError("ACID joint partition disjointness is invalid")
    expected = deterministic_partition(
        union, evaluation_scenes=evaluation_scenes or ["__evaluation_not_loaded__"]
    )
    for split in SPLITS:
        if expected[split] != partition[split]:
            raise JointCalibrationContractError("ACID joint partition is not deterministic")
    if evaluation_scenes is not None and set(union) & set(evaluation_scenes):
        raise JointCalibrationContractError("ACID joint partition overlaps evaluation scenes")
    return {split: split_scenes[split] for split in SPLITS}


def validate_plan(
    plan: Mapping[str, Any],
    *,
    prepared_root: Path | None = None,
    evaluation_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a plan without allowing it to become claim evidence."""
    if not isinstance(plan, Mapping):
        raise JointCalibrationContractError("ACID joint plan must be an object")
    required_fields = {
        "schema_version",
        "kind",
        "status",
        "protocol_id",
        "dataset",
        "model_scope",
        "author_side_prerequisite",
        "paper_result_eligible",
        "mechanism_config_write_allowed",
        "dl3dv_quality_gate_authorized",
        "source",
        "evaluation",
        "partition",
        "context_only_sidecars",
        "checkpoint_binding",
        "teacher_runtime_isolation",
        "holdout_policy",
        "plan_sha256",
    }
    if set(plan) != required_fields:
        raise JointCalibrationContractError("ACID joint plan has unexpected fields")
    if (
        plan.get("schema_version") != "1.0"
        or plan.get("kind") != PLAN_KIND
        or plan.get("status") != PLAN_STATUS
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("dataset") != DATASET
        or plan.get("model_scope") != list(MODELS)
        or plan.get("author_side_prerequisite") is not True
        or plan.get("paper_result_eligible") is not False
        or plan.get("mechanism_config_write_allowed") is not False
        or plan.get("dl3dv_quality_gate_authorized") is not False
    ):
        raise JointCalibrationContractError("ACID joint plan violates its author-side scope")
    source = plan.get("source")
    if not isinstance(source, Mapping) or set(source) != _PLAN_SOURCE_FIELDS:
        raise JointCalibrationContractError("ACID joint plan source binding is invalid")
    for field in _PLAN_SOURCE_FIELDS - {"prepared_tree_revalidation_required"}:
        _sha256(source.get(field), f"ACID joint source {field}")
    if source.get("prepared_tree_revalidation_required") is not True:
        raise JointCalibrationContractError("ACID joint plan must revalidate the prepared tree")
    evaluation = plan.get("evaluation")
    evaluation_fields = {
        "index_path",
        "source_index_sha256",
        "sample_selection_sha256",
        "sample_count",
        "source_entry_count",
        "null_entry_count",
        "model_pairs",
        "evaluation_scene_set_sha256",
    }
    if not isinstance(evaluation, Mapping) or set(evaluation) != evaluation_fields:
        raise JointCalibrationContractError("ACID joint plan evaluation binding is invalid")
    for field in (
        "source_index_sha256",
        "sample_selection_sha256",
        "evaluation_scene_set_sha256",
    ):
        _sha256(evaluation.get(field), f"ACID joint evaluation {field}")
    if (
        not isinstance(evaluation.get("index_path"), str)
        or not evaluation["index_path"]
        or evaluation.get("model_pairs") != [f"{model}/{DATASET}" for model in MODELS]
        or any(
            isinstance(evaluation.get(field), bool)
            or not isinstance(evaluation.get(field), int)
            or evaluation[field] < 0
            for field in ("sample_count", "source_entry_count", "null_entry_count")
        )
    ):
        raise JointCalibrationContractError("ACID joint evaluation metadata is invalid")
    evaluation_scenes = None
    if evaluation_rows is not None:
        evaluation_scenes = _scenes(
            [row.get("scene") for row in evaluation_rows], "ACID evaluation selection"
        )
        if evaluation["evaluation_scene_set_sha256"] != canonical_sha256(
            sorted(evaluation_scenes)
        ):
            raise JointCalibrationContractError("ACID joint evaluation scene set hash mismatch")
    split_scenes = _validate_partition(
        plan.get("partition"), evaluation_scenes=evaluation_scenes
    )
    if plan.get("context_only_sidecars") != _sidecar_requirements():
        raise JointCalibrationContractError("ACID joint sidecar requirements were changed")
    if plan.get("checkpoint_binding") != _checkpoint_requirements():
        raise JointCalibrationContractError("ACID joint checkpoint requirements were changed")
    if plan.get("teacher_runtime_isolation") != _teacher_runtime_isolation():
        raise JointCalibrationContractError("ACID teacher/runtime isolation was changed")
    if plan.get("holdout_policy") != _holdout_policy():
        raise JointCalibrationContractError("ACID holdout policy was changed")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != canonical_sha256(payload):
        raise JointCalibrationContractError("ACID joint plan SHA256 is invalid")
    if prepared_root is not None:
        prepared = _load_prepared_selection(prepared_root)
        expected_source = {
            "archive_sha256": prepared["source"]["archive_sha256"],
            "full_train_index_sha256": prepared["source"]["full_train_index_sha256"],
            "selected_index_sha256": prepared["source"]["selected_index_sha256"],
            "source_provenance_sha256": sha256_file(prepared["source_path"]),
            "prepared_manifest_sha256": sha256_file(prepared["manifest_path"]),
            "prepared_tree_sha256": prepared["manifest"]["tree_sha256"],
            "prepared_tree_revalidation_required": True,
        }
        if dict(source) != expected_source:
            raise JointCalibrationContractError("ACID joint plan source does not match prepared tree")
        planned_scenes = [
            *split_scenes[TRAIN_SPLIT],
            *split_scenes[HOLDOUT_SPLIT],
        ]
        if set(planned_scenes) != set(prepared["selected_scenes"]):
            raise JointCalibrationContractError("ACID joint plan does not cover the prepared selection")
    return {
        "plan_sha256": str(plan["plan_sha256"]),
        "train_scene_count": len(split_scenes[TRAIN_SPLIT]),
        "holdout_scene_count": len(split_scenes[HOLDOUT_SPLIT]),
        "evaluation_disjoint": True,
        "paper_result_eligible": False,
    }


def validate_plan_file(
    plan_path: Path,
    *,
    prepared_root: Path,
    evaluation_protocol: Path = ROOT / "artifact" / "evaluation_protocol.json",
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Recheck a saved plan against the live prepared and evaluation identities."""
    rows, evaluation = load_acid_evaluation_contract(
        evaluation_protocol, repository_root=repository_root
    )
    plan = _read_json(plan_path, "ACID joint plan")
    identity = validate_plan(plan, prepared_root=prepared_root, evaluation_rows=rows)
    if dict(plan["evaluation"]) != evaluation:
        raise JointCalibrationContractError("ACID joint plan evaluation identity is stale")
    return identity


def _validate_execution_plan(
    plan: Mapping[str, Any],
    *,
    prepared_root: Path | None = None,
    evaluation_rows: Sequence[Mapping[str, Any]] | None = None,
    evaluation_protocol: Path = ROOT / "artifact" / "evaluation_protocol.json",
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Require a fresh official evaluation check before sidecar execution."""
    rows = evaluation_rows
    if rows is None:
        rows, evaluation = load_acid_evaluation_contract(
            evaluation_protocol, repository_root=repository_root
        )
        if dict(plan.get("evaluation", {})) != evaluation:
            raise JointCalibrationContractError("ACID joint plan evaluation identity is stale")
    return validate_plan(plan, prepared_root=prepared_root, evaluation_rows=rows)


def _load_selected_examples(
    prepared: Mapping[str, Any], scenes: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    """Load source chunks only during author-side sidecar materialization."""
    try:
        import torch
    except ImportError as exc:
        raise JointCalibrationContractError("context-only sidecar materialization requires torch") from exc
    train_root = Path(prepared["root"]) / "train"
    index = prepared["selected_index"]
    by_chunk: dict[Path, set[str]] = {}
    for scene in scenes:
        by_chunk.setdefault(_safe_chunk_path(train_root, index[scene]), set()).add(scene)
    examples: dict[str, Mapping[str, Any]] = {}
    for path, required in sorted(by_chunk.items(), key=lambda item: str(item[0])):
        chunk = torch.load(path, map_location="cpu")
        if not isinstance(chunk, list):
            raise JointCalibrationContractError(f"ACID selected chunk is not a list: {path}")
        for example in chunk:
            if not isinstance(example, Mapping) or example.get("key") not in required:
                continue
            examples[str(example["key"])] = example
    missing = sorted(set(scenes) - set(examples))
    if missing:
        raise JointCalibrationContractError(
            f"ACID selected scene is absent from its chunk: {missing[0]}"
        )
    return examples


def _clone_context_value(value: Any, *, label: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise JointCalibrationContractError("context-only sidecar materialization requires torch") from exc
    if getattr(torch, "is_tensor", lambda _: False)(value):
        return value.detach().cpu().clone()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise JointCalibrationContractError(f"ACID {label} has an unsupported serialized type")


def _context_camera_rows(cameras: Any, *, view_count: int, indices: list[int]) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise JointCalibrationContractError("context-only sidecar materialization requires torch") from exc
    if getattr(torch, "is_tensor", lambda _: False)(cameras):
        if cameras.ndim != 2 or cameras.shape[0] != view_count:
            raise JointCalibrationContractError("ACID source has invalid camera geometry")
        return torch.stack(
            [cameras[index].detach().cpu().clone() for index in indices], dim=0
        )
    if not isinstance(cameras, (list, tuple)) or len(cameras) != view_count:
        raise JointCalibrationContractError("ACID source has invalid camera geometry")
    return [copy.deepcopy(cameras[index]) for index in indices]


def _context_record(example: Mapping[str, Any], scene: str) -> dict[str, Any]:
    if example.get("key") != scene:
        raise JointCalibrationContractError("ACID source example does not match its scene")
    images = example.get("images")
    cameras = example.get("cameras")
    if images is None or not hasattr(images, "__len__"):
        raise JointCalibrationContractError(f"ACID scene has no image sequence: {scene}")
    view_count = len(images)
    selection = deterministic_views(scene, view_count)
    context_indices = _indices(selection["context"], "ACID context")
    context_images = [
        _clone_context_value(images[index], label="context image")
        for index in context_indices
    ]
    return {
        "schema_version": "1.0",
        "kind": CONTEXT_RECORD_KIND,
        "key": scene,
        "source_view_count": view_count,
        "context_indices": context_indices,
        "context_cameras": _context_camera_rows(
            cameras, view_count=view_count, indices=context_indices
        ),
        "context_images": context_images,
    }


def _input_source_binding(plan: Mapping[str, Any], split_record: Mapping[str, Any]) -> dict[str, Any]:
    source = plan["source"]
    return {
        "prepared_tree_sha256": source["prepared_tree_sha256"],
        "prepared_manifest_sha256": source["prepared_manifest_sha256"],
        "prepared_source_provenance_sha256": source["source_provenance_sha256"],
        "partition_scene_set_sha256": split_record["scene_set_sha256"],
    }


def _write_context_only_sidecar(
    output_root: Path,
    *,
    split: str,
    scenes: list[str],
    examples: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise JointCalibrationContractError("context-only sidecar materialization requires torch") from exc
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"ACID context-only sidecar already exists: {output_root}")
    test_root = output_root / "test"
    test_root.mkdir(parents=True, exist_ok=False)
    index: dict[str, str] = {}
    selection_rows: list[dict[str, Any]] = []
    for ordinal, scene in enumerate(scenes):
        record = _context_record(examples[scene], scene)
        name = f"{ordinal:06d}.torch"
        torch.save([record], test_root / name)
        index[scene] = name
        selection_rows.append(
            {
                "scene": scene,
                "source_view_count": record["source_view_count"],
                "context_indices": record["context_indices"],
            }
        )
    index_path = test_root / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    opened_file_manifest = [
        {
            "path": "test/index.json",
            "role": "context_only_sidecar_index",
            "sha256": sha256_file(index_path),
        },
        *[
            {
                "path": f"test/{index[scene]}",
                "role": "context_only_sidecar_record",
                "sha256": sha256_file(test_root / index[scene]),
            }
            for scene in scenes
        ],
    ]
    split_record = plan["partition"][split]
    input_record = {
        "schema_version": "1.0",
        "kind": CONTEXT_INPUT_KIND,
        "status": "PASS",
        "dataset": DATASET,
        "split": split,
        "plan_sha256": plan["plan_sha256"],
        "scene_count": len(scenes),
        "scene_set_sha256": split_record["scene_set_sha256"],
        "context_view_rule": CONTEXT_VIEW_RULE,
        "selection_sha256": canonical_sha256(selection_rows),
        "source_binding": _input_source_binding(plan, split_record),
        "opened_file_manifest": opened_file_manifest,
        "opened_file_manifest_sha256": canonical_sha256(opened_file_manifest),
        "target_rgb_included": False,
        "target_camera_metadata_included": False,
        "target_index_included": False,
        "expected_results_included": False,
        "teacher_artifact_included": False,
    }
    provenance_path = output_root / ".scarf-acid-joint-input.json"
    provenance_path.write_text(
        json.dumps(input_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = build_dataset_manifest(
        output_root,
        f"acid-joint-{split}-context-only",
        "author-side ACID joint-calibration context-only sidecar",
        str(plan["plan_sha256"]),
    )
    manifest_path = output_root / ".scarf-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "sidecar_root": f"inputs/{split}",
        "tree_sha256": manifest["tree_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "input_provenance_sha256": sha256_file(provenance_path),
        "selection_sha256": input_record["selection_sha256"],
        "scene_count": len(scenes),
        "target_rgb_included": False,
        "target_camera_metadata_included": False,
        "target_index_included": False,
        "expected_results_included": False,
        "teacher_artifact_included": False,
    }


def _safe_sidecar_chunk_path(test_root: Path, value: Any) -> Path:
    path = _safe_relative_path(test_root, value, "ACID context-only sidecar")
    if not path.is_file():
        raise FileNotFoundError("ACID context-only sidecar chunk is missing")
    return path


def _reject_forbidden_sidecar_paths(root: Path) -> None:
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().casefold()
        if any(token in relative for token in ("target", "expected_results", "teacher")):
            raise JointCalibrationContractError(
                f"ACID context-only sidecar contains a forbidden artifact path: {relative}"
            )


def _context_camera_count(value: Any) -> int:
    if not hasattr(value, "__len__"):
        raise JointCalibrationContractError("ACID context-only record has no context cameras")
    return len(value)


def validate_context_only_sidecar(
    root: Path,
    *,
    plan: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    """Reject all target and teacher fields from one materialized sidecar."""
    if split not in SPLITS:
        raise JointCalibrationContractError(f"unknown ACID joint split: {split}")
    validate_plan(plan)
    root = Path(root).resolve()
    _reject_forbidden_sidecar_paths(root)
    provenance_path = root / ".scarf-acid-joint-input.json"
    source = _read_json(provenance_path, "ACID context-only sidecar provenance")
    if set(source) != _CONTEXT_INPUT_FIELDS:
        raise JointCalibrationContractError("ACID context-only sidecar provenance has unexpected fields")
    split_record = plan["partition"][split]
    expected_source = {
        "schema_version": "1.0",
        "kind": CONTEXT_INPUT_KIND,
        "status": "PASS",
        "dataset": DATASET,
        "split": split,
        "plan_sha256": plan["plan_sha256"],
        "scene_count": SPLIT_COUNTS[split],
        "scene_set_sha256": split_record["scene_set_sha256"],
        "context_view_rule": CONTEXT_VIEW_RULE,
        "source_binding": _input_source_binding(plan, split_record),
        "target_rgb_included": False,
        "target_camera_metadata_included": False,
        "target_index_included": False,
        "expected_results_included": False,
        "teacher_artifact_included": False,
    }
    for key, value in expected_source.items():
        if source.get(key) != value:
            raise JointCalibrationContractError(
                f"ACID context-only sidecar violates {key} isolation requirement"
            )
    _sha256(source.get("selection_sha256"), "ACID context-only selection")
    opened = source.get("opened_file_manifest")
    if (
        not isinstance(opened, list)
        or len(opened) != SPLIT_COUNTS[split] + 1
        or source.get("opened_file_manifest_sha256") != canonical_sha256(opened)
    ):
        raise JointCalibrationContractError("ACID context-only opened-file manifest is invalid")
    test_root = root / "test"
    index = _read_json(test_root / "index.json", "ACID context-only sidecar index")
    scenes = _scenes(split_record["scenes"], f"ACID {split} partition", count=SPLIT_COUNTS[split])
    if set(index) != set(scenes):
        raise JointCalibrationContractError("ACID context-only sidecar scene set does not match partition")
    if (
        any(not isinstance(value, str) or not value for value in index.values())
        or len(set(index.values())) != len(index)
    ):
        raise JointCalibrationContractError("ACID context-only sidecar index is invalid")
    expected_paths = {"test/index.json"}
    expected_paths.update(f"test/{name}" for name in index.values())
    seen_paths = set()
    for item in opened:
        if not isinstance(item, Mapping):
            raise JointCalibrationContractError("ACID context-only opened-file record is invalid")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or relative not in expected_paths
            or "target" in relative.casefold()
            or "expected_results" in relative.casefold()
            or "teacher" in relative.casefold()
            or item.get("role") not in {"context_only_sidecar_index", "context_only_sidecar_record"}
            or not isinstance(item.get("sha256"), str)
            or SHA256.fullmatch(item["sha256"]) is None
        ):
            raise JointCalibrationContractError("ACID context-only opened-file record is unsafe")
        path = root / relative
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise JointCalibrationContractError("ACID context-only opened-file hash mismatch")
        seen_paths.add(relative)
    if seen_paths != expected_paths:
        raise JointCalibrationContractError("ACID context-only opened-file allowlist is incomplete")

    try:
        import torch
    except ImportError as exc:
        raise JointCalibrationContractError("context-only sidecar validation requires torch") from exc
    selection_rows = []
    for scene in scenes:
        chunk_path = _safe_sidecar_chunk_path(test_root, index.get(scene))
        chunk = torch.load(chunk_path, map_location="cpu")
        if not isinstance(chunk, list) or len(chunk) != 1 or not isinstance(chunk[0], Mapping):
            raise JointCalibrationContractError("ACID context-only sidecar chunk is invalid")
        record = chunk[0]
        if (
            set(record) != _CONTEXT_RECORD_FIELDS
            or record.get("schema_version") != "1.0"
            or record.get("kind") != CONTEXT_RECORD_KIND
            or record.get("key") != scene
        ):
            raise JointCalibrationContractError("ACID context-only record has forbidden fields")
        view_count = record.get("source_view_count")
        if isinstance(view_count, bool) or not isinstance(view_count, int) or view_count < 5:
            raise JointCalibrationContractError("ACID context-only record has invalid source view count")
        context_indices = _indices(record.get("context_indices"), "ACID context-only record")
        if context_indices != deterministic_views(scene, view_count)["context"]:
            raise JointCalibrationContractError("ACID context-only record has non-deterministic context views")
        cameras = record.get("context_cameras")
        images = record.get("context_images")
        if (
            _context_camera_count(cameras) != len(context_indices)
            or not isinstance(images, list)
            or len(images) != len(context_indices)
        ):
            raise JointCalibrationContractError("ACID context-only record has invalid context payload")
        for image in images:
            _clone_context_value(image, label="context image")
        selection_rows.append(
            {
                "scene": scene,
                "source_view_count": view_count,
                "context_indices": context_indices,
            }
        )
    if source.get("selection_sha256") != canonical_sha256(selection_rows):
        raise JointCalibrationContractError("ACID context-only sidecar selection hash mismatch")
    tree = verify_tree_manifest(root, root / ".scarf-manifest.json")
    return {
        "split": split,
        "scene_count": len(scenes),
        "tree_sha256": tree["tree_sha256"],
        "manifest_sha256": tree["manifest_sha256"],
        "input_provenance_sha256": sha256_file(provenance_path),
        "selection_sha256": source["selection_sha256"],
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "teacher_artifact_accessed": False,
        "expected_results_accessed": False,
    }


def materialize_context_only_sidecars(
    prepared_root: Path,
    *,
    plan: Mapping[str, Any],
    output_root: Path,
    evaluation_rows: Sequence[Mapping[str, Any]] | None = None,
    evaluation_protocol: Path = ROOT / "artifact" / "evaluation_protocol.json",
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Materialize both ACID split sidecars after rehashing the prepared tree."""
    _validate_execution_plan(
        plan,
        prepared_root=prepared_root,
        evaluation_rows=evaluation_rows,
        evaluation_protocol=evaluation_protocol,
        repository_root=repository_root,
    )
    prepared = _load_prepared_selection(prepared_root)
    prepared_tree = verify_tree_manifest(
        prepared["root"], prepared["root"] / ".scarf-manifest.json"
    )
    if (
        prepared_tree["tree_sha256"] != plan["source"]["prepared_tree_sha256"]
        or prepared_tree["manifest_sha256"] != plan["source"]["prepared_manifest_sha256"]
    ):
        raise JointCalibrationContractError("ACID prepared tree does not match the frozen plan")
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"ACID joint sidecar output already exists: {output_root}")
    all_scenes = [
        *plan["partition"][TRAIN_SPLIT]["scenes"],
        *plan["partition"][HOLDOUT_SPLIT]["scenes"],
    ]
    examples = _load_selected_examples(prepared, all_scenes)
    output_root.mkdir(parents=True, exist_ok=False)
    sidecars: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        scenes = list(plan["partition"][split]["scenes"])
        sidecars[split] = _write_context_only_sidecar(
            output_root / "inputs" / split,
            split=split,
            scenes=scenes,
            examples=examples,
            plan=plan,
        )
    materialization = {
        "schema_version": "1.0",
        "kind": MATERIALIZATION_KIND,
        "status": "PASS",
        "plan_sha256": plan["plan_sha256"],
        "prepared_tree_sha256": prepared_tree["tree_sha256"],
        "prepared_manifest_sha256": prepared_tree["manifest_sha256"],
        "sidecars": sidecars,
        "paper_result_eligible": False,
        "mechanism_config_write_allowed": False,
    }
    materialization_path = output_root / "materialization.json"
    materialization_path.write_text(
        json.dumps(materialization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = build_dataset_manifest(
        output_root,
        "acid-joint-calibration-context-only-sidecars",
        "author-side ACID joint-calibration sidecars",
        str(plan["plan_sha256"]),
    )
    manifest_path = output_root / ".scarf-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **materialization,
        "tree_sha256": manifest["tree_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
    }


def validate_materialization(
    root: Path,
    *,
    plan: Mapping[str, Any],
    evaluation_rows: Sequence[Mapping[str, Any]] | None = None,
    evaluation_protocol: Path = ROOT / "artifact" / "evaluation_protocol.json",
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Validate both sidecars and keep their teacher/runtime boundary closed."""
    _validate_execution_plan(
        plan,
        evaluation_rows=evaluation_rows,
        evaluation_protocol=evaluation_protocol,
        repository_root=repository_root,
    )
    root = Path(root).resolve()
    record = _read_json(root / "materialization.json", "ACID joint materialization")
    if set(record) != _MATERIALIZATION_FIELDS:
        raise JointCalibrationContractError("ACID joint materialization has unexpected fields")
    if (
        record.get("schema_version") != "1.0"
        or record.get("kind") != MATERIALIZATION_KIND
        or record.get("status") != "PASS"
        or record.get("plan_sha256") != plan["plan_sha256"]
        or record.get("prepared_tree_sha256") != plan["source"]["prepared_tree_sha256"]
        or record.get("prepared_manifest_sha256") != plan["source"]["prepared_manifest_sha256"]
        or record.get("paper_result_eligible") is not False
        or record.get("mechanism_config_write_allowed") is not False
        or not isinstance(record.get("sidecars"), Mapping)
        or set(record["sidecars"]) != set(SPLITS)
    ):
        raise JointCalibrationContractError("ACID joint materialization violates its scope")
    identities = {}
    for split in SPLITS:
        identity = validate_context_only_sidecar(
            root / "inputs" / split, plan=plan, split=split
        )
        declared = record["sidecars"].get(split)
        required = {
            "sidecar_root": f"inputs/{split}",
            "tree_sha256": identity["tree_sha256"],
            "manifest_sha256": identity["manifest_sha256"],
            "input_provenance_sha256": identity["input_provenance_sha256"],
            "selection_sha256": identity["selection_sha256"],
            "scene_count": SPLIT_COUNTS[split],
            "target_rgb_included": False,
            "target_camera_metadata_included": False,
            "target_index_included": False,
            "expected_results_included": False,
            "teacher_artifact_included": False,
        }
        if declared != required:
            raise JointCalibrationContractError("ACID joint materialization sidecar identity mismatch")
        identities[split] = identity
    tree = verify_tree_manifest(root, root / ".scarf-manifest.json")
    return {
        "plan_sha256": plan["plan_sha256"],
        "tree_sha256": tree["tree_sha256"],
        "manifest_sha256": tree["manifest_sha256"],
        "sidecars": identities,
        "paper_result_eligible": False,
    }


def validate_execution_access_audit(
    audit: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed when a later train, holdout, or runtime stage crosses isolation."""
    validate_plan(plan)
    required_fields = {
        "schema_version",
        "kind",
        "plan_sha256",
        "stage",
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
        "expected_results_accessed",
        "evaluation_scene_accessed",
        "descriptor_model_id_accessed",
        "descriptor_dataset_id_accessed",
        "teacher_files_opened",
        "teacher_files_runtime_accessible",
        "runtime_teacher_path",
        "optimizer_executed",
        "asset_updated",
        "rerank_executed",
        "partition_reshuffled",
    }
    if not isinstance(audit, Mapping) or set(audit) != required_fields:
        raise JointCalibrationContractError("ACID joint access audit has unexpected fields")
    stage = audit.get("stage")
    if (
        audit.get("schema_version") != "1.0"
        or audit.get("kind") != ACCESS_AUDIT_KIND
        or audit.get("plan_sha256") != plan["plan_sha256"]
        or stage not in {"teacher", "train", "holdout", "runtime"}
        or audit.get("runtime_teacher_path") is not None
    ):
        raise JointCalibrationContractError("ACID joint access audit identity is invalid")
    for field in (
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
        "expected_results_accessed",
        "evaluation_scene_accessed",
        "descriptor_model_id_accessed",
        "descriptor_dataset_id_accessed",
        "teacher_files_runtime_accessible",
    ):
        if audit.get(field) is not False:
            raise JointCalibrationContractError(f"ACID joint access audit illegally opened {field}")
    for field in (
        "teacher_files_opened",
        "optimizer_executed",
        "asset_updated",
        "rerank_executed",
        "partition_reshuffled",
    ):
        if not isinstance(audit.get(field), bool):
            raise JointCalibrationContractError(f"ACID joint access audit has invalid {field}")
    if stage == "runtime" and (
        audit["teacher_files_opened"]
        or audit["optimizer_executed"]
        or audit["asset_updated"]
        or audit["rerank_executed"]
        or audit["partition_reshuffled"]
    ):
        raise JointCalibrationContractError("ACID deployed runtime crossed the offline boundary")
    if stage == "holdout" and (
        audit["optimizer_executed"]
        or audit["asset_updated"]
        or audit["rerank_executed"]
        or audit["partition_reshuffled"]
    ):
        raise JointCalibrationContractError("ACID holdout attempted to train or retune")
    if stage == "teacher" and (audit["optimizer_executed"] or audit["asset_updated"]):
        raise JointCalibrationContractError("ACID teacher stage attempted to update the asset")
    return {"stage": str(stage), "plan_sha256": str(plan["plan_sha256"]), "status": "PASS"}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--prepared-root", type=Path, required=True)
    plan_parser.add_argument("--evaluation-protocol", type=Path, default=ROOT / "artifact" / "evaluation_protocol.json")
    plan_parser.add_argument("--repository-root", type=Path, default=ROOT)
    plan_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate-plan")
    validate_parser.add_argument("--plan", type=Path, required=True)
    validate_parser.add_argument("--prepared-root", type=Path, required=True)
    validate_parser.add_argument("--evaluation-protocol", type=Path, default=ROOT / "artifact" / "evaluation_protocol.json")
    validate_parser.add_argument("--repository-root", type=Path, default=ROOT)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--plan", type=Path, required=True)
    materialize_parser.add_argument("--prepared-root", type=Path, required=True)
    materialize_parser.add_argument("--output-root", type=Path, required=True)

    validate_materialization_parser = subparsers.add_parser("validate-materialization")
    validate_materialization_parser.add_argument("--plan", type=Path, required=True)
    validate_materialization_parser.add_argument("--root", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "plan":
            record = plan_acid_joint_calibration(
                args.prepared_root,
                evaluation_protocol=args.evaluation_protocol,
                repository_root=args.repository_root,
            )
            _write_json(args.output, record)
            print(args.output)
        elif args.command == "validate-plan":
            record = validate_plan_file(
                args.plan,
                prepared_root=args.prepared_root,
                evaluation_protocol=args.evaluation_protocol,
                repository_root=args.repository_root,
            )
            print(json.dumps(record, indent=2, sort_keys=True))
        elif args.command == "materialize":
            plan = _read_json(args.plan, "ACID joint plan")
            record = materialize_context_only_sidecars(
                args.prepared_root,
                plan=plan,
                output_root=args.output_root,
            )
            print(json.dumps(record, indent=2, sort_keys=True))
        else:
            plan = _read_json(args.plan, "ACID joint plan")
            record = validate_materialization(args.root, plan=plan)
            print(json.dumps(record, indent=2, sort_keys=True))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
