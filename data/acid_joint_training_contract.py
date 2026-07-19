#!/usr/bin/env python3
"""Freeze and validate ACID joint-materialization calibration training.

This is a second protocol layer over ``plan_acid_joint_calibration.py``.  The
first layer pins the evaluation-disjoint ACID scenes and emits context-only
sidecars.  This layer pins the *one* shared calibrator training recipe before
an optimizer may run.  It is deliberately not a training runner and cannot
authorize a DL3DV quality experiment.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.frozen_audit_contract import require_fixed_file_sha256, sha256_file
from data.plan_acid_joint_calibration import (
    HOLDOUT_SPLIT,
    MODELS,
    PROTOCOL_ID,
    TRAIN_SPLIT,
    JointCalibrationContractError,
    validate_materialization,
    validate_plan,
)
from scripts.calibration_contract import canonical_sha256
from scripts.saes_execution_identity import (
    build_saes_execution_identity,
    validate_saes_execution_identity,
)
from saes.hardware_accounting import LEDGER_VERSION


CONTRACT_ID = "saes-joint-materialization-calibration-acid-v1-training-v5"
CONTRACT_KIND = "acid_joint_materialization_training_contract"
CONTRACT_STATUS = "FROZEN_PENDING_EXECUTION"
RESULT_KIND = "acid_joint_materialization_teacher_fidelity_v5"
ACCESS_AUDIT_KIND = "acid_joint_calibration_access_audit_v5"
CONTRACT_SCHEMA_VERSION = "5.0"
TEACHER_CACHE_KIND = "acid_joint_materialization_teacher_cache_v5"
TEACHER_CACHE_EXAMPLE_KIND = "acid_joint_materialization_teacher_example_v5"
TEACHER_CACHE_SCHEMA_VERSION = "5.0"
RUNTIME_EVIDENCE_KIND = "acid_joint_materialization_runtime_controls_v5"
RUNTIME_EVIDENCE_SCHEMA_VERSION = "5.0"
CANDIDATE_PROVENANCE_KIND = "acid_joint_materialization_candidate_provenance_v1"
CANDIDATE_PROVENANCE_SCHEMA_VERSION = "1.0"
CANDIDATE_VERIFICATION_KIND = "acid_joint_materialization_candidate_verification_v1"
CANDIDATE_VERIFICATION_SCHEMA_VERSION = "1.0"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")

DEFAULT_PLAN_PATH = ROOT / "artifact" / "protocol" / "acid_joint_calibration_plan.json"
DEFAULT_MATERIALIZATION_ROOT = (
    ROOT / "outputs" / "calibration" / "acid_joint_calibration_v1_context_only"
)
DEFAULT_CHECKPOINT_MANIFEST = ROOT / "artifact" / "manifests" / "checkpoints.json"
DEFAULT_TRAINING_CONTRACT_PATH = (
    ROOT / "artifact" / "protocol" / "acid_joint_materialization_training_contract_v5.json"
)
DEFAULT_TRAINING_OUTPUT_ROOT = ROOT / "outputs" / "calibration" / "acid_joint_materialization_training_v5"
DEFAULT_ASSET_PATH = (
    "outputs/calibration/acid_joint_materialization_training_v5/"
    "frozen_joint_materialization_calibrator.pt"
)
RESULT_RECORD_PATHS = {
    TRAIN_SPLIT: "outputs/calibration/acid_joint_materialization_training_v5/train_teacher_fidelity.json",
    HOLDOUT_SPLIT: "outputs/calibration/acid_joint_materialization_training_v5/holdout_teacher_fidelity.json",
}
RUNTIME_EVIDENCE_RECORDS = {
    split: {
        model: (
            "outputs/calibration/acid_joint_materialization_training_v5/"
            f"runtime_controls/{split}/{model}.json"
        )
        for model in MODELS
    }
    for split in (TRAIN_SPLIT, HOLDOUT_SPLIT)
}

# These are the only local implementation files allowed to participate in the
# author-side cache/train/holdout line. The generated contract records each
# byte hash, and the live preflight rejects any drift before cache or optimizer
# execution. The runner path is added once that source file exists.
IMPLEMENTATION_PATHS = (
    "data/acid_joint_training_contract.py",
    "data/frozen_audit_contract.py",
    "data/build_manifest.py",
    "data/verify_prepared_dataset.py",
    "data/plan_acid_joint_calibration.py",
    "integration/acid_joint_context.py",
    "integration/acid_joint_model_context.py",
    "integration/model_loader.py",
    "saes/probe_layout.py",
    "saes/progressive_saes.py",
    "saes/guard_policy.py",
    "saes/joint_materialization_calibrator.py",
    "saes/joint_materialization_teacher.py",
    "saes/selected_output_replay.py",
    "saes/selected_output_execution.py",
    "saes/probe_first_schedule.py",
    "saes/depthsplat_s3_footprint.py",
    "saes/hardware_accounting.py",
    "scripts/ae_config.py",
    "scripts/calibration_contract.py",
    "scripts/saes_execution_identity.py",
    "scripts/compile_calibration.py",
    "scripts/compile_protocol.py",
    "scripts/acid_joint_teacher_cache_worker.py",
    "scripts/acid_joint_calibration_runner.py",
    "scripts/acid_joint_runtime_control_evidence.py",
)

CHECKPOINT_REQUESTS: dict[str, dict[str, str]] = {
    "transplat": {
        "checkpoint_dataset": "acid",
        "checkpoint_path": "transplat/checkpoints/acid.ckpt",
    },
    "mvsplat": {
        "checkpoint_dataset": "acid",
        "checkpoint_path": "mvsplat/checkpoints/acid.ckpt",
    },
    "depthsplat": {
        "checkpoint_dataset": "re10k",
        "checkpoint_path": "depthsplat/checkpoints/re10k.ckpt",
    },
}

MODEL_CONFIG_REQUESTS: dict[str, dict[str, Any]] = {
    "transplat": {
        "entrypoint_path": "transplat/config/experiment/acid.yaml",
        "source_paths": [
            "transplat/config/main.yaml",
            "transplat/config/experiment/acid.yaml",
            "transplat/config/dataset/re10k.yaml",
            "transplat/config/dataset/view_sampler/bounded.yaml",
            "transplat/config/dataset/view_sampler_dataset_specific_config/bounded_re10k.yaml",
            "transplat/config/model/encoder/trans.yaml",
            "transplat/config/model/decoder/splatting_cuda.yaml",
            "transplat/config/loss/mse.yaml",
            "transplat/config/loss/lpips.yaml",
        ],
        "hydra_overrides": [],
    },
    "mvsplat": {
        "entrypoint_path": "mvsplat/config/experiment/acid.yaml",
        "source_paths": [
            "mvsplat/config/main.yaml",
            "mvsplat/config/experiment/acid.yaml",
            "mvsplat/config/dataset/re10k.yaml",
            "mvsplat/config/dataset/view_sampler/bounded.yaml",
            "mvsplat/config/dataset/view_sampler_dataset_specific_config/bounded_re10k.yaml",
            "mvsplat/config/model/encoder/costvolume.yaml",
            "mvsplat/config/model/decoder/splatting_cuda.yaml",
            "mvsplat/config/loss/mse.yaml",
            "mvsplat/config/loss/lpips.yaml",
        ],
        "hydra_overrides": [],
    },
    "depthsplat": {
        "entrypoint_path": "depthsplat/config/experiment/re10k.yaml",
        "source_paths": [
            "depthsplat/config/main.yaml",
            "depthsplat/config/experiment/re10k.yaml",
            "depthsplat/config/dataset/re10k.yaml",
            "depthsplat/config/dataset/view_sampler/bounded.yaml",
            "depthsplat/config/dataset/view_sampler_dataset_specific_config/bounded_re10k.yaml",
            "depthsplat/config/model/encoder/depthsplat.yaml",
            "depthsplat/config/model/decoder/splatting_cuda.yaml",
            "depthsplat/config/loss/mse.yaml",
            "depthsplat/config/loss/lpips.yaml",
        ],
        "hydra_overrides": [
            "model.encoder.num_scales=2",
            "model.encoder.upsample_factor=2",
            "model.encoder.lowest_feature_resolution=4",
            "model.encoder.monodepth_vit_type=vitl",
        ],
    },
}

MODEL_SOURCE_ROOTS = {model: model for model in MODELS}

# The loader imports every model's local ``src`` package. DepthSplat also
# constructs its DINOv2 backbone through a pinned local torch.hub checkout;
# bind that executable tree as part of the same runtime identity.
MODEL_RUNTIME_IMPORT_ROOTS: dict[str, tuple[str, ...]] = {
    "transplat": ("transplat/src",),
    "mvsplat": ("mvsplat/src",),
    "depthsplat": (
        "depthsplat/src",
        "assets/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    ),
}

MODEL_ENVIRONMENT_PROFILES = {
    "transplat": "classic",
    "mvsplat": "classic",
    "depthsplat": "depthsplat",
}

ACID_ISOLATED_MODEL_ENV = "SCARF_ACID_ISOLATED_MODEL"
_ISOLATED_MODEL_PROCESS: tuple[int, str] | None = None

PREPROCESSING_CONTRACT = {
    "identifier": "upstream_lanczos_rescale_center_crop_intrinsics_v1",
    "source_image_shape": [360, 640],
    "target_image_shape": [256, 256],
    "rescale_rule": "scale=max(target_h/source_h,target_w/source_w); rounded_scaled_shape",
    "interpolation": "Pillow.Image.LANCZOS_uint8_roundtrip",
    "crop_rule": "center_crop_after_rescale",
    "intrinsics_rule": "upstream_normalized_fx_fy_scale_by_precrop_width_height_over_target",
    "patch_alignment_rule": "upstream_center_crop_to_encoder_patch_multiple_with_intrinsics_scale",
    "target_rgb_or_camera_access": False,
    "crop_shim_paths": {
        "transplat": "transplat/src/dataset/shims/crop_shim.py",
        "mvsplat": "mvsplat/src/dataset/shims/crop_shim.py",
        "depthsplat": "depthsplat/src/dataset/shims/crop_shim.py",
    },
    "patch_shim_paths": {
        "transplat": "transplat/src/dataset/shims/patch_shim.py",
        "mvsplat": "mvsplat/src/dataset/shims/patch_shim.py",
        "depthsplat": "depthsplat/src/dataset/shims/patch_shim.py",
    },
    "per_model_patch_size": {"transplat": 16, "mvsplat": 16, "depthsplat": 16},
}

def _routing_parameters_for_identity(execution_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the one registered route in cache/runtime-specific terminology."""

    identity = validate_saes_execution_identity(execution_identity)
    return {
        "tile_size": identity["tile_size"],
        "feature_threshold_tau_f": identity["feature_threshold"],
        "depth_threshold_tau_d": identity["depth_threshold"],
        "decision_semantics": identity["decision_semantics"],
        "depth_routing_semantics": identity["depth_routing_semantics"],
        "route_order": identity["route_order"],
        "l0_retained_positions": identity["l0_anchor_count"],
        "l1_retained_positions": identity["l1_anchor_count"],
        "l0_anchor_layout": identity["l0_anchor_layout"],
        "l0_anchor_positions": identity["l0_anchor_positions"],
        "l1_anchor_layout": identity["l1_anchor_layout"],
        "l1_anchor_positions": identity["l1_anchor_positions"],
        "l0_budget": "K(T)",
        "l1_budget": "fixed-12-anchor-boundary",
        "full_budget": identity["full_tile_mode"],
        "beta_x": 0.5,
        "beta_f": 0.1,
        "beta_d": 1.0,
        "cross_check_threshold": identity["cross_check_threshold"],
        "materialization": identity["materialization"],
        "l1_depth_reference": identity["l1_depth_reference"],
        "moment_geometry": identity["moment_geometry"],
        "materialization_guard": identity["materialization_guard"],
        "materialization_guard_policy": identity["materialization_guard_policy"],
        "materialization_guard_min_covariance_cosine": identity[
            "materialization_guard_min_covariance_cosine"
        ],
        "materialization_guard_min_harmonic_cosine": identity[
            "materialization_guard_min_harmonic_cosine"
        ],
        "materialization_guard_max_opacity_distance": identity[
            "materialization_guard_max_opacity_distance"
        ],
        "context_safety_guard": identity["context_safety_guard"],
        "context_guard_policy": identity["context_guard_policy"],
        "context_guard_max_footprint_ratio": identity[
            "context_guard_max_footprint_ratio"
        ],
        "context_guard_max_relative_depth_span": identity[
            "context_guard_max_relative_depth_span"
        ],
        "context_guard_max_center_mahalanobis": identity[
            "context_guard_max_center_mahalanobis"
        ],
        "guard_nonprobe_s3_attribute_reads_must_equal": 0,
        "offline_capture_only": True,
        "joint_calibrator_during_cache_capture": False,
    }


# Kept as a public fixture helper for the synthetic contract tests. Production
# binding always recreates this mapping from the registered execution identity.
SAES_ROUTING_PARAMETERS = _routing_parameters_for_identity(
    build_saes_execution_identity()
)

SAES_MODEL_ROUTING = {
    "transplat": {"gpp": 1, "num_depth_candidates": 128, "ray_depth_mode": "euclidean"},
    "mvsplat": {"gpp": 1, "num_depth_candidates": 128, "ray_depth_mode": "euclidean"},
    "depthsplat": {"gpp": 1, "num_depth_candidates": 128, "ray_depth_mode": "z"},
}

CALIBRATOR_ARCHITECTURE = {
    "schema_version": "saes-joint-materialization-calibrator-v1",
    "kind": "saes-joint-materialization-calibrator-state-dict",
    "descriptor_dim": 32,
    "bottleneck_dim": 8,
    "joint_output_dim": 40,
    "parameter_count": 624,
    "network_macs_per_descriptor": 576,
    "shared_across_models_and_datasets": True,
}

OPTIMIZATION_RECIPE = {
    "optimizer": "AdamW",
    "seed": 20260718,
    "training_dtype": "float32",
    "deterministic_algorithms_required": True,
    "batch_size": 256,
    "total_updates": 12000,
    "warmup_updates": 500,
    "learning_rate": 0.001,
    "final_learning_rate": 0.00001,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "weight_decay": 1e-5,
    "gradient_clip_norm": 1.0,
    "schedule": "linear_warmup_then_cosine_decay",
    "model_cache_execution": "one-model-per-subprocess",
    "model_sampling": "round_robin_transplat_mvsplat_depthsplat",
    "scene_order": "sidecar_ordinal_ascending",
    "anchor_order": "stable_retained_anchor_index_ascending",
    "resume_allowed": False,
    "early_stopping_allowed": False,
    "hyperparameter_search_allowed": False,
}

TEACHER_OBJECTIVE = {
    "offline_teacher_only": True,
    "teacher_source": "dense_adaptor_output",
    "teacher_storage_scope": "author_training_only",
    "teacher_cache_root": "outputs/calibration/acid_joint_materialization_training_v5/teacher_cache",
    "teacher_cache_generation": "one-model-per-subprocess_from_context_only_sidecars",
    "teacher_target": "assignment_aligned_dense_adaptor_correction_packet",
    "offline_teacher_dense_nonprobe_attributes_allowed": True,
    "route_and_assignments_frozen_before_teacher_access": True,
    "dense_nonprobe_attributes_persisted": False,
    "target_camera_or_render_allowed": False,
    "target_rgb_or_ground_truth_allowed": False,
    "runtime_teacher_access_allowed": False,
    "teacher_components": ["mean", "covariance", "opacity", "sh"],
    "loss": {
        "name": "identity_normalized_joint_packet_mse",
        "component_weights": {
            "mean": 1.0,
            "covariance": 1.0,
            "opacity": 1.0,
            "sh": 1.0,
        },
        "identity_baseline": "selected_only_ordinary_representative_reconstruction_zero_correction",
        "normalizer": "per_model_per_component_identity_mse",
        "normalizer_floor": 1e-8,
        "aggregate": "unweighted_mean_of_component_relative_mse",
    },
    "allowed_inputs": [
        "context_only_sidecar_records",
        "frozen_model_checkpoint",
        "probe_attributes",
        "context_feature_statistics",
        "context_depth_statistics",
        "assignment_statistics",
        "context_camera_geometry",
        "dense_adaptor_output",
    ],
    "forbidden_inputs": [
        "target_rgb",
        "target_camera_metadata",
        "target_indices",
        "ground_truth_metrics",
        "artifact/expected_results.json",
        "paper_tables",
        "evaluation_scenes",
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
}

RUNTIME_PRECONDITIONS = {
    "route_mask_unchanged_required": True,
    "retained_counts_unchanged_required": True,
    "full_passthrough_required": True,
    "selected_only_s3_required": True,
    "two_finite_skipped_s3_sentinels_required": True,
    "selected_head_event_contract_required": True,
    "cost_ledger_required": True,
    "all_three_models_required": True,
}

TEACHER_FIDELITY_GATES = {
    "all_models_required": True,
    "all_components_nonworsening_max_relative_mse": 1.0,
    "train_joint_relative_mse_max": 0.9,
    "holdout_joint_relative_mse_max": 0.95,
    "finite_required": True,
    "covariance_psd_required": True,
    "runtime_preconditions": RUNTIME_PRECONDITIONS,
}

HOLDOUT_POLICY = {
    "optimizer_allowed": False,
    "asset_update_allowed": False,
    "rerank_allowed": False,
    "partition_reshuffle_allowed": False,
    "threshold_or_budget_change_allowed": False,
    "frozen_train_asset_required": True,
}

# A local provenance JSON can bind bytes to inputs, but it cannot prove that
# an optimizer or cache generator actually produced those bytes. v5 therefore
# starts fail-closed: future protocol revisions must pin either a deterministic
# replay implementation or an external verifier public key before promotion.
CANDIDATE_VERIFICATION_POLICY = {
    "schema_version": "1.0",
    "candidate_runtime_and_promotion_verification_required": True,
    "local_hash_chain_is_not_cryptographic_proof": True,
    "registered_deterministic_replay_verifier": None,
    "registered_external_attestation_public_key_sha256": None,
}

_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "contract_id",
        "protocol_id",
        "author_side_prerequisite",
        "paper_result_eligible",
        "mechanism_config_write_allowed",
        "dl3dv_quality_gate_authorized",
        "source_plan",
        "context_only_inputs",
        "checkpoint_binding",
        "model_config_binding",
        "preprocessing",
        "saes_routing",
        "implementation_binding",
        "shared_asset",
        "result_records",
        "runtime_evidence_records",
        "candidate_verification_policy",
        "optimization_recipe",
        "teacher_objective",
        "teacher_fidelity_gates",
        "holdout_policy",
        "contract_sha256",
    }
)
_METRIC_COMPONENTS = ("mean", "covariance", "opacity", "sh")


class JointTrainingContractError(ValueError):
    """Raised when the frozen shared-calibrator training line is invalid."""


def require_isolated_model_process(model: str) -> None:
    """Require a worker API to run in one explicit model-scoped process."""

    global _ISOLATED_MODEL_PROCESS
    if model not in MODELS or os.environ.get(ACID_ISOLATED_MODEL_ENV) != model:
        raise JointTrainingContractError(
            "ACID model work must run in its dedicated isolated subprocess"
        )
    current = (os.getpid(), model)
    if _ISOLATED_MODEL_PROCESS is not None and _ISOLATED_MODEL_PROCESS != current:
        raise JointTrainingContractError(
            "one ACID worker process may not execute more than one model"
        )
    _ISOLATED_MODEL_PROCESS = current


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointTrainingContractError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise JointTrainingContractError(f"{label} must be an object")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise JointTrainingContractError(f"{label} must be a SHA256 digest")
    return value


def _canonical_json_value(value: Any) -> Any:
    """Convert a typed Hydra configuration into deterministic JSON data."""

    if is_dataclass(value):
        return _canonical_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return _canonical_json_value(dict(value.items()))
    if hasattr(value, "__dict__"):
        return _canonical_json_value(vars(value))
    raise TypeError(f"resolved configuration contains unsupported {type(value).__name__}")


def resolved_config_sha256(config: Any) -> str:
    """Hash the actual typed model configuration used by an encoder worker."""

    return canonical_sha256(_canonical_json_value(config))


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JointTrainingContractError(f"{label} must be a positive integer")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise JointTrainingContractError(f"{label} must be finite and nonnegative")
    return float(value)


def _safe_repository_path(repository_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise JointTrainingContractError(f"{label} path is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise JointTrainingContractError(f"{label} path is unsafe")
    root = Path(repository_root).resolve()
    resolved = (root / Path(*pure.parts)).resolve()
    if root not in resolved.parents and resolved != root:
        raise JointTrainingContractError(f"{label} path escapes the repository")
    return resolved


def _relative_to_root(path: Path, repository_root: Path, label: str) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repository_root).resolve()).as_posix()
    except ValueError as exc:
        raise JointTrainingContractError(f"{label} must be inside the repository") from exc


def _runtime_import_tree_binding(
    repository_root: Path, relative_root: str, *, model: str
) -> dict[str, Any]:
    """Hash one local tree that can be imported by a model runtime."""

    source_root = _safe_repository_path(
        repository_root, relative_root, f"{model} runtime import root"
    )
    if not source_root.is_dir():
        raise JointTrainingContractError(f"{model} runtime import root is missing: {relative_root}")
    files: list[dict[str, str]] = []
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        relative_parts = path.relative_to(source_root).parts
        if (
            "__pycache__" in relative_parts
            or ".git" in relative_parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise JointTrainingContractError(
                f"{model} runtime import tree contains an unsafe entry"
            )
        resolved = path.resolve()
        if source_root not in resolved.parents:
            raise JointTrainingContractError(
                f"{model} runtime import tree entry escapes its root"
            )
        files.append(
            {
                "path": _relative_to_root(path, repository_root, f"{model} runtime source file"),
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise JointTrainingContractError(
            f"{model} runtime import root has no executable files: {relative_root}"
        )
    return {
        "path": relative_root,
        "tree_sha256": canonical_sha256(files),
        "file_count": len(files),
    }


def _model_source_binding(repository_root: Path, model: str) -> dict[str, Any]:
    """Bind every source tree dynamically imported by one model loader."""

    if model not in MODELS:
        raise JointTrainingContractError(f"unknown model source binding: {model}")
    repository_path = MODEL_SOURCE_ROOTS[model]
    model_root = _safe_repository_path(
        repository_root, repository_path, f"{model} source repository"
    )
    if not model_root.is_dir():
        raise JointTrainingContractError(f"{model} source repository is missing")
    process = subprocess.run(
        ["git", "-C", str(model_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = process.stdout.strip()
    if process.returncode != 0 or GIT_COMMIT.fullmatch(commit) is None:
        raise JointTrainingContractError(f"{model} source commit is unavailable")

    roots = [
        _runtime_import_tree_binding(repository_root, relative, model=model)
        for relative in MODEL_RUNTIME_IMPORT_ROOTS[model]
    ]
    src = roots[0]
    expected_src = f"{repository_path}/src"
    if src["path"] != expected_src:
        raise JointTrainingContractError(f"{model} runtime source root no longer starts at src")
    return {
        "repository_path": repository_path,
        "repository_commit": commit,
        "src_path": expected_src,
        "src_tree_sha256": src["tree_sha256"],
        "src_file_count": src["file_count"],
        "runtime_import_roots": roots,
    }


def _profile_python(profile: str) -> Path:
    variable = f"SCARF_PYTHON_{profile.upper()}"
    configured = os.environ.get(variable)
    candidate = Path(configured).expanduser() if configured else Path(sys.executable)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise JointTrainingContractError(
            f"{variable} or the invoking Python must name an executable interpreter"
        )
    return candidate.resolve()


def _profile_interpreter_binding(profile: str) -> dict[str, str]:
    """Bind the exact interpreter used to resolve and execute one profile."""

    executable = _profile_python(profile)
    process = subprocess.run(
        [
            str(executable),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    version = process.stdout.strip()
    if process.returncode != 0 or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise JointTrainingContractError(f"{profile} profile interpreter version is unavailable")
    return {
        "executable_sha256": sha256_file(executable),
        "python_version": version,
    }


def _runtime_profile_interpreter_binding(profile: str) -> dict[str, str]:
    """Bind the active worker interpreter and reject a mismatched profile."""

    expected = _profile_python(profile)
    actual = Path(sys.executable).resolve()
    if actual != expected:
        raise JointTrainingContractError(
            f"{profile} runtime must execute with its configured profile interpreter"
        )
    return {
        "executable_sha256": sha256_file(actual),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
    }


def _model_config_source_files(repository_root: Path, model: str) -> list[dict[str, str]]:
    """Rehash the exact Hydra source-file set declared for one model."""

    if model not in MODELS:
        raise JointTrainingContractError(f"unknown model config binding: {model}")
    request = MODEL_CONFIG_REQUESTS[model]
    files: list[dict[str, str]] = []
    for relative in request["source_paths"]:
        path = _safe_repository_path(repository_root, relative, f"{model} config")
        if not path.is_file():
            raise JointTrainingContractError(f"{model} config source is missing: {relative}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    if [item["path"] for item in files] != request["source_paths"]:
        raise JointTrainingContractError(f"{model} config source order changed")
    return files


def _resolved_model_config_sha256s(repository_root: Path) -> dict[str, str]:
    """Resolve each model config in a fresh profile interpreter before freezing."""

    if Path(repository_root).resolve() != ROOT:
        raise JointTrainingContractError(
            "resolved model configuration requires the canonical repository root"
        )
    from scripts.ae_config import resolve_experiment

    records: dict[str, str] = {}
    script = Path(__file__).resolve()
    for model in MODELS:
        profile = resolve_experiment(model, "acid", repository_root).environment_profile
        process = subprocess.run(
            [
                str(_profile_python(profile)),
                str(script),
                "resolve-model-config",
                "--model",
                model,
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip()
            raise JointTrainingContractError(
                f"{model} resolved configuration failed in the {profile} profile: {detail}"
            )
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise JointTrainingContractError(
                f"{model} resolved configuration emitted invalid JSON"
            ) from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"model", "resolved_config_sha256"}
            or value.get("model") != model
        ):
            raise JointTrainingContractError(
                f"{model} resolved configuration has an invalid schema"
            )
        records[model] = _sha256(
            value.get("resolved_config_sha256"),
            f"{model} resolved configuration SHA256",
        )
    return records


def _resolved_model_config_sha256_here(model: str) -> str:
    """Resolve one model config in its isolated worker process."""

    if model not in MODELS:
        raise JointTrainingContractError("model configuration model is invalid")
    from integration.model_loader import create_model_loader
    from scripts.ae_config import resolve_experiment

    experiment = resolve_experiment(model, "acid", ROOT)
    config = create_model_loader(model).resolve_model_config(
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
    )
    return resolved_config_sha256(config)


def _checkpoint_records(
    checkpoint_manifest_path: Path, *, repository_root: Path
) -> dict[str, dict[str, Any]]:
    """Resolve the three checkpoint identities without trusting path labels."""
    manifest = _read_json(checkpoint_manifest_path, "checkpoint manifest")
    records = manifest.get("files")
    if manifest.get("schema_version") != "1.0" or not isinstance(records, list):
        raise JointTrainingContractError("checkpoint manifest has an invalid schema")
    resolved: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        request = CHECKPOINT_REQUESTS[model]
        candidates = [
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("model") == model
            and record.get("dataset") == request["checkpoint_dataset"]
            and record.get("path") == request["checkpoint_path"]
        ]
        if len(candidates) != 1:
            raise JointTrainingContractError(f"checkpoint manifest has no unique {model} binding")
        record = candidates[0]
        expected_fields = {"checkpoint_dataset", "checkpoint_path", "checkpoint_sha256", "checkpoint_size"}
        candidate = {
            "checkpoint_dataset": record.get("dataset"),
            "checkpoint_path": record.get("path"),
            "checkpoint_sha256": record.get("sha256"),
            "checkpoint_size": record.get("size"),
        }
        if set(candidate) != expected_fields:
            raise JointTrainingContractError(f"checkpoint manifest has malformed {model} binding")
        _sha256(candidate["checkpoint_sha256"], f"{model} checkpoint SHA256")
        _positive_int(candidate["checkpoint_size"], f"{model} checkpoint size")
        checkpoint_path = _safe_repository_path(
            repository_root, candidate["checkpoint_path"], f"{model} checkpoint"
        )
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size != candidate["checkpoint_size"]:
            raise JointTrainingContractError(f"{model} checkpoint is missing or has a different byte count")
        require_fixed_file_sha256(
            checkpoint_path,
            candidate["checkpoint_sha256"],
            label=f"{model} checkpoint",
        )
        resolved[model] = candidate
    return resolved


def validate_runtime_checkpoint_binding(
    contract: Mapping[str, Any],
    *,
    model: str,
    checkpoint_path: Path,
    repository_root: Path = ROOT,
) -> str:
    """Rehash a checkpoint immediately before or after model construction."""

    validate_training_contract(contract)
    repository_root = Path(repository_root).resolve()
    if repository_root != ROOT or model not in MODELS:
        raise JointTrainingContractError(
            "runtime checkpoint binding requires the canonical repository and model"
        )
    expected = contract["checkpoint_binding"]["models"][model]
    expected_path = _safe_repository_path(
        repository_root, expected["checkpoint_path"], f"{model} runtime checkpoint"
    )
    actual_path = Path(checkpoint_path).resolve()
    if actual_path != expected_path:
        raise JointTrainingContractError(
            f"{model} runtime checkpoint path differs from training contract"
        )
    if not actual_path.is_file() or actual_path.stat().st_size != expected["checkpoint_size"]:
        raise JointTrainingContractError(
            f"{model} runtime checkpoint byte count differs from training contract"
        )
    require_fixed_file_sha256(
        actual_path, expected["checkpoint_sha256"], label=f"{model} runtime checkpoint"
    )
    return str(expected["checkpoint_sha256"])


def _model_config_records(repository_root: Path) -> dict[str, dict[str, Any]]:
    """Bind all configuration sources before model-specific cache extraction."""

    resolved_configs = _resolved_model_config_sha256s(repository_root)
    resolved: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        request = MODEL_CONFIG_REQUESTS[model]
        files = _model_config_source_files(repository_root, model)
        entrypoint = request["entrypoint_path"]
        entrypoint_hash = next(
            item["sha256"] for item in files if item["path"] == entrypoint
        )
        profile = MODEL_ENVIRONMENT_PROFILES[model]
        resolved[model] = {
            "entrypoint_path": entrypoint,
            "entrypoint_sha256": entrypoint_hash,
            "source_files": files,
            "source_manifest_sha256": canonical_sha256(files),
            "hydra_overrides": list(request["hydra_overrides"]),
            "environment_profile": profile,
            "profile_interpreter": _profile_interpreter_binding(profile),
            "runtime_source": _model_source_binding(repository_root, model),
            "resolved_config_sha256": resolved_configs[model],
        }
    return resolved


def _preprocessing_binding(repository_root: Path) -> dict[str, Any]:
    value = {
        **PREPROCESSING_CONTRACT,
        "crop_shim_sha256": {},
        "patch_shim_sha256": {},
    }
    for model, relative in PREPROCESSING_CONTRACT["crop_shim_paths"].items():
        path = _safe_repository_path(repository_root, relative, f"{model} crop shim")
        if not path.is_file():
            raise JointTrainingContractError(f"{model} crop shim is missing")
        value["crop_shim_sha256"][model] = sha256_file(path)
    for model, relative in PREPROCESSING_CONTRACT["patch_shim_paths"].items():
        path = _safe_repository_path(repository_root, relative, f"{model} patch shim")
        if not path.is_file():
            raise JointTrainingContractError(f"{model} patch shim is missing")
        value["patch_shim_sha256"][model] = sha256_file(path)
    return value


def _saes_routing_binding(repository_root: Path) -> dict[str, Any]:
    """Bind the exact L0/L1/Full routing used to construct teacher caches."""
    mechanism_path = _safe_repository_path(
        repository_root, "artifact/mechanism_config.json", "mechanism configuration"
    )
    implementation_path = _safe_repository_path(
        repository_root, "saes/progressive_saes.py", "SAES routing implementation"
    )
    if not mechanism_path.is_file() or not implementation_path.is_file():
        raise JointTrainingContractError("SAES routing sources are missing")
    execution_identity = build_saes_execution_identity()
    try:
        mechanism_config = _read_json(mechanism_path, "mechanism configuration")
        configured_identity = validate_saes_execution_identity(
            mechanism_config.get("saes_execution_identity")
        )
    except (OSError, ValueError) as exc:
        raise JointTrainingContractError(
            "mechanism configuration does not bind the registered SAES execution route"
        ) from exc
    if configured_identity != execution_identity:
        raise JointTrainingContractError(
            "mechanism configuration SAES execution route differs from the training route"
        )
    value = {
        "execution_identity": execution_identity,
        "execution_route_sha256": execution_identity["route_sha256"],
        "parameters": _routing_parameters_for_identity(execution_identity),
        "per_model": {model: dict(SAES_MODEL_ROUTING[model]) for model in MODELS},
        "mechanism_config_path": "artifact/mechanism_config.json",
        "mechanism_config_sha256": sha256_file(mechanism_path),
        "implementation_path": "saes/progressive_saes.py",
        "implementation_sha256": sha256_file(implementation_path),
    }
    value["routing_sha256"] = canonical_sha256(value)
    return value


def _implementation_binding(repository_root: Path) -> dict[str, Any]:
    """Hash every executable component of the frozen calibration line."""
    files: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_PATHS:
        path = _safe_repository_path(repository_root, relative, "training implementation")
        if not path.is_file():
            raise JointTrainingContractError(
                f"frozen training implementation is missing: {relative}"
            )
        files.append({"path": relative, "sha256": sha256_file(path)})
    value = {"files": files}
    value["implementation_tree_sha256"] = canonical_sha256(files)
    return value


def _context_input_binding(
    materialization_root: Path,
    *,
    plan: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Rehash every context-only input before it is admitted to training."""
    try:
        identity = validate_materialization(materialization_root, plan=plan)
    except (JointCalibrationContractError, FileNotFoundError, OSError) as exc:
        raise JointTrainingContractError("context-only ACID materialization is invalid") from exc
    root = Path(materialization_root).resolve()
    materialization_path = root / "materialization.json"
    if not materialization_path.is_file():
        raise JointTrainingContractError("context-only materialization record is missing")
    splits: dict[str, dict[str, Any]] = {}
    for split in (TRAIN_SPLIT, HOLDOUT_SPLIT):
        sidecar = identity["sidecars"].get(split)
        if not isinstance(sidecar, Mapping):
            raise JointTrainingContractError(f"context-only {split} identity is missing")
        required = {
            "split",
            "tree_sha256",
            "manifest_sha256",
            "input_provenance_sha256",
            "selection_sha256",
            "scene_count",
            "target_rgb_accessed",
            "target_camera_metadata_accessed",
            "target_index_accessed",
            "teacher_artifact_accessed",
            "expected_results_accessed",
        }
        if set(sidecar) != required:
            raise JointTrainingContractError(f"context-only {split} identity has unexpected fields")
        for field in (
            "tree_sha256",
            "manifest_sha256",
            "input_provenance_sha256",
            "selection_sha256",
        ):
            _sha256(sidecar[field], f"context-only {split} {field}")
        if sidecar["split"] != split:
            raise JointTrainingContractError(f"context-only {split} has the wrong split identity")
        for field in (
            "target_rgb_accessed",
            "target_camera_metadata_accessed",
            "target_index_accessed",
            "teacher_artifact_accessed",
            "expected_results_accessed",
        ):
            if sidecar[field] is not False:
                raise JointTrainingContractError(f"context-only {split} accessed {field}")
        splits[split] = dict(sidecar)
    return {
        "materialization_root": _relative_to_root(
            root, repository_root, "context-only materialization root"
        ),
        "outer_tree_sha256": identity["tree_sha256"],
        "outer_manifest_sha256": identity["manifest_sha256"],
        "materialization_sha256": sha256_file(materialization_path),
        "splits": splits,
    }


def _source_plan_binding(plan_path: Path, plan: Mapping[str, Any], *, repository_root: Path) -> dict[str, str]:
    return {
        "path": _relative_to_root(plan_path, repository_root, "ACID joint plan"),
        "file_sha256": sha256_file(plan_path),
        "plan_sha256": str(plan["plan_sha256"]),
        "protocol_id": str(plan["protocol_id"]),
    }


def _shared_asset_contract() -> dict[str, Any]:
    return {
        "asset_path": DEFAULT_ASSET_PATH,
        "single_global_asset_required": True,
        "per_model_or_dataset_assets_forbidden": True,
        "runtime_hash_pinning_required": True,
        "architecture": dict(CALIBRATOR_ARCHITECTURE),
    }


def build_training_contract(
    *,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
    checkpoint_manifest_path: Path = DEFAULT_CHECKPOINT_MANIFEST,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Build a new immutable pre-optimization contract from live identities."""
    repository_root = Path(repository_root).resolve()
    plan_path = Path(plan_path).resolve()
    if repository_root != ROOT or plan_path != DEFAULT_PLAN_PATH:
        raise JointTrainingContractError(
            "joint training contracts may only freeze the canonical repository and ACID plan"
        )
    plan = _read_json(plan_path, "ACID joint plan")
    try:
        validate_plan(plan)
    except JointCalibrationContractError as exc:
        raise JointTrainingContractError("ACID joint plan is invalid") from exc
    if plan.get("protocol_id") != PROTOCOL_ID:
        raise JointTrainingContractError("ACID joint plan has the wrong protocol id")
    materialization_root = Path(materialization_root).resolve()
    checkpoint_manifest_path = Path(checkpoint_manifest_path).resolve()
    if (
        materialization_root != DEFAULT_MATERIALIZATION_ROOT
        or checkpoint_manifest_path != DEFAULT_CHECKPOINT_MANIFEST
    ):
        raise JointTrainingContractError(
            "joint training contracts may only freeze canonical ACID inputs and checkpoints"
        )
    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "status": CONTRACT_STATUS,
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "author_side_prerequisite": True,
        "paper_result_eligible": False,
        "mechanism_config_write_allowed": False,
        "dl3dv_quality_gate_authorized": False,
        "source_plan": _source_plan_binding(plan_path, plan, repository_root=repository_root),
        "context_only_inputs": _context_input_binding(
            materialization_root, plan=plan, repository_root=repository_root
        ),
        "checkpoint_binding": {
            "checkpoint_manifest_path": _relative_to_root(
                checkpoint_manifest_path, repository_root, "checkpoint manifest"
            ),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
            "models": _checkpoint_records(
                checkpoint_manifest_path, repository_root=repository_root
            ),
            "model_order": list(MODELS),
            "shared_asset_model_independent": True,
        },
        "model_config_binding": _model_config_records(repository_root),
        "preprocessing": _preprocessing_binding(repository_root),
        "saes_routing": _saes_routing_binding(repository_root),
        "implementation_binding": _implementation_binding(repository_root),
        "shared_asset": _shared_asset_contract(),
        "result_records": dict(RESULT_RECORD_PATHS),
        "runtime_evidence_records": {
            split: dict(records) for split, records in RUNTIME_EVIDENCE_RECORDS.items()
        },
        "candidate_verification_policy": dict(CANDIDATE_VERIFICATION_POLICY),
        "optimization_recipe": dict(OPTIMIZATION_RECIPE),
        "teacher_objective": dict(TEACHER_OBJECTIVE),
        "teacher_fidelity_gates": dict(TEACHER_FIDELITY_GATES),
        "holdout_policy": dict(HOLDOUT_POLICY),
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    validate_training_contract(contract)
    return contract


def _validate_source_plan_binding(value: Any) -> dict[str, str]:
    required = {"path", "file_sha256", "plan_sha256", "protocol_id"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("training contract source plan binding is invalid")
    if (
        value.get("path") != DEFAULT_PLAN_PATH.relative_to(ROOT).as_posix()
        or _sha256(value.get("file_sha256"), "training contract plan file SHA256") is None
        or _sha256(value.get("plan_sha256"), "training contract plan SHA256") is None
        or value.get("protocol_id") != PROTOCOL_ID
    ):
        raise JointTrainingContractError("training contract source plan binding is invalid")
    return dict(value)


def _validate_context_input_binding(value: Any) -> dict[str, Any]:
    required = {
        "materialization_root",
        "outer_tree_sha256",
        "outer_manifest_sha256",
        "materialization_sha256",
        "splits",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("training contract context-only input binding is invalid")
    if value.get("materialization_root") != DEFAULT_MATERIALIZATION_ROOT.relative_to(ROOT).as_posix():
        raise JointTrainingContractError("training contract materialization root is invalid")
    for field in ("outer_tree_sha256", "outer_manifest_sha256", "materialization_sha256"):
        _sha256(value.get(field), f"training contract {field}")
    splits = value.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {TRAIN_SPLIT, HOLDOUT_SPLIT}:
        raise JointTrainingContractError("training contract has the wrong context-only splits")
    expected_scene_counts = {TRAIN_SPLIT: 24, HOLDOUT_SPLIT: 8}
    expected_fields = {
        "split",
        "tree_sha256",
        "manifest_sha256",
        "input_provenance_sha256",
        "selection_sha256",
        "scene_count",
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
        "teacher_artifact_accessed",
        "expected_results_accessed",
    }
    for split in (TRAIN_SPLIT, HOLDOUT_SPLIT):
        sidecar = splits.get(split)
        if not isinstance(sidecar, Mapping) or set(sidecar) != expected_fields:
            raise JointTrainingContractError(f"training contract {split} identity is invalid")
        for field in (
            "tree_sha256",
            "manifest_sha256",
            "input_provenance_sha256",
            "selection_sha256",
        ):
            _sha256(sidecar.get(field), f"training contract {split} {field}")
        if sidecar.get("split") != split:
            raise JointTrainingContractError(f"training contract {split} split identity changed")
        if sidecar.get("scene_count") != expected_scene_counts[split]:
            raise JointTrainingContractError(f"training contract {split} scene count changed")
        for field in expected_fields - {
            "split",
            "tree_sha256",
            "manifest_sha256",
            "input_provenance_sha256",
            "selection_sha256",
            "scene_count",
        }:
            if sidecar.get(field) is not False:
                raise JointTrainingContractError(f"training contract {split} crossed {field}")
    return {key: value[key] for key in required}


def _validate_checkpoint_binding(value: Any) -> dict[str, Any]:
    required = {
        "checkpoint_manifest_path",
        "checkpoint_manifest_sha256",
        "models",
        "model_order",
        "shared_asset_model_independent",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("training contract checkpoint binding is invalid")
    if (
        value.get("checkpoint_manifest_path")
        != DEFAULT_CHECKPOINT_MANIFEST.relative_to(ROOT).as_posix()
        or value.get("model_order") != list(MODELS)
        or value.get("shared_asset_model_independent") is not True
    ):
        raise JointTrainingContractError("training contract checkpoint binding changed")
    _sha256(value.get("checkpoint_manifest_sha256"), "training checkpoint manifest SHA256")
    models = value.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODELS):
        raise JointTrainingContractError("training contract has the wrong checkpoint models")
    expected_fields = {
        "checkpoint_dataset",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_size",
    }
    for model in MODELS:
        record = models[model]
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise JointTrainingContractError(f"training contract {model} checkpoint schema is invalid")
        request = CHECKPOINT_REQUESTS[model]
        if (
            record.get("checkpoint_dataset") != request["checkpoint_dataset"]
            or record.get("checkpoint_path") != request["checkpoint_path"]
        ):
            raise JointTrainingContractError(f"training contract {model} checkpoint changed")
        _sha256(record.get("checkpoint_sha256"), f"training contract {model} checkpoint SHA256")
        _positive_int(record.get("checkpoint_size"), f"training contract {model} checkpoint size")
    return dict(value)


def _validate_model_source_binding(value: Any, *, model: str) -> dict[str, Any]:
    required = {
        "repository_path",
        "repository_commit",
        "src_path",
        "src_tree_sha256",
        "src_file_count",
        "runtime_import_roots",
    }
    expected_root = MODEL_SOURCE_ROOTS[model]
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError(f"training contract {model} runtime-source schema is invalid")
    if (
        value.get("repository_path") != expected_root
        or value.get("src_path") != f"{expected_root}/src"
        or not isinstance(value.get("repository_commit"), str)
        or GIT_COMMIT.fullmatch(value["repository_commit"]) is None
    ):
        raise JointTrainingContractError(f"training contract {model} runtime-source changed")
    _sha256(value.get("src_tree_sha256"), f"training contract {model} source-tree SHA256")
    _positive_int(value.get("src_file_count"), f"training contract {model} source-file count")
    roots = value.get("runtime_import_roots")
    expected_paths = list(MODEL_RUNTIME_IMPORT_ROOTS[model])
    if not isinstance(roots, list) or len(roots) != len(expected_paths):
        raise JointTrainingContractError(
            f"training contract {model} runtime import roots are invalid"
        )
    if [item.get("path") if isinstance(item, Mapping) else None for item in roots] != expected_paths:
        raise JointTrainingContractError(
            f"training contract {model} runtime import roots changed"
        )
    for item in roots:
        if not isinstance(item, Mapping) or set(item) != {"path", "tree_sha256", "file_count"}:
            raise JointTrainingContractError(
                f"training contract {model} runtime import-root schema is invalid"
            )
        _sha256(item.get("tree_sha256"), f"training contract {model} runtime import SHA256")
        _positive_int(item.get("file_count"), f"training contract {model} runtime import count")
    if (
        roots[0]["path"] != value["src_path"]
        or roots[0]["tree_sha256"] != value["src_tree_sha256"]
        or roots[0]["file_count"] != value["src_file_count"]
    ):
        raise JointTrainingContractError(
            f"training contract {model} primary src root differs from runtime imports"
        )
    return dict(value)


def _validate_profile_interpreter_binding(value: Any, *, model: str) -> dict[str, str]:
    required = {"executable_sha256", "python_version"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError(
            f"training contract {model} profile-interpreter schema is invalid"
        )
    _sha256(value.get("executable_sha256"), f"training contract {model} interpreter SHA256")
    version = value.get("python_version")
    if not isinstance(version, str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise JointTrainingContractError(
            f"training contract {model} interpreter version is invalid"
        )
    return {"executable_sha256": str(value["executable_sha256"]), "python_version": version}


def _validate_model_config_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(MODELS):
        raise JointTrainingContractError("training contract model-config binding is invalid")
    required = {
        "entrypoint_path",
        "entrypoint_sha256",
        "source_files",
        "source_manifest_sha256",
        "hydra_overrides",
        "environment_profile",
        "profile_interpreter",
        "runtime_source",
        "resolved_config_sha256",
    }
    for model in MODELS:
        record = value[model]
        request = MODEL_CONFIG_REQUESTS[model]
        if not isinstance(record, Mapping) or set(record) != required:
            raise JointTrainingContractError(f"training contract {model} config schema is invalid")
        if (
            record.get("entrypoint_path") != request["entrypoint_path"]
            or record.get("hydra_overrides") != request["hydra_overrides"]
            or record.get("environment_profile") != MODEL_ENVIRONMENT_PROFILES[model]
        ):
            raise JointTrainingContractError(f"training contract {model} config selection changed")
        _sha256(record.get("entrypoint_sha256"), f"training contract {model} entrypoint SHA256")
        _sha256(record.get("source_manifest_sha256"), f"training contract {model} config manifest SHA256")
        files = record.get("source_files")
        if not isinstance(files, list) or len(files) != len(request["source_paths"]):
            raise JointTrainingContractError(f"training contract {model} config sources are invalid")
        expected_paths = request["source_paths"]
        if [item.get("path") if isinstance(item, Mapping) else None for item in files] != expected_paths:
            raise JointTrainingContractError(f"training contract {model} config source order changed")
        for item in files:
            if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
                raise JointTrainingContractError(f"training contract {model} config source schema is invalid")
            _sha256(item.get("sha256"), f"training contract {model} config source SHA256")
        if (
            record["entrypoint_sha256"]
            != next(item["sha256"] for item in files if item["path"] == request["entrypoint_path"])
            or record["source_manifest_sha256"] != canonical_sha256(files)
        ):
            raise JointTrainingContractError(f"training contract {model} config hashes are invalid")
        _validate_model_source_binding(record.get("runtime_source"), model=model)
        _validate_profile_interpreter_binding(record.get("profile_interpreter"), model=model)
        _sha256(
            record.get("resolved_config_sha256"),
            f"training contract {model} resolved-config SHA256",
        )
    return {model: dict(value[model]) for model in MODELS}


def _validate_preprocessing(value: Any) -> dict[str, Any]:
    required = {*PREPROCESSING_CONTRACT, "crop_shim_sha256", "patch_shim_sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("training contract preprocessing schema is invalid")
    for field, expected in PREPROCESSING_CONTRACT.items():
        if value.get(field) != expected:
            raise JointTrainingContractError(f"training contract preprocessing {field} changed")
    crop_shim_sha256 = value.get("crop_shim_sha256")
    patch_shim_sha256 = value.get("patch_shim_sha256")
    if (
        not isinstance(crop_shim_sha256, Mapping)
        or set(crop_shim_sha256) != set(MODELS)
        or not isinstance(patch_shim_sha256, Mapping)
        or set(patch_shim_sha256) != set(MODELS)
    ):
        raise JointTrainingContractError("training contract preprocessing shim set is invalid")
    for model in MODELS:
        _sha256(crop_shim_sha256.get(model), f"training contract {model} crop shim SHA256")
        _sha256(patch_shim_sha256.get(model), f"training contract {model} patch shim SHA256")
    return dict(value)


def _validate_saes_routing(value: Any) -> dict[str, Any]:
    required = {
        "execution_identity",
        "execution_route_sha256",
        "parameters",
        "per_model",
        "mechanism_config_path",
        "mechanism_config_sha256",
        "implementation_path",
        "implementation_sha256",
        "routing_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("training contract SAES routing schema is invalid")
    try:
        execution_identity = validate_saes_execution_identity(
            value.get("execution_identity")
        )
    except ValueError as exc:
        raise JointTrainingContractError("training contract SAES execution route changed") from exc
    if (
        value.get("execution_route_sha256") != execution_identity["route_sha256"]
        or value.get("parameters") != _routing_parameters_for_identity(execution_identity)
        or value.get("per_model") != SAES_MODEL_ROUTING
        or value.get("mechanism_config_path") != "artifact/mechanism_config.json"
        or value.get("implementation_path") != "saes/progressive_saes.py"
    ):
        raise JointTrainingContractError("training contract SAES routing changed")
    for field in ("mechanism_config_sha256", "implementation_sha256"):
        _sha256(value.get(field), f"training contract SAES {field}")
    payload = {key: item for key, item in value.items() if key != "routing_sha256"}
    if value.get("routing_sha256") != canonical_sha256(payload):
        raise JointTrainingContractError("training contract SAES routing SHA256 is invalid")
    return dict(value)


def _validate_implementation_binding(value: Any) -> dict[str, Any]:
    required = {"files", "implementation_tree_sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("training implementation binding schema is invalid")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(IMPLEMENTATION_PATHS):
        raise JointTrainingContractError("training implementation binding has the wrong file count")
    paths = [item.get("path") if isinstance(item, Mapping) else None for item in files]
    if paths != list(IMPLEMENTATION_PATHS):
        raise JointTrainingContractError("training implementation binding file order changed")
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise JointTrainingContractError("training implementation binding file schema is invalid")
        _sha256(item.get("sha256"), "training implementation SHA256")
    if value.get("implementation_tree_sha256") != canonical_sha256(files):
        raise JointTrainingContractError("training implementation binding SHA256 is invalid")
    return {"files": [dict(item) for item in files], "implementation_tree_sha256": value["implementation_tree_sha256"]}


def _validate_shared_asset(value: Any) -> dict[str, Any]:
    if value != _shared_asset_contract():
        raise JointTrainingContractError("training contract shared asset definition changed")
    return dict(value)


def _validate_result_records(value: Any) -> dict[str, str]:
    if value != RESULT_RECORD_PATHS:
        raise JointTrainingContractError("training contract result-record paths changed")
    return dict(value)


def _validate_runtime_evidence_records(value: Any) -> dict[str, dict[str, str]]:
    """Require one canonical source-bound runtime proof for every model/split."""
    if value != RUNTIME_EVIDENCE_RECORDS:
        raise JointTrainingContractError("training contract runtime-evidence paths changed")
    return {
        split: {model: str(path) for model, path in records.items()}
        for split, records in RUNTIME_EVIDENCE_RECORDS.items()
    }


def _validate_candidate_verification_policy(value: Any) -> dict[str, Any]:
    if value != CANDIDATE_VERIFICATION_POLICY:
        raise JointTrainingContractError("training contract candidate-verification policy changed")
    return dict(value)


def validate_training_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the static immutable contract without opening author-side data."""
    if not isinstance(contract, Mapping) or set(contract) != _CONTRACT_FIELDS:
        raise JointTrainingContractError("training contract has unexpected fields")
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("status") != CONTRACT_STATUS
        or contract.get("contract_id") != CONTRACT_ID
        or contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("author_side_prerequisite") is not True
        or contract.get("paper_result_eligible") is not False
        or contract.get("mechanism_config_write_allowed") is not False
        or contract.get("dl3dv_quality_gate_authorized") is not False
    ):
        raise JointTrainingContractError("training contract violates its author-side scope")
    source_plan = _validate_source_plan_binding(contract.get("source_plan"))
    context_inputs = _validate_context_input_binding(contract.get("context_only_inputs"))
    checkpoint_binding = _validate_checkpoint_binding(contract.get("checkpoint_binding"))
    _validate_model_config_binding(contract.get("model_config_binding"))
    _validate_preprocessing(contract.get("preprocessing"))
    saes_routing = _validate_saes_routing(contract.get("saes_routing"))
    _validate_implementation_binding(contract.get("implementation_binding"))
    _validate_shared_asset(contract.get("shared_asset"))
    _validate_result_records(contract.get("result_records"))
    _validate_runtime_evidence_records(contract.get("runtime_evidence_records"))
    _validate_candidate_verification_policy(contract.get("candidate_verification_policy"))
    if contract.get("optimization_recipe") != OPTIMIZATION_RECIPE:
        raise JointTrainingContractError("training contract optimizer schedule changed")
    if contract.get("teacher_objective") != TEACHER_OBJECTIVE:
        raise JointTrainingContractError("training contract teacher objective changed")
    if contract.get("teacher_fidelity_gates") != TEACHER_FIDELITY_GATES:
        raise JointTrainingContractError("training contract teacher-fidelity gates changed")
    if contract.get("holdout_policy") != HOLDOUT_POLICY:
        raise JointTrainingContractError("training contract holdout policy changed")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if contract.get("contract_sha256") != canonical_sha256(payload):
        raise JointTrainingContractError("training contract SHA256 is invalid")
    return {
        "contract_sha256": str(contract["contract_sha256"]),
        "plan_sha256": source_plan["plan_sha256"],
        "context_outer_tree_sha256": context_inputs["outer_tree_sha256"],
        "checkpoint_models": list(checkpoint_binding["model_order"]),
        "saes_execution_route_sha256": saes_routing["execution_route_sha256"],
        "status": CONTRACT_STATUS,
        "paper_result_eligible": False,
    }


def _canonical_live_contract_path(repository_root: Path) -> Path:
    try:
        relative = DEFAULT_TRAINING_CONTRACT_PATH.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise JointTrainingContractError(
            "canonical training contract path is outside the canonical repository"
        ) from exc
    return _safe_repository_path(
        repository_root, relative.as_posix(), "canonical joint training contract"
    )


def _require_canonical_live_contract(
    contract: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Reject caller-supplied contract objects that differ from the frozen file."""

    canonical_path = _canonical_live_contract_path(repository_root)
    if not canonical_path.is_file():
        raise JointTrainingContractError("canonical joint training contract is missing")
    durable = _read_json(canonical_path, "canonical joint training contract")
    if durable != contract:
        raise JointTrainingContractError(
            "caller-supplied training contract differs from the canonical frozen file"
        )
    return durable


def validate_live_training_contract(
    contract: Mapping[str, Any], *, repository_root: Path = ROOT
) -> dict[str, Any]:
    """Rehash plan, sidecars, and checkpoints before an optimizer is launched."""
    repository_root = Path(repository_root).resolve()
    if repository_root != ROOT:
        raise JointTrainingContractError(
            "joint training contract live validation requires the canonical repository root"
        )
    contract = _require_canonical_live_contract(contract, repository_root=repository_root)
    identity = validate_training_contract(contract)
    source_plan = contract["source_plan"]
    plan_path = _safe_repository_path(repository_root, source_plan["path"], "ACID joint plan")
    if not plan_path.is_file():
        raise JointTrainingContractError("ACID joint plan is missing")
    require_fixed_file_sha256(
        plan_path, source_plan["file_sha256"], label="ACID joint plan"
    )
    plan = _read_json(plan_path, "ACID joint plan")
    try:
        validate_plan(plan)
    except JointCalibrationContractError as exc:
        raise JointTrainingContractError("ACID joint plan no longer validates") from exc
    if (
        plan.get("plan_sha256") != source_plan["plan_sha256"]
        or plan.get("protocol_id") != source_plan["protocol_id"]
    ):
        raise JointTrainingContractError("ACID joint plan identity differs from training contract")
    materialization_root = _safe_repository_path(
        repository_root,
        contract["context_only_inputs"]["materialization_root"],
        "context-only materialization",
    )
    actual_context = _context_input_binding(
        materialization_root, plan=plan, repository_root=repository_root
    )
    if actual_context != contract["context_only_inputs"]:
        raise JointTrainingContractError("context-only input identity differs from training contract")
    checkpoint_binding = contract["checkpoint_binding"]
    manifest_path = _safe_repository_path(
        repository_root,
        checkpoint_binding["checkpoint_manifest_path"],
        "checkpoint manifest",
    )
    require_fixed_file_sha256(
        manifest_path,
        checkpoint_binding["checkpoint_manifest_sha256"],
        label="checkpoint manifest",
    )
    actual_checkpoints = _checkpoint_records(manifest_path, repository_root=repository_root)
    if actual_checkpoints != checkpoint_binding["models"]:
        raise JointTrainingContractError("checkpoint identity differs from training contract")
    if _model_config_records(repository_root) != contract["model_config_binding"]:
        raise JointTrainingContractError("model configuration identity differs from training contract")
    if _preprocessing_binding(repository_root) != contract["preprocessing"]:
        raise JointTrainingContractError("preprocessing identity differs from training contract")
    if _saes_routing_binding(repository_root) != contract["saes_routing"]:
        raise JointTrainingContractError("SAES routing identity differs from training contract")
    if _implementation_binding(repository_root) != contract["implementation_binding"]:
        raise JointTrainingContractError("training implementation identity differs from training contract")
    return {
        **identity,
        "live_plan_sha256": str(plan["plan_sha256"]),
        "live_context_outer_tree_sha256": actual_context["outer_tree_sha256"],
        "status": "PASS_PREOPTIMIZATION_REHASH",
        "paper_result_eligible": False,
    }


def validate_runtime_model_binding(
    contract: Mapping[str, Any],
    *,
    model: str,
    resolved_config_sha256: str,
    repository_root: Path = ROOT,
    require_loaded_src: bool = False,
) -> dict[str, Any]:
    """Reject a worker whose loaded source or typed config drifted after preflight."""

    repository_root = Path(repository_root).resolve()
    if repository_root != ROOT or model not in MODELS or not isinstance(require_loaded_src, bool):
        raise JointTrainingContractError("runtime model binding requires the canonical repository and model")
    contract = _require_canonical_live_contract(contract, repository_root=repository_root)
    validate_training_contract(contract)
    expected = contract["model_config_binding"][model]
    actual_source = _model_source_binding(repository_root, model)
    if actual_source != expected["runtime_source"]:
        raise JointTrainingContractError(f"{model} runtime source identity differs from training contract")
    actual_files = _model_config_source_files(repository_root, model)
    actual_config_source_sha256 = canonical_sha256(actual_files)
    if (
        actual_files != expected["source_files"]
        or actual_config_source_sha256 != expected["source_manifest_sha256"]
    ):
        raise JointTrainingContractError(
            f"{model} runtime config sources differ from training contract"
        )
    profile = str(expected["environment_profile"])
    actual_profile_interpreter = _runtime_profile_interpreter_binding(profile)
    if actual_profile_interpreter != expected["profile_interpreter"]:
        raise JointTrainingContractError(
            f"{model} runtime profile interpreter differs from training contract"
        )
    if require_loaded_src:
        expected_src = _safe_repository_path(
            repository_root, expected["runtime_source"]["src_path"], f"{model} runtime src"
        )
        imported_origins: list[Path] = []
        for module_name, module in tuple(sys.modules.items()):
            if module_name != "src" and not module_name.startswith("src."):
                continue
            origin = getattr(module, "__file__", None)
            if isinstance(origin, str) and origin:
                resolved_origin = Path(origin).resolve()
                if expected_src not in resolved_origin.parents:
                    raise JointTrainingContractError(
                        f"{model} imported src module escaped the declared model source root"
                    )
                imported_origins.append(resolved_origin)
                continue
            namespace_paths = getattr(module, "__path__", None)
            if isinstance(namespace_paths, (str, bytes)):
                namespace_paths = None
            try:
                resolved_namespace_paths = tuple(namespace_paths)
            except TypeError:
                resolved_namespace_paths = ()
            if not resolved_namespace_paths:
                raise JointTrainingContractError(
                    f"{model} imported src module has no constrained file or namespace path: {module_name}"
                )
            for namespace_path in resolved_namespace_paths:
                if not isinstance(namespace_path, str) or not namespace_path:
                    raise JointTrainingContractError(
                        f"{model} imported src namespace path is invalid: {module_name}"
                    )
                resolved_namespace_path = Path(namespace_path).resolve()
                if (
                    resolved_namespace_path != expected_src
                    and expected_src not in resolved_namespace_path.parents
                ):
                    raise JointTrainingContractError(
                        f"{model} imported src module escaped the declared model source root"
                    )
                imported_origins.append(resolved_namespace_path)
        if not imported_origins:
            raise JointTrainingContractError(
                f"{model} worker loaded no declared model src modules"
            )
        if model == "depthsplat":
            dino_root = _safe_repository_path(
                repository_root,
                expected["runtime_source"]["runtime_import_roots"][1]["path"],
                "DepthSplat pinned DINOv2 source",
            )
            dino_origins: list[Path] = []
            for module_name, module in tuple(sys.modules.items()):
                if module_name != "dinov2" and not module_name.startswith("dinov2."):
                    continue
                origin = getattr(module, "__file__", None)
                if not isinstance(origin, str) or not origin:
                    raise JointTrainingContractError(
                        f"DepthSplat DINOv2 module has no file origin: {module_name}"
                    )
                resolved_origin = Path(origin).resolve()
                if dino_root not in resolved_origin.parents:
                    raise JointTrainingContractError(
                        "DepthSplat imported DINOv2 module escaped the pinned runtime source root"
                    )
                dino_origins.append(resolved_origin)
            if not dino_origins:
                raise JointTrainingContractError(
                    "DepthSplat worker loaded no pinned DINOv2 modules"
                )
    digest = _sha256(
        resolved_config_sha256,
        f"{model} runtime resolved-config SHA256",
    )
    if digest != expected["resolved_config_sha256"]:
        raise JointTrainingContractError(
            f"{model} runtime resolved configuration differs from training contract"
        )
    return {
        "runtime_source": actual_source,
        "config_source_sha256": actual_config_source_sha256,
        "environment_profile": profile,
        "profile_interpreter": actual_profile_interpreter,
        "resolved_config_sha256": digest,
    }


def _validate_asset_identity(value: Any) -> dict[str, Any]:
    required = {
        "asset_path",
        "schema_version",
        "kind",
        "sha256",
        "state_sha256",
        "byte_count",
        "descriptor_dim",
        "bottleneck_dim",
        "joint_output_dim",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("teacher-fidelity asset identity is invalid")
    if value.get("asset_path") != DEFAULT_ASSET_PATH:
        raise JointTrainingContractError("teacher-fidelity asset path changed")
    for field in ("sha256", "state_sha256"):
        _sha256(value.get(field), f"teacher-fidelity asset {field}")
    _positive_int(value.get("byte_count"), "teacher-fidelity asset byte count")
    architecture = CALIBRATOR_ARCHITECTURE
    for field in ("schema_version", "kind", "descriptor_dim", "bottleneck_dim", "joint_output_dim"):
        if value.get(field) != architecture[field]:
            raise JointTrainingContractError(f"teacher-fidelity asset {field} changed")
    return dict(value)


def candidate_asset_path_for_contract(contract: Mapping[str, Any]) -> str:
    """Return the only permitted staged-asset path for a frozen contract."""

    validate_training_contract(contract)
    return f"{contract['shared_asset']['asset_path']}.candidate"


def candidate_provenance_path_for_contract(contract: Mapping[str, Any]) -> str:
    """Return the immutable provenance sidecar paired with a staged asset."""

    return f"{candidate_asset_path_for_contract(contract)}.provenance.json"


def candidate_verification_path_for_contract(contract: Mapping[str, Any]) -> str:
    """Return the canonical external-verifier record path for a candidate."""

    return f"{candidate_asset_path_for_contract(contract)}.verification.json"


def _validate_candidate_cache_manifests(
    value: Any, *, contract: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(MODELS):
        raise JointTrainingContractError("candidate provenance has an invalid cache-manifest set")
    root = str(contract["teacher_objective"]["teacher_cache_root"])
    result: dict[str, dict[str, str]] = {}
    for model in MODELS:
        item = value[model]
        required = {"path", "sha256"}
        expected_path = f"{root}/{model}/{TRAIN_SPLIT}/manifest.json"
        if (
            not isinstance(item, Mapping)
            or set(item) != required
            or item.get("path") != expected_path
        ):
            raise JointTrainingContractError(
                f"candidate provenance {model} cache-manifest binding is invalid"
            )
        result[model] = {
            "path": expected_path,
            "sha256": _sha256(item.get("sha256"), f"candidate provenance {model} cache SHA256"),
        }
    return result


def validate_candidate_provenance(
    value: Any,
    *,
    contract: Mapping[str, Any],
    candidate_asset: Mapping[str, Any],
    cache_manifests: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a local, non-cryptographic binding for one staged candidate."""

    validate_training_contract(contract)
    asset = _validate_asset_identity(candidate_asset)
    required = {
        "schema_version",
        "kind",
        "contract_sha256",
        "plan_sha256",
        "candidate_asset_path",
        "candidate_asset_sha256",
        "candidate_asset_state_sha256",
        "candidate_asset_byte_count",
        "cache_manifests",
        "optimization_recipe_sha256",
        "seed",
        "total_updates",
        "local_hash_chain_is_not_cryptographic_proof",
        "verification_status",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("candidate provenance has an invalid schema")
    if (
        value.get("schema_version") != CANDIDATE_PROVENANCE_SCHEMA_VERSION
        or value.get("kind") != CANDIDATE_PROVENANCE_KIND
        or value.get("contract_sha256") != contract["contract_sha256"]
        or value.get("plan_sha256") != contract["source_plan"]["plan_sha256"]
        or value.get("candidate_asset_path") != candidate_asset_path_for_contract(contract)
        or value.get("candidate_asset_sha256") != asset["sha256"]
        or value.get("candidate_asset_state_sha256") != asset["state_sha256"]
        or value.get("candidate_asset_byte_count") != asset["byte_count"]
        or value.get("optimization_recipe_sha256")
        != canonical_sha256(contract["optimization_recipe"])
        or value.get("seed") != contract["optimization_recipe"]["seed"]
        or value.get("total_updates") != contract["optimization_recipe"]["total_updates"]
        or value.get("local_hash_chain_is_not_cryptographic_proof") is not True
        or value.get("verification_status") != "BLOCKED_NO_VERIFIED_EXECUTION"
    ):
        raise JointTrainingContractError("candidate provenance differs from the frozen execution")
    manifests = _validate_candidate_cache_manifests(
        value.get("cache_manifests"), contract=contract
    )
    expected_manifests = _validate_candidate_cache_manifests(
        cache_manifests, contract=contract
    )
    if manifests != expected_manifests:
        raise JointTrainingContractError("candidate provenance cache manifests changed")
    return {
        "candidate_asset_path": str(value["candidate_asset_path"]),
        "candidate_asset_sha256": str(value["candidate_asset_sha256"]),
        "candidate_asset_state_sha256": str(value["candidate_asset_state_sha256"]),
        "cache_manifests": manifests,
    }


def validate_candidate_verification(
    value: Any,
    *,
    contract: Mapping[str, Any],
    provenance_path: str,
    provenance_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject candidate verifier records under the deliberately blocked v5 policy.

    A verifier identity hash is not itself a signature or deterministic replay
    proof. v5 therefore accepts no verifier mode at all. A future protocol
    needs a new schema and implementation that actually verifies an external
    signature or independently reruns the pinned optimizer before it may
    accept either mode.
    """

    validate_training_contract(contract)
    required = {
        "schema_version",
        "kind",
        "verification_mode",
        "candidate_provenance_path",
        "candidate_provenance_sha256",
        "candidate_asset_sha256",
        "contract_sha256",
        "cache_manifests",
        "verifier_identity",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("candidate verification has an invalid schema")
    mode = value.get("verification_mode")
    if mode not in {"deterministic_replay_verifier", "external_trusted_attestation"}:
        raise JointTrainingContractError("candidate verification mode is unsupported")
    _sha256(provenance_sha256, "candidate provenance file SHA256")
    _sha256(value.get("candidate_provenance_sha256"), "candidate verification provenance SHA256")
    _sha256(value.get("candidate_asset_sha256"), "candidate verification asset SHA256")
    if (
        value.get("schema_version") != CANDIDATE_VERIFICATION_SCHEMA_VERSION
        or value.get("kind") != CANDIDATE_VERIFICATION_KIND
        or value.get("candidate_provenance_path") != provenance_path
        or value.get("candidate_provenance_sha256") != provenance_sha256
        or value.get("candidate_asset_sha256") != provenance["candidate_asset_sha256"]
        or value.get("contract_sha256") != contract["contract_sha256"]
        or _validate_candidate_cache_manifests(value.get("cache_manifests"), contract=contract)
        != provenance["cache_manifests"]
    ):
        raise JointTrainingContractError("candidate verification differs from provenance")
    policy = contract["candidate_verification_policy"]
    if (
        policy["registered_deterministic_replay_verifier"] is not None
        or policy["registered_external_attestation_public_key_sha256"] is not None
    ):
        raise JointTrainingContractError(
            "v5 candidate verification policy cannot register an authority; use a new verifier schema"
        )
    raise JointTrainingContractError(
        "v5 candidate verification is BLOCKED_NO_VERIFIED_EXECUTION; identity hashes are not proof"
    )


def _validate_result_access_audit(
    value: Any, *, contract: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    """Enforce the plan's isolation audit without reopening a raw ACID record."""
    required = {
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
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError("teacher-fidelity access audit has unexpected fields")
    expected_stage = "train" if stage == TRAIN_SPLIT else "holdout"
    if (
        value.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or value.get("kind") != ACCESS_AUDIT_KIND
        or value.get("plan_sha256") != contract["source_plan"]["plan_sha256"]
        or value.get("stage") != expected_stage
        or value.get("runtime_teacher_path") is not None
    ):
        raise JointTrainingContractError("teacher-fidelity access audit identity is invalid")
    for field in (
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
        "expected_results_accessed",
        "evaluation_scene_accessed",
        "descriptor_model_id_accessed",
        "descriptor_dataset_id_accessed",
        "teacher_files_runtime_accessible",
        "rerank_executed",
        "partition_reshuffled",
    ):
        if value.get(field) is not False:
            raise JointTrainingContractError(f"teacher-fidelity access audit illegally set {field}")
    for field in ("teacher_files_opened", "optimizer_executed", "asset_updated"):
        if not isinstance(value.get(field), bool):
            raise JointTrainingContractError(f"teacher-fidelity access audit has invalid {field}")
    if value["teacher_files_opened"] is not True:
        raise JointTrainingContractError("teacher-fidelity record did not use the registered offline teacher")
    if stage == TRAIN_SPLIT:
        if value["optimizer_executed"] is not True or value["asset_updated"] is not True:
            raise JointTrainingContractError("training result did not execute the registered optimizer")
    elif value["optimizer_executed"] is not False or value["asset_updated"] is not False:
        raise JointTrainingContractError("holdout attempted to optimize or update the asset")
    return dict(value)


def _validate_metrics(value: Any, *, stage: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_METRIC_COMPONENTS):
        raise JointTrainingContractError("teacher-fidelity metrics have the wrong components")
    floor = float(TEACHER_OBJECTIVE["loss"]["normalizer_floor"])
    relative: dict[str, float] = {}
    for component in _METRIC_COMPONENTS:
        record = value[component]
        if not isinstance(record, Mapping) or set(record) != {"identity_mse", "calibrated_mse"}:
            raise JointTrainingContractError(f"teacher-fidelity {component} metric is invalid")
        identity_mse = _finite_nonnegative(record["identity_mse"], f"{component} identity MSE")
        calibrated_mse = _finite_nonnegative(record["calibrated_mse"], f"{component} calibrated MSE")
        ratio = calibrated_mse / max(identity_mse, floor)
        if ratio > float(TEACHER_FIDELITY_GATES["all_components_nonworsening_max_relative_mse"]):
            raise JointTrainingContractError(f"teacher-fidelity {component} regressed from identity")
        relative[component] = ratio
    joint = sum(relative.values()) / len(relative)
    limit = float(
        TEACHER_FIDELITY_GATES[
            "train_joint_relative_mse_max" if stage == TRAIN_SPLIT else "holdout_joint_relative_mse_max"
        ]
    )
    if joint > limit:
        raise JointTrainingContractError("teacher-fidelity joint metric did not meet the frozen gate")
    return {**relative, "joint": joint}


def _expected_input_identity(contract: Mapping[str, Any], split: str) -> dict[str, Any]:
    sidecar = contract["context_only_inputs"]["splits"][split]
    return {
        "tree_sha256": sidecar["tree_sha256"],
        "manifest_sha256": sidecar["manifest_sha256"],
        "input_provenance_sha256": sidecar["input_provenance_sha256"],
        "selection_sha256": sidecar["selection_sha256"],
        "scene_count": sidecar["scene_count"],
    }


def _validate_runtime_evidence_identity(
    value: Any, *, contract: Mapping[str, Any], model: str, stage: str
) -> dict[str, str]:
    """Bind a fidelity record to one immutable, model/split runtime proof."""
    required = {
        "path",
        "sha256",
        "selected_head_sha256",
        "ledger_sha256",
        "asset_sha256",
        "asset_state_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError(f"teacher-fidelity {model} runtime evidence is invalid")
    expected_path = contract["runtime_evidence_records"][stage][model]
    if value.get("path") != expected_path:
        raise JointTrainingContractError(f"teacher-fidelity {model} runtime evidence path changed")
    for field in required - {"path"}:
        _sha256(value.get(field), f"teacher-fidelity {model} runtime evidence {field}")
    return {key: str(value[key]) for key in required}


def _validate_model_result(
    value: Any, *, contract: Mapping[str, Any], model: str, stage: str
) -> dict[str, Any]:
    required = {
        "checkpoint_sha256",
        "config_source_sha256",
        "runtime_source",
        "environment_profile",
        "profile_interpreter",
        "resolved_config_sha256",
        "preprocessing_id",
        "prepared_patch_size",
        "saes_routing_sha256",
        "saes_execution_route_sha256",
        "input_identity",
        "metrics",
        "controls",
        "runtime_evidence",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise JointTrainingContractError(f"teacher-fidelity {model} result is invalid")
    expected_checkpoint = contract["checkpoint_binding"]["models"][model]["checkpoint_sha256"]
    if value.get("checkpoint_sha256") != expected_checkpoint:
        raise JointTrainingContractError(f"teacher-fidelity {model} checkpoint changed")
    expected_config = contract["model_config_binding"][model]
    if value.get("config_source_sha256") != expected_config["source_manifest_sha256"]:
        raise JointTrainingContractError(f"teacher-fidelity {model} config sources changed")
    if value.get("runtime_source") != expected_config["runtime_source"]:
        raise JointTrainingContractError(f"teacher-fidelity {model} runtime source changed")
    if value.get("environment_profile") != expected_config["environment_profile"]:
        raise JointTrainingContractError(f"teacher-fidelity {model} environment profile changed")
    if value.get("profile_interpreter") != expected_config["profile_interpreter"]:
        raise JointTrainingContractError(f"teacher-fidelity {model} profile interpreter changed")
    if value.get("resolved_config_sha256") != expected_config["resolved_config_sha256"]:
        raise JointTrainingContractError(
            f"teacher-fidelity {model} resolved config differs from training contract"
        )
    if value.get("preprocessing_id") != contract["preprocessing"]["identifier"]:
        raise JointTrainingContractError(f"teacher-fidelity {model} preprocessing changed")
    if value.get("prepared_patch_size") != contract["preprocessing"]["per_model_patch_size"][model]:
        raise JointTrainingContractError(f"teacher-fidelity {model} patch preprocessing changed")
    if value.get("saes_routing_sha256") != contract["saes_routing"]["routing_sha256"]:
        raise JointTrainingContractError(f"teacher-fidelity {model} SAES routing changed")
    if (
        value.get("saes_execution_route_sha256")
        != contract["saes_routing"]["execution_route_sha256"]
    ):
        raise JointTrainingContractError(
            f"teacher-fidelity {model} SAES execution route changed"
        )
    input_identity = value.get("input_identity")
    if input_identity != _expected_input_identity(contract, stage):
        raise JointTrainingContractError(f"teacher-fidelity {model} input identity changed")
    metrics = _validate_metrics(value.get("metrics"), stage=stage)
    runtime_evidence = _validate_runtime_evidence_identity(
        value.get("runtime_evidence"), contract=contract, model=model, stage=stage
    )
    controls = value.get("controls")
    if not isinstance(controls, Mapping) or set(controls) != set(RUNTIME_PRECONDITIONS):
        raise JointTrainingContractError(f"teacher-fidelity {model} controls are invalid")
    for field, required_value in RUNTIME_PRECONDITIONS.items():
        if controls.get(field) is not required_value:
            raise JointTrainingContractError(f"teacher-fidelity {model} failed {field}")
    return {
        "metrics": metrics,
        "resolved_config_sha256": str(value["resolved_config_sha256"]),
        "runtime_evidence": runtime_evidence,
    }


def _live_candidate_cache_manifest_identities(
    contract: Mapping[str, Any], *, repository_root: Path
) -> dict[str, dict[str, str]]:
    """Rehash the canonical train-cache manifests named by candidate provenance."""

    root = str(contract["teacher_objective"]["teacher_cache_root"])
    identities: dict[str, dict[str, str]] = {}
    for model in MODELS:
        relative = f"{root}/{model}/{TRAIN_SPLIT}/manifest.json"
        path = _safe_repository_path(
            repository_root, relative, f"{model} candidate cache manifest"
        )
        if not path.is_file():
            raise JointTrainingContractError(
                f"{model} candidate cache manifest is unavailable"
            )
        identities[model] = {"path": relative, "sha256": sha256_file(path)}
    return identities


def _validate_result_candidate_execution(
    value: Any,
    *,
    stage: str,
    asset: Mapping[str, Any],
    contract: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any] | None:
    if stage == HOLDOUT_SPLIT:
        if value is not None:
            raise JointTrainingContractError(
                "holdout result may not claim staged-candidate verification"
            )
        return None
    if not isinstance(value, Mapping) or set(value) != {"provenance", "verification"}:
        raise JointTrainingContractError("training result has no candidate-execution verification")
    provenance = value.get("provenance")
    verification = value.get("verification")
    required_provenance = {
        "path",
        "sha256",
        "candidate_asset_sha256",
        "candidate_asset_state_sha256",
        "cache_manifests",
        "local_hash_chain_is_not_cryptographic_proof",
    }
    required_verification = {
        "path",
        "sha256",
        "verification_mode",
        "candidate_provenance_path",
        "candidate_provenance_sha256",
        "candidate_asset_sha256",
    }
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != required_provenance
        or not isinstance(verification, Mapping)
        or set(verification) != required_verification
        or provenance.get("local_hash_chain_is_not_cryptographic_proof") is not True
        or verification.get("verification_mode")
        not in {"deterministic_replay_verifier", "external_trusted_attestation"}
    ):
        raise JointTrainingContractError("training result candidate-execution verification is invalid")
    for field in (
        "sha256",
        "candidate_asset_sha256",
        "candidate_asset_state_sha256",
    ):
        _sha256(provenance.get(field), f"training result candidate provenance {field}")
    for field in (
        "sha256",
        "candidate_provenance_sha256",
        "candidate_asset_sha256",
    ):
        _sha256(verification.get(field), f"training result candidate verification {field}")
    if (
        not isinstance(provenance.get("path"), str)
        or not isinstance(verification.get("path"), str)
        or verification["candidate_provenance_path"] != provenance["path"]
        or verification["candidate_provenance_sha256"] != provenance["sha256"]
        or provenance["candidate_asset_sha256"] != asset["sha256"]
        or provenance["candidate_asset_state_sha256"] != asset["state_sha256"]
        or verification["candidate_asset_sha256"] != asset["sha256"]
    ):
        raise JointTrainingContractError("training result candidate verification is not asset-bound")
    if not isinstance(provenance.get("cache_manifests"), Mapping):
        raise JointTrainingContractError("training result candidate cache manifests are invalid")
    expected_provenance_path = candidate_provenance_path_for_contract(contract)
    expected_verification_path = candidate_verification_path_for_contract(contract)
    if (
        provenance["path"] != expected_provenance_path
        or verification["path"] != expected_verification_path
    ):
        raise JointTrainingContractError(
            "training result candidate sidecar path differs from the frozen contract"
        )
    provenance_path = _safe_repository_path(
        repository_root, expected_provenance_path, "candidate provenance"
    )
    verification_path = _safe_repository_path(
        repository_root, expected_verification_path, "candidate verification"
    )
    try:
        require_fixed_file_sha256(
            provenance_path, provenance["sha256"], label="candidate provenance"
        )
        require_fixed_file_sha256(
            verification_path, verification["sha256"], label="candidate verification"
        )
    except (OSError, ValueError) as exc:
        raise JointTrainingContractError(
            "training result candidate provenance or verification sidecar is unavailable"
        ) from exc
    cache_manifests = _live_candidate_cache_manifest_identities(
        contract, repository_root=repository_root
    )
    sidecar_provenance = validate_candidate_provenance(
        _read_json(provenance_path, "candidate provenance"),
        contract=contract,
        candidate_asset=asset,
        cache_manifests=cache_manifests,
    )
    if (
        sidecar_provenance["candidate_asset_sha256"] != provenance["candidate_asset_sha256"]
        or sidecar_provenance["candidate_asset_state_sha256"]
        != provenance["candidate_asset_state_sha256"]
        or sidecar_provenance["cache_manifests"] != provenance["cache_manifests"]
    ):
        raise JointTrainingContractError(
            "training result candidate provenance differs from its canonical sidecar"
        )
    # v5 always raises here: an identity hash is not a verifier. Keeping this
    # call after both sidecar rehashes prevents a hand-written inline record
    # from bypassing the canonical audit boundary.
    validate_candidate_verification(
        _read_json(verification_path, "candidate verification"),
        contract=contract,
        provenance_path=expected_provenance_path,
        provenance_sha256=provenance["sha256"],
        provenance=sidecar_provenance,
    )
    return {
        "provenance": dict(provenance),
        "verification": dict(verification),
    }


def validate_teacher_fidelity_result(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    frozen_train_result: Mapping[str, Any] | None = None,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Validate a train/holdout teacher-fidelity record under the frozen gates.

    A holdout record must be submitted with its previously validated training
    record.  The check makes the asset identity equal, so no holdout fitting,
    reranking, or replacement asset can be smuggled into the result.
    """
    validate_training_contract(contract)
    repository_root = Path(repository_root).resolve()
    required = {
        "schema_version",
        "kind",
        "contract_sha256",
        "plan_sha256",
        "stage",
        "frozen_train_result_sha256",
        "asset",
        "candidate_execution_verification",
        "models",
        "access_audit",
    }
    if not isinstance(result, Mapping) or set(result) != required:
        raise JointTrainingContractError("teacher-fidelity result has unexpected fields")
    stage = result.get("stage")
    if stage not in {TRAIN_SPLIT, HOLDOUT_SPLIT}:
        raise JointTrainingContractError("teacher-fidelity result has an invalid stage")
    if (
        result.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or result.get("kind") != RESULT_KIND
        or result.get("contract_sha256") != contract["contract_sha256"]
        or result.get("plan_sha256") != contract["source_plan"]["plan_sha256"]
    ):
        raise JointTrainingContractError("teacher-fidelity result does not bind the frozen contract")
    asset = _validate_asset_identity(result.get("asset"))
    candidate_execution_verification = _validate_result_candidate_execution(
        result.get("candidate_execution_verification"),
        stage=stage,
        asset=asset,
        contract=contract,
        repository_root=repository_root,
    )
    frozen_train_sha256 = result.get("frozen_train_result_sha256")
    if stage == TRAIN_SPLIT:
        if frozen_train_sha256 is not None:
            raise JointTrainingContractError("training result must not bind a prior train result")
    else:
        _sha256(frozen_train_sha256, "holdout frozen training-result SHA256")
    models = result.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODELS):
        raise JointTrainingContractError("teacher-fidelity result has the wrong model set")
    model_metrics = {
        model: _validate_model_result(models[model], contract=contract, model=model, stage=stage)
        for model in MODELS
    }
    for model in MODELS:
        evidence = model_metrics[model]["runtime_evidence"]
        if (
            evidence["asset_sha256"] != asset["sha256"]
            or evidence["asset_state_sha256"] != asset["state_sha256"]
        ):
            raise JointTrainingContractError(
                f"teacher-fidelity {model} runtime evidence is not bound to the frozen asset"
            )
    audit = _validate_result_access_audit(
        result.get("access_audit"), contract=contract, stage=stage
    )
    if stage == TRAIN_SPLIT:
        pass
    else:
        if frozen_train_result is None:
            raise JointTrainingContractError("holdout requires the frozen validated training result")
        train_identity = validate_teacher_fidelity_result(
            frozen_train_result, contract=contract, repository_root=repository_root
        )
        if train_identity["stage"] != TRAIN_SPLIT or asset != train_identity["asset"]:
            raise JointTrainingContractError("holdout asset differs from frozen training asset")
        for model in MODELS:
            if (
                model_metrics[model]["resolved_config_sha256"]
                != train_identity["model_relative_mse"][model]["resolved_config_sha256"]
            ):
                raise JointTrainingContractError(
                    f"holdout {model} resolved config differs from training"
                )
        if audit["rerank_executed"] or audit["partition_reshuffled"]:
            raise JointTrainingContractError("holdout attempted to optimize or retune")
    return {
        "stage": stage,
        "asset": asset,
        "frozen_train_result_sha256": frozen_train_sha256,
        "model_relative_mse": model_metrics,
        "candidate_execution_verification": candidate_execution_verification,
        "contract_sha256": str(contract["contract_sha256"]),
        "paper_result_eligible": False,
    }


def _validate_live_runtime_evidence_reference(
    identity: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    model: str,
    stage: str,
    repository_root: Path,
    candidate_execution_verification: Mapping[str, Any] | None,
) -> None:
    """Rehash and identity-check the runtime proof referenced by one result row."""
    path = _safe_repository_path(
        repository_root,
        identity["path"],
        f"{model} {stage} runtime evidence",
    )
    require_fixed_file_sha256(
        path, identity["sha256"], label=f"{model} {stage} runtime evidence"
    )
    evidence = _read_json(path, f"{model} {stage} runtime evidence")
    required_evidence_fields = {
        "schema_version",
        "kind",
        "contract_sha256",
        "plan_sha256",
        "model",
        "split",
        "checkpoint_sha256",
        "config_source_sha256",
        "runtime_source",
        "environment_profile",
        "profile_interpreter",
        "resolved_config_sha256",
        "preprocessing_id",
        "prepared_patch_size",
        "saes_routing_sha256",
        "saes_execution_route_sha256",
        "input_identity",
        "asset",
        "candidate_execution_verification",
        "execution_boundary",
        "selected_head",
        "ledger",
        "access_audit",
        "scene_records",
    }
    if set(evidence) != required_evidence_fields:
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence schema is invalid"
        )
    expected = {
        "schema_version": RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "kind": RUNTIME_EVIDENCE_KIND,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "model": model,
        "split": stage,
        "checkpoint_sha256": contract["checkpoint_binding"]["models"][model]["checkpoint_sha256"],
        "config_source_sha256": contract["model_config_binding"][model]["source_manifest_sha256"],
        "runtime_source": contract["model_config_binding"][model]["runtime_source"],
        "environment_profile": contract["model_config_binding"][model]["environment_profile"],
        "profile_interpreter": contract["model_config_binding"][model]["profile_interpreter"],
        "resolved_config_sha256": contract["model_config_binding"][model][
            "resolved_config_sha256"
        ],
        "preprocessing_id": contract["preprocessing"]["identifier"],
        "prepared_patch_size": contract["preprocessing"]["per_model_patch_size"][model],
        "saes_routing_sha256": contract["saes_routing"]["routing_sha256"],
        "saes_execution_route_sha256": contract["saes_routing"][
            "execution_route_sha256"
        ],
        "input_identity": _expected_input_identity(contract, stage),
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            raise JointTrainingContractError(
                f"{model} {stage} runtime evidence differs at {field}"
            )
    if evidence.get("candidate_execution_verification") != candidate_execution_verification:
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence candidate verification changed"
        )
    selected_head = evidence.get("selected_head")
    ledger = evidence.get("ledger")
    asset = evidence.get("asset")
    if (
        not isinstance(selected_head, Mapping)
        or not isinstance(ledger, Mapping)
        or not isinstance(asset, Mapping)
        or canonical_sha256(selected_head) != identity["selected_head_sha256"]
        or canonical_sha256(ledger) != identity["ledger_sha256"]
        or asset.get("sha256") != identity["asset_sha256"]
        or asset.get("state_sha256") != identity["asset_state_sha256"]
    ):
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence replay or asset identity changed"
        )
    try:
        verified_asset = _validate_asset_identity(asset)
    except JointTrainingContractError as exc:
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence asset manifest is invalid"
        ) from exc
    if (
        verified_asset["sha256"] != identity["asset_sha256"]
        or verified_asset["state_sha256"] != identity["asset_state_sha256"]
    ):
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence asset identity changed"
        )
    expected_boundary = {
        "dense_route_prepass_for_validation": True,
        "selected_head_only": True,
        "s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "renderer_executed": False,
        "quality_metrics_computed": False,
    }
    if evidence.get("execution_boundary") != expected_boundary:
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence overstates its execution boundary"
        )
    if (
        set(selected_head)
        != {"model", "contract_version", "dense_head_macs", "replayed_head_macs"}
        or selected_head.get("model") != model
        or selected_head.get("contract_version") != "saes-selected-output-replay-v1"
    ):
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence selected-head schema is invalid"
        )
    dense_macs = selected_head.get("dense_head_macs")
    replayed_macs = selected_head.get("replayed_head_macs")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (dense_macs, replayed_macs)
        )
        or dense_macs <= replayed_macs
    ):
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence selected-head MACs are invalid"
        )
    events = ledger.get("events") if isinstance(ledger, Mapping) else None
    required_events = {
        "joint_calibrator_calls",
        "joint_calibrator_l0_calls",
        "joint_calibrator_l1_calls",
        "joint_calibrator_full_calls",
        "joint_calibrator_selected_descriptor_reads",
        "joint_calibrator_skipped_head_macs",
    }
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("ledger_version") != LEDGER_VERSION
        or not isinstance(events, Mapping)
        or not required_events <= set(events)
    ):
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence ledger is invalid"
        )
    if any(
        isinstance(events[field], bool)
        or not isinstance(events[field], int)
        or events[field] < 0
        for field in required_events
    ):
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence ledger counters are invalid"
        )
    if (
        events["joint_calibrator_calls"]
        != events["joint_calibrator_selected_descriptor_reads"]
        or events["joint_calibrator_calls"]
        != events["joint_calibrator_l0_calls"]
        + events["joint_calibrator_l1_calls"]
        + events["joint_calibrator_full_calls"]
        or events["joint_calibrator_full_calls"] != 0
        or events["joint_calibrator_skipped_head_macs"] <= 0
    ):
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence ledger does not charge the calibrator"
        )
    records = evidence.get("scene_records")
    expected_count = contract["context_only_inputs"]["splits"][stage]["scene_count"]
    if not isinstance(records, list) or len(records) != expected_count:
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence scene coverage is incomplete"
        )
    expected_record_fields = {
        "sample_index",
        "scene",
        "route_mask_sha256",
        "selected_head_sha256",
        "saes_stats_sha256",
    }
    for ordinal, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or set(record) != expected_record_fields
            or record.get("sample_index") != ordinal
            or not isinstance(record.get("scene"), str)
            or not record["scene"]
        ):
            raise JointTrainingContractError(
                f"{model} {stage} runtime evidence scene order is invalid"
            )
        for field in expected_record_fields - {"sample_index", "scene"}:
            _sha256(record.get(field), f"{model} {stage} runtime evidence {field}")
    if len({record["scene"] for record in records}) != len(records):
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence has duplicate scenes"
        )
    audit = evidence.get("access_audit")
    required_audit = {
        "target_rgb_accessed",
        "target_camera_metadata_accessed",
        "target_index_accessed",
        "expected_results_accessed",
        "evaluation_scene_accessed",
        "teacher_files_opened",
        "teacher_files_runtime_accessible",
        "runtime_teacher_path",
        "descriptor_model_id_accessed",
        "descriptor_dataset_id_accessed",
        "optimizer_executed",
        "asset_updated",
        "renderer_executed",
        "quality_metrics_computed",
    }
    forbidden = required_audit - {"runtime_teacher_path"}
    if (
        not isinstance(audit, Mapping)
        or set(audit) != required_audit
        or any(audit.get(field) is not False for field in forbidden)
        or audit.get("runtime_teacher_path") is not None
    ):
        raise JointTrainingContractError(
            f"{model} {stage} runtime evidence crossed an isolation boundary"
        )


def validate_live_teacher_fidelity_result(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    contract: Mapping[str, Any],
    repository_root: Path = ROOT,
    frozen_train_result: Mapping[str, Any] | None = None,
    frozen_train_result_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a durable fidelity record, its frozen asset, and holdout ancestry."""
    repository_root = Path(repository_root).resolve()
    contract = _require_canonical_live_contract(contract, repository_root=repository_root)
    validate_live_training_contract(contract, repository_root=repository_root)
    stage = result.get("stage") if isinstance(result, Mapping) else None
    if stage not in {TRAIN_SPLIT, HOLDOUT_SPLIT}:
        raise JointTrainingContractError("teacher-fidelity result has an invalid stage")
    expected_result_path = _safe_repository_path(
        repository_root, contract["result_records"][stage], f"{stage} fidelity result"
    )
    if Path(result_path).resolve() != expected_result_path:
        raise JointTrainingContractError(f"{stage} fidelity record path differs from the frozen contract")
    if not expected_result_path.is_file():
        raise JointTrainingContractError(f"{stage} fidelity record is missing")
    durable_result = _read_json(expected_result_path, f"{stage} fidelity result")
    if durable_result != result:
        raise JointTrainingContractError(
            f"{stage} fidelity result object differs from its canonical file"
        )
    canonical_train_result: Mapping[str, Any] | None = None
    expected_train_path: Path | None = None
    if stage == HOLDOUT_SPLIT:
        if frozen_train_result_path is None:
            raise JointTrainingContractError("holdout requires the frozen training-result path")
        expected_train_path = _safe_repository_path(
            repository_root,
            contract["result_records"][TRAIN_SPLIT],
            "frozen training fidelity result",
        )
        if Path(frozen_train_result_path).resolve() != expected_train_path or not expected_train_path.is_file():
            raise JointTrainingContractError("holdout training-result path differs from the frozen contract")
        canonical_train_result = _read_json(expected_train_path, "frozen train result")
        if canonical_train_result.get("stage") != TRAIN_SPLIT:
            raise JointTrainingContractError("canonical frozen train result has the wrong stage")
        if frozen_train_result is not None and frozen_train_result != canonical_train_result:
            raise JointTrainingContractError(
                "holdout frozen training-result object differs from its canonical file"
            )
        # This is deliberately recursive only in the holdout-to-train
        # direction. A blocked train candidate must block any derived holdout.
        validate_live_teacher_fidelity_result(
            canonical_train_result,
            result_path=expected_train_path,
            contract=contract,
            repository_root=repository_root,
        )
    result_identity = validate_teacher_fidelity_result(
        durable_result,
        contract=contract,
        frozen_train_result=canonical_train_result,
        repository_root=repository_root,
    )
    asset = result_identity["asset"]
    asset_path = _safe_repository_path(
        repository_root, asset["asset_path"], "frozen joint calibrator asset"
    )
    try:
        from saes.joint_materialization_calibrator import (
            load_calibrator_asset,
            validate_calibrator_asset,
        )

        manifest = {
            "schema_version": asset["schema_version"],
            "kind": asset["kind"],
            "sha256": asset["sha256"],
            "byte_count": asset["byte_count"],
            "descriptor_dim": asset["descriptor_dim"],
            "bottleneck_dim": asset["bottleneck_dim"],
            "joint_output_dim": asset["joint_output_dim"],
        }
        verified_asset = validate_calibrator_asset(asset_path, manifest)
        calibrator = load_calibrator_asset(asset_path, verified_asset)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise JointTrainingContractError("frozen joint calibrator asset does not validate") from exc
    if calibrator.runtime_state_sha256 != asset["state_sha256"]:
        raise JointTrainingContractError("frozen joint calibrator state SHA256 differs from result")
    for model in MODELS:
        _validate_live_runtime_evidence_reference(
            result_identity["model_relative_mse"][model]["runtime_evidence"],
            contract=contract,
            model=model,
            stage=stage,
            repository_root=repository_root,
            candidate_execution_verification=result_identity[
                "candidate_execution_verification"
            ],
        )
    train_result_sha256 = None
    if stage == HOLDOUT_SPLIT:
        assert expected_train_path is not None
        train_result_sha256 = sha256_file(frozen_train_result_path)
        if train_result_sha256 != result_identity["frozen_train_result_sha256"]:
            raise JointTrainingContractError("holdout does not bind the immutable training-result bytes")
    return {
        **result_identity,
        "asset_verified": True,
        "asset_state_sha256": calibrator.runtime_state_sha256,
        "result_sha256": sha256_file(expected_result_path),
        "frozen_train_result_sha256": train_result_sha256,
        "paper_result_eligible": False,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen training contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    freeze.add_argument("--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION_ROOT)
    freeze.add_argument("--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST)
    freeze.add_argument("--repository-root", type=Path, default=ROOT)

    validate = subparsers.add_parser("validate-contract")
    validate.add_argument("--contract", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path, default=ROOT)

    validate_result = subparsers.add_parser("validate-result")
    validate_result.add_argument("--contract", type=Path, required=True)
    validate_result.add_argument("--result", type=Path, required=True)
    validate_result.add_argument("--frozen-train-result", type=Path)
    validate_result.add_argument("--repository-root", type=Path, default=ROOT)

    resolve_config = subparsers.add_parser("resolve-model-config")
    resolve_config.add_argument("--model", choices=MODELS, required=True)

    args = parser.parse_args()
    try:
        if args.command == "freeze":
            contract = build_training_contract(
                plan_path=args.plan,
                materialization_root=args.materialization_root,
                checkpoint_manifest_path=args.checkpoint_manifest,
                repository_root=args.repository_root,
            )
            _write_json(args.output, contract)
            record = {
                "output": str(args.output),
                **validate_training_contract(contract),
            }
        elif args.command == "validate-contract":
            contract = _read_json(args.contract, "training contract")
            record = validate_live_training_contract(
                contract, repository_root=args.repository_root
            )
        elif args.command == "validate-result":
            contract = _read_json(args.contract, "training contract")
            result = _read_json(args.result, "teacher-fidelity result")
            frozen_train = (
                _read_json(args.frozen_train_result, "frozen training result")
                if args.frozen_train_result is not None
                else None
            )
            record = validate_live_teacher_fidelity_result(
                result,
                result_path=args.result,
                contract=contract,
                repository_root=args.repository_root,
                frozen_train_result=frozen_train,
                frozen_train_result_path=args.frozen_train_result,
            )
        else:
            record = {
                "model": args.model,
                "resolved_config_sha256": _resolved_model_config_sha256_here(args.model),
            }
    except (ImportError, JointTrainingContractError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
