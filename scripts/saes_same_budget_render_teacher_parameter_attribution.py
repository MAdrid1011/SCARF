#!/usr/bin/env python3
"""Attribute frozen render-teacher fidelity to representative parameter families.

This post-hoc capacity diagnostic consumes only frozen oracle artifacts, two
context RGB inputs, and frozen target-camera metadata.  It never loads target
RGB, creates an optimizer, changes routing, or evaluates ground-truth quality.
"""

# ruff: noqa: E402

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

from data.frozen_audit_contract import (
    require_fixed_file_sha256,
    require_frozen_identity,
)

from integration import create_model_loader
from integration.model_loader import _calibration_camera_geometry, _decode_calibration_context_images
from saes.frozen_audit_preflight import (
    SENTINEL_SPECS,
    assert_selected_only_commit_equal,
    bind_posthoc_full_s3_observation,
    poison_skipped_depths,
    poison_skipped_s3,
    require_posthoc_full_s3_exception,
    selected_only_commit_payload,
    selected_s3_read_evidence,
    wrap_selected_s3_reads,
)
from scripts.calibration_inputs import load_target_free_record, validate_target_free_input_root
from scripts.saes_same_budget_dense_oracle import MATERIALIZATION, SEED, _oracle_output_accounting
from scripts.saes_same_budget_render_teacher_oracle import (
    ATTRIBUTE_NAMES,
    FIXED_INPUT_IDENTITY,
    FIXED_PARTITION,
    BoundedRepresentativeParameters,
    _assert_compact_full_passthrough,
    _assert_fixed_partition,
    _camera_metadata_sha256,
    _render_one,
    _representative_mask,
    _sha256_tensors,
    _teacher_fidelity,
)
from scripts.saes_selected_output_quality_gate import (
    _apply_fixed_saes,
    _retain_renderable_gaussians,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution
from scripts.saes_target_free_materialization_audit import (
    _capture_encoder_execution,
    _clone_gaussians,
)


KIND = "saes_same_budget_render_teacher_parameter_attribution"
# v3 is an index-domain-only technical replay of the frozen v2 protocol.
# Keeping both identities in each record prevents the new durable output from
# being confused with the immutable v2 failure.
ATTRIBUTION_ID = "same-budget-render-teacher-parameter-attribution-v3"
FROZEN_PROTOCOL_ID = "same-budget-render-teacher-parameter-attribution-v2"
FAMILIES = ("means", "covariances", "opacities", "harmonics")
SELECTED_ONLY_PREFLIGHT_MATERIALIZATION = (
    "conditional-adapter-offset-attribute-transport-diagnostic"
)
ORACLE_ROOT = ROOT / (
    "outputs/ae_dl3dv_repair_diagnostics/"
    "transplat_sample0_same_budget_render_teacher_oracle_v3_frozen_contract"
)
CAMERA_INPUT_ROOT = ROOT / (
    "outputs/ae_dl3dv_repair_diagnostics/"
    "transplat_sample0_l1_primary_reference_target_free_input_v1"
)

EXPECTED_ORACLE_RESULTS_SHA256 = (
    "e4c110a087c8f19126f235a88091383e9b85e34d713ffd70a5960ad91e5ae295"
)
EXPECTED_OPTIMIZED_REPRESENTATIVES_SHA256 = (
    "9a784eff7a2c9150aa0f5fece4a32d667dc98666991049bfeace0dc34e7433d6"
)
EXPECTED_OPTIMIZED_VALUES_SHA256 = (
    "6d1fe17b2109713cd8b34658f48d4d09a15cd43cc449c331082b8fc68443f29c"
)
EXPECTED_DENSE_TEACHER_SHA256 = (
    "0d766a203ad00aff6596405ac3d35a04c93b92e95806c9b01d1fdf17a2193597"
)
EXPECTED_CAMERA_METADATA_SHA256 = (
    "626f8871946983a9d1da19dc604e7024156709edd36314a17ece0334cd41ceb9"
)
EXPECTED_CAMERA_AUDIT_INPUT_SHA256 = (
    "cfa8cd54c216d9f59558dd19c7d3c519a517b583ec10acc30dc757d75abb12d2"
)
EXPECTED_CAMERA_RECORD_SHA256 = (
    "c90664e89009005145a3c3f6257e7549cc35e9860bb185d3fe5d79b018da813b"
)
EXPECTED_CAMERA_SIDECAR_TREE_SHA256 = (
    "ae043f7584dec864af7a29d5f0c74a871974e627e36b0e23de955c7141b82fd9"
)
METRIC_TOLERANCES = {"psnr_db": 0.05, "ssim": 5.0e-4, "lpips": 5.0e-4}


def _variant_specs() -> tuple[dict[str, Any], ...]:
    """Return the exact, pre-registered family substitutions in run order."""
    return (
        {"name": "initial_compact", "teacher_families": (), "reverted_families": ()},
        {"name": "teacher_means", "teacher_families": ("means",), "reverted_families": ()},
        {
            "name": "teacher_covariances",
            "teacher_families": ("covariances",),
            "reverted_families": (),
        },
        {
            "name": "teacher_opacities",
            "teacher_families": ("opacities",),
            "reverted_families": (),
        },
        {
            "name": "teacher_harmonics",
            "teacher_families": ("harmonics",),
            "reverted_families": (),
        },
        {
            "name": "teacher_means_covariances",
            "teacher_families": ("means", "covariances"),
            "reverted_families": (),
        },
        {
            "name": "teacher_opacities_harmonics",
            "teacher_families": ("opacities", "harmonics"),
            "reverted_families": (),
        },
        {"name": "all_teacher", "teacher_families": FAMILIES, "reverted_families": ()},
        {
            "name": "all_teacher_minus_means",
            "teacher_families": tuple(family for family in FAMILIES if family != "means"),
            "reverted_families": ("means",),
        },
        {
            "name": "all_teacher_minus_covariances",
            "teacher_families": tuple(
                family for family in FAMILIES if family != "covariances"
            ),
            "reverted_families": ("covariances",),
        },
        {
            "name": "all_teacher_minus_opacities",
            "teacher_families": tuple(family for family in FAMILIES if family != "opacities"),
            "reverted_families": ("opacities",),
        },
        {
            "name": "all_teacher_minus_harmonics",
            "teacher_families": tuple(family for family in FAMILIES if family != "harmonics"),
            "reverted_families": ("harmonics",),
        },
    )


def _expected_variant_names() -> tuple[str, ...]:
    return tuple(str(spec["name"]) for spec in _variant_specs())


def _assert_optimized_values(
    optimized: dict[str, torch.Tensor], representative_global: torch.Tensor
) -> dict[str, torch.Tensor]:
    expected = {"global_indices", *ATTRIBUTE_NAMES}
    if set(optimized) != expected:
        raise RuntimeError("frozen optimized representative archive has an invalid schema")
    global_indices = optimized["global_indices"]
    if (
        not torch.is_tensor(global_indices)
        or global_indices.dtype != torch.long
        or not torch.equal(global_indices.detach().cpu(), representative_global.detach().cpu())
    ):
        raise RuntimeError("frozen optimized representative indices drifted from the partition")
    values: dict[str, torch.Tensor] = {}
    count = int(representative_global.numel())
    for name in ATTRIBUTE_NAMES:
        value = optimized[name]
        if not torch.is_tensor(value) or value.shape[0] != count or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"frozen optimized {name} values are invalid")
        values[name] = value
    if _sha256_tensors(values) != EXPECTED_OPTIMIZED_VALUES_SHA256:
        raise RuntimeError("frozen optimized representative values SHA256 drifted")
    covariance = values["covariances"]
    symmetric = (covariance + covariance.mT) * 0.5
    if not torch.allclose(covariance, symmetric, rtol=0.0, atol=1.0e-5):
        raise RuntimeError("frozen optimized covariance is not symmetric")
    if bool((torch.linalg.eigvalsh(symmetric) < -1.0e-6).any()):
        raise RuntimeError("frozen optimized covariance is not PSD")
    return values


def _load_frozen_oracle() -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor]:
    """Load only hash-bound post-hoc teacher artifacts before model loading."""
    results_path = ORACLE_ROOT / "results.json"
    require_fixed_file_sha256(
        results_path, EXPECTED_ORACLE_RESULTS_SHA256, label="frozen render teacher result"
    )
    record = json.loads(results_path.read_text(encoding="utf-8"))
    require_frozen_identity(
        record,
        {
            "kind": "saes_same_budget_render_teacher_oracle",
            "paper_result_eligible": False,
            "quality_retry_authorized": False,
            "fixed_input_identity": {"expected": FIXED_INPUT_IDENTITY, "verified": True},
            "fixed_partition": {"expected": FIXED_PARTITION},
            "execution_boundary": {
                "dense_renderer_output_is_teacher": True,
                "target_rgb_used": False,
                "runtime_execution": False,
            },
        },
        label="frozen render teacher result",
    )
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("frozen render teacher result has no artifact manifest")
    optimized_info = artifacts.get("optimized_representatives")
    teacher_info = artifacts.get("dense_render_teacher")
    if not isinstance(optimized_info, dict) or not isinstance(teacher_info, dict):
        raise RuntimeError("frozen render teacher artifact manifest is incomplete")
    if optimized_info.get("sha256") != EXPECTED_OPTIMIZED_REPRESENTATIVES_SHA256:
        raise RuntimeError("frozen optimized representative manifest drifted")
    if teacher_info.get("sha256") != EXPECTED_DENSE_TEACHER_SHA256:
        raise RuntimeError("frozen dense teacher manifest drifted")
    optimized_path = ORACLE_ROOT / str(optimized_info.get("path"))
    teacher_path = ORACLE_ROOT / str(teacher_info.get("path"))
    require_fixed_file_sha256(
        optimized_path,
        EXPECTED_OPTIMIZED_REPRESENTATIVES_SHA256,
        label="frozen optimized representatives",
    )
    require_fixed_file_sha256(
        teacher_path, EXPECTED_DENSE_TEACHER_SHA256, label="frozen dense render teacher"
    )
    optimized = torch.load(optimized_path, map_location="cpu", weights_only=True)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
    if not isinstance(optimized, dict) or not torch.is_tensor(teacher):
        raise RuntimeError("frozen render teacher artifacts have invalid types")
    if list(teacher.shape) != [4, 3, 256, 256] or not bool(torch.isfinite(teacher).all()):
        raise RuntimeError("frozen dense render teacher has the wrong tensor contract")
    if teacher_info.get("shape") != list(teacher.shape):
        raise RuntimeError("frozen dense render teacher shape drifted from manifest")
    return record, optimized, teacher


def _load_frozen_camera_record() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a camera-only target-free source sidecar and enforce its identity."""
    audit_path = CAMERA_INPUT_ROOT / "audit-input.json"
    require_fixed_file_sha256(
        audit_path,
        EXPECTED_CAMERA_AUDIT_INPUT_SHA256,
        label="frozen camera audit input",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require_frozen_identity(
        audit,
        {
            "kind": "dl3dv_target_free_l1_primary_reference_audit_input",
            "status": "PASS",
            "paper_result_eligible": False,
            "model": "transplat",
            "dataset": "dl3dv",
            "source_sample_index": 0,
            "target_rgb_included": False,
            "target_rgb_opened": False,
            "target_rgb_paths_passed_to_encoder": False,
            "selected_sample": {
                "scene": FIXED_INPUT_IDENTITY["scene"],
                "context_indices": FIXED_INPUT_IDENTITY["context_indices"],
                "target_indices": FIXED_INPUT_IDENTITY["target_indices"],
            },
            "canonical_protocol": {
                "source_index_sha256": FIXED_INPUT_IDENTITY["evaluation_index_sha256"],
            },
        },
        label="frozen camera audit input",
    )
    source_identity = validate_target_free_input_root(CAMERA_INPUT_ROOT / "sidecar", "dl3dv")
    require_frozen_identity(
        source_identity,
        {"tree_sha256": EXPECTED_CAMERA_SIDECAR_TREE_SHA256},
        label="frozen target-free camera sidecar",
    )
    record = load_target_free_record(CAMERA_INPUT_ROOT / "sidecar", FIXED_INPUT_IDENTITY["scene"])
    allowed = {
        "schema_version",
        "kind",
        "key",
        "cameras",
        "context_indices",
        "target_indices",
        "context_images",
    }
    if set(record) != allowed:
        raise RuntimeError("frozen camera sidecar has an unexpected payload field")
    if record["context_indices"] != FIXED_INPUT_IDENTITY["context_indices"] or record[
        "target_indices"
    ] != FIXED_INPUT_IDENTITY["target_indices"]:
        raise RuntimeError("frozen camera sidecar input indices drifted")
    index = json.loads(
        (CAMERA_INPUT_ROOT / "sidecar" / "test" / "index.json").read_text(encoding="utf-8")
    )
    chunk = CAMERA_INPUT_ROOT / "sidecar" / "test" / str(index[FIXED_INPUT_IDENTITY["scene"]])
    require_fixed_file_sha256(chunk, EXPECTED_CAMERA_RECORD_SHA256, label="frozen camera record")
    return record, {"audit_input_sha256": EXPECTED_CAMERA_AUDIT_INPUT_SHA256, **source_identity}


def _build_frozen_target_free_batch(loader: Any, model_bundle: Any, source_record: dict[str, Any]) -> dict[str, Any]:
    """Mirror the native crop/data shim using a zero-only target shape carrier."""
    loader._setup_imports()
    try:
        context_indices = list(source_record["context_indices"])
        target_indices = list(source_record["target_indices"])
        context_images = _decode_calibration_context_images(source_record["context_images"])
        source_height, source_width = context_images.shape[-2:]
        extrinsics, intrinsics = _calibration_camera_geometry(source_record["cameras"])
        if max([*context_indices, *target_indices]) >= extrinsics.shape[0]:
            raise RuntimeError("frozen target-free camera index exceeds source geometry")
        dataset_cfg = model_bundle.config.dataset
        context_extrinsics = extrinsics[context_indices]
        scale: torch.Tensor | float = 1.0
        if len(context_indices) == 2 and bool(getattr(dataset_cfg, "make_baseline_1", False)):
            scale = (context_extrinsics[0, :3, 3] - context_extrinsics[1, :3, 3]).norm()
            if float(scale) < float(getattr(dataset_cfg, "baseline_epsilon", 0.0)):
                raise RuntimeError("frozen target-free input has an insufficient baseline")
            extrinsics[:, :3, 3] /= scale
        near_value = float(getattr(dataset_cfg, "near", -1.0))
        far_value = float(getattr(dataset_cfg, "far", -1.0))
        near_value = 0.1 if near_value == -1.0 else near_value
        far_value = 1000.0 if far_value == -1.0 else far_value
        nf_scale: torch.Tensor | float = (
            scale if bool(getattr(dataset_cfg, "baseline_scale_bounds", True)) else 1.0
        )
        target_shape_carrier = context_images.new_zeros(
            (len(target_indices), 3, source_height, source_width)
        )
        batch = {
            "context": {
                "extrinsics": extrinsics[context_indices].unsqueeze(0),
                "intrinsics": intrinsics[context_indices].unsqueeze(0),
                "image": context_images.unsqueeze(0),
                "near": torch.full((1, len(context_indices)), near_value) / nf_scale,
                "far": torch.full((1, len(context_indices)), far_value) / nf_scale,
                "index": torch.tensor(context_indices, dtype=torch.long).unsqueeze(0),
            },
            "target": {
                "extrinsics": extrinsics[target_indices].unsqueeze(0),
                "intrinsics": intrinsics[target_indices].unsqueeze(0),
                "image": target_shape_carrier.unsqueeze(0),
                "near": torch.full((1, len(target_indices)), near_value) / nf_scale,
                "far": torch.full((1, len(target_indices)), far_value) / nf_scale,
                "index": torch.tensor(target_indices, dtype=torch.long).unsqueeze(0),
            },
            "scene": [source_record["key"]],
        }
        from src.dataset.data_module import get_data_shim
        from src.dataset.shims.crop_shim import apply_crop_shim

        batch = apply_crop_shim(batch, tuple(model_bundle.config.dataset.image_shape))
        batch = get_data_shim(model_bundle.encoder)(batch)
        placeholder = batch["target"].pop("image", None)
        if not torch.is_tensor(placeholder) or bool(torch.count_nonzero(placeholder)):
            raise RuntimeError("frozen target-free batch did not preserve a zero-only shape carrier")
        return batch
    finally:
        loader._restore_cwd()


def _load_model_and_inputs(device: torch.device) -> tuple[Any, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Load the full renderer from a no-RGB source sidecar and native shim."""
    from scripts.ae_config import resolve_claim_selection, resolve_experiment

    source_record, camera_source_identity = _load_frozen_camera_record()
    experiment = resolve_experiment("transplat", "dl3dv", ROOT)
    selection = resolve_claim_selection("transplat", "dl3dv", ROOT)
    checkpoint_sha256 = require_fixed_file_sha256(
        experiment.checkpoint,
        FIXED_INPUT_IDENTITY["checkpoint_sha256"],
        label="frozen attribution checkpoint",
    )
    evaluation_index_sha256 = require_fixed_file_sha256(
        selection.index_path,
        FIXED_INPUT_IDENTITY["evaluation_index_sha256"],
        label="frozen attribution evaluation index",
    )
    loader = create_model_loader("transplat")
    bundle = loader.load_model(
        str(experiment.checkpoint),
        device=device,
        experiment_name=experiment.experiment,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        hydra_overrides=experiment.hydra_overrides,
        encoder_only=False,
    )
    if bundle.decoder is None:
        raise RuntimeError("teacher attribution requires the native decoder")
    batch = _build_frozen_target_free_batch(loader, bundle, source_record)
    if "image" in batch["target"]:
        raise RuntimeError("frozen attribution target mapping still carries RGB")
    context = {
        key: value.to(bundle.device) if torch.is_tensor(value) else value
        for key, value in batch["context"].items()
    }
    target = {key: value.to(bundle.device) for key, value in batch["target"].items()}
    if list(context["image"].shape) != FIXED_INPUT_IDENTITY["context_shape"]:
        raise RuntimeError("frozen attribution context image shape drifted")
    if _camera_metadata_sha256(target) != EXPECTED_CAMERA_METADATA_SHA256:
        raise RuntimeError("frozen target camera metadata drifted")
    identity = {
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_index_sha256": evaluation_index_sha256,
        "scene": str(batch["scene"][0]),
        "context_indices": [int(value) for value in context["index"][0].detach().cpu().tolist()],
        "target_indices": [int(value) for value in target["index"][0].detach().cpu().tolist()],
        "context_shape": list(context["image"].shape),
        "teacher_views": int(target["extrinsics"].shape[1]),
    }
    require_frozen_identity(identity, FIXED_INPUT_IDENTITY, label="frozen attribution input")
    return bundle.model, context, target, {
        "fixed_input": identity,
        "camera_source": camera_source_identity,
        "target_camera_metadata_sha256": _camera_metadata_sha256(target),
        "target_rgb_accessed": False,
        "target_camera_mapping_present": True,
        "target_mapping_carries_rgb": False,
    }


def _require_clean_source_identity() -> dict[str, Any]:
    """Bind every technical replay to a committed, clean source identity."""
    from scripts.result_record import source_identity

    source = source_identity()
    if source.get("git_dirty") is not False:
        raise RuntimeError("frozen teacher attribution requires a clean source tree before execution")
    return source


def _build_compact_index_maps(
    *,
    dense: Any,
    compact: Any,
    modified: torch.Tensor,
    representative_global: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map frozen dense representative IDs to validated compact slots."""
    if not torch.is_tensor(modified) or modified.ndim != 1 or modified.dtype != torch.bool:
        raise RuntimeError("frozen compact index map has an invalid modified mask")
    if not torch.is_tensor(representative_global) or (
        representative_global.ndim != 1
        or representative_global.dtype != torch.long
        or representative_global.numel() < 1
    ):
        raise RuntimeError("frozen compact index map has invalid representative IDs")
    dense_means = getattr(dense, "means", None)
    compact_means = getattr(compact, "means", None)
    if (
        not torch.is_tensor(dense_means)
        or not torch.is_tensor(compact_means)
        or dense_means.ndim < 2
        or compact_means.ndim < 2
    ):
        raise RuntimeError("frozen compact index map has an invalid Gaussian layout")
    dense_slots = int(dense_means.shape[1])
    compact_slots = int(compact_means.shape[1])
    if (
        modified.numel() != dense_slots
        or modified.device != dense_means.device
        or representative_global.device != dense_means.device
        or compact_means.device != dense_means.device
    ):
        raise RuntimeError("frozen compact index map has inconsistent devices or slot counts")
    for family in ATTRIBUTE_NAMES:
        dense_value = getattr(dense, family, None)
        compact_value = getattr(compact, family, None)
        if (
            not torch.is_tensor(dense_value)
            or not torch.is_tensor(compact_value)
            or dense_value.ndim < 2
            or compact_value.ndim < 2
            or dense_value.shape[:2] != dense_means.shape[:2]
            or compact_value.shape[:2] != compact_means.shape[:2]
            or dense_value.device != dense_means.device
            or compact_value.device != compact_means.device
        ):
            raise RuntimeError(f"frozen compact index map has an invalid {family} layout")
    if (
        bool((representative_global < 0).any())
        or bool((representative_global >= dense_slots).any())
        or (
            representative_global.numel() > 1
            and not bool((representative_global[1:] > representative_global[:-1]).all())
        )
    ):
        raise RuntimeError("frozen compact index map has unordered or out-of-range representatives")
    retained_global = torch.nonzero(~modified, as_tuple=False).flatten()
    if compact_slots != retained_global.numel():
        raise RuntimeError("frozen compact reconstruction has the wrong slot count")
    global_to_local = torch.full(
        (dense_slots,), -1, dtype=torch.long, device=dense_means.device
    )
    global_to_local[retained_global] = torch.arange(
        retained_global.numel(), dtype=torch.long, device=dense_means.device
    )
    representative_local = global_to_local[representative_global]
    if (
        bool((representative_local < 0).any())
        or bool((representative_local >= compact_slots).any())
        or torch.unique(representative_local).numel() != representative_local.numel()
        or not torch.equal(retained_global[representative_local], representative_global)
    ):
        raise RuntimeError("frozen representative global-to-local map drifted")
    return retained_global, global_to_local, representative_local


def _build_parameter_variants(
    initial: Any,
    optimized_values: dict[str, torch.Tensor],
    representative_local: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    """Inject only saved frozen values into representative-local slots."""
    if set(optimized_values) != set(ATTRIBUTE_NAMES):
        raise ValueError("optimized values must contain every descriptor family")
    count = int(representative_local.numel())
    slot_count = int(initial.means.shape[1])
    if (
        representative_local.ndim != 1
        or representative_local.dtype != torch.long
        or count < 1
        or representative_local.device != initial.means.device
        or bool((representative_local < 0).any())
        or bool((representative_local >= slot_count).any())
    ):
        raise ValueError("representative local indices are invalid")
    if torch.unique(representative_local).numel() != count:
        raise ValueError("representative local indices alias one another")
    for family in ATTRIBUTE_NAMES:
        values = getattr(initial, family, None)
        if (
            not torch.is_tensor(values)
            or values.ndim < 2
            or values.shape[:2] != initial.means.shape[:2]
            or values.device != initial.means.device
        ):
            raise ValueError(f"initial compact has an invalid {family} layout")
    variants: dict[str, dict[str, Any]] = {}
    all_slots = torch.arange(initial.means.shape[1], device=representative_local.device)
    frozen = torch.ones_like(all_slots, dtype=torch.bool)
    frozen[representative_local] = False
    for spec in _variant_specs():
        active = tuple(spec["teacher_families"])
        candidate = _clone_gaussians(initial)
        for family in active:
            value = optimized_values[family].to(
                device=getattr(candidate, family).device,
                dtype=getattr(candidate, family).dtype,
            )
            if value.shape != getattr(candidate, family)[0, representative_local].shape:
                raise RuntimeError(f"frozen optimized {family} shape does not match representatives")
            getattr(candidate, family)[0, representative_local] = value
        for family in ATTRIBUTE_NAMES:
            actual = getattr(candidate, family)
            baseline = getattr(initial, family)
            if not torch.equal(actual[:, frozen], baseline[:, frozen]):
                raise RuntimeError(f"{spec['name']} changed a non-representative {family}")
            expected = (
                optimized_values[family].to(device=actual.device, dtype=actual.dtype)
                if family in active
                else baseline[0, representative_local]
            )
            if not torch.equal(actual[0, representative_local], expected):
                raise RuntimeError(f"{spec['name']} has the wrong representative {family}")
        variants[str(spec["name"])] = {"gaussians": candidate, **spec}
    if tuple(variants) != _expected_variant_names():
        raise RuntimeError("parameter attribution variant registry drifted")
    return variants


def _representative_level_masks(
    modified: torch.Tensor, representative_mask: torch.Tensor, *, views: int, height: int, width: int
) -> dict[str, torch.Tensor]:
    """Recover L0/L1 representative groups from the immutable K/2K partition."""
    if height % 4 or width % 4:
        raise RuntimeError("frozen attribution grid is not divisible by the tile size")
    l0 = torch.zeros_like(representative_mask)
    l1 = torch.zeros_like(representative_mask)
    per_view = height * width
    for view in range(views):
        for row in range(0, height, 4):
            for column in range(0, width, 4):
                local = torch.tensor(
                    [
                        view * per_view + (row + dy) * width + column + dx
                        for dy in range(4)
                        for dx in range(4)
                    ],
                    dtype=torch.long,
                    device=modified.device,
                )
                removed = int(modified[local].sum().item())
                representatives = local[representative_mask[local]]
                if removed == 0:
                    if representatives.numel():
                        raise RuntimeError("a Full tile has a representative marker")
                elif removed == 12:
                    if representatives.numel() != 4:
                        raise RuntimeError("an L0 tile has the wrong representative count")
                    l0[representatives] = True
                elif removed == 8:
                    if representatives.numel() != 8:
                        raise RuntimeError("an L1 tile has the wrong representative count")
                    l1[representatives] = True
                else:
                    raise RuntimeError("frozen partition has an unsupported tile budget")
    if bool((l0 & l1).any()) or not torch.equal(l0 | l1, representative_mask):
        raise RuntimeError("L0/L1 representative groups do not cover the frozen partition")
    return {"L0": l0, "L1": l1}


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().flatten().float().cpu()
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise RuntimeError("parameter attribution summary is empty or non-finite")
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "maximum": float(values.max().item()),
    }


def _grouped_parameter_changes(
    initial: Any,
    optimized_values: dict[str, torch.Tensor],
    representative_global: torch.Tensor,
    representative_local: torch.Tensor,
    groups: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Report frozen initial-to-teacher absolute and relative family changes."""
    if (
        representative_global.ndim != 1
        or representative_global.dtype != torch.long
        or representative_global.device != initial.means.device
        or representative_local.ndim != 1
        or representative_local.dtype != torch.long
        or representative_local.device != initial.means.device
        or representative_local.shape != representative_global.shape
        or representative_global.numel() < 1
        or (
            representative_global.numel() > 1
            and not bool((representative_global[1:] > representative_global[:-1]).all())
        )
        or bool((representative_local < 0).any())
        or bool((representative_local >= initial.means.shape[1]).any())
        or torch.unique(representative_local).numel() != representative_local.numel()
    ):
        raise RuntimeError("representative global-to-local map is invalid")
    output: dict[str, Any] = {}
    for level, mask in groups.items():
        if (
            not torch.is_tensor(mask)
            or mask.ndim != 1
            or mask.dtype != torch.bool
            or mask.device != representative_global.device
            or mask.numel() <= int(representative_global[-1].item())
        ):
            raise RuntimeError(f"{level} representative group has an invalid global layout")
        selected_global = torch.nonzero(mask, as_tuple=False).flatten()
        locations = torch.searchsorted(representative_global, selected_global)
        if bool((locations >= representative_global.numel()).any()) or not torch.equal(
            representative_global[locations], selected_global
        ):
            raise RuntimeError(f"{level} representatives are not in the frozen archive")
        selected_local = representative_local[locations]
        family_values: dict[str, Any] = {}
        for family in FAMILIES:
            baseline = getattr(initial, family)[0, selected_local]
            teacher = optimized_values[family].to(device=baseline.device, dtype=baseline.dtype)[locations]
            difference = (teacher - baseline).reshape(baseline.shape[0], -1)
            reference = baseline.reshape(baseline.shape[0], -1)
            absolute = torch.linalg.vector_norm(difference, dim=1)
            relative = absolute / torch.linalg.vector_norm(reference, dim=1).clamp_min(1.0e-8)
            family_values[family] = {
                "absolute_change": _summary(absolute),
                "relative_change": _summary(relative),
            }
        output[level] = {"representative_count": int(selected_global.numel()), "families": family_values}
    return output


def _render_variant_fidelity(
    model: Any,
    variants: dict[str, dict[str, Any]],
    target: dict[str, torch.Tensor],
    teacher: torch.Tensor,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Render every immutable substitution against the saved dense teacher."""
    teacher = teacher.to(device=target["extrinsics"].device)
    results: dict[str, Any] = {}
    with torch.no_grad():
        for name, entry in variants.items():
            images = torch.stack(
                [
                    _render_one(model, entry["gaussians"], target, view, image_shape)
                    for view in range(teacher.shape[0])
                ]
            )
            results[name] = {
                "teacher_parameter_families": list(entry["teacher_families"]),
                "reverted_from_all_teacher": list(entry["reverted_families"]),
                "direct_teacher_fidelity": _teacher_fidelity(images, teacher),
            }
    return results


def _sanity_match(observed: dict[str, float], expected: dict[str, float]) -> dict[str, Any]:
    delta = {key: float(observed[key] - expected[key]) for key in METRIC_TOLERANCES}
    return {
        "expected": dict(expected),
        "observed": dict(observed),
        "delta": delta,
        "tolerances": dict(METRIC_TOLERANCES),
        "passed": all(abs(delta[key]) <= METRIC_TOLERANCES[key] for key in METRIC_TOLERANCES),
    }


def _restoration_summary(variants: dict[str, Any]) -> dict[str, Any]:
    initial = variants["initial_compact"]["direct_teacher_fidelity"]["mean"]
    full = variants["all_teacher"]["direct_teacher_fidelity"]["mean"]
    denominator = full["psnr_db"] - initial["psnr_db"]
    if denominator <= 0.0:
        raise RuntimeError("frozen teacher control has no PSNR restoration span")
    result: dict[str, Any] = {}
    for name, value in variants.items():
        mean = value["direct_teacher_fidelity"]["mean"]
        result[name] = {
            "psnr_restoration_fraction": float((mean["psnr_db"] - initial["psnr_db"]) / denominator),
            "psnr_gap_to_all_teacher_db": float(full["psnr_db"] - mean["psnr_db"]),
            "ssim_gap_to_all_teacher": float(full["ssim"] - mean["ssim"]),
            "lpips_gap_to_all_teacher": float(mean["lpips"] - full["lpips"]),
        }
    return result


def _run_two_sentinel_selected_only_preflight(
    *, model: Any, context: dict[str, torch.Tensor], device: torch.device
) -> dict[str, Any]:
    """Prove selected descriptor access before the post-hoc oracle is rebuilt."""
    if SELECTED_ONLY_PREFLIGHT_MATERIALIZATION == MATERIALIZATION:
        raise RuntimeError("selected-only preflight cannot use the post-hoc full-S3 oracle")
    _, views, _, height, width = context["image"].shape
    with strict_fp32_convolution_execution() as numerical_execution:
        reference_gaussians, reference_features, reference_depths = _capture_encoder_execution(
            model, context
        )
        reference_trace: list[dict[str, Any]] = []
        reference_mask, reference_stats = _apply_fixed_saes(
            reference_gaussians,
            features=reference_features,
            depths=reference_depths,
            context=context,
            height=height,
            width=width,
            views=views,
            materialization=SELECTED_ONLY_PREFLIGHT_MATERIALIZATION,
            tile_trace=reference_trace,
        )
        if reference_stats.get("guard_nonprobe_s3_attribute_reads") != 0:
            raise RuntimeError("selected-only reference read a skipped S3 descriptor")
        reference = selected_only_commit_payload(
            gaussians=reference_gaussians,
            modified_mask=reference_mask,
            stats=reference_stats,
            tile_trace=reference_trace,
        )
        selected = torch.nonzero(~reference_mask, as_tuple=False).flatten()
        if selected.numel() == 0:
            raise RuntimeError("selected-only reference has no retained descriptor")
        replays: list[dict[str, Any]] = []
        for sentinel_name, sentinel in SENTINEL_SPECS:
            gaussians, features, depths = _capture_encoder_execution(model, context)
            poison_skipped_s3(gaussians, reference_mask, sentinel=sentinel)
            poison_skipped_depths(
                depths,
                reference_mask,
                height=height,
                width=width,
                primitives_per_pixel=1,
                value=sentinel["depth"],
            )
            guarded, fields = wrap_selected_s3_reads(gaussians, selected)
            trace: list[dict[str, Any]] = []
            replay_mask, replay_stats = _apply_fixed_saes(
                guarded,
                features=features,
                depths=depths,
                context=context,
                height=height,
                width=width,
                views=views,
                materialization=SELECTED_ONLY_PREFLIGHT_MATERIALIZATION,
                tile_trace=trace,
            )
            if replay_stats.get("guard_nonprobe_s3_attribute_reads") != 0:
                raise RuntimeError(f"{sentinel_name} read a skipped S3 descriptor")
            replay = selected_only_commit_payload(
                gaussians=gaussians,
                modified_mask=replay_mask,
                stats=replay_stats,
                tile_trace=trace,
            )
            deltas = assert_selected_only_commit_equal(
                reference, replay, label=sentinel_name
            )
            replays.append(
                {
                    "sentinel": sentinel_name,
                    "route_trace_identical": True,
                    "all_stats_identical": True,
                    "committed_outputs_identical": True,
                    "full_outputs_identical": True,
                    "maximum_absolute_delta": deltas,
                    "selected_s3_read_guard": selected_s3_read_evidence(
                        fields, label=sentinel_name
                    ),
                }
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return {
        "kind": "two-sentinel-selected-only-s3-preflight-v1",
        "passed": True,
        "scope": "SAES descriptor access after dense encoder capture only",
        "materialization": SELECTED_ONLY_PREFLIGHT_MATERIALIZATION,
        "sentinel_count": len(SENTINEL_SPECS),
        "finite_s2_s3_sentinels": [name for name, _ in SENTINEL_SPECS],
        "reference_modified_mask_sha256": hashlib.sha256(
            reference["modified_mask"].numpy().tobytes()
        ).hexdigest(),
        "reference_retained_descriptor_count": int(reference["retained_indices"].numel()),
        "reference_full_descriptor_count": int(reference_stats["full_stage3_gaussians"]),
        "route_trace_identical": True,
        "all_stats_identical": True,
        "committed_outputs_identical": True,
        "full_outputs_identical": True,
        "selected_s3_read_guard_enforced": True,
        "proves_dense_oracle_selected_only": False,
        "proves_s3_compute_savings": False,
        "target_rgb_accessed": False,
        "replays": replays,
        "numerical_execution": numerical_execution,
    }


def collect_parameter_attribution(*, device: torch.device) -> dict[str, Any]:
    """Execute the one fixed frozen-oracle attribution without target RGB."""
    source = _require_clean_source_identity()
    s3_access_contract = require_posthoc_full_s3_exception(
        materialization=MATERIALIZATION,
        runtime_execution=False,
        paper_result_eligible=False,
        quality_gate_authorized=False,
    )
    model, context, target, input_provenance = _load_model_and_inputs(device)
    model.eval()
    selected_only_descriptor_access_preflight = _run_two_sentinel_selected_only_preflight(
        model=model, context=context, device=device
    )
    oracle_record, optimized_archive, teacher = _load_frozen_oracle()
    _, views, _, height, width = context["image"].shape
    if views != 2:
        raise RuntimeError("frozen attribution expects exactly two context views")
    with strict_fp32_convolution_execution() as numerical_execution:
        dense, features, depths = _capture_encoder_execution(model, context)
        canonical = _clone_gaussians(dense)
        modified, stats = _apply_fixed_saes(
            canonical,
            features=features,
            depths=depths,
            context=context,
            height=height,
            width=width,
            views=views,
            materialization=MATERIALIZATION,
        )
        accounting = _oracle_output_accounting(
            modified=modified, stats=stats, full_gaussians=int(dense.means.shape[1])
        )
        s3_access_preflight = bind_posthoc_full_s3_observation(
            s3_access_contract,
            full_stage3_reads=int(stats["same_budget_dense_oracle_full_stage3_reads"]),
        )
        representative_mask = _representative_mask(modified, views=views, height=height, width=width)
        full_global_mask, observed_partition = _assert_fixed_partition(
            dense=dense,
            canonical=canonical,
            modified=modified,
            representative_mask=representative_mask,
            stats=stats,
        )
        representative_global = torch.nonzero(representative_mask, as_tuple=False).flatten()
        optimized_values = _assert_optimized_values(optimized_archive, representative_global)
        compact = _retain_renderable_gaussians(canonical, ~modified)
        _retained_global, global_to_local, representative_local = _build_compact_index_maps(
            dense=dense,
            compact=compact,
            modified=modified,
            representative_global=representative_global,
        )
        full_local = _assert_compact_full_passthrough(
            compact=compact,
            dense=dense,
            full_global_mask=full_global_mask,
            global_to_local=global_to_local,
        )
        module = BoundedRepresentativeParameters(
            compact,
            dense,
            representative_global,
            representative_local,
            height=height,
            width=width,
        ).to(device)
        with torch.no_grad():
            initial = _clone_gaussians(module.compose())
        module.assert_immutable_slots(initial)
        variants = _build_parameter_variants(initial, optimized_values, representative_local)
        for entry in variants.values():
            for family in ATTRIBUTE_NAMES:
                if not torch.equal(
                    getattr(entry["gaussians"], family)[0, full_local],
                    getattr(dense, family)[0, torch.nonzero(full_global_mask, as_tuple=False).flatten()],
                ):
                    raise RuntimeError(f"{entry['name']} changed a Full passthrough {family}")
        level_masks = _representative_level_masks(
            modified, representative_mask, views=views, height=height, width=width
        )
        parameter_group_relative_changes = _grouped_parameter_changes(
            initial,
            optimized_values,
            representative_global,
            representative_local,
            level_masks,
        )
        fidelity = _render_variant_fidelity(
            model,
            variants,
            target,
            teacher,
            (height, width),
        )
    archived_teacher = oracle_record["teacher"]
    full_sanity = _sanity_match(
        fidelity["all_teacher"]["direct_teacher_fidelity"]["mean"],
        archived_teacher["final_fidelity"]["mean"],
    )
    initial_sanity = _sanity_match(
        fidelity["initial_compact"]["direct_teacher_fidelity"]["mean"],
        archived_teacher["initial_fidelity"]["mean"],
    )
    if not full_sanity["passed"] or not initial_sanity["passed"]:
        raise RuntimeError("frozen teacher fidelity sanity control drifted")
    return {
        "schema_version": "1.0",
        "kind": KIND,
        "audit_id": ATTRIBUTION_ID,
        "frozen_protocol_id": FROZEN_PROTOCOL_ID,
        "technical_replay_of": FROZEN_PROTOCOL_ID,
        "status": "COMPLETED",
        "paper_result_eligible": False,
        "quality_gate_authorized": False,
        "quality_metrics_computed": False,
        "direct_teacher_metrics_computed": True,
        "optimizer_executed": False,
        "renderer_executed": True,
        "teacher_oracle_accessed": True,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": True,
        "model": "transplat",
        "dataset": "dl3dv",
        "sample_index": 0,
        "scene": input_provenance["fixed_input"]["scene"],
        "context_indices": input_provenance["fixed_input"]["context_indices"],
        "target_indices": input_provenance["fixed_input"]["target_indices"],
        "execution_boundary": {
            "native_dataloader_executed": False,
            "target_camera_mapping_present": True,
            "target_mapping_carries_rgb": False,
            "target_rgb_loaded": False,
            "target_rgb_transferred": False,
            "target_rgb_used": False,
            "target_camera_metadata_used_for_posthoc_teacher_render": True,
            "dense_renderer_output_reused_as_teacher": True,
            "teacher_reoptimized": False,
            "routing_recomputed_only_for_fixed_partition_verification": True,
            "runtime_execution": False,
            "s2_s3_savings_claimed": 0.0,
        },
        "frozen_oracle": {
            "root": str(ORACLE_ROOT),
            "results_sha256": EXPECTED_ORACLE_RESULTS_SHA256,
            "optimized_representatives_sha256": EXPECTED_OPTIMIZED_REPRESENTATIVES_SHA256,
            "optimized_values_sha256": EXPECTED_OPTIMIZED_VALUES_SHA256,
            "dense_render_teacher_sha256": EXPECTED_DENSE_TEACHER_SHA256,
            "target_camera_metadata_sha256": EXPECTED_CAMERA_METADATA_SHA256,
        },
        "input_provenance": input_provenance,
        "selected_only_descriptor_access_preflight": selected_only_descriptor_access_preflight,
        "s3_access_preflight": s3_access_preflight,
        "routing": {
            "materialization": MATERIALIZATION,
            "committed_modified_mask_sha256": observed_partition["modified_mask_sha256"],
            "representative_mask_sha256": observed_partition["representative_mask_sha256"],
            "stats": stats,
        },
        "same_budget_output_accounting": accounting,
        "fixed_partition": observed_partition,
        "parameter_group_relative_changes": parameter_group_relative_changes,
        "variants": fidelity,
        "restoration_summary": _restoration_summary(fidelity),
        "sanity_controls": {
            "initial_compact": initial_sanity,
            "all_teacher": full_sanity,
        },
        "next_action": "interpret-frozen-parameter-attribution-before-any-new-candidate",
        "source": source,
        "numerical_execution": numerical_execution,
    }


def smoke_initial_compact_render(*, device: torch.device) -> dict[str, Any]:
    """Exercise the fixed compact reconstruction and one render without metrics."""
    source = _require_clean_source_identity()
    s3_access_contract = require_posthoc_full_s3_exception(
        materialization=MATERIALIZATION,
        runtime_execution=False,
        paper_result_eligible=False,
        quality_gate_authorized=False,
    )
    model, context, target, input_provenance = _load_model_and_inputs(device)
    model.eval()
    selected_only_descriptor_access_preflight = _run_two_sentinel_selected_only_preflight(
        model=model, context=context, device=device
    )
    _oracle_record, optimized_archive, _teacher = _load_frozen_oracle()
    _, views, _, height, width = context["image"].shape
    with strict_fp32_convolution_execution() as numerical_execution:
        dense, features, depths = _capture_encoder_execution(model, context)
        canonical = _clone_gaussians(dense)
        modified, stats = _apply_fixed_saes(
            canonical,
            features=features,
            depths=depths,
            context=context,
            height=height,
            width=width,
            views=views,
            materialization=MATERIALIZATION,
        )
        accounting = _oracle_output_accounting(
            modified=modified, stats=stats, full_gaussians=int(dense.means.shape[1])
        )
        s3_access_preflight = bind_posthoc_full_s3_observation(
            s3_access_contract,
            full_stage3_reads=int(stats["same_budget_dense_oracle_full_stage3_reads"]),
        )
        representative_mask = _representative_mask(modified, views=views, height=height, width=width)
        full_global_mask, observed_partition = _assert_fixed_partition(
            dense=dense,
            canonical=canonical,
            modified=modified,
            representative_mask=representative_mask,
            stats=stats,
        )
        representative_global = torch.nonzero(representative_mask, as_tuple=False).flatten()
        _assert_optimized_values(optimized_archive, representative_global)
        compact = _retain_renderable_gaussians(canonical, ~modified)
        _retained_global, global_to_local, representative_local = _build_compact_index_maps(
            dense=dense,
            compact=compact,
            modified=modified,
            representative_global=representative_global,
        )
        full_local = _assert_compact_full_passthrough(
            compact=compact,
            dense=dense,
            full_global_mask=full_global_mask,
            global_to_local=global_to_local,
        )
        module = BoundedRepresentativeParameters(
            compact,
            dense,
            representative_global,
            representative_local,
            height=height,
            width=width,
        ).to(device)
        with torch.no_grad():
            initial = _clone_gaussians(module.compose())
            image = _render_one(model, initial, target, 0, (height, width))
        module.assert_immutable_slots(initial)
        for family in ATTRIBUTE_NAMES:
            if not bool(torch.isfinite(getattr(initial, family)).all()):
                raise RuntimeError(f"initial compact has non-finite {family}")
            if not torch.equal(
                getattr(initial, family)[0, full_local],
                getattr(dense, family)[0, torch.nonzero(full_global_mask, as_tuple=False).flatten()],
            ):
                raise RuntimeError(f"initial compact changed a Full passthrough {family}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if list(image.shape) != [3, height, width] or not bool(torch.isfinite(image).all()):
            raise RuntimeError("initial compact smoke render has an invalid output")
    return {
        "kind": KIND,
        "audit_id": ATTRIBUTION_ID,
        "frozen_protocol_id": FROZEN_PROTOCOL_ID,
        "technical_replay_of": FROZEN_PROTOCOL_ID,
        "status": "PASS",
        "output_written": False,
        "teacher_metrics_computed": False,
        "optimizer_executed": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": True,
        "rendered_target_view": 0,
        "render_shape": list(image.shape),
        "fixed_partition": observed_partition,
        "same_budget_output_accounting": accounting,
        "selected_only_descriptor_access_preflight": selected_only_descriptor_access_preflight,
        "s3_access_preflight": s3_access_preflight,
        "input_provenance": input_provenance,
        "source": source,
        "numerical_execution": numerical_execution,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-initial-render", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_initial_render and args.output_dir is not None:
        parser.error("--smoke-initial-render does not write an output directory")
    if not args.smoke_initial_render and args.output_dir is None:
        parser.error("--output-dir is required unless --smoke-initial-render is set")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if args.smoke_initial_render:
        print(json.dumps(smoke_initial_compact_render(device=device), sort_keys=True))
        return 0
    assert args.output_dir is not None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        record = collect_parameter_attribution(device=device)
    except Exception as exc:
        record = {
            "schema_version": "1.0",
            "kind": KIND,
            "audit_id": ATTRIBUTION_ID,
            "frozen_protocol_id": FROZEN_PROTOCOL_ID,
            "technical_replay_of": FROZEN_PROTOCOL_ID,
            "status": "FAILED",
            "paper_result_eligible": False,
            "quality_gate_authorized": False,
            "quality_metrics_computed": False,
            "direct_teacher_metrics_computed": False,
            "optimizer_executed": False,
            "target_rgb_accessed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (args.output_dir / "results.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(args.output_dir / "results.json", file=sys.stderr)
        return 2
    (args.output_dir / "results.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
