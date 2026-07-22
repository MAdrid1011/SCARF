"""Prepare a context-only ACID joint-calibration record for one model worker.

The generic ACID loader intentionally has no model dependency. This companion
is imported only inside a one-model subprocess after that model's checkpoint
has been hash-validated. It mirrors the existing context-view crop and patch
shims without constructing a target mapping or placeholder.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping

import torch

def _finite_positive_scale(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.numel() != 1 or not bool(torch.isfinite(value).all()) or float(value) <= 0.0:
        raise ValueError(f"ACID joint {label} must be finite and positive")
    return value


def _clone_context_tensors(context: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    required = {"image", "extrinsics", "intrinsics", "index"}
    if set(context) != required:
        raise ValueError("ACID joint raw context has an unexpected schema")
    cloned = {}
    for key in required:
        value = context[key]
        if not torch.is_tensor(value):
            raise TypeError(f"ACID joint raw context {key} must be a tensor")
        cloned[key] = value.detach().clone()
    if cloned["image"].ndim != 5 or cloned["image"].shape[:3] != (1, 2, 3):
        raise ValueError("ACID joint raw context image must have shape [1,2,3,H,W]")
    if cloned["extrinsics"].shape != (1, 2, 4, 4):
        raise ValueError("ACID joint raw context extrinsics must have shape [1,2,4,4]")
    if cloned["intrinsics"].shape != (1, 2, 3, 3):
        raise ValueError("ACID joint raw context intrinsics must have shape [1,2,3,3]")
    if cloned["index"].shape != (1, 2) or cloned["index"].dtype != torch.long:
        raise ValueError("ACID joint raw context index must have shape [1,2] long")
    return cloned


def prepare_acid_joint_model_context(
    source: Any,
    *,
    dataset_cfg: Any,
    encoder_cfg: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return one model-ready context mapping and a target-free preparation audit.

    No target camera, index, image, model identity, or dataset identity is
    added to the context mapping. The returned audit is logging provenance
    only; it is never supplied to the calibrator descriptor.
    """
    # Tests and archive validation intentionally reload the context module.
    # Resolve the class at the boundary so a valid freshly loaded record does
    # not fail an otherwise strict type check against a stale module object.
    from integration.acid_joint_context import AcidJointContext

    if not isinstance(source, AcidJointContext):
        raise TypeError("ACID joint source must be an AcidJointContext")
    context = _clone_context_tensors(source.context)
    source_shape = [int(value) for value in context["image"].shape[-2:]]
    scale: torch.Tensor | float = 1.0
    if bool(getattr(dataset_cfg, "make_baseline_1", False)):
        scale = _finite_positive_scale(
            (context["extrinsics"][0, 0, :3, 3] - context["extrinsics"][0, 1, :3, 3]).norm(),
            label="context baseline",
        )
        context["extrinsics"][:, :, :3, 3] /= scale
    near_value = float(getattr(dataset_cfg, "near", -1.0))
    far_value = float(getattr(dataset_cfg, "far", -1.0))
    near_value = 0.1 if near_value == -1.0 else near_value
    far_value = 1000.0 if far_value == -1.0 else far_value
    if not (0.0 < near_value < far_value):
        raise ValueError("ACID joint model has invalid context near/far bounds")
    nf_scale: torch.Tensor | float = (
        scale if bool(getattr(dataset_cfg, "baseline_scale_bounds", True)) else 1.0
    )
    context["near"] = torch.full((1, 2), near_value, dtype=torch.float32) / nf_scale
    context["far"] = torch.full((1, 2), far_value, dtype=torch.float32) / nf_scale

    image_shape = tuple(getattr(dataset_cfg, "image_shape", ()))
    if len(image_shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in image_shape):
        raise ValueError("ACID joint model has an invalid image_shape")
    crop_module = importlib.import_module("src.dataset.shims.crop_shim")
    patch_module = importlib.import_module("src.dataset.shims.patch_shim")
    crop = getattr(crop_module, "apply_crop_shim_to_views", None)
    patch = getattr(patch_module, "apply_patch_shim_to_views", None)
    if not callable(crop) or not callable(patch):
        raise RuntimeError("ACID joint model is missing context-view shims")
    context = crop(context, image_shape)
    patch_size = int(getattr(encoder_cfg, "shim_patch_size", 1)) * int(
        getattr(encoder_cfg, "downscale_factor", 1)
    )
    if patch_size < 1:
        raise ValueError("ACID joint model has an invalid patch size")
    context = patch(context, patch_size)
    if set(context) != {"image", "extrinsics", "intrinsics", "index", "near", "far"}:
        raise RuntimeError("ACID joint context shims changed the target-free schema")
    prepared = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in context.items()
    }
    if any(not torch.is_tensor(value) for value in prepared.values()):
        raise RuntimeError("ACID joint context shims emitted a non-tensor value")
    return prepared, {
        "source_image_shape": source_shape,
        "prepared_image_shape": [int(value) for value in prepared["image"].shape[-2:]],
        "context_view_count": 2,
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "baseline_normalized": bool(getattr(dataset_cfg, "make_baseline_1", False)),
        "patch_size": patch_size,
    }


__all__ = ["prepare_acid_joint_model_context"]
