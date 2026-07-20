"""Source-faithful selected-output support for DepthSplat's native color path.

The DepthSplat encoder has a dense ``gaussian_regressor`` followed by a
replicate-padded ``gaussian_head``.  A repeated L0/L1 layout closes the
regressor's spatial dependency, so this module keeps that preceding path
native dense and limits selected execution to the final head.  It then carries
the selected context RGB through the original GaussianAdapter as an explicit
keyword argument, preserving its SH DC initialization and z-depth geometry.

Nothing here claims sparse S1/S2/S3 execution.  Its purpose is to make the
selected raw-output and Adapter boundary auditable before a nonzero merge is
allowed to affect a renderer.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from saes.depthsplat_backend import (
    canonical_json_sha256,
    source_native_depthsplat_coordinates,
)
from saes.selected_output_replay import (
    FP32_ATOL,
    FP32_RTOL,
    replay_two_conv_selected_outputs,
    unpack_two_conv_head,
)


DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT = (
    "depthsplat-native-dense-regressor-selected-head-rgb-adapter-v1"
)
DEPTHSPLAT_NATIVE_DENSE_FALLBACK_ATOL = 2.0e-5
DEPTHSPLAT_NATIVE_DENSE_FALLBACK_RTOL = FP32_RTOL


@dataclass(frozen=True)
class DepthSplatAdapterInputs:
    """The dense native Adapter call captured from one encoder invocation."""

    extrinsics: torch.Tensor
    intrinsics: torch.Tensor
    coordinates: torch.Tensor
    depths: torch.Tensor
    opacities: torch.Tensor
    raw_body: torch.Tensor
    image_shape: tuple[int, int]
    input_images: torch.Tensor


@dataclass(frozen=True)
class DepthSplatNativeExecution:
    """One target-free dense DepthSplat encoder invocation and its boundaries."""

    dense_gaussians: Any
    gaussian_head_input: torch.Tensor
    dense_raw_head: torch.Tensor
    adapter_inputs: DepthSplatAdapterInputs
    sample_image_grid: Any
    events: dict[str, Any]
    # The router consumes these source tensors instead of dense raw-head
    # outputs. They are retained separately to keep the materializer target-
    # free and selected-output-only.
    routing_features: torch.Tensor | None = None
    routing_z_depths: torch.Tensor | None = None


@dataclass(frozen=True)
class DepthSplatSelectedHeadReplay:
    """Selected raw head outputs in their original ``[V,C,H,W]`` layout."""

    values: torch.Tensor
    selection_mask: torch.Tensor
    events: dict[str, Any]
    equivalence: dict[str, Any]
    # Full positions are copied from the captured source head rather than
    # reconstructed by the compact replay.  Keep the actual mask alongside
    # its ledger so packet construction can prove that boundary.
    native_full_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class DepthSplatSparseRawPacket:
    """Selected DepthSplat raw outputs plus source-bound Adapter side inputs.

    The raw descriptor layout is exactly ``[opacity_logit, offset_x, offset_y,
    adapter_body...]``.  It intentionally differs from the two-offset classic
    packet layout, which has no raw opacity channel and cannot carry RGB/SH.
    """

    descriptor_keys: torch.Tensor
    raw_head_descriptors: torch.Tensor
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor
    coordinates: torch.Tensor
    depths: torch.Tensor
    mapped_opacities: torch.Tensor
    source_rgb: torch.Tensor
    dense_slots: torch.Tensor
    source_trace: Mapping[str, Any]


@dataclass(frozen=True)
class DepthSplatPackedGaussianAttributes:
    """Variable-length selected native Adapter attributes in decoder slot order."""

    dense_slots: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor
    source_trace: dict[str, Any]
    source_trace_sha256: str
    # Digest of the attributes currently stored in this packet. Native Adapter
    # packets set this equal to the trace's native-adapter binding; a later
    # materialized packet records a new current binding while retaining that
    # source binding in its trace.
    attribute_binding_sha256: str | None = None

    def as_single_batch(self, gaussians_type: type[Any]) -> Any:
        """Build a native decoder Gaussian container without dense padding."""

        return gaussians_type(
            means=self.means.unsqueeze(0),
            covariances=self.covariances.unsqueeze(0),
            harmonics=self.harmonics.unsqueeze(0),
            opacities=self.opacities.unsqueeze(0),
        )


def _tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("DepthSplat tensor digest requires a tensor")
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    # ``Tensor.numpy`` does not support bfloat16. Hash physical bytes so an
    # audit reports a deterministic value before its strict-FP32 gate decides
    # whether the execution is eligible for materialization.
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not torch.is_tensor(value):
            raise TypeError("DepthSplat module state contains a non-tensor entry")
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def depthsplat_attribute_binding_sha256(
    *,
    dense_slots: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    harmonics: torch.Tensor,
    opacities: torch.Tensor,
) -> str:
    """Bind slot order and every packed Gaussian attribute to one packet."""

    return canonical_json_sha256(
        {
            "dense_slots_sha256": _tensor_sha256(dense_slots),
            "means_sha256": _tensor_sha256(means),
            "covariances_sha256": _tensor_sha256(covariances),
            "harmonics_sha256": _tensor_sha256(harmonics),
            "opacities_sha256": _tensor_sha256(opacities),
        }
    )


def _require_context(context: Mapping[str, Any]) -> tuple[int, int, int]:
    required = {"image", "extrinsics", "intrinsics", "near", "far"}
    if not isinstance(context, Mapping) or not required.issubset(context):
        raise ValueError("DepthSplat capture requires source context tensors only")
    image = context["image"]
    if not torch.is_tensor(image) or image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("DepthSplat capture requires one [B=1,V,3,H,W] context image")
    _, views, channels, height, width = image.shape
    if channels != 3 or min(views, height, width) < 1:
        raise ValueError("DepthSplat context image has an invalid native shape")
    for name, shape in (
        ("extrinsics", (1, views, 4, 4)),
        ("intrinsics", (1, views, 3, 3)),
    ):
        value = context[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"DepthSplat context {name} has an invalid shape")
    if any(context[name].device != image.device for name in required if torch.is_tensor(context[name])):
        raise ValueError("DepthSplat context tensors must share one device")
    return views, height, width


def _unwrap_gaussians(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = value.get("gaussians")
    required = ("means", "covariances", "harmonics", "opacities")
    if any(not torch.is_tensor(getattr(value, name, None)) for name in required):
        raise ValueError("DepthSplat encoder did not return native Gaussian attributes")
    return value


def _loaded_sample_image_grid(encoder: Any, *, source_root: Path | None) -> Any:
    module = sys.modules.get(type(encoder).__module__)
    source = getattr(module, "__file__", None) if module is not None else None
    function = getattr(module, "sample_image_grid", None) if module is not None else None
    if not isinstance(source, str) or not callable(function):
        raise RuntimeError("DepthSplat encoder module lacks source sample_image_grid")
    source_path = Path(source).resolve()
    if source_root is not None and Path(source_root).resolve() not in source_path.parents:
        raise RuntimeError("DepthSplat encoder module was loaded from a foreign source tree")
    return function


def _validate_encoder_structure(encoder: Any) -> tuple[nn.Module, nn.Module, Any]:
    regressor = getattr(encoder, "gaussian_regressor", None)
    head = getattr(encoder, "gaussian_head", None)
    adapter = getattr(encoder, "gaussian_adapter", None)
    if not isinstance(regressor, nn.Module) or not isinstance(head, nn.Module):
        raise ValueError("DepthSplat encoder lacks its native regressor or Gaussian head")
    regressor_first, _, regressor_second = unpack_two_conv_head(regressor)
    head_first, _, head_second = unpack_two_conv_head(head)
    if regressor_first.padding_mode != "zeros" or regressor_second.padding_mode != "zeros":
        raise ValueError("DepthSplat gaussian_regressor padding contract changed")
    if head_first.padding_mode != "replicate" or head_second.padding_mode != "replicate":
        raise ValueError("DepthSplat gaussian_head padding contract changed")
    if not callable(adapter):
        raise ValueError("DepthSplat encoder lacks its native GaussianAdapter")
    cfg = getattr(encoder, "cfg", None)
    if (
        getattr(cfg, "num_surfaces", None) != 1
        or getattr(cfg, "gaussians_per_pixel", None) != 1
        or getattr(cfg, "init_sh_input_img", None) is not True
    ):
        raise ValueError("DepthSplat selected packet supports only the native S=1/P=1/RGB-SH route")
    return regressor, head, adapter


def _capture_adapter_inputs(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> DepthSplatAdapterInputs:
    if len(args) != 7:
        raise RuntimeError("DepthSplat GaussianAdapter did not receive seven positional inputs")
    (
        extrinsics,
        intrinsics,
        coordinates,
        depths,
        opacities,
        raw_body,
        image_shape,
    ) = args
    input_images = kwargs.get("input_images")
    if not isinstance(image_shape, tuple) or len(image_shape) != 2:
        raise RuntimeError("DepthSplat GaussianAdapter has an invalid image shape")
    if not torch.is_tensor(input_images):
        raise RuntimeError("DepthSplat GaussianAdapter did not receive source RGB input_images")
    tensors = (extrinsics, intrinsics, coordinates, depths, opacities, raw_body)
    if any(not torch.is_tensor(value) for value in tensors):
        raise RuntimeError("DepthSplat GaussianAdapter inputs must be tensors")
    return DepthSplatAdapterInputs(
        extrinsics=extrinsics.detach(),
        intrinsics=intrinsics.detach(),
        coordinates=coordinates.detach(),
        depths=depths.detach(),
        opacities=opacities.detach(),
        raw_body=raw_body.detach(),
        image_shape=image_shape,
        input_images=input_images.detach(),
    )


def _validate_dense_adapter_inputs(
    adapter_inputs: DepthSplatAdapterInputs, *, views: int, height: int, width: int
) -> None:
    pixels = height * width
    if (
        adapter_inputs.extrinsics.shape != (1, views, 1, 1, 1, 4, 4)
        or adapter_inputs.intrinsics.shape != (1, views, 1, 1, 1, 3, 3)
        or adapter_inputs.coordinates.shape != (1, views, pixels, 1, 1, 2)
        or adapter_inputs.depths.shape != (1, views, pixels, 1, 1)
        or adapter_inputs.opacities.shape != (1, views, pixels, 1, 1)
        or adapter_inputs.raw_body.ndim != 6
        or adapter_inputs.raw_body.shape[:5] != (1, views, pixels, 1, 1)
        or adapter_inputs.image_shape != (height, width)
        or adapter_inputs.input_images.shape != (1, views, 3, height, width)
    ):
        raise RuntimeError("DepthSplat native Adapter layout changed")
    tensors = (
        adapter_inputs.extrinsics,
        adapter_inputs.intrinsics,
        adapter_inputs.coordinates,
        adapter_inputs.depths,
        adapter_inputs.opacities,
        adapter_inputs.raw_body,
        adapter_inputs.input_images,
    )
    if any(value.device != adapter_inputs.raw_body.device for value in tensors):
        raise RuntimeError("DepthSplat native Adapter inputs use mixed devices")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("DepthSplat native Adapter inputs are non-finite")


def _adapter_inputs_binding_sha256(adapter_inputs: DepthSplatAdapterInputs) -> str:
    """Bind every same-invocation native Adapter side input."""

    if not isinstance(adapter_inputs, DepthSplatAdapterInputs):
        raise TypeError("DepthSplat Adapter binding requires captured inputs")
    return canonical_json_sha256(
        {
            "extrinsics_sha256": _tensor_sha256(adapter_inputs.extrinsics),
            "intrinsics_sha256": _tensor_sha256(adapter_inputs.intrinsics),
            "coordinates_sha256": _tensor_sha256(adapter_inputs.coordinates),
            "depths_sha256": _tensor_sha256(adapter_inputs.depths),
            "opacities_sha256": _tensor_sha256(adapter_inputs.opacities),
            "raw_body_sha256": _tensor_sha256(adapter_inputs.raw_body),
            "input_images_sha256": _tensor_sha256(adapter_inputs.input_images),
            "image_shape": list(adapter_inputs.image_shape),
        }
    )


def _dense_gaussian_attribute_binding_sha256(
    dense_gaussians: Any, *, slots: int
) -> str:
    """Digest the source-native dense Adapter output in decoder slot order."""

    dense = _unwrap_gaussians(dense_gaussians)
    if (
        dense.means.shape != (1, slots, 3)
        or dense.covariances.shape != (1, slots, 3, 3)
        or dense.harmonics.ndim != 4
        or dense.harmonics.shape[:3] != (1, slots, 3)
        or dense.opacities.shape != (1, slots)
    ):
        raise ValueError("DepthSplat dense Adapter attributes have an invalid shape")
    dense_slots = torch.arange(slots, device=dense.means.device, dtype=torch.int64)
    return depthsplat_attribute_binding_sha256(
        dense_slots=dense_slots,
        means=dense.means[0],
        covariances=dense.covariances[0],
        harmonics=dense.harmonics[0],
        opacities=dense.opacities[0],
    )


def capture_depthsplat_native_execution(
    encoder: Any,
    context: Mapping[str, Any],
    *,
    source_root: Path | None = None,
) -> DepthSplatNativeExecution:
    """Capture one target-free native encoder pass at the Adapter boundary."""

    views, height, width = _require_context(context)
    if bool(getattr(encoder, "training", False)):
        raise ValueError("DepthSplat native capture requires eval mode")
    regressor, head, adapter = _validate_encoder_structure(encoder)
    feature_upsampler = getattr(encoder, "feature_upsampler", None)
    if not isinstance(feature_upsampler, nn.Module):
        raise ValueError("DepthSplat encoder lacks its native feature upsampler")
    sample_image_grid = _loaded_sample_image_grid(encoder, source_root=source_root)
    captured: dict[str, Any] = {}

    def capture_regressor(
        _module: Any, inputs: tuple[Any, ...], output: Any
    ) -> None:
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            raise RuntimeError("DepthSplat gaussian_regressor boundary changed")
        if "regressor" in captured:
            raise RuntimeError("DepthSplat gaussian_regressor ran more than once")
        captured["regressor"] = {
            "input_shape": list(inputs[0].shape),
            "output_shape": list(output.shape),
            "input_sha256": _tensor_sha256(inputs[0]),
            "output_sha256": _tensor_sha256(output),
        }

    def capture_head(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            raise RuntimeError("DepthSplat gaussian_head boundary changed")
        if "head" in captured:
            raise RuntimeError("DepthSplat gaussian_head ran more than once")
        captured["head"] = (inputs[0].detach(), output.detach())

    def capture_features(
        _module: Any, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        if not torch.is_tensor(output):
            raise RuntimeError("DepthSplat feature upsampler boundary changed")
        if "features" in captured:
            raise RuntimeError("DepthSplat feature upsampler ran more than once")
        captured["features"] = output.detach()

    def capture_adapter(
        inputs: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> None:
        if "adapter" in captured:
            raise RuntimeError("DepthSplat GaussianAdapter ran more than once")
        captured["adapter"] = _capture_adapter_inputs(inputs, kwargs)

    # ``EncoderDepthSplat`` invokes ``self.gaussian_adapter.forward(...)``
    # directly. A module pre-hook is intentionally not enough because it only
    # observes ``Module.__call__``. Wrap the bound native method for this one
    # scoped source invocation and restore it before returning to the caller.
    original_adapter_forward = adapter.forward

    def captured_adapter_forward(*args: Any, **kwargs: Any) -> Any:
        capture_adapter(args, kwargs)
        return original_adapter_forward(*args, **kwargs)

    regressor_handle = regressor.register_forward_hook(capture_regressor)
    head_handle = head.register_forward_hook(capture_head)
    feature_handle = feature_upsampler.register_forward_hook(capture_features)
    adapter.forward = captured_adapter_forward
    try:
        with torch.no_grad():
            dense_gaussians = _unwrap_gaussians(
                encoder(context, global_step=0, deterministic=True)
            )
    finally:
        regressor_handle.remove()
        head_handle.remove()
        feature_handle.remove()
        adapter.forward = original_adapter_forward
    if set(captured) != {"regressor", "head", "adapter", "features"}:
        raise RuntimeError("DepthSplat native capture did not reach every required boundary")
    head_input, dense_raw_head = captured["head"]
    adapter_inputs = captured["adapter"]
    routing_features = captured["features"]
    if (
        head_input.ndim != 4
        or dense_raw_head.ndim != 4
        or head_input.shape[0] != views
        or dense_raw_head.shape[0] != views
        or tuple(head_input.shape[-2:]) != (height, width)
        or tuple(dense_raw_head.shape[-2:]) != (height, width)
    ):
        raise RuntimeError("DepthSplat raw head layout changed")
    regressor_channels = int(captured["regressor"]["output_shape"][1])
    feature_start = regressor_channels + 3
    feature_end = feature_start + int(routing_features.shape[1])
    if (
        routing_features.ndim != 4
        or routing_features.shape[0] != views
        or tuple(routing_features.shape[-2:]) != (height, width)
        or feature_start < 0
        or feature_end + 1 != head_input.shape[1]
        or not torch.allclose(
            head_input[:, feature_start:feature_end],
            routing_features,
            rtol=FP32_RTOL,
            atol=FP32_ATOL,
        )
    ):
        raise RuntimeError("DepthSplat routing features drift from gaussian-head input")
    _validate_dense_adapter_inputs(
        adapter_inputs, views=views, height=height, width=width
    )
    routing_z_depths = adapter_inputs.depths[:, :, :, 0, 0].reshape(
        1, views, height, width
    )
    body_width = getattr(adapter, "d_in", None)
    if not isinstance(body_width, int) or dense_raw_head.shape[1] != body_width + 3:
        raise RuntimeError("DepthSplat raw-head opacity/offset/body layout changed")
    slots = views * height * width
    if (
        dense_gaussians.means.shape != (1, slots, 3)
        or dense_gaussians.covariances.shape != (1, slots, 3, 3)
        or dense_gaussians.harmonics.ndim != 4
        or dense_gaussians.harmonics.shape[:3] != (1, slots, 3)
        or dense_gaussians.opacities.shape != (1, slots)
    ):
        raise RuntimeError("DepthSplat final Gaussian layout changed")
    dense_attribute_binding_sha256 = _dense_gaussian_attribute_binding_sha256(
        dense_gaussians, slots=slots
    )
    adapter_inputs_binding_sha256 = _adapter_inputs_binding_sha256(adapter_inputs)
    events = {
            "contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
            "source_bound": True,
            "execution_scope": "depthsplat-dense-regressor-selected-gaussian-head-only",
            "gaussian_regressor": {
                "native_dense_executed": True,
                "padding_mode": "zeros",
                "weight_sha256": _module_state_sha256(regressor),
                **captured["regressor"],
            },
            "gaussian_head": {
                "padding_mode": "replicate",
                "weight_sha256": _module_state_sha256(head),
                "input_shape": list(head_input.shape),
                "output_shape": list(dense_raw_head.shape),
                "dense_output_sha256": _tensor_sha256(dense_raw_head),
            },
            "adapter": {
                "source_rgb_keyword": "input_images",
                "source_rgb_shape": list(adapter_inputs.input_images.shape),
                "image_shape": list(adapter_inputs.image_shape),
                "z_depth_geometry": True,
                "adapter_body_width": body_width,
                "dense_attribute_binding_sha256": dense_attribute_binding_sha256,
                "dense_inputs_binding_sha256": adapter_inputs_binding_sha256,
            },
            "routing": {
                "source_module": "feature_upsampler",
                "head_input_channel_slice": [feature_start, feature_end],
                "features_shape": list(routing_features.shape),
                "features_sha256": _tensor_sha256(routing_features),
                "z_depth_shape": list(routing_z_depths.shape),
                "z_depth_sha256": _tensor_sha256(routing_z_depths),
            },
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
    }
    events["native_execution_sha256"] = canonical_json_sha256(events)
    return DepthSplatNativeExecution(
        dense_gaussians=dense_gaussians,
        gaussian_head_input=head_input,
        dense_raw_head=dense_raw_head,
        adapter_inputs=adapter_inputs,
        sample_image_grid=sample_image_grid,
        events=events,
        routing_features=routing_features.unsqueeze(0),
        routing_z_depths=routing_z_depths,
    )


def _validate_selection(
    selection_mask: torch.Tensor, *, views: int, height: int, width: int
) -> torch.Tensor:
    if (
        not torch.is_tensor(selection_mask)
        or selection_mask.dtype != torch.bool
        or selection_mask.shape != (views, height, width)
    ):
        raise ValueError("DepthSplat selection must have shape [V,H,W] and bool dtype")
    if not bool(selection_mask.any()):
        raise ValueError("DepthSplat selected replay requires at least one output")
    return selection_mask


def _head_event_from_dense_source(
    head: nn.Module,
    *,
    height: int,
    width: int,
    batch_item: int,
    selected_final_output_positions: int | None = None,
) -> dict[str, Any]:
    first, _, second = unpack_two_conv_head(head)
    positions = height * width
    macs = positions * 9 * (
        first.in_channels * first.out_channels + second.in_channels * second.out_channels
    )
    return {
        "batch_item": batch_item,
        "source_native_dense_head_capture": True,
        "dense_spatial_positions": positions,
        "selected_final_output_positions": (
            positions
            if selected_final_output_positions is None
            else selected_final_output_positions
        ),
        "first_conv_required_output_positions": positions,
        "first_conv_dense_closure": True,
        "second_conv_selected_only": False,
        "dense_head_macs": macs,
        "replayed_head_macs": macs,
        "head_mac_saving": 0.0,
        "padding_mode": "replicate",
    }


def _numeric_equivalence_report(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare selected raw descriptors without emitting non-finite evidence."""

    if reference.shape != candidate.shape:
        raise ValueError("DepthSplat selected replay comparison shapes differ")
    finite = bool(torch.isfinite(reference).all()) and bool(torch.isfinite(candidate).all())
    report: dict[str, Any] = {
        "atol": atol,
        "rtol": rtol,
        "finite": finite,
    }
    if not finite:
        report.update(
            {
                "maximum_absolute_delta": None,
                "mean_absolute_delta": None,
                "equivalent": False,
            }
        )
        return report
    delta = (reference - candidate).abs()
    report.update(
        {
            "maximum_absolute_delta": float(delta.max().item()),
            "mean_absolute_delta": float(delta.mean().item()),
            "equivalent": bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol)),
        }
    )
    return report


def replay_depthsplat_selected_head(
    head: nn.Module,
    head_input: torch.Tensor,
    dense_raw_head: torch.Tensor,
    selection_mask: torch.Tensor,
    *,
    native_full_mask: torch.Tensor | None = None,
) -> DepthSplatSelectedHeadReplay:
    """Replay compact outputs and pass source-native Full outputs through.

    A Full tile is not an approximation and must retain the same raw head
    values the source encoder produced.  ``native_full_mask`` makes that
    exception explicit while compact L0/L1 positions continue through the
    replicate-padded selected-output replay. If a finite compact replay falls
    just outside the strict FP32 envelope but inside the fixed native-fallback
    envelope, that entire view's compact positions are copied from the source
    capture and charged as dense native work.
    """

    if (
        head_input.ndim != 4
        or dense_raw_head.ndim != 4
        or head_input.shape[0] != dense_raw_head.shape[0]
        or tuple(head_input.shape[-2:]) != tuple(dense_raw_head.shape[-2:])
    ):
        raise ValueError("DepthSplat selected replay head inputs are inconsistent")
    first, _, second = unpack_two_conv_head(head)
    if first.padding_mode != "replicate" or second.padding_mode != "replicate":
        raise ValueError("DepthSplat selected replay requires replicate-padded gaussian_head")
    views, channels, height, width = dense_raw_head.shape
    if channels != second.out_channels or head_input.shape[1] != first.in_channels:
        raise ValueError("DepthSplat selected replay head channels changed")
    selection_mask = _validate_selection(
        selection_mask, views=views, height=height, width=width
    ).to(head_input.device)
    if native_full_mask is None:
        native_full_mask = torch.zeros_like(selection_mask)
    elif (
        not torch.is_tensor(native_full_mask)
        or native_full_mask.dtype != torch.bool
        or native_full_mask.shape != selection_mask.shape
    ):
        raise ValueError("DepthSplat native Full mask must match selected outputs")
    else:
        native_full_mask = native_full_mask.to(head_input.device)
    if bool((native_full_mask & ~selection_mask).any()):
        raise ValueError("DepthSplat native Full mask requests an omitted output")
    output = torch.zeros_like(dense_raw_head)
    per_view: list[dict[str, Any]] = []
    selected_reference: list[torch.Tensor] = []
    selected_values: list[torch.Tensor] = []
    strict_failure_views = 0
    candidate_positions = 0
    candidate_replay_macs = 0
    actual_compact_replay_positions = 0
    fallback_views = 0
    fallback_positions = 0
    full_passthrough_bitwise = True
    for view in range(views):
        mask = selection_mask[view]
        full = native_full_mask[view]
        compact = mask & ~full
        selected = int(mask.sum().item())
        full_count = int(full.sum().item())
        compact_count = int(compact.sum().item())
        if selected == 0:
            per_view.append(
                {
                    "batch_item": view,
                    "source_native_dense_head_capture": False,
                    "selected_final_output_positions": 0,
                    "selected_compact_requested_positions": 0,
                    "compact_replay_candidate_positions": 0,
                    "candidate_replay_macs": 0,
                    "selected_compact_replay_positions": 0,
                    "compact_replay_candidate_finite": True,
                    "compact_replay_candidate_maximum_absolute_delta": None,
                    "compact_replay_candidate_mean_absolute_delta": None,
                    "compact_replay_strict_equivalent": True,
                    "compact_replay_fallback_envelope_equivalent": True,
                    "native_dense_fallback_applied": False,
                    "native_dense_fallback_compact_positions": 0,
                    "native_full_passthrough_bitwise": True,
                    "dense_head_macs": 0,
                    "replayed_head_macs": 0,
                    "head_mac_saving": 0.0,
                    "padding_mode": "replicate",
                }
            )
            continue
        full_bitwise = True
        if full_count:
            output[view, :, full] = dense_raw_head[view, :, full]
            full_bitwise = bool(
                torch.equal(output[view, :, full], dense_raw_head[view, :, full])
            )
        full_passthrough_bitwise &= full_bitwise
        candidate_count = 0
        candidate_macs = 0
        actual_replay_count = 0
        fallback_count = 0
        candidate_report: dict[str, Any] | None = None
        fallback_report: dict[str, Any] | None = None
        fallback_applied = False
        if compact_count == 0:
            event = {
                "batch_item": view,
                "source_native_dense_head_capture": bool(full_count),
                "selected_final_output_positions": selected,
                "dense_head_macs": 0,
                "replayed_head_macs": 0,
                "head_mac_saving": 0.0,
                "padding_mode": "replicate",
            }
        elif compact_count == height * width:
            output[view] = dense_raw_head[view]
            event = _head_event_from_dense_source(
                head, height=height, width=width, batch_item=view
            )
        else:
            replay = replay_two_conv_selected_outputs(
                head, head_input[view : view + 1], compact
            )
            candidate = replay.values[0]
            source = dense_raw_head[view, :, compact]
            candidate_count = compact_count
            candidate_positions += compact_count
            candidate_macs = int(replay.events["replayed_head_macs"])
            candidate_replay_macs += candidate_macs
            candidate_report = _numeric_equivalence_report(
                source, candidate, atol=FP32_ATOL, rtol=FP32_RTOL
            )
            fallback_report = _numeric_equivalence_report(
                source,
                candidate,
                atol=DEPTHSPLAT_NATIVE_DENSE_FALLBACK_ATOL,
                rtol=DEPTHSPLAT_NATIVE_DENSE_FALLBACK_RTOL,
            )
            if not candidate_report["equivalent"]:
                strict_failure_views += 1
            if (
                candidate_report["finite"]
                and not candidate_report["equivalent"]
                and fallback_report["equivalent"]
            ):
                output[view, :, compact] = source
                fallback_applied = True
                fallback_count = compact_count
                fallback_views += 1
                fallback_positions += compact_count
                event = _head_event_from_dense_source(
                    head,
                    height=height,
                    width=width,
                    batch_item=view,
                    selected_final_output_positions=selected,
                )
            else:
                output[view, :, replay.coordinates[:, 0], replay.coordinates[:, 1]] = candidate
                actual_replay_count = compact_count
                actual_compact_replay_positions += compact_count
                event = {"batch_item": view, **replay.events}
        event.update(
            {
                "source_native_full_passthrough_positions": full_count,
                "selected_compact_requested_positions": compact_count,
                "compact_replay_candidate_positions": candidate_count,
                "candidate_replay_macs": candidate_macs,
                "selected_compact_replay_positions": actual_replay_count,
                "compact_replay_candidate_finite": (
                    True if candidate_report is None else candidate_report["finite"]
                ),
                "compact_replay_candidate_maximum_absolute_delta": (
                    None
                    if candidate_report is None
                    else candidate_report["maximum_absolute_delta"]
                ),
                "compact_replay_candidate_mean_absolute_delta": (
                    None
                    if candidate_report is None
                    else candidate_report["mean_absolute_delta"]
                ),
                "compact_replay_strict_equivalent": (
                    True if candidate_report is None else candidate_report["equivalent"]
                ),
                "compact_replay_fallback_envelope_equivalent": (
                    True if fallback_report is None else fallback_report["equivalent"]
                ),
                "native_dense_fallback_applied": fallback_applied,
                "native_dense_fallback_compact_positions": fallback_count,
                "native_full_passthrough_bitwise": full_bitwise,
            }
        )
        per_view.append(event)
        selected_reference.append(dense_raw_head[view, :, mask])
        selected_values.append(output[view, :, mask])
    reference = torch.cat(selected_reference, dim=1)
    replayed = torch.cat(selected_values, dim=1)
    equivalence = _numeric_equivalence_report(
        reference, replayed, atol=FP32_ATOL, rtol=FP32_RTOL
    )
    equivalence["equivalent"] = bool(
        equivalence["equivalent"] and full_passthrough_bitwise
    )
    dense_macs = sum(int(event["dense_head_macs"]) for event in per_view)
    actual_macs = sum(int(event["replayed_head_macs"]) for event in per_view)
    return DepthSplatSelectedHeadReplay(
        values=output,
        selection_mask=selection_mask,
        events={
            "contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
            "padding_mode": "replicate",
            "batch_size": views,
            "head_cost_semantics": "logical-route-cost-excludes-fallback-validation-v1",
            "dense_head_macs": dense_macs,
            "actual_head_macs": actual_macs,
            "head_mac_delta": dense_macs - actual_macs,
            "head_mac_saving": 1.0 - actual_macs / dense_macs if dense_macs else 0.0,
            "selected_final_output_positions": int(selection_mask.sum().item()),
            "omitted_final_output_positions": views * height * width - int(selection_mask.sum().item()),
            "native_full_passthrough_mask_sha256": _tensor_sha256(
                native_full_mask.to(dtype=torch.uint8)
            ),
            "native_full_passthrough_positions": int(native_full_mask.sum().item()),
            "native_full_passthrough_bitwise": full_passthrough_bitwise,
            "selected_compact_requested_positions": int(
                (selection_mask & ~native_full_mask).sum().item()
            ),
            "compact_replay_candidate_positions": candidate_positions,
            "candidate_replay_macs": candidate_replay_macs,
            "selected_compact_replay_positions": actual_compact_replay_positions,
            "compact_replay_strict_failure_view_count": strict_failure_views,
            "native_dense_fallback_envelope_atol": DEPTHSPLAT_NATIVE_DENSE_FALLBACK_ATOL,
            "native_dense_fallback_envelope_rtol": DEPTHSPLAT_NATIVE_DENSE_FALLBACK_RTOL,
            "native_dense_fallback_view_count": fallback_views,
            "native_dense_fallback_compact_positions": fallback_positions,
            "per_view": per_view,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        },
        equivalence=equivalence,
        native_full_mask=native_full_mask,
    )


def _canonical_descriptor_keys(
    positions: torch.Tensor, *, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = positions[:, 1] * width + positions[:, 2]
    keys = torch.stack(
        (
            torch.zeros_like(pixels),
            positions[:, 0],
            pixels,
            torch.zeros_like(pixels),
        ),
        dim=1,
    ).to(dtype=torch.int64)
    return keys, pixels.to(dtype=torch.int64)


def _source_selected_side_inputs(
    execution: DepthSplatNativeExecution, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    adapter_inputs = execution.adapter_inputs
    height, width = adapter_inputs.image_shape
    positions = positions.to(adapter_inputs.raw_body.device)
    views = positions[:, 0]
    pixels = positions[:, 1] * width + positions[:, 2]
    return (
        adapter_inputs.extrinsics[0, views, 0, 0, 0],
        adapter_inputs.intrinsics[0, views, 0, 0, 0],
        adapter_inputs.depths[0, views, pixels, 0, 0],
        adapter_inputs.opacities[0, views, pixels, 0, 0],
        adapter_inputs.input_images[0, views, :, positions[:, 1], positions[:, 2]],
    )


def build_depthsplat_sparse_raw_packet(
    execution: DepthSplatNativeExecution,
    replay: DepthSplatSelectedHeadReplay,
) -> DepthSplatSparseRawPacket:
    """Bind selected replay outputs to their same-invocation native side inputs."""

    if not isinstance(execution, DepthSplatNativeExecution) or not isinstance(
        replay, DepthSplatSelectedHeadReplay
    ):
        raise TypeError("DepthSplat packet requires native execution and selected replay")
    if replay.events.get("contract_version") != DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT:
        raise ValueError("DepthSplat selected replay contract changed")
    if replay.equivalence.get("equivalent") is not True:
        raise ValueError("DepthSplat selected replay is not equivalent to its dense raw head")
    views, channels, height, width = execution.dense_raw_head.shape
    selection_mask = _validate_selection(
        replay.selection_mask, views=views, height=height, width=width
    )
    if replay.native_full_mask is None:
        native_full_mask = torch.zeros_like(selection_mask)
    elif (
        not torch.is_tensor(replay.native_full_mask)
        or replay.native_full_mask.dtype != torch.bool
        or replay.native_full_mask.shape != selection_mask.shape
    ):
        raise ValueError("DepthSplat packet native Full mask is invalid")
    else:
        native_full_mask = replay.native_full_mask.to(selection_mask.device)
    if bool((native_full_mask & ~selection_mask).any()):
        raise ValueError("DepthSplat packet native Full mask requests an omitted output")
    native_full_mask_sha256 = _tensor_sha256(native_full_mask.to(dtype=torch.uint8))
    native_full_positions = int(native_full_mask.sum().item())
    if (
        replay.events.get("native_full_passthrough_mask_sha256") is not None
        and replay.events.get("native_full_passthrough_mask_sha256")
        != native_full_mask_sha256
    ) or (
        replay.events.get("native_full_passthrough_positions") is not None
        and replay.events.get("native_full_passthrough_positions") != native_full_positions
    ):
        raise ValueError("DepthSplat packet native Full replay ledger drifted")
    if replay.values.shape != execution.dense_raw_head.shape:
        raise ValueError("DepthSplat selected replay output shape changed")
    if native_full_positions and not torch.equal(
        replay.values.permute(0, 2, 3, 1)[native_full_mask],
        execution.dense_raw_head.permute(0, 2, 3, 1)[native_full_mask],
    ):
        raise RuntimeError("DepthSplat packet native Full descriptors drifted from source head")
    positions = selection_mask.nonzero(as_tuple=False).to(replay.values.device)
    raw = replay.values.permute(0, 2, 3, 1)[
        positions[:, 0], positions[:, 1], positions[:, 2]
    ]
    dense_raw = execution.dense_raw_head.permute(0, 2, 3, 1)[
        positions[:, 0], positions[:, 1], positions[:, 2]
    ]
    if not bool(torch.allclose(raw, dense_raw, rtol=FP32_RTOL, atol=FP32_ATOL)):
        raise RuntimeError("DepthSplat packet selected descriptors drifted from source head")
    body_width = execution.adapter_inputs.raw_body.shape[-1]
    if channels != body_width + 3:
        raise ValueError("DepthSplat packet raw descriptor layout changed")
    extrinsics, intrinsics, depths, dense_opacities, source_rgb = _source_selected_side_inputs(
        execution, positions
    )
    coordinates = source_native_depthsplat_coordinates(
        raw, selection_mask, sample_image_grid=execution.sample_image_grid
    )
    native_coordinates = execution.adapter_inputs.coordinates[
        0,
        positions[:, 0],
        positions[:, 1] * width + positions[:, 2],
        0,
        0,
    ]
    mapped_opacities = raw[:, 0].sigmoid()
    if not bool(torch.allclose(coordinates, native_coordinates, rtol=FP32_RTOL, atol=FP32_ATOL)):
        raise RuntimeError("DepthSplat packet coordinates drifted from native z-depth Adapter inputs")
    if not bool(torch.allclose(mapped_opacities, dense_opacities, rtol=FP32_RTOL, atol=FP32_ATOL)):
        raise RuntimeError("DepthSplat packet opacity mapping drifted from native Adapter inputs")
    keys, pixels = _canonical_descriptor_keys(positions, width=width)
    dense_slots = (positions[:, 0] * (height * width) + pixels).to(dtype=torch.int64)
    trace = {
        "contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
        "source_bound": True,
        "execution_scope": "depthsplat-dense-regressor-selected-gaussian-head-only",
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
        "source_rgb_keyword": "input_images",
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "head_forward_invocations": 1,
        "head_final_positions_executed": int(selection_mask.sum().item()),
        "head_weight_sha256": execution.events["gaussian_head"]["weight_sha256"],
        "regressor_weight_sha256": execution.events["gaussian_regressor"]["weight_sha256"],
        "source_view_count": views,
        "source_image_shape": [height, width],
        "selection_mask_sha256": _tensor_sha256(selection_mask.to(dtype=torch.uint8)),
        "native_full_passthrough_mask_sha256": native_full_mask_sha256,
        "native_full_passthrough_positions": native_full_positions,
        "selected_descriptor_sha256": _tensor_sha256(raw),
        "selected_rgb_sha256": _tensor_sha256(source_rgb),
        "adapter_body_width": body_width,
        "native_execution_sha256": execution.events.get(
            "native_execution_sha256", canonical_json_sha256(execution.events)
        ),
    }
    if execution.routing_features is not None or execution.routing_z_depths is not None:
        if (
            not torch.is_tensor(execution.routing_features)
            or not torch.is_tensor(execution.routing_z_depths)
            or execution.routing_features.shape[0] != 1
            or execution.routing_features.shape[1] != views
            or tuple(execution.routing_features.shape[-2:]) != (height, width)
            or execution.routing_z_depths.shape != (1, views, height, width)
        ):
            raise ValueError("DepthSplat execution routing tensors are invalid")
        trace.update(
            {
                "routing_features_sha256": _tensor_sha256(execution.routing_features),
                "routing_z_depths_sha256": _tensor_sha256(execution.routing_z_depths),
            }
        )
    return DepthSplatSparseRawPacket(
        descriptor_keys=keys,
        raw_head_descriptors=raw,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        coordinates=coordinates,
        depths=depths,
        mapped_opacities=mapped_opacities,
        source_rgb=source_rgb,
        dense_slots=dense_slots,
        source_trace=trace,
    )


def subset_depthsplat_sparse_raw_packet(
    packet: DepthSplatSparseRawPacket,
    selection_mask: torch.Tensor,
) -> DepthSplatSparseRawPacket:
    """Keep a final decoder subset from an already source-selected packet.

    The incremental plan may prefetch L1 anchors for a possible L0 promotion.
    These producer-only rows cannot escape into an accepted L0 decoder packet.
    This function selects only already-replayed rows and never inspects a dense
    omitted raw descriptor or dense Gaussian attribute.
    """

    if not isinstance(packet, DepthSplatSparseRawPacket):
        raise TypeError("DepthSplat packet subset requires a selected raw packet")
    if (
        not torch.is_tensor(selection_mask)
        or selection_mask.dtype != torch.bool
        or selection_mask.ndim != 3
        or not bool(selection_mask.any())
    ):
        raise ValueError("DepthSplat packet subset requires a nonempty [V,H,W] bool mask")
    views, height, width = selection_mask.shape
    positions = selection_mask.nonzero(as_tuple=False)
    requested_slots = (
        positions[:, 0] * (height * width) + positions[:, 1] * width + positions[:, 2]
    ).to(device=packet.dense_slots.device, dtype=torch.int64)
    if (
        packet.dense_slots.ndim != 1
        or packet.dense_slots.dtype != torch.int64
        or packet.dense_slots.numel() == 0
        or packet.dense_slots.numel() != packet.raw_head_descriptors.shape[0]
        or packet.source_trace.get("source_bound") is not True
        or packet.source_trace.get("source_view_count") != views
        or packet.source_trace.get("source_image_shape") != [height, width]
    ):
        raise ValueError("DepthSplat packet subset source packet is invalid")
    source_slots = packet.dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    keys = packet.descriptor_keys
    pixels_per_view = height * width
    if (
        keys.shape != (len(source_slots), 4)
        or keys.dtype != torch.int64
        or not bool((keys[:, 0] == 0).all())
        or not bool((keys[:, 3] == 0).all())
        or not torch.equal(
            keys[:, 1], packet.dense_slots // pixels_per_view
        )
        or not torch.equal(
            keys[:, 2], packet.dense_slots % pixels_per_view
        )
    ):
        raise ValueError("DepthSplat packet subset descriptor keys drifted from source slots")
    slot_to_index = {int(slot): index for index, slot in enumerate(source_slots)}
    requested_list = requested_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(slot_to_index) != len(source_slots) or any(
        int(slot) not in slot_to_index for slot in requested_list
    ):
        raise ValueError("DepthSplat packet subset requests an unreplayed descriptor")
    indices = torch.tensor(
        [slot_to_index[int(slot)] for slot in requested_list],
        device=packet.dense_slots.device,
        dtype=torch.long,
    )
    trace = dict(packet.source_trace)
    trace.update(
        {
            "producer_request_mask_sha256": trace.get("selection_mask_sha256"),
            "selection_mask_sha256": _tensor_sha256(selection_mask.to(dtype=torch.uint8)),
            "selected_descriptor_sha256": _tensor_sha256(packet.raw_head_descriptors[indices]),
            "selected_rgb_sha256": _tensor_sha256(packet.source_rgb[indices]),
            "packet_selection_kind": "depthsplat-final-selected-output-mask-v1",
            "selection_views": views,
        }
    )
    return DepthSplatSparseRawPacket(
        descriptor_keys=packet.descriptor_keys[indices].clone(),
        raw_head_descriptors=packet.raw_head_descriptors[indices].clone(),
        extrinsics=packet.extrinsics[indices].clone(),
        intrinsics=packet.intrinsics[indices].clone(),
        coordinates=packet.coordinates[indices].clone(),
        depths=packet.depths[indices].clone(),
        mapped_opacities=packet.mapped_opacities[indices].clone(),
        source_rgb=packet.source_rgb[indices].clone(),
        dense_slots=packet.dense_slots[indices].clone(),
        source_trace=trace,
    )


def _strictly_lexicographic(keys: torch.Tensor) -> bool:
    if keys.shape[0] < 2:
        return True
    previous, following = keys[:-1], keys[1:]
    equal_prefix = torch.ones(previous.shape[0], dtype=torch.bool, device=keys.device)
    greater = torch.zeros_like(equal_prefix)
    for column in range(keys.shape[1]):
        greater |= equal_prefix & (following[:, column] > previous[:, column])
        equal_prefix &= following[:, column] == previous[:, column]
    return bool(greater.all())


class DepthSplatPackedGaussianConsumer:
    """Rebatch selected DepthSplat descriptors through the native RGB Adapter."""

    def __init__(self, adapter: Any):
        if not callable(adapter):
            raise TypeError("DepthSplat packed consumer requires a callable Adapter")
        self._adapter = adapter

    def _validate_packet(
        self, packet: DepthSplatSparseRawPacket
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(packet, DepthSplatSparseRawPacket):
            raise TypeError("DepthSplat packed consumer requires DepthSplatSparseRawPacket")
        tensors = (
            packet.descriptor_keys,
            packet.raw_head_descriptors,
            packet.extrinsics,
            packet.intrinsics,
            packet.coordinates,
            packet.depths,
            packet.mapped_opacities,
            packet.source_rgb,
            packet.dense_slots,
        )
        if any(not torch.is_tensor(value) for value in tensors):
            raise ValueError("DepthSplat packet fields must all be tensors")
        (
            keys,
            raw,
            extrinsics,
            intrinsics,
            coordinates,
            depths,
            opacities,
            source_rgb,
            dense_slots,
        ) = tensors
        count = keys.shape[0]
        if (
            keys.dtype != torch.int64
            or keys.shape != (count, 4)
            or raw.ndim != 2
            or raw.shape[0] != count
            or extrinsics.shape != (count, 4, 4)
            or intrinsics.shape != (count, 3, 3)
            or coordinates.shape != (count, 2)
            or depths.shape != (count,)
            or opacities.shape != (count,)
            or source_rgb.shape != (count, 3)
            or dense_slots.dtype != torch.int64
            or dense_slots.shape != (count,)
            or count == 0
        ):
            raise ValueError("DepthSplat packet tensor dimensions are inconsistent")
        if any(value.device != raw.device for value in tensors[2:]):
            raise ValueError("DepthSplat packet tensors must share one device")
        if any(not value.is_floating_point() for value in tensors[1:8]):
            raise ValueError("DepthSplat packet values must be floating point")
        if any(not bool(torch.isfinite(value).all()) for value in tensors[1:8]):
            raise ValueError("DepthSplat packet values must be finite")
        if not _strictly_lexicographic(keys):
            raise ValueError("DepthSplat descriptor keys must be strictly canonical")
        if keys[:, 0].min().item() < 0 or keys[:, 0].max().item() != 0 or not bool((keys[:, 3] == 0).all()):
            raise ValueError("DepthSplat packet requires B=1/S=1 descriptors")
        if dense_slots.numel() > 1 and not bool((dense_slots[1:] > dense_slots[:-1]).all()):
            raise ValueError("DepthSplat dense slots must be strictly increasing")
        adapter_width = getattr(self._adapter, "d_in", None)
        if not isinstance(adapter_width, int) or raw.shape[1] != adapter_width + 3:
            raise ValueError("DepthSplat raw head width must equal Adapter body width plus 3")
        if not bool(torch.equal(opacities, raw[:, 0].sigmoid())):
            raise ValueError("DepthSplat packet opacity must be sigmoid(raw_head[:,0])")
        trace = packet.source_trace
        required_trace = {
            "contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
            "source_bound": True,
            "execution_scope": "depthsplat-dense-regressor-selected-gaussian-head-only",
            "adapter_side_inputs_source_bound": True,
            "adapter_side_inputs_same_scoped_invocation": True,
            "source_rgb_keyword": "input_images",
            "source_rgb_sh_initialization": True,
            "z_depth_geometry": True,
            "head_forward_invocations": 1,
        }
        if not isinstance(trace, Mapping) or any(trace.get(key) != value for key, value in required_trace.items()):
            raise ValueError("DepthSplat packet lacks a source-bound RGB/z-depth trace")
        return tensors

    def convert(
        self,
        packet: DepthSplatSparseRawPacket,
        *,
        image_shape: tuple[int, int],
        native_execution: DepthSplatNativeExecution | None = None,
        native_full_mask: torch.Tensor | None = None,
    ) -> DepthSplatPackedGaussianAttributes:
        """Run selected Adapter work and retain Full attributes from its capture.

        A compact L0/L1 position is converted through the selected RGB Adapter.
        A Full position is native fallback, so its final Gaussian attributes
        are gathered from the same dense encoder invocation rather than being
        recomputed with a differently shaped Adapter call.
        """

        if (
            not isinstance(image_shape, tuple)
            or len(image_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_shape)
        ):
            raise ValueError("DepthSplat image_shape must contain two positive integers")
        (
            _keys,
            raw,
            extrinsics,
            intrinsics,
            coordinates,
            depths,
            opacities,
            source_rgb,
            dense_slots,
        ) = self._validate_packet(packet)
        trace = packet.source_trace
        if (native_execution is None) != (native_full_mask is None):
            raise ValueError(
                "DepthSplat native Full attributes require both execution and mask"
            )
        full_indices: torch.Tensor | None = None
        full_slots: torch.Tensor | None = None
        native_dense: Any | None = None
        native_full_attribute_binding_sha256: str | None = None
        native_execution_sha256: str | None = None
        if native_execution is not None:
            if not isinstance(native_execution, DepthSplatNativeExecution):
                raise TypeError("DepthSplat native Full execution is invalid")
            if (
                not torch.is_tensor(native_full_mask)
                or native_full_mask.dtype != torch.bool
                or native_full_mask.ndim != 3
                or tuple(native_full_mask.shape[-2:]) != image_shape
            ):
                raise ValueError("DepthSplat native Full mask has an invalid shape")
            views, height, width = native_full_mask.shape
            native_execution_sha256 = native_execution.events.get(
                "native_execution_sha256", canonical_json_sha256(native_execution.events)
            )
            if (
                trace.get("native_execution_sha256") != native_execution_sha256
                or trace.get("source_view_count") != views
                or trace.get("source_image_shape") != [height, width]
            ):
                raise ValueError("DepthSplat native Full execution differs from its packet")
            full_mask = native_full_mask.to(device=dense_slots.device)
            full_mask_sha256 = _tensor_sha256(full_mask.to(dtype=torch.uint8))
            full_count = int(full_mask.sum().item())
            if (
                trace.get("native_full_passthrough_mask_sha256") != full_mask_sha256
                or trace.get("native_full_passthrough_positions") != full_count
            ):
                raise ValueError("DepthSplat native Full mask differs from its packet ledger")
            positions = full_mask.nonzero(as_tuple=False)
            full_slots = (
                positions[:, 0] * (height * width)
                + positions[:, 1] * width
                + positions[:, 2]
            ).to(dtype=torch.int64)
            full_indices = torch.searchsorted(dense_slots, full_slots)
            if bool((full_indices >= dense_slots.numel()).any()) or not torch.equal(
                dense_slots[full_indices], full_slots
            ):
                raise ValueError("DepthSplat native Full mask requests an omitted packet slot")
            native_dense = _unwrap_gaussians(native_execution.dense_gaussians)
            total_slots = views * height * width
            if (
                native_dense.means.shape != (1, total_slots, 3)
                or native_dense.covariances.shape != (1, total_slots, 3, 3)
                or native_dense.harmonics.ndim != 4
                or native_dense.harmonics.shape[:3] != (1, total_slots, 3)
                or native_dense.opacities.shape != (1, total_slots)
            ):
                raise ValueError("DepthSplat native Full execution attributes changed")
            adapter_events = native_execution.events.get("adapter")
            if (
                not isinstance(adapter_events, Mapping)
                or adapter_events.get("dense_inputs_binding_sha256")
                != _adapter_inputs_binding_sha256(native_execution.adapter_inputs)
                or adapter_events.get("dense_attribute_binding_sha256")
                != _dense_gaussian_attribute_binding_sha256(
                    native_dense, slots=total_slots
                )
            ):
                raise ValueError("DepthSplat native Full capture binding drifted")
            native_full_attribute_binding_sha256 = depthsplat_attribute_binding_sha256(
                dense_slots=full_slots,
                means=native_dense.means[0, full_slots],
                covariances=native_dense.covariances[0, full_slots],
                harmonics=native_dense.harmonics[0, full_slots],
                opacities=native_dense.opacities[0, full_slots],
            )
        elif trace.get("native_full_passthrough_positions", 0) != 0:
            raise ValueError(
                "DepthSplat packet with Full slots requires native dense attributes"
            )
        count = raw.shape[0]
        all_indices = torch.arange(count, device=dense_slots.device, dtype=torch.long)
        if full_indices is None:
            compact_indices = all_indices
        else:
            compact_mask = torch.ones(count, device=dense_slots.device, dtype=torch.bool)
            compact_mask[full_indices] = False
            compact_indices = compact_mask.nonzero(as_tuple=False).reshape(-1)
        compact_count = int(compact_indices.numel())
        compact_means: torch.Tensor | None = None
        compact_covariances: torch.Tensor | None = None
        compact_harmonics: torch.Tensor | None = None
        compact_opacities: torch.Tensor | None = None
        if compact_count:
            compact_raw = raw[compact_indices]
            compact_dtype = compact_raw.dtype
            adapter_result = self._adapter(
                extrinsics[compact_indices].reshape(1, 1, compact_count, 1, 1, 4, 4),
                intrinsics[compact_indices].reshape(1, 1, compact_count, 1, 1, 3, 3),
                coordinates[compact_indices].reshape(1, 1, compact_count, 1, 1, 2),
                depths[compact_indices]
                .to(dtype=compact_dtype)
                .reshape(1, 1, compact_count, 1, 1),
                opacities[compact_indices]
                .to(dtype=compact_dtype)
                .reshape(1, 1, compact_count, 1, 1),
                compact_raw[:, 3:].reshape(1, 1, compact_count, 1, 1, -1),
                image_shape,
                input_images=source_rgb[compact_indices]
                .to(dtype=compact_dtype)
                .transpose(0, 1)
                .reshape(1, 1, 3, 1, compact_count),
            )
            required = ("means", "covariances", "harmonics", "opacities")
            if any(
                not torch.is_tensor(getattr(adapter_result, name, None))
                for name in required
            ):
                raise ValueError("DepthSplat Adapter result lacks Gaussian attributes")
            compact_means = adapter_result.means.reshape(compact_count, 3)
            compact_covariances = adapter_result.covariances.reshape(
                compact_count, 3, 3
            )
            compact_harmonics = adapter_result.harmonics.reshape(compact_count, 3, -1)
            compact_opacities = adapter_result.opacities.reshape(compact_count)
            for value in (
                compact_means,
                compact_covariances,
                compact_harmonics,
                compact_opacities,
            ):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("DepthSplat Adapter produced non-finite selected attributes")
        if full_indices is None:
            if (
                compact_means is None
                or compact_covariances is None
                or compact_harmonics is None
                or compact_opacities is None
            ):
                raise RuntimeError("DepthSplat selected Adapter unexpectedly had no rows")
            means = compact_means
            covariances = compact_covariances
            harmonics = compact_harmonics
            output_opacities = compact_opacities
        else:
            if full_slots is None or native_dense is None:
                raise RuntimeError("DepthSplat native Full attributes were not captured")
            means = native_dense.means[0, dense_slots].clone()
            covariances = native_dense.covariances[0, dense_slots].clone()
            harmonics = native_dense.harmonics[0, dense_slots].clone()
            output_opacities = native_dense.opacities[0, dense_slots].clone()
            if compact_count:
                if (
                    compact_means is None
                    or compact_covariances is None
                    or compact_harmonics is None
                    or compact_opacities is None
                ):
                    raise RuntimeError("DepthSplat compact Adapter attributes are missing")
                native_values = (
                    means,
                    covariances,
                    harmonics,
                    output_opacities,
                )
                adapter_values = (
                    compact_means,
                    compact_covariances,
                    compact_harmonics,
                    compact_opacities,
                )
                if any(
                    native.device != value.device or native.dtype != value.dtype
                    for native, value in zip(native_values, adapter_values)
                ):
                    raise ValueError("DepthSplat native Full attributes have mixed precision")
                means[compact_indices] = compact_means
                covariances[compact_indices] = compact_covariances
                harmonics[compact_indices] = compact_harmonics
                output_opacities[compact_indices] = compact_opacities
        trace = dict(trace)
        if native_full_attribute_binding_sha256 is not None:
            trace.update(
                {
                    "native_full_adapter_attribute_execution_sha256": native_execution_sha256,
                    "native_full_adapter_attribute_passthrough_mask_sha256": trace[
                        "native_full_passthrough_mask_sha256"
                    ],
                    "native_full_adapter_attribute_binding_sha256": (
                        native_full_attribute_binding_sha256
                    ),
                    "native_full_adapter_attribute_passthrough_count": int(
                        full_slots.numel()
                    ),
                    "selected_native_rgb_adapter_compact_count": compact_count,
                    "selected_native_rgb_adapter_executed": compact_count > 0,
                }
            )
        attribute_binding_sha256 = depthsplat_attribute_binding_sha256(
            dense_slots=dense_slots,
            means=means,
            covariances=covariances,
            harmonics=harmonics,
            opacities=output_opacities,
        )
        trace["native_adapter_attribute_binding_sha256"] = attribute_binding_sha256
        return DepthSplatPackedGaussianAttributes(
            dense_slots=dense_slots.clone(),
            means=means,
            covariances=covariances,
            harmonics=harmonics,
            opacities=output_opacities,
            source_trace=trace,
            source_trace_sha256=canonical_json_sha256(trace),
            attribute_binding_sha256=attribute_binding_sha256,
        )


def compare_depthsplat_packed_to_dense(
    packed: DepthSplatPackedGaussianAttributes, dense_gaussians: Any
) -> dict[str, Any]:
    """Compare all selected native Adapter attributes with dense source slots."""

    if not isinstance(packed, DepthSplatPackedGaussianAttributes):
        raise TypeError("DepthSplat packed comparison requires packed attributes")
    dense = _unwrap_gaussians(dense_gaussians)
    slots = packed.dense_slots
    if slots.dtype != torch.int64 or slots.ndim != 1 or slots.numel() == 0:
        raise ValueError("DepthSplat packed comparison has invalid dense slots")
    count = slots.numel()
    fields = {
        "means": (packed.means, dense.means[0, slots]),
        "covariances": (packed.covariances, dense.covariances[0, slots]),
        "harmonics": (packed.harmonics, dense.harmonics[0, slots]),
        "opacities": (packed.opacities, dense.opacities[0, slots]),
    }
    report: dict[str, Any] = {"atol": FP32_ATOL, "rtol": FP32_RTOL, "count": int(count)}
    equivalent = True
    for name, (actual, expected) in fields.items():
        if actual.shape != expected.shape:
            raise ValueError(f"DepthSplat packed {name} shape differs from dense source")
        delta = (actual - expected).abs()
        field_equivalent = bool(torch.allclose(actual, expected, rtol=FP32_RTOL, atol=FP32_ATOL))
        equivalent &= field_equivalent
        report[name] = {
            "maximum_absolute_delta": float(delta.max().item()),
            "mean_absolute_delta": float(delta.mean().item()),
            "equivalent": field_equivalent,
        }
    report["equivalent"] = equivalent
    return report


def compare_depthsplat_full_passthrough_to_dense_bitwise(
    packed: DepthSplatPackedGaussianAttributes,
    dense_gaussians: Any,
    full_passthrough_mask: torch.Tensor,
) -> dict[str, Any]:
    """Require every declared Full slot to retain native Adapter bits exactly.

    Compact selected outputs are allowed the explicit FP32 replay tolerance
    used by ``compare_depthsplat_packed_to_dense``.  A Full tile is different:
    it is source-native fallback, so this check intentionally accepts no
    numerical drift in means, covariances, SH, or opacity.
    """

    if not isinstance(packed, DepthSplatPackedGaussianAttributes):
        raise TypeError("DepthSplat Full comparison requires packed attributes")
    if (
        not torch.is_tensor(full_passthrough_mask)
        or full_passthrough_mask.dtype != torch.bool
        or full_passthrough_mask.ndim != 3
    ):
        raise ValueError("DepthSplat Full comparison requires a [V,H,W] bool mask")
    views, height, width = full_passthrough_mask.shape
    trace = packed.source_trace
    if (
        not isinstance(trace, Mapping)
        or trace.get("source_view_count") != views
        or trace.get("source_image_shape") != [height, width]
        or trace.get("native_full_passthrough_mask_sha256")
        != _tensor_sha256(full_passthrough_mask.to(dtype=torch.uint8))
        or trace.get("native_full_passthrough_positions")
        != int(full_passthrough_mask.sum().item())
    ):
        raise ValueError("DepthSplat Full comparison packet trace is not bound to its mask")
    dense = _unwrap_gaussians(dense_gaussians)
    slots = packed.dense_slots
    if (
        slots.dtype != torch.int64
        or slots.ndim != 1
        or slots.numel() == 0
        or bool((slots[1:] <= slots[:-1]).any())
    ):
        raise ValueError("DepthSplat Full comparison packed slots are not strictly ordered")
    positions = full_passthrough_mask.nonzero(as_tuple=False).to(device=slots.device)
    if positions.numel() == 0:
        return {"count": 0, "bitwise_equivalent": True, "fields": {}}
    requested_slots = (
        positions[:, 0] * (height * width) + positions[:, 1] * width + positions[:, 2]
    ).to(dtype=torch.int64)
    indices = torch.searchsorted(slots, requested_slots)
    if bool((indices >= slots.numel()).any()) or not torch.equal(slots[indices], requested_slots):
        raise ValueError("DepthSplat Full comparison requests a missing packed slot")
    fields = {
        "means": (packed.means[indices], dense.means[0, requested_slots]),
        "covariances": (
            packed.covariances[indices],
            dense.covariances[0, requested_slots],
        ),
        "harmonics": (packed.harmonics[indices], dense.harmonics[0, requested_slots]),
        "opacities": (packed.opacities[indices], dense.opacities[0, requested_slots]),
    }
    report: dict[str, Any] = {"count": int(requested_slots.numel()), "fields": {}}
    equivalent = True
    for name, (actual, expected) in fields.items():
        if actual.shape != expected.shape:
            raise ValueError(f"DepthSplat Full comparison {name} shape differs from dense source")
        exact = torch.equal(actual, expected)
        equivalent &= exact
        report["fields"][name] = {
            "bitwise_equivalent": exact,
            "maximum_absolute_delta": float((actual - expected).abs().max().item()),
        }
    report["bitwise_equivalent"] = equivalent
    return report
