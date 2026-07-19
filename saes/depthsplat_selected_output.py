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


@dataclass(frozen=True)
class DepthSplatSelectedHeadReplay:
    """Selected raw head outputs in their original ``[V,C,H,W]`` layout."""

    values: torch.Tensor
    selection_mask: torch.Tensor
    events: dict[str, Any]
    equivalence: dict[str, Any]


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
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not torch.is_tensor(value):
            raise TypeError("DepthSplat module state contains a non-tensor entry")
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


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
    adapter.forward = captured_adapter_forward
    try:
        with torch.no_grad():
            dense_gaussians = _unwrap_gaussians(
                encoder(context, global_step=0, deterministic=True)
            )
    finally:
        regressor_handle.remove()
        head_handle.remove()
        adapter.forward = original_adapter_forward
    if set(captured) != {"regressor", "head", "adapter"}:
        raise RuntimeError("DepthSplat native capture did not reach every required boundary")
    head_input, dense_raw_head = captured["head"]
    adapter_inputs = captured["adapter"]
    if (
        head_input.ndim != 4
        or dense_raw_head.ndim != 4
        or head_input.shape[0] != views
        or dense_raw_head.shape[0] != views
        or tuple(head_input.shape[-2:]) != (height, width)
        or tuple(dense_raw_head.shape[-2:]) != (height, width)
    ):
        raise RuntimeError("DepthSplat raw head layout changed")
    _validate_dense_adapter_inputs(
        adapter_inputs, views=views, height=height, width=width
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
    return DepthSplatNativeExecution(
        dense_gaussians=dense_gaussians,
        gaussian_head_input=head_input,
        dense_raw_head=dense_raw_head,
        adapter_inputs=adapter_inputs,
        sample_image_grid=sample_image_grid,
        events={
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
            },
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        },
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
    head: nn.Module, *, height: int, width: int, batch_item: int
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
        "selected_final_output_positions": positions,
        "first_conv_required_output_positions": positions,
        "first_conv_dense_closure": True,
        "second_conv_selected_only": False,
        "dense_head_macs": macs,
        "replayed_head_macs": macs,
        "head_mac_saving": 0.0,
        "padding_mode": "replicate",
    }


def replay_depthsplat_selected_head(
    head: nn.Module,
    head_input: torch.Tensor,
    dense_raw_head: torch.Tensor,
    selection_mask: torch.Tensor,
) -> DepthSplatSelectedHeadReplay:
    """Replay selected native final-head outputs without reading omitted slots."""

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
    output = torch.zeros_like(dense_raw_head)
    per_view: list[dict[str, Any]] = []
    selected_reference: list[torch.Tensor] = []
    selected_values: list[torch.Tensor] = []
    for view in range(views):
        mask = selection_mask[view]
        selected = int(mask.sum().item())
        if selected == 0:
            per_view.append(
                {
                    "batch_item": view,
                    "source_native_dense_head_capture": False,
                    "selected_final_output_positions": 0,
                    "dense_head_macs": 0,
                    "replayed_head_macs": 0,
                    "head_mac_saving": 0.0,
                    "padding_mode": "replicate",
                }
            )
            continue
        if selected == height * width:
            output[view] = dense_raw_head[view]
            per_view.append(
                _head_event_from_dense_source(head, height=height, width=width, batch_item=view)
            )
            selected_reference.append(dense_raw_head[view].reshape(channels, -1))
            selected_values.append(dense_raw_head[view].reshape(channels, -1))
            continue
        replay = replay_two_conv_selected_outputs(head, head_input[view : view + 1], mask)
        output[view, :, replay.coordinates[:, 0], replay.coordinates[:, 1]] = replay.values[0]
        per_view.append({"batch_item": view, **replay.events})
        selected_reference.append(
            dense_raw_head[view, :, replay.coordinates[:, 0], replay.coordinates[:, 1]]
        )
        selected_values.append(replay.values[0])
    reference = torch.cat(selected_reference, dim=1)
    replayed = torch.cat(selected_values, dim=1)
    delta = (reference - replayed).abs()
    equivalent = bool(torch.allclose(reference, replayed, rtol=FP32_RTOL, atol=FP32_ATOL))
    dense_macs = sum(int(event["dense_head_macs"]) for event in per_view)
    actual_macs = sum(int(event["replayed_head_macs"]) for event in per_view)
    return DepthSplatSelectedHeadReplay(
        values=output,
        selection_mask=selection_mask,
        events={
            "contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
            "padding_mode": "replicate",
            "batch_size": views,
            "dense_head_macs": dense_macs,
            "actual_head_macs": actual_macs,
            "head_mac_delta": dense_macs - actual_macs,
            "head_mac_saving": 1.0 - actual_macs / dense_macs if dense_macs else 0.0,
            "selected_final_output_positions": int(selection_mask.sum().item()),
            "omitted_final_output_positions": views * height * width - int(selection_mask.sum().item()),
            "per_view": per_view,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
        },
        equivalence={
            "atol": FP32_ATOL,
            "rtol": FP32_RTOL,
            "maximum_absolute_delta": float(delta.max().item()),
            "mean_absolute_delta": float(delta.mean().item()),
            "equivalent": equivalent,
        },
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
    if replay.values.shape != execution.dense_raw_head.shape:
        raise ValueError("DepthSplat selected replay output shape changed")
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
        "selection_mask_sha256": _tensor_sha256(selection_mask.to(dtype=torch.uint8)),
        "selected_descriptor_sha256": _tensor_sha256(raw),
        "selected_rgb_sha256": _tensor_sha256(source_rgb),
        "adapter_body_width": body_width,
    }
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

    def convert(self, packet: DepthSplatSparseRawPacket, *, image_shape: tuple[int, int]) -> DepthSplatPackedGaussianAttributes:
        """Run one rebatched native Adapter call using only selected RGB pixels."""

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
        count = raw.shape[0]
        dtype = raw.dtype
        adapter_result = self._adapter(
            extrinsics.reshape(1, 1, count, 1, 1, 4, 4),
            intrinsics.reshape(1, 1, count, 1, 1, 3, 3),
            coordinates.reshape(1, 1, count, 1, 1, 2),
            depths.to(dtype=dtype).reshape(1, 1, count, 1, 1),
            opacities.to(dtype=dtype).reshape(1, 1, count, 1, 1),
            raw[:, 3:].reshape(1, 1, count, 1, 1, -1),
            image_shape,
            input_images=source_rgb.to(dtype=dtype).transpose(0, 1).reshape(1, 1, 3, 1, count),
        )
        required = ("means", "covariances", "harmonics", "opacities")
        if any(not torch.is_tensor(getattr(adapter_result, name, None)) for name in required):
            raise ValueError("DepthSplat Adapter result lacks Gaussian attributes")
        means = adapter_result.means.reshape(count, 3)
        covariances = adapter_result.covariances.reshape(count, 3, 3)
        harmonics = adapter_result.harmonics.reshape(count, 3, -1)
        output_opacities = adapter_result.opacities.reshape(count)
        for value in (means, covariances, harmonics, output_opacities):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("DepthSplat Adapter produced non-finite selected attributes")
        trace = dict(packet.source_trace)
        return DepthSplatPackedGaussianAttributes(
            dense_slots=dense_slots.clone(),
            means=means,
            covariances=covariances,
            harmonics=harmonics,
            opacities=output_opacities,
            source_trace=trace,
            source_trace_sha256=canonical_json_sha256(trace),
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
