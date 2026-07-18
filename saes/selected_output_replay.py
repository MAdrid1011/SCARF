"""Exact same-weight replay for retained outputs of a two-convolution head.

TranSplat and MVSplat emit raw Gaussian descriptors through the same spatial
head: ``Conv3x3 -> GELU -> Conv3x3``.  A retained output only needs a 5x5
input halo, but a repeated four-corner-per-T=4 selection makes the first
convolution's closure dense.  This module executes that closure explicitly,
then executes the second convolution only for retained outputs.  It never
changes, trains, or approximates the source head's parameters.

The replay is an execution primitive and an event-accounting boundary.  It is
not a quality result and does not imply that upstream S1/S2/refinement work is
sparse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


REPLAY_CONTRACT_VERSION = "saes-selected-output-replay-v1"
# These fixed tolerances are intentionally not CLI parameters.  They cover
# normal FP32 library-kernel accumulation differences without becoming a
# quality or evaluation tolerance.
FP32_ATOL = 1.0e-5
FP32_RTOL = 1.0e-5


@dataclass(frozen=True)
class SelectedOutputReplay:
    """Selected head outputs and the target-free event ledger."""

    coordinates: torch.Tensor
    values: torch.Tensor
    events: dict[str, Any]


@dataclass(frozen=True)
class SelectedOutputReplayMap:
    """A full-shape raw head map with only selected positions executed."""

    values: torch.Tensor
    selection_mask: torch.Tensor
    events: dict[str, Any]


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _require_supported_conv(conv: nn.Conv2d, *, name: str) -> None:
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"{name} must be nn.Conv2d")
    if (
        _pair(conv.kernel_size) != (3, 3)
        or _pair(conv.stride) != (1, 1)
        or _pair(conv.dilation) != (1, 1)
        or _pair(conv.padding) != (1, 1)
        or conv.groups != 1
        or conv.padding_mode != "zeros"
    ):
        raise ValueError(
            f"{name} must be an ungrouped zero-padded stride-1 3x3 convolution"
        )


def unpack_two_conv_head(head: nn.Module) -> tuple[nn.Conv2d, nn.Module, nn.Conv2d]:
    """Validate and return the exact ``Conv3x3 -> activation -> Conv3x3`` head."""
    if not isinstance(head, nn.Sequential) or len(head) != 3:
        raise ValueError(
            "selected-output replay requires a three-stage nn.Sequential head"
        )
    first, activation, second = tuple(head)
    _require_supported_conv(first, name="head[0]")
    _require_supported_conv(second, name="head[2]")
    if not isinstance(activation, nn.GELU):
        raise ValueError("selected-output replay requires the original nn.GELU")
    if first.out_channels != second.in_channels:
        raise ValueError("head convolution channel dimensions are inconsistent")
    return first, activation, second


def selected_output_coordinates(selection_mask: torch.Tensor) -> torch.Tensor:
    """Return row-major retained coordinates from a single ``[H,W]`` mask."""
    if selection_mask.ndim != 2 or selection_mask.dtype != torch.bool:
        raise ValueError("selection_mask must be one two-dimensional bool tensor")
    coordinates = selection_mask.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("selected-output replay requires at least one retained output")
    return coordinates


def _same_conv3_closure(
    coordinates: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    """Return row-major valid input positions for a zero-padded 3x3 output set."""
    offsets = torch.arange(-1, 2, device=coordinates.device)
    rows = coordinates[:, 0, None, None] + offsets[None, :, None]
    columns = coordinates[:, 1, None, None] + offsets[None, None, :]
    valid = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    linear = rows * width + columns
    return torch.unique(linear[valid], sorted=True)


def _linear_to_coordinates(linear: torch.Tensor, *, width: int) -> torch.Tensor:
    return torch.stack((torch.div(linear, width, rounding_mode="floor"), linear % width), dim=1)


def _gather_zero_padded_patches(
    activations: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    """Gather one native 3x3 patch per valid output coordinate.

    Padding is materialized only as a one-pixel border.  No dense hidden
    activation map is evaluated by this helper.
    """
    if activations.ndim != 4:
        raise ValueError("head input must be [N,C,H,W]")
    height, width = activations.shape[-2:]
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must be [K,2]")
    padded = F.pad(activations, (1, 1, 1, 1))
    offsets = torch.arange(3, device=activations.device)
    rows = coordinates[:, 0, None, None] + offsets[None, :, None]
    columns = coordinates[:, 1, None, None] + offsets[None, None, :]
    if bool((rows < 0).any()) or bool((rows >= height + 2).any()):
        raise ValueError("patch coordinate is outside the zero-padded activation")
    if bool((columns < 0).any()) or bool((columns >= width + 2).any()):
        raise ValueError("patch coordinate is outside the zero-padded activation")
    # Advanced indexing yields [N,C,K,3,3]; the convolution batch is K.
    return padded[:, :, rows, columns].permute(0, 2, 1, 3, 4).contiguous()


def _apply_conv_to_patches(conv: nn.Conv2d, patches: torch.Tensor) -> torch.Tensor:
    """Run the original kernel on independent 3x3 patches."""
    batch, count, channels, patch_h, patch_w = patches.shape
    if channels != conv.in_channels or (patch_h, patch_w) != (3, 3):
        raise ValueError("patches do not match the original convolution input")
    output = F.conv2d(
        patches.reshape(batch * count, channels, patch_h, patch_w),
        conv.weight,
        conv.bias,
    )
    return output.reshape(batch, count, conv.out_channels)


def _hidden_patches_for_selected_outputs(
    hidden_values: torch.Tensor,
    hidden_linear: torch.Tensor,
    selected_coordinates: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Gather zero-padded hidden 3x3 neighborhoods for selected final outputs."""
    batch, hidden_count, channels = hidden_values.shape
    if hidden_count != hidden_linear.numel():
        raise ValueError("hidden values and coordinate map disagree")
    lookup = torch.full(
        (height * width,), -1, device=hidden_values.device, dtype=torch.long
    )
    lookup[hidden_linear] = torch.arange(
        hidden_count, device=hidden_values.device, dtype=torch.long
    )
    offsets = torch.arange(-1, 2, device=hidden_values.device)
    rows = selected_coordinates[:, 0, None, None] + offsets[None, :, None]
    columns = selected_coordinates[:, 1, None, None] + offsets[None, None, :]
    valid = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    safe_linear = (
        rows.clamp(0, height - 1) * width + columns.clamp(0, width - 1)
    )
    slots = lookup[safe_linear]
    if bool((slots[valid] < 0).any()):
        raise RuntimeError("first-convolution closure omitted a required hidden output")
    gathered = hidden_values[:, slots.clamp_min(0).reshape(-1), :]
    patches = gathered.reshape(batch, -1, 3, 3, channels).permute(0, 1, 4, 2, 3)
    return patches * valid.to(dtype=patches.dtype).view(1, -1, 1, 3, 3)


def _conv_macs_per_position(conv: nn.Conv2d) -> int:
    return conv.in_channels * conv.out_channels * 3 * 3


def replay_two_conv_selected_outputs(
    head: nn.Module,
    head_input: torch.Tensor,
    selection_mask: torch.Tensor,
) -> SelectedOutputReplay:
    """Replay retained outputs with the original head weights and activation.

    The returned tensor has shape ``[N,C,K]`` in the row-major coordinate order
    returned alongside it.  The first convolution executes only the exact
    closure needed by the second convolution; the second executes only ``K``
    retained output positions.
    """
    if head_input.ndim != 4:
        raise ValueError("head_input must have shape [N,C,H,W]")
    first, activation, second = unpack_two_conv_head(head)
    if head_input.shape[1] != first.in_channels:
        raise ValueError("head_input channel count does not match head[0]")
    height, width = head_input.shape[-2:]
    if tuple(selection_mask.shape) != (height, width):
        raise ValueError("selection_mask must match head_input spatial dimensions")
    if selection_mask.device != head_input.device:
        selection_mask = selection_mask.to(head_input.device)
    selected_coordinates = selected_output_coordinates(selection_mask)
    selected_linear = selected_coordinates[:, 0] * width + selected_coordinates[:, 1]
    hidden_linear = _same_conv3_closure(
        selected_coordinates, height=height, width=width
    )
    hidden_coordinates = _linear_to_coordinates(hidden_linear, width=width)

    first_patches = _gather_zero_padded_patches(head_input, hidden_coordinates)
    hidden_values = activation(_apply_conv_to_patches(first, first_patches))
    second_patches = _hidden_patches_for_selected_outputs(
        hidden_values,
        hidden_linear,
        selected_coordinates,
        height=height,
        width=width,
    )
    selected_values = _apply_conv_to_patches(second, second_patches).transpose(1, 2)

    input_halo_linear = _same_conv3_closure(
        hidden_coordinates, height=height, width=width
    )
    dense_positions = height * width
    first_macs = _conv_macs_per_position(first)
    second_macs = _conv_macs_per_position(second)
    replay_macs = hidden_linear.numel() * first_macs + selected_linear.numel() * second_macs
    dense_macs = dense_positions * (first_macs + second_macs)
    events = {
        "contract_version": REPLAY_CONTRACT_VERSION,
        "head_structure": "Conv3x3->GELU->Conv3x3",
        "dense_spatial_positions": dense_positions,
        "selected_final_output_positions": int(selected_linear.numel()),
        "first_conv_required_output_positions": int(hidden_linear.numel()),
        "first_conv_required_input_halo_positions": int(input_halo_linear.numel()),
        "first_conv_dense_closure": int(hidden_linear.numel()) == dense_positions,
        "second_conv_selected_only": True,
        "dense_head_macs": dense_macs,
        "replayed_head_macs": replay_macs,
        "head_mac_saving": 1.0 - replay_macs / dense_macs,
        "upstream_s2_saving": 0.0,
    }
    return SelectedOutputReplay(
        coordinates=selected_coordinates,
        values=selected_values,
        events=events,
    )


def dense_selected_outputs(
    dense_head_output: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    """Extract dense reference values in the replay's coordinate order."""
    if dense_head_output.ndim != 4:
        raise ValueError("dense_head_output must be [N,C,H,W]")
    return dense_head_output[:, :, coordinates[:, 0], coordinates[:, 1]]


def compare_selected_outputs(
    dense_head_output: torch.Tensor, replay: SelectedOutputReplay
) -> dict[str, Any]:
    """Return the fixed FP32 numerical-equivalence verdict for one replay."""
    reference = dense_selected_outputs(dense_head_output, replay.coordinates)
    if reference.shape != replay.values.shape:
        raise ValueError("dense and replayed selected output shapes differ")
    delta = (reference - replay.values).abs()
    if dense_head_output.dtype not in (torch.float32, torch.float64):
        raise ValueError("selected-output equivalence audit requires FP32 or FP64")
    allclose = torch.allclose(
        reference, replay.values, rtol=FP32_RTOL, atol=FP32_ATOL
    )
    return {
        "reference_shape": list(reference.shape),
        "dtype": str(dense_head_output.dtype).replace("torch.", ""),
        "atol": FP32_ATOL,
        "rtol": FP32_RTOL,
        "maximum_absolute_delta": float(delta.max().item()),
        "mean_absolute_delta": float(delta.mean().item()),
        "equivalent": bool(allclose),
    }


def replay_two_conv_selected_output_maps(
    head: nn.Module,
    head_input: torch.Tensor,
    selection_mask: torch.Tensor,
    *,
    dense_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> SelectedOutputReplayMap:
    """Execute a possibly different retained mask for every head batch item.

    Classic encoders flatten context views into the head batch. A two-dimensional
    mask would incorrectly impose one view's route on every other view, so this
    wrapper consumes ``[N,H,W]`` masks. Unselected output slots are zero-filled;
    a correct later materialization path must never read those descriptors.
    """
    if head_input.ndim != 4:
        raise ValueError("head_input must have shape [N,C,H,W]")
    first, activation, second = unpack_two_conv_head(head)
    batch, channels, height, width = head_input.shape
    if channels != first.in_channels:
        raise ValueError("head_input channel count does not match head[0]")
    if selection_mask.ndim == 2:
        selection_mask = selection_mask.unsqueeze(0).expand(batch, -1, -1)
    if (
        selection_mask.ndim != 3
        or selection_mask.shape != (batch, height, width)
        or selection_mask.dtype != torch.bool
    ):
        raise ValueError("selection_mask must have shape [N,H,W] and bool dtype")
    selection_mask = selection_mask.to(head_input.device)
    output = torch.zeros(
        (batch, second.out_channels, height, width),
        dtype=head_input.dtype,
        device=head_input.device,
    )
    per_item_events = []
    for item in range(batch):
        item_mask = selection_mask[item]
        selected = int(item_mask.sum().item())
        if selected == height * width:
            # Full tiles use the native dense head and are charged as such.
            dense = (
                dense_forward(head_input[item : item + 1])
                if dense_forward is not None
                else head(head_input[item : item + 1])
            )
            if dense.shape != (1, second.out_channels, height, width):
                raise ValueError("dense head execution returned an incompatible shape")
            output[item : item + 1] = dense
            first_macs = _conv_macs_per_position(first)
            second_macs = _conv_macs_per_position(second)
            event = {
                "dense_spatial_positions": height * width,
                "selected_final_output_positions": selected,
                "first_conv_required_output_positions": height * width,
                "first_conv_required_input_halo_positions": height * width,
                "first_conv_dense_closure": True,
                "second_conv_selected_only": False,
                "dense_head_macs": (height * width) * (first_macs + second_macs),
                "replayed_head_macs": (height * width) * (first_macs + second_macs),
                "head_mac_saving": 0.0,
            }
        else:
            replay = replay_two_conv_selected_outputs(
                head, head_input[item : item + 1], item_mask
            )
            coordinates = replay.coordinates
            output[item, :, coordinates[:, 0], coordinates[:, 1]] = replay.values[0]
            event = dict(replay.events)
        event["batch_item"] = item
        per_item_events.append(event)

    dense_head_macs = sum(int(event["dense_head_macs"]) for event in per_item_events)
    actual_head_macs = sum(int(event["replayed_head_macs"]) for event in per_item_events)
    selected_positions = sum(
        int(event["selected_final_output_positions"]) for event in per_item_events
    )
    return SelectedOutputReplayMap(
        values=output,
        selection_mask=selection_mask,
        events={
            "contract_version": REPLAY_CONTRACT_VERSION,
            "batch_size": batch,
            "head_structure": "Conv3x3->GELU->Conv3x3",
            "dense_head_macs": dense_head_macs,
            "actual_head_macs": actual_head_macs,
            "head_mac_delta": dense_head_macs - actual_head_macs,
            "head_mac_saving": 1.0 - actual_head_macs / dense_head_macs,
            "selected_final_output_positions": selected_positions,
            "omitted_final_output_positions": batch * height * width - selected_positions,
            "dense_batch_items": sum(
                not event["second_conv_selected_only"] for event in per_item_events
            ),
            "selected_output_batch_items": sum(
                event["second_conv_selected_only"] for event in per_item_events
            ),
            "upstream_s2_saving": 0.0,
            "per_item": per_item_events,
        },
    )
