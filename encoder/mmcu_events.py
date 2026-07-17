"""Event-level MMCU slot accounting for executed tensor operations."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import math
from collections.abc import Iterable, Iterator
from typing import Any, Callable

import torch
from torch import nn


_STAGES = ("s1", "s2", "s3")
_CURRENT_STAGE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "scarf_mmcu_stage", default=None
)
_EVENTS: dict[str, dict[str, int]] = {}


def reset_mmcu_events() -> None:
    global _EVENTS
    _EVENTS = {
        stage: {"useful_mmcu_slots": 0, "scheduled_mmcu_slots": 0, "operations": 0}
        for stage in _STAGES
    }


@contextlib.contextmanager
def mmcu_stage(stage: str) -> Iterator[None]:
    if stage not in _STAGES:
        raise ValueError(f"invalid MMCU stage: {stage}")
    token = _CURRENT_STAGE.set(stage)
    try:
        yield
    finally:
        _CURRENT_STAGE.reset(token)


def recording_stage(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with mmcu_stage(stage):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def record_mmcu_slots(useful: int, scheduled: int) -> None:
    stage = _CURRENT_STAGE.get()
    if stage is None:
        return
    if (
        not isinstance(useful, int)
        or isinstance(useful, bool)
        or not isinstance(scheduled, int)
        or isinstance(scheduled, bool)
        or useful <= 0
        or scheduled < useful
    ):
        raise ValueError("MMCU slots must satisfy 0 < useful <= scheduled")
    if not _EVENTS:
        reset_mmcu_events()
    _EVENTS[stage]["useful_mmcu_slots"] += useful
    _EVENTS[stage]["scheduled_mmcu_slots"] += scheduled
    _EVENTS[stage]["operations"] += 1


def record_conv_tensors(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    array_size: int = 32,
) -> None:
    batch = int(output.shape[0])
    m = int(output.shape[2]) * int(output.shape[3])
    n = int(output.shape[1])
    k = int(weight.shape[1]) * int(weight.shape[2]) * int(weight.shape[3])
    record_gemm_shape(
        batch=batch,
        m=m,
        n=n,
        k=k,
        tile_m=array_size,
        tile_n=array_size,
    )


def record_gemm_shape(
    *, batch: int, m: int, n: int, k: int, tile_m: int = 32, tile_n: int = 32
) -> None:
    useful = int(batch) * int(m) * int(n) * int(k)
    tile_count = int(batch) * math.ceil(m / tile_m) * math.ceil(n / tile_n)
    scheduled_cycles = tile_count * (k + tile_m + tile_n)
    record_mmcu_slots(useful, scheduled_cycles * tile_m * tile_n)


def record_linear_tensors(input_tensor: torch.Tensor, layer: nn.Linear) -> None:
    m = input_tensor.numel() // layer.in_features
    record_gemm_shape(batch=1, m=m, n=layer.out_features, k=layer.in_features)


@contextlib.contextmanager
def trace_torch_mmcu_modules(
    module: nn.Module,
    stage: str,
    *,
    exclude_modules: Iterable[nn.Module] = (),
) -> Iterator[None]:
    """Count executed Conv/Linear shapes in an accurate PyTorch module path."""
    excluded = {
        id(child)
        for excluded_module in exclude_modules
        for child in excluded_module.modules()
    }
    handles = []

    def hook(layer: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            return
        with mmcu_stage(stage):
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                record_conv_tensors(inputs[0], layer.weight, output)
            elif isinstance(layer, nn.Linear):
                record_linear_tensors(inputs[0], layer)

    for child in module.modules():
        if id(child) not in excluded and isinstance(
            child, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
        ):
            handles.append(child.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def mmcu_stage_records(cycles: dict[str, int]) -> dict[str, dict[str, Any]]:
    if not _EVENTS:
        reset_mmcu_events()
    records = {}
    for stage, component in zip(("s1", "s2", "s3", "s4"), ("feature", "depth", "gaussian", "ggu")):
        event = _EVENTS.get(stage)
        available = bool(event and event["scheduled_mmcu_slots"] > 0)
        records[stage] = {
            "cycles": int(cycles[component]),
            "useful_mmcu_slots": event["useful_mmcu_slots"] if available else 0,
            "scheduled_mmcu_slots": event["scheduled_mmcu_slots"] if available else 0,
            "mmcu_slots_available": available,
            "source": (
                "executed_tensor_shape_mmcu_schedule"
                if available
                else "stage_has_no_recorded_mmcu_operation"
            ),
        }
    return records


reset_mmcu_events()
