"""Trace executed PyTorch layers into deterministic SCARF hardware cycles."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch import nn

from encoder import ActivationType, ActivationUnit, ConvEngine
from encoder.mmcu_events import mmcu_stage


@dataclass(frozen=True)
class ModuleCycleTrace:
    output: Any
    total_cycles: int
    breakdown: dict[str, int]


def run_module_with_cycle_trace(
    module: nn.Module, *args: Any, mmcu_stage_name: str | None = None, **kwargs: Any
) -> ModuleCycleTrace:
    """Execute a module once and count its mapped Conv/activation layers."""
    conv = ConvEngine()
    activations = {
        nn.ReLU: ("relu", ActivationUnit(ActivationType.RELU)),
        nn.GELU: ("gelu", ActivationUnit(ActivationType.GELU)),
        nn.SiLU: ("silu", ActivationUnit(ActivationType.SILU)),
        nn.Sigmoid: ("sigmoid", ActivationUnit(ActivationType.SIGMOID)),
    }
    breakdown = {
        "conv2d": 0,
        "conv_transpose2d": 0,
        "relu": 0,
        "gelu": 0,
        "silu": 0,
        "sigmoid": 0,
    }
    handles = []

    def hook(layer: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            return
        input_tensor = inputs[0]
        if isinstance(layer, nn.Conv2d):
            context = (
                mmcu_stage(mmcu_stage_name)
                if mmcu_stage_name is not None
                else contextlib.nullcontext()
            )
            with context:
                stats = conv._compute_cycles(
                    input_tensor, layer.weight, output, layer.stride[0]
                )
            breakdown["conv2d"] += stats.total_cycles
        elif isinstance(layer, nn.ConvTranspose2d):
            context = (
                mmcu_stage(mmcu_stage_name)
                if mmcu_stage_name is not None
                else contextlib.nullcontext()
            )
            with context:
                stats = conv._compute_transposed_cycles(
                    input_tensor, layer.weight, output
                )
            breakdown["conv_transpose2d"] += stats.total_cycles
        else:
            for layer_type, (name, unit) in activations.items():
                if isinstance(layer, layer_type):
                    breakdown[name] += unit._compute_cycles(output).total_cycles
                    break

    for child in module.modules():
        if isinstance(
            child,
            (nn.Conv2d, nn.ConvTranspose2d, nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid),
        ):
            handles.append(child.register_forward_hook(hook))

    try:
        output = module(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    total_cycles = sum(breakdown.values())
    if total_cycles <= 0:
        raise ValueError("module cycle trace executed no supported hardware layers")
    return ModuleCycleTrace(
        output=output,
        total_cycles=total_cycles,
        breakdown={key: value for key, value in breakdown.items() if value > 0},
    )


def run_callable_with_module_cycle_trace(
    function: Callable[[], Any], modules: Iterable[nn.Module], *, mmcu_stage_name: str | None = None
) -> ModuleCycleTrace:
    """Trace selected module trees while a larger callable executes once."""
    conv = ConvEngine()
    activations = {
        nn.ReLU: ("relu", ActivationUnit(ActivationType.RELU)),
        nn.GELU: ("gelu", ActivationUnit(ActivationType.GELU)),
        nn.SiLU: ("silu", ActivationUnit(ActivationType.SILU)),
        nn.Sigmoid: ("sigmoid", ActivationUnit(ActivationType.SIGMOID)),
    }
    breakdown = {
        "conv2d": 0,
        "conv_transpose2d": 0,
        "relu": 0,
        "gelu": 0,
        "silu": 0,
        "sigmoid": 0,
    }
    handles = []
    registered: set[int] = set()

    def hook(layer: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            return
        input_tensor = inputs[0]
        if isinstance(layer, nn.Conv2d):
            context = (
                mmcu_stage(mmcu_stage_name)
                if mmcu_stage_name is not None
                else contextlib.nullcontext()
            )
            with context:
                stats = conv._compute_cycles(
                    input_tensor, layer.weight, output, layer.stride[0]
                )
            breakdown["conv2d"] += stats.total_cycles
        elif isinstance(layer, nn.ConvTranspose2d):
            context = (
                mmcu_stage(mmcu_stage_name)
                if mmcu_stage_name is not None
                else contextlib.nullcontext()
            )
            with context:
                stats = conv._compute_transposed_cycles(
                    input_tensor, layer.weight, output
                )
            breakdown["conv_transpose2d"] += stats.total_cycles
        else:
            for layer_type, (name, unit) in activations.items():
                if isinstance(layer, layer_type):
                    breakdown[name] += unit._compute_cycles(output).total_cycles
                    break

    supported = (
        nn.Conv2d,
        nn.ConvTranspose2d,
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.Sigmoid,
    )
    for module in modules:
        for child in module.modules():
            if isinstance(child, supported) and id(child) not in registered:
                registered.add(id(child))
                handles.append(child.register_forward_hook(hook))

    try:
        output = function()
    finally:
        for handle in handles:
            handle.remove()

    total_cycles = sum(breakdown.values())
    if total_cycles <= 0:
        raise ValueError("selected module trace executed no supported hardware layers")
    return ModuleCycleTrace(
        output=output,
        total_cycles=total_cycles,
        breakdown={key: value for key, value in breakdown.items() if value > 0},
    )
