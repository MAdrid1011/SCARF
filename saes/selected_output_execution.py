"""Scoped native-module execution for same-weight selected-output replay."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
import torch.nn as nn

from saes.selected_output_replay import replay_two_conv_selected_output_maps


@dataclass
class SelectedOutputExecutionTrace:
    """Actual head calls made while a scoped replay override is installed."""

    selection_mask: torch.Tensor
    invocations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def events(self) -> dict[str, Any]:
        if len(self.invocations) != 1:
            raise RuntimeError(
                "selected-output quality execution requires exactly one raw-head call"
            )
        return dict(self.invocations[0])


@contextmanager
def selected_output_head_execution(
    head: nn.Module, selection_mask: torch.Tensor
) -> Iterator[SelectedOutputExecutionTrace]:
    """Replace one native head call with a per-view selected-output execution.

    This changes only the head's execution schedule. The original module object,
    weights, activation, and all upstream encoder modules remain intact, and
    the original ``forward`` method is restored even if the enclosing encoder
    raises.
    """
    if not isinstance(head, nn.Module):
        raise TypeError("selected-output execution requires an nn.Module head")
    if selection_mask.ndim != 3 or selection_mask.dtype != torch.bool:
        raise ValueError("selected-output execution requires a [N,H,W] bool mask")
    original_forward = head.forward
    had_instance_forward = "forward" in head.__dict__
    trace = SelectedOutputExecutionTrace(selection_mask=selection_mask.detach().clone())

    def sparse_forward(head_input: torch.Tensor) -> torch.Tensor:
        replay = replay_two_conv_selected_output_maps(
            head,
            head_input,
            selection_mask,
            dense_forward=original_forward,
        )
        trace.invocations.append(dict(replay.events))
        return replay.values

    head.forward = sparse_forward  # type: ignore[method-assign]
    try:
        yield trace
    finally:
        if had_instance_forward:
            head.forward = original_forward  # type: ignore[method-assign]
        else:
            delattr(head, "forward")
