"""Small runtime helpers for reproducible paired neural executions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchRNGState:
    cpu: torch.Tensor
    cuda: tuple[torch.Tensor, ...]


def capture_torch_rng_state() -> TorchRNGState:
    """Capture the CPU and all visible CUDA random-number streams."""
    cuda = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return TorchRNGState(cpu=torch.random.get_rng_state().clone(), cuda=cuda)


def restore_torch_rng_state(state: TorchRNGState) -> None:
    """Restore a state captured before a paired reference execution."""
    torch.random.set_rng_state(state.cpu)
    if state.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("captured CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(list(state.cuda))
