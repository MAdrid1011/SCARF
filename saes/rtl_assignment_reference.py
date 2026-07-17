"""Exact integer reference for the staged SAES assignment normalizer.

The caller supplies nonnegative bilateral-kernel scores. This module neither
computes those scores nor reads features, depths, target RGB, quality metrics,
or paper results. It only makes the finite-precision normalization and residual
rule explicit for the Chisel data-path contract.
"""

from __future__ import annotations

from collections.abc import Sequence


MAX_ANCHORS = 8


def normalize_assignment_scores(
    scores: Sequence[int],
    *,
    anchor_count: int,
    fraction_bits: int = 16,
) -> dict[str, int | tuple[int, ...]]:
    """Normalize active nonnegative scores into one exact Q0.fraction_bits sum."""
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int):
        raise ValueError("anchor_count must be an integer")
    if not 1 <= anchor_count <= MAX_ANCHORS:
        raise ValueError(f"anchor_count must be in [1, {MAX_ANCHORS}]")
    if isinstance(fraction_bits, bool) or not isinstance(fraction_bits, int):
        raise ValueError("fraction_bits must be an integer")
    if fraction_bits <= 0:
        raise ValueError("fraction_bits must be positive")
    if len(scores) < anchor_count:
        raise ValueError("scores must cover every active anchor")

    active_scores: list[int] = []
    for index, value in enumerate(scores[:anchor_count]):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"scores[{index}] must be a nonnegative integer")
        active_scores.append(value)
    score_sum = sum(active_scores)
    if score_sum == 0:
        raise ValueError("active score sum must be positive")

    scale = 1 << fraction_bits
    weights = [(score * scale) // score_sum for score in active_scores]
    residual_anchor = max(
        range(anchor_count), key=lambda index: (active_scores[index], -index)
    )
    weights[residual_anchor] += scale - sum(weights)
    return {
        "weights": tuple(weights + [0] * (MAX_ANCHORS - anchor_count)),
        "weight_sum": sum(weights),
        "residual_anchor": residual_anchor,
    }
