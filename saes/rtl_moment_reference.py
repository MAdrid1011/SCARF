"""Exact integer reference for the staged SAES scalar RTL moment lane.

This helper consumes only synthetic or already-selected descriptor values and
assignment weights. It has no routing, image, target-RGB, quality, or paper
result input. The representation unit is deliberately caller-defined: means
use one signed fixed-point unit, variances use its square, and all weights use
one shared nonnegative unit.
"""

from __future__ import annotations

from collections.abc import Iterable


def _integer(value: int, name: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _trunc_div(numerator: int, denominator: int) -> int:
    """Integer division with the signed hardware contract's zero truncation."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return numerator // denominator if numerator >= 0 else -((-numerator) // denominator)


def scalar_moment_merge(
    *,
    base_weight: int,
    base_mean: int,
    base_variance: int,
    updates: Iterable[tuple[int, int, int]],
) -> dict[str, int]:
    """Merge one scalar Gaussian coordinate by first/second moments.

    Args:
        base_weight: Positive anchor mass in the shared integer weight unit.
        base_mean: Signed base descriptor value.
        base_variance: Nonnegative intrinsic variance in squared value units.
        updates: Ordered ``(weight, mean, variance)`` pseudo-descriptor tuples.

    Returns:
        Integer mass, zero-truncated mean, clamped variance, and accepted count.
    """
    base_weight = _integer(base_weight, "base_weight", nonnegative=True)
    if base_weight == 0:
        raise ValueError("base_weight must be positive")
    base_mean = _integer(base_mean, "base_mean")
    base_variance = _integer(base_variance, "base_variance", nonnegative=True)

    total_weight = base_weight
    first_moment = base_weight * base_mean
    second_moment = base_weight * (base_variance + base_mean * base_mean)
    accepted_updates = 0
    for index, update in enumerate(updates):
        if not isinstance(update, tuple) or len(update) != 3:
            raise ValueError(f"updates[{index}] must be a (weight, mean, variance) tuple")
        weight, mean, variance = update
        weight = _integer(weight, f"updates[{index}].weight", nonnegative=True)
        mean = _integer(mean, f"updates[{index}].mean")
        variance = _integer(
            variance, f"updates[{index}].variance", nonnegative=True
        )
        total_weight += weight
        first_moment += weight * mean
        second_moment += weight * (variance + mean * mean)
        accepted_updates += 1

    mean = _trunc_div(first_moment, total_weight)
    variance = max(0, _trunc_div(second_moment, total_weight) - mean * mean)
    return {
        "total_weight": total_weight,
        "mean": mean,
        "variance": variance,
        "accepted_updates": accepted_updates,
    }
