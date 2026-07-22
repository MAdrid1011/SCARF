"""Exact spatial-footprint gate for DepthSplat's fixed S3 adaptor stack.

DepthSplat's submitted adaptor is a chain of four same-resolution 3x3
convolutions: two in ``gaussian_regressor`` and two in ``gaussian_head``.
This module propagates the declared SAES retained-output positions backwards
through that exact chain using replicate-padding geometry. It establishes only
which spatial activations must exist for bit-identical retained raw-head
values; it neither executes a model nor uses images, targets, metrics, or
paper results.
"""

from __future__ import annotations

from collections.abc import Iterable

from saes.rtl_retained_scheduler_reference import (
    SUPPORTED_TILE_SIZE,
    retained_output_positions,
)


DEPTHSPLAT_S3_CONV3_LAYERS = (
    "gaussian_regressor.0",
    "gaussian_regressor.2",
    "gaussian_head.0",
    "gaussian_head.2",
)

Position = tuple[int, int]


def _validate_geometry(*, height: int, width: int) -> None:
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if height % SUPPORTED_TILE_SIZE or width % SUPPORTED_TILE_SIZE:
        raise ValueError("DepthSplat S3 footprint requires dimensions divisible by T=4")


def retained_output_footprint(*, height: int, width: int, level: str) -> frozenset[Position]:
    """Return the repeated T=4 L0/L1 native raw-head output positions."""
    _validate_geometry(height=height, width=width)
    local_positions = retained_output_positions(level)
    return frozenset(
        (tile_row + local_row, tile_column + local_column)
        for tile_row in range(0, height, SUPPORTED_TILE_SIZE)
        for tile_column in range(0, width, SUPPORTED_TILE_SIZE)
        for local_row, local_column in local_positions
    )


def backward_same_conv3_footprint(
    positions: Iterable[Position], *, height: int, width: int
) -> frozenset[Position]:
    """Propagate required outputs through one replicate-padded 3x3 convolution."""
    _validate_geometry(height=height, width=width)
    required: set[Position] = set()
    for row, column in positions:
        if not 0 <= row < height or not 0 <= column < width:
            raise ValueError("convolution footprint position is outside the feature map")
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                required.add(
                    (
                        min(max(row + delta_row, 0), height - 1),
                        min(max(column + delta_column, 0), width - 1),
                    )
                )
    return frozenset(required)


def depthsplat_s3_required_footprint(
    *, height: int, width: int, level: str
) -> dict[str, object]:
    """Trace retained output requirements backwards through the fixed S3 stack."""
    output_positions = retained_output_footprint(
        height=height, width=width, level=level
    )
    reverse_layers: list[dict[str, object]] = []
    current = output_positions
    for layer in reversed(DEPTHSPLAT_S3_CONV3_LAYERS):
        input_positions = backward_same_conv3_footprint(
            current, height=height, width=width
        )
        reverse_layers.append(
            {
                "layer": layer,
                "required_output_positions": len(current),
                "required_input_positions": len(input_positions),
            }
        )
        current = input_positions
    total_positions = height * width
    return {
        "model": "depthsplat",
        "level": level,
        "tile_size": SUPPORTED_TILE_SIZE,
        "spatial_size": [height, width],
        "total_spatial_positions": total_positions,
        "retained_final_head_positions": len(output_positions),
        "reverse_conv3_requirements": tuple(reverse_layers),
        "all_preceding_s3_positions_required": len(current) == total_positions,
    }
