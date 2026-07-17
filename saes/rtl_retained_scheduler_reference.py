"""Pure reference for the staged T=4 SAES retained-output scheduler.

The scheduler is a control boundary only: it requests native S2/S3 outputs
for the positions selected by an already-completed SAES route.  It neither
routes tiles nor synthesizes skipped descriptors.
"""

from __future__ import annotations

from saes.probe_layout import (
    compute_lightweight_positions,
    compute_probe_positions,
)


SUPPORTED_TILE_SIZE = 4


def retained_output_positions(level: str, *, tile_size: int = SUPPORTED_TILE_SIZE) -> tuple[tuple[int, int], ...]:
    """Return the ordered native-output requests for one accepted sparse route."""
    if tile_size != SUPPORTED_TILE_SIZE:
        raise ValueError(
            f"RTL retained-output scheduler supports only T={SUPPORTED_TILE_SIZE}"
        )
    if level == "L0":
        return tuple(compute_probe_positions(tile_size))
    if level == "L1":
        return tuple(compute_lightweight_positions(tile_size))
    raise ValueError("retained-output scheduler requires an L0 or L1 route")


def retained_output_indices(level: str, *, tile_size: int = SUPPORTED_TILE_SIZE) -> tuple[int, ...]:
    """Return row-major tile-local indices for the fixed retained request order."""
    return tuple(
        row * tile_size + column
        for row, column in retained_output_positions(level, tile_size=tile_size)
    )
