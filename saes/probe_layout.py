"""Pure, deterministic SAES probe and retained-anchor layouts.

These helpers define only the geometric K(T)/2K(T) layout.  They deliberately
have no Torch dependency so CPU-side schema, accounting, and release checks
can validate the declared SAES path without loading a model runtime.
"""

from __future__ import annotations

import math
from typing import List, Tuple


Position = Tuple[int, int]


def compute_probe_positions(tile_size: int) -> List[Position]:
    """Compute the declared K(T) primary probe positions in one square tile.

    K(T) is four corner probes plus ``ceil(2 * log2(T / 4))`` interior probes
    for ``T > 4``.  Interior probes are selected one per uniform-subgrid cell
    at the nearest unselected pixel to that cell's centre.
    """
    T = tile_size
    if T <= 4:
        probe_count = 4
    else:
        probe_count = 4 + math.ceil(2 * math.log2(T / 4))

    corners: List[Position] = [(0, 0), (0, T - 1), (T - 1, 0), (T - 1, T - 1)]
    probes: List[Position] = list(corners)
    probes_set = set(corners)

    additional_count = probe_count - 4
    if additional_count > 0 and T > 2:
        grid_side = math.ceil(math.sqrt(additional_count))
        added = 0
        for grid_row in range(grid_side):
            if added >= additional_count:
                break
            for grid_column in range(grid_side):
                if added >= additional_count:
                    break
                centre_row = 1.0 + (grid_row + 0.5) * (T - 2) / grid_side
                centre_column = 1.0 + (grid_column + 0.5) * (T - 2) / grid_side
                best_distance = float("inf")
                best_position: Position | None = None
                for row in range(1, T - 1):
                    for column in range(1, T - 1):
                        if (row, column) not in probes_set:
                            distance = (
                                (row - centre_row) ** 2
                                + (column - centre_column) ** 2
                            )
                            if distance < best_distance:
                                best_distance = distance
                                best_position = (row, column)
                if best_position is not None:
                    probes.append(best_position)
                    probes_set.add(best_position)
                    added += 1

    return probes


def compute_lightweight_positions(tile_size: int) -> List[Position]:
    """Return the declared 2K(T) L1 anchors, preserving primary probes first."""
    primary = compute_probe_positions(tile_size)
    target = min(tile_size * tile_size, 2 * len(primary))
    positions = list(primary)
    selected = set(positions)
    candidates = [
        (row, column)
        for row in range(tile_size)
        for column in range(tile_size)
        if (row, column) not in selected
    ]
    while len(positions) < target:
        best = max(
            candidates,
            key=lambda point: (
                min(
                    (point[0] - anchor[0]) ** 2
                    + (point[1] - anchor[1]) ** 2
                    for anchor in positions
                ),
                -point[0],
                -point[1],
            ),
        )
        positions.append(best)
        selected.add(best)
        candidates.remove(best)
    return positions


def compute_nonprobe_positions(tile_size: int) -> List[Position]:
    """Return all tile positions not retained by the declared L0 probe layout."""
    probes = set(compute_probe_positions(tile_size))
    return [
        (row, column)
        for row in range(tile_size)
        for column in range(tile_size)
        if (row, column) not in probes
    ]
