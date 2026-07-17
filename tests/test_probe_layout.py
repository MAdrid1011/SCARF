import subprocess
import sys

import pytest


def test_t4_layout_is_fixed_and_l1_preserves_primary_prefix():
    from saes.probe_layout import (
        compute_lightweight_positions,
        compute_nonprobe_positions,
        compute_probe_positions,
    )

    primary = compute_probe_positions(4)
    lightweight = compute_lightweight_positions(4)

    assert primary == [(0, 0), (0, 3), (3, 0), (3, 3)]
    assert lightweight == [
        (0, 0),
        (0, 3),
        (3, 0),
        (3, 3),
        (1, 1),
        (2, 2),
        (0, 1),
        (0, 2),
    ]
    assert compute_nonprobe_positions(4) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
    ]


def test_layout_capacity_and_l1_prefix_are_preserved_for_declared_tiles():
    from saes.probe_layout import (
        compute_lightweight_positions,
        compute_probe_positions,
    )

    assert [len(compute_probe_positions(tile_size)) for tile_size in (4, 8, 16)] == [
        4,
        6,
        8,
    ]
    for tile_size in (4, 8, 16):
        primary = compute_probe_positions(tile_size)
        lightweight = compute_lightweight_positions(tile_size)
        assert lightweight[: len(primary)] == primary
        assert len(lightweight) == min(tile_size * tile_size, 2 * len(primary))
        assert len(set(primary)) == len(primary)
        assert len(set(lightweight)) == len(lightweight)


def test_torch_progressive_saes_wrappers_delegate_to_the_pure_layout():
    pytest.importorskip("torch")

    from saes.probe_layout import (
        compute_lightweight_positions,
        compute_probe_positions,
    )
    from saes.progressive_saes import ProgressiveSAES

    for tile_size in (4, 5, 8, 16):
        assert ProgressiveSAES.compute_probe_positions(tile_size) == compute_probe_positions(
            tile_size
        )
        assert (
            ProgressiveSAES.compute_lightweight_positions(tile_size)
            == compute_lightweight_positions(tile_size)
        )


def test_layout_and_event_accounting_import_and_execute_without_torch():
    code = """
import builtins

original_import = builtins.__import__

def import_without_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError('Torch must not be imported by CPU SAES helpers')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_torch
from saes.probe_layout import compute_lightweight_positions, compute_probe_positions
from saes.hardware_accounting import build_saes_event_ledger

assert len(compute_probe_positions(4)) == 4
assert len(compute_lightweight_positions(4)) == 8
ledger = build_saes_event_ledger(
    {
        'total_tiles_processed': 1,
        'level0_tiles': 1,
        'level1_tiles': 0,
        'full_tiles': 0,
        'l0_representatives': 4,
    },
    feature_dim=128,
    tile_size=4,
    sh_degree=4,
)
assert ledger['events']['l0_retained_anchors'] == 4
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=__file__.rsplit("/tests/", 1)[0],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
