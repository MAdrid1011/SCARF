"""Unit coverage for the MVSplat native raw-cost-volume audit boundary."""

import pytest


torch = pytest.importorskip("torch")


def test_fixed_native_cv_phase_masks_preserve_primary_l1_prefix_and_full_coverage():
    from scripts.saes_mvsplat_raw_cost_volume_audit import (
        build_fixed_native_cv_phase_masks,
    )

    primary, secondary, full = build_fixed_native_cv_phase_masks(
        batch=2,
        height=8,
        width=8,
        tile_size=4,
        device=torch.device("cpu"),
    )

    assert primary.shape == secondary.shape == full.shape == (2, 8, 8)
    assert int(primary.sum()) == 2 * 4 * 4
    assert int(secondary.sum()) == 2 * 8 * 4
    assert not bool((primary & secondary).any())
    assert bool(full.all())


def test_fixed_native_cv_phase_masks_reject_non_native_grid_shape():
    from scripts.saes_mvsplat_raw_cost_volume_audit import (
        build_fixed_native_cv_phase_masks,
    )

    with pytest.raises(ValueError, match="tiled exactly"):
        build_fixed_native_cv_phase_masks(
            batch=1,
            height=7,
            width=8,
            tile_size=4,
            device=torch.device("cpu"),
        )
