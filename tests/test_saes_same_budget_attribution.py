from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _gaussians(values):
    values = torch.tensor(values, dtype=torch.float32)
    count = values.numel()
    return SimpleNamespace(
        means=values.reshape(1, count, 1).repeat(1, 1, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, count, 1, 1),
        harmonics=values.reshape(1, count, 1, 1).repeat(1, 1, 3, 2),
        opacities=(values / 100).reshape(1, count, 1),
    )


def test_three_way_attribution_shares_one_mask_and_isolates_updates():
    from scripts.saes_same_budget_attribution import _build_attribution_variants

    baseline = _gaussians(range(16))
    merged = _gaussians(range(16))
    representative = torch.zeros(16, dtype=torch.bool)
    representative[torch.tensor((0, 3, 12, 15))] = True
    modified = ~representative
    for name in ("means", "covariances", "harmonics", "opacities"):
        getattr(merged, name)[:, representative] += 10.0
    merged.opacities[:, modified] = 0.0

    variants, accounting = _build_attribution_variants(
        baseline,
        merged,
        modified,
        views=1,
        height=4,
        width=4,
        stats={
            "zeroed_gaussians": 12,
            "same_budget_dense_oracle_output_gaussians": 4,
            "full_stage3_gaussians": 0,
            "level0_tiles": 1,
            "level1_tiles": 0,
        },
    )
    retained = ~modified

    assert accounting == {
        "full_gaussians": 16,
        "removed_nonprobes": 12,
        "retained_gaussians": 4,
        "representative_slots": 4,
        "representative_slots_with_nonzero_update": 4,
        "full_passthrough_gaussians": 0,
        "unchanged_full_or_retained_gaussians": 12,
    }
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(
            getattr(variants["drop_only"], name), getattr(baseline, name)[:, retained]
        )
        torch.testing.assert_close(
            getattr(variants["drop_merge"], name), getattr(merged, name)[:, retained]
        )
        torch.testing.assert_close(
            getattr(variants["merge_only"], name)[:, ~representative],
            getattr(baseline, name)[:, ~representative],
        )
        torch.testing.assert_close(
            getattr(variants["merge_only"], name)[:, representative],
            getattr(merged, name)[:, representative],
        )


def test_three_way_attribution_partitions_mixed_l0_l1_full_tiles():
    from saes.progressive_saes import ProgressiveSAES
    from scripts.saes_same_budget_attribution import _build_attribution_variants

    height, width = 4, 12
    baseline = _gaussians(range(height * width))
    merged = _gaussians(range(height * width))
    modified = torch.zeros(height * width, dtype=torch.bool)
    l0 = set((0, 3, 12, 15))
    l1 = {
        row * 4 + column
        for row, column in ProgressiveSAES.compute_lightweight_positions(4)
    }
    for local in range(16):
        row, column = divmod(local, 4)
        if local not in l0:
            modified[row * width + column] = True
        if local not in l1:
            modified[row * width + 4 + column] = True
    representatives = ~modified
    # The third tile is Full, so only the first two tiles may change.
    representatives[torch.arange(height * width).reshape(height, width)[:, 8:].reshape(-1)] = False
    for name in ("means", "covariances", "harmonics", "opacities"):
        getattr(merged, name)[:, representatives] += 5.0
    merged.opacities[:, modified] = 0.0

    _, accounting = _build_attribution_variants(
        baseline,
        merged,
        modified,
        views=1,
        height=height,
        width=width,
        stats={
            "zeroed_gaussians": 16,
            "same_budget_dense_oracle_output_gaussians": 16,
            "full_stage3_gaussians": 16,
            "level0_tiles": 1,
            "level1_tiles": 1,
        },
    )

    assert accounting["removed_nonprobes"] == 16
    assert accounting["representative_slots"] == 16
    assert accounting["full_passthrough_gaussians"] == 16
