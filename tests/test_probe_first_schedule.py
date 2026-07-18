import pytest


torch = pytest.importorskip("torch")


def _features(l0_tiles: set[tuple[int, int]]) -> torch.Tensor:
    feature = torch.zeros(1, 1, 2, 8, 8)
    for tile_y in range(2):
        for tile_x in range(2):
            vectors = (
                ((1.0, 0.0),) * 4
                if (tile_y, tile_x) in l0_tiles
                else ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0))
            )
            y, x = tile_y * 4, tile_x * 4
            for (row, column), vector in zip(
                ((0, 0), (0, 3), (3, 0), (3, 3)), vectors
            ):
                feature[0, 0, :, y + row, x + column] = torch.tensor(vector)
    return feature


def _depths(uniform: set[tuple[int, int]]) -> torch.Tensor:
    depth = torch.ones(1, 1, 8, 8)
    for tile_y in range(2):
        for tile_x in range(2):
            if (tile_y, tile_x) not in uniform:
                depth[0, 0, tile_y * 4 + 3, tile_x * 4 + 3] = 2.0
    return depth


def test_probe_first_schedule_keeps_l1_fallback_anchors_for_l0_tiles():
    from saes.probe_first_schedule import build_conservative_probe_first_schedule

    schedule = build_conservative_probe_first_schedule(
        _features({(0, 0), (1, 0)}),
        _depths({(0, 0), (0, 1)}),
        height=8,
        width=8,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
    )

    mask = schedule.selection_mask[0]
    assert int(mask[:4, :4].sum()) == 8
    assert int(mask[:4, 4:].sum()) == 8
    assert int(mask[4:, :4].sum()) == 4
    assert int(mask[4:, 4:].sum()) == 16
    assert schedule.events["potential_l0_tiles"] == 2
    assert schedule.events["potential_l1_tiles"] == 1
    assert schedule.events["potential_full_tiles"] == 1
    assert schedule.events["gaussian_attributes_accessed"] is False


def test_retained_mask_requires_exact_one_primitive_layout():
    from saes.probe_first_schedule import retained_mask_from_saes_modified

    modified = torch.zeros(32, dtype=torch.bool)
    modified[2] = True
    retained = retained_mask_from_saes_modified(modified, views=2, height=4, width=4)
    assert retained.shape == (2, 4, 4)
    assert retained[0, 0, 2].item() is False
    with pytest.raises(ValueError, match="one-primitive"):
        retained_mask_from_saes_modified(modified, views=1, height=4, width=4)
