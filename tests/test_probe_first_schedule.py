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
    assert int(mask[:4, :4].sum()) == 12
    assert int(mask[:4, 4:].sum()) == 12
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


def test_paper_normalized_feature_mode_records_tau_f_in_its_normalized_unit():
    from saes.probe_first_schedule import build_incremental_probe_first_plan

    features = torch.zeros(1, 1, 2, 4, 4)
    for position, vector in {
        (0, 0): (1.0, 0.0),
        (0, 3): (3.0, 0.0),
        (3, 0): (1.0, 0.0),
        (3, 3): (3.0, 0.0),
    }.items():
        features[0, 0, :, position[0], position[1]] = torch.tensor(vector)
    depths = torch.ones(1, 1, 4, 4)

    normalized = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="paper-probe-normalized-feature-first-hit",
    )
    literal = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="paper-probe-feature-variance-first-hit",
    )

    assert normalized.events["feature_threshold"] == pytest.approx(0.2)
    assert (
        normalized.events["feature_statistic"]
        == "normalized-probe-vector-standard-deviation"
    )
    assert normalized.tile_trace[0]["feature_score"] == pytest.approx(0.0)
    assert normalized.events["potential_l0_tiles"] == 1
    assert literal.events["feature_statistic"] == "raw-probe-mean-channel-variance"
    assert literal.tile_trace[0]["feature_score"] == pytest.approx(0.5)
    assert literal.events["potential_l1_tiles"] == 1


def test_incremental_probe_first_plan_partitions_primary_secondary_and_full_requests():
    from saes.probe_first_schedule import build_incremental_probe_first_plan

    plan = build_incremental_probe_first_plan(
        _features({(0, 0), (1, 0)}),
        _depths({(0, 0), (0, 1)}),
        height=8,
        width=8,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
    )

    primary = plan.primary_mask[0]
    secondary = plan.secondary_mask[0]
    full = plan.full_mask[0]
    assert int(primary[:4, :4].sum()) == 4
    assert int(primary[:4, 4:].sum()) == 4
    assert int(primary[4:, :4].sum()) == 4
    assert int(primary[4:, 4:].sum()) == 4
    assert int(secondary[:4, :4].sum()) == 8
    assert int(secondary[:4, 4:].sum()) == 8
    assert int(secondary[4:, :4].sum()) == 0
    assert int(secondary[4:, 4:].sum()) == 0
    assert int(full[:4, :4].sum()) == 0
    assert int(full[4:, 4:].sum()) == 16
    assert int(plan.selection_mask.sum()) == 44
    assert plan.events["primary_head_final_positions"] == 16
    assert plan.events["secondary_head_final_positions"] == 16
    assert plan.events["full_head_final_positions"] == 16
    assert plan.tile_trace[3]["pre_guard_route"] == "Full"
    assert len(plan.events["primary_mask_sha256"]) == 64
    assert len(plan.events["secondary_mask_sha256"]) == 64
    assert len(plan.events["full_mask_sha256"]) == 64
    assert len(plan.events["selection_mask_sha256"]) == 64
    assert len(plan.events["tile_trace_sha256"]) == 64


def test_balanced_l1_semantics_stage_the_declared_interior_anchors():
    from saes.probe_first_schedule import (
        BALANCED_L1_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
    )

    plan = build_incremental_probe_first_plan(
        _features({(0, 0)}),
        _depths({(0, 0), (0, 1), (1, 0), (1, 1)}),
        height=8,
        width=8,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
        l1_anchor_semantics=BALANCED_L1_ANCHOR_SEMANTICS,
    )

    observed = {
        tuple(position.tolist())
        for position in plan.selection_mask[0, :4, :4].nonzero(as_tuple=False)
    }
    assert observed == {
        (0, 0),
        (0, 1),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 2),
        (3, 3),
    }
    assert plan.events["l1_anchor_semantics"] == BALANCED_L1_ANCHOR_SEMANTICS
    assert plan.events["l1_anchor_count"] == 12


def test_adaptive_l1_15_keeps_legacy_boundary_anchors_and_omits_one_s1_center():
    from saes.probe_first_schedule import (
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
        ADAPTIVE_L1_15_SELECTION_SEMANTICS,
        build_incremental_probe_first_plan,
        l1_local_positions_for_tile,
    )
    from saes.probe_layout import compute_lightweight_positions, compute_probe_positions

    features = torch.zeros(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 4, 4)
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )
    record = plan.tile_trace[0]
    anchors = l1_local_positions_for_tile(
        record,
        tile_size=4,
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )

    assert record["adaptive_l1_omitted_local_position"] == [1, 1]
    assert record["adaptive_l1_best_to_second_residual_ratio"] == pytest.approx(1.0)
    assert anchors[:4] == compute_probe_positions(4)
    assert anchors[:12] == compute_lightweight_positions(4)
    assert len(anchors) == 15
    assert len(set(anchors)) == 15
    assert set(((1, 1), (1, 2), (2, 1), (2, 2))) - set(anchors) == {(1, 1)}
    assert int(plan.primary_mask.sum()) == 4
    assert int(plan.secondary_mask.sum()) == 11
    assert int(plan.selection_mask.sum()) == 15
    assert plan.events["l1_anchor_count"] == 15
    assert plan.events["l1_anchor_selection"] == ADAPTIVE_L1_15_SELECTION_SEMANTICS
    assert plan.events["l1_anchor_selection_uses_s1_only"] is True

    changed_features = features.clone()
    changed_features[0, 0, :, 1, 1] = 10.0
    changed = build_incremental_probe_first_plan(
        changed_features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )
    assert changed.tile_trace[0]["adaptive_l1_omitted_local_position"] != [1, 1]

    changed_depths = depths.clone()
    changed_depths[0, 0, 0, 0] = 2.0
    depth_changed = build_incremental_probe_first_plan(
        changed_features,
        changed_depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )
    assert (
        depth_changed.tile_trace[0]["adaptive_l1_omitted_local_position"]
        == changed.tile_trace[0]["adaptive_l1_omitted_local_position"]
    )
