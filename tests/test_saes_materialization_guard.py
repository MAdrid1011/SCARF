from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _gaussians():
    means = []
    covariances = []
    harmonics = []
    opacities = []
    for row in range(4):
        for column in range(4):
            means.append((float(column), float(row), 1.0))
            covariances.append(torch.eye(3) * 0.01)
            harmonics.append(torch.full((3, 1), 0.1 + 0.01 * (row + column)))
            opacities.append(0.2 + 0.01 * (row + column))
    return SimpleNamespace(
        means=torch.tensor(means).unsqueeze(0),
        covariances=torch.stack(covariances).unsqueeze(0),
        harmonics=torch.stack(harmonics).unsqueeze(0),
        opacities=torch.tensor(opacities).unsqueeze(0),
    )


def _uniform_inputs():
    return (
        torch.ones(1, 1, 2, 4, 4),
        torch.ones(1, 1, 16, 1, 1),
    )


def _l1_inputs():
    features = torch.zeros(1, 1, 2, 4, 4)
    for (row, column), value in {
        (0, 0): (0.0, 0.0),
        (0, 3): (2.0, 0.0),
        (3, 0): (0.0, 2.0),
        (3, 3): (2.0, 2.0),
    }.items():
        features[0, 0, :, row, column] = torch.tensor(value)
    return features, torch.ones(1, 1, 16, 1, 1)


def test_l0_guard_rejection_still_attempts_l1_before_full_fallback():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    # The feature route selects L0, but one primary anchor violates the SH
    # cosine guard. L1 must still be checked before the tile becomes Full.
    gaussians.harmonics[0, 0] *= -1.0
    features, depths = _uniform_inputs()

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
    )

    assert stats["l0_guard_rejections"] == 1
    assert stats["l1_guard_attempts_after_l0_rejection"] == 1
    assert stats["l1_guard_rejections"] == 1
    assert stats["full_tiles"] == 1
    assert stats["materialization_guard_enabled"] is True


def test_primary_cross_check_fails_closed_before_l1_widening():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    # Every pair remains above the ordinary covariance-cosine guard's 0.7
    # floor, but the leave-one-out primary prediction is inconsistent with a
    # held-out anchor. L1 preserves this primary prefix, so it cannot recover
    # a failed primary cross-check by widening to boundary anchors.
    for index, diagonal in zip(
        (0, 3, 12, 15),
        ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0), (1.0, 0.5, 0.5)),
    ):
        gaussians.covariances[0, index] = torch.diag(torch.tensor(diagonal))
    features, depths = _uniform_inputs()
    tile_trace = []

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        cross_check_threshold=0.015,
        tile_trace=tile_trace,
    )

    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 0
    assert stats["full_tiles"] == 1
    assert stats["probe_cross_check_l0_checks"] == 1
    assert stats["probe_cross_check_l0_rejections"] == 1
    assert stats["l1_guard_attempts_after_l0_rejection"] == 0
    check = tile_trace[0]["guard_checks"][0]["probe_cross_check"]
    assert check["checked"] is True
    assert check["passed"] is False
    assert check["error_max"] > check["threshold"]
    assert check["nonprobe_s3_attribute_reads"] == 0


def test_l0_center_separation_short_circuits_impossible_l1_guard():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    features, depths = _uniform_inputs()
    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
        context_safety_guard=True,
    )

    # L1 includes the same four primary anchors. A failed maximum pairwise
    # center distance is monotonic as more anchors are added, so reevaluating
    # L1 cannot recover this tile.
    assert stats["l0_guard_checks"] == 1
    assert stats["l0_guard_rejections"] == 1
    assert stats["l1_guard_attempts_after_l0_rejection"] == 0
    assert stats["l1_guard_checks"] == 0
    assert stats["full_tiles"] == 1


def test_tile_trace_matches_guard_counters_for_rejected_to_full_route():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    # A failed primary-anchor SH comparison rejects L0. Those same primary
    # anchors are part of the declared L1 set, so the fail-closed route ends
    # at Full after both guard checks.
    gaussians.harmonics[0, 0] *= -1.0
    features, depths = _uniform_inputs()
    tile_trace = []

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        tile_trace=tile_trace,
    )

    assert len(tile_trace) == stats["total_tiles_processed"] == 1
    record = tile_trace[0]
    assert set(record) == {
        "view_index",
        "tile_row",
        "tile_column",
        "feature_variance",
        "feature_candidate",
        "depth_candidate",
        "guard_enabled",
            "guard_checks",
            "routing_level_before_materialization",
            "final_route",
    }
    assert record["feature_candidate"] is True
    assert record["depth_candidate"] is True
    assert record["guard_enabled"] is True
    assert record["routing_level_before_materialization"] == "Full"
    assert record["final_route"] == "Full"

    checks = record["guard_checks"]
    assert [check["level"] for check in checks] == ["L0", "L1"]
    assert [check["anchor_count"] for check in checks] == [4, 12]
    assert all(check["passed"] is False for check in checks)
    assert sum(check["level"] == "L0" for check in checks) == stats[
        "l0_guard_checks"
    ]
    assert sum(check["level"] == "L1" for check in checks) == stats[
        "l1_guard_checks"
    ]

    allowed_guard_scalars = {
        "level",
        "anchor_count",
        "passed",
        "covariance_cosine_minimum",
        "harmonic_cosine_minimum",
        "opacity_distance_maximum",
        "nonprobe_s3_attribute_reads",
        "probe_cross_check",
    }
    for check in checks:
        assert set(check) == allowed_guard_scalars
        assert check["nonprobe_s3_attribute_reads"] == 0
        cross_check = check["probe_cross_check"]
        assert set(cross_check) == {
            "checked",
            "passed",
            "error_max",
            "threshold",
            "primary_anchor_count",
            "nonprobe_s3_attribute_reads",
        }
        assert cross_check["checked"] is False
        assert cross_check["passed"] is None
        assert cross_check["error_max"] is None
        assert cross_check["nonprobe_s3_attribute_reads"] == 0


def test_disabled_materialization_guard_preserves_boolean_provenance():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    features, depths = _uniform_inputs()
    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        materialization_guard=False,
    )

    assert stats["materialization_guard_enabled"] is False
    assert stats["l0_guard_checks"] == 0
    assert stats["guard_anchor_attribute_reads"] == 0


def test_l1_guard_rejection_falls_closed_to_full():
    from saes.probe_layout import compute_lightweight_positions
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    # The first extra L1 anchor is not a primary probe. Its failed SH guard
    # proves L1 validates its full 2K(T) anchor set before materialization.
    row, column = compute_lightweight_positions(4)[4]
    gaussians.harmonics[0, row * 4 + column] *= -1.0
    features, depths = _l1_inputs()

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        decision_semantics="probe-vector-first-hit",
    )

    assert stats["level0_tiles"] == 0
    assert stats["l1_guard_checks"] == 1
    assert stats["l1_guard_rejections"] == 1
    assert stats["level1_tiles"] == 0
    assert stats["full_tiles"] == 1


def test_probe_only_guard_is_invariant_to_nonprobe_s3_attributes():
    from saes.probe_layout import compute_probe_positions
    from saes.progressive_saes import ProgressiveSAES

    gaussians = _gaussians()
    saes = ProgressiveSAES(4, 4)
    probes = [row * 4 + column for row, column in compute_probe_positions(4)]
    baseline = saes.probe_materialization_validity(gaussians, probes, level="L0")

    nonprobes = sorted(set(range(16)) - set(probes))
    gaussians.covariances[0, nonprobes] = float("nan")
    gaussians.harmonics[0, nonprobes] = float("nan")
    gaussians.opacities[0, nonprobes] = float("nan")
    poisoned = saes.probe_materialization_validity(gaussians, probes, level="L0")

    assert poisoned == baseline
    assert poisoned["anchor_indices"] == probes
    assert poisoned["nonprobe_s3_attribute_reads"] == 0


def test_declared_l1_layouts_are_nested_and_event_conserving():
    from saes.probe_layout import compute_lightweight_positions, compute_probe_positions
    from saes.progressive_saes import apply_progressive_saes

    assert len(compute_probe_positions(4)) == 4
    assert len(compute_lightweight_positions(4)) == 12
    assert len(compute_probe_positions(8)) == 6
    assert len(compute_lightweight_positions(8)) == 12
    assert compute_lightweight_positions(8)[:6] == compute_probe_positions(8)

    gaussians = _gaussians()
    features, depths = _l1_inputs()
    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        decision_semantics="probe-vector-first-hit",
        num_depth_candidates=4,
    )

    assert stats["level1_tiles"] == 1
    assert stats["l1_lightweight_anchors"] == 12
    assert stats["executed_s2_evaluations"] == 12 * 4
    assert stats["full_s2_evaluations"] == 16 * 4


def test_guard_accounting_charges_only_selected_anchor_control_work():
    from saes.hardware_accounting import build_saes_event_ledger

    ledger = build_saes_event_ledger(
        {
            "total_tiles_processed": 2,
            "level0_tiles": 1,
            "level1_tiles": 1,
            "full_tiles": 0,
            "l0_representatives": 4,
            "l1_lightweight_anchors": 12,
            "materialization_guard_enabled": True,
            "l0_guard_checks": 1,
            "l1_guard_checks": 1,
            "l0_guard_rejections": 0,
            "l1_guard_rejections": 0,
            "guard_anchor_attribute_reads": 3 * (4 + 12),
            "guard_nonprobe_s3_attribute_reads": 0,
        },
        feature_dim=128,
        tile_size=4,
        sh_degree=2,
    )

    assert ledger["events"]["guard_anchor_descriptors"] == 16
    assert ledger["events"]["guard_nonprobe_s3_attribute_reads"] == 0
    assert ledger["cycles"]["materialization_guard"] > 0
    assert ledger["traffic_bytes"]["materialization_guard_descriptor_read"] > 0
