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
    assert len(compute_lightweight_positions(4)) == 8
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
    assert stats["l1_lightweight_anchors"] == 8
    assert stats["executed_s2_evaluations"] == 8 * 4
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
            "l1_lightweight_anchors": 8,
            "materialization_guard_enabled": True,
            "l0_guard_checks": 1,
            "l1_guard_checks": 1,
            "l0_guard_rejections": 0,
            "l1_guard_rejections": 0,
            "guard_anchor_attribute_reads": 3 * (4 + 8),
            "guard_nonprobe_s3_attribute_reads": 0,
        },
        feature_dim=128,
        tile_size=4,
        sh_degree=2,
    )

    assert ledger["events"]["guard_anchor_descriptors"] == 12
    assert ledger["events"]["guard_nonprobe_s3_attribute_reads"] == 0
    assert ledger["cycles"]["materialization_guard"] > 0
    assert ledger["traffic_bytes"]["materialization_guard_descriptor_read"] > 0
