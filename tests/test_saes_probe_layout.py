"""Contract tests for the fixed claim-path SAES L1 anchor layout."""


def test_t4_l1_uses_four_primary_probes_and_eight_boundary_anchors():
    from saes.probe_layout import compute_lightweight_positions, compute_probe_positions

    anchors = compute_lightweight_positions(4)

    assert anchors == [
        (0, 0),
        (0, 3),
        (3, 0),
        (3, 3),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 3),
        (2, 0),
        (2, 3),
        (3, 1),
        (3, 2),
    ]
    assert anchors[:4] == compute_probe_positions(4)
    assert len(anchors) == 12
    assert len(set(anchors)) == len(anchors)
    assert all(row in {0, 3} or column in {0, 3} for row, column in anchors[4:])
