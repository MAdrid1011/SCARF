import pytest


torch = pytest.importorskip("torch")


def test_fsdr_frame_keeps_feature_and_depth_on_the_same_context_view():
    from scripts.fsdr_trace import prepare_fsdr_frame

    features = torch.zeros(1, 2, 3, 2, 2)
    features[:, 0].fill_(1.0)
    features[:, 1].fill_(9.0)
    depths = torch.zeros(1, 2, 16, 1, 1)
    depths[:, 0].fill_(2.0)
    depths[:, 1].fill_(8.0)

    feature_frame, depth_frame = prepare_fsdr_frame(
        features, depths, height=4, width=4
    )

    assert feature_frame.shape == (16, 3)
    assert depth_frame.shape == (16,)
    assert torch.equal(feature_frame, torch.ones_like(feature_frame))
    assert torch.equal(depth_frame, torch.full_like(depth_frame, 2.0))


def test_tile_probe_order_runs_corners_before_row_major_remainder():
    from scripts.fsdr_trace import tile_probe_pixel_order

    order = tile_probe_pixel_order(height=4, width=8, tile_size=4)

    assert order[:4] == [0, 3, 24, 27]
    assert order[4:16] == [1, 2, 8, 9, 10, 11, 16, 17, 18, 19, 25, 26]
    assert order[16:20] == [4, 7, 28, 31]
    assert sorted(order) == list(range(32))


def test_candidate_frame_uses_full_search_argmax_and_candidate_expectation():
    from scripts.fsdr_trace import prepare_fsdr_candidate_frame

    features = torch.zeros(1, 2, 3, 2, 2)
    features[:, 0].fill_(1.0)
    features[:, 1].fill_(9.0)
    probabilities = torch.zeros(1, 2, 4, 2, 2)
    probabilities[:, 0, 2] = 0.75
    probabilities[:, 0, 1] = 0.25
    probabilities[:, 1, 3] = 1.0
    candidates = torch.tensor(
        [[[[[1.0]], [[2.0]], [[4.0]], [[8.0]]],
          [[[10.0]], [[20.0]], [[40.0]], [[80.0]]]]]
    )

    feature_frame, anchors, top1, candidate_frame, shape = (
        prepare_fsdr_candidate_frame(features, probabilities, candidates)
    )

    assert shape == (2, 2)
    assert feature_frame.shape == (4, 3)
    assert torch.equal(feature_frame, torch.ones_like(feature_frame))
    assert torch.equal(top1, torch.full((4,), 2, dtype=torch.long))
    assert torch.allclose(anchors, torch.full((4,), 3.5))
    assert torch.equal(candidate_frame[0], torch.tensor([1.0, 2.0, 4.0, 8.0]))


def test_candidate_frame_rejects_probability_candidate_mismatch():
    from scripts.fsdr_trace import prepare_fsdr_candidate_frame

    features = torch.zeros(1, 2, 3, 2, 2)
    probabilities = torch.ones(1, 2, 4, 2, 2) / 4
    candidates = torch.ones(1, 2, 3, 1, 1)

    with pytest.raises(ValueError, match="candidate count"):
        prepare_fsdr_candidate_frame(features, probabilities, candidates)
