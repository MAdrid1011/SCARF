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


def test_candidate_frames_cover_every_context_view():
    from scripts.fsdr_trace import prepare_fsdr_candidate_frames

    features = torch.zeros(1, 2, 3, 2, 2)
    features[:, 0].fill_(1.0)
    features[:, 1].fill_(9.0)
    probabilities = torch.zeros(1, 2, 4, 2, 2)
    probabilities[:, 0, 1] = 1.0
    probabilities[:, 1, 3] = 1.0
    candidates = torch.tensor(
        [[[[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
          [[[10.0]], [[20.0]], [[30.0]], [[40.0]]]]]
    )

    frames = prepare_fsdr_candidate_frames(features, probabilities, candidates)

    assert len(frames) == 2
    assert torch.equal(frames[0][0], torch.ones_like(frames[0][0]))
    assert torch.equal(frames[1][0], torch.full_like(frames[1][0], 9.0))
    assert torch.equal(frames[0][2], torch.ones(4, dtype=torch.long))
    assert torch.equal(frames[1][2], torch.full((4,), 3, dtype=torch.long))


def test_fsdr_feature_source_keeps_pipeline_features_for_classic_models():
    from scripts.fsdr_trace import select_fsdr_feature_tensor

    pipeline = torch.zeros(1, 2, 128, 8, 8)

    selected, source = select_fsdr_feature_tensor(
        model="mvsplat",
        source="pipeline",
        pipeline_features=pipeline,
        mono_features=None,
    )

    assert selected is pipeline
    assert source == "pipeline"


def test_fsdr_feature_source_reshapes_authentic_depthsplat_mono_features():
    from scripts.fsdr_trace import select_fsdr_feature_tensor

    pipeline = torch.zeros(1, 2, 128, 16, 16)
    mono = torch.arange(2 * 1024 * 8 * 8, dtype=torch.float32).reshape(
        2, 1024, 8, 8
    )

    selected, source = select_fsdr_feature_tensor(
        model="depthsplat",
        source="depthsplat-mono",
        pipeline_features=pipeline,
        mono_features=mono,
    )

    assert selected.shape == (1, 2, 1024, 8, 8)
    assert torch.equal(selected[0, 0], mono[0])
    assert torch.equal(selected[0, 1], mono[1])
    assert source == "depthsplat-mono"


def test_fsdr_feature_source_rejects_mono_features_for_other_models():
    from scripts.fsdr_trace import select_fsdr_feature_tensor

    with pytest.raises(ValueError, match="only valid for DepthSplat"):
        select_fsdr_feature_tensor(
            model="transplat",
            source="depthsplat-mono",
            pipeline_features=torch.zeros(1, 2, 128, 8, 8),
            mono_features=torch.zeros(2, 384, 8, 8),
        )


def test_runtime_fsdr_feature_dim_uses_executed_tensor_channels():
    from scripts.fsdr_trace import runtime_fsdr_feature_dim

    assert runtime_fsdr_feature_dim(torch.zeros(1, 2, 1024, 8, 8)) == 1024

    with pytest.raises(ValueError, match="feature tensor"):
        runtime_fsdr_feature_dim(torch.zeros(2, 1024, 8, 8))
