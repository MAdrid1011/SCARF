from depth_predictor.depthsplat_predictor import DepthSplatDepthPredictorSim


def test_depthsplat_plane_sweep_counts_all_reference_source_candidate_positions():
    cycles, candidates = DepthSplatDepthPredictorSim._plane_sweep_cycles(
        batch_size=1,
        view_count=2,
        feature_channels=128,
        depth_candidates=128,
        height=32,
        width=56,
    )

    assert candidates == 2 * 128 * 32 * 56
    # Four 32-channel batches run the five-cycle grid sampler per candidate.
    assert cycles > candidates * 20


def test_depthsplat_plane_sweep_scales_linearly_with_candidate_count():
    full_cycles, full_candidates = DepthSplatDepthPredictorSim._plane_sweep_cycles(
        batch_size=1,
        view_count=2,
        feature_channels=128,
        depth_candidates=128,
        height=32,
        width=56,
    )
    narrow_cycles, narrow_candidates = DepthSplatDepthPredictorSim._plane_sweep_cycles(
        batch_size=1,
        view_count=2,
        feature_channels=128,
        depth_candidates=32,
        height=32,
        width=56,
    )

    assert narrow_candidates == full_candidates // 4
    # The depth-independent inverse-K setup remains, all candidate-plane work
    # contracts exactly with the narrowed search window.
    assert narrow_cycles < full_cycles
    source_pairs = 1 * 2 * (2 - 1)
    backproject_cycles = source_pairs * 3 * 3 * 32 * 56 // 1024
    assert full_cycles - backproject_cycles == 4 * (
        narrow_cycles - backproject_cycles
    )
