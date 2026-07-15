import pytest


torch = pytest.importorskip("torch")


def test_batched_lsh_matches_scalar_signatures():
    from fsdr.lsh_hasher import LSHHasher
    from fsdr.types import FSDRConfig

    generator = torch.Generator().manual_seed(4)
    features = torch.randn(37, 8, generator=generator)
    hasher = LSHHasher(FSDRConfig(feature_dim=8, seed=9))

    scalar = torch.tensor([hasher.hash(feature) for feature in features])
    batched = hasher.hash_batch(features)

    assert torch.equal(batched.cpu(), scalar)


def test_frame_processing_preserves_raster_cache_semantics():
    from fsdr import FSDRSimulator

    generator = torch.Generator().manual_seed(7)
    features = torch.randn(30, 8, generator=generator)
    depths = torch.linspace(1.0, 1.3, 30)
    scalar = FSDRSimulator(
        feature_dim=8, cache_size=8, hamming_threshold=3, seed=11
    )
    scalar_paths = []
    for pixel_index, (feature, depth) in enumerate(zip(features, depths)):
        y, x = divmod(pixel_index, 6)
        path, _, _ = scalar.process_pixel(
            feature, float(depth), (y, x), pixel_index
        )
        scalar_paths.append(path)

    batched = FSDRSimulator(
        feature_dim=8, cache_size=8, hamming_threshold=3, seed=11
    )
    batched_paths = batched.process_frame(features, depths, width=6)

    assert batched_paths == scalar_paths
    assert batched.get_summary() == scalar.get_summary()
    assert batched.reuse_data == scalar.reuse_data


def test_frame_processing_honors_an_explicit_pixel_schedule():
    from fsdr import FSDRSimulator

    generator = torch.Generator().manual_seed(17)
    features = torch.randn(16, 8, generator=generator)
    depths = torch.linspace(1.0, 1.2, 16)
    order = [0, 3, 12, 15, *[index for index in range(16) if index not in {0, 3, 12, 15}]]
    scalar = FSDRSimulator(
        feature_dim=8, cache_size=8, hamming_threshold=3, seed=19
    )
    scalar_paths = [None] * 16
    for pixel_index in order:
        y, x = divmod(pixel_index, 4)
        path, _, _ = scalar.process_pixel(
            features[pixel_index],
            float(depths[pixel_index]),
            (y, x),
            pixel_index,
        )
        scalar_paths[pixel_index] = path

    batched = FSDRSimulator(
        feature_dim=8, cache_size=8, hamming_threshold=3, seed=19
    )
    batched_paths = batched.process_frame(
        features, depths, width=4, pixel_order=order
    )

    assert batched_paths == scalar_paths
    assert batched.get_summary() == scalar.get_summary()


def test_cache_hit_inserts_the_current_signature_and_depth():
    from fsdr import FSDRSimulator

    simulator = FSDRSimulator(
        feature_dim=8,
        cache_size=2,
        hamming_threshold=3,
        reuse_hamming=3,
        seed=23,
    )
    simulator.process_signature(0b0000, 1.0, (0, 0), 0)
    simulator.process_signature(0b0001, 2.0, (0, 1), 1)

    entries = [entry for entry in simulator.cache.entries if entry is not None]
    assert len(entries) == 2
    current = next(entry for entry in entries if entry.signature == 0b0001)
    assert current.best_depth == pytest.approx(2.0)
    assert current.position == (0, 1)


def test_every_rtl_cache_hit_uses_the_narrow_path():
    from fsdr import FSDRSimulator

    simulator = FSDRSimulator(
        feature_dim=8,
        cache_size=2,
        hamming_threshold=3,
        reuse_hamming=3,
        reuse_confidence=1.0,
        seed=29,
    )
    first = simulator.process_signature(0b0011, 1.0, (0, 0), 0)
    second = simulator.process_signature(0b0011, 1.0, (0, 1), 1)

    assert first[0] == "full_compute"
    assert second[0] == "guided"


def test_discrete_top1_coverage_uses_the_actual_nearest_candidate_subset():
    from fsdr import FSDRSimulator

    simulator = FSDRSimulator(
        feature_dim=2,
        cache_size=1,
        hamming_threshold=3,
        num_depth_candidates=4,
        seed=31,
    )
    features = torch.ones(3, 2)
    anchors = torch.tensor([0.0, 3.0, 3.0])
    top1 = torch.tensor([0, 3, 3])
    candidates = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]]
    )

    paths = simulator.process_discrete_frame(
        features,
        anchors,
        top1,
        candidates,
        width=3,
    )
    summary = simulator.get_summary()

    assert paths == ["full_compute", "guided", "guided"]
    assert summary["guided_top1_covered"] == 1
    assert summary["guided_top1_missed"] == 1
    assert summary["top1_coverage"] == pytest.approx(0.5)
    assert summary["discrete_candidate_evidence"] is True
