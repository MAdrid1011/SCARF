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


def test_lsh_projection_is_fp16_seeded_and_does_not_mutate_global_rng():
    from fsdr import FSDRSimulator
    from fsdr.lsh_hasher import LSHHasher
    from fsdr.types import FSDRConfig

    torch.manual_seed(123)
    expected = torch.randn(4)
    torch.manual_seed(123)
    hasher = LSHHasher(FSDRConfig(feature_dim=8, seed=42))
    actual = torch.randn(4)

    assert torch.equal(actual, expected)
    assert hasher.get_projection_matrix().dtype == torch.float16
    assert torch.equal(
        hasher.get_projection_matrix(),
        LSHHasher(FSDRConfig(feature_dim=8, seed=42)).get_projection_matrix(),
    )
    assert FSDRSimulator(feature_dim=8).config.seed == 42


def test_fp16_lsh_operands_decode_exactly_to_q24():
    from fsdr.lsh_hasher import fp16_to_q24

    smallest_subnormal = torch.tensor([1], dtype=torch.int16).view(torch.float16)
    values = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float16)
    decoded = fp16_to_q24(torch.cat((values, smallest_subnormal)))

    assert decoded.tolist() == [0, 1 << 24, -(1 << 24), 1 << 23, 1]


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


def test_historical_depth_guard_falls_back_on_a_locally_inconsistent_anchor():
    from fsdr import FSDRSimulator

    simulator = FSDRSimulator(
        feature_dim=8,
        cache_size=8,
        hamming_threshold=0,
        guidance_policy="historical-depth-guard",
        seed=30,
    )
    simulator.process_signature(0x0000, 10.0, (10, 10), 0)
    for index, signature in enumerate((0xFFFF, 0xFFFE, 0xFFFC), start=1):
        simulator.process_signature(signature, 1.0, (0, index + 3), index)

    guarded = simulator.process_signature(0x0000, 1.0, (0, 7), 4)
    summary = simulator.get_summary()

    assert guarded[0] == "hit_no_guide"
    assert guarded[1] == simulator.num_depth_candidates
    assert summary["depth_inconsistent"] == 1
    assert summary["hit_no_guide"] == 1
    assert summary["guidance_policy"] == "historical-depth-guard"


def test_claim_local_validity_guard_falls_back_and_records_discrete_events():
    from fsdr import FSDRSimulator

    simulator = FSDRSimulator(
        feature_dim=8,
        cache_size=8,
        hamming_threshold=0,
        depth_consistency_threshold=0.10,
        guidance_policy="paper-hamming-local-validity",
        seed=30,
    )
    simulator.process_signature(0x0000, 10.0, (10, 10), 0)
    for index, signature in enumerate((0xFFFF, 0xFFFE, 0xFFFC), start=1):
        simulator.process_signature(signature, 1.0, (0, index + 3), index)

    guarded = simulator.process_signature(0x0000, 1.0, (0, 7), 4)
    summary = simulator.get_summary()

    assert guarded[0] == "hit_no_guide"
    assert summary["hamming_hits"] == 1
    assert summary["local_valid_hits"] == 0
    assert summary["local_invalid_fallbacks"] == 1
    assert summary["depth_consistency_threshold"] == pytest.approx(0.10)


def test_local_validity_threshold_must_be_in_the_registered_range():
    from fsdr import FSDRSimulator

    for invalid in (0.0, -0.1, 1.0, float("inf")):
        with pytest.raises(ValueError, match="depth consistency"):
            FSDRSimulator(feature_dim=8, depth_consistency_threshold=invalid)


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


def test_begin_frame_clears_frame_local_state_but_preserves_aggregate_counts():
    from fsdr import FSDRSimulator

    simulator = FSDRSimulator(
        feature_dim=2,
        cache_size=2,
        hamming_threshold=3,
        num_depth_candidates=4,
        seed=37,
    )
    simulator.process_signature(0b0011, 1.0, (0, 0), 0)
    simulator.reuse_data[7] = {"depth_ratio": 1.0, "in_window": True}

    simulator.begin_frame()

    assert len(simulator.cache) == 0
    assert simulator.recent_depths == {}
    assert simulator.reuse_data == {}
    assert simulator.stats["total_pixels"] == 1
    assert simulator.stats["frames_started"] == 1


def test_candidate_and_feature_buffer_events_are_counted_from_executed_paths():
    from fsdr import FSDRSimulator

    simulator = FSDRSimulator(
        feature_dim=2,
        cache_size=1,
        hamming_threshold=3,
        num_depth_candidates=4,
        seed=41,
    )
    for pixel_index in range(3):
        simulator.process_signature(0b0011, 1.0, (0, pixel_index), pixel_index)

    summary = simulator.get_summary()
    assert summary["full_depth_evaluations"] == 12
    assert summary["executed_depth_evaluations"] == 6
    assert summary["depth_evaluations_available"] is True
    assert summary["feature_buffer_bytes_baseline"] == 48
    assert summary["feature_buffer_bytes_actual"] == 24
    assert summary["feature_buffer_bytes_available"] is True
    assert summary["feature_buffer_element_bytes"] == 2
