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
