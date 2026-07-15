import pytest


torch = pytest.importorskip("torch")


def test_batched_tile_variance_matches_direct_formula():
    from saes.progressive_saes import ProgressiveSAES

    generator = torch.Generator().manual_seed(12)
    features = torch.randn(1, 2, 6, 8, 12, generator=generator)
    variances, normalized = ProgressiveSAES.classify_tiles_by_features(
        features, 8, 12, tile_size=4
    )

    for tile_y in range(2):
        for tile_x in range(3):
            tile = normalized[
                :,
                tile_y * 4 : (tile_y + 1) * 4,
                tile_x * 4 : (tile_x + 1) * 4,
            ].reshape(normalized.shape[0], -1)
            assert variances[(tile_y, tile_x)] == pytest.approx(
                tile.std(dim=1).mean().item(), abs=1e-7
            )
