import pytest
from types import SimpleNamespace


torch = pytest.importorskip("torch")


def _gaussians(view_count: int = 1):
    h = w = 4
    means = []
    harmonics = []
    opacities = []
    for view in range(view_count):
        for y in range(h):
            for x in range(w):
                means.append([float(x), float(y), float(view + 1)])
                value = 0.1 + 0.01 * (x + y)
                harmonics.append([[value], [value + 0.1], [value + 0.2]])
                opacities.append(0.2 + 0.01 * (x + y))
    count = len(means)
    return SimpleNamespace(
        means=torch.tensor(means).unsqueeze(0),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, count, 1, 1) * 0.01,
        harmonics=torch.tensor(harmonics).unsqueeze(0),
        opacities=torch.tensor(opacities).unsqueeze(0),
    )


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


def test_representative_path_performs_full_gaussian_moment_matching():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    original_means = gaussians.means.clone()
    original_harmonics = gaussians.harmonics.clone()
    original_opacities = gaussians.opacities.clone()
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    mask, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=features,
        depths=depths,
        cross_check_threshold=2.0,
    )

    probes = torch.tensor([0, 3, 12, 15])
    non_probes = torch.tensor([i for i in range(16) if i not in probes.tolist()])
    assert stats["level0_tiles"] == 1
    assert mask[non_probes].all()
    assert torch.count_nonzero(gaussians.opacities[0, non_probes]) == 0
    assert not torch.equal(gaussians.means[0, probes], original_means[0, probes])
    assert not torch.equal(
        gaussians.harmonics[0, probes], original_harmonics[0, probes]
    )
    assert not torch.equal(
        gaussians.opacities[0, probes], original_opacities[0, probes]
    )
    assert torch.all((gaussians.opacities[0, probes] >= 0.0))
    assert torch.all((gaussians.opacities[0, probes] <= 1.0))
    eigenvalues = torch.linalg.eigvalsh(gaussians.covariances[0, probes])
    assert torch.all(eigenvalues >= -1e-7)


def test_saes_processes_every_context_view_in_flattened_gaussian_order():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians(view_count=2)
    features = torch.ones(1, 2, 2, 4, 4)
    depths = torch.ones(1, 2, 16, 1, 1)

    mask, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=features,
        depths=depths,
        cross_check_threshold=2.0,
        view_count=2,
    )

    non_probes = [i for i in range(16) if i not in {0, 3, 12, 15}]
    expected_zero = torch.tensor(non_probes + [16 + i for i in non_probes])
    assert stats["total_tiles_processed"] == 2
    assert stats["level0_tiles"] == 2
    assert stats["zeroed_gaussians"] == 24
    assert mask[expected_zero].all()
    assert torch.count_nonzero(gaussians.opacities[0, expected_zero]) == 0


def test_dense_diagnostic_does_not_claim_gaussian_pruning():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    original_harmonics = gaussians.harmonics.clone()
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    mask, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=features,
        depths=depths,
        cross_check_threshold=2.0,
        materialization="dense-diagnostic",
    )

    non_probes = torch.tensor([i for i in range(16) if i not in {0, 3, 12, 15}])
    assert mask[non_probes].all()
    assert stats["zeroed_gaussians"] == 0
    assert stats["effective_gaussians"] == 16
    assert torch.count_nonzero(gaussians.opacities[0, non_probes]) == 12
    assert not torch.equal(
        gaussians.harmonics[0, non_probes], original_harmonics[0, non_probes]
    )
