from types import SimpleNamespace
import json

import pytest


torch = pytest.importorskip("torch")


def _gaussians():
    return SimpleNamespace(
        means=torch.arange(48, dtype=torch.float32).reshape(1, 16, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1),
        harmonics=torch.ones(1, 16, 3, 1),
        opacities=torch.full((1, 16), 0.25),
    )


def test_representative_indices_follow_modified_tile_layout():
    from scripts.saes_diagnostics import representative_indices

    modified = torch.ones(16, dtype=torch.bool)
    modified[torch.tensor([0, 3, 12, 15])] = False

    indices = representative_indices(
        modified,
        view_count=1,
        height=4,
        width=4,
        tile_size=4,
        primitives_per_pixel=1,
    )

    assert indices.tolist() == [0, 3, 12, 15]


def test_coverage_variant_changes_only_retained_representatives():
    from scripts.saes_diagnostics import build_coverage_variant

    gaussians = _gaussians()
    representatives = torch.tensor([0, 3, 12, 15])
    variant = build_coverage_variant(
        gaussians,
        representatives,
        covariance_scale=4.0,
        opacity_scale=2.0,
    )

    assert torch.equal(variant.means, gaussians.means)
    assert torch.equal(variant.harmonics, gaussians.harmonics)
    assert torch.allclose(
        variant.covariances[0, representatives],
        gaussians.covariances[0, representatives] * 4.0,
    )
    expected_opacity = 1.0 - (1.0 - 0.25) ** 2.0
    assert torch.allclose(
        variant.opacities[0, representatives],
        torch.full((4,), expected_opacity),
    )
    untouched = torch.tensor([index for index in range(16) if index not in representatives])
    assert torch.equal(
        variant.covariances[0, untouched], gaussians.covariances[0, untouched]
    )
    assert torch.equal(variant.opacities[0, untouched], gaussians.opacities[0, untouched])
    assert variant.covariances.data_ptr() != gaussians.covariances.data_ptr()


def test_decision_statistics_use_vector_variance_and_absolute_depth_spread():
    from scripts.saes_diagnostics import decision_statistics

    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (0.0, 0.0),
        (0, 3): (2.0, 0.0),
        (3, 0): (0.0, 2.0),
        (3, 3): (2.0, 2.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)
    depths[0, 0, 15, 0, 0] = 3.0

    record = decision_statistics(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
    )

    assert record["tile_count"] == 1
    assert record["feature"]["raw_vector_variance"]["mean"] == pytest.approx(2.0)
    assert record["depth"]["absolute_std"]["mean"] == pytest.approx(3**0.5 / 2)
    assert record["depth"]["relative_std"]["mean"] == pytest.approx(3**0.5 / 3)


def test_candidate_coordinate_depth_and_first_hit_are_near_far_normalized():
    from scripts.saes_diagnostics import decision_statistics, normalize_inverse_depth

    near = torch.tensor([[1.0]])
    far = torch.tensor([[5.0]])
    metric = torch.tensor([[[[[1.0]], [[5.0]], [[5.0 / 3.0]]]]])
    normalized = normalize_inverse_depth(metric, near=near, far=far)
    assert normalized.flatten().tolist() == pytest.approx([1.0, 0.0, 0.5])

    features = torch.zeros(1, 1, 2, 4, 8)
    for (y, x), value in {
        (0, 4): (0.0, 0.0),
        (0, 7): (2.0, 0.0),
        (3, 4): (0.0, 2.0),
        (3, 7): (2.0, 2.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    midpoint_depth = 1.0 / ((1.0 / 5.0) + 0.5 * ((1.0 / 1.0) - (1.0 / 5.0)))
    depths = torch.full((1, 1, 32, 1, 1), midpoint_depth)

    record = decision_statistics(
        features,
        depths,
        height=4,
        width=8,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        near=near,
        far=far,
    )

    assert record["depth"]["candidate_coordinate_std"]["mean"] == pytest.approx(0.0)
    first_hit = record["first_hit"]["raw_vector_variance"][
        "candidate_coordinate_std"
    ]
    assert first_hit["l0_count"] == 1
    assert first_hit["l1_count"] == 1
    assert first_hit["full_count"] == 0


def test_probe_feature_variances_are_keyed_by_view_and_tile():
    from scripts.saes_diagnostics import probe_feature_variances

    features = torch.zeros(1, 1, 2, 4, 8)
    for (y, x), value in {
        (0, 0): (0.0, 0.0),
        (0, 3): (2.0, 0.0),
        (3, 0): (0.0, 2.0),
        (3, 3): (2.0, 2.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)

    scores = probe_feature_variances(
        features,
        height=4,
        width=8,
        tile_size=4,
    )

    assert scores[(0, 0, 0)] == pytest.approx(2.0)
    assert scores[(0, 0, 1)] == pytest.approx(0.0)


def test_ranked_tile_subset_changes_only_lowest_score_eligible_tile():
    from scripts.saes_diagnostics import build_ranked_tile_subset_variant

    original = SimpleNamespace(
        means=torch.zeros(1, 32, 3),
        covariances=torch.zeros(1, 32, 3, 3),
        harmonics=torch.zeros(1, 32, 3, 1),
        opacities=torch.zeros(1, 32),
    )
    sparse = SimpleNamespace(
        means=torch.ones(1, 32, 3),
        covariances=torch.ones(1, 32, 3, 3),
        harmonics=torch.ones(1, 32, 3, 1),
        opacities=torch.ones(1, 32),
    )
    modified = torch.zeros(32, dtype=torch.bool)
    modified[[index for index in range(16) if index not in {0, 3, 12, 15}]] = True
    modified[
        [16 + index for index in range(16) if index not in {0, 3, 12, 15}]
    ] = True

    variant, metadata = build_ranked_tile_subset_variant(
        original,
        sparse,
        modified,
        {(0, 0, 0): 0.8, (0, 0, 1): 0.2},
        view_count=1,
        height=4,
        width=8,
        tile_size=4,
        primitives_per_pixel=1,
        target_fraction=0.5,
        ranking_statistic="test_score",
    )

    first_tile = torch.tensor(
        [y * 8 + x for y in range(4) for x in range(4)]
    )
    second_tile = torch.tensor(
        [y * 8 + x for y in range(4) for x in range(4, 8)]
    )
    assert torch.count_nonzero(variant.opacities[0, first_tile]) == 0
    assert torch.count_nonzero(variant.opacities[0, second_tile]) == 16
    assert metadata["selected_tile_count"] == 1
    assert metadata["eligible_tile_count"] == 2
    assert metadata["selected_fraction"] == pytest.approx(0.5)


def test_declared_level0_rate_comes_from_expected_results(tmp_path):
    from scripts.saes_diagnostics import declared_level0_rate

    expected = tmp_path / "expected.json"
    expected.write_text(
        json.dumps(
            {
                "mechanisms": {
                    "transplat/re10k": {"level0_rate": 0.172},
                }
            }
        ),
        encoding="utf-8",
    )

    assert declared_level0_rate(expected, "transplat", "re10k") == pytest.approx(
        0.172
    )


def test_probe_cross_check_errors_are_target_free_tile_scores():
    from scripts.saes_diagnostics import probe_cross_check_errors

    gaussians = SimpleNamespace(
        means=torch.zeros(1, 32, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 32, 1, 1),
        harmonics=torch.ones(1, 32, 3, 1),
        opacities=torch.full((1, 32), 0.25),
    )
    gaussians.harmonics[0, 4] *= -1.0

    scores = probe_cross_check_errors(
        gaussians,
        view_count=1,
        height=4,
        width=8,
        tile_size=4,
        primitives_per_pixel=1,
    )

    assert scores[(0, 0, 0)] == pytest.approx(0.0, abs=1e-6)
    assert scores[(0, 0, 1)] > scores[(0, 0, 0)]


def test_component_variant_restores_only_named_representative_attributes():
    from scripts.saes_diagnostics import build_component_variant

    original = _gaussians()
    sparse = _gaussians()
    sparse.means += 100.0
    sparse.covariances *= 3.0
    sparse.harmonics *= 4.0
    sparse.opacities *= 2.0
    representatives = torch.tensor([0, 3, 12, 15])

    variant = build_component_variant(
        sparse,
        original,
        representatives,
        restore=("means", "opacities"),
    )

    assert torch.equal(variant.means[0, representatives], original.means[0, representatives])
    assert torch.equal(
        variant.opacities[0, representatives], original.opacities[0, representatives]
    )
    assert torch.equal(
        variant.covariances[0, representatives], sparse.covariances[0, representatives]
    )
    assert torch.equal(
        variant.harmonics[0, representatives], sparse.harmonics[0, representatives]
    )
    untouched = torch.tensor([index for index in range(16) if index not in representatives])
    assert torch.equal(variant.means[0, untouched], sparse.means[0, untouched])
