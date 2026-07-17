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


def test_representative_indices_preserve_all_l1_native_anchors():
    from saes.progressive_saes import ProgressiveSAES
    from scripts.saes_diagnostics import representative_indices

    retained = [
        row * 4 + column
        for row, column in ProgressiveSAES.compute_lightweight_positions(4)
    ]
    modified = torch.ones(16, dtype=torch.bool)
    modified[torch.tensor(retained)] = False

    indices = representative_indices(
        modified,
        view_count=1,
        height=4,
        width=4,
        tile_size=4,
        primitives_per_pixel=1,
    )

    assert indices.tolist() == sorted(retained)


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


def test_runtime_diagnostics_do_not_expose_paper_target_lookup():
    import scripts.saes_diagnostics as diagnostics

    assert not hasattr(diagnostics, "declared_level0_rate")


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


def test_materialization_attribute_audit_is_target_free_and_tracks_l0_errors():
    from scripts.saes_diagnostics import materialization_attribute_audit

    original = SimpleNamespace(
        means=torch.ones(1, 16, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1),
        harmonics=torch.full((1, 16, 3, 1), 0.5),
        opacities=torch.full((1, 16), 0.25),
    )
    sparse = SimpleNamespace(
        means=original.means.clone(),
        covariances=original.covariances.clone(),
        harmonics=original.harmonics.clone(),
        opacities=original.opacities.clone(),
    )
    non_probes = torch.tensor([index for index in range(16) if index not in {0, 3, 12, 15}])
    sparse.opacities[0, non_probes] = 0.0
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    record = materialization_attribute_audit(
        original,
        sparse,
        features=features,
        depths=depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        view_count=1,
        decision_semantics="current",
    )

    assert record["paper_result_eligible"] is False
    assert record["routing_signal_used"] is False
    assert record["full_stage3_reference_use"] == "posthoc-diagnostic-only"
    assert record["levels"]["L0"]["tiles"] == 1
    assert record["levels"]["L1"]["tiles"] == 0
    assert record["levels"]["L0"]["non_probe_attribute_error"]["mean_relative"]["count"] == 12
    assert record["levels"]["L0"]["non_probe_attribute_error"]["mean_relative"]["maximum"] == pytest.approx(0.0)
    assert record["levels"]["L0"]["non_probe_attribute_error"]["opacity_absolute"]["maximum"] == pytest.approx(0.0)
    assert record["full_oracle_representative_error_l0"]["mean_relative"]["maximum"] == pytest.approx(0.0)
    assert record["full_oracle_representative_error_l0"]["covariance_relative"]["maximum"] == pytest.approx(0.0)
    assert record["full_oracle_representative_error_l0"]["harmonic_relative"]["maximum"] == pytest.approx(0.0)
    assert record["full_oracle_representative_error_l0"]["opacity_absolute"]["maximum"] == pytest.approx(0.0)
    # Four retained alpha=0.25 anchors represent only one quarter of the
    # full tile's optical depth.  The audit must expose that loss in the
    # alpha-compositing domain instead of treating alpha averages as mass.
    optical_depth = record["levels"]["L0"]["optical_depth_conservation"]
    assert optical_depth["signed_error"]["count"] == 1
    assert optical_depth["signed_error"]["p50"] == pytest.approx(-0.75)
    assert optical_depth["relative_error"]["p50"] == pytest.approx(0.75)


def test_materialization_audit_compares_corner_depth_reconstruction_posthoc():
    from scripts.saes_diagnostics import materialization_attribute_audit

    original = SimpleNamespace(
        means=torch.ones(1, 16, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1),
        harmonics=torch.full((1, 16, 3, 1), 0.5),
        opacities=torch.full((1, 16), 0.25),
    )
    sparse = SimpleNamespace(
        means=original.means.clone(),
        covariances=original.covariances.clone(),
        harmonics=original.harmonics.clone(),
        opacities=original.opacities.clone(),
    )
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.empty(1, 1, 16, 1, 1)
    for row in range(4):
        for column in range(4):
            depths[0, 0, row * 4 + column, 0, 0] = 1.0 + row + 2.0 * column

    record = materialization_attribute_audit(
        original,
        sparse,
        features=features,
        depths=depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        view_count=1,
        decision_semantics="current",
    )

    depth_oracle = record["posthoc_nonprobe_s2_depth_reconstruction"]
    assert depth_oracle["reference_use"] == "posthoc full S2 non-probe depth only"
    assert (
        depth_oracle["assignment_weighted_relative_error"]["L0"]["p50"]
        > 0.1
    )
    assert depth_oracle["corner_bilinear_relative_error"]["L0"]["count"] == 12
    assert (
        depth_oracle["corner_bilinear_relative_error"]["L0"]["maximum"]
        < 1e-6
    )


def test_materialization_attribute_audit_builds_selected_l1_anchor_oracle():
    from saes.progressive_saes import apply_progressive_saes
    from scripts.saes_diagnostics import materialization_attribute_audit

    # Constant Stage-3 attributes make the exact full oracle coincide with the
    # probe-only materialization.  The deliberately varying probe features
    # bypass L0 and the uniform S2 depths select L1 instead.
    original = SimpleNamespace(
        means=torch.ones(1, 16, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1),
        harmonics=torch.full((1, 16, 3, 1), 0.5),
        opacities=torch.full((1, 16), 0.25),
    )
    sparse = SimpleNamespace(
        means=original.means.clone(),
        covariances=original.covariances.clone(),
        harmonics=original.harmonics.clone(),
        opacities=original.opacities.clone(),
    )
    features = torch.zeros(1, 1, 2, 4, 4)
    for (row, column), vector in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, row, column] = torch.tensor(vector)
    depths = torch.ones(1, 1, 16, 1, 1)

    _, stats, _ = apply_progressive_saes(
        sparse,
        H=4,
        W=4,
        tile_size=4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        decision_semantics="current",
    )
    record = materialization_attribute_audit(
        original,
        sparse,
        features=features,
        depths=depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        view_count=1,
        decision_semantics="current",
    )

    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 1
    assert record["levels"]["L1"]["tiles"] == 1
    assert record["full_oracle_l1_anchor_semantics"].startswith("posthoc full Stage-3")
    for key in ("mean_relative", "covariance_relative", "harmonic_relative", "opacity_absolute"):
        assert record["full_oracle_representative_error_l1"][key]["count"] == 8
        assert record["full_oracle_representative_error_l1"][key]["maximum"] == pytest.approx(
            0.0, abs=1e-6
        )
        for anchor_kind in ("primary_probe", "selected_lightweight_anchor"):
            assert (
                record["full_oracle_representative_error_l1_by_anchor_kind"]
                [anchor_kind][key]["count"]
                == 4
            )
    mixture = record["full_oracle_tile_mixture_error"]["L1"]
    for key in (
        "mean_relative",
        "covariance_relative",
        "harmonic_relative",
        "opacity_average_absolute",
    ):
        assert mixture[key]["count"] == 1
        assert mixture[key]["maximum"] == pytest.approx(0.0, abs=1e-6)
    assert mixture["optical_depth_relative"]["p50"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "materialization",
    (
        "virtual-reconstruction-diagnostic",
    ),
)
def test_materialization_attribute_audit_compares_virtual_outputs_at_their_pixels(
    materialization,
):
    from scripts.saes_diagnostics import materialization_attribute_audit

    original = SimpleNamespace(
        means=torch.full((1, 16, 3), 2.0),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1),
        harmonics=torch.full((1, 16, 3, 1), 0.5),
        opacities=torch.full((1, 16), 0.25),
    )
    sparse = SimpleNamespace(
        means=original.means.clone(),
        covariances=original.covariances.clone(),
        harmonics=original.harmonics.clone(),
        opacities=original.opacities.clone(),
    )
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    record = materialization_attribute_audit(
        original,
        sparse,
        features=features,
        depths=depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        view_count=1,
        decision_semantics="current",
        materialization=materialization,
    )

    assert record["materialization"] == materialization
    assert record["full_oracle_representative_error_applicable"] is False
    assert record["non_probe_comparison"].startswith("direct reconstructed")
    errors = record["levels"]["L0"]["virtual_non_probe_reconstruction_error"]
    for key in ("mean_relative", "covariance_relative", "harmonic_relative", "opacity_absolute"):
        assert errors[key]["count"] == 12
        assert errors[key]["maximum"] == pytest.approx(0.0)
    assert record["levels"]["L0"]["optical_depth_conservation"]["relative_error"]["maximum"] == pytest.approx(0.0)


def test_raw_s3_probe_interpolation_audit_is_posthoc_and_constant_preserving():
    from scripts.saes_diagnostics import raw_s3_probe_interpolation_audit

    raw = torch.zeros(1, 1, 16, 1, 10)
    raw[..., 3:7] = torch.tensor((0.0, 0.0, 0.0, 1.0))
    raw[..., 7:] = 0.5
    depths = torch.ones(1, 1, 16, 1, 1)
    opacities = torch.full((1, 1, 16, 1, 1), 0.25)
    coordinates = torch.zeros(1, 1, 16, 1, 2)
    # TranSplat's matching features are lower resolution than the S2/S3 grid;
    # the audit must use the same explicit bilinear router resampling as SAES.
    features = torch.zeros(1, 1, 2, 2, 2)

    record = raw_s3_probe_interpolation_audit(
        raw,
        depths,
        opacities,
        coordinates,
        features=features,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="current",
    )

    assert record["kind"] == "saes_s3_raw_descriptor_audit"
    assert record["target_rgb_accessed"] is False
    assert record["full_s3_reference_use"] == "posthoc-diagnostic-only"
    assert record["aggregate_moment_matching_executed"] is False
    assert record["retained_descriptor_output_emitted"] is False
    assert "not a retained-anchor" in record["audit_scope"]
    assert record["tile_counts"] == {"L0": 1, "L1": 0}
    assert record["feature_grid"] == {
        "input_height": 2,
        "input_width": 2,
        "routing_height": 4,
        "routing_width": 4,
        "resampling": "bilinear-align_corners-false-via-ProgressiveSAES",
    }
    assert record["assignment_weight_sum_error_max"] <= 1e-6
    for method in ("assignment_interpolation", "nearest_anchor"):
        for metric, summary in record["errors"]["L0"][method].items():
            assert summary["count"] == 12
            assert summary["maximum"] == pytest.approx(0.0, abs=1e-6), metric
