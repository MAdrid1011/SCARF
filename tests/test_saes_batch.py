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


def _adapter_compatible_gaussians():
    """Return a 4x4 fixture whose means follow the adapter ray contract."""
    gaussians = _gaussians()
    depth = 2.0
    for row in range(4):
        for column in range(4):
            offset = torch.tensor(
                (
                    0.04 if (row + column) % 2 else -0.03,
                    -0.02 if row % 2 else 0.03,
                )
            )
            coordinate = torch.tensor(
                ((column + 0.5) / 4, (row + 0.5) / 4, 1.0)
            )
            coordinate[:2] += offset
            gaussians.means[0, row * 4 + column] = (
                coordinate / coordinate.norm() * depth
            )
    return gaussians


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


def test_probe_vector_variance_uses_only_paper_probes_per_view():
    from saes.progressive_saes import ProgressiveSAES

    features = torch.zeros(1, 2, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (0.0, 0.0),
        (0, 3): (2.0, 0.0),
        (3, 0): (0.0, 2.0),
        (3, 3): (2.0, 2.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    features[0, 0, :, 1:3, 1:3] = 1000.0
    features[0, 1] = 7.0

    variances, _ = ProgressiveSAES.classify_tiles_by_features(
        features,
        4,
        4,
        tile_size=4,
        per_view=True,
        statistic="raw-probe-vector-variance",
    )

    assert variances == pytest.approx({(0, 0, 0): 2.0, (1, 0, 0): 0.0})


def test_normalized_probe_total_variance_matches_reference_vectors():
    from saes.progressive_saes import ProgressiveSAES

    features = torch.zeros(1, 1, 2, 4, 4)
    probes = {(0, 0): (1.0, 0.0), (0, 3): (0.0, 1.0),
              (3, 0): (-1.0, 0.0), (3, 3): (0.0, -1.0)}
    for position, value in probes.items():
        features[0, 0, :, position[0], position[1]] = torch.tensor(value)

    variances, _ = ProgressiveSAES.classify_tiles_by_features(
        features,
        4,
        4,
        tile_size=4,
        per_view=True,
        statistic="normalized-probe-total-variance",
    )

    assert variances[(0, 0, 0)] == pytest.approx(1.0)


def test_normalized_probe_vector_standard_deviation_is_the_variance_square_root():
    from saes.progressive_saes import ProgressiveSAES

    features = torch.zeros(1, 1, 2, 4, 4)
    probes = {(0, 0): (1.0, 0.0), (0, 3): (0.0, 1.0),
              (3, 0): (-1.0, 0.0), (3, 3): (0.0, -1.0)}
    for position, value in probes.items():
        features[0, 0, :, position[0], position[1]] = torch.tensor(value)

    variances, _ = ProgressiveSAES.classify_tiles_by_features(
        features,
        4,
        4,
        tile_size=4,
        per_view=True,
        statistic="normalized-probe-total-variance",
    )
    standard_deviations, _ = ProgressiveSAES.classify_tiles_by_features(
        features,
        4,
        4,
        tile_size=4,
        per_view=True,
        statistic="normalized-probe-vector-standard-deviation",
    )

    assert standard_deviations[(0, 0, 0)] == pytest.approx(
        variances[(0, 0, 0)] ** 0.5
    )


def test_raw_probe_mean_channel_variance_is_channel_width_invariant():
    from saes.progressive_saes import ProgressiveSAES

    features = torch.zeros(1, 1, 2, 4, 4)
    probes = {(0, 0): (0.0, 0.0), (0, 3): (2.0, 2.0),
              (3, 0): (0.0, 0.0), (3, 3): (2.0, 2.0)}
    for position, value in probes.items():
        features[0, 0, :, position[0], position[1]] = torch.tensor(value)
    repeated = features.repeat(1, 1, 8, 1, 1)

    base, _ = ProgressiveSAES.classify_tiles_by_features(
        features,
        4,
        4,
        tile_size=4,
        per_view=True,
        statistic="raw-probe-mean-channel-variance",
    )
    wide, _ = ProgressiveSAES.classify_tiles_by_features(
        repeated,
        4,
        4,
        tile_size=4,
        per_view=True,
        statistic="raw-probe-mean-channel-variance",
    )

    assert base[(0, 0, 0)] == pytest.approx(1.0)
    assert wide[(0, 0, 0)] == pytest.approx(base[(0, 0, 0)])


def test_camera_world_point_matches_the_shared_c2w_ray_convention():
    from saes.progressive_saes import ProgressiveSAES

    extrinsics = torch.eye(4).reshape(1, 1, 4, 4)
    extrinsics[0, 0, :3, 3] = torch.tensor((2.0, -1.0, 0.5))
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)

    euclidean = ProgressiveSAES.camera_world_point(
        extrinsics,
        intrinsics,
        view_index=0,
        row=2,
        column=1,
        height=4,
        width=4,
        depth=torch.tensor(3.0),
        ray_depth_mode="euclidean",
    )
    z_depth = ProgressiveSAES.camera_world_point(
        extrinsics,
        intrinsics,
        view_index=0,
        row=2,
        column=1,
        height=4,
        width=4,
        depth=torch.tensor(3.0),
        ray_depth_mode="z",
    )

    raw_direction = torch.tensor((0.375, 0.625, 1.0))
    origin = torch.tensor((2.0, -1.0, 0.5))
    torch.testing.assert_close(
        euclidean, origin + 3.0 * raw_direction / raw_direction.norm()
    )
    torch.testing.assert_close(z_depth, origin + 3.0 * raw_direction)


def test_camera_geometry_requires_a_complete_c2w_intrinsics_pair():
    from saes.progressive_saes import apply_progressive_saes

    with pytest.raises(ValueError, match="both context extrinsics and intrinsics"):
        apply_progressive_saes(
            _gaussians(),
            4,
            4,
            features=torch.ones(1, 1, 2, 4, 4),
            depths=torch.ones(1, 1, 16, 1, 1),
            context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        )


def test_camera_aware_moment_matching_uses_rays_without_nonprobe_stage3_reads():
    from saes.progressive_saes import apply_progressive_saes

    plain = _gaussians()
    camera_aware = _gaussians()
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)
    options = {
        "feature_var_threshold": 1.0,
        "depth_std_threshold": 1.0,
        "features": features,
        "depths": depths,
    }
    apply_progressive_saes(plain, 4, 4, **options)
    _, stats, _ = apply_progressive_saes(
        camera_aware,
        4,
        4,
        **options,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
    )

    probes = torch.tensor([0, 3, 12, 15])
    assert stats["camera_aware_moment_matching"] is True
    assert not torch.equal(camera_aware.means[0, probes], plain.means[0, probes])


def test_camera_aware_interpolation_preserves_probe_subpixel_offsets():
    from saes.progressive_saes import ProgressiveSAES

    saes = ProgressiveSAES(
        4,
        4,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
    )
    positions = [(0, 0), (0, 3)]
    depths = torch.tensor((2.0, 2.0))
    source_rays = torch.stack(
        [saes._camera_world_point(0, row, column, depth) for (row, column), depth in zip(positions, depths)]
    )
    offset = torch.tensor((0.03, -0.02, 0.01))
    source_means = source_rays + offset
    weights = torch.tensor((0.25, 0.75))

    actual = saes._camera_aware_interpolated_mean(
        source_means,
        depths,
        positions,
        view_index=0,
        target_row=2,
        target_column=1,
        weights=weights,
    )
    expected = saes._camera_world_point(0, 2, 1, torch.tensor(2.0)) + offset

    torch.testing.assert_close(actual, expected)


def test_virtual_covariance_interpolation_keeps_only_intrinsic_shape():
    from saes.progressive_saes import ProgressiveSAES

    source_covariances = torch.stack(
        (
            torch.diag(torch.tensor((1.0, 2.0, 3.0))),
            torch.diag(torch.tensor((5.0, 7.0, 11.0))),
        )
    )
    weights = torch.tensor((0.25, 0.75))

    actual = ProgressiveSAES._interpolate_intrinsic_covariance(
        source_covariances, weights
    )
    expected = 0.25 * source_covariances[0] + 0.75 * source_covariances[1]

    torch.testing.assert_close(actual, expected)
    assert torch.all(torch.linalg.eigvalsh(actual) > 0.0)


def test_default_first_hit_routing_uses_no_gaussian_similarity_gate():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        cross_check_threshold=0.0,
    )

    assert stats["level0_tiles"] == 1
    assert stats["level1_tiles"] == 0
    assert stats["full_tiles"] == 0


def test_probe_vector_first_hit_routes_l1_without_unpublished_gaussian_gate():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (0.0, 0.0),
        (0, 3): (2.0, 0.0),
        (3, 0): (0.0, 2.0),
        (3, 3): (2.0, 2.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)

    mask, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        cross_check_threshold=0.0,
        decision_semantics="probe-vector-first-hit",
    )

    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 1
    assert stats["full_tiles"] == 0
    assert stats["decision_semantics"] == "probe-vector-first-hit"
    assert stats["zeroed_gaussians"] == 8
    assert stats["effective_gaussians"] == 8
    assert torch.count_nonzero(mask) == 8


def test_normalized_probe_standard_deviation_is_a_first_hit_diagnostic():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    features = torch.zeros(1, 1, 2, 4, 4)
    probes = {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.8, 0.6),
        (3, 0): (1.0, 0.0),
        (3, 3): (0.8, 0.6),
    }
    for (y, x), value in probes.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        decision_semantics="probe-normalized-std-first-hit",
    )

    assert stats["feature_statistic"] == "normalized-probe-vector-standard-deviation"
    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 1
    assert stats["full_tiles"] == 0


def test_standard_deviation_decision_squares_back_to_kernel_variance():
    from saes.progressive_saes import ProgressiveSAES

    standard_deviation = ProgressiveSAES(
        4, 4, decision_semantics="probe-normalized-std-first-hit"
    )
    variance = ProgressiveSAES(4, 4, decision_semantics="current")

    assert standard_deviation._assignment_feature_variance(0.2) == pytest.approx(0.04)
    assert variance._assignment_feature_variance(0.2) == pytest.approx(0.2)
    assert standard_deviation._assignment_feature_variance(float("inf")) == float("inf")


def test_inverse_depth_candidate_routing_uses_s2_coordinate_without_changing_merge_depths():
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    features = torch.zeros(1, 1, 2, 4, 4)
    for (row, column), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (1.0, 0.0),
        (3, 3): (0.0, 1.0),
    }.items():
        features[0, 0, :, row, column] = torch.tensor(value)
    depths = torch.full((1, 1, 16, 1, 1), 100.0)
    depths[0, 0, 3, 0, 0] = 120.0
    depths[0, 0, 12, 0, 0] = 120.0
    near = torch.tensor([[0.1]])
    far = torch.tensor([[1000.0]])

    metric_mask, metric_stats, _ = apply_progressive_saes(
        _gaussians(),
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        decision_semantics="probe-normalized-std-first-hit",
    )
    candidate_mask, candidate_stats, _ = apply_progressive_saes(
        _gaussians(),
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        decision_semantics="probe-normalized-std-first-hit",
        depth_routing_semantics="inverse-depth-candidate-coordinate-standard-deviation",
        depth_near=near,
        depth_far=far,
    )

    assert metric_stats["full_tiles"] == 1
    assert candidate_stats["level1_tiles"] == 1
    assert candidate_stats["depth_statistic"] == (
        "inverse-depth-candidate-coordinate-standard-deviation"
    )
    assert not bool(metric_mask.any())
    assert bool(candidate_mask.any())
    normalized = ProgressiveSAES.inverse_depth_candidate_coordinate(
        depths, near=near, far=far
    )
    assert normalized[0, 0, 0, 0, 0] != normalized[0, 0, 3, 0, 0]


def test_inverse_depth_candidate_routing_requires_context_bounds():
    from saes.progressive_saes import apply_progressive_saes

    with pytest.raises(ValueError, match="requires depth_near and depth_far"):
        apply_progressive_saes(
            _gaussians(),
            4,
            4,
            features=torch.ones(1, 1, 2, 4, 4),
            depths=torch.ones(1, 1, 16, 1, 1),
            depth_routing_semantics="inverse-depth-candidate-coordinate-standard-deviation",
        )


def test_pre_fallback_routing_ledger_keeps_thresholds_fixed_and_separates_units():
    from scripts.saes_target_free_materialization_audit import (
        _pre_fallback_routing_summary,
    )

    features = torch.zeros(1, 1, 2, 4, 4)
    for (row, column), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (1.0, 0.0),
        (3, 3): (0.0, 1.0),
    }.items():
        features[0, 0, :, row, column] = torch.tensor(value)
    depths = torch.full((1, 1, 16, 1, 1), 100.0)
    depths[0, 0, 3, 0, 0] = 120.0
    depths[0, 0, 12, 0, 0] = 120.0

    report = _pre_fallback_routing_summary(
        features,
        depths,
        near=torch.tensor([[0.1]]),
        far=torch.tensor([[1000.0]]),
        height=4,
        width=4,
    )

    assert report["metric-depth-standard-deviation"] == {
        "diagnostic_only": True,
        "tile_count": 1,
        "l0_count": 0,
        "l1_count": 0,
        "full_count": 1,
        "l0_rate": 0.0,
        "l1_rate": 0.0,
        "full_rate": 1.0,
    }
    assert report["inverse-depth-candidate-coordinate-standard-deviation"]["l1_count"] == 1


def test_retained_anchor_attribute_diagnostic_excludes_skipped_descriptors():
    from scripts.saes_target_free_materialization_audit import (
        _retained_anchor_attribute_diagnostic,
    )

    source = _gaussians()
    materialized = _gaussians()
    retained = torch.tensor([0, 3, 12, 15])
    materialized.means[0, 0, 0] += 2.0
    materialized.covariances[0, 0] *= 4.0
    materialized.opacities[0, 0] *= 0.5

    report = _retained_anchor_attribute_diagnostic(source, materialized, retained)

    assert report["changed_retained_anchor_count"] == 1
    assert report["source_or_skipped_descriptor_access"] is False
    assert report["mean_displacement_l2"]["p50"] == pytest.approx(2.0)
    assert report["covariance_determinant_ratio"]["p50"] == pytest.approx(64.0)
    assert report["source_covariance_min_eigenvalue"]["p50"] == pytest.approx(0.01)
    assert report["source_covariance_asymmetry_frobenius"]["p50"] == pytest.approx(0.0)
    assert report["covariance_increment_min_eigenvalue"]["p50"] == pytest.approx(0.03)
    assert report["opacity_ratio"]["p50"] == pytest.approx(0.5)


def test_l1_lightweight_positions_double_the_representative_anchors():
    from saes.progressive_saes import ProgressiveSAES

    representative = ProgressiveSAES.compute_probe_positions(4)
    lightweight = ProgressiveSAES.compute_lightweight_positions(4)

    assert len(representative) == 4
    assert len(lightweight) == 8
    assert lightweight[:4] == representative
    assert len(set(lightweight)) == len(lightweight)


def test_probe_cross_check_exposes_the_continuous_error():
    from saes.progressive_saes import ProgressiveSAES

    gaussians = _gaussians()
    probes = [0, 3, 12, 15]
    saes = ProgressiveSAES(4, 4, cross_check_threshold=1e-6)

    baseline_error = saes.probe_cross_check_error(gaussians, probes)
    gaussians.harmonics[0, probes[0]] *= -1.0
    changed_error = saes.probe_cross_check_error(gaussians, probes)

    assert baseline_error >= 0.0
    assert changed_error > baseline_error
    assert saes.probe_cross_check(gaussians, probes) == (
        changed_error <= saes.cross_check_threshold
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
        materialization_guard=False,
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
    assert stats["assignment_weight_sum_error_max"] <= 1e-6
    assert stats["opacity_transmittance_error_max"] <= 1e-6
    assert stats["covariance_psd_violations"] == 0


def test_representative_path_preserves_constant_sh_and_range_bounded_opacity():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    probes = torch.tensor([0, 3, 12, 15])
    gaussians.harmonics[:] = 0.375
    gaussians.opacities[0, probes] = torch.tensor((0.10, 0.20, 0.30, 0.40))
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=features,
        depths=depths,
        materialization_guard=False,
    )

    # With identical probe features the bilateral assignment is uniform. Each
    # representative averages its own opacity (weight 1) with 12 virtual
    # values (weight 1/4), keeping the result inside the contributor range.
    expected_opacity = (torch.tensor((0.10, 0.20, 0.30, 0.40)) + 0.75) / 4.0
    torch.testing.assert_close(gaussians.opacities[0, probes], expected_opacity)
    assert torch.all(gaussians.opacities[0, probes] >= 0.10)
    assert torch.all(gaussians.opacities[0, probes] <= 0.40)
    torch.testing.assert_close(
        gaussians.harmonics[0, probes], torch.full_like(gaussians.harmonics[0, probes], 0.375)
    )
    assert stats["opacity_transmittance_error_max"] <= 1e-6


def test_representative_path_never_reads_non_probe_stage3_attributes():
    from saes.progressive_saes import apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    probes = torch.tensor([0, 3, 12, 15])
    non_probes = torch.tensor([i for i in range(16) if i not in probes.tolist()])
    perturbed.means[0, non_probes] = 1e4
    perturbed.covariances[0, non_probes] = -1e4
    perturbed.harmonics[0, non_probes] = 1e4
    perturbed.opacities[0, non_probes] = -1e4
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=1.0,
            depth_std_threshold=1.0,
            features=features,
            depths=depths,
        )

    assert torch.equal(baseline.means[0, probes], perturbed.means[0, probes])
    assert torch.equal(
        baseline.covariances[0, probes], perturbed.covariances[0, probes]
    )
    assert torch.equal(baseline.harmonics[0, probes], perturbed.harmonics[0, probes])
    assert torch.equal(baseline.opacities[0, probes], perturbed.opacities[0, probes])


def test_conditional_anchor_transport_avoids_cross_anchor_attribute_leakage():
    from saes.progressive_saes import apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    mixture = _gaussians()
    probes = torch.tensor([0, 3, 12, 15])
    non_probes = torch.tensor([i for i in range(16) if i not in probes.tolist()])
    original = {
        name: getattr(baseline, name)[0, probes].clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    perturbed.means[0, non_probes] = 1e4
    perturbed.covariances[0, non_probes] = -1e4
    perturbed.harmonics[0, non_probes] = 1e4
    perturbed.opacities[0, non_probes] = 0.99
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        mask, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=1.0,
            depth_std_threshold=1.0,
            features=features,
            depths=depths,
            materialization="conditional-anchor-transport-diagnostic",
        )
        assert stats["merge_semantics"] == "conditional-anchor-transport"
        assert stats["level0_tiles"] == 1
        assert mask[non_probes].all()
        assert torch.count_nonzero(gaussians.opacities[0, non_probes]) == 0
        assert stats["covariance_psd_violations"] == 0

    # With no camera geometry, a conditional transport has each probe absorb
    # only copies of itself.  It must not mix any other selected anchor's SH or
    # opacity, and it must not depend on withheld non-probe S3 attributes.
    for name, value in original.items():
        torch.testing.assert_close(getattr(baseline, name)[0, probes], value)
        torch.testing.assert_close(
            getattr(baseline, name)[0, probes], getattr(perturbed, name)[0, probes]
        )

    apply_progressive_saes(
        mixture,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=features,
        depths=depths,
    )
    assert not torch.allclose(mixture.harmonics[0, probes], original["harmonics"])
    assert not torch.allclose(mixture.opacities[0, probes], original["opacities"])


def test_conditional_optical_mass_is_single_assignment_psd_and_target_free():
    from saes.progressive_saes import apply_progressive_saes

    baseline = _gaussians()
    poisoned = _gaussians()
    probes = torch.tensor([0, 3, 12, 15])
    non_probes = torch.tensor([i for i in range(16) if i not in probes.tolist()])
    for gaussians in (baseline, poisoned):
        gaussians.means.fill_(1.5)
        gaussians.covariances[:] = torch.eye(3) * 0.25
        gaussians.harmonics.fill_(0.375)
        gaussians.opacities.fill_(0.25)
    poisoned.means[0, non_probes] = 1e4
    poisoned.covariances[0, non_probes] = -1e4
    poisoned.harmonics[0, non_probes] = 1e4
    poisoned.opacities[0, non_probes] = 0.99
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    expected_mass = 16 * (
        -torch.log1p(torch.tensor(-0.25)) * torch.sqrt(torch.tensor(0.25**3))
    )
    for gaussians in (baseline, poisoned):
        mask, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=1.0,
            depth_std_threshold=1.0,
            features=features,
            depths=depths,
            materialization="conditional-optical-mass-diagnostic",
        )
        assert stats["merge_semantics"] == "conditional-optical-mass"
        assert stats["conditional_mass_fallback_tiles"] == 0
        assert stats["conditional_assignment_uses"] == 12 * 4
        assert stats["conditional_mass_conservation_error_max"] <= 1e-6
        assert mask[non_probes].all()
        assert torch.count_nonzero(gaussians.opacities[0, non_probes]) == 0
        assert torch.all(torch.linalg.eigvalsh(gaussians.covariances[0, probes]) >= -1e-7)
        observed_mass = (
            -torch.log1p(-gaussians.opacities[0, probes])
            * torch.sqrt(torch.linalg.det(gaussians.covariances[0, probes] + torch.eye(3) * 1e-8))
        ).sum()
        torch.testing.assert_close(observed_mass, expected_mass, rtol=1e-5, atol=1e-6)

    # Poisoning withheld descriptors cannot alter retained anchor attributes.
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(
            getattr(baseline, name)[0, probes], getattr(poisoned, name)[0, probes]
        )


def test_conditional_optical_mass_one_hot_assignment_only_updates_receiving_anchor():
    from saes.progressive_saes import ProgressiveSAES

    gaussians = _gaussians()
    gaussians.covariances[:] = torch.eye(3) * 0.25
    gaussians.opacities.fill_(0.25)
    probes = [0, 3, 12, 15]
    non_probe_items = [
        ((row, column), row * 4 + column)
        for row in range(4)
        for column in range(4)
        if row * 4 + column not in probes
    ]
    original = {name: getattr(gaussians, name)[0, probes].clone() for name in (
        "means", "covariances", "harmonics", "opacities"
    )}
    assignments = torch.zeros(len(non_probe_items), len(probes))
    assignments[:, 0] = 1.0
    saes = ProgressiveSAES(4, 4)
    fallback = saes._conditional_optical_mass_merge(
        means=gaussians.means[0],
        covariances=gaussians.covariances[0],
        harmonics=gaussians.harmonics[0],
        opacities=gaussians.opacities[0],
        source_means=original["means"],
        source_covariances=original["covariances"],
        source_harmonics=original["harmonics"],
        source_opacities=original["opacities"],
        probe_indices=probes,
        probe_depths=torch.ones(4),
        probe_positions=[(0, 0), (0, 3), (3, 0), (3, 3)],
        non_probe_items=non_probe_items,
        assignment_matrix=assignments,
        view_index=0,
    )

    assert fallback is False
    assert saes.stats["conditional_assignment_uses"] == 12 * 4
    torch.testing.assert_close(gaussians.means[0, probes[1:]], original["means"][1:])
    torch.testing.assert_close(
        gaussians.covariances[0, probes[1:]], original["covariances"][1:]
    )
    torch.testing.assert_close(
        gaussians.harmonics[0, probes[1:]], original["harmonics"][1:]
    )
    torch.testing.assert_close(
        gaussians.opacities[0, probes[1:]], original["opacities"][1:]
    )
    expected_alpha = 1.0 - (1.0 - 0.25) ** 13
    assert gaussians.opacities[0, probes[0]].item() == pytest.approx(expected_alpha)


def test_conditional_optical_mass_falls_back_to_full_on_non_psd_anchor_input():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    probes = torch.tensor([0, 3, 12, 15])
    before = gaussians.opacities.clone()
    gaussians.covariances[0, probes] = -torch.eye(3)
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
        materialization="conditional-optical-mass-diagnostic",
    )

    assert stats["conditional_mass_fallback_tiles"] == 1
    assert stats["level0_tiles"] == 0
    assert stats["full_tiles"] == 1
    assert not bool(mask.any())
    torch.testing.assert_close(gaussians.opacities, before)


def test_conditional_optical_mass_allows_second_moment_covariance_expansion(monkeypatch):
    from saes.progressive_saes import ProgressiveSAES

    gaussians = _gaussians()
    gaussians.covariances[:] = torch.eye(3) * 0.25
    gaussians.opacities.fill_(0.25)
    probes = [0, 3, 12, 15]
    non_probe_items = [
        ((row, column), row * 4 + column)
        for row in range(4)
        for column in range(4)
        if row * 4 + column not in probes
    ]
    saes = ProgressiveSAES(4, 4)

    def transported(source_mean, _depth, _position, target_positions, *, view_index):
        del view_index
        return source_mean.expand(len(target_positions), -1) + torch.tensor((10.0, 0.0, 0.0))

    monkeypatch.setattr(saes, "_anchor_conditioned_transport_means", transported)
    fallback = saes._conditional_optical_mass_merge(
        means=gaussians.means[0],
        covariances=gaussians.covariances[0],
        harmonics=gaussians.harmonics[0],
        opacities=gaussians.opacities[0],
        source_means=gaussians.means[0, probes].clone(),
        source_covariances=gaussians.covariances[0, probes].clone(),
        source_harmonics=gaussians.harmonics[0, probes].clone(),
        source_opacities=gaussians.opacities[0, probes].clone(),
        probe_indices=probes,
        probe_depths=torch.ones(4),
        probe_positions=[(0, 0), (0, 3), (3, 0), (3, 3)],
        non_probe_items=non_probe_items,
        assignment_matrix=torch.full((len(non_probe_items), len(probes)), 0.25),
        view_index=0,
    )

    assert fallback is False
    assert saes.stats["conditional_range_fallback_tiles"] == 0
    assert torch.linalg.det(gaussians.covariances[0, probes[0]]) > torch.det(
        torch.eye(3) * 0.25
    )
    assert torch.all(torch.linalg.eigvalsh(gaussians.covariances[0, probes]) >= -1e-7)


def test_context_projected_footprint_ignores_pure_depth_axis_expansion():
    from saes.progressive_saes import ProgressiveSAES

    saes = ProgressiveSAES(
        4,
        4,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
    )
    means = torch.tensor([[0.0, 0.0, 2.0]])
    planar = torch.diag(torch.tensor((0.25, 0.25, 0.25))).unsqueeze(0)
    depth_stretched = torch.diag(torch.tensor((0.25, 0.25, 25.0))).unsqueeze(0)

    planar_scale = saes._context_projected_footprint_scales(
        means, planar, view_index=0
    )
    stretched_scale = saes._context_projected_footprint_scales(
        means, depth_stretched, view_index=0
    )

    torch.testing.assert_close(planar_scale, torch.tensor((0.0625,)))
    torch.testing.assert_close(stretched_scale, planar_scale)


def test_context_projected_footprint_accepts_valid_sub_3d_epsilon_area():
    from saes.progressive_saes import ProgressiveSAES

    saes = ProgressiveSAES(
        4,
        4,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
    )
    scale = saes._context_projected_footprint_scales(
        torch.tensor([[0.0, 0.0, 2.0]]),
        (torch.eye(3) * 1.0e-10).unsqueeze(0),
        view_index=0,
    )

    assert scale is not None
    assert scale.item() == pytest.approx(2.5e-11, rel=1e-5)


def test_conditional_projected_optical_mass_requires_context_camera_geometry():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    before = gaussians.opacities.clone()
    _mask, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=torch.ones(1, 1, 2, 4, 4),
        depths=torch.ones(1, 1, 16, 1, 1),
        materialization="conditional-projected-optical-mass-diagnostic",
    )

    assert stats["conditional_mass_fallback_tiles"] == 1
    assert stats["full_tiles"] == 1
    torch.testing.assert_close(gaussians.opacities, before)


def test_target_free_materialization_helpers_clone_and_poison_only_skipped_rows():
    from scripts.saes_target_free_materialization_audit import (
        _clone_gaussians,
        _gaussians_on_cpu,
        _poison_skipped_descriptors,
    )

    source = _gaussians()
    clone = _clone_gaussians(source)
    cpu_clone = _gaussians_on_cpu(source)
    skipped = torch.tensor([1, 5, 9])
    retained = torch.tensor([0, 2, 3])
    _poison_skipped_descriptors(clone, skipped)

    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(getattr(source, name)[0, retained], getattr(clone, name)[0, retained])
        assert getattr(cpu_clone, name).device.type == "cpu"
    assert not torch.equal(source.means[0, skipped], clone.means[0, skipped])


def test_l0_range_envelope_diagnostic_reports_excursions_without_changing_route():
    from scripts.saes_target_free_materialization_audit import (
        _clone_gaussians,
        _l0_range_envelope_summary,
    )

    source = _gaussians()
    materialized = _clone_gaussians(source)
    anchors = torch.tensor([0, 3, 12, 15])
    materialized.covariances[0, anchors] *= 10.0
    materialized.opacities[0, anchors] = 0.99
    mask = torch.ones(16, dtype=torch.bool)
    mask[anchors] = False

    report = _l0_range_envelope_summary(
        source,
        materialized,
        mask,
        height=4,
        width=4,
        views=1,
        stats={"level1_tiles": 0, "full_tiles": 0},
    )

    assert report == {
        "applicable": True,
        "tile_count": 1,
        "determinant_above_selected_anchor_max_tiles": 1,
        "determinant_above_selected_anchor_max_rate": 1.0,
        "opacity_outside_selected_anchor_range_tiles": 1,
        "opacity_outside_selected_anchor_range_rate": 1.0,
    }


def test_anchor_conditioned_transport_preserves_one_anchor_camera_residual():
    from saes.progressive_saes import ProgressiveSAES

    saes = ProgressiveSAES(
        4,
        4,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
    )
    source_position = (0, 0)
    source_depth = torch.tensor(2.0)
    source_ray = saes._camera_world_point(0, *source_position, source_depth)
    residual = torch.tensor((0.03, -0.02, 0.01))
    actual = saes._anchor_conditioned_transport_means(
        source_ray + residual,
        source_depth,
        source_position,
        [(2, 1)],
        view_index=0,
    )
    expected = saes._camera_world_point(0, 2, 1, source_depth) + residual

    torch.testing.assert_close(actual[0], expected)


def test_adapter_offset_transport_matches_transplat_adapter_subpixel_ray():
    """The diagnostic must reproduce offset-before-normalization geometry."""
    from saes.progressive_saes import ProgressiveSAES
    from transplat.src.model.encoder.common.gaussian_adapter import (
        GaussianAdapter,
        GaussianAdapterCfg,
    )

    height = width = 4
    angle = torch.tensor(0.31)
    rotation = torch.tensor(
        (
            (torch.cos(angle), -torch.sin(angle), 0.0),
            (torch.sin(angle), torch.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    extrinsic = torch.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = torch.tensor((0.2, -0.3, 0.4))
    intrinsic = torch.tensor(
        ((2.3, 0.1, 0.05), (0.0, 1.7, -0.03), (0.0, 0.0, 1.0))
    )
    adapter = GaussianAdapter(GaussianAdapterCfg(0.01, 0.10, 0))
    depth = torch.tensor((2.4,))
    opacity = torch.tensor((0.30,))
    raw_gaussian = torch.zeros(1, adapter.d_in)
    source_position = (0, 1)
    target_position = (2, 2)
    offset_xy = torch.tensor((0.73, 0.31))
    pixel_size = torch.tensor((1 / width, 1 / height))

    def adapter_coordinate(position):
        row, column = position
        centre = torch.tensor(((column + 0.5) / width, (row + 0.5) / height))
        return centre + (offset_xy - 0.5) * pixel_size

    source = adapter.forward(
        extrinsic.unsqueeze(0),
        intrinsic.unsqueeze(0),
        adapter_coordinate(source_position).unsqueeze(0),
        depth,
        opacity,
        raw_gaussian,
        (height, width),
    )
    expected = adapter.forward(
        extrinsic.unsqueeze(0),
        intrinsic.unsqueeze(0),
        adapter_coordinate(target_position).unsqueeze(0),
        depth,
        opacity,
        raw_gaussian,
        (height, width),
    ).means[0]
    saes = ProgressiveSAES(
        height,
        width,
        context_extrinsics=extrinsic.reshape(1, 1, 4, 4),
        context_intrinsics=intrinsic.reshape(1, 1, 3, 3),
    )

    actual = saes._adapter_offset_transport_means(
        source.means[0],
        depth[0],
        source_position,
        [target_position],
        view_index=0,
    )
    residual_transport = saes._anchor_conditioned_transport_means(
        source.means[0],
        depth[0],
        source_position,
        [target_position],
        view_index=0,
    )

    assert actual is not None
    torch.testing.assert_close(actual[0], expected, rtol=1e-5, atol=1e-6)
    # A constant world-space residual is the old diagnostic, not the adapter.
    assert not torch.allclose(actual, residual_transport)


def test_adapter_offset_transport_is_target_free_psd_and_attribute_preserving():
    from saes.progressive_saes import apply_progressive_saes

    baseline = _adapter_compatible_gaussians()
    poisoned = _adapter_compatible_gaussians()
    probes = torch.tensor((0, 3, 12, 15))
    non_probes = torch.tensor(
        [index for index in range(16) if index not in probes.tolist()]
    )
    original = {
        name: getattr(baseline, name)[0, probes].clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    poisoned.means[0, non_probes] = 1e4
    poisoned.covariances[0, non_probes] = -1e4
    poisoned.harmonics[0, non_probes] = 1e4
    poisoned.opacities[0, non_probes] = 0.99
    options = {
        "feature_var_threshold": 1.0,
        "depth_std_threshold": 1.0,
        "features": torch.ones(1, 1, 2, 4, 4),
        "depths": torch.full((1, 1, 16, 1, 1), 2.0),
        "context_extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
        "context_intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
        "materialization": "conditional-adapter-offset-transport-diagnostic",
        "materialization_guard": False,
    }

    for gaussians in (baseline, poisoned):
        mask, stats, _ = apply_progressive_saes(gaussians, 4, 4, **options)
        assert stats["merge_semantics"] == "conditional-adapter-offset-transport"
        assert stats["adapter_offset_transport_uses"] == 12 * 4
        assert stats["adapter_offset_transport_fallback_tiles"] == 0
        assert stats["level0_tiles"] == 1
        assert stats["covariance_psd_violations"] == 0
        assert mask[non_probes].all()
        assert torch.count_nonzero(gaussians.opacities[0, non_probes]) == 0
        assert torch.all(torch.linalg.eigvalsh(gaussians.covariances[0, probes]) >= -1e-7)
        # Each conditional descriptor originates from the receiving anchor,
        # so no other anchor (or skipped S3 descriptor) can alter SH/opacity.
        torch.testing.assert_close(gaussians.harmonics[0, probes], original["harmonics"])
        torch.testing.assert_close(gaussians.opacities[0, probes], original["opacities"])

    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(
            getattr(baseline, name)[0, probes],
            getattr(poisoned, name)[0, probes],
        )


def test_adapter_offset_transport_preserves_router_and_execution_events():
    from saes.progressive_saes import apply_progressive_saes
    from saes.hardware_accounting import build_saes_event_ledger

    adapter_path = _adapter_compatible_gaussians()
    residual_path = _adapter_compatible_gaussians()
    options = {
        "feature_var_threshold": 1.0,
        "depth_std_threshold": 1.0,
        "features": torch.ones(1, 1, 2, 4, 4),
        "depths": torch.full((1, 1, 16, 1, 1), 2.0),
        "context_extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
        "context_intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
        "materialization_guard": False,
    }
    _, adapter_stats, _ = apply_progressive_saes(
        adapter_path,
        4,
        4,
        materialization="conditional-adapter-offset-transport-diagnostic",
        **options,
    )
    _, residual_stats, _ = apply_progressive_saes(
        residual_path,
        4,
        4,
        materialization="conditional-anchor-transport-diagnostic",
        **options,
    )

    # The adapter repair may only change pseudo-mean geometry.  It must not
    # change the L0/L1/Full router, selected-anchor count, or S2/S3 ledger.
    event_keys = (
        "total_tiles_processed",
        "level0_tiles",
        "level1_tiles",
        "full_tiles",
        "level0_pixels",
        "level1_pixels",
        "l0_representatives",
        "l1_lightweight_anchors",
        "full_stage3_gaussians",
        "zeroed_gaussians",
        "effective_gaussians",
        "full_s2_evaluations",
        "executed_s2_evaluations",
    )
    for key in event_keys:
        assert adapter_stats[key] == residual_stats[key]
    assert adapter_stats["executed_s2_evaluations"] == 4
    assert adapter_stats["full_s2_evaluations"] == 16
    assert build_saes_event_ledger(
        adapter_stats, feature_dim=2, tile_size=4, sh_degree=0
    ) == build_saes_event_ledger(
        residual_stats, feature_dim=2, tile_size=4, sh_degree=0
    )


def test_adapter_offset_transport_fails_closed_without_context_geometry():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _adapter_compatible_gaussians()
    before = {
        name: getattr(gaussians, name).clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    mask, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=torch.ones(1, 1, 2, 4, 4),
        depths=torch.full((1, 1, 16, 1, 1), 2.0),
        materialization="conditional-adapter-offset-transport-diagnostic",
        materialization_guard=False,
    )

    assert stats["adapter_offset_transport_fallback_tiles"] == 1
    assert stats["level0_tiles"] == 0
    assert stats["full_tiles"] == 1
    assert not bool(mask.any())
    for name, value in before.items():
        torch.testing.assert_close(getattr(gaussians, name), value)


def test_transmittance_diagnostic_conserves_constant_l0_optical_depth_without_nonprobe_reads():
    from saes.progressive_saes import apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    probes = torch.tensor([0, 3, 12, 15])
    non_probes = torch.tensor([i for i in range(16) if i not in probes.tolist()])
    baseline.opacities.fill_(0.25)
    perturbed.opacities.fill_(0.25)
    perturbed.means[0, non_probes] = 1e4
    perturbed.covariances[0, non_probes] = -1e4
    perturbed.harmonics[0, non_probes] = 1e4
    perturbed.opacities[0, non_probes] = 0.99
    full_optical_depth = -torch.log1p(-baseline.opacities).sum()
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        mask, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=1.0,
            depth_std_threshold=1.0,
            features=features,
            depths=depths,
            materialization="transmittance-diagnostic",
        )
        assert stats["level0_tiles"] == 1
        assert stats["optical_depth_assignment_error_max"] <= 1e-6
        assert mask[non_probes].all()
        assert torch.count_nonzero(gaussians.opacities[0, non_probes]) == 0

    observed_optical_depth = -torch.log1p(-baseline.opacities).sum()
    torch.testing.assert_close(observed_optical_depth, full_optical_depth)
    assert torch.equal(baseline.means[0, probes], perturbed.means[0, probes])
    assert torch.equal(baseline.covariances[0, probes], perturbed.covariances[0, probes])
    assert torch.equal(baseline.harmonics[0, probes], perturbed.harmonics[0, probes])
    assert torch.equal(baseline.opacities[0, probes], perturbed.opacities[0, probes])


@pytest.mark.parametrize(
    "materialization",
    (
        "virtual-reconstruction-diagnostic",
    ),
)
def test_virtual_reconstruction_l0_is_constant_preserving_and_uses_no_nonprobe_stage3(
    materialization,
):
    from saes.progressive_saes import apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    probes = torch.tensor([0, 3, 12, 15])
    non_probes = torch.tensor([i for i in range(16) if i not in probes.tolist()])
    constant_mean = torch.tensor((1.0, -2.0, 3.0))
    constant_covariance = torch.eye(3) * 0.25
    for gaussians in (baseline, perturbed):
        gaussians.means[:] = constant_mean
        gaussians.covariances[:] = constant_covariance
        gaussians.harmonics.fill_(0.375)
        gaussians.opacities.fill_(0.25)
    expected = {
        name: getattr(baseline, name).clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    perturbed.means[0, non_probes] = 1e4
    perturbed.covariances[0, non_probes] = -1e4
    perturbed.harmonics[0, non_probes] = 1e4
    perturbed.opacities[0, non_probes] = 0.99
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        mask, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=1.0,
            depth_std_threshold=1.0,
            features=features,
            depths=depths,
            materialization=materialization,
            materialization_guard=False,
        )
        assert stats["level0_tiles"] == 1
        assert stats["virtual_reconstructed_gaussians"] == 12
        assert stats["effective_gaussians"] == 16
        assert stats["zeroed_gaussians"] == 0
        assert mask[non_probes].all()

    for name, value in expected.items():
        torch.testing.assert_close(getattr(baseline, name), value)
        torch.testing.assert_close(getattr(baseline, name), getattr(perturbed, name))
    assert torch.all(torch.linalg.eigvalsh(baseline.covariances[0]) >= -1e-7)


@pytest.mark.parametrize(
    "materialization",
    (
        "virtual-reconstruction-diagnostic",
    ),
)
def test_virtual_reconstruction_uses_c2w_ray_for_each_nonprobe_mean(materialization):
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    source_positions = ((0, 0), (0, 3), (3, 0), (3, 3))

    def unit_ray(row: int, column: int) -> torch.Tensor:
        direction = torch.tensor(
            ((column + 0.5) / 4.0, (row + 0.5) / 4.0, 1.0)
        )
        return direction / direction.norm()

    # The selected anchors have zero sub-pixel residual. Every skipped mean is
    # poisoned so the only correct output is the target pixel's C2W ray.
    gaussians.means.fill_(-100.0)
    for row, column in source_positions:
        gaussians.means[0, row * 4 + column] = unit_ray(row, column)
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=features,
        depths=depths,
        materialization=materialization,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
    )

    assert stats["camera_aware_moment_matching"] is True
    torch.testing.assert_close(gaussians.means[0, 1 * 4 + 1], unit_ray(1, 1))


def test_l1_selected_native_anchors_do_not_read_unselected_stage3_attributes():
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    retained = torch.tensor(
        [y * 4 + x for y, x in ProgressiveSAES.compute_lightweight_positions(4)]
    )
    unselected = torch.tensor([index for index in range(16) if index not in retained.tolist()])
    perturbed.means[0, unselected] = 1e4
    perturbed.covariances[0, unselected] = -1e4
    perturbed.harmonics[0, unselected] = 1e4
    perturbed.opacities[0, unselected] = -1e4
    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        _, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=0.2,
            depth_std_threshold=0.1,
            features=features,
            depths=depths,
            materialization_guard=False,
        )
        assert stats["level0_tiles"] == 0
        assert stats["level1_tiles"] == 1
        assert stats["l1_lightweight_anchors"] == 8

    assert torch.equal(baseline.means[0, retained], perturbed.means[0, retained])
    assert torch.equal(
        baseline.covariances[0, retained], perturbed.covariances[0, retained]
    )
    assert torch.equal(
        baseline.harmonics[0, retained], perturbed.harmonics[0, retained]
    )
    assert torch.equal(
        baseline.opacities[0, retained], perturbed.opacities[0, retained]
    )


def test_transmittance_diagnostic_conserves_constant_l1_optical_depth_without_unselected_reads():
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    retained = torch.tensor(
        [y * 4 + x for y, x in ProgressiveSAES.compute_lightweight_positions(4)]
    )
    unselected = torch.tensor([index for index in range(16) if index not in retained.tolist()])
    baseline.opacities.fill_(0.25)
    perturbed.opacities.fill_(0.25)
    perturbed.means[0, unselected] = 1e4
    perturbed.covariances[0, unselected] = -1e4
    perturbed.harmonics[0, unselected] = 1e4
    perturbed.opacities[0, unselected] = 0.99
    full_optical_depth = -torch.log1p(-baseline.opacities).sum()
    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        mask, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=0.2,
            depth_std_threshold=0.1,
            features=features,
            depths=depths,
            materialization="transmittance-diagnostic",
        )
        assert stats["level0_tiles"] == 0
        assert stats["level1_tiles"] == 1
        assert stats["l1_lightweight_anchors"] == 8
        assert stats["optical_depth_assignment_error_max"] <= 1e-6
        assert mask[unselected].all()
        assert torch.count_nonzero(gaussians.opacities[0, unselected]) == 0

    observed_optical_depth = -torch.log1p(-baseline.opacities).sum()
    torch.testing.assert_close(observed_optical_depth, full_optical_depth)
    assert torch.equal(baseline.means[0, retained], perturbed.means[0, retained])
    assert torch.equal(
        baseline.covariances[0, retained], perturbed.covariances[0, retained]
    )
    assert torch.equal(baseline.harmonics[0, retained], perturbed.harmonics[0, retained])
    assert torch.equal(baseline.opacities[0, retained], perturbed.opacities[0, retained])


@pytest.mark.parametrize(
    "materialization",
    (
        "virtual-reconstruction-diagnostic",
    ),
)
def test_virtual_reconstruction_l1_uses_only_native_selected_anchors(materialization):
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    retained = torch.tensor(
        [y * 4 + x for y, x in ProgressiveSAES.compute_lightweight_positions(4)]
    )
    unselected = torch.tensor([index for index in range(16) if index not in retained.tolist()])
    constant_mean = torch.tensor((1.0, -2.0, 3.0))
    constant_covariance = torch.eye(3) * 0.25
    for gaussians in (baseline, perturbed):
        gaussians.means[:] = constant_mean
        gaussians.covariances[:] = constant_covariance
        gaussians.harmonics.fill_(0.375)
        gaussians.opacities.fill_(0.25)
    perturbed.means[0, unselected] = 1e4
    perturbed.covariances[0, unselected] = -1e4
    perturbed.harmonics[0, unselected] = 1e4
    perturbed.opacities[0, unselected] = 0.99
    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        mask, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=0.2,
            depth_std_threshold=0.1,
            features=features,
            depths=depths,
            materialization=materialization,
        )
        assert stats["level0_tiles"] == 0
        assert stats["level1_tiles"] == 1
        assert stats["l1_lightweight_anchors"] == 8
        assert stats["virtual_reconstructed_gaussians"] == 8
        assert stats["effective_gaussians"] == 16
        assert stats["zeroed_gaussians"] == 0
        assert mask[unselected].all()

    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(getattr(baseline, name), getattr(perturbed, name))
    assert torch.all(torch.linalg.eigvalsh(baseline.covariances[0]) >= -1e-7)


@pytest.mark.parametrize(
    "materialization",
    (
        "virtual-reconstruction-diagnostic",
    ),
)
def test_virtual_reconstruction_l1_consumes_the_extra_native_anchor_outputs(
    materialization,
):
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    retained = torch.tensor(
        [y * 4 + x for y, x in ProgressiveSAES.compute_lightweight_positions(4)]
    )
    primary = set(
        y * 4 + x for y, x in ProgressiveSAES.compute_probe_positions(4)
    )
    extra = torch.tensor([index for index in retained.tolist() if index not in primary])
    unselected = torch.tensor([index for index in range(16) if index not in retained.tolist()])
    perturbed.means[0, extra] += 10.0
    perturbed.covariances[0, extra] *= 3.0
    perturbed.harmonics[0, extra] *= -1.0
    perturbed.opacities[0, extra] = 0.9
    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        _, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=0.2,
            depth_std_threshold=0.1,
            features=features,
            depths=depths,
            materialization=materialization,
            materialization_guard=False,
        )
        assert stats["level0_tiles"] == 0
        assert stats["level1_tiles"] == 1
        assert stats["l1_lightweight_anchors"] == 8

    assert not torch.allclose(baseline.means[0, unselected], perturbed.means[0, unselected])
    assert not torch.allclose(
        baseline.covariances[0, unselected], perturbed.covariances[0, unselected]
    )
    assert not torch.allclose(
        baseline.harmonics[0, unselected], perturbed.harmonics[0, unselected]
    )
    assert not torch.allclose(
        baseline.opacities[0, unselected], perturbed.opacities[0, unselected]
    )


def test_l1_selected_lightweight_anchors_are_native_stage3_outputs():
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    retained = torch.tensor(
        [y * 4 + x for y, x in ProgressiveSAES.compute_lightweight_positions(4)]
    )
    primary = set(y * 4 + x for y, x in ProgressiveSAES.compute_probe_positions(4))
    extra = torch.tensor([index for index in retained.tolist() if index not in primary])
    perturbed.means[0, extra] += 10.0
    perturbed.covariances[0, extra] *= 3.0
    perturbed.harmonics[0, extra] *= -1.0
    perturbed.opacities[0, extra] = 0.9
    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        _, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=0.2,
            depth_std_threshold=0.1,
            features=features,
            depths=depths,
            materialization_guard=False,
        )
        assert stats["level0_tiles"] == 0
        assert stats["level1_tiles"] == 1
        assert stats["l1_lightweight_anchors"] == 8

    assert not torch.allclose(baseline.means[0, retained], perturbed.means[0, retained])
    assert not torch.allclose(
        baseline.covariances[0, retained], perturbed.covariances[0, retained]
    )
    assert not torch.allclose(
        baseline.harmonics[0, retained], perturbed.harmonics[0, retained]
    )
    assert not torch.allclose(
        baseline.opacities[0, retained], perturbed.opacities[0, retained]
    )


def test_l1_selected_anchors_do_not_read_unselected_depths_after_probe_decision():
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    retained = torch.tensor(
        [y * 4 + x for y, x in ProgressiveSAES.compute_lightweight_positions(4)]
    )
    unselected = torch.tensor([index for index in range(16) if index not in retained.tolist()])
    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (0.0, 0.0),
        (0, 3): (2.0, 0.0),
        (3, 0): (0.0, 2.0),
        (3, 3): (2.0, 2.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)
    poisoned_depths = depths.clone()
    poisoned_depths[0, 0, unselected, 0, 0] = torch.linspace(
        10.0, 17.0, unselected.numel()
    )

    for gaussians, candidate_depths in (
        (baseline, depths),
        (perturbed, poisoned_depths),
    ):
        _, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=0.2,
            depth_std_threshold=0.1,
            features=features,
            depths=candidate_depths,
            decision_semantics="probe-vector-first-hit",
        )
        assert stats["level0_tiles"] == 0
        assert stats["level1_tiles"] == 1

    torch.testing.assert_close(baseline.means[0, retained], perturbed.means[0, retained])
    torch.testing.assert_close(
        baseline.covariances[0, retained], perturbed.covariances[0, retained]
    )
    torch.testing.assert_close(
        baseline.harmonics[0, retained], perturbed.harmonics[0, retained]
    )
    torch.testing.assert_close(baseline.opacities[0, retained], perturbed.opacities[0, retained])


def test_probe_spread_uses_no_non_probe_stage3_attributes():
    from saes.progressive_saes import apply_progressive_saes

    baseline = _gaussians()
    perturbed = _gaussians()
    probes = torch.tensor([0, 3, 12, 15])
    non_probes = torch.tensor([i for i in range(16) if i not in probes.tolist()])
    original_means = baseline.means[0, probes].clone()
    original_covariances = baseline.covariances[0, probes].clone()
    original_harmonics = baseline.harmonics[0, probes].clone()
    original_opacities = baseline.opacities[0, probes].clone()

    perturbed.means[0, non_probes] = 1e4
    perturbed.covariances[0, non_probes] = -1e4
    perturbed.harmonics[0, non_probes] = 1e4
    perturbed.opacities[0, non_probes] = -1e4
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 16, 1, 1)

    for gaussians in (baseline, perturbed):
        mask, stats, _ = apply_progressive_saes(
            gaussians,
            4,
            4,
            feature_var_threshold=1.0,
            depth_std_threshold=1.0,
            features=features,
            depths=depths,
            cross_check_threshold=2.0,
            materialization="probe-spread-diagnostic",
        )
        assert stats["level0_tiles"] == 1
        assert mask[non_probes].all()
        assert torch.count_nonzero(gaussians.opacities[0, non_probes]) == 0

    assert torch.equal(baseline.means[0, probes], original_means)
    assert torch.equal(baseline.harmonics[0, probes], original_harmonics)
    assert torch.equal(baseline.opacities[0, probes], original_opacities)
    assert not torch.equal(baseline.covariances[0, probes], original_covariances)
    assert torch.equal(baseline.means[0, probes], perturbed.means[0, probes])
    assert torch.equal(
        baseline.covariances[0, probes], perturbed.covariances[0, probes]
    )
    assert torch.equal(baseline.harmonics[0, probes], perturbed.harmonics[0, probes])
    assert torch.equal(baseline.opacities[0, probes], perturbed.opacities[0, probes])


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


def test_paper_assignment_modulates_spatial_term_by_feature_variance():
    from saes.progressive_saes import paper_assignment_weights

    spatial = torch.tensor([0.0, 1.0])
    feature = torch.zeros(2)
    smooth = paper_assignment_weights(
        spatial,
        feature,
        feature_variance=0.0,
        beta_x=0.5,
        beta_f=0.1,
        level="L0",
    )
    textured = paper_assignment_weights(
        spatial,
        feature,
        feature_variance=1.0,
        beta_x=0.5,
        beta_f=0.1,
        level="L0",
    )

    assert smooth.tolist() == pytest.approx([0.5, 0.5])
    assert textured.tolist() == pytest.approx(torch.softmax(torch.tensor([0.0, -4.0]), dim=0).tolist())


def test_paper_l1_reliability_uses_probe_depth_mean_and_std():
    from saes.progressive_saes import paper_assignment_weights

    depths = torch.tensor([0.0, 0.0, 0.0, 1.0])
    weights = paper_assignment_weights(
        torch.zeros(4),
        torch.zeros(4),
        feature_variance=0.0,
        beta_x=0.5,
        beta_f=0.1,
        level="L1",
        probe_depths=depths,
        beta_d=1.0,
    )
    expected = torch.softmax(
        -(depths - depths.mean()).abs() / depths.std(unbiased=False), dim=0
    )

    assert weights.tolist() == pytest.approx(expected.tolist())
    assert weights[-1] < weights[0]


def test_l1_reliability_can_use_primary_probe_depth_reference():
    from saes.progressive_saes import paper_assignment_weights

    selected_anchor_depths = torch.tensor([0.0, 0.0, 0.0, 1.0, 3.0, 4.0])
    primary_probe_depths = selected_anchor_depths[:4]
    weights = paper_assignment_weights(
        torch.zeros(selected_anchor_depths.numel()),
        torch.zeros(selected_anchor_depths.numel()),
        feature_variance=0.0,
        beta_x=0.5,
        beta_f=0.1,
        level="L1",
        probe_depths=selected_anchor_depths,
        depth_reference_depths=primary_probe_depths,
        beta_d=1.0,
    )
    expected = torch.softmax(
        -(
            selected_anchor_depths - primary_probe_depths.mean()
        ).abs()
        / primary_probe_depths.std(unbiased=False),
        dim=0,
    )

    assert weights.tolist() == pytest.approx(expected.tolist())
    assert weights[-1] < weights[-2] < weights[0]


def test_legacy_l1_primary_depth_reference_alias_matches_default():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        materialization="l1-primary-depth-reference-diagnostic",
    )

    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 1
    assert stats["l1_depth_reference"] == "primary-probes"


def test_claim_path_uses_absolute_probe_depth_standard_deviation():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    features = torch.zeros(1, 1, 2, 4, 4)
    for (y, x), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, y, x] = torch.tensor(value)
    depths = torch.full((1, 1, 16, 1, 1), 10.0)
    depths[0, 0, 15, 0, 0] = 10.5

    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
    )

    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 0
    assert stats["full_tiles"] == 1
