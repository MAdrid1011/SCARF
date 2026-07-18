from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


MATERIALIZATION = "same-budget-dense-oracle-diagnostic"


def _gaussians():
    """A positive-depth 4x4 dense Stage-3 tile with one primitive per pixel."""
    count = 16
    means = torch.tensor((0.0, 0.0, 2.0)).reshape(1, 1, 3).repeat(1, count, 1)
    covariances = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, count, 1, 1) * 0.01
    harmonics = torch.zeros(1, count, 3, 1)
    for index in range(count):
        harmonics[0, index].fill_(0.1 + 0.01 * index)
    return SimpleNamespace(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=torch.full((1, count, 1), 0.2),
    )


def _features_for_l1():
    features = torch.zeros(1, 1, 2, 4, 4)
    for (row, column), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, row, column] = torch.tensor(value)
    return features


def _run(gaussians, *, level):
    from saes.progressive_saes import apply_progressive_saes

    if level == "L0":
        features = torch.ones(1, 1, 2, 4, 4)
        feature_threshold = 1.0
    elif level == "L1":
        features = _features_for_l1()
        feature_threshold = 0.2
    else:
        raise ValueError(f"unsupported level: {level}")
    return apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=feature_threshold,
        depth_std_threshold=0.1,
        features=features,
        depths=torch.full((1, 1, 16, 1, 1), 2.0),
        materialization=MATERIALIZATION,
        materialization_guard=False,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
    )


@pytest.mark.parametrize(
    ("level", "retained_count"), (("L0", 4), ("L1", 8))
)
def test_same_budget_dense_oracle_keeps_only_l0_or_l1_output_budget(
    level, retained_count
):
    gaussians = _gaussians()
    modified, stats, _ = _run(gaussians, level=level)

    assert stats[f"level{0 if level == 'L0' else 1}_tiles"] == 1
    assert stats["same_budget_dense_oracle_tiles"] == 1
    assert stats["same_budget_dense_oracle_full_stage3_reads"] == 16
    assert stats["same_budget_dense_oracle_output_gaussians"] == retained_count
    assert stats["same_budget_dense_oracle_runtime_eligible"] is False
    assert stats["effective_gaussians"] == retained_count
    assert stats["zeroed_gaussians"] == 16 - retained_count
    assert int(modified.sum().item()) == 16 - retained_count
    assert stats["same_budget_dense_oracle_mass_construction_error_max"] <= 1e-5
    assert (
        stats["same_budget_dense_oracle_projected_moment_construction_error_max"]
        <= 1e-3
    )
    assert torch.count_nonzero(gaussians.opacities[0, modified]) == 0
    assert bool(torch.isfinite(gaussians.covariances).all())
    assert bool((torch.linalg.eigvalsh(gaussians.covariances[0, ~modified]) >= -1e-7).all())


def test_same_budget_dense_oracle_uses_full_nonprobe_stage3_attributes():
    baseline = _gaussians()
    changed = _gaussians()
    selected = torch.tensor((0, 3, 12, 15))
    skipped = torch.tensor([index for index in range(16) if index not in selected.tolist()])

    # Keep values within the selected-anchor SH range, so the range constraint
    # cannot hide whether the dense skipped descriptors were actually used.
    for index, value in zip(selected.tolist(), (0.1, 0.3, 0.5, 0.7)):
        baseline.harmonics[0, index].fill_(value)
        changed.harmonics[0, index].fill_(value)
    baseline.harmonics[0, skipped] = 0.2
    changed.harmonics[0, skipped] = 0.6

    baseline_modified, baseline_stats, _ = _run(baseline, level="L0")
    changed_modified, changed_stats, _ = _run(changed, level="L0")

    assert torch.equal(baseline_modified, changed_modified)
    assert baseline_stats["same_budget_dense_oracle_full_stage3_reads"] == 16
    assert changed_stats["same_budget_dense_oracle_full_stage3_reads"] == 16
    assert not torch.allclose(
        baseline.harmonics[0, ~baseline_modified],
        changed.harmonics[0, ~changed_modified],
    )


def test_same_budget_dense_oracle_fails_closed_without_partial_writes():
    gaussians = _gaussians()
    before = {
        name: getattr(gaussians, name).clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    gaussians.means[0, 1, 0] = float("nan")

    modified, stats, _ = _run(gaussians, level="L0")

    assert stats["same_budget_dense_oracle_tiles"] == 0
    assert stats["same_budget_dense_oracle_fallback_tiles"] == 1
    assert stats["same_budget_dense_oracle_full_stage3_reads"] == 16
    assert stats["full_tiles"] == 1
    assert stats["effective_gaussians"] == 16
    assert not bool(modified.any())
    for name, value in before.items():
        expected = value.clone()
        if name == "means":
            expected[0, 1, 0] = float("nan")
        torch.testing.assert_close(
            getattr(gaussians, name), expected, equal_nan=True
        )


def test_same_budget_dense_oracle_accounting_requires_k_2k_full_partition():
    from scripts.saes_same_budget_dense_oracle import _oracle_output_accounting

    modified = torch.zeros(16, dtype=torch.bool)
    modified[4:] = True
    stats = {
        "l0_representatives": 4,
        "l1_lightweight_anchors": 0,
        "full_stage3_gaussians": 0,
        "effective_gaussians": 4,
        "same_budget_dense_oracle_output_gaussians": 4,
        "same_budget_dense_oracle_runtime_eligible": False,
    }

    accounting = _oracle_output_accounting(
        modified=modified, stats=stats, full_gaussians=16
    )

    assert accounting["retained_gaussians"] == 4
    assert accounting["same_budget_output_count_matches_l0_l1"] is True
    assert accounting["effective_gaussians_less_than_full"] is True


def test_same_budget_dense_oracle_rejects_baseline_identical_all_full_result():
    from scripts.saes_same_budget_dense_oracle import _oracle_success_verdict

    all_full = _oracle_success_verdict(
        {"pass": True}, {"effective_gaussians_less_than_full": False}
    )
    reduced = _oracle_success_verdict(
        {"pass": True}, {"effective_gaussians_less_than_full": True}
    )

    assert all_full["quality_limit_pass"] is True
    assert all_full["same_budget_dense_oracle_meets_quality_limit"] is False
    assert reduced["same_budget_dense_oracle_meets_quality_limit"] is True


def test_projected_moment_helper_matches_small_covariance_finite_difference():
    from saes.progressive_saes import ProgressiveSAES

    angle = torch.tensor(0.23)
    rotation = torch.tensor(
        (
            (torch.cos(angle), -torch.sin(angle), 0.0),
            (torch.sin(angle), torch.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    extrinsic = torch.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = torch.tensor((0.3, -0.2, 0.4))
    intrinsic = torch.tensor(((1.7, 0.1, 0.0), (0.0, 2.3, 0.0), (0.0, 0.0, 1.0)))
    camera_mean = torch.tensor((0.2, -0.1, 2.0))
    camera_covariance = torch.diag(torch.tensor((2e-4, 3e-4, 5e-5)))
    mean = (rotation @ camera_mean + extrinsic[:3, 3]).reshape(1, 3)
    covariance = (rotation @ camera_covariance @ rotation.mT).reshape(1, 3, 3)
    saes = ProgressiveSAES(
        4,
        4,
        context_extrinsics=extrinsic.reshape(1, 1, 4, 4),
        context_intrinsics=intrinsic.reshape(1, 1, 3, 3),
    )

    projected = saes._context_projected_moments(mean, covariance, view_index=0)
    assert projected is not None
    _, expected_covariance, _, _, _ = projected

    torch.manual_seed(7)
    perturbations = torch.randn(80_000, 3) @ torch.linalg.cholesky(camera_covariance).mT
    camera_samples = camera_mean + perturbations
    homogeneous = camera_samples @ intrinsic.mT
    projected_samples = homogeneous[:, :2] / homogeneous[:, 2:]
    observed_covariance = torch.cov(projected_samples.mT, correction=0)

    torch.testing.assert_close(
        observed_covariance, expected_covariance[0], rtol=0.035, atol=2e-6
    )
