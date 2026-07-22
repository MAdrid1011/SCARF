"""Progressive SAES safety invariants for the claim-path implementation."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _gaussians() -> SimpleNamespace:
    means = torch.zeros(1, 16, 3)
    means[..., 2] = 2.0
    return SimpleNamespace(
        means=means,
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1)
        * 0.01,
        harmonics=torch.ones(1, 16, 3, 1),
        opacities=torch.full((1, 16), 0.2),
    )


def _copy(gaussians: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        means=gaussians.means.clone(),
        covariances=gaussians.covariances.clone(),
        harmonics=gaussians.harmonics.clone(),
        opacities=gaussians.opacities.clone(),
    )


def _uniform_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, 1, 2, 4, 4), torch.ones(1, 1, 16, 1, 1)


def test_context_safety_guard_fails_closed_without_camera_and_preserves_full_tile():
    from saes.progressive_saes import apply_progressive_saes

    source = _gaussians()
    candidate = _copy(source)
    features, depths = _uniform_inputs()

    modified, stats, _ = apply_progressive_saes(
        candidate,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        context_safety_guard=True,
    )

    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 0
    assert stats["full_tiles"] == 1
    assert stats["context_guard_checks"] == 1
    assert stats["context_guard_rejections"] == 1
    assert stats["context_guard_missing_geometry"] == 1
    assert not bool(modified.any())
    for name in ("means", "covariances", "harmonics", "opacities"):
        assert torch.equal(getattr(candidate, name), getattr(source, name))


def test_context_safety_guard_accepts_finite_uniform_probe_geometry():
    from saes.progressive_saes import ProgressiveSAES

    gaussians = _gaussians()
    router = ProgressiveSAES(
        4,
        4,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
        context_safety_guard=True,
    )
    result = router.context_coverage_occlusion_safety(
        gaussians,
        [0, 3, 12, 15],
        torch.ones(4),
        view_index=0,
        level="L0",
    )

    assert result["passed"] is True
    assert result["coverage_passed"] is True
    assert result["occlusion_passed"] is True
    assert result["nonprobe_s3_attribute_reads"] == 0


def test_context_safety_guard_rejects_projected_anchor_center_separation():
    from saes.progressive_saes import ProgressiveSAES

    gaussians = _gaussians()
    # The selected anchors are individually valid, but their producer-camera
    # centers are much farther apart than their combined projected support.
    # The guard may inspect only these anchors and must retain the dense tile.
    gaussians.means[0, [0, 3, 12, 15], 0] = torch.tensor((0.0, 4.0, 0.0, 4.0))
    router = ProgressiveSAES(
        4,
        4,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
        context_safety_guard=True,
    )

    result = router.context_coverage_occlusion_safety(
        gaussians,
        [0, 3, 12, 15],
        torch.ones(4),
        view_index=0,
        level="L0",
    )

    assert result["center_overlap_passed"] is False
    assert result["projected_center_mahalanobis_max"] > 2.0
    assert result["reason"] == "center_separation"
    assert result["passed"] is False
    assert result["nonprobe_s3_attribute_reads"] == 0


def test_runtime_stats_preserve_the_cross_check_threshold():
    from saes.progressive_saes import apply_progressive_saes

    gaussians = _gaussians()
    features, depths = _uniform_inputs()
    _, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        cross_check_threshold=0.015,
        features=features,
        depths=depths,
    )

    assert stats["cross_check_threshold"] == 0.015


def test_uncertified_representative_deletion_falls_back_to_full():
    from saes.progressive_saes import apply_progressive_saes

    source = _gaussians()
    candidate = _copy(source)
    features, depths = _uniform_inputs()

    modified, stats, _ = apply_progressive_saes(
        candidate,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        require_deletion_certificate=True,
    )

    assert stats["deletion_certificate_required"] is True
    assert stats["uncertified_deletion_fallback_tiles"] == 1
    assert stats["deletion_certificate_source_status"] == "untrusted_source_kind"
    assert stats["deletion_certificate_rejection_reasons"] == {
        "untrusted_source_kind": 1
    }
    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 0
    assert stats["full_tiles"] == 1
    assert not bool(modified.any())
    for name in ("means", "covariances", "harmonics", "opacities"):
        assert torch.equal(getattr(candidate, name), getattr(source, name))


def test_exact_zero_source_opacity_certificate_deletes_without_merging():
    from saes.progressive_saes import (
        DELETION_CERTIFICATE_SOURCE_KIND,
        apply_progressive_saes,
    )

    source = _gaussians()
    anchors = [0, 3, 12, 15]
    nonprobes = [index for index in range(16) if index not in anchors]
    source.opacities.zero_()
    source.opacities[0, anchors] = 0.2
    candidate = _copy(source)
    # A valid certificate must not inspect or repair withheld Stage-3 fields.
    candidate.means[0, nonprobes] = torch.nan
    candidate.covariances[0, nonprobes] = torch.nan
    candidate.harmonics[0, nonprobes] = torch.nan
    anchor_snapshot = {
        name: getattr(candidate, name)[0, anchors].clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    source_opacities = source.opacities.reshape(1, 1, 16, 1, 1).clone()
    features, depths = _uniform_inputs()

    modified, stats, _ = apply_progressive_saes(
        candidate,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        require_deletion_certificate=True,
        source_opacities=source_opacities,
        source_opacity_certificate_kind=DELETION_CERTIFICATE_SOURCE_KIND,
    )

    assert stats["deletion_certificate_source_status"] == "ready"
    assert stats["deletion_certificate_accepted_tiles"] == 1
    assert stats["deletion_certificate_rejected_tiles"] == 0
    assert stats["deletion_certificate_zero_opacity_gaussians"] == 12
    assert stats["deletion_certificate_nonprobe_s3_attribute_reads"] == 0
    assert stats["uncertified_deletion_fallback_tiles"] == 0
    assert stats["level0_tiles"] == 1
    assert stats["full_tiles"] == 0
    assert stats["zeroed_gaussians"] == 12
    assert torch.equal(modified, torch.tensor([index in nonprobes for index in range(16)]))
    for name, before in anchor_snapshot.items():
        assert torch.equal(getattr(candidate, name)[0, anchors], before)
    assert bool(torch.isnan(candidate.means[0, nonprobes]).all())
    assert bool(torch.isnan(candidate.covariances[0, nonprobes]).all())
    assert bool(torch.isnan(candidate.harmonics[0, nonprobes]).all())


@pytest.mark.parametrize(
    ("source_opacities", "source_kind", "expected_reason"),
    (
        (None, "s2-density-adapter-opacity-v1", "missing_source_opacities"),
        (torch.zeros(1, 16), "s2-density-adapter-opacity-v1", "invalid_source_layout"),
        (
            torch.full((1, 1, 16, 1, 1), float("nan")),
            "s2-density-adapter-opacity-v1",
            "invalid_source_opacity_values",
        ),
    ),
)
def test_exact_zero_source_opacity_certificate_fails_closed_on_invalid_source(
    source_opacities, source_kind, expected_reason
):
    from saes.progressive_saes import apply_progressive_saes

    source = _gaussians()
    candidate = _copy(source)
    features, depths = _uniform_inputs()

    modified, stats, _ = apply_progressive_saes(
        candidate,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        require_deletion_certificate=True,
        source_opacities=source_opacities,
        source_opacity_certificate_kind=source_kind,
    )

    assert stats["deletion_certificate_rejected_tiles"] == 1
    assert stats["deletion_certificate_rejection_reasons"] == {expected_reason: 1}
    assert stats["full_tiles"] == 1
    assert stats["zeroed_gaussians"] == 0
    assert not bool(modified.any())
    for name in ("means", "covariances", "harmonics", "opacities"):
        assert torch.equal(getattr(candidate, name), getattr(source, name))


def test_exact_zero_source_opacity_certificate_rejects_anchor_mapping_mismatch():
    from saes.progressive_saes import (
        DELETION_CERTIFICATE_SOURCE_KIND,
        apply_progressive_saes,
    )

    source = _gaussians()
    candidate = _copy(source)
    source_opacities = torch.zeros(1, 1, 16, 1, 1)
    source_opacities[0, 0, [0, 3, 12, 15], 0, 0] = 0.1
    features, depths = _uniform_inputs()

    modified, stats, _ = apply_progressive_saes(
        candidate,
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=features,
        depths=depths,
        require_deletion_certificate=True,
        source_opacities=source_opacities,
        source_opacity_certificate_kind=DELETION_CERTIFICATE_SOURCE_KIND,
    )

    assert stats["deletion_certificate_rejection_reasons"] == {
        "source_anchor_mapping_mismatch": 1
    }
    assert stats["full_tiles"] == 1
    assert stats["zeroed_gaussians"] == 0
    assert not bool(modified.any())
