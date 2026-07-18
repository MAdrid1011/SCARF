from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


H = 4
W = 8
VIEWS = 2
TANGENT_MATERIALIZATION = "multicontext-tangent-plane-diagnostic"
BASE_MATERIALIZATION = "conditional-adapter-offset-attribute-transport-diagnostic"


def _context_geometry():
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, VIEWS, 1, 1)
    extrinsics[0, 1, :3, 3] = torch.tensor((0.1, 0.0, 0.0))
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, VIEWS, 1, 1)
    return extrinsics, intrinsics


def _gaussians():
    from saes.progressive_saes import ProgressiveSAES

    extrinsics, intrinsics = _context_geometry()
    means = []
    for view in range(VIEWS):
        for row in range(H):
            for column in range(W):
                means.append(
                    ProgressiveSAES.camera_world_point(
                        extrinsics,
                        intrinsics,
                        view_index=view,
                        row=row,
                        column=column,
                        height=H,
                        width=W,
                        depth=torch.tensor(2.0),
                        ray_depth_mode="euclidean",
                    )
                )
    count = len(means)
    return SimpleNamespace(
        means=torch.stack(means).unsqueeze(0),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, count, 1, 1) * 0.01,
        harmonics=torch.linspace(0.1, 0.3, count)
        .reshape(1, count, 1, 1)
        .repeat(1, 1, 3, 2),
        opacities=torch.full((1, count), 0.25),
    )


def _clone(gaussians):
    return SimpleNamespace(
        means=gaussians.means.clone(),
        covariances=gaussians.covariances.clone(),
        harmonics=gaussians.harmonics.clone(),
        opacities=gaussians.opacities.clone(),
    )


def _options():
    extrinsics, intrinsics = _context_geometry()
    features = torch.ones(1, VIEWS, 2, H, W)
    depths = torch.full((1, VIEWS, H * W, 1, 1), 2.0)
    # The right tile has a nonuniform primary-probe feature/depth signature,
    # so it remains Full while the left tile is the only L0 candidate.
    for view in range(VIEWS):
        for (row, column), vector, depth in (
            ((0, 4), (1.0, 0.0), 1.0),
            ((0, 7), (0.0, 1.0), 2.0),
            ((3, 4), (-1.0, 0.0), 3.0),
            ((3, 7), (0.0, -1.0), 4.0),
        ):
            features[0, view, :, row, column] = torch.tensor(vector)
            depths[0, view, row * W + column, 0, 0] = depth
    return {
        "feature_var_threshold": 0.1,
        "depth_std_threshold": 0.1,
        "features": features,
        "depths": depths,
        "view_count": VIEWS,
        "context_extrinsics": extrinsics,
        "context_intrinsics": intrinsics,
        "decision_semantics": "probe-normalized-std-first-hit",
        "materialization_guard": False,
    }


def _indices_for_left_tile(*, include_probes):
    probes = {(0, 0), (0, 3), (3, 0), (3, 3)}
    positions = [
        (row, column)
        for row in range(H)
        for column in range(4)
        if ((row, column) in probes) == include_probes
    ]
    return torch.tensor(
        [
            view * H * W + row * W + column
            for view in range(VIEWS)
            for row, column in positions
        ]
    )


def _full_indices():
    return torch.tensor(
        [
            view * H * W + row * W + column
            for view in range(VIEWS)
            for row in range(H)
            for column in range(4, W)
        ]
    )


def test_multicontext_tangent_materialization_preserves_route_full_slots_and_selected_only_access():
    from saes.progressive_saes import apply_progressive_saes

    original = _gaussians()
    base = _clone(original)
    tangent = _clone(original)
    poisoned = _clone(original)
    options = _options()
    skipped = _indices_for_left_tile(include_probes=False)
    full = _full_indices()
    for name, value in (
        ("means", 1.0e4),
        ("covariances", -1.0e4),
        ("harmonics", 1.0e4),
        ("opacities", 0.99),
    ):
        getattr(poisoned, name)[0, skipped] = value
    poisoned_options = {**options, "depths": options["depths"].clone()}
    poisoned_options["depths"][0, :, skipped % (H * W), 0, 0] = 1.0e4

    base_mask, base_stats, _ = apply_progressive_saes(
        base, H, W, materialization=BASE_MATERIALIZATION, **options
    )
    tangent_mask, tangent_stats, _ = apply_progressive_saes(
        tangent, H, W, materialization=TANGENT_MATERIALIZATION, **options
    )
    poisoned_mask, poisoned_stats, _ = apply_progressive_saes(
        poisoned, H, W, materialization=TANGENT_MATERIALIZATION, **poisoned_options
    )

    assert torch.equal(base_mask, tangent_mask)
    assert torch.equal(tangent_mask, poisoned_mask)
    route_keys = (
        "total_tiles_processed",
        "level0_tiles",
        "level1_tiles",
        "full_tiles",
        "l0_representatives",
        "l1_lightweight_anchors",
        "full_stage3_gaussians",
        "zeroed_gaussians",
        "effective_gaussians",
        "full_s2_evaluations",
        "executed_s2_evaluations",
        "adapter_offset_transport_uses",
        "adapter_offset_attribute_transport_uses",
    )
    for key in route_keys:
        assert tangent_stats[key] == base_stats[key]
        assert poisoned_stats[key] == tangent_stats[key]
    assert tangent_stats["multicontext_tangent_enabled"] is True
    assert tangent_stats["multicontext_tangent_attempts"] > 0
    assert tangent_stats["multicontext_tangent_accepted"] > 0
    assert (
        tangent_stats["multicontext_tangent_accepted"]
        + tangent_stats["multicontext_tangent_local_fallbacks"]
        == tangent_stats["multicontext_tangent_attempts"]
    )
    assert tangent_stats["multicontext_tangent_context_camera_reads"] > 0
    assert tangent_stats["multicontext_tangent_constraint_count"] > 0
    assert tangent_stats["multicontext_tangent_runtime_eligible"] is False
    # The residual is a floating-point diagnostic, whereas all route/event
    # counters must remain exactly invariant under skipped S2/S3 poisoning.
    tangent_stable_stats = dict(tangent_stats)
    poisoned_stable_stats = dict(poisoned_stats)
    tangent_residual = tangent_stable_stats.pop("multicontext_tangent_residual_max")
    poisoned_residual = poisoned_stable_stats.pop("multicontext_tangent_residual_max")
    assert poisoned_stable_stats == tangent_stable_stats
    assert poisoned_residual == pytest.approx(tangent_residual, rel=1e-3, abs=1e-8)

    # Only covariance is eligible to differ from the fixed current path.
    for name in ("means", "harmonics", "opacities"):
        torch.testing.assert_close(getattr(tangent, name), getattr(base, name))
    # Removed slots preserve their backing-buffer contents, but never enter
    # the emitted Gaussian set. Poisoning them must not affect valid slots.
    valid = ~tangent_mask
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(
            getattr(poisoned, name)[0, valid],
            getattr(tangent, name)[0, valid],
            rtol=2e-5,
            atol=1e-6,
        )
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(
            getattr(tangent, name)[0, full], getattr(original, name)[0, full]
        )
        torch.testing.assert_close(
            getattr(poisoned, name)[0, full], getattr(original, name)[0, full]
        )
    assert bool((torch.linalg.eigvalsh(tangent.covariances[0, valid]) >= -1e-7).all())
    assert bool((torch.linalg.eigvalsh(poisoned.covariances[0, valid]) >= -1e-7).all())


@pytest.mark.parametrize(
    "materialization", (BASE_MATERIALIZATION, TANGENT_MATERIALIZATION)
)
def test_selected_only_s3_read_guard_accepts_current_and_tangent_paths(materialization):
    from saes.progressive_saes import apply_progressive_saes
    from scripts.saes_multicontext_directional_audit import _wrap_selected_s3_reads

    source = _gaussians()
    expected = _clone(source)
    options = _options()
    expected_mask, _, _ = apply_progressive_saes(
        expected, H, W, materialization=materialization, **options
    )
    selected = torch.nonzero(~expected_mask, as_tuple=False).flatten()
    guarded, fields = _wrap_selected_s3_reads(source, selected)

    actual_mask, _, _ = apply_progressive_saes(
        guarded, H, W, materialization=materialization, **options
    )

    assert torch.equal(actual_mask, expected_mask)
    for field in fields.values():
        assert field.read_indices
        assert set(field.read_indices) <= set(selected.tolist())
