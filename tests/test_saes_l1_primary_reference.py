"""Target-free L1 depth-reference invariants for the declared 2K path."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _gaussians() -> SimpleNamespace:
    return SimpleNamespace(
        means=torch.zeros(1, 16, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1),
        harmonics=torch.ones(1, 16, 3, 1),
        opacities=torch.full((1, 16), 0.2),
    )


def _l1_features() -> torch.Tensor:
    features = torch.zeros(1, 1, 2, 4, 4)
    for (row, column), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, row, column] = torch.tensor(value)
    return features


def _l1_depths(extra_anchor_depths: tuple[float, float, float, float]) -> torch.Tensor:
    from saes.probe_layout import compute_lightweight_positions

    depths = torch.full((1, 1, 16, 1, 1), 1.0)
    primary_depths = (1.00, 1.01, 0.99, 1.02)
    positions = compute_lightweight_positions(4)
    for (row, column), depth in zip(positions[:4], primary_depths):
        depths[0, 0, row * 4 + column, 0, 0] = depth
    for (row, column), depth in zip(positions[4:], extra_anchor_depths):
        depths[0, 0, row * 4 + column, 0, 0] = depth
    return depths


def _run_l1(depths: torch.Tensor):
    from saes.progressive_saes import apply_progressive_saes

    return apply_progressive_saes(
        _gaussians(),
        4,
        4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        features=_l1_features(),
        depths=depths,
        materialization_guard=False,
    )


@pytest.mark.parametrize(
    "materialization",
    (
        "representative",
        "dense-diagnostic",
        "transmittance-diagnostic",
        "conditional-anchor-transport-diagnostic",
        "conditional-adapter-offset-transport-diagnostic",
    ),
)
def test_all_normal_saes_materializations_use_primary_l1_reference(materialization):
    from saes.progressive_saes import ProgressiveSAES

    saes = ProgressiveSAES(4, 4, materialization=materialization)

    assert saes.l1_depth_reference == "primary-probes"


def test_l1_assignment_uses_primary_depth_reference_with_2k_anchors(monkeypatch):
    import saes.progressive_saes as progressive

    original = progressive.paper_assignment_weights
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(*args, **kwargs):
        if kwargs["level"] == "L1":
            captured.append(
                (
                    kwargs["probe_depths"].detach().clone(),
                    kwargs["depth_reference_depths"].detach().clone(),
                )
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(progressive, "paper_assignment_weights", capture)
    _, stats, _ = _run_l1(_l1_depths((4.0, 5.0, 6.0, 7.0)))

    assert stats["l1_depth_reference"] == "primary-probes"
    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 1
    assert stats["l1_lightweight_anchors"] == 8
    assert captured
    for selected_depths, reference_depths in captured:
        assert selected_depths.numel() == 8
        torch.testing.assert_close(reference_depths, selected_depths[:4])
        torch.testing.assert_close(
            reference_depths, torch.tensor((1.00, 1.01, 0.99, 1.02))
        )


def test_extra_l1_anchor_depths_do_not_change_route_or_anchor_counts():
    _, baseline, _ = _run_l1(_l1_depths((1.0, 1.0, 1.0, 1.0)))
    _, changed, _ = _run_l1(_l1_depths((10.0, 20.0, 30.0, 40.0)))

    for key in (
        "level0_tiles",
        "level1_tiles",
        "full_tiles",
        "l0_representatives",
        "l1_lightweight_anchors",
        "full_stage3_gaussians",
        "executed_s2_evaluations",
    ):
        assert changed[key] == baseline[key]
    assert baseline["l1_depth_reference"] == "primary-probes"
    assert changed["l1_depth_reference"] == "primary-probes"
