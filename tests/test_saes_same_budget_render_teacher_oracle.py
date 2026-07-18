from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _dense_gaussians(*, height=4, width=8):
    count = height * width
    means = torch.zeros(1, count, 3)
    for index in range(count):
        row, column = divmod(index, width)
        means[0, index] = torch.tensor((0.01 * column, 0.01 * row, 2.0))
    covariance = torch.tensor(
        ((0.02, 0.001, 0.0), (0.001, 0.018, 0.0005), (0.0, 0.0005, 0.016))
    )
    return SimpleNamespace(
        means=means,
        covariances=covariance.reshape(1, 1, 3, 3).repeat(1, count, 1, 1),
        harmonics=torch.linspace(0.0, 0.2, count).reshape(1, count, 1, 1).repeat(1, 1, 3, 4),
        opacities=torch.full((1, count), 0.2),
    )


def _compact(dense, retained):
    return SimpleNamespace(
        means=dense.means[:, retained].clone(),
        covariances=dense.covariances[:, retained].clone(),
        harmonics=dense.harmonics[:, retained].clone(),
        opacities=dense.opacities[:, retained].clone(),
    )


def test_bounded_render_oracle_keeps_full_slots_frozen_and_gradients_finite():
    from scripts.saes_same_budget_render_teacher_oracle import (
        BoundedRepresentativeParameters,
    )

    dense = _dense_gaussians()
    representative_global = torch.tensor((0, 3, 24, 27))
    removed = torch.tensor(
        [
            row * 8 + column
            for row in range(4)
            for column in range(4)
            if row * 8 + column not in representative_global.tolist()
        ]
    )
    retained_mask = torch.ones(32, dtype=torch.bool)
    retained_mask[removed] = False
    retained = torch.nonzero(retained_mask, as_tuple=False).flatten()
    compact = _compact(dense, retained)
    global_to_local = torch.full((32,), -1, dtype=torch.long)
    global_to_local[retained] = torch.arange(retained.numel())
    representative_local = global_to_local[representative_global]
    assert bool((representative_local >= 0).all())

    module = BoundedRepresentativeParameters(
        compact,
        dense,
        representative_global,
        representative_local,
        height=4,
        width=8,
    )
    candidate = module.compose()
    module.assert_immutable_slots(candidate)
    torch.testing.assert_close(
        candidate.means[:, representative_local],
        compact.means[:, representative_local],
        rtol=1e-5,
        atol=1e-5,
    )
    assert bool((torch.linalg.eigvalsh(candidate.covariances[0]) >= -1e-7).all())
    assert bool(((candidate.opacities >= 0.0) & (candidate.opacities <= 1.0)).all())

    loss = (
        candidate.means.square().sum()
        + candidate.covariances.square().sum()
        + candidate.harmonics.square().sum()
        + candidate.opacities.square().sum()
    )
    loss.backward()
    for _name, parameter in module.named_parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.count_nonzero(parameter.grad))
