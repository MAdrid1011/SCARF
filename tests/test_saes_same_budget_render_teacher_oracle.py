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
    from scripts.saes_target_free_materialization_audit import _clone_gaussians

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

    final_candidate = _clone_gaussians(module.compose())
    module.assert_immutable_slots(final_candidate)
    assert all(
        not getattr(final_candidate, name).requires_grad
        for name in ("means", "covariances", "harmonics", "opacities")
    )


def test_fixed_partition_rejects_routing_or_full_passthrough_drift():
    from scripts.saes_same_budget_render_teacher_oracle import (
        _assert_compact_full_passthrough,
        _assert_fixed_partition,
        _sha256_mask,
    )

    dense = _dense_gaussians()
    canonical = _compact(dense, torch.ones(32, dtype=torch.bool))
    representatives = torch.zeros(32, dtype=torch.bool)
    representatives[torch.tensor((0, 3, 24, 27))] = True
    left_tile = torch.tensor(
        [row * 8 + column for row in range(4) for column in range(4)]
    )
    modified = torch.zeros(32, dtype=torch.bool)
    modified[left_tile[~representatives[left_tile]]] = True
    expected = {
        "full_gaussians": 32,
        "removed_gaussians": 12,
        "representative_gaussians": 4,
        "full_passthrough_gaussians": 16,
        "retained_gaussians": 20,
        "l0_representatives": 4,
        "l1_lightweight_anchors": 0,
        "modified_mask_sha256": _sha256_mask(modified),
        "representative_mask_sha256": _sha256_mask(representatives),
    }
    stats = {
        "l0_representatives": 4,
        "l1_lightweight_anchors": 0,
    }

    full_mask, observed = _assert_fixed_partition(
        dense=dense,
        canonical=canonical,
        modified=modified,
        representative_mask=representatives,
        stats=stats,
        expected=expected,
    )
    assert int(full_mask.sum()) == 16
    assert observed == expected

    drifted_expected = dict(expected)
    drifted_expected["modified_mask_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="pre-registered"):
        _assert_fixed_partition(
            dense=dense,
            canonical=canonical,
            modified=modified,
            representative_mask=representatives,
            stats=stats,
            expected=drifted_expected,
        )

    canonical.means[0, 4, 0] += 1.0
    with pytest.raises(RuntimeError, match="Full passthrough means"):
        _assert_fixed_partition(
            dense=dense,
            canonical=canonical,
            modified=modified,
            representative_mask=representatives,
            stats=stats,
            expected=expected,
        )

    canonical = _compact(dense, torch.ones(32, dtype=torch.bool))
    retained = ~modified
    compact = _compact(canonical, retained)
    global_to_local = torch.full((32,), -1, dtype=torch.long)
    global_to_local[retained] = torch.arange(int(retained.sum()))
    full_local = _assert_compact_full_passthrough(
        compact=compact,
        dense=dense,
        full_global_mask=full_mask,
        global_to_local=global_to_local,
    )
    compact.means[0, full_local[0], 0] += 1.0
    with pytest.raises(RuntimeError, match="compacted oracle changed Full passthrough means"):
        _assert_compact_full_passthrough(
            compact=compact,
            dense=dense,
            full_global_mask=full_mask,
            global_to_local=global_to_local,
        )


def test_oracle_artifact_serializes_the_rendered_representatives(tmp_path):
    from scripts.saes_same_budget_render_teacher_oracle import (
        _save_oracle_artifacts,
        _sha256_tensors,
    )

    values = {
        "means": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "covariances": torch.eye(3).reshape(1, 3, 3).repeat(4, 1, 1),
        "harmonics": torch.arange(48, dtype=torch.float32).reshape(4, 3, 4),
        "opacities": torch.full((4,), 0.2),
    }
    teacher = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4)
    artifacts = _save_oracle_artifacts(
        tmp_path,
        representative_global=torch.tensor((0, 3, 24, 27)),
        final_representative_values=values,
        teacher=teacher,
    )
    saved = torch.load(tmp_path / "optimized_representatives.pt")
    assert torch.equal(saved["global_indices"], torch.tensor((0, 3, 24, 27)))
    for name, value in values.items():
        assert torch.equal(saved[name], value)
    assert artifacts["optimized_representatives"]["values_sha256"] == _sha256_tensors(
        values
    )
