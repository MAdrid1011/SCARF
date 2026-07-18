from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


MATERIALIZATION = "assignment-consensus-adapter-pseudo-descriptor-diagnostic"


class _SelectedReadTensor:
    """Permit selected S3 reads and skipped-output writes only."""

    def __init__(self, tensor, selected_indices):
        self._tensor = tensor
        self._selected_indices = set(int(index) for index in selected_indices)
        self.read_indices = []

    def __getattr__(self, name):
        return getattr(self._tensor, name)

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 2 or index[0] != 0:
            raise AssertionError(f"unexpected S3 tensor read index: {index!r}")
        requested = index[1]
        if isinstance(requested, int):
            indices = [requested]
        elif torch.is_tensor(requested):
            indices = [int(value) for value in requested.reshape(-1).tolist()]
        else:
            indices = [int(value) for value in requested]
        if not set(indices) <= self._selected_indices:
            raise AssertionError(f"skipped S3 descriptor read: {indices!r}")
        self.read_indices.extend(indices)
        return self._tensor[index]

    def __setitem__(self, index, value):
        self._tensor[index] = value


def _clone_gaussians(gaussians):
    return SimpleNamespace(
        means=gaussians.means.clone(),
        covariances=gaussians.covariances.clone(),
        harmonics=gaussians.harmonics.clone(),
        opacities=gaussians.opacities.clone(),
    )


def _adapter_compatible_gaussians(*, primitives_per_pixel: int = 1):
    """Build 4x4 descriptors compatible with the selected adapter contract."""
    means = []
    harmonics = []
    opacities = []
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
            means.append((coordinate / coordinate.norm() * 2.0).tolist())
            value = 0.1 + 0.01 * (row + column)
            harmonics.append([[value], [value + 0.1], [value + 0.2]])
            opacities.append(0.2 + 0.01 * (row + column))
    base = SimpleNamespace(
        means=torch.tensor(means).unsqueeze(0),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1)
        * 0.01,
        harmonics=torch.tensor(harmonics).unsqueeze(0),
        opacities=torch.tensor(opacities).unsqueeze(0),
    )
    if primitives_per_pixel == 1:
        return base
    return SimpleNamespace(
        means=base.means.repeat_interleave(primitives_per_pixel, dim=1),
        covariances=base.covariances.repeat_interleave(
            primitives_per_pixel, dim=1
        ),
        harmonics=base.harmonics.repeat_interleave(primitives_per_pixel, dim=1),
        opacities=base.opacities.repeat_interleave(primitives_per_pixel, dim=1),
    )


def _identity_saes():
    from saes.progressive_saes import ProgressiveSAES

    return ProgressiveSAES(
        4,
        4,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
    )


def _selected_geometry(saes, *, depths, offsets, positions):
    means = []
    for depth, offset, position in zip(depths, offsets, positions):
        lifted = saes._lift_adapter_offsets(
            [position],
            depth.reshape(1),
            offset.reshape(1, 2),
            view_index=0,
        )
        assert lifted is not None
        means.append(lifted[0])
    return torch.stack(means)


def _l1_features():
    features = torch.zeros(1, 1, 2, 4, 4)
    for (row, column), value in {
        (0, 0): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (3, 0): (-1.0, 0.0),
        (3, 3): (0.0, -1.0),
    }.items():
        features[0, 0, :, row, column] = torch.tensor(value)
    return features


def _options(*, level):
    if level == "L0":
        return {
            "feature_var_threshold": 1.0,
            "depth_std_threshold": 1.0,
            "features": torch.ones(1, 1, 2, 4, 4),
            "depths": torch.full((1, 1, 16, 1, 1), 2.0),
        }
    if level == "L1":
        return {
            "feature_var_threshold": 0.2,
            "depth_std_threshold": 0.1,
            "features": _l1_features(),
            "depths": torch.full((1, 1, 16, 1, 1), 2.0),
        }
    raise ValueError(f"unsupported test level: {level}")


def test_assignment_consensus_helper_constant_one_hot_and_permutation_invariant():
    saes = _identity_saes()
    positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
    depths = torch.full((4,), 2.0)
    offsets = torch.zeros(4, 2)
    means = _selected_geometry(
        saes, depths=depths, offsets=offsets, positions=positions
    )
    covariances = torch.eye(3).expand(4, -1, -1).clone() * 0.25
    targets = [(1, 1), (2, 2)]
    uniform = torch.full((2, 4), 0.25)

    output = saes._assignment_consensus_adapter_pseudo_geometry(
        means, covariances, depths, positions, uniform, targets, view_index=0
    )
    assert output is not None
    consensus_means, consensus_covariances = output
    expected_means = saes._lift_adapter_offsets(
        targets, torch.full((2,), 2.0), torch.zeros(2, 2), view_index=0
    )
    assert expected_means is not None
    torch.testing.assert_close(consensus_means, expected_means)
    torch.testing.assert_close(
        consensus_covariances,
        torch.eye(3).expand(2, -1, -1) * 0.25,
    )

    one_hot = torch.tensor(((0.0, 1.0, 0.0, 0.0),))
    one_hot_output = saes._assignment_consensus_adapter_pseudo_geometry(
        means, covariances, depths, positions, one_hot, [targets[0]], view_index=0
    )
    assert one_hot_output is not None
    expected_one_hot = saes._lift_adapter_offsets(
        [targets[0]], depths[1].reshape(1), offsets[1].reshape(1, 2), view_index=0
    )
    assert expected_one_hot is not None
    torch.testing.assert_close(one_hot_output[0], expected_one_hot)
    torch.testing.assert_close(one_hot_output[1][0], covariances[1])

    permutation = torch.tensor((2, 0, 3, 1))
    permuted_output = saes._assignment_consensus_adapter_pseudo_geometry(
        means[permutation],
        covariances[permutation],
        depths[permutation],
        [positions[index] for index in permutation.tolist()],
        uniform[:, permutation],
        targets,
        view_index=0,
    )
    assert permuted_output is not None
    torch.testing.assert_close(permuted_output[0], consensus_means)
    torch.testing.assert_close(permuted_output[1], consensus_covariances)


def test_assignment_consensus_helper_applies_assignment_once_and_rejects_bad_simplex():
    saes = _identity_saes()
    positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
    depths = torch.tensor((1.6, 2.0, 2.4, 2.8))
    offsets = torch.tensor(((-0.04, 0.02), (0.03, -0.01), (0.01, 0.04), (-0.02, -0.03)))
    means = _selected_geometry(
        saes, depths=depths, offsets=offsets, positions=positions
    )
    covariances = torch.stack(
        tuple(torch.eye(3) * value for value in (0.05, 0.10, 0.15, 0.20))
    )
    assignment = torch.tensor(((0.10, 0.20, 0.30, 0.40),))
    target = [(1, 2)]

    output = saes._assignment_consensus_adapter_pseudo_geometry(
        means, covariances, depths, positions, assignment, target, view_index=0
    )
    assert output is not None
    expected_depth = assignment @ depths
    expected_offset = assignment @ offsets
    expected_mean = saes._lift_adapter_offsets(
        target, expected_depth, expected_offset, view_index=0
    )
    assert expected_mean is not None
    conditional_means = torch.stack(
        [
            saes._lift_adapter_offsets(
                target,
                depths[index].reshape(1),
                offsets[index].reshape(1, 2),
                view_index=0,
            )[0]
            for index in range(4)
        ],
        dim=0,
    )
    displacements = conditional_means - expected_mean[0]
    expected_covariance = torch.einsum(
        "k,kij->ij",
        assignment[0],
        covariances + torch.einsum("ki,kj->kij", displacements, displacements),
    )
    torch.testing.assert_close(output[0], expected_mean)
    torch.testing.assert_close(output[1][0], expected_covariance)

    squared = assignment.square()
    squared = squared / squared.sum(dim=1, keepdim=True)
    squared_mean = saes._lift_adapter_offsets(
        target, squared @ depths, squared @ offsets, view_index=0
    )
    assert squared_mean is not None
    assert not torch.allclose(output[0], squared_mean)
    assert (
        saes._assignment_consensus_adapter_pseudo_geometry(
            means,
            covariances,
            depths,
            positions,
            torch.tensor(((0.1, 0.2, 0.3, 0.1),)),
            target,
            view_index=0,
        )
        is None
    )
    invalid_covariances = covariances.clone()
    invalid_covariances[0, 0, 0] = -1.0
    assert (
        saes._assignment_consensus_adapter_pseudo_geometry(
            means,
            invalid_covariances,
            depths,
            positions,
            assignment,
            target,
            view_index=0,
        )
        is None
    )
    assert (
        saes._assignment_consensus_adapter_pseudo_geometry(
            means,
            covariances,
            depths,
            positions,
            torch.tensor(((0.2, 0.2, 0.8, -0.2),)),
            target,
            view_index=0,
        )
        is None
    )


def test_assignment_consensus_matches_nonidentity_transplat_adapter_geometry():
    from saes.progressive_saes import ProgressiveSAES
    from transplat.src.model.encoder.common.gaussian_adapter import (
        GaussianAdapter,
        GaussianAdapterCfg,
    )

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
    intrinsic = torch.tensor(((2.3, 0.1, 0.05), (0.0, 1.7, -0.03), (0.0, 0.0, 1.0)))
    saes = ProgressiveSAES(
        4,
        4,
        context_extrinsics=extrinsic.reshape(1, 1, 4, 4),
        context_intrinsics=intrinsic.reshape(1, 1, 3, 3),
    )
    adapter = GaussianAdapter(GaussianAdapterCfg(0.01, 0.10, 0))
    positions = [(0, 1), (3, 2)]
    target = [(2, 2)]
    depths = torch.tensor((2.1, 2.7))
    offsets = torch.tensor(((0.035, -0.020), (-0.030, 0.040)))
    assignment = torch.tensor(((0.25, 0.75),))
    raw_gaussian = torch.zeros(1, adapter.d_in)

    def adapter_mean(position, depth, offset):
        row, column = position
        coordinate = torch.tensor(((column + 0.5) / 4, (row + 0.5) / 4)) + offset
        return adapter.forward(
            extrinsic.unsqueeze(0),
            intrinsic.unsqueeze(0),
            coordinate.unsqueeze(0),
            depth.reshape(1),
            torch.tensor((0.30,)),
            raw_gaussian,
            (4, 4),
        ).means[0]

    means = torch.stack(
        [adapter_mean(position, depth, offset) for position, depth, offset in zip(positions, depths, offsets)]
    )
    covariances = torch.stack((torch.eye(3) * 0.10, torch.eye(3) * 0.20))
    output = saes._assignment_consensus_adapter_pseudo_geometry(
        means, covariances, depths, positions, assignment, target, view_index=0
    )
    assert output is not None

    consensus_depth = assignment @ depths
    consensus_offset = assignment @ offsets
    expected_mean = adapter_mean(target[0], consensus_depth[0], consensus_offset[0])
    conditional_means = torch.stack(
        [adapter_mean(target[0], depths[index], offsets[index]) for index in range(2)]
    )
    displacement = conditional_means - expected_mean
    expected_covariance = torch.einsum(
        "k,kij->ij",
        assignment[0],
        covariances + torch.einsum("ki,kj->kij", displacement, displacement),
    )
    torch.testing.assert_close(output[0][0], expected_mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        output[1][0], expected_covariance, rtol=1e-5, atol=1e-6
    )


def test_assignment_consensus_l0_preserves_constants_and_skipped_poison_invariance():
    from saes.progressive_saes import apply_progressive_saes

    baseline = _adapter_compatible_gaussians()
    poisoned = _adapter_compatible_gaussians()
    selected = torch.tensor((0, 3, 12, 15))
    skipped = torch.tensor([index for index in range(16) if index not in selected.tolist()])
    for gaussians in (baseline, poisoned):
        gaussians.harmonics.fill_(0.375)
        gaussians.opacities.fill_(1.0)
    poisoned.means[0, skipped] = 1.0e4
    poisoned.covariances[0, skipped] = -1.0e4
    poisoned.harmonics[0, skipped] = 1.0e4
    poisoned.opacities[0, skipped] = 0.99
    baseline_selected = {
        name: getattr(baseline, name)[0, selected].clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    options = {
        **_options(level="L0"),
        "context_extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
        "context_intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
        "materialization": MATERIALIZATION,
        "materialization_guard": False,
    }
    poisoned_options = {**options, "depths": options["depths"].clone()}
    poisoned_options["depths"][0, 0, skipped, 0, 0] = 1.0e4

    baseline_mask, baseline_stats, _ = apply_progressive_saes(
        baseline, 4, 4, **options
    )
    poisoned_mask, poisoned_stats, _ = apply_progressive_saes(
        poisoned, 4, 4, **poisoned_options
    )

    assert torch.equal(baseline_mask, poisoned_mask)
    assert baseline_stats == poisoned_stats
    assert baseline_mask[skipped].all()
    assert baseline_stats["level0_tiles"] == 1
    assert baseline_stats["assignment_consensus_pseudo_outputs"] == 12
    assert baseline_stats["assignment_consensus_anchor_pairs"] == 48
    assert baseline_stats["assignment_consensus_offset_recoveries"] == 4
    assert baseline_stats["assignment_consensus_target_lifts"] == 60
    assert baseline_stats["zeroed_gaussians"] == 0
    assert baseline_stats["effective_gaussians"] == 16
    for name, source in baseline_selected.items():
        torch.testing.assert_close(getattr(baseline, name)[0, selected], source)
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(getattr(baseline, name), getattr(poisoned, name))
    torch.testing.assert_close(
        baseline.harmonics, torch.full_like(baseline.harmonics, 0.375)
    )
    torch.testing.assert_close(
        baseline.opacities, torch.ones_like(baseline.opacities)
    )
    assert torch.all(torch.linalg.eigvalsh(baseline.covariances[0]) >= -1e-7)


def test_assignment_consensus_l1_keeps_route_and_charges_virtual_outputs():
    from saes.hardware_accounting import build_saes_event_ledger
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    frozen_path = _adapter_compatible_gaussians()
    baseline = _adapter_compatible_gaussians()
    poisoned = _adapter_compatible_gaussians()
    selected = torch.tensor(
        [row * 4 + column for row, column in ProgressiveSAES.compute_lightweight_positions(4)]
    )
    skipped = torch.tensor([index for index in range(16) if index not in selected.tolist()])
    selected_snapshot = {
        name: getattr(baseline, name)[0, selected].clone()
        for name in ("means", "covariances", "harmonics", "opacities")
    }
    poisoned.means[0, skipped] = 1.0e4
    poisoned.covariances[0, skipped] = -1.0e4
    poisoned.harmonics[0, skipped] = 1.0e4
    poisoned.opacities[0, skipped] = 0.99
    options = {
        **_options(level="L1"),
        "context_extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
        "context_intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
        "materialization_guard": False,
    }
    frozen_mask, frozen_stats, _ = apply_progressive_saes(
        frozen_path,
        4,
        4,
        materialization="conditional-adapter-offset-attribute-transport-diagnostic",
        **options,
    )
    baseline_mask, baseline_stats, _ = apply_progressive_saes(
        baseline, 4, 4, materialization=MATERIALIZATION, **options
    )
    poisoned_options = {**options, "depths": options["depths"].clone()}
    poisoned_options["depths"][0, 0, skipped, 0, 0] = 1.0e4
    poisoned_mask, poisoned_stats, _ = apply_progressive_saes(
        poisoned, 4, 4, materialization=MATERIALIZATION, **poisoned_options
    )

    assert torch.equal(baseline_mask, frozen_mask)
    assert torch.equal(baseline_mask, poisoned_mask)
    for key in (
        "level0_tiles",
        "level1_tiles",
        "full_tiles",
        "l0_representatives",
        "l1_lightweight_anchors",
        "full_stage3_gaussians",
        "full_s2_evaluations",
        "executed_s2_evaluations",
    ):
        assert baseline_stats[key] == frozen_stats[key]
    assert baseline_stats == poisoned_stats
    assert baseline_stats["level1_tiles"] == 1
    assert baseline_stats["assignment_consensus_pseudo_outputs"] == 8
    assert baseline_stats["assignment_consensus_anchor_pairs"] == 64
    assert baseline_stats["assignment_consensus_offset_recoveries"] == 8
    assert baseline_stats["assignment_consensus_target_lifts"] == 72
    for name, source in selected_snapshot.items():
        torch.testing.assert_close(getattr(baseline, name)[0, selected], source)
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(getattr(baseline, name), getattr(poisoned, name))

    ledger = build_saes_event_ledger(
        baseline_stats, feature_dim=2, tile_size=4, sh_degree=0
    )
    assert ledger["events"]["assignment_consensus_pseudo_outputs"] == 8
    assert ledger["events"]["assignment_consensus_anchor_pairs"] == 64
    assert ledger["cycles"]["moment_matching_total"] == 0
    assert ledger["cycles"]["assignment_consensus_total"] > 0
    assert ledger["traffic_bytes"]["assignment_consensus_virtual_output_write"] == (
        8 * ledger["inputs"]["descriptor_bytes"]
    )
    assert ledger["traffic_bytes"]["assignment_consensus_virtual_output_write"] > 0


@pytest.mark.parametrize("level", ("L0", "L1"))
def test_assignment_consensus_uses_each_selected_anchor_depth_and_attributes(level):
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    baseline = _adapter_compatible_gaussians()
    perturbed = _adapter_compatible_gaussians()
    positions = (
        ProgressiveSAES.compute_probe_positions(4)
        if level == "L0"
        else ProgressiveSAES.compute_lightweight_positions(4)
    )
    selected = torch.tensor([row * 4 + column for row, column in positions])
    skipped = torch.tensor([index for index in range(16) if index not in selected.tolist()])
    changed_position = positions[-1]
    changed_index = selected[-1].item()
    geometry = _identity_saes()
    old_depth = torch.tensor(2.0)
    offset = geometry._recover_adapter_offset(
        perturbed.means[0, changed_index], old_depth, changed_position, view_index=0
    )
    assert offset is not None
    new_depth = torch.tensor(2.6)
    lifted = geometry._lift_adapter_offsets(
        [changed_position], new_depth.reshape(1), offset.reshape(1, 2), view_index=0
    )
    assert lifted is not None
    perturbed.means[0, changed_index] = lifted[0]
    perturbed.covariances[0, changed_index] *= 2.0
    perturbed.harmonics[0, changed_index] += 0.5
    perturbed.opacities[0, changed_index] = 0.7
    options = {
        **_options(level=level),
        "context_extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
        "context_intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
        "materialization": MATERIALIZATION,
        "materialization_guard": False,
    }
    perturbed_options = {**options, "depths": options["depths"].clone()}
    perturbed_options["depths"][0, 0, changed_index, 0, 0] = new_depth
    baseline_mask, baseline_stats, _ = apply_progressive_saes(
        baseline, 4, 4, **options
    )
    perturbed_mask, perturbed_stats, _ = apply_progressive_saes(
        perturbed, 4, 4, **perturbed_options
    )

    assert torch.equal(baseline_mask, perturbed_mask)
    for key in (
        "level0_tiles",
        "level1_tiles",
        "full_tiles",
        "l0_representatives",
        "l1_lightweight_anchors",
        "full_stage3_gaussians",
        "full_s2_evaluations",
        "executed_s2_evaluations",
    ):
        assert baseline_stats[key] == perturbed_stats[key]
    assert not torch.allclose(
        baseline.means[0, skipped], perturbed.means[0, skipped]
    )
    assert not torch.allclose(
        baseline.harmonics[0, skipped], perturbed.harmonics[0, skipped]
    )


def test_assignment_consensus_full_tile_fallback_restores_every_primitive_slot():
    from saes.hardware_accounting import build_saes_event_ledger
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    gaussians = _adapter_compatible_gaussians(primitives_per_pixel=2)
    original = _clone_gaussians(gaussians)
    selected_positions = ProgressiveSAES.compute_probe_positions(4)
    bad_slot_indices = torch.tensor(
        [(row * 4 + column) * 2 + 1 for row, column in selected_positions]
    )
    gaussians.means[0, bad_slot_indices] = 0.0
    original = _clone_gaussians(gaussians)
    depths = torch.full((1, 1, 16, 2, 1), 2.0)
    mask, stats, _ = apply_progressive_saes(
        gaussians,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=torch.ones(1, 1, 2, 4, 4),
        depths=depths,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
        materialization=MATERIALIZATION,
        materialization_guard=False,
    )

    assert stats["full_tiles"] == 1
    assert stats["level0_tiles"] == 0
    assert stats["assignment_consensus_fallback_tiles"] == 1
    assert stats["assignment_consensus_l0_fallback_tiles"] == 1
    assert stats["assignment_consensus_l1_fallback_tiles"] == 0
    assert stats["assignment_consensus_pseudo_outputs"] == 0
    assert not mask.any()
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(getattr(gaussians, name), getattr(original, name))

    ledger = build_saes_event_ledger(
        stats,
        feature_dim=2,
        tile_size=4,
        sh_degree=0,
        primitives_per_pixel=2,
    )
    assert ledger["events"]["assignment_consensus_fallback_tiles"] == 1
    assert ledger["events"]["assignment_consensus_l0_fallback_tiles"] == 1
    assert (
        ledger["events"]["assignment_consensus_fallback_attempted_pseudo_outputs"]
        == 24
    )
    assert (
        ledger["events"]["assignment_consensus_fallback_attempted_anchor_pairs"]
        == 96
    )
    assert ledger["cycles"]["assignment_consensus_fallback_assignment"] > 0
    assert ledger["cycles"]["assignment_consensus_total"] > 0
    assert (
        ledger["traffic_bytes"][
            "assignment_consensus_fallback_assignment_feature_read"
        ]
        > 0
    )
    assert (
        ledger["traffic_bytes"]["assignment_consensus_fallback_selected_descriptor_read"]
        > 0
    )


@pytest.mark.parametrize(
    ("primitives_per_pixel", "materialization_guard"),
    ((1, False), (1, True), (2, False), (2, True)),
)
def test_assignment_consensus_never_reads_skipped_s3_descriptors(
    primitives_per_pixel,
    materialization_guard,
):
    from saes.progressive_saes import ProgressiveSAES, apply_progressive_saes

    source = _adapter_compatible_gaussians(
        primitives_per_pixel=primitives_per_pixel
    )
    selected_positions = ProgressiveSAES.compute_probe_positions(4)
    selected = [
        (row * 4 + column) * primitives_per_pixel + slot
        for slot in range(primitives_per_pixel)
        for row, column in selected_positions
    ]
    if primitives_per_pixel == 2:
        bad_slot = [
            (row * 4 + column) * primitives_per_pixel + 1
            for row, column in selected_positions
        ]
        source.means[0, bad_slot] = 0.0
    guarded = SimpleNamespace(
        means=_SelectedReadTensor(source.means, selected),
        covariances=_SelectedReadTensor(source.covariances, selected),
        harmonics=_SelectedReadTensor(source.harmonics, selected),
        opacities=_SelectedReadTensor(source.opacities, selected),
    )
    depths = torch.full((1, 1, 16, primitives_per_pixel, 1), 2.0)
    mask, stats, _ = apply_progressive_saes(
        guarded,
        4,
        4,
        feature_var_threshold=1.0,
        depth_std_threshold=1.0,
        features=torch.ones(1, 1, 2, 4, 4),
        depths=depths,
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
        materialization=MATERIALIZATION,
        materialization_guard=materialization_guard,
    )

    for field in ("means", "covariances", "harmonics", "opacities"):
        observed = getattr(guarded, field).read_indices
        assert observed
        assert set(observed) <= set(selected)
    if materialization_guard:
        assert stats["l0_guard_checks"] == 1
    if primitives_per_pixel == 1:
        assert stats["level0_tiles"] == 1
        assert mask.any()
    else:
        assert stats["full_tiles"] == 1
        assert stats["assignment_consensus_fallback_tiles"] == 1
        assert not mask.any()


@pytest.mark.parametrize("level", ("L0", "L1"))
def test_assignment_consensus_full_probe_layout_is_a_valid_noop(level):
    from saes.progressive_saes import apply_progressive_saes

    source = _adapter_compatible_gaussians()
    gaussians = SimpleNamespace(
        means=source.means[:, :4].clone(),
        covariances=source.covariances[:, :4].clone(),
        harmonics=source.harmonics[:, :4].clone(),
        opacities=source.opacities[:, :4].clone(),
    )
    original = _clone_gaussians(gaussians)
    features = torch.ones(1, 1, 2, 2, 2)
    feature_threshold = 1.0
    if level == "L1":
        # Equality with the strict first-hit threshold rejects L0 but leaves
        # the uniform depth probes eligible for L1.
        feature_threshold = 0.0
    mask, stats, _ = apply_progressive_saes(
        gaussians,
        2,
        2,
        tile_size=2,
        feature_var_threshold=feature_threshold,
        depth_std_threshold=1.0,
        features=features,
        depths=torch.full((1, 1, 4, 1, 1), 2.0),
        context_extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        context_intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
        materialization=MATERIALIZATION,
        materialization_guard=False,
    )

    assert stats["assignment_consensus_pseudo_outputs"] == 0
    assert stats["level0_tiles"] == int(level == "L0")
    assert stats["level1_tiles"] == int(level == "L1")
    assert not mask.any()
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(getattr(gaussians, name), getattr(original, name))
