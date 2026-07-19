from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _gaussians():
    generator = torch.Generator().manual_seed(23)
    count = 16
    factors = torch.randn(count, 3, 3, generator=generator)
    return SimpleNamespace(
        means=torch.randn(1, count, 3, generator=generator),
        covariances=(
            factors @ factors.mT + torch.eye(3).reshape(1, 3, 3) * 0.05
        ).unsqueeze(0),
        harmonics=torch.randn(1, count, 3, 25, generator=generator),
        opacities=torch.full((1, count, 1), 0.2),
    )


def _copy_gaussians(gaussians):
    return SimpleNamespace(
        means=gaussians.means.clone(),
        covariances=gaussians.covariances.clone(),
        harmonics=gaussians.harmonics.clone(),
        opacities=gaussians.opacities.clone(),
    )


def _joint_calibrator(tmp_path, *, name: str = "joint-calibrator.pt"):
    from saes.joint_materialization_calibrator import (
        JointMaterializationCalibrator,
        calibrator_asset_manifest,
        load_calibrator_asset,
    )

    path = tmp_path / name
    source = JointMaterializationCalibrator()
    torch.save({"state_dict": source.state_dict()}, path)
    return load_calibrator_asset(path, calibrator_asset_manifest(path))


def _selected_head_events(*, model="transplat"):
    return {
        "model": model,
        "contract_version": "saes-selected-output-replay-v1",
        "dense_head_macs": 200_000,
        "replayed_head_macs": 0,
    }


def _l0_options(calibrator=None):
    return {
        "tile_size": 4,
        "gpp": 1,
        "features": torch.zeros(1, 1, 2, 4, 4),
        "depths": torch.ones(1, 1, 4, 4),
        "feature_var_threshold": 0.2,
        "depth_std_threshold": 0.1,
        "materialization_guard": False,
        "joint_calibrator": calibrator,
        "joint_calibrator_model": "transplat" if calibrator is not None else None,
        "joint_calibrator_selected_head": (
            _selected_head_events() if calibrator is not None else None
        ),
    }


def test_zero_joint_calibrator_preserves_normal_representative_route_and_output(tmp_path):
    from saes.progressive_saes import apply_progressive_saes
    from saes.hardware_accounting import build_saes_event_ledger

    source = _gaussians()
    baseline = _copy_gaussians(source)
    calibrated = _copy_gaussians(source)
    baseline_mask, baseline_stats, _ = apply_progressive_saes(
        baseline, 4, 4, **_l0_options()
    )
    calibrated_mask, calibrated_stats, _ = apply_progressive_saes(
        calibrated, 4, 4, **_l0_options(_joint_calibrator(tmp_path))
    )

    assert torch.equal(calibrated_mask, baseline_mask)
    assert calibrated_stats["level0_tiles"] == baseline_stats["level0_tiles"] == 1
    assert calibrated_stats["level1_tiles"] == baseline_stats["level1_tiles"] == 0
    assert calibrated_stats["full_tiles"] == baseline_stats["full_tiles"] == 0
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(getattr(calibrated, name), getattr(baseline, name))
    contract = calibrated_stats["joint_calibrator"]
    assert contract["calls"] == contract["l0_calls"] == 4
    assert contract["l1_calls"] == contract["full_calls"] == 0
    assert contract["selected_descriptor_reads"] == 4
    ledger = build_saes_event_ledger(
        calibrated_stats, feature_dim=2, tile_size=4, sh_degree=4
    )
    assert ledger["events"]["joint_calibrator_calls"] == 4
    assert ledger["events"]["joint_calibrator_full_calls"] == 0
    assert ledger["cycles"]["joint_calibrator"] > 0


def test_joint_calibrator_isolated_from_two_finite_skipped_s3_sentinels(tmp_path):
    from saes.progressive_saes import apply_progressive_saes

    source = _gaussians()
    clean = _copy_gaussians(source)
    poisoned = _copy_gaussians(source)
    skipped = torch.tensor([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
    poisoned.means[0, skipped] = 1.0e4
    poisoned.covariances[0, skipped] = -1.0e4
    poisoned.harmonics[0, skipped] = -1.0e4
    poisoned.opacities[0, skipped] = 1.0e4

    calibrator = _joint_calibrator(tmp_path)
    clean_mask, clean_stats, _ = apply_progressive_saes(
        clean, 4, 4, **_l0_options(calibrator)
    )
    poisoned_mask, poisoned_stats, _ = apply_progressive_saes(
        poisoned, 4, 4, **_l0_options(calibrator)
    )

    assert torch.equal(clean_mask, poisoned_mask)
    assert clean_stats["joint_calibrator"] == poisoned_stats["joint_calibrator"]
    selected = ~clean_mask
    for name in ("means", "covariances", "harmonics", "opacities"):
        torch.testing.assert_close(
            getattr(clean, name)[0, selected], getattr(poisoned, name)[0, selected]
        )


def test_joint_calibrator_runs_once_for_each_l1_lightweight_anchor(tmp_path):
    from saes.progressive_saes import apply_progressive_saes

    l1 = _gaussians()
    features = torch.zeros(1, 1, 1, 4, 4)
    features[0, 0, 0, 0, 0] = 0.0
    features[0, 0, 0, 0, 3] = 1.0
    features[0, 0, 0, 3, 0] = 2.0
    features[0, 0, 0, 3, 3] = 3.0
    mask, stats, _ = apply_progressive_saes(
        l1,
        4,
        4,
        tile_size=4,
        gpp=1,
        features=features,
        depths=torch.ones(1, 1, 4, 4),
        feature_var_threshold=0.01,
        depth_std_threshold=0.1,
        decision_semantics="probe-vector-first-hit",
        materialization_guard=False,
        joint_calibrator=_joint_calibrator(tmp_path),
        joint_calibrator_model="transplat",
        joint_calibrator_selected_head=_selected_head_events(),
    )

    assert bool(mask.any())
    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 1
    assert stats["full_tiles"] == 0
    assert stats["joint_calibrator"]["calls"] == 12
    assert stats["joint_calibrator"]["l0_calls"] == 0
    assert stats["joint_calibrator"]["l1_calls"] == 12


def test_joint_calibrator_never_runs_or_modifies_a_full_tile(tmp_path):
    from saes.progressive_saes import apply_progressive_saes

    source = _gaussians()
    full = _copy_gaussians(source)
    features = torch.zeros(1, 1, 1, 4, 4)
    for column, value in zip((0, 3, 0, 3), (0.0, 1.0, 2.0, 3.0)):
        row = 0 if column in (0, 3) and value < 2.0 else 3
        features[0, 0, 0, row, column] = value
    depths = torch.zeros(1, 1, 4, 4)
    depths[0, 0, 0, 0] = 1.0
    depths[0, 0, 0, 3] = 2.0
    depths[0, 0, 3, 0] = 3.0
    depths[0, 0, 3, 3] = 4.0

    mask, stats, _ = apply_progressive_saes(
        full,
        4,
        4,
        tile_size=4,
        gpp=1,
        features=features,
        depths=depths,
        feature_var_threshold=0.01,
        depth_std_threshold=0.01,
        decision_semantics="probe-vector-first-hit",
        materialization_guard=False,
        joint_calibrator=_joint_calibrator(tmp_path),
        joint_calibrator_model="transplat",
        joint_calibrator_selected_head=_selected_head_events(),
    )

    assert not bool(mask.any())
    assert stats["full_tiles"] == 1
    assert stats["joint_calibrator"]["calls"] == 0
    assert stats["joint_calibrator"]["full_calls"] == 0
    for name in ("means", "covariances", "harmonics", "opacities"):
        assert torch.equal(getattr(full, name), getattr(source, name))


def test_offline_capture_records_only_selected_l0_representatives_and_isolates_skipped_s3():
    """The author-side cache hook cannot observe finite skipped descriptors."""
    from saes.progressive_saes import apply_progressive_saes

    clean = _gaussians()
    poisoned = _copy_gaussians(clean)
    skipped = torch.tensor([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
    poisoned.means[0, skipped] = 1.0e4
    poisoned.covariances[0, skipped] = -1.0e4
    poisoned.harmonics[0, skipped] = -1.0e4
    poisoned.opacities[0, skipped] = 1.0e4
    clean_capture = []
    poisoned_capture = []

    clean_mask, clean_stats, _ = apply_progressive_saes(
        clean,
        4,
        4,
        **{
            **_l0_options(),
            "offline_joint_calibration_capture": clean_capture.append,
        },
    )
    poisoned_mask, poisoned_stats, _ = apply_progressive_saes(
        poisoned,
        4,
        4,
        **{
            **_l0_options(),
            "offline_joint_calibration_capture": poisoned_capture.append,
        },
    )

    assert torch.equal(clean_mask, poisoned_mask)
    assert clean_stats["offline_joint_calibration_capture_calls"] == 4
    assert poisoned_stats["offline_joint_calibration_capture_calls"] == 4
    assert len(clean_capture) == len(poisoned_capture) == 4
    for clean_sample, poisoned_sample in zip(clean_capture, poisoned_capture):
        assert clean_sample["level"] == poisoned_sample["level"] == "L0"
        assert set(clean_sample) == {
            "descriptor",
            "means",
            "covariances",
            "harmonics",
            "opacities",
            "level",
            "anchor_index",
            "teacher_nonprobe_indices",
            "teacher_assignment_weights",
        }
        assert clean_sample["anchor_index"] == poisoned_sample["anchor_index"]
        assert (
            clean_sample["teacher_nonprobe_indices"]
            == poisoned_sample["teacher_nonprobe_indices"]
        )
        for name in ("descriptor", "means", "covariances", "harmonics", "opacities"):
            torch.testing.assert_close(clean_sample[name], poisoned_sample[name])
        torch.testing.assert_close(
            clean_sample["teacher_assignment_weights"],
            poisoned_sample["teacher_assignment_weights"],
        )


def test_offline_capture_does_not_run_on_full_tiles_or_combine_with_runtime_calibrator(tmp_path):
    from saes.progressive_saes import apply_progressive_saes

    full = _gaussians()
    captured = []
    _, stats, _ = apply_progressive_saes(
        full,
        4,
        4,
        **{
            **_l0_options(),
            "features": torch.ones(1, 1, 2, 4, 4),
            "depths": torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
            "feature_var_threshold": 0.0,
            "depth_std_threshold": 0.0,
            "offline_joint_calibration_capture": captured.append,
        },
    )
    assert stats["full_tiles"] == 1
    assert stats["offline_joint_calibration_capture_calls"] == 0
    assert captured == []

    with pytest.raises(ValueError, match="cannot run with a calibrator"):
        apply_progressive_saes(
            _gaussians(),
            4,
            4,
            **{
                **_l0_options(_joint_calibrator(tmp_path)),
                "offline_joint_calibration_capture": lambda _sample: None,
            },
        )


def test_joint_calibrator_requires_a_pinned_asset_and_selected_head_contract(tmp_path):
    from saes.progressive_saes import apply_progressive_saes

    unpinned = _gaussians()
    from saes.joint_materialization_calibrator import JointMaterializationCalibrator

    with pytest.raises(ValueError, match="hash-pinned asset"):
        apply_progressive_saes(
            unpinned,
            4,
            4,
            **_l0_options(JointMaterializationCalibrator()),
        )

    depthsplat = _gaussians()
    calibrator = _joint_calibrator(tmp_path)
    _, stats, _ = apply_progressive_saes(
        depthsplat,
        4,
        4,
        **{
            **_l0_options(calibrator),
            "joint_calibrator_model": "depthsplat",
            "joint_calibrator_selected_head": _selected_head_events(model="depthsplat"),
        },
    )
    assert stats["joint_calibrator"]["model"] == "depthsplat"

    unknown = _gaussians()
    with pytest.raises(ValueError, match="model-specific selected-head"):
        apply_progressive_saes(
            unknown,
            4,
            4,
            **{
                **_l0_options(calibrator),
                "joint_calibrator_model": "unknown",
                "joint_calibrator_selected_head": _selected_head_events(model="unknown"),
            },
        )
