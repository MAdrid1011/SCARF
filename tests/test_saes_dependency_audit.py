"""Tests for target-free dense S2/S3 dependency-audit helpers."""

import pytest


torch = pytest.importorskip("torch")


def test_probe_mask_keeps_only_the_four_corners_of_each_tile():
    from scripts.saes_dependency_audit import build_probe_mask

    mask = build_probe_mask(height=8, width=8, tile_size=4)

    assert mask.shape == (8, 8)
    assert int(mask.sum()) == 16
    assert bool(mask[0, 0]) and bool(mask[0, 3])
    assert bool(mask[3, 0]) and bool(mask[3, 3])
    assert bool(mask[4, 4]) and bool(mask[7, 7])
    assert not bool(mask[1, 1]) and not bool(mask[6, 5])


def test_dependency_summary_reports_only_retained_probe_raw_head_changes():
    from scripts.saes_dependency_audit import (
        build_probe_mask,
        summarize_probe_dependency,
    )

    baseline = torch.zeros(2, 3, 4, 4)
    perturbed = baseline.clone()
    perturbed[0, 0, 0, 0] = 2.0
    perturbed[1, 1, 1, 1] = 7.0  # Non-probe location: excluded from the summary.
    mask = build_probe_mask(height=4, width=4, tile_size=4)

    summary = summarize_probe_dependency(baseline, perturbed, mask)

    assert summary["probe_value_count"] == 24
    assert summary["probe_changed_value_count"] == 1
    assert summary["dependency_detected"] is True
    assert summary["maximum_absolute_delta"] == pytest.approx(2.0)
    assert summary["nonprobe_delta_not_reported"] is True


def test_context_transfer_excludes_the_native_target_mapping():
    from scripts.saes_dependency_audit import _context_on_device

    batch = {
        "context": {"image": torch.ones(1), "index": torch.tensor([0])},
        "target": {"image": torch.full((1,), 7.0)},
    }

    context = _context_on_device(batch, torch.device("cpu"))

    assert set(context) == {"image", "index"}
    assert "target" not in context


def test_dependency_audit_removes_target_rgb_before_context_execution():
    from scripts.saes_dependency_audit import remove_target_rgb

    batch = {
        "target": {
            "image": torch.ones(1),
            "extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
        }
    }

    assert remove_target_rgb(batch) is True
    assert "image" not in batch["target"]
    assert "extrinsics" in batch["target"]


def test_dependency_audit_maps_each_model_to_its_native_dense_stage():
    from types import SimpleNamespace

    from scripts.saes_dependency_audit import _raw_head_modules

    classic_predictor = SimpleNamespace(to_gaussians=object(), refine_unet=object())
    classic = SimpleNamespace(
        encoder=SimpleNamespace(depth_predictor=classic_predictor)
    )
    depthsplat = SimpleNamespace(
        encoder=SimpleNamespace(gaussian_head=object(), gaussian_regressor=object())
    )

    assert _raw_head_modules(classic, "transplat")[2] == "DepthPredictorTrans.refine_unet"
    assert _raw_head_modules(classic, "mvsplat")[2] == "DepthPredictorMultiView.refine_unet"
    assert _raw_head_modules(depthsplat, "depthsplat")[2] == (
        "EncoderDepthSplat.gaussian_regressor"
    )
