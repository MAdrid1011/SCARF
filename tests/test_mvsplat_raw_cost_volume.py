"""Regression coverage for the diagnostic-only MVSplat raw-CV primitive."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


torch = pytest.importorskip("torch")


def _inputs(*, batch: int = 2, height: int = 4, width: int = 4):
    torch.manual_seed(211)
    reference = torch.randn(batch, 3, height, width)
    sources = (torch.randn_like(reference), torch.randn_like(reference))
    intrinsics = torch.eye(3).reshape(1, 3, 3).repeat(batch, 1, 1)
    intrinsics[:, 0, 0] = 2.0
    intrinsics[:, 1, 1] = 2.0
    intrinsics[:, 0, 2] = (width - 1) / 2
    intrinsics[:, 1, 2] = (height - 1) / 2
    pose_one = torch.eye(4).reshape(1, 4, 4).repeat(batch, 1, 1)
    pose_two = pose_one.clone()
    pose_one[:, 0, 3] = 0.15
    pose_two[:, 1, 3] = -0.1
    candidates = torch.tensor((0.4, 0.7, 1.0)).reshape(1, 3, 1, 1).repeat(batch, 1, 1, 1)
    return reference, sources, intrinsics, (pose_one, pose_two), candidates


def test_incremental_raw_cost_volume_matches_dense_and_never_replays_positions():
    from saes.mvsplat_raw_cost_volume import (
        MVSplatSelectedRawCostVolumeProducer,
        dense_mvsplat_raw_cost_volume,
    )

    reference, sources, intrinsics, poses, candidates = _inputs()
    dense = dense_mvsplat_raw_cost_volume(
        reference, sources, intrinsics, poses, candidates
    )
    producer = MVSplatSelectedRawCostVolumeProducer(
        reference, sources, intrinsics, poses, candidates, tile_size=4
    )
    primary = torch.zeros(2, 4, 4, dtype=torch.bool)
    primary[:, 0, 0] = True
    primary[:, 0, 3] = True
    primary[:, 3, 0] = True
    primary[:, 3, 3] = True
    secondary = torch.zeros_like(primary)
    secondary[:, 1, 1] = True
    secondary[:, 2, 2] = True
    full = torch.ones_like(primary)

    primary_event = producer.execute("primary", primary)
    secondary_event = producer.execute("secondary", secondary)
    full_event = producer.execute("full", full)
    replay = producer.finalize(full)
    comparison = producer.verify_against_dense(dense, mask=full)

    assert primary_event["raw_candidate_positions_executed"] == int(primary.sum())
    assert secondary_event["raw_candidate_positions_executed"] == int(secondary.sum())
    assert full_event["raw_candidate_positions_reused"] == int((primary | secondary).sum())
    assert full_event["raw_candidate_positions_executed"] == int((~(primary | secondary)).sum())
    assert replay.events["raw_candidate_positions_executed"] == int(full.sum())
    assert replay.events["whole_pipeline_s2_s3_sparse_execution_verified"] is False
    assert replay.events["s2_s3_savings_claimed"] is False
    assert replay.events["native_s2_depth_output_available"] is False
    assert comparison["equivalent"] is True
    assert comparison["whole_pipeline_s2_s3_sparse_execution_verified"] is False
    torch.testing.assert_close(replay.values, dense, rtol=1.0e-5, atol=1.0e-6)


def test_raw_cost_volume_rejects_out_of_order_or_full_resolution_masks():
    from saes.mvsplat_raw_cost_volume import MVSplatSelectedRawCostVolumeProducer

    reference, sources, intrinsics, poses, candidates = _inputs()
    producer = MVSplatSelectedRawCostVolumeProducer(
        reference, sources, intrinsics, poses, candidates
    )
    native = torch.zeros(2, 4, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="primary -> secondary -> full"):
        producer.execute("secondary", native)
    with pytest.raises(ValueError, match="native grid"):
        producer.execute("primary", torch.zeros(2, 16, 16, dtype=torch.bool))

    with pytest.raises(ValueError, match="native FP32"):
        MVSplatSelectedRawCostVolumeProducer(
            reference.half(),
            tuple(source.half() for source in sources),
            intrinsics.half(),
            tuple(pose.half() for pose in poses),
            candidates.half(),
        )


def test_raw_cost_volume_dense_comparison_detects_changed_selected_values():
    from saes.mvsplat_raw_cost_volume import (
        MVSplatSelectedRawCostVolumeProducer,
        dense_mvsplat_raw_cost_volume,
    )

    reference, sources, intrinsics, poses, candidates = _inputs(batch=1)
    dense = dense_mvsplat_raw_cost_volume(
        reference, sources, intrinsics, poses, candidates
    )
    producer = MVSplatSelectedRawCostVolumeProducer(
        reference, sources, intrinsics, poses, candidates
    )
    one = torch.zeros(1, 4, 4, dtype=torch.bool)
    one[:, 0, 0] = True
    zero = torch.zeros_like(one)
    producer.execute("primary", one)
    producer.execute("secondary", zero)
    producer.execute("full", zero)
    replay = producer.finalize(one)
    corrupted = dense.clone()
    corrupted[0, 0, 0, 0] += 1.0

    comparison = producer.verify_against_dense(corrupted, mask=one)

    assert comparison["equivalent"] is False
    assert comparison["maximum_absolute_error"] == pytest.approx(1.0)
    assert replay.events["unrequested_diagnostic_positions"] == 15
    assert replay.events["not_native_pipeline_work_avoided"] is True
    assert replay.events["s2_s3_savings_claimed"] is False


def test_dense_raw_cost_volume_matches_the_native_mvsplat_warp_formula():
    """Use a fresh interpreter so MVSplat's ``src`` package cannot collide with TranSplat."""

    root = Path(__file__).resolve().parents[1]
    code = """
import math
import torch

from saes.mvsplat_raw_cost_volume import dense_mvsplat_raw_cost_volume
from src.model.encoder.costvolume.depth_predictor_multiview import (
    warp_with_pose_depth_candidates,
)

torch.manual_seed(313)
reference = torch.randn(2, 3, 4, 4)
sources = (torch.randn_like(reference), torch.randn_like(reference))
intrinsics = torch.eye(3).reshape(1, 3, 3).repeat(2, 1, 1)
intrinsics[:, 0, 0] = 2.0
intrinsics[:, 1, 1] = 2.0
intrinsics[:, 0, 2] = 1.5
intrinsics[:, 1, 2] = 1.5
pose_one = torch.eye(4).reshape(1, 4, 4).repeat(2, 1, 1)
pose_two = pose_one.clone()
pose_one[:, 0, 3] = 0.2
pose_two[:, 1, 3] = -0.1
poses = (pose_one, pose_two)
candidates = torch.tensor((0.4, 0.7, 1.0)).reshape(1, 3, 1, 1).repeat(2, 1, 1, 1)

contributions = []
for source, pose in zip(sources, poses):
    warped = warp_with_pose_depth_candidates(
        source,
        intrinsics,
        pose,
        1.0 / candidates.repeat(1, 1, 4, 4),
        warp_padding_mode=\"zeros\",
    )
    contributions.append((reference.unsqueeze(2) * warped).sum(dim=1) / math.sqrt(3))
native = torch.stack(contributions, dim=0).mean(dim=0)
actual = dense_mvsplat_raw_cost_volume(reference, sources, intrinsics, poses, candidates)
torch.testing.assert_close(actual, native, rtol=1.0e-5, atol=1.0e-6)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "mvsplat"), str(root), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_selected_raw_cost_volume_matches_native_mvsplat_preparation_layout():
    """Exercise native camera scaling and ``(v b)`` source ordering directly."""

    root = Path(__file__).resolve().parents[1]
    code = """
import math
import torch
from einops import rearrange

from saes.mvsplat_raw_cost_volume import (
    MVSplatSelectedRawCostVolumeProducer,
    dense_mvsplat_raw_cost_volume,
)
from src.model.encoder.costvolume.depth_predictor_multiview import (
    prepare_feat_proj_data_lists,
    warp_with_pose_depth_candidates,
)

torch.manual_seed(419)
b, v, c, h, w, d = 2, 3, 3, 4, 4, 3
features = torch.randn(b, v, c, h, w)
normalized_intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(b, v, 1, 1)
normalized_intrinsics[:, :, 0, 0] = 0.75
normalized_intrinsics[:, :, 1, 1] = 0.8
normalized_intrinsics[:, :, 0, 2] = 0.5
normalized_intrinsics[:, :, 1, 2] = 0.5
extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(b, v, 1, 1)
for batch_index in range(b):
    for view_index in range(v):
        extrinsics[batch_index, view_index, 0, 3] = 0.2 * (batch_index + view_index)
        extrinsics[batch_index, view_index, 1, 3] = -0.1 * view_index
near = torch.full((b, v), 1.0)
far = torch.full((b, v), 4.0)
feat_lists, feature_pixel_intrinsics, relative_poses, candidates = prepare_feat_proj_data_lists(
    features, normalized_intrinsics, extrinsics, near, far, num_samples=d
)
contributions = []
for source_features, relative_pose in zip(feat_lists[1:], relative_poses):
    warped = warp_with_pose_depth_candidates(
        source_features,
        feature_pixel_intrinsics,
        relative_pose,
        1.0 / candidates.repeat(1, 1, h, w),
        warp_padding_mode=\"zeros\",
    )
    contributions.append((feat_lists[0].unsqueeze(2) * warped).sum(dim=1) / math.sqrt(c))
native = torch.stack(contributions, dim=0).mean(dim=0)
actual = dense_mvsplat_raw_cost_volume(
    feat_lists[0],
    tuple(feat_lists[1:]),
    feature_pixel_intrinsics,
    tuple(relative_poses),
    candidates,
)
torch.testing.assert_close(actual, native, rtol=1.0e-5, atol=1.0e-6)
assert feature_pixel_intrinsics.shape == (v * b, 3, 3)
assert not torch.equal(feature_pixel_intrinsics, rearrange(normalized_intrinsics, \"b v ... -> (v b) ...\"))

producer = MVSplatSelectedRawCostVolumeProducer(
    reference_features=feat_lists[0],
    source_features=tuple(feat_lists[1:]),
    feature_pixel_intrinsics=feature_pixel_intrinsics,
    relative_reference_to_source_poses=tuple(relative_poses),
    inverse_depth_candidates=candidates,
)
primary = torch.zeros(v * b, h, w, dtype=torch.bool)
primary[:, 0, 0] = True
secondary = torch.zeros_like(primary)
secondary[:, h - 1, w - 1] = True
full = torch.ones_like(primary)
producer.execute(\"primary\", primary)
producer.execute(\"secondary\", secondary)
producer.execute(\"full\", full)
producer.finalize(full)
comparison = producer.verify_against_dense(native, mask=full)
assert comparison[\"equivalent\"] is True
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "mvsplat"), str(root), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_raw_cost_volume_diagnostic_cannot_enable_mvsplat_s2_s3_savings():
    from saes.execution_dependency import (
        resolve_s2_s3_execution_contract,
        s2_s3_saving_ratio,
    )

    contract = resolve_s2_s3_execution_contract("mvsplat")

    assert contract["s2_s3_sparse_execution_verified"] is False
    assert "selected raw-correlation diagnostic" in contract["stages"][
        "s2_candidate_search"
    ]["reason"]
    assert s2_s3_saving_ratio(0.75, contract) == 0.0
