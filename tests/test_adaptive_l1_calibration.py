"""Tests for the target-free adaptive L1 residual calibration record."""

from __future__ import annotations

import json

import pytest

from saes.adaptive_l1_calibration import build_calibration_record, load_frozen_threshold


def _record() -> dict:
    return build_calibration_record(
        sample_records=[
            {
                "sample_index": 1,
                "scene": "scene-one",
                "residuals": [0.1, 0.2, 0.3],
                "access": {
                    "target_rgb_accessed": False,
                    "target_camera_accessed": False,
                    "skipped_s3_attributes_accessed": False,
                },
            },
            {
                "sample_index": 2,
                "scene": "scene-two",
                "residuals": [0.4, 0.5, 0.6],
                "access": {
                    "target_rgb_accessed": False,
                    "target_camera_accessed": False,
                    "skipped_s3_attributes_accessed": False,
                },
            },
        ],
        checkpoint_sha256="a" * 64,
        source_index_sha256="b" * 64,
        sample_selection_sha256="c" * 64,
    )


def test_frozen_threshold_binds_source_and_excludes_evaluation_sample(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")

    loaded = load_frozen_threshold(
        path,
        evaluation_sample_index=0,
        checkpoint_sha256="a" * 64,
        source_index_sha256="b" * 64,
        sample_selection_sha256="c" * 64,
    )

    assert loaded["threshold_value"] == pytest.approx(0.3)
    assert loaded["access"] == {
        "target_rgb_accessed": False,
        "target_camera_accessed": False,
        "skipped_s3_attributes_accessed": False,
    }


def test_frozen_threshold_rejects_hash_drift_and_evaluation_overlap(tmp_path):
    path = tmp_path / "calibration.json"
    record = _record()
    record["threshold"]["value"] = 99.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_frozen_threshold(
            path,
            evaluation_sample_index=0,
            checkpoint_sha256="a" * 64,
            source_index_sha256="b" * 64,
            sample_selection_sha256="c" * 64,
        )

    path.write_text(json.dumps(_record()), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        load_frozen_threshold(
            path,
            evaluation_sample_index=1,
            checkpoint_sha256="a" * 64,
            source_index_sha256="b" * 64,
            sample_selection_sha256="c" * 64,
        )
