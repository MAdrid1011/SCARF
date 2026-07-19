"""Tests for frozen V4 selected-anchor attribute replay calibration records."""

from __future__ import annotations

import json

import pytest

from saes.adaptive_l1_v4_attribute_calibration import (
    build_calibration_record,
    load_frozen_threshold,
)


HASH = "a" * 64
V15_HASH = "b" * 64


def _record(index: int, risks: list[float]) -> dict:
    return {
        "sample_index": index,
        "scene": f"scene-{index}",
        "risks": risks,
        "access": {
            "target_rgb_accessed": False,
            "target_camera_accessed": False,
            "skipped_s3_attributes_accessed": False,
        },
    }


def _build() -> dict:
    return build_calibration_record(
        sample_records=[_record(1, [0.1, 0.2, 0.4, 0.5]), _record(2, [0.2, 0.3, 0.6])],
        checkpoint_sha256=HASH,
        source_index_sha256=HASH,
        sample_selection_sha256=HASH,
        v15_calibration_sha256=V15_HASH,
    )


def test_v4_attribute_calibration_freezes_minimum_scene_q25(tmp_path):
    record = _build()
    assert record["threshold"]["rule"] == "minimum-per-scene-q25"
    assert record["threshold"]["value"] == 0.2
    path = tmp_path / "v16.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    loaded = load_frozen_threshold(
        path,
        evaluation_sample_index=0,
        checkpoint_sha256=HASH,
        source_index_sha256=HASH,
        sample_selection_sha256=HASH,
        v15_calibration_sha256=V15_HASH,
    )
    assert loaded["threshold_value"] == 0.2


def test_v4_attribute_calibration_rejects_overlap_and_source_drift(tmp_path):
    path = tmp_path / "v16.json"
    path.write_text(json.dumps(_build()), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        load_frozen_threshold(
            path,
            evaluation_sample_index=1,
            checkpoint_sha256=HASH,
            source_index_sha256=HASH,
            sample_selection_sha256=HASH,
            v15_calibration_sha256=V15_HASH,
        )
    with pytest.raises(ValueError, match="source binding"):
        load_frozen_threshold(
            path,
            evaluation_sample_index=0,
            checkpoint_sha256=HASH,
            source_index_sha256=HASH,
            sample_selection_sha256=HASH,
            v15_calibration_sha256="c" * 64,
        )


def test_v4_attribute_calibration_rejects_target_access():
    invalid = _record(1, [0.1])
    invalid["access"]["target_rgb_accessed"] = True
    with pytest.raises(ValueError, match="target-free"):
        build_calibration_record(
            sample_records=[invalid, _record(2, [0.2])],
            checkpoint_sha256=HASH,
            source_index_sha256=HASH,
            sample_selection_sha256=HASH,
            v15_calibration_sha256=V15_HASH,
        )
