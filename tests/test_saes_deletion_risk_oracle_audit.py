"""Unit coverage for the source-only side of the deletion-risk audit."""

import pytest


def test_source_scalar_trace_excludes_dense_oracle_values():
    from scripts.saes_deletion_risk_oracle_audit import _source_scalars_from_trace

    trace = [
        {
            "view_index": 0,
            "tile_row": 0,
            "tile_column": 0,
            "feature_variance": 0.01,
            "feature_candidate": True,
            "depth_candidate": None,
            "routing_level_before_materialization": "L0",
            "guard_checks": [
                {
                    "level": "L0",
                    "passed": True,
                    "anchor_count": 4,
                    "covariance_cosine_minimum": 0.99,
                    "harmonic_cosine_minimum": 0.98,
                    "opacity_distance_maximum": 0.01,
                    "probe_cross_check": {
                        "checked": True,
                        "error_max": 0.005,
                    },
                }
            ],
        },
        {
            "view_index": 0,
            "tile_row": 0,
            "tile_column": 1,
            "feature_variance": 0.3,
            "feature_candidate": False,
            "depth_candidate": False,
            "routing_level_before_materialization": "Full",
            "guard_checks": [],
        },
    ]

    records = _source_scalars_from_trace(trace)

    assert set(records) == {(0, 0, 0), (0, 0, 1)}
    l0 = records[(0, 0, 0)]
    assert l0 == {
        "feature_variance": 0.01,
        "feature_candidate": True,
        "depth_candidate": None,
        "routing_level": "L0",
        "guard_accepted": True,
        "anchor_count": 4,
        "covariance_cosine_minimum": 0.99,
        "harmonic_cosine_minimum": 0.98,
        "opacity_distance_maximum": 0.01,
        "probe_cross_check_checked": True,
        "probe_cross_check_error_max": 0.005,
    }
    full = records[(0, 0, 1)]
    assert full["guard_accepted"] is False
    assert full["covariance_cosine_minimum"] is None
    assert "dense_s3" not in full


def test_source_scalar_trace_rejects_sparse_route_without_accepted_guard():
    from scripts.saes_deletion_risk_oracle_audit import _source_scalars_from_trace

    with pytest.raises(RuntimeError, match="no accepted source guard"):
        _source_scalars_from_trace(
            [
                {
                    "view_index": 0,
                    "tile_row": 0,
                    "tile_column": 0,
                    "feature_variance": 0.01,
                    "feature_candidate": True,
                    "depth_candidate": None,
                    "routing_level_before_materialization": "L0",
                    "guard_checks": [],
                }
            ]
        )
