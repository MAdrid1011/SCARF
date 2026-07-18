import copy

import pytest


torch = pytest.importorskip("torch", reason="SAES guard-partition audit requires torch")


def _guard_check(level: str, *, passed: bool, anchor_count: int) -> dict:
    return {
        "level": level,
        "anchor_count": anchor_count,
        "passed": passed,
        "covariance_cosine_minimum": 1.0 if passed else -1.0,
        "harmonic_cosine_minimum": 1.0 if passed else -1.0,
        "opacity_distance_maximum": 0.0 if passed else 1.0,
        "nonprobe_s3_attribute_reads": 0,
    }


def _trace_record(*, route: str, checks: list[dict], feature_candidate: bool = True) -> dict:
    return {
        "view_index": 0,
        "tile_row": 0,
        "tile_column": 0,
        "feature_variance": 0.1,
        "feature_candidate": feature_candidate,
        "depth_candidate": True if checks else False,
        "guard_enabled": True,
        "guard_checks": checks,
        "routing_level_before_materialization": route,
    }


def _guarded_stats(
    *,
    l0_checks: int,
    l1_checks: int,
    l0_rejections: int,
    l1_rejections: int,
    route: str = "Full",
):
    route_counts = {"L0": 0, "L1": 0, "Full": 0}
    route_counts[route] = 1
    return {
        "materialization_guard_enabled": True,
        "total_tiles_processed": 1,
        "level0_tiles": route_counts["L0"],
        "level1_tiles": route_counts["L1"],
        "full_tiles": route_counts["Full"],
        "l0_guard_checks": l0_checks,
        "l1_guard_checks": l1_checks,
        "l0_guard_rejections": l0_rejections,
        "l1_guard_rejections": l1_rejections,
        "l1_guard_attempts_after_l0_rejection": l0_rejections,
        "guard_anchor_attribute_reads": 3 * (l0_checks * 4 + l1_checks * 8),
        "guard_nonprobe_s3_attribute_reads": 0,
    }


def test_scalar_trace_validator_matches_runtime_guard_counters():
    from scripts.saes_l1_primary_reference_guard_partition_audit import _validate_trace

    trace = [
        _trace_record(
            route="Full",
            checks=[
                _guard_check("L0", passed=False, anchor_count=4),
                _guard_check("L1", passed=False, anchor_count=8),
            ],
        )
    ]
    summary = _validate_trace(
        trace,
        _guarded_stats(
            l0_checks=1, l1_checks=1, l0_rejections=1, l1_rejections=1
        ),
        guard_enabled=True,
    )

    assert summary["tile_count"] == 1
    assert summary["pre_materialization_routes"] == {"L0": 0, "L1": 0, "Full": 1}
    assert summary["guard_anchor_descriptors"] == 12
    assert summary["guard_nonprobe_s3_attribute_reads"] == 0


def test_scalar_trace_validator_rejects_non_scalar_attribute_payload():
    from scripts.saes_l1_primary_reference_guard_partition_audit import _validate_trace

    trace = [_trace_record(route="L0", checks=[_guard_check("L0", passed=True, anchor_count=4)])]
    trace[0]["guard_checks"][0]["anchor_indices"] = [0, 3, 12, 15]

    with pytest.raises(RuntimeError, match="unsupported guard field"):
        _validate_trace(
            trace,
            _guarded_stats(
                l0_checks=1,
                l1_checks=0,
                l0_rejections=0,
                l1_rejections=0,
                route="L0",
            ),
            guard_enabled=True,
        )


def test_partition_classifier_separates_fixed_guard_outcomes():
    from scripts.saes_l1_primary_reference_guard_partition_audit import _partition_trace

    accepted = _trace_record(route="L0", checks=[_guard_check("L0", passed=True, anchor_count=4)])
    l1_after_rejection = copy.deepcopy(accepted)
    l1_after_rejection.update(
        {
            "tile_column": 1,
            "routing_level_before_materialization": "L1",
            "guard_checks": [
                _guard_check("L0", passed=False, anchor_count=4),
                _guard_check("L1", passed=True, anchor_count=8),
            ],
        }
    )
    rejected_full = copy.deepcopy(accepted)
    rejected_full.update(
        {
            "tile_column": 2,
            "routing_level_before_materialization": "Full",
            "guard_checks": [
                _guard_check("L0", passed=False, anchor_count=4),
                _guard_check("L1", passed=False, anchor_count=8),
            ],
        }
    )
    noncandidate_full = copy.deepcopy(accepted)
    noncandidate_full.update(
        {
            "tile_column": 3,
            "feature_candidate": False,
            "depth_candidate": False,
            "routing_level_before_materialization": "Full",
            "guard_checks": [],
        }
    )

    labels, counts = _partition_trace(
        [accepted, l1_after_rejection, rejected_full, noncandidate_full]
    )

    assert labels[(0, 0, 0)] == "guard_accepted"
    assert labels[(0, 0, 1)] == "l0_rejected_l1_accepted"
    assert labels[(0, 0, 2)] == "rejected_to_full"
    assert labels[(0, 0, 3)] == "noncandidate_full"
    assert counts == {
        "guard_accepted": 1,
        "l0_rejected_l1_accepted": 1,
        "rejected_to_full": 1,
        "noncandidate_full": 1,
    }


def test_shadow_trace_allows_only_first_hit_depth_short_circuit():
    from scripts.saes_l1_primary_reference_guard_partition_audit import (
        _assert_shadow_candidates_match,
    )

    guarded = [
        _trace_record(
            route="L1",
            checks=[
                _guard_check("L0", passed=False, anchor_count=4),
                _guard_check("L1", passed=True, anchor_count=8),
            ],
        )
    ]
    shadow = [_trace_record(route="L0", checks=[])]
    shadow[0]["guard_enabled"] = False
    shadow[0]["depth_candidate"] = None

    _assert_shadow_candidates_match(guarded, shadow)

    shadow[0]["feature_candidate"] = False
    with pytest.raises(RuntimeError, match="changed an S1 candidate"):
        _assert_shadow_candidates_match(guarded, shadow)


def test_shadow_trace_requires_all_runtime_guard_counters_to_be_zero():
    from scripts.saes_l1_primary_reference_guard_partition_audit import _validate_trace

    shadow = [_trace_record(route="L0", checks=[])]
    shadow[0]["guard_enabled"] = False
    shadow[0]["depth_candidate"] = None
    stats = {
        "materialization_guard_enabled": False,
        "total_tiles_processed": 1,
        "level0_tiles": 1,
        "level1_tiles": 0,
        "full_tiles": 0,
        "l0_guard_checks": 0,
        "l1_guard_checks": 0,
        "l0_guard_rejections": 0,
        "l1_guard_rejections": 0,
        "l1_guard_attempts_after_l0_rejection": 0,
        "guard_anchor_attribute_reads": 0,
        "guard_nonprobe_s3_attribute_reads": 0,
    }
    _validate_trace(shadow, stats, guard_enabled=False)

    stats["guard_anchor_attribute_reads"] = 12
    with pytest.raises(RuntimeError, match="disabled SAES guard emitted guard work"):
        _validate_trace(shadow, stats, guard_enabled=False)


def test_trace_validator_rejects_route_counter_mismatch():
    from scripts.saes_l1_primary_reference_guard_partition_audit import _validate_trace

    trace = [_trace_record(route="L0", checks=[_guard_check("L0", passed=True, anchor_count=4)])]
    stats = _guarded_stats(
        l0_checks=1,
        l1_checks=0,
        l0_rejections=0,
        l1_rejections=0,
        route="Full",
    )

    with pytest.raises(RuntimeError, match="routes disagree"):
        _validate_trace(trace, stats, guard_enabled=True)


def test_shadow_partition_labels_preserve_canonical_guard_outcomes():
    from scripts.saes_l1_primary_reference_guard_partition_audit import (
        _shadow_partition_labels,
    )

    labels = {
        (0, 0, 0): "guard_accepted",
        (0, 0, 1): "rejected_to_full",
        (0, 0, 2): "noncandidate_full",
    }

    assert _shadow_partition_labels(labels) == labels
    assert _shadow_partition_labels(labels) is not labels


def test_programmatic_collector_rejects_nonfixed_input(tmp_path):
    from scripts.saes_l1_primary_reference_guard_partition_audit import (
        collect_guard_partition_audit,
    )

    with pytest.raises(RuntimeError, match="only accepts its fixed target-free input"):
        collect_guard_partition_audit(device=torch.device("cpu"), input_root=tmp_path)


@pytest.mark.parametrize(
    "flag",
    (
        "--input-root",
        "--sample-index",
        "--feature-threshold",
        "--depth-threshold",
        "--materialization",
        "--materialization-guard",
    ),
)
def test_guard_partition_cli_rejects_fixed_contract_overrides(flag):
    from scripts.saes_l1_primary_reference_guard_partition_audit import main

    with pytest.raises(SystemExit) as exc:
        main(["--output-dir", "outputs/fixed", flag, "override"])
    assert exc.value.code == 2
