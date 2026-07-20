"""CPU contracts for the coverage-enriched DepthSplat context-only audit."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _sha(value: str) -> str:
    return value * 64


def _identity(*, sample_index: int = 0):
    return {
        "scene": "scene-fixed",
        "context_indices": [0, 9],
        "tree_sha256": _sha("a"),
        "manifest_sha256": _sha("b"),
        "audit_input_sha256": _sha("c"),
        "source_sample_index": sample_index,
        "source_binding": {
            "canonical_index_sha256": _sha("d"),
            "canonical_sample_selection_sha256": _sha("e"),
            "canonical_selection_sha256": _sha("f"),
            "source_audit_input_sha256": _sha("1"),
            "source_audit_tree_sha256": _sha("2"),
            "source_sidecar_tree_sha256": _sha("3"),
        },
        "sidecar": {"index_sha256": _sha("4"), "record_sha256": _sha("5")},
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_mapping_present": False,
        "target_index_accessed": False,
    }


def _context():
    return {
        "image": torch.zeros(1, 2, 3, 4, 8),
        "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
        "near": torch.full((1, 2), 0.1),
        "far": torch.full((1, 2), 100.0),
        "index": torch.tensor([[0, 9]], dtype=torch.long),
    }


def _batch(identity):
    return {
        "context": _context(),
        "scene": [identity["scene"]],
        "calibration": {
            **identity,
            "target_mapping_present": False,
            "native_preprocessing": {
                "crop_image_shape": [4, 8],
                "patch_size": 1,
                "prepared_image_shape": [4, 8],
            },
        },
    }


def test_coverage_enriched_profile_is_fixed_scale_and_precalibration_only():
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE,
    )
    from saes.probe_first_schedule import (
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_PLAN_CONTRACT,
        coverage_enriched_t4_route_config_sha256,
    )
    from scripts.saes_depthsplat_coverage_enriched_t4_context_only_audit import (
        coverage_enriched_t4_profile,
    )

    profile = coverage_enriched_t4_profile()

    assert profile["materialization_profile"] == (
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE
    )
    assert profile["route_plan_contract"] == DEPTHSPLAT_COVERAGE_ENRICHED_T4_PLAN_CONTRACT
    assert profile["maximum_coverage_covariance_scale"] == 1.0
    assert profile["owner_support_guard"] is True
    assert profile["attribute_loo_guard"] is None
    assert profile["precalibration_only"] is True
    assert profile["coverage_certificate"] == (
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MOMENT_CERTIFICATE
    )
    assert profile["route_plan_config_sha256"] == (
        coverage_enriched_t4_route_config_sha256(profile)
    )


def test_support_basis_profile_is_separate_and_fixed_scale():
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE,
    )
    from saes.probe_first_schedule import (
        DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT,
        support_basis_t4_route_config_sha256,
    )
    from scripts.saes_depthsplat_coverage_enriched_t4_context_only_audit import (
        SUPPORT_BASIS_MECHANISM,
        _profile_for_mechanism,
        support_basis_t4_profile,
    )

    profile = support_basis_t4_profile()

    assert _profile_for_mechanism(SUPPORT_BASIS_MECHANISM) == profile
    assert profile["materialization_profile"] == (
        DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE
    )
    assert profile["route_plan_contract"] == DEPTHSPLAT_SUPPORT_BASIS_T4_PLAN_CONTRACT
    assert profile["maximum_coverage_covariance_scale"] == 1.0
    assert profile["support_basis_guard"] is True
    assert profile["attribute_loo_guard"] is None
    assert profile["coverage_certificate"] == DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE
    assert profile["route_plan_config_sha256"] == (
        support_basis_t4_route_config_sha256(profile)
    )


def test_coverage_enriched_context_identity_rejects_nonzero_or_target_access():
    from scripts.saes_depthsplat_coverage_enriched_t4_context_only_audit import (
        _require_formal_context_identity,
    )

    assert _require_formal_context_identity(_identity())["source_sample_index"] == 0
    with pytest.raises(ValueError, match="sample zero"):
        _require_formal_context_identity(_identity(sample_index=1))

    target_bearing = _identity()
    target_bearing["target_rgb_accessed"] = True
    with pytest.raises(ValueError, match="context-only"):
        _require_formal_context_identity(target_bearing)

    target_index_bearing = _identity()
    target_index_bearing["target_index_accessed"] = True
    with pytest.raises(ValueError, match="context-only"):
        _require_formal_context_identity(target_index_bearing)


def test_coverage_enriched_loaded_batch_rejects_target_mapping_and_drift():
    from scripts.saes_depthsplat_coverage_enriched_t4_context_only_audit import (
        _require_loaded_context_batch,
    )

    identity = _identity()
    context, calibration = _require_loaded_context_batch(_batch(identity), identity)
    assert set(context) == {"image", "extrinsics", "intrinsics", "near", "far", "index"}
    assert calibration["target_mapping_present"] is False

    target_bearing = _batch(identity)
    target_bearing["target"] = {"image": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="target mapping"):
        _require_loaded_context_batch(target_bearing, identity)

    drifted = _batch(identity)
    drifted["calibration"] = deepcopy(drifted["calibration"])
    drifted["calibration"]["source_sample_index"] = 1
    with pytest.raises(RuntimeError, match="changed input identity"):
        _require_loaded_context_batch(drifted, identity)


def test_coverage_enriched_parser_defaults_to_v2_and_has_no_literal_override():
    from scripts.saes_depthsplat_coverage_enriched_t4_context_only_audit import (
        DEFAULT_INPUT_ROOT,
        build_parser,
    )

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--output-dir", "result"])

    assert args.input_root == DEFAULT_INPUT_ROOT
    assert args.input_root.name == "depthsplat_sample0_l0_l1_context_only_v2"
    assert not hasattr(args, "sample_index")
    assert not hasattr(args, "model")
    assert not hasattr(args, "literal_t4_v16_record")
    assert not hasattr(args, "literal_t4_acid_plan_path")
    assert args.mechanism == "owner-support-v1"


def test_coverage_enriched_plan_is_profile_bound_and_target_free():
    from saes.probe_first_schedule import (
        build_depthsplat_coverage_enriched_t4_probe_first_plan,
    )
    from scripts.saes_depthsplat_coverage_enriched_t4_context_only_audit import (
        _require_coverage_enriched_plan,
    )

    plan = build_depthsplat_coverage_enriched_t4_probe_first_plan(
        torch.zeros(1, 1, 2, 4, 4),
        torch.ones(1, 1, 4, 4),
        height=4,
        width=4,
        feature_threshold=0.20,
        depth_threshold=0.10,
    )

    profile, trace_sha256 = _require_coverage_enriched_plan(plan)
    assert profile["l0_anchor_count"] == 4
    assert profile["l1_anchor_count"] == 12
    assert trace_sha256 == plan.events["tile_trace_sha256"]
    assert plan.events["target_rgb_accessed"] is False
    assert plan.events["gaussian_attributes_accessed"] is False

    drifted = SimpleNamespace(
        events={**plan.events, "l1_anchor_count": 4}, tile_trace=plan.tile_trace
    )
    with pytest.raises(RuntimeError, match="plan changed"):
        _require_coverage_enriched_plan(drifted)

    plan.tile_trace[0]["pre_guard_route"] = "Full"
    with pytest.raises(RuntimeError, match="plan trace hash changed"):
        _require_coverage_enriched_plan(plan)


def test_support_basis_plan_is_profile_bound_and_target_free():
    from saes.probe_first_schedule import build_depthsplat_support_basis_t4_probe_first_plan
    from scripts.saes_depthsplat_coverage_enriched_t4_context_only_audit import (
        _require_coverage_enriched_plan,
        support_basis_t4_profile,
    )

    plan = build_depthsplat_support_basis_t4_probe_first_plan(
        torch.zeros(1, 1, 2, 4, 4),
        torch.ones(1, 1, 4, 4),
        height=4,
        width=4,
        feature_threshold=0.20,
        depth_threshold=0.10,
    )
    profile, trace_sha256 = _require_coverage_enriched_plan(
        plan, profile=support_basis_t4_profile()
    )

    assert profile["l0_anchor_count"] == 4
    assert profile["l1_anchor_count"] == 12
    assert trace_sha256 == plan.events["tile_trace_sha256"]
    assert plan.events["target_rgb_accessed"] is False
    assert plan.events["gaussian_attributes_accessed"] is False


def test_preflight_contract_requires_no_literal_guard_and_route_summary_is_explicit():
    from scripts.saes_depthsplat_coverage_enriched_t4_context_only_audit import (
        _preflight_route_counts,
        _require_coverage_enriched_final_route,
        _require_coverage_enriched_preflight,
        coverage_enriched_t4_profile,
    )
    from saes.depthsplat_backend import canonical_json_sha256

    profile = coverage_enriched_t4_profile()
    preflight_trace = (
        {"attempted": True, "accepted": True, "accepted_level": "L1"},
        {
            "attempted": True,
            "accepted": False,
            "accepted_level": None,
            "reason": "owner support rejected",
        },
        {"attempted": False, "accepted": True, "accepted_level": "Full"},
    )
    preflight_trace_sha256 = canonical_json_sha256(preflight_trace)
    certificate_payload = {
        "schema": profile["coverage_certificate"],
        "fixed_moment_covariance_scale": 1.0,
        "support_containment_guard": True,
        "tile_trace_sha256": preflight_trace_sha256,
        "per_update": [],
    }
    events = {
        "execution_profile": profile["materialization_profile"],
        "maximum_coverage_covariance_scale": 1.0,
        "coverage_certificate": profile["coverage_certificate"],
        "coverage_enriched_owner_support_guard": True,
        "formal_paper_selected_probe_only": False,
        "selected_anchor_attribute_loo_collect_only": False,
        "selected_anchor_attribute_loo_guard": False,
        "selected_anchor_attribute_loo_frozen_guard": None,
        "selected_anchor_attribute_loo_aggregate": None,
        "target_rgb_accessed": False,
        "target_camera_accessed_before_commit": False,
        "skipped_s3_attributes_accessed": False,
        "coverage_certificate_payload": certificate_payload,
        "coverage_certificate_sha256": canonical_json_sha256(certificate_payload),
        "materialization_session_sha256": _sha("b"),
        "tile_trace_sha256": preflight_trace_sha256,
        "coverage_enriched_l0_to_l1_tile_count": 1,
        "promoted_full_tiles": 1,
    }
    preflight = SimpleNamespace(
        events=events,
        tile_trace=preflight_trace,
    )
    plan = SimpleNamespace(
        tile_trace=(
            {"pre_guard_route": "L0"},
            {"pre_guard_route": "L0"},
            {"pre_guard_route": "Full"},
        )
    )
    final_trace = (
        {"planned_route": "L0", "final_route": "L1"},
        {"planned_route": "L0", "final_route": "Full"},
        {"planned_route": "Full", "final_route": "Full"},
    )
    final_route = SimpleNamespace(
        events={
            "route_counts": {"L0": 0, "L1": 1, "Full": 2},
            "tile_trace_sha256": canonical_json_sha256(final_trace),
            "preflight_trace_sha256": preflight_trace_sha256,
            "coverage_certificate_sha256": events["coverage_certificate_sha256"],
        },
        tile_trace=final_trace,
    )

    assert _require_coverage_enriched_preflight(
        profile=profile, preflight=preflight
    ) == preflight_trace_sha256
    assert _require_coverage_enriched_final_route(
        preflight=preflight,
        preflight_trace_sha256=preflight_trace_sha256,
        final_route=final_route,
    ) == final_route.events["tile_trace_sha256"]
    assert _preflight_route_counts(
        plan=plan, preflight=preflight, final_route=final_route
    ) == {
        "planned": {"L0": 2, "L1": 0, "Full": 1},
        "accepted_before_final_full_promotion": {"L0": 0, "L1": 1, "Full": 1},
        "final": {"L0": 0, "L1": 1, "Full": 2},
        "l0_to_l1_enriched": 1,
        "promoted_full": 1,
        "rejection_reasons": {"owner support rejected": 1},
    }

    guarded_events = {**events, "selected_anchor_attribute_loo_frozen_guard": {}}
    with pytest.raises(RuntimeError, match="preflight changed"):
        _require_coverage_enriched_preflight(
            profile=profile,
            preflight=SimpleNamespace(
                events=guarded_events, tile_trace=preflight_trace
            ),
        )

    tampered_preflight = list(preflight_trace)
    tampered_preflight[0] = {**tampered_preflight[0], "accepted_level": "L0"}
    with pytest.raises(RuntimeError, match="preflight trace hash changed"):
        _require_coverage_enriched_preflight(
            profile=profile,
            preflight=SimpleNamespace(
                events=events, tile_trace=tuple(tampered_preflight)
            ),
        )

    tampered_final = list(final_trace)
    tampered_final[0] = {**tampered_final[0], "final_route": "L0"}
    with pytest.raises(RuntimeError, match="final route trace hash changed"):
        _require_coverage_enriched_final_route(
            preflight=preflight,
            preflight_trace_sha256=preflight_trace_sha256,
            final_route=SimpleNamespace(
                events=final_route.events, tile_trace=tuple(tampered_final)
            ),
        )
