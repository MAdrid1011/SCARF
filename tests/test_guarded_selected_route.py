"""Target-free guard resolution for the selected raw-head route."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


def _l0_inputs():
    return torch.ones(1, 1, 2, 4, 4), torch.ones(1, 1, 16, 1, 1)


def _l1_inputs(*, nonuniform_depth: bool = False):
    features = torch.zeros(1, 1, 2, 4, 4)
    for (row, column), value in {
        (0, 0): (0.0, 0.0),
        (0, 3): (2.0, 0.0),
        (3, 0): (0.0, 2.0),
        (3, 3): (2.0, 2.0),
    }.items():
        features[0, 0, :, row, column] = torch.tensor(value)
    depths = torch.ones(1, 1, 16, 1, 1)
    if nonuniform_depth:
        depths[0, 0, 15, 0, 0] = 2.0
    return features, depths


def _plan(features, depths):
    from saes.probe_first_schedule import build_incremental_probe_first_plan

    return build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
    )


def _mask_digest(mask):
    value = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _packed(
    plan,
    *,
    harmonic_slot: int | None = None,
    cross_check_covariances: bool = False,
    trace_update=None,
):
    from saes.sparse_gaussian_consumer import PackedGaussianAttributes

    positions = plan.selection_mask.nonzero(as_tuple=False)
    slots = positions[:, 0] * 16 + positions[:, 1] * 4 + positions[:, 2]
    count = int(slots.numel())
    means = torch.stack(
        (
            (positions[:, 2].to(torch.float32) + 0.5) / 4.0,
            (positions[:, 1].to(torch.float32) + 0.5) / 4.0,
            torch.ones(count),
        ),
        dim=1,
    )
    # The spatially distributed synthetic anchors share broad enough projected
    # support to exercise the accepted selected-only route. Separation itself
    # is covered by the progressive guard test.
    covariances = torch.eye(3).expand(count, -1, -1).clone() * 0.25
    if cross_check_covariances:
        for slot, diagonal in zip(
            (0, 3, 12, 15),
            (
                (1.0, 0.5, 0.5),
                (0.5, 1.0, 0.5),
                (0.5, 0.5, 1.0),
                (1.0, 0.5, 0.5),
            ),
        ):
            position = (slots == slot).nonzero(as_tuple=False)
            if position.numel():
                covariances[position.item()] = torch.diag(torch.tensor(diagonal))
    harmonics = torch.ones(count, 3, 1)
    if harmonic_slot is not None:
        harmonics[(slots == harmonic_slot).nonzero(as_tuple=False).item()] *= -1.0
    source_trace = {
        "schema_version": "saes-incremental-selected-output-adapter-audit-v1",
        "source_bound": True,
        "execution_scope": "s3_raw_gaussian_head_only",
        "head_forward_invocations": 1,
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
        "route_plan_contract_version": plan.events["contract_version"],
        "route_tile_trace_sha256": plan.events["tile_trace_sha256"],
        "route_primary_mask_sha256": _mask_digest(plan.primary_mask),
        "route_secondary_mask_sha256": _mask_digest(plan.secondary_mask),
        "route_full_mask_sha256": _mask_digest(plan.full_mask),
        "route_selection_mask_sha256": _mask_digest(plan.selection_mask),
    }
    if trace_update is not None:
        source_trace.update(trace_update)
    source_trace_sha256 = hashlib.sha256(
        json.dumps(source_trace, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PackedGaussianAttributes(
        batch_indices=torch.zeros(count, dtype=torch.int64),
        dense_slots=slots.to(dtype=torch.int64),
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=torch.full((count,), 0.2),
        source_trace=source_trace,
        source_trace_sha256=source_trace_sha256,
    )


def _cameras():
    return (
        torch.eye(4).reshape(1, 1, 4, 4),
        torch.eye(3).reshape(1, 1, 3, 3),
    )


def _source_opacities(*, primary_only_zero: bool):
    values = torch.full((1, 1, 16, 1, 1), 0.2)
    if primary_only_zero:
        primary_slots = torch.tensor((0, 3, 12, 15), dtype=torch.int64)
        values.reshape(1, 16)[0, :] = 0.0
        values.reshape(1, 16)[0, primary_slots] = 0.2
    return values


def test_exact_zero_source_opacity_certificate_preserves_selected_l0_packet():
    from saes.guarded_selected_route import resolve_guarded_selected_route
    from saes.progressive_saes import DELETION_CERTIFICATE_SOURCE_KIND

    features, depths = _l0_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        source_opacities=_source_opacities(primary_only_zero=True),
        source_opacity_certificate_kind=DELETION_CERTIFICATE_SOURCE_KIND,
        require_deletion_certificate=True,
    )

    assert resolved.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert int(resolved.selected_output_mask.sum()) == 4
    assert int(resolved.additional_full_mask.sum()) == 0
    assert resolved.events["deletion_certificate_checks"] == 1
    assert resolved.events["deletion_certificate_accepted_tiles"] == 1
    assert resolved.events["deletion_certificate_zero_opacity_gaussians"] == 12
    certificate = resolved.tile_trace[0]["deletion_certificate"]
    assert certificate["passed"] is True
    assert certificate["reason"] == "exact_zero_alpha"
    assert certificate["nonprobe_s3_attribute_reads"] == 0
    assert certificate["nonprobe_source_opacity"] == {
        "count": 12,
        "strictly_positive_count": 0,
        "minimum": 0.0,
        "p50": 0.0,
        "p95": 0.0,
        "maximum": 0.0,
        "sum": 0.0,
    }


def test_nonzero_source_opacity_certificate_fails_closed_to_full():
    from saes.guarded_selected_route import resolve_guarded_selected_route
    from saes.progressive_saes import DELETION_CERTIFICATE_SOURCE_KIND

    features, depths = _l0_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        source_opacities=_source_opacities(primary_only_zero=False),
        source_opacity_certificate_kind=DELETION_CERTIFICATE_SOURCE_KIND,
        require_deletion_certificate=True,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert int(resolved.selected_output_mask.sum()) == 16
    assert int(resolved.additional_full_mask.sum()) == 4
    assert resolved.events["deletion_certificate_rejected_tiles"] == 1
    assert resolved.events["uncertified_deletion_fallback_tiles"] == 1
    assert resolved.events["deletion_certificate_rejection_reasons"] == {
        "nonzero_source_opacity": 1
    }
    certificate = resolved.tile_trace[0]["deletion_certificate"]
    assert certificate["passed"] is False
    assert certificate["reason"] == "nonzero_source_opacity"
    assert certificate["nonprobe_source_opacity"]["strictly_positive_count"] == 12
    assert certificate["nonprobe_source_opacity"]["minimum"] == pytest.approx(0.2)


def test_l0_guard_accepts_only_primary_selected_attributes():
    from saes.guarded_selected_route import resolve_guarded_selected_route

    features, depths = _l0_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )

    assert resolved.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert int(resolved.selected_output_mask.sum()) == 4
    assert int(resolved.additional_full_mask.sum()) == 0
    assert resolved.events["available_selected_descriptor_count"] == 12
    assert resolved.events["whole_pipeline_s2_s3_sparse_execution_verified"] is False


@pytest.mark.parametrize(
    ("decision_semantics", "feature_statistic"),
    (
        ("paper-probe-feature-variance-first-hit", "raw-probe-mean-channel-variance"),
        (
            "paper-probe-normalized-feature-first-hit",
            "normalized-probe-vector-standard-deviation",
        ),
    ),
)
def test_paper_nonzero_policy_is_explicit_and_excludes_lossless_certificate(
    decision_semantics, feature_statistic
):
    from saes.guarded_selected_route import (
        PAPER_NONZERO_DEV_POLICY,
        resolve_guarded_selected_route,
    )
    from saes.progressive_saes import DELETION_CERTIFICATE_SOURCE_KIND
    from saes.probe_first_schedule import (
        PAPER_KP_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
    )

    features, depths = _l0_inputs()
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics=decision_semantics,
        l1_anchor_semantics=PAPER_KP_ANCHOR_SEMANTICS,
    )
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        execution_policy=PAPER_NONZERO_DEV_POLICY,
    )
    assert resolved.events["execution_policy"] == PAPER_NONZERO_DEV_POLICY
    assert plan.events["feature_statistic"] == feature_statistic
    assert resolved.events["not_lossless_deletion"] is True
    assert resolved.events["source_nonprobe_s3_attribute_reads"] == 0
    assert resolved.events["target_rgb_accessed_before_commit"] is False
    assert resolved.events["attribute_guard_mode"] == "paper-formula-no-extra-attribute-guard"
    assert resolved.events["context_safety_guard"] is False
    assert resolved.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert resolved.tile_trace[0]["guard_checks"] == []

    with pytest.raises(ValueError, match="cannot claim a lossless certificate"):
        resolve_guarded_selected_route(
            plan,
            _packed(plan),
            features=features,
            depths=depths,
            context_extrinsics=extrinsics,
            context_intrinsics=intrinsics,
            source_opacities=_source_opacities(primary_only_zero=True),
            source_opacity_certificate_kind=DELETION_CERTIFICATE_SOURCE_KIND,
            require_deletion_certificate=True,
            execution_policy=PAPER_NONZERO_DEV_POLICY,
        )


def test_l0_depth_continuity_widens_preloaded_l1_anchors_without_new_request():
    from saes.guarded_selected_route import (
        ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
        resolve_guarded_selected_route,
    )

    features, depths = _l0_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        execution_policy=ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    assert int(plan.secondary_mask.sum()) == 8
    assert int(resolved.selected_output_mask.sum()) == 12
    assert int(resolved.additional_full_mask.sum()) == 0
    assert resolved.events["l0_depth_continuity_guard"] is True
    assert resolved.events["l0_depth_continuity_widened_tiles"] == 1
    assert resolved.events["l0_depth_continuity_promoted_full_tiles"] == 0
    assert resolved.tile_trace[0]["l0_depth_continuity"] == {
        "checked": True,
        "depth_uniform": True,
        "action": "widen_l1",
    }


def test_l0_depth_continuity_promotes_nonuniform_geometry_to_full():
    from saes.guarded_selected_route import (
        ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
        resolve_guarded_selected_route,
    )

    features, depths = _l0_inputs()
    depths = depths.clone()
    depths[0, 0, 15, 0, 0] = 2.0
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        execution_policy=ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert int(plan.secondary_mask.sum()) == 0
    assert int(resolved.selected_output_mask.sum()) == 16
    assert int(resolved.additional_full_mask.sum()) == 12
    assert resolved.events["l0_depth_continuity_widened_tiles"] == 0
    assert resolved.events["l0_depth_continuity_promoted_full_tiles"] == 1
    assert resolved.tile_trace[0]["l0_depth_continuity"] == {
        "checked": True,
        "depth_uniform": False,
        "action": "promote_full",
    }


def test_l0_depth_continuity_leaves_original_l1_first_hit_unchanged():
    from saes.guarded_selected_route import (
        ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
        resolve_guarded_selected_route,
    )

    features, depths = _l1_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        execution_policy=ENGINEERING_L1_12_CONTINUITY_DEV_POLICY,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    assert int(resolved.selected_output_mask.sum()) == 12
    assert resolved.tile_trace[0]["l0_depth_continuity"] is None
    assert resolved.tile_trace[0]["guard_checks"] == []


def test_balanced_continuity_policy_preserves_the_balanced_l1_packet():
    from saes.guarded_selected_route import (
        ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY,
        resolve_guarded_selected_route,
    )
    from saes.probe_first_schedule import (
        BALANCED_L1_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
    )

    features, depths = _l1_inputs()
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
        l1_anchor_semantics=BALANCED_L1_ANCHOR_SEMANTICS,
    )
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        execution_policy=ENGINEERING_L1_12_BALANCED_CONTINUITY_DEV_POLICY,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    assert resolved.events["l1_anchor_semantics"] == BALANCED_L1_ANCHOR_SEMANTICS
    assert resolved.events["not_lossless_deletion"] is True
    assert int(resolved.additional_full_mask.sum()) == 0
    assert torch.equal(resolved.selected_output_mask, plan.selection_mask)
    assert {
        tuple(position.tolist())
        for position in resolved.selected_output_mask[0].nonzero(as_tuple=False)
    } == {
        (0, 0),
        (0, 1),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 2),
        (3, 3),
    }


def test_adaptive_l1_15_continuity_policy_preserves_the_bound_tile_layout():
    from saes.guarded_selected_route import (
        ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY,
        resolve_guarded_selected_route,
    )
    from saes.probe_first_schedule import (
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
    )

    features, depths = _l1_inputs()
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        execution_policy=ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    assert resolved.events["l1_anchor_semantics"] == ADAPTIVE_L1_15_ANCHOR_SEMANTICS
    assert resolved.events["l1_anchor_count"] == 15
    assert resolved.events["not_lossless_deletion"] is True
    assert int(resolved.additional_full_mask.sum()) == 0
    assert torch.equal(resolved.selected_output_mask, plan.selection_mask)
    assert (
        resolved.tile_trace[0]["retained_local_positions"]
        == plan.tile_trace[0]["l1_anchor_local_positions"]
    )
    assert len(resolved.tile_trace[0]["retained_local_positions"]) == 15


def test_adaptive_l1_15_confidence_policy_promotes_an_ambiguous_center_to_full():
    from saes.guarded_selected_route import (
        ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY,
        resolve_guarded_selected_route,
    )
    from saes.probe_first_schedule import (
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
    )

    features, depths = _l0_inputs()
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )
    assert plan.tile_trace[0]["adaptive_l1_best_to_second_residual_ratio"] == pytest.approx(
        1.0
    )
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        execution_policy=ENGINEERING_L1_15_ADAPTIVE_CONFIDENCE_DEV_POLICY,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert int(resolved.selected_output_mask.sum()) == 16
    assert int(resolved.additional_full_mask.sum()) == 1
    assert resolved.events["adaptive_l1_confidence_guard"] is True
    assert resolved.events["adaptive_l1_confidence_checked_tiles"] == 1
    assert resolved.events["adaptive_l1_confidence_retained_l1_tiles"] == 0
    assert resolved.events["adaptive_l1_confidence_promoted_full_tiles"] == 1
    confidence = resolved.tile_trace[0]["adaptive_l1_confidence"]
    assert confidence["checked"] is True
    assert confidence["leave_one_out_residual"] == pytest.approx(0.0)
    assert confidence["best_to_second_residual_ratio"] == pytest.approx(1.0)
    assert confidence["maximum_ratio"] == 0.5
    assert confidence["passed"] is False
    assert confidence["action"] == "promote_full"


def test_adaptive_l1_15_absolute_residual_policy_promotes_only_the_missing_center():
    from saes.guarded_selected_route import (
        ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
        resolve_guarded_selected_route,
    )
    from saes.probe_first_schedule import (
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
    )

    features, depths = _l1_inputs()
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )
    residual = plan.tile_trace[0]["adaptive_l1_leave_one_out_residual"]
    assert residual > 0.0
    extrinsics, intrinsics = _cameras()
    promoted = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        execution_policy=ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
        adaptive_l1_maximum_leave_one_out_residual=0.0,
    )
    retained = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        execution_policy=ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
        adaptive_l1_maximum_leave_one_out_residual=float(residual),
    )

    assert promoted.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert int(promoted.additional_full_mask.sum()) == 1
    assert promoted.events["adaptive_l1_absolute_residual_promoted_full_tiles"] == 1
    assert retained.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    assert int(retained.additional_full_mask.sum()) == 0
    assert retained.events["adaptive_l1_absolute_residual_retained_l1_tiles"] == 1


def test_l0_primary_cross_check_adds_only_the_missing_full_positions():
    from saes.guarded_selected_route import resolve_guarded_selected_route

    features, depths = _l0_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan, cross_check_covariances=True),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    # The conservative L0 plan already staged the eight L1 boundary anchors
    # because its S2 probe depths permit a later L1 promotion. Only the four
    # interior slots remain for the fail-closed Full extension.
    assert int(resolved.additional_full_mask.sum()) == 4
    assert resolved.events["probe_cross_check_checks"] == 1
    assert resolved.events["probe_cross_check_rejections"] == 1
    cross_check = resolved.tile_trace[0]["guard_checks"][0]["probe_cross_check"]
    assert cross_check["checked"] is True
    assert cross_check["passed"] is False
    assert cross_check["error"] > cross_check["threshold"]
    assert cross_check["nonprobe_s3_attribute_reads"] == 0


def test_dense_and_selected_routes_match_on_primary_cross_check_fallback():
    from saes.guarded_selected_route import resolve_guarded_selected_route
    from saes.progressive_saes import apply_progressive_saes

    features, depths = _l0_inputs()
    plan = _plan(features, depths)
    packed = _packed(plan, cross_check_covariances=True)
    extrinsics, intrinsics = _cameras()
    selected = resolve_guarded_selected_route(
        plan,
        packed,
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )

    means = torch.zeros(1, 16, 3)
    covariances = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1) * 0.25
    harmonics = torch.ones(1, 16, 3, 1)
    opacities = torch.full((1, 16), 0.2)
    means[0, packed.dense_slots] = packed.means
    covariances[0, packed.dense_slots] = packed.covariances
    harmonics[0, packed.dense_slots] = packed.harmonics
    opacities[0, packed.dense_slots] = packed.opacities
    dense = SimpleNamespace(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )
    modified, stats, _ = apply_progressive_saes(
        dense,
        4,
        4,
        tile_size=4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        cross_check_threshold=0.015,
        features=features,
        depths=depths,
        decision_semantics="probe-normalized-std-first-hit",
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        materialization_guard=True,
        context_safety_guard=True,
    )

    assert stats["level0_tiles"] == 0
    assert stats["level1_tiles"] == 0
    assert stats["full_tiles"] == 1
    assert stats["probe_cross_check_l0_rejections"] == 1
    assert not bool(modified.any())
    assert bool(selected.raw_head_request_mask.all())
    assert selected.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}


def test_l1_guard_reads_only_the_twelve_selected_anchors():
    from saes.guarded_selected_route import resolve_guarded_selected_route

    features, depths = _l1_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    assert int(plan.selection_mask.sum()) == 12
    assert int(resolved.selected_output_mask.sum()) == 12
    assert int(resolved.additional_full_mask.sum()) == 0
    assert resolved.events["unavailable_dense_descriptor_count"] == 4
    assert resolved.tile_trace[0]["guard_checks"][0]["anchor_count"] == 12


def test_l1_primary_cross_check_uses_the_primary_prefix_not_skipped_slots():
    from saes.guarded_selected_route import resolve_guarded_selected_route

    features, depths = _l1_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan, cross_check_covariances=True),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert int(resolved.additional_full_mask.sum()) == 4
    cross_check = resolved.tile_trace[0]["guard_checks"][0]["probe_cross_check"]
    assert cross_check["anchor_count"] == 4
    assert cross_check["passed"] is False
    assert cross_check["nonprobe_s3_attribute_reads"] == 0


def test_l1_guard_rejection_appends_only_missing_full_positions():
    from saes.guarded_selected_route import resolve_guarded_selected_route

    features, depths = _l1_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    # The first non-primary L1 anchor is selected but makes the selected-only
    # materialization guard fail, requiring the unproduced interior positions.
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan, harmonic_slot=1),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert int(resolved.selected_output_mask.sum()) == 16
    assert int(resolved.additional_full_mask.sum()) == 4
    assert resolved.events["requires_incremental_full_dispatch"] is True
    assert not bool((resolved.additional_full_mask & plan.selection_mask).any())
    assert torch.equal(
        resolved.raw_head_request_mask,
        plan.selection_mask | resolved.additional_full_mask,
    )
    assert torch.equal(resolved.selected_output_mask, resolved.raw_head_request_mask)
    assert resolved.additional_full_mask.nonzero(as_tuple=False).tolist() == [
        [0, 1, 1],
        [0, 1, 2],
        [0, 2, 1],
        [0, 2, 2],
    ]
    assert resolved.events["additional_full_mask_sha256"] == _mask_digest(
        resolved.additional_full_mask
    )
    assert resolved.events["raw_head_request_mask_sha256"] == _mask_digest(
        resolved.raw_head_request_mask
    )
    assert resolved.tile_trace[0]["final_route"] == "Full"


def test_l1_guard_rejection_executes_appended_full_once_without_replaying_work():
    from saes.incremental_selected_output_execution import (
        IncrementalSelectedOutputProducer,
    )
    from saes.guarded_selected_route import resolve_guarded_selected_route

    features, depths = _l1_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan, harmonic_slot=1),
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )
    torch.manual_seed(113)
    head = nn.Sequential(
        nn.Conv2d(3, 5, 3, 1, 1),
        nn.GELU(),
        nn.Conv2d(5, 2, 3, 1, 1),
    ).eval()
    inputs = torch.randn(1, 3, 4, 4)
    dense = head(inputs)
    producer = IncrementalSelectedOutputProducer(head, inputs, tile_size=4)

    producer.execute("primary", plan.primary_mask)
    producer.execute("secondary", plan.secondary_mask)
    producer.execute("full", plan.full_mask)
    extension = producer.execute("full_extension", resolved.additional_full_mask)
    replay = producer.finalize(resolved.raw_head_request_mask)

    assert extension["mask_sha256"] == resolved.events["additional_full_mask_sha256"]
    assert extension["head_final_positions_requested"] == 4
    assert extension["head_final_positions_reused"] == 0
    assert extension["head_final_positions_executed"] == 4
    assert extension["second_conv_positions_executed"] == 4
    assert extension["first_conv_positions_executed"] == 0
    assert extension["full_tile_native_identity_verified"] is False
    assert sum(
        event["head_final_positions_executed"] for event in replay.events["phases"]
    ) == 16
    assert torch.equal(replay.computed_mask, resolved.raw_head_request_mask)
    assert replay.events["computed_mask_sha256"] == resolved.events[
        "raw_head_request_mask_sha256"
    ]
    torch.testing.assert_close(replay.values, dense, rtol=1.0e-5, atol=1.0e-5)


def test_missing_context_geometry_fails_closed_to_full():
    from saes.guarded_selected_route import resolve_guarded_selected_route

    features, depths = _l0_inputs()
    plan = _plan(features, depths)
    resolved = resolve_guarded_selected_route(
        plan,
        _packed(plan),
        features=features,
        depths=depths,
        context_extrinsics=None,
        context_intrinsics=None,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert int(resolved.additional_full_mask.sum()) == 4
    assert resolved.tile_trace[0]["guard_checks"][0]["context_safety"]["reason"] == "missing_geometry"


def test_preplanned_full_tile_is_selected_attribute_passthrough_only():
    from saes.guarded_selected_route import resolve_guarded_selected_route

    features, depths = _l1_inputs(nonuniform_depth=True)
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    packed = _packed(plan)
    resolved = resolve_guarded_selected_route(
        plan,
        packed,
        features=features,
        depths=depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
    )

    assert resolved.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert torch.equal(resolved.selected_output_mask, plan.selection_mask)
    assert int(resolved.additional_full_mask.sum()) == 0
    assert resolved.events["full_passthrough_dense_slots"] == packed.dense_slots.tolist()
    assert resolved.events["full_tile_native_identity_verified"] is False


def test_resolver_rejects_plan_trace_or_slot_binding_drift():
    from saes.guarded_selected_route import resolve_guarded_selected_route

    features, depths = _l0_inputs()
    plan = _plan(features, depths)
    extrinsics, intrinsics = _cameras()
    with pytest.raises(ValueError, match="route tile trace"):
        resolve_guarded_selected_route(
            plan,
            _packed(plan, trace_update={"route_tile_trace_sha256": "0" * 64}),
            features=features,
            depths=depths,
            context_extrinsics=extrinsics,
            context_intrinsics=intrinsics,
        )
