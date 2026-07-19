"""Focused source-contract tests for DepthSplat's nonzero L0/L1 merge path."""

from __future__ import annotations

import hashlib
import json

import pytest


torch = pytest.importorskip("torch")


def _mask_sha256(mask):
    value = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _plan(
    features,
    z_depths,
    *,
    literal_paper_t4=False,
    feature_threshold=0.1,
    depth_threshold=0.1,
):
    from saes.probe_first_schedule import (
        build_incremental_probe_first_plan,
        build_literal_paper_t4_probe_first_plan,
    )

    if literal_paper_t4:
        return build_literal_paper_t4_probe_first_plan(
            features,
            z_depths,
            height=4,
            width=4,
            feature_threshold=feature_threshold,
            depth_threshold=depth_threshold,
        )

    return build_incremental_probe_first_plan(
        features,
        z_depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=feature_threshold,
        depth_threshold=depth_threshold,
        decision_semantics="probe-normalized-std-first-hit",
    )


def _selected_packet_and_packed(
    mask,
    z_depths,
    *,
    routing_features,
    opacity_logit=-1.0,
    harmonic_delta=0.0,
    covariance_scale=1.0,
):
    from depthsplat.src.geometry.projection import get_world_rays, sample_image_grid
    from saes.depthsplat_backend import canonical_json_sha256, source_native_depthsplat_coordinates
    from saes.depthsplat_l0_l1_materializer import depthsplat_z_depth_world_means
    from saes.depthsplat_selected_output import (
        DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
        DepthSplatPackedGaussianAttributes,
        DepthSplatSparseRawPacket,
        _tensor_sha256,
        depthsplat_attribute_binding_sha256,
    )

    positions = mask.nonzero(as_tuple=False)
    count = positions.shape[0]
    raw = torch.zeros(count, 7, dtype=torch.float32)
    raw[:, 0] = opacity_logit
    raw[:, 1] = torch.linspace(-0.2, 0.2, count)
    raw[:, 2] = torch.linspace(0.2, -0.2, count)
    raw[:, 3:] = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) / 100.0
    coordinates = source_native_depthsplat_coordinates(
        raw, mask, sample_image_grid=sample_image_grid
    )
    extrinsics = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).repeat(count, 1, 1)
    intrinsics = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(count, 1, 1)
    depths = z_depths[0, positions[:, 0], positions[:, 1], positions[:, 2]].clone()
    means = depthsplat_z_depth_world_means(
        coordinates,
        extrinsics,
        intrinsics,
        depths,
        source_get_world_rays=get_world_rays,
    )
    rgb = torch.stack(
        (
            torch.linspace(0.1, 0.9, count),
            torch.linspace(0.9, 0.1, count),
            torch.full((count,), 0.4),
        ),
        dim=1,
    )
    harmonics = torch.zeros(count, 3, 2, dtype=torch.float32)
    harmonics[:, :, 0] = rgb + harmonic_delta
    harmonics[:, :, 1] = rgb * 0.2
    opacities = raw[:, 0].sigmoid()
    height, width = mask.shape[-2:]
    pixels = positions[:, 1] * width + positions[:, 2]
    slots = (positions[:, 0] * (height * width) + pixels).to(dtype=torch.int64)
    keys = torch.stack(
        (torch.zeros_like(pixels), positions[:, 0], pixels, torch.zeros_like(pixels)), dim=1
    ).to(dtype=torch.int64)
    trace = {
        "contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
        "source_bound": True,
        "execution_scope": "depthsplat-dense-regressor-selected-gaussian-head-only",
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
        "source_rgb_keyword": "input_images",
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "head_forward_invocations": 1,
        "selection_mask_sha256": _mask_sha256(mask),
        "selected_descriptor_sha256": _tensor_sha256(raw),
        "selected_rgb_sha256": _tensor_sha256(rgb),
        "native_execution_sha256": "e" * 64,
        "source_view_count": int(mask.shape[0]),
        "source_image_shape": [int(mask.shape[1]), int(mask.shape[2])],
        "routing_features_sha256": _tensor_sha256(routing_features),
        "routing_z_depths_sha256": _tensor_sha256(z_depths),
        "native_full_passthrough_mask_sha256": _mask_sha256(torch.zeros_like(mask)),
        "native_full_passthrough_positions": 0,
    }
    packet = DepthSplatSparseRawPacket(
        descriptor_keys=keys,
        raw_head_descriptors=raw,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        coordinates=coordinates,
        depths=depths,
        mapped_opacities=opacities,
        source_rgb=rgb,
        dense_slots=slots,
        source_trace=trace,
    )
    covariances = (
        torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(count, 1, 1)
        * float(covariance_scale)
    )
    attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=slots,
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )
    empty_slots = slots[:0]
    empty_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=empty_slots,
        means=means[:0],
        covariances=torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)[:0],
        harmonics=harmonics[:0],
        opacities=opacities[:0],
    )
    packed_trace = {
        **trace,
        "native_adapter_attribute_binding_sha256": attribute_binding,
        "native_full_adapter_attribute_execution_sha256": trace["native_execution_sha256"],
        "native_full_adapter_attribute_passthrough_mask_sha256": trace[
            "native_full_passthrough_mask_sha256"
        ],
        "native_full_adapter_attribute_binding_sha256": empty_attribute_binding,
        "native_full_adapter_attribute_passthrough_count": 0,
        "selected_native_rgb_adapter_compact_count": count,
        "selected_native_rgb_adapter_executed": True,
    }
    packed = DepthSplatPackedGaussianAttributes(
        dense_slots=slots.clone(),
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        source_trace=packed_trace,
        source_trace_sha256=canonical_json_sha256(packed_trace),
        attribute_binding_sha256=attribute_binding,
    )
    return packet, packed


def _preflight(
    features,
    z_depths,
    *,
    opacity_logit=-1.0,
    harmonic_delta=0.0,
    covariance_scale=1.0,
    literal_paper_t4=False,
    feature_threshold=0.1,
    depth_threshold=0.1,
    collect_selected_anchor_attribute_loo_risk=False,
    selected_anchor_attribute_loo_frozen_guard=None,
    selected_anchor_attribute_loo_maximum_risk=None,
    preflight_source_sample_image_grid=None,
    maximum_coverage_covariance_scale=None,
):
    from depthsplat.src.geometry.projection import get_world_rays, sample_image_grid
    from saes.depthsplat_l0_l1_materializer import preflight_depthsplat_l0_l1_materialization

    plan = _plan(
        features,
        z_depths,
        literal_paper_t4=literal_paper_t4,
        feature_threshold=feature_threshold,
        depth_threshold=depth_threshold,
    )
    packet, packed = _selected_packet_and_packed(
        plan.selection_mask,
        z_depths,
        routing_features=features,
        opacity_logit=opacity_logit,
        harmonic_delta=harmonic_delta,
        covariance_scale=covariance_scale,
    )
    preflight = preflight_depthsplat_l0_l1_materialization(
        packet,
        packed,
        plan,
        features,
        z_depths,
        source_sample_image_grid=(
            sample_image_grid
            if preflight_source_sample_image_grid is None
            else preflight_source_sample_image_grid
        ),
        source_get_world_rays=get_world_rays,
        maximum_coverage_covariance_scale=(
            (1.0 if literal_paper_t4 else 16.0)
            if maximum_coverage_covariance_scale is None
            else maximum_coverage_covariance_scale
        ),
        execution_profile=(
            "depthsplat-literal-paper-t4-selected-probe-moment-v1"
            if literal_paper_t4
            else "depthsplat-development-omitted-z-alpha-union-v1"
        ),
        collect_selected_anchor_attribute_loo_risk=(
            collect_selected_anchor_attribute_loo_risk
        ),
        selected_anchor_attribute_loo_frozen_guard=(
            selected_anchor_attribute_loo_frozen_guard
        ),
        selected_anchor_attribute_loo_maximum_risk=(
            selected_anchor_attribute_loo_maximum_risk
        ),
    )
    return plan, packet, packed, preflight


def _literal_frozen_guard(plan, *, threshold_value=30.0):
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_LITERAL_PAPER_T4_V16_RECORD_KIND,
        DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_GUARD_SCHEMA,
        DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
    )

    return {
        "schema_version": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_GUARD_SCHEMA,
        "frozen_record_kind": DEPTHSPLAT_LITERAL_PAPER_T4_V16_RECORD_KIND,
        "frozen_record_sha256": "1" * 64,
        "threshold_value": threshold_value,
        "threshold_rule": "test-maximum-held-out-risk-v1",
        "risk_metric": DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
        "materialization_profile": DEPTHSPLAT_LITERAL_PAPER_T4_MATERIALIZATION_PROFILE,
        "route_plan_contract": plan.events["contract_version"],
        "route_plan_config_sha256": plan.events[
            "literal_paper_t4_route_config_sha256"
        ],
        "acid_binding_sha256": "2" * 64,
        "application_sha256": "3" * 64,
    }


def _authenticated_literal_frozen_guard(plan, *, threshold_value=30.0):
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    return calibration._issue_verified_literal_t4_materializer_guard(
        _literal_frozen_guard(plan, threshold_value=threshold_value)
    )


def test_z_depth_lift_is_not_euclidean_unit_ray_lift():
    from depthsplat.src.geometry.projection import get_world_rays
    from saes.depthsplat_l0_l1_materializer import depthsplat_z_depth_world_means

    coordinates = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    extrinsics = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    intrinsics = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
    z_depth = torch.tensor([2.0], dtype=torch.float32)

    actual = depthsplat_z_depth_world_means(
        coordinates,
        extrinsics,
        intrinsics,
        z_depth,
        source_get_world_rays=get_world_rays,
    )
    _origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
    euclidean = directions / directions.norm(dim=1, keepdim=True) * z_depth.unsqueeze(1)

    torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0, 2.0]]))
    assert not torch.allclose(actual, euclidean)


def test_shape_aware_coverage_expands_for_virtual_gaussian_extent_not_just_center():
    from saes.depthsplat_l0_l1_materializer import _coverage_closed_covariance

    covariance = torch.eye(3, dtype=torch.float32)
    mean = torch.zeros(3, dtype=torch.float32)
    virtual_means = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    virtual_covariances = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)

    closed, certificate = _coverage_closed_covariance(
        covariance,
        mean,
        virtual_means,
        virtual_covariances,
        maximum_scale=16.0,
    )

    # Centre-only containment would leave scale at one.  A 2-sigma source
    # ellipsoid centred one sigma away needs (1 + 2)^2 / 2^2 = 2.25.
    torch.testing.assert_close(closed, torch.eye(3) * 2.25)
    assert certificate["containment_radius"] == 2.0
    assert certificate["maximum_containment_lhs_before_scale"] == pytest.approx(3.0)
    assert certificate["maximum_containment_lhs_after_scale"] == pytest.approx(2.0)
    assert certificate["maximum_whitened_virtual_covariance_eigenvalue"] == pytest.approx(1.0)
    assert certificate["moment_covariance_scale"] == pytest.approx(2.25)

    far_closed, far_certificate = _coverage_closed_covariance(
        covariance,
        mean,
        torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32),
        virtual_covariances,
        maximum_scale=16.0,
    )
    # The virtual outer extent reaches five sigma from the merged centre.
    torch.testing.assert_close(far_closed, torch.eye(3) * 6.25)
    assert far_certificate["maximum_containment_lhs_before_scale"] == pytest.approx(5.0)
    assert far_certificate["maximum_containment_lhs_after_scale"] == pytest.approx(2.0)
    assert far_certificate["moment_covariance_scale"] == pytest.approx(6.25)


def test_l0_preflight_merges_nonzero_virtual_domain_and_discards_prefetched_l1_rows():
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(features, z_depths)

    assert plan.tile_trace[0]["pre_guard_route"] == "L0"
    assert preflight.events["nonzero_direct_deletion"] is False
    assert preflight.events["skipped_s3_attributes_accessed"] is False
    assert preflight.events["accepted_tiles"] == 1
    assert preflight.update_dense_slots.numel() == 4
    assert torch.linalg.eigvalsh(preflight.covariances).min() >= -1e-6

    route = resolve_depthsplat_compact_final_route(plan, preflight)
    assert route.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert int(plan.selection_mask.sum()) == 12
    assert int(route.selected_output_mask.sum()) == 4
    assert int(route.raw_head_request_mask.sum()) == 12
    assert route.events["producer_only_prefetch_descriptor_count"] == 8


def test_rgb_derived_sh_field_changes_the_nonzero_merge_update():
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    _plan_a, _packet_a, _packed_a, preflight_a = _preflight(features, z_depths)
    _plan_b, _packet_b, _packed_b, preflight_b = _preflight(
        features, z_depths, harmonic_delta=0.25
    )

    assert preflight_a.update_dense_slots.numel() == preflight_b.update_dense_slots.numel()
    assert not torch.allclose(preflight_a.harmonics, preflight_b.harmonics)
    torch.testing.assert_close(
        (preflight_b.harmonics - preflight_a.harmonics)[:, :, 0],
        torch.full_like(preflight_a.harmonics[:, :, 0], 0.25),
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.count_nonzero(
        (preflight_b.harmonics - preflight_a.harmonics)[:, :, 1]
    ) == 0


def test_virtual_geometry_uses_source_s2_z_depth_at_the_skipped_position():
    features = torch.zeros(1, 1, 3, 4, 4)
    flat_z_depths = torch.full((1, 1, 4, 4), 2.0)
    sloped_z_depths = flat_z_depths.clone()
    # The four L0 probes remain unchanged, so this is not a route change. It
    # only verifies that the virtual skipped Gaussian uses the source S2 z at
    # its own pixel rather than copying an anchor's z plane.
    sloped_z_depths[0, 0, 1, 1] = 4.0
    _plan_a, _packet_a, _packed_a, preflight_a = _preflight(features, flat_z_depths)
    plan_b, _packet_b, _packed_b, preflight_b = _preflight(features, sloped_z_depths)

    assert plan_b.tile_trace[0]["pre_guard_route"] == "L0"
    assert preflight_b.tile_trace[0]["virtual_z_depth_max"] == pytest.approx(4.0)
    assert not torch.allclose(preflight_a.means, preflight_b.means)


def test_continuity_failure_promotes_the_entire_tile_full_without_omitted_attributes():
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    features = torch.zeros(1, 1, 3, 4, 4)
    features[:, :, :, 1, 1] = 10.0
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(features, z_depths)
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert plan.tile_trace[0]["pre_guard_route"] == "L0"
    assert preflight.events["promoted_full_tiles"] == 1
    assert "continuity" in preflight.tile_trace[0]["reason"]
    assert int(route.selected_output_mask.sum()) == 16
    assert int(route.additional_full_mask.sum()) == 4
    assert preflight.events["source_nonprobe_s3_attribute_reads"] == 0


def test_endpoint_failure_promotes_full_and_apply_preserves_all_full_attributes_bitwise():
    from saes.depthsplat_backend import canonical_json_sha256
    from saes.depthsplat_l0_l1_materializer import (
        apply_depthsplat_compact_l0_l1_materialization,
        resolve_depthsplat_compact_final_route,
    )
    from saes.depthsplat_selected_output import (
        DepthSplatPackedGaussianAttributes,
        depthsplat_attribute_binding_sha256,
        subset_depthsplat_sparse_raw_packet,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(features, z_depths, opacity_logit=100.0)
    route = resolve_depthsplat_compact_final_route(plan, preflight)
    source_packet, source_packed = _selected_packet_and_packed(
        route.selected_output_mask,
        z_depths,
        routing_features=features,
        opacity_logit=100.0,
    )
    final_packet = subset_depthsplat_sparse_raw_packet(
        source_packet, route.selected_output_mask
    )
    attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=source_packed.dense_slots,
        means=source_packed.means,
        covariances=source_packed.covariances,
        harmonics=source_packed.harmonics,
        opacities=source_packed.opacities,
    )
    full_mask_sha256 = _mask_sha256(route.full_passthrough_mask)
    full_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=source_packed.dense_slots,
        means=source_packed.means,
        covariances=source_packed.covariances,
        harmonics=source_packed.harmonics,
        opacities=source_packed.opacities,
    )
    final_trace = {
        **final_packet.source_trace,
        "native_full_passthrough_mask_sha256": full_mask_sha256,
        "native_full_passthrough_positions": int(route.full_passthrough_mask.sum()),
        "native_adapter_attribute_binding_sha256": attribute_binding,
        "native_full_adapter_attribute_execution_sha256": "e" * 64,
        "native_full_adapter_attribute_passthrough_mask_sha256": full_mask_sha256,
        "native_full_adapter_attribute_binding_sha256": full_attribute_binding,
        "native_full_adapter_attribute_passthrough_count": int(
            route.full_passthrough_mask.sum()
        ),
        "selected_native_rgb_adapter_compact_count": 0,
        "selected_native_rgb_adapter_executed": False,
    }
    final_packed = DepthSplatPackedGaussianAttributes(
        dense_slots=source_packed.dense_slots.clone(),
        means=source_packed.means.clone(),
        covariances=source_packed.covariances.clone(),
        harmonics=source_packed.harmonics.clone(),
        opacities=source_packed.opacities.clone(),
        source_trace=final_trace,
        source_trace_sha256=canonical_json_sha256(final_trace),
        attribute_binding_sha256=attribute_binding,
    )
    result = apply_depthsplat_compact_l0_l1_materialization(final_packed, preflight, route)

    assert preflight.tile_trace[0]["reason"] == "native_opacity_endpoint_requires_full"
    assert route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert torch.equal(result.means, final_packed.means)
    assert torch.equal(result.covariances, final_packed.covariances)
    assert torch.equal(result.harmonics, final_packed.harmonics)
    assert torch.equal(result.opacities, final_packed.opacities)


def test_apply_rejects_final_full_route_without_matching_native_full_provenance():
    from saes.depthsplat_backend import canonical_json_sha256
    from saes.depthsplat_l0_l1_materializer import (
        apply_depthsplat_compact_l0_l1_materialization,
        resolve_depthsplat_compact_final_route,
    )
    from saes.depthsplat_selected_output import (
        DepthSplatPackedGaussianAttributes,
        depthsplat_attribute_binding_sha256,
        subset_depthsplat_sparse_raw_packet,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(features, z_depths, opacity_logit=100.0)
    route = resolve_depthsplat_compact_final_route(plan, preflight)
    source_packet, source_packed = _selected_packet_and_packed(
        route.selected_output_mask,
        z_depths,
        routing_features=features,
        opacity_logit=100.0,
    )
    final_packet = subset_depthsplat_sparse_raw_packet(
        source_packet, route.selected_output_mask
    )
    attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=source_packed.dense_slots,
        means=source_packed.means,
        covariances=source_packed.covariances,
        harmonics=source_packed.harmonics,
        opacities=source_packed.opacities,
    )
    forged_trace = {
        **final_packet.source_trace,
        "native_adapter_attribute_binding_sha256": attribute_binding,
        "native_full_adapter_attribute_execution_sha256": "e" * 64,
        "native_full_adapter_attribute_passthrough_mask_sha256": _mask_sha256(
            torch.zeros_like(route.full_passthrough_mask)
        ),
        "native_full_adapter_attribute_binding_sha256": attribute_binding,
        "native_full_adapter_attribute_passthrough_count": 0,
    }
    forged = DepthSplatPackedGaussianAttributes(
        dense_slots=source_packed.dense_slots.clone(),
        means=source_packed.means.clone(),
        covariances=source_packed.covariances.clone(),
        harmonics=source_packed.harmonics.clone(),
        opacities=source_packed.opacities.clone(),
        source_trace=forged_trace,
        source_trace_sha256=canonical_json_sha256(forged_trace),
        attribute_binding_sha256=attribute_binding,
    )

    with pytest.raises(ValueError, match="native replay route"):
        apply_depthsplat_compact_l0_l1_materialization(forged, preflight, route)


def test_subset_excludes_l0_prefetch_rows_and_rejects_unreplayed_extension():
    from saes.depthsplat_selected_output import subset_depthsplat_sparse_raw_packet

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, packet, _packed, preflight = _preflight(features, z_depths)
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    route = resolve_depthsplat_compact_final_route(plan, preflight)
    final_packet = subset_depthsplat_sparse_raw_packet(packet, route.selected_output_mask)
    assert final_packet.dense_slots.numel() == 4
    assert final_packet.source_trace["packet_selection_kind"] == "depthsplat-final-selected-output-mask-v1"

    extension = torch.ones_like(route.selected_output_mask)
    with pytest.raises(ValueError, match="unreplayed descriptor"):
        subset_depthsplat_sparse_raw_packet(packet, extension)


def test_route_rejects_a_mutated_preflight_tile_decision():
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(features, z_depths)
    preflight.tile_trace[0]["accepted"] = False

    with pytest.raises(ValueError, match="tile decision diverged"):
        resolve_depthsplat_compact_final_route(plan, preflight)


def test_literal_t4_materialization_ignores_poisoned_omitted_depths_and_source_grid():
    features = torch.full((1, 1, 3, 4, 4), 2.0)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    poisoned_depths = z_depths.clone()
    for row in range(4):
        for column in range(4):
            if (row, column) not in {(0, 0), (0, 3), (3, 0), (3, 3)}:
                poisoned_depths[0, 0, row, column] = 100.0 + row * 4 + column

    def unexpected_source_grid(*_args, **_kwargs):
        raise AssertionError("literal selected-probe materialization must not open source grid")

    plan_a, _packet_a, _packed_a, preflight_a = _preflight(
        features,
        z_depths,
        literal_paper_t4=True,
        preflight_source_sample_image_grid=unexpected_source_grid,
    )
    plan_b, _packet_b, _packed_b, preflight_b = _preflight(
        features,
        poisoned_depths,
        literal_paper_t4=True,
        preflight_source_sample_image_grid=unexpected_source_grid,
    )

    assert plan_a.tile_trace[0]["pre_guard_route"] == "L0"
    assert plan_b.tile_trace[0]["pre_guard_route"] == "L0"
    assert plan_a.tile_trace[0]["depth_uniform"] is None
    assert plan_b.tile_trace[0]["depth_uniform"] is None
    torch.testing.assert_close(preflight_a.update_dense_slots, preflight_b.update_dense_slots)
    torch.testing.assert_close(preflight_a.means, preflight_b.means)
    torch.testing.assert_close(preflight_a.covariances, preflight_b.covariances)
    torch.testing.assert_close(preflight_a.harmonics, preflight_b.harmonics)
    torch.testing.assert_close(preflight_a.opacities, preflight_b.opacities)
    for preflight in (preflight_a, preflight_b):
        assert preflight.events["formal_paper_selected_probe_only"] is True
        assert preflight.events["source_pixel_grid_sha256"] is None
        assert preflight.events["omitted_routing_z_depth_reads"] == 0
        assert preflight.events["assignment_feature_semantics"] == "raw-bilinear-s1-v1"
        assert preflight.events["initial_binding"]["assignment_feature_semantics"] == "raw-bilinear-s1-v1"
        assert preflight.events["maximum_coverage_covariance_scale"] == pytest.approx(1.0)
        assert preflight.events["coverage_max_moment_covariance_scale"] <= 1.0 + 1e-6


def test_literal_t4_accepts_finite_psd_moment_merges_without_support_containment():
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_LITERAL_PAPER_T4_MOMENT_CERTIFICATE,
        resolve_depthsplat_compact_final_route,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, literal_paper_t4=True
    )

    # The selected-only spatial virtual field is finite PSD but its virtual
    # ellipsoids do not satisfy the old, non-paper 2-sigma containment rule.
    # Literal Section 3 only asks for probe-constrained moment matching.
    assert plan.tile_trace[0]["pre_guard_route"] == "L0"
    assert preflight.events["accepted_tiles"] == 1
    assert preflight.events["promoted_full_tiles"] == 0
    assert preflight.tile_trace[0]["reason"] == "accepted"
    assert preflight.update_dense_slots.numel() == 4
    assert preflight.events["coverage_certificate"] == (
        DEPTHSPLAT_LITERAL_PAPER_T4_MOMENT_CERTIFICATE
    )
    assert preflight.events["coverage_max_containment_lhs_after_scale"] is None
    assert preflight.events["literal_finite_psd_moment_merge"] is True
    assert preflight.events["literal_support_containment_guard"] is False
    certificate = preflight.events["coverage_certificate_payload"]
    assert certificate["finite_psd_moment_merge"] is True
    assert certificate["fixed_moment_covariance_scale"] == pytest.approx(1.0)
    assert certificate["support_containment_guard"] is False
    assert all(
        row["moment_covariance_scale"] == pytest.approx(1.0)
        and row["support_containment_guard"] is False
        for row in certificate["per_update"]
    )
    assert torch.linalg.eigvalsh(preflight.covariances).min() >= -1e-6

    route = resolve_depthsplat_compact_final_route(plan, preflight)
    assert route.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert route.events["coverage_certificate_geometry"] == (
        "gaussian-parameter-space-first-second-moment-v1"
    )


def test_literal_t4_invalid_moment_merge_promotes_the_entire_tile_full(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    def invalid_moment(*_args, **_kwargs):
        raise ValueError("DepthSplat formal paper moment merge is non-finite")

    monkeypatch.setattr(
        materializer, "_literal_finite_psd_moment_merge", invalid_moment
    )
    _plan_value, _packet, _packed, preflight = _preflight(
        features, z_depths, literal_paper_t4=True
    )

    assert preflight.events["accepted_tiles"] == 0
    assert preflight.events["promoted_full_tiles"] == 1
    assert preflight.tile_trace[0]["reason"] == (
        "DepthSplat formal paper moment merge is non-finite"
    )
    assert int(preflight.promote_full_mask.sum()) == 16


def test_literal_t4_rejects_covariance_expansion_before_materialization():
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    with pytest.raises(ValueError, match="forbids covariance expansion"):
        _preflight(
            features,
            z_depths,
            literal_paper_t4=True,
            maximum_coverage_covariance_scale=1.01,
        )


def test_literal_t4_frozen_guard_uses_maximum_held_out_risk_not_q75(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan = _plan(features, z_depths, literal_paper_t4=True)
    guard = _authenticated_literal_frozen_guard(plan, threshold_value=30.0)

    def fixed_loo(**_kwargs):
        return {
            "checked": True,
            "scorable": True,
            "status": "scored",
            "certificate": materializer.DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_CERTIFICATE,
            "policy": materializer.DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_POLICY,
            "risk_metric": materializer.DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_RISK_METRIC,
            "level": "L0",
            "anchor_count": 4,
            "held_out_anchor_count": 4,
            "held_out_anchor_records": [
                {
                    "held_out_local_position": [0, index],
                    "harmonic_relative_error": 0.0,
                    "opacity_logit_relative_error": 0.0,
                    "risk": risk,
                }
                for index, risk in enumerate((0.0, 0.0, 0.0, 100.0))
            ],
            "q75_risk": 25.0,
            "maximum_held_out_risk": 100.0,
            "selected_anchor_native_attribute_label_reads": 4,
            "selected_anchor_native_attribute_endpoint_reads": 0,
            "source_nonprobe_s3_attribute_reads": 0,
            "nonzero_direct_deletion": False,
        }

    monkeypatch.setattr(
        materializer, "depthsplat_selected_anchor_attribute_loo_certificate", fixed_loo
    )
    _plan_value, _packet, _packed, preflight = _preflight(
        features,
        z_depths,
        literal_paper_t4=True,
        collect_selected_anchor_attribute_loo_risk=True,
        selected_anchor_attribute_loo_frozen_guard=guard,
    )

    loo = preflight.tile_trace[0]["selected_anchor_attribute_loo"]
    assert loo["q75_risk"] == pytest.approx(25.0)
    assert loo["maximum_held_out_risk"] == pytest.approx(100.0)
    assert loo["maximum_allowed_risk"] == pytest.approx(30.0)
    assert loo["passed"] is False
    assert loo["action"] == "promote_full"
    assert preflight.events["promoted_full_tiles"] == 1
    assert int(preflight.promote_full_mask.sum()) == 16
    assert preflight.events["selected_anchor_attribute_loo_aggregate"][
        "guard_promoted_full_tile_count"
    ] == 1


def test_literal_t4_loo_endpoint_collection_records_unscorable_full_promotion():
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_ENDPOINT_UNSCORABLE,
        NATIVE_OPACITY_ENDPOINT_FULL_REASON,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    _plan_value, _packet, _packed, preflight = _preflight(
        features,
        z_depths,
        literal_paper_t4=True,
        opacity_logit=100.0,
        collect_selected_anchor_attribute_loo_risk=True,
    )

    loo = preflight.tile_trace[0]["selected_anchor_attribute_loo"]
    assert preflight.tile_trace[0]["reason"] == NATIVE_OPACITY_ENDPOINT_FULL_REASON
    assert loo["checked"] is False
    assert loo["scorable"] is False
    assert loo["status"] == DEPTHSPLAT_SELECTED_ANCHOR_ATTRIBUTE_LOO_ENDPOINT_UNSCORABLE
    assert loo["action"] == "promote_full_unscorable"
    assert loo["maximum_held_out_risk"] is None
    assert preflight.events["promoted_full_tiles"] == 1
    aggregate = preflight.events["selected_anchor_attribute_loo_aggregate"]
    assert aggregate["scorable_tile_count"] == 0
    assert aggregate["unscorable_promoted_full_tile_count"] == 1


def test_scalar_loo_threshold_is_rejected_without_a_frozen_guard():
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    with pytest.raises(ValueError, match="requires a frozen V16 guard"):
        _preflight(
            features,
            z_depths,
            literal_paper_t4=True,
            selected_anchor_attribute_loo_maximum_risk=30.0,
        )


def test_literal_t4_rejects_a_syntactically_valid_forged_mapping_guard():
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan = _plan(features, z_depths, literal_paper_t4=True)

    with pytest.raises(TypeError, match="authenticated V16T4 guard"):
        _preflight(
            features,
            z_depths,
            literal_paper_t4=True,
            selected_anchor_attribute_loo_frozen_guard=_literal_frozen_guard(plan),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "frozen_record_kind",
            "depthsplat-nonzero-l0-l1-acid-disjoint-v16d",
            "formal V16L/T4 LOO guard binding changed",
        ),
        (
            "materialization_profile",
            "depthsplat-development-omitted-z-alpha-union-v1",
            "formal V16L/T4 LOO guard binding changed",
        ),
        (
            "route_plan_contract",
            "saes-incremental-probe-first-plan-v1",
            "formal V16L/T4 LOO guard binding changed",
        ),
        (
            "route_plan_config_sha256",
            "0" * 64,
            "formal V16L/T4 LOO guard binding changed",
        ),
    ],
)
def test_literal_t4_rejects_nonliteral_frozen_guard_projection(field, value, error):
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan = _plan(features, z_depths, literal_paper_t4=True)
    projection = _literal_frozen_guard(plan)
    projection[field] = value
    import saes.depthsplat_literal_t4_acid_calibration as calibration

    guard = calibration._issue_verified_literal_t4_materializer_guard(projection)

    with pytest.raises(ValueError, match=error):
        _preflight(
            features,
            z_depths,
            literal_paper_t4=True,
            selected_anchor_attribute_loo_frozen_guard=guard,
        )
