"""Focused source-contract tests for DepthSplat's nonzero L0/L1 merge path."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

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
    coverage_enriched_t4=False,
    support_basis_t4=False,
    soft_mixture_t4=False,
    soft_mixture_normalized_t4=False,
    soft_mixture_kernel_closure_t4=False,
    feature_threshold=0.1,
    depth_threshold=0.1,
):
    from saes.probe_first_schedule import (
        build_depthsplat_coverage_enriched_t4_probe_first_plan,
        build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan,
        build_depthsplat_soft_mixture_normalized_t4_probe_first_plan,
        build_depthsplat_soft_mixture_t4_probe_first_plan,
        build_depthsplat_support_basis_t4_probe_first_plan,
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
    if coverage_enriched_t4:
        return build_depthsplat_coverage_enriched_t4_probe_first_plan(
            features,
            z_depths,
            height=4,
            width=4,
            feature_threshold=feature_threshold,
            depth_threshold=depth_threshold,
        )
    if support_basis_t4:
        return build_depthsplat_support_basis_t4_probe_first_plan(
            features,
            z_depths,
            height=4,
            width=4,
            feature_threshold=feature_threshold,
            depth_threshold=depth_threshold,
        )
    if soft_mixture_t4:
        return build_depthsplat_soft_mixture_t4_probe_first_plan(
            features,
            z_depths,
            height=4,
            width=4,
            feature_threshold=feature_threshold,
            depth_threshold=depth_threshold,
        )
    if soft_mixture_normalized_t4:
        return build_depthsplat_soft_mixture_normalized_t4_probe_first_plan(
            features,
            z_depths,
            height=4,
            width=4,
            feature_threshold=feature_threshold,
            depth_threshold=depth_threshold,
        )
    if soft_mixture_kernel_closure_t4:
        return build_depthsplat_soft_mixture_kernel_closure_t4_probe_first_plan(
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
    native_full_mask=None,
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

    if native_full_mask is None:
        native_full_mask = torch.zeros_like(mask)
    if (
        native_full_mask.shape != mask.shape
        or native_full_mask.dtype != torch.bool
        or bool((native_full_mask & ~mask).any())
    ):
        raise ValueError("test packet native Full mask is invalid")
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
    selected_full = native_full_mask[
        positions[:, 0], positions[:, 1], positions[:, 2]
    ]
    full_slots = slots[selected_full]
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
        "native_full_passthrough_mask_sha256": _mask_sha256(native_full_mask),
        "native_full_passthrough_positions": int(full_slots.numel()),
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
    full_attribute_binding = depthsplat_attribute_binding_sha256(
        dense_slots=full_slots,
        means=means[selected_full],
        covariances=covariances[selected_full],
        harmonics=harmonics[selected_full],
        opacities=opacities[selected_full],
    )
    packed_trace = {
        **trace,
        "native_adapter_attribute_binding_sha256": attribute_binding,
        "native_full_adapter_attribute_execution_sha256": trace["native_execution_sha256"],
        "native_full_adapter_attribute_passthrough_mask_sha256": trace[
            "native_full_passthrough_mask_sha256"
        ],
        "native_full_adapter_attribute_binding_sha256": full_attribute_binding,
        "native_full_adapter_attribute_passthrough_count": int(full_slots.numel()),
        "selected_native_rgb_adapter_compact_count": count - int(full_slots.numel()),
        "selected_native_rgb_adapter_executed": count > int(full_slots.numel()),
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
    coverage_enriched_t4=False,
    support_basis_t4=False,
    soft_mixture_t4=False,
    soft_mixture_normalized_t4=False,
    soft_mixture_kernel_closure_t4=False,
    feature_threshold=0.1,
    depth_threshold=0.1,
    collect_selected_anchor_attribute_loo_risk=False,
    selected_anchor_attribute_loo_frozen_guard=None,
    selected_anchor_attribute_loo_maximum_risk=None,
    mixture_kernel_closure_frozen_guard=None,
    route_isolation="l0_l1",
    preflight_source_sample_image_grid=None,
    maximum_coverage_covariance_scale=None,
    execution_profile=None,
):
    from depthsplat.src.geometry.projection import get_world_rays, sample_image_grid
    from saes.depthsplat_l0_l1_materializer import preflight_depthsplat_l0_l1_materialization

    plan = _plan(
        features,
        z_depths,
        literal_paper_t4=literal_paper_t4,
        coverage_enriched_t4=coverage_enriched_t4,
        support_basis_t4=support_basis_t4,
        soft_mixture_t4=soft_mixture_t4,
        soft_mixture_normalized_t4=soft_mixture_normalized_t4,
        soft_mixture_kernel_closure_t4=soft_mixture_kernel_closure_t4,
        feature_threshold=feature_threshold,
        depth_threshold=depth_threshold,
    )
    packet, packed = _selected_packet_and_packed(
        plan.selection_mask,
        z_depths,
        routing_features=features,
        native_full_mask=plan.full_mask,
        opacity_logit=opacity_logit,
        harmonic_delta=harmonic_delta,
        covariance_scale=covariance_scale,
    )
    profile = (
        "depthsplat-literal-paper-t4-selected-probe-moment-v1"
        if literal_paper_t4
        else "depthsplat-coverage-enriched-t4-balanced-l1-owner-support-v1"
        if coverage_enriched_t4
        else "depthsplat-support-basis-t4-balanced-l1-cooperative-v1"
        if support_basis_t4
        else "depthsplat-soft-mixture-t4-balanced-l1-source-only-moment-replay-v1"
        if soft_mixture_t4
        else "depthsplat-soft-mixture-normalized-t4-balanced-l1-source-only-moment-replay-v2"
        if soft_mixture_normalized_t4
        else "depthsplat-soft-mixture-kernel-closure-t4-balanced-l1-source-only-v3"
        if soft_mixture_kernel_closure_t4
        else "depthsplat-development-omitted-z-alpha-union-v1"
    )
    if execution_profile is not None:
        profile = execution_profile
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
            (1.0 if profile != "depthsplat-development-omitted-z-alpha-union-v1" else 16.0)
            if maximum_coverage_covariance_scale is None
            else maximum_coverage_covariance_scale
        ),
        execution_profile=profile,
        collect_selected_anchor_attribute_loo_risk=(
            collect_selected_anchor_attribute_loo_risk
        ),
        selected_anchor_attribute_loo_frozen_guard=(
            selected_anchor_attribute_loo_frozen_guard
        ),
        selected_anchor_attribute_loo_maximum_risk=(
            selected_anchor_attribute_loo_maximum_risk
        ),
        mixture_kernel_closure_frozen_guard=(
            mixture_kernel_closure_frozen_guard
        ),
        route_isolation=route_isolation,
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


def test_conditional_transport_does_not_leak_an_anchor_offset_to_other_outputs():
    """Each receiving anchor must own its transported virtual geometry."""

    from depthsplat.src.geometry.projection import get_world_rays, sample_image_grid
    from saes.depthsplat_backend import source_native_depthsplat_coordinates
    from saes.depthsplat_l0_l1_materializer import (
        _build_tile_updates,
        depthsplat_z_depth_world_means,
    )

    features = torch.zeros((1, 1, 2, 4, 4), dtype=torch.float32)
    z_depths = torch.full((1, 1, 4, 4), 2.0, dtype=torch.float32)
    plan = _plan(
        features,
        z_depths,
        feature_threshold=0.1,
        depth_threshold=0.1,
    )
    assert plan.tile_trace[0]["pre_guard_route"] == "L0"
    packet, packed = _selected_packet_and_packed(
        plan.selection_mask,
        z_depths,
        routing_features=features,
        native_full_mask=plan.full_mask,
    )
    slot_to_index = {
        int(slot): index for index, slot in enumerate(packet.dense_slots.tolist())
    }
    source_grid, _ = sample_image_grid((4, 4), packet.coordinates.device)

    def build(source_packet, source_packed=packed):
        return _build_tile_updates(
            packet=source_packet,
            packed=source_packed,
            slot_to_index=slot_to_index,
            feature_map=features[0, 0],
            z_depth_map=z_depths[0, 0],
            record=plan.tile_trace[0],
            level="L0",
            semantics="engineering-lightweight-12-balanced-v1",
            view=0,
            tile_y=0,
            tile_x=0,
            height=4,
            width=4,
            feature_statistic=plan.events["feature_statistic"],
            source_grid=source_grid,
            source_get_world_rays=get_world_rays,
            maximum_feature_relative_residual=1.0,
            maximum_coverage_covariance_scale=16.0,
        )[0]

    baseline = {slot: mean for slot, mean, *_ in build(packet)}
    perturbed_raw = packet.raw_head_descriptors.clone()
    # Slot 3 is the top-right L0 corner. Keep the offset inside the native
    # half-pixel envelope while changing only its conditional transport.
    perturbed_raw[slot_to_index[3], 1] += 0.1
    perturbed_packet = replace(
        packet,
        raw_head_descriptors=perturbed_raw,
        coordinates=source_native_depthsplat_coordinates(
            perturbed_raw,
            plan.selection_mask,
            sample_image_grid=sample_image_grid,
        ),
    )
    perturbed_means = packed.means.clone()
    index = slot_to_index[3]
    perturbed_means[index] = depthsplat_z_depth_world_means(
        perturbed_packet.coordinates[index : index + 1],
        perturbed_packet.extrinsics[index : index + 1],
        perturbed_packet.intrinsics[index : index + 1],
        perturbed_packet.depths[index : index + 1],
        source_get_world_rays=get_world_rays,
    )[0]
    perturbed_packed = replace(packed, means=perturbed_means)
    perturbed = {
        slot: mean for slot, mean, *_ in build(perturbed_packet, perturbed_packed)
    }

    assert torch.allclose(baseline[0], perturbed[0], rtol=0.0, atol=1.0e-7)
    assert not torch.allclose(baseline[3], perturbed[3], rtol=0.0, atol=1.0e-7)

def test_fixed_scale_conditional_transport_isolated_per_receiving_anchor():
    """The fixed-scale SAES path must retain conditional offset ownership."""

    from depthsplat.src.geometry.projection import get_world_rays, sample_image_grid
    from saes.depthsplat_backend import source_native_depthsplat_coordinates
    from saes.depthsplat_l0_l1_materializer import (
        _build_fixed_scale_selected_only_tile_updates,
        depthsplat_z_depth_world_means,
    )

    features = torch.zeros((1, 1, 2, 4, 4), dtype=torch.float32)
    z_depths = torch.full((1, 1, 4, 4), 2.0, dtype=torch.float32)
    plan = _plan(
        features,
        z_depths,
        soft_mixture_normalized_t4=True,
        feature_threshold=0.1,
        depth_threshold=0.1,
    )
    assert plan.tile_trace[0]["pre_guard_route"] == "L0"
    packet, packed = _selected_packet_and_packed(
        plan.selection_mask,
        z_depths,
        routing_features=features,
        native_full_mask=plan.full_mask,
    )
    slot_to_index = {
        int(slot): index for index, slot in enumerate(packet.dense_slots.tolist())
    }

    def build(source_packet, source_packed=packed):
        return _build_fixed_scale_selected_only_tile_updates(
            packet=source_packet,
            packed=source_packed,
            slot_to_index=slot_to_index,
            raw_feature_map=features[0, 0],
            record=plan.tile_trace[0],
            level="L0",
            semantics="engineering-lightweight-12-balanced-v1",
            view=0,
            tile_y=0,
            tile_x=0,
            height=4,
            width=4,
            feature_statistic=plan.events["feature_statistic"],
            assignment_feature_semantics="unit-normalized-bilinear-s1-v1",
            coverage_enriched=False,
            conditional_anchor_transport=True,
            routing_z_depth_map=z_depths[0, 0],
            source_get_world_rays=get_world_rays,
        )[0]

    baseline_updates = {slot: values for slot, *values in build(packet)}
    baseline = {slot: values[0] for slot, values in baseline_updates.items()}
    perturbed_raw = packet.raw_head_descriptors.clone()
    perturbed_raw[slot_to_index[3], 1] += 0.1
    perturbed_packet = replace(
        packet,
        raw_head_descriptors=perturbed_raw,
        coordinates=source_native_depthsplat_coordinates(
            perturbed_raw,
            plan.selection_mask,
            sample_image_grid=sample_image_grid,
        ),
    )
    perturbed_means = packed.means.clone()
    index = slot_to_index[3]
    perturbed_means[index] = depthsplat_z_depth_world_means(
        perturbed_packet.coordinates[index : index + 1],
        perturbed_packet.extrinsics[index : index + 1],
        perturbed_packet.intrinsics[index : index + 1],
        perturbed_packet.depths[index : index + 1],
        source_get_world_rays=get_world_rays,
    )[0]
    perturbed = {
        slot: mean
        for slot, mean, *_ in build(
            perturbed_packet, replace(packed, means=perturbed_means)
        )
    }

    assert torch.allclose(baseline[0], perturbed[0], rtol=0.0, atol=1.0e-7)
    assert not torch.allclose(baseline[3], perturbed[3], rtol=0.0, atol=1.0e-7)

    perturbed_harmonics = packed.harmonics.clone()
    perturbed_harmonics[slot_to_index[3]] += 0.1
    perturbed_opacities = packed.opacities.clone()
    perturbed_opacities[slot_to_index[3]] += 0.05
    attribute_perturbed = {
        slot: values
        for slot, *values in build(
            packet,
            replace(
                packed,
                harmonics=perturbed_harmonics,
                opacities=perturbed_opacities,
            ),
        )
    }

    assert torch.allclose(
        baseline_updates[0][2], attribute_perturbed[0][2], rtol=0.0, atol=1.0e-7
    )
    assert torch.allclose(
        baseline_updates[0][3], attribute_perturbed[0][3], rtol=0.0, atol=1.0e-7
    )
    assert not torch.allclose(
        baseline_updates[3][2], attribute_perturbed[3][2], rtol=0.0, atol=1.0e-7
    )
    assert not torch.allclose(
        baseline_updates[3][3], attribute_perturbed[3][3], rtol=0.0, atol=1.0e-7
    )


def test_direct_conditional_profile_materializes_normalized_l0_without_kernel_guard():
    """The real direct route keeps normalized L0 without certificate promotion."""

    from depthsplat.src.geometry.projection import get_world_rays, sample_image_grid
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_COVERAGE_CERTIFICATE,
        DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
        preflight_depthsplat_l0_l1_materialization,
        resolve_depthsplat_compact_final_route,
    )

    features = torch.zeros((1, 1, 2, 4, 4), dtype=torch.float32)
    z_depths = torch.full((1, 1, 4, 4), 2.0, dtype=torch.float32)
    plan = _plan(
        features,
        z_depths,
        soft_mixture_normalized_t4=True,
        feature_threshold=0.1,
        depth_threshold=0.1,
    )
    packet, packed = _selected_packet_and_packed(
        plan.selection_mask,
        z_depths,
        routing_features=features,
        native_full_mask=plan.full_mask,
    )
    preflight = preflight_depthsplat_l0_l1_materialization(
        packet,
        packed,
        plan,
        features,
        z_depths,
        source_sample_image_grid=sample_image_grid,
        source_get_world_rays=get_world_rays,
        maximum_coverage_covariance_scale=16.0,
        execution_profile=DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE,
    )
    final_route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert preflight.events["execution_profile"] == (
        DEPTHSPLAT_DIRECT_CONDITIONAL_T4_MATERIALIZATION_PROFILE
    )
    assert preflight.events["soft_mixture_certificate_aggregate"] is None
    assert preflight.events["mixture_kernel_closure_aggregate"] is None
    assert preflight.events["coverage_certificate"] == DEPTHSPLAT_COVERAGE_CERTIFICATE
    assert preflight.events["maximum_coverage_covariance_scale"] == pytest.approx(16.0)
    assert preflight.events["coverage_max_containment_lhs_after_scale"] <= 2.0 + 1e-5
    assert 1.0 <= preflight.events["coverage_max_moment_covariance_scale"] <= 16.0
    assert preflight.tile_trace[0]["assignment_transport"] == (
        "bilateral-soft-moment-v1"
    )
    assert final_route.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert preflight.update_dense_slots.numel() == 4


def _kernel_risk_frozen_guard(plan, *, threshold_value=0.1):
    from saes.depthsplat_mixture_kernel_acid_calibration import (
        KERNEL_RISK_GUARD_SCHEMA,
        KERNEL_RISK_KIND,
        KERNEL_RISK_METRIC,
        KERNEL_RISK_THRESHOLD_RULE,
    )
    from saes.depthsplat_mixture_kernel_guard import KIND, POLICY, SCHEMA_VERSION
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
    )

    return {
        "schema_version": KERNEL_RISK_GUARD_SCHEMA,
        "frozen_record_kind": KERNEL_RISK_KIND,
        "frozen_record_sha256": "a" * 64,
        "threshold_value": threshold_value,
        "threshold_rule": KERNEL_RISK_THRESHOLD_RULE,
        "risk_metric": KERNEL_RISK_METRIC,
        "materialization_profile": (
            DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE
        ),
        "route_plan_contract": plan.events["contract_version"],
        "route_plan_config_sha256": plan.events[
            "soft_mixture_kernel_closure_t4_route_config_sha256"
        ],
        "kernel_closure_schema_version": SCHEMA_VERSION,
        "kernel_closure_kind": KIND,
        "kernel_closure_policy": POLICY,
        "acid_binding_sha256": "b" * 64,
        "application_sha256": "c" * 64,
    }


def _authenticated_kernel_risk_frozen_guard(plan, *, threshold_value=0.1):
    import saes.depthsplat_mixture_kernel_acid_calibration as calibration

    return calibration._issue_verified_mixture_kernel_risk_guard(
        _kernel_risk_frozen_guard(plan, threshold_value=threshold_value)
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


def _owner_support_result(*, owner_count, virtual_owner_indices, passed):
    from saes.depthsplat_l0_l1_materializer import _tensor_sha256
    from saes.depthsplat_owner_coverage import (
        AUDIT_KIND,
        AUDIT_SCHEMA_VERSION,
        OWNER_ASSIGNMENT_POLICY,
    )

    owner_assignment_counts = [
        int((virtual_owner_indices == owner_index).sum().item())
        for owner_index in range(owner_count)
    ]
    owners = []
    for owner_index in range(owner_count):
        assigned = owner_assignment_counts[owner_index]
        owners.append(
            {
                "owner_index": owner_index,
                "source_anchor_count": 1,
                "source_anchor_support_included": True,
                "source_anchor_support_passed": passed,
                "assigned_virtual_count": assigned,
                "dense_descriptor_count": assigned + 1,
                "active_virtual_count": assigned,
                "active_source_anchor_count": 1,
                "active_dense_descriptor_count": assigned + 1,
                "checked": True,
                "audit_valid": True,
                "passed": passed,
                "reason": None if passed else "uncontained-assigned-virtuals",
                "coverage": {
                    "valid": True,
                    "dense_descriptor_count": assigned + 1,
                    "active_dense_descriptor_count": assigned + 1,
                    "hole_count": 0 if passed else 1,
                },
            }
        )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "sigma": 2.0,
        "passed": passed,
        "source_only": {
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "source_camera_only": True,
            "input_covariances_mutated": False,
            "owner_source_anchor_support_included": True,
        },
        "ownership": {
            "policy": OWNER_ASSIGNMENT_POLICY,
            "virtual_owner_indices_sha256": _tensor_sha256(virtual_owner_indices),
            "owner_assignment_counts": owner_assignment_counts,
        },
        "summary": {
            "input_valid": True,
            "owner_count": owner_count,
            "source_anchor_count": owner_count,
            "virtual_primitive_count": int(virtual_owner_indices.numel()),
            "assigned_owner_count": sum(count > 0 for count in owner_assignment_counts),
            "empty_owner_count": sum(count == 0 for count in owner_assignment_counts),
            "checked_owner_count": owner_count,
            "invalid_owner_assignment_count": 0,
            "failed_owner_count": 0 if passed else owner_count,
            "all_active_virtual_2sigma_supports_contained": passed,
            "all_active_owner_anchor_and_virtual_2sigma_supports_contained": passed,
            "hole_count": 0 if passed else owner_count,
        },
        "owners": owners,
    }


def _constructed_support_basis_coverage(*, candidate_means, hole_count=0, **_kwargs):
    """Supply only the geometric verdict for a route-binding contract test."""

    from saes.projected_domain_coverage_audit import (
        AUDIT_SCHEMA_VERSION,
        FIXED_SIGMA,
        ProjectedDomainCoverage,
    )

    candidate_count = int(candidate_means.shape[0])
    contained = 0.0 if hole_count else 1.0
    return ProjectedDomainCoverage(
        valid=True,
        reason=None,
        schema_version=AUDIT_SCHEMA_VERSION,
        sigma=FIXED_SIGMA,
        dense_descriptor_count=1,
        active_dense_descriptor_count=1,
        candidate_descriptor_count=candidate_count,
        active_candidate_descriptor_count=candidate_count,
        dense_optical_mass=1.0,
        contained_optical_mass=contained,
        mass_weighted_recall=contained,
        count_recall=contained,
        hole_count=hole_count,
        dense_offscreen_count=0,
        candidate_offscreen_count=0,
    )


def test_coverage_enriched_l0_pass_discards_balanced_l1_prefetch(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    monkeypatch.setattr(
        materializer,
        "audit_depthsplat_owner_coverage",
        lambda **kwargs: _owner_support_result(
            owner_count=int(kwargs["merged_means"].shape[0]),
            virtual_owner_indices=kwargs["virtual_owner_indices"],
            passed=True,
        ),
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, coverage_enriched_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert plan.tile_trace[0]["pre_guard_route"] == "L0"
    assert int(plan.selection_mask.sum()) == 12
    assert preflight.tile_trace[0]["accepted_level"] == "L0"
    assert preflight.events["coverage_enriched_l0_to_l1_tile_count"] == 0
    assert route.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert int(route.selected_output_mask.sum()) == 4
    assert int(route.raw_head_request_mask.sum()) == 12
    assert route.events["producer_only_prefetch_descriptor_count"] == 8
    assert preflight.events["skipped_s3_attributes_accessed"] is False
    assert preflight.events["target_rgb_accessed"] is False


def test_coverage_enriched_l0_failure_retries_prefetched_balanced_l1(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    monkeypatch.setattr(
        materializer,
        "audit_depthsplat_owner_coverage",
        lambda **kwargs: _owner_support_result(
            owner_count=int(kwargs["merged_means"].shape[0]),
            virtual_owner_indices=kwargs["virtual_owner_indices"],
            passed=int(kwargs["merged_means"].shape[0]) == 12,
        ),
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, coverage_enriched_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert preflight.tile_trace[0]["planned_route"] == "L0"
    assert preflight.tile_trace[0]["accepted_level"] == "L1"
    assert preflight.tile_trace[0]["l1_enrichment_prefetched"] is True
    assert preflight.events["coverage_enriched_l0_to_l1_tile_count"] == 1
    assert preflight.update_dense_slots.numel() == 12
    assert route.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    assert int(route.selected_output_mask.sum()) == 12
    assert int(route.raw_head_request_mask.sum()) == 12
    assert int(route.additional_full_mask.sum()) == 0
    assert route.events["producer_only_prefetch_descriptor_count"] == 0


def test_coverage_enriched_l1_failure_promotes_the_entire_tile_full(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    monkeypatch.setattr(
        materializer,
        "audit_depthsplat_owner_coverage",
        lambda **kwargs: _owner_support_result(
            owner_count=int(kwargs["merged_means"].shape[0]),
            virtual_owner_indices=kwargs["virtual_owner_indices"],
            passed=False,
        ),
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, coverage_enriched_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert preflight.tile_trace[0]["accepted"] is False
    assert preflight.tile_trace[0]["accepted_level"] is None
    assert int(preflight.promote_full_mask.sum()) == 16
    assert preflight.update_dense_slots.numel() == 0
    assert route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert int(route.selected_output_mask.sum()) == 16
    assert int(route.additional_full_mask.sum()) == 4


def test_coverage_enriched_rejects_incomplete_owner_assignment_evidence(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    def incomplete_owner_audit(**kwargs):
        result = _owner_support_result(
            owner_count=int(kwargs["merged_means"].shape[0]),
            virtual_owner_indices=kwargs["virtual_owner_indices"],
            passed=True,
        )
        result["ownership"]["owner_assignment_counts"][0] += 1
        return result

    monkeypatch.setattr(
        materializer, "audit_depthsplat_owner_coverage", incomplete_owner_audit
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, coverage_enriched_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert preflight.tile_trace[0]["accepted"] is False
    assert "owner evidence binding changed" in preflight.tile_trace[0]["reason"]
    assert route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}


def test_coverage_enriched_route_rebuilds_owner_certificate_from_live_trace(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_backend import canonical_json_sha256
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    monkeypatch.setattr(
        materializer,
        "audit_depthsplat_owner_coverage",
        lambda **kwargs: _owner_support_result(
            owner_count=int(kwargs["merged_means"].shape[0]),
            virtual_owner_indices=kwargs["virtual_owner_indices"],
            passed=True,
        ),
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, coverage_enriched_t4=True
    )
    payload = preflight.events["coverage_certificate_payload"]
    payload["per_update"][0]["owner_index"] = 12345
    preflight.events["coverage_certificate_sha256"] = canonical_json_sha256(payload)

    with pytest.raises(ValueError, match="certificate binding changed"):
        resolve_depthsplat_compact_final_route(plan, preflight)


def test_coverage_enriched_route_rejects_post_preflight_plan_contract_drift(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    monkeypatch.setattr(
        materializer,
        "audit_depthsplat_owner_coverage",
        lambda **kwargs: _owner_support_result(
            owner_count=int(kwargs["merged_means"].shape[0]),
            virtual_owner_indices=kwargs["virtual_owner_indices"],
            passed=True,
        ),
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, coverage_enriched_t4=True
    )
    plan.events["contract_version"] = "saes-incremental-probe-first-plan-v1"

    with pytest.raises(ValueError, match="execution profile does not match plan contract"):
        resolve_depthsplat_compact_final_route(plan, preflight)


def test_coverage_enriched_plan_cannot_use_development_materialization_profile():
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    with pytest.raises(ValueError, match="execution profile does not match plan contract"):
        _preflight(
            features,
            z_depths,
            coverage_enriched_t4=True,
            execution_profile=DEPTHSPLAT_DEVELOPMENT_MATERIALIZATION_PROFILE,
        )


def test_support_basis_profile_has_its_own_plan_and_fixed_scale_certificate():
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE,
        resolve_depthsplat_compact_final_route,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, support_basis_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert plan.events["contract_version"] == (
        "saes-depthsplat-support-basis-t4-probe-first-plan-v1"
    )
    assert preflight.events["execution_profile"] == (
        DEPTHSPLAT_SUPPORT_BASIS_T4_MATERIALIZATION_PROFILE
    )
    assert preflight.events["coverage_certificate"] == (
        DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE
    )
    assert preflight.events["maximum_coverage_covariance_scale"] == pytest.approx(1.0)
    assert preflight.events["support_basis_guard"] is True
    assert preflight.events["coverage_enriched_owner_support_guard"] is None
    assert route.events["coverage_certificate_geometry"] == (
        "source-camera-composed-soft-ledger-same-tile-projected-2sigma-v1"
    )
    assert route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}


def test_support_basis_l0_compact_route_binds_composed_certificate(monkeypatch):
    import saes.depthsplat_support_basis_coverage as support_coverage
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE,
        resolve_depthsplat_compact_final_route,
    )

    monkeypatch.setattr(
        support_coverage,
        "audit_projected_dense_domain_coverage",
        _constructed_support_basis_coverage,
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, support_basis_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert preflight.tile_trace[0]["accepted_level"] == "L0"
    assert preflight.update_dense_slots.numel() == 4
    assert route.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert int(route.selected_output_mask.sum()) == 4
    assert int(route.raw_head_request_mask.sum()) == 12
    certificate = preflight.events["coverage_certificate_payload"]
    assert certificate["schema"] == DEPTHSPLAT_SUPPORT_BASIS_T4_MOMENT_CERTIFICATE
    assert len(certificate["per_update"]) == 4
    assert all(row["support_basis_passed"] is True for row in certificate["per_update"])


def test_support_basis_l0_failure_retries_prefetched_l1_compact_route(monkeypatch):
    import saes.depthsplat_support_basis_coverage as support_coverage
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    def l1_only_coverage(*, candidate_means, **kwargs):
        return _constructed_support_basis_coverage(
            candidate_means=candidate_means,
            hole_count=0 if int(candidate_means.shape[0]) == 12 else 1,
            **kwargs,
        )

    monkeypatch.setattr(
        support_coverage,
        "audit_projected_dense_domain_coverage",
        l1_only_coverage,
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, support_basis_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert preflight.tile_trace[0]["planned_route"] == "L0"
    assert preflight.tile_trace[0]["accepted_level"] == "L1"
    assert preflight.tile_trace[0]["l1_enrichment_prefetched"] is True
    assert preflight.events["support_basis_l0_to_l1_tile_count"] == 1
    assert preflight.update_dense_slots.numel() == 12
    assert route.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    assert int(route.selected_output_mask.sum()) == 12


def test_support_basis_route_rejects_rehashed_internal_ledger_tamper(monkeypatch):
    import saes.depthsplat_support_basis_coverage as support_coverage
    from saes.depthsplat_backend import canonical_json_sha256
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    monkeypatch.setattr(
        support_coverage,
        "audit_projected_dense_domain_coverage",
        _constructed_support_basis_coverage,
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, support_basis_t4=True
    )
    payload = preflight.events["coverage_certificate_payload"]
    payload["per_update"][0]["support_basis_sha256"] = "0" * 64
    preflight.events["coverage_certificate_sha256"] = canonical_json_sha256(payload)

    with pytest.raises(ValueError, match="certificate binding changed"):
        resolve_depthsplat_compact_final_route(plan, preflight)


def test_support_basis_plan_cannot_run_under_the_owner_profile():
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    with pytest.raises(ValueError, match="execution profile does not match plan contract"):
        _preflight(
            features,
            z_depths,
            support_basis_t4=True,
            execution_profile=DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
        )


def test_soft_mixture_normalized_profile_rebuilds_its_own_normalized_plan(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
        resolve_depthsplat_compact_final_route,
    )

    raw_builder_calls = []
    normalized_builder_calls = []
    original_normalized_builder = (
        materializer.build_depthsplat_soft_mixture_normalized_t4_probe_first_plan
    )

    def unexpected_raw_builder(*args, **kwargs):
        raw_builder_calls.append((args, kwargs))
        raise AssertionError("normalized-v2 profile must not rebuild with raw-v1")

    def tracking_normalized_builder(*args, **kwargs):
        normalized_builder_calls.append((args, kwargs))
        return original_normalized_builder(*args, **kwargs)

    monkeypatch.setattr(
        materializer,
        "build_depthsplat_soft_mixture_t4_probe_first_plan",
        unexpected_raw_builder,
    )
    monkeypatch.setattr(
        materializer,
        "build_depthsplat_soft_mixture_normalized_t4_probe_first_plan",
        tracking_normalized_builder,
    )
    features = torch.zeros(1, 1, 2, 4, 4)
    for position, vector in {
        (0, 0): (1.0, 0.0),
        (0, 3): (3.0, 0.0),
        (3, 0): (1.0, 0.0),
        (3, 3): (3.0, 0.0),
    }.items():
        features[0, 0, :, position[0], position[1]] = torch.tensor(vector)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    plan, _packet, _packed, preflight = _preflight(
        features,
        z_depths,
        soft_mixture_normalized_t4=True,
        feature_threshold=0.2,
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert raw_builder_calls == []
    assert len(normalized_builder_calls) == 1
    assert plan.tile_trace[0]["pre_guard_route"] == "L0"
    assert preflight.events["execution_profile"] == (
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE
    )
    assert preflight.events["assignment_feature_semantics"] == (
        "unit-normalized-bilinear-s1-v1"
    )
    assert preflight.tile_trace[0]["assignment_feature_semantics"] == (
        "unit-normalized-bilinear-s1-v1"
    )
    assert preflight.events["soft_mixture_guard"] is True
    assert route.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert route.events["assignment_feature_semantics"] == (
        "unit-normalized-bilinear-s1-v1"
    )
    assert route.events["assignment_feature_map_sha256"] == (
        preflight.events["assignment_feature_map_sha256"]
    )


def test_soft_mixture_normalized_plan_cannot_bind_the_raw_v1_profile():
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    with pytest.raises(ValueError, match="execution profile does not match plan contract"):
        _preflight(
            features,
            z_depths,
            soft_mixture_normalized_t4=True,
            execution_profile=DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
        )
    with pytest.raises(ValueError, match="execution profile does not match plan contract"):
        _preflight(
            features,
            z_depths,
            soft_mixture_t4=True,
            execution_profile=(
                DEPTHSPLAT_SOFT_MIXTURE_NORMALIZED_T4_MATERIALIZATION_PROFILE
            ),
        )


def test_soft_mixture_normalized_native_full_tile_stays_passthrough():
    from saes.depthsplat_l0_l1_materializer import (
        resolve_depthsplat_compact_final_route,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    features[0, 0, :, 0, 0] = torch.tensor((1.0, 0.0, 0.0))
    features[0, 0, :, 0, 3] = torch.tensor((0.0, 1.0, 0.0))
    features[0, 0, :, 3, 0] = torch.tensor((1.0, 0.0, 0.0))
    features[0, 0, :, 3, 3] = torch.tensor((0.0, 1.0, 0.0))
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    z_depths[0, 0, 0, 0] = 1.0
    z_depths[0, 0, 0, 3] = 3.0
    z_depths[0, 0, 3, 0] = 1.0
    z_depths[0, 0, 3, 3] = 3.0

    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, soft_mixture_normalized_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert plan.tile_trace[0]["pre_guard_route"] == "Full"
    assert preflight.tile_trace[0]["accepted_level"] == "Full"
    assert route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert route.tile_trace[0]["final_route"] == "Full"
    assert bool(route.full_passthrough_mask.all())


def test_soft_mixture_normalized_assignment_binding_cannot_drift():
    from saes.depthsplat_l0_l1_materializer import (
        resolve_depthsplat_compact_final_route,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, soft_mixture_normalized_t4=True
    )
    preflight.events["initial_binding"]["assignment_feature_semantics"] = (
        "raw-bilinear-s1-v1"
    )

    with pytest.raises(ValueError, match="preflight binding changed"):
        resolve_depthsplat_compact_final_route(plan, preflight)


def test_soft_mixture_profile_replays_s_r_without_projected_support_routing(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE,
        DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
        resolve_depthsplat_compact_final_route,
    )

    def unexpected_projected_guard(**_kwargs):
        raise AssertionError("soft-mixture profile must not invoke projected coverage")

    monkeypatch.setattr(
        materializer, "audit_depthsplat_owner_coverage", unexpected_projected_guard
    )
    monkeypatch.setattr(
        materializer, "audit_depthsplat_tile_support_basis", unexpected_projected_guard
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, soft_mixture_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert plan.events["contract_version"] == (
        "saes-depthsplat-soft-mixture-t4-probe-first-plan-v1"
    )
    assert preflight.events["execution_profile"] == (
        DEPTHSPLAT_SOFT_MIXTURE_T4_MATERIALIZATION_PROFILE
    )
    assert preflight.events["coverage_certificate"] == (
        DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE
    )
    assert preflight.events["soft_mixture_guard"] is True
    assert preflight.events["soft_mixture_projected_domain_guard_used"] is False
    assert preflight.events["coverage_max_containment_lhs_after_scale"] is None
    certificate = preflight.tile_trace[0]["soft_mixture_certificate"]
    assert certificate["passed"] is True
    assert certificate["source_only"]["projected_domain_guard_used"] is False
    assert certificate["source_only"]["boolean_owner_assignment_used"] is False
    aggregate = preflight.events["soft_mixture_certificate_aggregate"]
    assert aggregate["tile_certificate_attempt_count"] == 1
    assert aggregate["passed_tile_certificate_count"] == 1
    assert aggregate["failed_tile_certificate_count"] == 0
    assert route.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}
    assert route.events["coverage_certificate_geometry"] == (
        "source-only-spatial-s-bilateral-r-first-second-moment-replay-v1"
    )


def test_soft_mixture_l0_replay_failure_retries_prefetched_l1(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    original_certificate = materializer.certify_depthsplat_tile_soft_mixture

    def l1_only_certificate(**kwargs):
        result = original_certificate(**kwargs)
        if int(kwargs["anchor_source_means"].shape[0]) == 4:
            result["passed"] = False
            result["summary"] = {
                **result["summary"],
                "reason": "synthetic-l0-replay-failure",
            }
        return result

    monkeypatch.setattr(
        materializer, "certify_depthsplat_tile_soft_mixture", l1_only_certificate
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, soft_mixture_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    assert preflight.tile_trace[0]["accepted_level"] == "L1"
    assert preflight.tile_trace[0]["l1_enrichment_prefetched"] is True
    assert preflight.events["soft_mixture_l0_to_l1_tile_count"] == 1
    assert preflight.update_dense_slots.numel() == 12
    assert route.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    aggregate = preflight.events["soft_mixture_certificate_aggregate"]
    assert aggregate["tile_certificate_attempt_count"] == 2
    assert aggregate["passed_tile_certificate_count"] == 1
    assert aggregate["failed_tile_certificate_count"] == 1


def _forced_kernel_closure_result(result, *, passed):
    """Keep a source-only guard fixture internally self-consistent."""

    forced = copy.deepcopy(result)
    risk = 0.0 if passed else 1.0
    summary = dict(forced["summary"])
    summary.update(
        {
            "input_valid": True,
            "reason": None if passed else "kernel-closure-risk-exceeds-strict-limit",
            "strict_maximum_relative_risk": 0.1,
            "maximum_world_kernel_risk": risk,
            "maximum_source_kernel_risk": risk,
            "maximum_kernel_risk": risk,
            "maximum_source_log_depth_rms": 0.0,
            "per_output": [
                {
                    **dict(row),
                    "world_kernel_risk": risk,
                    "source_kernel_risk": risk,
                    "maximum_kernel_risk": risk,
                    "source_log_depth_rms": 0.0,
                    "passed": passed,
                }
                for row in summary["per_output"]
            ],
        }
    )
    forced["summary"] = summary
    forced["passed"] = passed
    return forced


def test_soft_mixture_kernel_closure_default_is_strict_fp32_abstention():
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, soft_mixture_kernel_closure_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    aggregate = preflight.events["mixture_kernel_closure_aggregate"]
    assert aggregate["strict_maximum_relative_risk"] == pytest.approx(
        1024.0 * torch.finfo(torch.float32).eps
    )
    assert aggregate["tile_kernel_closure_attempt_count"] == 2
    assert aggregate["passed_tile_kernel_closure_count"] == 0
    assert aggregate["failed_tile_kernel_closure_count"] == 2
    s_r_aggregate = preflight.events["soft_mixture_certificate_aggregate"]
    assert s_r_aggregate["tile_certificate_attempt_count"] == 2
    assert s_r_aggregate["passed_tile_certificate_count"] == 2
    assert s_r_aggregate["failed_tile_certificate_count"] == 0
    assert preflight.update_dense_slots.numel() == 0
    assert bool(preflight.promote_full_mask.all())
    assert route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}


def test_soft_mixture_kernel_closure_retries_l0_on_existing_l1_prefetch(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    original_guard = materializer.assess_depthsplat_tile_kernel_closure

    def l1_only_kernel_closure(**kwargs):
        result = original_guard(**kwargs)
        return _forced_kernel_closure_result(
            result,
            passed=int(kwargs["anchor_source_means"].shape[0]) == 12,
        )

    monkeypatch.setattr(
        materializer, "assess_depthsplat_tile_kernel_closure", l1_only_kernel_closure
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, soft_mixture_kernel_closure_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    record = preflight.tile_trace[0]
    assert plan.tile_trace[0]["pre_guard_route"] == "L0"
    assert record["accepted_level"] == "L1"
    assert record["l1_enrichment_prefetched"] is True
    assert record["l0_mixture_kernel_closure_failure"]["passed"] is False
    assert record["mixture_kernel_closure"]["passed"] is True
    assert preflight.events["mixture_kernel_closure_l0_to_l1_tile_count"] == 1
    assert preflight.events["mixture_kernel_closure_full_promotion_tile_count"] == 0
    aggregate = preflight.events["mixture_kernel_closure_aggregate"]
    assert aggregate["tile_kernel_closure_attempt_count"] == 2
    assert aggregate["passed_tile_kernel_closure_count"] == 1
    assert aggregate["failed_tile_kernel_closure_count"] == 1
    assert route.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}
    assert route.events["mixture_kernel_closure_aggregate_sha256"] == (
        preflight.events["mixture_kernel_closure_aggregate_sha256"]
    )


def test_kernel_closure_route_isolation_forces_requested_level_to_full(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    original_guard = materializer.assess_depthsplat_tile_kernel_closure

    def accepting_kernel_closure(**kwargs):
        return _forced_kernel_closure_result(original_guard(**kwargs), passed=True)

    monkeypatch.setattr(
        materializer, "assess_depthsplat_tile_kernel_closure", accepting_kernel_closure
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    plan, _packet, _packed, preflight = _preflight(
        features,
        z_depths,
        soft_mixture_kernel_closure_t4=True,
        route_isolation="l1_only",
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)
    assert preflight.tile_trace[0]["reason"] == "route-isolation-l0-forced-full"
    assert route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}

    plan, _packet, _packed, preflight = _preflight(
        features,
        z_depths,
        soft_mixture_kernel_closure_t4=True,
        route_isolation="l0_only",
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)
    assert route.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0}

    plan, _packet, _packed, preflight = _preflight(
        features,
        z_depths,
        soft_mixture_kernel_closure_t4=True,
        route_isolation="l0_to_l1",
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)
    assert preflight.tile_trace[0]["accepted_level"] == "L1"
    assert preflight.tile_trace[0]["l0_to_l1_coordinated"] is True
    assert route.events["route_counts"] == {"L0": 0, "L1": 1, "Full": 0}


def test_kernel_risk_guard_is_authenticated_and_binds_the_calibrated_threshold(
    monkeypatch,
):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route
    from scripts.saes_depthsplat_soft_mixture_sample0_quality_gate import (
        _source_summary,
        soft_mixture_kernel_closure_t4_profile,
    )

    original_guard = materializer.assess_depthsplat_tile_kernel_closure

    def accepting_kernel_closure(**kwargs):
        assert kwargs["strict_maximum_relative_risk"] == pytest.approx(0.1)
        return _forced_kernel_closure_result(
            original_guard(**kwargs), passed=True
        )

    monkeypatch.setattr(
        materializer, "assess_depthsplat_tile_kernel_closure", accepting_kernel_closure
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan = _plan(
        features,
        z_depths,
        soft_mixture_kernel_closure_t4=True,
        feature_threshold=0.2,
        depth_threshold=0.1,
    )
    guard = _authenticated_kernel_risk_frozen_guard(plan, threshold_value=0.1)
    _plan_value, _packet, _packed, preflight = _preflight(
        features,
        z_depths,
        soft_mixture_kernel_closure_t4=True,
        feature_threshold=0.2,
        depth_threshold=0.1,
        mixture_kernel_closure_frozen_guard=guard,
    )
    route = resolve_depthsplat_compact_final_route(_plan_value, preflight)
    profile = soft_mixture_kernel_closure_t4_profile(kernel_risk_guard=guard)
    summary = _source_summary(
        plan=_plan_value,
        preflight=preflight,
        final_route=route,
        profile=profile,
    )

    assert preflight.events["mixture_kernel_closure_calibrated_threshold"] is True
    assert preflight.events["mixture_kernel_closure_frozen_guard"] == dict(guard)
    assert route.events["mixture_kernel_closure_frozen_guard"] == dict(guard)
    assert summary["kernel_closure"]["calibrated_threshold"] is True
    assert summary["kernel_closure"]["strict_maximum_relative_risk"] == pytest.approx(
        0.1
    )

    with pytest.raises(TypeError, match="authenticated ACID guard"):
        _preflight(
            features,
            z_depths,
            soft_mixture_kernel_closure_t4=True,
            feature_threshold=0.2,
            depth_threshold=0.1,
            mixture_kernel_closure_frozen_guard=dict(guard),
        )


def test_soft_mixture_kernel_closure_promotes_l1_failure_to_full(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    original_guard = materializer.assess_depthsplat_tile_kernel_closure

    def rejecting_kernel_closure(**kwargs):
        return _forced_kernel_closure_result(original_guard(**kwargs), passed=False)

    monkeypatch.setattr(
        materializer, "assess_depthsplat_tile_kernel_closure", rejecting_kernel_closure
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, soft_mixture_kernel_closure_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    record = preflight.tile_trace[0]
    assert record["accepted"] is False
    assert record["l0_mixture_kernel_closure_failure"]["passed"] is False
    assert record["l1_mixture_kernel_closure_failure"]["passed"] is False
    for level in ("l0", "l1"):
        certificate = record[f"{level}_soft_mixture_certificate_before_kernel_closure"]
        assert certificate["passed"] is True
        assert record[
            f"{level}_soft_mixture_certificate_before_kernel_closure_sha256"
        ] == materializer._canonical_sha256(certificate)
    assert preflight.update_dense_slots.numel() == 0
    assert bool(preflight.promote_full_mask.all())
    assert preflight.events["mixture_kernel_closure_l0_to_l1_tile_count"] == 0
    assert preflight.events["mixture_kernel_closure_full_promotion_tile_count"] == 1
    aggregate = preflight.events["mixture_kernel_closure_aggregate"]
    assert aggregate["tile_kernel_closure_attempt_count"] == 2
    assert aggregate["passed_tile_kernel_closure_count"] == 0
    assert aggregate["failed_tile_kernel_closure_count"] == 2
    s_r_aggregate = preflight.events["soft_mixture_certificate_aggregate"]
    assert s_r_aggregate["tile_certificate_attempt_count"] == 2
    assert s_r_aggregate["passed_tile_certificate_count"] == 2
    assert s_r_aggregate["failed_tile_certificate_count"] == 0
    assert route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}

    preflight.events["mixture_kernel_closure_aggregate"]["maximum_kernel_risk"] = 0.0
    with pytest.raises(ValueError, match="kernel-closure aggregate binding changed"):
        resolve_depthsplat_compact_final_route(plan, preflight)


@pytest.mark.parametrize(
    "validator_name",
    (
        "_validate_soft_mixture_evidence",
        "_validate_mixture_kernel_closure_evidence",
    ),
)
def test_soft_mixture_kernel_contract_failures_abort_instead_of_promoting_full(
    monkeypatch, validator_name
):
    import saes.depthsplat_l0_l1_materializer as materializer

    def forced_contract_failure(*_args, **_kwargs):
        raise ValueError("forced-kernel-contract-failure")

    monkeypatch.setattr(materializer, validator_name, forced_contract_failure)
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    with pytest.raises(ValueError, match="forced-kernel-contract-failure"):
        _preflight(features, z_depths, soft_mixture_kernel_closure_t4=True)


def test_kernel_closure_runner_binds_each_closure_to_a_passed_s_r_certificate():
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route
    from scripts.saes_depthsplat_soft_mixture_sample0_quality_gate import (
        _source_summary,
        soft_mixture_kernel_closure_t4_profile,
    )

    profile = soft_mixture_kernel_closure_t4_profile()
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features,
        z_depths,
        soft_mixture_kernel_closure_t4=True,
        feature_threshold=profile["feature_threshold"],
        depth_threshold=profile["depth_threshold"],
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    summary = _source_summary(
        plan=plan,
        preflight=preflight,
        final_route=route,
        profile=profile,
    )
    assert summary["certificate"]["summary"]["passed_tile_certificate_count"] == (
        summary["kernel_closure"]["summary"]["tile_kernel_closure_attempt_count"]
    )

    corrupted = copy.deepcopy(preflight.events["soft_mixture_certificate_aggregate"])
    corrupted["passed_tile_certificate_count"] -= 1
    corrupted["failed_tile_certificate_count"] += 1
    preflight.events["soft_mixture_certificate_aggregate"] = corrupted
    preflight.events["soft_mixture_certificate_aggregate_sha256"] = (
        materializer._canonical_sha256(corrupted)
    )

    with pytest.raises(RuntimeError, match="lack matching passed S/R certificates"):
        _source_summary(
            plan=plan,
            preflight=preflight,
            final_route=route,
            profile=profile,
        )


def test_kernel_closure_runner_distinguishes_identity_fallback_from_compact_pass():
    from scripts.saes_depthsplat_soft_mixture_sample0_quality_gate import _quality_status

    assert _quality_status(verdict={"pass": True}, update_count=0) == (
        "PASS_IDENTITY_FALLBACK"
    )
    assert _quality_status(verdict={"pass": True}, update_count=1) == "PASS"
    assert _quality_status(verdict={"pass": False}, update_count=0) == "QUALITY_FAILED"


def test_soft_mixture_kernel_closure_records_unscorable_integrals_and_promotes_full(
    monkeypatch,
):
    import saes.depthsplat_mixture_kernel_guard as kernel_guard
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    monkeypatch.setattr(
        kernel_guard, "_relative_kernel_risk", lambda **_kwargs: float("inf")
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, soft_mixture_kernel_closure_t4=True
    )
    route = resolve_depthsplat_compact_final_route(plan, preflight)

    aggregate = preflight.events["mixture_kernel_closure_aggregate"]
    assert aggregate["tile_kernel_closure_attempt_count"] == 2
    assert aggregate["failed_tile_kernel_closure_count"] == 2
    assert aggregate["maximum_kernel_risk_distribution"]["finite_tile_risk_count"] == 0
    assert aggregate["maximum_kernel_risk_distribution"]["unscorable_tile_count"] == 2
    assert aggregate["reason_counts"] == {"kernel-integral": 2}
    assert route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}


def test_soft_mixture_kernel_closure_forbids_covariance_expansion():
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)

    with pytest.raises(ValueError, match="fixed-scale profile forbids covariance expansion"):
        _preflight(
            features,
            z_depths,
            soft_mixture_kernel_closure_t4=True,
            maximum_coverage_covariance_scale=2.0,
        )


def test_coverage_enriched_route_rejects_l0_to_l1_trace_mutation(monkeypatch):
    import saes.depthsplat_l0_l1_materializer as materializer
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    monkeypatch.setattr(
        materializer,
        "audit_depthsplat_owner_coverage",
        lambda **kwargs: _owner_support_result(
            owner_count=int(kwargs["merged_means"].shape[0]),
            virtual_owner_indices=kwargs["virtual_owner_indices"],
            passed=True,
        ),
    )
    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(
        features, z_depths, coverage_enriched_t4=True
    )
    assert preflight.tile_trace[0]["accepted_level"] == "L0"
    preflight.tile_trace[0]["accepted_level"] = "L1"

    with pytest.raises(ValueError, match="preflight tile trace hash changed"):
        resolve_depthsplat_compact_final_route(plan, preflight)


def test_route_rejects_mutated_plan_trace_before_preflight_output_is_consumed():
    from saes.depthsplat_l0_l1_materializer import resolve_depthsplat_compact_final_route

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, _packed, preflight = _preflight(features, z_depths)
    plan.tile_trace[0]["pre_guard_route"] = "Full"

    with pytest.raises(ValueError, match="plan tile trace hash changed"):
        resolve_depthsplat_compact_final_route(plan, preflight)


def test_preflight_rejects_a_mutated_coverage_plan_trace():
    from depthsplat.src.geometry.projection import get_world_rays, sample_image_grid
    from saes.depthsplat_l0_l1_materializer import (
        DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
        preflight_depthsplat_l0_l1_materialization,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan = _plan(features, z_depths, coverage_enriched_t4=True)
    packet, packed = _selected_packet_and_packed(
        plan.selection_mask, z_depths, routing_features=features
    )
    plan.tile_trace[0]["depth_uniform"] = False

    with pytest.raises(ValueError, match="plan tile trace hash changed"):
        preflight_depthsplat_l0_l1_materialization(
            packet,
            packed,
            plan,
            features,
            z_depths,
            source_sample_image_grid=sample_image_grid,
            source_get_world_rays=get_world_rays,
            maximum_coverage_covariance_scale=1.0,
            execution_profile=DEPTHSPLAT_COVERAGE_ENRICHED_T4_MATERIALIZATION_PROFILE,
        )


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

    with pytest.raises(ValueError, match="preflight tile trace hash changed"):
        resolve_depthsplat_compact_final_route(plan, preflight)


def test_apply_rejects_a_mutated_final_route_trace():
    from saes.depthsplat_l0_l1_materializer import (
        apply_depthsplat_compact_l0_l1_materialization,
        resolve_depthsplat_compact_final_route,
    )

    features = torch.zeros(1, 1, 3, 4, 4)
    z_depths = torch.full((1, 1, 4, 4), 2.0)
    plan, _packet, packed, preflight = _preflight(features, z_depths)
    route = resolve_depthsplat_compact_final_route(plan, preflight)
    route.tile_trace[0]["final_route"] = "Full"

    with pytest.raises(ValueError, match="final route tile trace hash changed"):
        apply_depthsplat_compact_l0_l1_materialization(packed, preflight, route)


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
