"""Paper-kp compact packet materialization tests."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _hash_trace(trace):
    return hashlib.sha256(
        json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _features(level):
    values = torch.zeros(1, 1, 2, 4, 4)
    if level == "L1":
        for (row, column), value in {
            (0, 0): (0.0, 0.0),
            (0, 3): (2.0, 0.0),
            (3, 0): (0.0, 2.0),
            (3, 3): (2.0, 2.0),
        }.items():
            values[0, 0, :, row, column] = torch.tensor(value)
    return values


def _plan(level):
    from saes.probe_first_schedule import (
        PAPER_KP_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
    )

    features = _features(level)
    depths = torch.ones(1, 1, 16, 1, 1)
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=4,
        width=4,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="paper-probe-feature-variance-first-hit",
        l1_anchor_semantics=PAPER_KP_ANCHOR_SEMANTICS,
    )
    record = plan.tile_trace[0]
    assert record["pre_guard_route"] == level
    assert plan.events["l0_anchor_count"] == 4
    assert plan.events["l1_anchor_count"] == 4
    assert int(plan.primary_mask.sum()) == 4
    assert int(plan.secondary_mask.sum()) == 0
    assert int(plan.selection_mask.sum()) == 4
    return plan, features


def _source_trace():
    return {
        "source_bound": True,
        "execution_scope": "s3_raw_gaussian_head_only",
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
    }


def _packet_and_packed(mask):
    from saes.sparse_gaussian_consumer import PackedGaussianAttributes, SparseRawGaussianPacket

    positions = mask.nonzero(as_tuple=False)
    count = int(positions.shape[0])
    pixels = positions[:, 1] * 4 + positions[:, 2]
    slots = positions[:, 0] * 16 + pixels
    raw = torch.zeros(count, 8)
    raw[:, 2:] = torch.linspace(-0.2, 0.2, count).unsqueeze(1)
    trace = _source_trace()
    packet = SparseRawGaussianPacket(
        descriptor_keys=torch.stack(
            (torch.zeros(count, dtype=torch.int64), positions[:, 0], pixels, torch.zeros(count, dtype=torch.int64)),
            dim=1,
        ),
        raw_descriptors=raw,
        primitive_keys=torch.stack(
            (
                torch.zeros(count, dtype=torch.int64),
                positions[:, 0],
                pixels,
                torch.zeros(count, dtype=torch.int64),
                torch.zeros(count, dtype=torch.int64),
            ),
            dim=1,
        ),
        primitive_to_descriptor=torch.arange(count, dtype=torch.int64),
        coordinates=torch.stack(
            ((positions[:, 2].to(torch.float32) + 0.5) / 4.0, (positions[:, 1].to(torch.float32) + 0.5) / 4.0),
            dim=1,
        ),
        depths=torch.ones(count),
        mapped_opacities=torch.full((count,), 0.2),
        dense_slots=slots.to(torch.int64),
        source_trace=trace,
    )
    ray_coordinates = packet.coordinates
    means = torch.cat(
        (ray_coordinates, torch.ones(count, 1)),
        dim=1,
    )
    means = means / means.norm(dim=1, keepdim=True)
    packed = PackedGaussianAttributes(
        batch_indices=torch.zeros(count, dtype=torch.int64),
        dense_slots=slots.to(torch.int64),
        means=means,
        covariances=torch.eye(3).expand(count, -1, -1).clone() * 0.02,
        harmonics=torch.arange(count * 6, dtype=torch.float32).reshape(count, 3, 2) / 10.0,
        opacities=torch.linspace(0.1, 0.4, count),
        source_trace=trace,
        source_trace_sha256=_hash_trace(trace),
    )
    return packet, packed


def _route(plan, level):
    from saes.guarded_selected_route import GuardedSelectedRoute

    selected = plan.primary_mask.clone()
    return GuardedSelectedRoute(
        selected_output_mask=selected,
        additional_full_mask=torch.zeros_like(selected),
        raw_head_request_mask=selected.clone(),
        tile_trace=(
            {
                "view": 0,
                "tile_y": 0,
                "tile_x": 0,
                "pre_guard_route": level,
                "depth_uniform": True,
                "guard_checks": [],
                "final_route": level,
            },
        ),
        events={
            "schema_version": "synthetic-guard-v1",
            "execution_policy": "paper-nonzero-dev",
            "route_counts": {level: 1},
        },
    )


def _context():
    return torch.eye(4).reshape(1, 1, 4, 4), torch.eye(3).reshape(1, 1, 3, 3)


def _native_adapter():
    from transplat.src.model.encoder.common.gaussian_adapter import (
        GaussianAdapter,
        GaussianAdapterCfg,
    )

    return GaussianAdapter(
        GaussianAdapterCfg(
            gaussian_scale_min=0.01,
            gaussian_scale_max=0.02,
            sh_degree=0,
        )
    ).eval()


def _native_geometry_packet_and_packed():
    """Build four selected corner descriptors through the real CPU Adapter."""
    from saes.sparse_gaussian_consumer import PackedGaussianConsumer, SparseRawGaussianPacket
    from transplat.src.model.encoder.encoder_trans import (
        gaussian_adapter_coordinates_from_raw_offsets,
    )

    height = width = 4
    mask = torch.zeros(1, height, width, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 3] = True
    mask[0, 3, 0] = True
    mask[0, 3, 3] = True
    positions = mask.nonzero(as_tuple=False)
    count = int(positions.shape[0])
    pixels = positions[:, 1] * width + positions[:, 2]
    slots = positions[:, 0] * (height * width) + pixels

    # The first two raw-head values are TranSplat's offset logits. The rest
    # match a degree-zero GaussianAdapter raw body: scales, quaternion, and SH.
    raw = torch.zeros(count, 12)
    raw[:, :2] = torch.tensor(
        ((-2.0, 1.7), (1.2, -1.8), (-1.5, -1.3), (2.1, 1.4))
    )
    raw[:, 2:5] = torch.tensor(
        ((-0.4, 0.1, 0.2), (0.3, -0.2, 0.1), (0.2, 0.4, -0.3), (-0.1, 0.2, 0.5))
    )
    raw[:, 5] = 1.0
    raw[:, 9:] = torch.tensor(
        ((0.2, 0.1, -0.1), (-0.2, 0.3, 0.1), (0.1, -0.3, 0.2), (0.3, 0.2, -0.2))
    )

    coordinates = gaussian_adapter_coordinates_from_raw_offsets(
        raw[:, :2].sigmoid(),
        image_shape=(height, width),
        pixel_indices=pixels.to(dtype=torch.int64),
    )
    trace = {
        **_source_trace(),
        "head_forward_invocations": 1,
        "head_final_positions_executed": count,
    }
    packet = SparseRawGaussianPacket(
        descriptor_keys=torch.stack(
            (
                torch.zeros(count, dtype=torch.int64),
                positions[:, 0],
                pixels,
                torch.zeros(count, dtype=torch.int64),
            ),
            dim=1,
        ),
        raw_descriptors=raw,
        primitive_keys=torch.stack(
            (
                torch.zeros(count, dtype=torch.int64),
                positions[:, 0],
                pixels,
                torch.zeros(count, dtype=torch.int64),
                torch.zeros(count, dtype=torch.int64),
            ),
            dim=1,
        ),
        primitive_to_descriptor=torch.arange(count, dtype=torch.int64),
        coordinates=coordinates,
        depths=torch.tensor((1.1, 1.7, 2.2, 2.8)),
        mapped_opacities=torch.tensor((0.10, 0.20, 0.30, 0.40)),
        dense_slots=slots.to(dtype=torch.int64),
        source_trace=trace,
    )

    extrinsics = torch.eye(4).reshape(1, 1, 4, 4)
    extrinsics[0, 0, :3, :3] = torch.tensor(
        ((0.92106099, -0.38941834, 0.0), (0.38941834, 0.92106099, 0.0), (0.0, 0.0, 1.0))
    )
    extrinsics[0, 0, :3, 3] = torch.tensor((0.25, -0.40, 1.10))
    intrinsics = torch.tensor(
        ([1.45, 0.08, 0.03], [0.0, 1.20, -0.04], [0.0, 0.0, 1.0])
    ).reshape(1, 1, 3, 3)
    packed = PackedGaussianConsumer(_native_adapter()).convert(
        packet,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_shape=(height, width),
    )
    return packet, packed, extrinsics, intrinsics


def _native_world_points(coordinates, depths, extrinsics, intrinsics):
    from transplat.src.geometry.projection import get_world_rays

    count = int(depths.numel())
    origins, directions = get_world_rays(
        coordinates,
        extrinsics[0, 0].expand(count, -1, -1),
        intrinsics[0, 0].expand(count, -1, -1),
    )
    return origins + directions * depths.reshape(-1, 1)


def _packet_pixel_centres(packet):
    pixels = packet.descriptor_keys[:, 2]
    rows = pixels // 4
    columns = pixels % 4
    return torch.stack(
        ((columns.to(torch.float32) + 0.5) / 4.0, (rows.to(torch.float32) + 0.5) / 4.0),
        dim=1,
    )


def _selected_packet_after_dense_poison(packet):
    """Model an S3 map whose non-selected descriptors are unreadable."""
    from saes.sparse_gaussian_consumer import SparseRawGaussianPacket

    pixels = packet.descriptor_keys[:, 2]
    rows = pixels // 4
    columns = pixels % 4
    raw_map = torch.full((1, 1, 4, 4, packet.raw_descriptors.shape[1]), torch.nan)
    raw_map[0, 0, rows, columns] = packet.raw_descriptors
    selected_raw = raw_map[0, 0, rows, columns]
    assert bool(torch.isfinite(selected_raw).all())
    selected_mask = torch.zeros(4, 4, dtype=torch.bool)
    selected_mask[rows, columns] = True
    assert bool(torch.isnan(raw_map[0, 0][~selected_mask]).all())
    return SparseRawGaussianPacket(
        **{**packet.__dict__, "raw_descriptors": selected_raw}
    )


def test_selected_packet_native_offset_coordinates_lift_to_packed_means():
    packet, packed, extrinsics, intrinsics = _native_geometry_packet_and_packed()

    centres = _packet_pixel_centres(packet)
    assert float((packet.coordinates - centres).abs().amax()) > 0.01
    expected_means = _native_world_points(
        packet.coordinates,
        packet.depths,
        extrinsics,
        intrinsics,
    )
    torch.testing.assert_close(packed.means, expected_means, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(
        packed.means,
        _native_world_points(centres, packet.depths, extrinsics, intrinsics),
    )


def test_center_ray_residual_pseudo_is_distinct_from_native_coordinate_lift():
    from saes.packed_l0_l1_materializer import _camera_residual_pseudo_means

    packet, packed, extrinsics, intrinsics = _native_geometry_packet_and_packed()
    assignment = torch.tensor([[0.71, 0.17, 0.08, 0.04]])
    residual_pseudo = _camera_residual_pseudo_means(
        source_means=packed.means,
        source_depths=packet.depths,
        source_positions=((0, 0), (0, 3), (3, 0), (3, 3)),
        assignment_matrix=assignment,
        target_positions=((1, 1),),
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        view=0,
        width=4,
        height=4,
    )

    centres = _packet_pixel_centres(packet)
    weighted_offset = assignment @ (packet.coordinates - centres)
    target_coordinate = torch.tensor(((1.5 / 4.0, 1.5 / 4.0))) + weighted_offset
    native_coordinate_pseudo = _native_world_points(
        target_coordinate,
        assignment @ packet.depths,
        extrinsics,
        intrinsics,
    )
    assert float((residual_pseudo - native_coordinate_pseudo).norm()) > 1e-4


def test_native_geometry_audit_is_invariant_to_skipped_raw_poison():
    from saes.packed_l0_l1_materializer import _camera_residual_pseudo_means
    from saes.sparse_gaussian_consumer import PackedGaussianConsumer

    packet, packed, extrinsics, intrinsics = _native_geometry_packet_and_packed()
    poisoned_packet = _selected_packet_after_dense_poison(packet)
    poisoned_packed = PackedGaussianConsumer(_native_adapter()).convert(
        poisoned_packet,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_shape=(4, 4),
    )
    torch.testing.assert_close(poisoned_packet.coordinates, packet.coordinates)
    torch.testing.assert_close(poisoned_packed.means, packed.means, rtol=1e-6, atol=1e-6)

    assignment = torch.tensor([[0.40, 0.30, 0.20, 0.10]])
    baseline_pseudo = _camera_residual_pseudo_means(
        source_means=packed.means,
        source_depths=packet.depths,
        source_positions=((0, 0), (0, 3), (3, 0), (3, 3)),
        assignment_matrix=assignment,
        target_positions=((2, 1),),
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        view=0,
        width=4,
        height=4,
    )
    poisoned_pseudo = _camera_residual_pseudo_means(
        source_means=poisoned_packed.means,
        source_depths=poisoned_packet.depths,
        source_positions=((0, 0), (0, 3), (3, 0), (3, 3)),
        assignment_matrix=assignment,
        target_positions=((2, 1),),
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        view=0,
        width=4,
        height=4,
    )
    torch.testing.assert_close(poisoned_pseudo, baseline_pseudo, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("level", ("L0", "L1"))
def test_paper_kp_materializer_merges_nonzero_anchor_packet(level):
    from saes.packed_l0_l1_materializer import (
        apply_compact_l0_l1_materialization,
        preflight_compact_l0_l1_materialization,
        resolve_compact_final_route,
        SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION,
    )

    plan, features = _plan(level)
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, intrinsics = _context()
    preflight = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        _route(plan, level),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION,
    )
    assert preflight.events["execution_policy"] == "paper-nonzero-dev"
    assert preflight.events["not_lossless_deletion"] is True
    assert preflight.events["source_nonprobe_s3_attribute_reads"] == 0
    assert preflight.events["accepted_tiles"] == 1
    assert (
        preflight.events["aggregation"]
        == "probe-spatial-virtual-single-assignment-coverage-closed-v3"
    )
    assert preflight.events["coverage_certificate"] == "selected-only-spatial-virtual-intra-tile-2sigma-v1"
    assert preflight.tile_trace[0]["coverage"]["passed"] is True
    assert int(preflight.promote_full_mask.sum()) == 0
    assert preflight.update_dense_slots.numel() == 4

    final_route = resolve_compact_final_route(_route(plan, level), plan, preflight)
    assert final_route.events["route_counts"] == {"L0": 1, "L1": 0, "Full": 0} if level == "L0" else {"L0": 0, "L1": 1, "Full": 0}
    materialized = apply_compact_l0_l1_materialization(packed, preflight, final_route)
    assert materialized.dense_slots.numel() == 4
    assert materialized.source_trace["packet_selection_kind"] == "guard_selected_output_mask"
    assert materialized.source_trace["compact_materialization_skipped_s3_attributes_accessed"] is False
    assert bool(torch.isfinite(materialized.means).all())
    assert bool(torch.isfinite(materialized.covariances).all())
    assert bool((torch.linalg.eigvalsh(materialized.covariances) >= -1e-6).all())
    assert bool((materialized.opacities >= packed.opacities.min()).all())
    assert bool((materialized.opacities <= packed.opacities.max()).all())
    assert bool((materialized.harmonics >= packed.harmonics.amin(dim=0)).all())
    assert bool((materialized.harmonics <= packed.harmonics.amax(dim=0)).all())


def test_adaptive_l1_15_materializer_merges_only_the_bound_center_omission():
    from saes.guarded_selected_route import (
        ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY,
        GuardedSelectedRoute,
    )
    from saes.packed_l0_l1_materializer import (
        CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
        _mask_sha256,
        preflight_compact_l0_l1_materialization,
    )
    from saes.probe_first_schedule import (
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
    )

    features = _features("L1")
    depths = torch.ones(1, 1, 16, 1, 1)
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
    assert plan.tile_trace[0]["pre_guard_route"] == "L1"
    packet, packed = _packet_and_packed(plan.selection_mask)
    retained = plan.tile_trace[0]["l1_anchor_local_positions"]
    route = GuardedSelectedRoute(
        selected_output_mask=plan.selection_mask.clone(),
        additional_full_mask=torch.zeros_like(plan.selection_mask),
        raw_head_request_mask=plan.selection_mask.clone(),
        tile_trace=(
            {
                "view": 0,
                "tile_y": 0,
                "tile_x": 0,
                "pre_guard_route": "L1",
                "depth_uniform": True,
                "retained_local_positions": retained,
                "final_route": "L1",
            },
        ),
        events={
            "execution_policy": ENGINEERING_L1_15_ADAPTIVE_CONTINUITY_DEV_POLICY,
            "source_tile_trace_sha256": plan.events["tile_trace_sha256"],
            "source_selection_mask_sha256": _mask_sha256(plan.selection_mask),
            "l1_anchor_semantics": ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
            "l1_anchor_count": 15,
        },
    )
    extrinsics, intrinsics = _context()
    preflight = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        route,
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
    )

    assert preflight.events["l1_anchor_count"] == 15
    assert preflight.events["accepted_tiles"] == 1
    assert preflight.update_dense_slots.numel() == 15
    assert preflight.tile_trace[0]["anchor_count"] == 15
    assert preflight.tile_trace[0]["nonprobe_count"] == 1
    assert preflight.events["source_nonprobe_s3_attribute_reads"] == 0


def test_adaptive_l1_15_v4_attribute_loo_replays_selected_centers_only():
    from saes.packed_l0_l1_materializer import (
        selected_anchor_v4_attribute_loo_certificate,
    )
    from saes.probe_first_schedule import (
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
        l1_local_positions_for_tile,
    )

    features = _features("L1")
    depths = torch.ones(1, 1, 16, 1, 1)
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
    packet, packed = _packet_and_packed(plan.selection_mask)
    anchors = l1_local_positions_for_tile(
        plan.tile_trace[0],
        tile_size=4,
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )
    slots = packet.dense_slots.tolist()
    certificate = selected_anchor_v4_attribute_loo_certificate(
        packet=packet,
        packed=packed,
        slot_to_index={int(slot): index for index, slot in enumerate(slots)},
        view=0,
        tile_y=0,
        tile_x=0,
        height=4,
        width=4,
        tile_size=4,
        anchor_positions=anchors,
    )

    assert certificate["checked"] is True
    assert certificate["held_out_center_count"] == 3
    assert certificate["selected_anchor_s3_attribute_label_reads"] == 3
    assert certificate["nonprobe_s3_attribute_reads"] == 0
    assert certificate["q75_risk"] >= 0.0
    assert len(certificate["held_out_center_records"]) == 3
    omitted_y, omitted_x = certificate["actual_omitted_local_position"]
    assert omitted_y * 4 + omitted_x not in packet.dense_slots.tolist()


def test_adaptive_l1_15_v4_attribute_loo_accepts_a_constant_selected_field():
    from saes.packed_l0_l1_materializer import selected_anchor_v4_attribute_loo_certificate
    from saes.probe_first_schedule import (
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
        l1_local_positions_for_tile,
    )
    from saes.sparse_gaussian_consumer import PackedGaussianAttributes

    features = _features("L1")
    depths = torch.ones(1, 1, 16, 1, 1)
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
    packet, packed = _packet_and_packed(plan.selection_mask)
    constant = PackedGaussianAttributes(
        batch_indices=packed.batch_indices,
        dense_slots=packed.dense_slots,
        means=packed.means,
        covariances=packed.covariances,
        harmonics=torch.full_like(packed.harmonics, 0.25),
        opacities=torch.full_like(packed.opacities, 0.2),
        source_trace=packed.source_trace,
        source_trace_sha256=packed.source_trace_sha256,
    )
    anchors = l1_local_positions_for_tile(
        plan.tile_trace[0],
        tile_size=4,
        l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    )
    certificate = selected_anchor_v4_attribute_loo_certificate(
        packet=packet,
        packed=constant,
        slot_to_index={int(slot): index for index, slot in enumerate(packet.dense_slots.tolist())},
        view=0,
        tile_y=0,
        tile_x=0,
        height=4,
        width=4,
        tile_size=4,
        anchor_positions=anchors,
    )

    assert certificate["q75_risk"] <= 1e-3


def test_v4_attribute_loo_rejection_promotes_one_adaptive_l1_omission_to_full():
    from saes.guarded_selected_route import (
        ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
        GuardedSelectedRoute,
    )
    from saes.packed_l0_l1_materializer import (
        _mask_sha256,
        preflight_compact_l0_l1_materialization,
        resolve_compact_final_route,
    )
    from saes.probe_first_schedule import (
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
        build_incremental_probe_first_plan,
    )
    from saes.sparse_gaussian_consumer import PackedGaussianAttributes

    features = _features("L1")
    depths = torch.ones(1, 1, 16, 1, 1)
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
    packet, packed = _packet_and_packed(plan.selection_mask)
    retained = plan.tile_trace[0]["l1_anchor_local_positions"]
    sharp_harmonics = packed.harmonics.clone()
    center_position = next(
        tuple(position)
        for position in retained
        if tuple(position) in {(1, 1), (1, 2), (2, 1), (2, 2)}
    )
    center_slot = center_position[0] * 4 + center_position[1]
    center_index = int((packed.dense_slots == center_slot).nonzero(as_tuple=False).item())
    sharp_harmonics[center_index] += 100.0
    sharp = PackedGaussianAttributes(
        batch_indices=packed.batch_indices,
        dense_slots=packed.dense_slots,
        means=packed.means,
        covariances=packed.covariances,
        harmonics=sharp_harmonics,
        opacities=packed.opacities,
        source_trace=packed.source_trace,
        source_trace_sha256=packed.source_trace_sha256,
    )
    route = GuardedSelectedRoute(
        selected_output_mask=plan.selection_mask.clone(),
        additional_full_mask=torch.zeros_like(plan.selection_mask),
        raw_head_request_mask=plan.selection_mask.clone(),
        tile_trace=(
            {
                "view": 0,
                "tile_y": 0,
                "tile_x": 0,
                "pre_guard_route": "L1",
                "depth_uniform": True,
                "retained_local_positions": retained,
                "final_route": "L1",
            },
        ),
        events={
            "execution_policy": ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
            "source_tile_trace_sha256": plan.events["tile_trace_sha256"],
            "source_selection_mask_sha256": _mask_sha256(plan.selection_mask),
            "l1_anchor_semantics": ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
            "l1_anchor_count": 15,
        },
    )
    extrinsics, intrinsics = _context()
    preflight = preflight_compact_l0_l1_materialization(
        packet,
        sharp,
        route,
        plan,
        features,
        extrinsics,
        intrinsics,
        selected_anchor_v4_attribute_loo_maximum_risk=0.0,
    )

    assert preflight.events["selected_anchor_v4_attribute_loo_guard"] is True
    assert preflight.events["selected_anchor_v4_attribute_loo_checked_tiles"] == 1
    assert preflight.events["selected_anchor_v4_attribute_loo_promoted_full_tiles"] == 1
    assert int(preflight.promote_full_mask.sum()) == 16
    assert preflight.tile_trace[0]["accepted"] is False
    assert preflight.tile_trace[0]["selected_anchor_v4_attribute_loo"]["passed"] is False
    final_route = resolve_compact_final_route(route, plan, preflight)
    assert int(final_route.selected_output_mask.sum()) == 16
    assert int(final_route.additional_full_mask.sum()) == 1


def test_spatial_virtual_field_is_continuous_and_changes_range_constrained_attributes():
    from saes.packed_l0_l1_materializer import (
        _spatial_virtual_weights,
        preflight_compact_l0_l1_materialization,
        SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION,
    )

    weights = _spatial_virtual_weights(
        [(0, 0), (1, 1), (3, 3)],
        [(0, 0), (0, 3), (3, 0), (3, 3)],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(weights[0], torch.tensor((1.0, 0.0, 0.0, 0.0)))
    torch.testing.assert_close(weights[2], torch.tensor((0.0, 0.0, 0.0, 1.0)))
    torch.testing.assert_close(
        weights[1], torch.tensor((4.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0, 1.0 / 9.0))
    )

    plan, features = _plan("L0")
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, intrinsics = _context()
    preflight = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        _route(plan, "L0"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION,
    )
    assert not torch.allclose(preflight.harmonics, packed.harmonics)
    assert not torch.allclose(preflight.opacities, packed.opacities)
    assert bool((preflight.harmonics >= packed.harmonics.amin(dim=0)).all())
    assert bool((preflight.harmonics <= packed.harmonics.amax(dim=0)).all())
    assert bool((preflight.opacities >= packed.opacities.min()).all())
    assert bool((preflight.opacities <= packed.opacities.max()).all())


def test_direct_merge_keeps_nonreceiver_attributes_isolated():
    from saes.packed_l0_l1_materializer import (
        CONDITIONAL_DIRECT_AGGREGATION,
        preflight_compact_l0_l1_materialization,
    )
    from saes.sparse_gaussian_consumer import PackedGaussianAttributes

    plan, features = _plan("L0")
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, intrinsics = _context()
    baseline = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        _route(plan, "L0"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_DIRECT_AGGREGATION,
    )
    changed_harmonics = packed.harmonics.clone()
    changed_opacities = packed.opacities.clone()
    changed_harmonics[1] += 3.0
    changed_opacities[1] = 0.75
    changed = PackedGaussianAttributes(
        batch_indices=packed.batch_indices,
        dense_slots=packed.dense_slots,
        means=packed.means,
        covariances=packed.covariances,
        harmonics=changed_harmonics,
        opacities=changed_opacities,
        source_trace=packed.source_trace,
        source_trace_sha256=packed.source_trace_sha256,
    )
    candidate = preflight_compact_l0_l1_materialization(
        packet,
        changed,
        _route(plan, "L0"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_DIRECT_AGGREGATION,
    )
    torch.testing.assert_close(candidate.means[0], baseline.means[0])
    torch.testing.assert_close(candidate.covariances[0], baseline.covariances[0])
    torch.testing.assert_close(candidate.harmonics[0], baseline.harmonics[0])
    torch.testing.assert_close(candidate.opacities[0], baseline.opacities[0])
    assert not torch.allclose(candidate.harmonics[1], baseline.harmonics[1])
    assert not torch.allclose(candidate.opacities[1], baseline.opacities[1])


def test_direct_geometry_spatial_attribute_merge_changes_only_attributes():
    from saes.packed_l0_l1_materializer import (
        CONDITIONAL_DIRECT_AGGREGATION,
        CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
        preflight_compact_l0_l1_materialization,
    )
    from saes.sparse_gaussian_consumer import PackedGaussianAttributes

    plan, features = _plan("L0")
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, intrinsics = _context()
    direct = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        _route(plan, "L0"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_DIRECT_AGGREGATION,
    )
    spatial_attributes = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        _route(plan, "L0"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
    )
    torch.testing.assert_close(spatial_attributes.means, direct.means)
    torch.testing.assert_close(spatial_attributes.covariances, direct.covariances)
    assert not torch.allclose(spatial_attributes.harmonics, direct.harmonics)
    assert not torch.allclose(spatial_attributes.opacities, direct.opacities)

    changed_harmonics = packed.harmonics.clone()
    changed_opacities = packed.opacities.clone()
    changed_harmonics[1] += 3.0
    changed_opacities[1] = 0.75
    changed = PackedGaussianAttributes(
        batch_indices=packed.batch_indices,
        dense_slots=packed.dense_slots,
        means=packed.means,
        covariances=packed.covariances,
        harmonics=changed_harmonics,
        opacities=changed_opacities,
        source_trace=packed.source_trace,
        source_trace_sha256=packed.source_trace_sha256,
    )
    changed_preflight = preflight_compact_l0_l1_materialization(
        packet,
        changed,
        _route(plan, "L0"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
    )
    torch.testing.assert_close(changed_preflight.means, spatial_attributes.means)
    torch.testing.assert_close(changed_preflight.covariances, spatial_attributes.covariances)
    assert not torch.allclose(changed_preflight.harmonics[0], spatial_attributes.harmonics[0])
    assert not torch.allclose(changed_preflight.opacities[0], spatial_attributes.opacities[0])
    assert bool((changed_preflight.harmonics >= changed.harmonics.amin(dim=0)).all())
    assert bool((changed_preflight.harmonics <= changed.harmonics.amax(dim=0)).all())
    assert bool((changed_preflight.opacities >= changed.opacities.min()).all())
    assert bool((changed_preflight.opacities <= changed.opacities.max()).all())


def test_l1_plane_intersection_stays_on_the_fitted_surface():
    from saes.packed_l0_l1_materializer import (
        _fit_l1_anchor_plane,
        _intersect_context_rays_with_plane,
        _lift_context_coordinates,
    )

    source_means = torch.tensor(
        (
            (-0.4, -0.3, 2.0),
            (0.5, -0.3, 2.0),
            (-0.4, 0.4, 2.0),
            (0.5, 0.4, 2.0),
        )
    )
    extrinsics, intrinsics = _context()
    origin, normal, metadata = _fit_l1_anchor_plane(source_means)
    coordinates = torch.tensor(((0.25, 0.25), (0.75, 0.25), (0.50, 0.75)))
    depths = _intersect_context_rays_with_plane(
        coordinates,
        plane_origin=origin,
        plane_normal=normal,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        view=0,
    )
    lifted = _lift_context_coordinates(
        coordinates,
        depths,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        view=0,
    )
    torch.testing.assert_close(lifted[:, 2], torch.full((3,), 2.0), atol=1e-6, rtol=1e-6)
    assert metadata["maximum_plane_residual"] <= 1e-6
    assert metadata["planarity_eigenvalue_ratio"] <= 1e-6


def test_l1_plane_merge_keeps_l0_direct_geometry_and_materializes_l1_plane():
    from saes.packed_l0_l1_materializer import (
        CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
        CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION,
        preflight_compact_l0_l1_materialization,
    )

    extrinsics, intrinsics = _context()
    l0_plan, l0_features = _plan("L0")
    l0_packet, l0_packed = _packet_and_packed(l0_plan.selection_mask)
    v4_l0 = preflight_compact_l0_l1_materialization(
        l0_packet,
        l0_packed,
        _route(l0_plan, "L0"),
        l0_plan,
        l0_features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
    )
    v5_l0 = preflight_compact_l0_l1_materialization(
        l0_packet,
        l0_packed,
        _route(l0_plan, "L0"),
        l0_plan,
        l0_features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION,
    )
    torch.testing.assert_close(v5_l0.means, v4_l0.means)
    torch.testing.assert_close(v5_l0.covariances, v4_l0.covariances)
    torch.testing.assert_close(v5_l0.harmonics, v4_l0.harmonics)
    torch.testing.assert_close(v5_l0.opacities, v4_l0.opacities)

    l1_plan, l1_features = _plan("L1")
    l1_packet, l1_packed = _packet_and_packed(l1_plan.selection_mask)
    v5_l1 = preflight_compact_l0_l1_materialization(
        l1_packet,
        l1_packed,
        _route(l1_plan, "L1"),
        l1_plan,
        l1_features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION,
    )
    assert v5_l1.events["accepted_tiles"] == 1
    assert (
        v5_l1.tile_trace[0]["coverage"]["scope"]
        == "selected-only-l1-plane-direct-geometry-spatial-attribute-intra-tile-2sigma"
    )
    assert v5_l1.tile_trace[0]["coverage"]["l1_plane_maximum_residual"] >= 0.0
    assert bool((torch.linalg.eigvalsh(v5_l1.covariances) >= -1e-6).all())


def test_spatial_attribute_l0_continuity_guard_promotes_incoherent_tile():
    from saes.packed_l0_l1_materializer import (
        CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
        preflight_compact_l0_l1_materialization,
    )

    plan, features = _plan("L0")
    features = features.clone()
    features[0, 0, :, 1, 1] = torch.tensor((10.0, -10.0))
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, intrinsics = _context()
    preflight = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        _route(plan, "L0"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
    )
    assert preflight.events["accepted_tiles"] == 0
    assert int(preflight.promote_full_mask.sum()) == 16
    assert "feature continuity" in next(iter(preflight.events["rejection_reasons"]))
    assert preflight.events["source_nonprobe_s3_attribute_reads"] == 0


def test_spatial_attribute_continuity_guard_promotes_incoherent_l1_tile():
    from saes.packed_l0_l1_materializer import (
        CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION,
        preflight_compact_l0_l1_materialization,
    )

    plan, features = _plan("L1")
    features = features.clone()
    features.zero_()
    for row, column in ((0, 0), (0, 3), (3, 0), (3, 3)):
        features[0, 0, :, row, column] = torch.tensor((1.0, 0.0))
    features[0, 0, :, 1, 1] = torch.tensor((-1.0, 0.0))
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, intrinsics = _context()
    preflight = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        _route(plan, "L1"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION,
    )
    assert preflight.events["accepted_tiles"] == 0
    assert int(preflight.promote_full_mask.sum()) == 16
    assert "feature continuity" in next(iter(preflight.events["rejection_reasons"]))
    assert preflight.events["source_nonprobe_s3_attribute_reads"] == 0


def test_direct_merge_promotes_tile_when_coverage_expansion_is_unsafe(monkeypatch):
    import saes.packed_l0_l1_materializer as materializer

    plan, features = _plan("L0")
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, intrinsics = _context()
    monkeypatch.setattr(materializer, "COMPACT_COVERAGE_MAX_COVARIANCE_SCALE", 1.0)
    preflight = materializer.preflight_compact_l0_l1_materialization(
        packet,
        packed,
        _route(plan, "L0"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=materializer.CONDITIONAL_DIRECT_AGGREGATION,
    )
    assert preflight.events["accepted_tiles"] == 0
    assert int(preflight.promote_full_mask.sum()) == 16
    assert "required coverage expansion" in next(iter(preflight.events["rejection_reasons"]))


def test_failed_preflight_promotes_whole_tile_and_preserves_full_attributes():
    from saes.packed_l0_l1_materializer import (
        apply_compact_l0_l1_materialization,
        preflight_compact_l0_l1_materialization,
        resolve_compact_final_route,
    )

    plan, features = _plan("L0")
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, _intrinsics = _context()
    invalid_intrinsics = torch.zeros(1, 1, 3, 3)
    guard = _route(plan, "L0")
    preflight = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        guard,
        plan,
        features,
        extrinsics,
        invalid_intrinsics,
    )
    assert int(preflight.promote_full_mask.sum()) == 16
    assert preflight.update_dense_slots.numel() == 0
    final_route = resolve_compact_final_route(guard, plan, preflight)
    assert final_route.events["route_counts"] == {"L0": 0, "L1": 0, "Full": 1}
    assert int(final_route.selected_output_mask.sum()) == 16
    assert int(final_route.additional_full_mask.sum()) == 12

    _full_packet, full_packed = _packet_and_packed(torch.ones(1, 4, 4, dtype=torch.bool))
    output = apply_compact_l0_l1_materialization(full_packed, preflight, final_route)
    assert torch.equal(output.means, full_packed.means)
    assert torch.equal(output.covariances, full_packed.covariances)
    assert torch.equal(output.harmonics, full_packed.harmonics)
    assert torch.equal(output.opacities, full_packed.opacities)


def test_final_packet_rejects_guard_only_request_slots():
    from saes.packed_l0_l1_materializer import (
        apply_compact_l0_l1_materialization,
        preflight_compact_l0_l1_materialization,
        resolve_compact_final_route,
    )

    plan, features = _plan("L0")
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, intrinsics = _context()
    guard = _route(plan, "L0")
    preflight = preflight_compact_l0_l1_materialization(
        packet, packed, guard, plan, features, extrinsics, intrinsics
    )
    final_route = resolve_compact_final_route(guard, plan, preflight)
    _request_packet, request_packed = _packet_and_packed(torch.ones(1, 4, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="selected_output_mask"):
        apply_compact_l0_l1_materialization(request_packed, preflight, final_route)


def test_packet_updates_match_existing_assignment_mixture_on_selected_anchors():
    from saes.packed_l0_l1_materializer import (
        ASSIGNMENT_MIXTURE_AGGREGATION,
        preflight_compact_l0_l1_materialization,
    )
    from saes.probe_layout import compute_probe_positions
    from saes.progressive_saes import ProgressiveSAES

    plan, features = _plan("L0")
    packet, packed = _packet_and_packed(plan.selection_mask)
    extrinsics, intrinsics = _context()
    preflight = preflight_compact_l0_l1_materialization(
        packet,
        packed,
        _route(plan, "L0"),
        plan,
        features,
        extrinsics,
        intrinsics,
        aggregation_semantics=ASSIGNMENT_MIXTURE_AGGREGATION,
    )
    anchors = packet.dense_slots.tolist()
    dense = SimpleNamespace(
        means=torch.zeros(1, 16, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1) * 0.02,
        harmonics=torch.zeros(1, 16, 3, 2),
        opacities=torch.full((1, 16), 0.2),
    )
    dense.means[0, anchors] = packed.means
    dense.covariances[0, anchors] = packed.covariances
    dense.harmonics[0, anchors] = packed.harmonics
    dense.opacities[0, anchors] = packed.opacities
    _scores, feature_norm = ProgressiveSAES.classify_tiles_by_features(
        features,
        4,
        4,
        4,
        threshold=0.2,
        per_view=True,
        statistic="raw-probe-mean-channel-variance",
    )
    router = ProgressiveSAES(
        4,
        4,
        initial_tile_size=4,
        feature_var_threshold=0.2,
        depth_std_threshold=0.1,
        view_count=1,
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        decision_semantics="paper-probe-feature-variance-first-hit",
    )
    primary = compute_probe_positions(4)
    router._weighted_moment_match(
        dense,
        anchors,
        {
            (row, column): row * 4 + column
            for row in range(4)
            for column in range(4)
            if (row, column) not in set(primary)
        },
        feature_norm,
        torch.ones(1, 1, 16, 1, 1),
        "L0",
        0,
        0,
        view_index=0,
        retained_positions=primary,
        feature_variance=float(plan.tile_trace[0]["feature_score"]),
    )
    torch.testing.assert_close(preflight.means, dense.means[0, anchors], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        preflight.covariances, dense.covariances[0, anchors], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        preflight.harmonics, dense.harmonics[0, anchors], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        preflight.opacities, dense.opacities[0, anchors], rtol=2e-5, atol=2e-5
    )
