"""Paper-bound compact L0/L1 aggregation for selected Gaussian packets.

This module is deliberately separate from the frozen direct-deletion route.
It never reads a skipped Stage-3 attribute: every non-probe contributor is
constructed from selected probe outputs, S1 features, selected S2 depths, and
the source-native GaussianAdapter ray convention.  A tile is either planned
atomically or promoted to Full before a final packet is produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from saes.guarded_selected_route import (
    ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY,
    GuardedSelectedRoute,
)
from saes.probe_first_schedule import (
    ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    BALANCED_L1_ANCHOR_SEMANTICS,
    LEGACY_L1_ANCHOR_SEMANTICS,
    PAPER_KP_ANCHOR_SEMANTICS,
    IncrementalProbeFirstPlan,
    l1_local_positions_for_tile,
)
from saes.probe_layout import compute_probe_positions
from saes.progressive_saes import (
    PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
    ProgressiveSAES,
    paper_assignment_weights,
)
from saes.sparse_gaussian_consumer import PackedGaussianAttributes, SparseRawGaussianPacket


MATERIALIZER_SCHEMA_VERSION = "saes-packed-l0-l1-materializer-v4"

# A native producer can legitimately saturate its floating-point sigmoid at
# exactly one.  Compact attribute interpolation and V4's logit-domain replay
# cannot represent that endpoint without changing it, so the whole tile keeps
# the native Full path instead.
NATIVE_OPACITY_ENDPOINT_FULL_REASON = "native_opacity_endpoint_requires_full"

# v1 is retained only to reproduce and diagnose the already-recorded failed
# candidate. It synthesizes a skipped descriptor from every probe and then
# applies the same assignment again while updating a receiver.
ASSIGNMENT_MIXTURE_AGGREGATION = (
    "probe-camera-residual-assignment-mixture-first-second-moment-range-constrained-v1"
)

# v2 is the selected-only closure used by the active repair line. A skipped
# position contributes to receiver p only through p's own Adapter offset,
# depth, and attributes, weighted once by r(i -> p).
CONDITIONAL_DIRECT_AGGREGATION = (
    "probe-conditional-adapter-offset-direct-merge-coverage-closed-v2"
)

# v3 forms one spatially continuous virtual Gaussian field from selected probe
# outputs before applying the paper's bilateral assignment once. Its spatial
# interpolation weights are deliberately independent of r(i -> p), avoiding
# the rejected R^T R construction while allowing SH and alpha to vary inside
# a tile.
SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION = (
    "probe-spatial-virtual-single-assignment-coverage-closed-v3"
)

# v4 keeps v2's native per-receiver geometry, but restores the paper's
# range-constrained SH/opacity aggregation from a spatially continuous field.
# This deliberately does not revive v3's rejected spatial geometry.
CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION = (
    "probe-conditional-adapter-offset-spatial-attribute-merge-coverage-closed-v4"
)

# v5 retains the v4 attribute field and replaces L1's constant-depth ray
# transport with the local planar surface described by the selected anchors.
# The GaussianAdapter scales linearly with depth, so virtual covariances are
# scaled by the squared ray-plane depth ratio before moment matching.
CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION = (
    "probe-conditional-adapter-offset-l1-plane-spatial-attribute-merge-coverage-closed-v5"
)

# v6 retains v4's direct native geometry and applies the target-free S1
# interpolation-continuity closure to every compact level. It is deliberately
# separate from v5 because the plane audit did not materially improve coverage.
CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION = (
    "probe-conditional-adapter-offset-spatial-attribute-continuity-merge-coverage-closed-v6"
)
# Keep the layout ablation isolated from V6's all-level continuity guard. V4
# is the last quality-measured aggregation and the only changed mechanism in
# the balanced experiment is the selected L1 anchor geometry.
DEFAULT_COMPACT_AGGREGATION = CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION

# A fixed source-context support closure may expand a direct moment covariance
# only enough to contain its assigned two-sigma virtual supports. Larger
# expansions are treated as geometric uncertainty and promote the tile Full.
COMPACT_COVERAGE_MAX_COVARIANCE_SCALE = 16.0

# A selected-anchor spatial reconstruction must explain every skipped S1
# feature within the anchor set's own feature diameter. This scale-free
# continuity condition has no target or dense-S3 input; ambiguous tiles are
# materialized on Full rather than absorbing an unsupported attribute field.
COMPACT_FEATURE_INTERPOLATION_MAX_RELATIVE_RESIDUAL = 1.0

# V16 does not inspect the true omitted descriptor.  It hides each of the
# three retained adaptive-center anchors and checks whether V4's selected-only
# spatial SH/opacity field can replay that selected anchor from the other 14.
SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CERTIFICATE = (
    "selected-anchor-v4-spatial-attribute-center-loo-v1"
)
SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CENTER_POLICY = (
    "adaptive-retained-center-triplet-v1"
)
SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CENTERS = ((1, 1), (1, 2), (2, 1), (2, 2))


@dataclass(frozen=True)
class CompactMaterializationPreflight:
    """Immutable tile updates derived only from selected L0/L1 anchors."""

    update_dense_slots: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor
    promote_full_mask: torch.Tensor
    tile_trace: tuple[dict[str, Any], ...]
    events: dict[str, Any]


@dataclass(frozen=True)
class CompactFinalRoute:
    """Final output and producer masks after compact-materializer preflight."""

    selected_output_mask: torch.Tensor
    additional_full_mask: torch.Tensor
    raw_head_request_mask: torch.Tensor
    tile_trace: tuple[dict[str, Any], ...]
    events: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("compact materialization trace must be JSON-serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def _mask_sha256(mask: torch.Tensor) -> str:
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise ValueError("compact materialization mask must be boolean")
    value = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _full_positions(tile_size: int) -> list[tuple[int, int]]:
    return [(row, column) for row in range(tile_size) for column in range(tile_size)]


def _tile_slots(
    *,
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    tile_size: int,
    positions: list[tuple[int, int]],
) -> list[int]:
    slots: list[int] = []
    for local_y, local_x in positions:
        if not 0 <= local_y < tile_size or not 0 <= local_x < tile_size:
            raise ValueError("compact materializer received an invalid local anchor")
        row = tile_y * tile_size + local_y
        column = tile_x * tile_size + local_x
        if not 0 <= row < height or not 0 <= column < width:
            raise ValueError("compact materializer anchor falls outside its image")
        slots.append(view * (height * width) + row * width + column)
    return slots


def _mark_tile(mask: torch.Tensor, *, view: int, tile_y: int, tile_x: int, tile_size: int) -> None:
    mask[
        view,
        tile_y * tile_size : (tile_y + 1) * tile_size,
        tile_x * tile_size : (tile_x + 1) * tile_size,
    ] = True


def _require_fixed_plan(plan: IncrementalProbeFirstPlan) -> tuple[int, int, int, str]:
    if not isinstance(plan, IncrementalProbeFirstPlan):
        raise TypeError("compact materializer requires an incremental probe-first plan")
    events = plan.events
    if events.get("contract_version") != "saes-incremental-probe-first-plan-v1":
        raise ValueError("compact materializer requires a source-bound probe-first plan")
    tile_size = events.get("tile_size")
    if not isinstance(tile_size, int) or tile_size != 4:
        raise ValueError("compact materializer currently supports exactly T=4")
    if any(
        not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 3
        for mask in (plan.primary_mask, plan.secondary_mask, plan.full_mask, plan.selection_mask)
    ):
        raise ValueError("compact materializer plan masks are invalid")
    if not torch.equal(
        plan.selection_mask, plan.primary_mask | plan.secondary_mask | plan.full_mask
    ):
        raise ValueError("compact materializer plan mask union is inconsistent")
    views, height, width = plan.selection_mask.shape
    if height % tile_size or width % tile_size:
        raise ValueError("compact materializer requires tile-aligned dimensions")
    semantics = events.get("l1_anchor_semantics", LEGACY_L1_ANCHOR_SEMANTICS)
    if semantics not in {
        PAPER_KP_ANCHOR_SEMANTICS,
        LEGACY_L1_ANCHOR_SEMANTICS,
        BALANCED_L1_ANCHOR_SEMANTICS,
        ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
    }:
        raise ValueError("compact materializer received an unknown L1 anchor semantics")
    expected_l1 = (
        4
        if semantics == PAPER_KP_ANCHOR_SEMANTICS
        else 15
        if semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS
        else 12
    )
    if events.get("l0_anchor_count") != 4 or events.get("l1_anchor_count") != expected_l1:
        raise ValueError("compact materializer plan anchor counts do not match its semantics")
    return views, height, width, semantics


def _validate_selected_source(
    packet: SparseRawGaussianPacket,
    packed: PackedGaussianAttributes,
    plan: IncrementalProbeFirstPlan,
    *,
    views: int,
    height: int,
    width: int,
) -> dict[int, int]:
    if not isinstance(packet, SparseRawGaussianPacket) or not isinstance(
        packed, PackedGaussianAttributes
    ):
        raise TypeError("compact materializer requires selected packet and packed attributes")
    count = int(packet.dense_slots.numel())
    if (
        count < 1
        or packet.dense_slots.shape != (count,)
        or packed.dense_slots.shape != (count,)
        or not torch.equal(packet.dense_slots, packed.dense_slots)
        or packet.raw_descriptors.ndim != 2
        or packet.raw_descriptors.shape[0] != count
        or packet.raw_descriptors.shape[1] < 2
        or packet.coordinates.shape != (count, 2)
        or packet.depths.shape != (count,)
        or packed.batch_indices.shape != (count,)
        or packed.means.shape != (count, 3)
        or packed.covariances.shape != (count, 3, 3)
        or packed.harmonics.ndim != 3
        or packed.harmonics.shape[:2] != (count, 3)
        or packed.opacities.shape != (count,)
    ):
        raise ValueError("compact materializer selected packet tensors are inconsistent")
    values = (
        packet.raw_descriptors,
        packet.coordinates,
        packet.depths,
        packed.means,
        packed.covariances,
        packed.harmonics,
        packed.opacities,
    )
    if any(value.device != packet.raw_descriptors.device for value in values):
        raise ValueError("compact materializer selected inputs must share one device")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("compact materializer selected inputs must be finite")
    if not bool((packet.depths > 0.0).all()):
        raise ValueError("compact materializer requires positive selected probe depths")
    if not bool((packed.opacities >= 0.0).all()) or not bool((packed.opacities <= 1.0).all()):
        raise ValueError("compact materializer requires selected opacities in [0, 1]")
    if not bool((packed.batch_indices == 0).all()):
        raise ValueError("compact materializer currently supports B=1 only")
    expected = plan.selection_mask.nonzero(as_tuple=False)
    expected_slots = expected[:, 0] * (height * width) + expected[:, 1] * width + expected[:, 2]
    if not torch.equal(packet.dense_slots, expected_slots.to(packet.dense_slots.device)):
        raise ValueError("compact materializer packet does not match plan selection")
    if packet.source_trace.get("source_bound") is not True:
        raise ValueError("compact materializer packet is not source-bound")
    if packed.source_trace.get("source_bound") is not True:
        raise ValueError("compact materializer attributes are not source-bound")
    if packet.dense_slots.numel() > 1 and not bool(
        (packet.dense_slots[1:] > packet.dense_slots[:-1]).all()
    ):
        raise ValueError("compact materializer packet slots must be strictly increasing")
    total_slots = views * height * width
    slots = packet.dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if any(slot < 0 or slot >= total_slots for slot in slots) or len(set(slots)) != len(slots):
        raise ValueError("compact materializer packet slots are invalid")
    return {slot: index for index, slot in enumerate(slots)}


def _tile_feature_variance(
    record: Mapping[str, Any],
    decision_semantics: str,
    feature_statistic: str,
) -> float:
    score = record.get("feature_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("compact materializer tile has no feature score")
    score = float(score)
    if not torch.isfinite(torch.tensor(score)) or score < 0.0:
        raise ValueError("compact materializer feature score is invalid")
    if feature_statistic == "normalized-probe-vector-standard-deviation":
        return score * score
    if decision_semantics == PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS:
        raise ValueError("normalized paper route has an inconsistent feature statistic")
    return score


def _project_psd(covariance: torch.Tensor) -> torch.Tensor:
    covariance = (covariance + covariance.mT) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if not bool(torch.isfinite(eigenvalues).all()):
        raise ValueError("compact materializer covariance eigendecomposition is non-finite")
    projected = eigenvectors @ torch.diag(eigenvalues.clamp_min(1e-8)) @ eigenvectors.mT
    return (projected + projected.mT) * 0.5


def _validate_source_covariances(covariances: torch.Tensor) -> None:
    if covariances.ndim != 3 or covariances.shape[1:] != (3, 3):
        raise ValueError("compact materializer covariance shape is invalid")
    symmetric = (covariances - covariances.mT).abs().amax()
    if not bool(torch.isfinite(symmetric)) or float(symmetric.item()) > 1e-4:
        raise ValueError("compact materializer source covariance is not symmetric")
    values = torch.linalg.eigvalsh((covariances + covariances.mT) * 0.5)
    if not bool(torch.isfinite(values).all()) or bool((values < -1e-6).any()):
        raise ValueError("compact materializer source covariance is not PSD")


def _camera_residual_pseudo_means(
    *,
    source_means: torch.Tensor,
    source_depths: torch.Tensor,
    source_positions: list[tuple[int, int]],
    assignment_matrix: torch.Tensor,
    target_positions: list[tuple[int, int]],
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    view: int,
    width: int,
    height: int,
) -> torch.Tensor:
    """Reproduce the existing selected-anchor camera-residual mixture."""
    if not target_positions:
        return source_means.new_empty((0, 3))
    count = source_means.shape[0]
    if (
        source_means.shape != (count, 3)
        or source_depths.shape != (count,)
        or len(source_positions) != count
        or assignment_matrix.shape != (len(target_positions), count)
    ):
        raise ValueError("compact materializer camera-residual inputs are inconsistent")
    device = source_means.device
    dtype = source_means.dtype
    extrinsic = context_extrinsics[0, view].to(device=device, dtype=dtype)
    intrinsic = context_intrinsics[0, view].to(device=device, dtype=dtype)
    if not bool(torch.isfinite(extrinsic).all()) or not bool(torch.isfinite(intrinsic).all()):
        raise ValueError("compact materializer camera geometry is non-finite")

    def world_points(
        positions: list[tuple[int, int]], depths: torch.Tensor
    ) -> torch.Tensor:
        rows = torch.tensor([row for row, _ in positions], device=device, dtype=dtype)
        columns = torch.tensor(
            [column for _, column in positions], device=device, dtype=dtype
        )
        coordinates = torch.stack(
            (
                (columns + 0.5) / width,
                (rows + 0.5) / height,
                torch.ones_like(rows),
            ),
            dim=1,
        )
        try:
            directions = torch.linalg.solve(intrinsic, coordinates.mT).mT
        except RuntimeError as error:
            raise ValueError("compact materializer camera solve failed") from error
        norms = directions.norm(dim=1, keepdim=True)
        if not bool(torch.isfinite(directions).all()) or bool((norms <= 1e-8).any()):
            raise ValueError("compact materializer camera ray is invalid")
        directions = directions / norms
        world_directions = torch.einsum("ij,nj->ni", extrinsic[:3, :3], directions)
        values = extrinsic[:3, 3].unsqueeze(0) + world_directions * depths.unsqueeze(1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("compact materializer camera lift is non-finite")
        return values

    source_rays = world_points(source_positions, source_depths)
    target_depths = assignment_matrix @ source_depths
    target_rays = world_points(target_positions, target_depths)
    residuals = source_means - source_rays
    values = target_rays + assignment_matrix @ residuals
    if not bool(torch.isfinite(values).all()):
        raise ValueError("compact materializer camera residual mixture is non-finite")
    return values


def _pixel_centres(
    positions: list[tuple[int, int]],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not positions:
        return torch.empty(0, 2, device=device, dtype=dtype)
    if any(row < 0 or row >= height or column < 0 or column >= width for row, column in positions):
        raise ValueError("compact materializer position falls outside the image")
    rows = torch.tensor([row for row, _ in positions], device=device, dtype=dtype)
    columns = torch.tensor(
        [column for _, column in positions], device=device, dtype=dtype
    )
    return torch.stack(
        ((columns + 0.5) / width, (rows + 0.5) / height),
        dim=1,
    )


def _lift_context_coordinates(
    coordinates: torch.Tensor,
    depths: torch.Tensor,
    *,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    view: int,
) -> torch.Tensor:
    """Lift source-native image coordinates with TranSplat's ray convention."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("compact materializer coordinates must be [N,2]")
    depths = depths.reshape(-1).to(device=coordinates.device, dtype=coordinates.dtype)
    if depths.numel() != coordinates.shape[0]:
        raise ValueError("compact materializer coordinate/depth counts do not match")
    if (
        not bool(torch.isfinite(coordinates).all())
        or not bool(torch.isfinite(depths).all())
        or bool((depths <= 0.0).any())
    ):
        raise ValueError("compact materializer native coordinate lift is invalid")
    extrinsic = context_extrinsics[0, view].to(
        device=coordinates.device, dtype=coordinates.dtype
    )
    intrinsic = context_intrinsics[0, view].to(
        device=coordinates.device, dtype=coordinates.dtype
    )
    if not bool(torch.isfinite(extrinsic).all()) or not bool(torch.isfinite(intrinsic).all()):
        raise ValueError("compact materializer camera geometry is non-finite")
    try:
        determinant = torch.linalg.det(intrinsic)
    except RuntimeError as error:
        raise ValueError("compact materializer intrinsic determinant failed") from error
    tiny = torch.as_tensor(
        torch.finfo(coordinates.dtype).eps,
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    if not bool(torch.isfinite(determinant)) or bool(determinant.abs() <= tiny):
        raise ValueError("compact materializer intrinsic is singular")
    homogeneous = torch.cat(
        (
            coordinates,
            torch.ones((coordinates.shape[0], 1), device=coordinates.device, dtype=coordinates.dtype),
        ),
        dim=1,
    )
    try:
        directions = torch.linalg.solve(intrinsic, homogeneous.mT).mT
    except RuntimeError as error:
        raise ValueError("compact materializer camera solve failed") from error
    norms = directions.norm(dim=1, keepdim=True)
    if not bool(torch.isfinite(directions).all()) or bool((norms <= tiny).any()):
        raise ValueError("compact materializer camera ray is invalid")
    directions = directions / norms
    world_directions = torch.einsum("ij,nj->ni", extrinsic[:3, :3], directions)
    values = extrinsic[:3, 3].unsqueeze(0) + world_directions * depths.unsqueeze(1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("compact materializer native coordinate lift is non-finite")
    return values


def _fit_l1_anchor_plane(
    source_means: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Fit one finite local plane from the selected L1 anchor means.

    The fit is deliberately geometry-only. It uses no skipped descriptor,
    target camera, or target RGB. A rank-deficient selected layout cannot
    identify a plane and must stay on the Full path.
    """
    if (
        source_means.ndim != 2
        or source_means.shape[1] != 3
        or source_means.shape[0] < 3
        or not bool(torch.isfinite(source_means).all())
    ):
        raise ValueError("compact materializer L1 plane anchors are invalid")
    origin = source_means.mean(dim=0)
    centered = source_means - origin
    covariance = centered.mT @ centered / float(source_means.shape[0])
    covariance = (covariance + covariance.mT) * 0.5
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    except RuntimeError as error:
        raise ValueError("compact materializer L1 plane eigendecomposition failed") from error
    if not bool(torch.isfinite(eigenvalues).all()):
        raise ValueError("compact materializer L1 plane spectrum is non-finite")
    scale = eigenvalues[-1].abs().clamp_min(1.0)
    numerical_rank_floor = torch.finfo(source_means.dtype).eps * scale * 64.0
    if bool(eigenvalues[1] <= numerical_rank_floor):
        raise ValueError("compact materializer L1 plane anchors are rank-deficient")
    normal = eigenvectors[:, 0]
    normal_norm = normal.norm()
    if not bool(torch.isfinite(normal_norm)) or bool(normal_norm <= numerical_rank_floor):
        raise ValueError("compact materializer L1 plane normal is invalid")
    normal = normal / normal_norm
    residual = (centered @ normal).abs().amax()
    if not bool(torch.isfinite(residual)):
        raise ValueError("compact materializer L1 plane residual is non-finite")
    return origin, normal, {
        "maximum_plane_residual": float(residual.item()),
        "planarity_eigenvalue_ratio": float(
            (eigenvalues[0].clamp_min(0.0) / eigenvalues[1]).item()
        ),
    }


def _intersect_context_rays_with_plane(
    coordinates: torch.Tensor,
    *,
    plane_origin: torch.Tensor,
    plane_normal: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    view: int,
) -> torch.Tensor:
    """Return positive Euclidean distances where producer rays hit one plane."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("compact materializer L1 plane coordinates are invalid")
    if plane_origin.shape != (3,) or plane_normal.shape != (3,):
        raise ValueError("compact materializer L1 plane parameters are invalid")
    if not bool(torch.isfinite(coordinates).all()) or not bool(torch.isfinite(plane_origin).all()) or not bool(torch.isfinite(plane_normal).all()):
        raise ValueError("compact materializer L1 plane inputs are non-finite")
    dtype = coordinates.dtype
    device = coordinates.device
    extrinsic = context_extrinsics[0, view].to(device=device, dtype=dtype)
    intrinsic = context_intrinsics[0, view].to(device=device, dtype=dtype)
    if not bool(torch.isfinite(extrinsic).all()) or not bool(torch.isfinite(intrinsic).all()):
        raise ValueError("compact materializer L1 plane camera is non-finite")
    tiny = torch.as_tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
    try:
        intrinsic_determinant = torch.linalg.det(intrinsic)
    except RuntimeError as error:
        raise ValueError("compact materializer L1 plane intrinsic determinant failed") from error
    if not bool(torch.isfinite(intrinsic_determinant)) or bool(intrinsic_determinant.abs() <= tiny):
        raise ValueError("compact materializer L1 plane intrinsic is singular")
    homogeneous = torch.cat(
        (coordinates, torch.ones((coordinates.shape[0], 1), device=device, dtype=dtype)),
        dim=1,
    )
    try:
        camera_directions = torch.linalg.solve(intrinsic, homogeneous.mT).mT
    except RuntimeError as error:
        raise ValueError("compact materializer L1 plane camera solve failed") from error
    norms = camera_directions.norm(dim=1, keepdim=True)
    if not bool(torch.isfinite(camera_directions).all()) or bool((norms <= tiny).any()):
        raise ValueError("compact materializer L1 plane ray is invalid")
    world_directions = torch.einsum(
        "ij,nj->ni", extrinsic[:3, :3], camera_directions / norms
    )
    normal = plane_normal.to(device=device, dtype=dtype)
    numerator = torch.dot(
        plane_origin.to(device=device, dtype=dtype) - extrinsic[:3, 3], normal
    )
    denominator = world_directions @ normal
    parallel_tolerance = torch.finfo(dtype).eps * 64.0
    if (
        not bool(torch.isfinite(numerator))
        or not bool(torch.isfinite(denominator).all())
        or bool((denominator.abs() <= parallel_tolerance).any())
    ):
        raise ValueError("compact materializer L1 plane ray is parallel")
    depths = numerator / denominator
    if not bool(torch.isfinite(depths).all()) or bool((depths <= tiny).any()):
        raise ValueError("compact materializer L1 plane ray intersection is invalid")
    return depths


def _selected_native_offsets(
    *,
    packet: SparseRawGaussianPacket,
    packed: PackedGaussianAttributes,
    anchor_indices: list[int],
    source_positions: list[tuple[int, int]],
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    view: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Validate selected Adapter geometry and return each bounded raw offset."""
    count = len(anchor_indices)
    if count < 1 or len(source_positions) != count:
        raise ValueError("compact materializer selected native geometry is incomplete")
    indices = torch.tensor(
        anchor_indices, device=packet.raw_descriptors.device, dtype=torch.long
    )
    source_coordinates = packet.coordinates[indices]
    source_depths = packet.depths[indices].to(
        device=source_coordinates.device, dtype=source_coordinates.dtype
    )
    source_means = packed.means[indices].to(
        device=source_coordinates.device, dtype=source_coordinates.dtype
    )
    if not bool(torch.isfinite(source_coordinates).all()):
        raise ValueError("compact materializer selected coordinates are non-finite")

    # Import the producer's shared coordinate routine rather than duplicating
    # the raw-offset convention. This reads only selected raw descriptors.
    from transplat.src.model.encoder.encoder_trans import (
        gaussian_adapter_coordinates_from_raw_offsets,
    )

    pixel_indices = (packet.dense_slots[indices] % (height * width)).to(
        device=packet.raw_descriptors.device,
        dtype=torch.int64,
    )
    expected_coordinates = gaussian_adapter_coordinates_from_raw_offsets(
        packet.raw_descriptors[indices, :2].sigmoid(),
        image_shape=(height, width),
        pixel_indices=pixel_indices,
    )
    if not torch.allclose(
        source_coordinates,
        expected_coordinates.to(device=source_coordinates.device, dtype=source_coordinates.dtype),
        rtol=2e-5,
        atol=2e-5,
    ):
        raise ValueError("compact materializer selected coordinates drift from native offsets")
    lifted_means = _lift_context_coordinates(
        source_coordinates,
        source_depths,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        view=view,
    )
    if not torch.allclose(source_means, lifted_means, rtol=5e-5, atol=5e-5):
        raise ValueError("compact materializer selected means drift from native geometry")
    centres = _pixel_centres(
        source_positions,
        height=height,
        width=width,
        device=source_coordinates.device,
        dtype=source_coordinates.dtype,
    )
    offsets = source_coordinates - centres
    limits = torch.tensor(
        (0.5 / width, 0.5 / height),
        device=offsets.device,
        dtype=offsets.dtype,
    )
    tolerance = torch.as_tensor(2e-5, device=offsets.device, dtype=offsets.dtype)
    if not bool(torch.isfinite(offsets).all()) or bool((offsets.abs() > limits + tolerance).any()):
        raise ValueError("compact materializer selected native offset is out of range")
    return offsets


def _spatial_virtual_weights(
    target_positions: list[tuple[int, int]],
    source_positions: list[tuple[int, int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a deterministic spatial field independent of bilateral assignment.

    Four corner probes use bilinear interpolation, which is continuous across
    the tile and exact at each corner. Larger engineering anchor layouts use a
    fixed inverse-squared-distance partition of unity; this remains separate
    from the paper's feature/depth bilateral assignment used for absorption.
    """
    if not target_positions or not source_positions:
        raise ValueError("compact materializer spatial virtual field is empty")
    if len(set(source_positions)) != len(source_positions):
        raise ValueError("compact materializer spatial virtual anchors are duplicated")
    rows = [row for row, _ in source_positions]
    columns = [column for _, column in source_positions]
    top, bottom = min(rows), max(rows)
    left, right = min(columns), max(columns)
    corners = {(top, left), (top, right), (bottom, left), (bottom, right)}
    if len(source_positions) == 4 and set(source_positions) == corners and top < bottom and left < right:
        weights = torch.zeros(
            (len(target_positions), len(source_positions)), device=device, dtype=dtype
        )
        source_to_index = {position: index for index, position in enumerate(source_positions)}
        for target_index, (row, column) in enumerate(target_positions):
            u = torch.as_tensor((column - left) / (right - left), device=device, dtype=dtype)
            v = torch.as_tensor((row - top) / (bottom - top), device=device, dtype=dtype)
            weights[target_index, source_to_index[(top, left)]] = (1.0 - u) * (1.0 - v)
            weights[target_index, source_to_index[(top, right)]] = u * (1.0 - v)
            weights[target_index, source_to_index[(bottom, left)]] = (1.0 - u) * v
            weights[target_index, source_to_index[(bottom, right)]] = u * v
    else:
        target = torch.tensor(target_positions, device=device, dtype=dtype)
        source = torch.tensor(source_positions, device=device, dtype=dtype)
        distances = (target.unsqueeze(1) - source.unsqueeze(0)).square().sum(dim=2)
        exact = distances <= torch.finfo(dtype).eps
        inverse = (distances + 1e-6).reciprocal()
        weights = inverse / inverse.sum(dim=1, keepdim=True)
        if bool(exact.any()):
            exact_rows = exact.any(dim=1)
            weights[exact_rows] = exact[exact_rows].to(dtype=dtype)
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights < 0.0).any())
        or not torch.allclose(
            weights.sum(dim=1),
            torch.ones(weights.shape[0], device=device, dtype=dtype),
            atol=1e-5,
            rtol=1e-5,
        )
    ):
        raise ValueError("compact materializer spatial virtual weights are invalid")
    return weights


def _anchor_support_scale(values: torch.Tensor) -> torch.Tensor:
    """Return a local, selected-anchor-only scale for a replay residual."""
    if values.ndim < 2 or values.shape[0] < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("compact materializer replay support is invalid")
    flattened = values.reshape(values.shape[0], -1)
    distances = torch.pdist(flattened)
    if distances.numel() == 0 or not bool(torch.isfinite(distances).all()):
        raise ValueError("compact materializer replay support distances are invalid")
    magnitude = flattened.abs().amax().clamp_min(1.0)
    numerical_floor = torch.finfo(flattened.dtype).eps * 1024.0 * magnitude
    return torch.maximum(distances.median(), numerical_floor)


def _finite_scalar_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if not all(torch.isfinite(torch.tensor(value)) and value >= 0.0 for value in ordered):
        raise ValueError("compact materializer replay summary is invalid")

    def quantile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(ordered),
        "minimum": quantile(0.0),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "maximum": quantile(1.0),
    }


def _opacity_logits(opacities: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(opacities).all()) or bool((opacities < 0.0).any()) or bool(
        (opacities >= 1.0).any()
    ):
        raise ValueError("compact materializer replay opacity is invalid")
    epsilon = torch.finfo(opacities.dtype).eps * 16.0
    bounded = opacities.clamp(min=epsilon, max=1.0 - epsilon)
    return torch.log(bounded) - torch.log1p(-bounded)


def selected_anchor_v4_attribute_loo_certificate(
    *,
    packet: SparseRawGaussianPacket,
    packed: PackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    tile_size: int,
    anchor_positions: list[tuple[int, int]],
) -> dict[str, Any]:
    """Replay the three retained adaptive centers through V4's attribute field.

    The native values used as labels are already selected L1 anchors.  The
    fourth center, which is the real compact omission, is deliberately never
    looked up in ``packed`` or ``packet``.
    """
    if tile_size != 4 or len(anchor_positions) != 15:
        raise ValueError("compact materializer V4 replay requires adaptive L1-15 anchors")
    retained_centers = [
        position for position in SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CENTERS if position in anchor_positions
    ]
    omitted_centers = [
        position
        for position in SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CENTERS
        if position not in anchor_positions
    ]
    if len(retained_centers) != 3 or len(omitted_centers) != 1:
        raise ValueError("compact materializer V4 replay has invalid adaptive center layout")
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=tile_size,
        positions=anchor_positions,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("compact materializer V4 replay lacks a selected anchor")
    anchor_indices = [slot_to_index[slot] for slot in anchor_slots]
    source_harmonics = packed.harmonics[anchor_indices]
    source_opacities = packed.opacities[anchor_indices]
    if not bool(torch.isfinite(source_harmonics).all()):
        raise ValueError("compact materializer V4 replay harmonics are non-finite")
    _opacity_logits(source_opacities)
    global_positions = [
        (tile_y * tile_size + row, tile_x * tile_size + column)
        for row, column in anchor_positions
    ]
    records: list[dict[str, float | list[int]]] = []
    risks: list[torch.Tensor] = []
    for held_out in retained_centers:
        held_out_offset = anchor_positions.index(held_out)
        source_offsets = [
            offset for offset in range(len(anchor_positions)) if offset != held_out_offset
        ]
        source_positions = [global_positions[offset] for offset in source_offsets]
        target_position = [
            (tile_y * tile_size + held_out[0], tile_x * tile_size + held_out[1])
        ]
        spatial_weights = _spatial_virtual_weights(
            target_position,
            source_positions,
            device=source_harmonics.device,
            dtype=source_harmonics.dtype,
        )
        flattened = source_harmonics[source_offsets].reshape(len(source_offsets), -1)
        predicted_harmonics = (spatial_weights @ flattened).reshape_as(
            source_harmonics[held_out_offset]
        )
        predicted_opacity = (spatial_weights @ source_opacities[source_offsets]).reshape(())
        actual_harmonics = source_harmonics[held_out_offset]
        actual_opacity = source_opacities[held_out_offset]
        harmonic_error = (predicted_harmonics - actual_harmonics).norm() / _anchor_support_scale(
            flattened
        )
        source_logits = _opacity_logits(source_opacities[source_offsets]).reshape(-1, 1)
        opacity_error = (
            (_opacity_logits(predicted_opacity.reshape(1)) - _opacity_logits(actual_opacity.reshape(1)))
            .abs()
            .reshape(())
            / _anchor_support_scale(source_logits)
        )
        risk = torch.maximum(harmonic_error, opacity_error)
        if not bool(torch.isfinite(risk)) or bool(risk < 0.0):
            raise ValueError("compact materializer V4 replay risk is invalid")
        risks.append(risk)
        records.append(
            {
                "held_out_local_position": [int(held_out[0]), int(held_out[1])],
                "harmonic_relative_error": float(harmonic_error.item()),
                "opacity_logit_relative_error": float(opacity_error.item()),
                "risk": float(risk.item()),
            }
        )
    values = torch.stack(risks).to(dtype=torch.float32)
    q75_risk = torch.quantile(values, 0.75)
    if not bool(torch.isfinite(q75_risk)) or bool(q75_risk < 0.0):
        raise ValueError("compact materializer V4 replay quantile is invalid")
    return {
        "checked": True,
        "certificate": SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CERTIFICATE,
        "center_policy": SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CENTER_POLICY,
        "held_out_center_count": len(records),
        "held_out_center_records": records,
        "q75_risk": float(q75_risk.item()),
        "selected_anchor_s3_attribute_label_reads": len(records),
        "nonprobe_s3_attribute_reads": 0,
        "actual_omitted_local_position": [
            int(omitted_centers[0][0]),
            int(omitted_centers[0][1]),
        ],
    }


def _feature_interpolation_continuity(
    *,
    view_features: torch.Tensor,
    source_positions: list[tuple[int, int]],
    target_positions: list[tuple[int, int]],
    spatial_weights: torch.Tensor,
) -> dict[str, float | bool]:
    """Check whether selected anchors explain skipped S1 features continuously.

    The comparison is normalized by the selected anchors' own feature
    diameter. It is an engineering safety closure around the paper's
    probe-only aggregation, not a new L0/L1 routing statistic. A zero-diameter
    anchor field permits only numerical roundoff at skipped positions.
    """
    if (
        not torch.is_tensor(view_features)
        or view_features.ndim != 3
        or not source_positions
        or not target_positions
        or spatial_weights.shape != (len(target_positions), len(source_positions))
        or spatial_weights.device != view_features.device
    ):
        raise ValueError("compact materializer feature continuity inputs are invalid")
    height, width = view_features.shape[1:]
    if any(
        row < 0 or row >= height or column < 0 or column >= width
        for row, column in (*source_positions, *target_positions)
    ):
        raise ValueError("compact materializer feature continuity position is invalid")
    anchors = torch.stack(
        [view_features[:, row, column] for row, column in source_positions], dim=0
    )
    targets = torch.stack(
        [view_features[:, row, column] for row, column in target_positions], dim=0
    )
    if not bool(torch.isfinite(anchors).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("compact materializer feature continuity is non-finite")
    reconstruction = spatial_weights @ anchors
    residuals = (targets - reconstruction).norm(dim=1)
    if anchors.shape[0] > 1:
        diameter = torch.pdist(anchors).amax()
    else:
        diameter = anchors.new_zeros(())
    scale = anchors.norm(dim=1).amax().clamp_min(1.0)
    numerical_tolerance = torch.finfo(anchors.dtype).eps * scale * 64.0
    permitted = torch.maximum(diameter, numerical_tolerance)
    maximum_residual = residuals.amax()
    relative_residual = maximum_residual / permitted
    if not bool(torch.isfinite(relative_residual)):
        raise ValueError("compact materializer feature continuity residual is non-finite")
    return {
        "passed": bool(
            relative_residual
            <= COMPACT_FEATURE_INTERPOLATION_MAX_RELATIVE_RESIDUAL
        ),
        "anchor_feature_diameter": float(diameter.item()),
        "maximum_interpolation_residual": float(maximum_residual.item()),
        "maximum_relative_interpolation_residual": float(relative_residual.item()),
    }


def _project_context_gaussians(
    means: torch.Tensor,
    covariances: torch.Tensor,
    *,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    view: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project finite world-space Gaussian support into one source context."""
    if means.ndim != 2 or means.shape[1] != 3 or covariances.shape != (means.shape[0], 3, 3):
        raise ValueError("compact materializer projected coverage inputs are invalid")
    if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(covariances).all()):
        raise ValueError("compact materializer projected coverage is non-finite")
    extrinsic = context_extrinsics[0, view].to(device=means.device, dtype=means.dtype)
    intrinsic = context_intrinsics[0, view].to(device=means.device, dtype=means.dtype)
    rotation = extrinsic[:3, :3]
    origin = extrinsic[:3, 3]
    camera_points = (means - origin) @ rotation
    tiny = torch.as_tensor(torch.finfo(means.dtype).tiny, device=means.device, dtype=means.dtype)
    if not bool(torch.isfinite(camera_points).all()) or bool((camera_points[:, 2] <= tiny).any()):
        raise ValueError("compact materializer projected coverage has invalid depth")
    homogeneous = camera_points @ intrinsic.mT
    denominator = homogeneous[:, 2]
    if not bool(torch.isfinite(homogeneous).all()) or bool((denominator.abs() <= tiny).any()):
        raise ValueError("compact materializer projected coverage has invalid projection")
    centres = homogeneous[:, :2] / denominator.unsqueeze(1)
    derivative_camera = torch.empty((means.shape[0], 2, 3), device=means.device, dtype=means.dtype)
    for axis in range(2):
        derivative_camera[:, axis] = (
            intrinsic[axis].unsqueeze(0) * denominator.unsqueeze(1)
            - homogeneous[:, axis].unsqueeze(1) * intrinsic[2].unsqueeze(0)
        ) / denominator.square().unsqueeze(1)
    jacobians = derivative_camera @ rotation.mT
    projected = jacobians @ covariances @ jacobians.mT
    projected = (projected + projected.mT) * 0.5
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(projected)
    except RuntimeError as error:
        raise ValueError("compact materializer projected coverage eigendecomposition failed") from error
    if not bool(torch.isfinite(eigenvalues).all()) or bool((eigenvalues <= tiny).any()):
        raise ValueError("compact materializer projected coverage is not positive definite")
    projected = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.mT
    return centres, (projected + projected.mT) * 0.5


def _ellipse_containment_certificate(
    *,
    virtual_means: torch.Tensor,
    virtual_covariances: torch.Tensor,
    owner_indices: torch.Tensor,
    merged_means: torch.Tensor,
    merged_covariances: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    view: int,
) -> dict[str, Any]:
    """Require each virtual two-sigma support ellipse to fit its merged owner."""
    if owner_indices.ndim != 1 or owner_indices.shape[0] != virtual_means.shape[0]:
        raise ValueError("compact materializer coverage owners are invalid")
    if owner_indices.numel() == 0:
        return {
            "checked": True,
            "passed": True,
            "virtual_count": 0,
            "failed_count": 0,
            "maximum_containment_lhs": 0.0,
            "required_covariance_scales": [],
        }
    if owner_indices.min().item() < 0 or owner_indices.max().item() >= merged_means.shape[0]:
        raise ValueError("compact materializer coverage owner is out of range")
    virtual_centres, virtual_projected = _project_context_gaussians(
        virtual_means,
        virtual_covariances,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        view=view,
    )
    merged_centres, merged_projected = _project_context_gaussians(
        merged_means,
        merged_covariances,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        view=view,
    )
    radius = 2.0
    tolerance = torch.as_tensor(1e-5, device=virtual_means.device, dtype=virtual_means.dtype)
    maximum_lhs = 0.0
    failed = 0
    required_scales = torch.ones(
        merged_means.shape[0], device=virtual_means.device, dtype=virtual_means.dtype
    )
    for index, owner in enumerate(owner_indices.detach().to(device="cpu", dtype=torch.int64).tolist()):
        container = merged_projected[owner]
        source = virtual_projected[index]
        delta = virtual_centres[index] - merged_centres[owner]
        try:
            cholesky = torch.linalg.cholesky(container)
            mahalanobis = torch.linalg.solve_triangular(
                cholesky, delta.unsqueeze(1), upper=False
            ).square().sum().clamp_min(0.0).sqrt()
            left = torch.linalg.solve_triangular(cholesky, source, upper=False)
            whitened = torch.linalg.solve_triangular(
                cholesky, left.mT, upper=False
            ).mT
            maximum_eigenvalue = torch.linalg.eigvalsh(
                (whitened + whitened.mT) * 0.5
            ).amax()
        except RuntimeError as error:
            raise ValueError("compact materializer coverage containment failed") from error
        if not bool(torch.isfinite(mahalanobis)) or not bool(torch.isfinite(maximum_eigenvalue)):
            raise ValueError("compact materializer coverage containment is non-finite")
        lhs = mahalanobis + radius * maximum_eigenvalue.clamp_min(0.0).sqrt()
        maximum_lhs = max(maximum_lhs, float(lhs.item()))
        required_scales[owner] = torch.maximum(
            required_scales[owner],
            (lhs / radius).square(),
        )
        if bool(lhs > radius + tolerance):
            failed += 1
    return {
        "checked": True,
        "passed": failed == 0,
        "virtual_count": int(owner_indices.numel()),
        "failed_count": failed,
        "maximum_containment_lhs": maximum_lhs,
        "required_covariance_scales": [
            float(value)
            for value in required_scales.detach().to(device="cpu", dtype=torch.float64).tolist()
        ],
    }


def _build_tile_update(
    *,
    packet: SparseRawGaussianPacket,
    packed: PackedGaussianAttributes,
    slot_to_index: Mapping[int, int],
    view_features: torch.Tensor,
    feature_variance: float,
    view: int,
    tile_y: int,
    tile_x: int,
    height: int,
    width: int,
    tile_size: int,
    level: str,
    anchor_positions: list[tuple[int, int]],
    primary_positions: list[tuple[int, int]],
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    aggregation_semantics: str,
) -> tuple[
    list[int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    anchor_slots = _tile_slots(
        view=view,
        tile_y=tile_y,
        tile_x=tile_x,
        height=height,
        width=width,
        tile_size=tile_size,
        positions=anchor_positions,
    )
    if any(slot not in slot_to_index for slot in anchor_slots):
        raise ValueError("compact materializer tile lacks a selected anchor")
    anchor_indices = [slot_to_index[slot] for slot in anchor_slots]
    source_means = packed.means[anchor_indices]
    source_covariances = packed.covariances[anchor_indices]
    source_harmonics = packed.harmonics[anchor_indices]
    source_opacities = packed.opacities[anchor_indices]
    _validate_source_covariances(source_covariances)
    if not bool(torch.isfinite(source_harmonics).all()):
        raise ValueError("compact materializer source harmonics are non-finite")

    all_positions = _full_positions(tile_size)
    anchor_set = set(anchor_positions)
    nonprobe_positions = [position for position in all_positions if position not in anchor_set]
    if not nonprobe_positions:
        return (
            anchor_slots,
            source_means.clone(),
            source_covariances.clone(),
            source_harmonics.clone(),
            source_opacities.clone(),
            {
                "checked": True,
                "passed": True,
                "virtual_count": 0,
                "failed_count": 0,
                "maximum_containment_lhs": 0.0,
                "scope": "no-nonprobe",
            },
        )
    coordinate_scale = max(tile_size - 1, 1)
    probe_features = torch.stack(
        [
            view_features[:, tile_y * tile_size + row, tile_x * tile_size + column]
            for row, column in anchor_positions
        ],
        dim=0,
    )
    assignments: list[torch.Tensor] = []
    for local_y, local_x in nonprobe_positions:
        target_feature = view_features[
            :, tile_y * tile_size + local_y, tile_x * tile_size + local_x
        ]
        spatial_distances = torch.tensor(
            [
                ((local_y - anchor_y) / coordinate_scale) ** 2
                + ((local_x - anchor_x) / coordinate_scale) ** 2
                for anchor_y, anchor_x in anchor_positions
            ],
            device=source_means.device,
            dtype=source_means.dtype,
        )
        feature_distances = (probe_features - target_feature.unsqueeze(0)).square().sum(dim=1)
        kwargs: dict[str, Any] = {}
        if level == "L1":
            primary_count = len(primary_positions)
            if anchor_positions[:primary_count] != primary_positions:
                raise ValueError("compact materializer L1 anchors lost their primary prefix")
            kwargs = {
                "probe_depths": packet.depths[anchor_indices],
                "depth_reference_depths": packet.depths[anchor_indices[:primary_count]],
                "beta_d": 1.0,
            }
        weights = paper_assignment_weights(
            spatial_distances,
            feature_distances.to(dtype=source_means.dtype),
            feature_variance=feature_variance,
            beta_x=0.50,
            beta_f=0.10,
            level=level,
            **kwargs,
        )
        if not bool(torch.isfinite(weights).all()) or not torch.allclose(
            weights.sum(), torch.ones((), device=weights.device, dtype=weights.dtype), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("compact materializer assignment is invalid")
        assignments.append(weights)
    assignment_matrix = torch.stack(assignments, dim=0)
    source_positions = [
        (tile_y * tile_size + row, tile_x * tile_size + column)
        for row, column in anchor_positions
    ]
    target_positions = [
        (tile_y * tile_size + row, tile_x * tile_size + column)
        for row, column in nonprobe_positions
    ]
    if aggregation_semantics in {
        CONDITIONAL_DIRECT_AGGREGATION,
        CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
        CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION,
        CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION,
        SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION,
    }:
        source_offsets = _selected_native_offsets(
            packet=packet,
            packed=packed,
            anchor_indices=anchor_indices,
            source_positions=source_positions,
            context_extrinsics=context_extrinsics,
            context_intrinsics=context_intrinsics,
            view=view,
            height=height,
            width=width,
        )
        source_depths = packet.depths[anchor_indices].to(
            device=source_means.device,
            dtype=source_means.dtype,
        )
        target_centres = _pixel_centres(
            target_positions,
            height=height,
            width=width,
            device=source_means.device,
            dtype=source_means.dtype,
        )
        spatial_virtual_geometry = (
            aggregation_semantics == SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION
        )
        spatial_attribute_field = aggregation_semantics in {
            CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
            CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION,
            CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION,
            SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION,
        }
        l1_plane_geometry = (
            aggregation_semantics
            == CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION
            and level == "L1"
        )
        feature_continuity: dict[str, float | bool] | None = None
        if spatial_attribute_field:
            virtual_weights = _spatial_virtual_weights(
                target_positions,
                source_positions,
                device=source_means.device,
                dtype=source_means.dtype,
            )
            # V4 established the L0-only closure. V6 extends the same
            # target-free criterion to the remaining compact L1 tiles.
            use_feature_continuity = (
                aggregation_semantics
                == CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION
                or (
                    aggregation_semantics
                    == CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION
                    and level == "L0"
                )
            )
            if use_feature_continuity:
                feature_continuity = _feature_interpolation_continuity(
                    view_features=view_features,
                    source_positions=source_positions,
                    target_positions=target_positions,
                    spatial_weights=virtual_weights,
                )
                if feature_continuity["passed"] is not True:
                    raise ValueError("compact materializer feature continuity rejected tile")
            flat_harmonics = source_harmonics.reshape(len(anchor_indices), -1)
            virtual_harmonics = virtual_weights @ flat_harmonics
            virtual_opacities = virtual_weights @ source_opacities
            if (
                not bool(torch.isfinite(virtual_harmonics).all())
                or not bool(torch.isfinite(virtual_opacities).all())
                or bool((virtual_opacities < 0.0).any())
                or bool((virtual_opacities >= 1.0).any())
            ):
                raise ValueError("compact materializer spatial virtual attributes are invalid")
        if spatial_virtual_geometry:
            virtual_offsets = virtual_weights @ source_offsets
            virtual_depths = virtual_weights @ source_depths
            virtual_means = _lift_context_coordinates(
                target_centres + virtual_offsets,
                virtual_depths,
                context_extrinsics=context_extrinsics,
                context_intrinsics=context_intrinsics,
                view=view,
            )
            virtual_covariances = torch.einsum(
                "nk,kij->nij", virtual_weights, source_covariances
            )
            virtual_covariances = (
                virtual_covariances + virtual_covariances.mT
            ) * 0.5
            conditional_means_by_anchor = None
            conditional_covariances_by_anchor = None
            l1_plane = None
        else:
            l1_plane = None
            conditional_covariances_by_anchor = None
            if l1_plane_geometry:
                plane_origin, plane_normal, l1_plane = _fit_l1_anchor_plane(source_means)
                conditional_depths_by_anchor = torch.stack(
                    [
                        _intersect_context_rays_with_plane(
                            target_centres + source_offsets[anchor_offset].unsqueeze(0),
                            plane_origin=plane_origin,
                            plane_normal=plane_normal,
                            context_extrinsics=context_extrinsics,
                            context_intrinsics=context_intrinsics,
                            view=view,
                        )
                        for anchor_offset in range(len(anchor_indices))
                    ],
                    dim=0,
                )
                conditional_means_by_anchor = torch.stack(
                    [
                        _lift_context_coordinates(
                            target_centres + source_offsets[anchor_offset].unsqueeze(0),
                            conditional_depths_by_anchor[anchor_offset],
                            context_extrinsics=context_extrinsics,
                            context_intrinsics=context_intrinsics,
                            view=view,
                        )
                        for anchor_offset in range(len(anchor_indices))
                    ],
                    dim=0,
                )
                depth_ratios = conditional_depths_by_anchor / source_depths[:, None]
                if (
                    not bool(torch.isfinite(depth_ratios).all())
                    or bool((depth_ratios <= 0.0).any())
                ):
                    raise ValueError("compact materializer L1 plane depth ratio is invalid")
                conditional_covariances_by_anchor = (
                    source_covariances[:, None]
                    * depth_ratios.square().unsqueeze(-1).unsqueeze(-1)
                )
            else:
                conditional_means_by_anchor = torch.stack(
                    [
                        _lift_context_coordinates(
                            target_centres + source_offsets[anchor_offset].unsqueeze(0),
                            source_depths[anchor_offset].expand(len(target_positions)),
                            context_extrinsics=context_extrinsics,
                            context_intrinsics=context_intrinsics,
                            view=view,
                        )
                        for anchor_offset in range(len(anchor_indices))
                    ],
                    dim=0,
                )
        output_means: list[torch.Tensor] = []
        output_covariances: list[torch.Tensor] = []
        output_harmonics: list[torch.Tensor] = []
        output_opacities: list[torch.Tensor] = []
        for anchor_offset, _anchor_index in enumerate(anchor_indices):
            absorbed = assignment_matrix[:, anchor_offset]
            weights = torch.cat(
                (torch.ones(1, device=absorbed.device, dtype=absorbed.dtype), absorbed)
            )
            normalized = weights / weights.sum().clamp_min(1e-8)
            if spatial_virtual_geometry:
                contributor_means = torch.cat(
                    (source_means[anchor_offset].unsqueeze(0), virtual_means),
                    dim=0,
                )
                contributor_covariances = torch.cat(
                    (source_covariances[anchor_offset].unsqueeze(0), virtual_covariances),
                    dim=0,
                )
            else:
                if conditional_means_by_anchor is None:
                    raise RuntimeError("compact materializer direct virtual geometry is missing")
                contributor_means = torch.cat(
                    (
                        source_means[anchor_offset].unsqueeze(0),
                        conditional_means_by_anchor[anchor_offset],
                    ),
                    dim=0,
                )
                conditional_covariances = (
                    source_covariances[anchor_offset]
                    .unsqueeze(0)
                    .expand(len(target_positions), -1, -1)
                    if conditional_covariances_by_anchor is None
                    else conditional_covariances_by_anchor[anchor_offset]
                )
                contributor_covariances = torch.cat(
                    (
                        source_covariances[anchor_offset].unsqueeze(0),
                        conditional_covariances,
                    ),
                    dim=0,
                )
            mean = torch.einsum("n,ni->i", normalized, contributor_means)
            centered = contributor_means - mean
            covariance = torch.einsum(
                "n,nij->ij",
                normalized,
                contributor_covariances
                + torch.einsum("ni,nj->nij", centered, centered),
            )
            covariance = _project_psd(covariance)
            if spatial_attribute_field:
                harmonic_minimum = source_harmonics.amin(dim=0)
                harmonic_maximum = source_harmonics.amax(dim=0)
                harmonics = torch.einsum(
                    "n,nk->k",
                    normalized,
                    torch.cat(
                        (
                            flat_harmonics[anchor_offset].unsqueeze(0),
                            virtual_harmonics,
                        ),
                        dim=0,
                    ),
                ).reshape_as(source_harmonics[anchor_offset])
                harmonics = torch.maximum(
                    torch.minimum(harmonics, harmonic_maximum), harmonic_minimum
                )
                opacity = torch.dot(
                    normalized,
                    torch.cat(
                        (source_opacities[anchor_offset].reshape(1), virtual_opacities)
                    ),
                ).clamp(
                    min=float(source_opacities.amin().item()),
                    max=float(source_opacities.amax().item()),
                )
            else:
                # The Adapter's SH and alpha do not depend on image-plane
                # coordinates. With no skipped S3 attributes, the v2 direct
                # closure retains the receiver's range-constrained value.
                harmonics = source_harmonics[anchor_offset].clone()
                opacity = source_opacities[anchor_offset].clone()
            if (
                not bool(torch.isfinite(mean).all())
                or not bool(torch.isfinite(covariance).all())
                or not bool(torch.isfinite(harmonics).all())
                or not bool(torch.isfinite(opacity))
                or bool(torch.linalg.eigvalsh(covariance).min() < -1e-6)
                or bool(opacity < 0.0)
                or bool(opacity >= 1.0)
            ):
                raise ValueError("compact materializer direct output validation failed")
            output_means.append(mean)
            output_covariances.append(covariance)
            output_harmonics.append(harmonics)
            output_opacities.append(opacity)
        merged_means = torch.stack(output_means)
        merged_covariances = torch.stack(output_covariances)
        owner_indices = assignment_matrix.argmax(dim=1)
        row_indices = torch.arange(
            len(target_positions), device=owner_indices.device, dtype=torch.long
        )
        if spatial_virtual_geometry:
            coverage_virtual_means = virtual_means
            coverage_virtual_covariances = virtual_covariances
        else:
            if conditional_means_by_anchor is None:
                raise RuntimeError("compact materializer direct coverage geometry is missing")
            coverage_virtual_means = conditional_means_by_anchor[owner_indices, row_indices]
            coverage_virtual_covariances = (
                source_covariances[owner_indices]
                if conditional_covariances_by_anchor is None
                else conditional_covariances_by_anchor[owner_indices, row_indices]
            )
        initial_coverage = _ellipse_containment_certificate(
            virtual_means=coverage_virtual_means,
            virtual_covariances=coverage_virtual_covariances,
            owner_indices=owner_indices,
            merged_means=merged_means,
            merged_covariances=merged_covariances,
            context_extrinsics=context_extrinsics,
            context_intrinsics=context_intrinsics,
            view=view,
        )
        required_scales = torch.tensor(
            initial_coverage["required_covariance_scales"],
            device=merged_covariances.device,
            dtype=merged_covariances.dtype,
        )
        if (
            required_scales.shape != (len(anchor_indices),)
            or not bool(torch.isfinite(required_scales).all())
            or bool((required_scales > COMPACT_COVERAGE_MAX_COVARIANCE_SCALE).any())
        ):
            raise ValueError("compact materializer required coverage expansion is unsafe")
        merged_covariances = torch.stack(
            [
                _project_psd(covariance * scale)
                for covariance, scale in zip(merged_covariances, required_scales)
            ]
        )
        coverage = _ellipse_containment_certificate(
            virtual_means=coverage_virtual_means,
            virtual_covariances=coverage_virtual_covariances,
            owner_indices=owner_indices,
            merged_means=merged_means,
            merged_covariances=merged_covariances,
            context_extrinsics=context_extrinsics,
            context_intrinsics=context_intrinsics,
            view=view,
        )
        coverage["scope"] = (
            "selected-only-spatial-virtual-intra-tile-2sigma"
            if spatial_virtual_geometry
            else "selected-only-l1-plane-direct-geometry-spatial-attribute-intra-tile-2sigma"
            if l1_plane_geometry
            else "selected-only-direct-geometry-spatial-attribute-continuity-intra-tile-2sigma"
            if aggregation_semantics
            == CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION
            else "selected-only-direct-geometry-spatial-attribute-intra-tile-2sigma"
            if spatial_attribute_field
            else "selected-only-intra-tile-virtual-2sigma"
        )
        coverage["moment_covariance_scale_max"] = float(required_scales.max().item())
        if l1_plane is not None:
            coverage["l1_plane_maximum_residual"] = l1_plane["maximum_plane_residual"]
            coverage["l1_plane_planarity_eigenvalue_ratio"] = l1_plane[
                "planarity_eigenvalue_ratio"
            ]
        if feature_continuity is not None:
            coverage["feature_continuity_passed"] = feature_continuity["passed"]
            coverage["feature_continuity_anchor_diameter"] = feature_continuity[
                "anchor_feature_diameter"
            ]
            coverage["feature_continuity_maximum_residual"] = feature_continuity[
                "maximum_interpolation_residual"
            ]
            coverage["feature_continuity_maximum_relative_residual"] = feature_continuity[
                "maximum_relative_interpolation_residual"
            ]
        if coverage["passed"] is not True:
            raise ValueError("compact materializer virtual support is not covered")
        return (
            anchor_slots,
            merged_means,
            merged_covariances,
            torch.stack(output_harmonics),
            torch.stack(output_opacities),
            coverage,
        )
    if aggregation_semantics != ASSIGNMENT_MIXTURE_AGGREGATION:
        raise ValueError("compact materializer aggregation semantics are unsupported")
    pseudo_means = _camera_residual_pseudo_means(
        source_means=source_means,
        source_depths=packet.depths[anchor_indices].to(
            device=source_means.device, dtype=source_means.dtype
        ),
        source_positions=source_positions,
        assignment_matrix=assignment_matrix,
        target_positions=target_positions,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        view=view,
        width=width,
        height=height,
    )
    pseudo_covariances = torch.einsum("nk,kij->nij", assignment_matrix, source_covariances)
    pseudo_covariances = (pseudo_covariances + pseudo_covariances.mT) * 0.5
    flat_harmonics = source_harmonics.reshape(len(anchor_indices), -1)
    pseudo_harmonics = assignment_matrix @ flat_harmonics
    pseudo_opacities = assignment_matrix @ source_opacities
    if (
        not bool(torch.isfinite(pseudo_harmonics).all())
        or not bool(torch.isfinite(pseudo_opacities).all())
        or bool((pseudo_opacities < 0.0).any())
        or bool((pseudo_opacities >= 1.0).any())
    ):
        raise ValueError("compact materializer pseudo attributes are invalid")

    output_means: list[torch.Tensor] = []
    output_covariances: list[torch.Tensor] = []
    output_harmonics: list[torch.Tensor] = []
    output_opacities: list[torch.Tensor] = []
    harmonic_minimum = source_harmonics.amin(dim=0)
    harmonic_maximum = source_harmonics.amax(dim=0)
    opacity_minimum = source_opacities.amin()
    opacity_maximum = source_opacities.amax()
    for anchor_offset, _anchor_index in enumerate(anchor_indices):
        absorbed = assignment_matrix[:, anchor_offset]
        weights = torch.cat(
            (torch.ones(1, device=absorbed.device, dtype=absorbed.dtype), absorbed)
        )
        normalized = weights / weights.sum().clamp_min(1e-8)
        contributor_means = torch.cat(
            (source_means[anchor_offset].unsqueeze(0), pseudo_means),
            dim=0,
        )
        contributor_covariances = torch.cat(
            (
                source_covariances[anchor_offset].unsqueeze(0),
                pseudo_covariances,
            ),
            dim=0,
        )
        mean = torch.einsum("n,ni->i", normalized, contributor_means)
        centered = contributor_means - mean
        covariance = torch.einsum(
            "n,nij->ij",
            normalized,
            contributor_covariances
            + torch.einsum("ni,nj->nij", centered, centered),
        )
        covariance = _project_psd(covariance)
        harmonics = torch.einsum(
            "n,nk->k",
            normalized,
            torch.cat((flat_harmonics[anchor_offset].unsqueeze(0), pseudo_harmonics), dim=0),
        ).reshape_as(source_harmonics[anchor_offset])
        harmonics = torch.maximum(torch.minimum(harmonics, harmonic_maximum), harmonic_minimum)
        opacity = torch.dot(
            normalized,
            torch.cat((source_opacities[anchor_offset].reshape(1), pseudo_opacities)),
        ).clamp(min=float(opacity_minimum.item()), max=float(opacity_maximum.item()))
        if (
            not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(covariance).all())
            or not bool(torch.isfinite(harmonics).all())
            or not bool(torch.isfinite(opacity))
            or bool(torch.linalg.eigvalsh(covariance).min() < -1e-6)
            or bool(opacity < 0.0)
            or bool(opacity >= 1.0)
        ):
            raise ValueError("compact materializer output validation failed")
        output_means.append(mean)
        output_covariances.append(covariance)
        output_harmonics.append(harmonics)
        output_opacities.append(opacity)
    return (
        anchor_slots,
        torch.stack(output_means),
        torch.stack(output_covariances),
        torch.stack(output_harmonics),
        torch.stack(output_opacities),
        {
            "checked": False,
            "passed": False,
            "virtual_count": len(target_positions),
            "failed_count": len(target_positions),
            "maximum_containment_lhs": float("inf"),
            "scope": "legacy-assignment-mixture-no-coverage-certificate",
        },
    )


def preflight_compact_l0_l1_materialization(
    initial_packet: SparseRawGaussianPacket,
    initial_packed: PackedGaussianAttributes,
    guarded_route: GuardedSelectedRoute,
    plan: IncrementalProbeFirstPlan,
    features: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    *,
    aggregation_semantics: str = DEFAULT_COMPACT_AGGREGATION,
    selected_anchor_v4_attribute_loo_maximum_risk: float | None = None,
    collect_selected_anchor_v4_attribute_loo_risk: bool = False,
) -> CompactMaterializationPreflight:
    """Plan all selected-anchor updates without mutating a final packet.

    The only tile-local failure action is promotion to Full.  This preserves a
    single source-bound producer extension and prevents partially updated
    anchor packets from escaping a failed preflight.
    """
    views, height, width, semantics = _require_fixed_plan(plan)
    if aggregation_semantics not in {
        ASSIGNMENT_MIXTURE_AGGREGATION,
        CONDITIONAL_DIRECT_AGGREGATION,
        CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION,
        CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION,
        CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION,
        SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION,
    }:
        raise ValueError("compact materializer aggregation semantics are unsupported")
    if not isinstance(guarded_route, GuardedSelectedRoute):
        raise TypeError("compact materializer requires a guarded selected route")
    execution_policy = guarded_route.events.get("execution_policy")
    if not isinstance(execution_policy, str) or not execution_policy:
        raise ValueError("compact materializer route has no execution policy")
    v4_attribute_loo_guard = (
        execution_policy
        == ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
    )
    if v4_attribute_loo_guard and (
        semantics != ADAPTIVE_L1_15_ANCHOR_SEMANTICS
        or aggregation_semantics != CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION
    ):
        raise ValueError("compact materializer V4 replay policy requires V4 adaptive L1-15")
    if not isinstance(collect_selected_anchor_v4_attribute_loo_risk, bool):
        raise ValueError("compact materializer V4 replay collection must be boolean")
    if v4_attribute_loo_guard:
        if (
            isinstance(selected_anchor_v4_attribute_loo_maximum_risk, bool)
            or not isinstance(selected_anchor_v4_attribute_loo_maximum_risk, (int, float))
            or not torch.isfinite(
                torch.tensor(float(selected_anchor_v4_attribute_loo_maximum_risk))
            )
            or float(selected_anchor_v4_attribute_loo_maximum_risk) < 0.0
        ):
            raise ValueError("compact materializer V4 replay policy requires a finite threshold")
    elif selected_anchor_v4_attribute_loo_maximum_risk is not None:
        raise ValueError("compact materializer V4 replay threshold requires its dedicated policy")
    collect_v4_attribute_loo = (
        collect_selected_anchor_v4_attribute_loo_risk or v4_attribute_loo_guard
    )
    if (
        guarded_route.selected_output_mask.shape != plan.selection_mask.shape
        or guarded_route.additional_full_mask.shape != plan.selection_mask.shape
        or guarded_route.raw_head_request_mask.shape != plan.selection_mask.shape
    ):
        raise ValueError("compact materializer route masks do not match the plan")
    if semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS:
        if (
            guarded_route.events.get("source_tile_trace_sha256")
            != plan.events.get("tile_trace_sha256")
            or guarded_route.events.get("source_selection_mask_sha256")
            != _mask_sha256(plan.selection_mask)
            or guarded_route.events.get("l1_anchor_semantics") != semantics
            or guarded_route.events.get("l1_anchor_count") != 15
        ):
            raise ValueError("adaptive L1-15 route is not bound to its source plan")
    if not torch.is_tensor(features) or features.ndim != 5 or features.shape[:2] != (1, views):
        raise ValueError("compact materializer requires B=1 aligned S1 features")
    if features.device != initial_packet.raw_descriptors.device or not bool(torch.isfinite(features).all()):
        raise ValueError("compact materializer S1 features must be finite and source-aligned")
    if (
        not torch.is_tensor(context_extrinsics)
        or not torch.is_tensor(context_intrinsics)
        or context_extrinsics.shape != (1, views, 4, 4)
        or context_intrinsics.shape != (1, views, 3, 3)
        or context_extrinsics.device != initial_packet.raw_descriptors.device
        or context_intrinsics.device != initial_packet.raw_descriptors.device
        or not bool(torch.isfinite(context_extrinsics).all())
        or not bool(torch.isfinite(context_intrinsics).all())
    ):
        raise ValueError("compact materializer requires finite source-bound context geometry")
    slot_to_index = _validate_selected_source(
        initial_packet,
        initial_packed,
        plan,
        views=views,
        height=height,
        width=width,
    )
    native_opacity_endpoint_selected_count = int(
        (initial_packed.opacities == 1.0).sum().item()
    )
    primary_positions = compute_probe_positions(4)
    expected_l1_count = int(plan.events["l1_anchor_count"])
    if len(primary_positions) != 4 or expected_l1_count not in {4, 12, 15}:
        raise RuntimeError("compact materializer has an invalid fixed anchor layout")
    decision_semantics = str(plan.events.get("decision_semantics"))
    feature_statistic = str(plan.events.get("feature_statistic"))
    if feature_statistic not in {
        "raw-probe-vector-variance",
        "raw-probe-mean-channel-variance",
        "normalized-probe-total-variance",
        "normalized-probe-vector-standard-deviation",
    }:
        raise ValueError("compact materializer plan has an unknown feature statistic")
    _scores, feature_norm = ProgressiveSAES.classify_tiles_by_features(
        features,
        height,
        width,
        4,
        threshold=float(plan.events["feature_threshold"]),
        per_view=True,
        statistic=feature_statistic,
    )
    if feature_norm is None or feature_norm.shape != (views, features.shape[2], height, width):
        raise RuntimeError("compact materializer could not construct aligned S1 features")
    route_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in guarded_route.tile_trace
    }
    plan_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in plan.tile_trace
    }
    expected_records = views * (height // 4) * (width // 4)
    if len(route_records) != expected_records or len(plan_records) != expected_records:
        raise ValueError("compact materializer received incomplete tile traces")

    updates: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    promote_full_mask = torch.zeros_like(plan.selection_mask)
    trace: list[dict[str, Any]] = []
    attempted = 0
    accepted = 0
    rejection_reasons: dict[str, int] = {}
    for view in range(views):
        for tile_y in range(height // 4):
            for tile_x in range(width // 4):
                key = (view, tile_y, tile_x)
                route_record = route_records[key]
                plan_record = plan_records[key]
                level = route_record.get("final_route")
                if level not in {"L0", "L1", "Full"}:
                    raise ValueError("compact materializer route has an invalid final level")
                tile_entry: dict[str, Any] = {
                    "view": view,
                    "tile_y": tile_y,
                    "tile_x": tile_x,
                    "guarded_route": level,
                    "attempted": level in {"L0", "L1"},
                    "accepted": False,
                    "reason": "full_passthrough" if level == "Full" else None,
                    "anchor_count": 0,
                    "nonprobe_count": 0,
                    "coverage": None,
                    "selected_anchor_v4_attribute_loo": None,
                    "native_opacity_endpoint_anchor_count": 0,
                }
                if level == "Full":
                    trace.append(tile_entry)
                    continue
                attempted += 1
                anchors = (
                    primary_positions
                    if level == "L0"
                    else l1_local_positions_for_tile(
                        plan_record,
                        tile_size=4,
                        l1_anchor_semantics=semantics,
                    )
                )
                if len(anchors) != (4 if level == "L0" else expected_l1_count):
                    raise ValueError("compact materializer tile has an invalid anchor count")
                if semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS and level == "L1":
                    expected_retained = [list(position) for position in anchors]
                    if route_record.get("retained_local_positions") != expected_retained:
                        raise ValueError("adaptive L1-15 route anchors do not match its plan")
                tile_entry["anchor_count"] = len(anchors)
                tile_entry["nonprobe_count"] = 16 - len(anchors)
                try:
                    anchor_slots = _tile_slots(
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        height=height,
                        width=width,
                        tile_size=4,
                        positions=anchors,
                    )
                    if any(slot not in slot_to_index for slot in anchor_slots):
                        raise ValueError("compact materializer tile anchor is absent from selection")
                    endpoint_anchor_count = int(
                        (
                            initial_packed.opacities[
                                [slot_to_index[slot] for slot in anchor_slots]
                            ]
                            == 1.0
                        )
                        .sum()
                        .item()
                    )
                    tile_entry["native_opacity_endpoint_anchor_count"] = endpoint_anchor_count
                    if endpoint_anchor_count:
                        raise ValueError(NATIVE_OPACITY_ENDPOINT_FULL_REASON)
                    feature_variance = _tile_feature_variance(
                        plan_record,
                        decision_semantics,
                        feature_statistic,
                    )
                    if collect_v4_attribute_loo and level == "L1":
                        replay = selected_anchor_v4_attribute_loo_certificate(
                            packet=initial_packet,
                            packed=initial_packed,
                            slot_to_index=slot_to_index,
                            view=view,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            height=height,
                            width=width,
                            tile_size=4,
                            anchor_positions=anchors,
                        )
                        if v4_attribute_loo_guard:
                            maximum_risk = float(
                                selected_anchor_v4_attribute_loo_maximum_risk
                            )
                            replay["maximum_risk"] = maximum_risk
                            replay["passed"] = replay["q75_risk"] <= maximum_risk
                            replay["action"] = (
                                "retain_l1"
                                if replay["passed"] is True
                                else "promote_full"
                            )
                        else:
                            replay["maximum_risk"] = None
                            replay["passed"] = None
                            replay["action"] = "observed_only"
                        tile_entry["selected_anchor_v4_attribute_loo"] = replay
                        if replay["passed"] is False:
                            raise ValueError("compact materializer V4 replay rejected tile")
                    slots, means, covariances, harmonics, opacities, coverage = _build_tile_update(
                        packet=initial_packet,
                        packed=initial_packed,
                        slot_to_index=slot_to_index,
                        view_features=feature_norm[view],
                        feature_variance=feature_variance,
                        view=view,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        height=height,
                        width=width,
                        tile_size=4,
                        level=level,
                        anchor_positions=anchors,
                        primary_positions=primary_positions,
                        context_extrinsics=context_extrinsics,
                        context_intrinsics=context_intrinsics,
                        aggregation_semantics=aggregation_semantics,
                    )
                except (RuntimeError, ValueError) as error:
                    _mark_tile(promote_full_mask, view=view, tile_y=tile_y, tile_x=tile_x, tile_size=4)
                    reason = str(error) or type(error).__name__
                    tile_entry["reason"] = reason
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                else:
                    for slot, mean, covariance, harmonic, opacity in zip(
                        slots, means, covariances, harmonics, opacities
                    ):
                        updates.append((slot, mean, covariance, harmonic, opacity))
                    tile_entry["accepted"] = True
                    tile_entry["reason"] = "accepted"
                    tile_entry["coverage"] = coverage
                    accepted += 1
                trace.append(tile_entry)
    if updates:
        ordered = sorted(updates, key=lambda value: value[0])
        update_slots = torch.tensor(
            [value[0] for value in ordered],
            device=initial_packed.dense_slots.device,
            dtype=torch.int64,
        )
        if len(set(int(slot) for slot in update_slots.detach().cpu().tolist())) != len(ordered):
            raise RuntimeError("compact materializer generated duplicate anchor updates")
        means = torch.stack([value[1] for value in ordered])
        covariances = torch.stack([value[2] for value in ordered])
        harmonics = torch.stack([value[3] for value in ordered])
        opacities = torch.stack([value[4] for value in ordered])
    else:
        dtype = initial_packed.means.dtype
        device = initial_packed.means.device
        update_slots = torch.empty(0, device=device, dtype=torch.int64)
        means = torch.empty(0, 3, device=device, dtype=dtype)
        covariances = torch.empty(0, 3, 3, device=device, dtype=dtype)
        harmonics = torch.empty(
            0,
            3,
            initial_packed.harmonics.shape[2],
            device=device,
            dtype=initial_packed.harmonics.dtype,
        )
        opacities = torch.empty(0, device=device, dtype=initial_packed.opacities.dtype)
    coverage_records = [
        record["coverage"]
        for record in trace
        if isinstance(record.get("coverage"), Mapping)
    ]
    coverage_scales = [
        float(record.get("moment_covariance_scale_max", 1.0))
        for record in coverage_records
        if isinstance(record.get("moment_covariance_scale_max", 1.0), (int, float))
    ]
    coverage_lhs = [
        float(record["maximum_containment_lhs"])
        for record in coverage_records
        if isinstance(record.get("maximum_containment_lhs"), (int, float))
    ]
    v4_attribute_loo_records = [
        record["selected_anchor_v4_attribute_loo"]
        for record in trace
        if isinstance(record.get("selected_anchor_v4_attribute_loo"), Mapping)
    ]
    v4_attribute_loo_risks = [
        float(record["q75_risk"])
        for record in v4_attribute_loo_records
        if isinstance(record.get("q75_risk"), (int, float))
    ]
    if len(v4_attribute_loo_risks) != len(v4_attribute_loo_records):
        raise ValueError("compact materializer V4 replay trace is incomplete")
    trace_sha256 = _canonical_sha256(trace)
    events = {
        "schema_version": MATERIALIZER_SCHEMA_VERSION,
        "execution_policy": execution_policy,
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "target_rgb_accessed_before_commit": False,
        "skipped_s3_attributes_accessed": False,
        "source_nonprobe_s3_attribute_reads": 0,
        "not_lossless_deletion": True,
        "nonzero_direct_deletion": False,
        "aggregation": aggregation_semantics,
        "coverage_certificate": (
            "selected-only-spatial-virtual-intra-tile-2sigma-v1"
            if aggregation_semantics == SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION
            else "selected-only-direct-geometry-spatial-attribute-intra-tile-2sigma-v1"
            if aggregation_semantics == CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION
            else "selected-only-l1-plane-direct-geometry-spatial-attribute-intra-tile-2sigma-v1"
            if aggregation_semantics == CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION
            else "selected-only-direct-geometry-spatial-attribute-continuity-intra-tile-2sigma-v1"
            if aggregation_semantics
            == CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION
            else "selected-only-intra-tile-virtual-2sigma-v1"
            if aggregation_semantics == CONDITIONAL_DIRECT_AGGREGATION
            else "not-available-for-legacy-assignment-mixture"
        ),
        "coverage_checked_tiles": len(coverage_records),
        "coverage_max_moment_covariance_scale": max(coverage_scales, default=0.0),
        "coverage_max_containment_lhs": max(coverage_lhs, default=0.0),
        "selected_anchor_v4_attribute_loo_guard": v4_attribute_loo_guard,
        "selected_anchor_v4_attribute_loo_collect_only": (
            collect_v4_attribute_loo and not v4_attribute_loo_guard
        ),
        "selected_anchor_v4_attribute_loo_maximum_risk": (
            float(selected_anchor_v4_attribute_loo_maximum_risk)
            if v4_attribute_loo_guard
            else None
        ),
        "selected_anchor_v4_attribute_loo_checked_tiles": len(v4_attribute_loo_records),
        "selected_anchor_v4_attribute_loo_retained_l1_tiles": sum(
            record.get("action") == "retain_l1" for record in v4_attribute_loo_records
        ),
        "selected_anchor_v4_attribute_loo_promoted_full_tiles": sum(
            record.get("action") == "promote_full" for record in v4_attribute_loo_records
        ),
        "selected_anchor_v4_attribute_loo_risk_summary": _finite_scalar_summary(
            v4_attribute_loo_risks
        ),
        "selected_anchor_v4_attribute_label_reads": sum(
            int(record.get("selected_anchor_s3_attribute_label_reads", 0))
            for record in v4_attribute_loo_records
        ),
        "native_opacity_endpoint_selected_count": native_opacity_endpoint_selected_count,
        "native_opacity_endpoint_promoted_full_tiles": sum(
            record.get("reason") == NATIVE_OPACITY_ENDPOINT_FULL_REASON
            for record in trace
        ),
        "l1_anchor_semantics": semantics,
        "paper_l1_retention_interpretation": (
            "literal-probe-set-only-not-specified-by-paper"
            if semantics == PAPER_KP_ANCHOR_SEMANTICS
            else "balanced-engineering-layout"
            if semantics == BALANCED_L1_ANCHOR_SEMANTICS
            else "adaptive-center-s1-single-omission-engineering-layout"
            if semantics == ADAPTIVE_L1_15_ANCHOR_SEMANTICS
            else "legacy-engineering-layout"
        ),
        "l0_anchor_count": len(primary_positions),
        "l1_anchor_count": expected_l1_count,
        "attempted_tiles": attempted,
        "accepted_tiles": accepted,
        "promoted_full_tiles": attempted - accepted,
        "update_anchor_count": int(update_slots.numel()),
        "promote_full_mask_sha256": _mask_sha256(promote_full_mask),
        "tile_trace_sha256": trace_sha256,
        "rejection_reasons": rejection_reasons,
    }
    return CompactMaterializationPreflight(
        update_dense_slots=update_slots,
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        promote_full_mask=promote_full_mask,
        tile_trace=tuple(trace),
        events=events,
    )


def resolve_compact_final_route(
    guarded_route: GuardedSelectedRoute,
    plan: IncrementalProbeFirstPlan,
    preflight: CompactMaterializationPreflight,
) -> CompactFinalRoute:
    """Promote failed compact tiles into the existing one-shot Full extension."""
    views, height, width, semantics = _require_fixed_plan(plan)
    if not isinstance(guarded_route, GuardedSelectedRoute) or not isinstance(
        preflight, CompactMaterializationPreflight
    ):
        raise TypeError("compact final route requires guard and preflight records")
    promote = preflight.promote_full_mask
    if (
        promote.shape != plan.selection_mask.shape
        or promote.dtype != torch.bool
        or promote.device != plan.selection_mask.device
    ):
        raise ValueError("compact final route promotion mask is invalid")
    selected = guarded_route.selected_output_mask | promote
    additional = guarded_route.additional_full_mask | (
        promote & ~guarded_route.raw_head_request_mask
    )
    raw_request = plan.selection_mask | additional
    if bool((selected & ~raw_request).any()):
        raise RuntimeError("compact final route output exceeds its producer request")
    preflight_records = {
        (int(record["view"]), int(record["tile_y"]), int(record["tile_x"])): record
        for record in preflight.tile_trace
    }
    if len(preflight_records) != views * (height // 4) * (width // 4):
        raise ValueError("compact final route preflight trace is incomplete")
    trace: list[dict[str, Any]] = []
    route_counts = {"L0": 0, "L1": 0, "Full": 0}
    for record in guarded_route.tile_trace:
        key = (int(record["view"]), int(record["tile_y"]), int(record["tile_x"]))
        preflight_record = preflight_records[key]
        final_route = str(record["final_route"])
        if preflight_record["attempted"] and not preflight_record["accepted"]:
            final_route = "Full"
        if final_route not in route_counts:
            raise ValueError("compact final route has an invalid tile level")
        updated = dict(record)
        updated["compact_materialization"] = dict(preflight_record)
        updated["final_route"] = final_route
        route_counts[final_route] += 1
        trace.append(updated)
    trace_sha256 = _canonical_sha256(trace)
    events = dict(guarded_route.events)
    events.update(
        {
            "schema_version": "saes-compact-l0-l1-route-v1",
            "base_guard_schema_version": guarded_route.events.get("schema_version"),
            "paper_result_eligible": False,
            "target_rgb_accessed": False,
            "target_rgb_accessed_before_commit": False,
            "execution_policy": preflight.events["execution_policy"],
            "not_lossless_deletion": True,
            "source_nonprobe_s3_attribute_reads": 0,
            "nonzero_direct_deletion": False,
            "compact_materialization_enabled": True,
            "compact_materialization_schema_version": MATERIALIZER_SCHEMA_VERSION,
            "compact_materialization_preflight_trace_sha256": preflight.events[
                "tile_trace_sha256"
            ],
            "compact_materialization_promoted_full_tiles": preflight.events[
                "promoted_full_tiles"
            ],
            "l1_anchor_semantics": semantics,
            "route_counts": route_counts,
            "selected_output_mask_sha256": _mask_sha256(selected),
            "additional_full_mask_sha256": _mask_sha256(additional),
            "raw_head_request_mask_sha256": _mask_sha256(raw_request),
            "selected_output_descriptor_count": int(selected.sum().item()),
            "additional_full_descriptor_count": int(additional.sum().item()),
            "requires_incremental_full_dispatch": bool(additional.any()),
            "tile_trace_sha256": trace_sha256,
        }
    )
    return CompactFinalRoute(
        selected_output_mask=selected,
        additional_full_mask=additional,
        raw_head_request_mask=raw_request,
        tile_trace=tuple(trace),
        events=events,
    )


def _validate_final_slots(
    final_packed: PackedGaussianAttributes,
    final_route: CompactFinalRoute,
) -> dict[int, int]:
    if not isinstance(final_packed, PackedGaussianAttributes):
        raise TypeError("compact materializer final packet must be packed attributes")
    expected_positions = final_route.selected_output_mask.nonzero(as_tuple=False)
    _, height, width = final_route.selected_output_mask.shape
    expected_slots = (
        expected_positions[:, 0] * (height * width)
        + expected_positions[:, 1] * width
        + expected_positions[:, 2]
    ).to(final_packed.dense_slots.device, dtype=torch.int64)
    if not torch.equal(final_packed.dense_slots, expected_slots):
        raise ValueError("compact materializer final packet does not use selected_output_mask")
    slots = final_packed.dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    return {slot: index for index, slot in enumerate(slots)}


def apply_compact_l0_l1_materialization(
    final_packed: PackedGaussianAttributes,
    preflight: CompactMaterializationPreflight,
    final_route: CompactFinalRoute,
) -> PackedGaussianAttributes:
    """Apply validated L0/L1 anchor updates while preserving Full attributes bitwise."""
    if not isinstance(preflight, CompactMaterializationPreflight) or not isinstance(
        final_route, CompactFinalRoute
    ):
        raise TypeError("compact materializer requires preflight and final route")
    slot_to_index = _validate_final_slots(final_packed, final_route)
    update_count = int(preflight.update_dense_slots.numel())
    if (
        preflight.means.shape != (update_count, 3)
        or preflight.covariances.shape != (update_count, 3, 3)
        or preflight.harmonics.shape != (update_count, 3, final_packed.harmonics.shape[2])
        or preflight.opacities.shape != (update_count,)
    ):
        raise ValueError("compact materializer preflight update shapes are inconsistent")
    update_slots = preflight.update_dense_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    if any(slot not in slot_to_index for slot in update_slots):
        raise ValueError("compact materializer update does not belong to final output")
    if update_count and (
        not bool(torch.isfinite(preflight.opacities).all())
        or not bool((preflight.opacities >= 0.0).all())
        or not bool((preflight.opacities < 1.0).all())
    ):
        raise ValueError("compact materializer updates require opacities in [0, 1)")
    means = final_packed.means.clone()
    covariances = final_packed.covariances.clone()
    harmonics = final_packed.harmonics.clone()
    opacities = final_packed.opacities.clone()
    if update_count:
        indices = torch.tensor(
            [slot_to_index[slot] for slot in update_slots],
            device=means.device,
            dtype=torch.int64,
        )
        means[indices] = preflight.means.to(device=means.device, dtype=means.dtype)
        covariances[indices] = preflight.covariances.to(
            device=covariances.device, dtype=covariances.dtype
        )
        harmonics[indices] = preflight.harmonics.to(
            device=harmonics.device, dtype=harmonics.dtype
        )
        opacities[indices] = preflight.opacities.to(
            device=opacities.device, dtype=opacities.dtype
        )
    if (
        not bool(torch.isfinite(means).all())
        or not bool(torch.isfinite(covariances).all())
        or not bool(torch.isfinite(harmonics).all())
        or not bool(torch.isfinite(opacities).all())
        or not bool((opacities >= 0.0).all())
        or not bool((opacities <= 1.0).all())
        or bool((torch.linalg.eigvalsh((covariances + covariances.mT) * 0.5) < -1e-6).any())
    ):
        raise ValueError("compact materializer final attributes violate their contract")
    source_trace = dict(final_packed.source_trace)
    source_trace.update(
        {
            "compact_materialization_schema_version": MATERIALIZER_SCHEMA_VERSION,
            "packet_selection_kind": "guard_selected_output_mask",
            "compact_materialization_preflight_trace_sha256": preflight.events[
                "tile_trace_sha256"
            ],
            "compact_materialization_final_route_trace_sha256": final_route.events[
                "tile_trace_sha256"
            ],
            "compact_materialization_update_anchor_count": update_count,
            "compact_materialization_full_passthrough_count": int(means.shape[0]) - update_count,
            "compact_materialization_native_opacity_endpoint_selected_count": preflight.events[
                "native_opacity_endpoint_selected_count"
            ],
            "compact_materialization_native_opacity_endpoint_promoted_full_tiles": preflight.events[
                "native_opacity_endpoint_promoted_full_tiles"
            ],
            "compact_materialization_aggregation": preflight.events["aggregation"],
            "compact_materialization_coverage_certificate": preflight.events[
                "coverage_certificate"
            ],
            "compact_materialization_selected_anchor_v4_attribute_loo_guard": preflight.events[
                "selected_anchor_v4_attribute_loo_guard"
            ],
            "compact_materialization_selected_anchor_v4_attribute_loo_maximum_risk": preflight.events[
                "selected_anchor_v4_attribute_loo_maximum_risk"
            ],
            "compact_materialization_skipped_s3_attributes_accessed": False,
            "compact_materialization_nonzero_direct_deletion": False,
            "compact_materialization_execution_policy": preflight.events[
                "execution_policy"
            ],
            "compact_materialization_not_lossless_deletion": True,
        }
    )
    source_trace_sha256 = _canonical_sha256(source_trace)
    return PackedGaussianAttributes(
        batch_indices=final_packed.batch_indices.clone(),
        dense_slots=final_packed.dense_slots.clone(),
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        source_trace=source_trace,
        source_trace_sha256=source_trace_sha256,
    )


__all__ = [
    "CompactFinalRoute",
    "CompactMaterializationPreflight",
    "ASSIGNMENT_MIXTURE_AGGREGATION",
    "CONDITIONAL_DIRECT_AGGREGATION",
    "CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_AGGREGATION",
    "CONDITIONAL_DIRECT_SPATIAL_ATTRIBUTE_CONTINUITY_AGGREGATION",
    "CONDITIONAL_L1_PLANE_SPATIAL_ATTRIBUTE_AGGREGATION",
    "DEFAULT_COMPACT_AGGREGATION",
    "SPATIAL_VIRTUAL_SINGLE_ASSIGNMENT_AGGREGATION",
    "LEGACY_L1_ANCHOR_SEMANTICS",
    "MATERIALIZER_SCHEMA_VERSION",
    "NATIVE_OPACITY_ENDPOINT_FULL_REASON",
    "PAPER_KP_ANCHOR_SEMANTICS",
    "SELECTED_ANCHOR_V4_ATTRIBUTE_LOO_CERTIFICATE",
    "apply_compact_l0_l1_materialization",
    "preflight_compact_l0_l1_materialization",
    "resolve_compact_final_route",
    "selected_anchor_v4_attribute_loo_certificate",
]
