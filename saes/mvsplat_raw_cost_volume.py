"""Selected-query execution for MVSplat's native raw correlation volume.

This module intentionally stops before ``corr_refine_net``.  It computes the
same warp-and-dot-product primitive as MVSplat for explicitly requested
low-resolution positions, caches primary/secondary/Full work, and exposes a
comparison against a dense raw correlation volume.  Callers must supply the
FP32 feature-pixel intrinsics and reference-to-source relative poses emitted
by MVSplat's ``prepare_feat_proj_data_lists``.  It is not a sparse S2 depth
producer: GroupNorm, U-Net, and cross-view attention still require the native
refinement path to run densely.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


MVSPLAT_RAW_COST_VOLUME_EXECUTION_VERSION = "mvsplat-selected-raw-cost-volume-v1"
_PHASE_ORDER = {"primary": 0, "secondary": 1, "full": 2}


@dataclass(frozen=True)
class MVSplatRawCostVolumeReplay:
    """A sparse raw-correlation map together with its phase ledger."""

    values: torch.Tensor
    computed_mask: torch.Tensor
    events: dict[str, Any]


@dataclass(frozen=True)
class _MVSplatRawCostVolumeInputs:
    reference_features: torch.Tensor
    source_features: tuple[torch.Tensor, ...]
    feature_pixel_intrinsics: torch.Tensor
    relative_reference_to_source_poses: tuple[torch.Tensor, ...]
    inverse_depth_candidates: torch.Tensor


def _tensor_sha256(value: torch.Tensor) -> str:
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mask_sha256(mask: torch.Tensor) -> str:
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise ValueError("mask digest requires a bool tensor")
    return _tensor_sha256(mask.to(dtype=torch.uint8))


def _validate_inputs(
    reference_features: torch.Tensor,
    source_features: Sequence[torch.Tensor],
    feature_pixel_intrinsics: torch.Tensor,
    relative_reference_to_source_poses: Sequence[torch.Tensor],
    inverse_depth_candidates: torch.Tensor,
) -> _MVSplatRawCostVolumeInputs:
    if (
        not torch.is_tensor(reference_features)
        or reference_features.ndim != 4
        or not reference_features.is_floating_point()
    ):
        raise ValueError("reference_features must be a floating [N,C,H,W] tensor")
    batch, channels, height, width = reference_features.shape
    if reference_features.dtype != torch.float32:
        raise ValueError("selected raw-cost-volume execution is restricted to native FP32 tensors")
    if min(batch, channels, height, width) <= 0 or min(height, width) == 1:
        raise ValueError("raw cost-volume inputs require nonempty spatial dimensions above one")
    if not source_features or len(source_features) != len(relative_reference_to_source_poses):
        raise ValueError(
            "source_features and relative_reference_to_source_poses must be nonempty and paired"
        )
    sources = tuple(source_features)
    poses = tuple(relative_reference_to_source_poses)
    for source in sources:
        if (
            not torch.is_tensor(source)
            or source.shape != reference_features.shape
            or source.dtype != reference_features.dtype
            or source.device != reference_features.device
        ):
            raise ValueError("each source feature must match reference [N,C,H,W], dtype, and device")
    if (
        not torch.is_tensor(feature_pixel_intrinsics)
        or feature_pixel_intrinsics.shape != (batch, 3, 3)
        or feature_pixel_intrinsics.dtype != reference_features.dtype
        or feature_pixel_intrinsics.device != reference_features.device
    ):
        raise ValueError(
            "feature_pixel_intrinsics from native preparation must match [N,3,3], dtype, and device"
        )
    for pose in poses:
        if (
            not torch.is_tensor(pose)
            or pose.shape != (batch, 4, 4)
            or pose.dtype != reference_features.dtype
            or pose.device != reference_features.device
        ):
            raise ValueError(
                "each relative reference-to-source pose must match [N,4,4], dtype, and device"
            )
    candidates = inverse_depth_candidates
    if candidates.ndim == 2:
        candidates = candidates[:, :, None, None]
    if (
        not torch.is_tensor(candidates)
        or candidates.ndim != 4
        or candidates.shape[0] != batch
        or candidates.shape[1] <= 0
        or candidates.shape[2:] != (1, 1)
        or candidates.dtype != reference_features.dtype
        or candidates.device != reference_features.device
        or not torch.isfinite(candidates).all()
        or bool((candidates <= 0).any())
    ):
        raise ValueError(
            "inverse_depth_candidates must be positive finite [N,D,1,1] values on the feature device"
        )
    return _MVSplatRawCostVolumeInputs(
        reference_features=reference_features,
        source_features=sources,
        feature_pixel_intrinsics=feature_pixel_intrinsics,
        relative_reference_to_source_poses=poses,
        inverse_depth_candidates=candidates,
    )


def _project_grid(
    feature_pixel_intrinsics: torch.Tensor,
    relative_reference_to_source_pose: torch.Tensor,
    inverse_depth_candidates: torch.Tensor,
    pixels: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reproduce MVSplat's camera projection for ``[N,3,P]`` pixel coordinates."""

    batch, _, positions = pixels.shape
    depth_count = inverse_depth_candidates.shape[1]
    points = torch.inverse(feature_pixel_intrinsics).bmm(pixels)
    metric_depth = inverse_depth_candidates.reciprocal().reshape(
        batch, 1, depth_count, 1
    )
    points = torch.bmm(relative_reference_to_source_pose[:, :3, :3], points).unsqueeze(2) * metric_depth
    points = points + relative_reference_to_source_pose[:, :3, -1:].unsqueeze(-1)
    points = torch.bmm(feature_pixel_intrinsics, points.reshape(batch, 3, -1)).reshape(
        batch, 3, depth_count, positions
    )
    denominator = points[:, 2].clamp(min=1e-3)
    x_grid = 2 * points[:, 0] / denominator / (width - 1) - 1
    y_grid = 2 * points[:, 1] / denominator / (height - 1) - 1
    return torch.stack((x_grid, y_grid), dim=-1)


def _dense_warp(
    source_features: torch.Tensor,
    feature_pixel_intrinsics: torch.Tensor,
    relative_reference_to_source_pose: torch.Tensor,
    inverse_depth_candidates: torch.Tensor,
) -> torch.Tensor:
    """Build MVSplat's native full raw warp ``[N,C,D,H,W]``."""

    batch, _channels, height, width = source_features.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=source_features.device, dtype=source_features.dtype),
        torch.arange(width, device=source_features.device, dtype=source_features.dtype),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=0).reshape(1, 3, -1)
    pixels = pixels.expand(batch, -1, -1)
    grid = _project_grid(
        feature_pixel_intrinsics,
        relative_reference_to_source_pose,
        inverse_depth_candidates,
        pixels,
        height=height,
        width=width,
    )
    depth_count = inverse_depth_candidates.shape[1]
    return F.grid_sample(
        source_features,
        grid.reshape(batch, depth_count * height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch, source_features.shape[1], depth_count, height, width)


def dense_mvsplat_raw_cost_volume(
    reference_features: torch.Tensor,
    source_features: Sequence[torch.Tensor],
    feature_pixel_intrinsics: torch.Tensor,
    relative_reference_to_source_poses: Sequence[torch.Tensor],
    inverse_depth_candidates: torch.Tensor,
) -> torch.Tensor:
    """Return MVSplat's pre-refinement raw correlation volume.

    The result corresponds to the first ``D`` channels supplied to
    ``DepthPredictorMultiView.corr_refine_net`` when the standard cost-volume
    path is enabled.  It deliberately excludes the concatenated reference
    feature channels and every downstream S2 operation.
    """

    inputs = _validate_inputs(
        reference_features,
        source_features,
        feature_pixel_intrinsics,
        relative_reference_to_source_poses,
        inverse_depth_candidates,
    )
    contributions = []
    for source, pose in zip(
        inputs.source_features, inputs.relative_reference_to_source_poses
    ):
        warped = _dense_warp(
            source,
            inputs.feature_pixel_intrinsics,
            pose,
            inputs.inverse_depth_candidates,
        )
        contributions.append(
            (inputs.reference_features.unsqueeze(2) * warped).sum(dim=1)
            / math.sqrt(inputs.reference_features.shape[1])
        )
    return torch.mean(torch.stack(contributions, dim=0), dim=0, keepdim=False)


class MVSplatSelectedRawCostVolumeProducer:
    """Execute selected MVSplat raw-correlation positions without replay.

    Masks operate only in the native low-resolution cost-volume grid.  A
    full-resolution Gaussian-head route mask is intentionally rejected rather
    than being implicitly resampled, because that mapping changes route
    semantics and has not been validated.
    """

    def __init__(
        self,
        reference_features: torch.Tensor,
        source_features: Sequence[torch.Tensor],
        feature_pixel_intrinsics: torch.Tensor,
        relative_reference_to_source_poses: Sequence[torch.Tensor],
        inverse_depth_candidates: torch.Tensor,
        *,
        tile_size: int | None = None,
    ) -> None:
        self._inputs = _validate_inputs(
            reference_features,
            source_features,
            feature_pixel_intrinsics,
            relative_reference_to_source_poses,
            inverse_depth_candidates,
        )
        batch, _channels, height, width = reference_features.shape
        if tile_size is not None and (
            isinstance(tile_size, bool)
            or not isinstance(tile_size, int)
            or tile_size <= 0
            or height % tile_size
            or width % tile_size
        ):
            raise ValueError("tile_size must divide the native cost-volume dimensions")
        self._batch = batch
        self._height = height
        self._width = width
        self._depth_count = int(self._inputs.inverse_depth_candidates.shape[1])
        self._tile_size = tile_size or max(height, width)
        self._values = torch.zeros(
            (batch, self._depth_count, height, width),
            dtype=reference_features.dtype,
            device=reference_features.device,
        )
        self._computed_mask = torch.zeros(
            (batch, height, width), dtype=torch.bool, device=reference_features.device
        )
        self._events: list[dict[str, Any]] = []
        self._seen_phases: set[str] = set()
        self._sealed = False

    @property
    def values(self) -> torch.Tensor:
        """Current sparse raw-correlation map; inspect with ``computed_mask``."""

        return self._values

    @property
    def computed_mask(self) -> torch.Tensor:
        """Native cost-volume positions completed by prior phases."""

        return self._computed_mask

    def _validate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if (
            not torch.is_tensor(mask)
            or mask.dtype != torch.bool
            or mask.shape != (self._batch, self._height, self._width)
            or mask.device != self._values.device
        ):
            raise ValueError(
                "selected raw-cost-volume masks must be [N,h,w] bool tensors on the native grid device"
            )
        return mask.detach().clone()

    def _validate_phase(self, phase: str) -> None:
        if phase not in _PHASE_ORDER:
            raise ValueError("phase must be primary, secondary, or full")
        if phase in self._seen_phases:
            raise ValueError(f"{phase} phase was already executed")
        expected = len(self._seen_phases)
        if _PHASE_ORDER[phase] != expected:
            raise ValueError("raw-cost-volume phases must execute primary -> secondary -> full")

    def _selected_correlation(
        self, batch_index: int, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Return ``[D,K]`` native raw correlations for one view-major item."""

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("selected coordinates must be [K,2] row/column pairs")
        rows = coordinates[:, 0]
        columns = coordinates[:, 1]
        pixels = torch.stack(
            (
                columns.to(dtype=self._values.dtype),
                rows.to(dtype=self._values.dtype),
                torch.ones_like(columns, dtype=self._values.dtype),
            ),
            dim=0,
        ).unsqueeze(0)
        reference = self._inputs.reference_features[batch_index, :, rows, columns]
        contributions = []
        for source, pose in zip(
            self._inputs.source_features,
            self._inputs.relative_reference_to_source_poses,
        ):
            grid = _project_grid(
                self._inputs.feature_pixel_intrinsics[batch_index : batch_index + 1],
                pose[batch_index : batch_index + 1],
                self._inputs.inverse_depth_candidates[batch_index : batch_index + 1],
                pixels,
                height=self._height,
                width=self._width,
            )
            warped = F.grid_sample(
                source[batch_index : batch_index + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0]
            contributions.append((reference.unsqueeze(1) * warped).sum(dim=0) / math.sqrt(reference.shape[0]))
        return torch.mean(torch.stack(contributions, dim=0), dim=0, keepdim=False)

    def execute(self, phase: str, selection_mask: torch.Tensor) -> dict[str, Any]:
        """Perform one logical phase and record reused versus new query work."""

        if self._sealed:
            raise RuntimeError("raw-cost-volume producer has already been finalized")
        self._validate_phase(phase)
        requested = self._validate_mask(selection_mask)
        missing = requested & ~self._computed_mask
        for batch_index in range(self._batch):
            coordinates = missing[batch_index].nonzero(as_tuple=False)
            if coordinates.numel() == 0:
                continue
            values = self._selected_correlation(batch_index, coordinates)
            rows, columns = coordinates.unbind(dim=1)
            self._values[batch_index, :, rows, columns] = values
        self._computed_mask |= missing
        requested_count = int(requested.sum().item())
        executed_count = int(missing.sum().item())
        event = {
            "phase": phase,
            "mask_sha256": _mask_sha256(requested),
            "raw_candidate_positions_requested": requested_count,
            "raw_candidate_positions_executed": executed_count,
            "raw_candidate_positions_reused": requested_count - executed_count,
            "depth_candidates_per_position": self._depth_count,
            "source_views_per_position": len(self._inputs.source_features),
            "candidate_evaluations_executed": executed_count * self._depth_count,
            "source_view_warp_evaluations_executed": (
                executed_count * self._depth_count * len(self._inputs.source_features)
            ),
        }
        self._events.append(event)
        self._seen_phases.add(phase)
        return dict(event)

    def finalize(self, required_mask: torch.Tensor | None = None) -> MVSplatRawCostVolumeReplay:
        """Seal the primary -> secondary -> Full execution trace."""

        if self._sealed:
            raise RuntimeError("raw-cost-volume producer has already been finalized")
        if self._seen_phases != set(_PHASE_ORDER):
            raise ValueError("primary, secondary, and full phases must all execute before finalizing")
        required = self._computed_mask if required_mask is None else self._validate_mask(required_mask)
        if bool((required & ~self._computed_mask).any()):
            raise ValueError("required raw-cost-volume positions have not been executed")
        events = [dict(event) for event in self._events]
        executed = int(self._computed_mask.sum().item())
        dense_positions = self._batch * self._height * self._width
        ledger = {
            "schema_version": MVSPLAT_RAW_COST_VOLUME_EXECUTION_VERSION,
            "available": True,
            "execution_scope": "mvsplat_raw_cost_volume_pre_refinement_only",
            "claim_scope": "diagnostic_only",
            "paper_result_eligible": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "s2_s3_savings_claimed": False,
            "native_s2_depth_output_available": False,
            "dense_refinement_required": True,
            "dense_refinement_reason": (
                "corr_refine_net and the full-resolution refinement path contain "
                "global normalization and cross-view closure"
            ),
            "identity_requires_external_dense_reference": True,
            "batch_size": self._batch,
            "native_cost_volume_height": self._height,
            "native_cost_volume_width": self._width,
            "depth_candidates": self._depth_count,
            "source_view_count": len(self._inputs.source_features),
            "dense_raw_candidate_positions": dense_positions,
            "raw_candidate_positions_executed": executed,
            "unrequested_diagnostic_positions": dense_positions - executed,
            "not_native_pipeline_work_avoided": True,
            "reference_feature_sha256": _tensor_sha256(self._inputs.reference_features),
            "source_feature_sha256": [
                _tensor_sha256(source) for source in self._inputs.source_features
            ],
            "feature_pixel_intrinsics_sha256": _tensor_sha256(
                self._inputs.feature_pixel_intrinsics
            ),
            "relative_reference_to_source_pose_sha256": [
                _tensor_sha256(pose)
                for pose in self._inputs.relative_reference_to_source_poses
            ],
            "inverse_depth_candidates_sha256": _tensor_sha256(
                self._inputs.inverse_depth_candidates
            ),
            "computed_mask_sha256": _mask_sha256(self._computed_mask),
            "phases": events,
            "phase_trace_sha256": _canonical_sha256(events),
        }
        self._sealed = True
        return MVSplatRawCostVolumeReplay(
            values=self._values.clone(),
            computed_mask=self._computed_mask.clone(),
            events=ledger,
        )

    def verify_against_dense(
        self,
        dense_raw_cost_volume: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-6,
    ) -> dict[str, Any]:
        """Compare completed selected values to a native dense raw-CV reference.

        This is the only identity check this primitive can provide.  A passing
        comparison does not establish native S2 depth, Full encoder, renderer,
        or quality equivalence.
        """

        if (
            not torch.is_tensor(dense_raw_cost_volume)
            or dense_raw_cost_volume.shape != self._values.shape
            or dense_raw_cost_volume.dtype != self._values.dtype
            or dense_raw_cost_volume.device != self._values.device
        ):
            raise ValueError("dense raw-cost-volume reference must match producer values")
        requested = self._computed_mask if mask is None else self._validate_mask(mask)
        if bool((requested & ~self._computed_mask).any()):
            raise ValueError("cannot verify raw-cost-volume positions that were not executed")
        expanded = requested.unsqueeze(1).expand_as(self._values)
        actual = self._values.masked_select(expanded)
        expected = dense_raw_cost_volume.masked_select(expanded)
        if actual.numel() == 0:
            raise ValueError("raw-cost-volume identity comparison requires at least one position")
        difference = (actual - expected).abs()
        return {
            "comparison_scope": "mvsplat_raw_cost_volume_selected_positions_only",
            "comparison_mask_sha256": _mask_sha256(requested),
            "compared_positions": int(requested.sum().item()),
            "compared_values": int(actual.numel()),
            "dense_raw_cost_volume_sha256": _tensor_sha256(dense_raw_cost_volume),
            "maximum_absolute_error": float(difference.max().item()),
            "equivalent": bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
        }
