"""
SAES: Scene-Adaptive Early Sparsification — Multi-Level v4 Simulator

Progressive Adaptive Early Sparsification for SCARF with 2-level tile classification
aligned with the SCARF Dataflow specification (L0 + L1 only):

  Level 0 (Feature Pre-Filter): Tiles with low feature variance after S1.
    Process K(T) probes through S2+S3 and merge assigned non-probe Gaussians
    with first/second-moment matching. Set non-probe opacity=0.
    Effective Gaussian count reduced to K(T).

  Level 1 (Depth-Based): Tiles that pass L0 but have uniform primary-probe
    depths. Use the paper's probe-constrained decision and depth-reliability
    term, then execute a deterministic 2K(T) native-anchor lightweight path.
    Set the remaining non-anchor opacity=0. At T=4 this retains 8 of 16
    Gaussians, making L1 less compressive than the 4-of-16 L0 path.

  Full: Tiles that fail both checks — no SAES savings.

The L0 path reads Stage-3 attributes only for the K(T) declared probe
Gaussians. L1 routing adds primary-probe depth evidence and a depth-reliability
factor; after that routing decision it executes the declared deterministic
2K(T) native anchors. Non-anchor assignment uses S1 features, tile coordinates,
and only those executed-anchor values; it never reads the remaining tile
positions.

K(T) probe count formula (Spec §Stage 3):
    K(T) = 4 + ceil(2·log₂(T/4))   for T ≥ 4
    e.g. T=4 → 4 probes, T=8 → 6, T=16 → 8
    Additional M=K(T)−4 probes selected from a G×G uniform subgrid (G=⌈√M⌉)
    of the non-corner interior, one probe per cell, nearest to cell centre.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from saes.probe_layout import (
    compute_lightweight_positions as _compute_lightweight_positions,
)
from saes.probe_layout import compute_probe_positions as _compute_probe_positions


def paper_assignment_weights(
    spatial_distances: torch.Tensor,
    feature_distances: torch.Tensor,
    *,
    feature_variance: float,
    beta_x: float,
    beta_f: float,
    level: str,
    probe_depths: torch.Tensor | None = None,
    depth_reference_depths: torch.Tensor | None = None,
    beta_d: float | None = None,
) -> torch.Tensor:
    """Compute the Section 3 bilateral L0/L1 probe-assignment weights.

    The L0 spatial term is explicitly modulated by the tile's probe-feature
    variance.  L1 reuses that bilateral term and multiplies it by the paper's
    probe-depth reliability factor, expressed in log space before softmax.
    """
    if spatial_distances.ndim != 1 or feature_distances.shape != spatial_distances.shape:
        raise ValueError("assignment distances must be equal-length vectors")
    if not math.isfinite(feature_variance) or feature_variance < 0.0:
        raise ValueError("feature_variance must be finite and nonnegative")
    if level not in {"L0", "L1"}:
        raise ValueError("assignment level must be L0 or L1")
    if beta_x <= 0.0 or beta_f <= 0.0:
        raise ValueError("assignment bandwidths must be positive")
    if not torch.isfinite(spatial_distances).all() or not torch.isfinite(feature_distances).all():
        raise ValueError("assignment distances must be finite")

    logits = -(
        float(feature_variance) * spatial_distances / (beta_x**2 + 1e-8)
        + feature_distances / (beta_f**2 + 1e-8)
    )
    if level == "L1":
        if probe_depths is None or probe_depths.shape != spatial_distances.shape:
            raise ValueError("L1 assignment requires one probe depth per weight")
        if beta_d is None or beta_d <= 0.0 or not torch.isfinite(probe_depths).all():
            raise ValueError("L1 assignment requires finite depths and positive beta_d")
        reference_depths = (
            probe_depths
            if depth_reference_depths is None
            else depth_reference_depths.reshape(-1).to(
                device=probe_depths.device, dtype=probe_depths.dtype
            )
        )
        if (
            reference_depths.numel() < 1
            or not torch.isfinite(reference_depths).all()
        ):
            raise ValueError("L1 depth reference must contain finite values")
        depth_mean = reference_depths.mean()
        depth_std = reference_depths.std(unbiased=False)
        logits = logits - (probe_depths - depth_mean).abs() / (beta_d * depth_std + 1e-8)
    return torch.softmax(logits, dim=0)


class ProgressiveSAES:
    """
    Progressive Adaptive Early Sparsification for SCARF (v4: Dataflow-aligned).

    Key changes vs v3:
      - K(T) adaptive probe count replaces fixed 4-corner probes.
      - Weighted moment matching (L0/L1) absorbs non-anchor Gaussians into
        the selected outputs.
      - Single-gate update (L2) blends toward selected-anchor means.
      - Non-anchor pixels have opacity set to 0 after an early stop. L0 retains
        K(T) representatives and the less-compressive L1 path retains 2K(T).
    """

    # ------------------------------------------------------------------ #
    # Static helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_probe_positions(tile_size: int) -> List[Tuple[int, int]]:
        """
        Compute K(T) adaptive probe positions inside a T×T tile.

        Spec: K(T) = 4 + ceil(2·log₂(T/4)), minimum 4 for T=4.
        Layout:
          - 4 corner probes: (0,0), (0,T-1), (T-1,0), (T-1,T-1)
          - M = K(T)-4 additional probes from a G×G uniform subgrid of the
            non-corner interior (rows 1..T-2, cols 1..T-2), with G=ceil(√M).
            For each cell one probe is placed at the pixel nearest to the cell
            centre; at most one probe per cell (spatial dispersion guarantee).
        """
        return _compute_probe_positions(tile_size)

    @staticmethod
    def compute_lightweight_positions(tile_size: int) -> List[Tuple[int, int]]:
        """Return deterministic L1 anchors with twice the L0 probe capacity.

        The paper leaves the lightweight retained-output count unspecified.
        This declared engineering detail keeps the primary probes first and
        uses farthest-point sampling for the additional anchors.  The route
        itself still depends only on the primary probes; after an L1 decision
        the 2K positions are executed as native S2/S3 outputs.
        """
        return _compute_lightweight_positions(tile_size)

    @staticmethod
    def _validate_camera_geometry(
        context_extrinsics: torch.Tensor,
        context_intrinsics: torch.Tensor,
        *,
        view_count: int,
        ray_depth_mode: str,
    ) -> None:
        if not torch.is_tensor(context_extrinsics) or not torch.is_tensor(
            context_intrinsics
        ):
            raise ValueError("context camera geometry must be tensors")
        if context_extrinsics.shape != (1, view_count, 4, 4):
            raise ValueError("context extrinsics must have shape [1, views, 4, 4]")
        if context_intrinsics.shape != (1, view_count, 3, 3):
            raise ValueError("context intrinsics must have shape [1, views, 3, 3]")
        if context_extrinsics.device != context_intrinsics.device:
            raise ValueError("context camera geometry must share one device")
        if ray_depth_mode not in {"euclidean", "z"}:
            raise ValueError("ray depth mode must be euclidean or z")

    @staticmethod
    def camera_world_point(
        context_extrinsics: torch.Tensor,
        context_intrinsics: torch.Tensor,
        *,
        view_index: int,
        row: int,
        column: int,
        height: int,
        width: int,
        depth: torch.Tensor,
        ray_depth_mode: str,
    ) -> torch.Tensor:
        """Lift one pixel-depth pair with the upstream normalized C2W convention."""
        ProgressiveSAES._validate_camera_geometry(
            context_extrinsics,
            context_intrinsics,
            view_count=context_extrinsics.shape[1]
            if torch.is_tensor(context_extrinsics) and context_extrinsics.ndim >= 2
            else 0,
            ray_depth_mode=ray_depth_mode,
        )
        if not isinstance(view_index, int) or not 0 <= view_index < context_extrinsics.shape[1]:
            raise ValueError("camera view index is out of range")
        if height <= 0 or width <= 0 or not 0 <= row < height or not 0 <= column < width:
            raise ValueError("camera pixel coordinate is out of range")
        if not torch.is_tensor(depth) or depth.numel() != 1:
            raise ValueError("camera depth must be a scalar tensor")

        dtype = context_extrinsics.dtype
        device = context_extrinsics.device
        coordinate = torch.tensor(
            ((column + 0.5) / width, (row + 0.5) / height, 1.0),
            device=device,
            dtype=dtype,
        )
        camera_direction = torch.linalg.solve(
            context_intrinsics[0, view_index], coordinate
        )
        if ray_depth_mode == "euclidean":
            camera_direction = camera_direction / camera_direction.norm().clamp_min(1e-8)
        else:
            denominator = camera_direction[2]
            if denominator.abs() < 1e-8:
                denominator = torch.full_like(denominator, 1e-8)
            camera_direction = camera_direction / denominator
        transform = context_extrinsics[0, view_index]
        world_direction = transform[:3, :3] @ camera_direction
        return transform[:3, 3] + world_direction * depth.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Initialisation                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        H: int,
        W: int,
        initial_tile_size: int = 4,
        feature_var_threshold: float = None,
        depth_std_threshold: float = None,
        cross_check_threshold: float = 0.02,
        view_count: int = 1,
        primitives_per_pixel: int = 1,
        materialization: str = "representative",
        decision_semantics: str = "current",
        beta_x: float = 0.50,
        beta_f: float = 0.10,
        beta_d: float = 1.00,
        num_depth_candidates: int = 1,
        context_extrinsics: torch.Tensor | None = None,
        context_intrinsics: torch.Tensor | None = None,
        ray_depth_mode: str = "euclidean",
    ):
        self.H = H
        self.W = W
        self.initial_tile_size = initial_tile_size
        # L0: tighter default so only truly flat/uniform regions early-stop.
        self.feature_var_threshold = (feature_var_threshold
                                      if feature_var_threshold is not None
                                      else 0.004)
        # L1: 4% relative depth std allows depth-uniform tiles to be caught.
        self.depth_std_threshold = (depth_std_threshold
                                    if depth_std_threshold is not None
                                    else 0.04)
        self.cross_check_threshold = cross_check_threshold
        if view_count < 1 or primitives_per_pixel < 1:
            raise ValueError("view_count and primitives_per_pixel must be positive")
        self.view_count = view_count
        self.primitives_per_pixel = primitives_per_pixel
        if materialization not in (
            "representative",
            "dense-diagnostic",
            "probe-spread-diagnostic",
            "transmittance-diagnostic",
            "virtual-reconstruction-diagnostic",
            "l1-primary-depth-reference-diagnostic",
            "conditional-anchor-transport-diagnostic",
        ):
            raise ValueError(f"unsupported SAES materialization: {materialization}")
        self.materialization = materialization
        # The paper's soft assignment associates a skipped position with its
        # receiving probe.  The historical implementation first formed an
        # unconditional mixture over *all* probes and then applied the same
        # assignment again while updating each probe.  This diagnostic instead
        # transports each selected anchor conditionally to its assigned target
        # position, avoiding an undocumented r_i,p * r_i,q cross-anchor term.
        self.merge_semantics = (
            "conditional-anchor-transport"
            if materialization == "conditional-anchor-transport-diagnostic"
            else "assignment-mixture"
        )
        # The paper defines the L1 depth reliability statistic over its primary
        # probe set.  The 2K native-anchor expansion is an explicitly declared
        # engineering detail, so this diagnostic keeps those extra anchors for
        # aggregation but anchors their reliability normalization to the K
        # routing probes instead of letting them redefine the L1 evidence.
        self.l1_depth_reference = (
            "primary-probes"
            if materialization == "l1-primary-depth-reference-diagnostic"
            else "retained-anchors"
        )
        if decision_semantics not in (
            "current",
            "probe-vector-first-hit",
            "probe-channel-variance-first-hit",
            "probe-normalized-std-first-hit",
        ):
            raise ValueError(f"unsupported SAES decision semantics: {decision_semantics}")
        self.decision_semantics = decision_semantics
        for label, value in (("beta_x", beta_x), ("beta_f", beta_f), ("beta_d", beta_d)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{label} must be finite and positive")
        self.beta_x = float(beta_x)
        self.beta_f = float(beta_f)
        self.beta_d = float(beta_d)
        if (
            isinstance(num_depth_candidates, bool)
            or not isinstance(num_depth_candidates, int)
            or num_depth_candidates <= 0
        ):
            raise ValueError("num_depth_candidates must be positive")
        self.num_depth_candidates = num_depth_candidates
        if (context_extrinsics is None) != (context_intrinsics is None):
            raise ValueError(
                "both context extrinsics and intrinsics are required for camera geometry"
            )
        self.context_extrinsics = context_extrinsics
        self.context_intrinsics = context_intrinsics
        self.ray_depth_mode = ray_depth_mode
        self._camera_origins = None
        self._camera_directions = None
        if context_extrinsics is not None:
            self._validate_camera_geometry(
                context_extrinsics,
                context_intrinsics,
                view_count=view_count,
                ray_depth_mode=ray_depth_mode,
            )
            self._camera_origins, self._camera_directions = self._build_camera_rays()

        # --- Adaptive probe positions (K(T) formula) ---
        T = initial_tile_size
        self.probe_positions: List[Tuple[int, int]] = self.compute_probe_positions(T)
        self.lightweight_positions: List[Tuple[int, int]] = (
            self.compute_lightweight_positions(T)
        )
        _probe_set = set(self.probe_positions)
        self.non_probe_positions: List[Tuple[int, int]] = [
            (r, c) for r in range(T) for c in range(T)
            if (r, c) not in _probe_set
        ]

        # Backward-compatible class-level aliases (read by external tooling)
        self.__class__.PROBE_POSITIONS = self.probe_positions
        self.__class__.NON_PROBE_POSITIONS = self.non_probe_positions

        # Statistics
        self.stats: Dict = {
            'decision_semantics': self.decision_semantics,
            'l1_depth_reference': self.l1_depth_reference,
            'merge_semantics': self.merge_semantics,
            'feature_statistic': (
                'raw-probe-mean-channel-variance'
                if self.decision_semantics == 'probe-channel-variance-first-hit'
                else 'raw-probe-vector-variance'
                if self.decision_semantics == 'probe-vector-first-hit'
                else 'normalized-probe-vector-standard-deviation'
                if self.decision_semantics == 'probe-normalized-std-first-hit'
                else 'normalized-probe-total-variance'
            ),
            'camera_aware_moment_matching': context_extrinsics is not None,
            'total_tiles_processed': 0,
            'level0_tiles': 0,
            'level1_tiles': 0,
            'full_tiles': 0,
            'level0_pixels': 0,
            'level1_pixels': 0,
            'pixels_original': 0,
            'total_modified_pixels': 0,
            'effective_gaussians': 0,
            'zeroed_gaussians': 0,
            'l0_representatives': 0,
            'l1_lightweight_anchors': 0,
            'full_stage3_gaussians': 0,
            'assignment_weight_sum_error_max': 0.0,
            'opacity_transmittance_error_max': 0.0,
            # Only the diagnostic optical-depth path uses this identity.  It
            # measures arithmetic agreement with the selected-anchor
            # surrogate, not agreement with withheld Stage-3 outputs.
            'optical_depth_assignment_error_max': 0.0,
            # Virtual reconstruction retains a descriptor at every pixel but
            # obtains its non-anchor descriptor from fixed-function anchor
            # interpolation rather than an S3 network evaluation.
            'virtual_reconstructed_gaussians': 0,
            'covariance_psd_violations': 0,
        }

    def _build_camera_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build C2W rays once for the resident model/sample session."""
        if self.context_extrinsics is None or self.context_intrinsics is None:
            raise RuntimeError("camera geometry was not configured")
        device = self.context_extrinsics.device
        dtype = self.context_extrinsics.dtype
        rows = (torch.arange(self.H, device=device, dtype=dtype) + 0.5) / self.H
        columns = (torch.arange(self.W, device=device, dtype=dtype) + 0.5) / self.W
        grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
        coordinates = torch.stack(
            (grid_x, grid_y, torch.ones_like(grid_x)), dim=-1
        )
        camera_directions = torch.einsum(
            "vij,hwj->vhwi",
            torch.linalg.inv(self.context_intrinsics[0]),
            coordinates,
        )
        if self.ray_depth_mode == "euclidean":
            camera_directions = camera_directions / camera_directions.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        else:
            denominator = camera_directions[..., 2:3]
            denominator = torch.where(
                denominator.abs() < 1e-8,
                torch.full_like(denominator, 1e-8),
                denominator,
            )
            camera_directions = camera_directions / denominator
        world_directions = torch.einsum(
            "vij,vhwj->vhwi", self.context_extrinsics[0, :, :3, :3], camera_directions
        )
        return self.context_extrinsics[0, :, :3, 3], world_directions

    def _camera_world_point(
        self, view_index: int, row: int, column: int, depth: torch.Tensor
    ) -> torch.Tensor:
        if self._camera_origins is None or self._camera_directions is None:
            raise RuntimeError("camera geometry was not configured")
        return (
            self._camera_origins[view_index]
            + self._camera_directions[view_index, row, column]
            * depth.to(
                device=self._camera_directions.device,
                dtype=self._camera_directions.dtype,
            )
        )

    def _camera_aware_interpolated_mean(
        self,
        source_means: torch.Tensor,
        source_depths: torch.Tensor,
        source_positions: List[Tuple[int, int]],
        *,
        view_index: int,
        target_row: int,
        target_column: int,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate a probe Gaussian at a new pixel without discarding offsets.

        Gaussian adaptors predict a sub-pixel image-plane offset in addition to
        depth.  A C2W ray reconstructed only from a target pixel centre loses
        that probe-produced geometric attribute.  Preserve it by transporting
        the assignment-weighted residual between each probe mean and its
        depth-only C2W ray to the target ray.  All inputs are probe outputs or
        S1/S2-derived quantities; this helper never needs a non-probe S3 value.
        """
        if self._camera_directions is None:
            raise RuntimeError("camera geometry was not configured")
        count = len(source_positions)
        if (
            source_means.shape != (count, 3)
            or source_depths.numel() != count
            or weights.numel() != count
        ):
            raise ValueError("camera interpolation inputs must align with probes")
        source_depths = source_depths.reshape(-1).to(
            device=source_means.device, dtype=source_means.dtype
        )
        weights = weights.reshape(-1).to(
            device=source_means.device, dtype=source_means.dtype
        )
        source_rays = torch.stack(
            [
                self._camera_world_point(view_index, row, column, depth)
                for (row, column), depth in zip(source_positions, source_depths)
            ],
            dim=0,
        )
        residuals = source_means - source_rays
        target_depth = torch.dot(weights, source_depths)
        target_ray = self._camera_world_point(
            view_index, target_row, target_column, target_depth
        )
        return target_ray + torch.einsum("n,ni->i", weights, residuals)

    def _anchor_conditioned_transport_means(
        self,
        source_mean: torch.Tensor,
        source_depth: torch.Tensor,
        source_position: Tuple[int, int],
        target_positions: List[Tuple[int, int]],
        *,
        view_index: int,
    ) -> torch.Tensor:
        """Transport one selected anchor to a batch of assigned positions.

        This is the conditional counterpart of the existing mixture transport:
        ``g_{i|p}`` uses only anchor ``p``'s depth and adaptor residual.  It
        never reads a skipped S2/S3 value and deliberately does not mix another
        anchor into p before the paper assignment weight is applied.
        """
        if source_mean.shape != (3,) or source_depth.numel() != 1:
            raise ValueError("anchor-conditioned transport requires one 3D source")
        if not target_positions:
            return source_mean.new_empty((0, 3))
        if self._camera_directions is None or self._camera_origins is None:
            return source_mean.unsqueeze(0).expand(len(target_positions), -1)
        source_row, source_column = source_position
        source_ray = self._camera_world_point(
            view_index, source_row, source_column, source_depth.reshape(())
        ).to(device=source_mean.device, dtype=source_mean.dtype)
        residual = source_mean - source_ray
        rows = torch.tensor(
            [row for row, _ in target_positions],
            device=self._camera_directions.device,
            dtype=torch.long,
        )
        columns = torch.tensor(
            [column for _, column in target_positions],
            device=self._camera_directions.device,
            dtype=torch.long,
        )
        directions = self._camera_directions[view_index, rows, columns].to(
            device=source_mean.device, dtype=source_mean.dtype
        )
        origin = self._camera_origins[view_index].to(
            device=source_mean.device, dtype=source_mean.dtype
        )
        return origin.unsqueeze(0) + directions * source_depth.reshape(1, 1).to(
            device=source_mean.device, dtype=source_mean.dtype
        ) + residual.unsqueeze(0)

    @staticmethod
    def _interpolate_intrinsic_covariance(
        source_covariances: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate only local Gaussian shape before the final moment merge.

        A selected-anchor-derived pseudo Gaussian already has its own C2W
        mean. Adding anchor-to-pseudo mean dispersion here and adding
        pseudo-to-anchor dispersion again in the final second-moment update
        counts the same spatial support twice. Keep the pseudo covariance
        intrinsic; the outer-product term belongs exactly once in the
        receiving anchor's first/second-moment match.
        """
        if (
            source_covariances.ndim != 3
            or source_covariances.shape[1:] != (3, 3)
            or weights.numel() != source_covariances.shape[0]
        ):
            raise ValueError("intrinsic covariance interpolation inputs must align")
        weights = weights.reshape(-1).to(
            device=source_covariances.device, dtype=source_covariances.dtype
        )
        covariance = torch.einsum("n,nij->ij", weights, source_covariances)
        return (covariance + covariance.mT) * 0.5

    def _assignment_feature_variance(self, decision_statistic: float) -> float:
        """Convert the decision statistic to the paper kernel's variance unit."""
        if math.isinf(decision_statistic) and decision_statistic > 0.0:
            return decision_statistic
        if not math.isfinite(decision_statistic) or decision_statistic < 0.0:
            raise ValueError("feature decision statistic must be finite and nonnegative")
        if self.decision_semantics == "probe-normalized-std-first-hit":
            return float(decision_statistic) ** 2
        return float(decision_statistic)

    # ------------------------------------------------------------------ #
    # Feature / depth classification helpers                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_tiles_by_features(
        features,
        h: int,
        w: int,
        tile_size: int,
        threshold: float = 0.02,
        per_view: bool = False,
        statistic: str = "current-channel-std",
    ) -> Tuple[Dict[Tuple[int, int], float], 'torch.Tensor']:
        """
        Classify tiles by feature variance from S1 feature maps.

        Returns:
            tile_variances: dict (th, tw) -> feature_variance (float)
            feat_norm:      normalised upsampled features [C, H, W]
        """
        if features.dim() == 5:
            feat = features[0]
        elif features.dim() == 4:
            feat = features
        else:
            return {}, None

        if not per_view:
            feat = feat.mean(dim=0, keepdim=True)
        feat_up = F.interpolate(
            feat, size=(h, w), mode='bilinear', align_corners=False
        )  # [V, C, H, W] or [1, C, H, W]
        feat_norm = feat_up / (feat_up.norm(dim=1, keepdim=True) + 1e-8)

        tiles_h = h // tile_size
        tiles_w = w // tile_size
        if statistic not in (
            "current-channel-std",
            "raw-probe-vector-variance",
            "raw-probe-mean-channel-variance",
            "normalized-probe-total-variance",
            "normalized-probe-vector-standard-deviation",
        ):
            raise ValueError(f"unsupported feature statistic: {statistic}")
        statistic_features = (
            feat_up
            if statistic in {
                "raw-probe-vector-variance",
                "raw-probe-mean-channel-variance",
            }
            else feat_norm
        )
        tiled = (
            statistic_features[:, :, : tiles_h * tile_size, : tiles_w * tile_size]
            .unfold(2, tile_size, tile_size)
            .unfold(3, tile_size, tile_size)
            .contiguous()
        )
        if statistic in (
            "raw-probe-vector-variance",
            "raw-probe-mean-channel-variance",
            "normalized-probe-total-variance",
            "normalized-probe-vector-standard-deviation",
        ):
            probes = ProgressiveSAES.compute_probe_positions(tile_size)
            probe_vectors = torch.stack(
                [tiled[..., local_y, local_x] for local_y, local_x in probes],
                dim=-1,
            )
            probe_mean = probe_vectors.mean(dim=-1, keepdim=True)
            squared_deviation = (probe_vectors - probe_mean).square()
            if statistic == "raw-probe-mean-channel-variance":
                # Sum-of-channel variance makes tau_f depend on the model's
                # feature width. The paper's scalar variance is the mean
                # component variance across the probe vectors.
                values = squared_deviation.mean(dim=(1, -1)).detach().cpu()
            else:
                values = squared_deviation.sum(dim=1).mean(dim=-1)
                if statistic == "normalized-probe-vector-standard-deviation":
                    values = values.sqrt()
                values = values.detach().cpu()
        else:
            values = (
                tiled.reshape(tiled.shape[0], tiled.shape[1], tiles_h, tiles_w, -1)
                .std(dim=-1)
                .mean(dim=1)
                .detach()
                .cpu()
            )
        if per_view:
            tile_variances = {
                (view, th, tw): float(values[view, th, tw])
                for view in range(values.shape[0])
                for th in range(tiles_h)
                for tw in range(tiles_w)
            }
            return tile_variances, feat_norm
        tile_variances = {
            (th, tw): float(values[0, th, tw])
            for th in range(tiles_h)
            for tw in range(tiles_w)
        }
        return tile_variances, feat_norm[0]

    @staticmethod
    def check_depth_uniformity(
        depths,
        th: int,
        tw: int,
        tile_size: int,
        h: int,
        w: int,
        threshold: float = 0.03,
        probe_positions: List[Tuple[int, int]] = None,
        view_index: int = 0,
        primitive_slot: int = 0,
        relative: bool = True,
    ) -> bool:
        """
        Check if probe pixel depths within a tile are uniform.

        probe_positions: list of (local_row, local_col) inside the tile.
        Falls back to 4-corner positions if not supplied.
        """
        if probe_positions is None:
            probe_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]

        tile_y, tile_x = th * tile_size, tw * tile_size
        probe_depths = []

        for (ly, lx) in probe_positions:
            gy, gx = tile_y + ly, tile_x + lx
            if gy >= h or gx >= w:
                return False
            if depths is None:
                return False
            if depths.dim() == 5:
                pixel_idx = gy * w + gx
                values = depths[0, view_index, pixel_idx].reshape(-1)
                if primitive_slot >= values.numel():
                    return False
                d = values[primitive_slot].item()
            elif depths.dim() == 4:
                d = depths[0, view_index, gy, gx].item()
            else:
                return False
            probe_depths.append(d)

        if len(probe_depths) < 2:
            return False

        d_mean = sum(probe_depths) / len(probe_depths)
        d_std = (sum((d - d_mean) ** 2 for d in probe_depths)
                 / len(probe_depths)) ** 0.5
        if not relative:
            return d_std < threshold
        ordered = sorted(probe_depths)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        scale = abs(median)
        if scale < 1e-8:
            return d_std < threshold
        normalized = [depth / scale for depth in probe_depths]
        normalized_mean = sum(normalized) / len(normalized)
        normalized_std = (
            sum((depth - normalized_mean) ** 2 for depth in normalized)
            / len(normalized)
        ) ** 0.5
        return normalized_std < threshold

    @staticmethod
    def compute_tile_similarity(
        gaussians_full,
        probe_indices: List[int],
        include_position: bool = True,
    ) -> float:
        """
        Compute appearance/geometry similarity among probe Gaussians.

        Weighted metric:
          With position (L2):  cov(0.30) + SH(0.30) + opacity(0.15) + pos(0.25)
          Without position:    cov(0.40) + SH(0.40) + opacity(0.20)
        """
        n = len(probe_indices)
        if n < 2:
            return 1.0

        total_sim = 0.0
        count = 0

        means = gaussians_full.means[0]
        covs = gaussians_full.covariances[0]
        harmo = gaussians_full.harmonics[0]
        opacs = gaussians_full.opacities[0]

        if include_position:
            probe_means = means[probe_indices]
            pos_std = probe_means.std(dim=0).mean().item()
            pos_range = probe_means.abs().max().item() + 1e-8
            pos_sim = 1.0 - min(pos_std / pos_range, 1.0)

        for i in range(n):
            for j in range(i + 1, n):
                idx_i, idx_j = probe_indices[i], probe_indices[j]

                cov1 = covs[idx_i].flatten()
                cov2 = covs[idx_j].flatten()
                cov_sim = (F.cosine_similarity(
                    cov1.unsqueeze(0), cov2.unsqueeze(0)).item() + 1) / 2

                sh1 = harmo[idx_i].flatten()
                sh2 = harmo[idx_j].flatten()
                sh_sim = (F.cosine_similarity(
                    sh1.unsqueeze(0), sh2.unsqueeze(0)).item() + 1) / 2

                op1 = (opacs[idx_i].item() if opacs[idx_i].dim() == 0
                       else opacs[idx_i].squeeze().item())
                op2 = (opacs[idx_j].item() if opacs[idx_j].dim() == 0
                       else opacs[idx_j].squeeze().item())
                opacity_sim = 1.0 - min(abs(op1 - op2), 1.0)

                if include_position:
                    pair_sim = (0.3 * cov_sim + 0.3 * sh_sim
                                + 0.15 * opacity_sim + 0.25 * pos_sim)
                else:
                    pair_sim = 0.4 * cov_sim + 0.4 * sh_sim + 0.2 * opacity_sim
                total_sim += pair_sim
                count += 1

        return total_sim / count if count > 0 else 1.0

    def probe_cross_check_error(
        self,
        gaussians_full,
        probe_indices: List[int],
    ) -> float:
        """
        Return the ASIC-implementable leave-one-out probe prediction error.

        For each probe, predict it from the average of the other probes and
        return the maximum covariance or harmonic cosine error.
        """
        K = len(probe_indices)
        if K < 2:
            return float("inf")

        harmo = gaussians_full.harmonics[0]
        covs = gaussians_full.covariances[0]

        max_err = 0.0
        for leave_out in range(K):
            others = [probe_indices[j] for j in range(K) if j != leave_out]

            pred_h = sum(harmo[o] for o in others) / len(others)
            actual_h = harmo[probe_indices[leave_out]]
            h_flat_pred = pred_h.flatten()
            h_flat_actual = actual_h.flatten()
            if h_flat_actual.norm() > 1e-8 and h_flat_pred.norm() > 1e-8:
                h_sim = F.cosine_similarity(
                    h_flat_pred.unsqueeze(0),
                    h_flat_actual.unsqueeze(0)).item()
                max_err = max(max_err, 1.0 - max(h_sim, 0.0))

            pred_c = sum(covs[o] for o in others) / len(others)
            actual_c = covs[probe_indices[leave_out]]
            c_flat_pred = pred_c.flatten()
            c_flat_actual = actual_c.flatten()
            if c_flat_actual.norm() > 1e-8 and c_flat_pred.norm() > 1e-8:
                c_sim = F.cosine_similarity(
                    c_flat_pred.unsqueeze(0),
                    c_flat_actual.unsqueeze(0)).item()
                max_err = max(max_err, 1.0 - max(c_sim, 0.0))

        return max_err

    def probe_cross_check(
        self,
        gaussians_full,
        probe_indices: List[int],
    ) -> bool:
        """Return whether the continuous probe error passes the configured gate."""
        return (
            self.probe_cross_check_error(gaussians_full, probe_indices)
            <= self.cross_check_threshold
        )

    # ------------------------------------------------------------------ #
    # Core interpolation / update methods                                  #
    # ------------------------------------------------------------------ #

    def _weighted_moment_match(
        self,
        gaussians_full,
        probe_indices: List[int],
        non_probe_map: Dict[Tuple[int, int], int],
        feat_norm,
        depths,
        level: str,
        tile_y: int,
        tile_x: int,
        view_index: int = 0,
        primitive_slot: int = 0,
        feature_variance: float = 0.0,
        retained_positions: List[Tuple[int, int]] = None,
        opacity_aggregation: str = "range-constrained-average",
        output_style: str = "representative",
        merge_semantics: str = "assignment-mixture",
    ):
        """Absorb a tile using only selected-anchor Stage-3 attributes.

        Non-anchor pixels contribute S1 feature and coordinate evidence. Their
        Gaussian moments are interpolated from the selected L0/L1 outputs
        before the paper's assignment and moment-conservation update.

        ``assignment-weighted-optical-depth`` is intentionally diagnostic
        only. It keeps the same router, selected anchors, and assignment
        matrix, but aggregates inferred selected-anchor optical depth rather
        than alpha averages. It never reads a skipped Stage-3 Gaussian.
        """
        T = self.initial_tile_size
        device = gaussians_full.means.device
        K = len(probe_indices)
        if K < 1:
            return
        if opacity_aggregation not in (
            "range-constrained-average",
            "assignment-weighted-optical-depth",
        ):
            raise ValueError(f"unsupported opacity aggregation: {opacity_aggregation}")
        if output_style not in ("representative", "virtual-reconstruction"):
            raise ValueError(f"unsupported SAES output style: {output_style}")
        if merge_semantics not in (
            "assignment-mixture",
            "conditional-anchor-transport",
        ):
            raise ValueError(f"unsupported SAES merge semantics: {merge_semantics}")
        if (
            output_style == "virtual-reconstruction"
            and merge_semantics != "assignment-mixture"
        ):
            raise ValueError(
                "virtual reconstruction requires unconditional anchor interpolation"
            )

        means = gaussians_full.means[0]
        covs = gaussians_full.covariances[0]
        harmonics = gaussians_full.harmonics[0]
        opacities = gaussians_full.opacities[0]

        # Flat Gaussian indices contain the view and primitive slot, so spatial
        # probe coordinates must be recovered from tile-local positions.
        retained_positions = retained_positions or self.probe_positions
        if len(retained_positions) != K:
            raise ValueError("retained position count does not match probe indices")
        probe_gy = [tile_y + local_y for local_y, _ in retained_positions]
        probe_gx = [tile_x + local_x for _, local_x in retained_positions]

        probe_feats = []
        probe_depths = []
        view_features = None
        if feat_norm is not None:
            view_features = feat_norm[view_index] if feat_norm.dim() == 4 else feat_norm
        for k in range(K):
            gy, gx = probe_gy[k], probe_gx[k]
            probe_feats.append(
                view_features[:, gy, gx]
                if (view_features is not None
                    and gy < view_features.shape[1]
                    and gx < view_features.shape[2])
                else None
            )
            if depths is not None:
                if depths.dim() == 5:
                    pixel_index = gy * self.W + gx
                    values = depths[0, view_index, pixel_index].reshape(-1)
                    slot = min(primitive_slot, values.numel() - 1)
                    probe_depths.append(values[slot].item())
                elif depths.dim() == 4:
                    probe_depths.append(depths[0, view_index, gy, gx].item())
                else:
                    probe_depths.append(1.0)
            else:
                probe_depths.append(1.0)

        probe_depth_tensor = torch.tensor(
            probe_depths, device=device, dtype=means.dtype
        )
        depth_reference_tensor = probe_depth_tensor
        if level == "L1" and self.l1_depth_reference == "primary-probes":
            primary_count = len(self.probe_positions)
            if (
                retained_positions[:primary_count] != self.probe_positions
                or primary_count > probe_depth_tensor.numel()
            ):
                raise ValueError(
                    "L1 retained anchors must begin with the primary probe set"
                )
            depth_reference_tensor = probe_depth_tensor[:primary_count]

        source_means = means[probe_indices].clone()
        source_covs = covs[probe_indices].clone()
        source_harmonics = harmonics[probe_indices].clone()
        source_opacities = opacities[probe_indices].clone()
        non_probe_items = list(non_probe_map.items())
        assignments = []
        coordinate_scale = max(T - 1, 1)
        for (local_y, local_x), _ in non_probe_items:
            gy = tile_y + local_y
            gx = tile_x + local_x
            feature_i = (view_features[:, gy, gx]
                  if view_features is not None
                  and gy < view_features.shape[1]
                  and gx < view_features.shape[2]
                  else None)
            spatial_distances = [
                ((gy - probe_gy[k]) / coordinate_scale) ** 2
                + ((gx - probe_gx[k]) / coordinate_scale) ** 2
                for k in range(K)
            ]
            feature_distances = []
            for k in range(K):
                feature = 0.0
                probe_feature = probe_feats[k]
                if (
                    feature_i is not None
                    and probe_feature is not None
                    and feature_i.shape == probe_feature.shape
                ):
                    feature = (feature_i - probe_feature).square().sum().item()
                feature_distances.append(feature)
            assignments.append(
                paper_assignment_weights(
                    torch.tensor(spatial_distances, device=device, dtype=means.dtype),
                    torch.tensor(feature_distances, device=device, dtype=means.dtype),
                    feature_variance=feature_variance,
                    beta_x=self.beta_x,
                    beta_f=self.beta_f,
                    level=level,
                    probe_depths=probe_depth_tensor if level == "L1" else None,
                    depth_reference_depths=(
                        depth_reference_tensor if level == "L1" else None
                    ),
                    beta_d=self.beta_d if level == "L1" else None,
                )
            )

        assignment_matrix = (
            torch.stack(assignments, dim=0)
            if assignments
            else torch.empty(0, K, device=device, dtype=means.dtype)
        )
        if assignment_matrix.numel():
            error = float(
                (assignment_matrix.sum(dim=1) - 1.0).abs().max().item()
            )
            self.stats['assignment_weight_sum_error_max'] = max(
                self.stats['assignment_weight_sum_error_max'], error
            )

        flat_harmonics = source_harmonics.reshape(K, -1)
        flat_opacities = source_opacities.reshape(K, -1).clamp(0.0, 1.0 - 1e-6)
        source_optical_depth = -torch.log1p(-flat_opacities)
        pseudo_means = None
        pseudo_harmonics = None
        pseudo_opacities = None
        pseudo_optical_depth = None
        pseudo_covariances = []
        if merge_semantics == "assignment-mixture" and assignments:
            pseudo_means = assignment_matrix @ source_means
            if self._camera_directions is not None:
                pseudo_means = torch.stack(
                    [
                        self._camera_aware_interpolated_mean(
                            source_means,
                            probe_depth_tensor,
                            list(zip(probe_gy, probe_gx)),
                            view_index=view_index,
                            target_row=tile_y + local_y,
                            target_column=tile_x + local_x,
                            weights=assignment,
                        )
                        for assignment, ((local_y, local_x), _) in zip(
                            assignment_matrix, non_probe_items
                        )
                    ],
                    dim=0,
                )
            pseudo_harmonics = assignment_matrix @ flat_harmonics
            pseudo_opacities = assignment_matrix @ flat_opacities
            pseudo_optical_depth = assignment_matrix @ source_optical_depth
            pseudo_covariances = [
                self._interpolate_intrinsic_covariance(source_covs, row)
                for row in assignment_matrix
            ]

        if output_style == "virtual-reconstruction":
            # Preserve a Gaussian descriptor at every skipped pixel while
            # avoiding its neural S2/S3 evaluation. The descriptor is fully
            # reconstructed from selected-anchor S1/S2/S3 values: C2W-aware
            # means, intrinsic SPD covariance, SH, and alpha. This is a
            # diagnostic engineering path only until its fixed-function work,
            # storage traffic, and quality are all independently audited.
            harmonic_shape = source_harmonics.shape[1:]
            opacity_shape = source_opacities.shape[1:]
            for pseudo_index, ((_, output_index), pseudo_covariance) in enumerate(
                zip(non_probe_items, pseudo_covariances)
            ):
                eigenvalues, eigenvectors = torch.linalg.eigh(pseudo_covariance)
                eigenvalues = eigenvalues.clamp_min(1e-8)
                reconstructed_covariance = (
                    eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.mT
                )
                if bool(
                    (torch.linalg.eigvalsh(reconstructed_covariance) < -1e-7)
                    .any()
                    .item()
                ):
                    self.stats['covariance_psd_violations'] += 1
                means[output_index] = pseudo_means[pseudo_index]
                covs[output_index] = reconstructed_covariance
                harmonics[output_index] = pseudo_harmonics[pseudo_index].reshape(
                    harmonic_shape
                )
                opacities[output_index] = pseudo_opacities[pseudo_index].clamp(
                    0.0, 1.0 - 1e-6
                ).reshape(opacity_shape)
            self.stats['virtual_reconstructed_gaussians'] += len(non_probe_items)
            return

        for probe_offset, probe_index in enumerate(probe_indices):
            absorbed = (
                assignment_matrix[:, probe_offset]
                if assignments
                else torch.empty(0, device=device, dtype=means.dtype)
            )
            weights = torch.cat((torch.ones(1, device=device, dtype=means.dtype), absorbed))
            normalized = weights / weights.sum().clamp_min(1e-8)
            if assignments and merge_semantics == "conditional-anchor-transport":
                target_positions = [position for position, _ in non_probe_items]
                conditional_means = self._anchor_conditioned_transport_means(
                    source_means[probe_offset],
                    probe_depth_tensor[probe_offset],
                    (probe_gy[probe_offset], probe_gx[probe_offset]),
                    [
                        (tile_y + local_y, tile_x + local_x)
                        for local_y, local_x in target_positions
                    ],
                    view_index=view_index,
                )
                conditional_covariances = source_covs[probe_offset].unsqueeze(0).expand(
                    len(non_probe_items), -1, -1
                )
                conditional_harmonics = flat_harmonics[probe_offset].unsqueeze(0).expand(
                    len(non_probe_items), -1
                )
                conditional_opacities = flat_opacities[probe_offset].unsqueeze(0).expand(
                    len(non_probe_items), -1
                )
                contributor_means = torch.cat(
                    (source_means[probe_offset].unsqueeze(0), conditional_means), dim=0
                )
                contributor_covs = torch.cat(
                    (source_covs[probe_offset].unsqueeze(0), conditional_covariances),
                    dim=0,
                )
                contributor_harmonics = torch.cat(
                    (flat_harmonics[probe_offset].unsqueeze(0), conditional_harmonics),
                    dim=0,
                )
                contributor_opacities = torch.cat(
                    (flat_opacities[probe_offset].unsqueeze(0), conditional_opacities),
                    dim=0,
                )
            else:
                contributor_means = (
                    torch.cat(
                        (source_means[probe_offset].unsqueeze(0), pseudo_means), dim=0
                    )
                    if assignments
                    else source_means[probe_offset].unsqueeze(0)
                )
                contributor_covs = (
                    torch.stack(
                        [source_covs[probe_offset], *pseudo_covariances], dim=0
                    )
                    if assignments
                    else source_covs[probe_offset].unsqueeze(0)
                )
                contributor_harmonics = (
                    torch.cat(
                        (flat_harmonics[probe_offset].unsqueeze(0), pseudo_harmonics),
                        dim=0,
                    )
                    if assignments
                    else flat_harmonics[probe_offset].unsqueeze(0)
                )
                contributor_opacities = (
                    torch.cat(
                        (flat_opacities[probe_offset].unsqueeze(0), pseudo_opacities),
                        dim=0,
                    )
                    if assignments
                    else flat_opacities[probe_offset].unsqueeze(0)
                )
            merged_mean = torch.einsum("n,ni->i", normalized, contributor_means)
            centered = contributor_means - merged_mean
            second_moment = contributor_covs + torch.einsum(
                "ni,nj->nij", centered, centered
            )
            merged_covariance = torch.einsum(
                "n,nij->ij", normalized, second_moment
            )
            merged_covariance = (
                merged_covariance + merged_covariance.mT
            ) * 0.5

            harmonic_shape = source_harmonics.shape[1:]
            merged_harmonics = torch.einsum(
                "n,nk->k", normalized, contributor_harmonics
            ).reshape(harmonic_shape)
            merged_harmonics = torch.maximum(
                torch.minimum(
                    merged_harmonics, source_harmonics.amax(dim=0)
                ),
                source_harmonics.amin(dim=0),
            )

            opacity_shape = source_opacities.shape[1:]
            if opacity_aggregation == "assignment-weighted-optical-depth":
                # Alpha compositing is additive in tau=-log(1-alpha).  Each
                # skipped position contributes an inferred tau, constructed
                # only from selected anchor opacities, to its soft-assigned
                # retained anchor. This restores the assignment-surrogate
                # mass without consulting the skipped Stage-3 outputs.
                if assignments and merge_semantics == "conditional-anchor-transport":
                    absorbed_optical_depth = (
                        absorbed.sum() * source_optical_depth[probe_offset]
                    )
                elif assignments:
                    absorbed_optical_depth = torch.einsum(
                        "n,nk->k", absorbed, pseudo_optical_depth
                    )
                else:
                    absorbed_optical_depth = torch.zeros_like(
                        source_optical_depth[probe_offset]
                    )
                merged_optical_depth = (
                    source_optical_depth[probe_offset] + absorbed_optical_depth
                )
                merged_opacity = (
                    -torch.expm1(-merged_optical_depth)
                ).clamp(0.0, 1.0 - 1e-6).reshape(opacity_shape)
            else:
                # Section 3 specifies a range-constrained opacity average. The
                # pseudo Gaussian contributes with the same assignment weight
                # used for its first/second moments; do not multiply
                # pseudo-opacity residuals into a new opaque cluster.
                merged_opacity = torch.einsum(
                    "n,nk->k", normalized, contributor_opacities
                )
                merged_opacity = torch.maximum(
                    torch.minimum(merged_opacity, contributor_opacities.amax(dim=0)),
                    contributor_opacities.amin(dim=0),
                ).clamp(0.0, 1.0 - 1e-6).reshape(opacity_shape)

            eigenvalues, eigenvectors = torch.linalg.eigh(merged_covariance)
            eigenvalues = eigenvalues.clamp_min(1e-8)
            merged_covariance = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.mT
            if bool((torch.linalg.eigvalsh(merged_covariance) < -1e-7).any().item()):
                self.stats['covariance_psd_violations'] += 1

            means[probe_index] = merged_mean
            covs[probe_index] = merged_covariance
            harmonics[probe_index] = merged_harmonics
            opacities[probe_index] = merged_opacity

        output_opacities = opacities[probe_indices].reshape(K, -1)
        if opacity_aggregation == "assignment-weighted-optical-depth":
            observed_optical_depth = -torch.log1p(
                -output_opacities.clamp(0.0, 1.0 - 1e-6)
            ).sum()
            expected_optical_depth = source_optical_depth.sum()
            if assignments and merge_semantics == "assignment-mixture":
                expected_optical_depth = expected_optical_depth + pseudo_optical_depth.sum()
            aggregation_error = float(
                (observed_optical_depth - expected_optical_depth).abs().item()
            )
            self.stats['optical_depth_assignment_error_max'] = max(
                self.stats['optical_depth_assignment_error_max'], aggregation_error
            )
        residual = 1.0 - output_opacities
        residual_error = float(
            torch.maximum((-residual).clamp_min(0.0), (residual - 1.0).clamp_min(0.0))
            .max()
            .item()
        )
        self.stats['opacity_transmittance_error_max'] = max(
            self.stats['opacity_transmittance_error_max'], residual_error
        )

    def _spread_probe_covariances(
        self,
        gaussians_full,
        probe_indices: List[int],
        non_probe_map: Dict[Tuple[int, int], int],
        feat_norm,
        depths,
        level: str,
        tile_y: int,
        tile_x: int,
        view_index: int = 0,
        primitive_slot: int = 0,
        retained_positions: List[Tuple[int, int]] = None,
    ) -> None:
        """Diagnostic probe-only covariance expansion from commit adc7092."""
        device = gaussians_full.means.device
        dtype = gaussians_full.means.dtype
        covariances = gaussians_full.covariances[0]
        retained_positions = retained_positions or self.probe_positions
        if len(retained_positions) != len(probe_indices):
            raise ValueError("retained position count does not match probe indices")
        probe_y = [tile_y + local_y for local_y, _ in retained_positions]
        probe_x = [tile_x + local_x for _, local_x in retained_positions]
        view_features = None
        if feat_norm is not None:
            view_features = feat_norm[view_index] if feat_norm.dim() == 4 else feat_norm

        probe_features = []
        probe_depths = []
        for gy, gx in zip(probe_y, probe_x):
            probe_features.append(
                view_features[:, gy, gx]
                if view_features is not None
                and gy < view_features.shape[1]
                and gx < view_features.shape[2]
                else None
            )
            if depths is not None and depths.dim() == 5:
                values = depths[0, view_index, gy * self.W + gx].reshape(-1)
                slot = min(primitive_slot, values.numel() - 1)
                probe_depths.append(float(values[slot].item()))
            elif depths is not None and depths.dim() == 4:
                probe_depths.append(float(depths[0, view_index, gy, gx].item()))
            else:
                probe_depths.append(1.0)

        count = len(probe_indices)
        spatial_bandwidth_sq = (self.initial_tile_size / 2.0) ** 2
        feature_bandwidth_sq = 0.09
        depth_bandwidth = 0.1
        depth_mean = sum(probe_depths) / max(count, 1)
        spread = [torch.zeros(3, 3, device=device, dtype=dtype) for _ in range(count)]
        weights = [0.0 for _ in range(count)]
        image_scale = float(max(self.W, self.H, 1))

        for local_y, local_x in non_probe_map:
            gy = tile_y + local_y
            gx = tile_x + local_x
            feature = (
                view_features[:, gy, gx]
                if view_features is not None
                and gy < view_features.shape[1]
                and gx < view_features.shape[2]
                else None
            )
            raw = []
            for index in range(count):
                spatial = math.exp(
                    -(
                        (gy - probe_y[index]) ** 2
                        + (gx - probe_x[index]) ** 2
                    )
                    / (spatial_bandwidth_sq + 1e-8)
                )
                probe_feature = probe_features[index]
                affinity = 1.0
                if (
                    feature is not None
                    and probe_feature is not None
                    and feature.shape == probe_feature.shape
                ):
                    affinity = math.exp(
                        -float((feature - probe_feature).square().sum().item())
                        / (feature_bandwidth_sq + 1e-8)
                    )
                weight = spatial * affinity
                if level == "L1":
                    weight *= math.exp(
                        -abs(probe_depths[index] - depth_mean)
                        / (depth_bandwidth + 1e-8)
                    )
                raw.append(weight)
            total = sum(raw) + 1e-8
            for index, value in enumerate(raw):
                normalized = value / total
                if normalized < 1e-6:
                    continue
                scale = max(probe_depths[index], 1e-4) / image_scale
                delta = torch.tensor(
                    [
                        (gx - probe_x[index]) * scale,
                        (gy - probe_y[index]) * scale,
                        0.0,
                    ],
                    device=device,
                    dtype=dtype,
                )
                spread[index] += normalized * torch.outer(delta, delta)
                weights[index] += normalized

        for index, probe_index in enumerate(probe_indices):
            if weights[index] <= 1e-8:
                continue
            addition = spread[index] / weights[index]
            addition = (addition + addition.mT) * 0.5
            addition += torch.eye(3, device=device, dtype=dtype) * 1e-6
            covariances[probe_index] += addition


    # ------------------------------------------------------------------ #
    # Deprecated / legacy                                                   #
    # ------------------------------------------------------------------ #

    def replicate_tile(self, gaussians_full, tile_y: int, tile_x: int,
                       probe_idx: int, other_indices: List[int]):
        """Legacy Level 0: replicate single probe to all other pixels (dead code)."""
        covs  = gaussians_full.covariances
        harmo = gaussians_full.harmonics
        opacs = gaussians_full.opacities
        p_cov  = covs[0, probe_idx]
        p_harm = harmo[0, probe_idx]
        p_opac = opacs[0, probe_idx]
        for idx in other_indices:
            covs[0, idx]  = p_cov * 1.02
            harmo[0, idx] = p_harm
            opacs[0, idx] = p_opac

    def _dense_interpolate_non_probes(
        self,
        gaussians_full,
        probe_indices: List[int],
        non_probe_map: Dict[Tuple[int, int], int],
    ) -> None:
        """Historical dense interpolation retained only for failure analysis."""
        if self.initial_tile_size != 4 or len(probe_indices) != 4:
            raise ValueError("dense diagnostic supports the default 4x4 tile only")
        covariances = gaussians_full.covariances[0]
        harmonics = gaussians_full.harmonics[0]
        opacities = gaussians_full.opacities[0]
        probe_covariances = covariances[probe_indices].clone()
        probe_harmonics = harmonics[probe_indices].clone()
        probe_opacities = opacities[probe_indices].clone()
        opacity_min = probe_opacities.amin(dim=0)
        opacity_max = probe_opacities.amax(dim=0)
        harmonic_norm = probe_harmonics.reshape(4, -1).norm(dim=1).mean()

        for (local_y, local_x), index in non_probe_map.items():
            y = local_y / 3.0
            x = local_x / 3.0
            weights = torch.tensor(
                [(1-y)*(1-x), (1-y)*x, y*(1-x), y*x],
                device=covariances.device,
                dtype=covariances.dtype,
            )
            covariances[index] = torch.einsum(
                "n,nij->ij", weights, probe_covariances
            )
            merged_harmonics = torch.einsum(
                "n,nk->k", weights, probe_harmonics.reshape(4, -1)
            ).reshape(probe_harmonics.shape[1:])
            merged_norm = merged_harmonics.norm().clamp_min(1e-8)
            merged_harmonics *= torch.clamp(
                harmonic_norm / merged_norm, 0.9, 1.1
            )
            harmonics[index] = merged_harmonics
            merged_opacity = torch.einsum(
                "n,nk->k", weights, probe_opacities.reshape(4, -1)
            ).reshape(probe_opacities.shape[1:])
            opacities[index] = torch.maximum(
                torch.minimum(merged_opacity, opacity_max), opacity_min
            )

    # ------------------------------------------------------------------ #
    # Main tile processing loop                                             #
    # ------------------------------------------------------------------ #

    def process_all_tiles(
        self,
        gaussians_full,
        gpp: int = 1,
        tile_variances: Dict = None,
        depths=None,
        feat_norm=None,
    ) -> Tuple['torch.Tensor', Dict]:
        """
        Multi-level SAES v4 tile processing (Dataflow-aligned, L0+L1 only).

        For each tile:
          L0 — feature variance check → weighted moment matching (space+feat)
          L1 — primary-probe depth check → execute 2K(T) lightweight anchors
          Full — no modification

        Non-anchor pixels in an early-stopped tile have their opacity set to 0
        after moment matching. L0 retains K(T); L1 retains up to 2K(T)
        deterministic native anchors.

        Returns:
            modified_mask: bool tensor, True for pixels modified (L0/L1)
            stats:         per-level statistics including effective_gaussians
        """
        gaussian_count = gaussians_full.means.shape[1]
        device = gaussians_full.means.device
        if (
            self._camera_directions is not None
            and self._camera_directions.device != device
        ):
            raise ValueError("camera geometry and Gaussian tensors must share one device")
        if (
            self._camera_directions is not None
            and self._camera_directions.dtype != gaussians_full.means.dtype
        ):
            raise ValueError("camera geometry and Gaussian tensors must share one dtype")
        position_count = self.view_count * self.H * self.W
        expected_gaussians = position_count * self.primitives_per_pixel
        if gaussian_count != expected_gaussians:
            raise ValueError(
                "flattened Gaussian count does not match view/pixel layout: "
                f"got {gaussian_count}, expected {expected_gaussians}"
            )

        modified_mask = torch.zeros(
            gaussian_count, dtype=torch.bool, device=device
        )
        for key in self.stats:
            if key not in {
                'decision_semantics',
                'feature_statistic',
                'camera_aware_moment_matching',
                'l1_depth_reference',
                'merge_semantics',
            }:
                self.stats[key] = 0

        tiles_h = self.H // self.initial_tile_size
        tiles_w = self.W // self.initial_tile_size
        tile_size = self.initial_tile_size
        total_zeroed = 0

        def flat_index(view: int, pixel: int, primitive_slot: int) -> int:
            return (
                (view * self.H * self.W + pixel) * self.primitives_per_pixel
                + primitive_slot
            )

        for view in range(self.view_count):
            for th in range(tiles_h):
                for tw in range(tiles_w):
                    self.stats['total_tiles_processed'] += 1
                    tile_y = th * tile_size
                    tile_x = tw * tile_size

                    def indices_for_positions(
                        slot: int, positions: List[Tuple[int, int]]
                    ) -> List[int]:
                        return [
                            flat_index(
                                view,
                                (tile_y + local_y) * self.W + tile_x + local_x,
                                slot,
                            )
                            for local_y, local_x in positions
                        ]

                    def non_anchor_map(
                        slot: int, positions: List[Tuple[int, int]]
                    ) -> Dict[Tuple[int, int], int]:
                        retained = set(positions)
                        return {
                            (local_y, local_x): flat_index(
                                view,
                                (tile_y + local_y) * self.W + tile_x + local_x,
                                slot,
                            )
                            for local_y in range(tile_size)
                            for local_x in range(tile_size)
                            if (local_y, local_x) not in retained
                        }

                    first_probes = indices_for_positions(0, self.probe_positions)
                    feature_key = (view, th, tw)
                    feature_variance = (
                        tile_variances.get(
                            feature_key, tile_variances.get((th, tw), float('inf'))
                        )
                        if tile_variances is not None
                        else float('inf')
                    )
                    assignment_feature_variance = self._assignment_feature_variance(
                        feature_variance
                    )

                    selected_level = None
                    if self.decision_semantics == "probe-vector-first-hit":
                        if feature_variance < self.feature_var_threshold:
                            selected_level = 'L0'
                        elif depths is not None and self.check_depth_uniformity(
                            depths,
                            th,
                            tw,
                            tile_size,
                            self.H,
                            self.W,
                            self.depth_std_threshold,
                            probe_positions=self.probe_positions,
                            view_index=view,
                            relative=False,
                        ):
                            selected_level = 'L1'
                    else:
                        if feature_variance < self.feature_var_threshold:
                            selected_level = 'L0'
                        elif depths is not None and self.check_depth_uniformity(
                            depths,
                            th,
                            tw,
                            tile_size,
                            self.H,
                            self.W,
                            self.depth_std_threshold,
                            probe_positions=self.probe_positions,
                            view_index=view,
                            relative=False,
                        ):
                            selected_level = 'L1'

                    if selected_level is not None:
                        retained_positions = (
                            self.probe_positions
                            if selected_level == 'L0'
                            else self.lightweight_positions
                        )
                        for slot in range(self.primitives_per_pixel):
                            if self.materialization == "dense-diagnostic":
                                materialized_positions = self.probe_positions
                            else:
                                materialized_positions = retained_positions
                            probes = indices_for_positions(slot, materialized_positions)
                            non_probes = non_anchor_map(slot, materialized_positions)
                            if self.materialization in (
                                "representative",
                                "transmittance-diagnostic",
                                "virtual-reconstruction-diagnostic",
                                "l1-primary-depth-reference-diagnostic",
                                "conditional-anchor-transport-diagnostic",
                            ):
                                # L1 routing uses only primary probes. After it
                                # succeeds, its deterministic 2K positions are
                                # charged native S2/S3 outputs; only the rest
                                # of the tile is skipped.
                                self._weighted_moment_match(
                                    gaussians_full,
                                    probes,
                                    non_probes,
                                    feat_norm,
                                    depths,
                                    selected_level,
                                    tile_y,
                                    tile_x,
                                    view_index=view,
                                    primitive_slot=slot,
                                    feature_variance=assignment_feature_variance,
                                    retained_positions=materialized_positions,
                                    opacity_aggregation=(
                                        "assignment-weighted-optical-depth"
                                        if self.materialization == "transmittance-diagnostic"
                                        else "range-constrained-average"
                                    ),
                                    output_style=(
                                        "virtual-reconstruction"
                                        if self.materialization
                                        == "virtual-reconstruction-diagnostic"
                                        else "representative"
                                    ),
                                    merge_semantics=self.merge_semantics,
                                )
                            elif self.materialization == "probe-spread-diagnostic":
                                self._spread_probe_covariances(
                                    gaussians_full,
                                    probes,
                                    non_probes,
                                    feat_norm,
                                    depths,
                                    selected_level,
                                    tile_y,
                                    tile_x,
                                    view_index=view,
                                    primitive_slot=slot,
                                    retained_positions=materialized_positions,
                                )
                            else:
                                self._dense_interpolate_non_probes(
                                    gaussians_full, probes, non_probes
                                )
                            preserves_virtual_outputs = self.materialization in (
                                "dense-diagnostic",
                                "virtual-reconstruction-diagnostic",
                            )
                            for index in non_probes.values():
                                if not preserves_virtual_outputs:
                                    gaussians_full.opacities[0, index] *= 0.0
                                modified_mask[index] = True
                            if not preserves_virtual_outputs:
                                total_zeroed += len(non_probes)

                        pixel_count = tile_size * tile_size - len(retained_positions)
                        if selected_level == 'L0':
                            self.stats['level0_tiles'] += 1
                            self.stats['level0_pixels'] += pixel_count
                            self.stats['l0_representatives'] += (
                                len(retained_positions) * self.primitives_per_pixel
                            )
                        else:
                            self.stats['level1_tiles'] += 1
                            self.stats['level1_pixels'] += pixel_count
                            self.stats['l1_lightweight_anchors'] += (
                                len(retained_positions) * self.primitives_per_pixel
                            )
                        self.stats['pixels_original'] += len(retained_positions)
                        continue

                    self.stats['full_tiles'] += 1
                    self.stats['full_stage3_gaussians'] += (
                        tile_size * tile_size * self.primitives_per_pixel
                    )
                    self.stats['pixels_original'] += tile_size * tile_size

        total_tiles = max(1, self.stats['total_tiles_processed'])
        self.stats['total_modified_pixels'] = (
            self.stats['level0_pixels'] + self.stats['level1_pixels']
        )
        self.stats['zeroed_gaussians'] = total_zeroed
        self.stats['effective_gaussians'] = max(0, gaussian_count - total_zeroed)
        self.stats['full_s2_evaluations'] = (
            self.stats['total_tiles_processed']
            * tile_size
            * tile_size
            * self.primitives_per_pixel
            * self.num_depth_candidates
        )
        self.stats['executed_s2_evaluations'] = (
            self.stats['l0_representatives']
            + self.stats['l1_lightweight_anchors']
            + self.stats['full_stage3_gaussians']
        ) * self.num_depth_candidates
        self.stats['s2_evaluations_available'] = True

        self.stats['level0_ratio'] = self.stats['level0_tiles'] / total_tiles
        self.stats['level1_ratio'] = self.stats['level1_tiles'] / total_tiles
        self.stats['full_ratio'] = self.stats['full_tiles'] / total_tiles
        self.stats['early_stop_ratio'] = 1.0 - self.stats['full_ratio']
        self.stats['modification_ratio'] = (
            self.stats['total_modified_pixels'] / position_count
            if position_count > 0
            else 0.0
        )

        # Backward-compatible keys
        self.stats['early_stop_phase1'] = (self.stats['level0_tiles'] +
                                           self.stats['level1_tiles'])
        self.stats['early_stop_phase2']    = 0
        self.stats['full_processed']       = self.stats['full_tiles']
        self.stats['pixels_interpolated']  = self.stats['total_modified_pixels']
        self.stats['interpolation_ratio']  = self.stats['modification_ratio']

        return modified_mask, self.stats


# -------------------------------------------------------------------- #
# Public API                                                              #
# -------------------------------------------------------------------- #

def apply_progressive_saes(
    gaussians_full,
    H: int,
    W: int,
    tile_size: int = 4,
    gpp: int = 1,
    feature_var_threshold: float = None,
    depth_std_threshold: float = None,
    features=None,
    depths=None,
    cross_check_threshold: float = 0.015,
    feat_norm=None,            # pre-computed [C, H, W]; computed internally if None
    collect_continue_pixels: bool = False,
    view_count: int = None,
    materialization: str = "representative",
    decision_semantics: str = "current",
    beta_x: float = 0.50,
    beta_f: float = 0.10,
    beta_d: float = 1.00,
    num_depth_candidates: int = 1,
    context_extrinsics: torch.Tensor | None = None,
    context_intrinsics: torch.Tensor | None = None,
    ray_depth_mode: str = "euclidean",
) -> Tuple['torch.Tensor', Dict, List]:
    """
    Apply progressive SAES v4 (Dataflow-aligned, L0+L1) to Gaussians.

    Two-level progressive early-stopping:
      - L0: feature variance check → weighted moment matching (space+feat)
      - L1: primary-probe depth check → execute 2K(T) anchor moment matching
      - Full: no modification

    Non-anchor opacities are zeroed in early-stopped tiles. L0 retains K(T)
    representatives and L1 retains up to 2K(T) deterministic native anchors.

    Args:
        features:  S1 features [B, V, C, H, W] for L0 classification
        depths:    S2 depths   [B, V, H, W] for L1 check
        feat_norm: pre-normalised features [C, H, W] (optional)

    Returns:
        modified_mask:   bool tensor (True = primitive was changed or removed)
        stats:           per-level statistics (incl. effective_gaussians)
        continue_pixels: list of (y, x, pixel_idx) for unmodified pixels
    """
    gaussian_count = gaussians_full.means.shape[1]
    if view_count is None:
        if features is not None and hasattr(features, 'dim') and features.dim() == 5:
            view_count = int(features.shape[1])
        elif depths is not None and hasattr(depths, 'dim') and depths.dim() >= 4:
            view_count = int(depths.shape[1])
        else:
            view_count = 1
    position_count = view_count * H * W
    if gaussian_count % position_count != 0:
        raise ValueError(
            "Gaussian count is not divisible by view_count * H * W: "
            f"{gaussian_count} vs {position_count}"
        )
    primitives_per_pixel = gaussian_count // position_count

    tile_variances = None
    _feat_norm = feat_norm  # use caller-supplied if available

    if features is not None and hasattr(features, 'shape'):
        _tv, _fn = ProgressiveSAES.classify_tiles_by_features(
            features, H, W, tile_size,
            threshold=(feature_var_threshold
                       if feature_var_threshold is not None else 0.012),
            per_view=True,
            statistic=(
                "raw-probe-vector-variance"
                if decision_semantics == "probe-vector-first-hit"
                else "raw-probe-mean-channel-variance"
                if decision_semantics == "probe-channel-variance-first-hit"
                else "normalized-probe-vector-standard-deviation"
                if decision_semantics == "probe-normalized-std-first-hit"
                else "normalized-probe-total-variance"
            ),
        )
        tile_variances = _tv
        if _feat_norm is None:
            _feat_norm = _fn

    saes = ProgressiveSAES(
        H, W, tile_size,
        feature_var_threshold=feature_var_threshold,
        depth_std_threshold=depth_std_threshold,
        cross_check_threshold=cross_check_threshold,
        view_count=view_count,
        primitives_per_pixel=primitives_per_pixel,
        materialization=materialization,
        decision_semantics=decision_semantics,
        beta_x=beta_x,
        beta_f=beta_f,
        beta_d=beta_d,
        num_depth_candidates=num_depth_candidates,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        ray_depth_mode=ray_depth_mode,
    )
    modified_mask, stats = saes.process_all_tiles(
        gaussians_full, gpp,
        tile_variances=tile_variances,
        depths=depths,
        feat_norm=_feat_norm,
    )

    continue_pixels = []
    if collect_continue_pixels:
        indices = torch.nonzero(~modified_mask, as_tuple=False).flatten().cpu().tolist()
        for idx in indices:
            pixel_idx = idx // primitives_per_pixel
            y, x = divmod(pixel_idx, W)
            y %= H
            if x < W:
                continue_pixels.append((y, x, pixel_idx))

    return modified_mask, stats, continue_pixels
