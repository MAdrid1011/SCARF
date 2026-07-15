"""
SAES: Scene-Adaptive Early Sparsification — Multi-Level v4 Simulator

Progressive Adaptive Early Sparsification for SCARF with 2-level tile classification
aligned with the SCARF Dataflow specification (L0 + L1 only):

  Level 0 (Feature Pre-Filter): Tiles with low feature variance after S1.
    Process K(T) probes through S2+S3 and merge assigned non-probe Gaussians
    with first/second-moment matching. Set non-probe opacity=0.
    Effective Gaussian count reduced to K(T).

  Level 1 (Depth-Based): Tiles that pass L0 but have uniform probe depths.
    Same spatial-coverage expansion with additional depth-derived weighting.
    Set non-probe opacity=0.

  Full: Tiles that fail both checks — no SAES savings.

NOTE: the representative quality simulator reads the complete pretrained
Gaussian tile to evaluate the paper's moment-matching equation. This is not
equivalent to proving that early hardware can obtain those non-probe attributes.
The public claim contract therefore does not claim the current sparse-SAES
Table 1 or mechanism rows. The dense diagnostic is explicitly non-claiming.

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


class ProgressiveSAES:
    """
    Progressive Adaptive Early Sparsification for SCARF (v4: Dataflow-aligned).

    Key changes vs v3:
      - K(T) adaptive probe count replaces fixed 4-corner probes.
      - Weighted moment matching (L0/L1) absorbs non-probe Gaussians into probes.
      - Single-gate update (L2) blends toward probe mean.
      - Non-probe pixels have opacity set to 0 after any early-stop, reducing
        the effective Gaussian count for early-stopped tiles to at most K(T).
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
        T = tile_size
        if T <= 4:
            K = 4
        else:
            K = 4 + math.ceil(2 * math.log2(T / 4))

        corners = [(0, 0), (0, T - 1), (T - 1, 0), (T - 1, T - 1)]
        probes: List[Tuple[int, int]] = list(corners)
        probes_set: set = set(corners)

        M = K - 4
        if M > 0 and T > 2:
            G = math.ceil(math.sqrt(M))
            added = 0
            for gi in range(G):
                if added >= M:
                    break
                for gj in range(G):
                    if added >= M:
                        break
                    # Cell centre in non-corner interior [1, T-2]
                    cy = 1.0 + (gi + 0.5) * (T - 2) / G
                    cx = 1.0 + (gj + 0.5) * (T - 2) / G
                    best_dist = float('inf')
                    best_pos: Tuple[int, int] = None
                    for r in range(1, T - 1):
                        for c in range(1, T - 1):
                            if (r, c) not in probes_set:
                                d = (r - cy) ** 2 + (c - cx) ** 2
                                if d < best_dist:
                                    best_dist = d
                                    best_pos = (r, c)
                    if best_pos is not None:
                        probes.append(best_pos)
                        probes_set.add(best_pos)
                        added += 1

        return probes

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
        if materialization not in ("representative", "dense-diagnostic"):
            raise ValueError(f"unsupported SAES materialization: {materialization}")
        self.materialization = materialization

        # --- Adaptive probe positions (K(T) formula) ---
        T = initial_tile_size
        self.probe_positions: List[Tuple[int, int]] = self.compute_probe_positions(T)
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
        }

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
        tiled = (
            feat_norm[:, :, : tiles_h * tile_size, : tiles_w * tile_size]
            .unfold(2, tile_size, tile_size)
            .unfold(3, tile_size, tile_size)
            .contiguous()
        )
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
        if abs(d_mean) < 1e-8:
            return True
        d_std = (sum((d - d_mean) ** 2 for d in probe_depths)
                 / len(probe_depths)) ** 0.5
        return d_std / (abs(d_mean) + 1e-8) < threshold

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

    def probe_cross_check(
        self,
        gaussians_full,
        probe_indices: List[int],
    ) -> bool:
        """
        ASIC-implementable quality validation: leave-one-out cross-check.

        For each probe, predict it from the average of the other probes.
        Returns True if max prediction error <= cross_check_threshold.
        Works for any K ≥ 2 probes.
        """
        K = len(probe_indices)
        if K < 2:
            return False

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

        return max_err <= self.cross_check_threshold

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
    ):
        """
        Merge non-probe Gaussians into probe representatives.

        The assignment is the paper's joint spatial/feature kernel. L1 also
        applies the depth reliability factor. Each representative is then
        updated with the weighted first moment and the law of total covariance;
        SH and opacity use range-constrained weighted averages. Source tensors
        are cloned before any in-place update so every representative observes
        the same pre-merge tile.
        """
        T = self.initial_tile_size
        # The paper's x_i is tile-normalized. Keeping pixel coordinates here
        # makes the spatial kernel about two orders of magnitude too broad and
        # collapses all representatives toward the tile centre.
        beta_x_sq = 0.05 ** 2
        beta_f_sq = 0.09
        beta_d = 1.0

        device = gaussians_full.means.device
        K = len(probe_indices)
        if K < 1:
            return

        means = gaussians_full.means[0]
        covs = gaussians_full.covariances[0]
        harmonics = gaussians_full.harmonics[0]
        opacities = gaussians_full.opacities[0]

        # Flat Gaussian indices contain the view and primitive slot, so spatial
        # probe coordinates must be recovered from tile-local positions.
        probe_gy = [tile_y + local_y for local_y, _ in self.probe_positions[:K]]
        probe_gx = [tile_x + local_x for _, local_x in self.probe_positions[:K]]

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

        non_probe_items = list(non_probe_map.items())
        assignments = []
        depth_mean = sum(probe_depths) / K
        depth_std = (
            sum((depth - depth_mean) ** 2 for depth in probe_depths) / K
        ) ** 0.5
        for (local_y, local_x), _ in non_probe_items:
            gy = tile_y + local_y
            gx = tile_x + local_x
            feature_i = (view_features[:, gy, gx]
                  if view_features is not None
                  and gy < view_features.shape[1]
                  and gx < view_features.shape[2]
                  else None)
            logits = []
            for k in range(K):
                qy, qx = probe_gy[k], probe_gx[k]
                coordinate_scale = max(T - 1, 1)
                spatial = feature_variance * (
                    ((gy - qy) / coordinate_scale) ** 2
                    + ((gx - qx) / coordinate_scale) ** 2
                ) / (beta_x_sq + 1e-8)
                feature = 0.0
                probe_feature = probe_feats[k]
                if (
                    feature_i is not None
                    and probe_feature is not None
                    and feature_i.shape == probe_feature.shape
                ):
                    feature = (feature_i - probe_feature).square().sum().item()
                    feature /= beta_f_sq + 1e-8
                log_weight = -(spatial + feature)
                if level == 'L1':
                    log_weight -= abs(probe_depths[k] - depth_mean) / (
                        beta_d * depth_std + 1e-8
                    )
                logits.append(log_weight)
            assignments.append(
                torch.softmax(
                    torch.tensor(logits, device=device, dtype=means.dtype), dim=0
                )
            )

        assignment_matrix = (
            torch.stack(assignments, dim=0)
            if assignments
            else torch.empty(0, K, device=device, dtype=means.dtype)
        )
        non_probe_indices = [flat_idx for _, flat_idx in non_probe_items]
        source_indices = probe_indices + non_probe_indices
        source_means = means[source_indices].clone()
        source_covs = covs[source_indices].clone()
        source_harmonics = harmonics[source_indices].clone()
        source_opacities = opacities[source_indices].clone()

        for probe_offset, probe_index in enumerate(probe_indices):
            weights = torch.cat(
                (
                    torch.ones(1, device=device, dtype=means.dtype),
                    assignment_matrix[:, probe_offset],
                )
            )
            contributor_indices = [probe_offset] + list(range(K, len(source_indices)))
            contributor = torch.tensor(
                contributor_indices, device=device, dtype=torch.long
            )
            contributor_means = source_means[contributor]
            contributor_opacities = source_opacities[contributor]
            opacity_flat = contributor_opacities.reshape(len(weights), -1).mean(dim=1)
            moment_weights = weights * opacity_flat.clamp_min(1e-6)
            normalized = moment_weights / moment_weights.sum().clamp_min(1e-8)
            merged_mean = torch.einsum("n,ni->i", normalized, contributor_means)
            centered = contributor_means - merged_mean
            second_moment = source_covs[contributor] + torch.einsum(
                "ni,nj->nij", centered, centered
            )
            merged_covariance = torch.einsum(
                "n,nij->ij", normalized, second_moment
            )
            merged_covariance = (
                merged_covariance + merged_covariance.mT
            ) * 0.5

            contributor_harmonics = source_harmonics[contributor]
            harmonic_shape = contributor_harmonics.shape[1:]
            merged_harmonics = torch.einsum(
                "n,nk->k", normalized, contributor_harmonics.reshape(len(weights), -1)
            ).reshape(harmonic_shape)
            merged_harmonics = torch.maximum(
                torch.minimum(
                    merged_harmonics, contributor_harmonics.amax(dim=0)
                ),
                contributor_harmonics.amin(dim=0),
            )

            opacity_shape = contributor_opacities.shape[1:]
            bounded_opacity = contributor_opacities.reshape(len(weights), -1).clamp(
                0.0, 1.0 - 1e-6
            )
            merged_opacity = (
                1.0
                - torch.exp(
                    torch.einsum(
                        "n,nk->k", weights, torch.log1p(-bounded_opacity)
                    )
                )
            ).reshape(opacity_shape)

            means[probe_index] = merged_mean
            covs[probe_index] = merged_covariance
            harmonics[probe_index] = merged_harmonics
            opacities[probe_index] = merged_opacity


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
          L1 — depth uniformity check → weighted moment matching (space+feat+depth)
          Full — no modification

        Non-probe pixels in any early-stopped tile have their opacity set to 0
        after moment-matching, reducing the effective Gaussian count for that
        tile to at most K(T).

        Returns:
            modified_mask: bool tensor, True for pixels modified (L0/L1)
            stats:         per-level statistics including effective_gaussians
        """
        gaussian_count = gaussians_full.means.shape[1]
        device = gaussians_full.means.device
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

                    def probe_indices(slot: int) -> List[int]:
                        return [
                            flat_index(
                                view,
                                (tile_y + local_y) * self.W + tile_x + local_x,
                                slot,
                            )
                            for local_y, local_x in self.probe_positions
                        ]

                    def non_probe_map(slot: int) -> Dict[Tuple[int, int], int]:
                        return {
                            (local_y, local_x): flat_index(
                                view,
                                (tile_y + local_y) * self.W + tile_x + local_x,
                                slot,
                            )
                            for local_y, local_x in self.non_probe_positions
                        }

                    first_probes = probe_indices(0)
                    feature_key = (view, th, tw)
                    feature_variance = (
                        tile_variances.get(
                            feature_key, tile_variances.get((th, tw), float('inf'))
                        )
                        if tile_variances is not None
                        else float('inf')
                    )

                    selected_level = None
                    if (
                        feature_variance < self.feature_var_threshold
                        and self.probe_cross_check(gaussians_full, first_probes)
                    ):
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
                    ):
                        probe_similarity = self.compute_tile_similarity(
                            gaussians_full, first_probes, include_position=False
                        )
                        if (
                            probe_similarity >= 0.90
                            and self.probe_cross_check(
                                gaussians_full, first_probes
                            )
                        ):
                            selected_level = 'L1'

                    if selected_level is not None:
                        for slot in range(self.primitives_per_pixel):
                            probes = probe_indices(slot)
                            non_probes = non_probe_map(slot)
                            if self.materialization == "representative":
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
                                    feature_variance=feature_variance,
                                )
                            else:
                                self._dense_interpolate_non_probes(
                                    gaussians_full, probes, non_probes
                                )
                            for index in non_probes.values():
                                if self.materialization == "representative":
                                    gaussians_full.opacities[0, index] *= 0.0
                                modified_mask[index] = True
                            if self.materialization == "representative":
                                total_zeroed += len(non_probes)

                        pixel_count = len(self.non_probe_positions)
                        if selected_level == 'L0':
                            self.stats['level0_tiles'] += 1
                            self.stats['level0_pixels'] += pixel_count
                        else:
                            self.stats['level1_tiles'] += 1
                            self.stats['level1_pixels'] += pixel_count
                        self.stats['pixels_original'] += len(self.probe_positions)
                        continue

                    self.stats['full_tiles'] += 1
                    self.stats['pixels_original'] += tile_size * tile_size

        total_tiles = max(1, self.stats['total_tiles_processed'])
        self.stats['total_modified_pixels'] = (
            self.stats['level0_pixels'] + self.stats['level1_pixels']
        )
        self.stats['zeroed_gaussians'] = total_zeroed
        self.stats['effective_gaussians'] = max(0, gaussian_count - total_zeroed)

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
) -> Tuple['torch.Tensor', Dict, List]:
    """
    Apply progressive SAES v4 (Dataflow-aligned, L0+L1) to Gaussians.

    Two-level progressive early-stopping:
      - L0: feature variance check → weighted moment matching (space+feat)
      - L1: depth uniformity check → weighted moment matching (space+feat+depth)
      - Full: no modification

    Non-probe opacities are zeroed in early-stopped tiles, reducing the
    effective Gaussian count to at most K(T) per tile.

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
