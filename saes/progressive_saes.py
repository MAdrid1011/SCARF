"""
SAES: Scene-Adaptive Early-Stopping — Multi-Level v4 Simulator

Progressive Adaptive Early-Stopping for SCARF with 2-level tile classification
aligned with the SCARF Dataflow specification (L0 + L1 only):

  Level 0 (Feature Pre-Filter): Tiles with low feature variance after S1.
    Process K(T) probes through S2+S3; keep probe Gaussian parameters unchanged;
    expand probe covariances to cover assigned spatial territory (space+feature
    weights; covariance spread computed from 2-D pixel offsets only — no non-probe
    Stage-3 data accessed).  Set non-probe opacity=0.
    Effective Gaussian count reduced to K(T).

  Level 1 (Depth-Based): Tiles that pass L0 but have uniform probe depths.
    Same spatial-coverage expansion with additional depth-derived weighting.
    Set non-probe opacity=0.

  Full: Tiles that fail both checks — no SAES savings.

NOTE: non-probe Stage-3 outputs (covariances, harmonics, opacities, 3-D means)
are NEVER read by this module; they are not computed by real hardware for
early-stopped tiles.  Only probe Stage-3 outputs and S1 features are used.

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
    Progressive Adaptive Early-Stopping for SCARF (v4: Dataflow-aligned).

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

        feat = feat.mean(dim=0)  # [C, H_feat, W_feat]
        feat_up = F.interpolate(
            feat.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        )[0]  # [C, H, W]
        feat_norm = feat_up / (feat_up.norm(dim=0, keepdim=True) + 1e-8)

        tiles_h = h // tile_size
        tiles_w = w // tile_size
        tile_variances: Dict[Tuple[int, int], float] = {}

        for th in range(tiles_h):
            for tw in range(tiles_w):
                y0, x0 = th * tile_size, tw * tile_size
                tile_feat = feat_norm[:, y0:y0 + tile_size, x0:x0 + tile_size]
                tile_flat = tile_feat.reshape(tile_feat.shape[0], -1)
                channel_std = tile_flat.std(dim=1)
                tile_variances[(th, tw)] = channel_std.mean().item()

        return tile_variances, feat_norm

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
                d = depths[0, 0, pixel_idx, 0, 0].item()
            elif depths.dim() == 4:
                d = depths[0, 0, gy, gx].item()
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
    ):
        """
        Hardware-honest probe covariance expansion.

        Does NOT read non-probe Stage-3 outputs (covariances, harmonics,
        opacities, 3-D means).  Real hardware never computes these for
        early-stopped tiles.  Only probe Stage-3 outputs and S1 features
        are used.

        For each non-probe pixel i, compute soft assignment weights using
        2-D pixel positions and S1 features (both always available):

            r_{i→k}^(L0) ∝ exp(-‖x_i-x_k‖²/σ_s²) · exp(-‖f_i-f_k‖²/σ_f²)
            r_{i→k}^(L1) ∝ r_{i→k}^(L0) · exp(-|d̄_T - d_k|/β_d)

        Covariance expansion: for each probe k, accumulate the 3-D spread
        from its assigned non-probe territory using 2-D pixel offsets scaled
        by probe depth (probe depth is from Stage 2, always available):

            Δ3D_i ≈ (Δx_pix · d_k/W,  Δy_pix · d_k/H,  0)
            Σ_k' = Σ_k + (Σ_i r_{i→k} · outer(Δ3D_i, Δ3D_i)) / Σ_i r_{i→k}

        Probe means, harmonics, and opacities are NOT modified.
        After this call the caller zeroes non-probe opacities.
        """
        T = self.initial_tile_size
        σ_s_sq = (T / 2.0) ** 2          # spatial bandwidth² (pixels²)
        σ_f_sq = 0.09                     # feature bandwidth² (normalised)
        β_d    = 0.1                      # depth bandwidth for L1

        device = gaussians_full.means.device
        K = len(probe_indices)

        # Only probe Stage-3 outputs are accessed (probes have run S2+S3).
        # Non-probe covariances / harmonics / opacities / 3-D means are NEVER
        # read — real hardware never computes them for early-stopped tiles.
        means = gaussians_full.means[0]       # probe means (read-only except spread)
        covs  = gaussians_full.covariances[0] # probe covs  (expanded in-place)
        # harmo and opacs: NOT modified — probe keeps its own Stage-3 values.

        # Probe 2-D pixel coordinates, S1 features, and Stage-2 depths
        probe_gy = [pid // self.W for pid in probe_indices]
        probe_gx = [pid % self.W  for pid in probe_indices]

        probe_feats = []
        probe_d     = []
        for k, pid in enumerate(probe_indices):
            gy, gx = probe_gy[k], probe_gx[k]
            probe_feats.append(
                feat_norm[:, gy, gx]
                if (feat_norm is not None
                    and gy < feat_norm.shape[1]
                    and gx < feat_norm.shape[2])
                else None
            )
            if depths is not None:
                if depths.dim() == 5:
                    probe_d.append(depths[0, 0, pid, 0, 0].item())
                elif depths.dim() == 4:
                    probe_d.append(depths[0, 0, gy, gx].item())
                else:
                    probe_d.append(1.0)
            else:
                probe_d.append(1.0)

        # Accumulate covariance-spread term from 2-D pixel offsets only.
        # For non-probe pixel i assigned to probe k with weight w_{i→k}:
        #   Δpix = (gx_i − gx_k, gy_i − gy_k)
        #   scale = d_p / max(W, H)   (depth × angular_size_per_pixel)
        #   delta_3d ≈ (Δx·scale, Δy·scale, 0)
        #   spread_acc[k] += w_{i→k} · outer(delta_3d, delta_3d)
        # This gives the 3-D covariance contribution from the territory assigned
        # to each probe without reading any non-probe Stage-3 tensor.
        spread_acc = [torch.zeros(3, 3, device=device, dtype=means.dtype)
                      for _ in range(K)]
        w_total    = [0.0 for _ in range(K)]

        img_dim = float(max(self.W, self.H, 1))

        for (local_y, local_x), flat_idx in non_probe_map.items():
            gy = tile_y + local_y
            gx = tile_x + local_x

            fi = (feat_norm[:, gy, gx]
                  if feat_norm is not None
                  and gy < feat_norm.shape[1]
                  and gx < feat_norm.shape[2]
                  else None)

            # Assignment weights r_{i→k} — uses only 2-D positions and S1 features
            raw_w = []
            for k in range(K):
                qy, qx = probe_gy[k], probe_gx[k]
                sp = math.exp(-((gy - qy) ** 2 + (gx - qx) ** 2) / (σ_s_sq + 1e-8))
                fp = probe_feats[k]
                if fi is not None and fp is not None and fi.shape == fp.shape:
                    fd = (fi - fp).norm().item() ** 2
                    ft = math.exp(-fd / (σ_f_sq + 1e-8))
                else:
                    ft = 1.0
                w = sp * ft
                if level == 'L1':
                    dp = probe_d[k]
                    d_est = sum(probe_d) / K
                    w *= math.exp(-abs(d_est - dp) / (β_d + 1e-8))
                raw_w.append(w)

            w_sum = sum(raw_w) + 1e-8
            norm_w = [w / w_sum for w in raw_w]

            # Accumulate 2-D-pixel-based spread (no non-probe S3 data read)
            for k in range(K):
                wk = norm_w[k]
                if wk < 1e-6:
                    continue
                d_p = max(probe_d[k], 1e-4)
                scale = d_p / img_dim
                dx = float(gx - probe_gx[k]) * scale
                dy = float(gy - probe_gy[k]) * scale
                delta = torch.tensor([dx, dy, 0.0],
                                     device=device, dtype=means.dtype)
                spread_acc[k] = spread_acc[k] + wk * torch.outer(delta, delta)
                w_total[k]   += wk

        # Expand probe covariances to cover their assigned spatial territory.
        # Probe means, harmonics, and opacities are NOT modified.
        for k, pidx in enumerate(probe_indices):
            tw = w_total[k]
            if tw < 1e-8:
                continue
            spread = spread_acc[k] / tw
            spread = (spread + spread.mT) / 2          # symmetrise
            spread = spread + torch.eye(3, device=device,
                                        dtype=means.dtype) * 1e-6
            covs[pidx] = covs[pidx] + spread


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
        N = gaussians_full.means.shape[1]
        device = gaussians_full.means.device

        modified_mask = torch.zeros(N, dtype=torch.bool, device=device)
        pixel_level   = torch.full((N,), -1, dtype=torch.int8, device=device)

        # Reset stats
        for k in self.stats:
            self.stats[k] = 0

        tiles_h   = self.H // self.initial_tile_size
        tiles_w   = self.W // self.initial_tile_size
        tile_size = self.initial_tile_size

        total_zeroed = 0

        for th in range(tiles_h):
            for tw in range(tiles_w):
                self.stats['total_tiles_processed'] += 1
                tile_y = th * tile_size
                tile_x = tw * tile_size

                # Helper: build probe flat indices for this tile
                def _get_probe_indices():
                    idxs = []
                    for (ly, lx) in self.probe_positions:
                        gy, gx = tile_y + ly, tile_x + lx
                        pidx = gy * self.W + gx
                        if pidx < N:
                            idxs.append(pidx)
                    return idxs

                # Helper: build non-probe map (local_pos -> flat_idx)
                def _get_non_probe_map():
                    npm = {}
                    for (ly, lx) in self.non_probe_positions:
                        gy, gx = tile_y + ly, tile_x + lx
                        pidx = gy * self.W + gx
                        if pidx < N:
                            npm[(ly, lx)] = pidx
                    return npm

                # Helper: zero non-probe opacity and update stats
                def _zero_non_probe_opacity(non_probe_map):
                    nonlocal total_zeroed
                    for flat_idx in non_probe_map.values():
                        gaussians_full.opacities[0, flat_idx] = (
                            gaussians_full.opacities[0, flat_idx] * 0.0)
                    total_zeroed += len(non_probe_map)

                # ---- Level 0: Feature Pre-Filter ----
                if tile_variances is not None and (th, tw) in tile_variances:
                    feat_var = tile_variances[(th, tw)]
                    if feat_var < self.feature_var_threshold:
                        probe_indices = _get_probe_indices()
                        K = len(probe_indices)
                        if K >= 2:
                            if self.probe_cross_check(gaussians_full, probe_indices):
                                non_probe_map = _get_non_probe_map()
                                # Weighted moment matching (L0: space+feature)
                                self._weighted_moment_match(
                                    gaussians_full, probe_indices, non_probe_map,
                                    feat_norm, depths, 'L0', tile_y, tile_x)
                                # Zero non-probe opacities
                                _zero_non_probe_opacity(non_probe_map)
                                for idx in non_probe_map.values():
                                    modified_mask[idx] = True
                                    pixel_level[idx] = 0
                                self.stats['level0_tiles']  += 1
                                self.stats['level0_pixels'] += len(non_probe_map)
                                self.stats['pixels_original'] += K
                                continue

                # ---- Level 1: Depth-Based ----
                if depths is not None:
                    depth_uniform = self.check_depth_uniformity(
                        depths, th, tw, tile_size, self.H, self.W,
                        self.depth_std_threshold,
                        probe_positions=self.probe_positions)
                    if depth_uniform:
                        probe_indices = _get_probe_indices()
                        K = len(probe_indices)
                        if K >= 2:
                            probe_sim = self.compute_tile_similarity(
                                gaussians_full, probe_indices,
                                include_position=False)
                            # L1: depth already verified uniform; probe_sim gates quality.
                            # 0.90 keeps only tiles where all corner probes are nearly
                            # identical — these are the safest to early-stop with
                            # opacity zeroing (low per-tile quality loss).
                            if probe_sim >= 0.90:
                                if self.probe_cross_check(gaussians_full, probe_indices):
                                    non_probe_map = _get_non_probe_map()
                                    # Weighted moment matching (L1: space+feature+depth)
                                    self._weighted_moment_match(
                                        gaussians_full, probe_indices, non_probe_map,
                                        feat_norm, depths, 'L1', tile_y, tile_x)
                                    _zero_non_probe_opacity(non_probe_map)
                                    for idx in non_probe_map.values():
                                        modified_mask[idx] = True
                                        pixel_level[idx] = 1
                                    self.stats['level1_tiles']  += 1
                                    self.stats['level1_pixels'] += len(non_probe_map)
                                    self.stats['pixels_original'] += K
                                    continue

                # ---- Full processing ----
                self.stats['full_tiles']      += 1
                self.stats['pixels_original'] += tile_size * tile_size

        # Finalize stats
        total_tiles = max(1, self.stats['total_tiles_processed'])
        self.stats['total_modified_pixels'] = (self.stats['level0_pixels'] +
                                               self.stats['level1_pixels'])
        self.stats['zeroed_gaussians']   = total_zeroed
        self.stats['effective_gaussians'] = max(0, N - total_zeroed)

        self.stats['level0_ratio']     = self.stats['level0_tiles'] / total_tiles
        self.stats['level1_ratio']     = self.stats['level1_tiles'] / total_tiles
        self.stats['full_ratio']       = self.stats['full_tiles']   / total_tiles
        self.stats['early_stop_ratio'] = 1.0 - self.stats['full_ratio']
        self.stats['modification_ratio'] = (
            self.stats['total_modified_pixels'] / N if N > 0 else 0.0)

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
        modified_mask:   bool tensor (True = pixel was modified / opacity zeroed)
        stats:           per-level statistics (incl. effective_gaussians)
        continue_pixels: list of (y, x, pixel_idx) for unmodified pixels
    """
    tile_variances = None
    _feat_norm = feat_norm  # use caller-supplied if available

    if features is not None and hasattr(features, 'shape'):
        _tv, _fn = ProgressiveSAES.classify_tiles_by_features(
            features, H, W, tile_size,
            threshold=(feature_var_threshold
                       if feature_var_threshold is not None else 0.012))
        tile_variances = _tv
        if _feat_norm is None:
            _feat_norm = _fn

    saes = ProgressiveSAES(
        H, W, tile_size,
        feature_var_threshold=feature_var_threshold,
        depth_std_threshold=depth_std_threshold,
        cross_check_threshold=cross_check_threshold,
    )
    modified_mask, stats = saes.process_all_tiles(
        gaussians_full, gpp,
        tile_variances=tile_variances,
        depths=depths,
        feat_norm=_feat_norm,
    )

    # Collect unmodified pixels (for FSGR)
    continue_pixels = []
    for idx in range(modified_mask.shape[0]):
        if not modified_mask[idx]:
            pixel_idx = idx // gpp
            y = pixel_idx // W
            x = pixel_idx % W
            if y < H and x < W:
                continue_pixels.append((y, x, pixel_idx))

    return modified_mask, stats, continue_pixels
