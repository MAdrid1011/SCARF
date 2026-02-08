"""
SAES: Scene-Adaptive Early-Stopping — Multi-Level v3 Simulator

Progressive Adaptive Early-Stopping for SCARF with 3-level tile classification
for aggressive S2+S3 savings:

  Level 0 (Feature Pre-Filter): Tiles with low feature variance after S1.
    Process 4 probes through S2+S3, interpolate appearance for 12 others.
    Saves 75% of S2+S3 per tile. Near-zero quality impact.

  Level 1 (Depth-Based): Tiles with moderate feature variance but uniform depth.
    Process 4 probes through S2, check depth std. If uniform, interpolate
    S3 appearance for remaining 12 pixels. Saves 75% of S2+S3 per tile.

  Level 2 (Gaussian Similarity): Remaining tiles checked via probe Gaussian
    similarity. If similar, interpolate from 4 probes. Saves 75% of S3.

  Full: Tiles that fail all checks. No SAES savings (FSGR may still apply).

ASIC-implementable quality validation:
  - Probe cross-check: leave-one-out prediction on 4 probes
  - No oracle/ground-truth data needed
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


class ProgressiveSAES:
    """
    Progressive Adaptive Early-Stopping for SCARF (v3: Multi-Level).

    3-level tile classification for aggressive S2+S3 savings:

      Level 0 (Feature Pre-Filter): Tiles with low feature variance after S1.
        Process 4 probes through S2+S3, interpolate appearance for 12 others.
        Saves 75% of S2+S3 per tile. Near-zero quality impact.

      Level 1 (Depth-Based): Tiles with moderate feature variance but uniform depth.
        Process 4 probes through S2, check depth std. If uniform, interpolate
        S3 appearance for remaining 12 pixels. Saves 75% of S2+S3 per tile.

      Level 2 (Gaussian Similarity): Remaining tiles checked via probe Gaussian
        similarity. If similar, interpolate from 4 probes. Saves 75% of S3.

      Full: Tiles that fail all checks. No SAES savings (FSGR may still apply).

    Processing order for 4x4 tile:
        1. Feature variance check (from S1 features)
        2. Depth uniformity check (from S2 probe depths)
        3. Gaussian similarity check (from S3 probe Gaussians)
    """

    # Probe pixel positions within a 4x4 tile (CORNER layout for true bilinear)
    # Corner probes ensure all 12 non-probe pixels lie INSIDE the convex hull,
    # enabling genuine bilinear blending rather than nearest-probe assignment.
    # With center probes (1,1),(1,2),(2,1),(2,2), clamped bilinear degenerates
    # to nearest-probe for ALL 12 outer pixels — no actual blending occurs.
    PROBE_POSITIONS = [(0, 0), (0, 3), (3, 0), (3, 3)]

    # Non-probe positions (the 12 pixels to interpolate via bilinear blending)
    NON_PROBE_POSITIONS = [
                (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
                (3, 1), (3, 2),
    ]

    def __init__(self, H: int, W: int, initial_tile_size: int = 4,
                 threshold: float = None,
                 feature_var_threshold: float = None,
                 depth_std_threshold: float = None,
                 cross_check_threshold: float = 0.02):
        self.H = H
        self.W = W
        self.initial_tile_size = initial_tile_size
        self.gauss_threshold = threshold if threshold is not None else 0.985
        self.feature_var_threshold = feature_var_threshold if feature_var_threshold is not None else 0.015
        self.depth_std_threshold = depth_std_threshold if depth_std_threshold is not None else 0.008
        # Probe cross-check: ASIC-implementable quality validation
        # For each probe, predict it from the other 3; if error > threshold, fall back
        self.cross_check_threshold = cross_check_threshold

        # Statistics
        self.stats = {
            'total_tiles_processed': 0,
            'level0_tiles': 0,        # Feature-uniform (4 probes, interpolate 12)
            'level1_tiles': 0,        # Depth-uniform (4 probes, interpolate 12)
            'level2_tiles': 0,        # Gaussian-similar (4 probes, interpolate 12)
            'full_tiles': 0,          # No optimization
            'level0_pixels': 0,       # 12 interpolated pixels per L0 tile (v2)
            'level1_pixels': 0,       # 12 interpolated pixels per L1 tile
            'level2_pixels': 0,       # 12 interpolated pixels per L2 tile
            'pixels_original': 0,     # Pixels kept as-is (probes + full tiles)
            'total_modified_pixels': 0,
        }

        # Precompute bilinear weights for all non-probe positions
        # Corner probes at (0,0),(0,3),(3,0),(3,3): normalized coords ty=py/3, tx=px/3
        # All non-probe pixels are INSIDE the convex hull → true bilinear blending
        self._interp_weights = {}
        for (py, px) in self.NON_PROBE_POSITIONS:
            ty = py / 3.0   # normalized [0, 1] between corners
            tx = px / 3.0   # normalized [0, 1] between corners
            w00 = (1.0 - ty) * (1.0 - tx)  # probe (0,0)
            w01 = (1.0 - ty) * tx           # probe (0,3)
            w10 = ty * (1.0 - tx)           # probe (3,0)
            w11 = ty * tx                   # probe (3,3)
            self._interp_weights[(py, px)] = (w00, w01, w10, w11)

    @staticmethod
    def classify_tiles_by_features(features, h: int, w: int, tile_size: int,
                                   threshold: float = 0.02) -> Tuple[Dict[Tuple[int, int], float], 'torch.Tensor']:
        """
        Classify tiles by feature variance from S1 feature maps.

        Args:
            features: S1 features [B, V, C, H_feat, W_feat] or [V, C, H_feat, W_feat]
            h, w: output image resolution
            tile_size: tile size (4)
            threshold: variance threshold for uniform classification

        Returns:
            tile_variances: dict (th, tw) -> feature_variance (float)
            feat_norm: normalized upsampled features [C, H, W] for FSGI
        """
        # Handle different feature shapes
        if features.dim() == 5:
            feat = features[0]  # [V, C, H_feat, W_feat]
        elif features.dim() == 4:
            feat = features     # [V, C, H_feat, W_feat]
        else:
            return {}, None

        # Average over views
        feat = feat.mean(dim=0)  # [C, H_feat, W_feat]

        # Upsample to output resolution
        feat_up = F.interpolate(
            feat.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        )[0]  # [C, H, W]

        # Normalize features for variance computation AND for FSGI similarity
        feat_norm = feat_up / (feat_up.norm(dim=0, keepdim=True) + 1e-8)

        tiles_h = h // tile_size
        tiles_w = w // tile_size
        tile_variances = {}

        for th in range(tiles_h):
            for tw in range(tiles_w):
                y0, x0 = th * tile_size, tw * tile_size
                # Extract tile features: [C, tile_size, tile_size]
                tile_feat = feat_norm[:, y0:y0+tile_size, x0:x0+tile_size]
                # Reshape to [C, tile_size*tile_size], compute per-channel std
                tile_flat = tile_feat.reshape(tile_feat.shape[0], -1)  # [C, 16]
                channel_std = tile_flat.std(dim=1)  # [C]
                # Mean std across channels
                var_score = channel_std.mean().item()
                tile_variances[(th, tw)] = var_score

        return tile_variances, feat_norm

    @staticmethod
    def check_depth_uniformity(depths, th: int, tw: int, tile_size: int,
                               h: int, w: int, threshold: float = 0.03) -> bool:
        """
        Check if probe pixel depths within a tile are uniform.

        Uses corner probe positions for maximum spatial coverage.
        If all 4 corners have similar depth, the interior is likely uniform.

        Args:
            depths: depth tensor [B, V, H, W] or [B, V, N, 1, 1]
            th, tw: tile coordinates
            tile_size: tile size
            h, w: image resolution
            threshold: relative depth std threshold

        Returns:
            True if tile has uniform depth
        """
        # Corner probes — same as PROBE_POSITIONS for consistency
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

        if len(probe_depths) < 4:
            return False

        d_mean = sum(probe_depths) / len(probe_depths)
        if abs(d_mean) < 1e-8:
            return True  # Near-zero depth → uniform
        d_std = (sum((d - d_mean) ** 2 for d in probe_depths) / len(probe_depths)) ** 0.5
        relative_std = d_std / (abs(d_mean) + 1e-8)

        return relative_std < threshold

    @staticmethod
    def compute_tile_similarity(gaussians_full, probe_indices: List[int],
                                include_position: bool = True) -> float:
        """
        Compute similarity among probe Gaussians for a tile.

        Weighted metric:
          With position (L2):  cov(0.30) + SH(0.30) + opacity(0.15) + position(0.25)
          Without position (L1): cov(0.40) + SH(0.40) + opacity(0.20)
            (L1 already checks depth uniformity; position term penalizes corner probes)
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
                cov_sim = F.cosine_similarity(cov1.unsqueeze(0), cov2.unsqueeze(0)).item()
                cov_sim = (cov_sim + 1) / 2

                sh1 = harmo[idx_i].flatten()
                sh2 = harmo[idx_j].flatten()
                sh_sim = F.cosine_similarity(sh1.unsqueeze(0), sh2.unsqueeze(0)).item()
                sh_sim = (sh_sim + 1) / 2

                op1 = opacs[idx_i].item() if opacs[idx_i].dim() == 0 else opacs[idx_i].squeeze().item()
                op2 = opacs[idx_j].item() if opacs[idx_j].dim() == 0 else opacs[idx_j].squeeze().item()
                opacity_sim = 1.0 - min(abs(op1 - op2), 1.0)

                if include_position:
                    pair_sim = 0.3 * cov_sim + 0.3 * sh_sim + 0.15 * opacity_sim + 0.25 * pos_sim
                else:
                    # Appearance-only metric for L1 (depth already validated)
                    pair_sim = 0.4 * cov_sim + 0.4 * sh_sim + 0.2 * opacity_sim
                total_sim += pair_sim
                count += 1

        return total_sim / count if count > 0 else 1.0

    def probe_cross_check(self, gaussians_full, probe_indices: List[int]) -> bool:
        """
        ASIC-implementable quality validation: leave-one-out cross-check.

        For each of the 4 probes, predict it from the other 3 using the same
        bilinear interpolation weights. If ANY probe's prediction error exceeds
        the threshold, the tile is NOT suitable for interpolation.

        This uses ONLY the 4 computed probe values — no oracle/original data needed.
        An ASIC implements this as 4 extra weighted-sum operations per tile.

        Returns:
            True if cross-check passes (safe to interpolate), False otherwise.
        """
        if len(probe_indices) != 4:
            return False

        harmo = gaussians_full.harmonics[0]
        covs = gaussians_full.covariances[0]

        max_err = 0.0
        for leave_out in range(4):
            # Predict leave_out from the other 3 (simple average)
            others = [probe_indices[j] for j in range(4) if j != leave_out]

            pred_h = (harmo[others[0]] + harmo[others[1]] + harmo[others[2]]) / 3.0
            actual_h = harmo[probe_indices[leave_out]]

            h_flat_pred = pred_h.flatten()
            h_flat_actual = actual_h.flatten()

            if h_flat_actual.norm() > 1e-8 and h_flat_pred.norm() > 1e-8:
                h_sim = F.cosine_similarity(
                    h_flat_pred.unsqueeze(0), h_flat_actual.unsqueeze(0)).item()
                h_err = 1.0 - max(h_sim, 0.0)
                max_err = max(max_err, h_err)

            # Also check covariance
            pred_c = (covs[others[0]] + covs[others[1]] + covs[others[2]]) / 3.0
            actual_c = covs[probe_indices[leave_out]]
            c_flat_pred = pred_c.flatten()
            c_flat_actual = actual_c.flatten()
            if c_flat_actual.norm() > 1e-8 and c_flat_pred.norm() > 1e-8:
                c_sim = F.cosine_similarity(
                    c_flat_pred.unsqueeze(0), c_flat_actual.unsqueeze(0)).item()
                c_err = 1.0 - max(c_sim, 0.0)
                max_err = max(max_err, c_err)

        return max_err <= self.cross_check_threshold

    def replicate_tile(self, gaussians_full, tile_y: int, tile_x: int,
                       probe_idx: int, other_indices: List[int]):
        """
        Level 0: Replicate single probe pixel's appearance to all other pixels.
        Positions are kept original. Modifies gaussians_full IN PLACE.
        """
        covs = gaussians_full.covariances
        harmo = gaussians_full.harmonics
        opacs = gaussians_full.opacities

        p_cov = covs[0, probe_idx]     # [...]
        p_harm = harmo[0, probe_idx]   # [...]
        p_opac = opacs[0, probe_idx]   # [...]

        for idx in other_indices:
            covs[0, idx] = p_cov * 1.02   # small safety factor
            harmo[0, idx] = p_harm
            opacs[0, idx] = p_opac

    def interpolate_tile(self, gaussians_full, tile_y: int, tile_x: int,
                         probe_indices: List[int],
                         non_probe_pixel_map: Dict[Tuple[int, int], int],
                         level: str = 'bilinear'):
        """
        Bilinear interpolation of appearance from 4 corner-probe Gaussians.
        
        With corner probes at (0,0),(0,3),(3,0),(3,3), all 12 non-probe pixels
        lie INSIDE the convex hull, enabling genuine bilinear blending.
        
        Quality improvements (ASIC-implementable, ~6 extra cycles/tile):
          1. Adaptive covariance safety: scales with actual probe variance
          2. SH magnitude preservation: interpolated SH preserves average brightness
          3. Opacity clamped to probe range: prevents transparency artifacts
        
        Positions (means) are kept original. Modifies gaussians_full IN PLACE.
        """
        covs = gaussians_full.covariances
        harmo = gaussians_full.harmonics
        opacs = gaussians_full.opacities

        p_covs = covs[0, probe_indices]   # [4, ...]
        p_harmo = harmo[0, probe_indices]  # [4, ...]
        p_opacs = opacs[0, probe_indices]  # [4, ...]

        # Adaptive covariance safety factor: higher when probes differ more
        # ASIC: one variance computation per tile (not per pixel)
        cov_var = p_covs.var(dim=0).mean().item()
        cov_mean = p_covs.abs().mean().item() + 1e-8
        cov_safety = 1.0 + min(0.03, 0.5 * cov_var / cov_mean)

        # SH DC magnitude for preservation
        if p_harmo.dim() >= 2:
            dc_norms = p_harmo.flatten(1).norm(dim=1)  # [4]
            avg_dc_norm = dc_norms.mean().item() + 1e-8
        else:
            avg_dc_norm = None

        # Opacity range for clamping
        opac_min = p_opacs.min()
        opac_max = p_opacs.max()

        for (local_y, local_x) in self.NON_PROBE_POSITIONS:
            if (local_y, local_x) not in non_probe_pixel_map:
                continue

            flat_idx = non_probe_pixel_map[(local_y, local_x)]
            w00, w01, w10, w11 = self._interp_weights[(local_y, local_x)]

            new_cov = (w00 * p_covs[0] + w01 * p_covs[1] +
                       w10 * p_covs[2] + w11 * p_covs[3]) * cov_safety
            new_harmo = (w00 * p_harmo[0] + w01 * p_harmo[1] +
                         w10 * p_harmo[2] + w11 * p_harmo[3])
            new_opac = (w00 * p_opacs[0] + w01 * p_opacs[1] +
                        w10 * p_opacs[2] + w11 * p_opacs[3])

            # SH magnitude preservation: rescale to preserve average brightness
            if avg_dc_norm is not None and new_harmo.dim() >= 1:
                new_norm = new_harmo.flatten().norm().item() + 1e-8
                scale = avg_dc_norm / new_norm
                # Only apply gentle correction (avoid amplifying noise)
                scale = max(0.9, min(1.1, scale))
                new_harmo = new_harmo * scale

            # Opacity: clamp to probe range to prevent artifacts
            new_opac = torch.clamp(new_opac, opac_min, opac_max)

            covs[0, flat_idx] = new_cov
            harmo[0, flat_idx] = new_harmo
            opacs[0, flat_idx] = new_opac

    def process_all_tiles(self, gaussians_full, gpp: int = 1,
                          tile_variances: Dict = None,
                          depths=None,
                          feat_norm=None) -> Tuple[torch.Tensor, Dict]:
        """
        Multi-level tile processing (SAES v3).

        For each tile:
          1. Check feature variance (Level 0) → replicate from 1 probe
          2. Check depth uniformity (Level 1) → interpolate from 4 probes
          3. Check Gaussian similarity (Level 2) → interpolate from 4 probes
          4. Full processing → no modification

        Args:
            gaussians_full: Gaussians to modify in-place
            gpp: Gaussians per pixel
            tile_variances: dict from classify_tiles_by_features()
            depths: depth tensor for Level 1 check
            feat_norm: normalized features [C, H, W] — currently unused,
                       reserved for future FSGI (Feature-Similarity Guided
                       Interpolation) where interpolation weights are derived
                       from feature-space distances instead of spatial bilinear.

        Returns:
            modified_mask: Boolean mask - True for pixels modified (L0/L1/L2)
            stats: Per-level processing statistics
        """
        N = gaussians_full.means.shape[1]
        device = gaussians_full.means.device

        modified_mask = torch.zeros(N, dtype=torch.bool, device=device)
        # Track which level each pixel was assigned to: 0,1,2 or -1 (original)
        pixel_level = torch.full((N,), -1, dtype=torch.int8, device=device)

        # Reset stats
        for k in self.stats:
            self.stats[k] = 0

        tiles_h = self.H // self.initial_tile_size
        tiles_w = self.W // self.initial_tile_size
        tile_size = self.initial_tile_size

        for th in range(tiles_h):
            for tw in range(tiles_w):
                self.stats['total_tiles_processed'] += 1

                tile_y = th * tile_size
                tile_x = tw * tile_size

                # === Level 0: Feature Pre-Filter (4-probe interpolation) ===
                # L0 v2: uses 4-probe bilinear interpolation instead of 1-probe replication
                # Quality is much better → can use more aggressive feature_var_threshold
                if tile_variances is not None and (th, tw) in tile_variances:
                    feat_var = tile_variances[(th, tw)]
                    if feat_var < self.feature_var_threshold:
                        # Feature-uniform tile: use 4-probe interpolation
                        probe_indices = []
                        for (ly, lx) in self.PROBE_POSITIONS:
                            gy, gx = tile_y + ly, tile_x + lx
                            pix_idx = gy * self.W + gx
                            if pix_idx < N:
                                probe_indices.append(pix_idx)

                        if len(probe_indices) == 4:
                            # Cross-check: verify probes are interpolation-consistent
                            if not self.probe_cross_check(gaussians_full, probe_indices):
                                pass  # Cross-check failed, try L1/L2
                            else:
                                # Build non-probe map and interpolate (same as L1/L2)
                                non_probe_map = {}
                                for (ly, lx) in self.NON_PROBE_POSITIONS:
                                    gy, gx = tile_y + ly, tile_x + lx
                                    pix_idx = gy * self.W + gx
                                    if pix_idx < N:
                                        non_probe_map[(ly, lx)] = pix_idx

                                self.interpolate_tile(gaussians_full, tile_y, tile_x,
                                                      probe_indices, non_probe_map,
                                                      level='bilinear')

                                for idx in non_probe_map.values():
                                    modified_mask[idx] = True
                                    pixel_level[idx] = 0

                                self.stats['level0_tiles'] += 1
                                self.stats['level0_pixels'] += len(non_probe_map)
                                self.stats['pixels_original'] += len(probe_indices)
                                continue

                # === Level 1: Depth-Based Early Stopping ===
                if depths is not None:
                    depth_uniform = self.check_depth_uniformity(
                        depths, th, tw, tile_size, self.H, self.W,
                        self.depth_std_threshold)

                    if depth_uniform:
                        # Get 4 probe indices
                        probe_indices = []
                        for (ly, lx) in self.PROBE_POSITIONS:
                            gy, gx = tile_y + ly, tile_x + lx
                            pix_idx = gy * self.W + gx
                            if pix_idx < N:
                                probe_indices.append(pix_idx)

                        if len(probe_indices) == 4:
                            # Secondary check: verify probe Gaussians are appearance-similar
                            # (depth uniform doesn't guarantee appearance uniform)
                            # Use appearance-only metric (no position) since corner probes
                            # are inherently far apart — position already checked via depth
                            probe_sim = self.compute_tile_similarity(
                                gaussians_full, probe_indices,
                                include_position=False)
                            if probe_sim < 0.90:
                                # Depth is uniform but appearance diverges - skip L1
                                pass
                            elif not self.probe_cross_check(gaussians_full, probe_indices):
                                # ASIC cross-check: leave-one-out prediction failed
                                pass
                            else:
                                # Build non-probe map and interpolate
                                non_probe_map = {}
                                for (ly, lx) in self.NON_PROBE_POSITIONS:
                                    gy, gx = tile_y + ly, tile_x + lx
                                    pix_idx = gy * self.W + gx
                                    if pix_idx < N:
                                        non_probe_map[(ly, lx)] = pix_idx

                                self.interpolate_tile(gaussians_full, tile_y, tile_x,
                                                      probe_indices, non_probe_map,
                                                      level='bilinear')

                                for idx in non_probe_map.values():
                                    modified_mask[idx] = True
                                    pixel_level[idx] = 1

                                self.stats['level1_tiles'] += 1
                                self.stats['level1_pixels'] += len(non_probe_map)
                                self.stats['pixels_original'] += len(probe_indices)
                                continue

                # === Level 2: Gaussian Similarity Check ===
                probe_indices = []
                for (ly, lx) in self.PROBE_POSITIONS:
                    gy, gx = tile_y + ly, tile_x + lx
                    pix_idx = gy * self.W + gx
                    if pix_idx < N:
                        probe_indices.append(pix_idx)

                if len(probe_indices) == 4:
                    similarity = self.compute_tile_similarity(gaussians_full, probe_indices)

                    if similarity >= self.gauss_threshold:
                        # ASIC cross-check: leave-one-out prediction validation
                        if not self.probe_cross_check(gaussians_full, probe_indices):
                            pass  # Cross-check failed, fall through to Full
                        else:
                            non_probe_map = {}
                            for (ly, lx) in self.NON_PROBE_POSITIONS:
                                gy, gx = tile_y + ly, tile_x + lx
                                pix_idx = gy * self.W + gx
                                if pix_idx < N:
                                    non_probe_map[(ly, lx)] = pix_idx

                            self.interpolate_tile(gaussians_full, tile_y, tile_x,
                                                  probe_indices, non_probe_map,
                                                  level='bilinear')

                            for idx in non_probe_map.values():
                                modified_mask[idx] = True
                                pixel_level[idx] = 2

                            self.stats['level2_tiles'] += 1
                            self.stats['level2_pixels'] += len(non_probe_map)
                            self.stats['pixels_original'] += len(probe_indices)
                            continue

                # === Full processing: no modification ===
                self.stats['full_tiles'] += 1
                self.stats['pixels_original'] += tile_size * tile_size

        # Final stats
        total_tiles = max(1, self.stats['total_tiles_processed'])
        self.stats['total_modified_pixels'] = (self.stats['level0_pixels'] +
                                                self.stats['level1_pixels'] +
                                                self.stats['level2_pixels'])
        self.stats['level0_ratio'] = self.stats['level0_tiles'] / total_tiles
        self.stats['level1_ratio'] = self.stats['level1_tiles'] / total_tiles
        self.stats['level2_ratio'] = self.stats['level2_tiles'] / total_tiles
        self.stats['full_ratio'] = self.stats['full_tiles'] / total_tiles
        self.stats['early_stop_ratio'] = 1.0 - self.stats['full_ratio']
        self.stats['modification_ratio'] = (
            self.stats['total_modified_pixels'] / N if N > 0 else 0.0
        )
        # Backward-compatible keys
        self.stats['early_stop_phase1'] = (self.stats['level0_tiles'] +
                                            self.stats['level1_tiles'] +
                                            self.stats['level2_tiles'])
        self.stats['early_stop_phase2'] = 0
        self.stats['full_processed'] = self.stats['full_tiles']
        self.stats['pixels_interpolated'] = self.stats['total_modified_pixels']
        self.stats['interpolation_ratio'] = self.stats['modification_ratio']

        return modified_mask, self.stats


def apply_progressive_saes(
    gaussians_full,
    H: int, W: int,
    tile_size: int = 4,
    gpp: int = 1,
    threshold: float = None,
    feature_var_threshold: float = None,
    depth_std_threshold: float = None,
    features=None,
    depths=None,
    cross_check_threshold: float = 0.015,
) -> Tuple[torch.Tensor, Dict, List]:
    """
    Apply progressive SAES v3 (multi-level) to Gaussians.

    v3: 3-level tile optimization (all levels use 4-probe interpolation).
      Level 0: Feature pre-filter (4 probes, interpolate 12)
      Level 1: Depth uniformity (4 probes, interpolate 12)
      Level 2: Gaussian similarity (4 probes, interpolate 12)

    Args:
        features: S1 features for Level 0 classification [B, V, C, H, W]
        depths: S2 depths for Level 1 check [B, V, H, W]
        cross_check_threshold: ASIC probe cross-check threshold

    Returns:
        modified_mask: Boolean mask (True = pixel was modified)
        stats: Per-level processing statistics
        continue_pixels: List of (y, x, pixel_idx) for unmodified pixels
    """
    # Classify tiles by feature variance (Level 0) + get normalized features for FSGI
    tile_variances = None
    feat_norm = None
    if features is not None and hasattr(features, 'shape'):
        tile_variances, feat_norm = ProgressiveSAES.classify_tiles_by_features(
            features, H, W, tile_size,
            threshold=feature_var_threshold if feature_var_threshold is not None else 0.012)

    saes = ProgressiveSAES(H, W, tile_size,
                           threshold=threshold,
                           feature_var_threshold=feature_var_threshold,
                           depth_std_threshold=depth_std_threshold,
                           cross_check_threshold=cross_check_threshold)
    modified_mask, stats = saes.process_all_tiles(
        gaussians_full, gpp,
        tile_variances=tile_variances,
        depths=depths,
        feat_norm=feat_norm)

    # Collect unmodified pixels (for FSGR)
    continue_pixels = []
    for idx in range(modified_mask.shape[0]):
        if not modified_mask[idx]:
            pixel_idx = idx // gpp
            y = pixel_idx // H if H > 0 else 0
            x = pixel_idx % W
            if y < H and x < W:
                continue_pixels.append((y, x, pixel_idx))

    return modified_mask, stats, continue_pixels
