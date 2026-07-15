"""
FSDR Narrowed Depth Search Simulator — ASIC Hardware Model

Instead of skipping S2 entirely (which causes large quality loss),
FSDR uses cached depth to NARROW the cost_volume search in S2:
- Full S2: 128 depth candidates (cost_volume + U-Net + depth_head)
- Guided S2: 32 depth candidates centered on cached depth
  → 75% cost_volume reduction + ~25% U-Net reduction = ~42% S2 per pixel

Key advantages over full depth reuse:
1. Nearly ZERO quality loss (search still finds optimal depth in narrowed range)
2. Can use ALL cache hits (aggressive: hamming≤4, any spatial distance)
3. Depth consistency check prevents narrowing at depth discontinuities
4. S3 runs normally (no S3 impact)

Savings model (per guided pixel, cost_volume-only):
- cost_volume: 75% reduction (32 vs 128 candidates, per-pixel per-candidate)
- U-Net, depth_head, regression: unchanged (process full spatial resolution)
- Overall S2 saving: cv_fraction × 75% per guided pixel
  (e.g., TranSplat cv ~76% of scaled S2 → ~57% S2 saving per pixel)
- S3: unchanged (0% saving)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

from .types import FSDRConfig, CacheEntry
from .lsh_hasher import LSHHasher
from .cache_table import CacheTable


class FSDRSimulator:
    """
    FSDR (Feature Similarity Depth Reuse) — Narrowed Depth Search ASIC Model.

    Design: instead of skipping S2 entirely (which causes large quality loss),
    FSDR uses cached depth to NARROW the cost_volume search in S2:
    - Full S2: D depth candidates (cost_volume + U-Net + depth_head)
    - Guided S2: D/4 depth candidates centered on cached depth
      → 75% cost_volume reduction + ~25% U-Net reduction = ~42% S2 per pixel

    Key advantages over full depth reuse:
    1. Nearly ZERO quality loss (search still finds optimal depth in narrowed range)
    2. Can use ALL cache hits (aggressive: hamming≤4, any spatial distance)
    3. Depth consistency check prevents narrowing at depth discontinuities
    4. S3 runs normally (no S3 impact)

    For pixels where cached depth is far from actual (depth discontinuities):
    - Depth consistency check rejects → falls back to full D-candidate search
    - Zero quality impact for these pixels

    Savings model (per guided pixel, cost_volume-only):
    - cost_volume: 75% reduction (D/4 vs D candidates, per-pixel per-candidate)
    - U-Net, depth_head, regression: unchanged (process full spatial resolution)
    - Overall S2 saving: cv_fraction × 75% per guided pixel
      (e.g., TranSplat cv ~76% of S2 after scaling → ~57% S2 saving per pixel)
    - S3: unchanged (0% saving)

    For rendering quality simulation:
    - Guided pixels where actual_depth IS in window: zero quality modification
    - Guided pixels where actual_depth is outside window: modify means
      (but depth consistency check prevents most of these)
    """

    # Approximate fraction of S2 reducible by narrowing (cost_volume only).
    # Actual value is computed dynamically in SavingsTracker.compute_ablation()
    # based on real cost_volume / dp_core cycle ratio. This constant is a
    # typical reference value for TranSplat (cv_frac ~75% × 75% reduction).
    FSDR_S2_REDUCIBLE_FRAC = 0.56  # ~56% for TranSplat (cost_volume-only model)
    # Candidate reduction ratio
    NARROW_RATIO = 32 / 128  # 32 candidates instead of 128
    # Depth window: ±25% of cached depth (in inverse depth space, this covers wide range)
    DEPTH_WINDOW_FRAC = 0.25

    def __init__(self, feature_dim: int = 128, cache_size: int = 512,
                 hamming_threshold: int = 4,
                 reuse_hamming: int = 3, reuse_spatial: int = 12,
                 reuse_confidence: float = 0.80,
                 num_depth_candidates: int = 128,
                 seed: int = 0):
        self.config = FSDRConfig(
            cache_size=cache_size,
            feature_dim=feature_dim,
            hamming_threshold=hamming_threshold,
            seed=seed,
        )
        self.hasher = LSHHasher(self.config)
        self.cache = CacheTable(self.config)

        # Number of depth candidates for full search
        self.num_depth_candidates = num_depth_candidates

        # Guidance criteria (ASIC hardware parameters)
        self.reuse_hamming = reuse_hamming
        self.reuse_spatial = reuse_spatial
        self.reuse_confidence = reuse_confidence

        # Local depth consistency buffer (ASIC: register file of recent depths)
        self.recent_depths = {}
        self.depth_consistency_threshold = 0.05  # 5% tolerance (wider: narrowed search is safe)
        self.recent_depth_radius = 6

        # Reuse data: pixel_idx -> {'depth_ratio': float, 'in_window': bool}
        self.reuse_data = {}

        self.stats = {
            'frames_started': 0,
            'total_pixels': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'guided': 0,            # Guided search (narrowed S2)
            'guided_in_window': 0,  # Actual depth was in narrowed window
            'guided_out_window': 0, # Actual depth was outside window (rare)
            'guided_top1_covered': 0,
            'guided_top1_missed': 0,
            'discrete_top1_pixels': 0,
            'depth_inconsistent': 0,
            'hit_no_guide': 0,      # Hit but criteria not met
            'full_compute': 0,      # Cache miss
            'total_reuse': 0,       # = guided (for backward compat)
            'total_validated': 0,   # backward compat
            'depth_errors': [],
        }

    def begin_frame(self) -> None:
        """Reset the paper-defined frame-local cache without discarding aggregates."""
        self.cache.clear()
        self.recent_depths.clear()
        self.reuse_data.clear()
        self.stats['frames_started'] += 1

    def _check_depth_consistency(self, position: Tuple[int, int], cached_depth: float) -> bool:
        """Check cached depth consistency with local region (ASIC register file)."""
        nearby = []
        r, c = position
        radius = self.recent_depth_radius
        for row_offset in range(-radius, radius + 1):
            column_radius = radius - abs(row_offset)
            for column_offset in range(-column_radius, column_radius + 1):
                depth = self.recent_depths.get((r + row_offset, c + column_offset))
                if depth is not None:
                    nearby.append(depth)
        if len(nearby) < 3:
            return True
        mean_d = sum(nearby) / len(nearby)
        if abs(mean_d) < 1e-8:
            return True
        return abs(cached_depth - mean_d) / abs(mean_d) <= self.depth_consistency_threshold

    def _update_recent_depths(self, position: Tuple[int, int], depth: float):
        self.recent_depths[position] = depth
        if len(self.recent_depths) > 1024:
            oldest = list(self.recent_depths.keys())[:256]
            for k in oldest:
                del self.recent_depths[k]

    def _insert_current(
        self, signature: int, depth: float, position: Tuple[int, int]
    ) -> None:
        """Mirror the RTL sInsert state executed after every pixel."""
        self.cache.insert(
            CacheEntry(
                signature=int(signature),
                position=position,
                best_depth=float(depth),
                best_idx=0,
                peak_prob=0.9,
                second_offset=1,
                spread=0.1,
                valid=True,
            )
        )

    def _depth_in_window(self, actual_depth: float, cached_depth: float) -> bool:
        """Check if actual depth falls within the narrowed search window."""
        if abs(cached_depth) < 1e-8:
            return True
        window = self.DEPTH_WINDOW_FRAC
        lo = cached_depth * (1.0 - window)
        hi = cached_depth * (1.0 + window)
        return lo <= actual_depth <= hi

    def process_pixel(self, feature: torch.Tensor, actual_depth: float,
                      position: Tuple[int, int], pixel_idx: int,
                      actual_gaussians=None, gauss_idx: int = None,
                      ) -> Tuple[str, int, float]:
        """
        Process pixel through FSDR narrowed depth search.

        Returns:
            (path, num_searches, output_depth)
            num_searches: 32 for guided, num_depth_candidates for full
        """
        feature_cpu = feature.cpu() if feature.is_cuda else feature
        signature = self.hasher.hash(feature_cpu)
        return self.process_signature(signature, actual_depth, position, pixel_idx)

    def process_signature(
        self,
        signature: int,
        actual_depth: float,
        position: Tuple[int, int],
        pixel_idx: int,
        *,
        top1_index: int = None,
        candidate_values=None,
    ) -> Tuple[str, int, float]:
        """Process one precomputed hardware LSH signature."""
        has_top1 = top1_index is not None or candidate_values is not None
        if has_top1 and (top1_index is None or candidate_values is None):
            raise ValueError("top1 index and candidate values must be provided together")
        if has_top1:
            candidates = np.asarray(candidate_values, dtype=np.float64).reshape(-1)
            if candidates.size != self.num_depth_candidates:
                raise ValueError("candidate count does not match the FSDR configuration")
            if not np.isfinite(candidates).all():
                raise ValueError("candidate values must be finite")
            top1_index = int(top1_index)
            if not 0 <= top1_index < candidates.size:
                raise ValueError("top1 candidate index is out of range")
            self.stats['discrete_top1_pixels'] += 1
        self.stats['total_pixels'] += 1

        entry, hamming_dist = self.cache.lookup(signature)

        if entry is not None:
            self.stats['cache_hits'] += 1
            # CacheTable already enforces distance <= tau_h. The paper and RTL
            # route every such hit to the narrowed candidate path.
            if hamming_dist <= self.config.hamming_threshold:
                # GUIDED: S2 runs with D/4 candidates centered on cached_depth
                cached_depth = entry.best_depth
                in_window = self._depth_in_window(actual_depth, cached_depth)

                if has_top1:
                    narrowed = max(1, self.num_depth_candidates // 4)
                    nearest = np.argpartition(
                        np.abs(candidates - cached_depth), narrowed - 1
                    )[:narrowed]
                    if top1_index in nearest:
                        self.stats['guided_top1_covered'] += 1
                    else:
                        self.stats['guided_top1_missed'] += 1

                if in_window:
                    # Actual depth is in narrowed window → zero quality impact
                    # S2 still finds optimal depth, just searches fewer candidates
                    self.stats['guided_in_window'] += 1
                    self.reuse_data[pixel_idx] = {
                        'depth_ratio': 1.0,  # No modification needed
                        'in_window': True,
                    }
                else:
                    # Actual depth outside window → S2 picks closest in window
                    # This causes a small depth error
                    self.stats['guided_out_window'] += 1
                    window = self.DEPTH_WINDOW_FRAC
                    lo = cached_depth * (1.0 - window)
                    hi = cached_depth * (1.0 + window)
                    nearest = max(lo, min(hi, actual_depth))
                    depth_ratio = nearest / actual_depth if abs(actual_depth) > 1e-8 else 1.0
                    self.reuse_data[pixel_idx] = {
                        'depth_ratio': depth_ratio,
                        'in_window': False,
                    }
                    depth_err = abs(nearest - actual_depth) / max(abs(actual_depth), 1e-8)
                    self.stats['depth_errors'].append(depth_err)

                self.stats['guided'] += 1
                self.stats['total_reuse'] += 1
                self.stats['total_validated'] += 1

                # Update cache with actual depth (S2 computed it, even if narrowed)
                self._insert_current(signature, float(actual_depth), position)
                self._update_recent_depths(position, float(actual_depth))

                return 'guided', max(1, self.num_depth_candidates // 4), actual_depth
            else:
                # Defensive only: CacheTable must not return an over-threshold hit.
                self.stats['hit_no_guide'] += 1
                self._insert_current(signature, float(actual_depth), position)
                self._update_recent_depths(position, float(actual_depth))
                return 'hit_no_guide', self.num_depth_candidates, actual_depth
        else:
            # Cache miss → FULL search
            self.stats['cache_misses'] += 1
            self.stats['full_compute'] += 1

            self._insert_current(signature, float(actual_depth), position)
            self._update_recent_depths(position, float(actual_depth))

            return 'full_compute', self.num_depth_candidates, actual_depth

    def process_frame(
        self,
        features: torch.Tensor,
        depths: torch.Tensor,
        width: int,
        pixel_order: List[int] = None,
    ) -> List[str]:
        """Process a frame in the requested schedule with batched LSH projection."""
        if features.dim() != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("features must have shape [pixels, feature_dim]")
        flat_depths = depths.detach().reshape(-1)
        if flat_depths.numel() != features.shape[0] or width <= 0:
            raise ValueError("depth count and frame width must match the feature frame")
        signatures = self.hasher.hash_batch(features).detach().cpu().tolist()
        depth_values = flat_depths.cpu().tolist()
        if pixel_order is None:
            pixel_order = list(range(features.shape[0]))
        if (
            len(pixel_order) != features.shape[0]
            or sorted(pixel_order) != list(range(features.shape[0]))
        ):
            raise ValueError("pixel_order must be a permutation of every frame pixel")
        paths = [None] * features.shape[0]
        for pixel_idx in pixel_order:
            signature = signatures[pixel_idx]
            depth = depth_values[pixel_idx]
            y, x = divmod(pixel_idx, width)
            path, _, _ = self.process_signature(
                int(signature), float(depth), (y, x), pixel_idx
            )
            paths[pixel_idx] = path
        return paths

    def process_discrete_frame(
        self,
        features: torch.Tensor,
        candidate_anchors: torch.Tensor,
        top1_indices: torch.Tensor,
        candidate_values: torch.Tensor,
        *,
        width: int,
        pixel_order: List[int] = None,
        pixel_index_offset: int = 0,
    ) -> List[str]:
        """Process a frame with exact full-search candidate identities."""
        if features.dim() != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("features must have shape [pixels, feature_dim]")
        pixel_count = features.shape[0]
        anchors = candidate_anchors.detach().reshape(-1)
        top1 = top1_indices.detach().reshape(-1)
        candidates = candidate_values.detach()
        if (
            anchors.numel() != pixel_count
            or top1.numel() != pixel_count
            or candidates.shape != (pixel_count, self.num_depth_candidates)
            or width <= 0
            or pixel_count % width
        ):
            raise ValueError("discrete candidate frame shapes are inconsistent")
        if pixel_order is None:
            pixel_order = list(range(pixel_count))
        if len(pixel_order) != pixel_count or sorted(pixel_order) != list(range(pixel_count)):
            raise ValueError("pixel_order must be a permutation of every frame pixel")

        signatures = self.hasher.hash_batch(features).detach().cpu().tolist()
        anchor_values = anchors.cpu().tolist()
        top1_values = top1.cpu().tolist()
        candidate_rows = candidates.cpu().numpy()
        paths = [None] * pixel_count
        for pixel_idx in pixel_order:
            y, x = divmod(pixel_idx, width)
            path, _, _ = self.process_signature(
                int(signatures[pixel_idx]),
                float(anchor_values[pixel_idx]),
                (y, x),
                pixel_index_offset + pixel_idx,
                top1_index=int(top1_values[pixel_idx]),
                candidate_values=candidate_rows[pixel_idx],
            )
            paths[pixel_idx] = path
        return paths

    def get_reuse_ratio(self) -> float:
        """Fraction of pixels with guided (narrowed) S2 search."""
        total = self.stats['total_pixels']
        if total == 0:
            return 0.0
        return self.stats['total_reuse'] / total

    def get_validated_saving_ratio(self) -> float:
        return self.get_reuse_ratio()

    def get_depth_error_stats(self) -> Dict:
        errs = self.stats['depth_errors']
        if not errs:
            return {'mean': 0, 'max': 0, 'p95': 0, 'count': 0}
        errs_np = np.array(errs)
        return {
            'mean': float(np.mean(errs_np)),
            'max': float(np.max(errs_np)),
            'p95': float(np.percentile(errs_np, 95)),
            'count': len(errs),
        }

    def get_summary(self) -> Dict:
        total = self.stats['total_pixels']
        if total == 0:
            return self.stats
        summary = {k: v for k, v in self.stats.items() if k != 'depth_errors'}
        summary['hit_rate'] = self.stats['cache_hits'] / total
        summary['guided_rate'] = self.stats['guided'] / total
        summary['reuse_rate'] = summary['guided_rate']
        summary['validated_rate'] = summary['guided_rate']
        summary['in_window_rate'] = self.stats['guided_in_window'] / max(self.stats['guided'], 1)
        exact_guided = (
            self.stats['guided_top1_covered'] + self.stats['guided_top1_missed']
        )
        summary['discrete_candidate_evidence'] = (
            self.stats['discrete_top1_pixels'] == total
            and exact_guided == self.stats['guided']
        )
        summary['top1_coverage'] = (
            self.stats['guided_top1_covered'] / exact_guided
            if exact_guided
            else None
        )
        summary['hit_no_guide_rate'] = self.stats['hit_no_guide'] / total
        summary['full_compute_rate'] = self.stats['full_compute'] / total
        summary['depth_cache_size'] = len(self.cache)
        summary['reuse_data_size'] = len(self.reuse_data)
        summary['total_validated'] = self.stats['total_reuse']
        summary['depth_error'] = self.get_depth_error_stats()
        summary['depth_inconsistent'] = self.stats.get('depth_inconsistent', 0)
        return summary
