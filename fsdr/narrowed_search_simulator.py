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
                 num_depth_candidates: int = 128):
        self.config = FSDRConfig(
            cache_size=cache_size,
            feature_dim=feature_dim,
            hamming_threshold=hamming_threshold,
        )
        self.hasher = LSHHasher(self.config)
        self.cache = CacheTable(self.config)

        # Number of depth candidates for full search
        self.num_depth_candidates = num_depth_candidates

        # Guidance criteria (ASIC hardware parameters)
        self.reuse_hamming = reuse_hamming
        self.reuse_spatial = reuse_spatial
        self.reuse_confidence = reuse_confidence

        # Depth cache: entry_sig_key -> cached depth value
        self.depth_cache = {}

        # Local depth consistency buffer (ASIC: register file of recent depths)
        self.recent_depths = {}
        self.depth_consistency_threshold = 0.05  # 5% tolerance (wider: narrowed search is safe)
        self.recent_depth_radius = 6

        # Reuse data: pixel_idx -> {'depth_ratio': float, 'in_window': bool}
        self.reuse_data = {}

        self.stats = {
            'total_pixels': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'guided': 0,            # Guided search (narrowed S2)
            'guided_in_window': 0,  # Actual depth was in narrowed window
            'guided_out_window': 0, # Actual depth was outside window (rare)
            'depth_inconsistent': 0,
            'hit_no_guide': 0,      # Hit but criteria not met
            'full_compute': 0,      # Cache miss
            'total_reuse': 0,       # = guided (for backward compat)
            'total_validated': 0,   # backward compat
            'depth_errors': [],
        }

    def _entry_sig_key(self, entry):
        return tuple(entry.signature.tolist()) if hasattr(entry.signature, 'tolist') else str(entry.signature)

    def _check_depth_consistency(self, position: Tuple[int, int], cached_depth: float) -> bool:
        """Check cached depth consistency with local region (ASIC register file)."""
        nearby = []
        r, c = position
        for (pr, pc), d in self.recent_depths.items():
            if abs(pr - r) + abs(pc - c) <= self.recent_depth_radius:
                nearby.append(d)
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
        self.stats['total_pixels'] += 1

        feature_cpu = feature.cpu() if feature.is_cuda else feature
        signature = self.hasher.hash(feature_cpu)
        sig_key = tuple(signature.tolist()) if hasattr(signature, 'tolist') else str(signature)
        entry, hamming_dist = self.cache.lookup(signature)

        if entry is not None:
            self.stats['cache_hits'] += 1
            entry_sig_key = self._entry_sig_key(entry)

            # GUIDANCE DECISION: can we narrow the search?
            can_guide = (
                hamming_dist <= self.reuse_hamming and
                entry.peak_prob > self.reuse_confidence and
                entry_sig_key in self.depth_cache
            )

            if can_guide:
                cached_depth = self.depth_cache[entry_sig_key]
                if not self._check_depth_consistency(position, cached_depth):
                    can_guide = False
                    self.stats['depth_inconsistent'] += 1

            if can_guide:
                # GUIDED: S2 runs with D/4 candidates centered on cached_depth
                cached_depth = self.depth_cache[entry_sig_key]
                in_window = self._depth_in_window(actual_depth, cached_depth)

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
                self.cache.update(entry, float(actual_depth))
                self.depth_cache[entry_sig_key] = float(actual_depth)
                self._update_recent_depths(position, float(actual_depth))

                return 'guided', 32, actual_depth  # 32 candidates searched
            else:
                # Hit but can't guide
                self.stats['hit_no_guide'] += 1
                self.cache.update(entry, float(actual_depth))
                self.depth_cache[entry_sig_key] = float(actual_depth)
                self._update_recent_depths(position, float(actual_depth))
                return 'hit_no_guide', self.num_depth_candidates, actual_depth
        else:
            # Cache miss → FULL search
            self.stats['cache_misses'] += 1
            self.stats['full_compute'] += 1

            new_entry = CacheEntry(
                signature=signature,
                position=position,
                best_depth=float(actual_depth),
                best_idx=0,
                peak_prob=0.9,
                second_offset=1,
                spread=0.1,
                valid=True,
            )
            self.cache.insert(new_entry)
            self.depth_cache[sig_key] = float(actual_depth)
            self._update_recent_depths(position, float(actual_depth))

            return 'full_compute', self.num_depth_candidates, actual_depth

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
        summary['hit_no_guide_rate'] = self.stats['hit_no_guide'] / total
        summary['full_compute_rate'] = self.stats['full_compute'] / total
        summary['depth_cache_size'] = len(self.depth_cache)
        summary['reuse_data_size'] = len(self.reuse_data)
        summary['total_validated'] = self.stats['total_reuse']
        summary['depth_error'] = self.get_depth_error_stats()
        summary['depth_inconsistent'] = self.stats.get('depth_inconsistent', 0)
        return summary
