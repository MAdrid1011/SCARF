"""
FSDR Processor

Main processing engine for Feature-Similarity Depth Reuse.
"""
import time
import torch
from typing import Callable, Tuple, Optional

from .types import FSDRConfig, CacheEntry, FSDRResult
from .lsh_hasher import LSHHasher
from .cache_table import CacheTable
from .depth_corrector import DepthCorrector
from .light_verifier import LightVerifier
from .profiler import FSDRProfiler


class FSDRProcessor:
    """
    Main FSDR processing engine.
    
    Orchestrates:
        - LSH signature generation
        - Cache lookup
        - Strategy decision
        - Depth correction/verification
        - Cache update/insertion
        - Performance profiling
    
    Processing Flow:
        1. Generate LSH signature from feature
        2. Lookup cache by signature similarity
        3. If hit:
           a. Decide strategy (direct_reuse/interpolation/light_verify)
           b. Execute strategy to get depth
           c. Update cache entry (EMA)
        4. If miss:
           a. Execute full depth search
           b. Extract statistics from distribution
           c. Insert new cache entry
        5. Return result with profiling info
    
    Hardware Mapping:
        - Instantiates all sub-components
        - FSM for control flow
        - Total: ~1,250 LUTs, 22 DSPs, 1.1KB SRAM, 0.5KB ROM
    
    Example:
        config = FSDRConfig()
        depth_candidates = torch.linspace(0.5, 10.0, 32)
        
        processor = FSDRProcessor(config, depth_candidates)
        
        # Process single pixel
        result = processor.process_pixel(
            feature, position, cost_fn, prob_fn
        )
        
        # Process tile
        results = processor.process_tile(
            features, positions, cost_fn, prob_fn
        )
        
        # Get profiling
        stats = processor.get_profiling()
    """
    
    def __init__(
        self,
        config: FSDRConfig,
        depth_candidates: torch.Tensor,
        enable_profiling: bool = True,
    ):
        """
        Initialize FSDR processor.
        
        Args:
            config: FSDR configuration
            depth_candidates: [D] depth candidate values
            enable_profiling: Whether to collect profiling data
        """
        self.config = config
        self.depth_candidates = depth_candidates
        self.num_depths = len(depth_candidates)
        self.enable_profiling = enable_profiling
        
        # Initialize components
        self.lsh_hasher = LSHHasher(config)
        self.cache_table = CacheTable(config)
        self.depth_corrector = DepthCorrector(config, depth_candidates)
        self.light_verifier = LightVerifier(config)
        
        # Profiler
        self.profiler = FSDRProfiler(num_depths=self.num_depths)
    
    def process_pixel(
        self,
        feature: torch.Tensor,
        position: Tuple[int, int],
        cost_fn: Callable[[torch.Tensor, int], float],
        prob_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> FSDRResult:
        """
        Process single pixel with FSDR.
        
        Args:
            feature: [C] reference feature vector
            position: (u, v) pixel coordinates
            cost_fn: fn(feature, depth_idx) -> cost (for light verify)
            prob_fn: fn(feature, depth_candidates) -> probabilities [D] (for full search)
        
        Returns:
            FSDRResult with depth and processing info
        
        Hardware:
            - Variable latency depending on path
            - Direct reuse: ~3 cycles
            - Interpolation: ~5 cycles
            - Light verify: ~60-140 cycles
            - Full search: ~472 cycles
        """
        start_time = time.perf_counter_ns()
        timing = {}
        
        # Phase 1: Signature generation
        t0 = time.perf_counter_ns()
        signature = self.lsh_hasher.hash(feature)
        timing['signature'] = time.perf_counter_ns() - t0
        
        # Phase 2: Cache lookup
        t0 = time.perf_counter_ns()
        entry, hamming_dist = self.cache_table.lookup(signature)
        timing['lookup'] = time.perf_counter_ns() - t0
        
        # Phase 3: Hit/Miss processing
        if entry is not None:
            # Cache hit
            result = self._handle_cache_hit(
                entry, hamming_dist, feature, cost_fn, timing
            )
        else:
            # Cache miss
            result = self._handle_cache_miss(
                feature, position, signature, prob_fn, timing
            )
        
        # Record total time
        result.timing_ns['total'] = time.perf_counter_ns() - start_time
        
        # Profile
        if self.enable_profiling:
            self.profiler.record(result)
        
        return result
    
    def _handle_cache_hit(
        self,
        entry: CacheEntry,
        hamming_dist: int,
        feature: torch.Tensor,
        cost_fn: Callable,
        timing: dict,
    ) -> FSDRResult:
        """Handle cache hit processing."""
        t0 = time.perf_counter_ns()
        
        # Decide strategy
        strategy = self.depth_corrector.decide_strategy(entry, hamming_dist)
        
        if strategy == 'direct_reuse':
            depth = self.depth_corrector.direct_reuse(entry)
            num_searches = 0
            
        elif strategy == 'interpolation':
            depth = self.depth_corrector.interpolate(entry, hamming_dist)
            num_searches = 0
            
        else:  # light_verify
            depth, num_searches = self.light_verifier.verify(
                entry, feature, cost_fn, self.depth_candidates
            )
            # Update cache with verified depth
            self.cache_table.update(entry, depth)
        
        timing['correction'] = time.perf_counter_ns() - t0
        
        return FSDRResult(
            depth=depth,
            source=strategy,
            cache_hit=True,
            hamming_distance=hamming_dist,
            num_searches=num_searches,
            timing_ns=timing,
        )
    
    def _handle_cache_miss(
        self,
        feature: torch.Tensor,
        position: Tuple[int, int],
        signature: int,
        prob_fn: Callable,
        timing: dict,
    ) -> FSDRResult:
        """Handle cache miss processing."""
        t0 = time.perf_counter_ns()
        
        # Full depth search
        probs = prob_fn(feature, self.depth_candidates)
        
        # Compute expected depth
        depth = float((probs * self.depth_candidates).sum())
        
        timing['full_search'] = time.perf_counter_ns() - t0
        
        # Extract statistics for cache entry
        t0 = time.perf_counter_ns()
        new_entry = self._create_cache_entry(signature, position, probs, depth)
        self.cache_table.insert(new_entry)
        timing['insert'] = time.perf_counter_ns() - t0
        
        return FSDRResult(
            depth=depth,
            source='full_search',
            cache_hit=False,
            hamming_distance=None,
            num_searches=self.num_depths,
            timing_ns=timing,
        )
    
    def _create_cache_entry(
        self,
        signature: int,
        position: Tuple[int, int],
        probs: torch.Tensor,
        depth: float,
    ) -> CacheEntry:
        """Create cache entry from probability distribution."""
        # Best depth index
        best_idx = int(torch.argmax(probs).item())
        peak_prob = float(probs[best_idx])
        
        # Second-best index
        probs_copy = probs.clone()
        probs_copy[best_idx] = -float('inf')
        second_idx = int(torch.argmax(probs_copy).item())
        second_offset = second_idx - best_idx
        
        # Clamp offset to valid range
        second_offset = max(-16, min(15, second_offset))
        
        # Spread (normalized standard deviation)
        mean_depth = float((probs * self.depth_candidates).sum())
        variance = float((probs * (self.depth_candidates - mean_depth) ** 2).sum())
        std_dev = variance ** 0.5
        depth_range = float(self.depth_candidates[-1] - self.depth_candidates[0])
        spread = min(1.0, std_dev / depth_range) if depth_range > 0 else 0.0
        
        return CacheEntry(
            signature=signature,
            position=position,
            best_depth=depth,
            best_idx=best_idx,
            peak_prob=peak_prob,
            second_offset=second_offset,
            spread=spread,
            valid=True,
        )
    
    def process_batch(
        self,
        features: torch.Tensor,
        positions: list,
        cost_fn: Callable,
        prob_fn: Callable,
    ) -> list:
        """
        Process batch of pixels.
        
        Args:
            features: [N, C] batch of features
            positions: List of (u, v) positions
            cost_fn: Cost function for light verify
            prob_fn: Probability function for full search
        
        Returns:
            List of FSDRResult
        """
        results = []
        for i in range(len(features)):
            result = self.process_pixel(
                features[i], positions[i], cost_fn, prob_fn
            )
            results.append(result)
        return results
    
    def get_profiling(self) -> dict:
        """Get profiling summary."""
        return self.profiler.get_summary_dict()
    
    def reset_profiling(self):
        """Reset profiling counters."""
        self.profiler.reset()
    
    def get_cache_stats(self) -> dict:
        """Get cache table statistics."""
        return self.cache_table.get_stats()
    
    def clear_cache(self):
        """Clear cache table."""
        self.cache_table.clear()
