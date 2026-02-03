"""
FSDR Profiler

Performance metrics collection for FSDR processing.
"""
import time
from typing import Dict, Optional
from dataclasses import dataclass, field

from .types import FSDRResult, FSDRProfilingResult


class FSDRProfiler:
    """
    Performance profiler for FSDR processing.
    
    Tracks:
        - Cache hit/miss statistics
        - Processing path distribution
        - Memory access counts
        - Timing per path
    
    Hardware Mapping:
        - Counters: 32-bit registers per metric
        - Updated on each pixel (single cycle increment)
        - Total: ~200 LUTs for counters, 64B SRAM
    
    Example:
        profiler = FSDRProfiler(num_depths=32)
        
        # In processing loop:
        profiler.start_pixel()
        # ... process pixel ...
        profiler.record(result)
        
        # After processing:
        stats = profiler.get_results()
        print(f"Hit rate: {stats.hit_rate:.2%}")
        print(f"Memory reduction: {stats.memory_reduction:.2%}")
    """
    
    def __init__(self, num_depths: int = 32):
        """
        Initialize profiler.
        
        Args:
            num_depths: Number of depth candidates (for baseline calculation)
        """
        self.num_depths = num_depths
        self._results = FSDRProfilingResult()
        self._pixel_start_time: Optional[int] = None
    
    def start_pixel(self):
        """Mark start of pixel processing."""
        self._pixel_start_time = time.perf_counter_ns()
    
    def record(self, result: FSDRResult):
        """
        Record result of pixel processing.
        
        Args:
            result: FSDR processing result
        """
        self._results.total_pixels += 1
        
        # Cache hit/miss
        if result.cache_hit:
            self._results.cache_hits += 1
        else:
            self._results.cache_misses += 1
        
        # Path counts
        if result.source in self._results.path_counts:
            self._results.path_counts[result.source] += 1
        
        # Memory access
        self._results.total_searches += result.num_searches
        self._results.baseline_searches += self.num_depths
        
        # Timing
        if self._pixel_start_time is not None:
            elapsed = time.perf_counter_ns() - self._pixel_start_time
            self._results.total_time_ns += elapsed
            
            if result.source in self._results.path_time_ns:
                self._results.path_time_ns[result.source] += elapsed
            
            self._pixel_start_time = None
    
    def record_simple(
        self,
        cache_hit: bool,
        source: str,
        num_searches: int,
    ):
        """
        Simplified recording without timing.
        
        Args:
            cache_hit: Whether cache was hit
            source: Processing path
            num_searches: Number of depth searches
        """
        self._results.total_pixels += 1
        
        if cache_hit:
            self._results.cache_hits += 1
        else:
            self._results.cache_misses += 1
        
        if source in self._results.path_counts:
            self._results.path_counts[source] += 1
        
        self._results.total_searches += num_searches
        self._results.baseline_searches += self.num_depths
    
    def get_results(self) -> FSDRProfilingResult:
        """
        Get aggregated profiling results.
        
        Returns:
            FSDRProfilingResult with all statistics
        """
        return self._results
    
    def reset(self):
        """Reset all counters."""
        self._results = FSDRProfilingResult()
        self._pixel_start_time = None
    
    def get_summary_dict(self) -> Dict:
        """
        Get summary as dictionary.
        
        Returns:
            Dictionary with key metrics
        """
        r = self._results
        return {
            'total_pixels': r.total_pixels,
            'hit_rate': r.hit_rate,
            'memory_reduction': r.memory_reduction,
            'direct_reuse_rate': r.direct_reuse_rate,
            'path_distribution': {
                k: v / r.total_pixels if r.total_pixels > 0 else 0
                for k, v in r.path_counts.items()
            },
            'avg_time_per_pixel_us': r.avg_time_per_pixel_ns / 1000,
        }
    
    def print_summary(self):
        """Print formatted summary to console."""
        r = self._results
        
        print("=" * 50)
        print("FSDR Profiling Summary")
        print("=" * 50)
        print(f"Total pixels processed: {r.total_pixels}")
        print(f"Cache hit rate: {r.hit_rate:.2%}")
        print(f"Memory reduction: {r.memory_reduction:.2%}")
        print()
        print("Path Distribution:")
        for path, count in r.path_counts.items():
            pct = count / r.total_pixels * 100 if r.total_pixels > 0 else 0
            print(f"  {path}: {count} ({pct:.1f}%)")
        print()
        print(f"Total searches: {r.total_searches}")
        print(f"Baseline searches: {r.baseline_searches}")
        print(f"Searches saved: {r.baseline_searches - r.total_searches}")
        print()
        if r.total_time_ns > 0:
            print(f"Total time: {r.total_time_ns / 1e6:.2f} ms")
            print(f"Avg time per pixel: {r.avg_time_per_pixel_ns / 1000:.2f} µs")
        print("=" * 50)


class TileProfiler:
    """
    Profiler for tile-level statistics.
    
    Tracks per-tile metrics for analysis of spatial patterns.
    """
    
    def __init__(self):
        """Initialize tile profiler."""
        self.tiles = []
        self._current_tile = None
    
    def start_tile(self, tile_idx: int, position: tuple):
        """
        Start profiling a new tile.
        
        Args:
            tile_idx: Tile index
            position: (row, col) tile position
        """
        self._current_tile = {
            'tile_idx': tile_idx,
            'position': position,
            'pixels': 0,
            'hits': 0,
            'searches': 0,
            'start_time': time.perf_counter_ns(),
        }
    
    def record_pixel(self, cache_hit: bool, num_searches: int):
        """Record pixel result for current tile."""
        if self._current_tile is not None:
            self._current_tile['pixels'] += 1
            if cache_hit:
                self._current_tile['hits'] += 1
            self._current_tile['searches'] += num_searches
    
    def end_tile(self):
        """End current tile and save statistics."""
        if self._current_tile is not None:
            self._current_tile['end_time'] = time.perf_counter_ns()
            self._current_tile['duration_ns'] = (
                self._current_tile['end_time'] - self._current_tile['start_time']
            )
            self.tiles.append(self._current_tile)
            self._current_tile = None
    
    def get_tile_stats(self) -> list:
        """Get list of tile statistics."""
        results = []
        for tile in self.tiles:
            pixels = tile['pixels']
            results.append({
                'tile_idx': tile['tile_idx'],
                'position': tile['position'],
                'hit_rate': tile['hits'] / pixels if pixels > 0 else 0,
                'avg_searches': tile['searches'] / pixels if pixels > 0 else 0,
                'duration_us': tile['duration_ns'] / 1000,
            })
        return results
