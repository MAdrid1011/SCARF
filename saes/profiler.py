"""
SAES Profiler

Tracks performance metrics for tile processing, including:
- Per-tile timing (probe/evaluate/decide phases)
- Path distribution (early-stop/sparse/full counts)
- Computation savings (pixels skipped)
- Overhead analysis
"""
import time
from typing import Dict, List, Tuple
from collections import defaultdict

from .types import TileProcessingResult, SAESProfilingResult


class SAESProfiler:
    """
    Performance metrics collector for SAES tile processing
    
    Tracks timing, path distribution, and computation savings across all tiles.
    Aggregates results into scene-level statistics.
    """
    
    def __init__(self, scene_name: str = "unknown"):
        """
        Initialize profiler
        
        Args:
            scene_name: Scene identifier for reporting
        """
        self.scene_name = scene_name
        self.tile_results: List[TileProcessingResult] = []
        
        # Current tile tracking
        self._current_tile_id = None
        self._current_tile_start_ns = None
        self._current_phase_timings = {}
        self._phase_start_ns = None
        
        # Debug mode
        self.enable_detailed_logging = False
    
    def start_tile(self, tile_id: tuple) -> None:
        """
        Begin timing for a tile
        
        Args:
            tile_id: (row, col) tile coordinates
        """
        self._current_tile_id = tile_id
        self._current_tile_start_ns = time.perf_counter_ns()
        self._current_phase_timings = {}
        
        if self.enable_detailed_logging:
            print(f"[Profiler] Starting tile {tile_id}")
    
    def start_phase(self, phase_name: str) -> None:
        """
        Begin timing for a phase within current tile
        
        Args:
            phase_name: 'probe', 'evaluate', or 'decide'
        """
        self._phase_start_ns = time.perf_counter_ns()
    
    def end_phase(self, phase_name: str) -> None:
        """
        End timing for a phase and record duration
        
        Args:
            phase_name: 'probe', 'evaluate', or 'decide'
        """
        if self._phase_start_ns is None:
            raise RuntimeError(f"Phase '{phase_name}' ended without start")
        
        duration_ns = time.perf_counter_ns() - self._phase_start_ns
        self._current_phase_timings[phase_name] = duration_ns
        self._phase_start_ns = None
        
        if self.enable_detailed_logging:
            print(f"[Profiler]   Phase {phase_name}: {duration_ns / 1e6:.2f} ms")
    
    def record_phase(self, phase_name: str, duration_ns: int) -> None:
        """
        Record phase timing (alternative to start_phase/end_phase)
        
        Args:
            phase_name: 'probe', 'evaluate', or 'decide'
            duration_ns: Phase duration in nanoseconds
        """
        self._current_phase_timings[phase_name] = duration_ns
    
    def finish_tile(self, result: TileProcessingResult) -> None:
        """
        Complete tile and store result
        
        Args:
            result: TileProcessingResult with path, pixels processed, etc.
        """
        self.tile_results.append(result)
        
        if self.enable_detailed_logging:
            print(
                f"[Profiler] Finished tile {result.tile_id}: "
                f"{result.path_taken}, {result.pixels_processed} pixels, "
                f"similarity={result.similarity_score:.3f}"
            )
        
        # Reset current tile state
        self._current_tile_id = None
        self._current_tile_start_ns = None
        self._current_phase_timings = {}
    
    def get_scene_summary(self, tile_size: int = 4) -> SAESProfilingResult:
        """
        Aggregate all tiles into scene-level statistics
        
        Args:
            tile_size: Pixels per tile edge (for computing savings)
        
        Returns:
            SAESProfilingResult with aggregated metrics
        """
        if not self.tile_results:
            raise ValueError("No tile results to aggregate")
        
        # Categorize tiles by path
        early_stop, sparse, full = self._categorize_tiles()
        
        # Compute savings
        total_pixels_saved, saving_ratio = self.compute_savings(tile_size)
        
        # Compute overhead
        overhead_ns = self.compute_overhead()
        
        # Compute net speedup (requires baseline timing estimate)
        net_speedup = self.compute_net_speedup(tile_size, saving_ratio, overhead_ns)
        
        return SAESProfilingResult(
            scene_name=self.scene_name,
            total_tiles=len(self.tile_results),
            early_stop_tiles=early_stop,
            sparse_continue_tiles=sparse,
            full_continue_tiles=full,
            total_pixels_saved=total_pixels_saved,
            computation_saving_ratio=saving_ratio,
            overhead_ns=overhead_ns,
            net_speedup=net_speedup,
            tile_results=self.tile_results,
        )
    
    def _categorize_tiles(self) -> Tuple[int, int, int]:
        """
        Count tiles by path type
        
        Returns:
            (early_stop_count, sparse_continue_count, full_continue_count)
        """
        early_stop = sum(1 for r in self.tile_results if r.path_taken == "early_stop")
        sparse = sum(1 for r in self.tile_results if r.path_taken == "sparse_continue")
        full = sum(1 for r in self.tile_results if r.path_taken == "full_continue")
        
        return early_stop, sparse, full
    
    def compute_savings(self, tile_size: int) -> Tuple[int, float]:
        """
        Compute total pixels saved and saving ratio
        
        Args:
            tile_size: Pixels per tile edge
        
        Returns:
            (total_pixels_saved, computation_saving_ratio)
        
        Calculation:
            For each tile:
            - pixels_saved = tile_size² - pixels_processed
            saving_ratio = sum(pixels_saved) / (total_tiles * tile_size²)
        """
        total_pixels = len(self.tile_results) * (tile_size ** 2)
        total_processed = sum(r.pixels_processed for r in self.tile_results)
        total_saved = total_pixels - total_processed
        
        saving_ratio = total_saved / total_pixels if total_pixels > 0 else 0.0
        
        return total_saved, saving_ratio
    
    def compute_overhead(self) -> int:
        """
        Compute total SAES overhead (evaluate + decide phases)
        
        Returns:
            Total overhead in nanoseconds
        
        Overhead includes:
        - Similarity evaluation time
        - Decision logic time
        
        Does NOT include:
        - Probe processing time (would happen in baseline too)
        - Gaussian merger time (only for early-stop, part of savings)
        """
        overhead_ns = 0
        
        for result in self.tile_results:
            # Add evaluate and decide phase timings
            overhead_ns += result.timing_ns.get("evaluate", 0)
            overhead_ns += result.timing_ns.get("decide", 0)
            
            # For early-stop tiles, also add merger overhead
            if result.path_taken == "early_stop":
                overhead_ns += result.timing_ns.get("merge", 0)
        
        return overhead_ns
    
    def compute_net_speedup(
        self,
        tile_size: int,
        saving_ratio: float,
        overhead_ns: int,
    ) -> float:
        """
        Compute net speedup considering overhead
        
        Args:
            tile_size: Pixels per tile edge
            saving_ratio: Computation saving ratio
            overhead_ns: SAES overhead in nanoseconds
        
        Returns:
            Net speedup factor
        
        Formula:
            Assume baseline processes all pixels at T ns/pixel
            baseline_time = N_pixels * T
            saes_time = N_pixels * (1 - saving_ratio) * T + overhead
            speedup = baseline_time / saes_time
            
        Simplified (assume T = 1 unit):
            speedup = 1 / ((1 - saving_ratio) + overhead_ratio)
            
        For estimation, assume overhead_ratio = overhead_ns / baseline_time
        and baseline_time ≈ total_tiles * tile_size² * T_per_pixel
        """
        # Estimate baseline time (assume 1ms per pixel as rough estimate)
        total_pixels = len(self.tile_results) * (tile_size ** 2)
        estimated_baseline_ns = total_pixels * 1_000_000  # 1ms per pixel
        
        # Compute overhead ratio
        overhead_ratio = overhead_ns / estimated_baseline_ns if estimated_baseline_ns > 0 else 0.0
        
        # Compute net speedup
        # baseline / (baseline * (1 - saving) + overhead)
        # = 1 / ((1 - saving) + overhead_ratio)
        denominator = (1 - saving_ratio) + overhead_ratio
        speedup = 1.0 / denominator if denominator > 0 else 1.0
        
        return speedup
    
    def reset(self) -> None:
        """Reset profiler state for new scene"""
        self.tile_results.clear()
        self._current_tile_id = None
        self._current_tile_start_ns = None
        self._current_phase_timings = {}
        self._phase_start_ns = None
    
    def get_statistics_dict(self) -> Dict:
        """
        Get profiling statistics as dictionary (for JSON serialization)
        
        Returns:
            Dictionary with all statistics
        """
        if not self.tile_results:
            return {}
        
        early, sparse, full = self._categorize_tiles()
        tile_size = 4  # Default
        total_saved, saving_ratio = self.compute_savings(tile_size)
        overhead = self.compute_overhead()
        speedup = self.compute_net_speedup(tile_size, saving_ratio, overhead)
        
        return {
            "scene_name": self.scene_name,
            "total_tiles": len(self.tile_results),
            "path_distribution": {
                "early_stop": early,
                "sparse_continue": sparse,
                "full_continue": full,
            },
            "computation": {
                "pixels_saved": total_saved,
                "saving_ratio": saving_ratio,
            },
            "timing": {
                "overhead_ns": overhead,
                "overhead_ms": overhead / 1e6,
            },
            "performance": {
                "net_speedup": speedup,
            },
        }
