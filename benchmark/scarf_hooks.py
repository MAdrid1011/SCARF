"""
SCARF Hooks

Hooks for integrating SCARF into Transplat encoder.
"""
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fsdr.fsdr_processor import FSDRProcessor
    from fsdr.types import FSDRResult
    from dsu.dsu_processor import DSUProcessor
    from dsu.types import DSUResult
    from ggu.ggu_processor import GGUProcessor
    from ggu.types import GaussianOutput
    from .cycle_counter import CycleCounter


@dataclass
class HookResult:
    """
    Result from SCARF hooks.
    
    Attributes:
        skip_dsu: Whether to skip DSU (cache hit)
        depth: Depth value if available
        confidence: Confidence score
        strategy: Strategy used ('cache_hit', 'full_search')
    """
    skip_dsu: bool = False
    depth: Optional[float] = None
    confidence: float = 0.0
    strategy: str = 'full_search'


class SCARFHooks:
    """
    Hooks for integrating SCARF into Transplat encoder.
    
    Provides pre/post hooks for depth search and Gaussian generation
    to integrate FSDR caching, DSU cycle counting, and GGU cycle counting.
    
    Example:
        hooks = SCARFHooks(
            fsdr_processor=FSDRProcessor(...),
            dsu_processor=DSUProcessor(...),
            ggu_processor=GGUProcessor(...),
            cycle_counter=CycleCounter(),
        )
        
        # In encoder forward:
        result = hooks.pre_depth_search(features, position)
        if not result.skip_dsu:
            depth = dsu.search_depth(...)
            hooks.post_depth_search(depth, features)
        else:
            depth = result.depth
    """
    
    def __init__(
        self,
        fsdr_processor: Optional['FSDRProcessor'] = None,
        dsu_processor: Optional['DSUProcessor'] = None,
        ggu_processor: Optional['GGUProcessor'] = None,
        cycle_counter: Optional['CycleCounter'] = None,
        enable_fsdr: bool = True,
        enable_cycle_counting: bool = True,
    ):
        """
        Initialize SCARF hooks.
        
        Args:
            fsdr_processor: FSDR processor for depth caching
            dsu_processor: DSU processor for depth search
            ggu_processor: GGU processor for Gaussian generation
            cycle_counter: Cycle counter for profiling
            enable_fsdr: Whether to enable FSDR caching
            enable_cycle_counting: Whether to enable cycle counting
        """
        self.fsdr_processor = fsdr_processor
        self.dsu_processor = dsu_processor
        self.ggu_processor = ggu_processor
        self.cycle_counter = cycle_counter
        
        self.enable_fsdr = enable_fsdr and fsdr_processor is not None
        self.enable_cycle_counting = enable_cycle_counting and cycle_counter is not None
        
        # Statistics
        self._stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def pre_depth_search(
        self,
        feature: torch.Tensor,
        position: torch.Tensor,
        prob_fn: Optional[Any] = None,
    ) -> HookResult:
        """
        Hook before depth search - check FSDR cache.
        
        Args:
            feature: Feature vector [C]
            position: Pixel position [2]
            prob_fn: Probability function for FSDR (if needed)
        
        Returns:
            HookResult indicating whether to skip DSU
        """
        self._stats['total_queries'] += 1
        
        if not self.enable_fsdr or self.fsdr_processor is None:
            return HookResult(skip_dsu=False, strategy='full_search')
        
        try:
            # Call FSDR processor
            result = self.fsdr_processor.process_pixel(
                feature=feature,
                position=position,
                prob_fn=prob_fn,
            )
            
            if result.is_cache_hit:
                self._stats['cache_hits'] += 1
                return HookResult(
                    skip_dsu=True,
                    depth=result.depth,
                    confidence=result.confidence,
                    strategy=result.strategy,
                )
            else:
                self._stats['cache_misses'] += 1
                return HookResult(skip_dsu=False, strategy='full_search')
                
        except Exception as e:
            # Fallback to full search on error
            self._stats['cache_misses'] += 1
            return HookResult(skip_dsu=False, strategy='full_search')
    
    def post_depth_search(
        self,
        depth: float,
        feature: torch.Tensor,
        position: torch.Tensor,
        probability: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Hook after depth search - update FSDR cache.
        
        Args:
            depth: Computed depth value
            feature: Feature vector
            position: Pixel position
            probability: Depth probability distribution
        """
        if not self.enable_fsdr or self.fsdr_processor is None:
            return
        
        # Note: Cache update is handled internally by FSDR processor
        # This hook is for any additional post-processing if needed
        pass
    
    def pre_gaussian_gen(
        self,
        depth: float,
        raw_gaussian: torch.Tensor,
        pixel_coord: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Optional['GaussianOutput']:
        """
        Hook before Gaussian generation.
        
        Args:
            depth: Depth value
            raw_gaussian: Raw Gaussian parameters from network
            pixel_coord: Pixel coordinate
            intrinsics: Camera intrinsics
            extrinsics: Camera extrinsics
        
        Returns:
            GaussianOutput if using GGU processor, None otherwise
        """
        if self.ggu_processor is None:
            return None
        
        try:
            return self.ggu_processor.generate_gaussian(
                pixel_coord=pixel_coord,
                depth=depth,
                raw_gaussian=raw_gaussian,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
        except Exception:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get hook statistics.
        
        Returns:
            Dictionary with query counts and hit rate
        """
        total = self._stats['total_queries']
        hits = self._stats['cache_hits']
        
        return {
            'total_queries': total,
            'cache_hits': hits,
            'cache_misses': self._stats['cache_misses'],
            'hit_rate': hits / total if total > 0 else 0.0,
        }
    
    def get_cycle_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get cycle count summary.
        
        Returns:
            Cycle summary if counting enabled, None otherwise
        """
        if not self.enable_cycle_counting or self.cycle_counter is None:
            return None
        
        return self.cycle_counter.to_dict()
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        if self.cycle_counter is not None:
            self.cycle_counter.reset()
