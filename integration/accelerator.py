"""
Accelerator

Complete SCARF accelerator integrating FSDR, DSU, GGU, and SAES.
"""
import torch
from typing import Optional, Tuple, List, Dict

from fsdr import FSDRProcessor, FSDRConfig
from dsu import DSUProcessor, DSUConfig
from ggu import GGUProcessor, GGUConfig
from adapters import create_adapter, BaseAdapter


class Accelerator:
    """
    Complete SCARF accelerator.
    
    Integrates:
    - FSDR: Feature-Similarity Depth Reuse
    - DSU: Depth Search Unit
    - GGU: Gaussian Generation Unit
    - Model adapters for Transplat/MVSplat/DepthSplat
    
    Example:
        accelerator = Accelerator(model_type='transplat')
        
        gaussians, profiling = accelerator.process_scene(
            ref_features=ref_features,
            target_features=target_features,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            near=0.5,
            far=10.0,
        )
    """
    
    def __init__(
        self,
        model_type: str = 'transplat',
        fsdr_config: Optional[FSDRConfig] = None,
        dsu_config: Optional[DSUConfig] = None,
        ggu_config: Optional[GGUConfig] = None,
        enable_fsdr: bool = True,
        enable_profiling: bool = True,
    ):
        """
        Initialize accelerator.
        
        Args:
            model_type: 'transplat', 'mvsplat', or 'depthsplat'
            fsdr_config: FSDR configuration (uses defaults if None)
            dsu_config: DSU configuration (uses defaults if None)
            ggu_config: GGU configuration (uses defaults if None)
            enable_fsdr: Whether to use FSDR optimization
            enable_profiling: Whether to collect profiling data
        """
        self.model_type = model_type
        self.enable_fsdr = enable_fsdr
        self.enable_profiling = enable_profiling
        
        # Create adapter
        self.adapter = create_adapter(model_type)
        
        # Apply adapter-specific config overrides
        if fsdr_config is None:
            fsdr_config = FSDRConfig(
                feature_dim=self.adapter.get_feature_dim(),
                **self.adapter.get_fsdr_config_overrides()
            )
        
        if dsu_config is None:
            dsu_config = DSUConfig(
                feature_dim=self.adapter.get_feature_dim(),
                cost_type=self.adapter.get_cost_type(),
            )
        
        if ggu_config is None:
            ggu_config = GGUConfig()
        
        self.fsdr_config = fsdr_config
        self.dsu_config = dsu_config
        self.ggu_config = ggu_config
        
        # Initialize processors (DSU and GGU always)
        self.dsu = DSUProcessor(dsu_config)
        self.ggu = GGUProcessor(ggu_config)
        
        # FSDR initialized per-scene with depth candidates
        self.fsdr: Optional[FSDRProcessor] = None
    
    def process_pixel(
        self,
        ref_feature: torch.Tensor,
        target_feature_map: torch.Tensor,
        pixel_coord: torch.Tensor,
        depth_candidates: torch.Tensor,
        raw_gaussian: torch.Tensor,
        density: float,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """
        Process single pixel through complete pipeline.
        
        Returns:
            (gaussian_dict, profiling_dict)
        """
        profiling = {}
        
        # 1. Depth estimation (with or without FSDR)
        if self.enable_fsdr and self.fsdr is not None:
            # Create FSDR callbacks
            cost_fn = self.dsu.create_fsdr_cost_fn(
                ref_feature, target_feature_map, pixel_coord,
                depth_candidates, ref_intrinsics, ref_extrinsics,
                tgt_intrinsics, tgt_extrinsics,
            )
            prob_fn = self.dsu.create_fsdr_prob_fn(
                target_feature_map, pixel_coord,
                ref_intrinsics, ref_extrinsics,
                tgt_intrinsics, tgt_extrinsics,
            )
            
            fsdr_result = self.fsdr.process_pixel(
                ref_feature, (int(pixel_coord[0]), int(pixel_coord[1])),
                cost_fn, prob_fn
            )
            depth = fsdr_result.depth
            profiling['fsdr'] = {
                'source': fsdr_result.source,
                'cache_hit': fsdr_result.cache_hit,
                'num_searches': fsdr_result.num_searches,
            }
        else:
            # Direct DSU search
            dsu_result = self.dsu.search_depth(
                ref_feature, target_feature_map, pixel_coord,
                depth_candidates, ref_intrinsics, ref_extrinsics,
                tgt_intrinsics, tgt_extrinsics,
            )
            depth = dsu_result.depth
            profiling['dsu'] = {
                'num_candidates': dsu_result.num_candidates,
            }
        
        # 2. Gaussian generation
        gaussian = self.ggu.generate_gaussian(
            pixel_coord, depth, raw_gaussian, density,
            ref_intrinsics, ref_extrinsics,
        )
        
        return gaussian.to_dict(), profiling
    
    def setup_fsdr(self, depth_candidates: torch.Tensor):
        """Initialize FSDR processor with depth candidates."""
        self.fsdr = FSDRProcessor(
            self.fsdr_config,
            depth_candidates,
            enable_profiling=self.enable_profiling,
        )
    
    def get_fsdr_profiling(self) -> Optional[dict]:
        """Get FSDR profiling summary."""
        if self.fsdr is not None:
            return self.fsdr.get_profiling()
        return None
    
    def reset(self):
        """Reset all state for new scene."""
        if self.fsdr is not None:
            self.fsdr.clear_cache()
            self.fsdr.reset_profiling()
