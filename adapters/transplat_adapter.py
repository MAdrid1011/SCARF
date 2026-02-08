"""
Transplat Adapter

Adapter for Transplat model.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from .base_adapter import BaseAdapter, GaussianParams


class TransplatAdapter(BaseAdapter):
    """
    Adapter for Transplat model.
    
    Key differences:
    - Cost volume uses negative values (lower = better)
    - Uses inverse depth (disparity) candidates
    - Softmax applied with negation
    """
    
    def __init__(self, feature_dim: int = 128):
        self._feature_dim = feature_dim
    
    def extract_depth_distribution(
        self,
        cost_volume: torch.Tensor,
        depth_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert cost volume to probabilities.
        
        Formula: probs = softmax(-cost_volume)
        
        The negation converts "lower is better" to "higher is better".
        """
        # Determine dimension to apply softmax
        if cost_volume.dim() == 4:  # [B, D, H, W]
            dim = 1
        elif cost_volume.dim() == 3:  # [D, H, W]
            dim = 0
        else:  # [D]
            dim = 0
        
        # Negate costs before softmax
        probs = F.softmax(-cost_volume, dim=dim)
        return probs
    
    def get_depth_candidates(
        self,
        near: float,
        far: float,
        num_candidates: int,
    ) -> torch.Tensor:
        """
        Generate inverse depth (disparity) candidates.
        
        Linear spacing in inverse depth space for better near-range resolution.
        """
        # Inverse depth (disparity) space
        disp_near = 1.0 / far   # Far → small disparity
        disp_far = 1.0 / near   # Near → large disparity
        
        # Linear spacing in disparity
        disparities = torch.linspace(disp_near, disp_far, num_candidates)
        
        # Convert back to depth
        depths = 1.0 / disparities
        
        return depths
    
    def project_to_target(
        self,
        ref_coords: torch.Tensor,
        depth: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Standard pinhole camera projection."""
        N = ref_coords.shape[0]
        device = ref_coords.device
        dtype = ref_coords.dtype
        
        # Unproject to 3D
        ones = torch.ones(N, 1, device=device, dtype=dtype)
        uv_homog = torch.cat([ref_coords, ones], dim=1)  # [N, 3]
        
        # K^{-1} @ [u, v, 1]^T
        K_inv = torch.linalg.inv(ref_intrinsics)
        rays = (K_inv @ uv_homog.T).T  # [N, 3]
        
        # Scale by depth
        points_cam = rays * depth.unsqueeze(1)  # [N, 3]
        
        # To world
        R_ref = ref_extrinsics[:3, :3]
        t_ref = ref_extrinsics[:3, 3]
        points_world = (R_ref @ points_cam.T).T + t_ref  # [N, 3]
        
        # To target camera
        R_tgt_inv = tgt_extrinsics[:3, :3].T
        t_tgt = tgt_extrinsics[:3, 3]
        points_tgt = (R_tgt_inv @ (points_world - t_tgt).T).T  # [N, 3]
        
        # Project to 2D
        points_tgt_proj = (tgt_intrinsics @ points_tgt.T).T  # [N, 3]
        tgt_coords = points_tgt_proj[:, :2] / (points_tgt_proj[:, 2:3] + 1e-8)
        
        return tgt_coords
    
    def get_feature_dim(self) -> int:
        return self._feature_dim
    
    def get_cost_type(self) -> str:
        return 'cost'
    
    def get_fsgr_config_overrides(self) -> Dict:
        return {
            'hamming_threshold': 4,
            'high_confidence_threshold': 0.8,
        }
    
    def get_scale_range(self) -> Tuple[float, float]:
        """Transplat uses (0.5, 15.0) scale range."""
        return (0.5, 15.0)
    
    def parse_raw_gaussian(
        self,
        raw_gaussian: torch.Tensor,
        sh_degree: int = 4,
    ) -> GaussianParams:
        """
        Parse Transplat's raw Gaussian format.
        
        Format: [scales(3), rotation(4), sh(3*num_sh)]
        """
        num_sh = (sh_degree + 1) ** 2
        
        return GaussianParams(
            raw_scales=raw_gaussian[:3],
            raw_rotation=raw_gaussian[3:7],
            raw_sh=raw_gaussian[7:7+3*num_sh],
        )
