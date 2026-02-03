"""
MVSplat Adapter

Adapter for MVSplat model.
"""
import torch
import torch.nn.functional as F
from typing import Dict

from .base_adapter import BaseAdapter


class MVSplatAdapter(BaseAdapter):
    """
    Adapter for MVSplat model.
    
    Key differences from Transplat:
    - Correlation volume uses positive values (higher = better)
    - Uses linear depth candidates
    - Direct softmax
    """
    
    def __init__(self, feature_dim: int = 64):
        self._feature_dim = feature_dim
    
    def extract_depth_distribution(
        self,
        cost_volume: torch.Tensor,
        depth_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert correlation volume to probabilities.
        
        Formula: probs = softmax(correlation)
        
        Direct softmax since higher correlation = better match.
        """
        if cost_volume.dim() == 4:
            dim = 1
        elif cost_volume.dim() == 3:
            dim = 0
        else:
            dim = 0
        
        probs = F.softmax(cost_volume, dim=dim)
        return probs
    
    def get_depth_candidates(
        self,
        near: float,
        far: float,
        num_candidates: int,
    ) -> torch.Tensor:
        """
        Generate linear depth candidates.
        """
        return torch.linspace(near, far, num_candidates)
    
    def project_to_target(
        self,
        ref_coords: torch.Tensor,
        depth: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Same projection as Transplat (standard pinhole model)."""
        N = ref_coords.shape[0]
        device = ref_coords.device
        dtype = ref_coords.dtype
        
        ones = torch.ones(N, 1, device=device, dtype=dtype)
        uv_homog = torch.cat([ref_coords, ones], dim=1)
        
        K_inv = torch.linalg.inv(ref_intrinsics)
        rays = (K_inv @ uv_homog.T).T
        
        points_cam = rays * depth.unsqueeze(1)
        
        R_ref = ref_extrinsics[:3, :3]
        t_ref = ref_extrinsics[:3, 3]
        points_world = (R_ref @ points_cam.T).T + t_ref
        
        R_tgt_inv = tgt_extrinsics[:3, :3].T
        t_tgt = tgt_extrinsics[:3, 3]
        points_tgt = (R_tgt_inv @ (points_world - t_tgt).T).T
        
        points_tgt_proj = (tgt_intrinsics @ points_tgt.T).T
        tgt_coords = points_tgt_proj[:, :2] / (points_tgt_proj[:, 2:3] + 1e-8)
        
        return tgt_coords
    
    def get_feature_dim(self) -> int:
        return self._feature_dim
    
    def get_cost_type(self) -> str:
        return 'correlation'
