"""
Depth Sampler

Projection and bilinear sampling for depth search.
"""
import torch
from typing import Tuple

from .types import DSUConfig


class DepthSampler:
    """
    Depth sampling through projection and bilinear interpolation.
    
    Operations:
        1. Project reference pixel to target view at given depth
        2. Sample target feature at projected location
    
    Hardware Mapping:
        - Projection: 3x3 matrix solve + 4x4 transforms
        - Bilinear: 4 memory reads + weighted sum
        - Total: ~500 LUTs, 8 DSPs, 6 cycles
    
    Example:
        sampler = DepthSampler(config)
        tgt_coord = sampler.project_to_target(
            ref_coord, depth, intrinsics, extrinsics
        )
        feature = sampler.bilinear_sample(feature_map, tgt_coord)
    """
    
    def __init__(self, config: DSUConfig):
        """Initialize depth sampler."""
        self.config = config
    
    def project_to_target(
        self,
        ref_coord: torch.Tensor,       # [2]
        depth: float,
        ref_intrinsics: torch.Tensor,  # [3, 3]
        ref_extrinsics: torch.Tensor,  # [4, 4]
        tgt_intrinsics: torch.Tensor,  # [3, 3]
        tgt_extrinsics: torch.Tensor,  # [4, 4]
    ) -> torch.Tensor:
        """
        Project reference pixel to target view at given depth.
        
        Args:
            ref_coord: (u, v) reference pixel coordinates
            depth: Depth value
            ref_intrinsics: Reference camera intrinsics [3, 3]
            ref_extrinsics: Reference camera extrinsics [4, 4] (c2w)
            tgt_intrinsics: Target camera intrinsics [3, 3]
            tgt_extrinsics: Target camera extrinsics [4, 4] (c2w)
        
        Returns:
            [2] target pixel coordinates (u', v')
        """
        # Unproject to 3D in camera space
        uv_homog = torch.tensor([ref_coord[0], ref_coord[1], 1.0])
        ray_cam = torch.linalg.solve(ref_intrinsics, uv_homog)
        ray_cam = ray_cam / (torch.norm(ray_cam) + 1e-8)
        point_cam = depth * ray_cam
        
        # Transform to world
        point_homog = torch.cat([point_cam, torch.tensor([1.0])])
        point_world = ref_extrinsics @ point_homog
        
        # Transform to target camera
        tgt_extrinsics_inv = torch.linalg.inv(tgt_extrinsics)
        point_tgt = tgt_extrinsics_inv @ point_world
        
        # Project to target image
        point_tgt_3d = point_tgt[:3]
        if abs(point_tgt_3d[2]) < 1e-8:
            # Behind camera or at camera, return center
            return torch.tensor([tgt_intrinsics[0, 2], tgt_intrinsics[1, 2]])
        
        point_tgt_normalized = point_tgt_3d / point_tgt_3d[2]
        projected = tgt_intrinsics @ point_tgt_normalized
        
        return projected[:2]
    
    def bilinear_sample(
        self,
        feature_map: torch.Tensor,  # [C, H, W]
        coord: torch.Tensor,        # [2]
    ) -> torch.Tensor:
        """
        Bilinear interpolation sampling.
        
        Args:
            feature_map: [C, H, W] target feature map
            coord: [2] (u, v) coordinates (possibly fractional)
        
        Returns:
            [C] sampled feature vector
        """
        C, H, W = feature_map.shape
        
        # Clamp to valid range
        u = torch.clamp(coord[0], 0, W - 1)
        v = torch.clamp(coord[1], 0, H - 1)
        
        # Integer and fractional parts
        u0, v0 = int(u.item()), int(v.item())
        u1 = min(u0 + 1, W - 1)
        v1 = min(v0 + 1, H - 1)
        
        du = (u - u0).item()
        dv = (v - v0).item()
        
        # Bilinear interpolation
        f00 = feature_map[:, v0, u0]
        f01 = feature_map[:, v1, u0]
        f10 = feature_map[:, v0, u1]
        f11 = feature_map[:, v1, u1]
        
        result = (
            f00 * (1 - du) * (1 - dv) +
            f10 * du * (1 - dv) +
            f01 * (1 - du) * dv +
            f11 * du * dv
        )
        
        return result
    
    def sample_target_features(
        self,
        target_feature_map: torch.Tensor,
        ref_coord: torch.Tensor,
        depth_candidates: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample target features at all depth candidates.
        
        Args:
            target_feature_map: [C, H, W]
            ref_coord: [2]
            depth_candidates: [D]
            *_intrinsics, *_extrinsics: Camera parameters
        
        Returns:
            [D, C] target features at each depth
        """
        D = len(depth_candidates)
        C = target_feature_map.shape[0]
        
        target_features = torch.zeros(D, C)
        
        for d in range(D):
            depth = float(depth_candidates[d])
            
            # Project
            tgt_coord = self.project_to_target(
                ref_coord, depth,
                ref_intrinsics, ref_extrinsics,
                tgt_intrinsics, tgt_extrinsics,
            )
            
            # Sample
            target_features[d] = self.bilinear_sample(target_feature_map, tgt_coord)
        
        return target_features
