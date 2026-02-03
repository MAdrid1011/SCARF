"""
Covariance Builder

Covariance matrix construction from scales and rotation.
"""
import torch
import numpy as np

from .types import GGUConfig


class CovarianceBuilder:
    """
    Build covariance matrix from scale and rotation parameters.
    
    Operations:
        1. Map raw scales to valid range via sigmoid
        2. Apply depth-adaptive scaling
        3. Convert quaternion to rotation matrix
        4. Build covariance: R @ diag(s²) @ R^T
        5. Transform to world space
    
    Hardware Mapping:
        - Sigmoid: LUT or piecewise approximation
        - Quaternion to rotation: 12 multiplies + adds
        - Matrix multiply: 27 multiplies
        - Total: ~400 LUTs, 12 DSPs, 5 cycles
    
    Example:
        builder = CovarianceBuilder(config)
        scales = builder.map_scales(raw_scales, depth)
        cov = builder.build_covariance(scales, quaternion)
        world_cov = builder.transform_to_world(cov, c2w_rotation)
    """
    
    def __init__(self, config: GGUConfig):
        """Initialize covariance builder."""
        self.config = config
    
    def map_scales(
        self,
        raw_scales: torch.Tensor,
        depth: float,
    ) -> torch.Tensor:
        """
        Map raw scales to valid range with depth adaptation.
        
        Args:
            raw_scales: [3] unbounded scale values from network
            depth: Depth value for adaptive scaling
        
        Returns:
            [3] mapped scale values
        """
        # Sigmoid mapping to [scale_min, scale_max]
        base_scales = (
            self.config.scale_min + 
            (self.config.scale_max - self.config.scale_min) * 
            torch.sigmoid(raw_scales)
        )
        
        # Depth-adaptive scaling
        scales = base_scales * depth * self.config.depth_scale_multiplier
        
        return scales
    
    def quaternion_to_rotation(
        self,
        quaternion: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
        
        Args:
            quaternion: [4] quaternion (w, x, y, z)
            eps: Numerical stability epsilon
        
        Returns:
            [3, 3] rotation matrix
        """
        # Normalize
        q = quaternion / (torch.norm(quaternion) + eps)
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Rotation matrix
        R = torch.tensor([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], dtype=quaternion.dtype)
        
        return R
    
    def build_covariance(
        self,
        scales: torch.Tensor,
        quaternion: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build 3x3 covariance matrix from scales and rotation.
        
        Formula: Σ = R @ diag(s²) @ R^T
        
        Args:
            scales: [3] scale values
            quaternion: [4] rotation quaternion
        
        Returns:
            [3, 3] covariance matrix
        """
        R = self.quaternion_to_rotation(quaternion)
        S = torch.diag(scales ** 2)
        cov = R @ S @ R.T
        return cov
    
    def transform_to_world(
        self,
        covariance: torch.Tensor,
        c2w_rotation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform covariance to world space.
        
        Formula: Σ_world = R_c2w @ Σ_local @ R_c2w^T
        
        Args:
            covariance: [3, 3] local covariance
            c2w_rotation: [3, 3] camera-to-world rotation
        
        Returns:
            [3, 3] world-space covariance
        """
        return c2w_rotation @ covariance @ c2w_rotation.T
