"""
Covariance Builder - Standalone Implementation

Covariance matrix construction matching Transplat's GaussianAdapter exactly.
No dependency on Transplat modules.
"""
import torch
from typing import Tuple

from .types import GGUConfig


class CovarianceBuilder:
    """
    Build covariance matrix from scale and rotation parameters.
    
    Matches Transplat's GaussianAdapter covariance computation exactly:
    1. scales = scale_min + (scale_max - scale_min) * sigmoid(raw_scales)
    2. scales = scales * depth * scale_multiplier(intrinsics)
    3. rotations = normalize(raw_rotations)
    4. cov = R @ S @ S^T @ R^T (using quaternion format: i, j, k, r)
    5. cov_world = R_c2w @ cov @ R_c2w^T
    
    Hardware Mapping:
        - Sigmoid: LUT (256 entries) or piecewise linear
        - Quaternion to rotation: 12 multiplies + adds
        - Scale matrix: 3 multiplies
        - Matrix multiply: 27 multiplies × 2
        - Total: ~400 LUTs, 12 DSPs, 5 cycles
    """
    
    def __init__(self, config: GGUConfig):
        """Initialize covariance builder."""
        self.config = config
    
    def get_scale_multiplier(
        self,
        intrinsics: torch.Tensor,    # [3, 3]
        image_shape: Tuple[int, int] = (256, 256),
        multiplier: float = 0.1,
    ) -> float:
        """
        Compute scale multiplier based on intrinsics.
        
        Matches Transplat's GaussianAdapter.get_scale_multiplier():
        xy_multipliers = multiplier * (K_inv[:2,:2] @ pixel_size)
        return xy_multipliers.sum()
        
        Args:
            intrinsics: [3, 3] camera intrinsics
            image_shape: (H, W)
            multiplier: Base multiplier (default 0.1)
        
        Returns:
            Scale multiplier value
        """
        h, w = image_shape
        pixel_size = torch.tensor([1.0 / w, 1.0 / h], dtype=intrinsics.dtype)
        
        # Get inverse of top-left 2x2
        K_2x2 = intrinsics[:2, :2]
        K_2x2_inv = torch.linalg.inv(K_2x2)
        
        # Compute multiplier
        xy_mult = multiplier * (K_2x2_inv @ pixel_size)
        return xy_mult.sum().item()
    
    def map_scales(
        self,
        raw_scales: torch.Tensor,  # [3]
        depth: float,
        intrinsics: torch.Tensor = None,
        image_shape: Tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """
        Map raw scales to valid range with depth and intrinsics adaptation.
        
        Matches Transplat's GaussianAdapter.forward():
        scales = scale_min + (scale_max - scale_min) * sigmoid(raw_scales)
        scales = scales * depth * scale_multiplier
        
        Args:
            raw_scales: [3] raw scale values from network
            depth: Depth value
            intrinsics: [3, 3] camera intrinsics (optional)
            image_shape: (H, W)
        
        Returns:
            [3] mapped scale values
        """
        # Sigmoid mapping to [scale_min, scale_max]
        base_scales = (
            self.config.scale_min + 
            (self.config.scale_max - self.config.scale_min) * 
            torch.sigmoid(raw_scales)
        )
        
        # Compute scale multiplier
        if intrinsics is not None:
            scale_mult = self.get_scale_multiplier(intrinsics, image_shape)
        else:
            scale_mult = self.config.depth_scale_multiplier
        
        # Apply depth and multiplier
        scales = base_scales * depth * scale_mult
        
        return scales
    
    def quaternion_to_rotation(
        self,
        quaternion: torch.Tensor,  # [4] in (i, j, k, r) format
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Matches Transplat's quaternion_to_matrix() in gaussians.py:
        Order: (i, j, k, r) = (x, y, z, w)
        
        Args:
            quaternion: [4] quaternion in (i, j, k, r) format
            eps: Numerical stability epsilon
        
        Returns:
            [3, 3] rotation matrix
        """
        # Normalize first
        q = quaternion / (torch.norm(quaternion) + eps)
        
        # Unpack in Transplat's order: i, j, k, r
        i, j, k, r = q[0], q[1], q[2], q[3]
        
        # Compute rotation matrix
        # Matches Transplat's quaternion_to_matrix exactly
        two_s = 2.0 / ((q * q).sum() + eps)
        
        R = torch.stack([
            1 - two_s * (j*j + k*k),
            two_s * (i*j - k*r),
            two_s * (i*k + j*r),
            two_s * (i*j + k*r),
            1 - two_s * (i*i + k*k),
            two_s * (j*k - i*r),
            two_s * (i*k - j*r),
            two_s * (j*k + i*r),
            1 - two_s * (i*i + j*j),
        ]).reshape(3, 3)
        
        return R
    
    def build_covariance(
        self,
        scales: torch.Tensor,        # [3] mapped scales
        raw_rotation: torch.Tensor,  # [4] raw rotation quaternion
    ) -> torch.Tensor:
        """
        Build 3x3 covariance matrix from scales and rotation.
        
        Matches Transplat's build_covariance():
        S = diag(scales)
        R = quaternion_to_matrix(rotation)
        cov = R @ S @ S^T @ R^T
        
        Args:
            scales: [3] mapped scale values
            raw_rotation: [4] rotation quaternion (i, j, k, r)
        
        Returns:
            [3, 3] covariance matrix
        """
        # Normalize rotation
        rotation = raw_rotation / (torch.norm(raw_rotation) + 1e-8)
        
        # Convert to rotation matrix
        R = self.quaternion_to_rotation(rotation)
        
        # Build scale matrix
        S = torch.diag(scales)
        
        # Covariance: R @ S @ S^T @ R^T
        cov = R @ S @ S.T @ R.T
        
        return cov
    
    def transform_to_world(
        self,
        covariance: torch.Tensor,    # [3, 3] local covariance
        c2w_rotation: torch.Tensor,  # [3, 3] camera-to-world rotation
    ) -> torch.Tensor:
        """
        Transform covariance to world space.
        
        Matches Transplat's:
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        
        Args:
            covariance: [3, 3] local covariance
            c2w_rotation: [3, 3] camera-to-world rotation matrix
        
        Returns:
            [3, 3] world-space covariance
        """
        return c2w_rotation @ covariance @ c2w_rotation.T
