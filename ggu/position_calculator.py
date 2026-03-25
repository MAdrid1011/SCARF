"""
Position Calculator - Standalone Implementation

3D position calculation matching TranSplat's GaussianAdapter exactly.
No dependency on TranSplat modules.
"""
import torch
from typing import Tuple

from .types import GGUConfig


class PositionCalculator:
    """
    Calculate 3D world position from pixel coordinates and depth.
    
    Matches TranSplat's get_world_rays() + mean = origin + direction * depth
    
    Hardware Mapping:
        - Matrix inverse: ~100 LUTs (or precomputed)
        - Matrix multiply: ~100 LUTs, 6 DSPs
        - Normalize: ~50 LUTs
        - Total: ~250 LUTs, 6 DSPs, 3 cycles
    """
    
    def __init__(self, config: GGUConfig):
        """Initialize position calculator."""
        self.config = config
    
    def unproject_to_ray(
        self,
        coordinates: torch.Tensor,  # [2] normalized (x, y) in [0, 1]
        intrinsics: torch.Tensor,   # [3, 3]
    ) -> torch.Tensor:
        """
        Compute normalized ray direction in camera space.
        
        Matches TranSplat's unproject() function.
        
        Args:
            coordinates: [2] normalized (x, y) in [0, 1]
            intrinsics: [3, 3] camera intrinsics
        
        Returns:
            [3] normalized ray direction in camera space
        """
        # Homogenize: (x, y) -> (x, y, 1)
        coords_homog = torch.tensor([
            coordinates[0].item(),
            coordinates[1].item(),
            1.0
        ], dtype=intrinsics.dtype)
        
        # Apply inverse intrinsics
        K_inv = torch.linalg.inv(intrinsics)
        ray_direction = K_inv @ coords_homog
        
        # Normalize
        ray_direction = ray_direction / (torch.norm(ray_direction) + 1e-8)
        
        return ray_direction
    
    def get_world_rays(
        self,
        coordinates: torch.Tensor,  # [2] normalized (x, y) in [0, 1]
        extrinsics: torch.Tensor,   # [4, 4] camera-to-world
        intrinsics: torch.Tensor,   # [3, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get world-space ray origin and direction.
        
        Matches TranSplat's get_world_rays() exactly.
        
        Args:
            coordinates: [2] normalized (x, y) in [0, 1]
            extrinsics: [4, 4] camera-to-world matrix
            intrinsics: [3, 3] camera intrinsics
        
        Returns:
            origins: [3] world-space origin
            directions: [3] world-space direction (normalized)
        """
        # Get camera-space ray direction
        direction_cam = self.unproject_to_ray(coordinates, intrinsics)
        
        # Transform direction to world space (homogeneous)
        # direction_homog = (dx, dy, dz, 0) for vectors
        direction_homog = torch.tensor([
            direction_cam[0].item(),
            direction_cam[1].item(), 
            direction_cam[2].item(),
            0.0
        ], dtype=extrinsics.dtype)
        
        # Apply camera-to-world transform
        direction_world = extrinsics @ direction_homog
        direction_world = direction_world[:3]
        
        # Origin is the camera position (last column of extrinsics)
        origin = extrinsics[:3, 3]
        
        return origin, direction_world
    
    def compute_position(
        self,
        pixel_coord: torch.Tensor,  # [2] (row, col) or (y, x) in pixels
        depth: float,
        intrinsics: torch.Tensor,   # [3, 3]
        extrinsics: torch.Tensor,   # [4, 4]
        image_shape: Tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """
        Compute 3D world position from pixel and depth.
        
        Matches TranSplat: mean = origin + direction * depth
        
        Args:
            pixel_coord: [2] (y, x) pixel coordinates
            depth: Depth value in world units
            intrinsics: [3, 3] camera intrinsics
            extrinsics: [4, 4] camera extrinsics (c2w)
            image_shape: (H, W) for normalization
        
        Returns:
            [3] world-space 3D position
        """
        h, w = image_shape
        
        # Convert pixel (y, x) to normalized (x, y) in [0, 1]
        # TranSplat uses: coordinates = (index + 0.5) / length
        normalized_coords = torch.tensor([
            (float(pixel_coord[1]) + 0.5) / w,  # x
            (float(pixel_coord[0]) + 0.5) / h,  # y
        ], dtype=intrinsics.dtype)
        
        # Get world rays
        origin, direction = self.get_world_rays(normalized_coords, extrinsics, intrinsics)
        
        # Compute position: mean = origin + direction * depth
        position = origin + direction * depth
        
        return position
