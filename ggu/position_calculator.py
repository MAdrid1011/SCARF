"""
Position Calculator

3D position calculation from pixel and depth.
"""
import torch

from .types import GGUConfig


class PositionCalculator:
    """
    Calculate 3D world position from pixel coordinates and depth.
    
    Hardware Mapping:
        - Matrix inverse/solve: ~100 LUTs
        - Matrix multiply: ~100 LUTs, 6 DSPs
        - Total: ~200 LUTs, 6 DSPs, 3 cycles
    """
    
    def __init__(self, config: GGUConfig):
        """Initialize position calculator."""
        self.config = config
    
    def get_ray_direction(
        self,
        pixel_coord: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalized ray direction in camera space."""
        uv_homog = torch.tensor([pixel_coord[0], pixel_coord[1], 1.0])
        ray = torch.linalg.solve(intrinsics, uv_homog)
        return ray / (torch.norm(ray) + 1e-8)
    
    def compute_position(
        self,
        pixel_coord: torch.Tensor,
        depth: float,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 3D world position from pixel and depth."""
        ray_cam = self.get_ray_direction(pixel_coord, intrinsics)
        R_c2w = extrinsics[:3, :3]
        t_c2w = extrinsics[:3, 3]
        ray_world = R_c2w @ ray_cam
        position = t_c2w + ray_world * depth
        return position
