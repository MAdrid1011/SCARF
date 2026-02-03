"""
GGU Processor

Main processing engine for Gaussian Generation Unit.
"""
import torch
from typing import Optional

from .types import GGUConfig, GaussianOutput
from .position_calculator import PositionCalculator
from .covariance_builder import CovarianceBuilder
from .sh_rotator import SHRotator


class GGUProcessor:
    """
    Main GGU processing engine.
    
    Orchestrates:
        - Position calculation from pixel + depth
        - Scale mapping and covariance building
        - Spherical harmonics rotation
        - Gaussian assembly
    
    Hardware Mapping:
        - Position: ~200 LUTs, 6 DSPs
        - Covariance: ~400 LUTs, 12 DSPs
        - SH rotation: ~300 LUTs, 18 DSPs
        - Total: ~900 LUTs, 36 DSPs, 16 cycles
    
    Example:
        processor = GGUProcessor(config)
        
        gaussian = processor.generate_gaussian(
            pixel_coord, depth, raw_gaussian, density,
            intrinsics, extrinsics
        )
    """
    
    def __init__(self, config: GGUConfig):
        """Initialize GGU processor."""
        self.config = config
        self.position_calc = PositionCalculator(config)
        self.cov_builder = CovarianceBuilder(config)
        self.sh_rotator = SHRotator(config)
    
    def generate_gaussian(
        self,
        pixel_coord: torch.Tensor,
        depth: float,
        raw_gaussian: torch.Tensor,
        density: float,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> GaussianOutput:
        """
        Generate complete Gaussian from network output.
        
        Args:
            pixel_coord: [2] (u, v) pixel coordinates
            depth: Depth value
            raw_gaussian: [C_raw] = scales(3) + rotation(4) + sh(3*D_sh)
            density: Raw density/opacity value
            intrinsics: [3, 3] camera intrinsics
            extrinsics: [4, 4] camera extrinsics (c2w)
        
        Returns:
            GaussianOutput with position, covariance, opacity, harmonics
        """
        # Parse raw features
        raw_scales = raw_gaussian[:3]
        raw_rotation = raw_gaussian[3:7]
        num_sh = self.config.num_sh_coeffs
        raw_sh = raw_gaussian[7:7+3*num_sh].reshape(3, num_sh)
        
        # 1. Compute position
        mean = self.position_calc.compute_position(
            pixel_coord, depth, intrinsics, extrinsics
        )
        
        # 2. Build covariance
        scales = self.cov_builder.map_scales(raw_scales, depth)
        local_cov = self.cov_builder.build_covariance(scales, raw_rotation)
        
        # 3. Transform to world space
        R_c2w = extrinsics[:3, :3]
        world_cov = self.cov_builder.transform_to_world(local_cov, R_c2w)
        
        # 4. Rotate spherical harmonics
        world_sh = self.sh_rotator.rotate(raw_sh, R_c2w)
        
        # 5. Compute opacity
        opacity = float(torch.sigmoid(torch.tensor(density)))
        
        return GaussianOutput(
            mean=mean,
            cov=world_cov,
            opacity=opacity,
            harmonics=world_sh,
        )
    
    def generate_batch(
        self,
        pixel_coords: torch.Tensor,
        depths: torch.Tensor,
        raw_gaussians: torch.Tensor,
        densities: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> list:
        """
        Generate batch of Gaussians.
        
        Args:
            pixel_coords: [N, 2]
            depths: [N]
            raw_gaussians: [N, C_raw]
            densities: [N]
            intrinsics: [3, 3]
            extrinsics: [4, 4]
        
        Returns:
            List of GaussianOutput
        """
        N = len(pixel_coords)
        gaussians = []
        
        for i in range(N):
            gaussian = self.generate_gaussian(
                pixel_coords[i],
                float(depths[i]),
                raw_gaussians[i],
                float(densities[i]),
                intrinsics,
                extrinsics,
            )
            gaussians.append(gaussian)
        
        return gaussians
