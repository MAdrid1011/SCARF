"""
GGU (Gaussian Generation Unit) Module

Hardware simulator for 3D Gaussian generation from depth and features.

Key Components:
- PositionCalculator: 3D position from pixel + depth
- CovarianceBuilder: Covariance matrix from scales/rotation
- SHRotator: Spherical harmonics rotation
- GGUProcessor: Main processing engine

Example:
    from ggu import GGUProcessor, GGUConfig
    
    config = GGUConfig()
    processor = GGUProcessor(config)
    
    gaussian = processor.generate_gaussian(
        pixel_coord, depth, raw_gaussian, density,
        intrinsics, extrinsics
    )
"""

from .types import GGUConfig, GaussianOutput
from .position_calculator import PositionCalculator
from .covariance_builder import CovarianceBuilder
from .sh_rotator import SHRotator
from .ggu_processor import GGUProcessor

__all__ = [
    'GGUConfig',
    'GaussianOutput',
    'PositionCalculator',
    'CovarianceBuilder',
    'SHRotator',
    'GGUProcessor',
]
