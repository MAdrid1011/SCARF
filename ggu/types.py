"""
GGU Data Types

Core data structures for the GGU module.
"""
from dataclasses import dataclass
import torch


@dataclass
class GGUConfig:
    """
    GGU configuration parameters.
    
    Attributes:
        scale_min: Minimum scale value (default: 0.0005)
        scale_max: Maximum scale value (default: 0.5)
        sh_degree: Spherical harmonics degree (default: 3)
        depth_scale_multiplier: Depth-adaptive scaling factor
    """
    scale_min: float = 0.0005
    scale_max: float = 0.5
    sh_degree: int = 3
    depth_scale_multiplier: float = 1.0
    
    def __post_init__(self):
        if self.scale_min < 0 or self.scale_max <= self.scale_min:
            raise ValueError("scale_min must be non-negative and < scale_max")
        if self.sh_degree < 0 or self.sh_degree > 4:
            raise ValueError("sh_degree must be in [0, 4]")
    
    @property
    def num_sh_coeffs(self) -> int:
        """Number of SH coefficients per color channel."""
        return (self.sh_degree + 1) ** 2


@dataclass
class GaussianOutput:
    """
    Output of GGU Gaussian generation.
    
    Attributes:
        mean: [3] 3D position in world space
        cov: [3, 3] covariance matrix in world space
        opacity: Opacity value in [0, 1]
        harmonics: [3, num_sh] spherical harmonics coefficients
    """
    mean: torch.Tensor
    cov: torch.Tensor
    opacity: float
    harmonics: torch.Tensor
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'mean': self.mean,
            'cov': self.cov,
            'opacity': self.opacity,
            'harmonics': self.harmonics,
        }
