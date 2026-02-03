"""
GGU Data Types

Core data structures for the GGU module.
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass
class GGUConfig:
    """
    GGU configuration parameters matching Transplat's GaussianAdapterCfg.
    
    Default values match common Transplat configurations.
    For specific models, override with actual values from GaussianAdapterCfg:
    - Transplat/MVSplat: scale_activation='sigmoid', use_depth_scaling=True
    - DepthSplat: scale_activation='softplus', use_depth_scaling=False
    
    Attributes:
        scale_min: Minimum scale value (Transplat: gaussian_scale_min)
        scale_max: Maximum scale value (Transplat: gaussian_scale_max)
        sh_degree: Spherical harmonics degree (Transplat: sh_degree)
        depth_scale_multiplier: Fallback multiplier if intrinsics not provided
        image_shape: Default image shape for coordinate normalization
        scale_activation: 'sigmoid' (Transplat/MVSplat) or 'softplus' (DepthSplat)
        use_depth_scaling: Whether to multiply scales by depth (False for DepthSplat)
        softplus_shift: Shift for softplus activation (DepthSplat uses -4.0)
    """
    scale_min: float = 0.5      # RE10K default
    scale_max: float = 15.0     # RE10K default
    sh_degree: int = 4          # RE10K default
    depth_scale_multiplier: float = 0.1
    image_shape: Tuple[int, int] = (256, 256)
    # Model-specific configurations
    scale_activation: str = 'sigmoid'  # 'sigmoid' or 'softplus'
    use_depth_scaling: bool = True     # True for Transplat/MVSplat, False for DepthSplat
    softplus_shift: float = -4.0       # Only used when scale_activation='softplus'
    direction_normalize: str = 'norm'  # 'norm' (Transplat) or 'z' (DepthSplat)
    
    def __post_init__(self):
        if self.scale_min < 0:
            raise ValueError("scale_min must be non-negative")
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must be > scale_min")
        if self.sh_degree < 0 or self.sh_degree > 4:
            raise ValueError("sh_degree must be in [0, 4]")
        if self.scale_activation not in ('sigmoid', 'softplus'):
            raise ValueError("scale_activation must be 'sigmoid' or 'softplus'")
    
    @property
    def num_sh_coeffs(self) -> int:
        """Number of SH coefficients per color channel."""
        return (self.sh_degree + 1) ** 2
    
    @property
    def raw_gaussian_dim(self) -> int:
        """
        Expected dimension of raw_gaussian input.
        
        Structure: scales(3) + rotation(4) + sh(3 * num_sh_coeffs)
        """
        return 3 + 4 + 3 * self.num_sh_coeffs
    
    @classmethod
    def from_transplat_config(cls, gaussian_adapter_cfg) -> 'GGUConfig':
        """
        Create GGUConfig from Transplat's GaussianAdapterCfg.
        
        Args:
            gaussian_adapter_cfg: GaussianAdapterCfg from Transplat
        
        Returns:
            GGUConfig with matching parameters
        """
        return cls(
            scale_min=gaussian_adapter_cfg.gaussian_scale_min,
            scale_max=gaussian_adapter_cfg.gaussian_scale_max,
            sh_degree=gaussian_adapter_cfg.sh_degree,
        )


@dataclass
class GaussianOutput:
    """
    Output of GGU Gaussian generation.
    
    Matches the structure needed by Transplat's decoder.
    
    Attributes:
        mean: [3] 3D position in world space
        cov: [3, 3] covariance matrix in world space
        opacity: Opacity value in [0, 1]
        harmonics: [3, num_sh] spherical harmonics coefficients
        scales: [3] scale values (for debugging/export)
        rotations: [4] quaternion (for debugging/export)
    """
    mean: torch.Tensor      # [3]
    cov: torch.Tensor       # [3, 3]
    opacity: float
    harmonics: torch.Tensor  # [3, num_sh]
    scales: Optional[torch.Tensor] = None    # [3]
    rotations: Optional[torch.Tensor] = None  # [4]
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'mean': self.mean,
            'cov': self.cov,
            'opacity': self.opacity,
            'harmonics': self.harmonics,
            'scales': self.scales,
            'rotations': self.rotations,
        }
