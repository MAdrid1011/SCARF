"""
Base Adapter

Abstract base class for model-specific adapters.
"""
from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class GaussianParams:
    """Parsed Gaussian parameters from raw network output."""
    raw_scales: torch.Tensor      # [3] or [N, 3]
    raw_rotation: torch.Tensor    # [4] or [N, 4] quaternion
    raw_sh: torch.Tensor          # [3*num_sh] or [N, 3*num_sh]
    extra: Dict[str, torch.Tensor] = None  # Model-specific extra params


class BaseAdapter(ABC):
    """
    Abstract base class for model-specific adapters.
    
    Isolates all model-specific logic from core SCARF components.
    Each adapter must implement methods for:
    - Converting cost/correlation volumes to probabilities
    - Generating depth candidates
    - Projecting points between views
    - Parsing raw Gaussian parameters
    """
    
    @abstractmethod
    def extract_depth_distribution(
        self,
        cost_volume: torch.Tensor,
        depth_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert model-specific cost/correlation volume to probability distribution.
        
        Args:
            cost_volume: [B, D, H, W] or [D, H, W] or [D]
            depth_candidates: [D] depth values
        
        Returns:
            probs: Same shape as cost_volume, sums to 1 along D dimension
        """
        pass
    
    @abstractmethod
    def get_depth_candidates(
        self,
        near: float,
        far: float,
        num_candidates: int,
    ) -> torch.Tensor:
        """
        Generate depth candidates for this model.
        
        Args:
            near: Near plane distance
            far: Far plane distance
            num_candidates: Number of depth samples
        
        Returns:
            candidates: [D] depth values
        """
        pass
    
    @abstractmethod
    def project_to_target(
        self,
        ref_coords: torch.Tensor,
        depth: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project reference coordinates to target view at given depths.
        
        Args:
            ref_coords: [N, 2] (u, v) coordinates
            depth: [N] depth values
            ref_intrinsics: [3, 3]
            ref_extrinsics: [4, 4] (c2w)
            tgt_intrinsics: [3, 3]
            tgt_extrinsics: [4, 4] (c2w)
        
        Returns:
            tgt_coords: [N, 2] target coordinates
        """
        pass
    
    def get_feature_dim(self) -> int:
        """Return feature dimension for this model."""
        return 128
    
    def get_cost_type(self) -> str:
        """Return cost type: 'correlation' or 'cost'."""
        return 'correlation'
    
    def supports_fsgr(self) -> bool:
        """Whether this model supports FSGR optimization."""
        return True
    
    def supports_saes(self) -> bool:
        """Whether this model supports SAES optimization."""
        return True
    
    def get_fsgr_config_overrides(self) -> Dict:
        """Get model-specific FSGR configuration overrides."""
        return {}
    
    def get_saes_config_overrides(self) -> Dict:
        """Get model-specific SAES configuration overrides."""
        return {}
    
    def parse_raw_gaussian(
        self,
        raw_gaussian: torch.Tensor,
        sh_degree: int = 4,
    ) -> GaussianParams:
        """
        Parse raw Gaussian parameters from network output.
        
        Default implementation assumes Transplat format:
        [scales(3), rotation(4), sh(3*num_sh)]
        
        Override for models with different formats.
        
        Args:
            raw_gaussian: [D] raw network output
            sh_degree: Spherical harmonics degree
        
        Returns:
            GaussianParams with parsed components
        """
        num_sh = (sh_degree + 1) ** 2
        
        raw_scales = raw_gaussian[:3]
        raw_rotation = raw_gaussian[3:7]
        raw_sh = raw_gaussian[7:7+3*num_sh]
        
        return GaussianParams(
            raw_scales=raw_scales,
            raw_rotation=raw_rotation,
            raw_sh=raw_sh,
        )
    
    def get_gaussian_raw_dim(self, sh_degree: int = 4) -> int:
        """
        Get dimension of raw Gaussian output.
        
        Args:
            sh_degree: Spherical harmonics degree
        
        Returns:
            Total dimension of raw Gaussian vector
        """
        num_sh = (sh_degree + 1) ** 2
        return 3 + 4 + 3 * num_sh  # scales + rotation + sh
    
    def get_scale_range(self) -> Tuple[float, float]:
        """Get (scale_min, scale_max) for this model."""
        return (0.5, 15.0)  # Default Transplat values
