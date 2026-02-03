"""
Base Adapter

Abstract base class for model-specific adapters.
"""
from abc import ABC, abstractmethod
import torch
from typing import Dict


class BaseAdapter(ABC):
    """
    Abstract base class for model-specific adapters.
    
    Isolates all model-specific logic from core SCARF components.
    Each adapter must implement methods for:
    - Converting cost/correlation volumes to probabilities
    - Generating depth candidates
    - Projecting points between views
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
    
    def supports_fsdr(self) -> bool:
        """Whether this model supports FSDR optimization."""
        return True
    
    def supports_saes(self) -> bool:
        """Whether this model supports SAES optimization."""
        return True
    
    def get_fsdr_config_overrides(self) -> Dict:
        """Get model-specific FSDR configuration overrides."""
        return {}
    
    def get_saes_config_overrides(self) -> Dict:
        """Get model-specific SAES configuration overrides."""
        return {}
