"""
DepthSplat Adapter

Adapter for DepthSplat model.
"""
import torch
from typing import Dict

from .mvsplat_adapter import MVSplatAdapter


class DepthSplatAdapter(MVSplatAdapter):
    """
    Adapter for DepthSplat model.
    
    Inherits from MVSplat since depth estimation is similar.
    
    Key difference:
    - DINOv2 features tend to be more semantically consistent
    - Larger feature dimension (384)
    - 3-view support
    """
    
    def __init__(self, feature_dim: int = 384):
        super().__init__(feature_dim)
    
    def get_fsgr_config_overrides(self) -> Dict:
        """
        DepthSplat-specific FSGR configuration.
        
        DINOv2 features are more semantically consistent,
        so we can use tighter thresholds for better cache hit rate.
        """
        return {
            'hamming_threshold': 3,
            'high_confidence_threshold': 0.85,
        }
    
    def get_saes_config_overrides(self) -> Dict:
        """DepthSplat-specific SAES configuration."""
        return {
            'early_stop_threshold': 0.92,  # Slightly higher
        }
    
    def supports_three_view(self) -> bool:
        """DepthSplat uses 3 views for robust depth."""
        return True
