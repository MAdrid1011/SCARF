"""
MVSplat Feature Extractor

Complete feature extraction for MVSplat model using SCARF hardware simulation.
Similar to Transplat but without DepthAnythingV2.

Components:
- CNN Encoder (ResNet-style, 128 channels)
- Multi-View Transformer (6 layers)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass

import sys
from pathlib import Path
SCARF_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from .transplat_extractor import TransplatFeatureExtractor, TransplatFeatureOutput


@dataclass
class MVSplatFeatureOutput:
    """Output from MVSplat feature extraction."""
    # Main transformer features [B, V, C, H/8, W/8]
    trans_features: torch.Tensor
    # CNN features (list of multi-scale if available)
    cnn_features: Optional[List[torch.Tensor]] = None
    # Cycle counts
    cnn_cycles: int = 0
    transformer_cycles: int = 0
    total_cycles: int = 0
    cycle_breakdown: Dict[str, int] = None


class MVSplatFeatureExtractor(TransplatFeatureExtractor):
    """
    Feature extractor for MVSplat model.
    
    Inherits from TransplatFeatureExtractor as the architecture is similar,
    but without DepthAnythingV2 components.
    
    Usage:
        extractor = MVSplatFeatureExtractor.from_encoder(model.encoder)
        output = extractor.forward(images, extrinsics)
        
        features = output.trans_features
        cycles = output.total_cycles
    """
    
    def __init__(self, device: torch.device = None):
        super().__init__(device)
        
        # MVSplat may have different transformer config
        self.transformer_config = {
            'num_layers': 6,
            'd_model': 128,
            'num_heads': 1,
            'ffn_expansion': 4,
            # MVSplat can disable cross-view attention
            'wo_cross_attn': False,
        }
    
    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> 'MVSplatFeatureExtractor':
        """
        Create extractor from MVSplat encoder.
        
        Args:
            encoder: MVSplat's encoder module
            
        Returns:
            Configured MVSplatFeatureExtractor
        """
        device = next(encoder.parameters()).device
        extractor = cls(device=device)
        
        # Extract backbone
        if hasattr(encoder, 'backbone'):
            extractor.backbone = encoder.backbone
            if hasattr(encoder.backbone, 'feature_channels'):
                extractor.feature_channels = encoder.backbone.feature_channels
        else:
            raise ValueError("Encoder does not have backbone attribute")
        
        # Check for cross-view attention config
        if hasattr(encoder, 'wo_backbone_cross_attn'):
            extractor.transformer_config['wo_cross_attn'] = encoder.wo_backbone_cross_attn
        
        return extractor
    
    def forward(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        attn_splits: int = 2,
    ) -> MVSplatFeatureOutput:
        """
        Extract features using hardware simulation.
        
        Args:
            images: Input images [B, V, 3, H, W]
            extrinsics: Camera extrinsics [B, V, 4, 4]
            attn_splits: Attention window splits
            
        Returns:
            MVSplatFeatureOutput with features and cycle counts
        """
        # Use parent class forward
        parent_output = super().forward(images, extrinsics, attn_splits)
        
        # Adjust transformer cycles if cross-attention is disabled
        transformer_cycles = parent_output.transformer_cycles
        if self.transformer_config.get('wo_cross_attn', False):
            # Roughly half the transformer cycles without cross-view attention
            transformer_cycles = transformer_cycles // 2
        
        cycle_breakdown = {
            'cnn': parent_output.cnn_cycles,
            'transformer': transformer_cycles,
        }
        
        return MVSplatFeatureOutput(
            trans_features=parent_output.trans_features,
            cnn_features=parent_output.cnn_features,
            cnn_cycles=parent_output.cnn_cycles,
            transformer_cycles=transformer_cycles,
            total_cycles=parent_output.cnn_cycles + transformer_cycles,
            cycle_breakdown=cycle_breakdown,
        )
