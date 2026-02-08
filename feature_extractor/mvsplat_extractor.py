"""
MVSplat Feature Extractor

Complete feature extraction for MVSplat model using SCARF hardware simulation.
Uses hardware simulators for actual feature computation.

Components:
- CNN Encoder → CNNEncoderSimulator
- Multi-View Transformer → TransformerSimulator
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass
from einops import rearrange

import sys
from pathlib import Path
SCARF_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from .cnn_simulator import CNNEncoderSimulator
from .transformer_simulator import TransformerSimulator
from .types import CNNConfig, TransformerConfig


@dataclass
class MVSplatFeatureOutput:
    """Output from MVSplat feature extraction."""
    # Main transformer features [B, V, C, H/8, W/8]
    trans_features: torch.Tensor
    # CNN features [BV, C, H/8, W/8]
    cnn_features: Optional[torch.Tensor] = None
    # Cycle counts
    cnn_cycles: int = 0
    transformer_cycles: int = 0
    total_cycles: int = 0
    cycle_breakdown: Dict[str, int] = None


class MVSplatFeatureExtractor:
    """
    Feature extractor for MVSplat model using SCARF hardware simulators.
    
    Uses CNNEncoderSimulator and TransformerSimulator for actual computation.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Original backbone (for reference and weight loading)
        self.backbone = None
        self.feature_channels = 128
        
        # Hardware simulators
        self.cnn_sim: Optional[CNNEncoderSimulator] = None
        self.transformer_sim: Optional[TransformerSimulator] = None
        
        # Config
        self.wo_cross_attn = False
        self.use_hardware = True
    
    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> 'MVSplatFeatureExtractor':
        """Create extractor from MVSplat encoder."""
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
            extractor.wo_cross_attn = encoder.wo_backbone_cross_attn
        
        # Initialize CNN simulator
        # MVSplat: encoder.backbone.backbone is CNNEncoder
        cnn_module = None
        if hasattr(encoder.backbone, 'backbone'):
            cnn_module = encoder.backbone.backbone
        elif hasattr(encoder.backbone, 'cnet'):
            cnn_module = encoder.backbone.cnet
        
        if cnn_module is not None:
            # MVSplat uses 4x downscale, so num_output_scales=0
            cnn_config = CNNConfig(
                input_channels=3,
                output_dim=extractor.feature_channels,
                num_output_scales=0,  # 4x downscale (not 8x)
            )
            extractor.cnn_sim = CNNEncoderSimulator(cnn_config, device)
            extractor.cnn_sim.load_from_pytorch(cnn_module)
        
        # Initialize Transformer simulator
        trans_module = None
        if hasattr(encoder.backbone, 'transformer'):
            trans_module = encoder.backbone.transformer
        
        if trans_module is not None:
            trans_config = TransformerConfig(
                d_model=extractor.feature_channels,
                num_layers=6,
                num_heads=1,
                ffn_dim_expansion=4,
            )
            extractor.transformer_sim = TransformerSimulator(trans_config, device)
            try:
                extractor.transformer_sim.load_from_pytorch(trans_module)
            except (AttributeError, TypeError):
                pass
        
        return extractor
    
    def forward(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        attn_splits: int = 2,
    ) -> MVSplatFeatureOutput:
        """
        Extract features using SCARF hardware simulators.
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not loaded. Use from_encoder() to create extractor.")
        
        b, v, c, h, w = images.shape
        
        cnn_cycles = 0
        transformer_cycles = 0
        cnn_features = None
        trans_features = None
        
        # Normalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1).to(images.device)
        images_norm = (images - mean) / std
        concat = rearrange(images_norm, 'b v c h w -> (b v) c h w')
        
        with torch.no_grad():
            # ============================================================
            # 1. CNN Feature Extraction using Hardware Simulator
            # ============================================================
            if self.cnn_sim is not None and self.use_hardware:
                cnn_features, cnn_cycles = self.cnn_sim.forward(concat)
            elif hasattr(self.backbone, 'backbone'):
                # Fallback to original (backbone.backbone is CNNEncoder)
                cnn_features = self.backbone.backbone(concat)
                cnn_cycles = self._estimate_cnn_cycles(h, w)
            elif hasattr(self.backbone, 'cnet'):
                cnn_features = self.backbone.cnet(concat)
                cnn_cycles = self._estimate_cnn_cycles(h, w)
            
            # ============================================================
            # 2. Transformer using Hardware Simulator
            # ============================================================
            cnn_features_bvchw = None  # [B, V, C, H, W] format for output
            if cnn_features is not None:
                feat_h, feat_w = cnn_features.shape[2], cnn_features.shape[3]
                features_per_view = rearrange(cnn_features, '(b v) c h w -> b v c h w', b=b, v=v)
                cnn_features_bvchw = features_per_view  # Store for output
                features_list = list(torch.unbind(features_per_view, dim=1))
                
                # Add position encoding before transformer (matches backbone.forward)
                from mvsplat.src.model.encoder.backbone.backbone_multiview import (
                    feature_add_position_list,
                )
                features_list = feature_add_position_list(
                    features_list, attn_splits, self.feature_channels
                )
                
                if self.transformer_sim is not None and len(self.transformer_sim.layers) > 0 and self.use_hardware:
                    features_list_out, transformer_cycles = self.transformer_sim.forward(features_list)
                    trans_features = rearrange(
                        torch.stack(features_list_out, dim=1),
                        'b v c h w -> b v c h w'
                    )
                elif hasattr(self.backbone, 'transformer'):
                    features_list_out = self.backbone.transformer(
                        features_list,
                        attn_num_splits=attn_splits,
                    )
                    trans_features = rearrange(
                        torch.stack(features_list_out, dim=1),
                        'b v c h w -> b v c h w'
                    )
                    transformer_cycles = self._estimate_transformer_cycles(feat_h, feat_w)
                else:
                    trans_features = features_per_view
                    transformer_cycles = 0
        
        # Adjust for cross-attention disabled
        if self.wo_cross_attn:
            transformer_cycles = transformer_cycles // 2
        
        cycle_breakdown = {
            'cnn': cnn_cycles,
            'transformer': transformer_cycles,
        }
        
        return MVSplatFeatureOutput(
            trans_features=trans_features,
            cnn_features=cnn_features_bvchw,  # [B, V, C, H, W] format
            cnn_cycles=cnn_cycles,
            transformer_cycles=transformer_cycles,
            total_cycles=cnn_cycles + transformer_cycles,
            cycle_breakdown=cycle_breakdown,
        )
    
    def _estimate_cnn_cycles(self, h: int, w: int) -> int:
        """Estimate CNN cycles when hardware simulator not available."""
        cycles = 0
        curr_h, curr_w = h, w
        
        cycles += (curr_h * curr_w * 3 * 64 * 49) // 256
        curr_h, curr_w = curr_h // 2, curr_w // 2
        
        for _ in range(2):
            cycles += 2 * (curr_h * curr_w * 64 * 64 * 9) // 256
        
        cycles += (curr_h * curr_w * 64 * 96 * 9) // 256
        curr_h, curr_w = curr_h // 2, curr_w // 2
        cycles += (curr_h * curr_w * 96 * 96 * 9) // 256
        
        cycles += (curr_h * curr_w * 96 * 128 * 9) // 256
        curr_h, curr_w = curr_h // 2, curr_w // 2
        cycles += (curr_h * curr_w * 128 * 128 * 9) // 256
        
        cycles += (curr_h * curr_w * 128 * 128) // 256
        
        return cycles
    
    def _estimate_transformer_cycles(self, h: int, w: int) -> int:
        """Estimate Transformer cycles when hardware simulator not available."""
        seq_len = h * w
        d_model = 128
        num_layers = 6
        
        cycles = 0
        for _ in range(num_layers):
            cycles += 10 * (seq_len * d_model * d_model) // 128
        
        return cycles
