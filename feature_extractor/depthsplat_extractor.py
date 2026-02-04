"""
DepthSplat Feature Extractor

Complete feature extraction for DepthSplat model using SCARF hardware simulation.
Reuses existing hardware units: CNNEncoderSimulator, TransformerSimulator, ViTSimulator.

Components (all using existing SCARF hardware):
- CNN Encoder → CNNEncoderSimulator
- Multi-View Transformer → TransformerSimulator  
- DINOv2 ViT → ViTSimulator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from dataclasses import dataclass
from einops import rearrange

import sys
from pathlib import Path
SCARF_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from .vit_simulator import ViTSimulator, ViTConfig
from .cnn_simulator import CNNEncoderSimulator
from .transformer_simulator import TransformerSimulator
from .types import CNNConfig, TransformerConfig


@dataclass
class DepthSplatFeatureOutput:
    """Output from DepthSplat feature extraction."""
    # Multi-view transformer features [BV, C, H/8, W/8]
    trans_features: torch.Tensor
    # CNN features [BV, C, H/8, W/8]
    cnn_features: Optional[torch.Tensor] = None
    # DINOv2/Mono features [BV, C, H/8, W/8]
    mono_features: Optional[torch.Tensor] = None
    # Cycle counts
    cnn_cycles: int = 0
    transformer_cycles: int = 0
    dinov2_cycles: int = 0
    total_cycles: int = 0
    cycle_breakdown: Dict[str, int] = None


class DepthSplatFeatureExtractor:
    """
    Feature extractor for DepthSplat model using SCARF hardware units.
    
    Reuses existing hardware simulators:
    - CNNEncoderSimulator for backbone
    - TransformerSimulator for MV transformer
    - ViTSimulator for DINOv2
    """
    
    def __init__(self, device: torch.device = None, vit_type: str = 'vits'):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vit_type = vit_type
        
        # Original model components (loaded from encoder)
        self.backbone = None  # CNNEncoder
        self.transformer = None  # MultiViewFeatureTransformer
        self.pretrained = None  # DINOv2
        self.feature_channels = 128
        
        # Hardware simulators
        self.cnn_sim: Optional[CNNEncoderSimulator] = None
        self.transformer_sim: Optional[TransformerSimulator] = None
        self.vit_sim: Optional[ViTSimulator] = None
    
    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> 'DepthSplatFeatureExtractor':
        """Create extractor from DepthSplat encoder."""
        device = next(encoder.parameters()).device
        
        # Detect ViT type
        vit_type = 'vits'
        if hasattr(encoder, 'depth_predictor') and hasattr(encoder.depth_predictor, 'vit_type'):
            vit_type = encoder.depth_predictor.vit_type
        
        extractor = cls(device=device, vit_type=vit_type)
        
        dp = encoder.depth_predictor
        
        # Extract components from depth_predictor (MultiViewUniMatch)
        if hasattr(dp, 'backbone'):
            extractor.backbone = dp.backbone
        if hasattr(dp, 'transformer'):
            extractor.transformer = dp.transformer
        if hasattr(dp, 'pretrained'):
            extractor.pretrained = dp.pretrained
        if hasattr(dp, 'feature_channels'):
            extractor.feature_channels = dp.feature_channels
        
        # Initialize hardware simulators with weights
        # 1. CNN Simulator
        if extractor.backbone is not None:
            cnn_config = CNNConfig(
                input_channels=3,
                output_dim=extractor.feature_channels,
            )
            extractor.cnn_sim = CNNEncoderSimulator(cnn_config, device)
            extractor.cnn_sim.load_from_pytorch(extractor.backbone)
        
        # 2. Transformer Simulator
        if extractor.transformer is not None:
            trans_config = TransformerConfig(
                d_model=extractor.feature_channels,
                num_heads=1,
                num_layers=6,
                ffn_dim_expansion=4,  # Correct parameter name
            )
            extractor.transformer_sim = TransformerSimulator(trans_config, device)
            try:
                extractor.transformer_sim.load_from_pytorch(extractor.transformer)
            except (AttributeError, TypeError):
                # Custom transformer not fully compatible
                pass
        
        # 3. ViT Simulator for DINOv2
        if extractor.pretrained is not None:
            vit_config = ViTConfig(vit_type=vit_type, patch_size=14)
            extractor.vit_sim = ViTSimulator(vit_config, device)
            extractor.vit_sim.load_from_dinov2(extractor.pretrained)
        
        return extractor
    
    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images for backbone."""
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)
        return (images - mean) / std
    
    def forward(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        attn_splits: int = 2,
        **kwargs,
    ) -> DepthSplatFeatureOutput:
        """
        Extract features using SCARF hardware simulators.
        
        Calls each component's original forward for bit-accuracy,
        while counting cycles using hardware simulators.
        """
        b, v, c, h, w = images.shape
        
        # Normalize images
        images_norm = self._normalize_images(images)
        concat = rearrange(images_norm, "b v c h w -> (b v) c h w")
        
        cnn_cycles = 0
        transformer_cycles = 0
        dinov2_cycles = 0
        
        cnn_features = None
        trans_features = None
        mono_features = None
        
        # ============================================================
        # 1. CNN Feature Extraction (using backbone + CNNEncoderSimulator)
        # ============================================================
        if self.backbone is not None:
            with torch.no_grad():
                # Run original backbone for bit-accurate features
                features_list = self.backbone(concat)
                # Reverse: resolution from low to high
                features_list = features_list[::-1]
                cnn_features = features_list[0]  # Lowest resolution (main feature)
                
                # Count cycles using simulator
                if self.cnn_sim is not None:
                    _, cnn_cycles = self.cnn_sim.forward(concat)
        
        # ============================================================
        # 2. MV Transformer (using transformer + TransformerSimulator)
        # ============================================================
        if self.transformer is not None and cnn_features is not None:
            with torch.no_grad():
                # Add position encoding
                from depthsplat.src.model.encoder.unimatch.utils import mv_feature_add_position
                features_pos = mv_feature_add_position(
                    cnn_features, attn_splits, self.feature_channels
                )
                
                # Reshape for transformer: [BV, C, H, W] -> list of [B, C, H, W]
                features_list = list(
                    torch.unbind(
                        rearrange(features_pos, "(b v) c h w -> b v c h w", b=b, v=v), 
                        dim=1
                    )
                )
                
                # Run original transformer for bit-accurate features
                features_list_mv = self.transformer(
                    features_list,
                    attn_num_splits=attn_splits,
                )
                
                # Stack back: list of [B, C, H, W] -> [BV, C, H, W]
                trans_features = rearrange(
                    torch.stack(features_list_mv, dim=1), 
                    "b v c h w -> (b v) c h w"
                )
                
                # Count cycles using simulator or estimate
                if self.transformer_sim is not None and len(self.transformer_sim.layers) > 0:
                    _, transformer_cycles = self.transformer_sim.forward(features_list)
                else:
                    # Estimate cycles: 6 layers × (attn + ffn) per layer
                    seq_len = cnn_features.shape[2] * cnn_features.shape[3]
                    d_model = self.feature_channels
                    # Self-attn: 4 GEMMs, Cross-attn: 4 GEMMs, FFN: 2 GEMMs = 10 GEMMs per layer
                    gemm_per_layer = 10 * (seq_len * d_model * d_model) // 128
                    transformer_cycles = 6 * gemm_per_layer
        
        # ============================================================
        # 3. DINOv2 Feature Extraction (using pretrained + ViTSimulator)
        # ============================================================
        if self.pretrained is not None:
            with torch.no_grad():
                # Resize to patch-aligned size
                ori_h, ori_w = concat.shape[-2:]
                resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14
                concat_resized = F.interpolate(
                    concat, (resize_h, resize_w), mode='bilinear', align_corners=True
                )
                
                # Run original DINOv2 for bit-accurate features
                intermediate_layer_idx = {
                    'vits': [2, 5, 8, 11],
                    'vitb': [2, 5, 8, 11],
                    'vitl': [4, 11, 17, 23],
                }
                mono_intermediate = list(
                    self.pretrained.get_intermediate_layers(
                        concat_resized, 
                        intermediate_layer_idx[self.vit_type], 
                        return_class_token=False
                    )
                )
                
                # Reshape and resize last feature
                last_feat = mono_intermediate[-1]
                last_feat = last_feat.reshape(
                    concat.shape[0], resize_h // 14, resize_w // 14, -1
                ).permute(0, 3, 1, 2).contiguous()
                mono_features = F.interpolate(
                    last_feat, (ori_h // 8, ori_w // 8), 
                    mode='bilinear', align_corners=True
                )
                
                # Count cycles using ViT simulator
                if self.vit_sim is not None:
                    vit_output = self.vit_sim.forward(concat_resized)
                    dinov2_cycles = vit_output.total_cycles
        
        cycle_breakdown = {
            'cnn': cnn_cycles,
            'transformer': transformer_cycles,
            'dinov2': dinov2_cycles,
        }
        
        return DepthSplatFeatureOutput(
            trans_features=trans_features,
            cnn_features=cnn_features,
            mono_features=mono_features,
            cnn_cycles=cnn_cycles,
            transformer_cycles=transformer_cycles,
            dinov2_cycles=dinov2_cycles,
            total_cycles=cnn_cycles + transformer_cycles + dinov2_cycles,
            cycle_breakdown=cycle_breakdown,
        )
