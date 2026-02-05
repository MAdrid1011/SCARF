"""
Transplat Feature Extractor

Complete feature extraction for Transplat model using SCARF hardware simulation.
Uses hardware simulators for actual feature computation.

Components:
- CNN Encoder → CNNEncoderSimulator
- Multi-View Transformer → TransformerSimulator
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

from .cnn_simulator import CNNEncoderSimulator
from .transformer_simulator import TransformerSimulator
from .types import CNNConfig, TransformerConfig


@dataclass
class TransplatFeatureOutput:
    """Output from Transplat feature extraction."""
    # Main transformer features [B, V, C, H/8, W/8]
    trans_features: torch.Tensor
    # CNN features [BV, C, H/8, W/8]
    cnn_features: Optional[torch.Tensor] = None
    # Cycle counts
    cnn_cycles: int = 0
    transformer_cycles: int = 0
    total_cycles: int = 0
    cycle_breakdown: Dict[str, int] = None


class TransplatFeatureExtractor:
    """
    Feature extractor for Transplat model using SCARF hardware simulators.
    
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
        
        # Fallback mode (use original PyTorch if hardware fails)
        self.use_hardware = True
    
    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> 'TransplatFeatureExtractor':
        """Create extractor from Transplat encoder."""
        device = next(encoder.parameters()).device
        extractor = cls(device=device)
        
        # Extract backbone
        if hasattr(encoder, 'backbone'):
            extractor.backbone = encoder.backbone
            if hasattr(encoder.backbone, 'feature_channels'):
                extractor.feature_channels = encoder.backbone.feature_channels
        else:
            raise ValueError("Encoder does not have backbone attribute")
        
        # Initialize CNN simulator
        # Transplat: encoder.backbone.backbone is CNNEncoder
        cnn_module = None
        if hasattr(encoder.backbone, 'backbone'):
            cnn_module = encoder.backbone.backbone
        elif hasattr(encoder.backbone, 'cnet'):
            cnn_module = encoder.backbone.cnet
        
        if cnn_module is not None:
            # Transplat uses downscale_factor=4, so num_output_scales=0
            # (no extra stride in layer3)
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
    ) -> TransplatFeatureOutput:
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
                # Reshape for transformer: [BV, C, H, W] -> [B, V, C, H, W]
                feat_h, feat_w = cnn_features.shape[2], cnn_features.shape[3]
                features_per_view = rearrange(cnn_features, '(b v) c h w -> b v c h w', b=b, v=v)
                cnn_features_bvchw = features_per_view  # Store for output
                features_list = list(torch.unbind(features_per_view, dim=1))
                
                if self.transformer_sim is not None and len(self.transformer_sim.layers) > 0 and self.use_hardware:
                    features_list_out, transformer_cycles = self.transformer_sim.forward(features_list)
                    trans_features = rearrange(
                        torch.stack(features_list_out, dim=1),
                        'b v c h w -> b v c h w'
                    )
                elif hasattr(self.backbone, 'transformer'):
                    # Fallback to original transformer
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
                    # No transformer, just reshape CNN features
                    trans_features = features_per_view
                    transformer_cycles = 0
        
        cycle_breakdown = {
            'cnn': cnn_cycles,
            'transformer': transformer_cycles,
        }
        
        return TransplatFeatureOutput(
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
        
        # Conv7x7 stride=2
        cycles += (curr_h * curr_w * 3 * 64 * 49) // 256
        curr_h, curr_w = curr_h // 2, curr_w // 2
        
        # Layer 1: 2x ResBlock (64ch)
        for _ in range(2):
            cycles += 2 * (curr_h * curr_w * 64 * 64 * 9) // 256
        
        # Layer 2: 2x ResBlock (96ch, stride=2 on first)
        cycles += (curr_h * curr_w * 64 * 96 * 9) // 256
        curr_h, curr_w = curr_h // 2, curr_w // 2
        cycles += (curr_h * curr_w * 96 * 96 * 9) // 256
        
        # Layer 3: 2x ResBlock (128ch, stride=2 on first)
        cycles += (curr_h * curr_w * 96 * 128 * 9) // 256
        curr_h, curr_w = curr_h // 2, curr_w // 2
        cycles += (curr_h * curr_w * 128 * 128 * 9) // 256
        
        # Conv1x1
        cycles += (curr_h * curr_w * 128 * 128) // 256
        
        return cycles
    
    def _estimate_transformer_cycles(self, h: int, w: int) -> int:
        """Estimate Transformer cycles when hardware simulator not available."""
        seq_len = h * w
        d_model = 128
        num_layers = 6
        
        cycles = 0
        for _ in range(num_layers):
            # Self-attn + Cross-attn + FFN
            cycles += 10 * (seq_len * d_model * d_model) // 128
        
        return cycles
    
    def _count_cnn_cycles(self, h: int, w: int) -> int:
        """
        Count CNN cycles using hardware model.
        
        Based on Transplat's CNNEncoder:
        - Conv7x7 stride=2 + InstanceNorm + ReLU
        - 2x ResidualBlock (64ch)
        - 2x ResidualBlock (96ch, stride=2)
        - 2x ResidualBlock (128ch, stride=2)
        - Conv1x1 output
        """
        cycles = 0
        
        # Input resolution
        curr_h, curr_w = h, w
        
        # Initial Conv7x7 stride=2
        cycles += self._conv_cycles(curr_h, curr_w, 3, 64, 7, 2)
        cycles += self._norm_cycles(curr_h // 2, curr_w // 2, 64)
        cycles += self._activation_cycles(curr_h // 2, curr_w // 2, 64)
        curr_h, curr_w = curr_h // 2, curr_w // 2
        
        # Layer 1: 2x ResBlock (64ch)
        for _ in range(2):
            cycles += self._resblock_cycles(curr_h, curr_w, 64, 64, stride=1)
        
        # Layer 2: 2x ResBlock (96ch, stride=2 on first)
        cycles += self._resblock_cycles(curr_h, curr_w, 64, 96, stride=2)
        curr_h, curr_w = curr_h // 2, curr_w // 2
        cycles += self._resblock_cycles(curr_h, curr_w, 96, 96, stride=1)
        
        # Layer 3: 2x ResBlock (128ch, stride=2 on first)
        cycles += self._resblock_cycles(curr_h, curr_w, 96, 128, stride=2)
        curr_h, curr_w = curr_h // 2, curr_w // 2
        cycles += self._resblock_cycles(curr_h, curr_w, 128, 128, stride=1)
        
        # Output Conv1x1
        cycles += self._conv_cycles(curr_h, curr_w, 128, 128, 1, 1)
        
        return cycles
    
    def _count_transformer_cycles(self, h: int, w: int) -> int:
        """
        Count Transformer cycles using hardware model.
        
        Based on Transplat's MultiViewFeatureTransformer:
        - 6 layers
        - Self-attention + Cross-view attention + FFN
        """
        cycles = 0
        seq_len = h * w
        d_model = self.transformer_config['d_model']
        num_layers = self.transformer_config['num_layers']
        ffn_expansion = self.transformer_config['ffn_expansion']
        
        for _ in range(num_layers):
            # Self-attention: Q,K,V projections (3 GEMMs) + attention (2 GEMMs)
            # QKV: [N, d] @ [d, d] -> 3 * N * d * d
            cycles += 3 * self._gemm_cycles(seq_len, d_model, d_model)
            # Attention: [N, N] compute + [N, d] output
            cycles += self._gemm_cycles(seq_len, seq_len, d_model // 8)  # Multi-head
            cycles += self._gemm_cycles(seq_len, d_model, d_model)  # Output proj
            
            # Cross-view attention (similar)
            cycles += 3 * self._gemm_cycles(seq_len, d_model, d_model)
            cycles += self._gemm_cycles(seq_len, seq_len, d_model // 8)
            cycles += self._gemm_cycles(seq_len, d_model, d_model)
            
            # FFN: [N, d] -> [N, 4d] -> [N, d]
            cycles += self._gemm_cycles(seq_len, d_model, d_model * ffn_expansion)
            cycles += self._activation_cycles_flat(seq_len * d_model * ffn_expansion)
            cycles += self._gemm_cycles(seq_len, d_model * ffn_expansion, d_model)
            
            # LayerNorm (2 per sublayer)
            cycles += 4 * self._norm_cycles_flat(seq_len, d_model)
        
        return cycles
    
    def _conv_cycles(self, h: int, w: int, cin: int, cout: int, k: int, s: int) -> int:
        """Conv2d cycles: MACs / 256 (systolic array throughput)."""
        out_h = (h + 2 * (k // 2) - k) // s + 1
        out_w = (w + 2 * (k // 2) - k) // s + 1
        macs = out_h * out_w * cin * cout * k * k
        return macs // 256 + 1
    
    def _resblock_cycles(self, h: int, w: int, cin: int, cout: int, stride: int) -> int:
        """ResidualBlock cycles."""
        cycles = 0
        # Conv1
        cycles += self._conv_cycles(h, w, cin, cout, 3, stride)
        cycles += self._norm_cycles(h // stride, w // stride, cout)
        cycles += self._activation_cycles(h // stride, w // stride, cout)
        # Conv2
        cycles += self._conv_cycles(h // stride, w // stride, cout, cout, 3, 1)
        cycles += self._norm_cycles(h // stride, w // stride, cout)
        # Downsample (if needed)
        if stride > 1 or cin != cout:
            cycles += self._conv_cycles(h, w, cin, cout, 1, stride)
            cycles += self._norm_cycles(h // stride, w // stride, cout)
        # Final ReLU
        cycles += self._activation_cycles(h // stride, w // stride, cout)
        return cycles
    
    def _norm_cycles(self, h: int, w: int, c: int) -> int:
        """InstanceNorm cycles: 5 * N elements."""
        return 5 * h * w * c
    
    def _norm_cycles_flat(self, seq_len: int, d: int) -> int:
        """LayerNorm cycles: 5 * N elements."""
        return 5 * seq_len * d
    
    def _activation_cycles(self, h: int, w: int, c: int) -> int:
        """ReLU cycles: 1 * N elements."""
        return h * w * c
    
    def _activation_cycles_flat(self, n: int) -> int:
        """GELU cycles: 2 * N elements (LUT lookup)."""
        return 2 * n
    
    def _gemm_cycles(self, m: int, k: int, n: int) -> int:
        """GEMM cycles: M * K * N / 128 (GEMM unit throughput)."""
        return (m * k * n) // 128 + 1
