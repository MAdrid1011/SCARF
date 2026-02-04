"""
DepthSplat Feature Extractor

Complete feature extraction for DepthSplat model using SCARF hardware simulation.
Uses DINOv2 (ViT) for monocular depth features.

Components:
- CNN Encoder (multi-scale)
- Multi-View Transformer
- DINOv2 ViT (vits/vitb/vitl)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from einops import rearrange

import sys
from pathlib import Path
SCARF_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from .vit_simulator import ViTSimulator, ViTConfig


@dataclass
class DepthSplatFeatureOutput:
    """Output from DepthSplat feature extraction."""
    # Multi-view transformer features [B, V, C, H/8, W/8]
    trans_features: torch.Tensor
    # CNN features (multi-scale list)
    cnn_features: Optional[List[torch.Tensor]] = None
    # DINOv2/Mono features [BV, C, H/8, W/8]
    mono_features: Optional[torch.Tensor] = None
    # Intermediate mono features (for DPT head)
    mono_intermediate: Optional[List[torch.Tensor]] = None
    # Cycle counts
    cnn_cycles: int = 0
    transformer_cycles: int = 0
    dinov2_cycles: int = 0
    total_cycles: int = 0
    cycle_breakdown: Dict[str, int] = None


class DepthSplatFeatureExtractor:
    """
    Feature extractor for DepthSplat model.
    
    Handles the complex multi-scale architecture with DINOv2.
    
    Usage:
        extractor = DepthSplatFeatureExtractor.from_encoder(model.encoder)
        output = extractor.forward(images, extrinsics)
        
        features = output.trans_features
        mono_features = output.mono_features
        cycles = output.total_cycles
    """
    
    def __init__(self, device: torch.device = None, vit_type: str = 'vits'):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Original encoder components
        self.depth_predictor = None  # MultiViewUniMatch
        self.vit_type = vit_type
        
        # ViT config for DINOv2
        self.vit_config = ViTConfig(vit_type=vit_type, patch_size=14)
        self.vit_sim: Optional[ViTSimulator] = None
        
        # CNN/Transformer config
        self.feature_channels = 128
        self.num_scales = 1
    
    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> 'DepthSplatFeatureExtractor':
        """
        Create extractor from DepthSplat encoder.
        
        Args:
            encoder: DepthSplat's encoder module
            
        Returns:
            Configured DepthSplatFeatureExtractor
        """
        device = next(encoder.parameters()).device
        
        # Detect ViT type
        vit_type = 'vits'
        if hasattr(encoder, 'depth_predictor') and hasattr(encoder.depth_predictor, 'vit_type'):
            vit_type = encoder.depth_predictor.vit_type
        
        extractor = cls(device=device, vit_type=vit_type)
        
        # Extract depth predictor (contains backbone, transformer, and DINOv2)
        if hasattr(encoder, 'depth_predictor'):
            extractor.depth_predictor = encoder.depth_predictor
            
            # Load DINOv2 weights into ViT simulator
            if hasattr(encoder.depth_predictor, 'pretrained'):
                extractor.vit_sim = ViTSimulator(extractor.vit_config, device=device)
                extractor.vit_sim.load_from_dinov2(encoder.depth_predictor.pretrained)
            
            # Get config from depth predictor
            if hasattr(encoder.depth_predictor, 'feature_channels'):
                extractor.feature_channels = encoder.depth_predictor.feature_channels
            if hasattr(encoder.depth_predictor, 'num_scales'):
                extractor.num_scales = encoder.depth_predictor.num_scales
        else:
            raise ValueError("Encoder does not have depth_predictor attribute")
        
        return extractor
    
    def forward(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        attn_splits: int = 2,
        **kwargs,
    ) -> DepthSplatFeatureOutput:
        """
        Extract features using hardware simulation.
        
        For DepthSplat, we only count cycles since the depth_predictor
        requires intrinsics which may not always be available.
        The actual features are captured from hooks in demo.py.
        
        Args:
            images: Input images [B, V, 3, H, W]
            extrinsics: Camera extrinsics [B, V, 4, 4]
            intrinsics: Camera intrinsics [B, V, 3, 3]
            attn_splits: Attention window splits
            
        Returns:
            DepthSplatFeatureOutput with cycle counts
        """
        b, v, c, h, w = images.shape
        
        # Count cycles (without running depth_predictor)
        cnn_cycles = self._count_cnn_cycles(h, w)
        transformer_cycles = self._count_transformer_cycles(h // 8, w // 8)
        dinov2_cycles = self._count_dinov2_cycles(h, w)
        
        cycle_breakdown = {
            'cnn': cnn_cycles,
            'transformer': transformer_cycles,
            'dinov2': dinov2_cycles,
        }
        
        return DepthSplatFeatureOutput(
            trans_features=None,  # Not computed - use hooks
            cnn_features=None,
            mono_features=None,
            mono_intermediate=None,
            cnn_cycles=cnn_cycles,
            transformer_cycles=transformer_cycles,
            dinov2_cycles=dinov2_cycles,
            total_cycles=cnn_cycles + transformer_cycles + dinov2_cycles,
            cycle_breakdown=cycle_breakdown,
        )
    
    def _count_cnn_cycles(self, h: int, w: int) -> int:
        """Count CNN cycles (multi-scale encoder)."""
        cycles = 0
        
        # Similar to Transplat but may have multi-scale output
        curr_h, curr_w = h, w
        
        # Initial Conv7x7 stride=2
        cycles += self._conv_cycles(curr_h, curr_w, 3, 64, 7, 2)
        cycles += self._norm_cycles(curr_h // 2, curr_w // 2, 64)
        curr_h, curr_w = curr_h // 2, curr_w // 2
        
        # ResBlocks with multi-scale output
        channels = [64, 96, 128]
        for i, cout in enumerate(channels):
            cin = 64 if i == 0 else channels[i - 1]
            stride = 1 if i == 0 else 2
            
            cycles += self._resblock_cycles(curr_h, curr_w, cin, cout, stride)
            if stride == 2:
                curr_h, curr_w = curr_h // 2, curr_w // 2
            cycles += self._resblock_cycles(curr_h, curr_w, cout, cout, 1)
        
        return cycles
    
    def _count_transformer_cycles(self, h: int, w: int) -> int:
        """Count Transformer cycles."""
        cycles = 0
        seq_len = h * w
        d_model = self.feature_channels
        num_layers = 6
        
        for _ in range(num_layers):
            # Self-attention + Cross-attention + FFN
            # Self-attention
            cycles += 3 * self._gemm_cycles(seq_len, d_model, d_model)
            cycles += 2 * self._gemm_cycles(seq_len, seq_len, d_model // 8)
            # Cross-attention
            cycles += 3 * self._gemm_cycles(seq_len, d_model, d_model)
            cycles += 2 * self._gemm_cycles(seq_len, seq_len, d_model // 8)
            # FFN
            cycles += self._gemm_cycles(seq_len, d_model, d_model * 4)
            cycles += self._gemm_cycles(seq_len, d_model * 4, d_model)
            # LayerNorms
            cycles += 4 * self._norm_cycles_flat(seq_len, d_model)
        
        return cycles
    
    def _count_dinov2_cycles(self, h: int, w: int) -> int:
        """
        Count DINOv2 (ViT) cycles using ViT simulator.
        
        Uses the ViTSimulator to get accurate cycle counts.
        """
        if self.vit_sim is not None:
            # Use ViT simulator's cycle model
            # Create dummy input to get cycles
            dummy_images = torch.zeros(1, 3, h // 14 * 14, w // 14 * 14, device=self.device)
            try:
                output = self.vit_sim.forward(dummy_images)
                return output.total_cycles
            except Exception:
                pass
        
        # Fallback: estimate based on ViT architecture
        patch_size = 14
        seq_len = (h // patch_size) * (w // patch_size)
        
        # ViT configs
        vit_configs = {
            'vits': {'d_model': 384, 'num_heads': 6, 'num_layers': 12},
            'vitb': {'d_model': 768, 'num_heads': 12, 'num_layers': 12},
            'vitl': {'d_model': 1024, 'num_heads': 16, 'num_layers': 24},
        }
        config = vit_configs.get(self.vit_type, vit_configs['vits'])
        
        d_model = config['d_model']
        num_layers = config['num_layers']
        
        cycles = 0
        
        # Patch embedding (Conv)
        cycles += self._conv_cycles(h, w, 3, d_model, patch_size, patch_size)
        
        # Transformer layers
        for _ in range(num_layers):
            # Self-attention
            cycles += 4 * self._gemm_cycles(seq_len, d_model, d_model)
            cycles += 2 * self._gemm_cycles(seq_len, seq_len, d_model // config['num_heads'])
            # FFN
            cycles += self._gemm_cycles(seq_len, d_model, d_model * 4)
            cycles += self._gemm_cycles(seq_len, d_model * 4, d_model)
            # LayerNorms
            cycles += 2 * self._norm_cycles_flat(seq_len, d_model)
        
        return cycles
    
    def _conv_cycles(self, h: int, w: int, cin: int, cout: int, k: int, s: int) -> int:
        """Conv2d cycles."""
        out_h = (h + 2 * (k // 2) - k) // s + 1
        out_w = (w + 2 * (k // 2) - k) // s + 1
        macs = out_h * out_w * cin * cout * k * k
        return macs // 256 + 1
    
    def _resblock_cycles(self, h: int, w: int, cin: int, cout: int, stride: int) -> int:
        """ResidualBlock cycles."""
        cycles = 0
        cycles += self._conv_cycles(h, w, cin, cout, 3, stride)
        cycles += self._norm_cycles(h // stride, w // stride, cout)
        cycles += self._conv_cycles(h // stride, w // stride, cout, cout, 3, 1)
        cycles += self._norm_cycles(h // stride, w // stride, cout)
        if stride > 1 or cin != cout:
            cycles += self._conv_cycles(h, w, cin, cout, 1, stride)
            cycles += self._norm_cycles(h // stride, w // stride, cout)
        return cycles
    
    def _norm_cycles(self, h: int, w: int, c: int) -> int:
        """InstanceNorm cycles."""
        return 5 * h * w * c
    
    def _norm_cycles_flat(self, seq_len: int, d: int) -> int:
        """LayerNorm cycles."""
        return 5 * seq_len * d
    
    def _gemm_cycles(self, m: int, k: int, n: int) -> int:
        """GEMM cycles."""
        return (m * k * n) // 128 + 1
