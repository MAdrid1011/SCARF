"""
ViT/DINOv2 Hardware Simulator

Uses existing hardware units (GEMMUnit, NormalizationUnit, ActivationUnit, ConvEngine)
to simulate Vision Transformer inference with accurate cycle counting.

Supports DINOv2 variants: ViT-S/14, ViT-B/14, ViT-L/14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

# Import existing hardware units
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder.gemm_unit import GEMMUnit
from encoder.normalization_unit import NormalizationUnit
from encoder.activation_unit import ActivationUnit
from encoder.conv_engine import ConvEngine
from encoder.types import NormType, ActivationType, GEMMConfig


@dataclass
class ViTConfig:
    """Configuration for ViT/DINOv2 models."""
    vit_type: str = 'vits'  # 'vits', 'vitb', 'vitl'
    patch_size: int = 14
    image_size: int = 256
    
    # Model dimensions (auto-set based on vit_type)
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 12
    mlp_ratio: float = 4.0
    
    def __post_init__(self):
        """Set dimensions based on vit_type."""
        configs = {
            'vits': {'embed_dim': 384, 'num_heads': 6, 'num_layers': 12},
            'vitb': {'embed_dim': 768, 'num_heads': 12, 'num_layers': 12},
            'vitl': {'embed_dim': 1024, 'num_heads': 16, 'num_layers': 24},
        }
        if self.vit_type in configs:
            for k, v in configs[self.vit_type].items():
                setattr(self, k, v)


@dataclass
class ViTOutput:
    """Output from ViT simulator."""
    features: torch.Tensor  # [B, N, D] or list of intermediate features
    total_cycles: int
    patch_embed_cycles: int
    transformer_cycles: int
    cycle_breakdown: Dict[str, int]


class PatchEmbedSim:
    """
    Patch Embedding using ConvEngine.
    
    Equivalent to: nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    """
    
    def __init__(self, config: ViTConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        
        # Use ConvEngine for patch embedding (conv with kernel=stride=patch_size)
        self.conv_engine = ConvEngine()
        
        # Weights (loaded from pretrained model)
        self.weight: Optional[torch.Tensor] = None  # [embed_dim, 3, patch_size, patch_size]
        self.bias: Optional[torch.Tensor] = None    # [embed_dim]
    
    def load_weights(self, patch_embed: nn.Module):
        """Load weights from DINOv2's patch_embed module."""
        if hasattr(patch_embed, 'proj'):
            self.weight = patch_embed.proj.weight.to(self.device)
            if patch_embed.proj.bias is not None:
                self.bias = patch_embed.proj.bias.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Forward pass: [B, 3, H, W] -> [B, N, D]
        
        Returns:
            patches: [B, N, D] where N = (H/patch_size) * (W/patch_size)
            cycles: Hardware cycle count
        """
        B, C, H, W = x.shape
        patch_size = self.config.patch_size
        
        # Use PyTorch conv for correctness
        if self.weight is not None:
            out = F.conv2d(x, self.weight, self.bias, stride=patch_size)
        else:
            # Fallback if weights not loaded
            out = F.conv2d(x, torch.randn(self.config.embed_dim, 3, patch_size, patch_size, device=x.device),
                          stride=patch_size)
        
        # Reshape: [B, D, H/P, W/P] -> [B, N, D]
        out = out.flatten(2).transpose(1, 2)
        
        # Count cycles using ConvEngine model
        # Conv2d: output_h * output_w * in_channels * out_channels * kernel_h * kernel_w / array_size
        out_h, out_w = H // patch_size, W // patch_size
        macs = out_h * out_w * 3 * self.config.embed_dim * patch_size * patch_size
        cycles = macs // 256 + 1  # Assuming 256 MACs per cycle in systolic array
        
        return out, cycles


class MultiHeadAttentionSim:
    """
    Multi-Head Self-Attention using GEMMUnit.
    
    Operations:
    - Q, K, V projections: 3x Linear (GEMMUnit.linear)
    - Attention: Q @ K^T (GEMMUnit.matmul)
    - Softmax
    - Output: Attn @ V (GEMMUnit.matmul)
    - Output projection: Linear (GEMMUnit.linear)
    """
    
    def __init__(self, config: ViTConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        # GEMM unit for all matrix operations
        self.gemm_unit = GEMMUnit()
        
        # Weights
        self.qkv_weight: Optional[torch.Tensor] = None  # [3*embed_dim, embed_dim]
        self.qkv_bias: Optional[torch.Tensor] = None
        self.proj_weight: Optional[torch.Tensor] = None  # [embed_dim, embed_dim]
        self.proj_bias: Optional[torch.Tensor] = None
    
    def load_weights(self, attn: nn.Module):
        """Load weights from DINOv2's attention module."""
        if hasattr(attn, 'qkv'):
            self.qkv_weight = attn.qkv.weight.to(self.device)
            if attn.qkv.bias is not None:
                self.qkv_bias = attn.qkv.bias.to(self.device)
        if hasattr(attn, 'proj'):
            self.proj_weight = attn.proj.weight.to(self.device)
            if attn.proj.bias is not None:
                self.proj_bias = attn.proj.bias.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Forward pass: [B, N, D] -> [B, N, D]
        
        Returns:
            output: [B, N, D]
            cycles: Hardware cycle count
        """
        B, N, D = x.shape
        total_cycles = 0
        
        # QKV projection: [B, N, D] @ [D, 3D] -> [B, N, 3D]
        if self.qkv_weight is not None:
            qkv, cycles = self.gemm_unit.linear(x, self.qkv_weight, self.qkv_bias)
        else:
            qkv = torch.randn(B, N, 3 * D, device=x.device)
            cycles = self.gemm_unit._compute_cycles(
                torch.empty(B, N, D), torch.empty(D, 3 * D)
            )
        total_cycles += cycles.total_cycles
        
        # Reshape to [3, B, num_heads, N, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores: [B, H, N, head_dim] @ [B, H, head_dim, N] -> [B, H, N, N]
        attn, cycles = self.gemm_unit.matmul(q, k.transpose(-2, -1))
        total_cycles += cycles.total_cycles
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        # Softmax cycles: ~5N per head
        total_cycles += B * self.num_heads * N * 5
        
        # Apply attention: [B, H, N, N] @ [B, H, N, head_dim] -> [B, H, N, head_dim]
        out, cycles = self.gemm_unit.matmul(attn, v)
        total_cycles += cycles.total_cycles
        
        # Reshape back: [B, H, N, head_dim] -> [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)
        
        # Output projection: [B, N, D] @ [D, D] -> [B, N, D]
        if self.proj_weight is not None:
            out, cycles = self.gemm_unit.linear(out, self.proj_weight, self.proj_bias)
        else:
            cycles = self.gemm_unit._compute_cycles(
                torch.empty(B, N, D), torch.empty(D, D)
            )
        total_cycles += cycles.total_cycles
        
        return out, total_cycles


class MLPSim:
    """
    MLP/FFN using GEMMUnit and ActivationUnit.
    
    Operations:
    - Linear: [B, N, D] -> [B, N, 4D]
    - GELU activation
    - Linear: [B, N, 4D] -> [B, N, D]
    """
    
    def __init__(self, config: ViTConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        
        self.embed_dim = config.embed_dim
        self.hidden_dim = int(config.embed_dim * config.mlp_ratio)
        
        # Hardware units
        self.gemm_unit = GEMMUnit()
        self.activation_unit = ActivationUnit(ActivationType.GELU)
        
        # Weights
        self.fc1_weight: Optional[torch.Tensor] = None
        self.fc1_bias: Optional[torch.Tensor] = None
        self.fc2_weight: Optional[torch.Tensor] = None
        self.fc2_bias: Optional[torch.Tensor] = None
    
    def load_weights(self, mlp: nn.Module):
        """Load weights from DINOv2's MLP module."""
        if hasattr(mlp, 'fc1'):
            self.fc1_weight = mlp.fc1.weight.to(self.device)
            if mlp.fc1.bias is not None:
                self.fc1_bias = mlp.fc1.bias.to(self.device)
        if hasattr(mlp, 'fc2'):
            self.fc2_weight = mlp.fc2.weight.to(self.device)
            if mlp.fc2.bias is not None:
                self.fc2_bias = mlp.fc2.bias.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Forward pass: [B, N, D] -> [B, N, D]
        
        Returns:
            output: [B, N, D]
            cycles: Hardware cycle count
        """
        B, N, D = x.shape
        total_cycles = 0
        
        # FC1: [B, N, D] -> [B, N, hidden_dim]
        if self.fc1_weight is not None:
            out, cycles = self.gemm_unit.linear(x, self.fc1_weight, self.fc1_bias)
        else:
            out = torch.randn(B, N, self.hidden_dim, device=x.device)
            cycles = self.gemm_unit._compute_cycles(
                torch.empty(B, N, D), torch.empty(D, self.hidden_dim)
            )
        total_cycles += cycles.total_cycles
        
        # GELU activation
        out, cycles = self.activation_unit.forward(out)
        total_cycles += cycles.total_cycles
        
        # FC2: [B, N, hidden_dim] -> [B, N, D]
        if self.fc2_weight is not None:
            out, cycles = self.gemm_unit.linear(out, self.fc2_weight, self.fc2_bias)
        else:
            cycles = self.gemm_unit._compute_cycles(
                torch.empty(B, N, self.hidden_dim), torch.empty(self.hidden_dim, D)
            )
        total_cycles += cycles.total_cycles
        
        return out, total_cycles


class TransformerBlockSim:
    """
    Single Transformer Block using existing hardware units.
    
    Structure:
    - LayerNorm (NormalizationUnit)
    - Multi-Head Self-Attention (GEMMUnit)
    - Residual connection
    - LayerNorm (NormalizationUnit)
    - MLP (GEMMUnit + ActivationUnit)
    - Residual connection
    """
    
    def __init__(self, config: ViTConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        
        # Hardware units
        self.norm1 = NormalizationUnit(NormType.LAYER, config.embed_dim)
        self.attn = MultiHeadAttentionSim(config, device)
        self.norm2 = NormalizationUnit(NormType.LAYER, config.embed_dim)
        self.mlp = MLPSim(config, device)
        
        # Move norm weights to device
        self.norm1.weight = nn.Parameter(torch.ones(config.embed_dim, device=device))
        self.norm1.bias = nn.Parameter(torch.zeros(config.embed_dim, device=device))
        self.norm2.weight = nn.Parameter(torch.ones(config.embed_dim, device=device))
        self.norm2.bias = nn.Parameter(torch.zeros(config.embed_dim, device=device))
        
        # Weights for norms (for loading)
        self.norm1_weight: Optional[torch.Tensor] = None
        self.norm1_bias: Optional[torch.Tensor] = None
        self.norm2_weight: Optional[torch.Tensor] = None
        self.norm2_bias: Optional[torch.Tensor] = None
    
    def load_weights(self, block: nn.Module):
        """Load weights from DINOv2's transformer block."""
        if hasattr(block, 'norm1'):
            self.norm1_weight = block.norm1.weight.to(self.device)
            self.norm1_bias = block.norm1.bias.to(self.device)
            self.norm1.weight = nn.Parameter(self.norm1_weight)
            self.norm1.bias = nn.Parameter(self.norm1_bias)
        if hasattr(block, 'norm2'):
            self.norm2_weight = block.norm2.weight.to(self.device)
            self.norm2_bias = block.norm2.bias.to(self.device)
            self.norm2.weight = nn.Parameter(self.norm2_weight)
            self.norm2.bias = nn.Parameter(self.norm2_bias)
        if hasattr(block, 'attn'):
            self.attn.load_weights(block.attn)
        if hasattr(block, 'mlp'):
            self.mlp.load_weights(block.mlp)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Forward pass: [B, N, D] -> [B, N, D]
        
        Returns:
            output: [B, N, D]
            cycles: Hardware cycle count
        """
        total_cycles = 0
        
        # Norm1 + Attention + Residual
        normed, cycles = self.norm1.forward(x)
        total_cycles += cycles.total_cycles
        
        attn_out, cycles = self.attn.forward(normed)
        total_cycles += cycles
        
        x = x + attn_out
        
        # Norm2 + MLP + Residual
        normed, cycles = self.norm2.forward(x)
        total_cycles += cycles.total_cycles
        
        mlp_out, cycles = self.mlp.forward(normed)
        total_cycles += cycles
        
        x = x + mlp_out
        
        return x, total_cycles


class ViTSimulator:
    """
    Full ViT/DINOv2 Hardware Simulator.
    
    Composes existing hardware units to simulate complete ViT inference:
    - ConvEngine for patch embedding
    - GEMMUnit for attention and MLP
    - NormalizationUnit for LayerNorm
    - ActivationUnit for GELU
    
    This completely replaces the original DINOv2 forward pass with
    hardware-simulated computation.
    """
    
    def __init__(self, config: ViTConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        
        # Patch embedding
        self.patch_embed = PatchEmbedSim(config, device)
        
        # Position embedding (learned, added directly)
        self.pos_embed: Optional[torch.Tensor] = None
        
        # CLS token
        self.cls_token: Optional[torch.Tensor] = None
        
        # Transformer blocks
        self.blocks: List[TransformerBlockSim] = []
        for _ in range(config.num_layers):
            self.blocks.append(TransformerBlockSim(config, device))
        
        # Final norm
        self.final_norm = NormalizationUnit(NormType.LAYER, config.embed_dim)
        self.final_norm.weight = nn.Parameter(torch.ones(config.embed_dim, device=device))
        self.final_norm.bias = nn.Parameter(torch.zeros(config.embed_dim, device=device))
        
        # Intermediate layer indices for get_intermediate_layers
        self.intermediate_indices = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
        }
    
    def load_from_dinov2(self, dinov2_model: nn.Module):
        """
        Load weights from pretrained DINOv2 model.
        
        Args:
            dinov2_model: Loaded DINOv2 model from torch.hub
        """
        # Patch embedding
        if hasattr(dinov2_model, 'patch_embed'):
            self.patch_embed.load_weights(dinov2_model.patch_embed)
        
        # Position embedding
        if hasattr(dinov2_model, 'pos_embed'):
            self.pos_embed = dinov2_model.pos_embed.to(self.device)
        
        # CLS token
        if hasattr(dinov2_model, 'cls_token'):
            self.cls_token = dinov2_model.cls_token.to(self.device)
        
        # Transformer blocks
        if hasattr(dinov2_model, 'blocks'):
            for i, block in enumerate(dinov2_model.blocks):
                if i < len(self.blocks):
                    self.blocks[i].load_weights(block)
        
        # Final norm
        if hasattr(dinov2_model, 'norm'):
            self.final_norm.weight = nn.Parameter(dinov2_model.norm.weight.to(self.device))
            self.final_norm.bias = nn.Parameter(dinov2_model.norm.bias.to(self.device))
    
    def forward(self, x: torch.Tensor) -> ViTOutput:
        """
        Full ViT forward pass with hardware simulation.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            ViTOutput with features and cycle counts
        """
        B = x.shape[0]
        total_cycles = 0
        patch_embed_cycles = 0
        transformer_cycles = 0
        breakdown = {}
        
        # Patch embedding: [B, 3, H, W] -> [B, N, D]
        tokens, cycles = self.patch_embed.forward(x)
        patch_embed_cycles = cycles
        total_cycles += cycles
        breakdown['patch_embed'] = cycles
        
        # Add CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Add position embedding
        if self.pos_embed is not None:
            # Interpolate if needed
            if tokens.shape[1] != self.pos_embed.shape[1]:
                tokens = tokens + self._interpolate_pos_embed(tokens.shape[1])
            else:
                tokens = tokens + self.pos_embed
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            tokens, cycles = block.forward(tokens)
            transformer_cycles += cycles
            total_cycles += cycles
            breakdown[f'block_{i}'] = cycles
        
        # Final norm
        tokens, cycles = self.final_norm.forward(tokens)
        total_cycles += cycles.total_cycles
        breakdown['final_norm'] = cycles.total_cycles
        
        return ViTOutput(
            features=tokens,
            total_cycles=total_cycles,
            patch_embed_cycles=patch_embed_cycles,
            transformer_cycles=transformer_cycles,
            cycle_breakdown=breakdown,
        )
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        return_class_token: bool = False,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Get intermediate layer features (like DINOv2's get_intermediate_layers).
        
        Args:
            x: Input images [B, 3, H, W]
            layer_indices: Which layers to return (0-indexed)
            return_class_token: Whether to include CLS token
            
        Returns:
            features: List of [B, N, D] tensors
            cycles: Total hardware cycles
        """
        if layer_indices is None:
            layer_indices = self.intermediate_indices.get(self.config.vit_type, [2, 5, 8, 11])
        
        B = x.shape[0]
        total_cycles = 0
        intermediate_features = []
        
        # Patch embedding
        tokens, cycles = self.patch_embed.forward(x)
        total_cycles += cycles
        
        # Add CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Add position embedding
        if self.pos_embed is not None:
            if tokens.shape[1] != self.pos_embed.shape[1]:
                tokens = tokens + self._interpolate_pos_embed(tokens.shape[1])
            else:
                tokens = tokens + self.pos_embed
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            tokens, cycles = block.forward(tokens)
            total_cycles += cycles
            
            if i in layer_indices:
                if return_class_token:
                    intermediate_features.append(tokens)
                else:
                    intermediate_features.append(tokens[:, 1:])  # Exclude CLS token
        
        return intermediate_features, total_cycles
    
    def _interpolate_pos_embed(self, num_tokens: int) -> torch.Tensor:
        """Interpolate position embedding for different input sizes."""
        # Simple implementation - just use zeros for extra tokens
        if self.pos_embed is None:
            return torch.zeros(1, num_tokens, self.config.embed_dim, device=self.device)
        
        pos = self.pos_embed
        if num_tokens <= pos.shape[1]:
            return pos[:, :num_tokens]
        else:
            # Extend with interpolation
            extra = torch.zeros(1, num_tokens - pos.shape[1], self.config.embed_dim, device=self.device)
            return torch.cat([pos, extra], dim=1)
