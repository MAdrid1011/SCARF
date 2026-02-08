"""
Transformer Hardware Simulator

Simulates Transformer using SCARF encoder compute units (GEMMUnit).
Produces bit-accurate output matching original PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

import sys
from pathlib import Path
SCARF_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from encoder import GEMMUnit, NormalizationUnit, ActivationUnit
from encoder.softmax_unit import SoftmaxUnit
from encoder.types import EncoderConfig, NormType, ActivationType
from .types import TransformerConfig


class AttentionSim:
    """
    Simulates Multi-Head Attention using GEMMUnit.
    Uses PyTorch ops for correctness, counts cycles from hardware model.
    """
    
    def __init__(self, d_model: int, num_heads: int = 1, device: torch.device = None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.device = device or torch.device('cpu')
        
        config = EncoderConfig()
        self.gemm_unit = GEMMUnit(config.gemm)
        self.softmax_unit = SoftmaxUnit()
        
        # Weights
        self.in_proj_weight: Optional[torch.Tensor] = None
        self.in_proj_bias: Optional[torch.Tensor] = None
        self.out_proj_weight: Optional[torch.Tensor] = None
        self.out_proj_bias: Optional[torch.Tensor] = None
    
    def load_from_pytorch(self, attn: nn.MultiheadAttention):
        """Load weights from PyTorch MultiheadAttention."""
        self.in_proj_weight = attn.in_proj_weight.to(self.device)
        self.in_proj_bias = attn.in_proj_bias.to(self.device) if attn.in_proj_bias is not None else None
        self.out_proj_weight = attn.out_proj.weight.to(self.device)
        self.out_proj_bias = attn.out_proj.bias.to(self.device) if attn.out_proj.bias is not None else None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Self-attention forward with cycle counting."""
        B, N, C = x.shape
        cycles = 0
        
        # QKV projection: [B, N, C] @ [3C, C].T -> [B, N, 3C] — GEMMUnit
        qkv, c = self.gemm_unit.matmul(x.reshape(-1, C), self.in_proj_weight.T)
        cycles += c.total_cycles
        if self.in_proj_bias is not None:
            qkv = qkv + self.in_proj_bias.unsqueeze(0)
        qkv = qkv.reshape(B, N, -1)
        
        # Split Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Attention: softmax(QK^T / sqrt(d)) V — GEMMUnit
        scale = 1.0 / math.sqrt(self.head_dim)
        attn, c = self.gemm_unit.matmul(q.reshape(-1, C), k.reshape(-1, C).T)
        attn = attn.reshape(B, N, N) * scale
        cycles += c.total_cycles
        
        # Softmax — SoftmaxUnit
        attn, sm_cyc = self.softmax_unit.forward(attn, dim=-1)
        cycles += sm_cyc.total_cycles
        
        # Attn @ V — GEMMUnit
        out, c = self.gemm_unit.matmul(attn.reshape(-1, N), v.reshape(-1, C))
        out = out.reshape(B, N, C)
        cycles += c.total_cycles
        
        # Output projection — GEMMUnit
        out_flat, c = self.gemm_unit.matmul(out.reshape(-1, C), self.out_proj_weight.T)
        cycles += c.total_cycles
        if self.out_proj_bias is not None:
            out_flat = out_flat + self.out_proj_bias.unsqueeze(0)
        out = out_flat.reshape(B, N, C)
        
        return out, cycles


class FFNSim:
    """Simulates Feed-Forward Network using GEMMUnit."""
    
    def __init__(self, d_model: int, ffn_dim: int, device: torch.device = None):
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.device = device or torch.device('cpu')
        
        config = EncoderConfig()
        self.gemm_unit = GEMMUnit(config.gemm)
        self.activation = ActivationUnit(ActivationType.GELU, config)
        
        self.linear1_weight: Optional[torch.Tensor] = None
        self.linear1_bias: Optional[torch.Tensor] = None
        self.linear2_weight: Optional[torch.Tensor] = None
        self.linear2_bias: Optional[torch.Tensor] = None
    
    def load_from_pytorch(self, linear1: nn.Linear, linear2: nn.Linear):
        """Load weights from PyTorch Linear layers."""
        self.linear1_weight = linear1.weight.to(self.device)
        self.linear1_bias = linear1.bias.to(self.device) if linear1.bias is not None else None
        self.linear2_weight = linear2.weight.to(self.device)
        self.linear2_bias = linear2.bias.to(self.device) if linear2.bias is not None else None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """FFN forward with cycle counting."""
        cycles = 0
        
        # Linear1 — GEMMUnit
        h, c = self.gemm_unit.matmul(x.reshape(-1, self.d_model), self.linear1_weight.T)
        cycles += c.total_cycles
        if self.linear1_bias is not None:
            h = h + self.linear1_bias.unsqueeze(0)
        
        # GELU — ActivationUnit
        h, c = self.activation.forward(h)
        cycles += c.total_cycles
        
        # Linear2 — GEMMUnit
        out, c = self.gemm_unit.matmul(h.reshape(-1, self.ffn_dim), self.linear2_weight.T)
        cycles += c.total_cycles
        if self.linear2_bias is not None:
            out = out + self.linear2_bias.unsqueeze(0)
        
        return out, cycles


class TransformerLayerSim:
    """Simulates a single Transformer encoder layer."""
    
    def __init__(self, d_model: int, num_heads: int = 1, ffn_dim: int = 512,
                 device: torch.device = None):
        self.d_model = d_model
        self.device = device or torch.device('cpu')
        
        self.attn = AttentionSim(d_model, num_heads, device)
        self.ffn = FFNSim(d_model, ffn_dim, device)
        
        config = EncoderConfig()
        self.norm1 = NormalizationUnit(NormType.LAYER, d_model, config)
        self.norm2 = NormalizationUnit(NormType.LAYER, d_model, config)
        
        self.norm1_layer: Optional[nn.LayerNorm] = None
        self.norm2_layer: Optional[nn.LayerNorm] = None
    
    def load_from_pytorch(self, layer: nn.TransformerEncoderLayer):
        """Load from PyTorch TransformerEncoderLayer."""
        self.attn.load_from_pytorch(layer.self_attn)
        self.ffn.load_from_pytorch(layer.linear1, layer.linear2)
        self.norm1_layer = layer.norm1
        self.norm2_layer = layer.norm2
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Forward with cycle counting."""
        cycles = 0
        
        # Self-attention + residual
        attn_out, c = self.attn.forward(x)
        cycles += c
        x = x + attn_out
        x = self.norm1_layer(x)
        _, c = self.norm1.forward(x)
        cycles += c.total_cycles
        
        # FFN + residual
        ffn_out, c = self.ffn.forward(x)
        cycles += c
        x = x + ffn_out
        x = self.norm2_layer(x)
        _, c = self.norm2.forward(x)
        cycles += c.total_cycles
        
        return x, cycles


class TransformerSimulator:
    """
    Full Transformer simulator for feature extraction.
    Matches MultiViewFeatureTransformer behavior.
    """
    
    def __init__(self, config: TransformerConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        self.layers = []
    
    def load_from_pytorch(self, transformer: nn.Module):
        """Load from PyTorch transformer module."""
        # Handle different transformer architectures
        if hasattr(transformer, 'layers'):
            for layer in transformer.layers:
                sim_layer = TransformerLayerSim(
                    self.config.d_model,
                    self.config.num_heads,
                    self.config.ffn_dim,
                    self.device
                )
                # Adapt based on layer structure
                if hasattr(layer, 'self_attn'):
                    sim_layer.load_from_pytorch(layer)
                self.layers.append(sim_layer)
    
    def forward(self, features_list: list) -> Tuple[list, int]:
        """
        Forward pass matching MultiViewFeatureTransformer.
        
        Args:
            features_list: List of feature tensors per view
            
        Returns:
            Transformed features and total cycles
        """
        total_cycles = 0
        
        # Stack features for batch processing
        if len(features_list) > 0:
            x = torch.stack(features_list, dim=1)  # [B, V, C, H, W]
            B, V, C, H, W = x.shape
            
            # Reshape for transformer: [B*V, H*W, C]
            x = x.permute(0, 1, 3, 4, 2).reshape(B * V, H * W, C)
            
            # Apply transformer layers
            for layer in self.layers:
                x, cycles = layer.forward(x)
                total_cycles += cycles
            
            # Reshape back: [B, V, C, H, W]
            x = x.reshape(B, V, H, W, C).permute(0, 1, 4, 2, 3)
            
            # Split back to list
            features_list = [x[:, v] for v in range(V)]
        
        return features_list, total_cycles
