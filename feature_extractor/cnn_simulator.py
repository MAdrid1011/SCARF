"""
CNN Encoder Hardware Simulator

Simulates CNN backbone using SCARF encoder compute units.
Produces bit-accurate output matching original PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass

import sys
from pathlib import Path
SCARF_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from encoder import ConvEngine, NormalizationUnit, ActivationUnit
from encoder.types import EncoderConfig, NormType, ActivationType
from .types import CNNConfig


@dataclass
class LayerCycles:
    """Cycle tracking for a layer."""
    conv: int = 0
    norm: int = 0
    activation: int = 0
    
    @property
    def total(self) -> int:
        return self.conv + self.norm + self.activation


class Conv7x7LayerSim:
    """
    Simulates Conv7x7 + InstanceNorm + ReLU layer.
    
    Uses ConvEngine, NormalizationUnit, ActivationUnit to match
    PyTorch Conv2d(7,7,s=2,p=3) + InstanceNorm2d + ReLU exactly.
    """
    
    def __init__(self, in_channels: int, out_channels: int, device: torch.device = None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device or torch.device('cpu')
        
        # Hardware units
        config = EncoderConfig()
        self.conv_engine = ConvEngine(config.conv)
        self.norm_unit = NormalizationUnit(NormType.INSTANCE, out_channels, config)
        self.activation_unit = ActivationUnit(ActivationType.RELU, config)
        
        # Weights (loaded from PyTorch model)
        self.conv_weight: Optional[torch.Tensor] = None
        self.norm_weight: Optional[torch.Tensor] = None
        self.norm_bias: Optional[torch.Tensor] = None
        self.norm_running_mean: Optional[torch.Tensor] = None
        self.norm_running_var: Optional[torch.Tensor] = None
    
    def load_weights(self, conv_weight: torch.Tensor, norm_layer: nn.Module):
        """Load weights from PyTorch modules."""
        self.conv_weight = conv_weight.to(self.device)
        if hasattr(norm_layer, 'weight') and norm_layer.weight is not None:
            self.norm_weight = norm_layer.weight.to(self.device)
        if hasattr(norm_layer, 'bias') and norm_layer.bias is not None:
            self.norm_bias = norm_layer.bias.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Forward pass with cycle counting.
        
        Uses PyTorch ops internally but counts cycles as if using hardware.
        This ensures bit-accurate output while tracking hardware cost.
        """
        cycles = LayerCycles()
        
        # Conv7x7 stride=2 padding=3 — ConvEngine
        out, conv_cycles = self.conv_engine.forward(x, self.conv_weight, stride=2, padding=3)
        cycles.conv = conv_cycles.total_cycles
        
        # InstanceNorm — NormalizationUnit
        self.norm_unit.dim = out.shape[1]
        self.norm_unit.weight = self.norm_weight
        self.norm_unit.bias = self.norm_bias
        out, norm_cycles = self.norm_unit.forward(out)
        cycles.norm = norm_cycles.total_cycles
        
        # ReLU — ActivationUnit
        out, relu_cycles = self.activation_unit.forward(out)
        cycles.activation = relu_cycles.total_cycles
        
        return out, cycles.total


class ResidualBlockSim:
    """
    Simulates ResidualBlock using hardware units.
    
    Structure: Conv3x3 -> Norm -> ReLU -> Conv3x3 -> Norm -> (+ residual) -> ReLU
    """
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, 
                 dilation: int = 1, device: torch.device = None):
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.device = device or torch.device('cpu')
        
        # Hardware units
        config = EncoderConfig()
        self.conv_engine = ConvEngine(config.conv)
        self.norm_unit = NormalizationUnit(NormType.INSTANCE, planes, config)
        self.activation_unit = ActivationUnit(ActivationType.RELU, config)
        
        # Weights
        self.conv1_weight: Optional[torch.Tensor] = None
        self.conv2_weight: Optional[torch.Tensor] = None
        self.norm1: Optional[nn.Module] = None
        self.norm2: Optional[nn.Module] = None
        self.downsample_conv: Optional[torch.Tensor] = None
        self.downsample_norm: Optional[nn.Module] = None
        self.has_downsample = stride != 1 or in_planes != planes
    
    def load_from_pytorch(self, block: nn.Module):
        """Load weights from PyTorch ResidualBlock."""
        self.conv1_weight = block.conv1.weight.to(self.device)
        self.conv2_weight = block.conv2.weight.to(self.device)
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        
        if block.downsample is not None:
            self.downsample_conv = block.downsample[0].weight.to(self.device)
            self.downsample_norm = block.downsample[1]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Forward pass with cycle counting."""
        cycles = LayerCycles()
        
        identity = x
        
        # Conv1 — ConvEngine
        y, conv1_c = self.conv_engine.forward(x, self.conv1_weight, stride=self.stride,
                                               padding=self.dilation, dilation=self.dilation)
        cycles.conv += conv1_c.total_cycles
        
        y = self.norm1(y)
        cycles.norm += 5 * y.numel()
        
        # ReLU — ActivationUnit
        y, relu_c = self.activation_unit.forward(y)
        cycles.activation += relu_c.total_cycles
        
        # Conv2 — ConvEngine
        y, conv2_c = self.conv_engine.forward(y, self.conv2_weight, padding=self.dilation,
                                               dilation=self.dilation)
        cycles.conv += conv2_c.total_cycles
        
        y = self.norm2(y)
        cycles.norm += 5 * y.numel()
        
        # ReLU — ActivationUnit
        y, relu_c = self.activation_unit.forward(y)
        cycles.activation += relu_c.total_cycles
        
        # Downsample residual if needed
        if self.has_downsample and self.downsample_conv is not None:
            identity, ds_c = self.conv_engine.forward(identity, self.downsample_conv, stride=self.stride)
            cycles.conv += ds_c.total_cycles
            
            identity = self.downsample_norm(identity)
            cycles.norm += 5 * identity.numel()
        
        # Add residual and final ReLU — ActivationUnit
        out, relu_c = self.activation_unit.forward(identity + y)
        cycles.activation += relu_c.total_cycles
        
        return out, cycles.total


class CNNEncoderSimulator:
    """
    Full CNN Encoder hardware simulator.
    
    Matches Transplat's CNNEncoder exactly while counting hardware cycles.
    """
    
    def __init__(self, config: CNNConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        
        # Layers (initialized when weights loaded)
        self.conv1_sim: Optional[Conv7x7LayerSim] = None
        self.layer1: List[ResidualBlockSim] = []
        self.layer2: List[ResidualBlockSim] = []
        self.layer3: List[ResidualBlockSim] = []
        self.conv2_weight: Optional[torch.Tensor] = None
        
        # Hardware units for final conv
        enc_config = EncoderConfig()
        self.conv_engine = ConvEngine(enc_config.conv)
    
    def load_from_pytorch(self, encoder: nn.Module):
        """Load weights from PyTorch CNNEncoder."""
        # Conv1 layer
        self.conv1_sim = Conv7x7LayerSim(3, self.config.feature_dims[0], self.device)
        self.conv1_sim.load_weights(encoder.conv1.weight, encoder.norm1)
        
        # Layer 1 (2 ResBlocks, 64 channels)
        self.layer1 = []
        for i, block in enumerate(encoder.layer1):
            sim = ResidualBlockSim(
                self.config.feature_dims[0] if i == 0 else self.config.feature_dims[0],
                self.config.feature_dims[0],
                stride=1,
                device=self.device
            )
            sim.load_from_pytorch(block)
            self.layer1.append(sim)
        
        # Layer 2 (2 ResBlocks, 96 channels, stride=2)
        self.layer2 = []
        for i, block in enumerate(encoder.layer2):
            sim = ResidualBlockSim(
                self.config.feature_dims[0] if i == 0 else self.config.feature_dims[1],
                self.config.feature_dims[1],
                stride=2 if i == 0 else 1,
                device=self.device
            )
            sim.load_from_pytorch(block)
            self.layer2.append(sim)
        
        # Layer 3 (2 ResBlocks, 128 channels, stride=2 if num_output_scales==1)
        stride3 = 2 if self.config.num_output_scales == 1 else 1
        self.layer3 = []
        for i, block in enumerate(encoder.layer3):
            sim = ResidualBlockSim(
                self.config.feature_dims[1] if i == 0 else self.config.feature_dims[2],
                self.config.feature_dims[2],
                stride=stride3 if i == 0 else 1,
                device=self.device
            )
            sim.load_from_pytorch(block)
            self.layer3.append(sim)
        
        # Final 1x1 conv (with bias!)
        self.conv2_weight = encoder.conv2.weight.to(self.device)
        self.conv2_bias = encoder.conv2.bias.to(self.device) if encoder.conv2.bias is not None else None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Forward pass matching CNNEncoder exactly.

        Returns:
            features: Output features
            total_cycles: Hardware cycle count (includes tile-layout writeback)
        """
        total_cycles = 0

        # Conv1 + Norm + ReLU
        x, cycles = self.conv1_sim.forward(x)
        total_cycles += cycles

        # Layer 1
        for block in self.layer1:
            x, cycles = block.forward(x)
            total_cycles += cycles

        # Layer 2
        for block in self.layer2:
            x, cycles = block.forward(x)
            total_cycles += cycles

        # Layer 3
        for block in self.layer3:
            x, cycles = block.forward(x)
            total_cycles += cycles

        # Final 1x1 conv (with bias) — ConvEngine
        out, conv_cycles = self.conv_engine.forward(x, self.conv2_weight, self.conv2_bias)
        total_cycles += conv_cycles.total_cycles

        # --- S1 Tile-Layout Writeback (Dataflow spec §Stage 1) ---
        # After feature extraction, the feature map is reorganised in SRAM so that
        # cross-view features belonging to the same spatial tile are laid out
        # contiguously (tile-aligned writeback).  This removes the need for S2 to
        # scatter-gather across views and is the key Stage 1 innovation.
        #
        # Cost model: each element in the output [B, C, H_out, W_out] is read once
        # and written once to the rearranged SRAM layout.  Two accesses per element,
        # one element per cycle (simple DMA-like rearrangement, no compute).
        #   writeback_cycles ≈ 2 × H_out × W_out × C
        _, C_out, H_out, W_out = out.shape
        tile_layout_writeback_cycles = 2 * H_out * W_out * C_out
        total_cycles += tile_layout_writeback_cycles

        return out, total_cycles
