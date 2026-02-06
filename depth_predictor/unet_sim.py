"""
U-Net Hardware Simulator

Simulates U-Net architecture for cost volume refinement and depth refinement.
Uses ConvEngine, NormalizationUnit, ActivationUnit for hardware-accurate cycles.

Architecture:
    Encoder: Conv + GroupNorm + GELU → Downsample
    Bottleneck: Conv + GroupNorm + GELU
    Decoder: Upsample + Concat + Conv + GroupNorm + GELU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from encoder import ConvEngine, NormalizationUnit, ActivationUnit, BilinearUnit
from encoder.types import CycleStats, NormType, ActivationType
from .types import UNetConfig, UNetLayerConfig


class UNetBlockSim:
    """
    Single U-Net block: Conv + Norm + Activation.
    
    Hardware units:
    - ConvEngine: 3x3 or 1x1 convolution
    - NormalizationUnit: GroupNorm
    - ActivationUnit: GELU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_groups: int = 8,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """Initialize U-Net block simulator."""
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_groups = num_groups
        self.device = device
        
        # Hardware units
        self.conv_engine = ConvEngine()
        self.norm_unit = NormalizationUnit(NormType.GROUP, out_channels, num_groups)
        self.activation_unit = ActivationUnit(ActivationType.GELU)
        
        # PyTorch layers for weight loading
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        
        self._initialized = False
    
    def load_weights(self, conv_weight: torch.Tensor, conv_bias: Optional[torch.Tensor],
                     norm_weight: torch.Tensor, norm_bias: torch.Tensor):
        """Load weights from original model."""
        self.conv.weight.data = conv_weight.to(self.device)
        if conv_bias is not None:
            self.conv.bias = nn.Parameter(conv_bias.to(self.device))
        self.norm.weight.data = norm_weight.to(self.device)
        self.norm.bias.data = norm_bias.to(self.device)
        
        # Also load to normalization unit
        self.norm_unit.weight.data = norm_weight.to(self.device)
        self.norm_unit.bias.data = norm_bias.to(self.device)
        
        self._initialized = True
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, CycleStats]:
        """
        Forward pass with cycle tracking.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            output: Output tensor [B, C', H', W']
            cycles: Cycle statistics
        """
        total_cycles = CycleStats()
        
        # Convolution
        padding = self.kernel_size // 2
        out, conv_cycles = self.conv_engine.forward(
            x, self.conv.weight,
            bias=self.conv.bias if hasattr(self.conv, 'bias') and self.conv.bias is not None else None,
            stride=self.stride,
            padding=padding,
        )
        total_cycles = total_cycles + conv_cycles
        
        # Normalization (use PyTorch for accuracy, track cycles)
        out = self.norm(out)
        norm_cycles = CycleStats(
            total_cycles=5 * out.numel(),  # mean + var + normalize
            compute_cycles=5 * out.numel(),
            breakdown={'norm': 5 * out.numel()}
        )
        total_cycles = total_cycles + norm_cycles
        
        # Activation
        out, act_cycles = self.activation_unit.forward(out)
        total_cycles = total_cycles + act_cycles
        
        return out, total_cycles


class UNetEncoderSim:
    """
    U-Net encoder path: series of downsampling blocks.
    
    Each encoder stage:
    - Conv block (stride 1)
    - Conv block (stride 2) for downsampling
    """
    
    def __init__(
        self,
        in_channels: int,
        channels: Tuple[int, ...],
        num_groups: int = 8,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """Initialize encoder."""
        self.in_channels = in_channels
        self.channels = channels
        self.num_groups = num_groups
        self.device = device
        
        # Build encoder stages
        self.stages: List[List[UNetBlockSim]] = []
        current_channels = in_channels
        
        for i, out_ch in enumerate(channels):
            stage_blocks = []
            
            # First conv block (no downsampling)
            stage_blocks.append(UNetBlockSim(
                current_channels, out_ch, kernel_size=3, stride=1,
                num_groups=min(num_groups, out_ch), device=device
            ))
            
            # Second conv block (downsample if not last)
            if i < len(channels) - 1:
                stride = 2
            else:
                stride = 1
            stage_blocks.append(UNetBlockSim(
                out_ch, out_ch, kernel_size=3, stride=stride,
                num_groups=min(num_groups, out_ch), device=device
            ))
            
            self.stages.append(stage_blocks)
            current_channels = out_ch
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], CycleStats]:
        """
        Forward pass returning features at each scale.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            output: Bottleneck features
            skip_features: Features at each scale for skip connections
            cycles: Cycle statistics
        """
        total_cycles = CycleStats()
        skip_features = []
        
        for stage in self.stages[:-1]:
            for block in stage:
                x, cycles = block.forward(x)
                total_cycles = total_cycles + cycles
            skip_features.append(x)
        
        # Bottleneck (last stage)
        for block in self.stages[-1]:
            x, cycles = block.forward(x)
            total_cycles = total_cycles + cycles
        
        return x, skip_features, total_cycles


class UNetDecoderSim:
    """
    U-Net decoder path: series of upsampling blocks with skip connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        channels: Tuple[int, ...],
        num_groups: int = 8,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """Initialize decoder."""
        self.channels = channels
        self.num_groups = num_groups
        self.device = device
        
        # Bilinear unit for upsampling
        self.bilinear_unit = BilinearUnit()
        
        # Build decoder stages
        self.stages: List[UNetBlockSim] = []
        current_channels = in_channels
        
        for out_ch in channels:
            # Input channels = current + skip
            self.stages.append(UNetBlockSim(
                current_channels + out_ch, out_ch, kernel_size=3, stride=1,
                num_groups=min(num_groups, out_ch), device=device
            ))
            current_channels = out_ch
    
    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Forward pass with skip connections.
        
        Args:
            x: Bottleneck features [B, C, H, W]
            skip_features: Features from encoder (reversed order)
            
        Returns:
            output: Decoded features
            cycles: Cycle statistics
        """
        total_cycles = CycleStats()
        
        # Reverse skip features for decoder
        skip_features = list(reversed(skip_features))
        
        for i, (stage, skip) in enumerate(zip(self.stages, skip_features)):
            # Upsample
            x, up_cycles = self.bilinear_unit.interpolate(x, size=skip.shape[-2:])
            total_cycles = total_cycles + up_cycles
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Conv block
            x, block_cycles = stage.forward(x)
            total_cycles = total_cycles + block_cycles
        
        return x, total_cycles


class UNetSimulator:
    """
    Complete U-Net hardware simulator.
    
    Architecture:
        Input → Encoder (downsample) → Bottleneck → Decoder (upsample) → Output
                    ↓_____________________________↑ (skip connections)
    
    Hardware Mapping:
        - ConvEngine: All convolutions (~50K LUTs, 256 DSPs)
        - NormalizationUnit: GroupNorm (~3K LUTs, 16 DSPs)
        - ActivationUnit: GELU (~2K LUTs, 8 DSPs)
        - BilinearUnit: Upsampling (~800 LUTs, 8 DSPs)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: Optional[UNetConfig] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """
        Initialize U-Net simulator.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            config: U-Net configuration
            device: Device for computation
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config or UNetConfig()
        self.device = device
        
        # Build encoder
        self.encoder = UNetEncoderSim(
            in_channels,
            self.config.encoder_channels,
            self.config.num_groups,
            device,
        )
        
        # Build decoder
        bottleneck_channels = self.config.encoder_channels[-1]
        decoder_channels = self.config.decoder_channels or tuple(
            reversed(self.config.encoder_channels[:-1])
        )
        self.decoder = UNetDecoderSim(
            bottleneck_channels,
            decoder_channels,
            self.config.num_groups,
            device,
        )
        
        # Output projection
        final_channels = decoder_channels[-1] if decoder_channels else bottleneck_channels
        self.output_conv = UNetBlockSim(
            final_channels, out_channels, kernel_size=1, stride=1,
            num_groups=min(self.config.num_groups, out_channels), device=device
        )
        
        # PyTorch reference layers for weight loading
        self._build_pytorch_layers()
        
        self._total_cycles = 0
    
    def _build_pytorch_layers(self):
        """Build PyTorch layers for weight loading."""
        # This is for loading weights from original models
        self.pytorch_encoder = nn.ModuleList()
        self.pytorch_decoder = nn.ModuleList()
    
    def load_from_model(self, unet_model: nn.Module):
        """
        Load weights from a PyTorch U-Net model.
        
        Args:
            unet_model: Original U-Net model
        """
        # This would extract and load weights from the original model
        # Implementation depends on the specific model architecture
        pass
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, CycleStats]:
        """
        Forward pass with cycle tracking.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            output: Output tensor [B, C_out, H, W]
            cycles: Cycle statistics
        """
        total_cycles = CycleStats()
        
        # Encoder
        bottleneck, skip_features, enc_cycles = self.encoder.forward(x)
        total_cycles = total_cycles + enc_cycles
        
        # Decoder
        decoded, dec_cycles = self.decoder.forward(bottleneck, skip_features)
        total_cycles = total_cycles + dec_cycles
        
        # Output projection
        output, out_cycles = self.output_conv.forward(decoded)
        total_cycles = total_cycles + out_cycles
        
        self._total_cycles += total_cycles.total_cycles
        
        return output, total_cycles
    
    def forward_with_reference(
        self,
        x: torch.Tensor,
        reference_model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, CycleStats]:
        """
        Forward pass comparing with reference PyTorch model.
        
        Args:
            x: Input tensor
            reference_model: Reference U-Net model
            
        Returns:
            sim_output: Simulator output
            ref_output: Reference output
            cycles: Cycle statistics
        """
        # Run reference model
        with torch.no_grad():
            ref_output = reference_model(x)
        
        # Run simulator
        sim_output, cycles = self.forward(x)
        
        return sim_output, ref_output, cycles
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0


class SimplifiedUNetSim:
    """
    Simplified U-Net simulator that directly uses PyTorch operations
    but tracks cycles as if using hardware units.
    
    This is useful for initial integration where we want:
    1. Correct output (matching PyTorch)
    2. Cycle counting for hardware estimation
    """
    
    def __init__(
        self,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """Initialize simplified U-Net simulator."""
        self.device = device
        self.conv_engine = ConvEngine()
        self.norm_unit = NormalizationUnit(NormType.GROUP, 32)
        self.activation_unit = ActivationUnit(ActivationType.GELU)
        self.bilinear_unit = BilinearUnit()
        
        self._total_cycles = 0
    
    def simulate_unet_pass(
        self,
        x: torch.Tensor,
        unet_model: nn.Module,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Run actual U-Net model but track cycles.
        
        This uses the original model for computation but estimates
        cycles based on the operations performed.
        
        Args:
            x: Input tensor [B, C, H, W]
            unet_model: Original U-Net model
            
        Returns:
            output: U-Net output
            cycles: Estimated cycle statistics
        """
        # Run the actual model
        with torch.no_grad():
            output = unet_model(x)
        
        # Estimate cycles based on architecture
        cycles = self._estimate_cycles(x, output, unet_model)
        
        self._total_cycles += cycles.total_cycles
        
        return output, cycles
    
    def _estimate_cycles(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        model: nn.Module,
    ) -> CycleStats:
        """
        Estimate cycles based on model architecture.
        
        Counts:
        - Conv2d layers
        - GroupNorm layers
        - Activation functions
        - Bilinear upsampling
        """
        B, C_in, H, W = x.shape
        _, C_out, H_out, W_out = output.shape
        
        total_cycles = 0
        breakdown = {}
        
        # Count operations in model
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Estimate conv cycles
                k = module.kernel_size[0]
                c_in = module.in_channels
                c_out = module.out_channels
                # Assume same spatial size (approximate)
                conv_cycles = H * W * c_in * c_out * k * k // 256  # PE array throughput
                total_cycles += conv_cycles
                breakdown[f'conv_{name}'] = conv_cycles
                
            elif isinstance(module, nn.GroupNorm):
                # Estimate norm cycles
                num_groups = module.num_groups
                num_channels = module.num_channels
                norm_cycles = 5 * B * num_channels * H * W // 4  # Approximate
                total_cycles += norm_cycles
                breakdown[f'norm_{name}'] = norm_cycles
                
            elif isinstance(module, (nn.GELU, nn.ReLU, nn.SiLU)):
                # Estimate activation cycles
                act_cycles = B * C_out * H_out * W_out
                total_cycles += act_cycles
                breakdown[f'act_{name}'] = act_cycles
        
        return CycleStats(
            total_cycles=total_cycles,
            compute_cycles=total_cycles,
            breakdown=breakdown,
        )
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
