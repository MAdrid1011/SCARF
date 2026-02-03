"""
Bilinear Interpolation Unit

Hardware simulator for bilinear upsampling.
Supports 2x/4x upsampling and arbitrary size interpolation.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union

from .types import (
    BilinearConfig,
    CycleStats,
    ResourceEstimate,
    ENCODER_CYCLES,
    ENCODER_RESOURCES,
)


class BilinearUnit:
    """
    Bilinear Interpolation Unit for feature upsampling.
    
    Performs 2D bilinear interpolation:
    - Coordinate calculation
    - 4-point sampling
    - Weighted average
    
    Supports:
    - Scale factor based upsampling (2x, 4x)
    - Target size based interpolation
    - align_corners mode
    """
    
    def __init__(self, config: Optional[BilinearConfig] = None):
        """
        Initialize bilinear unit.
        
        Args:
            config: Bilinear configuration
        """
        self.config = config or BilinearConfig()
        self._total_cycles = 0
    
    def interpolate(
        self,
        x: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[Union[int, float]] = None,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Perform bilinear interpolation with cycle tracking.
        
        Args:
            x: Input tensor [B, C, H, W]
            size: Target output size (H_out, W_out)
            scale_factor: Scale factor for upsampling
            
        Returns:
            output: Interpolated tensor [B, C, H_out, W_out]
            cycles: Cycle statistics
        """
        if size is None and scale_factor is None:
            raise ValueError("Either size or scale_factor must be provided")
        
        # Compute output using PyTorch
        output = F.interpolate(
            x, 
            size=size,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.config.align_corners,
        )
        
        # Calculate cycle count
        cycles = self._compute_cycles(x, output)
        self._total_cycles += cycles.total_cycles
        
        return output, cycles
    
    def _compute_cycles(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> CycleStats:
        """
        Compute cycle count for bilinear interpolation.
        
        Per output pixel:
        - Coordinate calculation: 1 cycle
        - 4-point sampling: 1 cycle (parallel)
        - Interpolation: 2 cycles (4 MACs)
        Total: 4 cycles per output pixel
        
        With parallel channel processing:
        - Process up to parallel_channels in parallel
        """
        B, C, H_out, W_out = output.shape
        
        # Cycles per output pixel (per channel batch)
        cycles_per_pixel = (
            ENCODER_CYCLES['bilinear_coord'] +
            ENCODER_CYCLES['bilinear_sample'] +
            ENCODER_CYCLES['bilinear_interpolate']
        )
        
        # Channel batches
        parallel_channels = self.config.parallel_channels
        channel_batches = (C + parallel_channels - 1) // parallel_channels
        
        # Total cycles
        output_pixels = B * H_out * W_out
        compute_cycles = output_pixels * channel_batches * cycles_per_pixel
        
        return CycleStats(
            total_cycles=compute_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=0,
            breakdown={
                'interpolate': compute_cycles,
                'output_pixels': output_pixels,
                'channel_batches': channel_batches,
            }
        )
    
    def get_resource_estimate(self) -> ResourceEstimate:
        """Get hardware resource estimate."""
        return ENCODER_RESOURCES['bilinear_unit']
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
