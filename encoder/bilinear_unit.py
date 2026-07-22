"""
Interpolation Unit

Hardware simulator for image resampling / upsampling.
Supports bilinear, nearest-neighbor, and bicubic interpolation modes.
Also supports grid_sample for warping operations.
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
    Interpolation Unit for feature resampling.
    
    Supports multiple interpolation modes:
    - nearest: Nearest-neighbor (cheapest, 1 cycle per pixel)
    - bilinear: 4-point weighted average (4 cycles per pixel)
    - bicubic: 16-point weighted average (16 cycles per pixel)
    
    Also supports grid_sample for spatial transformer / warping:
    - bilinear grid sampling with border handling
    """
    
    def __init__(self, config: Optional[BilinearConfig] = None):
        """
        Initialize interpolation unit.
        
        Args:
            config: Bilinear configuration
        """
        self.config = config or BilinearConfig()
        self._total_cycles = 0
    
    def interpolate(
        self,
        x: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[Union[int, float, Tuple[float, float]]] = None,
        mode: str = 'bilinear',
        align_corners: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Perform interpolation with cycle tracking.
        
        Args:
            x: Input tensor [B, C, H, W]
            size: Target output size (H_out, W_out)
            scale_factor: Scale factor for upsampling
            mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
            align_corners: Override default align_corners setting
            
        Returns:
            output: Interpolated tensor [B, C, H_out, W_out]
            cycles: Cycle statistics
        """
        if size is None and scale_factor is None:
            raise ValueError("Either size or scale_factor must be provided")
        
        ac = align_corners if align_corners is not None else self.config.align_corners
        
        # Compute output using PyTorch
        if mode == 'nearest':
            output = F.interpolate(x, size=size, scale_factor=scale_factor, mode='nearest')
        elif mode == 'bilinear':
            output = F.interpolate(x, size=size, scale_factor=scale_factor,
                                   mode='bilinear', align_corners=ac)
        elif mode == 'bicubic':
            output = F.interpolate(x, size=size, scale_factor=scale_factor,
                                   mode='bicubic', align_corners=ac)
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")
        
        # Calculate cycle count
        cycles = self._compute_cycles(x, output, mode)
        self._total_cycles += cycles.total_cycles
        
        return output, cycles
    
    def grid_sample(
        self,
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Perform grid-based sampling with cycle tracking.
        
        Hardware implementation: coordinate unit + bilinear sampler with border logic.
        
        Args:
            input: Input tensor [B, C, H_in, W_in]
            grid: Sampling grid [B, H_out, W_out, 2] with (x, y) coordinates in [-1, 1]
            mode: Sampling mode ('bilinear' or 'nearest')
            padding_mode: Border handling ('zeros', 'border', 'reflection')
            align_corners: Override default align_corners
            
        Returns:
            output: Sampled tensor [B, C, H_out, W_out]
            cycles: Cycle statistics
        """
        ac = align_corners if align_corners is not None else self.config.align_corners
        
        output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                               align_corners=ac)
        
        B, C, H_out, W_out = output.shape
        parallel_channels = self.config.parallel_channels
        channel_batches = (C + parallel_channels - 1) // parallel_channels
        output_pixels = B * H_out * W_out
        
        if mode == 'nearest':
            # Coordinate denormalization + nearest lookup
            cycles_per_pixel = ENCODER_CYCLES['bilinear_coord'] + 1
        else:
            # Coordinate denormalization + 4-point sample + interpolate + border check
            cycles_per_pixel = (
                ENCODER_CYCLES['bilinear_coord'] +
                ENCODER_CYCLES['bilinear_sample'] +
                ENCODER_CYCLES['bilinear_interpolate'] +
                1  # border check overhead
            )
        
        compute_cycles = output_pixels * channel_batches * cycles_per_pixel
        
        cycles = CycleStats(
            total_cycles=compute_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=0,
            breakdown={
                'grid_sample': compute_cycles,
                'output_pixels': output_pixels,
                'channel_batches': channel_batches,
                'mode': mode,
            }
        )
        self._total_cycles += cycles.total_cycles
        return output, cycles
    
    def _compute_cycles(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        mode: str = 'bilinear',
    ) -> CycleStats:
        """
        Compute cycle count for interpolation.
        
        Nearest:  1 cycle per output pixel (just address compute + copy)
        Bilinear: 4 cycles per output pixel (coord + 4-sample + interp)
        Bicubic: 16 cycles per output pixel (coord + 16-sample + interp)
        """
        B, C, H_out, W_out = output.shape
        
        parallel_channels = self.config.parallel_channels
        channel_batches = (C + parallel_channels - 1) // parallel_channels
        output_pixels = B * H_out * W_out
        
        if mode == 'nearest':
            cycles_per_pixel = 1  # floor + copy
        elif mode == 'bilinear':
            cycles_per_pixel = (
                ENCODER_CYCLES['bilinear_coord'] +
                ENCODER_CYCLES['bilinear_sample'] +
                ENCODER_CYCLES['bilinear_interpolate']
            )
        elif mode == 'bicubic':
            # 16-point sample + 16 weight multiplications + accumulation
            cycles_per_pixel = (
                ENCODER_CYCLES['bilinear_coord'] +
                4 * ENCODER_CYCLES['bilinear_sample'] +
                4 * ENCODER_CYCLES['bilinear_interpolate']
            )
        else:
            raise ValueError(f"unsupported interpolation mode: {mode}")
        
        compute_cycles = output_pixels * channel_batches * cycles_per_pixel
        
        return CycleStats(
            total_cycles=compute_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=0,
            breakdown={
                'interpolate': compute_cycles,
                'output_pixels': output_pixels,
                'channel_batches': channel_batches,
                'mode': mode,
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
