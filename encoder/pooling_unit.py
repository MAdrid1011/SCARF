"""
Pooling Unit

Hardware simulator for pooling operations.
Supports average pooling and max pooling with configurable kernel sizes and strides.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union

from .types import (
    CycleStats,
    ResourceEstimate,
    ENCODER_CYCLES,
    ENCODER_RESOURCES,
)


class PoolingUnit:
    """
    Pooling Unit for spatial down-sampling.
    
    Hardware implementation: sliding-window comparator (max) or accumulator (avg)
    operating over a configurable kernel window.
    
    Supports:
    - avg_pool2d: Average pooling with configurable kernel, stride, padding
    - max_pool2d: Max pooling with configurable kernel, stride, padding
    - adaptive_avg_pool2d: Adaptive average pooling to target size
    """
    
    def __init__(self):
        self._total_cycles = 0
    
    def avg_pool2d(
        self,
        x: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        count_include_pad: bool = True,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Average pooling with cycle tracking.
        
        Args:
            x: Input tensor [B, C, H, W]
            kernel_size: Pooling window size
            stride: Pooling stride (defaults to kernel_size)
            padding: Input padding
            count_include_pad: Include padding in average calculation
            
        Returns:
            output: Pooled tensor [B, C, H', W']
            cycles: Cycle statistics
        """
        output = F.avg_pool2d(x, kernel_size, stride=stride, padding=padding,
                              count_include_pad=count_include_pad)
        
        cycles = self._compute_cycles(x, output, kernel_size, pool_type='avg')
        self._total_cycles += cycles.total_cycles
        return output, cycles
    
    def max_pool2d(
        self,
        x: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Max pooling with cycle tracking.
        
        Args:
            x: Input tensor [B, C, H, W]
            kernel_size: Pooling window size
            stride: Pooling stride (defaults to kernel_size)
            padding: Input padding
            
        Returns:
            output: Pooled tensor [B, C, H', W']
            cycles: Cycle statistics
        """
        output = F.max_pool2d(x, kernel_size, stride=stride, padding=padding)
        
        cycles = self._compute_cycles(x, output, kernel_size, pool_type='max')
        self._total_cycles += cycles.total_cycles
        return output, cycles
    
    def adaptive_avg_pool2d(
        self,
        x: torch.Tensor,
        output_size: Union[int, Tuple[int, int]],
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Adaptive average pooling with cycle tracking.
        
        Args:
            x: Input tensor [B, C, H, W]
            output_size: Target output spatial size
            
        Returns:
            output: Pooled tensor [B, C, H', W']
            cycles: Cycle statistics
        """
        output = F.adaptive_avg_pool2d(x, output_size)
        
        # Effective kernel = input_size / output_size
        if isinstance(output_size, int):
            eff_k = x.shape[2] // output_size
        else:
            eff_k = x.shape[2] // output_size[0]
        
        cycles = self._compute_cycles(x, output, eff_k, pool_type='avg')
        self._total_cycles += cycles.total_cycles
        return output, cycles
    
    def _compute_cycles(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]],
        pool_type: str = 'avg',
    ) -> CycleStats:
        """
        Compute cycle count for pooling.
        
        Per output pixel:
        - avg: K*K accumulations + 1 division = K*K + 1 cycles
        - max: K*K comparisons = K*K cycles
        """
        B, C, H_out, W_out = output.shape
        if isinstance(kernel_size, int):
            K2 = kernel_size * kernel_size
        else:
            K2 = kernel_size[0] * kernel_size[1]
        
        output_pixels = B * C * H_out * W_out
        
        if pool_type == 'avg':
            cycles_per_pixel = K2 + 1  # accumulate + divide
        else:
            cycles_per_pixel = K2  # compare
        
        compute_cycles = output_pixels * cycles_per_pixel
        
        return CycleStats(
            total_cycles=compute_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=0,
            breakdown={
                'pooling': compute_cycles,
                'output_pixels': output_pixels,
                'kernel_size': K2,
                'pool_type': pool_type,
            }
        )
    
    def get_resource_estimate(self) -> ResourceEstimate:
        """Get hardware resource estimate."""
        return ENCODER_RESOURCES.get('pooling_unit', ResourceEstimate(
            luts=500, dsps=4, sram_bytes=512
        ))
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
