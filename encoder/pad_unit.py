"""
Padding Unit

Hardware simulator for tensor padding operations.
Supports constant, replicate, reflect, and circular padding modes.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Union, List

from .types import (
    CycleStats,
    ResourceEstimate,
    ENCODER_RESOURCES,
)


class PadUnit:
    """
    Padding Unit for spatial padding operations.
    
    Hardware implementation: address generator that remaps border reads
    to implement various padding modes without actually copying data.
    For constant padding, a mux selects the constant value at border positions.
    
    Supports:
    - constant: Pad with a constant value (default 0)
    - replicate: Pad by repeating edge values
    - reflect: Pad by mirroring values at border
    - circular: Pad by wrapping around
    """
    
    def __init__(self):
        self._total_cycles = 0
    
    def pad(
        self,
        x: torch.Tensor,
        pad: Union[Tuple[int, ...], List[int]],
        mode: str = 'constant',
        value: float = 0.0,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Apply padding with cycle tracking.
        
        Args:
            x: Input tensor
            pad: Padding sizes (left, right, top, bottom, ...)
                 Following PyTorch F.pad convention.
            mode: Padding mode ('constant', 'replicate', 'reflect', 'circular')
            value: Fill value for constant padding
            
        Returns:
            output: Padded tensor
            cycles: Cycle statistics
        """
        output = F.pad(x, pad, mode=mode, value=value)
        
        cycles = self._compute_cycles(x, output, mode)
        self._total_cycles += cycles.total_cycles
        return output, cycles
    
    def _compute_cycles(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        mode: str,
    ) -> CycleStats:
        """
        Compute cycle count for padding.
        
        Padding is primarily a memory/address operation:
        - constant: 1 cycle per padded element (mux select)
        - replicate/reflect/circular: 1 cycle per padded element (address remap)
        
        For in-flight padding (fused with convolution), this adds zero overhead.
        Standalone padding counted separately.
        """
        total_elements = output.numel()
        original_elements = input.numel()
        padded_elements = total_elements - original_elements
        
        # Address generation + data movement for padded elements
        # Original elements are a straight copy (1 cycle per element, pipelined)
        if mode == 'constant':
            compute_cycles = padded_elements  # mux select constant
        else:
            compute_cycles = padded_elements * 2  # address remap + read
        
        # Add copy overhead for original data (pipelined, so minimal)
        compute_cycles += original_elements // 8  # 8-wide data bus
        
        return CycleStats(
            total_cycles=compute_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=0,
            breakdown={
                'padding': compute_cycles,
                'padded_elements': padded_elements,
                'mode': mode,
            }
        )
    
    def get_resource_estimate(self) -> ResourceEstimate:
        """Get hardware resource estimate."""
        return ENCODER_RESOURCES.get('pad_unit', ResourceEstimate(
            luts=200, dsps=0, sram_bytes=0
        ))
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
