"""
Convolution Engine

Hardware simulator for systolic array based convolution.
Supports configurable kernel sizes (default: 1, 3, 7, 14) and transposed convolution.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

from .types import (
    ConvConfig,
    CycleStats,
    ResourceEstimate,
    ENCODER_CYCLES,
    ENCODER_RESOURCES,
)


class ConvEngine:
    """
    Convolution Engine with systolic array dataflow.
    
    Simulates a 2D systolic array for convolution, tracking:
    - Cycle-accurate compute time
    - Memory access patterns
    - Resource utilization
    
    Supports configurable kernel sizes and transposed convolution.
    """
    
    def __init__(self, config: Optional[ConvConfig] = None):
        """
        Initialize convolution engine.
        
        Args:
            config: Convolution engine configuration
        """
        self.config = config or ConvConfig()
        self._total_cycles = 0
    
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Perform convolution with cycle tracking.
        
        Args:
            input: Input tensor [B, Cin, H, W]
            weight: Weight tensor [Cout, Cin/groups, K, K]
            bias: Optional bias tensor [Cout]
            stride: Convolution stride
            padding: Convolution padding
            dilation: Convolution dilation
            groups: Number of groups for grouped convolution
            
        Returns:
            output: Output tensor [B, Cout, H', W']
            cycles: Cycle statistics
        """
        # Compute output using PyTorch (reference implementation)
        output = F.conv2d(
            input, weight, bias,
            stride=stride, padding=padding, dilation=dilation, groups=groups
        )
        
        # Calculate cycle count
        cycles = self._compute_cycles(input, weight, output, stride)
        self._total_cycles += cycles.total_cycles
        
        return output, cycles
    
    def forward_transposed(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Perform transposed convolution with cycle tracking.
        
        Hardware implementation: zero-insertion on input followed by standard convolution.
        
        Args:
            input: Input tensor [B, Cin, H, W]
            weight: Weight tensor [Cin, Cout/groups, K, K]
            bias: Optional bias tensor [Cout]
            stride: Transposed convolution stride
            padding: Transposed convolution padding
            output_padding: Additional output size adjustment
            groups: Number of groups
            
        Returns:
            output: Output tensor [B, Cout, H', W']
            cycles: Cycle statistics
        """
        output = F.conv_transpose2d(
            input, weight, bias,
            stride=stride, padding=padding, output_padding=output_padding, groups=groups
        )
        
        # Cycle model: transposed conv ≈ standard conv on zero-inserted input
        # Zero-insertion expands input by stride, then standard conv
        B, Cout, H_out, W_out = output.shape
        Cin = weight.shape[0]
        K = weight.shape[2]
        total_macs = B * H_out * W_out * Cin * Cout * K * K
        pe_throughput = self.config.pe_array_size ** 2
        compute_cycles = (total_macs + pe_throughput - 1) // pe_throughput
        # Extra overhead for zero-insertion address generation
        setup_cycles = ENCODER_CYCLES['conv_setup'] * 2
        weight_load_cycles = Cin * Cout * K * K // self.config.pe_array_size
        total = compute_cycles + setup_cycles + weight_load_cycles
        
        cycles = CycleStats(
            total_cycles=total,
            compute_cycles=compute_cycles,
            memory_cycles=setup_cycles + weight_load_cycles,
            breakdown={
                'compute': compute_cycles,
                'setup': setup_cycles,
                'weight_load': weight_load_cycles,
                'type': 'transposed',
            }
        )
        self._total_cycles += cycles.total_cycles
        return output, cycles
    
    def _compute_cycles(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        stride: int,
    ) -> CycleStats:
        """
        Compute cycle count for systolic array convolution.
        
        Cycle model:
        - Total MACs = H_out * W_out * Cin * Cout * K * K
        - Throughput = PE_array_size^2 MACs per cycle
        - Cycles = Total_MACs / Throughput + setup_overhead
        """
        B, Cout, H_out, W_out = output.shape
        _, Cin, K, _ = weight.shape
        
        # Total multiply-accumulate operations
        total_macs = B * H_out * W_out * Cin * Cout * K * K
        
        # Systolic array throughput (MACs per cycle)
        pe_throughput = self.config.pe_array_size ** 2
        
        # Compute cycles
        compute_cycles = (total_macs + pe_throughput - 1) // pe_throughput
        
        # Setup/overhead cycles
        setup_cycles = ENCODER_CYCLES['conv_setup']
        
        # Memory cycles (weight loading, input streaming)
        weight_load_cycles = Cout * Cin * K * K // self.config.pe_array_size
        
        total = compute_cycles + setup_cycles + weight_load_cycles
        
        return CycleStats(
            total_cycles=total,
            compute_cycles=compute_cycles,
            memory_cycles=setup_cycles + weight_load_cycles,
            breakdown={
                'compute': compute_cycles,
                'setup': setup_cycles,
                'weight_load': weight_load_cycles,
            }
        )
    
    def get_resource_estimate(self) -> ResourceEstimate:
        """Get hardware resource estimate."""
        return ENCODER_RESOURCES['conv_engine']
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
