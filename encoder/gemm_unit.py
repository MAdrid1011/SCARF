"""
GEMM Unit

Hardware simulator for matrix multiplication.
Supports Transformer attention and FFN operations.
"""

import torch
from typing import Tuple, Optional
from dataclasses import dataclass

from .types import (
    GEMMConfig,
    CycleStats,
    ResourceEstimate,
    ENCODER_CYCLES,
    ENCODER_RESOURCES,
)
from .mmcu_events import record_mmcu_slots


class GEMMUnit:
    """
    GEMM Unit with output-stationary systolic array.
    
    Simulates tiled matrix multiplication for:
    - Linear layers (FFN)
    - Attention score computation (Q @ K^T)
    - Attention output computation (Attn @ V)
    
    Dataflow: Output-stationary with K-dimension streaming
    """
    
    def __init__(self, config: Optional[GEMMConfig] = None):
        """
        Initialize GEMM unit.
        
        Args:
            config: GEMM unit configuration
        """
        self.config = config or GEMMConfig()
        self._total_cycles = 0
    
    def matmul(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Perform matrix multiplication with cycle tracking.
        
        Args:
            a: Left matrix [..., M, K]
            b: Right matrix [..., K, N]
            
        Returns:
            output: Result matrix [..., M, N]
            cycles: Cycle statistics
        """
        # Compute output using PyTorch
        output = torch.matmul(a, b)
        
        # Calculate cycle count
        cycles = self._compute_cycles(a, b)
        self._total_cycles += cycles.total_cycles
        
        return output, cycles
    
    def linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Perform linear transformation (input @ weight^T + bias).
        
        Args:
            input: Input tensor [..., in_features]
            weight: Weight tensor [out_features, in_features]
            bias: Optional bias [out_features]
            
        Returns:
            output: Output tensor [..., out_features]
            cycles: Cycle statistics
        """
        # Compute using matmul with transposed weight
        output, cycles = self.matmul(input, weight.t())
        
        if bias is not None:
            output = output + bias
        
        return output, cycles
    
    def _compute_cycles(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> CycleStats:
        """
        Compute cycle count for tiled GEMM.
        
        Cycle model:
        - Process output in tile_m × tile_n tiles
        - Each tile takes K + fill/drain cycles
        - Total = num_tiles_m × num_tiles_n × cycles_per_tile
        """
        # Get matrix dimensions (handle batched case)
        if a.dim() == 2:
            M, K = a.shape
            _, N = b.shape
            batch = 1
        else:
            *batch_dims, M, K = a.shape
            _, N = b.shape[-2:]
            batch = 1
            for d in batch_dims:
                batch *= d
        
        # Tile counts
        tile_m = self.config.tile_m
        tile_n = self.config.tile_n
        
        num_tiles_m = (M + tile_m - 1) // tile_m
        num_tiles_n = (N + tile_n - 1) // tile_n
        
        # Cycles per tile: K accumulations + pipeline fill/drain
        pipeline_depth = tile_m + tile_n
        cycles_per_tile = K + pipeline_depth
        
        # Total compute cycles
        compute_cycles = batch * num_tiles_m * num_tiles_n * cycles_per_tile
        useful_slots = batch * M * N * K
        scheduled_slots = compute_cycles * self.config.array_m * self.config.array_n
        record_mmcu_slots(useful_slots, scheduled_slots)
        
        # Overhead per batch
        overhead = batch * ENCODER_CYCLES['gemm_tile_overhead']
        
        total = compute_cycles + overhead
        
        return CycleStats(
            total_cycles=total,
            compute_cycles=compute_cycles,
            memory_cycles=overhead,
            breakdown={
                'compute': compute_cycles,
                'overhead': overhead,
                'num_tiles': num_tiles_m * num_tiles_n,
            }
        )
    
    def get_resource_estimate(self) -> ResourceEstimate:
        """Get hardware resource estimate."""
        return ENCODER_RESOURCES['gemm_unit']
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
