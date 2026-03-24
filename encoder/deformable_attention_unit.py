"""
Deformable Attention Unit

Hardware simulator for Multi-Scale Deformable Attention operation.
This is a key component for TranSplat's UVTransformer.

Reference: Deformable DETR (https://arxiv.org/pdf/2010.04159.pdf)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .types import CycleStats, ResourceEstimate


@dataclass
class DeformableAttentionConfig:
    """Configuration for Deformable Attention Unit."""
    # Parallel processing
    parallel_heads: int = 4  # Number of heads processed in parallel
    parallel_points: int = 4  # Number of sampling points processed in parallel
    parallel_channels: int = 32  # Channels processed in parallel
    
    # Precision
    coord_bits: int = 16  # Bits for coordinate representation
    weight_bits: int = 16  # Bits for attention weights
    
    # Memory bandwidth
    memory_bandwidth_bytes: int = 64  # Bytes per cycle for memory access


class DeformableAttentionUnit:
    """
    Hardware simulator for Multi-Scale Deformable Attention.
    
    Implements the following operations using hardware-realizable primitives:
    1. Coordinate transformation: sampling_grids = 2 * sampling_locations - 1
    2. Bilinear grid sampling at sampling locations
    3. Weighted sum with attention weights
    
    Hardware architecture:
    - Coordinate Unit: Fixed-point multiply-add for coordinate transform
    - Bilinear Sampler: 4-point sampling with interpolation weights
    - Accumulator: MAC operations for weighted sum
    """
    
    def __init__(self, config: Optional[DeformableAttentionConfig] = None):
        """
        Initialize deformable attention unit.
        
        Args:
            config: Configuration for the unit
        """
        self.config = config or DeformableAttentionConfig()
        self._total_cycles = 0
    
    def forward(
        self,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Perform multi-scale deformable attention with cycle tracking.
        
        This is a pure hardware simulation that replicates the exact computation
        of multi_scale_deformable_attn_pytorch from mmcv.
        
        Args:
            value: [bs, num_keys, num_heads, embed_dims//num_heads]
            value_spatial_shapes: [num_levels, 2] - (H, W) for each level
            sampling_locations: [bs, num_queries, num_heads, num_levels, num_points, 2]
            attention_weights: [bs, num_queries, num_heads, num_levels, num_points]
            
        Returns:
            output: [bs, num_queries, embed_dims]
            cycles: Cycle statistics
        """
        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
        
        # Step 1: Coordinate transformation
        # sampling_grids = 2 * sampling_locations - 1
        # Hardware: fixed-point multiply (shift by 1) and subtract
        sampling_grids = 2 * sampling_locations - 1
        coord_cycles = self._compute_coord_cycles(sampling_locations)
        
        # Step 2: Split value by spatial levels
        value_list = value.split(
            [int(H_ * W_) for H_, W_ in value_spatial_shapes.tolist()],
            dim=1
        )
        
        # Step 3: Bilinear sampling at each level
        sampling_value_list = []
        sample_cycles_total = 0
        
        for level, (H_, W_) in enumerate(value_spatial_shapes.tolist()):
            H_ = int(H_)
            W_ = int(W_)
            
            # Reshape value: [bs, H_*W_, num_heads, embed_dims] -> [bs*num_heads, embed_dims, H_, W_]
            value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
                bs * num_heads, embed_dims, H_, W_
            )
            
            # Get sampling grid for this level
            # [bs, num_queries, num_heads, num_points, 2] -> [bs*num_heads, num_queries, num_points, 2]
            sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            
            # Hardware bilinear sampling using grid_sample
            # This is the core operation - each output requires 4 memory reads + 4 MACs
            sampling_value_l_ = self._hw_grid_sample(
                value_l_,
                sampling_grid_l_,
            )
            
            sampling_value_list.append(sampling_value_l_)
            
            # Accumulate sampling cycles
            sample_cycles_total += self._compute_sample_cycles(
                bs * num_heads, embed_dims, num_queries, num_points, H_, W_
            )
        
        # Step 4: Weighted sum with attention weights
        # Reshape attention weights: [bs, num_queries, num_heads, num_levels, num_points]
        # -> [bs*num_heads, 1, num_queries, num_levels*num_points]
        attention_weights_reshaped = attention_weights.transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_levels * num_points
        )
        
        # Stack sampled values and multiply with attention weights
        # [bs*num_heads, embed_dims, num_queries, num_levels*num_points]
        stacked_values = torch.stack(sampling_value_list, dim=-2).flatten(-2)
        
        # Element-wise multiply and sum: output = sum(values * weights)
        # Hardware: MAC operations
        output = (stacked_values * attention_weights_reshaped).sum(-1)
        
        # Reshape to [bs, num_queries, embed_dims]
        output = output.view(bs, num_heads * embed_dims, num_queries)
        output = output.transpose(1, 2).contiguous()
        
        # Compute weighted sum cycles
        weighted_sum_cycles = self._compute_weighted_sum_cycles(
            bs, num_heads, embed_dims, num_queries, num_levels, num_points
        )
        
        # Total cycles
        total_cycles = coord_cycles + sample_cycles_total + weighted_sum_cycles
        self._total_cycles += total_cycles
        
        cycles = CycleStats(
            total_cycles=total_cycles,
            compute_cycles=coord_cycles + weighted_sum_cycles,
            memory_cycles=sample_cycles_total,
            breakdown={
                'coordinate_transform': coord_cycles,
                'bilinear_sampling': sample_cycles_total,
                'weighted_sum': weighted_sum_cycles,
                'num_levels': num_levels,
                'num_points': num_points,
            }
        )
        
        return output, cycles
    
    def _hw_grid_sample(
        self,
        input: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hardware-simulated grid sampling using bilinear interpolation.
        
        For each sampling point:
        1. Convert normalized coords [-1,1] to pixel coords
        2. Find 4 neighboring pixels
        3. Compute bilinear weights
        4. Weighted sum of 4 pixels
        
        Args:
            input: [N, C, H, W] - feature map
            grid: [N, Hout, Wout, 2] - sampling coordinates in [-1, 1]
            
        Returns:
            output: [N, C, Hout, Wout] - sampled features
        """
        # Use PyTorch's grid_sample which performs exact bilinear interpolation
        # This matches what hardware would compute
        output = F.grid_sample(
            input,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        return output
    
    def _compute_coord_cycles(self, sampling_locations: torch.Tensor) -> int:
        """
        Compute cycles for coordinate transformation.
        
        Operation: grid = 2 * locations - 1
        Hardware: 1 shift + 1 subtract per coordinate
        """
        total_coords = sampling_locations.numel()
        # Parallel coordinate processing
        parallel_ops = self.config.parallel_heads * self.config.parallel_points * 2
        cycles = (total_coords + parallel_ops - 1) // parallel_ops
        return cycles
    
    def _compute_sample_cycles(
        self,
        batch_heads: int,
        embed_dims: int,
        num_queries: int,
        num_points: int,
        H: int,
        W: int,
    ) -> int:
        """
        Compute cycles for bilinear sampling at one level.
        
        Per sampling point per channel:
        - Coordinate conversion: 2 cycles
        - 4 memory reads: 4 cycles (or 1 cycle with 4-way parallel read)
        - Bilinear interpolation: 4 MACs = 4 cycles (or 1 cycle with 4 parallel MACs)
        
        Total per point per channel: ~4 cycles with parallelism
        """
        total_samples = batch_heads * num_queries * num_points
        total_channels = embed_dims
        
        # Cycles per sample (with channel parallelism)
        channel_batches = (total_channels + self.config.parallel_channels - 1) // self.config.parallel_channels
        
        # 4 memory accesses + 4 MACs per sample per channel batch
        cycles_per_sample = 4 + 4  # Simplified: memory + compute
        
        # Parallel sampling points
        sample_batches = (total_samples + self.config.parallel_points - 1) // self.config.parallel_points
        
        return sample_batches * channel_batches * cycles_per_sample
    
    def _compute_weighted_sum_cycles(
        self,
        bs: int,
        num_heads: int,
        embed_dims: int,
        num_queries: int,
        num_levels: int,
        num_points: int,
    ) -> int:
        """
        Compute cycles for weighted sum.
        
        Operation: output = sum(values * weights)
        Per query, per head: num_levels * num_points MACs per channel
        """
        total_macs = bs * num_heads * num_queries * embed_dims * num_levels * num_points
        
        # Parallel MAC operations
        parallel_macs = self.config.parallel_channels * self.config.parallel_heads
        cycles = (total_macs + parallel_macs - 1) // parallel_macs
        
        return cycles
    
    def grid_sample(
        self,
        input: torch.Tensor,
        grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Standalone grid sampling operation with cycle tracking.
        
        Args:
            input: [N, C, H, W]
            grid: [N, Hout, Wout, 2]
            
        Returns:
            output: [N, C, Hout, Wout]
            cycles: Cycle statistics
        """
        N, C, H, W = input.shape
        _, Hout, Wout, _ = grid.shape
        
        output = self._hw_grid_sample(input, grid)
        
        cycles = self._compute_sample_cycles(N, C, Hout, Wout, H, W)
        self._total_cycles += cycles
        
        return output, CycleStats(
            total_cycles=cycles,
            compute_cycles=cycles // 2,
            memory_cycles=cycles // 2,
            breakdown={'grid_sample': cycles}
        )
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
    
    def get_resource_estimate(self) -> ResourceEstimate:
        """Get hardware resource estimate."""
        return ResourceEstimate(
            multipliers=self.config.parallel_channels * 4,  # For bilinear: 4 MACs per sample
            adders=self.config.parallel_channels * 4,
            registers=self.config.parallel_channels * 8,  # 4 values + 4 weights
            sram_bytes=64 * 1024,  # 64KB for value caching
            description="Deformable Attention Unit with bilinear sampler and accumulator"
        )


def create_deformable_attention_unit(
    parallel_heads: int = 4,
    parallel_points: int = 4,
    parallel_channels: int = 32,
) -> DeformableAttentionUnit:
    """
    Factory function to create a DeformableAttentionUnit.
    
    Args:
        parallel_heads: Number of attention heads to process in parallel
        parallel_points: Number of sampling points to process in parallel
        parallel_channels: Number of channels to process in parallel
        
    Returns:
        Configured DeformableAttentionUnit
    """
    config = DeformableAttentionConfig(
        parallel_heads=parallel_heads,
        parallel_points=parallel_points,
        parallel_channels=parallel_channels,
    )
    return DeformableAttentionUnit(config)
