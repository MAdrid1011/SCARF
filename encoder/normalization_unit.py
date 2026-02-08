"""
Normalization Unit

Hardware simulator for normalization operations.
Supports LayerNorm, BatchNorm, InstanceNorm, GroupNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .types import (
    NormType,
    EncoderConfig,
    CycleStats,
    ResourceEstimate,
    ENCODER_CYCLES,
    ENCODER_RESOURCES,
)


class NormalizationUnit:
    """
    Normalization Unit supporting multiple normalization types.
    
    Supports:
    - LayerNorm: Normalize over last dimension
    - BatchNorm: Normalize over batch (uses running stats in eval)
    - InstanceNorm: Normalize per sample
    - GroupNorm: Normalize within groups
    
    All use the formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    
    def __init__(
        self,
        norm_type: NormType,
        dim: int,
        num_groups: int = 8,
        config: Optional[EncoderConfig] = None,
    ):
        """
        Initialize normalization unit.
        
        Args:
            norm_type: Type of normalization
            dim: Dimension to normalize (channels for BN/IN/GN, feature dim for LN)
            num_groups: Number of groups for GroupNorm
            config: Encoder configuration
        """
        self.norm_type = norm_type
        self.dim = dim
        self.num_groups = num_groups
        self.config = config or EncoderConfig()
        self._total_cycles = 0
        self._training = True
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
        # Running stats for BatchNorm
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self._training = mode
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, CycleStats]:
        """
        Apply normalization with cycle tracking.
        
        Args:
            x: Input tensor
               - LayerNorm: [..., dim]
               - BatchNorm/InstanceNorm/GroupNorm: [B, C, H, W] or [B, C, L]
            
        Returns:
            output: Normalized tensor
            cycles: Cycle statistics
        """
        if self.norm_type == NormType.LAYER:
            output = F.layer_norm(x, (self.dim,), self.weight, self.bias, 
                                   self.config.norm_epsilon)
        elif self.norm_type == NormType.BATCH:
            if self._training:
                output = F.batch_norm(x, None, None, self.weight, self.bias,
                                       True, 0.1, self.config.norm_epsilon)
            else:
                output = F.batch_norm(x, self.running_mean, self.running_var,
                                       self.weight, self.bias,
                                       False, 0.1, self.config.norm_epsilon)
        elif self.norm_type == NormType.INSTANCE:
            output = F.instance_norm(x, weight=self.weight, bias=self.bias,
                                      eps=self.config.norm_epsilon)
        elif self.norm_type == NormType.GROUP:
            output = F.group_norm(x, self.num_groups, self.weight, self.bias,
                                   self.config.norm_epsilon)
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")
        
        # Calculate cycle count
        cycles = self._compute_cycles(x)
        self._total_cycles += cycles.total_cycles
        
        return output, cycles
    
    def _compute_cycles(self, x: torch.Tensor) -> CycleStats:
        """
        Compute cycle count for normalization.
        
        Cycle model:
        - Mean: N elements accumulation + 1 division
        - Var: N elements accumulation + 1 division
        - Normalize: N elements (subtract, multiply, add)
        - Total: ~3N + 2 (LayerNorm/InstanceNorm)
        
        BatchNorm (eval): Just scale/shift = N cycles
        """
        if self.norm_type == NormType.LAYER:
            # Normalize over last dim
            N = self.dim
            num_instances = x.numel() // N
        elif self.norm_type == NormType.BATCH:
            # Normalize per channel over batch
            N = x.shape[0] * x[0, 0].numel() if x.dim() > 2 else x.shape[0]
            num_instances = self.dim
        elif self.norm_type == NormType.INSTANCE:
            # Normalize per sample per channel
            if x.dim() == 4:
                N = x.shape[2] * x.shape[3]
            else:
                N = x.shape[-1]
            num_instances = x.shape[0] * self.dim
        elif self.norm_type == NormType.GROUP:
            # Normalize within groups
            channels_per_group = self.dim // self.num_groups
            if x.dim() == 4:
                N = channels_per_group * x.shape[2] * x.shape[3]
            else:
                N = channels_per_group * x.shape[-1]
            num_instances = x.shape[0] * self.num_groups
        
        # Cycle calculation
        if self.norm_type == NormType.BATCH and not self._training:
            # Uses running stats - just scale/shift
            compute_cycles = num_instances * N
        else:
            # Full computation: mean + var + normalize
            mean_cycles = N * ENCODER_CYCLES['norm_mean_per_element']
            var_cycles = N * ENCODER_CYCLES['norm_var_per_element']
            apply_cycles = N * ENCODER_CYCLES['norm_apply_per_element']
            overhead = ENCODER_CYCLES['norm_overhead']
            
            compute_cycles = num_instances * (mean_cycles + var_cycles + apply_cycles + overhead)
        
        return CycleStats(
            total_cycles=compute_cycles,
            compute_cycles=compute_cycles,
            memory_cycles=0,
            breakdown={
                'normalize': compute_cycles,
                'num_instances': num_instances,
                'elements_per_instance': N,
            }
        )
    
    def get_resource_estimate(self) -> ResourceEstimate:
        """Get hardware resource estimate."""
        return ENCODER_RESOURCES['normalization_unit']
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
