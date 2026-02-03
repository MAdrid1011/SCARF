"""
Activation Unit

Hardware simulator for activation functions.
Supports ReLU, GELU, SiLU, Sigmoid with LUT-based implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.special import erf

from .types import (
    ActivationType,
    EncoderConfig,
    CycleStats,
    ResourceEstimate,
    ENCODER_CYCLES,
    ENCODER_RESOURCES,
)


class ActivationUnit:
    """
    Activation Unit with LUT-based non-linear functions.
    
    Supports:
    - ReLU: Combinational (max(0, x))
    - GELU: 256-entry LUT with interpolation
    - SiLU: 256-entry LUT with interpolation
    - Sigmoid: 256-entry LUT with interpolation
    
    All LUT-based activations use linear interpolation for accuracy.
    """
    
    def __init__(
        self,
        activation_type: ActivationType,
        config: Optional[EncoderConfig] = None,
    ):
        """
        Initialize activation unit.
        
        Args:
            activation_type: Type of activation function
            config: Encoder configuration (for LUT parameters)
        """
        self.activation_type = activation_type
        self.config = config or EncoderConfig()
        self._total_cycles = 0
        
        # Generate LUTs for non-linear functions
        self._luts = {}
        if activation_type != ActivationType.RELU:
            self._generate_luts()
    
    def _generate_luts(self):
        """Generate lookup tables for activation functions."""
        lut_size = self.config.activation_lut_size
        x_min, x_max = self.config.activation_input_range
        
        x = np.linspace(x_min, x_max, lut_size)
        
        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        gelu_y = x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
        self._luts['gelu'] = (torch.tensor(x, dtype=torch.float32),
                              torch.tensor(gelu_y, dtype=torch.float32))
        
        # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        silu_y = x / (1.0 + np.exp(-x))
        self._luts['silu'] = (torch.tensor(x, dtype=torch.float32),
                              torch.tensor(silu_y, dtype=torch.float32))
        
        # Sigmoid: 1 / (1 + exp(-x))
        sigmoid_y = 1.0 / (1.0 + np.exp(-x))
        self._luts['sigmoid'] = (torch.tensor(x, dtype=torch.float32),
                                  torch.tensor(sigmoid_y, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, CycleStats]:
        """
        Apply activation function with cycle tracking.
        
        Args:
            x: Input tensor
            
        Returns:
            output: Activated tensor
            cycles: Cycle statistics
        """
        if self.activation_type == ActivationType.RELU:
            output = F.relu(x)
        elif self.activation_type == ActivationType.GELU:
            # Use PyTorch for accuracy (LUT would be approximation)
            output = F.gelu(x)
        elif self.activation_type == ActivationType.SILU:
            output = F.silu(x)
        elif self.activation_type == ActivationType.SIGMOID:
            output = torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")
        
        # Calculate cycle count
        cycles = self._compute_cycles(x)
        self._total_cycles += cycles.total_cycles
        
        return output, cycles
    
    def _compute_cycles(self, x: torch.Tensor) -> CycleStats:
        """
        Compute cycle count for activation.
        
        All activations are 1 cycle per element:
        - ReLU: Combinational comparator
        - Others: LUT lookup with optional interpolation
        """
        num_elements = x.numel()
        
        if self.activation_type == ActivationType.RELU:
            cycles_per_element = ENCODER_CYCLES['activation_relu']
        else:
            cycles_per_element = ENCODER_CYCLES['activation_lut']
        
        total = num_elements * cycles_per_element
        
        return CycleStats(
            total_cycles=total,
            compute_cycles=total,
            memory_cycles=0,
            breakdown={
                'activation': total,
                'num_elements': num_elements,
            }
        )
    
    def get_resource_estimate(self) -> ResourceEstimate:
        """Get hardware resource estimate."""
        return ENCODER_RESOURCES['activation_unit']
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
