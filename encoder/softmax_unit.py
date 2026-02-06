"""
Softmax Unit

Hardware simulator for softmax operation commonly used in depth regression.
Implements hardware-realizable softmax using:
1. Max subtraction for numerical stability
2. Piecewise linear approximation for exp()
3. Division via Newton-Raphson or LUT

This is a key component for cost volume based depth estimation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .types import (
    EncoderConfig,
    CycleStats,
    ResourceEstimate,
    ENCODER_CYCLES,
)


@dataclass
class SoftmaxConfig:
    """Configuration for Softmax hardware unit.
    
    Recommended: 'lut' with lut_size=1024 for near-exact precision.
    Hardware cost: address (2 cycles) + SRAM read (1 cycle) + interp (2 cycles) = 5 cycles.
    SRAM footprint: 1024 * 32-bit = 4KB.
    
    Alternative: 'piecewise' with num_segments=64 for minimal SRAM (512B).
    """
    # Exp approximation method: 'lut', 'piecewise', 'taylor'
    exp_method: str = 'lut'
    # LUT size for exp (if using LUT) - 1024 gives near-exact results
    lut_size: int = 1024
    # Input range for LUT (after max subtraction, all values <= 0)
    input_range: Tuple[float, float] = (-10.0, 0.0)
    # Number of segments for piecewise linear (64 gives ~0.1% relative error)
    num_segments: int = 64
    # Taylor series order
    taylor_order: int = 6


class SoftmaxUnit:
    """
    Hardware Softmax Unit for depth regression.
    
    Implements softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    Hardware Implementation:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Step 1: Find Max (Comparator Tree)                                     │
    │    max_val = max(x, dim=1)                                              │
    │    Hardware: Binary tree of comparators, O(log D) cycles                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Step 2: Subtract Max (Parallel Subtractors)                            │
    │    x_shifted = x - max_val                                              │
    │    Hardware: D parallel subtractors, 1 cycle                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Step 3: Exp Approximation (Piecewise Linear / LUT)                     │
    │    exp_x = exp_approx(x_shifted)                                        │
    │    Hardware: LUT with linear interpolation, 2 cycles per element        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Step 4: Sum (Adder Tree)                                               │
    │    sum_exp = sum(exp_x, dim=1)                                          │
    │    Hardware: Binary tree of adders, O(log D) cycles                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Step 5: Division (Newton-Raphson or LUT)                               │
    │    pdf = exp_x / sum_exp                                                │
    │    Hardware: Reciprocal LUT + multiplier, 3-5 cycles                    │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Total cycles ≈ log(D) + 1 + 2 + log(D) + 5 ≈ 2*log(D) + 8 per spatial position
    """
    
    def __init__(
        self,
        config: Optional[SoftmaxConfig] = None,
        encoder_config: Optional[EncoderConfig] = None,
    ):
        """
        Initialize Softmax Unit.
        
        Args:
            config: Softmax-specific configuration
            encoder_config: General encoder configuration
        """
        self.config = config or SoftmaxConfig()
        self.encoder_config = encoder_config or EncoderConfig()
        self._total_cycles = 0
        
        # Generate LUTs
        self._exp_lut = None
        self._reciprocal_lut = None
        # Piecewise linear coefficients (lazy init)
        self._pw_slopes = None
        self._pw_intercepts = None
        self._generate_luts()
    
    def _generate_luts(self):
        """Generate lookup tables for exp and reciprocal."""
        lut_size = self.config.lut_size
        x_min, x_max = self.config.input_range
        
        # Exp LUT: covers shifted range (always <= 0 after max subtraction)
        exp_x = np.linspace(x_min, 0.0, lut_size)
        exp_y = np.exp(exp_x)
        self._exp_lut = (
            torch.tensor(exp_x, dtype=torch.float32),
            torch.tensor(exp_y, dtype=torch.float32)
        )
        
        # Reciprocal LUT: for 1/sum normalization
        # Sum is always positive, typical range [1, D] for softmax
        recip_x = np.linspace(0.1, 256.0, lut_size)  # Avoid division by zero
        recip_y = 1.0 / recip_x
        self._reciprocal_lut = (
            torch.tensor(recip_x, dtype=torch.float32),
            torch.tensor(recip_y, dtype=torch.float32)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        dim: int = 1,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Apply softmax along specified dimension.
        
        Args:
            x: Input tensor [B, D, H, W] or [B, D]
            dim: Dimension to apply softmax (default: 1 for depth candidates)
            
        Returns:
            pdf: Probability distribution tensor
            cycles: Cycle statistics
        """
        # Use hardware-accurate implementation
        pdf = self._hw_softmax(x, dim)
        
        # Calculate cycles
        cycles = self._compute_cycles(x, dim)
        self._total_cycles += cycles.total_cycles
        
        return pdf, cycles
    
    def _hw_softmax(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Hardware-accurate softmax implementation.
        
        Uses piecewise linear approximation for exp() which is
        implementable in ASIC hardware.
        """
        device = x.device
        
        # Step 1: Max subtraction for numerical stability
        x_max = x.max(dim=dim, keepdim=True)[0]
        x_shifted = x - x_max  # Now all values <= 0
        
        # Step 2: Exp approximation
        if self.config.exp_method == 'piecewise':
            exp_x = self._piecewise_exp(x_shifted)
        elif self.config.exp_method == 'lut':
            exp_x = self._lut_exp(x_shifted, device)
        elif self.config.exp_method == 'taylor':
            exp_x = self._taylor_exp(x_shifted)
        else:
            # Fallback to PyTorch (reference)
            exp_x = torch.exp(x_shifted)
        
        # Step 3: Sum
        sum_exp = exp_x.sum(dim=dim, keepdim=True)
        
        # Step 4: Normalize (division)
        # In hardware, this would use reciprocal LUT + multiply
        pdf = exp_x / (sum_exp + 1e-10)
        
        return pdf
    
    def _piecewise_exp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Piecewise linear approximation of exp(x) for x <= 0.
        
        Hardware implementation:
        - Segment address: floor((x - x_min) / segment_width)
          For power-of-2 segment counts, this is a shift operation.
        - Lookup: Read (slope, intercept) from SRAM for the segment.
        - Compute: result = slope * x + intercept (1 MAC).
        - Total: log2(num_segments) comparator cycles + 1 MAC ≈ 7 cycles.
        
        Uses config.num_segments uniform segments over [-10, 0].
        With 64 segments (segment_width=0.15625), max relative error < 0.3%.
        """
        N = self.config.num_segments
        x_min = -10.0
        x_max = 0.0
        seg_width = (x_max - x_min) / N
        
        # Pre-compute slopes and intercepts for each segment
        # In hardware, these are stored in SRAM as fixed-point constants.
        if not hasattr(self, '_pw_slopes') or self._pw_slopes is None:
            slopes = []
            intercepts = []
            for i in range(N):
                lo = x_min + i * seg_width
                hi = lo + seg_width
                y_lo = np.exp(lo)
                y_hi = np.exp(hi)
                s = (y_hi - y_lo) / seg_width
                b = y_lo - s * lo
                slopes.append(s)
                intercepts.append(b)
            self._pw_slopes = torch.tensor(slopes, dtype=torch.float32)
            self._pw_intercepts = torch.tensor(intercepts, dtype=torch.float32)
        
        slopes = self._pw_slopes.to(x.device)
        intercepts = self._pw_intercepts.to(x.device)
        
        # Compute segment index: floor((x - x_min) / seg_width)
        x_clamped = x.clamp(min=x_min, max=x_max - 1e-7)
        seg_idx = ((x_clamped - x_min) / seg_width).long().clamp(0, N - 1)
        
        # Gather slope and intercept per element
        flat_idx = seg_idx.flatten()
        s = slopes[flat_idx].reshape(x.shape)
        b = intercepts[flat_idx].reshape(x.shape)
        
        # Linear approximation: exp(x) ≈ slope * x + intercept
        result = s * x_clamped + b
        
        # Handle x >= 0 exactly (after max subtraction, only the max element is 0)
        result = torch.where(x >= 0, torch.ones_like(x), result)
        
        # Handle very negative values
        result = torch.where(x < x_min, torch.full_like(x, 1e-10), result)
        
        return result.clamp(min=1e-10)
    
    def _lut_exp(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        LUT-based exp with linear interpolation.
        
        Hardware: Address decoder + 2 memory reads + interpolation logic
        """
        lut_x, lut_y = self._exp_lut
        lut_x = lut_x.to(device)
        lut_y = lut_y.to(device)
        
        # Clamp input to LUT range
        x_clamped = x.clamp(min=lut_x[0].item(), max=lut_x[-1].item())
        
        # Find LUT indices
        # Normalize to [0, lut_size-1]
        x_norm = (x_clamped - lut_x[0]) / (lut_x[-1] - lut_x[0])
        idx_float = x_norm * (len(lut_x) - 1)
        idx_lo = idx_float.long().clamp(0, len(lut_x) - 2)
        idx_hi = (idx_lo + 1).clamp(0, len(lut_x) - 1)
        
        # Linear interpolation
        frac = idx_float - idx_lo.float()
        y_lo = lut_y[idx_lo.flatten()].reshape(x.shape)
        y_hi = lut_y[idx_hi.flatten()].reshape(x.shape)
        
        result = y_lo + frac * (y_hi - y_lo)
        
        return result.clamp(min=1e-10)
    
    def _taylor_exp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Taylor series approximation of exp(x).
        
        exp(x) = 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5! + x⁶/6!
        
        Hardware: Multipliers + adders, computed iteratively
        """
        # Clamp to reasonable range
        x_clamped = x.clamp(min=-10.0, max=0.0)
        
        order = self.config.taylor_order
        result = torch.ones_like(x_clamped)
        term = torch.ones_like(x_clamped)
        
        for i in range(1, order + 1):
            term = term * x_clamped / i
            result = result + term
        
        return result.clamp(min=1e-10)
    
    def _compute_cycles(self, x: torch.Tensor, dim: int) -> CycleStats:
        """
        Compute cycle count for softmax operation.
        
        Hardware breakdown:
        - Max finding: O(log D) comparisons per spatial position
        - Subtraction: 1 cycle per element
        - Exp LUT: 2 cycles per element (address + read + interpolate)
        - Sum: O(log D) additions per spatial position  
        - Division: 3-5 cycles per element (reciprocal LUT + multiply)
        """
        shape = x.shape
        D = shape[dim]
        num_positions = x.numel() // D  # Number of softmax operations
        
        # Cycles per softmax operation
        max_cycles = int(np.ceil(np.log2(D))) + 1  # Comparator tree
        sub_cycles = D  # Parallel subtraction
        exp_cycles = D * 2  # LUT lookup + interpolation
        sum_cycles = int(np.ceil(np.log2(D))) + 1  # Adder tree
        div_cycles = D * 4  # Reciprocal + multiply
        
        cycles_per_softmax = max_cycles + sub_cycles + exp_cycles + sum_cycles + div_cycles
        total = num_positions * cycles_per_softmax
        
        return CycleStats(
            total_cycles=total,
            compute_cycles=total,
            memory_cycles=0,
            breakdown={
                'max_find': num_positions * max_cycles,
                'subtraction': num_positions * sub_cycles,
                'exp_lut': num_positions * exp_cycles,
                'sum': num_positions * sum_cycles,
                'division': num_positions * div_cycles,
                'num_softmax_ops': num_positions,
                'depth_candidates': D,
            }
        )
    
    def weighted_sum(
        self,
        pdf: torch.Tensor,
        candidates: torch.Tensor,
        dim: int = 1,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Compute weighted sum of candidates using probability distribution.
        
        result = sum(pdf * candidates, dim=dim)
        
        Hardware: D parallel multiply + adder tree
        
        Args:
            pdf: Probability distribution [B, D, H, W]
            candidates: Depth/disparity candidates [D] or [B, D, H, W]
            dim: Dimension to sum over
            
        Returns:
            result: Weighted sum [B, 1, H, W]
            cycles: Cycle statistics
        """
        # Broadcast candidates if needed
        if candidates.dim() == 1:
            shape = [1] * pdf.dim()
            shape[dim] = -1
            candidates = candidates.view(*shape)
            candidates = candidates.expand_as(pdf)
        
        # Weighted sum
        result = (pdf * candidates).sum(dim=dim, keepdim=True)
        
        # Cycles: multiply + sum
        D = pdf.shape[dim]
        num_positions = pdf.numel() // D
        
        multiply_cycles = pdf.numel()  # Element-wise multiply
        sum_cycles = num_positions * (int(np.ceil(np.log2(D))) + 1)  # Adder tree
        total = multiply_cycles + sum_cycles
        
        cycles = CycleStats(
            total_cycles=total,
            compute_cycles=total,
            memory_cycles=0,
            breakdown={
                'multiply': multiply_cycles,
                'sum': sum_cycles,
            }
        )
        self._total_cycles += total
        
        return result, cycles
    
    def softmax_regression(
        self,
        logits: torch.Tensor,
        candidates: torch.Tensor,
        dim: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, CycleStats]:
        """
        Combined softmax + weighted sum for depth regression.
        
        This is the common pattern in cost-volume based depth estimation:
        1. pdf = softmax(logits)
        2. depth = sum(pdf * candidates)
        
        Args:
            logits: Raw scores [B, D, H, W]
            candidates: Depth/disparity values [D]
            dim: Dimension for depth candidates
            
        Returns:
            pdf: Probability distribution
            result: Regressed depth/disparity
            cycles: Combined cycle statistics
        """
        # Softmax
        pdf, softmax_cycles = self.forward(logits, dim)
        
        # Weighted sum
        result, sum_cycles = self.weighted_sum(pdf, candidates, dim)
        
        # Combined cycles
        total_cycles = softmax_cycles.total_cycles + sum_cycles.total_cycles
        cycles = CycleStats(
            total_cycles=total_cycles,
            compute_cycles=total_cycles,
            memory_cycles=0,
            breakdown={
                'softmax': softmax_cycles.breakdown,
                'weighted_sum': sum_cycles.breakdown,
            }
        )
        
        return pdf, result, cycles
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0


# Factory function
def create_softmax_unit(
    exp_method: str = 'piecewise',
    **kwargs,
) -> SoftmaxUnit:
    """
    Create a SoftmaxUnit with specified configuration.
    
    Args:
        exp_method: 'piecewise', 'lut', or 'taylor'
        **kwargs: Additional config parameters
        
    Returns:
        Configured SoftmaxUnit
    """
    config = SoftmaxConfig(exp_method=exp_method, **kwargs)
    return SoftmaxUnit(config)
