"""
Encoder Compute Unit Types

Shared type definitions for encoder hardware simulators.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, Dict, Any


class ActivationType(Enum):
    """Supported activation functions."""
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"


class NormType(Enum):
    """Supported normalization types."""
    LAYER = "layer"
    BATCH = "batch"
    INSTANCE = "instance"
    GROUP = "group"


@dataclass
class CycleStats:
    """Cycle count statistics for hardware operations."""
    total_cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    
    # Detailed breakdown
    breakdown: Dict[str, int] = field(default_factory=dict)
    
    def __add__(self, other: 'CycleStats') -> 'CycleStats':
        """Add two CycleStats together."""
        new_breakdown = dict(self.breakdown)
        for k, v in other.breakdown.items():
            new_breakdown[k] = new_breakdown.get(k, 0) + v
        return CycleStats(
            total_cycles=self.total_cycles + other.total_cycles,
            compute_cycles=self.compute_cycles + other.compute_cycles,
            memory_cycles=self.memory_cycles + other.memory_cycles,
            breakdown=new_breakdown,
        )


@dataclass
class ResourceEstimate:
    """Hardware resource estimates."""
    luts: int = 0
    dsps: int = 0
    sram_bytes: int = 0
    rom_bytes: int = 0
    
    def __add__(self, other: 'ResourceEstimate') -> 'ResourceEstimate':
        """Add two ResourceEstimate together."""
        return ResourceEstimate(
            luts=self.luts + other.luts,
            dsps=self.dsps + other.dsps,
            sram_bytes=self.sram_bytes + other.sram_bytes,
            rom_bytes=self.rom_bytes + other.rom_bytes,
        )


@dataclass
class ConvConfig:
    """Convolution engine configuration."""
    # Systolic array dimensions
    pe_array_size: int = 16  # 16x16 PE array
    
    # Supported kernel sizes
    supported_kernels: Tuple[int, ...] = (1, 3, 7)
    
    # Memory configuration
    weight_buffer_kb: int = 64
    input_buffer_lines: int = 8


@dataclass
class GEMMConfig:
    """GEMM unit configuration."""
    # Tile dimensions
    tile_m: int = 8
    tile_n: int = 16
    
    # Array size
    array_m: int = 8
    array_n: int = 16
    
    # Memory
    buffer_kb: int = 32


@dataclass
class BilinearConfig:
    """Bilinear interpolation configuration."""
    # Parallel channels
    parallel_channels: int = 8
    
    # Coordinate precision (fixed-point bits)
    coord_frac_bits: int = 8
    
    # Default align_corners
    align_corners: bool = True


@dataclass
class EncoderConfig:
    """Unified configuration for all encoder compute units."""
    
    # Convolution Engine
    conv: ConvConfig = field(default_factory=ConvConfig)
    
    # GEMM Unit
    gemm: GEMMConfig = field(default_factory=GEMMConfig)
    
    # Activation Unit
    activation_lut_size: int = 256
    activation_input_range: Tuple[float, float] = (-4.0, 4.0)
    
    # Normalization Unit
    norm_epsilon: float = 1e-5
    norm_groups: int = 8  # Default for GroupNorm
    
    # Bilinear Unit
    bilinear: BilinearConfig = field(default_factory=BilinearConfig)
    
    @classmethod
    def transplat_preset(cls) -> 'EncoderConfig':
        """Transplat model preset."""
        return cls(
            norm_groups=8,
        )
    
    @classmethod
    def mvsplat_preset(cls) -> 'EncoderConfig':
        """MVSplat model preset."""
        return cls(
            norm_groups=8,
        )
    
    @classmethod
    def depthsplat_preset(cls) -> 'EncoderConfig':
        """DepthSplat model preset."""
        return cls(
            norm_groups=4,  # DepthSplat uses GroupNorm(4)
        )


# Cycle count constants for resource estimation
ENCODER_CYCLES = {
    # Convolution (per MAC)
    'conv_mac': 1,
    'conv_setup': 10,
    
    # GEMM (per tile)
    'gemm_tile_overhead': 10,
    
    # Activation (per element)
    'activation_relu': 1,
    'activation_lut': 1,
    
    # Normalization
    'norm_mean_per_element': 1,
    'norm_var_per_element': 1,
    'norm_apply_per_element': 1,
    'norm_overhead': 2,
    
    # Bilinear (per output pixel)
    'bilinear_coord': 1,
    'bilinear_sample': 1,
    'bilinear_interpolate': 2,
}


# Resource estimates
ENCODER_RESOURCES = {
    'conv_engine': ResourceEstimate(luts=50000, dsps=256, sram_bytes=65536),
    'gemm_unit': ResourceEstimate(luts=20000, dsps=128, sram_bytes=32768),
    'activation_unit': ResourceEstimate(luts=2000, dsps=8, sram_bytes=1024),
    'normalization_unit': ResourceEstimate(luts=3000, dsps=16, sram_bytes=2048),
    'bilinear_unit': ResourceEstimate(luts=800, dsps=8, sram_bytes=0),
}
