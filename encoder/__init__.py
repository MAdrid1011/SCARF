"""
SCARF Encoder Compute Units

Hardware simulators for 3DGS encoder computation units:
- ConvEngine: Convolution engine with systolic array
- GEMMUnit: Matrix multiplication unit
- ActivationUnit: ReLU, GELU, SiLU, Sigmoid
- NormalizationUnit: LayerNorm, BatchNorm, InstanceNorm, GroupNorm
- BilinearUnit: Bilinear interpolation for upsampling
"""

from .types import (
    EncoderConfig,
    CycleStats,
    ResourceEstimate,
    ConvConfig,
    GEMMConfig,
    ActivationType,
    NormType,
    BilinearConfig,
)
from .conv_engine import ConvEngine
from .gemm_unit import GEMMUnit
from .activation_unit import ActivationUnit
from .normalization_unit import NormalizationUnit
from .bilinear_unit import BilinearUnit

__all__ = [
    # Config and types
    'EncoderConfig',
    'CycleStats',
    'ResourceEstimate',
    'ConvConfig',
    'GEMMConfig',
    'ActivationType',
    'NormType',
    'BilinearConfig',
    # Units
    'ConvEngine',
    'GEMMUnit',
    'ActivationUnit',
    'NormalizationUnit',
    'BilinearUnit',
]
