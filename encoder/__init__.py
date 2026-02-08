"""
SCARF Encoder Compute Units

Hardware simulators for 3DGS encoder computation units used by GGU:
- GEMMUnit: Matrix multiplication unit
- ActivationUnit: ReLU, GELU, SiLU, Sigmoid
- PadUnit: Constant, replicate, reflect, circular padding
"""

from .types import (
    EncoderConfig,
    CycleStats,
    ResourceEstimate,
    GEMMConfig,
    ActivationType,
)
from .gemm_unit import GEMMUnit
from .activation_unit import ActivationUnit
from .pad_unit import PadUnit

__all__ = [
    # Config and types
    'EncoderConfig',
    'CycleStats',
    'ResourceEstimate',
    'GEMMConfig',
    'ActivationType',
    # Units
    'GEMMUnit',
    'ActivationUnit',
    'PadUnit',
]
