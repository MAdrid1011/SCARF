"""
SCARF Encoder Compute Units

Hardware simulators for 3DGS encoder computation units:
- ConvEngine: Convolution engine with systolic array (incl. transposed conv)
- GEMMUnit: Matrix multiplication unit
- ActivationUnit: ReLU, GELU, SiLU, Sigmoid
- NormalizationUnit: LayerNorm, BatchNorm, InstanceNorm, GroupNorm
- BilinearUnit: Interpolation (bilinear, nearest, bicubic) + grid_sample
- PoolingUnit: Average / max / adaptive pooling
- PadUnit: Constant, replicate, reflect, circular padding
- SoftmaxUnit: Softmax for depth regression
- DeformableAttentionUnit: Multi-scale deformable attention (for Transplat)
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
from .pooling_unit import PoolingUnit
from .pad_unit import PadUnit
from .softmax_unit import SoftmaxUnit, SoftmaxConfig, create_softmax_unit
from .deformable_attention_unit import (
    DeformableAttentionUnit,
    DeformableAttentionConfig,
    create_deformable_attention_unit,
)

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
    'SoftmaxConfig',
    'DeformableAttentionConfig',
    # Units
    'ConvEngine',
    'GEMMUnit',
    'ActivationUnit',
    'NormalizationUnit',
    'BilinearUnit',
    'PoolingUnit',
    'PadUnit',
    'SoftmaxUnit',
    'create_softmax_unit',
    'DeformableAttentionUnit',
    'create_deformable_attention_unit',
]
