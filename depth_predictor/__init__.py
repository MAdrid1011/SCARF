"""
Depth Predictor Hardware Simulator

Hardware simulators for depth prediction in 3DGS encoders.
Replaces PyTorch depth predictor inference with cycle-accurate
hardware compositions.

Supports:
- TranSplat: Transformer-based cost volume matching
- MVSplat: Direct correlation cost volume
- DepthSplat: Multi-scale correlation with DPT head

Architecture:
    Features → Cost Volume → U-Net Refinement → Depth Head → Softmax Regression
                                                        ↓
                                              Depth Refinement (optional)
                                                        ↓
                                              Gaussian Head → Raw Gaussians
"""

from .types import (
    DepthPredictorConfig,
    DepthPredictorOutput,
    CycleBreakdown,
    CostVolumeType,
    DepthRegressionType,
    UNetConfig,
    UNetLayerConfig,
)

from .base_predictor import BaseDepthPredictorSim, PassThroughDepthPredictorSim
from .hw_depth_predictor import HWDepthPredictor
from .transplat_predictor import TransplatDepthPredictorSim, create_transplat_predictor
from .mvsplat_predictor import MVSplatDepthPredictorSim, create_mvsplat_predictor
from .depthsplat_predictor import DepthSplatDepthPredictorSim, create_depthsplat_predictor

__all__ = [
    # Types
    'DepthPredictorConfig',
    'DepthPredictorOutput',
    'CycleBreakdown',
    'CostVolumeType',
    'DepthRegressionType',
    'UNetConfig',
    'UNetLayerConfig',
    
    # Base classes
    'BaseDepthPredictorSim',
    'PassThroughDepthPredictorSim',
    
    # Hardware depth predictor
    'HWDepthPredictor',
    
    # Model-specific predictors
    'TransplatDepthPredictorSim',
    'MVSplatDepthPredictorSim',
    'DepthSplatDepthPredictorSim',
    
    # Factory functions
    'create_transplat_predictor',
    'create_mvsplat_predictor',
    'create_depthsplat_predictor',
]
