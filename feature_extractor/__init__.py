"""
Feature Extractor Hardware Simulator

Simulates CNN backbone and Transformer feature extraction using
SCARF encoder compute units with cycle-accurate modeling.

Supports:
- CNN + Transformer (Transplat, MVSplat)
- ViT/DINOv2 (DepthSplat)
"""

from .types import (
    FeatureExtractorConfig,
    FeatureOutput,
    CNNConfig,
    TransformerConfig,
)
from .cnn_simulator import CNNEncoderSimulator
from .transformer_simulator import TransformerSimulator, AttentionSim, FFNSim
from .feature_extractor import FeatureExtractorSimulator
from .vit_simulator import ViTSimulator, ViTConfig, ViTOutput

__all__ = [
    'FeatureExtractorConfig',
    'FeatureOutput',
    'CNNConfig',
    'TransformerConfig',
    'CNNEncoderSimulator',
    'TransformerSimulator',
    'AttentionSim',
    'FFNSim',
    'FeatureExtractorSimulator',
    # ViT/DINOv2
    'ViTSimulator',
    'ViTConfig',
    'ViTOutput',
]
