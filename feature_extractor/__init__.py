"""
Feature Extractor Hardware Simulator

Simulates CNN backbone and Transformer feature extraction using
SCARF encoder compute units with cycle-accurate modeling.

Supports:
- CNN + Transformer (TranSplat, MVSplat)
- ViT/DINOv2 (DepthSplat)

Model-Specific Extractors:
- TransplatFeatureExtractor: CNN + Transformer + DepthAnythingV2
- MVSplatFeatureExtractor: CNN + Transformer
- DepthSplatFeatureExtractor: CNN + DINOv2 + Transformer
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

# Model-specific extractors
from .transplat_extractor import TransplatFeatureExtractor, TransplatFeatureOutput
from .mvsplat_extractor import MVSplatFeatureExtractor, MVSplatFeatureOutput
from .depthsplat_extractor import DepthSplatFeatureExtractor, DepthSplatFeatureOutput

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
    # Model-specific extractors
    'TransplatFeatureExtractor',
    'TransplatFeatureOutput',
    'MVSplatFeatureExtractor',
    'MVSplatFeatureOutput',
    'DepthSplatFeatureExtractor',
    'DepthSplatFeatureOutput',
]
