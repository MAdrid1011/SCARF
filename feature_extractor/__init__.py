"""
Feature Extractor Hardware Simulator

Simulates CNN backbone and Transformer feature extraction using
SCARF encoder compute units with cycle-accurate modeling.
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
]
