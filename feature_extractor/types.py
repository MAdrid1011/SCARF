"""
Feature Extractor Data Types

Configuration and output structures for the feature extraction hardware simulator.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch


@dataclass
class CNNConfig:
    """CNN Encoder configuration."""
    input_channels: int = 3
    feature_dims: Tuple[int, int, int] = (64, 96, 128)
    output_dim: int = 128
    num_output_scales: int = 1
    norm_type: str = 'instance'
    
    @property
    def downscale_factor(self) -> int:
        return 8 if self.num_output_scales == 1 else 4


@dataclass
class TransformerConfig:
    """Transformer configuration."""
    d_model: int = 128
    num_layers: int = 6
    num_heads: int = 1
    ffn_dim_expansion: int = 4
    attn_splits: int = 2
    
    @property
    def ffn_dim(self) -> int:
        return self.d_model * self.ffn_dim_expansion


@dataclass
class FeatureExtractorConfig:
    """Feature Extractor configuration."""
    model_type: str = 'transplat'
    feature_channels: int = 128
    num_transformer_layers: int = 6
    enable_cycle_counting: bool = True
    cnn_config: Optional[CNNConfig] = None
    transformer_config: Optional[TransformerConfig] = None
    
    def __post_init__(self):
        if self.cnn_config is None:
            self.cnn_config = CNNConfig(output_dim=self.feature_channels)
        if self.transformer_config is None:
            self.transformer_config = TransformerConfig(
                d_model=self.feature_channels,
                num_layers=self.num_transformer_layers
            )
    
    @classmethod
    def transplat_preset(cls) -> 'FeatureExtractorConfig':
        return cls(model_type='transplat', feature_channels=128, num_transformer_layers=6)
    
    @classmethod
    def mvsplat_preset(cls) -> 'FeatureExtractorConfig':
        return cls(model_type='mvsplat', feature_channels=128, num_transformer_layers=6)
    
    @classmethod
    def depthsplat_preset(cls) -> 'FeatureExtractorConfig':
        return cls(model_type='depthsplat', feature_channels=128, num_transformer_layers=0)


@dataclass
class FeatureOutput:
    """Output of feature extraction."""
    features: torch.Tensor
    cnn_cycles: int = 0
    transformer_cycles: int = 0
    total_cycles: int = 0
    cycle_breakdown: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.total_cycles == 0:
            self.total_cycles = self.cnn_cycles + self.transformer_cycles


@dataclass
class LayerCycles:
    """Cycle counts for a single layer."""
    conv_cycles: int = 0
    norm_cycles: int = 0
    activation_cycles: int = 0
    gemm_cycles: int = 0
    
    @property
    def total(self) -> int:
        return self.conv_cycles + self.norm_cycles + self.activation_cycles + self.gemm_cycles
