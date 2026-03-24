"""
Depth Predictor Hardware Simulator Types

Type definitions for depth prediction hardware simulators.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum
import torch


class CostVolumeType(Enum):
    """Cost volume construction method."""
    CORRELATION = "correlation"      # Direct correlation (MVSplat, DepthSplat)
    TRANSFORMER = "transformer"      # Transformer matching (TranSplat)


class DepthRegressionType(Enum):
    """Depth regression method."""
    SOFT_ARGMAX = "soft_argmax"     # Softmax + weighted sum
    ARGMAX = "argmax"               # Hard argmax


@dataclass
class DepthPredictorConfig:
    """Configuration for depth predictor simulator."""
    
    # Depth candidate settings
    num_depth_candidates: int = 32
    depth_range: Tuple[float, float] = (0.5, 100.0)
    use_inverse_depth: bool = True
    
    # Cost volume settings
    cost_volume_type: CostVolumeType = CostVolumeType.CORRELATION
    
    # U-Net settings
    use_cost_volume_refinement: bool = True
    unet_channels: Tuple[int, ...] = (32, 64, 128)
    num_groups: int = 8
    
    # Depth refinement settings
    use_depth_refinement: bool = True
    
    # Regression settings
    regression_type: DepthRegressionType = DepthRegressionType.SOFT_ARGMAX
    softmax_temperature: float = 1.0
    
    @classmethod
    def transplat_preset(cls) -> 'DepthPredictorConfig':
        """TranSplat model preset."""
        return cls(
            num_depth_candidates=32,
            use_inverse_depth=True,
            cost_volume_type=CostVolumeType.TRANSFORMER,
            use_cost_volume_refinement=True,
            unet_channels=(32, 64, 128),
            num_groups=8,
            use_depth_refinement=True,
        )
    
    @classmethod
    def mvsplat_preset(cls) -> 'DepthPredictorConfig':
        """MVSplat model preset."""
        return cls(
            num_depth_candidates=32,
            use_inverse_depth=False,  # MVSplat uses linear depth
            cost_volume_type=CostVolumeType.CORRELATION,
            use_cost_volume_refinement=True,
            unet_channels=(32, 64, 128),
            num_groups=8,
            use_depth_refinement=True,
        )
    
    @classmethod
    def depthsplat_preset(cls) -> 'DepthPredictorConfig':
        """DepthSplat model preset."""
        return cls(
            num_depth_candidates=128,
            use_inverse_depth=False,
            cost_volume_type=CostVolumeType.CORRELATION,
            use_cost_volume_refinement=True,
            unet_channels=(128, 256),
            num_groups=4,  # DepthSplat uses GroupNorm(4)
            use_depth_refinement=True,
        )


@dataclass
class CycleBreakdown:
    """Detailed cycle breakdown for depth prediction stages."""
    feature_extraction: int = 0   # S1-shared feature extraction (CNN/Transformer/DINOv2) inside DP
    cost_volume: int = 0
    unet_refinement: int = 0
    depth_head: int = 0
    softmax_regression: int = 0
    depth_refinement: int = 0
    upsampling: int = 0
    gaussian_head: int = 0
    
    @property
    def total(self) -> int:
        """Total cycles."""
        return (
            self.feature_extraction +
            self.cost_volume +
            self.unet_refinement +
            self.depth_head +
            self.softmax_regression +
            self.depth_refinement +
            self.upsampling +
            self.gaussian_head
        )
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            'feature_extraction': self.feature_extraction,
            'cost_volume': self.cost_volume,
            'unet_refinement': self.unet_refinement,
            'depth_head': self.depth_head,
            'softmax_regression': self.softmax_regression,
            'depth_refinement': self.depth_refinement,
            'upsampling': self.upsampling,
            'gaussian_head': self.gaussian_head,
            'total': self.total,
        }


@dataclass
class DepthPredictorOutput:
    """Output from depth predictor simulator."""
    
    # Core outputs
    depths: torch.Tensor               # [B, V, H, W] or [B, V, H*W, srf, gpp]
    densities: Optional[torch.Tensor]  # [B, V, H, W] or [B, V, H*W, srf, gpp]
    raw_gaussians: Optional[torch.Tensor]  # [B, V, H*W, d_in]
    
    # Intermediate outputs (for debugging/comparison)
    depth_probs: Optional[torch.Tensor] = None  # [B, V, D, H, W] - softmax probs
    cost_volume: Optional[torch.Tensor] = None  # [B, V, D, H, W] - raw cost volume
    
    # Cycle statistics
    total_cycles: int = 0
    cycle_breakdown: CycleBreakdown = field(default_factory=CycleBreakdown)
    
    def get_cycle_breakdown_dict(self) -> Dict[str, int]:
        """Get cycle breakdown as dictionary."""
        return self.cycle_breakdown.to_dict()


@dataclass
class UNetLayerConfig:
    """Configuration for a single U-Net layer."""
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    use_norm: bool = True
    use_activation: bool = True
    num_groups: int = 8


@dataclass
class UNetConfig:
    """Configuration for U-Net architecture."""
    
    # Encoder layers
    encoder_channels: Tuple[int, ...] = (32, 64, 128)
    
    # Decoder layers (auto-generated from encoder if not specified)
    decoder_channels: Optional[Tuple[int, ...]] = None
    
    # Common settings
    num_groups: int = 8
    kernel_size: int = 3
    
    def __post_init__(self):
        if self.decoder_channels is None:
            # Mirror encoder (excluding last)
            self.decoder_channels = tuple(reversed(self.encoder_channels[:-1]))
