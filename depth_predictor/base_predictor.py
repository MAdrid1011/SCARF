"""
Base Depth Predictor Simulator

Abstract base class for model-specific depth predictors.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

from .types import DepthPredictorConfig, DepthPredictorOutput, CycleBreakdown
from .cost_volume_sim import CostVolumeSimulator
from .unet_sim import SimplifiedUNetSim
from .depth_head_sim import SimplifiedDepthHeadSim
from encoder import BilinearUnit
from encoder.types import CycleStats


class BaseDepthPredictorSim(ABC):
    """
    Abstract base class for depth predictor simulators.
    
    Provides common interface and utility methods for all model-specific
    depth predictor implementations.
    
    Common pipeline:
        Features → Cost Volume → U-Net Refinement → Depth Head → 
        Softmax Regression → (optional) Depth Refinement → Gaussian Head
    """
    
    def __init__(
        self,
        config: DepthPredictorConfig,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """
        Initialize base depth predictor simulator.
        
        Args:
            config: Depth predictor configuration
            device: Computation device
        """
        self.config = config
        self.device = device
        
        # Common hardware units
        self.cost_volume_sim = CostVolumeSimulator(config, device)
        self.unet_sim = SimplifiedUNetSim(device)
        self.depth_head_sim = SimplifiedDepthHeadSim(device)
        self.bilinear_unit = BilinearUnit()
        
        # Cycle tracking
        self._cycle_breakdown = CycleBreakdown()
        self._initialized = False
    
    @abstractmethod
    def load_from_model(self, model: nn.Module) -> None:
        """
        Load weights from original model.
        
        Args:
            model: Original encoder model
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        **kwargs,
    ) -> DepthPredictorOutput:
        """
        Forward pass through depth predictor.
        
        Args:
            features: Input features [B, V, C, H, W]
            intrinsics: Camera intrinsics [B, V, 3, 3]
            extrinsics: Camera extrinsics [B, V, 4, 4]
            near: Near plane [B, V]
            far: Far plane [B, V]
            
        Returns:
            DepthPredictorOutput with depths, densities, raw_gaussians
        """
        pass
    
    def forward_with_reference(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        reference_model: nn.Module,
        **kwargs,
    ) -> Tuple[DepthPredictorOutput, Any]:
        """
        Forward pass with comparison to reference implementation.
        
        Args:
            features, intrinsics, extrinsics, near, far: Same as forward()
            reference_model: Original depth predictor model
            
        Returns:
            sim_output: Simulator output
            ref_output: Reference model output
        """
        # Run simulator
        sim_output = self.forward(
            features, intrinsics, extrinsics, near, far, **kwargs
        )
        
        # Run reference model
        with torch.no_grad():
            ref_output = reference_model(
                features, intrinsics, extrinsics, near, far, **kwargs
            )
        
        return sim_output, ref_output
    
    def _generate_depth_candidates(
        self,
        near: torch.Tensor,
        far: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Generate depth candidate values.
        
        Args:
            near: Near plane [B] or [B, V]
            far: Far plane [B] or [B, V]
            num_samples: Number of depth samples
            
        Returns:
            depth_candidates: [B, D] or [VB, D]
        """
        if self.config.use_inverse_depth:
            # Inverse depth (disparity) sampling
            min_depth = 1.0 / far
            max_depth = 1.0 / near
        else:
            # Linear depth sampling
            min_depth = near
            max_depth = far
        
        # Handle different input shapes
        if min_depth.dim() == 1:
            min_depth = min_depth.unsqueeze(1)
            max_depth = max_depth.unsqueeze(1)
        elif min_depth.dim() == 2:
            # [B, V] -> flatten to [B*V, 1]
            min_depth = min_depth.reshape(-1, 1)
            max_depth = max_depth.reshape(-1, 1)
        
        # Linear interpolation
        t = torch.linspace(0.0, 1.0, num_samples, device=min_depth.device)
        depth_candidates = min_depth + t.unsqueeze(0) * (max_depth - min_depth)
        
        return depth_candidates
    
    def get_cycle_breakdown(self) -> CycleBreakdown:
        """Get cycle breakdown for last forward pass."""
        return self._cycle_breakdown
    
    def get_total_cycles(self) -> int:
        """Get total cycles from all components."""
        return self._cycle_breakdown.total
    
    def reset_cycles(self):
        """Reset all cycle counters."""
        self._cycle_breakdown = CycleBreakdown()
        self.cost_volume_sim.reset_cycles()
        self.unet_sim.reset_cycles()
        self.depth_head_sim.reset_cycles()
        self.bilinear_unit.reset_cycles()


class PassThroughDepthPredictorSim(BaseDepthPredictorSim):
    """
    Pass-through depth predictor that uses original model
    but tracks cycles for hardware estimation.
    
    This is useful for initial integration where we want:
    1. Correct output (matching original model)
    2. Cycle counting for hardware estimation
    """
    
    def __init__(
        self,
        config: DepthPredictorConfig,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """Initialize pass-through predictor."""
        super().__init__(config, device)
        self._model = None
    
    def load_from_model(self, model: nn.Module) -> None:
        """Store reference to original model."""
        self._model = model
        self._initialized = True
    
    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        **kwargs,
    ) -> DepthPredictorOutput:
        """
        Forward pass using original model with cycle tracking.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_from_model first.")
        
        # Run original model
        with torch.no_grad():
            output = self._model(features, intrinsics, extrinsics, near, far, **kwargs)
        
        # Estimate cycles
        self._estimate_cycles(features)
        
        # Parse output based on model type
        if isinstance(output, tuple):
            if len(output) >= 3:
                depths, densities, raw_gaussians = output[:3]
            elif len(output) == 2:
                depths, raw_gaussians = output
                densities = None
            else:
                depths = output[0]
                densities = None
                raw_gaussians = None
        else:
            depths = output
            densities = None
            raw_gaussians = None
        
        return DepthPredictorOutput(
            depths=depths,
            densities=densities,
            raw_gaussians=raw_gaussians,
            total_cycles=self._cycle_breakdown.total,
            cycle_breakdown=self._cycle_breakdown,
        )
    
    def _estimate_cycles(self, features: torch.Tensor):
        """Estimate cycles based on input size."""
        B, V, C, H, W = features.shape
        D = self.config.num_depth_candidates
        
        # Rough estimates based on typical depth predictor
        self._cycle_breakdown = CycleBreakdown(
            cost_volume=V * D * H * W * C * 10,  # Warping + correlation
            unet_refinement=H * W * 128 * 128 * 9 // 256,  # Conv operations
            depth_head=D * H * W * 4 * 2,  # Conv + activation
            softmax_regression=D * H * W * 5,  # Softmax + weighted sum
            depth_refinement=H * W * 64 * 64 * 9 // 256 if self.config.use_depth_refinement else 0,
            upsampling=4 * 4 * H * W * 4,  # Bilinear 4x
            gaussian_head=H * W * 4 * 128 * 3 // 256,  # Output projection
        )
