"""
MVSplat Depth Predictor Simulator

Hardware simulator for MVSplat's DepthPredictorMultiView.

Pipeline:
1. Direct correlation cost volume (feature warping + dot product)
2. Cost volume refinement (2D U-Net)
3. Coarse depth estimation (depth head + softmax)
4. Depth refinement (refinement U-Net)
5. Gaussian head (output projection)

Simpler than Transplat: no transformer matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from einops import rearrange

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from .base_predictor import BaseDepthPredictorSim
from .types import DepthPredictorConfig, DepthPredictorOutput, CycleBreakdown
from .hw_depth_predictor import HWDepthPredictor
from encoder import ConvEngine, BilinearUnit, ActivationUnit
from encoder.types import CycleStats, ActivationType


class MVSplatDepthPredictorSim(BaseDepthPredictorSim):
    """
    Hardware simulator for MVSplat's depth predictor.
    
    MVSplat uses direct correlation for cost volume:
    - Feature warping via camera projection
    - Normalized dot product correlation
    - 2D U-Net for refinement
    - Softmax depth regression
    
    Hardware Mapping:
        - Feature warping: BilinearUnit (~800 LUTs, 8 DSPs)
        - Correlation: Element-wise multiply + sum (~500 LUTs, 32 DSPs)
        - U-Net: ConvEngine + NormUnit (~50K LUTs, 256 DSPs)
        - Depth head: ConvEngine (~10K LUTs, 64 DSPs)
    
    IMPORTANT: When use_hw_computation=True, uses ACTUAL hardware compute units.
    """
    
    def __init__(
        self,
        config: Optional[DepthPredictorConfig] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_hw_computation: bool = True,
    ):
        """
        Initialize MVSplat depth predictor simulator.
        
        Args:
            config: Depth predictor configuration (uses mvsplat preset if None)
            device: Computation device
            use_hw_computation: If True, use hardware units for actual computation.
        """
        config = config or DepthPredictorConfig.mvsplat_preset()
        super().__init__(config, device)
        
        self._use_hw_computation = use_hw_computation
        
        # Model reference (fallback)
        self._depth_predictor = None
        
        # Hardware depth predictor
        self._hw_predictor = HWDepthPredictor(config, device, model_type='mvsplat')
        
        # Hardware units
        self.conv_engine = ConvEngine()
        self.bilinear_unit = BilinearUnit()
        self.gelu_unit = ActivationUnit(ActivationType.GELU)
        self.sigmoid_unit = ActivationUnit(ActivationType.SIGMOID)
    
    def load_from_model(self, model: nn.Module) -> None:
        """
        Load weights from MVSplat encoder model.
        
        Args:
            model: MVSplat encoder with depth_predictor attribute
        """
        if hasattr(model, 'depth_predictor'):
            self._depth_predictor = model.depth_predictor
        else:
            self._depth_predictor = model
        
        # Load weights into hardware predictor
        self._hw_predictor.load_from_model(model)
        
        self._initialized = True
    
    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DepthPredictorOutput:
        """
        Forward pass through MVSplat depth predictor with cycle tracking.
        
        IMPORTANT: When use_hw_computation=True, uses ACTUAL hardware compute units
        (ConvEngine, GEMMUnit, BilinearUnit, etc.) for computation.
        
        Args:
            features: Input features [B, V, C, H, W]
            intrinsics: Camera intrinsics [B, V, 3, 3]
            extrinsics: Camera extrinsics [B, V, 4, 4]
            near: Near plane [B, V]
            far: Far plane [B, V]
            images: Input images [B, V, 3, H_full, W_full] (for refinement)
            
        Returns:
            DepthPredictorOutput with depths, densities, raw_gaussians
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_from_model first.")
        
        # Reset cycle tracking
        self.reset_cycles()
        
        B, V, C, H, W = features.shape
        
        if self._use_hw_computation:
            # ===== HARDWARE COMPUTATION PATH =====
            with torch.no_grad():
                output = self._hw_predictor.forward(
                    features, intrinsics, extrinsics, near, far, **kwargs
                )
            
            self._cycle_breakdown = output.cycle_breakdown
            return output
        else:
            # ===== PASS-THROUGH PATH (fallback) =====
            if self._depth_predictor is None:
                raise RuntimeError("Original model not loaded for pass-through mode.")
            
            with torch.no_grad():
                depths, densities, raw_gaussians = self._run_original_model(
                    features, intrinsics, extrinsics, near, far, images, **kwargs
                )
            
            self._estimate_pipeline_cycles(features, images)
            
            return DepthPredictorOutput(
                depths=depths,
                densities=densities,
                raw_gaussians=raw_gaussians,
                total_cycles=self._cycle_breakdown.total,
                cycle_breakdown=self._cycle_breakdown,
            )
    
    def _run_original_model(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run original MVSplat depth predictor - ACTUAL COMPUTATION.
        
        MVSplat's DepthPredictorMultiView does NOT accept 'images' argument.
        """
        
        # Filter kwargs to only supported ones
        model_kwargs = {}
        supported_kwargs = ['gaussians_per_pixel', 'deterministic', 'extra_info', 'cnn_features']
        for key in supported_kwargs:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]
        
        # Call original model - THIS IS ACTUAL COMPUTATION
        output = self._depth_predictor(
            features, intrinsics, extrinsics, near, far, **model_kwargs
        )
        
        if isinstance(output, tuple):
            if len(output) >= 3:
                depths, densities, raw_gaussians = output[:3]
            else:
                depths = output[0]
                densities = output[1] if len(output) > 1 else None
                raw_gaussians = None
        else:
            depths = output
            densities = None
            raw_gaussians = None
        
        return depths, densities, raw_gaussians
    
    def _estimate_pipeline_cycles(
        self,
        features: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ):
        """
        Estimate cycles for MVSplat depth prediction pipeline.
        
        Simpler than Transplat: direct correlation instead of transformer.
        """
        B, V, C, H, W = features.shape
        D = self.config.num_depth_candidates
        
        H_full = images.shape[-2] if images is not None else H * 4
        W_full = images.shape[-1] if images is not None else W * 4
        
        # Stage 1: Cost volume construction (correlation-based)
        # For each depth candidate:
        #   - Feature warping: bilinear interpolation
        #   - Correlation: normalized dot product
        
        # Warping cycles: per depth, per source view
        # BilinearUnit: 4 cycles per output pixel
        warp_cycles_per_depth = (V - 1) * B * 4 * H * W
        warp_cycles = D * warp_cycles_per_depth
        
        # Correlation cycles: normalize + dot product
        # Normalize: sqrt(sum_squares) + divide = C + C cycles
        # Dot product: C multiplies + C-1 adds
        corr_cycles_per_depth = (V - 1) * B * (2 * C + 2 * C - 1) * H * W
        corr_cycles = D * corr_cycles_per_depth
        
        cost_volume_cycles = warp_cycles + corr_cycles
        
        # Stage 2: Cost volume U-Net refinement
        # Input: [VB, D + C, H, W] (cost volume + features)
        input_channels = D + C
        unet_channels = [128, 256, 256]
        unet_cycles = 0
        
        h, w = H, W
        in_ch = input_channels
        for i, ch in enumerate(unet_channels):
            unet_cycles += V * B * in_ch * ch * 9 * h * w // 256
            in_ch = ch
            if i < len(unet_channels) - 1:
                h, w = h // 2, w // 2
        
        for i, ch in enumerate(reversed(unet_channels[:-1])):
            h, w = h * 2, w * 2
            unet_cycles += V * B * ch * ch * 9 * h * w // 256
        
        # Stage 3: Coarse depth estimation
        depth_head_cycles = V * B * (
            D * 2 * D * H * W // 256 +  # Conv1
            2 * D * H * W +              # GELU
            2 * D * D * H * W // 256     # Conv2
        )
        
        softmax_cycles = V * B * 3 * D * H * W
        regression_cycles = V * B * 2 * D * H * W
        
        upsampling_cycles = V * B * 4 * H * W * 4 * 4
        
        # Stage 4: Depth refinement
        if self.config.use_depth_refinement:
            upsampler_cycles = V * B * C * C * 9 * H * 2 * W * 2 // 256
            refine_channels = [64, 128, 128]
            refine_cycles = 0
            h, w = H_full // 4, W_full // 4
            for ch in refine_channels:
                refine_cycles += V * B * ch * ch * 9 * h * w // 256
            
            depth_refinement_cycles = upsampler_cycles + refine_cycles
        else:
            depth_refinement_cycles = 0
        
        # Stage 5: Gaussian head
        gaussian_raw_channels = 84
        gaussian_head_cycles = V * B * 64 * gaussian_raw_channels * 9 * H_full // 4 * W_full // 4 // 256
        
        self._cycle_breakdown = CycleBreakdown(
            cost_volume=cost_volume_cycles,
            unet_refinement=unet_cycles,
            depth_head=depth_head_cycles,
            softmax_regression=softmax_cycles + regression_cycles,
            depth_refinement=depth_refinement_cycles,
            upsampling=upsampling_cycles,
            gaussian_head=gaussian_head_cycles,
        )


def create_mvsplat_predictor(
    model: nn.Module,
    config: Optional[DepthPredictorConfig] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> MVSplatDepthPredictorSim:
    """
    Factory function to create MVSplat depth predictor simulator.
    
    Args:
        model: MVSplat encoder model
        config: Optional configuration
        device: Computation device
        
    Returns:
        Initialized depth predictor simulator
    """
    predictor = MVSplatDepthPredictorSim(config, device)
    predictor.load_from_model(model)
    return predictor
