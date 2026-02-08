"""
Transplat Depth Predictor Simulator

Hardware simulator for Transplat's DepthPredictorTrans.

Pipeline:
1. Transformer-based cost volume matching (UVTransformer)
2. Cost volume refinement (2D U-Net)
3. Coarse depth estimation (depth head + softmax)
4. Depth refinement (refinement U-Net)
5. Gaussian head (output projection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from einops import rearrange, repeat

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from .base_predictor import BaseDepthPredictorSim
from .types import DepthPredictorConfig, DepthPredictorOutput, CycleBreakdown
from .unet_sim import SimplifiedUNetSim
from .depth_head_sim import SimplifiedDepthHeadSim
from .hw_depth_predictor import HWDepthPredictor
from encoder import ConvEngine, BilinearUnit, ActivationUnit, NormalizationUnit
from encoder.types import CycleStats, ActivationType


class TransplatDepthPredictorSim(BaseDepthPredictorSim):
    """
    Hardware simulator for Transplat's depth predictor.
    
    Transplat uses transformer-based cost volume matching:
    - UVTransformer for coarse + fine matching
    - 2D U-Net for cost volume refinement
    - Depth head with softmax regression
    - Refinement U-Net for full-resolution depth
    - Gaussian head for output
    
    Hardware Mapping:
        - Transformer matching: GEMMUnit for attention (~20K LUTs, 128 DSPs)
        - U-Net refinement: ConvEngine + NormUnit (~50K LUTs, 256 DSPs)
        - Depth head: ConvEngine (~10K LUTs, 64 DSPs)
        - Upsampling: BilinearUnit (~800 LUTs, 8 DSPs)
    """
    
    def __init__(
        self,
        config: Optional[DepthPredictorConfig] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_hw_computation: bool = True,
    ):
        """
        Initialize Transplat depth predictor simulator.
        
        Args:
            config: Depth predictor configuration (uses transplat preset if None)
            device: Computation device
            use_hw_computation: If True, use hardware predictor (either simulated or original-based).
                               If False, use original PyTorch model directly (pass-through).
        """
        config = config or DepthPredictorConfig.transplat_preset()
        super().__init__(config, device)
        
        self._use_hw_computation = use_hw_computation
        
        # Model reference for pass-through computation (fallback)
        self._depth_predictor = None
        
        # Hardware depth predictor for actual computation
        # Transplat uses transformer-based matching
        self._hw_predictor = HWDepthPredictor(config, device, model_type='transplat')
        
        # Control whether hw_predictor uses original model internally
        # When True: hw_predictor uses original model for accurate output + cycle tracking
        # When False: hw_predictor uses simplified hardware simulation (may have quality loss)
        self._use_accurate_hw = True  # Default to accurate mode
        
        # Hardware units for cycle tracking
        self.conv_engine = ConvEngine()
        self.bilinear_unit = BilinearUnit()
        self.gelu_unit = ActivationUnit(ActivationType.GELU)
        self.sigmoid_unit = ActivationUnit(ActivationType.SIGMOID)
    
    def load_from_model(self, model: nn.Module) -> None:
        """
        Load weights from Transplat encoder model.
        
        Args:
            model: Transplat encoder with depth_predictor attribute
        """
        if hasattr(model, 'depth_predictor'):
            self._depth_predictor = model.depth_predictor
        else:
            self._depth_predictor = model
        
        # Load weights into hardware predictor
        self._hw_predictor.load_from_model(model)
        
        # Sync config.num_depth_candidates from hw_predictor
        # (hw_predictor updates this from actual model during load_from_model)
        self.config.num_depth_candidates = self._hw_predictor.config.num_depth_candidates
        
        # Set hw_predictor to use original model for accurate output
        self._hw_predictor.set_use_original(self._use_accurate_hw)
        
        self._initialized = True
    
    def set_accurate_mode(self, accurate: bool) -> None:
        """
        Set whether to use accurate mode (original model) or pure hardware simulation.
        
        Args:
            accurate: If True, use original model for computation (accurate output).
                     If False, use simplified hardware simulation (may have quality loss).
        """
        self._use_accurate_hw = accurate
        if self._hw_predictor is not None:
            self._hw_predictor.set_use_original(accurate)
    
    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        da_depth: Optional[torch.Tensor] = None,
        dino_feature: Optional[torch.Tensor] = None,
        cnn_features: Optional[torch.Tensor] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DepthPredictorOutput:
        """
        Forward pass through Transplat depth predictor with cycle tracking.
        
        IMPORTANT: When use_hw_computation=True, this uses ACTUAL hardware compute units
        (ConvEngine, GEMMUnit, BilinearUnit, etc.) for computation. Results may differ
        from the original PyTorch model due to hardware approximations.
        
        Args:
            features: Input features [B, V, C, H, W]
            intrinsics: Camera intrinsics [B, V, 3, 3]
            extrinsics: Camera extrinsics [B, V, 4, 4]
            near: Near plane [B, V]
            far: Far plane [B, V]
            images: Input images [B, V, 3, H_full, W_full] (for refinement)
            da_depth: DepthAnything depth [B, V, 1, H, W] (REQUIRED for Transplat)
            dino_feature: DINO features [B, V, C, H, W] (REQUIRED for Transplat)
            cnn_features: CNN features (optional)
            extra_info: Additional info dict (contains images for depth predictor)
            
        Returns:
            DepthPredictorOutput with depths, densities, raw_gaussians
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_from_model first.")
        
        # Reset cycle tracking
        self.reset_cycles()
        
        B, V, C, H, W = features.shape
        D = self.config.num_depth_candidates
        
        if self._use_hw_computation:
            # ===== HARDWARE COMPUTATION PATH =====
            # Use hardware predictor (either accurate or simulated based on _use_accurate_hw)
            with torch.no_grad():
                output = self._hw_predictor.forward(
                    features, intrinsics, extrinsics, near, far,
                    images=images,  # Pass images directly
                    da_depth=da_depth,
                    dino_feature=dino_feature,
                    cnn_features=cnn_features,
                    extra_info=extra_info,
                    **kwargs
                )
            
            # Get cycle breakdown from hardware predictor
            self._cycle_breakdown = output.cycle_breakdown
            
            return output
        else:
            # ===== PASS-THROUGH PATH (fallback) =====
            # Run original model for correct output
            if self._depth_predictor is None:
                raise RuntimeError("Original model not loaded for pass-through mode.")
            
            with torch.no_grad():
                depths, densities, raw_gaussians = self._run_original_model(
                    features, intrinsics, extrinsics, near, far,
                    images, da_depth, dino_feature, 
                    cnn_features=cnn_features, extra_info=extra_info, **kwargs
                )
            
            # Track cycles based on actual computation performed
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
        images: Optional[torch.Tensor],
        da_depth: Optional[torch.Tensor],
        dino_feature: Optional[torch.Tensor],
        cnn_features: Optional[torch.Tensor] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run original Transplat depth predictor - ACTUAL COMPUTATION.
        
        This executes the real PyTorch model to get correct outputs.
        The outputs are then used in the real pipeline.
        """
        
        # Build kwargs for original model
        model_kwargs = {}
        
        # Required for Transplat
        if da_depth is not None:
            model_kwargs['da_depth'] = da_depth
        if dino_feature is not None:
            model_kwargs['dino_feature'] = dino_feature
        if cnn_features is not None:
            model_kwargs['cnn_features'] = cnn_features
        
        # Extra info (contains images)
        if extra_info is not None:
            model_kwargs['extra_info'] = extra_info
        elif images is not None:
            # Build extra_info from images if not provided
            from einops import rearrange
            model_kwargs['extra_info'] = {'images': rearrange(images, 'b v c h w -> (v b) c h w')}
        
        # Standard kwargs
        model_kwargs['gaussians_per_pixel'] = kwargs.get('gaussians_per_pixel', 1)
        model_kwargs['deterministic'] = kwargs.get('deterministic', True)
        
        # Call original model - THIS IS THE ACTUAL COMPUTATION
        output = self._depth_predictor(
            features, intrinsics, extrinsics, near, far, **model_kwargs
        )
        
        # Parse output
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
        Estimate cycles for Transplat depth prediction pipeline.
        
        Pipeline stages:
        1. encoder_4b: Cost volume matching (UVTransformer)
        2. encoder_4c: Cost volume U-Net
        3. encoder_4d: Coarse depth estimation
        4. encoder_4e: Depth refinement U-Net
        5. encoder_4f: Gaussian head
        """
        B, V, C, H, W = features.shape
        D = self.config.num_depth_candidates
        
        # Full resolution if images provided
        H_full = images.shape[-2] if images is not None else H * 4
        W_full = images.shape[-1] if images is not None else W * 4
        
        # Stage 1: Cost volume matching (transformer-based)
        # UVTransformer: coarse (1 layer) + fine (2 layers)
        # Each layer: QKV projection + attention + FFN
        embed_dim = 128  # typical
        transformer_cycles = 0
        
        # Coarse transformer (1 layer)
        # QKV: 3 * C * embed_dim MACs
        # Attention: H*W * H*W * embed_dim MACs
        # FFN: 2 * embed_dim * 4*embed_dim MACs
        coarse_cycles = V * B * (
            3 * C * embed_dim * H * W // 256 +  # QKV
            H * W * H * W * embed_dim // 256 +   # Attention
            2 * embed_dim * 4 * embed_dim * H * W // 256  # FFN
        )
        transformer_cycles += coarse_cycles
        
        # Fine transformer (2 layers)
        fine_cycles = 2 * coarse_cycles
        transformer_cycles += fine_cycles
        
        cost_volume_cycles = transformer_cycles
        
        # Stage 2: Cost volume U-Net refinement
        # Typical U-Net: encoder + bottleneck + decoder
        # ~3M MACs for 128x128 feature map
        unet_channels = [128, 256, 256]
        unet_cycles = 0
        
        h, w = H, W
        for i, ch in enumerate(unet_channels):
            # Encoder conv
            unet_cycles += V * B * ch * ch * 9 * h * w // 256
            if i < len(unet_channels) - 1:
                h, w = h // 2, w // 2
        
        # Decoder (mirror)
        for i, ch in enumerate(reversed(unet_channels[:-1])):
            h, w = h * 2, w * 2
            unet_cycles += V * B * ch * ch * 9 * h * w // 256
        
        # Stage 3: Coarse depth estimation
        # depth_head: Conv(D, 2D, 1x1) + GELU + Conv(2D, D, 1x1)
        depth_head_cycles = V * B * (
            D * 2 * D * H * W // 256 +  # Conv1
            2 * D * H * W +              # GELU
            2 * D * D * H * W // 256     # Conv2
        )
        
        # Softmax + regression
        softmax_cycles = V * B * 3 * D * H * W  # exp + sum + div
        regression_cycles = V * B * 2 * D * H * W  # multiply + sum
        
        # Upsampling 4x
        upsampling_cycles = V * B * 4 * H * W * 4 * 4  # bilinear 4x
        
        # Stage 4: Depth refinement U-Net
        if self.config.use_depth_refinement:
            # Feature upsampler
            upsampler_cycles = V * B * C * C * 9 * H * 2 * W * 2 // 256
            
            # Refinement U-Net (at full resolution)
            refine_channels = [64, 128, 128]
            refine_cycles = 0
            h, w = H_full // 4, W_full // 4  # Quarter resolution
            for ch in refine_channels:
                refine_cycles += V * B * ch * ch * 9 * h * w // 256
            
            depth_refinement_cycles = upsampler_cycles + refine_cycles
        else:
            depth_refinement_cycles = 0
        
        # Stage 5: Gaussian head
        # Output projection: Conv(feat_dim, gaussian_raw_channels, 3x3)
        gaussian_raw_channels = 84  # typical
        gaussian_head_cycles = V * B * 64 * gaussian_raw_channels * 9 * H_full // 4 * W_full // 4 // 256
        
        # Store breakdown
        self._cycle_breakdown = CycleBreakdown(
            cost_volume=cost_volume_cycles,
            unet_refinement=unet_cycles,
            depth_head=depth_head_cycles,
            softmax_regression=softmax_cycles + regression_cycles,
            depth_refinement=depth_refinement_cycles,
            upsampling=upsampling_cycles,
            gaussian_head=gaussian_head_cycles,
        )


def create_transplat_predictor(
    model: nn.Module,
    config: Optional[DepthPredictorConfig] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> TransplatDepthPredictorSim:
    """
    Factory function to create Transplat depth predictor simulator.
    
    Args:
        model: Transplat encoder model
        config: Optional configuration
        device: Computation device
        
    Returns:
        Initialized depth predictor simulator
    """
    predictor = TransplatDepthPredictorSim(config, device)
    predictor.load_from_model(model)
    return predictor
