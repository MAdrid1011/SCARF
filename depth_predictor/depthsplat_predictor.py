"""
DepthSplat Depth Predictor Simulator

Hardware simulator for DepthSplat's MultiViewUniMatch depth predictor.

Pipeline (more complex than Transplat/MVSplat):
1. Multi-scale feature extraction (CNN + Transformer + ViT)
2. Multi-scale cost volume construction
3. Per-scale cost volume regression (U-Net)
4. Multi-scale depth head with softmax
5. DPTHead learned upsampler (simplified)
6. Gaussian head

DepthSplat uses multi-scale processing for higher accuracy.
"""

import logging

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List

logger = logging.getLogger(__name__)
from einops import rearrange

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from .base_predictor import BaseDepthPredictorSim
from .types import DepthPredictorConfig, DepthPredictorOutput, CycleBreakdown
from .hw_depth_predictor import HWDepthPredictor
from encoder import ConvEngine, BilinearUnit, ActivationUnit, GEMMUnit
from encoder.types import CycleStats, ActivationType


class DepthSplatDepthPredictorSim(BaseDepthPredictorSim):
    """
    Hardware simulator for DepthSplat's depth predictor.
    
    DepthSplat uses MultiViewUniMatch which has:
    - Multi-scale cost volume processing
    - CNN + Transformer + ViT feature extraction
    - DPTHead for learned upsampling
    
    More complex than Transplat/MVSplat but achieves higher accuracy.
    
    Hardware Mapping:
        - Multi-scale features: ConvEngine + GEMMUnit
        - Cost volume: BilinearUnit + correlation
        - Regression: ConvEngine + NormUnit
        - DPTHead: ConvEngine + BilinearUnit
    
    IMPORTANT: When use_hw_computation=True, uses ACTUAL hardware compute units.
    Note that DepthSplat is more complex and requires images as input.
    """
    
    def __init__(
        self,
        config: Optional[DepthPredictorConfig] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_hw_computation: bool = True,
    ):
        """
        Initialize DepthSplat depth predictor simulator.
        
        Args:
            config: Depth predictor configuration (uses depthsplat preset if None)
            device: Computation device
            use_hw_computation: If True, use hardware units for actual computation.
        """
        config = config or DepthPredictorConfig.depthsplat_preset()
        super().__init__(config, device)
        
        self._use_hw_computation = use_hw_computation
        
        # Model reference (for pass-through and feature extraction)
        self._depth_predictor = None
        
        # Hardware depth predictor
        self._hw_predictor = HWDepthPredictor(config, device, model_type='depthsplat')
        
        # Hardware units
        self.conv_engine = ConvEngine()
        self.bilinear_unit = BilinearUnit()
        self.gemm_unit = GEMMUnit()
        self.gelu_unit = ActivationUnit(ActivationType.GELU)
        self.sigmoid_unit = ActivationUnit(ActivationType.SIGMOID)
    
    def load_from_model(self, model: nn.Module) -> None:
        """
        Load weights from DepthSplat encoder model.
        
        Args:
            model: DepthSplat encoder with depth_predictor attribute
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
        cnn_features: Optional[torch.Tensor] = None,
        mv_features: Optional[torch.Tensor] = None,
        mono_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DepthPredictorOutput:
        """
        Forward pass through DepthSplat depth predictor with cycle tracking.
        
        IMPORTANT: When use_hw_computation=True:
            - If `features` is provided, uses hardware compute units
            - Otherwise, falls back to original model
        
        DepthSplat's MultiViewUniMatch has a DIFFERENT interface:
            - First arg is `images`, not `features`
            - Uses `min_depth`, `max_depth` (inverse depth), not `near`, `far`
            - Returns a dict, not a tuple
        
        Args:
            features: Combined features [B, V, C, H, W] - Used by HW predictor
            intrinsics: Camera intrinsics [B, V, 3, 3]
            extrinsics: Camera extrinsics [B, V, 4, 4]
            near: Near plane [B] or [B, V] - will be converted to max_depth=1/near
            far: Far plane [B] or [B, V] - will be converted to min_depth=1/far
            images: Input images [B, V, C, H, W] - REQUIRED for pass-through
            cnn_features: (unused, MultiViewUniMatch computes internally)
            mv_features: (unused, MultiViewUniMatch computes internally)
            mono_features: (unused, MultiViewUniMatch computes internally)
            
        Returns:
            DepthPredictorOutput with depths, densities (match_prob), raw_gaussians=None
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_from_model first.")
        
        # Reset cycle tracking
        self.reset_cycles()
        
        if self._use_hw_computation:
            # ===== HARDWARE COMPUTATION PATH =====
            # DepthSplat HW simulation processes images through the entire pipeline
            # using hardware units (CNN backbone + MV Transformer + DINOv2 + cost volume + etc.)
            # No original model forward calls are made.
            if features is None or (isinstance(features, str) and features == 'depthsplat_integrated'):
                if images is None:
                    raise ValueError("DepthSplat HW mode requires images")
                
                # Create dummy features tensor for interface compatibility
                # The actual computation uses images directly via _forward_depthsplat_hw
                B, V, C_img, H_img, W_img = images.shape
                feat_h = H_img // 4  # 1/4 resolution (upsample_factor=4)
                feat_w = W_img // 4
                features = torch.zeros(B, V, 128, feat_h, feat_w, device=images.device)
                logger.debug("DepthSplat HW: processing images %s through full HW pipeline", images.shape)
            
            # Determine target resolution from images
            target_resolution = None
            if images is not None:
                H_img, W_img = images.shape[-2:]
                target_resolution = (H_img, W_img)
            
            # Use pre-extracted features with hardware depth predictor
            with torch.no_grad():
                output = self._hw_predictor.forward(
                    features, intrinsics, extrinsics, near, far, 
                    images=images, target_resolution=target_resolution, **kwargs
                )
            
            self._cycle_breakdown = output.cycle_breakdown
            return output
        else:
            # ===== PASS-THROUGH PATH (original model) =====
            # DepthSplat requires images for the original model
            if images is None:
                raise ValueError("DepthSplat depth predictor requires 'images' argument for pass-through mode")
            
            if self._depth_predictor is None:
                raise RuntimeError("Original model not loaded for pass-through mode.")
            
            B, V, C, H, W = images.shape
            
            with torch.no_grad():
                depths, densities, raw_gaussians = self._run_original_model(
                    images, intrinsics, extrinsics, near, far, **kwargs
                )
            
            # Estimate cycles for each stage (use images shape)
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
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run original DepthSplat MultiViewUniMatch - ACTUAL COMPUTATION.
        
        MultiViewUniMatch.forward signature:
            forward(images, attn_splits_list=None, intrinsics=None, 
                    min_depth=1.0/0.5, max_depth=1.0/100, num_depth_candidates=128,
                    extrinsics=None, nn_matrix=None, **kwargs)
                    
        Returns results_dict with:
            - depth_preds: list of [B, V, H, W] depth maps (multi-scale)
            - match_probs: list of [BV, D, H, W] probability volumes
            - features_mono_intermediate, features_cnn_all_scales, features_mv, etc.
        """
        B, V, C, H, W = images.shape
        device = images.device
        
        # Convert near/far to inverse depth (min_depth, max_depth)
        # MultiViewUniMatch uses: min_depth = 1/far, max_depth = 1/near
        # Expected shape: [B, V] for broadcast to [BV]
        
        # Ensure near and far are [B, V] tensors
        if near.dim() == 0:
            near_t = near.expand(B, V)
        elif near.dim() == 1:
            if near.shape[0] == B:
                near_t = near.unsqueeze(1).expand(B, V)
            elif near.shape[0] == V:
                near_t = near.unsqueeze(0).expand(B, V)
            else:
                near_t = near.mean().unsqueeze(0).unsqueeze(0).expand(B, V)
        else:
            near_t = near
            
        if far.dim() == 0:
            far_t = far.expand(B, V)
        elif far.dim() == 1:
            if far.shape[0] == B:
                far_t = far.unsqueeze(1).expand(B, V)
            elif far.shape[0] == V:
                far_t = far.unsqueeze(0).expand(B, V)
            else:
                far_t = far.mean().unsqueeze(0).unsqueeze(0).expand(B, V)
        else:
            far_t = far
        
        # Ensure on correct device and clamp to avoid division by zero
        near_t = near_t.to(device).clamp(min=1e-6)
        far_t = far_t.to(device).clamp(min=1e-6)
        
        # Inverse depth: min_depth = 1/far, max_depth = 1/near
        min_depth = 1.0 / far_t  # [B, V]
        max_depth = 1.0 / near_t  # [B, V]
        
        # Build nn_matrix for nearest neighbor matching if V > 3
        nn_matrix = None
        if V > 3:
            xyzs = extrinsics[:, :, :3, -1].detach()
            cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
            cameras_dist_index = torch.argsort(cameras_dist_matrix)
            nn_matrix = cameras_dist_index[:, :, :4]  # top 4 nearest
        
        # Call MultiViewUniMatch.forward - THIS IS ACTUAL COMPUTATION
        results_dict = self._depth_predictor(
            images,
            attn_splits_list=[2],  # Standard setting
            intrinsics=intrinsics,
            min_depth=min_depth,
            max_depth=max_depth,
            extrinsics=extrinsics,
            nn_matrix=nn_matrix,
        )
        
        # Extract outputs from results_dict
        # depth_preds is a list of [B, V, H, W] tensors (multi-scale)
        depth_preds = results_dict.get('depth_preds', [])
        if len(depth_preds) > 0:
            # Use final (highest resolution) depth
            depth_final = depth_preds[-1]  # [B, V, H, W]
            # Convert to standard format [B, V, H*W, 1, 1]
            depths = rearrange(depth_final, "b v h w -> b v (h w) () ()")
        else:
            depths = None
        
        # match_probs contains the softmax probability over depth candidates
        # Use as densities (confidence)
        match_probs = results_dict.get('match_probs', [])
        if len(match_probs) > 0:
            # [BV, D, H, W] -> take max prob as density
            match_prob = match_probs[-1]
            match_prob_max = torch.max(match_prob, dim=1, keepdim=True)[0]  # [BV, 1, H, W]
            # Upsample to image resolution if needed (BilinearUnit)
            if match_prob_max.shape[-2:] != (H, W):
                match_prob_max, _ = self.bilinear_unit.interpolate(
                    match_prob_max, size=(H, W), mode='nearest'
                )
            # Convert to standard format [B, V, H*W, 1, 1]
            densities = rearrange(match_prob_max, "(b v) c h w -> b v (c h w) () ()", b=B, v=V)
        else:
            densities = None
        
        # DepthSplat's raw_gaussians are computed in encoder, not depth_predictor
        # Return None here; gaussian_head output is captured separately
        raw_gaussians = None
        
        return depths, densities, raw_gaussians
    
    def _estimate_pipeline_cycles(
        self,
        features: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ):
        """
        Estimate cycles for DepthSplat depth prediction pipeline.
        
        DepthSplat is more complex due to multi-scale processing.
        """
        B, V, C, H, W = features.shape
        D = self.config.num_depth_candidates  # 128 for DepthSplat
        
        H_full = images.shape[-2] if images is not None else H * 4
        W_full = images.shape[-1] if images is not None else W * 4
        
        # DepthSplat uses multiple scales
        scales = [1, 2, 4]  # 1/4, 1/8, 1/16 of original resolution
        
        total_cost_volume_cycles = 0
        total_unet_cycles = 0
        total_depth_head_cycles = 0
        total_softmax_cycles = 0
        
        for scale in scales:
            h_scale = H // scale
            w_scale = W // scale
            d_scale = D // scale  # Fewer depth candidates at coarser scales
            
            # Cost volume construction per scale
            # Warping: BilinearUnit
            warp_cycles = (V - 1) * B * d_scale * 4 * h_scale * w_scale
            
            # Correlation
            corr_cycles = (V - 1) * B * d_scale * 4 * C * h_scale * w_scale
            
            total_cost_volume_cycles += warp_cycles + corr_cycles
            
            # U-Net regression per scale
            unet_channels = [128, 256]
            h, w = h_scale, w_scale
            for ch in unet_channels:
                total_unet_cycles += V * B * ch * ch * 9 * h * w // 256
                h, w = max(1, h // 2), max(1, w // 2)
            
            # Depth head per scale
            total_depth_head_cycles += V * B * (
                d_scale * 2 * d_scale * h_scale * w_scale // 256 +
                2 * d_scale * h_scale * w_scale +
                2 * d_scale * d_scale * h_scale * w_scale // 256
            )
            
            # Softmax + regression per scale
            total_softmax_cycles += V * B * 5 * d_scale * h_scale * w_scale
        
        # DPTHead upsampler (learned upsampling)
        # Multi-scale feature fusion + convolutions
        dpt_channels = [256, 128, 64, 32]
        dpt_cycles = 0
        h, w = H // 4, W // 4
        
        for i, ch in enumerate(dpt_channels):
            # Fusion conv
            dpt_cycles += V * B * ch * ch * 3 * 3 * h * w // 256
            # Upsample
            if i < len(dpt_channels) - 1:
                h, w = h * 2, w * 2
        
        # Final depth refinement
        depth_refinement_cycles = V * B * 64 * 64 * 9 * H * W // 256
        
        # Gaussian head
        gaussian_raw_channels = 84
        gaussian_head_cycles = V * B * 64 * gaussian_raw_channels * 9 * H * W // 256
        
        self._cycle_breakdown = CycleBreakdown(
            cost_volume=total_cost_volume_cycles,
            unet_refinement=total_unet_cycles,
            depth_head=total_depth_head_cycles,
            softmax_regression=total_softmax_cycles,
            depth_refinement=depth_refinement_cycles + dpt_cycles,
            upsampling=dpt_cycles,  # DPTHead includes upsampling
            gaussian_head=gaussian_head_cycles,
        )


def create_depthsplat_predictor(
    model: nn.Module,
    config: Optional[DepthPredictorConfig] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> DepthSplatDepthPredictorSim:
    """
    Factory function to create DepthSplat depth predictor simulator.
    
    Args:
        model: DepthSplat encoder model
        config: Optional configuration
        device: Computation device
        
    Returns:
        Initialized depth predictor simulator
    """
    predictor = DepthSplatDepthPredictorSim(config, device)
    predictor.load_from_model(model)
    return predictor
