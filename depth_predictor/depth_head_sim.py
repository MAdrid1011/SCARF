"""
Depth Head Hardware Simulator

Simulates depth head: Conv + Activation + Conv + Softmax + Depth Regression.

Hardware Units:
- ConvEngine: 1x1 or 3x3 convolutions
- ActivationUnit: GELU activation
- Custom softmax + weighted sum for depth regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # Used in reference/comparison methods
from typing import Tuple, Optional, Dict

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from encoder import ConvEngine, ActivationUnit, SoftmaxUnit
from encoder.types import CycleStats, ActivationType
from .types import DepthPredictorConfig, DepthRegressionType


class DepthHeadSimulator:
    """
    Depth head hardware simulator.
    
    Architecture (typical):
        Input [B, D, H, W] → Conv1 [B, 2D, H, W] → GELU → Conv2 [B, D, H, W]
                                                              ↓
                                                    Softmax → Depth Regression
    
    Hardware Mapping:
        - Conv1, Conv2: ConvEngine (1x1 kernels)
        - GELU: ActivationUnit (LUT-based)
        - Softmax: Custom hardware (~300 LUTs)
        - Depth regression: Weighted sum (~100 LUTs)
    """
    
    def __init__(
        self,
        num_depth_candidates: int,
        config: Optional[DepthPredictorConfig] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """
        Initialize depth head simulator.
        
        Args:
            num_depth_candidates: Number of depth hypotheses (D)
            config: Depth predictor configuration
            device: Computation device
        """
        self.num_depth = num_depth_candidates
        self.config = config or DepthPredictorConfig()
        self.device = device
        
        # Hardware units
        self.conv_engine = ConvEngine()
        self.activation_unit = ActivationUnit(ActivationType.GELU)
        self.softmax_unit = SoftmaxUnit()
        
        # PyTorch layers for weight storage
        self.conv1 = nn.Conv2d(num_depth_candidates, 2 * num_depth_candidates, 1)
        self.conv2 = nn.Conv2d(2 * num_depth_candidates, num_depth_candidates, 1)
        
        self._initialized = False
        self._total_cycles = 0
    
    def load_weights(
        self,
        depth_head: nn.Module,
    ):
        """
        Load weights from original depth head module.
        
        Args:
            depth_head: Original depth head (typically nn.Sequential)
        """
        # Extract weights from typical depth head structure:
        # Sequential(Conv2d, GELU, Conv2d)
        if isinstance(depth_head, nn.Sequential):
            layers = list(depth_head.children())
            if len(layers) >= 3:
                if isinstance(layers[0], nn.Conv2d):
                    self.conv1.weight.data = layers[0].weight.data.to(self.device)
                    if layers[0].bias is not None:
                        self.conv1.bias.data = layers[0].bias.data.to(self.device)
                if isinstance(layers[2], nn.Conv2d):
                    self.conv2.weight.data = layers[2].weight.data.to(self.device)
                    if layers[2].bias is not None:
                        self.conv2.bias.data = layers[2].bias.data.to(self.device)
        elif hasattr(depth_head, 'conv1') and hasattr(depth_head, 'conv2'):
            self.conv1.weight.data = depth_head.conv1.weight.data.to(self.device)
            if depth_head.conv1.bias is not None:
                self.conv1.bias.data = depth_head.conv1.bias.data.to(self.device)
            self.conv2.weight.data = depth_head.conv2.weight.data.to(self.device)
            if depth_head.conv2.bias is not None:
                self.conv2.bias.data = depth_head.conv2.bias.data.to(self.device)
        
        self._initialized = True
    
    def forward(
        self,
        cost_volume: torch.Tensor,
        depth_candidates: torch.Tensor,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, CycleStats, Optional[torch.Tensor]]:
        """
        Forward pass with cycle tracking.
        
        Args:
            cost_volume: Cost/correlation volume [B, D, H, W]
            depth_candidates: Depth values [D] or [B, D]
            return_probs: Whether to return softmax probabilities
            
        Returns:
            depths: Regressed depth values [B, H, W]
            cycles: Cycle statistics
            probs: (optional) Softmax probabilities [B, D, H, W]
        """
        B, D, H, W = cost_volume.shape
        total_cycles = CycleStats()
        
        # Conv1: D → 2D
        x, conv1_cycles = self.conv_engine.forward(
            cost_volume, self.conv1.weight,
            bias=self.conv1.bias,
            stride=1, padding=0,
        )
        total_cycles = total_cycles + conv1_cycles
        
        # GELU activation
        x, act_cycles = self.activation_unit.forward(x)
        total_cycles = total_cycles + act_cycles
        
        # Conv2: 2D → D
        logits, conv2_cycles = self.conv_engine.forward(
            x, self.conv2.weight,
            bias=self.conv2.bias,
            stride=1, padding=0,
        )
        total_cycles = total_cycles + conv2_cycles
        
        # Softmax over depth dimension (SoftmaxUnit)
        probs, softmax_cycles = self.softmax_unit.forward_with_temperature(
            logits, self.config.softmax_temperature, dim=1
        )
        total_cycles = total_cycles + softmax_cycles
        
        # Depth regression: weighted sum
        depths, regression_cycles = self._depth_regression(probs, depth_candidates)
        total_cycles = total_cycles + regression_cycles
        
        self._total_cycles += total_cycles.total_cycles
        
        if return_probs:
            return depths, total_cycles, probs
        return depths, total_cycles, None
    
    def _depth_regression(
        self,
        probs: torch.Tensor,
        depth_candidates: torch.Tensor,
    ) -> Tuple[torch.Tensor, CycleStats]:
        """
        Depth regression via weighted sum.
        
        depth = sum(prob_d * depth_d) for d in [0, D)
        
        Args:
            probs: Softmax probabilities [B, D, H, W]
            depth_candidates: Depth values [D] or [B, D]
            
        Returns:
            depths: Regressed depths [B, H, W]
            cycles: Cycle statistics
        """
        B, D, H, W = probs.shape
        
        # Reshape depth candidates for broadcasting
        if depth_candidates.dim() == 1:
            depth_candidates = depth_candidates.view(1, D, 1, 1)
        elif depth_candidates.dim() == 2:
            depth_candidates = depth_candidates.view(B, D, 1, 1)
        
        # Weighted sum
        depths = (probs * depth_candidates).sum(dim=1)  # [B, H, W]
        
        # Cycle estimation: D multiplies + D-1 adds per pixel
        regression_cycles = CycleStats(
            total_cycles=(2 * D - 1) * H * W * B,
            compute_cycles=(2 * D - 1) * H * W * B,
            breakdown={'regression': (2 * D - 1) * H * W * B}
        )
        
        return depths, regression_cycles
    
    def forward_with_reference(
        self,
        cost_volume: torch.Tensor,
        depth_candidates: torch.Tensor,
        reference_head: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, CycleStats]:
        """
        Forward with comparison to reference implementation.
        
        Args:
            cost_volume: [B, D, H, W]
            depth_candidates: [D] or [B, D]
            reference_head: Original depth head module
            
        Returns:
            sim_depths: Simulator output
            ref_depths: Reference output  
            cycles: Cycle statistics
        """
        # Reference computation
        with torch.no_grad():
            ref_logits = reference_head(cost_volume)
            ref_probs = F.softmax(ref_logits, dim=1)
            
            if depth_candidates.dim() == 1:
                d_cand = depth_candidates.view(1, -1, 1, 1)
            else:
                d_cand = depth_candidates.view(cost_volume.shape[0], -1, 1, 1)
            
            ref_depths = (ref_probs * d_cand).sum(dim=1)
        
        # Simulator computation
        sim_depths, cycles, _ = self.forward(cost_volume, depth_candidates)
        
        return sim_depths, ref_depths, cycles
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0


class SimplifiedDepthHeadSim:
    """
    Simplified depth head that uses original model computation
    but tracks cycles for hardware estimation.
    """
    
    def __init__(
        self,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """Initialize simplified depth head."""
        self.device = device
        self.conv_engine = ConvEngine()
        self.activation_unit = ActivationUnit(ActivationType.GELU)
        self._total_cycles = 0
    
    def simulate(
        self,
        cost_volume: torch.Tensor,
        depth_candidates: torch.Tensor,
        depth_head: nn.Module,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, CycleStats]:
        """
        Run original depth head but track cycles.
        
        Args:
            cost_volume: [B, D, H, W]
            depth_candidates: [D] or [B, D]
            depth_head: Original depth head module
            temperature: Softmax temperature
            
        Returns:
            depths: Regressed depths [B, H, W]
            probs: Softmax probabilities [B, D, H, W]
            cycles: Estimated cycles
        """
        B, D, H, W = cost_volume.shape
        
        # Run original model
        with torch.no_grad():
            logits = depth_head(cost_volume)
            probs = F.softmax(logits / temperature, dim=1)
            
            if depth_candidates.dim() == 1:
                d_cand = depth_candidates.view(1, D, 1, 1)
            else:
                d_cand = depth_candidates.view(B, D, 1, 1)
            
            depths = (probs * d_cand).sum(dim=1)
        
        # Estimate cycles
        cycles = self._estimate_cycles(cost_volume, depth_head)
        self._total_cycles += cycles.total_cycles
        
        return depths, probs, cycles
    
    def _estimate_cycles(
        self,
        cost_volume: torch.Tensor,
        depth_head: nn.Module,
    ) -> CycleStats:
        """Estimate cycles for depth head."""
        B, D, H, W = cost_volume.shape
        total_cycles = 0
        breakdown = {}
        
        # Count conv layers
        for name, module in depth_head.named_modules():
            if isinstance(module, nn.Conv2d):
                k = module.kernel_size[0]
                c_in = module.in_channels
                c_out = module.out_channels
                conv_cycles = H * W * c_in * c_out * k * k // 256
                total_cycles += conv_cycles
                breakdown[f'conv_{name}'] = conv_cycles
        
        # Softmax cycles
        softmax_cycles = 3 * D * H * W * B
        total_cycles += softmax_cycles
        breakdown['softmax'] = softmax_cycles
        
        # Regression cycles
        regression_cycles = 2 * D * H * W * B
        total_cycles += regression_cycles
        breakdown['regression'] = regression_cycles
        
        return CycleStats(
            total_cycles=total_cycles,
            compute_cycles=total_cycles,
            breakdown=breakdown,
        )
    
    def get_total_cycles(self) -> int:
        """Get total accumulated cycles."""
        return self._total_cycles
    
    def reset_cycles(self):
        """Reset cycle counter."""
        self._total_cycles = 0
