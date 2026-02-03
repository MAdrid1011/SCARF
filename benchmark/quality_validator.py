"""
Quality Validator

PSNR/SSIM quality metric validation for SCARF benchmarks.
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class QualityMetrics:
    """
    Quality metrics for a benchmark run.
    
    Attributes:
        psnr_mean: Mean PSNR across samples
        psnr_std: Standard deviation of PSNR
        ssim_mean: Mean SSIM across samples
        ssim_std: Standard deviation of SSIM
        num_samples: Number of samples
    """
    psnr_mean: float
    psnr_std: float = 0.0
    ssim_mean: float = 0.0
    ssim_std: float = 0.0
    num_samples: int = 1
    
    def to_dict(self) -> Dict:
        return {
            'psnr_mean': self.psnr_mean,
            'psnr_std': self.psnr_std,
            'ssim_mean': self.ssim_mean,
            'ssim_std': self.ssim_std,
            'num_samples': self.num_samples,
        }


class QualityValidator:
    """
    Quality metric validator for SCARF benchmark.
    
    Computes PSNR and SSIM between rendered and ground truth images,
    and validates against acceptable degradation thresholds.
    
    Example:
        validator = QualityValidator(baseline_psnr=25.5, baseline_ssim=0.92)
        
        for rendered, gt in samples:
            validator.record_sample(rendered, gt)
        
        metrics = validator.get_metrics()
        print(f"PSNR: {metrics.psnr_mean:.2f} dB")
        print(f"SSIM: {metrics.ssim_mean:.4f}")
        
        if validator.check_quality_threshold():
            print("Quality acceptable!")
    """
    
    def __init__(
        self,
        baseline_psnr: Optional[float] = None,
        baseline_ssim: Optional[float] = None,
    ):
        """
        Initialize quality validator.
        
        Args:
            baseline_psnr: Baseline PSNR for comparison (optional)
            baseline_ssim: Baseline SSIM for comparison (optional)
        """
        self.baseline_psnr = baseline_psnr
        self.baseline_ssim = baseline_ssim
        self._psnr_values: List[float] = []
        self._ssim_values: List[float] = []
    
    def record_sample(
        self,
        rendered: torch.Tensor,
        ground_truth: torch.Tensor,
    ):
        """
        Record quality metrics for a sample.
        
        Args:
            rendered: Rendered image [C, H, W] or [H, W, C]
            ground_truth: Ground truth image, same shape
        """
        # Ensure tensors are on CPU
        if rendered.is_cuda:
            rendered = rendered.cpu()
        if ground_truth.is_cuda:
            ground_truth = ground_truth.cpu()
        
        # Compute metrics
        psnr = self._compute_psnr(rendered, ground_truth)
        ssim = self._compute_ssim(rendered, ground_truth)
        
        self._psnr_values.append(psnr)
        self._ssim_values.append(ssim)
    
    def _compute_psnr(
        self,
        rendered: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> float:
        """Compute PSNR between two images."""
        mse = torch.mean((rendered - ground_truth) ** 2).item()
        if mse == 0:
            return float('inf')
        return 10 * np.log10(1.0 / mse)
    
    def _compute_ssim(
        self,
        rendered: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> float:
        """
        Compute SSIM between two images.
        
        Uses a simplified implementation. For production,
        consider using skimage.metrics.structural_similarity.
        """
        # Convert to numpy
        rendered_np = rendered.numpy().flatten()
        gt_np = ground_truth.numpy().flatten()
        
        # Compute means
        mu_r = np.mean(rendered_np)
        mu_g = np.mean(gt_np)
        
        # Compute variances and covariance
        sigma_r = np.var(rendered_np)
        sigma_g = np.var(gt_np)
        sigma_rg = np.mean((rendered_np - mu_r) * (gt_np - mu_g))
        
        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu_r * mu_g + C1) * (2 * sigma_rg + C2)) / \
               ((mu_r ** 2 + mu_g ** 2 + C1) * (sigma_r + sigma_g + C2))
        
        return float(ssim)
    
    def get_metrics(self) -> QualityMetrics:
        """Get aggregated quality metrics."""
        if not self._psnr_values:
            return QualityMetrics(psnr_mean=0.0)
        
        return QualityMetrics(
            psnr_mean=np.mean(self._psnr_values),
            psnr_std=np.std(self._psnr_values),
            ssim_mean=np.mean(self._ssim_values),
            ssim_std=np.std(self._ssim_values),
            num_samples=len(self._psnr_values),
        )
    
    def check_quality_threshold(
        self,
        psnr_drop_threshold: float = 0.5,
        ssim_drop_threshold: float = 0.01,
    ) -> bool:
        """
        Check if quality degradation is within acceptable threshold.
        
        Args:
            psnr_drop_threshold: Maximum acceptable PSNR drop (dB)
            ssim_drop_threshold: Maximum acceptable SSIM drop
        
        Returns:
            True if quality is acceptable
        """
        if self.baseline_psnr is None or self.baseline_ssim is None:
            # No baseline to compare against
            return True
        
        metrics = self.get_metrics()
        
        psnr_drop = self.baseline_psnr - metrics.psnr_mean
        ssim_drop = self.baseline_ssim - metrics.ssim_mean
        
        psnr_ok = psnr_drop < psnr_drop_threshold
        ssim_ok = ssim_drop < ssim_drop_threshold
        
        return psnr_ok and ssim_ok
    
    def get_comparison(self) -> Dict:
        """Get comparison with baseline metrics."""
        metrics = self.get_metrics()
        
        result = {
            'current': metrics.to_dict(),
            'baseline': {
                'psnr': self.baseline_psnr,
                'ssim': self.baseline_ssim,
            },
            'delta': {},
        }
        
        if self.baseline_psnr is not None:
            result['delta']['psnr'] = metrics.psnr_mean - self.baseline_psnr
        if self.baseline_ssim is not None:
            result['delta']['ssim'] = metrics.ssim_mean - self.baseline_ssim
        
        return result
    
    def reset(self):
        """Clear all recorded samples."""
        self._psnr_values.clear()
        self._ssim_values.clear()
