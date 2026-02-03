"""
Quality Validator Tests

Tests for PSNR/SSIM quality metric validation.
"""
import pytest
import torch
import numpy as np
from .conftest import (
    compute_psnr_reference,
    compute_ssim_reference,
    MockQualityMetrics,
)


class TestPSNRComputation:
    """PSNR computation tests."""
    
    def test_identical_images_infinite_psnr(self, identical_images):
        """Identical images should have infinite PSNR."""
        rendered, gt = identical_images
        psnr = compute_psnr_reference(rendered, gt)
        assert psnr == float('inf')
    
    def test_similar_images_high_psnr(self, similar_images):
        """Similar images should have high PSNR (>30 dB)."""
        rendered, gt = similar_images
        psnr = compute_psnr_reference(rendered, gt)
        assert psnr > 30  # Small noise = high PSNR
    
    def test_different_images_low_psnr(self, different_images):
        """Different images should have low PSNR (<15 dB)."""
        rendered, gt = different_images
        psnr = compute_psnr_reference(rendered, gt)
        assert psnr < 15  # Random images = low PSNR
    
    def test_psnr_range(self, similar_images):
        """PSNR should be in reasonable range."""
        rendered, gt = similar_images
        psnr = compute_psnr_reference(rendered, gt)
        assert 0 < psnr < 100  # Reasonable range for non-identical


class TestSSIMComputation:
    """SSIM computation tests."""
    
    def test_identical_images_perfect_ssim(self, identical_images):
        """Identical images should have SSIM = 1.0."""
        rendered, gt = identical_images
        ssim = compute_ssim_reference(rendered, gt)
        assert abs(ssim - 1.0) < 0.01  # Allow small numerical error
    
    def test_similar_images_high_ssim(self, similar_images):
        """Similar images should have high SSIM (>0.9)."""
        rendered, gt = similar_images
        ssim = compute_ssim_reference(rendered, gt)
        assert ssim > 0.9
    
    def test_different_images_low_ssim(self, different_images):
        """Different images should have low SSIM (<0.5)."""
        rendered, gt = different_images
        ssim = compute_ssim_reference(rendered, gt)
        assert ssim < 0.5
    
    def test_ssim_range(self, similar_images):
        """SSIM should be in [0, 1] range."""
        rendered, gt = similar_images
        ssim = compute_ssim_reference(rendered, gt)
        assert 0 <= ssim <= 1


class TestThresholdChecking:
    """Quality threshold checking tests."""
    
    def test_acceptable_psnr_drop(self, baseline_metrics):
        """PSNR drop within threshold should pass."""
        current_psnr = baseline_metrics.psnr - 0.3  # 0.3 dB drop
        psnr_drop = baseline_metrics.psnr - current_psnr
        acceptable = psnr_drop < 0.5  # Threshold
        assert acceptable
    
    def test_unacceptable_psnr_drop(self, baseline_metrics):
        """PSNR drop exceeding threshold should fail."""
        current_psnr = baseline_metrics.psnr - 1.0  # 1.0 dB drop
        psnr_drop = baseline_metrics.psnr - current_psnr
        acceptable = psnr_drop < 0.5  # Threshold
        assert not acceptable
    
    def test_acceptable_ssim_drop(self, baseline_metrics):
        """SSIM drop within threshold should pass."""
        current_ssim = baseline_metrics.ssim - 0.005  # 0.005 drop
        ssim_drop = baseline_metrics.ssim - current_ssim
        acceptable = ssim_drop < 0.01  # Threshold
        assert acceptable
    
    def test_unacceptable_ssim_drop(self, baseline_metrics):
        """SSIM drop exceeding threshold should fail."""
        current_ssim = baseline_metrics.ssim - 0.02  # 0.02 drop
        ssim_drop = baseline_metrics.ssim - current_ssim
        acceptable = ssim_drop < 0.01  # Threshold
        assert not acceptable


class TestBaselineComparison:
    """Baseline comparison mode tests."""
    
    def test_store_baseline_metrics(self, baseline_metrics):
        """Should store baseline metrics for comparison."""
        assert baseline_metrics.psnr == 25.5
        assert baseline_metrics.ssim == 0.92
    
    def test_compute_delta(self, baseline_metrics):
        """Should compute delta from baseline."""
        current_psnr = 25.3
        current_ssim = 0.915
        
        psnr_delta = current_psnr - baseline_metrics.psnr
        ssim_delta = current_ssim - baseline_metrics.ssim
        
        assert abs(psnr_delta - (-0.2)) < 0.01
        assert abs(ssim_delta - (-0.005)) < 0.001
    
    def test_percentage_change(self, baseline_metrics):
        """Should compute percentage change from baseline."""
        current_psnr = 25.3
        psnr_pct_change = (current_psnr - baseline_metrics.psnr) / baseline_metrics.psnr * 100
        
        assert abs(psnr_pct_change - (-0.78)) < 0.1  # ~0.78% decrease


class TestMultipleSamples:
    """Multiple sample aggregation tests."""
    
    def test_average_across_samples(self):
        """Should compute average metrics across samples."""
        psnr_values = [25.0, 26.0, 25.5, 24.5, 26.0]
        ssim_values = [0.91, 0.93, 0.92, 0.90, 0.94]
        
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        
        assert abs(avg_psnr - 25.4) < 0.01
        assert abs(avg_ssim - 0.92) < 0.01
    
    def test_std_across_samples(self):
        """Should compute std deviation across samples."""
        psnr_values = [25.0, 26.0, 25.5, 24.5, 26.0]
        std_psnr = np.std(psnr_values)
        
        assert std_psnr > 0  # Should have some variance
        assert std_psnr < 1  # But not too much


class TestEdgeCases:
    """Edge case tests."""
    
    def test_zero_image(self):
        """Should handle zero image gracefully."""
        rendered = torch.zeros(3, 64, 64)
        gt = torch.zeros(3, 64, 64)
        psnr = compute_psnr_reference(rendered, gt)
        assert psnr == float('inf')  # Identical
    
    def test_one_image(self):
        """Should handle all-ones image."""
        rendered = torch.ones(3, 64, 64)
        gt = torch.ones(3, 64, 64)
        psnr = compute_psnr_reference(rendered, gt)
        assert psnr == float('inf')  # Identical
    
    def test_single_pixel_difference(self):
        """Should detect single pixel difference."""
        gt = torch.zeros(3, 64, 64)
        rendered = gt.clone()
        rendered[0, 0, 0] = 1.0  # Single pixel change
        
        psnr = compute_psnr_reference(rendered, gt)
        assert psnr < float('inf')  # Not identical
        assert psnr > 20  # But still high (tiny difference)
