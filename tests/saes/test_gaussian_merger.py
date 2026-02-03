"""
Unit tests for GaussianMerger

Tests Gaussian enlargement and merging logic for early-stop path,
including covariance scaling, position averaging, and attribute merging.
"""
import pytest
import torch
import numpy as np
from conftest import MockGaussian


# Import will fail until implementation exists - this is expected in TDD
try:
    from saes.gaussian_merger import GaussianMerger
    IMPLEMENTATION_EXISTS = True
except ImportError:
    IMPLEMENTATION_EXISTS = False
    # Create placeholder for test development
    class GaussianMerger:
        def __init__(self):
            pass
        def enlarge_and_merge(self, probe_gaussians, tile_coverage):
            raise NotImplementedError("Not yet implemented")


class TestSingleGaussianEnlargement:
    """Test enlarging a single probe Gaussian"""
    
    def test_enlarge_single_gaussian_to_tile(self):
        """Single probe Gaussian should be enlarged to cover tile area"""
        gaussian = MockGaussian(
            mean=torch.tensor([5.0, 5.0, 5.0]),
            cov=torch.eye(3) * 0.1,  # Small initial covariance
            opacity=0.8,
            harmonics=torch.zeros(3, 16),
        )
        
        merger = GaussianMerger()
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Tile covers 4×4 = 16 pixels, e.g., rows [0, 4), cols [0, 4)
        tile_coverage = (0, 4, 0, 4)
        
        result = merger.enlarge_and_merge([gaussian], tile_coverage)
        
        # Should return a single enlarged Gaussian
        assert len(result) == 1
        
        # Position should be preserved
        assert torch.allclose(result[0].mean, gaussian.mean, atol=0.01)
        
        # Covariance should be scaled up
        result_cov_scale = torch.diagonal(result[0].cov).mean()
        original_cov_scale = torch.diagonal(gaussian.cov).mean()
        assert result_cov_scale > original_cov_scale, "Covariance should be enlarged"


class TestMultipleGaussianMerging:
    """Test merging multiple probe Gaussians"""
    
    def test_merge_four_probe_gaussians(self, identical_gaussians):
        """Four probe Gaussians should be merged into representative Gaussian(s)"""
        merger = GaussianMerger()
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        tile_coverage = (0, 4, 0, 4)  # 4×4 tile
        
        result = merger.enlarge_and_merge(identical_gaussians, tile_coverage)
        
        # Should return merged Gaussian(s) - implementation may vary
        # Could be single merged or multiple enlarged
        assert len(result) >= 1, "Should return at least one Gaussian"
        
        # Total opacity should be preserved (approximately)
        original_total_opacity = sum(g.opacity for g in identical_gaussians)
        result_total_opacity = sum(g.opacity for g in result)
        assert result_total_opacity == pytest.approx(original_total_opacity, rel=0.1)


class TestCovarianceScaling:
    """Test covariance scale increase for tile coverage"""
    
    def test_covariance_scale_proportional_to_tile_area(self):
        """Covariance scale should increase proportionally to cover tile area"""
        gaussian = MockGaussian(
            mean=torch.tensor([2.0, 2.0, 5.0]),
            cov=torch.eye(3) * 0.05,  # Very small initial covariance
            opacity=0.8,
            harmonics=torch.zeros(3, 16),
        )
        
        merger = GaussianMerger()
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Test with different tile sizes
        tile_4x4 = (0, 4, 0, 4)  # Area = 16
        tile_8x8 = (0, 8, 0, 8)  # Area = 64
        
        result_4x4 = merger.enlarge_and_merge([gaussian], tile_4x4)
        result_8x8 = merger.enlarge_and_merge([gaussian], tile_8x8)
        
        cov_scale_4x4 = torch.diagonal(result_4x4[0].cov).mean()
        cov_scale_8x8 = torch.diagonal(result_8x8[0].cov).mean()
        
        # Larger tile should have larger covariance scale
        assert cov_scale_8x8 > cov_scale_4x4, "8×8 tile should have larger covariance than 4×4"
        
        # Ratio should be approximately sqrt(64/16) = 2
        ratio = cov_scale_8x8 / cov_scale_4x4
        assert 1.5 < ratio < 3.0, "Covariance scale ratio should be approximately 2 for 4× area"
    
    def test_covariance_remains_positive_definite(self):
        """Enlarged covariance should remain positive definite (valid)"""
        gaussian = MockGaussian(
            mean=torch.zeros(3),
            cov=torch.eye(3) * 0.1,
            opacity=0.8,
            harmonics=torch.zeros(3, 16),
        )
        
        merger = GaussianMerger()
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        tile_coverage = (0, 4, 0, 4)
        result = merger.enlarge_and_merge([gaussian], tile_coverage)
        
        # Check positive definiteness via Cholesky decomposition
        cov = result[0].cov
        try:
            torch.linalg.cholesky(cov)
            is_positive_definite = True
        except RuntimeError:
            is_positive_definite = False
        
        assert is_positive_definite, "Enlarged covariance must be positive definite"


class TestPositionAveraging:
    """Test position averaging across probe Gaussians"""
    
    def test_position_weighted_average(self):
        """Merged position should be weighted average of probe positions"""
        gaussians = [
            MockGaussian(torch.tensor([1.0, 1.0, 5.0]), torch.eye(3), 0.8, torch.zeros(3, 16)),
            MockGaussian(torch.tensor([3.0, 3.0, 5.0]), torch.eye(3), 0.8, torch.zeros(3, 16)),
        ]
        
        merger = GaussianMerger()
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        tile_coverage = (0, 4, 0, 4)
        result = merger.enlarge_and_merge(gaussians, tile_coverage)
        
        # Expected average position
        expected_mean = torch.tensor([2.0, 2.0, 5.0])
        
        # Result should have position near the average
        # (exact match if equal opacity weights, approximate if weighted by opacity)
        actual_mean = result[0].mean if len(result) == 1 else sum(g.mean * g.opacity for g in result) / sum(g.opacity for g in result)
        assert torch.allclose(actual_mean, expected_mean, atol=0.5)


class TestColorAveraging:
    """Test color (SH) averaging across probe Gaussians"""
    
    def test_color_averaging_sh_coefficients(self):
        """Merged SH coefficients should be averaged"""
        sh_red = torch.zeros(3, 16)
        sh_red[0, 0] = 1.0  # Red
        
        sh_blue = torch.zeros(3, 16)
        sh_blue[2, 0] = 1.0  # Blue
        
        gaussians = [
            MockGaussian(torch.zeros(3), torch.eye(3), 0.8, sh_red),
            MockGaussian(torch.zeros(3), torch.eye(3), 0.8, sh_blue),
        ]
        
        merger = GaussianMerger()
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        tile_coverage = (0, 4, 0, 4)
        result = merger.enlarge_and_merge(gaussians, tile_coverage)
        
        # Averaged color should be purple (red + blue) / 2
        expected_sh = (sh_red + sh_blue) / 2
        
        # Result should have averaged SH
        actual_sh = result[0].harmonics if len(result) == 1 else sum(g.harmonics * g.opacity for g in result) / sum(g.opacity for g in result)
        assert torch.allclose(actual_sh, expected_sh, atol=0.1)


class TestOpacityAveraging:
    """Test opacity averaging across probe Gaussians"""
    
    def test_opacity_averaging(self):
        """Merged opacity should be averaged (or weighted average)"""
        gaussians = [
            MockGaussian(torch.zeros(3), torch.eye(3), 0.6, torch.zeros(3, 16)),
            MockGaussian(torch.zeros(3), torch.eye(3), 0.8, torch.zeros(3, 16)),
        ]
        
        merger = GaussianMerger()
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        tile_coverage = (0, 4, 0, 4)
        result = merger.enlarge_and_merge(gaussians, tile_coverage)
        
        # Expected average opacity
        expected_opacity = (0.6 + 0.8) / 2
        
        # Result should have averaged opacity
        actual_opacity = result[0].opacity if len(result) == 1 else sum(g.opacity for g in result) / len(result)
        assert actual_opacity == pytest.approx(expected_opacity, abs=0.1)


class TestEdgeTiles:
    """Test handling of edge tiles with partial coverage"""
    
    def test_edge_tile_partial_coverage(self):
        """Edge tiles (partial pixel coverage) should be handled correctly"""
        gaussian = MockGaussian(
            mean=torch.tensor([3.0, 3.0, 5.0]),
            cov=torch.eye(3) * 0.1,
            opacity=0.8,
            harmonics=torch.zeros(3, 16),
        )
        
        merger = GaussianMerger()
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Edge tile: rows [0, 3), cols [0, 3) = 9 pixels (not 16)
        edge_tile_coverage = (0, 3, 0, 3)
        
        result = merger.enlarge_and_merge([gaussian], edge_tile_coverage)
        
        # Should handle gracefully
        assert len(result) >= 1, "Edge tiles should be processed"
        
        # Covariance should be scaled for actual coverage
        result_cov_scale = torch.diagonal(result[0].cov).mean()
        original_cov_scale = torch.diagonal(gaussian.cov).mean()
        assert result_cov_scale > original_cov_scale


class TestTileCoverageCalculation:
    """Test tile coverage area calculation"""
    
    def test_coverage_area_computation(self):
        """Verify tile coverage area is computed correctly"""
        gaussian = MockGaussian(torch.zeros(3), torch.eye(3) * 0.1, 0.8, torch.zeros(3, 16))
        merger = GaussianMerger()
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Test different tile coverages
        test_cases = [
            ((0, 4, 0, 4), 16),  # 4×4 = 16
            ((0, 8, 0, 8), 64),  # 8×8 = 64
            ((0, 2, 0, 2), 4),   # 2×2 = 4
            ((0, 3, 0, 5), 15),  # 3×5 = 15 (non-square)
        ]
        
        for tile_coverage, expected_area in test_cases:
            result = merger.enlarge_and_merge([gaussian], tile_coverage)
            
            # Covariance scale should correlate with area
            # Larger area → larger covariance
            result_cov_scale = torch.diagonal(result[0].cov).mean().item()
            
            # Just verify it runs without error and scales reasonably
            assert result_cov_scale > 0, f"Covariance should be positive for coverage {tile_coverage}"
