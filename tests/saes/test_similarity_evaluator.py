"""
Unit tests for GaussianSimilarityEvaluator

Tests the 3D Gaussian similarity computation component which evaluates
probe Gaussians on position, covariance, color, and opacity dispersions.
"""
import pytest
import torch
import numpy as np
from conftest import MockGaussian


# Import will fail until implementation exists - this is expected in TDD
try:
    from saes.similarity_evaluator import GaussianSimilarityEvaluator
    from saes.types import GaussianSimilarityMetrics
    IMPLEMENTATION_EXISTS = True
except ImportError:
    IMPLEMENTATION_EXISTS = False
    # Create placeholder for test development
    class GaussianSimilarityEvaluator:
        def __init__(self, scene_scale=None):
            self.scene_scale = scene_scale
        def evaluate(self, gaussians):
            raise NotImplementedError("Not yet implemented")


class TestIdenticalGaussians:
    """Test cases for identical Gaussians (should have similarity ≈ 1.0)"""
    
    def test_identical_gaussians_high_similarity(self, identical_gaussians):
        """Identical Gaussians should have similarity score ≈ 1.0"""
        evaluator = GaussianSimilarityEvaluator(scene_scale=1.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(identical_gaussians)
        
        # All dispersions should be zero or near-zero
        assert metrics.position_dispersion < 0.01, "Position dispersion should be near-zero"
        assert metrics.covariance_dispersion < 0.01, "Covariance dispersion should be near-zero"
        assert metrics.color_dispersion < 0.01, "Color dispersion should be near-zero"
        assert metrics.opacity_dispersion < 0.01, "Opacity dispersion should be near-zero"
        
        # Similarity score should be very high
        assert metrics.similarity_score > 0.99, "Similarity should be ≈ 1.0 for identical Gaussians"


class TestDifferentGaussians:
    """Test cases for very different Gaussians (should have similarity ≈ 0.0)"""
    
    def test_different_gaussians_low_similarity(self, different_gaussians):
        """Very different Gaussians should have low similarity score"""
        evaluator = GaussianSimilarityEvaluator(scene_scale=10.0)  # Large scene
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(different_gaussians)
        
        # Dispersions should be high
        assert metrics.position_dispersion > 0.5, "Position dispersion should be high"
        assert metrics.covariance_dispersion > 0.3, "Covariance dispersion should be moderate"
        
        # Similarity score should be low
        assert metrics.similarity_score < 0.5, "Similarity should be low for very different Gaussians"


class TestPositionDispersion:
    """Test position dispersion calculation"""
    
    def test_position_dispersion_dominates_large_spatial_spread(self):
        """When Gaussians are far apart, position dispersion should dominate"""
        gaussians = [
            MockGaussian(
                mean=torch.tensor([0.0, 0.0, 0.0]),
                cov=torch.eye(3) * 0.1,
                opacity=0.8,
                harmonics=torch.zeros(3, 16)
            ),
            MockGaussian(
                mean=torch.tensor([10.0, 10.0, 10.0]),  # Very far
                cov=torch.eye(3) * 0.1,
                opacity=0.8,
                harmonics=torch.zeros(3, 16)
            ),
        ]
        
        evaluator = GaussianSimilarityEvaluator(scene_scale=1.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(gaussians)
        
        # Position dispersion should be largest contributor
        assert metrics.position_dispersion > metrics.covariance_dispersion
        assert metrics.position_dispersion > metrics.color_dispersion
        assert metrics.position_dispersion > metrics.opacity_dispersion
    
    def test_position_dispersion_normalized_by_scene_scale(self):
        """Position dispersion should be normalized by scene scale"""
        gaussians = [
            MockGaussian(torch.tensor([0.0, 0.0, 0.0]), torch.eye(3), 0.8, torch.zeros(3, 16)),
            MockGaussian(torch.tensor([5.0, 0.0, 0.0]), torch.eye(3), 0.8, torch.zeros(3, 16)),
        ]
        
        # Same Gaussians, different scene scales
        evaluator_small = GaussianSimilarityEvaluator(scene_scale=1.0)
        evaluator_large = GaussianSimilarityEvaluator(scene_scale=10.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics_small = evaluator_small.evaluate(gaussians)
        metrics_large = evaluator_large.evaluate(gaussians)
        
        # Larger scene scale should reduce normalized dispersion
        assert metrics_small.position_dispersion > metrics_large.position_dispersion


class TestCovarianceDispersion:
    """Test covariance dispersion calculation"""
    
    def test_covariance_dispersion_with_different_scales(self):
        """Gaussians with different covariance scales should have high dispersion"""
        gaussians = [
            MockGaussian(torch.zeros(3), torch.eye(3) * 0.1, 0.8, torch.zeros(3, 16)),
            MockGaussian(torch.zeros(3), torch.eye(3) * 2.0, 0.8, torch.zeros(3, 16)),
        ]
        
        evaluator = GaussianSimilarityEvaluator(scene_scale=1.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(gaussians)
        
        # Covariance dispersion should be significant
        assert metrics.covariance_dispersion > 0.3, "Large covariance difference should yield high dispersion"


class TestColorDispersion:
    """Test color (SH) dispersion calculation"""
    
    def test_color_dispersion_with_sh_coefficients(self):
        """Gaussians with different SH coefficients should have color dispersion"""
        sh_red = torch.zeros(3, 16)
        sh_red[0, 0] = 1.0  # Red DC component
        
        sh_blue = torch.zeros(3, 16)
        sh_blue[2, 0] = 1.0  # Blue DC component
        
        gaussians = [
            MockGaussian(torch.zeros(3), torch.eye(3), 0.8, sh_red),
            MockGaussian(torch.zeros(3), torch.eye(3), 0.8, sh_blue),
        ]
        
        evaluator = GaussianSimilarityEvaluator(scene_scale=1.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(gaussians)
        
        # Color dispersion should reflect red vs. blue difference
        assert metrics.color_dispersion > 0.1, "Different colors should yield measurable dispersion"


class TestOpacityDispersion:
    """Test opacity dispersion calculation"""
    
    def test_opacity_dispersion_calculation(self):
        """Gaussians with different opacities should have opacity dispersion"""
        gaussians = [
            MockGaussian(torch.zeros(3), torch.eye(3), 0.1, torch.zeros(3, 16)),  # Nearly transparent
            MockGaussian(torch.zeros(3), torch.eye(3), 0.9, torch.zeros(3, 16)),  # Nearly opaque
        ]
        
        evaluator = GaussianSimilarityEvaluator(scene_scale=1.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(gaussians)
        
        # Opacity dispersion should be max - min = 0.9 - 0.1 = 0.8
        assert metrics.opacity_dispersion == pytest.approx(0.8, abs=0.01)


class TestWeightedAggregation:
    """Test weighted similarity aggregation formula"""
    
    def test_weighted_aggregation_formula(self, moderate_similarity_gaussians):
        """Verify similarity score uses correct weighting: 0.4*pos + 0.3*cov + 0.15*color + 0.15*opacity"""
        evaluator = GaussianSimilarityEvaluator(scene_scale=5.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(moderate_similarity_gaussians)
        
        # Manually compute expected dispersion
        expected_dispersion = (
            0.4 * metrics.position_dispersion +
            0.3 * metrics.covariance_dispersion +
            0.15 * metrics.color_dispersion +
            0.15 * metrics.opacity_dispersion
        )
        
        # Similarity = exp(-dispersion / 0.1)
        expected_similarity = np.exp(-expected_dispersion / 0.1)
        
        # Verify similarity score matches formula
        assert metrics.similarity_score == pytest.approx(expected_similarity, abs=0.01)
    
    def test_similarity_score_in_valid_range(self, moderate_similarity_gaussians):
        """Similarity score should always be in [0, 1]"""
        evaluator = GaussianSimilarityEvaluator(scene_scale=5.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(moderate_similarity_gaussians)
        
        assert 0.0 <= metrics.similarity_score <= 1.0, "Similarity score must be in [0, 1]"


class TestSceneScaleEstimation:
    """Test automatic scene scale estimation"""
    
    def test_auto_estimate_scene_scale_when_none_provided(self):
        """When scene_scale=None, evaluator should auto-estimate from Gaussian positions"""
        gaussians = [
            MockGaussian(torch.tensor([1.0, 2.0, 3.0]), torch.eye(3), 0.8, torch.zeros(3, 16)),
            MockGaussian(torch.tensor([2.0, 3.0, 4.0]), torch.eye(3), 0.8, torch.zeros(3, 16)),
            MockGaussian(torch.tensor([3.0, 4.0, 5.0]), torch.eye(3), 0.8, torch.zeros(3, 16)),
        ]
        
        # scene_scale=None triggers auto-estimation
        evaluator = GaussianSimilarityEvaluator(scene_scale=None)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(gaussians)
        
        # Should run without error and produce valid similarity score
        assert 0.0 <= metrics.similarity_score <= 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_gaussian_should_have_perfect_similarity(self):
        """Single Gaussian has no pairwise comparisons → perfect similarity"""
        gaussians = [
            MockGaussian(torch.zeros(3), torch.eye(3), 0.8, torch.zeros(3, 16))
        ]
        
        evaluator = GaussianSimilarityEvaluator(scene_scale=1.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(gaussians)
        
        # Single Gaussian should have zero dispersion
        assert metrics.position_dispersion == 0.0
        assert metrics.similarity_score == pytest.approx(1.0, abs=0.01)
    
    def test_two_identical_gaussians(self):
        """Two identical Gaussians should have perfect similarity"""
        gaussian = MockGaussian(torch.ones(3), torch.eye(3), 0.8, torch.zeros(3, 16))
        gaussians = [gaussian, gaussian]  # Same object
        
        evaluator = GaussianSimilarityEvaluator(scene_scale=1.0)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        metrics = evaluator.evaluate(gaussians)
        
        assert metrics.similarity_score > 0.99, "Identical Gaussians should have similarity ≈ 1.0"
