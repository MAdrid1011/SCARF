"""
DSU Processor Tests

Tests for the Depth Search Unit processor.
"""
import pytest
import torch
import numpy as np
from .conftest import (
    MockDSUConfig, project_point, bilinear_sample
)


class TestDepthSamplerProjection:
    """Tests for depth sampler projection."""
    
    def test_project_same_camera(
        self, reference_intrinsics, reference_extrinsics
    ):
        """Projection to same camera should return same point."""
        pixel = (320.0, 240.0)  # Center pixel
        depth = 5.0
        
        u_proj, v_proj = project_point(
            pixel, depth,
            reference_intrinsics, reference_extrinsics,
            reference_intrinsics, reference_extrinsics,  # Same camera
        )
        
        assert abs(u_proj - pixel[0]) < 1e-4
        assert abs(v_proj - pixel[1]) < 1e-4
    
    def test_project_translated_camera(
        self, reference_intrinsics, reference_extrinsics,
        target_intrinsics, target_extrinsics
    ):
        """Projection to translated camera should shift horizontally."""
        pixel = (320.0, 240.0)
        depth = 5.0
        
        u_proj, v_proj = project_point(
            pixel, depth,
            reference_intrinsics, reference_extrinsics,
            target_intrinsics, target_extrinsics,
        )
        
        # Camera translated 0.1m in x → pixel should shift left
        # Shift depends on depth and focal length
        expected_shift = 500.0 * 0.1 / depth  # fx * baseline / depth = 10 pixels
        
        assert abs(u_proj - (pixel[0] - expected_shift)) < 1.0
        assert abs(v_proj - pixel[1]) < 1e-4  # No vertical shift
    
    def test_project_closer_depth_larger_shift(
        self, reference_intrinsics, reference_extrinsics,
        target_intrinsics, target_extrinsics
    ):
        """Closer depth should result in larger shift."""
        pixel = (320.0, 240.0)
        
        # Project at different depths
        _, _ = project_point(
            pixel, 2.0,
            reference_intrinsics, reference_extrinsics,
            target_intrinsics, target_extrinsics,
        )
        u_near, _ = project_point(
            pixel, 2.0,
            reference_intrinsics, reference_extrinsics,
            target_intrinsics, target_extrinsics,
        )
        
        u_far, _ = project_point(
            pixel, 10.0,
            reference_intrinsics, reference_extrinsics,
            target_intrinsics, target_extrinsics,
        )
        
        # Shift at near depth (2m) vs far depth (10m)
        shift_near = abs(pixel[0] - u_near)
        shift_far = abs(pixel[0] - u_far)
        
        assert shift_near > shift_far


class TestDepthSamplerBilinear:
    """Tests for bilinear sampling."""
    
    def test_bilinear_integer_coords(self, random_feature_map):
        """Integer coordinates should return exact pixel."""
        u, v = 32, 32
        sampled = bilinear_sample(random_feature_map, float(u), float(v))
        expected = random_feature_map[:, v, u]
        
        assert torch.allclose(sampled, expected)
    
    def test_bilinear_interpolation(self, random_feature_map):
        """Non-integer coordinates should interpolate."""
        u, v = 32.5, 32.5
        sampled = bilinear_sample(random_feature_map, u, v)
        
        # Manual calculation
        f00 = random_feature_map[:, 32, 32]
        f01 = random_feature_map[:, 33, 32]
        f10 = random_feature_map[:, 32, 33]
        f11 = random_feature_map[:, 33, 33]
        
        expected = 0.25 * (f00 + f01 + f10 + f11)
        
        assert torch.allclose(sampled, expected)
    
    def test_bilinear_boundary_clamping(self, random_feature_map):
        """Out-of-bounds coordinates should be clamped."""
        # Beyond right boundary
        u, v = 100.0, 32.0  # 64x64 map, u=100 > 63
        sampled = bilinear_sample(random_feature_map, u, v)
        
        # Should sample from right edge
        expected = random_feature_map[:, 32, 63]
        assert torch.allclose(sampled, expected)
        
        # Negative coordinates
        u, v = -5.0, 32.0
        sampled = bilinear_sample(random_feature_map, u, v)
        expected = random_feature_map[:, 32, 0]
        assert torch.allclose(sampled, expected)


class TestCostVolumeComputation:
    """Tests for cost volume computation."""
    
    def test_correlation_higher_for_similar_features(
        self, reference_feature, random_feature_map
    ):
        """Similar features should have higher correlation."""
        def compute_cost(ref, tgt, cost_type='correlation'):
            ref_norm = ref / (torch.norm(ref) + 1e-8)
            tgt_norm = tgt / (torch.norm(tgt) + 1e-8)
            return float(torch.dot(ref_norm, tgt_norm))
        
        # Create target feature similar to reference
        similar_target = reference_feature + torch.randn_like(reference_feature) * 0.1
        random_target = torch.randn_like(reference_feature)
        
        cost_similar = compute_cost(reference_feature, similar_target)
        cost_random = compute_cost(reference_feature, random_target)
        
        assert cost_similar > cost_random
    
    def test_cost_volume_shape(
        self, reference_feature, random_feature_map, depth_candidates
    ):
        """Cost volume should have shape [D]."""
        def compute_cost_volume(ref_feat, tgt_map, candidates):
            D = len(candidates)
            costs = torch.zeros(D)
            
            for d in range(D):
                # Mock sampling at fixed location
                tgt_feat = tgt_map[:, 32, 32]
                costs[d] = torch.dot(ref_feat, tgt_feat) / (
                    torch.norm(ref_feat) * torch.norm(tgt_feat) + 1e-8
                )
            
            return costs
        
        costs = compute_cost_volume(
            reference_feature, random_feature_map, depth_candidates
        )
        
        assert costs.shape == (len(depth_candidates),)


class TestSoftmaxAggregation:
    """Tests for softmax aggregation."""
    
    def test_softmax_sums_to_one(self, peaked_cost_volume):
        """Softmax probabilities should sum to 1."""
        costs = peaked_cost_volume[:, 32, 32]  # Single pixel
        probs = torch.softmax(costs, dim=0)
        
        assert abs(probs.sum().item() - 1.0) < 1e-6
    
    def test_expected_depth_in_range(self, depth_candidates, peaked_cost_volume):
        """Expected depth should be within depth range."""
        costs = peaked_cost_volume[:, 32, 32]
        probs = torch.softmax(costs, dim=0)
        
        expected_depth = (probs * depth_candidates).sum()
        
        assert expected_depth >= depth_candidates[0]
        assert expected_depth <= depth_candidates[-1]
    
    def test_negative_softmax_for_costs(self, depth_candidates):
        """Transplat uses negative softmax (lower cost = higher prob)."""
        # Create costs where lower index = lower cost (better)
        costs = torch.arange(32).float()  # 0, 1, 2, ...
        
        # For cost (lower is better), use softmax(-costs)
        probs = torch.softmax(-costs, dim=0)
        
        # Probability should be highest at index 0 (lowest cost)
        assert probs[0] > probs[15]
        assert probs[0] > probs[-1]
    
    def test_direct_softmax_for_correlation(self, depth_candidates):
        """MVSplat uses direct softmax (higher correlation = higher prob)."""
        # Create correlation where higher index = higher correlation (better)
        correlations = torch.arange(32).float()
        
        # For correlation (higher is better), use direct softmax
        probs = torch.softmax(correlations, dim=0)
        
        # Probability should be highest at last index (highest correlation)
        assert probs[-1] > probs[15]
        assert probs[-1] > probs[0]


class TestStatisticsExtraction:
    """Tests for statistics extraction for FSDR."""
    
    def test_extract_peak_info(self, depth_candidates, mock_probability_distribution):
        """Should extract correct peak index and probability."""
        probs = mock_probability_distribution
        
        best_idx = int(torch.argmax(probs).item())
        peak_prob = float(probs[best_idx])
        
        assert best_idx == len(probs) // 2  # Our mock peaks at center
        assert peak_prob > 0
    
    def test_extract_second_best(self, depth_candidates):
        """Should find second-best index correctly."""
        # Create bimodal distribution
        probs = torch.zeros(32)
        probs[10] = 0.6  # Best
        probs[15] = 0.3  # Second best
        probs = probs / probs.sum()
        
        best_idx = int(torch.argmax(probs).item())
        
        probs_copy = probs.clone()
        probs_copy[best_idx] = -1
        second_idx = int(torch.argmax(probs_copy).item())
        
        assert best_idx == 10
        assert second_idx == 15
        assert second_idx - best_idx == 5  # Offset
    
    def test_extract_spread(self, depth_candidates):
        """Should calculate distribution spread."""
        # Narrow distribution
        probs_narrow = torch.zeros(32)
        probs_narrow[16] = 0.9
        probs_narrow[15] = 0.05
        probs_narrow[17] = 0.05
        
        # Wide distribution
        probs_wide = torch.ones(32) / 32
        
        def calc_spread(probs, candidates):
            mean = (probs * candidates).sum()
            variance = (probs * (candidates - mean) ** 2).sum()
            std = variance.sqrt()
            depth_range = candidates[-1] - candidates[0]
            return std / depth_range
        
        spread_narrow = calc_spread(probs_narrow, depth_candidates)
        spread_wide = calc_spread(probs_wide, depth_candidates)
        
        assert spread_narrow < spread_wide


class TestDSUProcessorIntegration:
    """Integration tests for complete DSU processor."""
    
    def test_full_depth_search(
        self, reference_feature, random_feature_map, depth_candidates,
        projection_params
    ):
        """Complete depth search should return valid depth."""
        def full_search(ref_feat, tgt_map, candidates, proj_params):
            D = len(candidates)
            H, W = tgt_map.shape[1], tgt_map.shape[2]
            pixel = (W // 2, H // 2)
            
            costs = torch.zeros(D)
            for d, depth in enumerate(candidates):
                # Project and sample
                u_proj, v_proj = project_point(
                    pixel, float(depth),
                    proj_params['ref_intrinsics'],
                    proj_params['ref_extrinsics'],
                    proj_params['tgt_intrinsics'],
                    proj_params['tgt_extrinsics'],
                )
                
                tgt_feat = bilinear_sample(tgt_map, u_proj, v_proj)
                
                # Compute cost (correlation)
                costs[d] = torch.dot(ref_feat, tgt_feat) / (
                    torch.norm(ref_feat) * torch.norm(tgt_feat) + 1e-8
                )
            
            # Softmax aggregation
            probs = torch.softmax(costs, dim=0)
            depth = float((probs * candidates).sum())
            
            return depth, probs
        
        depth, probs = full_search(
            reference_feature, random_feature_map,
            depth_candidates, projection_params
        )
        
        assert depth_candidates[0] <= depth <= depth_candidates[-1]
        assert abs(probs.sum().item() - 1.0) < 1e-6


class TestDSUForFSDRIntegration:
    """Tests for DSU functions needed by FSDR."""
    
    def test_single_depth_cost_function(
        self, reference_feature, random_feature_map, projection_params
    ):
        """Single depth cost function for FSDR light verify."""
        def single_cost(ref_feat, tgt_map, pixel, depth_idx, candidates, proj_params):
            depth = float(candidates[depth_idx])
            
            u_proj, v_proj = project_point(
                pixel, depth,
                proj_params['ref_intrinsics'],
                proj_params['ref_extrinsics'],
                proj_params['tgt_intrinsics'],
                proj_params['tgt_extrinsics'],
            )
            
            tgt_feat = bilinear_sample(tgt_map, u_proj, v_proj)
            
            cost = torch.dot(ref_feat, tgt_feat) / (
                torch.norm(ref_feat) * torch.norm(tgt_feat) + 1e-8
            )
            return float(cost)
        
        # Mock depth candidates
        candidates = torch.linspace(0.5, 10.0, 32)
        pixel = (32.0, 32.0)
        
        cost = single_cost(
            reference_feature, random_feature_map,
            pixel, 16, candidates, projection_params
        )
        
        assert -1.0 <= cost <= 1.0  # Cosine similarity range
    
    def test_probability_distribution_function(
        self, reference_feature, random_feature_map, depth_candidates,
        projection_params
    ):
        """Full probability distribution for FSDR cache miss."""
        def prob_distribution(ref_feat, tgt_map, pixel, candidates, proj_params):
            D = len(candidates)
            costs = torch.zeros(D)
            
            for d, depth in enumerate(candidates):
                u_proj, v_proj = project_point(
                    pixel, float(depth),
                    proj_params['ref_intrinsics'],
                    proj_params['ref_extrinsics'],
                    proj_params['tgt_intrinsics'],
                    proj_params['tgt_extrinsics'],
                )
                
                tgt_feat = bilinear_sample(tgt_map, u_proj, v_proj)
                
                costs[d] = torch.dot(ref_feat, tgt_feat) / (
                    torch.norm(ref_feat) * torch.norm(tgt_feat) + 1e-8
                )
            
            return torch.softmax(costs, dim=0)
        
        pixel = (32.0, 32.0)
        probs = prob_distribution(
            reference_feature, random_feature_map,
            pixel, depth_candidates, projection_params
        )
        
        assert probs.shape == (32,)
        assert abs(probs.sum().item() - 1.0) < 1e-6
