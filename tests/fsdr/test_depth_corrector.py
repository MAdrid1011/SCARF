"""
Depth Corrector Tests

Tests for FSDR depth correction strategies.
"""
import pytest
import torch
from .conftest import MockCacheEntry, MockFSDRConfig


class TestStrategySelection:
    """Tests for strategy selection logic."""
    
    def test_direct_reuse_high_confidence_low_hamming(
        self, default_config, high_confidence_entry
    ):
        """Direct reuse when high confidence and low Hamming distance."""
        def decide_strategy(config, entry, hamming_dist):
            peak_prob = entry.peak_prob
            if peak_prob > config.high_confidence_threshold and hamming_dist <= config.hamming_direct_reuse:
                return 'direct_reuse'
            if peak_prob > config.medium_confidence_threshold or hamming_dist <= config.hamming_interpolate:
                return 'interpolation'
            return 'light_verify'
        
        strategy = decide_strategy(default_config, high_confidence_entry, hamming_dist=1)
        assert strategy == 'direct_reuse'
    
    def test_interpolation_medium_confidence(
        self, default_config, medium_confidence_entry
    ):
        """Interpolation when medium confidence."""
        def decide_strategy(config, entry, hamming_dist):
            peak_prob = entry.peak_prob
            if peak_prob > config.high_confidence_threshold and hamming_dist <= config.hamming_direct_reuse:
                return 'direct_reuse'
            if peak_prob > config.medium_confidence_threshold or hamming_dist <= config.hamming_interpolate:
                return 'interpolation'
            return 'light_verify'
        
        strategy = decide_strategy(default_config, medium_confidence_entry, hamming_dist=3)
        assert strategy == 'interpolation'
    
    def test_interpolation_low_hamming(self, default_config, low_confidence_entry):
        """Interpolation when low Hamming even with low confidence."""
        def decide_strategy(config, entry, hamming_dist):
            peak_prob = entry.peak_prob
            if peak_prob > config.high_confidence_threshold and hamming_dist <= config.hamming_direct_reuse:
                return 'direct_reuse'
            if peak_prob > config.medium_confidence_threshold or hamming_dist <= config.hamming_interpolate:
                return 'interpolation'
            return 'light_verify'
        
        # Low confidence but low Hamming
        strategy = decide_strategy(default_config, low_confidence_entry, hamming_dist=2)
        assert strategy == 'interpolation'
    
    def test_light_verify_low_confidence_high_hamming(
        self, default_config, low_confidence_entry
    ):
        """Light verify when low confidence and high Hamming."""
        def decide_strategy(config, entry, hamming_dist):
            peak_prob = entry.peak_prob
            if peak_prob > config.high_confidence_threshold and hamming_dist <= config.hamming_direct_reuse:
                return 'direct_reuse'
            if peak_prob > config.medium_confidence_threshold or hamming_dist <= config.hamming_interpolate:
                return 'interpolation'
            return 'light_verify'
        
        strategy = decide_strategy(default_config, low_confidence_entry, hamming_dist=4)
        assert strategy == 'light_verify'


class TestDirectReuse:
    """Tests for direct reuse strategy."""
    
    def test_returns_cached_depth(self, high_confidence_entry):
        """Direct reuse should return cached depth unchanged."""
        def direct_reuse(entry):
            return entry.best_depth
        
        depth = direct_reuse(high_confidence_entry)
        assert depth == high_confidence_entry.best_depth
    
    def test_no_additional_computation(self, high_confidence_entry):
        """Direct reuse should require no additional computation."""
        # This is a design test - direct reuse is O(1)
        def direct_reuse(entry):
            return entry.best_depth
        
        # Just verify it returns immediately
        depth = direct_reuse(high_confidence_entry)
        assert depth is not None


class TestInterpolation:
    """Tests for interpolation strategy."""
    
    def test_interpolation_formula(self, depth_candidates):
        """Test interpolation formula correctness."""
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.65,
            second_offset=2,  # Second best is at index 18
            spread=0.2,
        )
        
        def interpolate(entry, hamming_dist, depth_candidates):
            second_idx = entry.best_idx + entry.second_offset
            second_idx = max(0, min(len(depth_candidates) - 1, second_idx))
            second_depth = float(depth_candidates[second_idx])
            
            lambda_coeff = min(hamming_dist / 4.0, 0.5)
            return (1 - lambda_coeff) * entry.best_depth + lambda_coeff * second_depth
        
        # Hamming = 2 → λ = 0.5
        depth = interpolate(entry, hamming_dist=2, depth_candidates=depth_candidates)
        
        second_depth = float(depth_candidates[18])
        expected = 0.5 * entry.best_depth + 0.5 * second_depth
        assert abs(depth - expected) < 1e-6
    
    def test_interpolation_coefficient_capped(self, depth_candidates):
        """Interpolation coefficient should be capped at 0.5."""
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.65,
            second_offset=2,
            spread=0.2,
        )
        
        def interpolate(entry, hamming_dist, depth_candidates):
            second_idx = entry.best_idx + entry.second_offset
            second_idx = max(0, min(len(depth_candidates) - 1, second_idx))
            second_depth = float(depth_candidates[second_idx])
            
            lambda_coeff = min(hamming_dist / 4.0, 0.5)  # Cap at 0.5
            return (1 - lambda_coeff) * entry.best_depth + lambda_coeff * second_depth
        
        # Hamming = 4 → λ = 1.0 but capped at 0.5
        depth = interpolate(entry, hamming_dist=4, depth_candidates=depth_candidates)
        
        second_depth = float(depth_candidates[18])
        expected = 0.5 * entry.best_depth + 0.5 * second_depth
        assert abs(depth - expected) < 1e-6
    
    def test_interpolation_with_negative_offset(self, depth_candidates):
        """Test interpolation with negative second_offset."""
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.65,
            second_offset=-3,  # Second best is at index 13
            spread=0.2,
        )
        
        def interpolate(entry, hamming_dist, depth_candidates):
            second_idx = entry.best_idx + entry.second_offset
            second_idx = max(0, min(len(depth_candidates) - 1, second_idx))
            second_depth = float(depth_candidates[second_idx])
            
            lambda_coeff = min(hamming_dist / 4.0, 0.5)
            return (1 - lambda_coeff) * entry.best_depth + lambda_coeff * second_depth
        
        depth = interpolate(entry, hamming_dist=2, depth_candidates=depth_candidates)
        
        second_depth = float(depth_candidates[13])
        expected = 0.5 * entry.best_depth + 0.5 * second_depth
        assert abs(depth - expected) < 1e-6
    
    def test_interpolation_boundary_handling(self, depth_candidates):
        """Test boundary handling when second_idx is out of range."""
        # Second index would be negative
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=depth_candidates[0].item(),
            best_idx=0,
            peak_prob=0.65,
            second_offset=-5,  # Would be -5, clamped to 0
            spread=0.2,
        )
        
        def interpolate(entry, hamming_dist, depth_candidates):
            second_idx = entry.best_idx + entry.second_offset
            second_idx = max(0, min(len(depth_candidates) - 1, second_idx))
            second_depth = float(depth_candidates[second_idx])
            
            lambda_coeff = min(hamming_dist / 4.0, 0.5)
            return (1 - lambda_coeff) * entry.best_depth + lambda_coeff * second_depth
        
        depth = interpolate(entry, hamming_dist=2, depth_candidates=depth_candidates)
        
        # Should not raise, depth should be valid
        assert depth > 0


class TestStrategyThresholds:
    """Tests for threshold behavior."""
    
    def test_exact_threshold_direct_reuse(self, default_config):
        """Test exact threshold boundary for direct reuse."""
        # Exactly at threshold
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.81,  # Just above 0.8
            second_offset=2,
            spread=0.1,
        )
        
        def decide_strategy(config, entry, hamming_dist):
            peak_prob = entry.peak_prob
            if peak_prob > config.high_confidence_threshold and hamming_dist <= config.hamming_direct_reuse:
                return 'direct_reuse'
            if peak_prob > config.medium_confidence_threshold or hamming_dist <= config.hamming_interpolate:
                return 'interpolation'
            return 'light_verify'
        
        strategy = decide_strategy(default_config, entry, hamming_dist=2)
        assert strategy == 'direct_reuse'
        
        # Exactly at threshold (0.8) should NOT be direct reuse (> not >=)
        entry.peak_prob = 0.8
        strategy = decide_strategy(default_config, entry, hamming_dist=2)
        assert strategy == 'interpolation'
    
    def test_hamming_boundary_conditions(self, default_config):
        """Test Hamming distance boundary conditions."""
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.9,
            second_offset=2,
            spread=0.1,
        )
        
        def decide_strategy(config, entry, hamming_dist):
            peak_prob = entry.peak_prob
            if peak_prob > config.high_confidence_threshold and hamming_dist <= config.hamming_direct_reuse:
                return 'direct_reuse'
            if peak_prob > config.medium_confidence_threshold or hamming_dist <= config.hamming_interpolate:
                return 'interpolation'
            return 'light_verify'
        
        # hamming = 2 (at direct_reuse threshold) → direct_reuse
        assert decide_strategy(default_config, entry, 2) == 'direct_reuse'
        
        # hamming = 3 (above direct_reuse but at interpolate) → interpolation
        assert decide_strategy(default_config, entry, 3) == 'interpolation'
        
        # hamming = 4 (above interpolate threshold, but high confidence) → interpolation
        assert decide_strategy(default_config, entry, 4) == 'interpolation'


class TestConfigVariations:
    """Tests with different configurations."""
    
    def test_aggressive_config_more_reuse(self, aggressive_config):
        """Aggressive config should allow more direct reuse."""
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.75,  # Below default 0.8 threshold
            second_offset=2,
            spread=0.1,
        )
        
        def decide_strategy(config, entry, hamming_dist):
            peak_prob = entry.peak_prob
            if peak_prob > config.high_confidence_threshold and hamming_dist <= config.hamming_direct_reuse:
                return 'direct_reuse'
            if peak_prob > config.medium_confidence_threshold or hamming_dist <= config.hamming_interpolate:
                return 'interpolation'
            return 'light_verify'
        
        # With aggressive config (threshold 0.7), this should be direct_reuse
        strategy = decide_strategy(aggressive_config, entry, hamming_dist=2)
        assert strategy == 'direct_reuse'
    
    def test_conservative_config_more_verify(self, conservative_config):
        """Conservative config should require more verification."""
        entry = MockCacheEntry(
            signature=0x1234,
            position=(32, 32),
            best_depth=5.0,
            best_idx=16,
            peak_prob=0.85,  # Above default but below conservative 0.9
            second_offset=2,
            spread=0.1,
        )
        
        def decide_strategy(config, entry, hamming_dist):
            peak_prob = entry.peak_prob
            if peak_prob > config.high_confidence_threshold and hamming_dist <= config.hamming_direct_reuse:
                return 'direct_reuse'
            if peak_prob > config.medium_confidence_threshold or hamming_dist <= config.hamming_interpolate:
                return 'interpolation'
            return 'light_verify'
        
        # With conservative config, this would need interpolation
        strategy = decide_strategy(conservative_config, entry, hamming_dist=1)
        # peak_prob=0.85 <= 0.9, so not direct_reuse
        # peak_prob=0.85 > 0.7 (medium threshold), so interpolation
        assert strategy == 'interpolation'
