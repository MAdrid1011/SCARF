"""
Unit tests for DecisionController

Tests the path selection logic based on similarity scores and
remaining pixel index generation for different processing paths.
"""
import pytest


# Import will fail until implementation exists - this is expected in TDD
try:
    from saes.decision_controller import DecisionController
    from saes.types import TileConfig
    IMPLEMENTATION_EXISTS = True
except ImportError:
    IMPLEMENTATION_EXISTS = False
    # Create placeholder for test development
    class DecisionController:
        def __init__(self, config):
            self.config = config
        def decide_path(self, similarity_score):
            raise NotImplementedError("Not yet implemented")
        def get_remaining_indices(self, path, tile_size, probe_size):
            raise NotImplementedError("Not yet implemented")


class TestPathSelection:
    """Test path selection based on similarity thresholds"""
    
    def test_high_similarity_triggers_early_stop(self, tile_config):
        """Similarity score > 0.85 should trigger early_stop path"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Test scores above high threshold
        assert controller.decide_path(0.90) == "early_stop"
        assert controller.decide_path(0.87) == "early_stop"
        assert controller.decide_path(0.86) == "early_stop"
    
    def test_high_similarity_exact_threshold_triggers_early_stop(self, tile_config):
        """Similarity score exactly at 0.85 should trigger early_stop (boundary case)"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Exact threshold should be early_stop (>= comparison)
        assert controller.decide_path(0.85) == "early_stop"
    
    def test_medium_similarity_triggers_sparse_continue(self, tile_config):
        """Similarity score in (0.60, 0.85) should trigger sparse_continue path"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Test scores in medium range
        assert controller.decide_path(0.80) == "sparse_continue"
        assert controller.decide_path(0.70) == "sparse_continue"
        assert controller.decide_path(0.65) == "sparse_continue"
        assert controller.decide_path(0.61) == "sparse_continue"
    
    def test_low_similarity_exact_threshold_triggers_sparse_continue(self, tile_config):
        """Similarity score exactly at 0.60 should trigger sparse_continue (boundary case)"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Exact threshold should be sparse_continue (>= comparison)
        assert controller.decide_path(0.60) == "sparse_continue"
    
    def test_low_similarity_triggers_full_continue(self, tile_config):
        """Similarity score < 0.60 should trigger full_continue path"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Test scores below low threshold
        assert controller.decide_path(0.59) == "full_continue"
        assert controller.decide_path(0.50) == "full_continue"
        assert controller.decide_path(0.30) == "full_continue"
        assert controller.decide_path(0.10) == "full_continue"
        assert controller.decide_path(0.00) == "full_continue"


class TestRemainingIndicesEarlyStop:
    """Test remaining pixel indices for early-stop path"""
    
    def test_early_stop_returns_empty_list(self, tile_config):
        """Early-stop path should return empty list (no more pixels to process)"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        indices = controller.get_remaining_indices("early_stop", tile_size=4, probe_size=4)
        
        assert indices == [], "Early-stop should skip all remaining pixels"
    
    def test_early_stop_with_different_tile_sizes(self):
        """Early-stop should always return empty regardless of tile size"""
        config = TileConfig(tile_size=8, probe_size=4)
        controller = DecisionController(config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        indices = controller.get_remaining_indices("early_stop", tile_size=8, probe_size=4)
        assert indices == []


class TestRemainingIndicesSparse:
    """Test remaining pixel indices for sparse-continue path"""
    
    def test_sparse_continue_returns_configured_indices(self, tile_config):
        """Sparse-continue should return sparse_indices from config"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        indices = controller.get_remaining_indices("sparse_continue", tile_size=4, probe_size=4)
        
        # Default config has sparse_indices = [4, 7, 11, 15]
        assert indices == [4, 7, 11, 15], "Sparse-continue should return configured sparse indices"
    
    def test_sparse_continue_with_custom_indices(self):
        """Sparse-continue should respect custom sparse_indices config"""
        config = TileConfig(
            tile_size=4,
            probe_size=4,
            sparse_indices=[5, 6, 9, 10],  # Custom center-weighted pattern
        )
        controller = DecisionController(config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        indices = controller.get_remaining_indices("sparse_continue", tile_size=4, probe_size=4)
        
        assert indices == [5, 6, 9, 10]
    
    def test_sparse_continue_8x8_tile(self):
        """Sparse-continue with 8×8 tile should use configured indices"""
        config = TileConfig(
            tile_size=8,
            probe_size=4,
            sparse_indices=[4, 11, 22, 33, 44, 55],  # Diagonal pattern for 8×8
        )
        controller = DecisionController(config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        indices = controller.get_remaining_indices("sparse_continue", tile_size=8, probe_size=4)
        
        assert indices == [4, 11, 22, 33, 44, 55]


class TestRemainingIndicesFull:
    """Test remaining pixel indices for full-continue path"""
    
    def test_full_continue_returns_all_remaining_pixels(self, tile_config):
        """Full-continue should return all pixels from probe_size to tile_size²"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        indices = controller.get_remaining_indices("full_continue", tile_size=4, probe_size=4)
        
        # Tile = 4×4 = 16 pixels, probe = 4, remaining = [4..15]
        expected = list(range(4, 16))
        assert indices == expected, "Full-continue should process all remaining pixels"
    
    def test_full_continue_8x8_tile(self):
        """Full-continue with 8×8 tile should return [4..63]"""
        config = TileConfig(tile_size=8, probe_size=4)
        controller = DecisionController(config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        indices = controller.get_remaining_indices("full_continue", tile_size=8, probe_size=4)
        
        # Tile = 8×8 = 64 pixels, probe = 4, remaining = [4..63]
        expected = list(range(4, 64))
        assert indices == expected
    
    def test_full_continue_with_larger_probe(self):
        """Full-continue with probe_size=8 should return [8..15] for 4×4 tile"""
        config = TileConfig(tile_size=4, probe_size=8)
        controller = DecisionController(config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        indices = controller.get_remaining_indices("full_continue", tile_size=4, probe_size=8)
        
        # Tile = 4×4 = 16 pixels, probe = 8, remaining = [8..15]
        expected = list(range(8, 16))
        assert indices == expected


class TestCustomThresholds:
    """Test custom threshold configurations"""
    
    def test_conservative_thresholds(self):
        """Test with conservative (quality-prioritized) thresholds"""
        config = TileConfig(
            high_similarity_threshold=0.90,
            low_similarity_threshold=0.70,
        )
        controller = DecisionController(config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Higher thresholds → fewer early-stops
        assert controller.decide_path(0.95) == "early_stop"
        assert controller.decide_path(0.85) == "sparse_continue"  # Would be early_stop with default
        assert controller.decide_path(0.75) == "sparse_continue"
        assert controller.decide_path(0.65) == "full_continue"  # Would be sparse_continue with default
    
    def test_aggressive_thresholds(self):
        """Test with aggressive (speed-prioritized) thresholds"""
        config = TileConfig(
            high_similarity_threshold=0.80,
            low_similarity_threshold=0.50,
        )
        controller = DecisionController(config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Lower thresholds → more early-stops
        assert controller.decide_path(0.85) == "early_stop"
        assert controller.decide_path(0.75) == "early_stop"  # Would be sparse_continue with default
        assert controller.decide_path(0.60) == "sparse_continue"
        assert controller.decide_path(0.45) == "full_continue"


class TestThresholdValidation:
    """Test threshold validation logic"""
    
    def test_invalid_thresholds_should_raise_error(self):
        """high_threshold must be > low_threshold"""
        # This test verifies that DecisionController validates thresholds at init
        invalid_config = TileConfig(
            high_similarity_threshold=0.60,
            low_similarity_threshold=0.85,  # Invalid: low > high
        )
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        with pytest.raises(ValueError, match="high_similarity_threshold.*low_similarity_threshold"):
            DecisionController(invalid_config)
    
    def test_equal_thresholds_should_raise_error(self):
        """high_threshold == low_threshold should be invalid"""
        invalid_config = TileConfig(
            high_similarity_threshold=0.75,
            low_similarity_threshold=0.75,  # Invalid: equal
        )
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        with pytest.raises(ValueError):
            DecisionController(invalid_config)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_similarity_score_exactly_one(self, tile_config):
        """Similarity = 1.0 (perfect) should trigger early_stop"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        assert controller.decide_path(1.0) == "early_stop"
    
    def test_similarity_score_exactly_zero(self, tile_config):
        """Similarity = 0.0 (no similarity) should trigger full_continue"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        assert controller.decide_path(0.0) == "full_continue"
    
    def test_invalid_path_name_raises_error(self, tile_config):
        """get_remaining_indices with invalid path should raise error"""
        controller = DecisionController(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        with pytest.raises(ValueError, match="Invalid path"):
            controller.get_remaining_indices("invalid_path", tile_size=4, probe_size=4)
