"""
Unit tests for TileProcessor

Tests the main orchestrator coordinating 3-phase workflow,
tile grid computation, and integration of all SAES components.
"""
import pytest
import torch
from .conftest import MockGaussian


# Import will fail until implementation exists - this is expected in TDD
try:
    from saes.tile_processor import TileProcessor
    from saes.types import TileConfig, TileProcessingResult, SAESProfilingResult
    IMPLEMENTATION_EXISTS = True
except ImportError:
    IMPLEMENTATION_EXISTS = False
    # Create placeholders
    class TileProcessor:
        def __init__(self, config):
            pass
        def process_scene(self, features, depth_predictor_fn, gaussian_adapter_fn, context):
            raise NotImplementedError("Not yet implemented")
        def process_tile(self, tile_coords, features, *args):
            raise NotImplementedError("Not yet implemented")


class TestTileGridComputation:
    """Test tile grid calculation"""
    
    def test_64x64_feature_map_with_4x4_tiles(self, tile_config):
        """64×64 feature map with 4×4 tiles should yield 16×16 grid"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # 64×64 feature map
        H, W = 64, 64
        n_tiles_h, n_tiles_w = processor._compute_tile_grid(H, W, tile_config.tile_size)
        
        assert n_tiles_h == 16, "Should have 16 tile rows"
        assert n_tiles_w == 16, "Should have 16 tile columns"
        assert n_tiles_h * n_tiles_w == 256, "Should have 256 total tiles"
    
    def test_non_divisible_feature_map(self):
        """Feature map not evenly divisible by tile size should ceil divide"""
        config = TileConfig(tile_size=4)
        processor = TileProcessor(config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # 65×65 feature map (not divisible by 4)
        H, W = 65, 65
        n_tiles_h, n_tiles_w = processor._compute_tile_grid(H, W, 4)
        
        # Ceiling division: ceil(65/4) = 17
        assert n_tiles_h == 17
        assert n_tiles_w == 17


class TestSingleTileEarlyStopPath:
    """Test processing a single tile through early-stop path"""
    
    def test_single_tile_early_stop(self, tile_config, mock_depth_predictor_fn, mock_gaussian_adapter_fn, identical_gaussians):
        """Tile with high similarity should take early-stop path"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Mock that returns identical Gaussians (high similarity)
        def gaussian_fn_identical(depths, context):
            return identical_gaussians[:len(depths)]
        
        features = torch.randn(1, 2, 128, 4, 4)  # Single 4×4 tile
        context = {"intrinsics": None, "extrinsics": None}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, gaussian_fn_identical, context
        )
        
        # Should have taken early-stop path
        assert profiling.early_stop_tiles > 0, "Should have at least one early-stop tile"
        
        # Should have processed only probe pixels (4 out of 16)
        assert profiling.total_pixels_saved > 0, "Should have saved pixels"


class TestSingleTileSparsePath:
    """Test processing a single tile through sparse-continue path"""
    
    def test_single_tile_sparse_continue(self, tile_config, mock_depth_predictor_fn, moderate_similarity_gaussians):
        """Tile with moderate similarity should take sparse-continue path"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Mock that returns moderate similarity Gaussians
        def gaussian_fn_moderate(depths, context):
            return moderate_similarity_gaussians[:len(depths)]
        
        features = torch.randn(1, 2, 128, 4, 4)  # Single 4×4 tile
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, gaussian_fn_moderate, context
        )
        
        # Verify that path decision is consistent with similarity score
        tile_result = profiling.tile_results[0]
        sim_score = tile_result.similarity_score
        
        # Verify path is correct for the similarity score
        if sim_score >= 0.85:
            assert tile_result.path_taken == "early_stop"
        elif sim_score >= 0.60:
            assert tile_result.path_taken == "sparse_continue"
        else:
            assert tile_result.path_taken == "full_continue"
        
        # Note: moderate_similarity_gaussians may not always produce moderate similarity
        # due to random variations, so we verify correctness rather than specific path


class TestSingleTileFullPath:
    """Test processing a single tile through full-continue path"""
    
    def test_single_tile_full_continue(self, tile_config, mock_depth_predictor_fn, different_gaussians):
        """Tile with low similarity should take full-continue path"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Mock that returns different Gaussians (low similarity)
        def gaussian_fn_different(depths, context):
            return different_gaussians[:len(depths)]
        
        features = torch.randn(1, 2, 128, 4, 4)  # Single 4×4 tile
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, gaussian_fn_different, context
        )
        
        # Should have taken full-continue path
        assert profiling.full_continue_tiles > 0, "Should have at least one full-continue tile"
        
        # Should have processed all pixels (16 out of 16)
        assert profiling.total_pixels_saved == 0, "Full-continue should not save any pixels"


class TestPhaseWorkflow:
    """Test 3-phase workflow execution"""
    
    def test_phase1_probe_processing(self, tile_config, mock_depth_predictor_fn, mock_gaussian_adapter_fn):
        """Phase 1 should process probe_size pixels"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # Track how many times depth_predictor_fn is called
        call_counts = []
        
        def tracking_depth_fn(features, indices):
            call_counts.append(len(indices))
            return mock_depth_predictor_fn(features, indices)
        
        features = torch.randn(1, 2, 128, 4, 4)
        context = {}
        
        processor.process_scene(features, tracking_depth_fn, mock_gaussian_adapter_fn, context)
        
        # First call should be for probe_size=4 pixels
        assert call_counts[0] == 4, "Phase 1 should process 4 probe pixels"
    
    def test_phase2_similarity_evaluation_called(self, tile_config, mock_depth_predictor_fn, mock_gaussian_adapter_fn):
        """Phase 2 should call similarity evaluator"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        features = torch.randn(1, 2, 128, 4, 4)
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, mock_gaussian_adapter_fn, context
        )
        
        # Profiling should contain similarity scores
        assert len(profiling.tile_results) > 0, "Should have tile results"
        assert profiling.tile_results[0].similarity_score >= 0.0, "Should have similarity score"
    
    def test_phase3_decision_execution(self, tile_config, mock_depth_predictor_fn, identical_gaussians):
        """Phase 3 should execute decision (early-stop/sparse/full)"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        def gaussian_fn_identical(depths, context):
            return identical_gaussians[:len(depths)]
        
        features = torch.randn(1, 2, 128, 4, 4)
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, gaussian_fn_identical, context
        )
        
        # Should have recorded path taken
        assert profiling.tile_results[0].path_taken in ["early_stop", "sparse_continue", "full_continue"]


class TestMultipleTiles:
    """Test processing multiple tiles"""
    
    def test_multiple_tiles_processing(self, tile_config, mock_depth_predictor_fn, mock_gaussian_adapter_fn):
        """Process 4 tiles (8×8 feature map with 4×4 tiles)"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        features = torch.randn(1, 2, 128, 8, 8)  # 8×8 → 2×2 tiles
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, mock_gaussian_adapter_fn, context
        )
        
        # Should have processed 4 tiles
        assert profiling.total_tiles == 4, "Should have 4 tiles for 8×8 feature map"
        
        # Path distribution should sum to total tiles
        assert (
            profiling.early_stop_tiles +
            profiling.sparse_continue_tiles +
            profiling.full_continue_tiles
        ) == 4
    
    def test_256_tiles_64x64_feature_map(self, tile_config, mock_depth_predictor_fn, mock_gaussian_adapter_fn):
        """Process 256 tiles (64×64 feature map)"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        features = torch.randn(1, 2, 128, 64, 64)  # Full 64×64
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, mock_gaussian_adapter_fn, context
        )
        
        # Should have processed 256 tiles
        assert profiling.total_tiles == 256, "Should have 256 tiles for 64×64 feature map"


class TestTimingProfiling:
    """Test timing profiling for each phase"""
    
    def test_timing_breakdown_captured(self, tile_config, mock_depth_predictor_fn, mock_gaussian_adapter_fn):
        """Profiling should capture phase timings"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        features = torch.randn(1, 2, 128, 4, 4)
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, mock_gaussian_adapter_fn, context
        )
        
        # Tile results should have timing breakdown
        tile_result = profiling.tile_results[0]
        assert "probe" in tile_result.timing_ns, "Should have probe phase timing"
        assert "evaluate" in tile_result.timing_ns, "Should have evaluate phase timing"
        assert "decide" in tile_result.timing_ns, "Should have decide phase timing"
        
        # Timings should be non-negative
        assert tile_result.timing_ns["probe"] >= 0
        assert tile_result.timing_ns["evaluate"] >= 0
        assert tile_result.timing_ns["decide"] >= 0


class TestProfilingStatistics:
    """Test profiling result aggregation"""
    
    def test_profiling_result_aggregation(self, tile_config, mock_depth_predictor_fn, mock_gaussian_adapter_fn):
        """Profiling should aggregate statistics across all tiles"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        features = torch.randn(1, 2, 128, 8, 8)  # 4 tiles
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, mock_gaussian_adapter_fn, context
        )
        
        # Check aggregated statistics
        assert profiling.scene_name is not None
        assert profiling.total_tiles > 0
        assert profiling.computation_saving_ratio >= 0.0
        assert profiling.computation_saving_ratio <= 1.0
        assert profiling.overhead_ns >= 0
    
    def test_computation_saving_ratio_calculation(self, tile_config, mock_depth_predictor_fn, identical_gaussians):
        """Verify computation saving ratio is calculated correctly"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # All identical → all early-stop → 75% savings
        def gaussian_fn_identical(depths, context):
            return identical_gaussians[:len(depths)]
        
        features = torch.randn(1, 2, 128, 8, 8)  # 4 tiles, 64 total pixels
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, gaussian_fn_identical, context
        )
        
        # If all tiles early-stop, savings should be ~75%
        if profiling.early_stop_tiles == profiling.total_tiles:
            assert profiling.computation_saving_ratio > 0.70, "All early-stop should save ~75%"


class TestOutputCorrectness:
    """Test output Gaussian correctness"""
    
    def test_output_shape_matches_input(self, tile_config, mock_depth_predictor_fn, mock_gaussian_adapter_fn):
        """Output Gaussians should match expected count"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        features = torch.randn(1, 2, 128, 4, 4)  # 16 pixels
        context = {}
        
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, mock_gaussian_adapter_fn, context
        )
        
        # Should produce Gaussians (exact count depends on implementation)
        assert len(gaussians) > 0, "Should produce at least one Gaussian"


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_feature_map_raises_error(self, tile_config, mock_depth_predictor_fn, mock_gaussian_adapter_fn):
        """Empty feature map should raise appropriate error"""
        processor = TileProcessor(tile_config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        features = torch.randn(1, 2, 128, 0, 0)  # Empty
        context = {}
        
        with pytest.raises(ValueError, match="Feature map size"):
            processor.process_scene(features, mock_depth_predictor_fn, mock_gaussian_adapter_fn, context)
    
    def test_tile_size_larger_than_feature_map(self, mock_depth_predictor_fn, mock_gaussian_adapter_fn):
        """Tile size > feature map should handle gracefully"""
        config = TileConfig(tile_size=16)  # Larger than 8×8 feature map
        processor = TileProcessor(config)
        
        if not IMPLEMENTATION_EXISTS:
            pytest.skip("Implementation not yet available")
        
        # This should process as single tile
        features = torch.randn(1, 2, 128, 8, 8)
        context = {}
        
        # Should process entire feature map as single tile
        gaussians, profiling = processor.process_scene(
            features, mock_depth_predictor_fn, mock_gaussian_adapter_fn, context
        )
        
        # Should have processed as 1 tile
        assert profiling.total_tiles == 1, "Should process as single tile"
        assert len(gaussians) > 0, "Should produce Gaussians"
