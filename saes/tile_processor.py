"""
Tile Processor - Main SAES Engine

Orchestrates tile-based processing with 3-phase feedback workflow:
1. Probe: Process first N pixels
2. Evaluate: Compute 3D Gaussian similarity
3. Decide: Select path (early-stop/sparse/full) and execute
"""
import torch
from torch import Tensor
from typing import Callable, Dict, List, Tuple
import time

from .types import Gaussian, TileConfig, TileProcessingResult, SAESProfilingResult
from .similarity_evaluator import GaussianSimilarityEvaluator
from .decision_controller import DecisionController
from .gaussian_merger import GaussianMerger
from .profiler import SAESProfiler


class TileProcessor:
    """
    Main SAES tile processing engine
    
    Coordinates 3-phase workflow for each tile and aggregates results.
    Maintains decoupling from transplat via callback functions.
    """
    
    def __init__(self, config: TileConfig):
        """
        Initialize tile processor with configuration
        
        Args:
            config: TileConfig with tile size, thresholds, etc.
        """
        self.config = config
        
        # Initialize components
        self.similarity_evaluator = GaussianSimilarityEvaluator()
        self.decision_controller = DecisionController(config)
        self.gaussian_merger = GaussianMerger()
        self.profiler = None  # Created per scene
    
    def process_scene(
        self,
        features: Tensor,
        depth_predictor_fn: Callable,
        gaussian_adapter_fn: Callable,
        context: Dict,
    ) -> Tuple[List[Gaussian], SAESProfilingResult]:
        """
        Process entire scene with tile-based SAES
        
        Args:
            features: [B, V, C, H, W] - Feature map from backbone
            depth_predictor_fn: fn(features, indices) -> depths
            gaussian_adapter_fn: fn(depths, context) -> List[Gaussian]
            context: Camera parameters (intrinsics, extrinsics, etc.)
        
        Returns:
            - List[Gaussian]: Final Gaussians for entire scene
            - SAESProfilingResult: Performance metrics
        
        Raises:
            ValueError: If feature map size is invalid
        """
        B, V, C, H, W = features.shape
        
        if H == 0 or W == 0:
            raise ValueError(f"Feature map size invalid: H={H}, W={W}")
        
        # Initialize profiler for this scene
        scene_name = context.get("scene_name", "unknown")
        self.profiler = SAESProfiler(scene_name)
        
        # Compute tile grid
        n_tiles_h, n_tiles_w = self._compute_tile_grid(H, W, self.config.tile_size)
        
        # Collect all Gaussians from all tiles
        all_gaussians = []
        
        # Process each tile
        for tile_row in range(n_tiles_h):
            for tile_col in range(n_tiles_w):
                tile_id = (tile_row, tile_col)
                tile_coords = self._tile_coordinates(tile_id, H, W, self.config.tile_size)
                
                # Process single tile
                tile_gaussians = self._process_tile(
                    tile_id,
                    tile_coords,
                    features,
                    depth_predictor_fn,
                    gaussian_adapter_fn,
                    context,
                )
                
                all_gaussians.extend(tile_gaussians)
        
        # Aggregate profiling results
        profiling_result = self.profiler.get_scene_summary(tile_size=self.config.tile_size)
        
        return all_gaussians, profiling_result
    
    def _process_tile(
        self,
        tile_id: Tuple[int, int],
        tile_coords: Tuple[int, int, int, int],
        features: Tensor,
        depth_predictor_fn: Callable,
        gaussian_adapter_fn: Callable,
        context: Dict,
    ) -> List[Gaussian]:
        """
        Process single tile through 3-phase workflow
        
        Args:
            tile_id: (row, col) tile coordinates
            tile_coords: (row_start, row_end, col_start, col_end) pixel coordinates
            features: [B, V, C, H, W] - Full feature map
            depth_predictor_fn: Depth prediction callback
            gaussian_adapter_fn: Gaussian generation callback
            context: Camera parameters
        
        Returns:
            List[Gaussian] for this tile
        """
        self.profiler.start_tile(tile_id)
        row_start, row_end, col_start, col_end = tile_coords
        
        # Extract tile features
        tile_features = self._extract_tile_features(features, tile_coords)
        
        # Phase 1: Probe - process first N pixels
        self.profiler.start_phase("probe")
        probe_indices = list(range(self.config.probe_size))
        probe_gaussians = self._process_pixels(
            tile_features,
            probe_indices,
            depth_predictor_fn,
            gaussian_adapter_fn,
            context,
        )
        self.profiler.end_phase("probe")
        
        # Phase 2: Evaluate - compute 3D similarity
        self.profiler.start_phase("evaluate")
        
        # Auto-estimate scene scale from probe Gaussians
        if self.similarity_evaluator.scene_scale is None:
            positions = torch.stack([g.mean for g in probe_gaussians])
            self.similarity_evaluator.scene_scale = positions.std().item()
        
        metrics = self.similarity_evaluator.evaluate(probe_gaussians)
        similarity_score = metrics.similarity_score
        self.profiler.end_phase("evaluate")
        
        # Phase 3: Decide - select path and execute
        self.profiler.start_phase("decide")
        path = self.decision_controller.decide_path(similarity_score)
        remaining_indices = self.decision_controller.get_remaining_indices(
            path, self.config.tile_size, self.config.probe_size
        )
        self.profiler.end_phase("decide")
        
        # Execute path
        if path == "early_stop":
            # Merge and enlarge probe Gaussians
            tile_gaussians = self.gaussian_merger.enlarge_and_merge(
                probe_gaussians, tile_coords
            )
            pixels_processed = self.config.probe_size
            gaussians_generated = len(tile_gaussians)
        
        elif path in ["sparse_continue", "full_continue"]:
            # Process remaining pixels
            remaining_gaussians = self._process_pixels(
                tile_features,
                remaining_indices,
                depth_predictor_fn,
                gaussian_adapter_fn,
                context,
            )
            
            # Combine probe + remaining
            tile_gaussians = probe_gaussians + remaining_gaussians
            pixels_processed = self.config.probe_size + len(remaining_indices)
            gaussians_generated = len(tile_gaussians)
        
        else:
            raise RuntimeError(f"Unexpected path: {path}")
        
        # Record tile result
        tile_result = TileProcessingResult(
            tile_id=tile_id,
            path_taken=path,
            pixels_processed=pixels_processed,
            gaussians_generated=gaussians_generated,
            similarity_score=similarity_score,
            timing_ns=self.profiler._current_phase_timings.copy(),
        )
        self.profiler.finish_tile(tile_result)
        
        return tile_gaussians
    
    def _process_pixels(
        self,
        tile_features: Tensor,
        indices: List[int],
        depth_predictor_fn: Callable,
        gaussian_adapter_fn: Callable,
        context: Dict,
    ) -> List[Gaussian]:
        """
        Process specified pixels through depth prediction + Gaussian generation
        
        Args:
            tile_features: [B, V, C, tile_h, tile_w] - Tile feature slice
            indices: Pixel indices to process (in flattened tile order)
            depth_predictor_fn: Depth prediction callback
            gaussian_adapter_fn: Gaussian generation callback
            context: Camera parameters
        
        Returns:
            List[Gaussian] for specified pixels
        """
        if not indices:
            return []
        
        # Call depth predictor for these indices
        # Note: Implementation depends on depth_predictor_fn interface
        # For now, assume it accepts features and indices
        depths = depth_predictor_fn(tile_features, indices)
        
        # Convert to Gaussians
        gaussians = gaussian_adapter_fn(depths, context)
        
        return gaussians
    
    def _extract_tile_features(
        self,
        features: Tensor,
        tile_coords: Tuple[int, int, int, int],
    ) -> Tensor:
        """
        Slice feature tensor for tile region
        
        Args:
            features: [B, V, C, H, W] - Full feature map
            tile_coords: (row_start, row_end, col_start, col_end)
        
        Returns:
            [B, V, C, tile_h, tile_w] - Tile feature slice
        """
        row_start, row_end, col_start, col_end = tile_coords
        return features[:, :, :, row_start:row_end, col_start:col_end]
    
    def _compute_tile_grid(self, H: int, W: int, tile_size: int) -> Tuple[int, int]:
        """
        Compute number of tiles needed to cover feature map
        
        Args:
            H, W: Feature map dimensions
            tile_size: Pixels per tile edge
        
        Returns:
            (n_tiles_h, n_tiles_w)
        
        Example:
            H=64, W=64, tile_size=4 → (16, 16)
            H=65, W=65, tile_size=4 → (17, 17)  # Ceiling division
        """
        # Ceiling division to handle non-divisible sizes
        n_tiles_h = (H + tile_size - 1) // tile_size
        n_tiles_w = (W + tile_size - 1) // tile_size
        
        return n_tiles_h, n_tiles_w
    
    def _tile_coordinates(
        self,
        tile_id: Tuple[int, int],
        H: int,
        W: int,
        tile_size: int,
    ) -> Tuple[int, int, int, int]:
        """
        Convert tile ID to pixel coordinates
        
        Args:
            tile_id: (row, col) in tile grid
            H, W: Feature map dimensions
            tile_size: Pixels per tile edge
        
        Returns:
            (row_start, row_end, col_start, col_end) in pixel coordinates
        
        Example:
            tile_id=(1, 2), tile_size=4 → (4, 8, 8, 12)
        """
        tile_row, tile_col = tile_id
        
        row_start = tile_row * tile_size
        row_end = min(row_start + tile_size, H)  # Clamp at edge
        
        col_start = tile_col * tile_size
        col_end = min(col_start + tile_size, W)  # Clamp at edge
        
        return row_start, row_end, col_start, col_end
