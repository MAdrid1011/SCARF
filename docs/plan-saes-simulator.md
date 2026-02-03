# Implementation Plan: SAES Hardware Simulator

## Goal

Implement a Python-based hardware simulator for SAES (Scene-Adaptive Early-Stopping Dataflow) that models the feedback-driven tile processing architecture described in Design2, enabling performance evaluation and validation on real RE10K datasets without actual hardware implementation.

**Success criteria:**
- SAES simulator accurately models tile-based processing with 3-phase workflow (probe → evaluate → decide)
- 3D Gaussian similarity evaluator computes position/covariance/color/opacity metrics as specified
- Decision controller implements 3-level path selection (early-stop / sparse-continue / full-continue)
- Modular architecture enables unit testing of each component independently
- Integration tests validate end-to-end pipeline using real RE10K scenes
- Performance metrics report computation savings, timing overhead, and quality impact

**Out of scope:**
- Actual hardware synthesis or RTL implementation
- FSDR (Feature-Similarity Depth Reuse) integration
- Real-time optimization or GPU kernel implementation
- Training or model modification (inference-only evaluation)

## Codebase Analysis

**Files to modify:**
- `src/model/encoder/encoder_trans.py:246-263` - Add SAES simulator integration point
- `src/model/model_wrapper.py:59-64` - Add SAES profiling configuration flags
- `scripts/run_all_timing_tests.sh:1-50` - Add SAES testing command options

**Files to create:**
- `SCARF/saes/__init__.py` - SAES package initialization (Est: 15 LOC)
- `SCARF/saes/tile_processor.py` - Tile-based processing engine (Est: 280 LOC)
- `SCARF/saes/similarity_evaluator.py` - 3D Gaussian similarity computation (Est: 180 LOC)
- `SCARF/saes/decision_controller.py` - Path selection logic (Est: 150 LOC)
- `SCARF/saes/gaussian_merger.py` - Gaussian enlargement for early-stop path (Est: 120 LOC)
- `SCARF/saes/profiler.py` - Performance metrics collection (Est: 200 LOC)
- `SCARF/saes/types.py` - Data structures and type definitions (Est: 100 LOC)
- `tests/saes/test_tile_processor.py` - Unit tests for tile processor (Est: 220 LOC)
- `tests/saes/test_similarity_evaluator.py` - Unit tests for similarity evaluator (Est: 180 LOC)
- `tests/saes/test_decision_controller.py` - Unit tests for decision controller (Est: 160 LOC)
- `tests/saes/test_gaussian_merger.py` - Unit tests for Gaussian merger (Est: 140 LOC)
- `tests/saes/test_integration_re10k.py` - Integration tests on RE10K (Est: 250 LOC)
- `docs/saes-architecture.md` - Architecture documentation (Est: 400 lines)
- `docs/saes-usage.md` - Usage guide and examples (Est: 250 lines)

**Current architecture notes:**
- Encoder pipeline follows: backbone → depth_predictor → gaussian_adapter
- Depth predictor generates raw Gaussians per-pixel sequentially
- Model wrapper provides benchmarker for stage-level timing
- Existing analyzer scripts (analyze_gaussian_smoothness.py) provide patterns for Gaussian analysis
- Test infrastructure uses real datasets via test_cfg configuration

## Interface Design

**New interfaces:**

```python
# SCARF/saes/types.py
@dataclass
class TileConfig:
    """Tile processing configuration"""
    tile_size: int = 4  # 4x4 tile
    probe_size: int = 4  # Number of pixels in probe phase
    sparse_indices: List[int] = field(default_factory=lambda: [4, 7, 11, 15])
    high_similarity_threshold: float = 0.85
    low_similarity_threshold: float = 0.60

@dataclass
class GaussianSimilarityMetrics:
    """3D Gaussian similarity evaluation metrics"""
    position_dispersion: float  # Max pairwise position difference (normalized)
    covariance_dispersion: float  # Max pairwise covariance difference
    color_dispersion: float  # Max pairwise SH color difference
    opacity_dispersion: float  # Max pairwise opacity difference
    similarity_score: float  # Overall similarity (0-1, higher is more similar)

@dataclass
class TileProcessingResult:
    """Result from processing a single tile"""
    tile_id: Tuple[int, int]  # (row, col)
    path_taken: str  # 'early_stop', 'sparse_continue', 'full_continue'
    pixels_processed: int  # Number of pixels that underwent depth search
    gaussians_generated: int  # Number of Gaussians generated
    similarity_score: float  # Similarity score from probe phase
    timing_ns: Dict[str, int]  # Timing breakdown by phase

@dataclass
class SAESProfilingResult:
    """Overall SAES profiling statistics"""
    scene_name: str
    total_tiles: int
    early_stop_tiles: int
    sparse_continue_tiles: int
    full_continue_tiles: int
    total_pixels_saved: int  # Pixels that skipped depth search
    computation_saving_ratio: float  # Percentage of saved computation
    overhead_ns: int  # SAES overhead (similarity eval + decision)
    net_speedup: float  # Overall speedup considering overhead

# SCARF/saes/tile_processor.py
class TileProcessor:
    """Tile-based processing engine with feedback control"""
    
    def __init__(self, config: TileConfig):
        self.config = config
        self.similarity_evaluator = GaussianSimilarityEvaluator()
        self.decision_controller = DecisionController(config)
        self.gaussian_merger = GaussianMerger()
        self.profiler = SAESProfiler()
    
    def process_scene(
        self,
        features: Tensor,  # [B, V, C, H, W]
        depth_predictor_fn: Callable,  # Function to perform depth search + Gaussian generation
        gaussian_adapter_fn: Callable,  # Function to convert to final Gaussians
        context: Dict,  # Camera parameters
    ) -> Tuple[Gaussians, SAESProfilingResult]:
        """Process entire scene with tile-based SAES"""
        ...

# SCARF/saes/similarity_evaluator.py
class GaussianSimilarityEvaluator:
    """3D Gaussian similarity computation"""
    
    def __init__(self, scene_scale: Optional[float] = None):
        self.scene_scale = scene_scale
    
    def evaluate(
        self,
        gaussians: List[Gaussian],  # Probe Gaussians
    ) -> GaussianSimilarityMetrics:
        """
        Compute similarity metrics among probe Gaussians
        Returns metrics with similarity_score in [0, 1]
        """
        ...

# SCARF/saes/decision_controller.py
class DecisionController:
    """Path selection based on similarity score"""
    
    def __init__(self, config: TileConfig):
        self.config = config
    
    def decide_path(
        self,
        similarity_score: float,
    ) -> str:
        """
        Returns: 'early_stop', 'sparse_continue', or 'full_continue'
        """
        ...
    
    def get_remaining_indices(
        self,
        path: str,
        tile_size: int,
        probe_size: int,
    ) -> List[int]:
        """Get pixel indices to process based on path"""
        ...

# SCARF/saes/gaussian_merger.py
class GaussianMerger:
    """Gaussian enlargement for early-stop path"""
    
    def enlarge_and_merge(
        self,
        probe_gaussians: Gaussians,
        tile_coverage: Tuple[int, int, int, int],  # (row_start, row_end, col_start, col_end)
    ) -> Gaussians:
        """
        Enlarge probe Gaussians to cover entire tile
        Strategy: increase covariance scale to cover tile area
        """
        ...
```

**Modified interfaces:**

```python
# src/model/encoder/encoder_trans.py
class EncoderTrans(Encoder[EncoderTransCfg]):
    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        benchmarker = None,
        analyze_feature_depth: bool = False,
        enable_saes: bool = False,  # NEW: Enable SAES simulation
        saes_config: Optional[TileConfig] = None,  # NEW: SAES configuration
    ) -> Gaussians:
        ...
        # Integration point after depth predictor
        if enable_saes:
            gaussians, saes_result = self.saes_processor.process_scene(
                trans_features, self.depth_predictor, self.gaussian_adapter, context
            )
        else:
            # Original path
            depths, densities, raw_gaussians = self.depth_predictor(...)
            gaussians = self.gaussian_adapter(...)
        ...

# src/model/model_wrapper.py
@dataclass
class TestCfg:
    ...
    analyze_gaussian_smoothness: bool = False
    enable_saes_simulation: bool = False  # NEW: Enable SAES hardware simulation
    saes_tile_size: int = 4  # NEW: SAES tile size
```

**Documentation changes:**
- `docs/saes-architecture.md` - New document detailing SAES simulator architecture
- `docs/saes-usage.md` - New document with usage examples and configuration guide
- `SCARF/README.md` - Add SAES simulator overview and quick start

## Test Strategy

**Test philosophy:**
- Unit tests validate individual components with synthetic data
- Integration tests use real RE10K scenes to validate end-to-end pipeline
- Performance tests measure overhead and computation savings

**Test data required:**
- Synthetic Gaussians: Generate simple 4×4 tile scenarios with known similarity patterns
- RE10K subset: Use 5-10 scenes from RE10K test set for integration testing
- Ground truth: Compare SAES output against baseline (no early stopping)

**Test modifications:**
None (no existing SAES tests to modify)

**New test files:**

1. `tests/saes/test_similarity_evaluator.py` (Est: 180 LOC)
   - Test case: Identical Gaussians → similarity score ≈ 1.0
   - Test case: Highly different Gaussians → similarity score ≈ 0.0
   - Test case: Position dispersion dominates when large position diff
   - Test case: Covariance dispersion computation with various scales
   - Test case: Color and opacity dispersion with SH coefficients
   - Test case: Weighted aggregation produces expected similarity score

2. `tests/saes/test_decision_controller.py` (Est: 160 LOC)
   - Test case: High similarity (>0.85) → early_stop path
   - Test case: Medium similarity (0.6-0.85) → sparse_continue path
   - Test case: Low similarity (<0.6) → full_continue path
   - Test case: get_remaining_indices returns correct sparse indices
   - Test case: get_remaining_indices returns correct full indices
   - Test case: Boundary cases at threshold values

3. `tests/saes/test_gaussian_merger.py` (Est: 140 LOC)
   - Test case: Enlarge 4 probe Gaussians to cover 4×4 tile
   - Test case: Covariance scale increases proportionally to tile area
   - Test case: Position means preserved during enlargement
   - Test case: Color and opacity weighted averaging
   - Test case: Edge tiles with partial coverage

4. `tests/saes/test_tile_processor.py` (Est: 220 LOC)
   - Test case: Process single tile through 3-phase workflow
   - Test case: Early-stop path skips remaining pixels
   - Test case: Sparse-continue path processes sparse indices
   - Test case: Full-continue path processes all pixels
   - Test case: Timing profiling captures phase durations
   - Test case: Multiple tiles processed correctly
   - Test case: Profiling result aggregates statistics

5. `tests/saes/test_integration_re10k.py` (Est: 250 LOC)
   - Test case: Run SAES on single RE10K scene, verify output shape matches
   - Test case: Compare SAES output vs baseline, compute PSNR/SSIM
   - Test case: Verify computation saving ratio in expected range (30-50%)
   - Test case: Measure overhead and validate net speedup
   - Test case: Process multiple scenes and aggregate statistics
   - Test case: Validate early-stop tiles are in smooth regions
   - Test case: Validate full-continue tiles are in complex regions

**Expected test coverage:**
- Unit tests: 100% line coverage for core SAES modules
- Integration tests: Validate on ≥5 RE10K scenes
- Performance metrics: Computation saving ratio, overhead, net speedup, quality metrics (PSNR/SSIM)

## Implementation Steps

### Phase 1: Documentation (Steps 1-2)

**Step 1: Create SAES architecture documentation** (Estimated: 400 lines)
- `docs/saes-architecture.md` - Create comprehensive architecture document
  - Section 1: Overview of SAES design and goals
  - Section 2: System architecture diagram and component breakdown
  - Section 3: Tile-based processing workflow (3 phases: probe, evaluate, decide)
  - Section 4: Data structures and interfaces (TileConfig, GaussianSimilarityMetrics, etc.)
  - Section 5: 3D Gaussian similarity evaluation algorithm
  - Section 6: Decision controller path selection logic
  - Section 7: Gaussian merger enlargement strategy
  - Section 8: Performance profiling and metrics collection
  - Include Design2.md equations and specifications with code mappings
Dependencies: None

**Step 2: Create SAES usage documentation** (Estimated: 250 lines)
- `docs/saes-usage.md` - Create usage guide with examples
  - Section 1: Quick start guide
  - Section 2: Configuration options (TileConfig parameters)
  - Section 3: Running SAES simulation via command line
  - Section 4: Interpreting profiling results
  - Section 5: Integration with existing encoder pipeline
  - Section 6: Example outputs and visualizations
  - Section 7: Troubleshooting common issues
- `SCARF/README.md` - Update with SAES overview
  - Add "SAES Hardware Simulator" section
  - Add installation and quick start instructions
  - Add links to detailed documentation
Dependencies: Step 1 (architecture must be documented first)

### Phase 2: Test Cases (Steps 3-7)

**Step 3: Create similarity evaluator unit tests** (Estimated: 180 LOC)
- `tests/saes/__init__.py` - Test package initialization (5 LOC)
- `tests/saes/test_similarity_evaluator.py` - Comprehensive similarity evaluator tests
  - Test: `test_identical_gaussians_high_similarity` - Verify score ≈ 1.0 for identical inputs
  - Test: `test_different_gaussians_low_similarity` - Verify score ≈ 0.0 for vastly different inputs
  - Test: `test_position_dispersion_dominates` - Large position diff → high dispersion
  - Test: `test_covariance_dispersion` - Verify covariance difference computation
  - Test: `test_color_dispersion` - Verify SH coefficient difference computation
  - Test: `test_opacity_dispersion` - Verify opacity difference computation
  - Test: `test_weighted_aggregation` - Verify similarity score formula (0.4×pos + 0.3×cov + 0.15×color + 0.15×opacity)
  - Test: `test_scene_scale_normalization` - Verify position normalized by scene scale
  - Fixtures: Generate synthetic Gaussians with controlled properties
Dependencies: Step 2 (documentation defines expected behavior)

**Step 4: Create decision controller unit tests** (Estimated: 160 LOC)
- `tests/saes/test_decision_controller.py` - Decision logic tests
  - Test: `test_high_similarity_early_stop` - Score 0.85 → early_stop
  - Test: `test_high_similarity_exact_threshold` - Score 0.85 (boundary) → early_stop
  - Test: `test_medium_similarity_sparse` - Score 0.70 → sparse_continue
  - Test: `test_low_similarity_full` - Score 0.50 → full_continue
  - Test: `test_low_similarity_exact_threshold` - Score 0.60 (boundary) → sparse_continue
  - Test: `test_get_remaining_indices_sparse` - Sparse path returns [4,7,11,15] for 4×4 tile
  - Test: `test_get_remaining_indices_full` - Full path returns [4-15] for 4×4 tile
  - Test: `test_get_remaining_indices_early_stop` - Early stop returns empty list
  - Test: `test_custom_thresholds` - Verify custom threshold configuration
  - Fixtures: TileConfig with various threshold settings
Dependencies: Step 2

**Step 5: Create Gaussian merger unit tests** (Estimated: 140 LOC)
- `tests/saes/test_gaussian_merger.py` - Gaussian enlargement tests
  - Test: `test_enlarge_single_gaussian` - Single probe Gaussian enlarged to tile coverage
  - Test: `test_enlarge_multiple_gaussians` - 4 probe Gaussians merged and enlarged
  - Test: `test_covariance_scale_increase` - Verify covariance scale grows with tile area
  - Test: `test_position_mean_preserved` - Weighted average of probe positions
  - Test: `test_color_averaging` - SH coefficients averaged across probes
  - Test: `test_opacity_averaging` - Opacity averaged across probes
  - Test: `test_edge_tile_partial_coverage` - Tile at edge with partial pixels
  - Test: `test_coverage_calculation` - Verify tile coverage computation
  - Fixtures: Create probe Gaussians with known properties
Dependencies: Step 2

**Step 6: Create tile processor unit tests** (Estimated: 220 LOC)
- `tests/saes/test_tile_processor.py` - Tile processing workflow tests
  - Test: `test_single_tile_early_stop_path` - Process tile, hit early stop, verify 4 pixels processed
  - Test: `test_single_tile_sparse_path` - Process tile, hit sparse continue, verify 8 pixels processed
  - Test: `test_single_tile_full_path` - Process tile, hit full continue, verify 16 pixels processed
  - Test: `test_phase1_probe_processing` - Verify probe phase processes first 4 pixels
  - Test: `test_phase2_similarity_evaluation` - Verify similarity evaluator called with probe Gaussians
  - Test: `test_phase3_decision_execution` - Verify decision controller determines path
  - Test: `test_early_stop_gaussian_merger_called` - Verify merger called for early stop path
  - Test: `test_multiple_tiles_processing` - Process 4 tiles (16×16 feature map)
  - Test: `test_timing_profiling` - Verify timing breakdown captured per phase
  - Test: `test_profiling_result_aggregation` - Verify statistics aggregated across tiles
  - Mock: depth_predictor_fn returns synthetic depth/Gaussians
  - Mock: gaussian_adapter_fn returns synthetic final Gaussians
Dependencies: Steps 3, 4, 5 (requires all component tests)

**Step 7: Create RE10K integration tests** (Estimated: 250 LOC)
- `tests/saes/test_integration_re10k.py` - End-to-end integration tests
  - Test: `test_single_scene_saes_vs_baseline` - Process 1 RE10K scene, compare shapes/values
  - Test: `test_computation_saving_ratio` - Verify saving ratio in range [0.3, 0.5]
  - Test: `test_output_quality_psnr_ssim` - Verify PSNR ≥ baseline - 0.5 dB, SSIM ≥ baseline - 0.02
  - Test: `test_overhead_measurement` - Measure SAES overhead, verify < 10% of baseline time
  - Test: `test_net_speedup` - Verify net speedup ≥ 1.2×
  - Test: `test_multiple_scenes` - Process 5 scenes, aggregate statistics
  - Test: `test_early_stop_in_smooth_regions` - Verify early stop tiles have low variability
  - Test: `test_full_continue_in_complex_regions` - Verify full continue tiles have high variability
  - Test: `test_tile_distribution` - Verify distribution across 3 paths matches expectations
  - Fixtures: Load RE10K test scenes (reuse existing dataset infrastructure)
  - Utilities: Compute PSNR/SSIM, load ground truth images
Dependencies: Step 6 (requires tile processor working end-to-end)

### Phase 3: Implementation (Steps 8-14)

**Step 8: Implement type definitions** (Estimated: 100 LOC)
- `SCARF/saes/__init__.py` - Package initialization, export main classes (15 LOC)
- `SCARF/saes/types.py` - Data structures
  - Class: `TileConfig` with default values (tile_size=4, probe_size=4, thresholds)
  - Class: `GaussianSimilarityMetrics` with 5 fields (dispersions + similarity_score)
  - Class: `TileProcessingResult` with tile_id, path, counts, timing
  - Class: `SAESProfilingResult` with scene-level statistics
  - Helper: Type aliases for Tensor shapes
Dependencies: Step 7 (tests define required types)

**Step 9: Implement similarity evaluator** (Estimated: 180 LOC)
- `SCARF/saes/similarity_evaluator.py` - 3D Gaussian similarity computation
  - Class: `GaussianSimilarityEvaluator.__init__` - Initialize with optional scene_scale
  - Method: `evaluate` - Main entry point, compute all metrics
  - Method: `_compute_position_dispersion` - Max pairwise position L2 norm / scene_scale
  - Method: `_compute_covariance_dispersion` - Max pairwise covariance Frobenius norm / avg_cov_norm
  - Method: `_compute_color_dispersion` - Max pairwise SH DC component L2 norm
  - Method: `_compute_opacity_dispersion` - Max pairwise opacity absolute difference
  - Method: `_compute_similarity_score` - Weighted aggregation: exp(-dispersion / 0.1)
  - Helper: `_pairwise_distances` - Compute all pairwise distances efficiently
  - Helper: `_estimate_scene_scale` - Auto-estimate from Gaussian positions if not provided
Dependencies: Step 8 (requires types)

**Step 10: Implement decision controller** (Estimated: 150 LOC)
- `SCARF/saes/decision_controller.py` - Path selection logic
  - Class: `DecisionController.__init__` - Initialize with TileConfig
  - Method: `decide_path` - Main decision logic with threshold comparisons
  - Method: `get_remaining_indices` - Return pixel indices based on path
    - Early stop: return []
    - Sparse continue: return sparse_indices from config
    - Full continue: return range(probe_size, tile_size²)
  - Method: `_validate_thresholds` - Ensure high_threshold > low_threshold
  - Property: `summary` - Human-readable decision summary
Dependencies: Step 8 (requires types)

**Step 11: Implement Gaussian merger** (Estimated: 120 LOC)
- `SCARF/saes/gaussian_merger.py` - Gaussian enlargement
  - Class: `GaussianMerger.__init__` - Initialize with optional config
  - Method: `enlarge_and_merge` - Main entry point for early-stop path
  - Method: `_compute_tile_coverage_area` - Calculate tile spatial extent
  - Method: `_compute_average_position` - Weighted average of probe positions
  - Method: `_enlarge_covariance` - Scale up covariance to cover tile area
  - Method: `_average_color` - Average SH coefficients across probes
  - Method: `_average_opacity` - Average opacities across probes
  - Helper: `_create_enlarged_gaussian` - Construct enlarged Gaussian object
Dependencies: Step 8 (requires types)

**Step 12: Implement profiler** (Estimated: 200 LOC)
- `SCARF/saes/profiler.py` - Performance metrics collection
  - Class: `SAESProfiler.__init__` - Initialize metrics accumulators
  - Method: `start_tile` - Begin timing for a tile
  - Method: `record_phase` - Record phase timing (probe/evaluate/decide)
  - Method: `finish_tile` - Complete tile and store TileProcessingResult
  - Method: `get_scene_summary` - Aggregate all tiles into SAESProfilingResult
  - Method: `compute_savings` - Calculate pixels_saved and computation_saving_ratio
  - Method: `compute_overhead` - Sum similarity evaluation and decision time
  - Method: `compute_net_speedup` - Factor in overhead to compute real speedup
  - Property: `tile_results` - List of TileProcessingResult
  - Helper: `_categorize_tiles` - Count tiles by path type
Dependencies: Step 8 (requires types)

**Step 13: Implement tile processor** (Estimated: 280 LOC)
- `SCARF/saes/tile_processor.py` - Main tile processing engine
  - Class: `TileProcessor.__init__` - Initialize with config, create sub-components
  - Method: `process_scene` - Main entry point for full scene
    - Parse feature map dimensions, compute tile grid
    - Loop over all tiles, call `process_tile` for each
    - Aggregate results via profiler
    - Return final Gaussians and SAESProfilingResult
  - Method: `process_tile` - Process single tile through 3 phases
    - Phase 1: Call depth_predictor_fn for probe pixels
    - Phase 2: Call similarity_evaluator.evaluate on probe Gaussians
    - Phase 3: Call decision_controller.decide_path
    - Execute path: early_stop (merger) / sparse/full (depth_predictor_fn)
  - Method: `_extract_tile_features` - Slice feature tensor for tile region
  - Method: `_collect_probe_gaussians` - Gather probe Gaussians from depth_predictor output
  - Method: `_process_remaining_pixels` - Process sparse or full indices
  - Method: `_merge_tile_gaussians` - Combine probe + remaining Gaussians
  - Helper: `_compute_tile_grid` - Determine number of tiles in H and W dimensions
  - Helper: `_tile_coordinates` - Convert tile_id to (row_start, row_end, col_start, col_end)
Dependencies: Steps 9, 10, 11, 12 (requires all components)

**Step 14: Integrate SAES into encoder pipeline** (Estimated: 120 LOC)
- `src/model/encoder/encoder_trans.py:240-270` - Add SAES integration
  - Import SAES modules at top of file (TileProcessor, TileConfig)
  - Add `enable_saes` and `saes_config` parameters to `forward` signature
  - Initialize `TileProcessor` if `enable_saes=True`
  - Replace depth_predictor + gaussian_adapter calls with SAES process_scene
  - Store SAES profiling result in instance variable for model_wrapper access
  - Conditionally benchmark SAES stages
- `src/model/model_wrapper.py:59-64` - Add SAES configuration
  - Add `enable_saes_simulation: bool = False` to TestCfg
  - Add `saes_tile_size: int = 4` to TestCfg
  - Add `saes_high_threshold: float = 0.85` to TestCfg
  - Add `saes_low_threshold: float = 0.60` to TestCfg
- `src/model/model_wrapper.py:200-220` - Pass SAES config to encoder
  - Construct TileConfig from test_cfg settings
  - Pass enable_saes flag to encoder.forward
  - Collect and report SAES profiling results in test_step
- `src/model/model_wrapper.py:440-450` - Print SAES summary statistics
  - Add SAES statistics printing after existing analysis reports
  - Display computation savings, overhead, net speedup
- `scripts/run_all_timing_tests.sh:1-50` - Add SAES command option
  - Add `--enable-saes` flag to testing commands
  - Document SAES-specific options in script comments
Dependencies: Step 13 (requires tile processor complete)

## Total Complexity Estimate

**Lines of code breakdown:**
- Documentation: 650 lines (Steps 1-2)
- Tests: 950 LOC (Steps 3-7)
- Implementation: 1150 LOC (Steps 8-14)
- **Total: 2750 LOC (Very Large feature)**

**Recommended approach:**
Use milestone commits for incremental progress tracking:
- Milestone 1 (Steps 1-2): Documentation complete
- Milestone 2 (Steps 3-7): All tests written (0 tests pass initially)
- Milestone 3 (Steps 8-10): Core components implemented (similarity evaluator, decision controller tests pass)
- Milestone 4 (Steps 11-12): Merger and profiler implemented (merger tests pass)
- Milestone 5 (Step 13): Tile processor implemented (tile processor unit tests pass)
- Milestone 6 (Step 14): Integration complete (RE10K integration tests pass)
- Delivery commit: All tests pass, ready for PR

**Development timeline guidance:**
Given the complexity, expect 3-5 development sessions across multiple context windows. The modular architecture allows parallel development of components after core types are defined.

## Risk Mitigation

**Technical risks:**
1. **Risk**: Mocking depth_predictor_fn for unit tests is complex
   - **Mitigation**: Create lightweight mock in conftest.py that returns synthetic Gaussians
   - **Mitigation**: Integration tests use real depth_predictor to validate

2. **Risk**: RE10K dataset loading may have path/dependency issues
   - **Mitigation**: Reuse existing test infrastructure from model_wrapper
   - **Mitigation**: Start with single scene, expand to multiple incrementally

3. **Risk**: Performance overhead of Python simulation may be too high
   - **Mitigation**: Use PyTorch operations for similarity computation (GPU accelerated)
   - **Mitigation**: Profile hot paths and optimize bottlenecks

4. **Risk**: SAES output quality degradation may be unexpected
   - **Mitigation**: Integration tests validate PSNR/SSIM against thresholds
   - **Mitigation**: Visualize early-stop tiles to verify they align with smooth regions

**Validation strategy:**
- Unit tests ensure each component works in isolation
- Integration tests validate end-to-end pipeline correctness
- Performance tests confirm computation savings meet expectations (30-50%)
- Quality tests ensure PSNR/SSIM degradation is acceptable (< 0.5 dB / 0.02)

## Success Metrics

**Functional success:**
- [ ] All unit tests pass (100% coverage for SAES modules)
- [ ] Integration tests pass on ≥5 RE10K scenes
- [ ] SAES output shape matches baseline (same number of Gaussians after early-stop merge)
- [ ] 3-level path distribution aligns with Design2 expectations (~30% early, ~40% sparse, ~30% full)

**Performance success:**
- [ ] Computation saving ratio: 30-50% (pixels that skip depth search)
- [ ] SAES overhead: < 10% of baseline encoder time
- [ ] Net speedup: ≥ 1.2× (after accounting for overhead)

**Quality success:**
- [ ] PSNR degradation: < 0.5 dB compared to baseline
- [ ] SSIM degradation: < 0.02 compared to baseline
- [ ] Visual quality: Early-stop tiles predominantly in smooth regions (verified by smoothness analyzer)

**Documentation success:**
- [ ] Architecture document comprehensively explains design
- [ ] Usage guide enables users to run SAES simulation independently
- [ ] Code comments explain non-obvious implementation choices
