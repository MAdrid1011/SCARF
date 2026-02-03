# Milestone 2 for Issue #1

**Branch:** issue-1  
**Created:** 2026-02-03  
**LOC Implemented:** ~3739 lines (documentation + tests)  
**Cumulative LOC:** 1919 (Milestone 1) + 1820 (Milestone 2) = 3739 lines  
**Test Status:** 0/60 tests passed (tests created but implementation not yet available)

## Work Completed

### Phase 2: Test Cases (Steps 3-7) ✅

**Step 3: Similarity Evaluator Unit Tests (285 lines)**
- `tests/saes/test_similarity_evaluator.py` - Comprehensive similarity evaluator tests
  - Test identical Gaussians → similarity ≈ 1.0
  - Test different Gaussians → similarity ≈ 0.0
  - Test position dispersion (dominance, scene scale normalization)
  - Test covariance dispersion with different scales
  - Test color dispersion with SH coefficients
  - Test opacity dispersion calculation
  - Test weighted aggregation formula (0.4×pos + 0.3×cov + 0.15×color + 0.15×opacity)
  - Test scene scale auto-estimation
  - Edge cases (single Gaussian, two identical)

**Step 4: Decision Controller Unit Tests (307 lines)**
- `tests/saes/test_decision_controller.py` - Decision logic tests
  - Test high similarity (>0.85) → early_stop
  - Test medium similarity (0.6-0.85) → sparse_continue
  - Test low similarity (<0.6) → full_continue
  - Test boundary cases at exact thresholds (0.85, 0.60)
  - Test remaining indices generation for each path
  - Test custom sparse indices configuration
  - Test threshold validation (high > low)
  - Test 8×8 tiles and different tile sizes
  - Edge cases (score = 0.0, 1.0, invalid path names)

**Step 5: Gaussian Merger Unit Tests (290 lines)**
- `tests/saes/test_gaussian_merger.py` - Gaussian enlargement tests
  - Test single Gaussian enlargement to tile coverage
  - Test merging four probe Gaussians
  - Test covariance scale increase proportional to tile area
  - Test covariance remains positive definite after enlargement
  - Test position weighted averaging
  - Test color (SH) averaging
  - Test opacity averaging
  - Test edge tiles with partial coverage (3×3, non-square)
  - Test tile coverage area computation

**Step 6: Tile Processor Unit Tests (393 lines)**
- `tests/saes/test_tile_processor.py` - Tile processing workflow tests
  - Test tile grid computation (64×64 → 16×16 tiles, non-divisible sizes)
  - Test single tile through each path (early-stop, sparse, full)
  - Test 3-phase workflow (probe → evaluate → decide)
  - Test multiple tiles processing (4 tiles, 256 tiles)
  - Test timing profiling (probe/evaluate/decide phases)
  - Test profiling result aggregation
  - Test computation saving ratio calculation
  - Test output shape correctness
  - Edge cases (empty features, tile_size > feature_map)

**Step 7: RE10K Integration Tests (380 lines)**
- `tests/saes/test_integration_re10k.py` - End-to-end integration tests
  - Test single scene SAES vs. baseline comparison
  - Test computation saving ratio in expected range [0.30, 0.50]
  - Test PSNR degradation < 0.5 dB
  - Test SSIM degradation < 0.02
  - Test SAES overhead < 10% of baseline
  - Test net speedup ≥ 1.2×
  - Test processing 5 scenes with aggregated statistics
  - Test early-stop in smooth regions
  - Test full-continue in complex regions
  - Test path distribution reasonableness

**Test Infrastructure:**
- `tests/__init__.py` (1 line) - Test package initialization
- `tests/saes/__init__.py` (1 line) - SAES test package
- `tests/saes/conftest.py` (163 lines) - Shared fixtures
  - MockGaussian dataclass (transplat-decoupled)
  - Fixtures: identical_gaussians, different_gaussians, moderate_similarity_gaussians
  - Mock callbacks: depth_predictor_fn, gaussian_adapter_fn
  - Synthetic feature maps and camera context
  - Helper utilities: assert_gaussians_similar, create_gaussian_grid

**Key Design Decisions:**
1. **TDD Approach**: All tests written before implementation, will initially fail (expected)
2. **Transplat Decoupling**: Core unit tests use MockGaussian (no transplat dependency)
3. **Integration Tests Conditional**: RE10K tests skip gracefully if transplat unavailable
4. **Comprehensive Coverage**: 60 test cases covering happy paths, edge cases, and boundary conditions
5. **Mock Strategy**: Mock depth_predictor_fn and gaussian_adapter_fn for unit testing, use real transplat functions for integration

**Test Count by Category:**
- Similarity Evaluator: ~12 tests
- Decision Controller: ~18 tests  
- Gaussian Merger: ~12 tests
- Tile Processor: ~13 tests
- RE10K Integration: ~5 tests
- **Total: ~60 test cases**

## Work Remaining

### Phase 3: Implementation (Steps 8-14) - Estimated 1150 LOC

**Step 8: Implement type definitions (100 LOC)**
- `saes/__init__.py`: Package initialization
- `saes/types.py`: TileConfig, GaussianSimilarityMetrics, TileProcessingResult, SAESProfilingResult, Gaussian

**Step 9: Implement similarity evaluator (180 LOC)**
- `saes/similarity_evaluator.py`: GaussianSimilarityEvaluator class
  - Methods: evaluate, _compute_position_dispersion, _compute_covariance_dispersion
  - Methods: _compute_color_dispersion, _compute_opacity_dispersion, _compute_similarity_score
  - Helpers: _pairwise_distances, _estimate_scene_scale

**Step 10: Implement decision controller (150 LOC)**
- `saes/decision_controller.py`: DecisionController class
  - Methods: decide_path, get_remaining_indices, _validate_thresholds

**Step 11: Implement Gaussian merger (120 LOC)**
- `saes/gaussian_merger.py`: GaussianMerger class
  - Methods: enlarge_and_merge, _compute_tile_coverage_area, _compute_average_position
  - Methods: _enlarge_covariance, _average_color, _average_opacity, _create_enlarged_gaussian

**Step 12: Implement profiler (200 LOC)**
- `saes/profiler.py`: SAESProfiler class
  - Methods: start_tile, record_phase, finish_tile, get_scene_summary
  - Methods: compute_savings, compute_overhead, compute_net_speedup, _categorize_tiles

**Step 13: Implement tile processor (280 LOC)**
- `saes/tile_processor.py`: TileProcessor class (main engine)
  - Methods: process_scene, process_tile, _extract_tile_features, _collect_probe_gaussians
  - Methods: _process_remaining_pixels, _merge_tile_gaussians
  - Helpers: _compute_tile_grid, _tile_coordinates

**Step 14: Integrate SAES into transplat (120 LOC)**
- `../transplat/src/model/encoder/encoder_trans.py`: Add SAES integration hooks
- `../transplat/src/model/model_wrapper.py`: Add SAES configuration flags
- `../transplat/scripts/run_all_timing_tests.sh`: Add SAES command options

**Total remaining:** ~1150 LOC

## Next File Changes (Estimated for Next Milestone)

**Immediate next steps (Steps 8-10):**
- `saes/__init__.py`: Package initialization (~15 LOC)
- `saes/types.py`: All data structures (~100 LOC)
- `saes/similarity_evaluator.py`: Complete implementation (~180 LOC)
- `saes/decision_controller.py`: Complete implementation (~150 LOC)

**Estimated for next milestone:** ~445 LOC (Steps 8-10)

After implementing these core components, expect:
- similarity_evaluator tests: ~12/12 passing
- decision_controller tests: ~18/18 passing
- Total: ~30/60 tests passing

## Test Status

**All tests currently failing (expected - implementation not yet created):**

**Similarity Evaluator (0/12 passing):**
- All tests skip with "Implementation not yet available"
- Tests ready to validate: dispersion calculations, weighted aggregation, scene scale estimation

**Decision Controller (0/18 passing):**
- All tests skip with "Implementation not yet available"
- Tests ready to validate: path selection logic, remaining indices generation, threshold validation

**Gaussian Merger (0/12 passing):**
- All tests skip with "Implementation not yet available"
- Tests ready to validate: enlargement, covariance scaling, averaging logic

**Tile Processor (0/13 passing):**
- All tests skip with "Implementation not yet available"
- Tests ready to validate: 3-phase workflow, tile grid computation, timing profiling

**RE10K Integration (0/5 passing):**
- All tests skip with "transplat not available or implementation not yet available"
- Tests ready to validate: end-to-end pipeline, quality metrics, performance metrics

**Total: 0/60 tests passed** (all skip due to missing implementation)

## Notes

- Phase 2 complete: All test cases created following TDD approach
- Tests are comprehensive and exceed planned LOC (1820 vs. 950) due to detailed coverage
- MockGaussian dataclass ensures transplat-decoupled unit testing
- Integration tests conditionally skip if transplat unavailable (graceful degradation)
- Ready to proceed with Phase 3 implementation (types → similarity → decision → merger → profiler → tile processor → integration)
- After implementing Steps 8-10, expect ~30/60 tests to start passing
