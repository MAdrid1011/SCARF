# Implementation Plan: SCARF Inference Benchmark on RE10K

## Goal

Implement end-to-end SCARF accelerator integration with transplat for real inference on RE10K dataset, providing both performance metrics (cycles, memory access) and quality metrics (PSNR, SSIM) to validate that hardware optimization does not significantly degrade rendering quality.

**Success criteria:**
- SCARF-accelerated inference produces valid 3D Gaussian outputs
- Cycle counts reported for FSDR, DSU, GGU, SAES components
- PSNR/SSIM metrics within acceptable threshold (PSNR drop < 0.5dB, SSIM drop < 0.01)
- Memory access reduction from FSDR quantified (target: 60%+ reduction)
- Benchmark results saved to JSON for analysis

**Out of scope:**
- Actual FPGA/ASIC implementation (simulation only)
- Optimizing transplat's backbone or decoder
- Supporting MVSplat/DepthSplat in this issue (future work)

## Codebase Analysis

**Files to modify:**

| File | Purpose |
|------|---------|
| `SCARF/integration/accelerator.py` | Add cycle counting and batch processing |
| `SCARF/fsdr/fsdr_processor.py` | Add cycle estimation per operation |
| `SCARF/dsu/dsu_processor.py` | Add cycle estimation |
| `SCARF/ggu/ggu_processor.py` | Add cycle estimation |
| `SCARF/saes/tile_processor.py` | Add cycle estimation |
| `transplat/src/model/encoder/encoder_trans.py` | Wire SCARF integration |
| `transplat/src/model/model_wrapper.py` | Add SCARF config flags |

**Files to create:**

| File | Purpose | Est. LOC |
|------|---------|----------|
| `SCARF/benchmark/__init__.py` | Benchmark module init | 20 |
| `SCARF/benchmark/cycle_counter.py` | Cycle counting infrastructure | 150 |
| `SCARF/benchmark/quality_validator.py` | Quality metric collection | 100 |
| `SCARF/benchmark/benchmark_runner.py` | End-to-end benchmark orchestration | 250 |
| `SCARF/benchmark/report_generator.py` | Generate benchmark reports | 150 |
| `SCARF/tests/benchmark/test_benchmark_runner.py` | Benchmark runner tests | 200 |
| `SCARF/tests/benchmark/conftest.py` | Benchmark test fixtures | 80 |
| `SCARF/scripts/run_re10k_benchmark.py` | CLI for running benchmark | 120 |
| `SCARF/docs/benchmark-guide.md` | Benchmark documentation | 200 |

## Interface Design

### New Interfaces

**CycleCounter**
```python
@dataclass
class CycleEstimate:
    component: str          # 'fsdr', 'dsu', 'ggu', 'saes'
    operation: str          # 'hash', 'lookup', 'search', etc.
    cycles: int             # Estimated hardware cycles
    memory_accesses: int    # Number of memory accesses

class CycleCounter:
    def __init__(self, clock_freq_mhz: int = 200)
    def record(self, component: str, operation: str, cycles: int, mem_accesses: int = 0)
    def get_summary(self) -> Dict[str, CycleSummary]
    def get_total_cycles(self) -> int
    def get_memory_accesses(self) -> int
    def reset()
```

**QualityValidator**
```python
class QualityValidator:
    def __init__(self, baseline_psnr: float = None, baseline_ssim: float = None)
    def record_sample(self, rendered: torch.Tensor, ground_truth: torch.Tensor)
    def get_metrics(self) -> Dict[str, float]  # {'psnr': X, 'ssim': Y}
    def check_quality_threshold(self, psnr_drop: float = 0.5, ssim_drop: float = 0.01) -> bool
```

**BenchmarkRunner**
```python
class BenchmarkRunner:
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        scarf_config: Dict,
        output_dir: str
    )
    
    def run_baseline(self, num_scenes: int = 100) -> BaselineResult
    def run_scarf(self, num_scenes: int = 100) -> SCARFResult
    def compare_results(self, baseline: BaselineResult, scarf: SCARFResult) -> ComparisonReport
    def save_report(self, report: ComparisonReport, filename: str)
```

### Modified Interfaces

**FSDRProcessor.process_pixel()** - Add cycle return
```python
def process_pixel(...) -> Tuple[FSDRResult, CycleEstimate]
```

**DSUProcessor.search_depth()** - Add cycle return
```python
def search_depth(...) -> Tuple[DSUResult, CycleEstimate]
```

**GGUProcessor.generate_gaussian()** - Add cycle return
```python
def generate_gaussian(...) -> Tuple[GaussianOutput, CycleEstimate]
```

**Accelerator.process_pixel()** - Add cycle aggregation
```python
def process_pixel(...) -> Tuple[GaussianOutput, Dict[str, CycleEstimate]]
```

## Test Strategy

**Test modifications:**

| File | Changes |
|------|---------|
| `tests/fsdr/test_fsdr_processor.py` | Add cycle estimation tests |
| `tests/dsu/test_dsu_processor.py` | Add cycle estimation tests |
| `tests/ggu/test_ggu_processor.py` | Add cycle estimation tests |

**New test files:**

**`tests/benchmark/conftest.py`** (Est: 80 LOC)
- Mock model wrapper for testing
- Sample RE10K scene data
- Baseline metrics fixtures

**`tests/benchmark/test_cycle_counter.py`** (Est: 120 LOC)
- Test cycle recording and aggregation
- Test component-level breakdown
- Test memory access counting

**`tests/benchmark/test_quality_validator.py`** (Est: 100 LOC)
- Test PSNR/SSIM computation
- Test threshold checking
- Test baseline comparison

**`tests/benchmark/test_benchmark_runner.py`** (Est: 200 LOC)
- Test end-to-end benchmark flow
- Test report generation
- Test comparison logic

## Implementation Steps

### Phase 1: Documentation (Est: 200 LOC)

**Step 1.1: Create benchmark documentation** (Est: 200 LOC)
- `SCARF/docs/benchmark-guide.md` - Comprehensive guide including:
  - Cycle model explanation (how hardware cycles are estimated)
  - Quality metrics methodology
  - How to run benchmarks
  - Interpreting results
  - Configuration options
- `SCARF/benchmark/README.md` - Module overview

Dependencies: None

### Phase 2: Test Infrastructure (Est: 500 LOC)

**Step 2.1: Create benchmark test fixtures** (Est: 80 LOC)
- `SCARF/tests/benchmark/__init__.py`
- `SCARF/tests/benchmark/conftest.py`:
  - `mock_model_output` fixture
  - `sample_re10k_scene` fixture
  - `baseline_metrics` fixture
  - Helper functions for test data generation

Dependencies: Step 1.1

**Step 2.2: Create CycleCounter tests** (Est: 120 LOC)
- `SCARF/tests/benchmark/test_cycle_counter.py`:
  - Test single operation recording
  - Test aggregation across components
  - Test memory access counting
  - Test reset functionality
  - Test summary generation

Dependencies: Step 2.1

**Step 2.3: Create QualityValidator tests** (Est: 100 LOC)
- `SCARF/tests/benchmark/test_quality_validator.py`:
  - Test PSNR computation accuracy
  - Test SSIM computation accuracy
  - Test threshold checking
  - Test baseline comparison mode

Dependencies: Step 2.1

**Step 2.4: Create BenchmarkRunner tests** (Est: 200 LOC)
- `SCARF/tests/benchmark/test_benchmark_runner.py`:
  - Test initialization
  - Test baseline run (mocked)
  - Test SCARF run (mocked)
  - Test comparison logic
  - Test report generation

Dependencies: Step 2.2, Step 2.3

### Phase 3: Core Implementation (Est: 770 LOC)

**Step 3.1: Implement CycleCounter** (Est: 150 LOC)
- `SCARF/benchmark/__init__.py`
- `SCARF/benchmark/cycle_counter.py`:
  - CycleEstimate dataclass
  - CycleSummary dataclass
  - CycleCounter class with:
    - `record()` - Record single operation
    - `get_summary()` - Aggregated stats by component
    - `get_total_cycles()` - Total cycle count
    - `get_memory_accesses()` - Memory access count
    - `reset()` - Clear all records
  - Hardware cycle constants (from architecture docs):
    - FSDR_HASH_CYCLES = 3
    - FSDR_LOOKUP_CYCLES = 5
    - DSU_PROJECT_CYCLES = 8
    - DSU_SAMPLE_CYCLES = 4
    - DSU_COST_CYCLES = 2
    - GGU_POSITION_CYCLES = 2
    - GGU_COVARIANCE_CYCLES = 5
    - GGU_SH_CYCLES = 3
    - SAES_COMPARE_CYCLES = 4

Dependencies: Step 2.2

**Step 3.2: Implement QualityValidator** (Est: 100 LOC)
- `SCARF/benchmark/quality_validator.py`:
  - QualityMetrics dataclass
  - QualityValidator class with:
    - `record_sample()` - Add rendered/GT pair
    - `compute_psnr()` - Calculate PSNR
    - `compute_ssim()` - Calculate SSIM
    - `get_metrics()` - Return all metrics
    - `check_quality_threshold()` - Validate acceptable degradation

Dependencies: Step 2.3

**Step 3.3: Implement ReportGenerator** (Est: 150 LOC)
- `SCARF/benchmark/report_generator.py`:
  - BenchmarkReport dataclass
  - ComparisonReport dataclass
  - ReportGenerator class with:
    - `generate_baseline_report()` - Baseline metrics
    - `generate_scarf_report()` - SCARF metrics with cycle breakdown
    - `generate_comparison()` - Side-by-side comparison
    - `to_json()` - JSON export
    - `to_markdown()` - Markdown summary

Dependencies: Step 3.1, Step 3.2

**Step 3.4: Implement BenchmarkRunner** (Est: 250 LOC)
- `SCARF/benchmark/benchmark_runner.py`:
  - BaselineResult dataclass
  - SCARFResult dataclass
  - BenchmarkRunner class with:
    - `__init__()` - Load model and dataset
    - `run_baseline()` - Run without SCARF
    - `run_scarf()` - Run with SCARF acceleration
    - `compare_results()` - Generate comparison
    - `save_report()` - Export results
  - Integration with transplat:
    - Model loading via Hydra config
    - Dataset loading via DatasetRE10k
    - Metric collection via existing infrastructure

Dependencies: Step 3.3, Step 2.4

**Step 3.5: Create benchmark CLI script** (Est: 120 LOC)
- `SCARF/scripts/run_re10k_benchmark.py`:
  - Argument parsing (model path, dataset path, output dir, num_scenes)
  - Initialize BenchmarkRunner
  - Run baseline and SCARF benchmarks
  - Generate comparison report
  - Print summary to console
  - Save detailed JSON report

Dependencies: Step 3.4

### Phase 4: SCARF Component Integration (Est: 400 LOC)

**Step 4.1: Add cycle estimation to FSDR** (Est: 80 LOC)
- `SCARF/fsdr/fsdr_processor.py`:
  - Add `enable_cycle_counting` parameter
  - Add `CycleEstimate` return to `process_pixel()`
  - Count cycles for: hash, lookup, correction, light_verify
  - Count memory accesses for cache and depth search

Dependencies: Step 3.1

**Step 4.2: Add cycle estimation to DSU** (Est: 60 LOC)
- `SCARF/dsu/dsu_processor.py`:
  - Add `enable_cycle_counting` parameter
  - Add `CycleEstimate` return to `search_depth()`
  - Count cycles for: project, sample, cost, softmax

Dependencies: Step 3.1

**Step 4.3: Add cycle estimation to GGU** (Est: 60 LOC)
- `SCARF/ggu/ggu_processor.py`:
  - Add `enable_cycle_counting` parameter
  - Add `CycleEstimate` return to `generate_gaussian()`
  - Count cycles for: position, covariance, sh_rotation

Dependencies: Step 3.1

**Step 4.4: Add cycle estimation to SAES** (Est: 60 LOC)
- `SCARF/saes/tile_processor.py`:
  - Add `enable_cycle_counting` parameter
  - Count cycles for: probe, compare, decide, merge

Dependencies: Step 3.1

**Step 4.5: Update Accelerator for cycle aggregation** (Est: 80 LOC)
- `SCARF/integration/accelerator.py`:
  - Initialize CycleCounter
  - Aggregate cycles from FSDR, DSU, GGU
  - Return cycle breakdown in `process_pixel()`
  - Add `get_total_cycles()` method

Dependencies: Step 4.1, Step 4.2, Step 4.3, Step 4.4

**Step 4.6: Wire SCARF into transplat encoder** (Est: 60 LOC)
- `transplat/src/model/encoder/encoder_trans.py`:
  - Add SCARF accelerator initialization in `__init__`
  - Call accelerator in `encoder_4_depth_predictor` region
  - Collect and log cycle counts
  - Pass quality validator for optional comparison

Dependencies: Step 4.5

## Hardware Cycle Model

Based on SCARF architecture documentation:

| Component | Operation | Cycles | Notes |
|-----------|-----------|--------|-------|
| **FSDR** | | | |
| | LSH hash | 3 | 16-bit signature generation |
| | Cache lookup | 5 | 256-entry parallel compare |
| | Direct reuse | 1 | Pass-through |
| | Interpolation | 4 | 2 muls, 1 add |
| | Light verify | 3-7 | Local search (3-7 depths) |
| **DSU** | | | |
| | Project | 8 | Matrix multiply |
| | Bilinear sample | 4 | 4 weighted adds |
| | Cost compute | 2 | Dot product |
| | Softmax | 6 | Exp LUT + division |
| **GGU** | | | |
| | Position | 2 | Unproject + scale |
| | Covariance | 5 | Quaternion + matrix |
| | SH rotation | 3 | Degree-1 rotation |
| **SAES** | | | |
| | Probe sample | 4 | Corner + center |
| | Similarity | 6 | 4 metrics |
| | Decision | 1 | Threshold compare |
| | Merge | 4 | Weighted average |

**Baseline comparison:**
- Full DSU search: 32 depths × (8+4+2) = 448 cycles
- With FSDR (68% cache hit): ~180 cycles average
- Expected speedup: ~2.5×

## Quality Degradation Analysis

**Expected sources of quality degradation:**
1. FSDR depth reuse - Minor (interpolation smooths transitions)
2. Fixed-point quantization - Minimal (16-bit depth precision)
3. SAES early stopping - Configurable (threshold tuning)

**Mitigation strategies:**
- Conservative FSDR thresholds initially
- SAES threshold tuning per-scene complexity
- Quality validation with automatic fallback

## Total Estimated Complexity

| Phase | LOC |
|-------|-----|
| Phase 1: Documentation | 200 |
| Phase 2: Test Infrastructure | 500 |
| Phase 3: Core Implementation | 770 |
| Phase 4: Component Integration | 400 |
| **Total** | **1,870** |

**Recommended approach:** Milestone commits for incremental progress

**Milestone strategy:**
- Milestone 1 (Phase 1-2): Documentation + test infrastructure (~700 LOC)
- Milestone 2 (Phase 3.1-3.3): Core benchmark classes (~400 LOC)
- Milestone 3 (Phase 3.4-3.5): Runner + CLI (~370 LOC)
- Milestone 4 (Phase 4): Component integration (~400 LOC)
- Delivery: All tests passing, benchmark working end-to-end

## Dependencies

- transplat model checkpoint: `checkpoints/re10k.ckpt`
- RE10K dataset: `datasets/re10k/`
- Evaluation index: `assets/evaluation_index_re10k.json`
- torch, numpy, scikit-image (for SSIM)
