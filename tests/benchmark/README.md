# Benchmark Tests

Unit tests for the SCARF benchmark module.

## Overview

Tests for cycle counting, quality validation, benchmark orchestration, and report generation.

## Test Files

| File | Description |
|------|-------------|
| `test_cycle_counter.py` | CycleCounter and cycle constant tests |
| `test_quality_validator.py` | PSNR/SSIM computation and threshold tests |
| `test_benchmark_runner.py` | End-to-end benchmark orchestration tests |
| `conftest.py` | Shared test fixtures |

## Running Tests

```bash
# Run all benchmark tests
cd SCARF
pytest tests/benchmark/ -v

# Run specific test file
pytest tests/benchmark/test_cycle_counter.py -v

# Run with coverage
pytest tests/benchmark/ --cov=benchmark --cov-report=html
```

## Test Categories

### Cycle Counter Tests (`test_cycle_counter.py`)

- Basic recording and aggregation
- Component-level breakdown
- Memory access counting
- Hardware constant validation
- Baseline cycle comparison

### Quality Validator Tests (`test_quality_validator.py`)

- PSNR computation (identical, similar, different images)
- SSIM computation
- Threshold checking
- Baseline comparison
- Edge cases (zero images, single pixel difference)

### Benchmark Runner Tests (`test_benchmark_runner.py`)

- Initialization and configuration
- Baseline run metrics
- SCARF run with cycle counts
- Result comparison and speedup calculation
- Report generation (JSON structure, serialization)
- FSDR statistics

## Fixtures

Key fixtures defined in `conftest.py`:

| Fixture | Description |
|---------|-------------|
| `sample_cycle_estimates` | Mixed FSDR/DSU/GGU cycle estimates |
| `fsdr_only_estimates` | Cache hit path cycles |
| `full_search_estimates` | Full DSU search cycles |
| `identical_images` | Identical rendered/GT pair |
| `similar_images` | Images with small noise |
| `different_images` | Random image pairs |
| `baseline_metrics` | Reference quality metrics |
| `mock_model_config` | Model configuration dict |
| `mock_scarf_config` | SCARF configuration dict |

## Helper Functions

| Function | Description |
|----------|-------------|
| `compute_psnr_reference()` | Reference PSNR for validation |
| `compute_ssim_reference()` | Reference SSIM for validation |
| `generate_mock_scene_data()` | Generate mock scene data |

## Test Expectations

- Total test count: ~54 tests
- Expected pass rate: 100%
- Coverage target: >90% for benchmark module
