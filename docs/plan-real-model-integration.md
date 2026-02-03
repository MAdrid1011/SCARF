# Implementation Plan: Real Transplat Model Integration for SCARF Benchmark

## Goal

Integrate real Transplat model inference with RE10K dataset into SCARF benchmark system, enabling actual rendering with quality metrics (PSNR, SSIM) and SCARF cycle counting, with rendered images saved to disk.

**Success criteria:**
- Run Transplat inference on RE10K test set using real model checkpoint
- Render actual images and save to `outputs/` directory
- Compute PSNR/SSIM between rendered and ground truth images
- Track SCARF component cycles during inference
- Compare baseline vs SCARF-accelerated performance

**Out of scope:**
- Training or fine-tuning models
- Supporting models other than Transplat (MVSplat/DepthSplat in future PRs)
- GPU/CUDA kernel optimization (focus on Python-level integration)

## Codebase Analysis

### Transplat Inference Flow

Based on `src/model/model_wrapper.py` and `src/main.py`:

```
1. Load checkpoint → ModelWrapper (encoder + decoder)
2. Load dataset → DataModule with RE10K dataset
3. For each batch:
   a. encoder(context) → Gaussians
   b. decoder(gaussians, target_views) → rendered images
   c. compute_psnr/ssim(rendered, ground_truth)
   d. save_image(rendered, path)
```

### Key Transplat Files

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point, model/data loading |
| `src/model/model_wrapper.py:185-323` | `test_step()` inference logic |
| `src/dataset/dataset_re10k.py` | RE10K dataset loader |
| `src/dataset/data_module.py` | PyTorch Lightning DataModule |
| `src/misc/image_io.py:57-68` | `save_image()` function |
| `src/evaluation/metrics.py` | PSNR/SSIM/LPIPS functions |
| `config/experiment/re10k.yaml` | RE10K experiment config |
| `assets/evaluation_index_re10k.json` | Evaluation view pairs |

### Files to Create

| File | Purpose | Estimated LOC |
|------|---------|---------------|
| `SCARF/benchmark/transplat_runner.py` | Real Transplat inference integration | 280 |
| `SCARF/benchmark/scarf_hooks.py` | SCARF hooks for encoder/decoder | 150 |
| `SCARF/tests/benchmark/test_transplat_runner.py` | Integration tests | 200 |
| `SCARF/tests/benchmark/test_scarf_hooks.py` | Hook tests | 120 |
| `SCARF/data/` | Symlinks to model/data | - |
| `SCARF/docs/real-model-benchmark.md` | Usage documentation | 150 |

### Files to Modify

| File | Changes | Estimated LOC |
|------|---------|---------------|
| `SCARF/benchmark/__init__.py` | Export TransplatRunner | +5 |
| `SCARF/benchmark/benchmark_runner.py` | Add real mode support | +80 |
| `SCARF/scripts/run_re10k_benchmark.py` | Add real inference path | +50 |

## Interface Design

### New Interfaces

#### TransplatRunner

```python
class TransplatRunner:
    """Real Transplat inference runner with SCARF integration."""
    
    def __init__(
        self,
        checkpoint_path: str,
        dataset_root: str,
        device: str = 'cuda',
        scarf_hooks: Optional[SCARFHooks] = None,
    )
    
    def load_model(self) -> ModelWrapper:
        """Load Transplat model from checkpoint."""
    
    def load_dataset(
        self,
        stage: str = 'test',
        num_scenes: Optional[int] = None,
    ) -> DataModule:
        """Load RE10K dataset."""
    
    def run_inference(
        self,
        batch: BatchedExample,
        enable_scarf: bool = False,
    ) -> InferenceResult:
        """Run single batch inference."""
    
    def run_benchmark(
        self,
        num_scenes: int = 100,
        save_images: bool = True,
        output_dir: str = 'outputs/real_benchmark/',
    ) -> BenchmarkResult:
        """Run full benchmark with metrics."""
```

#### SCARFHooks

```python
class SCARFHooks:
    """Hooks for integrating SCARF into Transplat encoder."""
    
    def __init__(
        self,
        fsdr_processor: Optional[FSDRProcessor] = None,
        dsu_processor: Optional[DSUProcessor] = None,
        ggu_processor: Optional[GGUProcessor] = None,
        cycle_counter: Optional[CycleCounter] = None,
    )
    
    def pre_depth_search(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
    ) -> Optional[FSDRResult]:
        """Hook before depth search - check FSDR cache."""
    
    def post_depth_search(
        self,
        result: DSUResult,
        features: torch.Tensor,
    ) -> None:
        """Hook after depth search - update FSDR cache."""
    
    def pre_gaussian_gen(
        self,
        depth: torch.Tensor,
        raw_gaussian: torch.Tensor,
    ) -> None:
        """Hook before Gaussian generation."""
```

#### InferenceResult

```python
@dataclass
class InferenceResult:
    rendered: torch.Tensor       # [V, C, H, W] rendered images
    ground_truth: torch.Tensor   # [V, C, H, W] GT images
    psnr: float                  # Scene PSNR
    ssim: float                  # Scene SSIM
    scene_name: str              # Scene identifier
    cycles: Optional[Dict]       # SCARF cycle breakdown
    fsdr_stats: Optional[Dict]   # FSDR cache statistics
```

### Modified Interfaces

#### BenchmarkRunner

Add real inference mode:

```python
class BenchmarkRunner:
    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        # NEW: Real mode configuration
        use_real_inference: bool = False,
        device: str = 'cuda',
        ...
    )
    
    # NEW: Real inference methods
    def _run_real_baseline_inference(self, scene_idx: int) -> tuple:
        """Run actual Transplat inference."""
    
    def _run_real_scarf_inference(self, scene_idx: int) -> tuple:
        """Run SCARF-accelerated Transplat inference."""
```

## Test Strategy

### Test Modifications

#### `tests/benchmark/test_benchmark_runner.py`

Add real inference tests (Estimated: +80 LOC):
- Test: `test_real_mode_initialization` - Verify real mode setup
- Test: `test_real_baseline_produces_images` - Verify image output
- Test: `test_real_scarf_produces_metrics` - Verify SCARF metrics

### New Test Files

#### `tests/benchmark/test_transplat_runner.py` (Estimated: 200 LOC)

- Test: `test_load_model_from_checkpoint` - Model loading
- Test: `test_load_re10k_dataset` - Dataset loading
- Test: `test_single_batch_inference` - Single batch run
- Test: `test_inference_produces_valid_images` - Image shape/range
- Test: `test_inference_with_scarf_hooks` - SCARF integration
- Test: `test_save_rendered_images` - Image saving
- Test: `test_metrics_computation` - PSNR/SSIM accuracy

#### `tests/benchmark/test_scarf_hooks.py` (Estimated: 120 LOC)

- Test: `test_fsdr_hook_cache_hit` - FSDR cache hit path
- Test: `test_fsdr_hook_cache_miss` - FSDR cache miss path
- Test: `test_dsu_hook_cycle_counting` - DSU cycle recording
- Test: `test_ggu_hook_cycle_counting` - GGU cycle recording
- Test: `test_hooks_disabled_passthrough` - Disabled hooks passthrough

### Test Data Required

- Model checkpoint: `checkpoints/re10k.ckpt` (symlink or copy)
- RE10K test subset: First 10 scenes for CI tests
- Mock checkpoint for unit tests (smaller weights)

## Implementation Steps

### Phase 1: Documentation (Estimated: 150 LOC)

**Step 1: Create usage documentation**
- `docs/real-model-benchmark.md` - Document:
  - Prerequisites (checkpoint, dataset)
  - Installation steps
  - CLI usage for real mode
  - Expected outputs and metrics
  - Troubleshooting common issues

Dependencies: None

### Phase 2: Test Infrastructure (Estimated: 320 LOC)

**Step 2: Create TransplatRunner tests**
- `tests/benchmark/test_transplat_runner.py`
  - Test fixtures for mock model/data
  - Tests for model loading, inference, image saving
  - Tests with skip decorators for CI without GPU

Dependencies: Step 1

**Step 3: Create SCARFHooks tests**
- `tests/benchmark/test_scarf_hooks.py`
  - Tests for each hook type
  - Tests for cycle counting integration
  - Tests for disabled/enabled hooks

Dependencies: Step 1

### Phase 3: Core Implementation (Estimated: 430 LOC)

**Step 4: Implement TransplatRunner**
- `benchmark/transplat_runner.py`
  - Model loading from checkpoint
  - Dataset loading with Hydra config
  - Single batch inference
  - Batch benchmark with metrics
  - Image saving logic

Dependencies: Steps 2-3

**Step 5: Implement SCARFHooks**
- `benchmark/scarf_hooks.py`
  - FSDR pre/post hooks
  - DSU cycle counting hooks
  - GGU cycle counting hooks
  - Hook manager for enable/disable

Dependencies: Steps 2-3

**Step 6: Integrate real mode into BenchmarkRunner**
- `benchmark/benchmark_runner.py:323-400`
  - Implement `_run_real_baseline_inference()`
  - Implement `_run_real_scarf_inference()`
  - Add `use_real_inference` flag handling
  - Update `__init__` for real mode

Dependencies: Steps 4-5

**Step 7: Update CLI for real mode**
- `scripts/run_re10k_benchmark.py`
  - Add `--real` flag for real inference mode
  - Add `--device` argument
  - Add `--save-images` flag
  - Update help text and examples

Dependencies: Step 6

### Phase 4: Data Setup (Estimated: 50 LOC)

**Step 8: Create data directory structure**
- `data/README.md` - Document data setup
- `data/.gitignore` - Ignore large files
- Setup script for symlinks:
  ```bash
  ln -s /path/to/checkpoints ./data/checkpoints
  ln -s /path/to/datasets/re10k ./data/re10k
  ```

Dependencies: Step 7

**Step 9: Update module exports**
- `benchmark/__init__.py`
  - Export TransplatRunner, SCARFHooks
  - Update `__all__` list

Dependencies: Steps 4-5

### Total Estimates

| Phase | LOC | Description |
|-------|-----|-------------|
| Phase 1: Documentation | 150 | Usage guide |
| Phase 2: Tests | 320 | Test infrastructure |
| Phase 3: Implementation | 430 | Core code |
| Phase 4: Data Setup | 50 | Data directory |
| **Total** | **950** | Large feature |

**Recommended approach:** Milestone commits for incremental progress

**Milestone strategy:**
- Milestone 1 (Steps 1-3): Documentation and tests complete (0/12 tests pass)
- Milestone 2 (Steps 4-5): TransplatRunner and hooks implemented (6/12 tests pass)
- Milestone 3 (Steps 6-7): BenchmarkRunner integration (10/12 tests pass)
- Delivery (Steps 8-9): Full integration, all tests pass (12/12 tests pass)

## Dependencies

### Python Packages

Required packages (already in transplat):
- `torch >= 1.9.0`
- `pytorch-lightning >= 1.6.0`
- `hydra-core >= 1.2.0`
- `einops >= 0.4.0`
- `jaxtyping >= 0.2.0`

### Data Dependencies

- **Checkpoint**: `checkpoints/re10k.ckpt` (~500MB)
- **Dataset**: `datasets/re10k/test/` (~10GB for test set)
- **Evaluation Index**: `assets/evaluation_index_re10k.json`

### Hardware Requirements

- GPU with >= 8GB VRAM for inference
- CPU fallback supported but slow
- ~20GB disk for outputs (rendered images)

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Checkpoint not available | Document download instructions, CI skips real tests |
| Dataset too large | Support subset testing (first N scenes) |
| Memory issues | Add batch size control, memory-efficient mode |
| SCARF integration breaks model | Non-invasive hooks, easy disable path |

## Related Documentation

- `docs/benchmark-guide.md` - Existing benchmark documentation
- `docs/fsdr-architecture.md` - FSDR design for hooks
- `docs/multi-model-integration.md` - Future MVSplat/DepthSplat support
