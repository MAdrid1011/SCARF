# SAES Usage Guide

## 1. Quick Start

### 1.1 Installation

```bash
# Clone SCARF repository
git clone git@github.com:MAdrid1011/SCARF.git
cd SCARF

# Install dependencies
pip install torch torchvision numpy
pip install pytest pytest-cov  # For testing
```

### 1.2 Basic Usage (Standalone)

```python
from saes import TileProcessor, TileConfig
from saes.types import Gaussian
import torch

# Configure SAES
config = TileConfig(
    tile_size=4,
    probe_size=4,
    high_similarity_threshold=0.85,
    low_similarity_threshold=0.60,
)

# Initialize processor
processor = TileProcessor(config)

# Define mock depth predictor and Gaussian adapter
def depth_predictor_fn(features, indices):
    # Your depth prediction logic here
    return torch.randn(len(indices), 1)  # Mock depths

def gaussian_adapter_fn(depths, context):
    # Your Gaussian generation logic here
    gaussians = []
    for d in depths:
        gaussians.append(Gaussian(
            mean=torch.randn(3),
            cov=torch.eye(3),
            opacity=0.8,
            harmonics=torch.randn(3, 16)
        ))
    return gaussians

# Process scene
features = torch.randn(1, 2, 128, 64, 64)  # [B, V, C, H, W]
context = {"intrinsics": ..., "extrinsics": ...}

gaussians, profiling_result = processor.process_scene(
    features, depth_predictor_fn, gaussian_adapter_fn, context
)

# Check results
print(f"Computation saving: {profiling_result.computation_saving_ratio:.2%}")
print(f"Early-stop tiles: {profiling_result.early_stop_tiles}")
print(f"Net speedup: {profiling_result.net_speedup:.2f}×")
```

### 1.3 Integration with Transplat

```bash
# In transplat repository
cd transplat

# Run with SAES enabled
python -m src.main +experiment=re10k \
    checkpointing.load=./checkpoints/re10k.ckpt \
    mode=test \
    test.enable_saes_simulation=true \
    test.saes_tile_size=4 \
    test.saes_high_threshold=0.85 \
    test.saes_low_threshold=0.60 \
    test.compute_scores=true
```

## 2. Configuration Options

### 2.1 TileConfig Parameters

```python
@dataclass
class TileConfig:
    """SAES configuration parameters"""
    
    tile_size: int = 4
    # Pixels per tile edge (4 means 4×4 = 16 pixels per tile)
    # Larger tiles → more potential savings but coarser granularity
    # Recommended: 4 (good balance)
    
    probe_size: int = 4
    # Number of pixels to process in probe phase
    # Must be ≤ tile_size²
    # Recommended: 4 (minimum for reliable similarity estimation)
    
    sparse_indices: List[int] = field(default_factory=lambda: [4, 7, 11, 15])
    # Pixel indices to process in sparse-continue path
    # Default selects 4 corners of remaining pixels
    # For 4×4 tile: indices 0-3 are probe, 4-15 are remaining
    
    high_similarity_threshold: float = 0.85
    # Similarity score above this triggers early-stop
    # Higher → more conservative (fewer early-stops, better quality)
    # Lower → more aggressive (more early-stops, more savings)
    # Recommended range: [0.80, 0.90]
    
    low_similarity_threshold: float = 0.60
    # Similarity score below this triggers full-continue
    # Between low and high → sparse-continue
    # Recommended range: [0.55, 0.65]
```

### 2.2 Choosing Thresholds

**Quality-prioritized (conservative):**
```python
config = TileConfig(
    high_similarity_threshold=0.90,  # Only very smooth regions
    low_similarity_threshold=0.70,   # Wide sparse-continue zone
)
# Expected: ~30% computation saving, <0.1 dB PSNR loss
```

**Balanced (default):**
```python
config = TileConfig(
    high_similarity_threshold=0.85,
    low_similarity_threshold=0.60,
)
# Expected: ~42% computation saving, ~0.15 dB PSNR loss
```

**Speed-prioritized (aggressive):**
```python
config = TileConfig(
    high_similarity_threshold=0.80,  # More early-stops
    low_similarity_threshold=0.50,   # Narrow sparse-continue zone
)
# Expected: ~55% computation saving, ~0.3 dB PSNR loss
```

### 2.3 Customizing Sparse Indices

For different tile sizes or sampling patterns:

```python
# 8×8 tile with diagonal sampling
config = TileConfig(
    tile_size=8,
    probe_size=4,
    sparse_indices=[4, 11, 22, 33, 44, 55],  # Diagonal pattern
)

# 4×4 tile with center-weighted sampling
config = TileConfig(
    tile_size=4,
    probe_size=4,
    sparse_indices=[5, 6, 9, 10],  # Center 2×2 region
)
```

## 3. Running Tests

### 3.1 Unit Tests

Test individual components:

```bash
cd SCARF

# Test similarity evaluator
pytest tests/saes/test_similarity_evaluator.py -v

# Test decision controller
pytest tests/saes/test_decision_controller.py -v

# Test Gaussian merger
pytest tests/saes/test_gaussian_merger.py -v

# Test tile processor
pytest tests/saes/test_tile_processor.py -v

# Run all unit tests
pytest tests/saes/ -v --cov=saes
```

### 3.2 Integration Tests

Test with real RE10K data (requires transplat):

```bash
cd SCARF

# Single scene integration test
pytest tests/saes/test_integration_re10k.py::test_single_scene_saes_vs_baseline -v

# Multiple scenes
pytest tests/saes/test_integration_re10k.py::test_multiple_scenes -v

# Quality validation
pytest tests/saes/test_integration_re10k.py::test_output_quality_psnr_ssim -v

# Full integration suite
pytest tests/saes/test_integration_re10k.py -v
```

### 3.3 Performance Benchmarking

```bash
# Run with profiling enabled
pytest tests/saes/test_integration_re10k.py::test_overhead_measurement -v

# Generate detailed profiling report
python -m saes.profiler --scene <scene_name> --output profiling_report.json
```

## 4. Interpreting Results

### 4.1 Profiling Output

```python
profiling_result = SAESProfilingResult(
    scene_name="5aca87f95a9412c6",
    total_tiles=256,
    early_stop_tiles=78,      # 30.5%
    sparse_continue_tiles=102, # 39.8%
    full_continue_tiles=76,   # 29.7%
    total_pixels_saved=1728,  # Out of 4096 total
    computation_saving_ratio=0.422,  # 42.2%
    overhead_ns=12500000,     # 12.5 ms
    net_speedup=1.31,         # 1.31× faster
)
```

**Interpretation:**
- **Early-stop tiles (30.5%)**: Smooth regions (walls, floors)
- **Sparse tiles (39.8%)**: Moderate texture (painted surfaces)
- **Full tiles (29.7%)**: Complex geometry (edges, details)
- **Computation saving (42.2%)**: 42% of pixels skipped depth search
- **Net speedup (1.31×)**: After accounting for SAES overhead

### 4.2 Quality Metrics

```python
# Compare SAES vs. baseline
baseline_psnr = 28.5 dB
saes_psnr = 28.35 dB
psnr_degradation = baseline_psnr - saes_psnr  # 0.15 dB

baseline_ssim = 0.912
saes_ssim = 0.905
ssim_degradation = baseline_ssim - saes_ssim  # 0.007

# Acceptable degradation thresholds
assert psnr_degradation < 0.5  # ✓ Pass
assert ssim_degradation < 0.02  # ✓ Pass
```

**Interpretation:**
- PSNR loss < 0.5 dB: Visually imperceptible
- SSIM loss < 0.02: Structural similarity preserved
- Trade-off: 42% computation savings for 0.5% quality loss

### 4.3 Tile Distribution Visualization

```python
from saes.profiler import visualize_tile_distribution

visualize_tile_distribution(
    profiling_result,
    output_path="tile_distribution.png"
)
```

This generates a heatmap showing which tiles took which path:
- Blue: Early-stop tiles (high similarity)
- Green: Sparse-continue tiles (medium similarity)
- Red: Full-continue tiles (low similarity)

## 5. Advanced Usage

### 5.1 Custom Similarity Evaluator

Customize similarity metrics for specific use cases:

```python
from saes.similarity_evaluator import GaussianSimilarityEvaluator

class CustomSimilarityEvaluator(GaussianSimilarityEvaluator):
    def evaluate(self, gaussians):
        # Override with custom logic
        metrics = super().evaluate(gaussians)
        
        # Example: Weight position dispersion more heavily
        dispersion = (
            0.6 * metrics.position_dispersion +  # Increased from 0.4
            0.2 * metrics.covariance_dispersion + # Decreased from 0.3
            0.1 * metrics.color_dispersion +
            0.1 * metrics.opacity_dispersion
        )
        
        metrics.similarity_score = torch.exp(-dispersion / 0.1)
        return metrics

# Use custom evaluator
processor = TileProcessor(config)
processor.similarity_evaluator = CustomSimilarityEvaluator()
```

### 5.2 Dynamic Threshold Adjustment

Adjust thresholds based on scene statistics:

```python
from saes import TileProcessor, TileConfig

def adaptive_thresholds(scene_features):
    """Compute thresholds based on scene complexity"""
    # Estimate scene complexity (e.g., feature variance)
    complexity = scene_features.std().item()
    
    if complexity < 0.1:  # Simple scene (smooth)
        return 0.80, 0.55  # Aggressive
    elif complexity < 0.3:  # Moderate scene
        return 0.85, 0.60  # Balanced
    else:  # Complex scene
        return 0.90, 0.65  # Conservative

# Use adaptive config
high_thresh, low_thresh = adaptive_thresholds(features)
config = TileConfig(
    high_similarity_threshold=high_thresh,
    low_similarity_threshold=low_thresh,
)
processor = TileProcessor(config)
```

### 5.3 Per-Tile Debugging

Debug individual tile processing:

```python
processor = TileProcessor(config)
processor.profiler.enable_detailed_logging = True

gaussians, profiling = processor.process_scene(...)

# Inspect specific tile
tile_id = (4, 7)  # Tile at row 4, col 7
tile_result = profiling.tile_results[tile_id]

print(f"Tile {tile_id}:")
print(f"  Path: {tile_result.path_taken}")
print(f"  Similarity: {tile_result.similarity_score:.3f}")
print(f"  Pixels processed: {tile_result.pixels_processed}/16")
print(f"  Probe time: {tile_result.timing_ns['probe'] / 1e6:.2f} ms")
print(f"  Evaluate time: {tile_result.timing_ns['evaluate'] / 1e6:.2f} ms")
```

## 6. Troubleshooting

### 6.1 Common Issues

**Issue: Computation saving ratio is lower than expected (<30%)**

Possible causes:
- Scene is highly complex (many edges/details)
- Thresholds too conservative
- Tile size too small

Solutions:
```python
# Try more aggressive thresholds
config.high_similarity_threshold = 0.80  # Down from 0.85
config.low_similarity_threshold = 0.55   # Down from 0.60

# Or increase tile size
config.tile_size = 8  # Up from 4
```

**Issue: Quality degradation exceeds 0.5 dB**

Possible causes:
- Thresholds too aggressive
- Gaussian merger not covering tile adequately
- Scene has fine details that need full processing

Solutions:
```python
# Use more conservative thresholds
config.high_similarity_threshold = 0.90  # Up from 0.85
config.low_similarity_threshold = 0.65   # Up from 0.60

# Or disable early-stop for critical scenes
config.high_similarity_threshold = 1.0  # Never early-stop
```

**Issue: SAES overhead exceeds 10%**

Possible causes:
- Similarity evaluation too slow
- Too many tiles (small tile size)
- Python interpreter overhead

Solutions:
```python
# Increase tile size to reduce number of tiles
config.tile_size = 8  # Reduces tiles by 4×

# Or use JIT compilation
import torch.jit
similarity_evaluator = torch.jit.script(similarity_evaluator)
```

### 6.2 Validation Checklist

Before deploying SAES, verify:

- [ ] Unit tests pass (100% coverage)
- [ ] Integration tests pass on ≥5 scenes
- [ ] Computation saving ratio in [0.30, 0.50]
- [ ] PSNR degradation < 0.5 dB
- [ ] SSIM degradation < 0.02
- [ ] SAES overhead < 10%
- [ ] Net speedup ≥ 1.2×
- [ ] Tile distribution reasonable (~30% early / ~40% sparse / ~30% full)

### 6.3 Debugging Tips

**Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

processor = TileProcessor(config)
processor.enable_debug_logging = True
```

**Profile hot paths:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

gaussians, result = processor.process_scene(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**Visualize similarity scores:**
```python
from saes.profiler import plot_similarity_heatmap

plot_similarity_heatmap(
    profiling_result,
    output_path="similarity_heatmap.png"
)
```

## 7. Examples

### 7.1 Example: Processing RE10K Scene

```python
import torch
from saes import TileProcessor, TileConfig

# Load RE10K scene (requires transplat)
from transplat_dataset import load_re10k_scene
scene = load_re10k_scene("5aca87f95a9412c6")

# Configure SAES
config = TileConfig(
    tile_size=4,
    probe_size=4,
    high_similarity_threshold=0.85,
    low_similarity_threshold=0.60,
)

# Initialize processor
processor = TileProcessor(config)

# Define callbacks (assuming transplat model is loaded)
def depth_fn(features, indices):
    return model.depth_predictor(features, indices=indices)

def gaussian_fn(depths, context):
    return model.gaussian_adapter(depths, context)

# Process scene
features = scene["features"]  # [1, 2, 128, 64, 64]
context = {"intrinsics": scene["intrinsics"], "extrinsics": scene["extrinsics"]}

gaussians, profiling = processor.process_scene(
    features, depth_fn, gaussian_fn, context
)

# Report results
print(f"Scene: {scene['name']}")
print(f"Tiles: {profiling.total_tiles}")
print(f"  Early-stop: {profiling.early_stop_tiles} ({profiling.early_stop_tiles/profiling.total_tiles:.1%})")
print(f"  Sparse: {profiling.sparse_continue_tiles} ({profiling.sparse_continue_tiles/profiling.total_tiles:.1%})")
print(f"  Full: {profiling.full_continue_tiles} ({profiling.full_continue_tiles/profiling.total_tiles:.1%})")
print(f"Computation saving: {profiling.computation_saving_ratio:.2%}")
print(f"Net speedup: {profiling.net_speedup:.2f}×")
```

### 7.2 Example: Batch Processing Multiple Scenes

```python
from pathlib import Path
import json

scenes = ["5aca87f95a9412c6", "1babb7c467c9e1b4", ...]
results = []

for scene_name in scenes:
    scene = load_re10k_scene(scene_name)
    gaussians, profiling = processor.process_scene(...)
    
    results.append({
        "scene": scene_name,
        "computation_saving": profiling.computation_saving_ratio,
        "net_speedup": profiling.net_speedup,
        "early_stop_ratio": profiling.early_stop_tiles / profiling.total_tiles,
    })

# Save results
with open("saes_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Compute aggregate statistics
avg_saving = sum(r["computation_saving"] for r in results) / len(results)
avg_speedup = sum(r["net_speedup"] for r in results) / len(results)

print(f"Average computation saving: {avg_saving:.2%}")
print(f"Average net speedup: {avg_speedup:.2f}×")
```

## 8. Performance Tips

### 8.1 Optimize for Throughput

```python
# Use larger batch sizes
features = torch.randn(8, 2, 128, 64, 64)  # Batch size 8

# Process in parallel
from concurrent.futures import ThreadPoolExecutor

def process_batch(batch_features):
    return processor.process_scene(batch_features, ...)

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_batch, batch) for batch in batches]
    results = [f.result() for f in futures]
```

### 8.2 Optimize for Latency

```python
# Use smaller tiles for faster feedback
config = TileConfig(tile_size=2)  # 2×2 tiles

# Reduce probe size
config.probe_size = 2  # Process only 2 pixels initially

# Use GPU acceleration
device = torch.device("cuda")
features = features.to(device)
```

### 8.3 Memory Optimization

```python
# Process tiles sequentially to reduce memory footprint
config = TileConfig(tile_size=4)
processor = TileProcessor(config)
processor.process_tiles_sequentially = True  # Disable tile parallelism

# Clear GPU cache periodically
torch.cuda.empty_cache()
```

## 9. Integration Checklist

### 9.1 Standalone Integration

To use SAES in your own project:

- [ ] Install SCARF package
- [ ] Implement depth_predictor_fn callback
- [ ] Implement gaussian_adapter_fn callback
- [ ] Configure TileConfig for your use case
- [ ] Run unit tests to verify correctness
- [ ] Benchmark on representative scenes

### 9.2 Transplat Integration

To use SAES with transplat:

- [ ] Clone both SCARF and transplat repositories
- [ ] Install dependencies for both projects
- [ ] Enable SAES in transplat config
- [ ] Run integration tests on RE10K
- [ ] Validate quality metrics (PSNR/SSIM)
- [ ] Profile performance (overhead, speedup)

## 10. FAQ

**Q: Can SAES work with models other than transplat?**
A: Yes, SAES is model-agnostic. It only requires depth_predictor_fn and gaussian_adapter_fn callbacks.

**Q: What if my tile size doesn't evenly divide the feature map?**
A: SAES handles edge tiles gracefully. Partial tiles at edges are processed as full-continue.

**Q: Can I use SAES for training?**
A: SAES is designed for inference. Using it during training would affect gradient flow and convergence.

**Q: How does SAES compare to pruning techniques?**
A: SAES is orthogonal to pruning. Pruning reduces model parameters, SAES reduces runtime computation dynamically.

**Q: Can SAES be combined with FSDR?**
A: Yes, FSDR and SAES are complementary. FSDR optimizes memory access in depth search, SAES optimizes computation via early stopping.

**Q: What's the minimum PyTorch version required?**
A: PyTorch 2.0+ is recommended for optimal performance. PyTorch 1.12+ is minimum requirement.

## 11. Additional Resources

- [SAES Architecture Documentation](saes-architecture.md)
- [Implementation Plan](plan-saes-simulator.md)
- [transplat Design2.md](../transplat/draft/Design2.md)
- [GitHub Issue #1](https://github.com/MAdrid1011/SCARF/issues/1)
- [Test Suite Documentation](../tests/saes/README.md)
