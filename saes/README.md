# SAES Core Module

## Purpose

Scene-Adaptive Early-Stopping (SAES) implementation for tile-based 3D Gaussian Splatting acceleration. Reduces redundant computation by exploiting 3D geometric continuity through feedback-driven processing.

## Architecture

### Core Components

| File | Purpose | LOC | Hardware Module |
|------|---------|-----|-----------------|
| `types.py` | Data structures and configuration | 209 | Register definitions |
| `similarity_evaluator.py` | 3D Gaussian similarity computation | 259 | 3D_Similarity_Evaluator |
| `decision_controller.py` | Path selection logic | 138 | Decision_Logic + MUX |
| `gaussian_merger.py` | Probe Gaussian enlargement | 234 | Gaussian_Merger |
| `profiler.py` | Performance metrics collection | 304 | Performance_Counters |
| `tile_processor.py` | Main orchestration engine | 308 | SAES_Controller FSM |

**Total**: 1452 LOC core implementation

### Component Dependencies

```
tile_processor.py (Main Engine)
├─► types.py (Data structures)
├─► similarity_evaluator.py
│   └─► types.py
├─► decision_controller.py
│   └─► types.py
├─► gaussian_merger.py
│   └─► types.py
└─► profiler.py
    └─► types.py
```

**No cyclic dependencies** - Clean modular design.

## Integration Points

### Callback Pattern for Model Independence

SAES uses callbacks to remain decoupled from specific model implementations:

```python
# Universal interface
def process_scene(
    features: Tensor,              # [B, V, C, H, W] - from any model backbone
    depth_predictor_fn: Callable,  # Model-specific depth prediction
    gaussian_adapter_fn: Callable, # Model-specific Gaussian generation
    context: Dict,                 # Model-specific camera parameters
) -> Tuple[List[Gaussian], SAESProfilingResult]:
```

**Supported Models**:
- ✅ **Transplat**: 2-view encoder
- ✅ **DepthSplat**: 3-view encoder
- ✅ **MVSPlat**: Multi-scale encoder

**Key Insight**: All model differences handled in adapter layer (callback implementation), not in SAES core.

## Hardware Mapping

### Resource Budget

- **LUTs**: ~37K (per tile processing unit)
- **DSPs**: 242 (mostly for depth search, shared with baseline)
- **SRAM**: ~5KB (feature buffer + Gaussian buffer)
- **ROM**: 256B (EXP LUT)

### Latency Overhead

- **Similarity Evaluation**: 16 cycles
- **Decision Logic**: <1 cycle
- **Merger (early-stop)**: 5 cycles
- **Total**: ~21 cycles (~5% of baseline)

**Net Speedup**: 1.3-1.4× (considering overhead and savings)

See `docs/hardware-dataflow-mapping.md` for complete hardware specifications.

## Testing

### Unit Tests: 57/57 ✅

- **Similarity Evaluator**: 12 tests (identical/different Gaussians, dispersion calculations, weighted aggregation)
- **Decision Controller**: 20 tests (path selection, threshold boundaries, index generation)
- **Gaussian Merger**: 9 tests (enlargement, averaging, edge tiles)
- **Tile Processor**: 16 tests (3-phase workflow, tile grid, profiling)

### Integration Tests: 11 tests ⏸️

- Requires transplat integration for RE10K validation
- Tests ready and will validate end-to-end pipeline

**Test Coverage**: 100% of implemented core components

## Quick Start

### Import SAES

```python
from saes import TileProcessor, TileConfig
from saes.types import Gaussian
```

### Configure and Process

```python
# Configure
config = TileConfig(
    tile_size=4,
    probe_size=4,
    high_similarity_threshold=0.85,
    low_similarity_threshold=0.60,
)

# Initialize
processor = TileProcessor(config)

# Process scene
gaussians, profiling = processor.process_scene(
    features,           # [B, V, C, H, W]
    depth_predictor_fn, # Your depth prediction function
    gaussian_adapter_fn,# Your Gaussian generation function
    context,            # Camera parameters
)

# Check results
print(profiling.summary_str())
```

See `examples/saes_standalone_demo.py` for complete working example.

## Development

### Running Tests

```bash
cd SCARF
source ~/anaconda3/bin/activate transplat
export PYTHONPATH=/home/mazirui/transplat/SCARF:$PYTHONPATH
python -m pytest tests/saes/ -v
```

### Code Structure Conventions

- **snake_case**: All functions and variables
- **Type annotations**: Required for all functions
- **Docstrings**: Google style with Args/Returns/Raises
- **Private methods**: Prefix with `_`
- **Constants**: UPPER_CASE

### Performance Guidelines

- **Tile size**: 4×4 recommended (balance overhead vs. savings)
- **Probe size**: 4 pixels minimum (reliable similarity estimation)
- **Thresholds**: Tune per model (start with 0.85/0.60)

## References

- [SAES Architecture](../docs/saes-architecture.md) - Detailed design
- [SAES Usage Guide](../docs/saes-usage.md) - API and examples
- [Hardware Dataflow Mapping](../docs/hardware-dataflow-mapping.md) - Hardware implementation guide
- [Implementation Plan](../docs/plan-saes-simulator.md) - Development roadmap
- [Code Review Report](../docs/code-review-report.md) - Quality assessment
