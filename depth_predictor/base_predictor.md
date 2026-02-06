# base_predictor.py

Abstract base class and pass-through implementation for depth predictor simulators.

## External Interface

### `BaseDepthPredictorSim` (ABC)
Abstract base class providing the common interface for all model-specific depth predictors.

**Constructor:**
```python
BaseDepthPredictorSim(config: DepthPredictorConfig, device: torch.device)
```

**Abstract methods:**
- `forward(features, images, extrinsics, intrinsics, near, far, **kwargs) -> DepthPredictorOutput` — Run depth prediction through hardware simulator.
- `load_from_model(depth_predictor: nn.Module) -> None` — Extract weights from a trained model into the simulator.

**Concrete methods:**
- `set_accurate_mode(accurate: bool) -> None` — Toggle between HW simulation and original model pass-through.
- `get_hardware_cycles() -> dict` — Return accumulated cycle breakdown.

### `PassThroughDepthPredictorSim`
Pass-through implementation that delegates to the original PyTorch model. Used when hardware simulation is disabled (`--no-depth`).

**Constructor:**
```python
PassThroughDepthPredictorSim(config: DepthPredictorConfig, device: torch.device)
```

## Internal Helpers

### `_build_depth_candidates(near, far, D, device) -> torch.Tensor`
Build uniformly-spaced inverse depth candidates between near and far planes.

### `_softmax_regression(logits, depth_candidates) -> torch.Tensor`
Softmax-based depth regression: weighted sum of depth candidates by softmax probabilities.
