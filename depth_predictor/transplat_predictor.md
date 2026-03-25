# transplat_predictor.py

TranSplat-specific depth predictor simulator.

## External Interface

### `TransplatDepthPredictorSim(BaseDepthPredictorSim)`
Hardware simulator for TranSplat's depth prediction pipeline.

**Constructor:**
```python
TransplatDepthPredictorSim(config: DepthPredictorConfig, device: torch.device)
```

**Key methods:**
- `forward(features, images, extrinsics, intrinsics, near, far, **kwargs) -> DepthPredictorOutput` — TranSplat depth prediction via HWDepthPredictor.
- `load_from_model(depth_predictor: nn.Module) -> None` — Extract TranSplat depth predictor weights.
- `set_accurate_mode(accurate: bool) -> None` — Toggle HW simulation vs original model.

### `create_transplat_predictor(device) -> TransplatDepthPredictorSim`
Factory function with TranSplat-specific defaults.

## Internal Helpers

### `_prepare_features(features) -> torch.Tensor`
Reshape features from `[B, V, C, H, W]` to per-view format expected by HWDepthPredictor.
