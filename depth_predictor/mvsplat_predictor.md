# mvsplat_predictor.py

MVSplat-specific depth predictor simulator.

## External Interface

### `MVSplatDepthPredictorSim(BaseDepthPredictorSim)`
Hardware simulator for MVSplat's depth prediction pipeline.

**Constructor:**
```python
MVSplatDepthPredictorSim(config: DepthPredictorConfig, device: torch.device)
```

**Key methods:**
- `forward(features, images, extrinsics, intrinsics, near, far, **kwargs) -> DepthPredictorOutput` — MVSplat depth prediction via HWDepthPredictor (batched stereo).
- `load_from_model(depth_predictor: nn.Module) -> None` — Extract MVSplat depth predictor weights.

### `create_mvsplat_predictor(device) -> MVSplatDepthPredictorSim`
Factory function with MVSplat-specific defaults.

## Internal Helpers

### `_prepare_features(features) -> torch.Tensor`
Reshape features from `[B, V, C, H, W]` to format expected by HWDepthPredictor stereo path.
