# depthsplat_predictor.py

DepthSplat-specific depth predictor simulator.

## External Interface

### `DepthSplatDepthPredictorSim(BaseDepthPredictorSim)`
Hardware simulator for DepthSplat's MultiViewUniMatch depth prediction pipeline.

DepthSplat integrates feature extraction with depth prediction: when features are marked as
`'depthsplat_integrated'`, the depth predictor handles CNN + DINOv2 + MV Transformer internally
via HWDepthPredictor's `_forward_depthsplat_hw` path.

**Constructor:**
```python
DepthSplatDepthPredictorSim(config: DepthPredictorConfig, device: torch.device)
```

**Key methods:**
- `forward(features, images, extrinsics, intrinsics, near, far, **kwargs) -> DepthPredictorOutput` — DepthSplat depth prediction. Accepts raw images when feature extraction is integrated.
- `load_from_model(depth_predictor: nn.Module) -> None` — Extract DepthSplat (MultiViewUniMatch) weights.

### `create_depthsplat_predictor(device) -> DepthSplatDepthPredictorSim`
Factory function with DepthSplat-specific defaults.

## Internal Helpers

### `_extract_multiscale_outputs(hw_output) -> dict`
Parse the HWDepthPredictor output into multi-scale depth maps and Gaussian parameters.
