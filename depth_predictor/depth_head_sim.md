# depth_head_sim.py

Depth head simulators for converting cost volume logits to depth via softmax regression.

## External Interface

### `DepthHeadSimulator`
Full depth head with convolution layers and softmax regression.

**Constructor:**
```python
DepthHeadSimulator(config: DepthPredictorConfig, device: torch.device)
```

**Key methods:**
- `forward(cost_volume, depth_candidates) -> Tuple[Tensor, int]` — Apply conv layers + softmax regression. Returns `(depth [B, 1, H, W], cycles)`.

### `SimplifiedDepthHeadSim`
Lightweight depth head for use with external weight loading.

**Constructor:**
```python
SimplifiedDepthHeadSim(device: torch.device)
```

**Key methods:**
- `forward_with_weights(logits, depth_candidates, conv_weight, conv_bias, temperature) -> Tuple[Tensor, int]` — Forward pass with explicit weights and temperature.
- `forward_softmax_regression(logits, depth_candidates, temperature) -> Tuple[Tensor, int]` — Softmax-weighted depth regression only.

## Internal Helpers

### `_softmax_regression(logits, candidates, temperature) -> Tensor`
Apply temperature-scaled softmax and compute weighted depth sum.
