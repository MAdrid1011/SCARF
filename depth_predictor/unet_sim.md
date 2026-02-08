# unet_sim.py

U-Net hardware simulators for cost volume refinement.

## External Interface

### `UNetSimulator`
Full U-Net simulator with configurable encoder-decoder architecture.

**Constructor:**
```python
UNetSimulator(config: UNetConfig, device: torch.device)
```

**Key methods:**
- `forward(x: Tensor) -> Tuple[Tensor, int]` — U-Net forward pass with skip connections. Returns `(output, cycles)`.

### `SimplifiedUNetSim`
Lightweight U-Net that operates with externally-provided weights (from `HWDepthPredictor.load_from_model`).

**Constructor:**
```python
SimplifiedUNetSim(device: torch.device)
```

**Key methods:**
- `forward_with_weights(x, encoder_weights, decoder_weights) -> Tuple[Tensor, int]` — Forward pass with explicit weight dictionaries.

### Supporting classes
- `UNetBlockSim` — Single conv-norm-activation block
- `UNetEncoderSim` — Encoder (downsampling) path
- `UNetDecoderSim` — Decoder (upsampling) path with skip connections

## Internal Helpers

### `_conv_gn_relu(x, weight, bias, num_groups) -> Tuple[Tensor, int]`
Single convolution + GroupNorm + ReLU block with cycle counting.
