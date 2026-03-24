# hw_depth_predictor.py

Core hardware depth predictor implementing all three model pipelines using only SCARF hardware units.

## External Interface

### `HWDepthPredictor`
Central engine for hardware-simulated depth prediction. Contains zero `F.*` or `torch.sigmoid/softplus` calls — all computation is routed through hardware units.

**Constructor:**
```python
HWDepthPredictor(device: torch.device, mode: str = 'warping')
```

**Hardware units instantiated:**
- `ConvEngine` — Convolution (kernel sizes 1–14, transposed)
- `GEMMUnit` — Matrix multiplication
- `BilinearUnit` — Interpolation (bilinear/nearest/bicubic) + grid_sample
- `ActivationUnit` (RELU, GELU, SILU, SIGMOID) — Non-linear activations
- `NormalizationUnit` (LAYER, BATCH, INSTANCE, GROUP) — Normalization
  - Separate DINOv2 unit with `eps=1e-6` (`_norm_ln_dinov2`)
- `PoolingUnit` — Average/max pooling
- `PadUnit` — Padding operations
- `SoftmaxUnit` — Depth regression softmax
- `DeformableAttentionUnit` — Multi-scale deformable attention (TranSplat)

**Key methods:**
- `load_from_model(model: nn.Module) -> None` — Extract all weights from trained model.
- `forward(features, images, extrinsics, intrinsics, near, far, ...) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]` — Run depth prediction, returns `(depths, densities, raw_gaussians, extra, total_cycles)`.
- `set_use_original(use_original: bool) -> None` — Toggle between HW simulation and original model.

### `HWUNetUnit`
Hardware U-Net implementation for cost volume refinement.

**Constructor:**
```python
HWUNetUnit(device: torch.device, base_channels: int = 64)
```

**Key methods:**
- `forward(x: Tensor) -> Tuple[Tensor, int]` — U-Net forward pass with cycle counting.
- `forward_with_weights(x, encoder_weights, decoder_weights, ...) -> Tuple[Tensor, int]` — Forward with explicit weight tensors.

## Internal Helpers

### Hardware routing helpers
Each wraps a hardware unit call and returns `(output, cycles)`:
- `_hw_group_norm(x, gn_layer)` — GroupNorm via NormalizationUnit
- `_hw_batch_norm(x, bn_layer)` — BatchNorm via NormalizationUnit
- `_hw_instance_norm(x, in_layer)` — InstanceNorm via NormalizationUnit
- `_hw_layer_norm(x, normalized_shape, weight, bias, eps=1e-5)` — LayerNorm via NormalizationUnit. Routes to DINOv2 unit (`eps=1e-6`) or default (`eps=1e-5`) automatically.
- `_hw_grid_sample(input, grid, mode, padding_mode, align_corners)` — grid_sample via BilinearUnit
- `_hw_pad(x, pad, mode, value)` — Padding via PadUnit
- `_hw_avg_pool2d(x, kernel_size, stride, padding)` — Avg pooling via PoolingUnit
- `_hw_conv_transpose(x, conv_layer)` — Transposed conv via ConvEngine
- `_hw_activation(x, activation_type)` — Activation via ActivationUnit
- `_hw_conv(x, weight, bias, stride, padding, groups)` — Conv2d via ConvEngine
- `_hw_bmm(a, b)` — Batch matmul via GEMMUnit
- `_hw_interpolate(x, size, scale_factor)` — Interpolation via BilinearUnit
- `_hw_linear(x, weight, bias)` — Linear layer via GEMMUnit
- `_hw_ffn(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, act_type)` — FFN via GEMMUnit + ActivationUnit

### Model-specific forward paths
- `_forward_transplat_hw(...)` — TranSplat: transformer matching → cost volume → U-Net → depth head
- `_forward_stereo_batched_hw(...)` — MVSplat: batched correlation cost volume → U-Net → depth head
- `_forward_depthsplat_hw(...)` — DepthSplat: DINOv2 + CNN + MV Transformer → multi-scale cost volume → DPT head → depth head

### DepthSplat sub-components
- `_hw_ds_dinov2(...)` — DINOv2 ViT forward with ConvEngine patch embed (k=14)
- `_hw_ds_vit_block(...)` — Single ViT block (LayerNorm→MHSA→MLP with eps=1e-6)
- `_hw_ds_mv_transformer(...)` — Multi-view Swin-like transformer
- `_hw_ds_transformer_layer(...)` — Single transformer layer with attention
- `_hw_ds_warp_features(...)` — Feature warping via grid_sample
- `_hw_ds_dpt_head(...)` — DPT learned upsampler
- `_hw_ds_cnn_backbone(...)` — CNN feature extraction

### TranSplat sub-components
- `_hw_uv_transformer(...)` — UV Transformer with deformable attention
- `_hw_uv_coarse_attention(...)` — Coarse-level attention
- `_hw_uv_self_attention(...)` — Self-attention block
- `_hw_uv_cross_attention(...)` — Cross-attention block
- `_hw_multi_scale_deformable_attn(...)` — Deformable attention sampling
- `_hw_corr_refine_net(...)` — Correlation refinement network
