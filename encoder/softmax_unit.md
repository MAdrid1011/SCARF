# softmax_unit.py

Hardware simulator for softmax operation used in depth regression.

## External Interface

### `SoftmaxConfig` (dataclass)
Configuration for softmax hardware unit.

**Fields:**
- `exp_method: str = 'lut'` — Exp approximation: `'lut'`, `'piecewise'`, `'taylor'`
- `lut_size: int = 1024` — LUT entries (1024 gives near-exact precision, 4KB SRAM)
- `input_range: Tuple[float, float] = (-10.0, 0.0)` — Input range after max subtraction
- `num_segments: int = 64` — Piecewise linear segments (512B SRAM)
- `taylor_order: int = 6` — Taylor series order

### `SoftmaxUnit`
Hardware-realizable softmax using max-subtraction + LUT-based exp + reciprocal division.

**Constructor:**
```python
SoftmaxUnit(config: Optional[SoftmaxConfig] = None)
```

**Methods:**
- `forward(x, dim=-1) -> Tuple[Tensor, CycleStats]` — Apply softmax along dimension with cycle tracking.
- `forward_with_temperature(x, temperature, dim=-1) -> Tuple[Tensor, CycleStats]` — Temperature-scaled softmax.
- `get_total_cycles() -> int` — Accumulated cycles.
- `reset_cycles() -> None` — Reset cycle counter.

### `create_softmax_unit(**kwargs) -> SoftmaxUnit`
Factory function for creating SoftmaxUnit with custom config.

## Internal Helpers

### `_exp_lut(x) -> Tensor`
LUT-based exp approximation with linear interpolation between entries.

### `_exp_piecewise(x) -> Tensor`
Piecewise linear exp approximation using segment boundaries.

### `_exp_taylor(x) -> Tensor`
Taylor series exp approximation to configured order.

### `_compute_cycles(x, dim) -> CycleStats`
Cycle model: max-subtract (1 pass) + exp (LUT: 5 cycles/element) + sum + divide.
