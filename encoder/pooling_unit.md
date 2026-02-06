# pooling_unit.py

Hardware simulator for spatial pooling operations.

## External Interface

### `PoolingUnit`
Sliding-window pooling unit supporting average and max pooling.

Hardware implementation: comparator tree (max) or accumulator + divider (avg).

**Constructor:**
```python
PoolingUnit()
```

**Methods:**
- `avg_pool2d(x, kernel_size, stride=None, padding=0) -> Tuple[Tensor, CycleStats]` — Average pooling with configurable kernel, stride, and padding.
- `max_pool2d(x, kernel_size, stride=None, padding=0) -> Tuple[Tensor, CycleStats]` — Max pooling with configurable kernel, stride, and padding.
- `adaptive_avg_pool2d(x, output_size) -> Tuple[Tensor, CycleStats]` — Adaptive average pooling to target spatial size.
- `get_total_cycles() -> int` — Accumulated cycles.
- `reset_cycles() -> None` — Reset cycle counter.

## Internal Helpers

### `_compute_cycles(x, output, mode) -> CycleStats`
Compute cycle count based on output size and pooling mode. Average pooling costs `kernel_size²` MACs per output pixel; max pooling costs `kernel_size²` comparisons.
