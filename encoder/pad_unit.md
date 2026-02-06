# pad_unit.py

Hardware simulator for tensor padding operations.

## External Interface

### `PadUnit`
Address-remapping padding unit supporting multiple border modes.

Hardware implementation: address generator that remaps border reads without copying data.
For constant padding, a mux selects the constant value at border positions.

**Constructor:**
```python
PadUnit()
```

**Methods:**
- `pad(x, pad, mode='constant', value=0.0) -> Tuple[Tensor, CycleStats]` — Pad tensor with specified mode.
  - `pad`: Tuple of padding sizes `(left, right, top, bottom, ...)`
  - `mode`: `'constant'` | `'replicate'` | `'reflect'` | `'circular'`
  - `value`: Fill value for constant mode
- `get_total_cycles() -> int` — Accumulated cycles.
- `reset_cycles() -> None` — Reset cycle counter.

## Internal Helpers

### `_compute_cycles(x, output) -> CycleStats`
Cycle cost is 1 cycle per padded element (address remap).
