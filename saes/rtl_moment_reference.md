# rtl_moment_reference.py

Target-free exact-integer reference for the staged SAES scalar RTL moment lane.
It contains no routing or quality logic.

## External Interface

### `scalar_moment_merge`

```python
scalar_moment_merge(
    *,
    base_weight: int,
    base_mean: int,
    base_variance: int,
    updates: Iterable[tuple[int, int, int]],
) -> dict[str, int]
```

Merges a base scalar descriptor and ordered weighted pseudo descriptors through
first/second moments. Means are signed caller-defined fixed-point integers,
variances use squared units, and all weights share one nonnegative integer
unit. The returned mean divides toward zero and variance clamps at zero after
finite-precision subtraction.

`base_weight` must be positive; every variance and update weight must be
nonnegative. Invalid inputs raise `ValueError`.

## Internal Helpers

- `_integer` validates integer fields without accepting booleans.
- `_trunc_div` implements signed truncation toward zero to match the Chisel
  signed-divider contract.
