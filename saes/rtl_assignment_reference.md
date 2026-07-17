# rtl_assignment_reference.py

Target-free exact-integer reference for the staged SAES bilateral-assignment
normalizer. It receives precomputed nonnegative kernel scores and does not
inspect routing inputs or evaluation data.

## External Interface

### `normalize_assignment_scores`

```python
normalize_assignment_scores(
    scores: Sequence[int], *, anchor_count: int, fraction_bits: int = 16
) -> dict[str, int | tuple[int, ...]]
```

Normalizes one to eight active scores into Q0.`fraction_bits` weights. Integer
floor division is followed by assignment of the residual to the lowest-index
maximum score, making the returned active weights sum exactly to `1 <<
fraction_bits`.

Invalid anchor counts, absent active scores, negative scores, and an all-zero
active score sum raise `ValueError`.
