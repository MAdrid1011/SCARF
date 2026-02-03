# Light Verifier

Local depth search for uncertain cache hits.

## External Interface

### LightVerifier

```python
class LightVerifier:
    def __init__(self, config: FSDRConfig, depth_candidates: torch.Tensor)
    
    def get_search_range(self, entry: CacheEntry) -> Tuple[int, int]
    def verify(
        self,
        entry: CacheEntry,
        cost_fn: Callable[[int], float],
        prob_fn: Callable[[int, int], torch.Tensor]
    ) -> Tuple[float, int]
```

**Constructor:**
- `config`: FSDR configuration
- `depth_candidates`: [D] depth values

**Methods:**

#### get_search_range(entry) -> (start_idx, end_idx)
Calculate reduced search range based on cached spread.

- **Input**: `entry` - CacheEntry with spread information
- **Output**: `(start_idx, end_idx)` indices into depth_candidates
- **Range Size**: 3-7 candidates (vs. 32 baseline)

**Range Calculation:**
```
center = entry.best_depth_idx
half_range = max(1, int(entry.spread * num_candidates / 2))
half_range = min(half_range, 3)  # Cap at 3
start = max(0, center - half_range)
end = min(num_candidates, center + half_range + 1)
```

#### verify(entry, cost_fn, prob_fn) -> (depth, num_searches)
Perform local depth search within reduced range.

- **Input**:
  - `entry` - CacheEntry providing search center
  - `cost_fn` - `fn(depth_idx) -> cost` for single depth
  - `prob_fn` - `fn(start, end) -> probs` for range
- **Output**: `(estimated_depth, num_searches_performed)`
- **Memory Saved**: 78-91% (3-7 searches vs. 32)

## Internal Helpers

### _local_argmin(costs, start_idx) -> int
Find depth index with minimum cost in local range.

### _weighted_depth(probs, start_idx) -> float
Compute expected depth from local probability distribution.

## Hardware Mapping

- Range calculation: ~50 LUTs
- Local search: Reuses DSU pipeline
- Memory reduction: 78-91%
- Total: ~50 LUTs + DSU reuse
