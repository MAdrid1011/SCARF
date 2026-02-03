# Depth Corrector

Three-level correction strategy for cache hits.

## External Interface

### DepthCorrector

```python
class DepthCorrector:
    def __init__(self, config: FSDRConfig, depth_candidates: torch.Tensor)
    
    def decide_strategy(self, entry: CacheEntry, hamming_dist: int) -> str
    def direct_reuse(self, entry: CacheEntry) -> float
    def interpolate(self, entry: CacheEntry, hamming_dist: int) -> float
```

**Constructor:**
- `config`: FSDR configuration with confidence thresholds
- `depth_candidates`: [D] depth values for interpolation

**Methods:**

#### decide_strategy(entry, hamming_dist) -> str
Select correction strategy based on entry confidence and Hamming distance.

- **Input**: 
  - `entry` - CacheEntry from lookup
  - `hamming_dist` - Hamming distance to query
- **Output**: One of `'direct_reuse'`, `'interpolation'`, `'light_verify'`

**Decision Logic:**
```
if peak_prob > 0.8 AND hamming_dist <= 2:
    return 'direct_reuse'
elif peak_prob > 0.5 OR hamming_dist <= 3:
    return 'interpolation'
else:
    return 'light_verify'
```

#### direct_reuse(entry) -> float
Return cached depth unchanged.

- **Input**: `entry` - CacheEntry
- **Output**: `entry.best_depth`
- **Memory Saved**: 100% (0 searches)

#### interpolate(entry, hamming_dist) -> float
Blend best and second-best depths.

- **Input**: `entry` - CacheEntry, `hamming_dist` - for weight calculation
- **Output**: Interpolated depth value
- **Formula**: `depth = w × best_depth + (1-w) × second_depth`
  - Weight `w = peak_prob × (1 - hamming_dist/threshold)`
- **Memory Saved**: 100% (0 searches)

## Internal Helpers

### get_second_depth(entry) -> (idx, depth)
Retrieve second-best depth from entry.

## Hardware Mapping

- Strategy decision: 4 comparators (~50 LUTs)
- Interpolation: 2 multiplies, 1 add (~100 LUTs, 4 DSPs)
- Total: ~150 LUTs, 4 DSPs, 1 cycle
