# FSDR Processor

Main processor orchestrating the FSDR workflow.

## External Interface

### FSDRProcessor

```python
class FSDRProcessor:
    def __init__(
        self,
        config: FSDRConfig,
        depth_candidates: torch.Tensor,
        enable_profiling: bool = True
    )
    
    def process_pixel(
        self,
        feature: torch.Tensor,
        pixel_coord: Tuple[int, int],
        cost_fn: Callable[[int], float],
        prob_fn: Callable[[int, int], torch.Tensor]
    ) -> FSDRResult
    
    def process_batch(
        self,
        features: torch.Tensor,
        pixel_coords: List[Tuple[int, int]],
        cost_fn: Callable,
        prob_fn: Callable
    ) -> List[FSDRResult]
    
    def get_profiling(self) -> dict
    def reset_profiling(self)
    def get_cache_stats(self) -> dict
    def clear_cache(self)
```

**Constructor:**
- `config`: FSDRConfig with all parameters
- `depth_candidates`: [D] depth values
- `enable_profiling`: Whether to collect timing/stats

**Methods:**

#### process_pixel(feature, pixel_coord, cost_fn, prob_fn) -> FSDRResult
Process single pixel through FSDR pipeline.

- **Input**:
  - `feature` - [D] feature vector
  - `pixel_coord` - (x, y) pixel location
  - `cost_fn` - `fn(depth_idx) -> cost` for DSU callback
  - `prob_fn` - `fn(start, end) -> probs[end-start]` for range search
- **Output**: FSDRResult with depth and metadata

**Pipeline:**
```
1. Hash feature → signature
2. Cache lookup → hit/miss
3. If hit: DepthCorrector decides strategy
   - direct_reuse: return cached depth
   - interpolation: blend depths
   - light_verify: local search
4. If miss: full DSU search
5. Update cache with new entry
6. Return result
```

#### process_batch(...) -> List[FSDRResult]
Process multiple pixels (for future parallelization).

#### get_profiling() -> dict
Return profiling statistics as dictionary.

#### get_cache_stats() -> dict
Return cache statistics.

## Internal Helpers

### _handle_cache_hit(entry, hamming_dist, cost_fn, prob_fn)
Route to appropriate correction strategy.

### _handle_cache_miss(feature, signature, cost_fn, prob_fn)
Perform full search and create cache entry.

### _create_cache_entry(signature, probs, depth_candidates)
Build CacheEntry from search results.

## Hardware Mapping

- Control FSM: ~200 LUTs
- Orchestrates LSHHasher, CacheTable, DepthCorrector, LightVerifier
- Total (including submodules): ~1,500 LUTs, 8 DSPs, 2.2KB SRAM
