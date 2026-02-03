# FSDR Profiler

Performance metrics collection for FSDR.

## External Interface

### FSDRProfiler

```python
class FSDRProfiler:
    def __init__(self)
    
    def start_pixel(self)
    def end_pixel(self, result: FSDRResult)
    def record_cache_hit(self, path: str)
    def record_cache_miss(self)
    def record_searches(self, count: int)
    
    def get_summary(self) -> FSDRProfilingResult
    def print_summary()
    def to_dict() -> dict
    def reset()
```

**Methods:**

#### start_pixel() / end_pixel(result)
Mark pixel processing boundaries for timing.

#### record_cache_hit(path)
Record cache hit with correction path used.

- **Input**: `path` - one of `'direct_reuse'`, `'interpolation'`, `'light_verify'`

#### record_cache_miss()
Record cache miss (full search performed).

#### record_searches(count)
Record number of depth searches performed.

- **Input**: `count` - searches for this pixel (0 for direct reuse, 3-7 for light verify, 32 for miss)

#### get_summary() -> FSDRProfilingResult
Aggregate all metrics into summary.

**Computed Metrics:**
- `hit_rate = cache_hits / total_pixels`
- `memory_reduction = 1 - (total_searches / baseline_searches)`
- `path_distribution` - count per path type

#### print_summary()
Print formatted summary to stdout.

#### to_dict() -> dict
Export metrics as JSON-serializable dictionary.

### TileProfiler

Per-tile profiling for batch processing.

```python
class TileProfiler:
    def start_tile(self, tile_id: int)
    def end_tile(self)
    def get_tile_stats() -> List[dict]
```

## Internal Helpers

### _compute_percentages(counts) -> Dict[str, float]
Convert path counts to percentages.

### _format_duration(ns) -> str
Format nanoseconds as human-readable string.

## Hardware Mapping

- Profiler is software-only for development
- Hardware uses simple counters for hit/miss rates
- ~100 LUTs for counter logic
