# SAESProfiler Interface Documentation

## External Interface

### SAESProfiler Class

**Purpose**: Collect and aggregate performance metrics for SAES tile processing.

**Initialization**:
```python
def __init__(self, scene_name: str = "unknown")
```
- **Args**: `scene_name` - Scene identifier for reporting
- **State**: Initializes empty tile_results list and timing trackers
- **Debug Mode**: `enable_detailed_logging` flag (default False)

---

### Tile Lifecycle Methods

#### start_tile

```python
def start_tile(self, tile_id: Tuple[int, int]) -> None
```

**Purpose**: Begin timing for a tile.

**Parameters**:
- `tile_id`: (row, col) tile coordinates

**Side Effects**:
- Sets `_current_tile_id`
- Records `_current_tile_start_ns` (perf_counter_ns)
- Resets `_current_phase_timings`

**Usage**: Call at beginning of tile processing.

---

#### start_phase / end_phase

```python
def start_phase(self, phase_name: str) -> None
def end_phase(self, phase_name: str) -> None
```

**Purpose**: Time individual phases within tile.

**Parameters**:
- `phase_name`: 'probe', 'evaluate', 'decide', or 'merge'

**Usage**:
```python
profiler.start_phase("probe")
# ... execute probe phase ...
profiler.end_phase("probe")
```

**Alternative**: Use `record_phase(phase_name, duration_ns)` if timing done externally.

---

#### finish_tile

```python
def finish_tile(self, result: TileProcessingResult) -> None
```

**Purpose**: Complete tile and store result.

**Parameters**:
- `result`: TileProcessingResult with path, pixels processed, similarity score, timing

**Side Effects**:
- Appends result to `tile_results` list
- Resets current tile state

**Usage**: Call after tile processing complete.

---

### Aggregation Method: get_scene_summary

```python
def get_scene_summary(self, tile_size: int = 4) -> SAESProfilingResult
```

**Purpose**: Aggregate all tiles into scene-level statistics.

**Parameters**:
- `tile_size`: Pixels per tile edge (for computing savings)

**Returns**: `SAESProfilingResult` with:
- Path distribution (early/sparse/full counts)
- Computation savings (pixels saved, saving ratio)
- Overhead (SAES-specific timing)
- Net speedup (considering overhead)
- Full tile_results list

**Raises**:
- `ValueError`: If no tile results available

**Usage**:
```python
profiling = profiler.get_scene_summary(tile_size=4)
print(profiling.summary_str())
```

---

### Utility Methods

#### reset

```python
def reset(self) -> None
```

**Purpose**: Clear profiler state for new scene.

**Side Effects**: Clears `tile_results` and resets all tracking state.

---

#### get_statistics_dict

```python
def get_statistics_dict(self) -> Dict
```

**Purpose**: Export statistics as dictionary for JSON serialization.

**Returns**: Dictionary with:
- `scene_name`, `total_tiles`
- `path_distribution`: {early_stop, sparse_continue, full_continue}
- `computation`: {pixels_saved, saving_ratio}
- `timing`: {overhead_ns, overhead_ms}
- `performance`: {net_speedup}

**Usage**:
```python
import json
stats = profiler.get_statistics_dict()
with open("profiling.json", "w") as f:
    json.dump(stats, f, indent=2)
```

---

## Internal Helpers

### _categorize_tiles

```python
def _categorize_tiles(self) -> Tuple[int, int, int]
```

**Purpose**: Count tiles by path type.

**Returns**: (early_stop_count, sparse_continue_count, full_continue_count)

**Algorithm**: Count occurrences of each path in `tile_results`

---

### compute_savings

```python
def compute_savings(self, tile_size: int) -> Tuple[int, float]
```

**Purpose**: Compute total pixels saved and saving ratio.

**Formula**:
```python
total_pixels = num_tiles * tile_size²
total_processed = sum(r.pixels_processed for r in tile_results)
total_saved = total_pixels - total_processed
saving_ratio = total_saved / total_pixels
```

**Returns**: (total_pixels_saved, computation_saving_ratio)

---

### compute_overhead

```python
def compute_overhead(self) -> int
```

**Purpose**: Compute SAES-specific overhead.

**Includes**:
- Evaluate phase timing (similarity computation)
- Decide phase timing (path selection)
- Merge timing (for early-stop tiles only)

**Excludes**:
- Probe phase timing (would happen in baseline too)

**Returns**: Total overhead in nanoseconds

---

### compute_net_speedup

```python
def compute_net_speedup(
    self,
    tile_size: int,
    saving_ratio: float,
    overhead_ns: int,
) -> float
```

**Purpose**: Estimate net speedup considering overhead.

**Formula**:
```python
# Assume baseline: 1ms per pixel
baseline_ns = num_tiles * tile_size² * 1_000_000
overhead_ratio = overhead_ns / baseline_ns
speedup = 1.0 / ((1 - saving_ratio) + overhead_ratio)
```

**Returns**: Net speedup factor

**Example**:
- 42% saving, 8% overhead → speedup = 1 / (0.58 + 0.08) = 1.52×
- 30% saving, 10% overhead → speedup = 1 / (0.70 + 0.10) = 1.25×

---

## Hardware Mapping

### SAESProfiler → Performance Counters

```verilog
module performance_counters (
    input  wire        clk,
    input  wire        rst_n,
    
    // Events
    input  wire        tile_start,
    input  wire [1:0]  path_taken,  // 2'b10=early, 2'b01=sparse, 2'b00=full
    input  wire [4:0]  pixels_processed,
    
    // Cycle counters
    input  wire        probe_active,
    input  wire        eval_active,
    input  wire        decide_active,
    
    // Output counters
    output reg [15:0]  total_tiles,
    output reg [15:0]  early_stop_count,
    output reg [15:0]  sparse_continue_count,
    output reg [15:0]  full_continue_count,
    output reg [31:0]  total_pixels_saved,
    output reg [31:0]  probe_cycles,
    output reg [31:0]  eval_cycles,
    output reg [31:0]  decide_cycles
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        total_tiles <= 0;
        early_stop_count <= 0;
        sparse_continue_count <= 0;
        full_continue_count <= 0;
        total_pixels_saved <= 0;
        probe_cycles <= 0;
        eval_cycles <= 0;
        decide_cycles <= 0;
    end else begin
        // Count tiles
        if (tile_start) begin
            total_tiles <= total_tiles + 1;
            case (path_taken)
                2'b10: early_stop_count <= early_stop_count + 1;
                2'b01: sparse_continue_count <= sparse_continue_count + 1;
                2'b00: full_continue_count <= full_continue_count + 1;
            endcase
        end
        
        // Count saved pixels
        if (tile_start) begin
            total_pixels_saved <= total_pixels_saved + (16 - pixels_processed);
        end
        
        // Count cycles
        if (probe_active) probe_cycles <= probe_cycles + 1;
        if (eval_active) eval_cycles <= eval_cycles + 1;
        if (decide_active) decide_cycles <= decide_cycles + 1;
    end
end

endmodule
```

**Resources**: ~200 LUTs, 8 counters (16-32 bits each)  
**Latency**: Continuous (no impact on critical path)

---

## Metrics Interpretation

### Path Distribution

**Expected Distribution** (RE10K):
- Early-stop: ~30% (smooth regions: walls, floors, ceilings)
- Sparse-continue: ~40% (moderate texture: painted surfaces, fabrics)
- Full-continue: ~30% (complex: edges, fine details, occlusions)

**Interpretation**:
- Too many early-stops (>50%): May indicate over-aggressive thresholds or very smooth scene
- Too many full-continues (>50%): Scene is complex or thresholds too conservative
- Balanced distribution: Indicates appropriate threshold tuning

---

### Computation Saving Ratio

**Expected**: 30-50% for balanced config

**Breakdown**:
```
Early-stop tiles:     30% × 75% saved = 22.5%
Sparse-continue tiles: 40% × 50% saved = 20.0%
Full-continue tiles:   30% ×  0% saved =  0.0%
──────────────────────────────────────────────
Total saving:                            42.5%
```

**Interpretation**:
- <30%: Scene too complex or thresholds too conservative
- 30-50%: Expected range (good)
- >50%: Scene very smooth or thresholds aggressive (check quality)

---

### Overhead Analysis

**Expected Overhead**: 5-10% of baseline time

**Components**:
- Evaluate phase: ~80% of overhead (similarity computation)
- Decide phase: ~5% (threshold comparators)
- Merge phase: ~15% (weighted averaging)

**Interpretation**:
- <5%: Very efficient (hardware-optimized)
- 5-10%: Expected range (good)
- >10%: Bottleneck in similarity evaluator (profile further)

---

### Net Speedup

**Formula**: `speedup = baseline_time / (saes_time + overhead)`

**Expected**: 1.2-1.4× for balanced config

**Quality-Speed Tradeoff**:
| Speedup | Saving | Overhead | PSNR Loss |
|---------|--------|----------|-----------|
| 1.4× | 50% | 5% | 0.25 dB |
| 1.31× | 42% | 8% | 0.15 dB |
| 1.2× | 30% | 10% | 0.10 dB |

---

## Example Usage

```python
# Initialize profiler
profiler = SAESProfiler(scene_name="5aca87f95a9412c6")
profiler.enable_detailed_logging = True

# Process tiles
for tile in tiles:
    profiler.start_tile(tile.id)
    
    profiler.start_phase("probe")
    # ... probe processing ...
    profiler.end_phase("probe")
    
    profiler.start_phase("evaluate")
    # ... similarity evaluation ...
    profiler.end_phase("evaluate")
    
    profiler.start_phase("decide")
    # ... path decision ...
    profiler.end_phase("decide")
    
    result = TileProcessingResult(...)
    profiler.finish_tile(result)

# Get summary
profiling = profiler.get_scene_summary(tile_size=4)
print(profiling.summary_str())

# Export to JSON
import json
with open("profiling.json", "w") as f:
    json.dump(profiler.get_statistics_dict(), f, indent=2)
```

---

## Design Rationale

**Why per-phase timing?**
- Debug: Identify bottlenecks (is evaluate too slow?)
- Validation: Verify overhead matches hardware estimates
- Optimization: Guide where to focus optimization efforts

**Why exclude probe from overhead?**
- Fair comparison: Probe pixels processed in baseline too
- Overhead = SAES-specific cost only
- Helps calculate net speedup accurately

**Why track tile_results list?**
- Detailed analysis: Per-tile inspection for debugging
- Visualization: Generate heatmaps of path distribution
- Validation: Verify path decisions align with similarity scores

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03
