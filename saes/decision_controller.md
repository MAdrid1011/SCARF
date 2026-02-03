# DecisionController Interface Documentation

## External Interface

### DecisionController Class

**Purpose**: Select processing path based on similarity score and generate remaining pixel indices.

**Initialization**:
```python
def __init__(self, config: TileConfig)
```
- **Args**: `config` - TileConfig with thresholds and sparse_indices
- **Validation**: Calls `_validate_thresholds()` to ensure high > low
- **Raises**: `ValueError` if thresholds invalid

---

### Primary Method: decide_path

```python
def decide_path(self, similarity_score: float) -> str
```

**Purpose**: Select processing path based on similarity threshold comparison.

**Parameters**:
- `similarity_score`: float in [0, 1] from similarity evaluator

**Returns**: One of:
- `"early_stop"` - Stop processing, use probe Gaussians only
- `"sparse_continue"` - Process sparse subset of remaining pixels
- `"full_continue"` - Process all remaining pixels

**Decision Logic**:
```python
if similarity_score >= high_threshold (0.85):
    return "early_stop"       # 75% pixels saved
elif similarity_score >= low_threshold (0.60):
    return "sparse_continue"  # 50% pixels saved
else:
    return "full_continue"    # 0% pixels saved
```

**Threshold Boundary Behavior**:
- score == 0.85 → `"early_stop"` (inclusive)
- score == 0.60 → `"sparse_continue"` (inclusive)
- score < 0.60 → `"full_continue"`

**Example**:
```python
controller = DecisionController(config)

path = controller.decide_path(0.92)  # → "early_stop"
path = controller.decide_path(0.75)  # → "sparse_continue"
path = controller.decide_path(0.45)  # → "full_continue"
```

---

### Secondary Method: get_remaining_indices

```python
def get_remaining_indices(
    self, 
    path: str, 
    tile_size: int, 
    probe_size: int
) -> List[int]
```

**Purpose**: Generate pixel indices to process based on path decision.

**Parameters**:
- `path`: Decision from `decide_path()` ('early_stop', 'sparse_continue', 'full_continue')
- `tile_size`: Pixels per tile edge (e.g., 4 → 16 total pixels)
- `probe_size`: Number of probe pixels already processed (e.g., 4)

**Returns**: List of remaining pixel indices

**Index Generation**:
- `early_stop`: `[]` - No more pixels
- `sparse_continue`: `config.sparse_indices` - Configured sparse pattern (e.g., [4,7,11,15])
- `full_continue`: `[probe_size, probe_size+1, ..., tile_size²-1]` - All remaining

**Raises**:
- `ValueError`: If path is not one of valid names

**Examples**:
```python
controller = DecisionController(TileConfig(sparse_indices=[4, 7, 11, 15]))

# Early-stop: skip remaining
indices = controller.get_remaining_indices("early_stop", tile_size=4, probe_size=4)
# → []

# Sparse: process 4 corners
indices = controller.get_remaining_indices("sparse_continue", tile_size=4, probe_size=4)
# → [4, 7, 11, 15]

# Full: process all remaining
indices = controller.get_remaining_indices("full_continue", tile_size=4, probe_size=4)
# → [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

---

### Property: summary

```python
@property
def summary(self) -> str
```

**Purpose**: Human-readable configuration summary.

**Returns**: Multi-line string with thresholds and sparse indices

**Example Output**:
```
DecisionController Configuration:
  High threshold (early-stop): ≥ 0.85
  Low threshold (full-continue): < 0.60
  Medium range (sparse-continue): [0.60, 0.85)
  Sparse indices: [4, 7, 11, 15]
```

---

## Internal Helpers

### _validate_thresholds

```python
def _validate_thresholds(self) -> None
```

**Purpose**: Validate threshold configuration.

**Checks**:
1. `high_threshold > low_threshold` (strict inequality)
2. Both thresholds in [0, 1]

**Raises**: `ValueError` with descriptive message if validation fails

**Called**: Automatically in `__init__()`

---

## Hardware Mapping

### Decision Logic → Hardware Comparators

```verilog
module decision_logic (
    input  wire [7:0]  similarity_score,  // Fixed-point [0, 255]
    input  wire [7:0]  high_threshold,    // Config register
    input  wire [7:0]  low_threshold,     // Config register
    output reg  [1:0]  path_select        // 2'b00=full, 2'b01=sparse, 2'b10=early
);

// Parallel comparators
wire high_compare = (similarity_score >= high_threshold);
wire low_compare = (similarity_score >= low_threshold);

// Combinational logic
always @(*) begin
    if (high_compare)
        path_select = 2'b10;  // early_stop
    else if (low_compare)
        path_select = 2'b01;  // sparse_continue
    else
        path_select = 2'b00;  // full_continue
end

endmodule
```

**Resources**: ~100 LUTs (2 comparators + mux)  
**Latency**: <1 cycle (combinational)

---

### Index Generator → Hardware MUX

```verilog
module index_generator (
    input  wire [1:0]  path_select,          // From decision_logic
    input  wire [3:0]  sparse_indices [0:15], // Config registers
    input  wire [7:0]  probe_size,           // Config register
    input  wire [7:0]  tile_size,            // Config register
    output reg  [3:0]  remaining_indices [0:15],
    output reg  [4:0]  num_remaining         // [0, 16]
);

always @(*) begin
    case (path_select)
        2'b10: begin  // early_stop
            num_remaining = 5'd0;
            // remaining_indices = don't care
        end
        
        2'b01: begin  // sparse_continue
            num_remaining = 5'd4;  // Or configurable
            remaining_indices[0] = sparse_indices[0];
            remaining_indices[1] = sparse_indices[1];
            remaining_indices[2] = sparse_indices[2];
            remaining_indices[3] = sparse_indices[3];
        end
        
        2'b00: begin  // full_continue
            num_remaining = tile_size * tile_size - probe_size;
            for (int i = 0; i < 16; i = i + 1) begin
                if (i < num_remaining)
                    remaining_indices[i] = probe_size + i;
            end
        end
    endcase
end

endmodule
```

**Resources**: ~100 LUTs (mux logic)  
**Latency**: <1 cycle (combinational)

---

## Multi-Model Considerations

### Threshold Tuning per Model

Different models may have different Gaussian smoothness characteristics:

**Transplat** (2-view, medium smoothness):
```python
config = TileConfig(
    high_similarity_threshold=0.85,
    low_similarity_threshold=0.60,
)
```

**DepthSplat** (3-view, potentially smoother):
```python
config = TileConfig(
    high_similarity_threshold=0.88,  # More conservative
    low_similarity_threshold=0.65,
)
```

**MVSPlat** (multi-scale, more variation):
```python
config = TileConfig(
    high_similarity_threshold=0.82,  # More aggressive
    low_similarity_threshold=0.55,
)
```

**Recommendation**: Profile each model on representative scenes to determine optimal thresholds.

---

### Sparse Pattern Customization

Different tile sizes or models may benefit from different sparse patterns:

**4×4 Tile - Corner Pattern** (default):
```
0  1  2  3
4  5  6  7
8  9  10 11
12 13 14 15

Probe: [0, 1, 2, 3]
Sparse: [4, 7, 11, 15]  # 4 corners of remaining
```

**4×4 Tile - Center Pattern**:
```
Sparse: [5, 6, 9, 10]  # Center 2×2
```

**8×8 Tile - Diagonal Pattern**:
```
Probe: [0, 1, 2, 3]
Sparse: [4, 11, 22, 33, 44, 55]  # Diagonal
```

---

## Design Rationale

**Why threshold-based decision?**
- Hardware-friendly: Single-cycle comparison
- Interpretable: Clear quality-speed tradeoff
- Configurable: Runtime adjustable via registers

**Why three paths?**
- Granular control: Not just on/off, but gradual adaptation
- Better quality-speed tradeoff: Sparse path provides middle ground
- Empirical validation: 3-level shows best results in profiling

**Why >= comparisons (inclusive)?**
- Consistent boundary behavior: score == threshold has predictable path
- Hardware simplicity: Single comparator with >= operator
- Avoids ambiguity: No "undefined" region between thresholds

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03
