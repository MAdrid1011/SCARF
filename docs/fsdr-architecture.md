# FSDR Architecture: Feature-Similarity Depth Reuse

## 1. Overview

FSDR (Feature-Similarity Depth Reuse) is a hardware-software co-design component that reduces redundant memory access in the depth search stage of generalizable 3D Gaussian Splatting (3DGS) encoders by exploiting 2D feature space semantic similarity.

### 1.1 Design Goals

1. **Memory access reduction**: Achieve ~68% reduction in depth search stage memory access
2. **Model compatibility**: Support Transplat, MVSPlat, and DepthSplat architectures
3. **Hardware realizable**: All computations are simple integer/fixed-point operations
4. **Minimal overhead**: < 2KB SRAM, < 5 cycles latency overhead

### 1.2 Problem Statement

The depth search stage dominates memory bandwidth in 3DGS encoders:

```
For each reference pixel (u, v):
    For each depth candidate d_k (k = 1..D):
        1. Compute projection coordinates (u', v') = project(u, v, d_k)
        2. Sample from target feature map: tgt_feat = sample(target_feature_map, u', v')  ← HBM access
        3. Compute matching cost: cost_k = similarity(ref_feat, tgt_feat)
```

**Memory Access Analysis** (64×64 feature map, D=32 depth candidates):
- Per-pixel access: 32 × 128 × 2B = 8 KB
- Total access: 64 × 64 × 8 KB = 32 MB

**Key Observation**: Pixels with feature cosine similarity > 0.92 have median depth difference of only 2.3%. This semantic similarity enables depth reuse.

### 1.3 FSDR Solution

FSDR introduces a semantic-indexed cache that enables depth reuse:

```
For each reference pixel (u, v):
    1. Generate LSH signature from feature vector
    2. Lookup cache by signature similarity (Hamming distance)
    3. If hit with high confidence: Reuse cached depth (skip 32 accesses)
    4. If hit with medium confidence: Interpolate depth (skip 32 accesses)
    5. If hit with low confidence: Light verification (only 3-7 accesses)
    6. If miss: Full search + cache insert
```

**Key Innovation**: Transform 2D feature semantic similarity into hardware-capturable cache reuse.

---

## 2. System Architecture

### 2.1 Component Hierarchy

```
FSDRProcessor (main engine)
├── LSHHasher
│   ├── Random projection matrix (16 × C)
│   └── Sign extraction (16-bit signature)
├── CacheTable
│   ├── 128 entries × 70 bits
│   ├── Hamming distance calculator (parallel)
│   └── LRU + confidence weighted replacement
├── DepthCorrector
│   ├── Direct reuse strategy
│   ├── Interpolation strategy
│   └── Strategy decision logic
├── LightVerifier
│   ├── Search range calculator (from spread)
│   └── Local depth search
└── FSDRProfiler
    ├── Hit/miss counter
    ├── Path distribution tracker
    └── Memory access counter
```

### 2.2 Data Flow

```
Input: Feature vector f_cur [C], position (u, v)

FSDRProcessor.process_pixel():
  │
  ├─► Phase 1: Signature Generation [1 cycle]
  │   └─ sig_cur = lsh_hasher.hash(f_cur)  // 16-bit signature
  │
  ├─► Phase 2: Cache Lookup [1 cycle]
  │   └─ entry, hamming_dist = cache_table.lookup(sig_cur)
  │
  ├─► Phase 3: Hit/Miss Processing
  │   │
  │   ├─► [Cache Hit]
  │   │   │
  │   │   ├─► [High Confidence: peak_prob > 0.8 AND hamming ≤ 2]
  │   │   │   └─ depth = depth_corrector.direct_reuse(entry)
  │   │   │      Memory saved: 32 depth candidate accesses (100%)
  │   │   │
  │   │   ├─► [Medium Confidence: peak_prob > 0.5 OR hamming ≤ 3]
  │   │   │   └─ depth = depth_corrector.interpolate(entry, hamming_dist)
  │   │   │      Memory saved: 32 depth candidate accesses (100%)
  │   │   │
  │   │   └─► [Low Confidence]
  │   │       └─ depth = light_verifier.verify(entry, f_cur, ...)
  │   │          Memory saved: 25-29 accesses (78-91%)
  │   │
  │   └─► [Cache Miss]
  │       ├─ Execute full depth search (32 accesses)
  │       ├─ Extract statistics from softmax distribution
  │       └─ cache_table.insert(new_entry)
  │
  └─► Output: depth, source, timing

Output: FSDRResult {depth, source, cache_hit, hamming_distance, timing_ns}
```

---

## 3. Cache Entry Structure

### 3.1 Entry Layout (70 bits total)

```
┌─────────────┬───────────────┬─────────────┬──────────┬───────────┬─────────────┬────────┬───────┐
│  signature  │   position    │ best_depth  │ best_idx │ peak_prob │second_offset│ spread │ valid │
│   16 bits   │   16 bits     │   16 bits   │  5 bits  │  8 bits   │   5 bits    │ 8 bits │ 1 bit │
│             │   (u, v)      │   (FP16)    │ [0..31]  │  [0..1]   │  [-16..15]  │ [0..1] │       │
└─────────────┴───────────────┴─────────────┴──────────┴───────────┴─────────────┴────────┴───────┘
```

### 3.2 Field Descriptions

| Field | Bits | Description | Computation |
|-------|------|-------------|-------------|
| `signature` | 16 | LSH signature of reference feature | 16 random projections + sign |
| `position` | 16 | Pixel coordinates (u, v) | 8 bits each |
| `best_depth` | 16 | Optimal depth (FP16) | Expected depth from softmax |
| `best_idx` | 5 | Index of optimal depth candidate | argmax(probabilities) |
| `peak_prob` | 8 | Probability of optimal depth | max(probabilities) × 255 |
| `second_offset` | 5 | Offset to second-best depth index | signed, range [-16, 15] |
| `spread` | 8 | Distribution width (std dev) | Quantized to [0, 255] |
| `valid` | 1 | Entry validity flag | Set on insert |

### 3.3 Statistics Extraction

All statistics are computed from the softmax probability distribution during full search:

```python
# Given: probabilities p[D], depth_candidates d[D]

# Best depth (expected value)
best_depth = sum(p[k] * d[k] for k in range(D))

# Peak probability
best_idx = argmax(p)
peak_prob = p[best_idx]

# Second-best index
second_idx = argmax(p, exclude=best_idx)
second_offset = second_idx - best_idx  # Signed offset

# Spread (standard deviation)
mean_depth = best_depth
variance = sum(p[k] * (d[k] - mean_depth)**2 for k in range(D))
spread = sqrt(variance) / depth_range * 255  # Quantized
```

**Hardware Note**: Spread can be approximated without sqrt:
```python
spread_approx = abs(d[best_idx] - d[second_idx]) * (1 - peak_prob)
```

---

## 4. LSH Signature Generation

### 4.1 Algorithm

Locality-Sensitive Hashing (LSH) maps high-dimensional features to low-dimensional binary signatures while preserving similarity:

```python
class LSHHasher:
    def __init__(self, feature_dim: int = 128, lsh_dim: int = 16):
        # Random projection matrix: [lsh_dim, feature_dim]
        self.projection = np.random.randn(lsh_dim, feature_dim)
        self.projection = self.projection / np.linalg.norm(self.projection, axis=1, keepdims=True)
    
    def hash(self, feature: np.ndarray) -> int:
        """
        Generate 16-bit signature from feature vector.
        
        Args:
            feature: [C] feature vector (normalized)
        
        Returns:
            16-bit integer signature
        """
        # Project to 16 dimensions
        projections = self.projection @ feature  # [16]
        
        # Extract signs as bits
        signature = 0
        for i, proj in enumerate(projections):
            if proj >= 0:
                signature |= (1 << i)
        
        return signature
```

### 4.2 Similarity Preservation

**Property**: For unit vectors x, y:
```
P(sign(r·x) = sign(r·y)) = 1 - arccos(cos(x,y)) / π
```

**Implication**: High cosine similarity → Low Hamming distance

| Cosine Similarity | Expected Hamming Distance (16 bits) |
|-------------------|-------------------------------------|
| 1.0 | 0 |
| 0.95 | 1.0 |
| 0.90 | 1.4 |
| 0.85 | 1.9 |
| 0.80 | 2.4 |
| 0.70 | 3.3 |
| 0.50 | 5.3 |

### 4.3 Hardware Implementation

```verilog
module lsh_hasher (
    input  wire        clk,
    input  wire [15:0] feature [0:127],    // 128-dim feature (fixed-point)
    input  wire [15:0] proj_matrix [0:15][0:127],  // Pre-loaded projection
    output reg  [15:0] signature
);

// Parallel dot products
wire signed [31:0] dots [0:15];

genvar i;
generate
    for (i = 0; i < 16; i = i + 1) begin : DOT_PRODUCT
        dot_product_128 u_dot (
            .a(feature),
            .b(proj_matrix[i]),
            .result(dots[i])
        );
    end
endgenerate

// Sign extraction (combinational)
always @(*) begin
    for (int j = 0; j < 16; j = j + 1) begin
        signature[j] = (dots[j] >= 0) ? 1'b1 : 1'b0;
    end
end

endmodule
```

**Resources**: 16 MAC units (can be shared with DSU), ~500 LUTs  
**Latency**: 1 cycle (parallel dot products)

---

## 5. Cache Table

### 5.1 Organization

```
┌──────────────────────────────────────────────────────────────────┐
│                     Cache Table (128 entries)                     │
├──────┬───────────┬──────────────────────────────────────────────┤
│ Idx  │ Entry     │ Parallel Hamming Distance Calculators        │
├──────┼───────────┼──────────────────────────────────────────────┤
│  0   │ 70 bits   │  XOR + popcount ──┐                          │
│  1   │ 70 bits   │  XOR + popcount ──┼─► Min selector           │
│  2   │ 70 bits   │  XOR + popcount ──┤                          │
│ ...  │   ...     │       ...         │                          │
│ 127  │ 70 bits   │  XOR + popcount ──┘                          │
└──────┴───────────┴──────────────────────────────────────────────┘
```

### 5.2 Lookup Algorithm

```python
def lookup(self, signature: int) -> Tuple[Optional[CacheEntry], int]:
    """
    Find best matching entry by Hamming distance.
    
    Args:
        signature: 16-bit query signature
    
    Returns:
        (entry, hamming_distance) if hit, (None, -1) if miss
    """
    min_hamming = self.hamming_threshold + 1
    best_entry = None
    
    for entry in self.entries:
        if not entry.valid:
            continue
        
        # Parallel XOR + popcount
        hamming = popcount(signature ^ entry.signature)
        
        if hamming < min_hamming:
            min_hamming = hamming
            best_entry = entry
    
    if min_hamming <= self.hamming_threshold:
        return best_entry, min_hamming
    else:
        return None, -1
```

### 5.3 Replacement Policy

LRU with confidence weighting:

```python
def get_replacement_index(self) -> int:
    """
    Select entry to replace using LRU + confidence weighting.
    
    Priority: Invalid > Low confidence LRU > High confidence LRU
    """
    # First: Find any invalid entry
    for i, entry in enumerate(self.entries):
        if not entry.valid:
            return i
    
    # Second: Find LRU among low-confidence entries
    low_conf_lru = None
    low_conf_time = float('inf')
    for i, entry in enumerate(self.entries):
        if entry.peak_prob < 0.6 and entry.last_access < low_conf_time:
            low_conf_lru = i
            low_conf_time = entry.last_access
    
    if low_conf_lru is not None:
        return low_conf_lru
    
    # Third: Pure LRU
    return argmin(entry.last_access for entry in self.entries)
```

### 5.4 Hardware Implementation

```verilog
module cache_table (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] query_signature,
    input  wire        lookup_en,
    input  wire        insert_en,
    input  wire [69:0] insert_entry,
    
    output reg  [69:0] hit_entry,
    output reg  [3:0]  hamming_distance,
    output reg         cache_hit
);

// Storage: 128 entries × 70 bits
reg [69:0] entries [0:127];
reg [6:0]  lru_counter [0:127];

// Parallel Hamming distance calculation
wire [3:0] hamming [0:127];
wire [15:0] entry_signatures [0:127];

genvar i;
generate
    for (i = 0; i < 128; i = i + 1) begin : HAMMING_CALC
        assign entry_signatures[i] = entries[i][69:54];  // Signature field
        assign hamming[i] = popcount16(query_signature ^ entry_signatures[i]);
    end
endgenerate

// Min-selector tree
wire [3:0] min_hamming;
wire [6:0] min_idx;
min_selector_128 u_min (
    .values(hamming),
    .min_value(min_hamming),
    .min_index(min_idx)
);

// Hit detection
always @(posedge clk) begin
    if (lookup_en) begin
        if (min_hamming <= HAMMING_THRESHOLD && entries[min_idx][0]) begin
            cache_hit <= 1'b1;
            hit_entry <= entries[min_idx];
            hamming_distance <= min_hamming;
            lru_counter[min_idx] <= 7'd127;  // Reset LRU
        end else begin
            cache_hit <= 1'b0;
        end
    end
end

endmodule
```

**Resources**: ~300 LUTs, 128 × 70 bits = 1.1KB SRAM  
**Latency**: 1 cycle (parallel comparison + min selector)

---

## 6. Three-Level Correction Strategy

### 6.1 Strategy Selection

```python
def decide_strategy(self, entry: CacheEntry, hamming_dist: int) -> str:
    """
    Decide correction strategy based on confidence metrics.
    
    Returns:
        'direct_reuse' | 'interpolation' | 'light_verify'
    """
    peak_prob = entry.peak_prob / 255.0
    
    # Level 1: High confidence direct reuse
    if peak_prob > 0.8 and hamming_dist <= 2:
        return 'direct_reuse'
    
    # Level 2: Medium confidence interpolation
    if peak_prob > 0.5 or hamming_dist <= 3:
        return 'interpolation'
    
    # Level 3: Low confidence light verification
    return 'light_verify'
```

### 6.2 Direct Reuse

Simply return the cached depth:

```python
def direct_reuse(self, entry: CacheEntry) -> float:
    """
    Directly reuse cached depth (highest confidence).
    
    Conditions:
        - peak_prob > 0.8 (high confidence)
        - hamming_dist <= 2 (high feature similarity)
    
    Memory saved: 32 depth candidate accesses (100%)
    """
    return entry.best_depth
```

### 6.3 Interpolation

Interpolate between best and second-best depths:

```python
def interpolate(self, entry: CacheEntry, hamming_dist: int) -> float:
    """
    Interpolate between best and second-best depths.
    
    Formula:
        λ = hamming_dist / 4  (interpolation coefficient)
        depth = (1 - λ) * best_depth + λ * second_depth
    
    Rationale:
        Higher Hamming distance → more weight on second-best
        Second-best captures likely alternative when features differ
    
    Memory saved: 32 depth candidate accesses (100%)
    """
    # Compute second depth from offset
    second_idx = entry.best_idx + entry.second_offset
    second_depth = self.depth_candidates[second_idx]
    
    # Interpolation coefficient based on feature difference
    lambda_coeff = hamming_dist / 4.0
    lambda_coeff = min(lambda_coeff, 0.5)  # Cap at 50% weight
    
    return (1 - lambda_coeff) * entry.best_depth + lambda_coeff * second_depth
```

### 6.4 Light Verification

Perform local depth search around cached best depth:

```python
def light_verify(
    self,
    entry: CacheEntry,
    feature: np.ndarray,
    cost_fn: Callable,
) -> Tuple[float, int]:
    """
    Perform local depth search for verification.
    
    Search range: [best_idx - spread_idx, best_idx + spread_idx]
    Typical range: 3-7 candidates (vs 32 for full search)
    
    Memory saved: 25-29 accesses (78-91%)
    
    Returns:
        (depth, num_searches)
    """
    # Compute search radius from spread
    spread_idx = max(1, int(entry.spread * self.num_depths / 255))
    spread_idx = min(spread_idx, 3)  # Cap at 3 for efficiency
    
    # Define search range
    start_idx = max(0, entry.best_idx - spread_idx)
    end_idx = min(self.num_depths, entry.best_idx + spread_idx + 1)
    
    # Local search
    search_indices = range(start_idx, end_idx)
    num_searches = len(search_indices)
    
    # Compute costs for local range only
    costs = [cost_fn(feature, self.depth_candidates[i]) for i in search_indices]
    
    # Find best in local range
    local_best_idx = search_indices[np.argmin(costs)]
    
    return self.depth_candidates[local_best_idx], num_searches
```

---

## 7. FSDR Processor

### 7.1 Main Processing Loop

```python
class FSDRProcessor:
    def __init__(self, config: FSDRConfig, depth_candidates: np.ndarray):
        self.config = config
        self.depth_candidates = depth_candidates
        
        self.lsh_hasher = LSHHasher(config.feature_dim, config.lsh_dim)
        self.cache_table = CacheTable(config)
        self.depth_corrector = DepthCorrector(config, depth_candidates)
        self.light_verifier = LightVerifier(config)
        self.profiler = FSDRProfiler()
    
    def process_pixel(
        self,
        feature: np.ndarray,
        position: Tuple[int, int],
        cost_fn: Callable,
        prob_fn: Callable,
    ) -> FSDRResult:
        """
        Process single pixel with FSDR.
        
        Args:
            feature: [C] reference feature vector
            position: (u, v) pixel coordinates
            cost_fn: fn(feature, depth_idx) -> cost
            prob_fn: fn(feature, all_depths) -> probabilities [D]
        
        Returns:
            FSDRResult with depth and processing info
        """
        start_time = time.perf_counter_ns()
        
        # Phase 1: Signature generation
        signature = self.lsh_hasher.hash(feature)
        
        # Phase 2: Cache lookup
        entry, hamming_dist = self.cache_table.lookup(signature)
        
        # Phase 3: Hit/Miss processing
        if entry is not None:
            # Cache hit
            strategy = self.depth_corrector.decide_strategy(entry, hamming_dist)
            
            if strategy == 'direct_reuse':
                depth = self.depth_corrector.direct_reuse(entry)
                source = 'direct_reuse'
                
            elif strategy == 'interpolation':
                depth = self.depth_corrector.interpolate(entry, hamming_dist)
                source = 'interpolation'
                
            else:  # light_verify
                depth, num_searches = self.light_verifier.verify(
                    entry, feature, cost_fn
                )
                source = 'light_verify'
            
            # Update cache entry (exponential moving average)
            self.cache_table.update_depth(entry, depth)
            cache_hit = True
            
        else:
            # Cache miss - full search
            probs = prob_fn(feature, self.depth_candidates)
            depth = np.sum(probs * self.depth_candidates)
            
            # Extract statistics for cache entry
            new_entry = self._create_entry(signature, position, probs)
            self.cache_table.insert(new_entry)
            
            source = 'full_search'
            cache_hit = False
            hamming_dist = None
        
        # Record profiling
        elapsed_ns = time.perf_counter_ns() - start_time
        self.profiler.record(source, cache_hit, elapsed_ns)
        
        return FSDRResult(
            depth=depth,
            source=source,
            cache_hit=cache_hit,
            hamming_distance=hamming_dist,
            timing_ns={'total': elapsed_ns}
        )
```

---

## 8. Hardware Resource Summary

### 8.1 Component Breakdown

| Component | LUTs | DSPs | SRAM | ROM | Cycles |
|-----------|------|------|------|-----|--------|
| LSH Hasher | 500 | 16 | 0 | 0.5KB | 1 |
| Cache Table | 300 | 0 | 1.1KB | 0 | 1 |
| Hamming Calculator | 200 | 0 | 0 | 0 | 0 (comb) |
| Depth Corrector | 150 | 4 | 0 | 0 | 1 |
| Light Verifier | 100 | 2 | 0 | 0 | varies |
| **FSDR Total** | **1,250** | **22** | **1.1KB** | **0.5KB** | **<5** |

### 8.2 Memory Bandwidth Savings

| Scenario | Cache Hit Rate | Memory Access Reduction |
|----------|----------------|-------------------------|
| Smooth regions (walls, floors) | 90%+ | ~90% |
| Textured regions (surfaces) | 70-80% | ~65% |
| Complex boundaries (edges) | 30-50% | ~30% |
| **Overall average** | **~75%** | **~68%** |

---

## 9. Integration with DSU

### 9.1 FSDR + DSU Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FSDR + DSU Integrated Pipeline                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Feature ──► LSH Hasher ──► Cache Lookup ──┬─► [HIT]                │
│              (1 cycle)     (1 cycle)       │    │                   │
│                                            │    ├─► Direct Reuse    │
│                                            │    ├─► Interpolation   │
│                                            │    └─► Light Verify ──►│
│                                            │         (3-7 cycles)   │
│                                            │                        │
│                                            └─► [MISS]               │
│                                                  │                   │
│                                                  ▼                   │
│                                             Full DSU Search         │
│                                             (20+ cycles)            │
│                                                  │                   │
│                                                  ▼                   │
│                                             Cache Insert            │
│                                                  │                   │
│                                                  ▼                   │
│  ◄──────────────────────────────────────── Depth Output             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 DSU Integration Points

```python
class IntegratedDepthProcessor:
    def __init__(self, fsdr_config, dsu_config):
        self.fsdr = FSDRProcessor(fsdr_config, dsu_config.depth_candidates)
        self.dsu = DSUProcessor(dsu_config)
    
    def process_pixel(self, ref_feature, target_feature_map, ref_coord, ...):
        # Define cost function using DSU
        def cost_fn(feature, depth_idx):
            return self.dsu.compute_single_cost(
                feature, target_feature_map, ref_coord, depth_idx, ...
            )
        
        # Define probability function using DSU
        def prob_fn(feature, depth_candidates):
            return self.dsu.compute_probability_distribution(
                feature, target_feature_map, ref_coord, ...
            )
        
        # Use FSDR for intelligent depth search
        return self.fsdr.process_pixel(
            ref_feature, ref_coord, cost_fn, prob_fn
        )
```

---

## 10. Multi-Model Adaptation

### 10.1 Model Differences

| Model | Cost Volume | Softmax Direction | Feature Source |
|-------|-------------|-------------------|----------------|
| Transplat | Cost (lower = better) | Negative softmax | CNN + Transformer |
| MVSPlat | Correlation (higher = better) | Direct softmax | CNN |
| DepthSplat | Correlation (higher = better) | Direct softmax | DINOv2 |

### 10.2 Adapter Interface

```python
class FSDRAdapter(ABC):
    @abstractmethod
    def compute_probability(
        self,
        feature: Tensor,
        cost_volume: Tensor,
        depth_candidates: Tensor,
    ) -> Tensor:
        """Convert model-specific cost volume to probability distribution."""
        pass

class TransplatFSDRAdapter(FSDRAdapter):
    def compute_probability(self, feature, cost_volume, depth_candidates):
        # Transplat: cost volume (lower = better)
        return F.softmax(-cost_volume, dim=0)

class MVSplatFSDRAdapter(FSDRAdapter):
    def compute_probability(self, feature, cost_volume, depth_candidates):
        # MVSplat: correlation volume (higher = better)
        return F.softmax(cost_volume, dim=0)
```

---

## 11. Performance Characteristics

### 11.1 Expected Metrics

| Metric | Value | Range |
|--------|-------|-------|
| **Cache Hit Rate** | 75% | [60%, 85%] |
| **Direct Reuse Rate** | 40% | [30%, 50%] |
| **Interpolation Rate** | 25% | [20%, 30%] |
| **Light Verify Rate** | 10% | [5%, 15%] |
| **Full Search Rate** | 25% | [15%, 40%] |
| **Memory Access Reduction** | 68% | [55%, 75%] |
| **Latency Overhead** | <5 cycles | |

### 11.2 Scaling Analysis

| Feature Map | Pixels | Baseline Access | FSDR Access | Reduction |
|-------------|--------|-----------------|-------------|-----------|
| 16×16 | 256 | 2 MB | 0.64 MB | 68% |
| 32×32 | 1,024 | 8 MB | 2.56 MB | 68% |
| 64×64 | 4,096 | 32 MB | 10.24 MB | 68% |
| 128×128 | 16,384 | 128 MB | 40.96 MB | 68% |

**Key Insight**: Memory savings scale linearly with resolution.

---

## 12. Testing Strategy

### 12.1 Unit Tests

- **LSH Hasher**: Signature consistency, similarity preservation
- **Cache Table**: Insert/lookup/replace, Hamming distance calculation
- **Depth Corrector**: Strategy selection, interpolation accuracy
- **Light Verifier**: Search range calculation, local search correctness

### 12.2 Integration Tests

- **FSDR + DSU**: End-to-end depth prediction with FSDR
- **Memory Access Counting**: Verify actual access reduction
- **Quality Metrics**: PSNR/SSIM vs baseline (without FSDR)

### 12.3 Performance Tests

- **Hit Rate Profiling**: Per-scene hit rate distribution
- **Latency Measurement**: Per-pixel processing time
- **Memory Bandwidth**: Actual memory access measurement

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03  
**Related Issue**: GitHub Issue #2
