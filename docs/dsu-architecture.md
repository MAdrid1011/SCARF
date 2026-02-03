# DSU Architecture: Depth Search Unit

## 1. Overview

The Depth Search Unit (DSU) is the core computational module for depth prediction in generalizable 3DGS encoders. It performs cost volume computation and depth estimation through feature matching across views.

### 1.1 Design Goals

1. **Unified interface**: Support Transplat, MVSPlat, and DepthSplat depth prediction
2. **Hardware mappable**: All operations suitable for hardware acceleration
3. **FSDR compatible**: Provide callbacks for FSDR integration
4. **Efficient sampling**: Bilinear interpolation with boundary handling

### 1.2 Core Functionality

```
Input: Reference feature f_ref, target feature map, projection params
Output: Depth estimate d*, probability distribution p[D]

For each depth candidate d_k (k = 1..D):
    1. Project reference pixel to target view
    2. Sample target feature at projected location
    3. Compute matching cost/correlation
    
Aggregate costs via softmax to get depth distribution
```

---

## 2. System Architecture

### 2.1 Component Hierarchy

```
DSUProcessor (main engine)
├── DepthSampler
│   ├── Projection calculator
│   └── Bilinear interpolator
├── CostVolume
│   ├── Dot product calculator
│   └── Cost aggregator
└── SoftmaxAggregator
    ├── Softmax computer
    ├── Expected depth calculator
    └── Statistics extractor
```

### 2.2 Data Flow

```
Input: ref_feature [C], target_feature_map [C, H, W], ref_coord [2]

DSUProcessor.search_depth():
  │
  ├─► DepthSampler.sample_target_features()
  │   │
  │   ├─► For each depth d_k:
  │   │   ├─ Compute projection: (u', v') = project(u, v, d_k)
  │   │   └─ Bilinear sample: tgt_feat_k = sample(target, u', v')
  │   │
  │   └─► Output: target_features [D, C]
  │
  ├─► CostVolume.compute_costs()
  │   │
  │   ├─► For each depth d_k:
  │   │   └─ cost_k = dot(ref_feature, tgt_feat_k)  // or negative for Transplat
  │   │
  │   └─► Output: costs [D]
  │
  └─► SoftmaxAggregator.aggregate()
      │
      ├─► probs = softmax(costs)  // [D]
      ├─► depth = sum(probs * depth_candidates)
      └─► statistics = extract(probs, depth_candidates)

Output: depth, probs, statistics
```

---

## 3. Depth Sampler

### 3.1 Projection Calculation

```python
class DepthSampler:
    def project_to_target(
        self,
        ref_coord: Tensor,           # [2] (u, v)
        depth: float,                # scalar
        ref_intrinsics: Tensor,      # [3, 3]
        ref_extrinsics: Tensor,      # [4, 4]
        tgt_intrinsics: Tensor,      # [3, 3]
        tgt_extrinsics: Tensor,      # [4, 4]
    ) -> Tensor:                     # [2] (u', v')
        """
        Project reference pixel to target view at given depth.
        
        Algorithm:
            1. Backproject to 3D: X = depth * K_ref^{-1} @ [u, v, 1]^T
            2. Transform to world: X_world = T_ref @ X
            3. Transform to target: X_tgt = T_tgt^{-1} @ X_world
            4. Project to 2D: [u', v', 1]^T = K_tgt @ X_tgt[:3]
        """
        # Step 1: Backproject to camera space
        uv_homog = torch.tensor([ref_coord[0], ref_coord[1], 1.0])
        ray_dir = torch.linalg.solve(ref_intrinsics, uv_homog)
        point_cam = depth * ray_dir
        
        # Step 2: Transform to world
        point_cam_homog = torch.cat([point_cam, torch.tensor([1.0])])
        point_world = ref_extrinsics @ point_cam_homog
        
        # Step 3: Transform to target camera
        tgt_extrinsics_inv = torch.linalg.inv(tgt_extrinsics)
        point_tgt = tgt_extrinsics_inv @ point_world
        
        # Step 4: Project to target image
        point_tgt_3d = point_tgt[:3] / point_tgt[2]
        projected = tgt_intrinsics @ point_tgt_3d
        
        return projected[:2]  # (u', v')
```

### 3.2 Bilinear Sampling

```python
def bilinear_sample(
    self,
    feature_map: Tensor,    # [C, H, W]
    coord: Tensor,          # [2] (u, v), possibly fractional
) -> Tensor:                # [C]
    """
    Bilinear interpolation sampling.
    
    Handles boundary conditions by clamping coordinates.
    """
    H, W = feature_map.shape[1], feature_map.shape[2]
    
    # Clamp coordinates to valid range
    u = torch.clamp(coord[0], 0, W - 1)
    v = torch.clamp(coord[1], 0, H - 1)
    
    # Get integer and fractional parts
    u0, v0 = int(u), int(v)
    u1, v1 = min(u0 + 1, W - 1), min(v0 + 1, H - 1)
    du, dv = u - u0, v - v0
    
    # Bilinear interpolation
    f00 = feature_map[:, v0, u0]
    f01 = feature_map[:, v1, u0]
    f10 = feature_map[:, v0, u1]
    f11 = feature_map[:, v1, u1]
    
    result = (
        f00 * (1 - du) * (1 - dv) +
        f10 * du * (1 - dv) +
        f01 * (1 - du) * dv +
        f11 * du * dv
    )
    
    return result
```

### 3.3 Batch Sampling

```python
def sample_target_features(
    self,
    target_feature_map: Tensor,  # [C, H, W]
    ref_coord: Tensor,           # [2]
    depth_candidates: Tensor,    # [D]
    projection_params: dict,     # intrinsics, extrinsics
) -> Tensor:                     # [D, C]
    """
    Sample target features at all depth candidates.
    
    Returns:
        target_features: [D, C] sampled features
    """
    D = len(depth_candidates)
    C = target_feature_map.shape[0]
    
    target_features = torch.zeros(D, C)
    
    for k, depth in enumerate(depth_candidates):
        # Project to target
        tgt_coord = self.project_to_target(
            ref_coord, depth, **projection_params
        )
        
        # Sample feature
        target_features[k] = self.bilinear_sample(
            target_feature_map, tgt_coord
        )
    
    return target_features
```

---

## 4. Cost Volume

### 4.1 Cost Computation

```python
class CostVolume:
    def __init__(self, config: DSUConfig):
        self.config = config
    
    def compute_costs(
        self,
        ref_feature: Tensor,       # [C]
        target_features: Tensor,   # [D, C]
        cost_type: str = 'correlation',
    ) -> Tensor:                   # [D]
        """
        Compute matching costs for all depth candidates.
        
        Args:
            ref_feature: Reference feature vector
            target_features: Target features at each depth
            cost_type: 'correlation' (higher=better) or 'cost' (lower=better)
        
        Returns:
            costs: [D] cost/correlation values
        """
        # Normalize features
        ref_norm = ref_feature / (torch.norm(ref_feature) + 1e-8)
        tgt_norms = target_features / (
            torch.norm(target_features, dim=1, keepdim=True) + 1e-8
        )
        
        # Dot product (cosine similarity)
        costs = tgt_norms @ ref_norm  # [D]
        
        return costs
```

### 4.2 Batch Cost Volume

```python
def compute_correlation_volume(
    self,
    ref_features: Tensor,         # [B, C, H, W]
    target_features: Tensor,      # [B, D, C, H, W]
) -> Tensor:                      # [B, D, H, W]
    """
    Compute full correlation volume for batch.
    
    Used for full-resolution depth prediction.
    """
    B, C, H, W = ref_features.shape
    D = target_features.shape[1]
    
    # Normalize
    ref_norm = F.normalize(ref_features, dim=1)  # [B, C, H, W]
    tgt_norm = F.normalize(target_features, dim=2)  # [B, D, C, H, W]
    
    # Correlation: sum over C dimension
    correlation = torch.einsum('bchw,bdchw->bdhw', ref_norm, tgt_norm)
    
    return correlation
```

---

## 5. Softmax Aggregator

### 5.1 Depth Estimation

```python
class SoftmaxAggregator:
    def __init__(self, config: DSUConfig):
        self.config = config
    
    def aggregate(
        self,
        costs: Tensor,               # [D]
        depth_candidates: Tensor,    # [D]
        temperature: float = 1.0,
        return_distribution: bool = False,
    ) -> Union[float, Tuple[float, Tensor]]:
        """
        Aggregate costs to depth estimate via soft argmax.
        
        Args:
            costs: Cost/correlation values
            depth_candidates: Depth values for each candidate
            temperature: Softmax temperature (lower = sharper)
            return_distribution: Whether to return probability distribution
        
        Returns:
            depth: Expected depth
            probs: (optional) Probability distribution [D]
        """
        # Softmax probabilities
        probs = F.softmax(costs / temperature, dim=0)
        
        # Expected depth
        depth = torch.sum(probs * depth_candidates)
        
        if return_distribution:
            return depth.item(), probs
        return depth.item()
```

### 5.2 Statistics Extraction

```python
def extract_statistics(
    self,
    probs: Tensor,               # [D]
    depth_candidates: Tensor,    # [D]
) -> Tuple[int, float, int, float]:
    """
    Extract statistics for FSDR cache entry.
    
    Returns:
        best_idx: Index of maximum probability
        peak_prob: Maximum probability value
        second_idx: Index of second maximum
        spread: Distribution width (std dev / range)
    """
    D = len(probs)
    
    # Best depth
    best_idx = torch.argmax(probs).item()
    peak_prob = probs[best_idx].item()
    
    # Second best (exclude best)
    probs_copy = probs.clone()
    probs_copy[best_idx] = -float('inf')
    second_idx = torch.argmax(probs_copy).item()
    
    # Spread (weighted standard deviation)
    mean_depth = torch.sum(probs * depth_candidates)
    variance = torch.sum(probs * (depth_candidates - mean_depth) ** 2)
    std_dev = torch.sqrt(variance)
    
    # Normalize by depth range
    depth_range = depth_candidates[-1] - depth_candidates[0]
    spread = (std_dev / depth_range).item()
    
    return best_idx, peak_prob, second_idx, spread
```

---

## 6. DSU Processor

### 6.1 Full Depth Search

```python
class DSUProcessor:
    def __init__(self, config: DSUConfig):
        self.config = config
        self.depth_sampler = DepthSampler(config)
        self.cost_volume = CostVolume(config)
        self.aggregator = SoftmaxAggregator(config)
    
    def search_depth(
        self,
        ref_feature: Tensor,          # [C]
        target_feature_map: Tensor,   # [C, H, W]
        ref_coord: Tensor,            # [2]
        depth_candidates: Tensor,     # [D]
        projection_params: dict,
    ) -> Tuple[float, Tensor]:
        """
        Full depth search for a single pixel.
        
        Returns:
            depth: Estimated depth
            probs: Probability distribution [D]
        """
        # Sample target features at all depths
        target_features = self.depth_sampler.sample_target_features(
            target_feature_map, ref_coord, depth_candidates, projection_params
        )
        
        # Compute costs
        costs = self.cost_volume.compute_costs(ref_feature, target_features)
        
        # Aggregate to depth
        depth, probs = self.aggregator.aggregate(
            costs, depth_candidates, return_distribution=True
        )
        
        return depth, probs
```

### 6.2 Range-Limited Search (for FSDR)

```python
def search_depth_range(
    self,
    ref_feature: Tensor,
    target_feature_map: Tensor,
    ref_coord: Tensor,
    depth_range: Tuple[int, int],  # (start_idx, end_idx)
    depth_candidates: Tensor,
    projection_params: dict,
) -> float:
    """
    Local depth search within specified range.
    
    Used by FSDR light verification.
    
    Args:
        depth_range: (start_idx, end_idx) indices into depth_candidates
    
    Returns:
        depth: Best depth within range
    """
    start_idx, end_idx = depth_range
    local_candidates = depth_candidates[start_idx:end_idx]
    
    # Sample only within range
    target_features = self.depth_sampler.sample_target_features(
        target_feature_map, ref_coord, local_candidates, projection_params
    )
    
    # Compute costs
    costs = self.cost_volume.compute_costs(ref_feature, target_features)
    
    # Find best (no softmax, just argmax for efficiency)
    best_local_idx = torch.argmax(costs).item()
    depth = local_candidates[best_local_idx].item()
    
    return depth
```

---

## 7. Hardware Mapping

### 7.1 Projection Unit

```verilog
module projection_unit (
    input  wire [15:0] ref_coord [0:1],      // (u, v)
    input  wire [15:0] depth,                // Fixed-point depth
    input  wire [15:0] intrinsics [0:8],     // 3x3 matrix
    input  wire [15:0] extrinsics [0:15],    // 4x4 matrix
    output reg  [15:0] tgt_coord [0:1]       // (u', v')
);

// Stage 1: Backproject to camera space (3 MACs)
wire [31:0] ray_dir [0:2];
// ... matrix solve ...

// Stage 2: Scale by depth (3 MULs)
wire [31:0] point_cam [0:2];
// ... depth multiplication ...

// Stage 3: Transform (16 MACs for 4x4 matrix)
wire [31:0] point_world [0:3];
// ... matrix multiply ...

// Stage 4: Project (6 MACs + 2 DIVs)
// ... perspective division ...

endmodule
```

**Resources**: ~300 LUTs, 8 DSPs  
**Latency**: 4 cycles (pipelined)

### 7.2 Bilinear Sampler

```verilog
module bilinear_sampler (
    input  wire [15:0] coord [0:1],          // Fractional (u, v)
    input  wire [15:0] feature_mem [0:127][0:63][0:63],  // [C, H, W]
    output reg  [15:0] sampled_feature [0:127]
);

// Extract integer and fractional parts
wire [5:0] u0 = coord[0][15:10];
wire [5:0] v0 = coord[1][15:10];
wire [9:0] du = coord[0][9:0];
wire [9:0] dv = coord[1][9:0];

// Parallel sampling for all C channels
genvar c;
generate
    for (c = 0; c < 128; c = c + 1) begin : CHANNEL
        // 4 memory reads + bilinear weights
        wire [15:0] f00 = feature_mem[c][v0][u0];
        wire [15:0] f01 = feature_mem[c][v0+1][u0];
        wire [15:0] f10 = feature_mem[c][v0][u0+1];
        wire [15:0] f11 = feature_mem[c][v0+1][u0+1];
        
        // Bilinear interpolation
        wire [31:0] result = 
            f00 * (1024 - du) * (1024 - dv) +
            f10 * du * (1024 - dv) +
            f01 * (1024 - du) * dv +
            f11 * du * dv;
        
        assign sampled_feature[c] = result[31:16];
    end
endgenerate

endmodule
```

**Resources**: ~200 LUTs per channel (shared), 4 memory ports  
**Latency**: 2 cycles

### 7.3 Cost Calculator

```verilog
module cost_calculator (
    input  wire [15:0] ref_feature [0:127],  // [C]
    input  wire [15:0] tgt_feature [0:127],  // [C]
    output reg  [31:0] cost                   // Dot product result
);

// Parallel dot product
wire [31:0] products [0:127];

genvar i;
generate
    for (i = 0; i < 128; i = i + 1) begin : MULTIPLY
        assign products[i] = ref_feature[i] * tgt_feature[i];
    end
endgenerate

// Adder tree (7 stages for 128 inputs)
// ... reduction tree ...

assign cost = reduced_sum;

endmodule
```

**Resources**: 128 DSPs (or 32 DSPs with 4x reuse), ~500 LUTs  
**Latency**: 8 cycles (including reduction)

---

## 8. Resource Summary

| Component | LUTs | DSPs | Memory | Cycles |
|-----------|------|------|--------|--------|
| Projection Unit | 300 | 8 | 0 | 4 |
| Bilinear Sampler | 500 | 8 | Feature cache | 2 |
| Cost Calculator | 500 | 32 | 0 | 8 |
| Softmax Unit | 300 | 4 | 0 | 6 |
| **DSU Total (×1)** | **1,600** | **52** | **Variable** | **~20** |
| **DSU Total (×4)** | **5,600** | **184** | **Variable** | **~20** |

---

## 9. Integration with FSDR

### 9.1 Callback Interface

```python
def create_fsdr_callbacks(dsu: DSUProcessor, target_feature_map, projection_params):
    """
    Create callback functions for FSDR integration.
    """
    def cost_fn(feature, depth_idx):
        """Single depth cost computation."""
        depth = dsu.config.depth_candidates[depth_idx]
        tgt_coord = dsu.depth_sampler.project_to_target(
            ref_coord, depth, **projection_params
        )
        tgt_feature = dsu.depth_sampler.bilinear_sample(
            target_feature_map, tgt_coord
        )
        return torch.dot(feature, tgt_feature).item()
    
    def prob_fn(feature, depth_candidates):
        """Full probability distribution."""
        _, probs = dsu.search_depth(
            feature, target_feature_map, ref_coord,
            depth_candidates, projection_params
        )
        return probs
    
    return cost_fn, prob_fn
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03  
**Related Issue**: GitHub Issue #2
