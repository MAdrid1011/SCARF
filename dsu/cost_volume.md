# Cost Volume

Feature matching cost computation.

## External Interface

### CostVolume

```python
class CostVolume:
    def __init__(self, config: DSUConfig)
    
    def compute_cost(
        self,
        ref_feature: torch.Tensor,
        tgt_feature: torch.Tensor
    ) -> float
    
    def compute_cost_volume(
        self,
        ref_feature: torch.Tensor,
        tgt_features: torch.Tensor
    ) -> torch.Tensor
```

**Methods:**

#### compute_cost(ref_feature, tgt_feature) -> float
Compute matching cost between two features.

- **Input**:
  - `ref_feature` - [D] reference feature
  - `tgt_feature` - [D] target feature
- **Output**: Scalar cost/correlation value

**Cost Types:**
- `correlation`: `dot(ref, tgt) / (||ref|| × ||tgt||)` (cosine similarity)
- `cost`: Negative correlation (for Transplat)

#### compute_cost_volume(ref_feature, tgt_features) -> Tensor
Compute costs for multiple target features.

- **Input**:
  - `ref_feature` - [D] reference feature
  - `tgt_features` - [N, D] target features at different depths
- **Output**: [N] cost values

## Internal Helpers

### _normalize(feature) -> Tensor
L2 normalize feature vector.

### _dot_product(a, b) -> float
Compute dot product of two vectors.

## Hardware Mapping

- Normalization: ~200 LUTs (reciprocal sqrt LUT)
- Dot product: D multiply-accumulate operations
- For D=128: ~400 LUTs, 8 DSPs
- Pipelined: 1 cost per cycle after initial latency
