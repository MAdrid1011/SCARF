# DSU Processor

Main processor for depth search.

## External Interface

### DSUProcessor

```python
class DSUProcessor:
    def __init__(self, config: DSUConfig)
    
    def search_depth(
        self,
        ref_feature: torch.Tensor,
        target_feature_map: torch.Tensor,
        pixel_coord: Tuple[float, float],
        depth_candidates: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor
    ) -> DSUResult
    
    def create_fsdr_cost_fn(self, ...) -> Callable[[int], float]
    def create_fsdr_prob_fn(self, ...) -> Callable[[int, int], torch.Tensor]
```

**Methods:**

#### search_depth(...) -> DSUResult
Perform full depth search across all candidates.

- **Input**:
  - `ref_feature` - [D] reference feature
  - `target_feature_map` - [D, H, W] target features
  - `pixel_coord` - (u, v) reference pixel
  - `depth_candidates` - [N] depth values to test
  - Camera parameters for both views
- **Output**: DSUResult with depth and statistics

**Pipeline:**
```
for each depth_candidate:
    1. Project pixel to target view
    2. Sample target feature (bilinear)
    3. Compute matching cost
probabilities = softmax(costs)
depth = expected_value(probabilities, candidates)
stats = extract_statistics(probabilities)
```

#### create_fsdr_cost_fn(...) -> Callable
Create callback for FSDR single-depth cost query.

- **Output**: Function `fn(depth_idx) -> cost`
- **Usage**: FSDR calls this to evaluate individual depths

#### create_fsdr_prob_fn(...) -> Callable
Create callback for FSDR range probability query.

- **Output**: Function `fn(start_idx, end_idx) -> probs[end-start]`
- **Usage**: FSDR light verifier calls this for local search

## Internal Helpers

### _project_and_sample(pixel, depth, ...) -> Tensor
Combined projection and sampling for single depth.

### _compute_all_costs(ref_feature, tgt_features) -> Tensor
Batch cost computation for all candidates.

## Hardware Mapping

- Full search: 32 cycles (1 depth/cycle, pipelined)
- FSDR callbacks: Reuse same hardware
- Total: ~1,800 LUTs, 28 DSPs (including submodules)
