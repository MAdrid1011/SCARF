# Softmax Aggregator

Probability distribution and depth estimation.

## External Interface

### SoftmaxAggregator

```python
class SoftmaxAggregator:
    def __init__(self, config: DSUConfig)
    
    def compute_probabilities(
        self,
        costs: torch.Tensor,
        negate: bool = False
    ) -> torch.Tensor
    
    def estimate_depth(
        self,
        probabilities: torch.Tensor,
        depth_candidates: torch.Tensor
    ) -> float
    
    def extract_statistics(
        self,
        probabilities: torch.Tensor,
        depth_candidates: torch.Tensor
    ) -> Dict
```

**Methods:**

#### compute_probabilities(costs, negate) -> Tensor
Convert costs to probability distribution via softmax.

- **Input**:
  - `costs` - [D] cost/correlation values
  - `negate` - If True, negate before softmax (for cost volumes)
- **Output**: [D] probabilities summing to 1

**Formula:**
```
if negate:
    costs = -costs
probs = softmax(costs / temperature)
```

#### estimate_depth(probabilities, depth_candidates) -> float
Compute expected depth from distribution.

- **Input**:
  - `probabilities` - [D] probability distribution
  - `depth_candidates` - [D] depth values
- **Output**: Expected depth value

**Formula:**
```
depth = sum(probs[i] × depths[i])
```

#### extract_statistics(probabilities, depth_candidates) -> Dict
Extract statistics for FSDR cache.

- **Output**:
  ```python
  {
      'best_idx': int,      # argmax(probs)
      'second_idx': int,    # second highest
      'peak_prob': float,   # max probability
      'spread': float,      # distribution width
  }
  ```

## Internal Helpers

### _softmax(x, temperature) -> Tensor
Temperature-scaled softmax.

### _compute_spread(probs) -> float
Estimate distribution spread (entropy-based or std-dev).

## Hardware Mapping

- Exponential: LUT-based approximation (~256 entries)
- Sum/Division: ~200 LUTs
- Expected value: D multiply-accumulate
- Total: ~600 LUTs, 4 DSPs
