# DSU Types

Data structures for Depth Search Unit.

## External Interface

### DSUConfig

Configuration dataclass for DSU processor.

```python
@dataclass
class DSUConfig:
    feature_dim: int = 128
    num_candidates: int = 32
    cost_type: str = 'correlation'  # 'correlation' or 'cost'
    temperature: float = 1.0
```

**Fields:**
- `feature_dim`: Feature vector dimension
- `num_candidates`: Number of depth candidates to evaluate
- `cost_type`: Matching cost type
  - `'correlation'`: Higher = better match (direct softmax)
  - `'cost'`: Lower = better match (negate before softmax)
- `temperature`: Softmax temperature for sharpness control

### DSUResult

Result from depth search.

```python
@dataclass
class DSUResult:
    depth: float
    depth_idx: int
    probability: torch.Tensor   # [D] distribution
    confidence: float           # Peak probability
    spread: float              # Distribution spread
    num_candidates: int
```

**Methods:**

#### get_fsdr_stats() -> Dict
Extract statistics for FSDR cache entry.

```python
{
    'best_depth_idx': int,
    'second_depth_idx': int,
    'peak_prob': float,
    'spread': float,
}
```

## Hardware Mapping

- DSUConfig: Compile-time parameters
- DSUResult: Pipeline registers
  - depth: 16-bit fixed-point
  - probability: 32 × 8-bit (256 bytes)
  - confidence/spread: 8-bit each
