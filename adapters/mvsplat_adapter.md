# MVSplat Adapter

Adapter for MVSplat model.

## External Interface

### MVSplatAdapter

```python
class MVSplatAdapter(BaseAdapter):
    def __init__(self, feature_dim: int = 64)
    
    def extract_depth_distribution(
        self,
        cost_volume: torch.Tensor,
        depth_candidates: torch.Tensor
    ) -> torch.Tensor
    
    def get_depth_candidates(
        self,
        near: float,
        far: float,
        num_candidates: int
    ) -> torch.Tensor
    
    def project_to_target(self, ...) -> torch.Tensor
    
    def get_feature_dim(self) -> int      # Returns 64
    def get_cost_type(self) -> str        # Returns 'correlation'
```

## Model-Specific Behavior

### Correlation Volume Handling
MVSplat uses **correlation volumes** where higher values indicate better matches.

```python
def extract_depth_distribution(self, cost_volume, depth_candidates):
    # Direct softmax: higher correlation → higher probability
    return F.softmax(cost_volume, dim=depth_dim)
```

### Depth Candidate Generation
MVSplat uses **linear depth** spacing.

```python
def get_depth_candidates(self, near, far, num_candidates):
    return torch.linspace(near, far, num_candidates)
```

## Feature Characteristics

- **Feature Dimension**: 64 (smaller than Transplat)
- **Cost Type**: Correlation (direct softmax)
- **Depth Spacing**: Linear
- **Supports FSDR**: Yes
- **Supports SAES**: Yes

## Comparison with Transplat

| Aspect | MVSplat | Transplat |
|--------|---------|-----------|
| Feature dim | 64 | 128 |
| Volume type | Correlation | Cost |
| Softmax | Direct | Negate first |
| Depth spacing | Linear | Inverse |
