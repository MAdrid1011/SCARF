# Transplat Adapter

Adapter for Transplat model.

## External Interface

### TransplatAdapter

```python
class TransplatAdapter(BaseAdapter):
    def __init__(self, feature_dim: int = 128)
    
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
    
    def get_feature_dim(self) -> int      # Returns 128
    def get_cost_type(self) -> str        # Returns 'cost'
    def get_fsdr_config_overrides(self) -> Dict
```

## Model-Specific Behavior

### Cost Volume Handling
Transplat uses **cost volumes** where lower values indicate better matches.

```python
def extract_depth_distribution(self, cost_volume, depth_candidates):
    # Negate before softmax: lower cost → higher probability
    return F.softmax(-cost_volume, dim=depth_dim)
```

### Depth Candidate Generation
Transplat uses **inverse depth (disparity)** spacing for better near-range resolution.

```python
def get_depth_candidates(self, near, far, num_candidates):
    disp_near = 1.0 / far   # Far → small disparity
    disp_far = 1.0 / near   # Near → large disparity
    disparities = torch.linspace(disp_near, disp_far, num_candidates)
    return 1.0 / disparities  # Convert back to depth
```

### FSDR Configuration
```python
def get_fsdr_config_overrides(self):
    return {
        'hamming_threshold': 4,
        'high_confidence_threshold': 0.8,
    }
```

## Feature Characteristics

- **Feature Dimension**: 128
- **Cost Type**: Cost (negate before softmax)
- **Depth Spacing**: Inverse (disparity)
- **Supports FSDR**: Yes
- **Supports SAES**: Yes
