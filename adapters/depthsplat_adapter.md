# DepthSplat Adapter

Adapter for DepthSplat model.

## External Interface

### DepthSplatAdapter

```python
class DepthSplatAdapter(MVSplatAdapter):
    def __init__(self, feature_dim: int = 384)
    
    def get_fsdr_config_overrides(self) -> Dict
    def get_saes_config_overrides(self) -> Dict
    def supports_three_view(self) -> bool
```

## Model-Specific Behavior

### Inheritance from MVSplat
DepthSplat extends MVSplat with same basic behavior:
- Correlation volumes (direct softmax)
- Linear depth spacing

### DINOv2 Features
DepthSplat uses DINOv2 features which are:
- **Higher dimensional**: 384 vs 64/128
- **More semantically consistent**: Better for FSDR caching

### FSDR Configuration
Tighter thresholds due to semantic feature consistency:

```python
def get_fsdr_config_overrides(self):
    return {
        'hamming_threshold': 3,           # Tighter (was 4)
        'high_confidence_threshold': 0.85, # Higher (was 0.8)
    }
```

### SAES Configuration
```python
def get_saes_config_overrides(self):
    return {
        'early_stop_threshold': 0.92,  # Higher threshold
    }
```

### Three-View Support
```python
def supports_three_view(self) -> bool:
    return True  # DepthSplat uses 3 views for robust depth
```

## Feature Characteristics

- **Feature Dimension**: 384 (DINOv2)
- **Cost Type**: Correlation (inherited)
- **Depth Spacing**: Linear (inherited)
- **Supports FSDR**: Yes (with tighter thresholds)
- **Supports SAES**: Yes (with higher early-stop)
- **Three-View**: Yes
