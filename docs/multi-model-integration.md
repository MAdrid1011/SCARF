# Multi-Model Integration Guide

## 1. Overview

This document describes how SCARF integrates with different generalizable 3DGS encoder architectures: Transplat, MVSPlat, and DepthSplat.

### 1.1 Supported Models

| Model | Feature Source | Depth Estimation | Views |
|-------|----------------|------------------|-------|
| **Transplat** | CNN + Transformer | Cost Volume + Refinement | 2 |
| **MVSPlat** | CNN | Plane-Sweep Stereo | 2+ |
| **DepthSplat** | DINOv2 | Similar to MVSPlat | 3 |

### 1.2 Integration Philosophy

SCARF uses an **adapter pattern** to isolate model-specific logic:

```
Model-specific code (adapters/)
      │
      ▼
Unified interface (BaseAdapter)
      │
      ▼
Generic SCARF components (fsdr/, dsu/, ggu/, saes/)
```

---

## 2. Architecture Differences

### 2.1 Transplat

**Architecture**:
```
Input Images [B, V, 3, H, W]
      │
      ▼
CNN Backbone + Transformer → Features [B, V, C, H/4, W/4]
      │
      ▼
DepthAnythingV2 → Depth Prior + DINO Features
      │
      ▼
Cost Volume Matching → Raw Correlation [B, D, H/4, W/4]
      │
      ▼
Cost Volume U-Net → Refined Correlation [B, D, H/4, W/4]
      │
      ▼
Softmax(-correlation) → Depth Distribution
      │
      ▼
Depth Refinement U-Net → Fine Depth [B, V, H, W]
      │
      ▼
Gaussian Head → Raw Gaussians [B, V, H*W, C_raw]
```

**Key Characteristics**:
- Cost volume uses **negative** values (lower = better match)
- Softmax applied with **negation**: `softmax(-cost)`
- Uses **inverse depth (disparity)** candidates
- 2-view stereo with deformable attention

### 2.2 MVSPlat

**Architecture**:
```
Input Images [B, V, 3, H, W]
      │
      ▼
CNN Encoder → Features [B, V, C, H/4, W/4]
      │
      ▼
Plane-Sweep Stereo → Correlation Volume [B, D, H/4, W/4]
      │
      ▼
Softmax(correlation) → Depth Distribution
      │
      ▼
Expected Depth → Depth [B, V, H, W]
      │
      ▼
Gaussian Decoder → Raw Gaussians
```

**Key Characteristics**:
- Correlation volume uses **positive** values (higher = better)
- Softmax applied **directly**: `softmax(correlation)`
- Uses **depth** candidates (not inverse)
- Multi-view (2+ views)

### 2.3 DepthSplat

**Architecture**:
```
Input Images [B, V, 3, H, W]
      │
      ▼
DINOv2 Backbone → Features [B, V, C_dino, H/14, W/14]
      │
      ▼
Feature Upsampling → Features [B, V, C, H/4, W/4]
      │
      ▼
(Similar to MVSPlat from here)
```

**Key Characteristics**:
- Uses **DINOv2** for feature extraction (self-supervised)
- Features tend to be more **semantically consistent**
- Higher feature similarity across views
- 3-view setup for robust depth

---

## 3. Adapter Interface

### 3.1 Base Adapter

```python
from abc import ABC, abstractmethod
from typing import Tensor, Tuple

class BaseAdapter(ABC):
    """
    Abstract base class for model-specific adapters.
    
    Isolates all model-specific logic from core SCARF components.
    """
    
    @abstractmethod
    def extract_depth_distribution(
        self,
        cost_volume: Tensor,           # [B, D, H, W] or similar
        depth_candidates: Tensor,      # [D]
    ) -> Tensor:
        """
        Convert model-specific cost/correlation volume to probability distribution.
        
        Returns:
            probs: [B, D, H, W] depth probabilities (sum to 1 along D)
        """
        pass
    
    @abstractmethod
    def get_depth_candidates(
        self,
        near: float,
        far: float,
        num_candidates: int,
    ) -> Tensor:
        """
        Generate depth candidates for this model.
        
        Returns:
            candidates: [D] depth values
        """
        pass
    
    @abstractmethod
    def project_to_target(
        self,
        ref_coords: Tensor,            # [N, 2] (u, v) coordinates
        depth: Tensor,                 # [N] depth values
        ref_intrinsics: Tensor,        # [3, 3]
        ref_extrinsics: Tensor,        # [4, 4]
        tgt_intrinsics: Tensor,        # [3, 3]
        tgt_extrinsics: Tensor,        # [4, 4]
    ) -> Tensor:
        """
        Project reference coordinates to target view at given depths.
        
        Returns:
            tgt_coords: [N, 2] target coordinates
        """
        pass
    
    def get_feature_dim(self) -> int:
        """Return feature dimension for this model."""
        return 128  # Default, override as needed
    
    def supports_fsdr(self) -> bool:
        """Whether this model supports FSDR optimization."""
        return True
    
    def supports_saes(self) -> bool:
        """Whether this model supports SAES optimization."""
        return True
```

### 3.2 Transplat Adapter

```python
import torch
import torch.nn.functional as F
from typing import Tensor

class TransplatAdapter(BaseAdapter):
    """
    Adapter for Transplat model.
    
    Key differences from base:
    - Cost volume uses negative values (lower = better)
    - Uses inverse depth (disparity) candidates
    - Softmax with negation
    """
    
    def __init__(self, feature_dim: int = 128):
        self._feature_dim = feature_dim
    
    def extract_depth_distribution(
        self,
        cost_volume: Tensor,       # [B, D, H, W] - costs (lower = better)
        depth_candidates: Tensor,  # [D] - disparity values
    ) -> Tensor:
        """
        Transplat: Convert cost volume to probabilities.
        
        Formula: probs = softmax(-cost_volume, dim=1)
        
        The negation converts "lower is better" to "higher is better"
        before softmax.
        """
        # Negate costs before softmax
        probs = F.softmax(-cost_volume, dim=1)
        return probs
    
    def get_depth_candidates(
        self,
        near: float,
        far: float,
        num_candidates: int,
    ) -> Tensor:
        """
        Transplat: Generate inverse depth (disparity) candidates.
        
        Linear spacing in inverse depth space for better near-range resolution.
        """
        # Inverse depth (disparity) space
        disp_near = 1.0 / far   # Far → small disparity
        disp_far = 1.0 / near   # Near → large disparity
        
        # Linear spacing in disparity
        disparities = torch.linspace(disp_near, disp_far, num_candidates)
        
        # Convert back to depth for interface consistency
        depths = 1.0 / disparities
        
        return depths
    
    def project_to_target(
        self,
        ref_coords: Tensor,
        depth: Tensor,
        ref_intrinsics: Tensor,
        ref_extrinsics: Tensor,
        tgt_intrinsics: Tensor,
        tgt_extrinsics: Tensor,
    ) -> Tensor:
        """
        Standard pinhole camera projection.
        """
        N = ref_coords.shape[0]
        device = ref_coords.device
        
        # Unproject to 3D
        ones = torch.ones(N, 1, device=device)
        uv_homog = torch.cat([ref_coords, ones], dim=1)  # [N, 3]
        
        # K^{-1} @ [u, v, 1]^T
        K_inv = torch.linalg.inv(ref_intrinsics)
        rays = (K_inv @ uv_homog.T).T  # [N, 3]
        
        # Scale by depth
        points_cam = rays * depth.unsqueeze(1)  # [N, 3]
        
        # To world
        R_ref = ref_extrinsics[:3, :3]
        t_ref = ref_extrinsics[:3, 3]
        points_world = (R_ref @ points_cam.T).T + t_ref  # [N, 3]
        
        # To target camera
        R_tgt_inv = tgt_extrinsics[:3, :3].T
        t_tgt = tgt_extrinsics[:3, 3]
        points_tgt = (R_tgt_inv @ (points_world - t_tgt).T).T  # [N, 3]
        
        # Project to 2D
        points_tgt_proj = (tgt_intrinsics @ points_tgt.T).T  # [N, 3]
        tgt_coords = points_tgt_proj[:, :2] / points_tgt_proj[:, 2:3]
        
        return tgt_coords
    
    def get_feature_dim(self) -> int:
        return self._feature_dim
```

### 3.3 MVSPlat Adapter

```python
class MVSplatAdapter(BaseAdapter):
    """
    Adapter for MVSPlat model.
    
    Key differences from Transplat:
    - Correlation volume uses positive values (higher = better)
    - Uses depth (not inverse) candidates
    - Direct softmax
    """
    
    def __init__(self, feature_dim: int = 64):
        self._feature_dim = feature_dim
    
    def extract_depth_distribution(
        self,
        cost_volume: Tensor,       # [B, D, H, W] - correlation (higher = better)
        depth_candidates: Tensor,  # [D] - depth values
    ) -> Tensor:
        """
        MVSPlat: Convert correlation volume to probabilities.
        
        Formula: probs = softmax(correlation, dim=1)
        
        Direct softmax since higher correlation = better match.
        """
        probs = F.softmax(cost_volume, dim=1)
        return probs
    
    def get_depth_candidates(
        self,
        near: float,
        far: float,
        num_candidates: int,
    ) -> Tensor:
        """
        MVSPlat: Generate linear depth candidates.
        
        Linear spacing in depth space.
        """
        depths = torch.linspace(near, far, num_candidates)
        return depths
    
    def project_to_target(
        self,
        ref_coords: Tensor,
        depth: Tensor,
        ref_intrinsics: Tensor,
        ref_extrinsics: Tensor,
        tgt_intrinsics: Tensor,
        tgt_extrinsics: Tensor,
    ) -> Tensor:
        """
        Same projection as Transplat (standard pinhole model).
        """
        # Reuse Transplat implementation
        return TransplatAdapter().project_to_target(
            ref_coords, depth,
            ref_intrinsics, ref_extrinsics,
            tgt_intrinsics, tgt_extrinsics,
        )
    
    def get_feature_dim(self) -> int:
        return self._feature_dim
```

### 3.4 DepthSplat Adapter

```python
class DepthSplatAdapter(MVSplatAdapter):
    """
    Adapter for DepthSplat model.
    
    Inherits from MVSplat since depth estimation is similar.
    
    Key difference:
    - DINOv2 features tend to be more semantically consistent
    - May benefit from different FSDR thresholds
    """
    
    def __init__(self, feature_dim: int = 384):  # DINOv2 typically larger
        super().__init__(feature_dim)
    
    def get_fsdr_config_overrides(self) -> dict:
        """
        DepthSplat-specific FSDR configuration.
        
        DINOv2 features are more semantically consistent,
        so we can use tighter thresholds.
        """
        return {
            'hamming_threshold': 3,  # Stricter (default: 4)
            'high_confidence_threshold': 0.85,  # Higher (default: 0.8)
        }
    
    def supports_three_view(self) -> bool:
        """DepthSplat uses 3 views for robust depth."""
        return True
```

---

## 4. Integration Examples

### 4.1 Creating Adapter for Model

```python
def create_adapter(model_type: str) -> BaseAdapter:
    """
    Factory function to create appropriate adapter.
    """
    adapters = {
        'transplat': TransplatAdapter,
        'mvsplat': MVSplatAdapter,
        'depthsplat': DepthSplatAdapter,
    }
    
    if model_type not in adapters:
        raise ValueError(f"Unknown model: {model_type}. "
                        f"Supported: {list(adapters.keys())}")
    
    return adapters[model_type]()
```

### 4.2 Using Adapter with FSDR

```python
from fsdr import FSDRProcessor, FSDRConfig
from adapters import create_adapter

# Create adapter for specific model
adapter = create_adapter('transplat')

# Configure FSDR with adapter settings
config = FSDRConfig(
    feature_dim=adapter.get_feature_dim(),
    # Apply model-specific overrides if any
    **getattr(adapter, 'get_fsdr_config_overrides', lambda: {})()
)

# Generate depth candidates
depth_candidates = adapter.get_depth_candidates(near=0.5, far=10.0, num_candidates=32)

# Create FSDR processor
fsdr = FSDRProcessor(config, depth_candidates)

# Define probability function using adapter
def prob_fn(feature, candidates):
    # This would come from actual model's cost volume
    cost_volume = compute_cost_volume(feature, candidates)
    return adapter.extract_depth_distribution(cost_volume, candidates)

# Process pixel with FSDR
result = fsdr.process_pixel(feature, position, cost_fn, prob_fn)
```

### 4.3 Using Adapter with Complete Pipeline

```python
from integration import Accelerator
from adapters import create_adapter

# Create accelerator with specific model type
accelerator = Accelerator(
    fsdr_config=FSDRConfig(),
    saes_config=TileConfig(),
    dsu_config=DSUConfig(),
    ggu_config=GGUConfig(),
    model_type='transplat',  # Internally uses TransplatAdapter
)

# Process scene
gaussians, profiling = accelerator.process_scene(
    features=model_features,
    intrinsics=camera_intrinsics,
    extrinsics=camera_extrinsics,
    near=near_planes,
    far=far_planes,
)
```

---

## 5. Model-Specific Optimizations

### 5.1 Transplat Optimizations

| Optimization | Description | Impact |
|--------------|-------------|--------|
| Inverse depth candidates | Better near-range resolution | Quality +5% |
| Negative softmax | Match cost convention | Required |
| FSDR standard thresholds | CNN features moderate similarity | Hit rate ~75% |

### 5.2 MVSPlat Optimizations

| Optimization | Description | Impact |
|--------------|-------------|--------|
| Linear depth candidates | Uniform depth sampling | Standard |
| Direct softmax | Match correlation convention | Required |
| Multi-view projection | Handle 2+ views | Accuracy +3% |

### 5.3 DepthSplat Optimizations

| Optimization | Description | Impact |
|--------------|-------------|--------|
| DINOv2 feature dim | Larger features (384 vs 128) | LSH adjustment |
| Tighter FSDR thresholds | More consistent features | Hit rate +10% |
| 3-view support | Robust depth from 3 views | Quality +8% |

---

## 6. Testing Model Adapters

### 6.1 Unit Tests

```python
def test_transplat_adapter():
    adapter = TransplatAdapter()
    
    # Test depth candidates (inverse depth)
    candidates = adapter.get_depth_candidates(0.5, 10.0, 32)
    assert candidates[0] > candidates[-1]  # Near > Far in depth
    assert len(candidates) == 32
    
    # Test probability extraction
    cost_volume = torch.randn(1, 32, 64, 64)  # Lower = better
    probs = adapter.extract_depth_distribution(cost_volume, candidates)
    assert probs.sum(dim=1).allclose(torch.ones(1, 64, 64))


def test_mvsplat_adapter():
    adapter = MVSplatAdapter()
    
    # Test depth candidates (linear)
    candidates = adapter.get_depth_candidates(0.5, 10.0, 32)
    assert candidates[0] < candidates[-1]  # Near < Far
    
    # Test probability extraction
    corr_volume = torch.randn(1, 32, 64, 64)  # Higher = better
    probs = adapter.extract_depth_distribution(corr_volume, candidates)
    assert probs.sum(dim=1).allclose(torch.ones(1, 64, 64))


def test_depthsplat_adapter():
    adapter = DepthSplatAdapter()
    
    # Should inherit MVSplat behavior
    assert adapter.supports_three_view()
    assert adapter.get_feature_dim() == 384


def test_projection_consistency():
    """All adapters should produce same projection results."""
    adapters = [TransplatAdapter(), MVSplatAdapter(), DepthSplatAdapter()]
    
    ref_coords = torch.tensor([[32.0, 32.0]])
    depth = torch.tensor([5.0])
    K = torch.eye(3)
    K[0, 0] = K[1, 1] = 500.0
    K[0, 2] = K[1, 2] = 32.0
    E = torch.eye(4)
    
    results = [
        adapter.project_to_target(ref_coords, depth, K, E, K, E)
        for adapter in adapters
    ]
    
    for r in results[1:]:
        assert torch.allclose(results[0], r)
```

### 6.2 Integration Tests

```python
def test_fsdr_with_transplat():
    """Test FSDR works correctly with Transplat adapter."""
    adapter = TransplatAdapter()
    config = FSDRConfig(feature_dim=adapter.get_feature_dim())
    candidates = adapter.get_depth_candidates(0.5, 10.0, 32)
    
    fsdr = FSDRProcessor(config, candidates)
    
    # Simulate processing
    feature = torch.randn(128)
    result = fsdr.process_pixel(feature, (32, 32), mock_cost_fn, mock_prob_fn)
    
    assert result.depth > 0
    assert result.source in ['direct_reuse', 'interpolation', 'light_verify', 'full_search']


def test_full_pipeline_all_models():
    """Test complete pipeline works with all model types."""
    for model_type in ['transplat', 'mvsplat', 'depthsplat']:
        accelerator = Accelerator(
            model_type=model_type,
            # ... other configs
        )
        
        # Process synthetic scene
        gaussians, profiling = accelerator.process_scene(
            features=torch.randn(1, 2, 128, 64, 64),
            # ... other params
        )
        
        assert len(gaussians) > 0
        assert profiling.computation_saving_ratio > 0
```

---

## 7. Adding New Model Support

### 7.1 Steps to Add New Model

1. **Create adapter class** inheriting from `BaseAdapter`
2. **Implement required methods**:
   - `extract_depth_distribution()`
   - `get_depth_candidates()`
   - `project_to_target()`
3. **Override optional methods** if needed:
   - `get_feature_dim()`
   - `supports_fsdr()`
   - `supports_saes()`
4. **Add to factory function** in `create_adapter()`
5. **Write unit tests** for new adapter
6. **Document model-specific behavior**

### 7.2 Example: Adding GaussianSplat Support

```python
class GaussianSplatAdapter(BaseAdapter):
    """
    Hypothetical adapter for another 3DGS model.
    """
    
    def __init__(self):
        self._feature_dim = 256
    
    def extract_depth_distribution(self, cost_volume, depth_candidates):
        # Model-specific conversion
        # Example: temperature-scaled softmax
        temperature = 0.5
        return F.softmax(cost_volume / temperature, dim=1)
    
    def get_depth_candidates(self, near, far, num_candidates):
        # Example: log-spaced depth
        return torch.logspace(
            torch.log10(torch.tensor(near)),
            torch.log10(torch.tensor(far)),
            num_candidates
        )
    
    def project_to_target(self, ...):
        # Standard projection or model-specific
        pass
```

---

## 8. Summary

### 8.1 Adapter Pattern Benefits

| Benefit | Description |
|---------|-------------|
| **Isolation** | Model-specific code contained in adapters |
| **Extensibility** | Easy to add new models |
| **Testability** | Each adapter tested independently |
| **Reuse** | Core SCARF components shared across models |

### 8.2 Supported Models

| Model | Adapter | Status |
|-------|---------|--------|
| Transplat | TransplatAdapter | ✅ Supported |
| MVSPlat | MVSplatAdapter | ✅ Supported |
| DepthSplat | DepthSplatAdapter | ✅ Supported |
| Future models | Custom adapter | Easy to add |

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03  
**Related Issue**: GitHub Issue #2
