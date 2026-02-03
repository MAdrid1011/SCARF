# Base Adapter

Abstract interface for model-specific adapters.

## External Interface

### BaseAdapter (Abstract)

```python
class BaseAdapter(ABC):
    @abstractmethod
    def extract_depth_distribution(
        self,
        cost_volume: torch.Tensor,
        depth_candidates: torch.Tensor
    ) -> torch.Tensor
    
    @abstractmethod
    def get_depth_candidates(
        self,
        near: float,
        far: float,
        num_candidates: int
    ) -> torch.Tensor
    
    @abstractmethod
    def project_to_target(
        self,
        ref_coords: torch.Tensor,
        depth: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor
    ) -> torch.Tensor
    
    def get_feature_dim(self) -> int
    def get_cost_type(self) -> str
    def supports_fsdr(self) -> bool
    def supports_saes(self) -> bool
    def get_fsdr_config_overrides(self) -> Dict
    def get_saes_config_overrides(self) -> Dict
```

**Abstract Methods (must implement):**

#### extract_depth_distribution(cost_volume, depth_candidates) -> Tensor
Convert model-specific cost/correlation to probability distribution.

- Different models use different softmax conventions
- Transplat: negate costs before softmax
- MVSplat/DepthSplat: direct softmax

#### get_depth_candidates(near, far, num_candidates) -> Tensor
Generate depth candidates for this model.

- Transplat: inverse depth (disparity) spacing
- MVSplat/DepthSplat: linear spacing

#### project_to_target(...) -> Tensor
Project reference coordinates to target view.

- Standard pinhole projection (shared implementation)

**Optional Methods (have defaults):**

#### get_feature_dim() -> int
Return feature dimension (default: 128).

#### get_cost_type() -> str
Return `'correlation'` or `'cost'` (default: `'correlation'`).

#### get_fsdr_config_overrides() -> Dict
Return model-specific FSDR config adjustments.

## Usage Pattern

```python
class MyModelAdapter(BaseAdapter):
    def extract_depth_distribution(self, cost_volume, depth_candidates):
        return F.softmax(cost_volume, dim=0)  # Model-specific
    
    def get_depth_candidates(self, near, far, num):
        return torch.linspace(near, far, num)  # Model-specific
```
