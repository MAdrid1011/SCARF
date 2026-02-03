# Accelerator

Complete SCARF accelerator integrating all components.

## External Interface

### Accelerator

```python
class Accelerator:
    def __init__(
        self,
        model_type: str = 'transplat',
        fsdr_config: Optional[FSDRConfig] = None,
        dsu_config: Optional[DSUConfig] = None,
        ggu_config: Optional[GGUConfig] = None,
        enable_fsdr: bool = True,
        enable_profiling: bool = True
    )
    
    def process_pixel(
        self,
        ref_feature: torch.Tensor,
        target_feature_map: torch.Tensor,
        pixel_coord: torch.Tensor,
        depth_candidates: torch.Tensor,
        raw_gaussian: torch.Tensor,
        density: float,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor
    ) -> Tuple[dict, dict]
    
    def setup_fsdr(self, depth_candidates: torch.Tensor)
    def get_fsdr_profiling(self) -> Optional[dict]
    def reset(self)
```

**Constructor:**
- `model_type`: `'transplat'`, `'mvsplat'`, or `'depthsplat'`
- `fsdr_config`: Override FSDR settings (uses adapter defaults if None)
- `enable_fsdr`: Toggle FSDR optimization
- `enable_profiling`: Toggle profiling collection

**Methods:**

#### process_pixel(...) -> (gaussian_dict, profiling_dict)
Process single pixel through complete pipeline.

- **Output**: 
  - `gaussian_dict`: Position, covariance, color, opacity
  - `profiling_dict`: FSDR stats (source, cache_hit, searches)

**Pipeline:**
```
1. If FSDR enabled:
   a. Create DSU callbacks
   b. FSDR process → depth + source
2. Else:
   a. DSU full search → depth
3. GGU generate → Gaussian
4. Return results
```

#### setup_fsdr(depth_candidates)
Initialize FSDR processor with depth values.

- Must be called before `process_pixel` if FSDR enabled

#### reset()
Clear all state for new scene.

## Internal Components

- `adapter`: Model-specific adapter (created from model_type)
- `fsdr`: FSDRProcessor (created by setup_fsdr)
- `dsu`: DSUProcessor
- `ggu`: GGUProcessor

## Hardware Mapping

Combined resources from all submodules:
- FSDR: ~1,500 LUTs, 8 DSPs, 2.2KB SRAM
- DSU: ~1,800 LUTs, 28 DSPs
- GGU: ~1,200 LUTs, 32 DSPs
- Control: ~200 LUTs
- **Total**: ~4,700 LUTs, 68 DSPs, 2.2KB SRAM
