# SAES Examples

## Standalone Demo

The standalone demo (`saes_standalone_demo.py`) demonstrates SAES usage with mock data.

### Running the Demo

```bash
cd SCARF
source /home/mazirui/anaconda3/bin/activate transplat
export PYTHONPATH=/home/mazirui/transplat/SCARF:$PYTHONPATH
python examples/saes_standalone_demo.py
```

### Expected Output

```
SAES Profiling for demo_scene:
  Total tiles: 16
  Early-stop: X tiles (Y%)
  Sparse: X tiles (Y%)
  Full: X tiles (Y%)
  Pixels saved: N
  Computation saving: X.XX%
  SAES overhead: X.XX ms
  Net speedup: X.XX×
```

**Note**: With random mock data, most tiles take full-continue path due to low similarity. Real scenes with geometric continuity show better savings (30-50%).

## Integration with Transplat

### Quick Integration

For quick transplat integration, wrap existing depth predictor and Gaussian adapter:

```python
from saes import TileProcessor, TileConfig

# In encoder_trans.py forward()
if enable_saes:
    # Configure SAES
    saes_config = TileConfig(tile_size=4, ...)
    processor = TileProcessor(saes_config)
    
    # Define callbacks (adapt to transplat's interface)
    def depth_fn(feats, indices):
        # Call transplat depth_predictor for specific indices
        return self.depth_predictor(feats, indices=indices, ...)
    
    def gaussian_fn(depths, ctx):
        # Call transplat gaussian_adapter
        return self.gaussian_adapter(depths, ctx, ...)
    
    # Use SAES
    gaussians, saes_result = processor.process_scene(
        trans_features, depth_fn, gaussian_fn, context
    )
else:
    # Original path
    depths, densities, raw_gaussians = self.depth_predictor(...)
    gaussians = self.gaussian_adapter(...)
```

### Full Integration

See `docs/saes-usage.md` Section 9 for complete integration checklist and transplat-specific modifications.

## Additional Examples (Coming Soon)

- `saes_re10k_evaluation.py` - Evaluate SAES on real RE10K scenes
- `saes_quality_comparison.py` - Compare SAES vs. baseline quality metrics
- `saes_performance_profiling.py` - Detailed performance analysis
