# __init__.py

Package initialization for depth predictor hardware simulators.

## External Interface

### Exports

**Types:** `DepthPredictorConfig`, `DepthPredictorOutput`, `CycleBreakdown`, `CostVolumeType`, `DepthRegressionType`, `UNetConfig`, `UNetLayerConfig`

**Base classes:** `BaseDepthPredictorSim`, `PassThroughDepthPredictorSim`

**Core engine:** `HWDepthPredictor`

**Model-specific predictors:**
- `TransplatDepthPredictorSim` — TranSplat depth pipeline
- `MVSplatDepthPredictorSim` — MVSplat depth pipeline
- `DepthSplatDepthPredictorSim` — DepthSplat depth pipeline

**Factory functions:** `create_transplat_predictor`, `create_mvsplat_predictor`, `create_depthsplat_predictor`

## Internal Helpers

None — this module only re-exports from submodules.
