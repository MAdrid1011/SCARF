# SCARF Architecture Overview

SCARF is a hardware-realizable accelerator for depth-guided generalizable 3D
Gaussian Splatting encoders. It targets the feed-forward encoder path that
constructs Gaussian primitives from multi-view images.

## 1. Motivation

Generalizable 3DGS encoders avoid per-scene optimization by predicting Gaussian
primitives in one forward pass. This makes them suitable for edge deployment,
but the encoder contains a costly sequence of feature extraction, depth
prediction, and Gaussian generation. SCARF focuses on this encoder path.

The accelerator is built around two workload properties:

1. Pixels with similar 2D features often share nearby depth estimates.
2. Locally continuous surfaces often produce redundant Gaussian primitives.

FSDR exploits the first property by narrowing repeated depth searches. SAES
exploits the second property by choosing a cheaper Gaussian-generation path for
regular tiles.

## 2. Top-Level Pipeline

```text
Input views
  -> S1 feature extraction
  -> S2 depth prediction
  -> S3 Gaussian generation
  -> GGU post-processing
  -> Gaussian primitive set
```

| Stage | Function | Main hardware |
|-------|----------|---------------|
| S1 | Extract CNN, Transformer, or ViT features | MVU |
| S2 | Build cost volume and regress depth | MVU, BilinearUnit, FSDR cache |
| S3 | Generate raw Gaussian attributes | MVU, VectorALU, SAES controller |
| GGU | Convert raw attributes to renderable primitives | GGU array |

## 3. Compute Substrate

The Matrix-Vector Unit (MVU) provides the shared compute substrate for S1 to S3.
It contains a 48 by 48 matrix-compute array, vector datapaths, normalization and
activation units, and the control logic needed to switch between convolution,
GEMM, and attention-style execution.

This shared substrate avoids provisioning separate arrays for each stage. Model
variation is represented by weights, tensor dimensions, and instruction streams.
The same datapath supports TranSplat, MVSplat, and DepthSplat.

## 4. Memory System

SCARF keeps high-reuse intermediate data on chip:

- `WeightBuffer` stores the active model weights.
- `FeatureBuffer` stores feature tiles and cross-view sampling data.
- `TileSPM` stores tile-local depth and Gaussian intermediates.
- `FSDRCache` stores feature signatures and depth anchors.

The memory hierarchy is designed to reduce LPDDR traffic during depth search and
Gaussian generation. FSDR cuts repeated feature sampling, while SAES reduces the
number of full Gaussian-generation paths.

## 5. FSDR Datapath

Feature Similarity Depth Reuse sits at the S2 front end. For each query pixel,
it hashes the feature vector into a compact signature and searches a CAM-style
cache. A hit provides a depth anchor. The depth candidate set is then narrowed
around that anchor.

If the feature match or depth-consistency check fails, the pixel falls back to
the full candidate set. This keeps the optimization conservative near depth
boundaries.

## 6. SAES Datapath

Scene-Adaptive Early Sparsification operates at tile granularity. It samples a
small probe set from each tile and computes feature, depth, and Gaussian
consistency statistics. The controller then selects one of three paths:

- L0 representative path for highly regular tiles.
- L1 lightweight path for depth-uniform tiles.
- Full path for irregular tiles.

The lower-cost paths avoid full per-pixel Gaussian materialization when the
local evidence supports the decision.

## 7. GGU Datapath

The Gaussian Generation Unit converts raw network outputs into renderable
Gaussian primitives. It computes 3D position from pixel coordinate and depth,
constructs covariance from scale and rotation parameters, rotates
spherical-harmonic coefficients, and emits opacity.

The GGU is separated from the MVU so primitive construction can overlap with
preparation of later tiles.

## 8. RTL and Simulator Relationship

The Chisel RTL mirrors the Python hardware simulators:

| RTL area | Python reference |
|----------|------------------|
| `compute/` | `encoder/` |
| `ggu/` | `ggu/` |
| `memory/FSDRCache.scala` | `fsdr/cache_table.py` |
| `control/FSDRController.scala` | `fsdr/narrowed_search_simulator.py` |
| `control/SAESController.scala` | SAES path-selection logic |
| `control/PipelineController.scala` | End-to-end stage orchestration |

The Python simulator is the functional and cycle-counting reference. The Chisel
implementation provides a hardware-structured RTL counterpart for synthesis and
inspection.

## 9. Related Documentation

- [Pipeline Architecture](pipeline-architecture.md)
- [FSDR and SAES Mechanisms](fsdr-saes-mechanisms.md)
- [GGU Architecture](ggu-architecture.md)
- [Chisel RTL](../chisel/README.md)
