# SCARF Pipeline Architecture

SCARF is an ASIC-oriented simulator for depth-guided generalizable 3D Gaussian
Splatting encoders. The pipeline maps encoder inference into hardware stages
that are shared across TranSplat, MVSplat, and DepthSplat.

## 1. Pipeline Overview

```text
Input images and camera parameters
  -> S1 feature extraction
  -> S2 depth prediction
  -> S3 Gaussian generation
  -> GGU Gaussian post-processing
  -> splatting renderer
```

| Stage | Main computation | Hardware units |
|-------|------------------|----------------|
| S1 | CNN, Transformer, optional ViT features | ConvEngine, GEMMUnit, VectorALU |
| S2 | Cost volume, U-Net refinement, depth head | BilinearUnit, ConvEngine, VectorALU |
| S3 | Gaussian parameter head | ConvEngine, GEMMUnit, VectorALU |
| GGU | Position, covariance, opacity, SH conversion | GGU processing elements |

## 2. Hardware Configuration

| Unit | Configuration | Role |
|------|---------------|------|
| ConvEngine | 48 by 48 MAC array | Convolutions and U-Net layers |
| GEMMUnit | 48 by 48 MAC array | Matrix multiply and attention projections |
| VectorALU | 64-wide SIMD | Softmax, normalization, reductions, path statistics |
| BilinearUnit | 32 parallel samplers | Cross-view feature sampling |
| GGU array | 32 processing elements | Gaussian post-processing |
| Clock | 1 GHz | Post-layout target in 28 nm |

## 3. S1 Feature Extraction

S1 extracts multi-scale features from the input views. CNN layers are mapped to
the ConvEngine through tiled convolution. Transformer and ViT projection layers
use the GEMMUnit. Normalization, activation, and softmax operations use the
VectorALU and dedicated element-wise units.

The same compute substrate supports all evaluated models. Model differences are
handled by weights, tensor shapes, and instruction sequences rather than by
model-specific datapaths.

## 4. S2 Depth Prediction

S2 builds a cost volume by projecting each reference pixel into the target view
over a set of depth hypotheses. The BilinearUnit handles irregular cross-view
sampling so the main matrix units can remain focused on refinement and depth
regression.

FSDR can reduce this stage by narrowing the candidate window for pixels that hit
in the feature-similarity cache. The U-Net and depth head still process the
selected candidates through the normal hardware path.

## 5. S3 Gaussian Generation

S3 converts depth-conditioned features into raw Gaussian attributes. SAES can
redirect regular tiles to lower-cost paths before full per-pixel materialization.
Irregular tiles use the full generation path.

The hardware reuses the ConvEngine, GEMMUnit, and VectorALU from earlier stages.
This avoids duplicating arrays for a stage that has lower duty cycle than the
feature and depth paths.

## 6. GGU Post-Processing

The Gaussian Generation Unit converts raw network outputs into renderable 3D
Gaussian primitives. It computes position from pixel coordinates and depth,
builds covariance from scale and rotation parameters, rotates spherical-harmonic
coefficients, and emits opacity.

GGU execution overlaps with later tile preparation whenever dependencies allow.
This lets the MVU advance to the next tile while Gaussian post-processing
continues.

## 7. Data Movement

SCARF keeps high-frequency feature, depth, and Gaussian data on chip:

- The FeatureBuffer stores feature tiles and cost-volume inputs.
- The WeightBuffer feeds convolution and matrix kernels.
- The TileSPM stores intermediate tile data between S2 and S3.
- DRAM traffic is reserved for input images, weights, and final outputs.

FSDR reduces repeated feature sampling. SAES reduces intermediate Gaussian work.
Together, these mechanisms reduce both compute cycles and off-chip memory energy.

## 8. Running the Pipeline

```bash
python scripts/demo.py --model transplat
python scripts/demo.py --model mvsplat
python scripts/demo.py --model depthsplat
```

The same script can disable mechanisms for ablation:

```bash
python scripts/demo.py --model transplat --no-fsdr --no-saes
```

## 9. Related Documentation

- [Architecture Overview](architecture.md)
- [FSDR and SAES Mechanisms](fsdr-saes-mechanisms.md)
- [GGU Architecture](ggu-architecture.md)
- [Multi-Model Demo Guide](multi-model-demo-guide.md)
