# FSDR and SAES Mechanisms

This document describes the two scene-adaptive mechanisms used by SCARF:
Feature Similarity Depth Reuse (FSDR) and Scene-Adaptive Early Sparsification
(SAES). Both mechanisms reduce encoder work before it reaches the most
expensive depth-search and Gaussian-generation paths.

## 1. Overview

| Mechanism | Redundancy source | Stage | Main action |
|-----------|-------------------|-------|-------------|
| FSDR | Feature-space similarity | S2 depth prediction | Reuse a cached depth anchor to narrow the candidate window |
| SAES | Local 3D continuity | S2 and S3 | Select a lower-cost Gaussian-generation path for regular tiles |

FSDR and SAES are applied to disjoint work. SAES first classifies a tile and
bypasses depth prediction or Gaussian generation when the probe evidence is
sufficient. FSDR then handles the remaining pixels that still require depth
prediction. This keeps their cycle savings additive in the simulator.

## 2. FSDR

FSDR targets the plane-sweep depth search in S2. A standard depth search
evaluates all depth hypotheses for each pixel. FSDR observes that pixels with
similar encoder features often have nearby depth estimates. It therefore caches
feature signatures and depth anchors from previously processed pixels.

### 2.1 Narrowed Depth Search

FSDR changes the depth-search window as follows:

```text
Baseline search: evaluate D depth candidates
FSDR hit:        evaluate D/R candidates around the cached depth anchor
FSDR miss:       evaluate D depth candidates
```

The default contraction ratio is 4. For a 128-candidate search, a guided pixel
therefore evaluates 32 candidates. The U-Net refinement and depth regression
still run normally, so FSDR only removes low-value cost-volume construction
work.

### 2.2 Feature Hashing

The FSDR cache is indexed by a compact locality-sensitive hash of the feature
vector:

```text
feature[128] -> random projection ROM -> sign bits -> 16-bit signature
```

The hardware compares the query signature against all valid cache entries in
parallel. The closest entry is selected by Hamming distance. A query is guided
only when the signature distance is within the threshold and the cached depth
passes the confidence and depth-consistency checks.

### 2.3 Safety Checks

FSDR avoids using cached depths at likely discontinuities. The simulator checks
recent neighboring depths against the cached anchor. If the relative difference
is too large, the pixel falls back to the full depth-search path. This prevents
a feature match from forcing a narrow search across object boundaries.

### 2.4 Hardware Structures

FSDR is implemented with a small hash-and-CAM subsystem:

- `LSHHashUnit` computes the feature signature.
- `FSDRCache` stores signatures, depth anchors, confidence values, and valid bits.
- A comparator tree selects the nearest valid signature.
- A small depth-history register file supports the discontinuity check.

## 3. SAES

SAES targets redundant Gaussian materialization in locally regular tiles. It
uses a small set of probes to estimate feature, depth, and Gaussian consistency.
The result selects one of three generation paths.

| Path | Trigger condition | Action |
|------|-------------------|--------|
| L0 representative path | Low feature variance | Use representative Gaussians for the tile |
| L1 lightweight path | Low depth variance | Reuse a lightweight geometry path |
| Full path | Irregular tile | Run full per-pixel Gaussian generation |

### 3.1 Probe Selection

Each tile uses corner probes plus a gradually increasing set of interior probes.
Interior probes are spread across a subgrid to avoid clustered decisions. This
keeps the path decision stable as tile size changes.

### 3.2 Path Decision

SAES computes three statistics:

- Feature variance for L0 decisions.
- Depth variance for L1 decisions.
- Gaussian similarity for validation and quality control.

The default thresholds are selected from the sensitivity study in the paper. A
tile only takes a lower-cost path when the probe statistics indicate enough local
regularity.

### 3.3 Representative Gaussian Generation

For L0 tiles, SAES builds representative Gaussians by weighted moment matching.
Position and covariance use first- and second-moment statistics. Opacity and
spherical-harmonic coefficients are averaged with range checks. This keeps the
representative path conservative in textured or geometrically complex areas.

## 4. Simulator Integration

The end-to-end demo enables both mechanisms by default:

```bash
python scripts/demo.py --model transplat
python scripts/demo.py --model mvsplat
python scripts/demo.py --model depthsplat
```

They can be disabled independently:

```bash
python scripts/demo.py --model transplat --no-fsdr
python scripts/demo.py --model transplat --no-saes
python scripts/demo.py --model transplat --no-fsdr --no-saes
```

The simulator reports cycle counts, path statistics, cache hit rates, Gaussian
counts, and image-quality metrics. These counters are used by the paper to
attribute speedup to hardware execution, FSDR, and SAES.

## 5. Related Documentation

- [Architecture Overview](architecture.md)
- [Pipeline Architecture](pipeline-architecture.md)
- [GGU Architecture](ggu-architecture.md)
- [Multi-Model Demo Guide](multi-model-demo-guide.md)
