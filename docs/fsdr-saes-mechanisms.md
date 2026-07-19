# FSDR and SAES Mechanisms

This document describes the two scene-adaptive mechanisms used by SCARF:
Feature Similarity Depth Reuse (FSDR) and Scene-Adaptive Early Sparsification
(SAES). FSDR narrows a depth-search window when its cache conditions pass. The
current SAES software path is a route and materialization diagnostic applied
after the dense model has completed S2/S3 and materialized full Gaussian
descriptors. It does not verify sparse S2/S3 execution, RTL-cycle timing, or an
SAES speedup.

## 1. Overview

| Mechanism | Redundancy source | Stage | Main action |
|-----------|-------------------|-------|-------------|
| FSDR | Feature-space similarity | S2 depth prediction | Reuse a cached depth anchor to narrow the candidate window |
| SAES | Local 3D continuity | After dense S2/S3 | Classify tiles and materialize diagnostic retained output from dense descriptors |

FSDR and SAES are represented as distinct mechanism paths. That separation does
not make their accounting results additive hardware evidence. In the current
demo, all S2/S3 model work completes before SAES receives cloned full Gaussian
descriptors. SAES therefore provides route and materialization diagnostics
rather than a measured bypass of depth prediction or Gaussian generation.

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

SAES specifies a route for redundant Gaussian materialization in locally regular
tiles. It uses a small set of probes and follows the published first-hit
decision order: feature statistic for L0, then depth statistic for L1 after an
L0 miss, then Full. The current software performs this classification after
dense Gaussian materialization and uses the available descriptors to construct
diagnostic output. Route counts and retained-descriptor counts therefore do not
show physical sparse S2/S3 work.

| Path | Trigger condition | Diagnostic action |
|------|-------------------|--------|
| L0 representative path | Low feature variance | Construct representative output from already materialized descriptors |
| L1 lightweight path | L0 miss and low probe-depth standard deviation | Retain the declared anchors for diagnostic materialization |
| Full path | Irregular tile | Keep the dense descriptor output |

### 3.1 Probe Selection

Each tile uses corner probes plus a gradually increasing set of interior probes.
Interior probes are spread across a subgrid to avoid clustered decisions. This
keeps the path decision stable as tile size changes.

### 3.2 Path Decision

SAES computes the two decision statistics specified by the mechanism:

- Feature variance for L0 decisions.
- Probe-depth standard deviation for L1 decisions after an L0 miss.

At T=4, L1 computes its depth-reliability mean and standard deviation from four
primary corner routing probes. If L1 is selected, it retains those four primary
anchors plus eight boundary anchors, for twelve anchors total. The boundary
anchors do not alter the L1 reference statistics.

A claim-run threshold must be selected through the disjoint training-calibration
contract. It may not use evaluation RGB, paper tables, or evaluation aggregates.
The checked-in configuration is preregistered without a selected tuple, so it
does not authorize an SAES quality, work-reduction, or speed claim.

### 3.3 Representative Gaussian Generation

For L0 diagnostic output, SAES builds representative Gaussians from already
materialized descriptors by weighted moment matching. Position and covariance
use first- and second-moment statistics. Opacity and spherical-harmonic
coefficients are averaged with range checks. This keeps the representative path
conservative in textured or geometrically complex areas.

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

The demo reports path statistics, cache hit rates, Gaussian counts, image
metrics, and analytical accounting. The hardware ledger is no-overlap analytic
accounting, not RTL-cycle-equivalent timing. Its derived S2-evaluation count is
not an observed sparse kernel invocation. Runtime statistics also omit direct
primary-probe, secondary-probe, and Full replay counts, so they cannot bind a
complete event schedule. A caller-configured stage-event schedule can audit
declared assumptions, but it is not RTL timing evidence.

### 4.1 Artifact claim boundary

The current artifact claims no sparse-SAES Table 1 or Tables 2--3 rows. The
current Python path materializes full S3 descriptor tensors before SAES and has
not verified sparse S2/S3 execution. The analytical ledger and any
caller-declared stage-event schedule are diagnostic only. Neither can support a
SAES timing, PPA, work-reduction, or quality claim.

### 4.2 Target-Free FSDR Audits

`--claim-run --fsdr-only` becomes a paper-eligible FSDR evidence path only after
the global configuration has been selected and frozen by the calibration
contract. The checked-in preregistered configuration does not meet that
condition. `--diagnostic-run --fsdr-only --image-output-policy none` instead
emits an `fsdr_target_free_audit`: it removes target RGB before device transfer
and records that RGB was not passed to the model, routing, or metrics. The FSDR
aggregator rejects this diagnostic kind, so it cannot become Table 2 evidence
without a fresh calibrated claim run.

## 5. Related Documentation

- [Architecture Overview](architecture.md)
- [Pipeline Architecture](pipeline-architecture.md)
- [GGU Architecture](ggu-architecture.md)
- [Multi-Model Demo Guide](multi-model-demo-guide.md)
