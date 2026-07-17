# SCARF Mechanism Calibration Contract

SCARF uses one global mechanism configuration for all model and dataset pairs.
Calibration resolves engineering details that the paper does not assign exact
numeric values to; it does not redefine FSDR or SAES and it does not use the
paper's reported results as an optimization target.

## Isolation Boundary

Calibration scenes come only from the official Re10K and ACID training splits.
The author-side recovery command is `bash data/download_calibration_splits.sh`;
it verifies the published archive byte counts before hashing. The current
official SimpleHTTP mirror does not support byte-range requests, so an
interrupted full-archive download is deliberately preserved rather than falsely
resumed; `SCARF_CALIBRATION_RESTART_PARTIAL=1` is required to discard it and
restart. Full training archives and prepared images are not redistributed in
the release artifact.

This is an author-side, pre-submission operation. It is deliberately separate
from `quick`, `pilot`, and both `all-eval` profiles: a reviewer never downloads
the full training archives or runs `calibrate`. Claim execution reads the
single frozen, SHA256-bound `artifact/mechanism_config.json`; the evidence
archive carries the calibration manifest and candidate-record digests needed to
audit that configuration without distributing upstream training images.

After the archives complete, prepare and execute calibration with:

```bash
bash data/download_calibration_splits.sh
"${SCARF_PYTHON_CLASSIC:-python3}" data/prepare_calibration_splits.py
SCARF_PYTHON_CLASSIC=/path/to/classic/bin/python \
  bash scripts/run_ae.sh calibrate --output-root outputs/calibration
```

`calibrate` runs its compiler, trace replay, and configuration writer with the
same locked classic profile that loads the Re10K/ACID chunks; it never relies
on whichever Python happens to launch the shell wrapper.

The preparation step extracts only the selected chunks into
`downloads/calibration/prepared/<dataset>/`. The compiler then writes a
separate target-free sidecar under the calibration output: it contains only
the selected context-image bytes, all camera metadata, and the fixed view
indices. Target RGB bytes are not copied into that tree and calibration replay
rejects a record that carries them. The source subset retains the full official
training index, the selected 32-scene index, archive hash, prepared-tree
manifest, and provenance record. The compiler rejects a subset unless it is
exactly the SHA256-ranked selection from that full index.

The compiler selects 32 scenes from each dataset by sorting
`SHA256("SCARF-AE-calibration-v1\0" + dataset + "\0" + scene)` and taking the
first 32 valid scenes. The compiled manifest records every scene/view and the
source-tree hash. A scene or view present in the finalized evaluation protocol
is rejected. DL3DV is evaluation-only and never participates in calibration.

The calibration process may read context images, model inputs, pretrained-model
outputs, hardware event traces, and baseline-versus-approximated renders. It
must not read target RGB images, ground-truth quality metrics,
`artifact/expected_results.json`, manuscript CSV files, or completed evaluation
results. The runner records the complete opened-input manifest.

## Fixed Search Space

The random-hyperplane projection is not calibrated. It remains the recorded
seed-42 FP16 ROM shared by software and RTL. The following finite engineering
space is registered before calibration:

- FSDR local depth-validity tolerance `gamma_depth`:
  `0.05, 0.075, 0.10, 0.15`.
- SAES tile-normalized spatial bandwidth `beta_x`:
  `0.25, 0.50, 1.00`.
- SAES normalized feature bandwidth `beta_f`:
  `0.05, 0.10, 0.20`.
- SAES depth-reliability multiplier `beta_d`:
  `0.50, 1.00, 2.00`.

FSDR cache size 32, Hamming threshold 3, contraction ratio 4, SAES feature
threshold 0.2, depth threshold 0.1, and tile size 4 are paper defaults and are
not selected by calibration. Per-model, per-dataset, per-scene, and per-sample
parameter overrides are forbidden in a claim run.

## Selection Rule

Each candidate is replayed over the same compiled calibration traces. A
candidate is feasible only when baseline-render versus approximated-render
degradation is at most 0.05 dB PSNR, 0.003 SSIM, and 0.003 LPIPS on every
calibration pair. Among feasible candidates, the compiler selects the tuple
with the largest event-derived S2+S3 work reduction. Ties select less
compression, then the lexicographically smallest tuple. The decision does not
compare Guided Rate, L0/L1 rate, speedup, or quality with manuscript values.

The selected record is written to `artifact/mechanism_config.json` together
with calibration manifest hashes and raw candidate records. It is frozen before
the first evaluation run; changing it changes source identity and invalidates
resume/evidence reuse.

## Faithfulness Boundary

FSDR remains random-hyperplane LSH plus associative cache lookup and a cached
anchor-centered `D/4` search. The local validity check conservatively rejects a
Hamming hit when already available probe/neighbor depth evidence conflicts with
the cached anchor; it does not introduce another reuse method.

SAES routing remains the published probe feature-variance then probe
depth-standard-deviation first-hit hierarchy. L0 uses probe-anchored bilateral
aggregation. L1 uses a less-compressive `2K(T)` lightweight probe-constrained
path. Neither path may inspect target images or full non-probe Stage-3 outputs.
For moment matching, the released implementation uses the already available
C2W camera transform, normalized intrinsics, assignment-weighted probe depth,
and the non-probe pixel coordinate to lift a pseudo 3D mean on that pixel's
camera ray. This is geometry construction for the published moment update, not
a routing signal or a substitute Gaussian-adaptor invocation.
