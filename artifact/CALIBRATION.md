# SCARF Mechanism Calibration Contract

SCARF uses one SHA256-bound mechanism configuration across all model and
dataset pairs. Calibration fixes implementation-level constants while preserving
the FSDR and SAES decision rules.

## DL3DV Split Contract

The calibration protocol selects 24 DL3DV calibration scenes and eight holdout
scenes from the pinned `DL3DV/DL3DV-ALL-480P` revision. Both sets are disjoint
from the 140-scene evaluation index. The plan records the dataset revision,
terms URL, archive tree hash, evaluation-index hash, selected archive hashes,
and prepared-tree provenance.

The published evaluation workflows consume the frozen configuration in
`artifact/mechanism_config.json`. Full training archives and prepared images
remain at their official source and are not included in the release package.

## Preparation

```bash
python data/download_dl3dv_calibration.py \
  --write-plan outputs/calibration/dl3dv-download-plan.json \
  --evaluation-index depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
  --revision 5902ed6d707cc13a7779907c1e096676f7707971

python data/download_dl3dv_calibration.py \
  --plan outputs/calibration/dl3dv-download-plan.json \
  --evaluation-index depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
  --revision 5902ed6d707cc13a7779907c1e096676f7707971 \
  --output-root downloads/calibration/dl3dv

python data/prepare_dl3dv_calibration_inputs.py \
  --raw-root downloads/calibration/dl3dv/prepared \
  --plan outputs/calibration/dl3dv-download-plan.json \
  --preparation-record downloads/calibration/dl3dv/.scarf-dl3dv-calibration-source.json \
  --output-dir outputs/calibration/dl3dv-protocol
```

The compiler emits native and Re10K-compatible context sidecars for the
calibration and holdout splits. It preserves camera metadata, fixed view
indices, source hashes, and split provenance.

## Fixed Search Space

The random-hyperplane projection remains the shared seed-42 FP16 ROM. The
finite search space is registered before execution:

- FSDR local depth-validity tolerance `gamma_depth`: `0.05`, `0.075`, `0.10`,
  `0.15`.
- SAES normalized spatial bandwidth `beta_x`: `0.25`, `0.50`, `1.00`.
- SAES normalized feature bandwidth `beta_f`: `0.05`, `0.10`, `0.20`.
- SAES depth-reliability multiplier `beta_d`: `0.50`, `1.00`, `2.00`.

FSDR cache size 32, Hamming threshold 3, contraction ratio 4, SAES feature
threshold 0.2, depth threshold 0.1, and tile size 4 are fixed mechanism
parameters.

## Selection and Freeze

```bash
python scripts/calibration_sweep.py \
  --manifest outputs/calibration/dl3dv-protocol/manifest.json \
  --output-dir outputs/calibration/dl3dv-sweep

python scripts/calibrate_mechanisms.py \
  --candidate-records outputs/calibration/dl3dv-sweep/candidates.json \
  --output-dir outputs/calibration/dl3dv-sweep \
  --config-output artifact/mechanism_config.json
```

Each tuple is evaluated on the same calibration traces. Feasible tuples meet
the configured PSNR, SSIM, and LPIPS limits; the selection rule chooses the
largest event-derived S2+S3 reduction, then the deterministic tie break. The
selected tuple is replayed unchanged on the holdout split and written to the
global configuration with its split, trace, and provenance hashes.

## Mechanism Boundary

FSDR remains random-hyperplane LSH with associative cache lookup and a cached
anchor-centered `D/4` search. SAES preserves the probe feature-variance then
probe depth-standard-deviation first-hit hierarchy. At `T=4`, L0 uses four
corner probes and L1 uses those corners plus eight fixed boundary anchors. The
configuration, route, retained descriptors, Gaussian materialization, and
execution trace are all bound into the resulting record.
