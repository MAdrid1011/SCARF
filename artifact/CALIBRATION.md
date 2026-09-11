# SCARF Mechanism Calibration Contract

SCARF uses one SHA256-bound mechanism configuration across all model and
dataset pairs. Calibration fixes implementation-level constants while preserving
the FSDR and SAES decision rules.

## Split Contract

The publication protocol is the 24-scene plus eight-scene DL3DV split described
below. The shipped 1.0.4 checkout currently uses the explicit
`acid_train_holdout_v1` fallback because DL3DV calibration is gated and was not
available in this environment. Its frozen bundle is measured from real ACID
target-free traces, has one disjoint train scene and one disjoint holdout scene,
and is not labeled as DL3DV evidence. A release that makes the full paper
calibration claim must replace it with a real `dl3dv_train_holdout_v1` export.

Both protocols require disjoint train/holdout scenes and real target-free
forward traces. `scripts/freeze_acid_calibration.py` rejects missing metrics,
target RGB access, overlapping scenes, and placeholder hashes.

The published evaluation workflows consume the frozen configuration in
`artifact/calibration/frozen/mechanism_config.json`. The bundle includes the
selected tuple, calibration result, manifest summary, and SHA256 bindings;
`scripts.mechanism_config.require_calibrated_mechanism()` verifies all of them
before a claim workflow starts. Full training archives and prepared images
remain at their official source and are not included in the release package.

Claim workflows do not calibrate on the evaluator's machine. The release
bundle already contains a `status: calibrated` configuration with train/holdout
hashes and companion provenance. The preregistered root file remains a
pre-calibration baseline and is intentionally rejected when passed explicitly
to `--claim-run`; the default claim path resolves the verified frozen bundle.

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

The equivalent one-command entry point is:

```bash
python scripts/run_calibration.py --output-root outputs/calibration
```

On success it writes the frozen configuration both to
`artifact/mechanism_config.json` and to
`outputs/calibration/calibration/mechanism_config.json`, together with a
SHA256 sidecar. The latter is the prerequisite file consumed by
`install_claim_prerequisites.py`.

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

Timing is a separate release input. Quality claims consume the frozen
`status: calibrated` configuration and do not require a timing manifest.
Mechanism and performance claims additionally prepare the source-bound RTL
timing bundle described in `docs/claim-timing-backend.md` and pass
`--claim-timing-manifest` to `scripts/run_ae.sh`. The available component
simulators and diagnostic stage scheduler do not satisfy this contract and are
never promoted automatically.
