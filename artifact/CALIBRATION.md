# SCARF Mechanism Calibration Contract

SCARF uses one global mechanism configuration for all model and dataset pairs.
Calibration resolves engineering details that the paper does not assign exact
numeric values to; it does not redefine FSDR or SAES and it does not use the
paper's reported results as an optimization target.

## Isolation Boundary

The primary calibration source is the official gated DL3DV corpus. Its plan
selects 24 calibration scenes and eight holdout scenes from the pinned
`DL3DV/DL3DV-ALL-480P` revision using only archive paths and hashes, and rejects
every scene in the 140-scene DL3DV evaluation index. The plan records the terms
URL, revision, complete listed-tree SHA256, evaluation-index SHA256, and the
exact selected archive hashes before any image is downloaded. The current
account must have upstream access to that gated source; the downloader fails
closed instead of reusing evaluation scenes or bypassing terms. Full training
archives and prepared images are not redistributed in the release artifact.

This is an author-side, pre-submission operation. It is deliberately separate
from `quick`, `pilot`, and both `all-eval` profiles: a reviewer never downloads
the full training archives or runs `calibrate`. Claim execution reads the
single frozen, SHA256-bound `artifact/mechanism_config.json`; the evidence
archive carries the calibration manifest and candidate-record digests needed to
audit that configuration without distributing upstream training images.

Before any gated image download, produce the source-bound selection plan. The
downloader obtains the official archive tree through the authenticated Hugging
Face API, requires every selected object to expose a stable upstream object id,
and writes the plan before it requests any archive bytes. It uses the LFS
SHA256 when the gated API exposes it; otherwise it binds the revision-pinned
Git blob id and records the archive's actual SHA256 after download:

```bash
python data/download_dl3dv_calibration.py \
  --write-plan outputs/calibration/dl3dv-download-plan.json \
  --evaluation-index depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
  --revision 5902ed6d707cc13a7779907c1e096676f7707971
```

The plan remains `PLANNED_AWAITING_UPSTREAM_ACCESS` until the selected archives
can be legally downloaded and their internal camera/image layout is verified.
The existing Re10K/ACID `calibrate` command is retained for Functional
regression only and must not be substituted for this DL3DV calibration contract.

After reviewing that immutable plan, download and safely extract all 32
selected archives. The tool verifies every archive's advertised byte count and
bound upstream object id, records the actual SHA256 of each downloaded ZIP,
and preserves the plan-bound 24-scene training and eight-scene holdout sets in
the prepared-tree provenance.

```bash
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

The compiler writes separate native and Re10K-compatible target-free sidecars
for `calibration_train` and `calibration_holdout`. Each contains only selected
context-image bytes, all camera metadata, and fixed view indices; target RGB
bytes are never copied into either sidecar and replay rejects a record that
carries them. The source subset retains the official archive-tree hash,
selected archive hashes, prepared-tree manifest, and split provenance. The
compiler rejects missing, overlapping, extra, or evaluation-scene sidecars.

The global grid consumes only the training sidecars: TranSplat and MVSplat use
the Re10K-compatible representation, while DepthSplat uses native DL3DV. The
holdout pass receives exactly the training-selected tuple and may not enumerate
or rerank the grid. The existing Re10K/ACID `run_ae.sh calibrate` command
remains Functional regression only and cannot freeze a Results Reproduced
configuration.

```bash
python scripts/calibration_sweep.py \
  --manifest outputs/calibration/dl3dv-protocol/manifest.json \
  --output-dir outputs/calibration/dl3dv-sweep

python scripts/calibrate_mechanisms.py \
  --candidate-records outputs/calibration/dl3dv-sweep/candidates.json \
  --output-dir outputs/calibration/dl3dv-sweep \
  --config-output artifact/mechanism_config.json
```

The sweep runs the full registered grid only on `calibration_train`, writes the
selected tuple and holdout request, then invokes every holdout trace with that
exact tuple. `calibrate_mechanisms.py` reopens and rehashes the trace files,
sidecar provenance, train decision, and holdout result before it writes a
calibrated configuration. A stale or same-selection trace from a different
sidecar is rejected even when a resumable pair directory already exists.

The DL3DV planner domain-separates a 24-scene calibration rank and an eight-scene
holdout rank. The compiled manifest records every scene/view and source-tree
hash. A scene present in the finalized 140-scene evaluation protocol is
rejected. Re10K/ACID training calibration remains a separately documented
fallback only if a legal public scene-disjoint multiview source is pinned; it
is not a substitute for DL3DV evaluation scenes.

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

Each registered candidate is replayed over the same compiled training traces. A
candidate is feasible only when baseline-render versus approximated-render
degradation is at most 0.05 dB PSNR, 0.003 SSIM, and 0.003 LPIPS on every
training pair. Among feasible candidates, the compiler selects the tuple with
the largest event-derived S2+S3 work reduction. Ties select less compression,
then the lexicographically smallest tuple. The selected tuple alone is replayed
on every holdout pair under the same quality constraints; the holdout never
reranks candidates. The decision does not compare Guided Rate, L0/L1 rate,
speedup, or quality with manuscript values.

The selected record is written to `artifact/mechanism_config.json` only after
the exact tuple passes holdout. It binds both split manifests, target-free
sidecar/provenance hashes, trace sets, and candidate records. It is frozen
before the first evaluation run; changing it changes source identity and
invalidates resume/evidence reuse.

## ACID Materialization Epoch Boundary

All earlier ACID joint-materialization contracts, caches, smoke runs,
runtime evidence, and result records are superseded. They must not be run,
resumed, promoted, or included in a reviewer-facing release.

ACID v5 is the only active frozen author-side epoch. Its local provenance was
generated and live-validated, but it is not execution proof. No registered
deterministic verifier or external authority exists, so its candidate runtime,
promotion, and result paths remain fail-closed; no cache, train, runtime, or
result evidence exists. It remains `paper_result_eligible=false`: it cannot
authorize a DL3DV target-RGB quality gate, replace the required DL3DV
train/holdout calibration, establish a global S2/S3 saving, or support any
Results Reproduced claim.

## Fixed Sample-0 SAES Diagnostic

The public 4-by-4, 12-anchor, guard-on selector
`bash scripts/run_ae.sh saes-quality --profile dl3dv-gate` is
calibration-gated. Until an evaluation-disjoint configuration is frozen, it
fails closed; when available, it produces only a non-claim DL3DV sample-0
diagnostic. It neither selects a configuration nor supplies a table or figure
aggregate.

The current v5 sample-0 record preserves quality by taking the Full path for
every tile. It demonstrates fallback fidelity only and supplies no SAES
reduction evidence for Table 3 or Figure 11.

## Faithfulness Boundary

FSDR remains random-hyperplane LSH plus associative cache lookup and a cached
anchor-centered `D/4` search. The local validity check conservatively rejects a
Hamming hit when already available probe/neighbor depth evidence conflicts with
the cached anchor; it does not introduce another reuse method.

SAES routing remains the published probe feature-variance then probe
depth-standard-deviation first-hit hierarchy. L0 uses probe-anchored bilateral
aggregation. At `T=4`, L1 uses the four primary corner probes plus the fixed
eight edge anchors, for twelve retained anchors total. Its depth-reliability
mean and standard deviation are computed from the primary four routing probes;
the extra L1 anchors are an execution and aggregation expansion, not a new
depth reference. Neither path may inspect target images or full non-probe
Stage-3 outputs. The context-only safety guard may use only selected probes,
their depths, and context camera projection data; missing or invalid geometry
must leave the tile on the Full path.
For moment matching, the released implementation uses the already available
C2W camera transform, normalized intrinsics, assignment-weighted probe depth,
and the non-probe pixel coordinate to lift a pseudo 3D mean on that pixel's
camera ray. This is geometry construction for the published moment update, not
a routing signal or a substitute Gaussian-adaptor invocation.
