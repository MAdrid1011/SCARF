# MICRO 2026 Artifact Implementation Checklist

## Planning

- [x] Claim boundary and public hardware substitution recorded.
- [x] Paper-result mapping and badge scope documented.
- [x] Implementation touchpoints and strict failure policy recorded.

### Active L0/L1 Repair

- [x] Confirm the original L0/L1 semantics: non-probes are soft-assigned and
  merged into retained probe anchors, not directly deleted when nonzero.
- [x] Separate producer request masks from final compact output masks.
- [x] Add selected-anchor-only L0/L1 moment materialization with PSD and
  range-constrained SH/opacity checks.
- [x] Add failure-first synthetic, poison, and Full-promotion tests.
- [x] Run the fixed local DL3DV sample-0 V8c compact quality gate after the
  unit and packet checks: 34.8391 -> 30.1855 dB, so the candidate is rejected.
- [x] Run the V9b target-free native-geometry/spatial-attribute/L0-depth-
  continuity packet and coverage audit: `118136/131072` descriptors and
  5,545 absolute dense-domain holes, with no skipped-S3 reads.
- [x] Run the fixed V9 sample-0 quality gate: `34.8391 -> 33.0140 dB`; it is
  improved but still outside every quality tolerance.
- [x] Run the V10 L1-plane target-free packet and coverage audit: only two
  fewer holes (`5,545 -> 5,543`), so no redundant quality gate was run.
- [x] Run the V12 balanced-12 target-free packet and coverage audit: holes
  worsened (`5,545 -> 6,607`) and omitted optical-mass recall fell
  (`55.96% -> 45.18%`), so no GT quality gate was run.
- [x] Implement and audit the S1-only adaptive L1-15 center single-omission
  merge: absolute holes fell (`5,545 -> 1,667`) and uncontained omitted
  optical mass fell (`0.00807 -> 0.00253`), while normalized recall fell
  (`55.96% -> 44.83%`).
- [x] Run the fixed V13 sample-0 quality gate: `33.0140 -> 33.9187 dB`, but
  PSNR and LPIPS remain outside tolerance.
- [x] Run the V14 S1-dominance L1-15 -> Full target-free audit: 3,231 of
  3,234 L1 tiles promoted to Full, yielding `131069/131072` descriptors and
  only two absolute holes without GT or skipped-S3 reads.
- [x] Run the fixed V14c sample-0 quality gate after correcting target access
  order: `34.8391 -> 34.8387 dB`, but it is an effectively Full upper bound
  with only three L1 tiles and must not tune the S1 dominance threshold.
- [x] Rework V16's selected-anchor-only V4 center leave-one-out replay
  certificate for L1-15 tiles as an evaluation-disjoint, immutable-manifest
  route; any failed tile promotes Full and its 24/8 ACID records are frozen.
- [x] Historical only: freeze a target-free four-scene S1 absolute-LOO V15
  diagnostic and run its quality gate. It retained `128846/131072` descriptors
  and reached 34.2030 dB, but its DL3DV sample-1--4 calibration cannot
  authorize the active evaluation-disjoint route.
- [x] Select V16 as the distinct complementary mechanism: a frozen,
  selected-anchor V4 center-LOO certificate. It is a risk filter only; V4's
  output formula and V15's frozen p50 threshold remain unchanged.
- [x] Repair the native `opacity=1.0` endpoint without clamping: preserve it
  on Full, promote an affected compact tile Full before V4/logit replay, keep
  compact updates open-interval, and reject values above one. Materializer and
  incremental-adapter CPU regressions pass.
- [x] Run the target-free ACID holdout endpoint smoke before the fresh V15/V16
  freeze: `4fa73a829dde9435` completed with three native endpoints, zero
  endpoint compact-anchor promotions, 4,019 risk tiles, a valid self-hash,
  and no target or skipped-S3 access.
- [x] Replace the legacy DL3DV sample-1--4 calibration records with the ACID
  24/8 context-only freeze: V15 `821ce...e9596`, V16 `a2786...1ba`, train-only
  thresholds, and fixed-threshold holdout verification.
- [x] Run one DL3DV sample-0 target-free audit under the frozen pair: PASS,
  self-hash `d09de...7589`, target-free access evidence, and source-bound
  packed Adapter equivalence are recorded.
- [x] Preserve `quality_v1` as an invalid pre-render preflight: route binding
  drift was rejected before metrics, so it supplies no quality result and may
  not be used for tuning or reporting.
- [x] Run the one permitted quality gate under the same V15/V16/audit SHA
  bindings after the quality runner consumes the exact audited sidecar before
  loading target data: `quality_v2` passed at `34.839137 -> 34.779134 dB`
  (PSNR loss `0.060003 dB`), with SSIM loss `0.0002273` and LPIPS increase
  `0.0004545`. The committed route is `L0/L1/Full=0/206/7986`; it authorizes
  only the fixed V16 8-scene development gate, not a sparse-execution claim or
  metric tuning rerun.
- [ ] Run the registered TranSplat/DL3DV fixed eight-scene V16 gate with
  source ordinals `0..7`. Each scene must first produce its own target-free
  source/context sidecar and packed Adapter audit, then pass the exact-audit
  quality gate. Require every scene plus 32-view pooled and scene-macro
  PSNR/SSIM/LPIPS gates to pass before any MVSplat, DepthSplat, 140-scene,
  Figure 11, or cross-dataset work.

## Documentation

- [x] AE guide, claims, hardware scope, appendix, HotCRP draft, and checklist.
- [x] Third-party dependency and license inventory.
- [ ] Fill all pending release fields only after real archive validation.

## Software

- [x] DeepScale implementation and result-schema validator unit tested.
- [x] AE orchestrator and shell compatibility entry pass tests.
- [x] Dataset-aware demo CLI and aggregate schema are unit tested.
- [x] Loader mapping is tested for all nine model/dataset pairs.
- [x] Quick mode uses a deterministic synthetic Functional fixture that claim
  runs reject.
- [x] Strict CUDA quick completes end to end with no simulator fallback and a
  schema-valid one-sample aggregate.
- [x] Classic and DepthSplat interpreters are resolved separately and validated
  before model loading.
- [x] The DepthSplat CUDA 12.1 profile is installed and passes its complete
  package, nvcc, xFormers, and rasterizer lock check.
- [x] DL3DV native and Re10K-compatible preparation paths are separate.
- [x] Every sampler-selected target view is recorded and included in sample quality.
- [x] Sensitivity uses one model trace per sample and replays all paper grids.
- [x] Upstream sample/view protocol is recovered, finalized, and hashed.
- [x] Stable source ordinals and prepared-chunk execution ordinals are both
  recorded without changing the canonical selection identity.
- [x] Re10K and ACID sources, terms, archive revisions, prepared tree hashes,
  and all protocol-selected scene/view bounds are verified.
- [x] All six accessible model and dataset pairs completed a historical strict one-sample
  pilot with three target views, full checkpoint coverage, and positive S1,
  S2, S3, and GGU cycles.
- [x] The invalid full Re10K/ACID run was stopped after four complete samples,
  archived, and classified as an SAES implementation mismatch.
- [x] Table 1 and Tables 2--3 were removed from the claim set instead of using
  dense interpolation, edited results, or relaxed tolerances.
- [x] DL3DV and Orin results are explicitly outside the current claim.
- [x] Six real FSDR-only pilots completed without fallback; all six Guided Rate
  rows fail Table 2 and are retained as negative evidence.
- [x] Preserve the historical FSDR exclusion: its six-pair and 32-sample runs
  used software hyperplanes while the then-tracked RTL ROM was all zero.
- [x] Replace the all-zero RTL with the manifest-hashed seed-42 FP16 ROM, exact
  Q1.24 operand decode, signed 16-MAC projection, and bit-equivalence tests.
- [ ] Rerun the six accessible FSDR rows under the fixed projection contract;
  historical negative evidence is not promoted retroactively.
- [x] Replace the continuous-depth-window Top-1 approximation with exact
  full-search argmax-in-candidate-subset evidence and rerun six bounded pilots.
- [x] Keep every Table 2 Top-1 row diagnostic-only: the exact tensors are
  available, but their guided denominator depends on missing LSH collateral.

## RTL And Hardware

- [x] Chisel behavior tests and SystemVerilog emission pass.
- [x] Verilator lint and representative switching VCD pass.
- [x] Result schema v2 records executed S1-S3 useful/scheduled MMCU slots; the
  strict pilot produces nontrivial ratios without reading paper targets.
- [x] Figure 12 compares the executed-slot evidence against the paper bars and
  fails honestly when the old heuristic estimate differs by more than 2pp.
- [x] Ramulator and DRAMPower LPDDR5 functional proxy passes.
- [x] Pinned Ramulator 2.1 and DRAMPower 6.0.2 binaries were built and the
  real trace-to-timing-to-energy chain produced relocatable PASS evidence.
- [x] Clean iFlow `04b4d98` front end through global placement was exercised.
- [x] Clean iFlow/ASAP7/container/collateral preflight passes from pinned
  commit `04b4d98`.
- [x] A clean unchanged ASAP7 attempt completed synthesis through filler and
  reached global routing, then stopped on verified persistent swap thrashing;
  its hashed `attempt-outcome.json` is `NOT_CLAIMED_RESOURCE_LIMIT`, and no
  partial run or manuscript constant is substituted.
- [x] SRAM proxy, PDN evidence, and excluded PHY/I/O fields are explicit.
- [x] 7-to-28 execution is `NOT_CLAIMED_NO_PHYSICAL_INPUT`; deterministic
  DeepScale tables, examples, and formulas remain unit-tested Functional code.

## Validation And Release

- [x] Every currently claimed result passes expected-result validation.
- [x] Release checker rejects paths, missing DOI values, bad hashes, and incomplete dataset licenses.
- [x] CPU, CUDA, RTL, DRAM, and evidence-bundle clean-room checks are recorded.
  Orin and routed physical results are explicitly not claimed.
- [ ] Release tag, submodule SHAs, archive SHA256, and Zenodo DOI agree.

## Current Frontier

The strict quick path and all six historical one-sample pilots are executable,
but the pilot gate originally checked schema rather than paper tolerance. A
corrected formal run and focused TranSplat/MVSplat diagnostics prove that
sparse SAES does not meet Table 1 or Tables 2--3. C1/C4 are suspended and the
dense diagnostic is forbidden in claim runs. Formal RTL/DRAM evidence remains
complete. The clean iFlow preflight passed, but a 180-design external Vivado
sweep and low available memory prevented a standard full run. The audited
low-memory attempt preserves the unchanged design and all completion gates.
C7/C8 are not claimed. Pre-release source/evidence bundles pass the
clean-room hash, validation, public-asset download, and strict CUDA quick
checks. Do not mark routed PPA or Results Reproduced evidence complete from
diagnostics, and repeat the clean-room checks on the final DOI-bound bundles.

### Third-Badge Closure

- [x] Add a global preregistered mechanism configuration and v2.1 provenance.
- [x] Refuse paper claim runs until the configuration is calibrated on a
  disjoint training split.
- [x] Implement reviewer/full frozen profiles and profile-aware sensitivity.
- [x] Add probe-only SAES materialization, FSDR local-validity event evidence,
  and matching RTL control tests.
- [x] Complete a strict v2.1 Functional quick run after the mechanism update.
- [ ] Complete the official Re10K training download and compile 32 scenes.
- [x] Verify the official ACID training archive and prepare its fixed
  evaluation-disjoint 32-scene calibration tree. Archive SHA256:
  ddecee0c6cbb3a5e4fa5cd0182b6b7e31dc7dc23e586437ad8489199f16a18d0;
  prepared-tree SHA256:
  9ba8600d156e90a17c98f6599bedb4a0c17ec370835978f5cc02961c83e6726d.
  This is author-side Functional calibration material only, not a
  Results-Reproduced configuration freeze.
- [x] Keep the 554 GB/174 GB full-training archives author-side only; reviewer
  commands are regression-tested not to reference the calibration path.
- [ ] Run all registered calibration candidates and freeze one global tuple.
- [ ] Pass the six-pair mechanism gate, reviewer profile, then full profile.
- [ ] Obtain independent Orin NX evaluator evidence for mandatory Figure 8.

### Paper-Formula SAES Audit

- [x] Run the bounded canonical TranSplat/Re10K diagnostic without changing
  the dataset, selected views, materialization policy, or quality metrics.
- [x] Preserve the failed output under
  `outputs/ae_failures/diagnostics/transplat-re10k-paper-formula-v1/`.
- [x] Stop wider reruns: L0 rose from 29.2% to 99.9%, L1 remained effectively
  zero, PSNR fell from 26.1724 to 23.5173 dB, and SSIM fell from 0.87879 to
  0.80966 relative to the prior sparse diagnostic.
- [x] Restore the last-known-good Functional implementation. The paper does
  not define the feature-vector variance reduction/normalization or justify
  the additional Gaussian gates well enough to recover the reported path mix.
- [x] Keep C1/C4 and Results Reproduced suspended; do not expand this failed
  hypothesis to MVSplat, ACID, DepthSplat, or the full matrix.

### SAES Recovery Campaign V2

- [x] Bind the campaign to the fixed TranSplat/Re10K sample and immutable
  quality/selection contract.
- [x] Prioritize decision-statistic and sparse-coverage attribution before
  another threshold or full-matrix retry.
- [x] Add failure-first tests for diagnostic CLI isolation, representative
  layout recovery, and covariance/opacity coverage transforms.
- [x] Emit raw/unit-normalized vector-variance, absolute/relative depth-spread,
  and current gate distributions from one real run.
- [x] Run a fixed covariance/opacity sweep without changing selected tiles or
  using a dense materialization.
- [x] Attribute representative attributes independently. All restorations
  failed, while the current moment-matched representatives remained the least
  damaging sparse variant.
- [x] Record the structural L1 blocker: all tiles are below the current L0
  statistic threshold, and L0/L1 share the same Gaussian cross-check, making
  L1 unreachable after an L0 miss.
- [x] Add a target-free retention-boundary diagnostic at one-quarter intervals
  through the paper-declared L0 rate, plus the current measured L0 rate.
- [x] Run that diagnostic in a new non-overwriting directory and decide whether
  4/16 sparse materialization has any path to the paper's declared L0 rate.
- [x] Test the manuscript's literal constrained-average SH/opacity wording.
  Preserve the -5.1204 dB failure and revert it instead of promoting a
  paper-aligned but nonfunctional implementation.
- [x] No theory-consistent SAES correction passed PSNR, SSIM, and LPIPS on the
  fixed sample; preserve the contradiction without promotion.
- [x] Do not restore the formal SAES matrix or C1/C4 claims because the existing
  validators do not pass unchanged.
- [x] Close the sparse-SAES line after both target-free rankings failed at every
  tested nonzero rate; keep C1/C4 not claimed.

### FSDR Table 2 Recovery

- [x] Define a Table 2-only result schema and validator using the existing
  expected values and absolute mechanism tolerance.
- [x] Add failure-first tests for FSDR-only CLI isolation, persistent model
  reuse, complete-sample resume, selection hash, and zero/reference fallback.
- [x] Implement a persistent FSDR-only pair runner that skips SAES and rendering
  while preserving the canonical model/data path.
- [x] Run one exact-discrete sample for TranSplat, MVSplat, and DepthSplat on
  Re10K and ACID under `outputs/ae_pilot_fsdr_v2/`.
- [x] Stop before the six full protocol rows: every Guided Rate fails and the
  paper/RTL guided set cannot be recovered from the all-zero projection ROM.

### Mechanism Recovery Campaign V3

- [x] Bind the campaign to commit `6f210ad`, the canonical fixed samples, and
  unchanged paper thresholds and validators.
- [x] Add failure-first tests for inverse-depth candidate normalization and
  aligned L0-to-L1 first-hit statistics.
- [x] Run the candidate-coordinate SAES diagnostic without changing sparse
  materialization or using target images for routing.
- [x] Reject candidate-coordinate routing: it yields 3.80% L0 and 94.89% L1 on
  the fixed TranSplat/Re10K sample, farther from the declared split.
- [x] Add failure-first tests for all-context FSDR measurement and per-frame
  cache reset.
- [x] Run the corrected TranSplat/Re10K bounded diagnostic over both context
  frames; 90.36% Guided Rate and 96.812% Top-1 Coverage still fail.
- [x] Rerun the six bounded FSDR pilots with authentic probabilities and
  candidate tensors from every context frame.
- [x] Run a fixed 32-sample canonical-prefix aggregate for all six pairs. Every
  row failed over 262,144 authentic pixels, confirming a systematic
  projection/feature mismatch; preserve `outputs/ae_pilot_fsdr_v3_32/`.
- [x] Add failure-first tests for model-specific FSDR feature-source selection
  and runtime feature dimension without changing the default feature path.
- [x] Run one diagnostic-only canonical DepthSplat sample using its authentic
  1,024-channel DINOv2 mono tensor. Guided Rate was 99.976% and exact Top-1
  Coverage was 68.791%; preserve the failed evidence and reject promotion.
- [x] Promote the projection implementation independently of paper result
  values: seed 42 is fixed by the source contract, the ROM is SHA256-bound, and
  Python/RTL basis-vector signatures are bit-exact.
- [ ] Resume six quality pilots and full matrices only after the corresponding
  claim-critical mechanism gates pass.
- [x] Add `--pairs` to the public runner so the six legally available pairs can
  be executed without inserting a gated DL3DV job; duplicate, unknown, and
  non-software uses fail before execution.
- [x] Run all six seed-42 one-sample mechanism pilots under one source-tree
  identity in `outputs/ae_v2_six_pair_pilot_v2/`.
- [x] Stop before full protocols: all six L1 rates are zero, every Guided Rate
  fails, all Re10K Top-1 rows fail, and combined PSNR deltas range from
  -0.5624 to -5.0876 dB under the unchanged implementation.

### SAES Probe-Vector Decision Audit

- [x] Bind the audit to one canonical TranSplat/Re10K sample and unchanged
  thresholds, materialization, quality metrics, and selection.
- [x] Add failure-first tests for raw probe-vector total variance, per-view
  probe selection, and L1 execution strictly after an L0 miss.
- [x] Add an explicitly non-claiming diagnostic CLI mode; keep the default SAES
  decision behavior compatible.
- [x] Run the fixed sample once and preserve the complete result under a new
  non-overwriting diagnostic directory.
- [x] Reject promotion: L1 reached 86.267%, while PSNR, SSIM, and LPIPS missed
  the unchanged gates by large margins. Keep six-pair pilots stopped.

### Paper Assignment-Formula Audit

- [x] Add failure-first tests for the Section 3 `sigma_feat^2` spatial term,
  L1 probe-depth reliability factor, and absolute depth-standard-deviation
  gate.
- [x] Implement those formulas without changing the frozen threshold, seed,
  selection, quality metric, or routing inputs.
- [x] Run the canonical non-claim TranSplat/Re10K sample under
  `outputs/ae_failures/diagnostics/transplat-re10k-paper-formula-v2/`.
- [x] Preserve the -6.3895 dB SAES-only quality failure and keep all wider
  mechanism/quality runs stopped.

### SAES Gaussian-Head Feature Audit

- [x] Bind the audit to the canonical TranSplat/Re10K sample and identify the
  authentic full-resolution Gaussian-head input in the upstream source.
- [x] Add failure-first tests for pre-hook capture, `[B,V,C,H,W]` alignment,
  strict-run rejection, and non-claim result eligibility.
- [x] Implement the diagnostic feature source without changing the default
  pipeline feature path.
- [x] Run one clean fixed sample and record feature statistics, L0/L1/Full
  rates, and unchanged PSNR/SSIM/LPIPS deltas.
- [x] Reject promotion because every quality gate failed; keep the
  full six-pair matrix stopped.
- [x] Preserve v1/v2 implementation failures, add regression tests for SAES/FSDR
  tensor isolation and retention-boundary selection, and complete v3 at clean
  commit `5cee926` without fallback.
- [x] Record the v3 contradiction: L0/L1/Full was 29.2%/0.0%/70.8%, SAES-only
  deltas were -0.8090 dB PSNR, -0.027802 SSIM, and +0.066505 LPIPS, and even
  the 4.2969% retention slice failed LPIPS at +0.019034.

### SAES Hardware-Honest Probe-Spread Audit

- [x] Trace the probe-only implementation to repository commit `adc7092` and
  confirm the final Dataflow figure contains no extra L1 retained-pixel path.
- [x] Add failure-first tests that reject the diagnostic in claim/Functional
  runs, preserve strict no-fallback execution for a diagnostic run, force
  non-claim eligibility, and prove invariance to non-probe Stage-3 attributes.
- [x] Implement the diagnostic without changing the default representative
  materialization or any quality tolerance.
- [x] Run the canonical TranSplat/Re10K sample under strict diagnostic
  execution in `outputs/ae_failures/diagnostics/transplat-re10k-saes-probe-spread-v1/`.
- [x] Reject promotion: L0/L1/Full was 99.988%/0.012%/0.0%, while SAES-only
  PSNR fell by 7.9188 dB and combined PSNR fell by 7.9286 dB. The result has
  no fallback stages, validates structurally, and remains non-claim evidence.

### SAES L1 Lightweight-Path Audit

- [x] Implement deterministic 2K(T) L1 anchors while retaining K(T) for L0;
  T=4 now uses 8/16 versus 4/16 Gaussians, with exact event counts.
- [x] Add tests for anchor uniqueness, L0-prefix preservation, L1 zeroed count,
  and L1 effective Gaussian count.
- [x] Run the canonical raw-vector/absolute-depth first-hit diagnostic under
  `outputs/ae_failures/diagnostics/transplat-re10k-l1-lightweight-v1/`.
- [x] Reject promotion: L0/L1/Full is 3.796%/86.267%/9.937%, final PSNR changes
  by -2.6195 dB, and LPIPS changes by +0.11396 despite the lighter L1 path.

### Historical FSDR Depth-Guard Audit

- [x] Isolate the commit-`adc7092` 5% local depth guard behind a non-claiming
  guidance policy; claim and Functional modes reject it.
- [x] Add tests proving locally inconsistent hits fall back to full search and
  the default paper/RTL policy still narrows every Hamming hit.
- [x] Run canonical TranSplat/Re10K with SAES disabled and preserve the result
  under `outputs/ae_failures/diagnostics/transplat-re10k-fsdr-depth-guard-v1/`.

### Three-Badge Faithful Engineering Closure

- [x] Define the calibration/evaluation isolation and one-global-config rule.
- [x] Document reviewer/full evidence profiles and mandatory independent Orin
  execution without promoting current missing evidence.
- [x] Add target-isolation, calibration compiler, reviewer selection and schema
  v2.1 failure tests.
- [x] Implement the SHA256-bound mechanism configuration and public
  `calibrate`, `pilot`, and `--profile` interfaces.
- [x] Bind FSDR to authentic cost-volume features and implement the disclosed
  local depth-validity check in software, events, and RTL.
- [x] Implement camera-aware L0 aggregation and probe-constrained `2K(T)` L1
  without reading non-probe full Stage-3 outputs; strict synthetic quick,
  calibration replay, and sensitivity replay all record the C2W-ray path.
- [ ] Pass synthetic properties, six one-sample pilots, and the fixed 32-scene
  gate before launching reviewer/full matrices.
- [ ] Complete independent Orin, routed ASAP7, report catalog, release and DOI
  validation gates.
- [x] Reject promotion: Top-1 and quality pass, but Guided Rate is 53.16% rather
  than 72.1%, and the guard is absent from the submitted RTL/method contract.

## Latest Local Verification

Verified on 2026-07-16 and 2026-07-17:

- Base `pytest -q`: 244 passed and 14 skipped in the dependency-light CPU
  schema environment.
- Locked classic profile: all 304 tests pass. Both classic and DepthSplat
  environment checkers pass with CUDA 12.1 on the declared compiler path.
- Strict quick passed on the RTX 3060 with the pinned classic profile. It loaded
  MVSplat once, selected all three declared target views, emitted positive
  feature/depth/Gaussian/GGU cycles, and produced a validator-PASS aggregate.
- The exact FSDR pilot completed all six accessible pairs. Exact Top-1 Coverage
  was 96.694/99.244%, 96.129/98.789%, and 96.706/100.000% for
  TranSplat, MVSplat, and DepthSplat on Re10K/ACID respectively. All six Guided
  Rates failed, and no row is promoted because the RTL projection ROM is empty.
- The corrected 32-sample prefix covers 262,144 pixels per pair and every row
  still fails. Directly hashing DepthSplat's authentic 1,024-channel ViT-L mono
  tensor also fails at 99.976% Guided Rate and 68.791% Top-1 Coverage, so the
  model-specific DINO hypothesis is not promoted.
- The clean probe-vector first-hit audit at commit `1aec94c` records
  L0/L1/Full=3.796%/86.267%/9.937% and quality deltas of -3.1708 dB PSNR,
  -0.103587 SSIM, and +0.180455 LPIPS. Its result and run log validate but are
  explicitly non-claim evidence; the interpretation is not promoted.
- The Gaussian-head feature audit at clean commit `5cee926` captured the real
  `[1,2,163,256,256]` head input but still produced
  L0/L1/Full=29.2%/0.0%/70.8%. SAES-only deltas were -0.8090 dB PSNR,
  -0.027802 SSIM, and +0.066505 LPIPS. The complete v3 result validates and is
  explicitly non-claim evidence; six-pair quality execution remains stopped.
- A fresh post-fix strict quick run also passed under `outputs/ae_regression`.
- The strict `probe-spread-diagnostic` TranSplat/Re10K run validates without a
  fallback, but its 75.0% pruning destroys quality (SAES-only PSNR -7.9188 dB);
  it is preserved as non-claim failure evidence and does not reopen the matrix.
- Release clean-room testing found that the source bundle omitted the synthetic
  quick dataset because all of `datasets/` was excluded. The archive now
  includes only the four tracked `datasets/quick-re10k` fixture files, which
  remain explicitly ineligible for paper results; the archive regression and
  repository strict quick tests pass.
- The first strict quick attempt exposed batched LPIPS aggregation drift. The
  result builder now derives every aggregate metric from the recorded per-view
  values; the regression test and repeated strict run pass.
- DepthSplat installation initially inherited GCC 13, which nvcc 12.1 rejects.
  The installer now selects the newest installed compatible GCC/G++ pair
  without unsafe compiler flags. The pinned rasterizer built with GCC 12, and
  PEP 440-normalized lock comparison accepts equivalent versions such as
  `plyfile` 1.1 and 1.1.0.
- TranSplat now declares the exact Depth Anything V2 Base asset required by its
  code. The public CC-BY-NC-4.0 file has size 389,961,218 bytes and SHA256
  `0d2b7002e62d39d655571c371333340bd88f67ab95050c03591555aa05645328`.
- The official DepthSplat Re10K checkpoint is the documented large ViT-L
  variant. Its Hydra overrides are fixed in the experiment contract; DINOv2 is
  constructed from the pinned local source with `pretrained=false`, and the
  complete checkpoint supplies the weights without runtime network access.
- The DepthSplat hardware path now implements the checkpoint's two-scale MV
  and mono feature pyramids, uses scale-specific 128/64-channel regressors, and
  traces S3 cycles from the executed feature upsampler, regressor, and Gaussian
  head. Re10K and ACID strict pilots both pass without fallback.
- Formal RTL evidence under `outputs/ae_v2_lsh_rtl_pilot/rtl` passes nine Chisel tests,
  SystemVerilog emission, Verilator lint, and VCD capture. Formal DRAM evidence
  under `outputs/ae/dram` passes the pinned LPDDR5 trace-to-energy chain. The
  DRAM public runner now discovers the repository-local pinned tool installs by
  default instead of requiring undocumented environment variables.
- The first formal quality attempt correctly rejected sample 1 when a concurrent
  RTL emitter temporarily changed the worktree identity. No mismatched sample
  was accepted; the clean rerun resumes from canonical sample boundaries.
- The second formal quality attempt exposed that source-file order is not the
  upstream dataloader's chunk traversal order. The runner now keeps source-file
  order for the stable selection hash and executes in sorted chunk order, with
  both ordinals retained in every sample's provenance.
- `cd chisel && sbt test`: all 9 tests passed, including seed-42 FP16/Q1.24
  LSH software/RTL signature equivalence. The ROM matrix SHA256 is
  `a9d3431fca57f8408237281c60f3bf3fbe572289abd0e6be3b05824afc428397`.
- The latest strict quick aggregate records exact FSDR Top-1 `7602/7794`,
  depth evaluations `300352/1048576`, feature traffic
  `76890112/268435456` bytes, and SAES S2 evaluations
  `376576/1048576`. S1/S2/S3 executed-slot utilization is
  90.38%/86.85%/84.51% on the synthetic Functional fixture.
- The same-source six-pair real mechanism pilot passes every v2 structural
  validator but fails the paper-result gates. Guided Rate is 86.51%--94.96%,
  all L1 rates are zero, and the measured combined quality deltas rule out a
  full-protocol launch from this implementation.
- Worst-case retention now reads the v2 ablation schema, survives the
  result-write/resume crash window, and binds selected source images and
  result JSON by SHA256.
- Resume provenance includes a source-tree SHA256 over HEAD, submodules,
  tracked diff, and untracked source. A changed-source rerun correctly produced
  `executed=1, resumed=0` instead of accepting stale pilot evidence.
- Python compilation, shell syntax, Markdown style, and `git diff --check`
  passed.
- Claim runs reject disabled feature/depth/GGU/SAES/FSDR stages and abort on
  simulator exceptions instead of recording a hidden GPU fallback.
- DRAM evidence reports 8 requests, 121 memory cycles, 55.17 average read
  latency cycles, and 7.199e-9 J from the labeled LPDDR5 smoke proxy.
- iFlow clean-worktree dry-run passes with a portable manifest bound to commit
  `04b4d98`, immutable container ID, repo digest, and ASAP7 collateral hashes.
- iFlow execution now uses the immutable image ID resolved by preflight, and
  the resource guard inspects full Vivado worker command lines.
- Historical release snapshot: all six public checkpoints and all three
  required runtime assets passed their pinned SHA256 checks. Dataset terms are
  recorded without redistribution permission. The legacy ACID identifiers in
  this historical paragraph do not bind the active full-training calibration
  archive; its current binding appears in the Third-Badge Closure checklist.
  Re10K archive `ce351771c966fb25ef41efc561a313ef40607c9aa8ea904ed8d582b361408097`
  and prepared tree `2866634245989caa455fb46e5991d4e4b51e643c3f3e024e8793ff2b664fadd4`
  passed complete validation. ACID archive
  `c9f0685175bda403991493332b5cc10785678a51fbd5c497401d92d176a1a4ac`
  and prepared tree `0e21d05f448675e160529881d583701a2d03f19dbaae1af7941f37137688c187`
  also passed. Paper-result reproduction remains outside the claim. Pre-release
  clean-room validation is complete; the DOI-bound rebuild and account-side
  publication remain pending.
- The corrected ASAP7 RC flow is `NOT_CLAIMED_RESOURCE_LIMIT`: a clean
  unchanged low-memory attempt completed synthesis through filler and reached
  global routing, but 15 seconds of routing caused 172,774 swap-in pages with
  less than 2 GiB available memory. The hashed attempt outcome, guard records,
  and successful clean dry-run are retained under the physical evidence output.
- Source-archive runs verify every file against `release-manifest.json` and no
  longer require Git metadata after Zenodo extraction.
- The paper builds successfully as a 15-page PDF. The Artifact Appendix occupies
  portions of pages 13--14, remains within the two-page limit, and has no LaTeX
  errors, undefined references, undefined citations, or status-name overflows.
- Staged reference payloads live under the ignored
  `artifact/reference_results/evidence/` tree and enter the evidence archive,
  not Git history. The compact manifest remains tracked.

### DL3DV Repair-First Recovery

- [x] Stop and verify absence of non-DL3DV dataset downloads.
- [x] Preserve strict diagnostic baselines for one official DL3DV sample from
  TranSplat, MVSplat, and DepthSplat in non-overwriting output directories.
- [x] Identify the non-degenerate routing contradiction: normalized probe
  variance saturates L0, while raw probe variance leaves unacceptable quality.
- [x] Add upstream cost-volume feature-source/equality tests for all three
  models, including every active DepthSplat scale. The latest contracts capture
  each live inner matcher input at DL3DV sample 0: TranSplat and MVSplat are at
  `outputs/ae_dl3dv_feature_contract/*_sample0_v3/`, and both DepthSplat
  cost-volume scales are at `depthsplat_sample0_v2/`.
- [x] Diagnose and repair DepthSplat ASIC-no-opt/GGU numerical inequivalence;
  the fresh DL3DV no-opt output equals the GPU baseline on all aggregate
  quality metrics.
- [x] Preserve the fixed normalized-probe-standard-deviation notation audit:
  TranSplat obtains a non-degenerate 12.9%/34.3%/52.8% L0/L1/Full split but
  fails with -13.4084 dB SAES-only PSNR. It is non-claim diagnostic evidence,
  not a threshold change, and it does not justify expanding to the other
  models or full DL3DV.
- [x] Repair probe-only C2W materialization without evaluation targets or
  expected-result values: virtual L1 anchor depths no longer read non-probe
  S2 outputs, and probe sub-pixel ray residuals are preserved. Target-free
  geometry, PSD, SH, transmittance, and no-read properties pass (58 tests).
- [x] Rerun fixed TranSplat DL3DV sample 0 in
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_geometry_v1/`; it
  remains a non-claim failure (-13.3514 dB SAES-only), so no wider run started.
- [x] Add and run a target-free full-Stage-3 L1 oracle on the same fixed sample.
  It preserves probe-derived assignment depths and shows virtual L1 anchors
  have higher covariance, harmonic, and opacity discrepancy than the charged
  primary anchors; evidence is in
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_l1_anchor_kinds_v1/`.
- [x] Declare and test the 2K-native-anchor L1 engineering detail. The primary
  probes alone route L1; the deterministic extra K anchors are then executed,
  event-counted, and never selected from target or evaluation data. Its
  non-claim audit/quality outputs remain preserved.
- [x] Run fixed covariance/opacity/component attribution on that same sample.
  No fixed pair or component restoration passes; the result is diagnostic only.
- [x] Identify and audit the optical-depth conservation defect without target
  RGB: L0/L1 median relative error falls from 74.76%/50.03% to 4.94%/1.42%
  under the fixed `transmittance-diagnostic` path. Its one permitted quality
  retry improves combined PSNR by 2.6511 dB but still fails at -7.9491 dB;
  preserve both new output trees and do not tune opacity or thresholds.
- [ ] Run the target-free direct-attribute audit for the new
  `virtual-reconstruction-diagnostic` path. It must prove selected-anchor-only
  C2W/PSD/SH/alpha reconstruction and explicitly charge retained output
  descriptors before any fresh quality retry.
- [x] Run the virtual-reconstruction audit and one fixed quality retry. It
  retains all descriptors and improves combined PSNR to 31.9587 dB, but remains
  -2.8794 dB from no-opt; preserve the output and do not promote it as a sparse
  or hardware result.
- [ ] Audit the paper-compatible native representative merge (K L0 / 2K L1)
  for remaining moment, opacity/transmittance, and C2W-geometry defects;
  do not reconstruct or retain skipped descriptors, alter the three-level
  router, or add evaluation-dependent parameters before any quality retry.
- [x] Reject the full pseudo-mixture covariance candidate: its synthetic
  second-moment property passed, but the fixed target-free DL3DV attribute
  audit worsened L0/L1 full-oracle covariance p50 to 0.6317/0.2638. Preserve
  `transplat_sample0_second_moment_audit_v2/`; do not run a quality retry.
- [x] Reject the depth-scaled covariance candidate: its adapter-derived
  property passed, but the fixed target-free DL3DV audit had no material,
  consistently positive oracle change. Preserve
  `transplat_sample0_depth_scaled_covariance_audit_v1/`; do not run a quality
  retry.
- [x] Reject the corner-depth geometry candidate: its post-hoc full-S2 oracle
  improved, but the one unchanged TranSplat/DL3DV quality retry reduced
  combined PSNR (24.2378 -> 24.2147 dB) and increased LPIPS
  (0.25121 -> 0.25221). Preserve the audit and quality trees; restore
  assignment-weighted probe depth in the runtime path.
- [x] Reject the L1 primary-probe merge interpretation: preserving the extra
  K native anchors improved only tile-mixture covariance p50
  (0.129934 -> 0.107015), while mean, SH, and average opacity all worsened.
  Preserve `transplat_sample0_l1_primary_merge_audit_v2/` and
  `transplat_sample0_l1_native_global_oracle_v1/`; remove the candidate and
  do not run quality.
- [x] Classify `transplat_sample0_s3_before_s4_audit_v1/` correctly as a
  target-free raw-S3 *per-skipped-position interpolation oracle*. It stops
  before GGU/rendering, removes target RGB before device transfer, and maps S1
  64x64 -> 256x256 through the existing bilinear route. Its worsening p95
  depth, scale, rotation, SH, and opacity errors reject only naïve raw-descriptor
  interpolation. It neither executes a retained-anchor moment merge nor emits
  retained output, so it cannot reject the distinct post-GGU primitive moment
  path or authorize a pre-GGU quality retry.
- [x] Reject the L1 primary-depth-reference diagnostic at
  `transplat_sample0_l1_primary_depth_reference_audit_v1/`. It correctly
  derives L1 reliability normalization from the K routing probes while
  retaining and charging the existing 2K anchors. L1 covariance and mean
  improve, but harmonic and opacity aggregate errors worsen; the all-attribute
  gate fails, so preserve the non-claim implementation/evidence and do not
  run a quality retry.
- [x] Make that paper-required K-probe reference the default L1 semantics while
  retaining the declared 2K anchors for execution and aggregation. The fresh
  target-free audit at
  `transplat_sample0_l1_primary_reference_corrected_audit_v1/` (SHA256
  `c93a6bc94e7e20098333a438f0256574c5946745d74fc0c4cf9a0fae26aefea1`)
  passes target isolation, PSD, event, and skipped-descriptor poison checks
  with nonzero sparse work. It does not satisfy the earlier all-attribute gate
  and therefore does not authorize an RGB quality rerun.
- [x] Preserve the target-free routing-coordinate audit: metric-depth remains
  the literal default; relative and candidate-coordinate variants are rejected
  because they route 92.8% and 99.98% of this fixed sample to L1 and would fit
  a paper aggregate rather than a general mechanism.
- [x] Add a deterministic SAES control/merge/storage ledger and make every
  nonzero SAES ablation saving require it. The ledger is recorded as analytic,
  no-overlap, and explicitly non-RTL-cycle-equivalent.
- [x] Run the fixed TranSplat/DL3DV target-free hardware audit at
  `transplat_sample0_saes_hardware_accounting_v3/`: it removes target RGB,
  avoids baseline/render/quality work, and records 4,293,120 charged analytic
  cycles and 45,883,392 bytes for the all-L0 trace. Preserve the v2 local
  counter-initialization failure separately; do not overwrite it.
- [x] Bind SAES S2/S3 accounting to a per-model target-free execution-dependency
  contract. TranSplat's fixed DL3DV sample-0 audit detects a dense dependency
  at every retained raw-head value (result SHA256
  `bcbde72ec4029d3b32465ff920df8746a8f636e4383d19f161e5fa906b80f621`);
  MVSplat and DepthSplat have no positive sparse-execution evidence. Therefore
  every current SAES S2/S3 saving is zero, while the analytic control/merge/
  storage ledger remains recorded. Schema-v2.1 result validation re-resolves
  the model contract and rejects missing, substituted, or nonzero-unverified
  S2/S3 savings. This does not authorize a quality retry or the 140-scene
  protocol.
- [x] Expose and regression-test `SAESController.decisionCycles` in Chisel:
  L0 is 2 cycles and L1/Full are 3 cycles from accepted start; export the
  valid-on-done signal at `ScarfTop` for later VCD reconciliation. This covers
  only routing control, not the missing merge/storage datapath.
- [x] Remove the unsound RTL `ProbeOnly` bypass: L0/L1 classifications now
  conservatively execute the baseline S2 and S3 controller sequence until a
  real probe/assignment/moment/buffer datapath exists. New Chisel regressions
  trace both sparse levels through CostVol, U-Net, depth regression, S3 refine,
  and Gaussian head before GGU; this intentionally claims no SAES hardware
  saving.
- [x] Implement and emit the standalone `SAESDescriptorBuffer`: ordered 128-bit
  writes retain only completed selected descriptors, reject malformed streams,
  and expose the ledger-derived 6/12-beat layouts for SH degree 2/4. It is not
  yet integrated into `ScarfTop`, so it carries neither PPA nor timing evidence.
- [x] Implement and emit the standalone `SAESScalarMomentAccumulator`: its
  target-free Python reference and Chisel test agree on exact integer
  first/second-moment and constant-descriptor vectors. It is not yet replicated
  across a packed Gaussian descriptor or driven by the bilateral assignment.
- [x] Implement and emit the standalone `SAESAssignmentNormalizer`: it maps
  precomputed positive bilateral scores to an exactly normalized Q0.16 mass
  with a deterministic residual anchor. Its Python and Chisel vectors cover
  residual assignment, ties, and zero-score rejection; it has no route/target
  input and remains unconnected to a score producer or tile pipeline.
- [ ] Implement SAES assignment/moment matching and retained-descriptor
  buffering in the tile pipeline, then prove per-path event/cycle agreement
  before treating the analytic ledger as hardware timing or using it for a
  paper speedup.
- [x] Run the fixed target-free TranSplat/DL3DV `refine_unet` locality audit:
  all nine source tiles in the predeclared 3x3 raster changed all 16,384
  retained-probe spatial positions across the complete 64x64 tile grid. The
  result SHA256 is
  `bb5c24557a39da455869ebdac8804cf3a9c883f9c0df1592297c051ce2583a56`.
  This rejects a finite-halo direct bypass for the current dense path; it
  cannot enable savings or a quality/full-protocol retry.
- [x] Reconcile the DepthSplat DL3DV FSDR feature with the 16x128 RTL ROM:
  `depthsplat_sample0_v3/feature_contract.json` (SHA256
  `d519d0a7a7bbc966c9ae4333c42e5ab4ca987f3cd813ae9083feae2fac05def8`)
  proves the first actual cost-volume input is 128 channels. Remove the
  non-cost-volume 1024-channel mono selector from `--fsdr-only`, so a claim
  record cannot use a software-only hash. This does not resolve Guided Rate.
- [x] `dl3dv-native-merge-conditional-v1` Slice A: selected-anchor-only,
  one-hot, constant-descriptor, PSD, and no-cross-anchor-leakage properties
  pass for the diagnostic-only conditional transport path.
- [x] `dl3dv-native-merge-conditional-v1` Slice B: exactly one target-free
  TranSplat/DL3DV audit compared the same all-L0 tiles under the current route.
  It is rejected: tile-mixture covariance p50/p95 worsen from
  0.235292/0.754145 to 0.348608/0.891179. Preserve the matched baseline
  `transplat_sample0_current_native_merge_audit_v1/` (SHA256
  `a3b1d19b0312fb493161a2f868220782744b19bcaa21e423c485792ca3cddbf5`) and
  candidate `transplat_sample0_conditional_anchor_transport_audit_v1/`
  (SHA256 `30876f42d1c9d40a534417247c955ee1e418e45c636952c48a77357612a0c05a`).
  Do not render quality.
- [x] `dl3dv-native-merge-conditional-v1` Slice C: correctly not run because
  Slice B failed its strict all-attribute gate; no quality exception is allowed.
- [x] Add a strict diagnostic-only FSDR path for DL3DV sample gates. It removes
  target RGB before device transfer, records that it was not passed to the
  model or routing, and emits `fsdr_target_free_audit`, which cannot be
  aggregated as claim evidence. Fixed sample-0 runs passed schema validation
  for TranSplat, MVSplat, and DepthSplat at
  `outputs/ae_dl3dv_fsdr_audit/`; their Guided Rates are 64.1235%, 58.4717%,
  and 43.6942%, with discrete Top-1 coverage 99.7716%, 99.8539%, and 99.9361%.
  These are context-only execution checks, not quality or Table 2 results.
- [x] Run `dl3dv-s2s3-direct-dependency-v1` at fixed DL3DV sample 0 for all
  three models. The target-free raw-head gate detects retained-probe changes
  for TranSplat (`2,752,512/2,752,512`), MVSplat (`2,752,511/2,752,512`), and
  DepthSplat (`2,121,728/2,121,728`), so all three direct sparse-bypass routes
  are closed. The results are preserved in new audit directories and do not
  authorize a quality retry or DL3DV expansion.
- [x] Move the declared K(T)/2K(T) SAES layout into a pure standard-library
  helper. Event accounting now executes under an explicit no-Torch import
  guard, while the Torch `ProgressiveSAES` wrappers preserve fixed layouts and
  pass their classic regression subset. This is CPU clean-room plumbing only;
  it does not verify sparse S2/S3 execution or permit nonzero SAES savings.
- [x] Add the staged T=4 retained-output scheduler and a pure software
  reference for the exact L0 `[0,3,12,15]` and L1
  `[0,3,12,15,5,10,1,2]` request order. It advances only when a real upstream
  descriptor is confirmed and is intentionally not connected to a dummy
  `ScarfTop` source; sparse execution, timing, and savings remain unclaimed.
- [x] Complete the target-free fixed locality gate for all models. MVSplat has
  the same whole-grid retained-probe dependency as TranSplat; DepthSplat has a
  one-tile envelope, but its fixed four-convolution S3 adaptor footprint covers
  every predecessor position after one backward 3x3 expansion. Preserve both
  records and do not treat final-head-only local emission as sparse S2/S3 work.
- [x] Reject direct sparse-producer integration for the unmodified upstream
  checkpoints. A future reopen requires an author-provided probe-first adaptor
  or exact sparse-compatible training configuration/checkpoint; a dummy
  bypass, dense activation masking, final-head-only emission, or a newly
  trained surrogate cannot support the submitted SAES reproduction claim.
- [ ] Rerun the three strict one-sample DL3DV diagnostics in fresh directories.
- [ ] Run the official 140-scene DL3DV protocol only after those gates pass.
- [x] Implement same-weight selected-output replay for the TranSplat/MVSplat
  two-convolution Gaussian head and prove it on clean DL3DV sample 0 without
  target RGB or rendering. The first-convolution closure is dense; only the
  final convolution is selected-output work, so global S2/S3 savings remain
  zero.
- [x] Run the one predeclared TranSplat/DL3DV quality gate for conditional
  optical-density materialization. It fails the unchanged quality tolerances
  by a wide margin (`34.8381 -> 7.8144` dB PSNR); preserve
  `transplat_sample0_conditional_optical_mass_quality_v1/` and do not expand
  the result.
- [x] Add the no-parameter covariance range Full fallback and verify on clean
  target-free DL3DV sample 0 that it routes all 8,192 unsafe L0 tiles to Full,
  with no skipped S2/S3 work. This is a correct fail-closed outcome, not a
  sparse SAES reproduction result.
- [ ] Do not launch 8/32/140 DL3DV SAES quality runs until a target-free,
  nonzero sparse candidate passes the same sample-0 gate.
- [x] Run the coordinate-explicit target-free TranSplat/DL3DV sample-0 gate:
  normalized probe-vector standard deviation for L0 and inverse-depth candidate
  coordinate standard deviation for L1, with the paper's unchanged 0.20/0.10
  thresholds. It failed closed: all 8,192 attempted tiles returned to Full and
  skipped zero descriptors. Preserve `transplat_sample0_coordinate_explicit_
  optical_mass_audit_v1/` (SHA256 `fca6d6237ef1839358c2ba11475caa50fe530b5c39de8eb9b734588ea1860936`);
  do not run quality.
- [ ] Re-run one target-free TranSplat/DL3DV sample-0 audit after removing the
  unsupported covariance-determinant envelope. Use the fixed normalized probe
  feature statistic and paper-literal metric probe-depth standard deviation;
  require nonzero sparse work and all retained existing numerical safeguards
  before pre-registering any quality run.
- [x] Pass that target-free gate in
  `transplat_sample0_metric_depth_optical_mass_audit_v1/` (SHA256
  `27ece34902cada6d20a745295dcf6f8c82bcdeff60b8e3265b7cff68fb417183`):
  35,184 descriptors are skipped with no PSD, mass, or poison-invariance
  failure.
- [x] Run exactly one pre-registered TranSplat/DL3DV sample-0 quality gate in
  `transplat_sample0_metric_depth_optical_mass_quality_v1/` (SHA256
  `88a78ec60326b008cbd8a8f8ccd1ae71babca78ebf2ca2193fe5660e3e8559c2`).
  It fails: PSNR `34.8381 -> 10.0265`, SSIM `0.97360 -> 0.44955`, and LPIPS
  `0.03276 -> 0.51326`; no 8/32/140-scene expansion is authorized.
- [ ] Diagnose retained-anchor attribute distortion on the same sample with a
  target-free audit before considering one new materialization correction.
- [x] Diagnose retained anchors in
  `transplat_sample0_metric_depth_optical_mass_attribute_profile_v1/` (SHA256
  `5f06fc833de6b5d9c786a48626d7f1510bf26a828e958439d65a07d222a65d04`):
  3D covariance-volume mass drives output opacity p50 to `0.00403` from source
  p50 `0.29721`, while SH is unchanged.
- [ ] Implement and test context-camera projected optical-footprint mass before
  any further quality run. It must use no target camera/RGB, add no route gate,
  and pass PSD, finite, single-assignment, opacity-domain, and poison checks.
- [x] Run the first projected-footprint target-free audit. Preserve
  `transplat_sample0_metric_depth_projected_optical_mass_audit_v1/` (SHA256
  `4f23429a8b3917a73aa71a5c85934be89ac295d40be0df2e1bf2198ccc9aa1b4`);
  it exposes an overly large numerical 2D determinant floor, so quality is not
  authorized.
- [ ] Lower the projected-footprint validity floor to dtype-minimum positive
  scale, rerun the target-free profile, and require a nontrivial sparse route
  before another quality gate.
- [x] Correct the projected-footprint floor and run v2 (SHA256
  `b6c6d4949ee5ff6d735caed7ffe153dd4bc76070449868bef7e7ec96bfdeb988`).
  It passes numerical checks but retains opacity collapse, so no quality gate
  is authorized for projected optical mass.
- [ ] Run a target-free conditional-anchor-transport attribute audit with the
  fixed normalized feature and metric-depth route; verify range-constrained
  opacity before considering one fresh quality gate.
- [x] Pass that target-free conditional-anchor audit (SHA256
  `900c8e968076dee17f16ed2d6ef0540e454a98da1c1266b474670c0020e910f0`):
  opacity remains exactly range-constrained, while covariance contraction is
  recorded as the remaining quality risk.
- [ ] Run exactly one pre-registered conditional-anchor sample-0 quality gate
  in `transplat_sample0_metric_depth_conditional_anchor_quality_v1/`; do not
  expand DL3DV unless unchanged PSNR/SSIM/LPIPS tolerances all pass.
- [x] Run that pre-registered gate (SHA256
  `bde1768c7edd077ee000cab60f632c121928182564237d8788ea0acb7abfd9b1`):
  PSNR `34.8381 -> 24.2559`, SSIM `0.97360 -> 0.83307`, LPIPS
  `0.03276 -> 0.24604`; no DL3DV expansion is authorized.
- [ ] Run a local target-free conditional covariance-moment diagnosis before
  considering another materialization correction or quality gate.
- [x] Run the covariance diagnosis v3 (SHA256
  `8545e6479c7cd496ceaa8ba4aa395b27e16ea055ef32cd02e52dcad7ab818ed1`):
  covariance moments expand as required; no local arithmetic repair remains.
- [x] Stop DL3DV quality expansion: the fixed route's 34.314% L1 prevalence
  conflicts with the paper's 10.1% DL3DV aggregate and cannot be changed from
  evaluation data. Preserve Results Reproduced as unclaimed.
- [x] Implement the diagnostic-only adapter-faithful subpixel-offset transport:
  retain only anchor mean/depth/context C2W/intrinsics, apply the recovered
  bounded offset before target-ray normalization, and fail-close incompatible
  tiles to Full. Synthetic adapter-equivalence, poison, PSD/SH/opacity, and
  event-ledger tests pass.
- [x] Run exactly one target-free TranSplat/DL3DV sample-0 adapter-offset
  attribute audit in
  `transplat_sample0_adapter_offset_transport_audit_v1/`; do not render or
  compare quality until that new output passes every target-free check.
- [x] Pass the fixed adapter-offset attribute audit (SHA256
  `9fd33bf2a0e1ab4abeacd347d0f5478eac296a8a25d3a0cce7e8d53e9352fa17`):
  26,496 descriptors are skipped with no adapter fallback, no skipped-S3
  reads, and identical poison-pass events. This is target-free provenance, not
  a quality claim.
- [x] Run exactly one fixed adapter-offset selected-output quality pilot in
  `transplat_sample0_adapter_offset_transport_quality_v1/` with the unchanged
  `0.15/0.005/0.005` limits; no further DL3DV expansion is authorized on any
  failure. Preserve SHA256
  `e1e21de6f3d5f5509e676415ecf7314743af1b3aae453d0f17036cc973c858b1`:
  PSNR loss `9.1464` dB, SSIM loss `0.12203`, and LPIPS increase `0.18550`.
  This candidate fails and must not be retried.
- [x] Implement the separate adapter-offset SH/opacity transport diagnostic
  without changing its geometry, covariance, router, thresholds, L1 primary-K
  reference, or 2K anchor layout. Synthetic constant/range/poison/PSD and
  analytic-accounting tests pass.
- [x] Pass the fixed target-free attribute-transport audit in
  transplat_sample0_l1_primary_reference_attribute_transport_target_free_attribute_audit_v2/
  (SHA256 d4ed058d8227be3e371919d7edc32585eaa11b5472084e423041e9d5db7fc4bc):
  26,496 skipped descriptors, zero target RGB and skipped-S3 reads, identical
  poison events, nonzero SH/opacity updates, and a charged 175,488-pair
  analytic attribute-reconstruction ledger. This is not a quality or saving
  claim.
- [x] Run exactly one pre-registered attribute-transport selected-output
  quality gate in transplat_sample0_adapter_offset_attribute_transport_quality_v1/
  through scripts/saes_adapter_offset_attribute_transport_quality_gate.py.
  Preserve SHA256 ef7535183d8c794b5a8458c0325363501b312f93b277912accaed09b2cd37308:
  clean target-RGB-after-mask provenance, equal selected-head route and
  retained attributes, but non-bit-identical decoder output plus PSNR loss
  9.1865 dB, SSIM loss 0.12356, and LPIPS increase 0.18844. The fixed
  candidate fails; no retry or DL3DV expansion is allowed.
- [ ] Diagnose the remaining materialization failure target-free before
  registering any new quality gate. The diagnosis must not read target RGB,
  select a new sample, change thresholds, or infer a quality tolerance.
- [x] Run the fixed guard-partition oracle audit with canonical guarded and
  unguarded shadow clones. Preserve
  `transplat_sample0_l1_primary_reference_guard_partition_audit_v1/`
  (results SHA256 `dfb94a26923345ab8de0fb1d112a365160a40fc0816b47001f5286d628aa9dde`,
  source `42edcce`): all 8,192 scalar trace records agree with runtime route
  and guard counters; poison leaves the mask, trace, statistics, and retained
  attributes identical; no target RGB, skipped-S3 read, decoder, renderer, or
  metric is used. The canonical result has 2,932 guard-accepted, 937
  rejected-to-Full, 4,323 noncandidate-Full, and zero-by-construction
  L0-rejected/L1-accepted tiles. The shadow keeps those labels, but dominant
  covariance/coverage error remains the same order as accepted tiles; reject
  guard-threshold repair and do not retry quality.
- [x] Correct and pass the synthetic-only assignment-consensus
  adapter-pseudo-descriptor geometry gate. The original `43dfd37` synthetic
  pass is retained as invalidated history: its atomic rollback snapshot read
  skipped raw S3 descriptors. `92dec5c` defers every virtual-output write until
  all primitive slots validate; `968563d` preserves valid no-op tiles where
  every position is an anchor. Fixed routing, thresholds, primary-K/2K anchors,
  direct virtual skipped outputs, no assignment squaring, constant/one-hot/
  PSD/C2W, L0/L1 selected/skipped poison, route/event, default-guard selected
  read, and atomic Full-fallback tests now pass. The full SAES collection is
  `164` passed; the path has zero compression, separately charged successful
  and aborted geometry/traffic work, is non-paper-eligible, and does not
  authorize a quality run.
- [x] Defer the zero-compression virtual-output audit. Its synthetic result is
  preserved, but it is not the capacity follow-up and cannot authorize a
  quality retry or DL3DV expansion.
- [x] Add a fail-closed DL3DV training-calibration archive path: enumerate the
  authenticated official tree, freeze an evaluation-disjoint 24+8 plan before
  download, verify selected ZIP byte counts and upstream object ids, record
  actual SHA256 values, and reject unsafe archive members or a changed source
  tree.
- [x] Add a DL3DV target-free calibration compiler: build native and
  Re10K-compatible sidecars for disjoint 24-scene training and eight-scene
  holdout sets, retain all camera geometry, omit target RGB, and bind
  plan/preparation/tree hashes.
- [x] Teach the calibration grid to route TranSplat/MVSplat to the Re10K
  sidecar and DepthSplat to the native sidecar; legacy Re10K/ACID calibration
  remains Functional-only.
- [x] Re-run the target-free TranSplat/DL3DV materialization guard gate after
  fixing its stats reset. Preserve v1 as invalid because it serialized an
  active guard as disabled; v2 records `true`, 1,058 L0 plus 2,878 L1 checks,
  zero non-probe S3 reads, and a ledger-consistent charged guard path. This is
  diagnostic provenance only and does not grant S2/S3 savings.
- [ ] Obtain official access to `DL3DV/DL3DV-ALL-480P`, write the source-bound
  archive plan, then download and verify its 24 calibration plus 8 holdout ZIPs.
  The 2026-07-18 tree request succeeds but its first revision-pinned archive
  resolve returns upstream `403 not in the authorized list`; this remains an
  external authorization task, not a downloader or network failure.
- [ ] Do not run DL3DV calibration or the 140-scene evaluation until a real
  paper-compatible SAES sparse path passes the existing synthetic and
  single-sample quality gates with nonzero verified savings.
- [x] `saes-selected-output-quality-pilot-v1`: add batched per-view
  selected-output replay, execute guard-resolved L0/L1/Full masks through the
  native adapter and decoder, charge both replay passes, and run only the fixed
  TranSplat/DL3DV sample-0 quality gate. Preserve the output as a non-claim
  failure: it violates all three fixed quality limits and transferred target
  RGB before mask commitment, so its provenance cannot support a claim.
- [x] `saes-selected-output-quality-pilot-v2`: rerun the fixed v1
  scene/seed/checkpoint/threshold contract with RGB delayed until final-mask
  commitment. Target RGB remained unread, and the selected head matched
  dense-SAes routing and retained attributes, but the native decoder still
  consumed zero-opacity removed descriptors with absent raw-head attributes.
  Preserve the log as a target-free S4 contract failure; it produced no
  quality verdict.
- [x] `saes-selected-output-quality-pilot-v3`: render only SAES-retained
  descriptors from both dense and selected-head controls. It still failed the
  strict target-free decoder-equivalence gate before RGB access; preserve its
  log and do not treat it as a quality result.
- [x] `saes-selected-output-quality-pilot-v4`: record selected-vs-dense
  decoder deltas alongside a repeat dense-control render. Dense repeat is
  bit-stable; selected-vs-dense has mean absolute delta `1.028e-7` and max
  `0.001052`, so the source is selected-head FP32 accumulation rather than
  rasterizer noise. This run did not read RGB or emit quality metrics.
- [x] `saes-selected-output-quality-pilot-v5`: run the unchanged fixed
  four-view quality measurement after output commitment. Preserve results
  SHA256 `d2c51c3e1f93bad4d488f4366f1640f93045683a5e47b5695348561e1fcc4269`:
  clean RGB provenance, equal masks and retained attributes, but nonzero
  selected-head decoder delta plus PSNR loss `9.1452 dB`, SSIM loss `0.12201`,
  and LPIPS increase `0.18548`. It is non-claim failure evidence.
- [x] Block all 8/32/140-scene DL3DV expansion for this mechanism. A next
  candidate must be target-free until it passes synthetic/descriptor gates,
  or use only the official evaluation-disjoint training calibration once
  upstream `DL3DV-ALL-480P` authorization is granted.
- [x] Complete the fixed K/2K render-teacher capacity diagnostic in
  `transplat_sample0_same_budget_render_teacher_oracle_v3_frozen_contract/`
  (results SHA256 `e4c110a087c8f19126f235a88091383e9b85e34d713ffd70a5960ad91e5ae295`,
  source `ce9349c`). It freezes input identity, mask hashes, and
  `131072/9432/4792/116848` dense/removed/representative/Full accounting;
  target RGB is removed before encoding and only dense renderer output teaches
  the 128-step post-hoc optimizer. Final teacher fidelity is `53.5922 dB`,
  `0.998050` SSIM, and `0.006741` LPIPS. This proves bounded-slot capacity,
  not target-free or runtime feasibility, and authorizes no quality retry.
- [x] Implement the context-only tangent-plane covariance diagnostic and pass
  its synthetic and route-invariance gates: projected-moment, camera-order,
  rigid-transform, invalid-camera, PSD/finite, one-hot/constant, skipped
  S2/S3 poison, fixed-mask/event, and Full-passthrough checks all pass. The
  code changes only retained covariance and records
  `runtime_eligible=false`; the full SAES test set passes (`181 passed`). No
  DL3DV descriptor audit, render, metric, or quality retry is implied.
- [x] Pre-register and execute exactly one context-only, target-free multi-view
  representative-moment audit before any new DL3DV quality measurement. Keep
  the fixed checkpoint, seed, router, thresholds, K/2K/Full mask,
  representative count, Full passthrough slots, and event accounting; compare
  the current single-producer merge with a deterministic context-projected /
  tangent-plane first/second-moment construction using only context cameras,
  selected anchors, and existing assignments. Remove target RGB and target
  camera metadata before device transfer; do not render, decode, compute
  quality metrics, inspect expected results, optimize parameters, select a
  sample, or fit a threshold. Dense skipped S3 may be read only after candidate
  output is committed as a post-hoc reference. Require PSD/finite,
  constant/one-hot, single-assignment, camera-equivariance, selected/skipped
  poison, fixed-mask/event, and atomic Full-fallback gates. Whatever its
  outcome remains descriptor-only and cannot itself be reported as quality
  evidence. A pass under its separately frozen directional gate may authorize
  only the pre-registration of one fixed DL3DV sample-0 quality gate; a failed
  or inconclusive audit authorizes neither that gate nor a full DL3DV retry.
  The preserved execution at
  `transplat_sample0_multicontext_tangent_target_free_audit_v1/` is a negative,
  non-authorizing result: aggregate covariance error worsened by `0.02354%`
  instead of improving by `>=20%`, and it recorded `1,158` local fallbacks.
  It did not render or compute quality. Its fixed-input and selected-only-S3
  enforcement was insufficient for a clean gate, so it is retained honestly
  as a diagnostic and the tangent candidate is retired rather than rerun.
- [x] Wire the common runner-invoked S3-access preflight. Selected-only audit
  paths use an allowlisted payload and two finite S2/S3 sentinels, requiring
  equal route trace, counters, committed outputs, and Full outputs. The frozen
  attribution runner invokes that preflight on the fixed attribute-transport path before
  reconstructing the oracle. Its separately declared full-S3 oracle read is
  recorded as a non-runtime `not-applicable` exception with an observed full
  read count, never as selected-only execution. This cannot authorize a
  quality retry, a threshold sweep, or revival of the retired tangent
  candidate.
- [x] Run the zero-output strict-FP32 initial-compact smoke in a fresh CUDA
  process. The 2026-07-18 command exited `0` with no output artifact, target
  RGB, teacher metric, or optimizer. It passed fixed input/checkpoint hashes,
  both selected-only sentinels, exact `131072/9432/4792/116848` partition,
  immutable Full slots, and finite target-view-0 `[3,256,256]` render. Its
  transient terminal log SHA256 is
  `ef3501040e0b5d92420fbcd8c2c9e27a2c9dbf59eadae7423d839071e2686475`;
  the record separately reports the non-runtime 46,896 full-S3 oracle reads.
- [x] Repeat the no-output smoke from the v3 mapping-repair commit
  `95fdc87ca4c37befb14140f177ee887adb00c4aa`. The fresh CUDA process passed
  input/checkpoint hashes, both sentinels, exact partition, Full passthrough,
  and finite `[3,256,256]` view-0 rendering without target RGB, teacher
  metrics, optimizer, or an output directory. Terminal SHA256:
  `18d1f15598613e9ecf8d3170dc6723db113f2e002cdf2e6218a3664253cb0cfe`.
- [x] Preserve the aborted `...parameter_attribution_v2/` record (results
  SHA256 `5c804d5b1445d8e5c5fb474318dddb8081bcd0d9430db14dc2b5dda8d9c29906`,
  terminal SHA256 `c912b34a0f2867d4d44187d879745844d73d6d2bf1e8eb996f517b00343056a8`).
  Its CUDA out-of-bounds occurred because dense global representative IDs were
  used against compact slots; no teacher metric was persisted, recovered, or
  interpreted. The immutable v1 record remains cause-undetermined.
- [x] Make fixed input/checkpoint hashes and the two-finite-sentinel
  selected-only S3 proof a shared mandatory precondition for every later frozen
  selected-output or descriptor audit in this line. The dense teacher remains
  an explicit non-runtime full-S3 exception.
- [x] Complete the sole frozen render-teacher representative-parameter
  technical replay at
  `transplat_sample0_same_budget_render_teacher_parameter_attribution_v3/`.
  Results SHA256:
  `9a69293a8f482f44573b2fe75f312dbce7b59196f2b79f5d2cd82037130679e4`.
  The clean-source v3 run passed all index, partition, sentinel, Full-slot,
  and frozen-teacher controls with all twelve variants, no target RGB and no
  optimizer. Its direct-teacher attribution requires all four representative
  families for full recovery: the best proper subset, all-teacher-minus-SH,
  still loses `5.430664 dB`; all other omissions lose `10.134862`--`14.336210`
  dB. This is capacity evidence only, not a GT quality result.
- [x] Close the frozen teacher-attribution campaign. No untrained
  materialization correction, tangent rerun, or DL3DV quality run is authorized
  by v3. Any later route must be separately registered as evaluation-disjoint
  joint-family calibration or a more conservative Full fallback.
- [x] Register `saes-joint-materialization-calibration-acid-v1` as the only
  active materialization route. The frozen v3 teacher attribution establishes
  same-budget capacity but requires all four representative families, so
  untrained single-family/two-family repairs remain closed. ACID is author-side
  evaluation-disjoint source material only; this registration neither changes
  `artifact/mechanism_config.json` nor authorizes a DL3DV quality run.
- [x] Compile and hash a deterministic ACID 24/8 train/holdout partition from
  the verified 32-scene prepared tree, with target-free sidecars, source and
  checkpoint-binding requirements, evaluation-disjoint proof, teacher/runtime
  isolation, and a holdout-no-training guard. The plan is
  `artifact/protocol/acid_joint_calibration_plan.json` (file SHA256
  `844710cd845ff65382f176ac1b936e770128bf021676a4ccc3ecf5368521143c`,
  embedded `ad8f551652429a11512f80d516a36cac2ebc6537b461def81148af9a4584eeb5`).
  The materialized context-only sidecars validate at outer tree
  `863c6f5b221e6ef5f6141a27d3a2e1370b106433595806848c0e0c1b57e9596a`;
  they are author-side only and do not bind checkpoint hashes until training.
- [x] Implement the default-off coupled 32-to-8-to-40 shared calibrator and
  prove route/K/2K/Full invariance, Full bit identity, selected-only S3
  isolation, PSD/finite covariance, bounded opacity, SH degree masking, asset
  hash binding, and fail-closed budget behavior. The synthetic runtime path
  rejects an unpinned asset, non-representative materialization, a missing
  selected-head trace, and a DepthSplat replicate-padding replay without its
  required source-bound selected-head runtime evidence.
- [x] Charge the calibrator's MACs, weight/activation traffic, selected reads,
  and cycles. Reject a zero-cost ledger, a Full-slot call, a call-count mismatch,
  or a missing/over-5% model-specific selected-head contract. Do not claim
  S2/S3 saving while the execution-dependency contract remains unverified.
  The ledger remains analytic and records no additional S2/S3 savings.
- [x] Freeze the author-side shared-asset training contract at
  `artifact/protocol/acid_joint_materialization_training_contract.json` before
  any cache extraction or optimizer launch. It binds both context-only trees,
  the exact TranSplat/ACID, MVSplat/ACID, and DepthSplat/Re10K checkpoints,
  entrypoint/config-source hashes, the 24-file implementation binding, the
  360x640-to-256x256 upstream LANCZOS crop/intrinsics plus patch-alignment
  shims, and a fixed `T=4`, `tau_f=0.20`, `tau_d=0.10`, cross-check `0.02`,
  representative `K/2K/Full` SAES route. It also freezes the one shared
  12,000-update AdamW objective and all train/holdout teacher-fidelity gates.
  The active embedded contract SHA256 is
  `b3d98c23b842a36b9e0905f913bd601ca9ff27660fe2c7e88fb0c5c6a336dc6a`.
  It supersedes the preliminary, pre-implementation-binding schema whose
  embedded hash was `86c23fe3f275ffcec90fbed9546271a521d3932b2ce8ad84ae69be86e0717aff`.
  This is `FROZEN_PENDING_EXECUTION`, not a successful optimizer run.
- [x] Freeze the candidate-to-promotion ordering: `prepare-train` may write
  only the hash-pinned `calibrator.pt.candidate`; each model/split then needs
  source-bound context-only runtime-control evidence with an actual
  same-weight selected-head replay; only `finalize-train` may validate teacher
  fidelity and atomically promote that candidate to the canonical shared asset
  and train result. The evidence records `selected_head_only=true`,
  `s2_s3_sparse_execution_verified=false`, and
  `global_s2_s3_savings_claimed=false`. At 2026-07-18 22:43 +0800, the
  canonical TranSplat/ACID training-cache compiler was launched after the
  three one-scene smoke caches passed. It writes only per-scene records under
  `teacher_cache/transplat/calibration_train.partial` and cannot be consumed
  by the runner until a complete manifest atomically promotes it. No complete
  cache, candidate asset, train result, holdout result, or runtime-evidence
  record exists yet.
- [x] Require durable train/holdout teacher-fidelity result records with one
  hash- and state-pinned asset. Every model must have non-worsening individual
  mean/covariance/opacity/SH MSE and joint relative MSE at most `0.90` train,
  `0.95` holdout; finite/PSD, route/count, Full, selected-only two-sentinel,
  selected-head, and cost gates are mandatory. The holdout validator rehashes
  the train result and rejects retraining, asset updates, reranking, reshuffle,
  target/evaluation/expected-result access, or a changed resolved config. Dense
  non-probe adaptor attributes are permitted only in the offline teacher after
  the selected-only route/assignments are frozen; they never enter descriptor,
  runtime, or persisted cache inputs.
- [x] Superseded for local diagnostics (2026-07-19): the former ordering that
  blocked every DL3DV quality/fallback check on ACID train/holdout is retained
  as a paper-promotion constraint only. It does not block the user-prioritized
  local DL3DV simulator-fidelity repair. No local diagnostic may be promoted to
  a paper claim until the corresponding claim gates pass.
- [x] Preserve the aborted `...parameter_attribution_v1/` record rather than
  overwriting it. It ended before any direct-teacher metric with a CUDA
  device-side assert; its retained record does not determine the cause. The
  designated v3 technical replay is hash-bound to the target-free source
  sidecar and must pass strict-FP32 exact-partition verification before
  rendering.

### Local DL3DV Simulator Recovery

- [x] Fix the local TranSplat DL3DV sample-0 simulator baseline with SAES/FSDR
  disabled: `34.8380575 dB` versus native GPU `34.8380585 dB`.
- [x] Add the uncertified-deletion Full fallback for the representative SAES
  simulator route and cover it with a 4x4 regression test.
- [x] Rerun SAES-only local sample-0 ablation: `34.8380575 dB`, zero deleted
  Gaussians, and no SAES saving claimed.
- [x] Replace the unconditional placeholder with the source-only
  `exact-source-opacity-zero-v1` certificate. It is limited to the direct S2
  density-to-opacity input of TranSplat/MVSplat, leaves retained anchors
  bit-identical, and fails closed for DepthSplat, malformed tensors, NaN,
  nonzero alpha, or an anchor mapping mismatch.
- [x] Verify that certificate on local sample-0 with `--no-fsdr`:
  `34.8380585 -> 34.8380575 dB`, zero deleted Gaussians, and no S2/S3 saving.
  All 3,066 L0/L1 candidates had nonzero source alpha and correctly fell back
  to Full; the result is
  `outputs/ae_dl3dv_local_repro_v2/transplat_sample0_saes_only_exact_zero_certificate_v1/`.
- [x] Record the historical, non-frozen no-FSDR quality regression for the fixed first eight local
  DL3DV scenes. The complete aggregate at
  `outputs/ae_dl3dv_local_repro_v2/transplat_saes_only_no_fsdr_8/results.json`
  is reproducible with no reference fallback: mean PSNR
  `26.8285265 -> 26.8285263 dB`, maximum per-scene PSNR delta
  `9.54e-7 dB`, and all eight routes retained every Gaussian. It uses the
  mutable diagnostic route and is not frozen-route SAES evidence.
- [x] Complete the development-only source-scalar versus posthoc dense-S3
  oracle audit at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_representative_deletion_risk_oracle_v1/`.
  It accepted 1,760 representative candidates (L0 400, L1 1,360), but dense
  S3 showed optical-depth relative error min/median `0.1579/0.2530` and median
  covariance relative error `0.2827`. It neither altered runtime routing nor
  produced a sparse quality/speedup claim.
- [x] Close the empirical nonzero-opacity risk-gate route. Existing
  source/anchor/feature/depth thresholds cannot certify deletion; retain only
  `exact-source-opacity-zero-v1` as the runtime deletion certificate.
- [x] Audit the existing local transmittance, conditional-optical-mass,
  teacher-attribution, and simulator traces. No numerical violation exists in
  the fixed Full/failed-certificate path; the actionable defect was that the
  old local command did not bind the frozen SAES identity.
- [x] Add `--frozen-saes-route` to bind the immutable route while permitting
  only `--no-fsdr` for local SAES-only fidelity checks. It rejects route drift,
  all other disabled hardware stages, and simulator fallback. Focused CLI and
  quality-contract tests pass (`67 passed`).
- [x] Run and validate the frozen-route TranSplat/DL3DV sample-0 simulator at
  `outputs/ae_dl3dv_local_repro_v3/transplat_sample0_frozen_identity_simulator_v1/`.
  It has no fallback, route SHA256 `fa2e79efbe89f54ed63ab9b6f083fa2a9af09a4d72cefa8218c9611911da0cbd`,
  PSNR `34.83805847 -> 34.83805752 dB`, and all 8,192 tiles Full.
- [x] Bind the same `exact-source-opacity-zero-v1` certificate to the
  source-bound packet guard. New 4x4 regressions prove zero-alpha L0 output
  remains compact while any nonzero source opacity fails closed to Full.
- [x] Run the fixed sample-0 packet-to-native-Adapter-to-native-decoder quality
  fidelity pilot at
  `outputs/ae_dl3dv_local_repro_v5/transplat_sample0_sparse_packet_full_route_quality_v1/`.
  It binds the frozen route SHA, has `8192/0/0` Full/L0/L1 and 131,072 decoder
  Gaussians, and passes PSNR/SSIM/LPIPS with deltas
  `-0.0000095/-0.000000134/+0.00000135`. It is explicitly zero deletion,
  zero S2/S3 saving, and non-claiming Full-route fidelity evidence.
- [ ] Rerun the fixed first eight local DL3DV scenes with
- [x] Rerun the fixed first eight local DL3DV scenes with
  `--frozen-saes-route --no-fsdr`, then validate aggregate provenance and
  quality. The aggregate at
  `outputs/ae_dl3dv_local_repro_v3/transplat_saes_only_frozen_identity_8/results.json`
  passes with mean PSNR `26.82852650 -> 26.82852632 dB`, maximum per-scene
  delta `9.5367e-7 dB`, 8/8 frozen route hashes, no fallback, and all tiles
  Full with zero S2/S3 saving.
- [x] Fix the aggregate path for scene-dependent SAES reason-counter maps and
  preserve hash-bound SAES identity fields verbatim. The targeted aggregate
  suite passes (`11 passed`); the repaired aggregate passes
  `validate_result.py` without changing any sample JSON.
- [ ] Do not launch a new external calibration, a 32/140-scene expansion, or
  any cross-dataset quality run before a genuine non-Full sparse mechanism
  independently passes the unchanged `<=0.15 dB` gate.
- [ ] Implement a final-selected representative-materialization packet consumer
  before treating a guard-expanded request packet as a nonzero deletion result.
  It must bind the exact-zero certificate, preserve source-native geometry,
  and pass the same local sample-0 quality gate before any wider DL3DV run.
- [ ] Repair the selected-output diagnostic so its final guard-resolved packet
  is converted and rendered without a dense Gaussian fallback. Require a
  source-bound final mask, NaN-poison isolation for omitted raw-head positions,
  a finite decoder render, and no target RGB, metric, or S2/S3 saving claim.
- [x] Run only the fixed local DL3DV sample-3 sparse-consumer renderer smoke
  after the CPU contract tests pass. Do not promote a finite render to a
  nonzero-deletion quality or performance result.
- [x] Run the local sample-3 packed renderer smoke. It proved NaN-poisoned
  omitted slots do not reach the packet consumer, then correctly rejected the
  source-bound selected-head route because Gaussian mean error reached
  `0.17310333` against the dense native Adapter. Keep it diagnostic-only; do
  not relax the attribute or image-quality tolerances.
