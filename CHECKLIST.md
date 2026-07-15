# MICRO 2026 Artifact Implementation Checklist

## Planning

- [x] Claim boundary and public hardware substitution recorded.
- [x] Paper-result mapping and badge scope documented.
- [x] Implementation touchpoints and strict failure policy recorded.

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
- [x] FSDR Guided Rate is excluded because the tracked RTL projection ROM is
  all zero and no authentic hyperplane/feature-vector collateral is available.
- [x] Replace the continuous-depth-window Top-1 approximation with exact
  full-search argmax-in-candidate-subset evidence and rerun six bounded pilots.
- [x] Keep every Table 2 Top-1 row diagnostic-only: the exact tensors are
  available, but their guided denominator depends on missing LSH collateral.

## RTL And Hardware

- [x] Chisel behavior tests and SystemVerilog emission pass.
- [x] Verilator lint and representative switching VCD pass.
- [x] Ramulator and DRAMPower LPDDR5 functional proxy passes.
- [x] Pinned Ramulator 2.1 and DRAMPower 6.0.2 binaries were built and the
  real trace-to-timing-to-energy chain produced relocatable PASS evidence.
- [x] Clean iFlow `04b4d98` front end through global placement was exercised.
- [x] Clean iFlow/ASAP7/container/collateral preflight passes from pinned
  commit `04b4d98`.
- [x] Routed ASAP7 PPA is explicitly `NOT_CLAIMED_RESOURCE_LIMIT`; no partial
  run or manuscript constant is substituted.
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
sweep and less than 48 GiB available memory triggered the mandatory resource
guard. C7/C8 are not claimed. Pre-release source/evidence bundles pass the
clean-room hash, validation, public-asset download, and strict CUDA quick
checks. Do not mark routed PPA or Results Reproduced evidence complete from
diagnostics, and repeat the clean-room checks on the final DOI-bound bundles.

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
- [ ] Promote a routing or projection fix only if it is derived independently
  of the paper result values and passes the unchanged fixed-sample gates.
- [ ] Resume six quality pilots and full matrices only after the corresponding
  claim-critical mechanism gates pass.

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

### SAES Gaussian-Head Feature Audit

- [x] Bind the audit to the canonical TranSplat/Re10K sample and identify the
  authentic full-resolution Gaussian-head input in the upstream source.
- [x] Add failure-first tests for pre-hook capture, `[B,V,C,H,W]` alignment,
  strict-run rejection, and non-claim result eligibility.
- [x] Implement the diagnostic feature source without changing the default
  pipeline feature path.
- [ ] Run one clean fixed sample and record feature statistics, L0/L1/Full
  rates, and unchanged PSNR/SSIM/LPIPS deltas.
- [ ] Promote only if all provenance and quality gates pass; otherwise keep the
  full six-pair matrix stopped.

## Latest Local Verification

Verified on 2026-07-16:

- Base `pytest -q`: 256 passed after the probe-vector first-hit diagnostic was
  isolated from claim and Functional runs.
- Locked classic profile: all 256 tests pass. Both classic and DepthSplat
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
- A fresh post-fix strict quick run also passed under `outputs/ae_regression`.
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
- Formal RTL evidence under `outputs/ae/rtl` passes eight Chisel tests,
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
- `cd chisel && sbt test`: all 8 tests passed.
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
- All six public checkpoints and all three required runtime assets passed their pinned
  SHA256 checks. Dataset terms are recorded without redistribution permission.
  Re10K archive `ce351771c966fb25ef41efc561a313ef40607c9aa8ea904ed8d582b361408097`
  and prepared tree `2866634245989caa455fb46e5991d4e4b51e643c3f3e024e8793ff2b664fadd4`
  passed complete validation. ACID archive
  `c9f0685175bda403991493332b5cc10785678a51fbd5c497401d92d176a1a4ac`
  and prepared tree `0e21d05f448675e160529881d583701a2d03f19dbaae1af7941f37137688c187`
  also passed. Paper-result reproduction remains outside the claim. Pre-release
  clean-room validation is complete; the DOI-bound rebuild and account-side
  publication remain pending.
- The corrected ASAP7 RC flow is `NOT_CLAIMED_RESOURCE_LIMIT`: an unrelated
  180-design Vivado sweep had completed 84 designs with 96 remaining, and
  available memory was 11.8--14.0 GiB. The guard failure and successful clean
  dry-run are retained under `outputs/ae_failures/physical/`.
- Source-archive runs verify every file against `release-manifest.json` and no
  longer require Git metadata after Zenodo extraction.
- The paper builds successfully as a 15-page PDF. The Artifact Appendix occupies
  portions of pages 13--14, remains within the two-page limit, and has no LaTeX
  errors, undefined references, undefined citations, or status-name overflows.
- Staged reference payloads live under the ignored
  `artifact/reference_results/evidence/` tree and enter the evidence archive,
  not Git history. The compact manifest remains tracked.
