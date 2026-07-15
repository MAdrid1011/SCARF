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
- [x] Re10K and ACID sources, terms, archive revisions, prepared tree hashes,
  and all protocol-selected scene/view bounds are verified.
- [x] All six claimed model and dataset pairs complete a strict one-sample
  pilot with three target views, full checkpoint coverage, and positive S1,
  S2, S3, and GGU cycles.
- [ ] Full claimed Re10K and ACID runs are complete.
- [x] DL3DV and Orin results are explicitly outside the current claim.

## RTL And Hardware

- [x] Chisel behavior tests and SystemVerilog emission pass.
- [x] Verilator lint and representative switching VCD pass.
- [x] Ramulator and DRAMPower LPDDR5 functional proxy passes.
- [x] Pinned Ramulator 2.1 and DRAMPower 6.0.2 binaries were built and the
  real trace-to-timing-to-energy chain produced relocatable PASS evidence.
- [x] Clean iFlow `04b4d98` front end through global placement was exercised.
- [ ] Corrected ASAP7 RC overlay is rerun from a clean iFlow worktree.
- [ ] Routed ASAP7 PPA reports and provenance are complete.
- [x] SRAM proxy, PDN evidence, and excluded PHY/I/O fields are explicit.
- [ ] 7-to-28 normalization of the real ASAP7 run is reproducible.

## Validation And Release

- [ ] Every claimed result passes expected-result validation.
- [x] Release checker rejects paths, missing DOI values, bad hashes, and incomplete dataset licenses.
- [ ] CPU, CUDA, Orin, and physical clean-room checks are recorded.
- [ ] Release tag, submodule SHAs, archive SHA256, and Zenodo DOI agree.

## Current Frontier

The strict quick path and all six Re10K/ACID one-sample pilots pass. Complete
the delivery regression and create a clean source commit, then start the full
six-pair quality run. The single quality run for each pair also emits the
complete ablation and mechanism record. Software may run while an unrelated
Vivado job is active, but the corrected iFlow overlay still requires that every
Vivado process has exited and that at least 48 GiB is available. Do not mark
routed PPA, clean-room, or DOI evidence complete from dry runs, smoke vectors,
or fixtures.

## Latest Local Verification

Verified on 2026-07-15:

- Base `pytest -q`: 184 passed and 6 skipped. The skipped tests require PyTorch.
- Locked classic profile: all 197 tests pass. Both classic and DepthSplat
  environment checkers pass with CUDA 12.1 on the declared compiler path.
- Strict quick passed on the RTX 3060 with the pinned classic profile. It loaded
  MVSplat once, selected all three declared target views, emitted positive
  feature/depth/Gaussian/GGU cycles, and produced a validator-PASS aggregate.
- A fresh post-fix strict quick run also passed under `outputs/ae_regression`.
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
  also passed. Real experiment evidence, clean-room results, and the DOI remain
  unresolved until their workflows run.
- The corrected ASAP7 RC flow has not been rerun because an unrelated Vivado
  job remains active and available memory is below 48 GiB. Software pilots no
  longer wait for that job; the physical resource guard remains mandatory.
- Source-archive runs verify every file against `release-manifest.json` and no
  longer require Git metadata after Zenodo extraction.
- The paper and one-page Artifact Appendix build successfully as a 15-page PDF
  with no LaTeX errors, undefined references, or undefined citations.
- Staged reference payloads live under the ignored
  `artifact/reference_results/evidence/` tree and enter the evidence archive,
  not Git history. The compact manifest remains tracked.
