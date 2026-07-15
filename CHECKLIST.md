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

- [ ] Every claimed result passes expected-result validation.
- [x] Release checker rejects paths, missing DOI values, bad hashes, and incomplete dataset licenses.
- [ ] CPU, CUDA, Orin, and physical clean-room checks are recorded.
- [ ] Release tag, submodule SHAs, archive SHA256, and Zenodo DOI agree.

## Current Frontier

The strict quick path and all six historical one-sample pilots are executable,
but the pilot gate originally checked schema rather than paper tolerance. A
corrected formal run and focused TranSplat/MVSplat diagnostics prove that
sparse SAES does not meet Table 1 or Tables 2--3. C1/C4 are suspended and the
dense diagnostic is forbidden in claim runs. Formal RTL/DRAM evidence remains
complete. The clean iFlow preflight passed, but a 180-design external Vivado
sweep and less than 48 GiB available memory triggered the mandatory resource
guard. C7/C8 are not claimed. Do not mark routed PPA, clean-room, DOI, or
Results Reproduced evidence complete from diagnostics.

## Latest Local Verification

Verified on 2026-07-15:

- Base `pytest -q`: 194 passed and 9 skipped. The skipped tests require optional
  PyTorch or external toolchains.
- Locked classic profile: all 215 tests pass. Both classic and DepthSplat
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
  also passed. Real experiment evidence, clean-room results, and the DOI remain
  unresolved until their workflows run.
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
