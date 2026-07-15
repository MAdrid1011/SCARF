# MICRO 2026 Artifact Implementation Plan

## 1. Objective

- Run ID: `micro2026-ae-reconstruction`
- Objective: turn the accepted SCARF paper repository into an independently
  executable, provenance-preserving artifact for the Available, Functional,
  and Results Reproduced badges.
- Non-negotiable boundary: public physical evidence is ASAP7 predictive 7 nm.
  DeepScale output is a 28 nm-equivalent estimate and must never be presented
  as the TSMC 28 nm post-layout measurement reported by the paper.
- Null hypothesis: the repository cannot reproduce its declared software,
  simulator, RTL, and public physical-flow claims from documented inputs.
- Alternative hypothesis: every declared claim can be regenerated or is
  explicitly scoped out, with complete inputs, raw records, and validation.

## 2. Baseline And Comparability

- Baseline: numbers and configurations in `micro59-submit/320.pdf`, mapped in
  `artifact/CLAIMS.md`.
- Active claim matrix: TranSplat, MVSplat, and DepthSplat on Re10K and ACID.
  DL3DV mappings remain executable but are not claimed without gated data.
- Required software metrics: PSNR, SSIM, LPIPS, positive simulator cycles, and
  FSDR/SAES mechanism statistics. Orin timing and sensitivity figures are not
  in the active claim set.
- Required hardware metrics: routed ASAP7 area, delay/frequency, dynamic and
  leakage power, utilization, route/DRC status, plus deterministic scaling.
- Comparability risks: unavailable ACID/DL3DV payloads and checkpoints,
  incompatible model environments, absent Orin measurements, absent iFlow
  reports, SRAM abstraction differences, and commercial TSMC28 exclusions.
- Recovered protocol identity: non-null entries from the committed upstream
  indices in source-file order. Re10K has 6,474 executable samples, ACID has
  1,595, and DL3DV has 140. Execution follows the prepared dataloader's sorted
  chunk traversal and preserves both the stable source ordinal and the runtime
  execution ordinal in provenance.

## 3. Code Translation Plan

| Area | Planned implementation | Acceptance signal |
|---|---|---|
| Contract/docs | Claims, hardware scope, appendix, HotCRP, licenses | no unsupported badge claim |
| Software CLI | dataset-aware loader, fixed result schema, strict errors | quick and nine-pair dry run pass |
| Orchestrator | all public modes, aggregation, expected-result validation | missing evidence returns nonzero |
| Data/env | pinned profiles and checksum manifests | clean download/install verification |
| RTL | repaired tests, SV emission, Verilator/VCD | RTL result schema passes |
| DRAM | Ramulator trace and DRAMPower wrappers | raw logs map to parsed results |
| Physical | clean iFlow `04b4d98` ASAP7 flow and parser | routed reports set `physical_valid` |
| Scaling | tested DeepScale tables and 7-to-28 conversion | raw/factor/scaled records preserved |
| Release | path/hash/license/archive checks | unpacked archive quick run passes |

## 4. Execution Design

- Minimal pilot: unit tests, orchestrator dry runs, result-schema validation,
  DeepScale examples, and RTL dry run.
- Full run: `bash scripts/run_ae.sh all`, then `physical`, `scale`, and
  `validate` using the declared hardware environments.
- Stop condition: all claimed rows are PASS and the clean-room package can
  reproduce them without author-local paths or undistributed commercial data.
- Abandonment condition for a claim: narrow the claim when a required legal
  input or physical machine is unavailable before the deadline. Do not
  fabricate evidence.
- Output root: `outputs/`.
- Permanent manifests and expected values: `artifact/`.

## 5. Runtime Strategy

- Unit smoke: `python -m pytest -q`.
- CLI smoke: `bash scripts/run_ae.sh quick`.
- Software smoke and model pilots may run while an unrelated Vivado synthesis
  is active. They remain sequential and use the assigned single GPU.
- Do not run `scripts/run_rtl.sh` concurrently with a clean claim pair. The RTL
  emitter temporarily recreates `chisel/generated`, so formal RTL and software
  evidence runs are serialized to keep Git identity stable.
- Full software: `bash scripts/run_ae.sh all`.
- Full mode reuses the embedded ablation and mechanism records from each quality run
  instead of repeating the same model and sample matrix.
- Hardware: `bash scripts/run_ae.sh rtl`, `bash scripts/run_ae.sh physical`,
  and `bash scripts/run_ae.sh scale`.
- The physical flow is stricter than software execution: no Vivado process may
  be active and `/proc/meminfo` must report at least 48 GiB available memory.
- Final gate: `bash scripts/run_ae.sh validate` and archive clean-room checks.
- Long GPU/Orin/iFlow runs must retain commands, environment snapshots, logs,
  hashes, timestamps, and explicit failure status under their output directory.

## 6. Fallbacks And Recovery

- Missing data/checkpoints: fail with the exact manifest/download instruction.
- Missing Orin: software functionality may be tested elsewhere, but the Orin
  performance claim remains unexecuted.
- Missing iFlow/ASAP7: scaling unit tests may pass, but physical reproduction
  remains unexecuted.
- TSMC28 collateral: never request it for publication. Compare against the
  paper only after reporting raw ASAP7 and the normalization model.

## 7. Checklist Link

- Living execution checklist: `CHECKLIST.md`.
- Submission gate checklist: `artifact/CHECKLIST.md`.

## 8. Revision Log

| Date | Change | Reason |
|---|---|---|
| 2026-07-13 | Initialized from the user-approved 25-stage plan | Preserve the claim boundary while implementation proceeds |
| 2026-07-13 | Split DL3DV into native and Re10K-compatible prepared trees | DepthSplat and the Re10K loaders require different source image shapes |
| 2026-07-13 | Aggregate all sampler-selected target views per sample | Remove the hidden first-view truncation while preserving the unresolved author protocol gate |
| 2026-07-13 | Finalize the recoverable upstream evaluation protocol | Shared committed indices provide stable scene and view selections without author input |
| 2026-07-14 | Make claim runs fail on every simulator fallback | A GPU fallback cannot carry RTL or cycle claims even when its output metrics look plausible |
| 2026-07-14 | Bind source-archive revisions and use relocatable evidence paths | Tree hashes alone do not identify the downloaded archive, and author-local paths break clean-room use |
| 2026-07-14 | Vectorize FSDR signatures and SAES tile classification | Preserve raster cache semantics while making the full protocol computationally feasible |
| 2026-07-14 | Add a synthetic strict Functional quick path | Permit small clean-room validation without redistributing Re10K |
| 2026-07-14 | Bind separate profile interpreters and preflight records | Prevent one Python environment from silently running incompatible models |
| 2026-07-14 | Reuse quality runs for ablation and mechanism evidence | Remove an identical second full model and sample matrix |
| 2026-07-14 | Verify source archives through the embedded release manifest | Make Zenodo extraction executable without Git metadata while detecting edits |
| 2026-07-14 | Bind validated Re10K and ACID archive and prepared-tree hashes | Make the six accessible claim pairs reject any changed dataset tree |
| 2026-07-15 | Start software pilots without waiting for unrelated Vivado work | Use the idle CUDA device while preserving the independent physical-flow guard |
| 2026-07-15 | Keep staged evidence payloads out of Git | Package full public evidence without bloating source history |
| 2026-07-15 | Pin the model-scoped Depth Anything V2 Base runtime asset | TranSplat requires the Base checkpoint; MVSplat must not inherit that requirement |
| 2026-07-15 | Bind DepthSplat checkpoints to their published Hydra variants | The Re10K/ACID checkpoint is ViT-L and the DL3DV checkpoint is ViT-B, not the default ViT-S |
| 2026-07-15 | Construct DINOv2 from pinned local source without pretrained downloads | Full DepthSplat checkpoints already contain the backbone and claim runs must not access the network |
| 2026-07-15 | Complete DepthSplat multi-scale S2 and traced S3 cycles | Remove one-scale assumptions and zero Gaussian cycles exposed by strict pilots |
| 2026-07-15 | Serialize the RTL emitter and software claim runner | A concurrent emitter correctly triggered the dirty-worktree evidence guard; no invalid sample was accepted |
| 2026-07-15 | Discover pinned DRAM tools from the repository install tree | Keep the public `run_ae.sh dram` entry functional without hidden environment variables |
| 2026-07-15 | Separate stable protocol identity from prepared-data execution order | Preserve the published selection hash while matching the upstream chunk dataloader exactly |
