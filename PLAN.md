# MICRO 2026 Artifact Implementation Plan

## 1. Objective

- Run ID: `micro2026-ae-reconstruction`
- Objective: turn the accepted SCARF paper repository into an independently
  executable, provenance-preserving artifact for the Available and Functional
  badges. Results Reproduced remains a future gate because current real
  sparse-SAES probes refute the paper-result contract.
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
- Active software claim matrix: none. Real sparse-SAES probes do not satisfy
  the paper quality or L1-rate contracts. The Re10K/ACID workflows remain
  executable diagnostics; DL3DV remains unavailable without gated data.
- Required Functional software evidence: numerically equivalent S1/S2/GGU,
  positive simulator cycles, FSDR evidence, strict sparse-SAES failure records,
  and a claim guard that rejects the dense diagnostic. Orin timing and
  sensitivity figures are not in the active claim set.
- Optional physical metrics: routed ASAP7 area, delay/frequency, dynamic and
  leakage power, utilization, route/DRC status, plus deterministic scaling.
  They are currently outside the claim because the 48 GiB/no-Vivado resource
  gate is not met; DeepScale table/formula tests remain Functional evidence.
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
- Declared workflow: `bash scripts/run_ae.sh all`. With the current claim
  status it explicitly skips paper-result software pairs and continues through
  RTL, DRAM, physical/scaling, report generation, and validation.
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
| 2026-07-15 | Stop the full matrix after four systematic SAES quality failures | Avoid spending weeks on a non-comparable run and preserve every raw result |
| 2026-07-15 | Pair reference numerics with executed hardware cycle traces | Remove approximate-weight quality drift while retaining positive S1/S2/S3/GGU evidence |
| 2026-07-15 | Suspend C1/C4 sparse-SAES claims | TranSplat and MVSplat real probes exceed quality tolerance and produce zero L1 tiles; dense interpolation is not accepted as pruning evidence |
| 2026-07-15 | Reopen one bounded paper-formula SAES audit | The implementation uses normalized full-tile feature variance, relative depth spread, and undocumented decision gates, while the manuscript specifies raw probe variance, absolute probe depth standard deviation, and first-hit threshold decisions |
| 2026-07-15 | Stop the paper-formula SAES audit after one discriminative run | Literal raw-probe variance made 99.9% of tiles L0, left L1 effectively zero, and worsened PSNR/SSIM substantially; the manuscript leaves the feature-vector reduction and normalization under-specified |
| 2026-07-15 | Include only the synthetic quick dataset in the source bundle | A clean-room reviewer could not run `quick` because the blanket dataset exclusion also removed its manifest and sample; real datasets remain excluded |
| 2026-07-16 | Preserve the SAES coverage/attribution contradiction | No mass-conserving coverage pair or representative-component restoration passed the fixed quality tolerances; moment matching is necessary but insufficient |
| 2026-07-16 | Add one retention-boundary diagnostic before changing L1 | The current L0 gate consumes every tile that can pass the shared Gaussian cross-check, making L1 structurally unreachable; first test whether the declared 17.2% L0 rate is itself compatible with 4/16 sparse materialization |
| 2026-07-16 | Reject literal SH/opacity averaging after one real run | The paper-aligned constant-preserving average worsened the canonical sparse result from -0.4236 dB to -5.1204 dB PSNR; restore alpha-union as the last-known-good diagnostic implementation |
| 2026-07-16 | Close sparse-SAES recovery and move the badge frontier to FSDR Table 2 | Raw-feature and probe-Gaussian rankings both fail even at one quarter of the declared L0 rate, and LightGaussian supplies pruning plus finetuning rather than the cited moment-matching operation |
| 2026-07-16 | Remove FSDR Guided Rate from the reproduction frontier | All six real one-sample rows differ materially from Table 2, and the RTL LSH projection ROM contains only zero initialization rather than the hyperplanes needed to reproduce the software signatures |
| 2026-07-16 | Replace continuous-depth Top-1 approximation with discrete candidate evidence | Table 2 defines coverage over the full-search argmax candidate and retained candidate subset; a +/-25% continuous-depth window is not that metric |

## 9. Paper-Formula SAES Audit

- Run ID: `saes-paper-formula-v1`.
- Research question: does matching the published L0/L1 decision formulas restore
  the nonzero L1 protocol and paper quality without changing the dataset,
  sampler, materialization policy, or metric definitions?
- Null hypothesis: decision-formula alignment does not improve both sparse
  quality and L0/L1 agreement.
- Alternative hypothesis: raw probe-feature variance, absolute probe-depth
  standard deviation, and threshold-only first-hit routing remove the observed
  protocol mismatch.
- Minimal experiment: one canonical TranSplat/Re10K sample, compared with the
  archived corrected sparse result.
- Acceptance keys: PSNR and SSIM deltas, L0 rate, L1 rate, effective Gaussian
  count, and `paper_result_eligible` provenance.
- Continue condition: quality and L0/L1 agreement both improve without dense
  materialization. Otherwise preserve the result as a refuted implementation
  hypothesis and keep C1/C4 suspended.
- Result: the alternative hypothesis is refuted on the bounded canonical
  sample. Relative to `transplat-re10k-moment-match-v6`, PSNR changed from
  26.1724 to 23.5173 dB, SSIM from 0.87879 to 0.80966, L0 from 29.2% to 99.9%,
  and L1 remained effectively zero (2/8192 tiles). The output is preserved at
  `outputs/ae_failures/diagnostics/transplat-re10k-paper-formula-v1/` with
  `git_dirty=true` and is not claim evidence.
- Decision: restore the last-known-good Functional implementation, make no
  wider SAES rerun, and keep Results Reproduced suspended. Raw-probe variance
  plus the published threshold is not a recoverable reproduction contract
  without an absent normalization/reduction definition or original logs.

## 10. SAES Recovery Campaign V2

- Campaign ID: `saes-recovery-v2`.
- Parent run: the preserved TranSplat/Re10K sparse diagnostics at commit
  `9e27f50`, especially `moment-match-v4`, `moment-match-v6`, and the dense
  diagnostic.
- Main claim under test: a paper-faithful sparse SAES implementation can keep
  PSNR within 0.15 dB, SSIM within 0.005, and LPIPS within 0.005 while producing
  the declared L0/L1 paths without retaining all Gaussians.
- Fixed conditions: canonical sample/view selection, checkpoint, dataset tree,
  target views, renderer, quality metrics, tile size, seed, and sparse output
  contract. Generated JSON and tolerances remain immutable.
- Selected paper reference: accepted `micro59-submit/320.pdf`, Section 3 and
  Table 3. There is no separate paper experiment matrix in this repository;
  this section is the executable matrix for C1/C4.

| Exp ID | Slice ID | Tier | Class | Question | Continue signal |
|---|---|---|---|---|---|
| C1-D1 | `decision-statistics` | main_required | auxiliary | Which scalar definition makes paper `tau_f=0.2` and `tau_d=0.1` meaningful on real probes? | A documented definition produces non-degenerate L0/L1 rates without post-hoc target access |
| C1-D2 | `coverage-attribution` | main_required | claim-carrying | Is the sparse quality gap caused by representative covariance/opacity coverage rather than tile selection? | A theory-consistent coverage transform materially removes the visible 4x4 grid and improves all quality metrics |
| C1-D3 | `retention-boundary` | main_required | claim-carrying | Can 4/16 sparse materialization meet the quality contract when limited to the paper's 17.2% TranSplat/Re10K L0 rate? | A target-free probe-statistic ranking at the declared rate passes all three unchanged tolerances |
| C4-D1 | `l1-semantics` | main_required | claim-carrying | Does L1 require a distinct lighter-retention path rather than the current L0-equivalent 4/16 materialization? | An explicit paper-compatible path explains nonzero L1 and preserves quality |
| C1-P1 | `six-pair-pilots` | main_required | claim-carrying | Does the corrected mechanism generalize to all Re10K/ACID pairs? | All six one-sample pilots pass the fixed quality and provenance gates |
| C1-M1 | `six-pair-matrix` | main_required | claim-carrying | Do the full six matrices reproduce Table 1 and Tables 2--3? | Every aggregate passes the existing expected-results validator |

- First execution: one canonical TranSplat/Re10K run that emits decision
  distributions and a fixed coverage-factor sweep. It is diagnostic-only and
  must set `paper_result_eligible=false`.
- Success condition: identify one mechanism-level correction that passes the
  canonical sample without dense materialization, then validate it across six
  pilots before any formal matrix run.
- Abandonment condition: two instrumented, one-factor retries after this audit
  produce no interpretable improvement or require target-image-driven routing.
- Monitoring: check long GPU runs at roughly 60, 120, 300, and 600 seconds;
  preserve every completed or failed output in a new non-overwriting directory.
- `decision-statistics` result: `tau_f=0.2` accepts 3.80% under raw
  vector variance, 99.99% under unit-vector variance, and 100.00% under the
  current channel-std statistic. `tau_d=0.1` accepts 90.06% under absolute
  depth std and 96.75% under relative depth std. The manuscript therefore does
  not define enough normalization/reduction detail to reproduce its path rates.
- `coverage-attribution` result: all 13 sparse variants failed. The best
  mass-conserving pair (`covariance=2`, alpha exponent `0.5`) reached deltas of
  -0.3811 dB PSNR, -0.005554 SSIM, and +0.007651 LPIPS. Restoring any original
  representative attribute worsened quality, and restoring all representative
  attributes while zeroing non-probes reached -11.5482 dB. Evidence is under
  `outputs/ae_failures/diagnostics/transplat-re10k-saes-recovery-v2-attribution/`.
- Structural finding: the current statistic places every tile below
  `tau_f=0.2`; L0 then accepts exactly the tiles passing the Gaussian
  cross-check. L1 reuses that same cross-check, so every L0 miss must also miss
  L1. This explains the observed zero L1 rate and is not fixable by changing
  `tau_d` alone.
- `retention-boundary` result: target-free raw-probe ranking still failed every
  nonzero tested rate. At one quarter of the declared L0 rate (4.30% of tiles),
  deltas were -0.0304 dB PSNR, -0.005078 SSIM, and +0.019556 LPIPS. At the
  declared 17.2% rate they were -0.1494 dB, -0.017557, and +0.054051. Thus the
  current 4/16 sparse materialization cannot meet all three tolerances merely by
  routing fewer low-variance tiles.
- Paper-average result: replacing alpha-union and opacity-weighted SH with
  literal constrained weighted averages preserved constants in the unit test
  but produced -5.1204 dB PSNR, -0.120117 SSIM, and +0.119476 LPIPS. This
  source correction is refuted and reverted; its evidence remains under
  `outputs/ae_failures/diagnostics/transplat-re10k-saes-recovery-v2-paper-average/`.
- Probe-error boundary result: sorting eligible tiles by the target-free
  Gaussian leave-one-out error did not improve the frontier. At 4.30% it
  reached -0.0529 dB PSNR, -0.006471 SSIM, and +0.026159 LPIPS; at 17.2% it
  reached -0.2204 dB, -0.019584, and +0.055766. The result is preserved under
  `outputs/ae_failures/diagnostics/transplat-re10k-saes-recovery-v2-probe-error-boundary/`.
- Citation audit: public LightGaussian commit
  `6676b983e77baadd909effc56a6aaadafa964dcc` implements importance pruning,
  finetuning/recovery, SH distillation, and vector quantization. It does not
  expose the first/second-moment Gaussian merge cited by the SCARF manuscript,
  so it cannot supply the missing SAES operation or normalization contract.
- Closing decision: keep C1/C4 sparse-SAES rows not claimed and do not launch
  another threshold, coverage, or routing retry. Continue Results Reproduced
  work on the independent FSDR columns of paper Table 2 for all six legally
  available Re10K/ACID pairs, followed by RTL and any valid ASAP7/DeepScale
  public hardware proxy.

## 11. FSDR Table 2 Recovery

- Run ID: `fsdr-table2-six-pair`.
- Research question: do the public checkpoints, canonical indices, and
  target-free FSDR implementation reproduce the Guided Rate, Top-1 Coverage,
  depth-evaluation savings, and feature-buffer reductions in paper Table 2?
- Fixed conditions: all Re10K/ACID protocol samples, declared checkpoints,
  feature/depth definitions, cache/window configuration, and existing absolute
  mechanism tolerance. DL3DV remains outside the claim.
- Execution strategy: add a persistent FSDR-only measurement path that loads a
  model once per pair, omits SAES and image rendering, emits one trace record per
  complete sample, and aggregates only after selection-hash validation.
- Success condition: all six rows pass the unchanged mechanism validator and
  provide complete provenance plus resumable per-sample traces.
- Failure condition: a completed row exceeds the existing tolerance. Preserve
  it, diagnose the implementation/protocol mismatch, and do not copy manuscript
  constants into generated evidence.
- Six-pair pilot result: every row completed with schema-valid, non-fallback
  evidence, but Guided Rate was 99.42%--99.79% versus 66.2%--87.0% in the
  accessible Table 2 rows. Guided Rate is therefore not reproducible from the
  public artifact. The pilot evidence is retained under
  `outputs/ae_pilot_fsdr_v1/`.
- RTL/source blocker: `LSHHashUnit.scala` initializes its complete 16x128
  projection ROM to zero, while the software creates seed-0 Gaussian
  hyperplanes. Neither the paper nor a tracked manifest supplies the projection
  values or an exact feature-vector reduction contract. The Guided Rate claim
  must remain `NOT_CLAIMED_MISSING_COLLATERAL`; no threshold or seed sweep is
  permitted as a substitute.
- Metric correction: the pilot's `guided_in_window / guided` value is only a
  continuous-depth diagnostic. It must not be labeled Top-1 Coverage. Claim
  evidence now requires the full-search probability volume, its corresponding
  candidate tensor, per-pixel argmax indices, and the exact nearest-`D/R`
  subset around the cached anchor.
- Remaining slice: expose those discrete tensors from each executed predictor,
  validate their view and cost-volume resolution, and run one clean sample per
  accessible pair. A pair is claimable only if all tensors are authentic and
  its exact Top-1 Coverage passes the unchanged 0.02 absolute tolerance. If a
  predictor cannot expose the tensors without reconstructing missing values,
  or any pilot fails, mark that pair `NOT_CLAIMED` and do not launch its full
  protocol.
- Exact-pilot result: all six predictors exposed executed probability volumes
  and matching inverse-depth candidates. Top-1 Coverage for Re10K was 96.694%
  (TranSplat), 96.129% (MVSplat), and 96.706% (DepthSplat), so all three miss
  the fixed tolerance. ACID reached 99.244%, 98.789%, and 100.000%, which is
  numerically within tolerance on the single pilot sample. Guided Rates were
  89.36%--95.61% across all rows and all failed their paper targets.
- Closing decision: do not run any full FSDR protocol. The Re10K pilot already
  fails the exact metric, every Guided Rate fails, and even numerically passing
  ACID Top-1 values are conditioned on a guided set generated with software
  hyperplanes that cannot be tied to the all-zero RTL ROM or the paper. Preserve
  `outputs/ae_pilot_fsdr_v2/` as diagnostic evidence and keep Table 2 outside
  the Results Reproduced claim.

## 12. Mechanism Recovery Campaign V3

- Campaign ID: `mechanism-recovery-v3`.
- Parent evidence: `saes-recovery-v2` and `fsdr-table2-six-pair` at commit
  `6f210ad`.
- Main question: do two concrete implementation-contract mismatches, rather
  than tuned thresholds, explain enough of the SAES/FSDR gap to restore a valid
  six-pair reproduction path?
- Fixed conditions: canonical sample/view selection, checkpoints, dataset tree,
  target views, paper thresholds (`tau_f=0.2`, `tau_d=0.1`, `tau_h=3`), cache
  size 32, candidate ratio `D/4`, quality metrics, tolerances, tile size, and
  seed. Generated results remain immutable and no paper table value may be used
  as an implementation constant.

| Exp ID | Slice ID | Question | Intervention | Continue signal |
|---|---|---|---|---|
| C1-D4 | `candidate-coordinate-depth` | Is the fixed L1 threshold intended for the model's dimensionless depth-candidate coordinate rather than metric depth? | Map each predicted depth through the exact upstream inverse-depth near/far parameterization and report aligned first-hit rates without changing SAES output | The paper-fixed thresholds produce a non-degenerate L0/L1 split and identify a falsifiable routing correction |
| C2-D3 | `all-context-frames` | Did the FSDR pilot measure only context view zero even though the paper metric covers frames and resets the cache per frame? | Process every authentic probability/candidate view, clear frame-local cache state at each frame boundary, and aggregate integer counts | All context pixels are covered and the corrected bounded pilots pass the unchanged Table 2 tolerance |
| C2-D3b | `canonical-prefix-stability` | Are the one-sample FSDR failures representative or merely sample variance relative to dataset aggregates? | Aggregate the first 32 canonical samples for all six accessible pairs with the corrected all-frame contract | The Guided Rate and Top-1 deltas move consistently toward the paper targets; otherwise diagnose projection/feature semantics before a full protocol |
| C1-D5 | `paper-routing-replay` | Does the candidate-coordinate result support a paper-defined L0/L1 router without the unpublished cross-check and similarity gates? | Apply only the published first-hit statistics on the fixed sample | Quality and path rates both pass without target-image access or extra thresholds |
| C2-D4 | `projection-collateral` | Can software and RTL share a reproducible random-hyperplane matrix without selecting a seed from paper results? | Export one manifest-hashed matrix generated independently of evaluation metrics and load the same quantized values in both implementations | RTL/software signatures are bit-exact and six bounded pilots pass without seed search |

- Order: run C1-D4 and C2-D3 first because both are correctness diagnostics.
  C1-D5 is allowed only if C1-D4 yields a coherent first-hit interpretation.
  C2-D4 is allowed only after multi-frame evidence shows that projection
  collateral, rather than measurement coverage, is the remaining blocker.
- Success condition: one paper-derived implementation passes the fixed
  TranSplat/Re10K quality gate and all six FSDR pilot rows pass the existing
  mechanism validator, after which the six quality pilots and full protocols
  may resume.
- Abandonment condition: corrected coordinate/frame semantics still fail the
  fixed bounded gates, or recovery requires choosing normalization, seed,
  routing, or retention constants by minimizing error to the paper tables.
- C1-D4 result: refuted. On the canonical TranSplat/Re10K sample, raw probe
  vector variance produced L0=3.80%. Among the remaining tiles, metric-depth
  standard deviation produced L1=86.27%, while the exact upstream normalized
  inverse-depth candidate coordinate produced L1=94.89%. The latter makes the
  split more degenerate rather than explaining the paper's 17.2%/14.8% rates.
  The diagnostic is preserved under
  `outputs/ae_failures/diagnostics/transplat-re10k-saes-candidate-coordinate-v3/`;
  candidate-coordinate normalization is not promoted into SAES routing.
- C2-D3 bounded TranSplat/Re10K result: all 8,192 pixels from both context
  frames were measured with a cache reset at each frame boundary. Guided Rate
  changed from the view-zero-only 89.36% to 90.36%, and exact Top-1 Coverage
  changed from 96.694% to 96.812%. Both still fail the unchanged 0.02 absolute
  tolerance against 72.1% and 99.91%. The coverage correction remains because
  it fixes the evidence definition; its dirty diagnostic is preserved under
  `outputs/ae_failures/diagnostics/transplat-re10k-fsdr-all-context-v3/`.
- C2-D3 six-pair clean pilot: all six canonical one-sample rows completed at
  commit `fcdd916` with 8,192 authentic pixels per row and no fallback. Guided
  Rate was 84.85%--95.09% and every row failed its paper target. Re10K exact
  Top-1 Coverage was 96.75%--98.04%; ACID was 99.24%--99.97%. Because these are
  single-sample diagnostics compared with dataset aggregates, C2-D3b measures a
  fixed 32-sample prefix before deciding whether a full run has information
  value. Evidence is under `outputs/ae_pilot_fsdr_v3/`.
