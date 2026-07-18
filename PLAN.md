# MICRO 2026 Artifact Implementation Plan

## 1. Objective

- Run ID: `micro2026-three-badge-closure`
- Objective: turn the accepted SCARF paper repository into an independently
  executable, provenance-preserving artifact for Artifact Available,
  Artifacts Evaluated - Functional, and Results Reproduced. The third badge is
  an active submission intent, not a completed result: only a calibrated global
  mechanism configuration and independent evaluator evidence may promote it.
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
- Active software claim matrix: all nine model/dataset pairs are retained in
  the claim contract. Existing Re10K/ACID probes remain negative diagnostic
  evidence until the new disjoint calibration and mechanism gates succeed;
  DL3DV's official gated data is prepared and re-verified, but it remains
  non-claiming until those same global calibration and mechanism gates succeed.
- Required Functional software evidence: numerically equivalent S1/S2/GGU,
  positive simulator cycles, FSDR evidence, strict sparse-SAES failure records,
  and a claim guard that rejects the dense diagnostic. Orin timing and
  sensitivity figures are not in the active claim set.
- Optional physical metrics: routed ASAP7 area, delay/frequency, dynamic and
  leakage power, utilization, route/DRC status, plus deterministic scaling.
  They are currently outside the claim because no complete routed evidence
  exists. The default flow recommends 48 GiB/no-Vivado; an explicit audited
  low-memory attempt may run the unchanged design. DeepScale table/formula
  tests remain Functional evidence.
- Comparability risks: reviewer-side gated DL3DV access, checkpoint
  availability,
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
- Full run: `bash scripts/run_ae.sh calibrate`, bounded diagnostic pilot,
  reviewer profile, full profile, then `all-eval`, `physical`, `scale`, and
  `validate --require-key-results` using the declared hardware environments.
- Stop condition: all claimed rows are PASS and the clean-room package can
  reproduce them without author-local paths or undistributed commercial data.
- A result is removed from the submission only when its legal input or real
  hardware evidence remains unavailable; do not fabricate evidence, edit
  generated records, or relax a tolerance.
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
  be active. Its default path requires 48 GiB `MemAvailable`; its explicit
  low-memory attempt retains resource/swap snapshots and cannot bypass report
  validation.
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
| 2026-07-16 | Isolate the remaining SAES vector-variance interpretation | Test raw probe-vector total variance and threshold-only first-hit routing as an explicitly non-claiming mode; do not change the default simulator unless the fixed sample passes every unchanged quality gate |
| 2026-07-16 | Add an explicit protocol-pair subset to the public runner | Let reviewers run legal/gated subsets without changing the default nine-pair claim matrix or bypassing canonical indices |
| 2026-07-16 | Stop full execution after the seed-42 six-pair mechanism pilot | All six schema-valid runs fail unchanged Table 1/Table 2/Table 3 gates; preserve the same-source outputs and diagnose SAES/FSDR semantics before spending full-protocol compute |
| 2026-07-16 | Add audited low-memory ASAP7 attempt mode | The 48 GiB threshold is a conservative host-preflight recommendation, not a substitute for real completion evidence; preserve no-Vivado exclusion and all routed-report gates while allowing the unchanged flow to be attempted. |
| 2026-07-16 | Add C2W-ray-aware SAES moment construction | Non-probe Stage-3 attributes remain unread; pseudo 3D means now use only static camera geometry, assignment weights, and probe depths, and the path is bound into calibration/sensitivity provenance. |
| 2026-07-16 | Materialize target-free calibration sidecars | Calibration replay now reads only selected context-image bytes plus target camera geometry; compiler and runner reject target-RGB-bearing inputs before a trace starts. |
| 2026-07-17 | Stop the hardware-honest probe-spread SAES branch | The strict canonical TranSplat/Re10K diagnostic pruned 75.0% of Gaussians but lost 7.9188 dB SAES-only PSNR, so this non-claiming historical coverage variant cannot justify another six-pair run. |
| 2026-07-18 | Make primary routing probes the default L1 reliability reference | Section 3 defines the L1 depth mean and standard deviation over the routing probe set; the declared 2K anchor expansion remains an execution and aggregation detail. A fresh target-free audit passed structural and poison-invariance checks, but it does not override the historical all-attribute failure or authorize a quality retry. |

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
- C2-D3b result: all six 32-sample canonical-prefix rows completed cleanly at
  commit `6853691`, covering 262,144 authentic pixels per row. Guided Rate was
  91.121/92.888% for TranSplat, 86.551/87.259% for MVSplat, and
  94.845/94.408% for DepthSplat on Re10K/ACID, versus paper targets
  72.1/76.0%, 66.2/71.0%, and 83.0/87.0%. Exact Top-1 Coverage was
  94.786/96.920%, 95.994/94.818%, and 98.637/99.304%, versus
  99.91/99.93%, 99.90/99.87%, and 99.89/99.95%. Every unchanged validator row
  failed, so the mismatch is systematic rather than one-sample variance. The
  evidence is preserved under `outputs/ae_pilot_fsdr_v3_32/`; no full FSDR
  protocol is launched from this result.
- Next bounded slice: test the model feature contract without changing the
  seed, cache, thresholds, schedule, candidates, or selection. DepthSplat
  exposes both 128-channel multi-view matching features and a separate DINOv2
  mono feature tensor, while the current FSDR path always hashes the former.
  Add a diagnostic-only selector that records its feature source and derives
  the hash input dimension from the executed tensor, then run one canonical
  DepthSplat sample with the authentic mono tensor. Promotion is allowed only
  if the paper supports the selected source and all fixed mechanism gates pass;
  otherwise preserve the null result and continue to projection semantics.
- DepthSplat mono-feature result: refuted. The official ViT-L execution exposed
  a 1,024-channel mono tensor (not the adapter's stale 384-channel default).
  On the canonical Re10K sample, hashing that authentic tensor yielded
  99.976% Guided Rate and 68.791% exact Top-1 Coverage, versus 83.0% and
  99.89%. Both unchanged checks failed, and Top-1 Coverage was substantially
  worse than the 128-channel pipeline-feature diagnostic. Preserve
  `outputs/ae_failures/diagnostics/depthsplat-re10k-fsdr-dino-v4/` and do not
  promote direct mono-feature hashing. Continue with a source-level audit of
  projection collateral and RTL/software signature equivalence without seed
  search or paper-target fitting.

## 13. SAES Probe-Vector Decision Audit

- Run ID: `saes-probe-vector-first-hit-v1`.
- Research question: does the manuscript's vector-valued feature variance,
  implemented as the mean squared L2 distance of the executed raw probe
  vectors, recover a usable L0/L1 split when the unpublished Gaussian gates are
  removed?
- Fixed conditions: canonical TranSplat/Re10K sample 0, context and target
  views, seed 0, checkpoint, raw S1 tensor, four corner probes at tile size 4,
  `tau_f=0.2`, `tau_d=0.1`, representative materialization, renderer, and the
  existing PSNR/SSIM/LPIPS tolerances.
- One-factor change: add a diagnostic decision mode in which L0 uses only raw
  probe-vector total variance and L1, strictly after an L0 miss, uses only the
  absolute population standard deviation of probe depths. The default formal
  simulator remains unchanged.
- Evidence boundary: the CLI must reject this mode for claim and Functional
  runs, and its generated result must set `paper_result_eligible=false`.
- Continue condition: the fixed sample has a non-degenerate first-hit split and
  passes all three unchanged tolerances: PSNR <= 0.15 dB absolute delta, SSIM <=
  0.005, and LPIPS <= 0.005.
- Abandonment condition: any quality gate fails, the split remains degenerate,
  or implementation requires a normalization, gate, threshold, or seed absent
  from the paper. Preserve the output and do not launch six-pair pilots.
- Physical flow remains independently gated on no Vivado process and at least
  48 GiB `MemAvailable`.
- Result: refuted at clean commit `1aec94c`. The fixed sample produced
  L0/L1/Full rates of 3.796%/86.267%/9.937%. Relative to the unmodified
  baseline, the combined path changed PSNR by -3.1708 dB, SSIM by -0.103587,
  and LPIPS by +0.180455, failing every unchanged tolerance. The schema-valid
  non-claim evidence and log are preserved under
  `outputs/ae_failures/diagnostics/transplat-re10k-saes-probe-vector-first-hit-v2/`.
- Decision: do not promote this decision mode and do not launch six-pair
  pilots. The available paper text does not define a route that simultaneously
  recovers its path rates and sparse quality without an unpublished
  normalization, gate, or materialization/recovery procedure.

## 14. SAES Gaussian-Head Feature Audit

- Run ID: `saes-gaussian-head-feature-v1`.
- Research question: does SAES currently measure the wrong executed feature
  tensor? The implementation uses the 1/4-resolution matching feature, while
  all three upstream encoders regress Gaussian attributes from a distinct
  full-resolution head input containing refined and upsampled encoder features.
- Paper basis: Section 2 states that predicted depth and encoder features are
  regressed into per-pixel Gaussian attributes; Sections 3--4 place the SAES
  decision and parameter prediction in S3. The input to the executed Gaussian
  head is therefore a source-derived candidate and requires no table-fitted
  constant.
- Fixed conditions: canonical TranSplat/Re10K sample 0, seed 0, checkpoint,
  context/target views, `tau_f=0.2`, `tau_d=0.1`, tile size 4, materialization,
  FSDR, renderer, and all quality metrics remain unchanged.
- One-factor change: capture the real `to_gaussians` pre-hook input and select
  it only through a diagnostic `--saes-feature-source` option. The public
  default remains `pipeline`; claim and Functional runs must reject the
  diagnostic source and generated evidence must set
  `paper_result_eligible=false`.
- First gate: the captured tensor must be full resolution, have a recorded
  channel count/source, cover every context view, and yield finite raw/unit
  probe-vector statistics without fallback.
- Promotion gate: the fixed sample must pass PSNR <= 0.15 dB, SSIM <= 0.005,
  and LPIPS <= 0.005 with a non-degenerate first-hit split. Otherwise preserve
  the result and do not launch six-pair pilots.
- Strongest alternative: if the head input still fails, the mismatch lies in
  sparse materialization or unpublished feature normalization rather than the
  selected tensor alone.
- Result: refuted at clean commit `5cee926`. The captured tensor was the
  authentic full-resolution `[1,2,163,256,256]` Gaussian-head input, and the
  schema-valid diagnostic preserved scene `5aca87f95a9412c6`, context views
  `[58,133]`, target views `[84,102,129]`, and no simulator fallback. The
  executed current decision still produced L0/L1/Full rates of
  29.2%/0.0%/70.8% because all 8,192 channel-standard-deviation scores were
  below `tau_f=0.2` and the Gaussian cross-check remained the effective L0
  gate.
- Quality result: SAES-only changed PSNR by -0.8090 dB, SSIM by -0.027802,
  and LPIPS by +0.066505. At the smallest target-free retained fraction,
  4.2969%, PSNR and SSIM were within tolerance but LPIPS still changed by
  +0.019034. Every coverage, component-attribution, and retention-boundary
  variant failed at least one unchanged tolerance.
- Failure recovery: v1 exposed FSDR reading the diagnostic SAES tensor, and v2
  exposed one stale retention-boundary variable. Both failures have dedicated
  regression tests; v1/v2 logs remain preserved, while the complete v3 result
  is under
  `outputs/ae_failures/diagnostics/transplat-re10k-saes-gaussian-head-feature-v3/`.
- Decision: do not promote the Gaussian-head source and do not resume the six
  quality pilots. The full-resolution source changes the raw feature
  distribution but does not recover either the paper path split or sparse
  quality, so the remaining mismatch is not feature-source selection alone.

## 15. Hardware-Honest Probe-Spread Audit

- Run ID: `saes-probe-spread-v1`.
- Research question: does the AE rewrite's oracle access to complete non-probe
  Stage-3 Gaussians hide the intended early-materialization semantics? The
  paper says non-probe adaptor execution is bypassed, while the current
  representative path reads their means, covariances, harmonics, and opacities.
- Source basis: repository commit `adc7092` explicitly aligns the simulator to
  the SCARF specification and implements probe-only covariance expansion from
  assigned 2D pixel territory. The final Dataflow figure likewise routes L0/L1
  from probe Gaussians through soft assignment and aggregation before S4.
- Fixed conditions: canonical TranSplat/Re10K sample 0, seed, checkpoint,
  context/target selection, default current path decisions, thresholds,
  feature source, FSDR, renderer, and PSNR/SSIM/LPIPS tolerances.
- One-factor change: add a diagnostic-only materialization that reads S1
  features, probe depths, pixel coordinates, and probe Gaussian outputs; it
  leaves probe means/SH/opacity unchanged, adds the author-history 2D territory
  covariance spread, and removes non-probe opacities. It must be invariant to
  arbitrary changes in non-probe Stage-3 attributes.
- Evidence boundary: claim and Functional modes reject the diagnostic, and its
  result sets `paper_result_eligible=false`. Historical bandwidth parameters
  are named, provenance-recorded configuration values from `adc7092`, not
  fitted against Table 1 or Table 3.
- Promotion gate: the fixed sample must pass all unchanged quality tolerances
  and avoid a degenerate L0/L1/Full split. Failure preserves the result and
  closes this materialization route without a threshold or parameter sweep.

## 16. Schema-v2 And Public-Evidence Milestone

- Run ID: `ae-v2-lsh-rtl-pilot`.
- Research question: can every Figure 10/Table 2/Table 3/Figure 12 primitive be
  generated from executed events, while binding software and RTL FSDR to one
  projection contract and keeping public physical proxies separate from the
  paper's commercial TSMC28 targets?
- Implementation result: yes for the evidence plumbing. The 13-result catalog,
  strict result schema, worst-view retention, exact candidate/traffic counts,
  SAES S2 evaluation counts, MMCU slot events, hierarchical Table 4, guarded
  Figure 9 counterpart, and workload-bound DRAM interface are implemented and
  covered by tests.
- LSH result: seed-42 normalized hyperplanes are stored as FP16 with matrix
  SHA256 `a9d3431fca57f8408237281c60f3bf3fbe572289abd0e6be3b05824afc428397`.
  Python and RTL decode normalized FP16 operands exactly to signed Q1.24. The
  RTL uses 16 MACs across 128 projection cycles and passes basis-vector
  signature equivalence (`ffff`, `e84b`, `3ee8`).
- Pilot result: `outputs/ae_v2_lsh_rtl_pilot` passes strict sample and aggregate
  validation. It records FSDR Top-1 `7602/7794`, `300352/1048576` depth
  evaluations, `76890112/268435456` feature bytes, SAES S2
  `376576/1048576` evaluations, and nonzero S1-S3 MMCU events. The synthetic
  fixture remains `paper_result_eligible=false`.
- Resume result: changing source after the first pilot changed
  `source_tree_sha256`; the second invocation reran the sample rather than
  accepting stale evidence. Figure 10 manifests were rebuilt from the complete
  result-write crash window and all selected artifacts are hash-bound.
- Verification at this milestone: CPU/schema `240 passed, 14 skipped`; locked classic `298
  passed`; Chisel `9 passed`; SystemVerilog emit and Verilator lint PASS.
- Claim decision: implementation readiness does not promote Results
  Reproduced. The six accessible real pairs must be rerun with this LSH
  contract, DL3DV and Orin evidence are still absent, Figure 12's old heuristic
  bars may contradict executed-slot measurements, and Figure 9/Table 4 require
  workload activity plus a routed ASAP7 run. No tolerance or generated result
  is changed to force acceptance.
- Physical decision: no Vivado worker was active at the latest check, but
  `MemAvailable=13.19 GiB`; keep the 48 GiB guard and do not launch or shrink
  iFlow.

## 17. Seed-42 Six-Pair Mechanism Pilot

- Run ID: `ae-v2-six-pair-pilot-v2`.
- Command: `run_ae.sh mechanisms --pairs transplat/re10k,transplat/acid,`
  `mvsplat/re10k,mvsplat/acid,depthsplat/re10k,depthsplat/acid`
  `--num-samples 1`.
- Evidence: `outputs/ae_v2_six_pair_pilot_v2/`. Every sample and aggregate
  passes the v2 schema, uses the canonical upstream index, and has the same
  source-tree SHA256
  `d092c6bd305ee1280f82af944de86ffc91224898e716679f4f3e6c9bbf81ffeb`.
- FSDR result: Guided Rate is 86.51%--94.96%, above every corresponding paper
  target. Re10K Top-1 Coverage is 96.06%--98.46% and fails the 0.02 absolute
  gate; ACID is 99.14%--99.97% and is closer, but its guided set still fails.
- SAES result: every pair routes zero tiles to L1. L0 ranges from 21.00% to
  48.50%, and the combined output changes PSNR by -0.5624 to -5.0876 dB,
  SSIM by -0.03129 to -0.12109, and LPIPS by +0.04468 to +0.27226.
- Figure 12 result: executed-slot utilization is 85.74%--90.38% for S1,
  82.84%--90.66% for S2, and 84.51%--85.94% for S3. These measurements are
  retained even though they differ from the manuscript's heuristic bars.
- Decision: no full quality, mechanism, ablation, or sensitivity matrix may
  start from this implementation. The next implementation change must explain
  the zero-L1 structure and preserve quality without reading paper targets;
  rerunning or adjusting tolerances is not an acceptable retry.

## 18. L1 Lightweight-Path Audit

- Run ID: `transplat-re10k-l1-lightweight-v1`.
- Structural correction: L0 retains the paper's K(T) representatives, while L1
  now retains 2K(T) deterministic farthest-point anchors. At T=4 these are
  4/16 and 8/16 paths. This is the only retention split consistent with the
  manuscript's statement that L1 is less compressive and with Table 3's
  reported L0/L1/Gaussians-Saved arithmetic.
- Isolation: anchor selection uses tile coordinates only. The audit keeps the
  previously isolated raw-probe-vector/absolute-depth first-hit decisions,
  canonical sample, checkpoint, seed, images, renderer, and tolerances. It is
  recorded with `paper_result_eligible=false`.
- Result: schema PASS, L0/L1/Full=3.796%/86.267%/9.937%, and 46.0% of Gaussians
  removed. Relative to the prior four-anchor first-hit audit, the final PSNR
  loss improves from about -3.17 dB to -2.6195 dB, but SSIM still changes by
  -0.05722 and LPIPS by +0.11396.
- Decision: retain the structurally correct L1 implementation and its tests,
  but reject the first-hit interpretation for claim execution. Do not run it
  across the six pairs; the remaining blocker is the unpublished decision
  statistic/normalization and the sparse materialization quality contract.

## 19. Historical FSDR Depth-Guard Audit

- Run ID: `transplat-re10k-fsdr-depth-guard-v1`.
- Source basis: commit `adc7092` contains a 5% local depth-consistency guard,
  while the final paper/RTL and current claim path route every Hamming hit. The
  guard is therefore exposed only by `--fsdr-guidance-policy
  historical-depth-guard`; claim and Functional modes reject it.
- Fixed conditions: canonical TranSplat/Re10K sample 0, seed-42 FP16/Q1.24 ROM,
  32 cache entries, Hamming threshold 3, D/4 candidate window, exact discrete
  Top-1 evidence, and unchanged quality metrics. SAES is disabled to isolate
  the FSDR decision.
- Result: schema PASS and `paper_result_eligible=false`. Guided Rate decreases
  from 89.00% to 53.16%; exact Top-1 Coverage increases from 96.06% to 99.38%.
  PSNR changes by +0.00010 dB, SSIM by -0.000137, and LPIPS by +0.000444, all
  within the existing quality tolerances.
- Decision: the historical guard explains the quality/Top-1 side of the paper
  result but misses its 72.1% Guided Rate by 18.94 percentage points and is not
  represented by the submitted RTL. Preserve the diagnostic; do not tune its
  5% threshold or silently promote it to the claim path.

## 20. Three-Badge Closure Experiment

- Run ID: `three-badge-faithful-engineering-v1`.
- Research question: can the missing implementation details in the published
  FSDR/SAES equations be fixed once on a disjoint training calibration set so
  the unchanged nine-pair result gates pass naturally?
- Fixed baseline: seed-42 FP16/Q1.24 projection, upstream checkpoints and
  evaluation indices, current quality metrics, event-derived cycles, and all
  tolerances in `artifact/expected_results.json`.
- Permitted changes: cost-volume feature binding, L2/relative-depth statistic
  normalization, one conservative FSDR hit-validity tolerance, the published
  bilateral/depth-reliability bandwidths, camera-aware moment conservation,
  and a probe-constrained `2K(T)` L1 path.
- Forbidden changes: expected-result access outside validators, evaluation-set
  calibration, target/GT routing, pair-specific parameters, projection seed
  search, edited generated JSON, reference cycles, or tolerance changes.
- Minimal gate: synthetic properties, one TranSplat/Re10K calibration sample,
  six one-sample pilots, then a fixed 32-scene six-pair gate.
- Main gate: reviewer profile over all nine pairs followed by the full upstream
  protocol. Orin Figure 8 is completed only by a real independent evaluator.
- Stop condition: a candidate that requires a new routing signal or evaluation-
  target fitting is rejected and preserved as a diagnostic; it is never
  promoted to recover a paper number.

## 21. Paper Assignment-Formula Audit

- Run ID: `transplat-re10k-paper-formula-v2`.
- Research question: does implementing the explicit Section 3 bilateral spatial
  modulation and L1 probe-depth reliability recover sparse-SAES fidelity without
  introducing a new routing signal?
- Fixed conditions: canonical TranSplat/Re10K sample 0, pipeline S1 features,
  `tau_f=0.2`, `tau_d=0.1`, tile size 4, seed 0, same checkpoint, same target
  views, original renderer, and unchanged quality gates. The run is diagnostic
  only and does not read expected results during execution.
- Implementation: the spatial logit now multiplies the normalized squared
  pixel distance by tile `sigma_feat^2`, and L1 uses the paper's per-probe
  `exp(-|d_p-mean(d)|/(beta_d*std(d)+epsilon))` reliability factor. The claim
  path also uses absolute probe-depth standard deviation.
- Result: schema-v2.1 execution completed without fallback. It routed
  8,191/8,192 tiles to L0 and 1/8,192 to L1; SAES-only PSNR changed by
  -6.3895 dB, SSIM by -0.19536, and LPIPS by +0.21078. Combined output changed
  PSNR by -6.4047 dB. The unchanged quality gate fails decisively.
- Decision: retain the formula correction and its unit tests because it matches
  the manuscript, but do not promote it, adjust thresholds, or run the six-pair
  matrix. Continue only with the preregistered evaluation-disjoint calibration
  once official training data is available.

## 22. DL3DV Repair-First Recovery

- Run ID: `dl3dv-repair-first-v1`.
- User priority: do not start, resume, or download any non-DL3DV dataset while
  this recovery line is active. The already prepared official DL3DV trees and
  pinned checkpoints are the only permitted data inputs.
- Baseline: strict one-sample DL3DV executions for TranSplat, MVSplat, and
  DepthSplat under `outputs/ae_dl3dv_repair_baseline/`. They are diagnostic
  only because the global calibration configuration is still preregistered.
- Observed failure: normalized probe-vector variance with the fixed
  `tau_f=0.2` routes nearly every tile to L0 (100.0%, 99.9%, and 100.0%),
  producing SAES+FSDR PSNR losses of -16.08, -16.99, and -8.35 dB. A raw-vector
  TranSplat diagnostic instead gives L0=0%, L1=39.5% and still loses -12.81 dB.
  Therefore neither representation may be promoted by switching a flag.
- First repair gate: prove, with hooks and tensor-equality tests, the source,
  layout, resolution, and numerical equality of the feature tensor that each
  upstream model actually supplies to its cost-volume matcher. DepthSplat must
  cover every used `features_mv` scale. Separately prove that DepthSplat's
  ASIC-no-opt/GGU path is numerically equivalent to its unmodified upstream
  Gaussian output before assessing SAES.
- Second repair gate: implement only a paper-supported probe statistic and
  measurement scale; add synthetic properties for variance, routing order,
  C2W moment construction, covariance PSD, opacity/transmittance, SH, and
  camera geometry. The repair must not inspect target RGB, expected results,
  or evaluation aggregates during routing.
- Execution ladder: unit/property tests -> three strict DL3DV one-sample
  reruns in new output directories -> all three mechanism/quality gates on
  those samples -> only then the fixed official 140-scene DL3DV protocol.
  Every failure remains preserved; no result JSON, tolerance, sample selection,
  or paper target is edited to make a row pass.
- DepthSplat no-opt repair result: the old run mixed approximate hardware S2
  depth with the original S3 Gaussian head. The repaired run uses one pinned
  upstream `MultiViewUniMatch` execution for final depth, density, the first
  cost-volume feature scale, and the S3 head inputs while retaining hardware
  stage cycles. Its `ASIC (no opt)` DL3DV sample now equals the GPU baseline at
  35.58808 dB / 0.973352 SSIM / 0.036621 LPIPS; the prior no-opt output was
  33.94345 dB / 0.966408 / 0.046635. The remaining failure is SAES routing and
  sparse quality, not the base DepthSplat numerical path.
- Cost-volume feature-contract result: the live inner matching calls are now
  captured and checked bitwise on DL3DV sample 0. TranSplat's 128-channel
  feature (`a8ae...dc80`) reaches `DepthPredictorTrans.match_two`; MVSplat's
  128-channel feature (`3d25...7bd4`) reaches its warped cost-volume path;
  and DepthSplat's two live scales (`1ea0...73f0`, 128x32x56; and
  `215f...aef8`, 64x64x112) reach the corresponding cost-volume stages. The
  records are preserved in `outputs/ae_dl3dv_feature_contract/` and show that
  the remaining `tau_f` contradiction is not caused by selecting an inactive
  feature tensor. These native-loader diagnostics record that target RGB was
  loaded by the ordinary evaluation dataloader but do not move it to the model
  or read it in the feature hook/statistics; they are not calibration evidence.
- Normalized-standard-deviation notation audit: this fixed, non-claiming
  alternative interprets the normalized probe-vector reduction as
  σ_feat rather than σ_feat^2 while retaining `tau_f=0.2`, all other
  defaults, the same sample, and no target-driven routing. It produces
  TranSplat L0/L1/Full = 12.9%/34.3%/52.8%, but loses 13.4084 dB PSNR on
  SAES-only and 13.4674 dB combined. The result at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_normalized_probe_std_v1/`
  fails the unchanged quality gate, so the branch stops before MVSplat,
  DepthSplat, or a 140-scene run. The immediate defect remains sparse
  materialization fidelity, not merely the feature-statistic scale.
- Probe-only geometry repair: L1's synthesized anchors previously caused the
  later moment matcher to read their native non-probe depth values. The repair
  propagates a probe-derived virtual depth vector instead. C2W interpolation
  now also carries the assignment-weighted probe residual from the depth-only
  ray, preserving adaptor-predicted sub-pixel offsets. Target-free properties
  cover the no-non-probe-depth rule, ray-offset equivariance, covariance PSD,
  constant SH, and global proxy transmittance. The locked classic profile
  passes 58 targeted tests. A single fixed rerun at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_geometry_v1/`
  improves SAES-only PSNR from -13.4084 dB to -13.3514 dB but still fails the
  unchanged gate; it is preserved as non-claim evidence and does not open the
  MVSplat, DepthSplat, or 140-scene stages.
- L1 retained-output repair contract: the event model already charges 2K(T)
  L1 anchors as executed S2/S3 work, but the former software path synthesized
  the latter K anchors from the primary K probes. A target-free full-Stage-3
  oracle on the fixed TranSplat/DL3DV sample confirms that this is a material
  approximation: virtual-anchor covariance error has p50 0.3541 versus 0.2367
  for primary probes, with larger harmonic and opacity errors as well. The
  next one-factor repair therefore preserves the 2K deterministic positions
  as selected native L1 outputs after the probe-only route has chosen L1. It
  may read S2/S3 only for those charged selected anchors; all remaining tile
  positions remain unread and are absorbed by the same C2W-aware moment match.
  It must retain thresholds, routing order, selections, quality tolerances,
  and target isolation. Gate order is synthetic no-unretained-read/event tests,
  a new target-free audit, then one fresh TranSplat/DL3DV quality diagnostic.
  If either audit or quality does not improve, preserve the output and do not
  broaden to MVSplat, DepthSplat, or 140 scenes.
- L1 retained-output result: the target-free audit passes its directional gate
  (L1 full-oracle covariance p50 0.2968 -> 0.2333; SH/mean/opacity p50 each
  improve by more than 58%). The fresh strict TranSplat/DL3DV quality run is
  preserved under
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_l1_native_quality_v1/`.
  SAES-only PSNR improves from 23.7823 to 24.2742 dB and combined PSNR from
  23.7508 to 24.2378 dB, but both remain roughly 10.6 dB below the unchanged
  no-opt reference. This is a real structural improvement, not a pass and not
  a basis to expand the protocol. The next bounded diagnostic is attribution
  only: fixed mass-conserving covariance/opacity pairs plus component restores
  on this same sample, recorded as non-claim outputs. It may identify a missing
  conservation invariant but may not select a new runtime parameter, alter the
  router, or relax a quality tolerance.
- Engineering-detail decision: `micro59-submit/Sections/section3.tex` fixes
  L1's probe-constrained *routing* and aggregation principle but leaves its
  lightweight retained-output count unspecified. Per the author's direction,
  the implementation therefore declares 2K deterministic native L1 anchors as
  an engineering detail: K primary probes make the L1 decision; only after it
  passes are the extra K farthest-point anchors executed, counted, and used in
  moment matching. This is not an evaluation-tuned choice: it resolves the
  prior mismatch between 2K event accounting and virtual attributes, improves
  target-free errors, and is covered by no-unselected-read and event-count
  properties. It remains non-claiming until all unchanged quality gates pass.
- Optical-depth conservation diagnostic: the fixed target-free audit at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_optical_depth_v1/`
  exposed a missing alpha-compositing invariant: representative L0/L1 output
  retained only 25.2%/50.0% of full-tile optical depth at the median. The
  one-factor `transmittance-diagnostic` variant kept the router, thresholds,
  2K L1 anchors, assignments, and event counts fixed, then accumulated only
  selected-anchor inferred optical depth. Its target-free audit reduces the
  L0/L1 relative-error medians to 4.94%/1.42% under
  `transplat_sample0_transmittance_audit_v1/`. The predeclared quality retry
  improves SAES-only PSNR 24.2742 -> 27.0397 dB and combined PSNR
  24.2378 -> 26.8889 dB, but still loses 7.9491 dB (22.8173%) against the
  unchanged no-opt reference. The output is preserved at
  `transplat_sample0_transmittance_quality_v1/` and rejected: opacity-mass
  conservation is necessary but not sufficient, and high-opacity tiles also
  expose finite-alpha saturation. No threshold, sample, or metric changed.
- Next representation contract: the historical `dense-diagnostic` is not a
  viable repair because it leaves skipped Gaussian means resident from the
  full S3 tensor. A new non-claim `virtual-reconstruction-diagnostic` instead
  reconstructs every skipped descriptor from selected anchors only: C2W-ray
  means with transported residuals, intrinsic PSD covariance, interpolated SH,
  and interpolated alpha. It retains every output descriptor and therefore
  explicitly reports zero Gaussian compression plus a virtual-materialization
  descriptor count; its fixed-function cycle and traffic model remains
  unclaimed until separately implemented. It is barred from claim and
  Functional runs. Synthetic L0/L1 tests
  prove constant preservation, selected-anchor-only access, C2W geometry, and
  PSD. Its next gate is a fresh target-free DL3DV attribute audit, followed by
  exactly one fixed quality run only if that audit improves direct per-pixel
  reconstruction error.
- Virtual-reconstruction result: the direct target-free audit passes its
  structural checks at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_virtual_reconstruction_audit_v1/`:
  it reconstructs 35,184 positions, keeps all 131,072 output descriptors,
  has zero PSD violations, and never transfers target RGB. Its one fixed
  quality run is preserved at
  `transplat_sample0_virtual_reconstruction_quality_v1/`. SAES-only/combined
  PSNR reaches 32.2223/31.9587 dB, a +7.9481/+7.7208 dB improvement over the
  native-anchor sparse path, but it still loses 2.6157/2.8794 dB against the
  unchanged no-opt reference. Thus virtual coverage is a necessary
  representation clue, not a pass, and cannot be used for a compression or
  cycle claim. The unrun geometric/level-aware virtual branch is excluded:
  reconstructing and retaining every skipped descriptor conflicts with the
  paper's sparse-representative semantics. The next target-free work is a
  paper-compatible audit of the native K/2K representative merge, restricted
  to its existing moments, opacity/transmittance handling, and C2W geometry;
  it must not add descriptors, alter the three-level router, or select
  evaluation-dependent parameters.
- Second-moment candidate (rejected): a synthetic conservation property tested
  whether each inferred non-probe should carry the full between-anchor mixture
  covariance before the existing representative merge. The property held, but
  the fixed target-free TranSplat/DL3DV audit at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_second_moment_audit_v2/`
  worsened full-oracle covariance p50 from 0.4203 to 0.6317 (L0) and 0.2333
  to 0.2638 (L1). It neither reads target RGB nor renders quality, so it is
  retained as a negative diagnostic and the native local-shape covariance
  merge remains the main path.
- Depth-scaled covariance candidate (rejected): the upstream adaptor's
  depth-squared covariance law passes its constant-local-shape property, but
  the same target-free audit at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_depth_scaled_covariance_audit_v1/`
  changed L0/L1 oracle covariance p50 only from 0.420318/0.233308 to
  0.420313/0.233323. This is not a material or consistently positive change,
  so it is preserved as negative evidence and does not receive a quality run.
- Corner-depth geometry candidate (rejected): a post-hoc full-S2 audit found
  that bilinear interpolation of the four existing corner depths reduces
  depth and attribute-oracle error. A single unchanged TranSplat/DL3DV
  quality run at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_corner_depth_geometry_quality_v1/`
  nevertheless reduced combined PSNR from 24.2378 to 24.2147 dB and increased
  LPIPS from 0.25121 to 0.25221. The runtime path was restored to
  assignment-weighted probe depth; the full-S2 calculation remains only a
  post-hoc diagnostic and is not used for routing, materialization, or
  parameter selection.
- L1 primary-probe merge candidate (rejected): the bounded
  `dl3dv-l1-primary-merge-v1` diagnostic kept the 2K native L1 outputs but
  let only the K routing probes absorb skipped positions. Its local skipped
  covariance error improved, but a new post-hoc tile-mixture oracle gives a
  fair comparison of the two legal groupings. Against the same native-2K
  baseline, L1 covariance p50 improved from 0.129934 to 0.107015, while mean,
  SH, and average-opacity p50 worsened from 0.000404/0.008279/0.003373 to
  0.000638/0.011399/0.005775; optical-depth error was effectively unchanged.
  The outputs are preserved at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_l1_primary_merge_audit_v2/`
  and `transplat_sample0_l1_native_global_oracle_v1/`. The implementation was
  removed and no quality run was launched. The generic tile-mixture oracle is
  retained as a post-hoc, target-free audit aid for future paper-compatible
  merge candidates.
- S3-before-S4 raw-descriptor interpolation oracle (completed, scope
  corrected):
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_s3_before_s4_audit_v1/`
  executes S1/S2/the raw S3 descriptor head, but not the baseline encoder,
  GGU/S4, decoder, or any quality metric. It removes target RGB before device
  transfer and records that the native loader had supplied it. The audit maps
  TranSplat's 64x64 matching features to the 256x256 S2/S3 grid via the same
  bilinear, `align_corners=False` mapping as `ProgressiveSAES`.

  Assignment interpolation improved all L0 p50 values, but it worsened L0
  p95 depth (0.307220 vs 0.213901), raw scale (1.400259 vs 0.901987),
  rotation (0.358311 vs 0.305182), SH (1.013573 vs 0.989953), and opacity
  (0.212972 vs 0.178449) against deterministic nearest-anchor reconstruction.
  This rejects only naïve *per-skipped-position raw descriptor interpolation*:
  the audit does not perform a retained-anchor first/second-moment update, does
  not output retained descriptors, and therefore cannot validate or refute the
  separate post-GGU primitive moment path. No generic pre-GGU quality retry is
  justified, but the result must not be treated as evidence against an actual
  paper-compatible aggregate implementation. L1 has encouraging robust deltas
  but cannot rescue an L0/L1-wide change; retain it only as diagnostic evidence
  unless a separately specified, paper-compatible L1-only hypothesis clears its
  own property and event-accounting gates.

- L1 primary-depth-reference correction: the paper defines the L1 reliability
  normalizer over the primary routing probes, while the 2K native L1 anchor
  expansion is an engineering detail. The former non-claim
  `l1-primary-depth-reference-diagnostic` therefore represented the required
  semantics and is now a legacy alias of the normal implementation: all normal
  paths retain and charge 2K anchors but normalize their depth reliability with
  the original K probe mean and standard deviation. The historical target-free
  result at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_l1_primary_depth_reference_audit_v1/`
  improves L1 tile-mixture covariance p50 (0.129935 -> 0.121641) and mean p50
  (0.000404 -> 0.000351), but worsens harmonic p50 (0.008279 -> 0.008306) and
  opacity-average p50 (0.003373 -> 0.003485). It fails the all-attribute gate;
  no quality retry is allowed. The fresh target-free structural audit at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_l1_primary_reference_corrected_audit_v1/`
  (SHA256 `c93a6bc94e7e20098333a438f0256574c5946745d74fc0c4cf9a0fae26aefea1`)
  confirms no target-RGB transfer, no skipped-S3 read, PSD-safe output, stable
  events under poisoned skipped descriptors, and nonzero sparse work. It is
  not an attribute-quality pass and cannot reopen the historical quality gate.

## 23. DL3DV SAES Routing and Hardware-Cost Closure

- Routing-coordinate result (rejected): the fixed target-free audit at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_routing_coordinate_audit_v1/`
  confirms that TranSplat searches candidate depth in normalized inverse-depth
  coordinates upstream but emits metric depth before SAES. On the fixed sample,
  the existing normalized-feature / metric-depth route yields L0/L1/Full =
  12.9%/34.3%/52.8%; relative metric depth would route 92.8% to L1 and the
  candidate-coordinate statistic 99.98%. None is a paper-supported,
  generalizable replacement, so no unit switch may be fitted to Table 3's
  average L1 rate.
- Hardware-accounting contract: L0/L1 bypasses must include (i) probe feature
  variance, (ii) L0-miss probe-depth standard deviation, (iii) the submitted
  `SAESController` first-hit FSM transitions, (iv) assignment softmax and
  moment matching, and (v) retained-descriptor buffer traffic. A nonzero SAES
  saving without this ledger now raises an error in `SavingsTracker`.
- Implementation: `saes/hardware_accounting.py` derives a deterministic event
  ledger from executed tile/anchor counters and the exact runtime feature,
  tile, SH, and primitive dimensions. It reports all required traffic but only
  charges retained-descriptor reads/rewrites and route records as incremental
  storage traffic, avoiding an unproven double-count of baseline S1/S2 reads.
  Its explicit no-overlap cycle sum is deliberately labelled
  `analytic_no_overlap_not_rtl_cycle_equivalent`; result records preserve it
  under `events.saes.hardware_accounting` when present.
- RTL route-event consistency: `SAESController` now exposes
  `decisionCycles`, and `ScarfTop` exposes that valid-on-done event for VCD
  reconciliation. Chisel proves L0 = 2 cycles and L1/Full = 3 cycles from an
  accepted start, matching the controller portion of the Python ledger; the
  emitted SystemVerilog includes the signal. This proves only classification
  control timing, not assignment, moment matching, or descriptor-buffer timing.
- Fixed real audit: the target-free TranSplat/DL3DV sample-0 run at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_saes_hardware_accounting_v3/results.json`
  has SHA256
  `3f3334e5c5a5cea73d1c1e169ee12f6bd3ac7a7c3ed380851af73257e22ec9d4`.
  The native loader had loaded target RGB, but the diagnostic removed it before
  device transfer; baseline encoding, rendering, quality metrics, and expected
  result access are all recorded as false. It executed S1--S4 only.
- Result: this default-route trace classified all 8,192 tiles as L0, retained
  32,768 anchors, and bypassed 98,304 positions. The ledger charges 4,293,120
  serialized analytic cycles (2,457,600 assignment, 851,968 moment matching,
  770,560 storage, and 212,992 decision cycles) and 45,883,392 bytes of
  required traffic. Using the same stage-cycle inputs, the illustrative
  SAES-only analytic speedup changes from 2.0331x with an invalid zero-cost
  assumption to 1.9207x after the charge. This is not a paper performance
  result, because the sparse quality gate still fails and the ledger is not
  RTL timing evidence.
- Failure preservation: the first v2 audit reached the same target-free SAES
  counters but failed before result writing because the audit-local GGU counter
  was initialized in the wrong branch. The failure is retained in
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_saes_hardware_accounting_v2/FAILED.md`;
  the v3 retry changed only that local initialization.
- Decision: this closes the zero-cost-accounting defect but does not promote
  any DL3DV claim or authorize the 140-scene run. Before a cycle-equivalent
  Figure/Table claim, implement SAES assignment/moment matching and retained
  descriptor buffering in RTL, establish event-by-event software/RTL agreement,
  and then rerun the fixed single-sample quality gate from a new directory.
- RTL safety repair: review exposed that the old `sS2S3_ProbeOnly` controller
  state performed one bilinear operation and entered GGU without executing the
  required probe S2/S3 path or any SAES assignment, moment matching, and
  retained-descriptor buffering. It was therefore neither a correct sparse
  implementation nor a valid hardware-saving path. The state is removed: L0
  and L1 now conservatively follow the ordinary FSDR (when enabled), CostVol,
  U-Net, depth head/regression, S3 refine, and Gaussian-head sequence. Chisel
  regressions cover both sparse levels. This is a Functional-safety correction
  only; it deliberately contributes zero SAES RTL savings until the missing
  numeric datapath and buffer are implemented and reconciled to software events.
- Descriptor-buffer increment: `SAESDescriptorBuffer` is a standalone emitted
  Chisel `SyncReadMem` primitive with 32 slots, ordered write completion, stale
  descriptor invalidation, and synchronous 128-bit reads. The packed layout is
  shared with `saes.hardware_accounting`: degree-2 and degree-4 descriptors
  require 92/188 bytes or 6/12 beats. Unit tests cover both layouts and reject
  out-of-range, skipped, and premature-terminal writes. The module is not yet
  connected to `ScarfTop` or a numeric merge core; it is therefore Functional
  plumbing only and does not change any SAES cycle/PPA claim.
- Scalar-moment increment: `SAESScalarMomentAccumulator` now implements the
  existing first/second-moment sufficient statistics over caller-defined signed
  fixed-point units and nonnegative assignment weights. Its target-free Python
  reference and Chisel tests agree on an exact merge vector and constant
  preservation. It has no feature, depth, tile, or target input, is not a new
  SAES decision, and remains unconnected to descriptor packing, bilateral
  assignment, and `ScarfTop`; it does not authorize a timing or quality claim.
- Assignment-normalization increment: `SAESAssignmentNormalizer` now maps one
  to eight already-computed nonnegative bilateral scores into an exact Q0.16
  distribution, placing finite-precision residual mass on the lowest-index
  maximum score. The target-free Python reference and Chisel vectors cover the
  residual, ties, and invalid zero mass. This declares only the quantized
  normalization boundary; feature/depth score formation, descriptor merge, and
  top-level scheduling remain unconnected, so it creates no route, cycle, PPA,
  or quality claim.
- S2/S3 dependency gate: the fixed target-free
  `transplat_sample0_s2s3_dependency_audit_v1` result (SHA256
  `bcbde72ec4029d3b32465ff920df8746a8f636e4383d19f161e5fa906b80f621`)
  changed every 2,752,512 retained-probe TranSplat raw-head value after zeroing
  only non-probe `refine_unet` inputs. `SavingsTracker` now records an explicit
  per-model execution-dependency contract: TranSplat is blocked by that
  counterexample and MVSplat/DepthSplat lack a positive dependency audit, so
  all current SAES S2/S3 savings are zero. The existing target-free analytic
  control/merge/storage ledger remains a recorded non-RTL penalty, and FSDR
  remains eligible on positions not removed by a *verified* SAES S2 bypass.
  Schema-v2.1 records carry the resolved model contract plus explicit S2/S3
  fractions; `validate_result.py` resolves it again and rejects missing,
  substituted, or nonzero-unverified savings evidence.
  This rejects only direct sparse execution accounting; it neither changes the
  paper router nor authorizes a quality retry or DL3DV expansion.
- Next diagnostic, `dl3dv-transplat-refine-dependency-locality-v1`: run exactly
  the fixed 3x3 raster of source tiles (top/center/bottom by left/center/right)
  on the already pinned DL3DV sample 0. In each separate context-only encoder
  execution, zero only the 12 non-probe `refine_unet` input activations of one
  4x4 source tile and capture the raw S3 head before S4. Report changed
  retained-probe positions and their tile-distance envelope, never target RGB,
  quality metrics, or a selected execution parameter. A result can only bound
  or reject a prospective finite-halo implementation; it cannot grant S2/S3
  savings, alter SAES routing, or authorize a quality retry. A future positive
  path would still require a fixed sparse implementation, full dependency
  proof, software/RTL event agreement, and the existing property gates.
- Result: `transplat_sample0_refine_dependency_locality_v1` completed with
  result SHA256 `bb5c24557a39da455869ebdac8804cf3a9c883f9c0df1592297c051ce2583a56`.
  It accessed context only and stopped at the raw S3 head. Each of the nine
  predeclared source tiles changed every retained-probe spatial position in the
  complete 64x64 tile grid; center-to-probe maximum Chebyshev distance was 32,
  and edge/corner source tiles reached 63. The finite-halo direct-bypass
  hypothesis is rejected for the current unmodified TranSplat refinement path.
  Preserve the diagnostic; do not create a sparse schedule, change a parameter,
  run quality, or expand DL3DV from this result.
- FSDR feature/ROM reconciliation: the fresh target-free
  `depthsplat_sample0_v3` feature contract (SHA256
  `d519d0a7a7bbc966c9ae4333c42e5ab4ca987f3cd813ae9083feae2fac05def8`)
  proves that DepthSplat's first executed cost-volume scale is
  `features_mv[0]` with shape `[1,2,128,32,56]`, exactly matching the
  seed-42 16x128 FP16/Q1.24 ROM. Its second scale is 64 channels and is
  separately recorded. The 1024-channel mono tensor is not a cost-volume
  input; remove its previous `--fsdr-only` claim-path selector so no software
  hash can evade the RTL dimension contract. This closes feature-source
  consistency only, not the unresolved Guided-Rate reproduction gate.

## 24. Native Representative-Merge Analysis Campaign

- Campaign ID: `dl3dv-native-merge-conditional-v1`.
- Parent evidence: `transplat_sample0_l1_native_quality_v1` establishes that
  native K/2K anchors improve over virtual anchors but still fail sparse
  quality by roughly 10.6 dB. This campaign is target-free until a strict
  all-attribute audit clears its predeclared gate.
- Research question: does the existing representative update accidentally
  apply an assignment twice by first forming an unconditional pseudo descriptor
  `g_i = sum_q r_iq g_q` and then adding it to representative `p` with
  `r_ip`? The paper says to merge a non-probe into its corresponding probe;
  it does not require cross-probe `r_ip*r_iq` leakage.
- One permitted intervention: for each selected anchor `p`, construct the
  conditional selected-anchor-only transport `g_{i|p}` from p's own depth,
  C2W residual, intrinsic covariance, SH, and opacity, then use `r_ip` once
  in p's first/second-moment update. The router, K/2K positions, all thresholds,
  feature/depth inputs, assignment weights, output count, and event ledger stay
  fixed. The path remains a non-claim diagnostic until all gates pass.
- Slice A (synthetic property): prove constant descriptors, one-hot assignment,
  selected-anchor-only access, PSD covariance, and no cross-anchor leakage.
  Failure means do not run on DL3DV.
- Slice B (target-free DL3DV audit): same fixed TranSplat sample 0, no target
  RGB, no renderer, no metrics. Compare L0 and L1 p50 and p95 mean/covariance/
  SH/opacity/optical-depth errors against the current native K/2K merge.
  Continue only if every required attribute improves or remains numerically
  unchanged within exact rounding tolerance; a mixed result is rejected.
- Slice C (conditional quality retry): only if Slice B passes, run exactly one
  fresh strict sample with the same command except for the declared diagnostic
  materialization. It remains in a new directory; quality tolerances and
  evaluation identity are unchanged. A failure terminates the campaign and
  leaves the 140-scene protocol closed.
- Result and decision: Slice A synthetic properties pass. Slice B was run once
  on TranSplat/DL3DV sample 0 with target RGB removed before execution, no
  renderer, and no quality metrics. The matched `representative` baseline and
  `conditional-anchor-transport-diagnostic` candidate have identical scene,
  seed, `current` decision semantics, feature statistic, all-L0 route (8,192
  tiles), and 32,768 retained anchors. The baseline results SHA256 is
  `a3b1d19b0312fb493161a2f868220782744b19bcaa21e423c485792ca3cddbf5`; the
  candidate results SHA256 is
  `30876f42d1c9d40a534417247c955ee1e418e45c636952c48a77357612a0c05a`.
  Although mean, SH, opacity, and optical-depth values are numerically near
  unchanged or locally better, L0 tile-mixture covariance worsens from p50/p95
  `0.235292/0.754145` to `0.348608/0.891179`. This is a mixed required
  attribute result, so the candidate is rejected, Slice C is not run, and the
  full DL3DV protocol remains closed. The diagnostic-only implementation and
  both result trees are preserved as negative evidence; no router, tolerance,
  selection, or evaluation-derived parameter was changed.

## 25. Target-Free DL3DV FSDR Gate

- Decision: retain `--fsdr-only --claim-run` as the calibrated, paper-eligible
  Table 2 path. Add the strict `--diagnostic-run --fsdr-only
  --image-output-policy none` path for context-only DL3DV audits. It emits
  `fsdr_target_free_audit`, which the FSDR aggregator rejects by kind and which
  requires target RGB to be removed before device transfer, never passed to the
  model, and never used for routing or metrics.
- Fixed sample-0 result: the same official DL3DV selection completed for all
  three encoders using their required environment profiles. The results are
  target-free execution evidence only: TranSplat `64.1235%` guided and
  `99.7716%` discrete Top-1 coverage (SHA256
  `d93a64b7cb24841df141a2af47bea5494759cfeafe064cd0966f39d5140e8d89`);
  MVSplat `58.4717%` / `99.8539%` (SHA256
  `b613c3cda8bdd18d675ba261673f2c207e281fcd8b2a5124f8e8fffc593ed5c3`);
  and native DepthSplat `43.6942%` / `99.9361%` (SHA256
  `bc401760daf1695d23f4cfdc200eb73dca85d23ff39e9b14292d378af559d1c6`).
- Decision: these records establish the target-free execution and discrete
  candidate boundary for the three DL3DV models. They neither select a
  mechanism configuration nor validate rendering quality, so they cannot
  promote FSDR, SAES, Table 2, or any Results Reproduced claim. The 140-scene
  DL3DV protocol remains closed on the existing SAES quality/dependency gates.

## 26. Multi-Model SAES Direct-Dependency Gate

- Run ID: `dl3dv-s2s3-direct-dependency-v1`.
- Research question: can MVSplat or DepthSplat preserve every selected 4x4
  corner-probe raw Gaussian-head value after its corresponding dense non-probe
  S3 input activation is zeroed?
- Fixed contract: use only sample 0 from the committed DL3DV selection and
  context tensors; capture the raw Gaussian head before S4, rendering, target
  metrics, or expected-result access. MVSplat perturbs `refine_unet` input and
  DepthSplat perturbs `gaussian_regressor` input. The tile mask, seed, model
  environment, input selection, and retained-probe definition are fixed.
- Null hypothesis: at least one retained raw-head value changes for each model;
  the dense implementation therefore cannot directly bypass non-probe S2/S3
  work. The alternative only reopens a later sparse-implementation audit; it
  never grants a saving by itself.
- Stop condition: any retained-probe delta records
  `dense_dependency_detected` with the immutable result hash. If no deltas are
  detected, stop before quality and require an independent sparse execution,
  software/RTL reconciliation, and existing SAES property gates.
- Result: the fixed context-only gate rejects direct bypass for all three
  models. TranSplat changed `2,752,512/2,752,512` retained raw-head values
  (SHA256 `8a73027989cfaafce2145b6370d2209450725eb33dfbd7c9afa0e5c5ead82679`),
  MVSplat changed `2,752,511/2,752,512`
  (`ac5610772240887f7f5004c050fbc0ad08261337dd18b02f3430c7cf00174ea5`),
  and DepthSplat changed `2,121,728/2,121,728`
  (`893747ecb7d3336f90b9f7afdf052cd3d946d8728b37ec5ebf558533a4befd76`).
  Every record removed target RGB before context-device transfer and stopped
  before S4, rendering, or quality. No quality retry or DL3DV expansion is
  authorized by this result.

## 27. CPU Clean-Room SAES Layout Separation

- Contract: K(T) primary probes and 2K(T) L1 anchor coordinates are structural
  geometry, not model execution. `saes.probe_layout` therefore implements the
  existing deterministic layout using only the Python standard library. The
  public `ProgressiveSAES` static methods retain their names and delegate to
  this canonical helper, while `saes.hardware_accounting` imports it directly.
- Evidence: fixed T=4 coordinates, L0/L1 cardinality and prefix invariants for
  T={4,8,16}, classic-wrapper equivalence for T={4,5,8,16}, and a subprocess
  that blocks every `torch` import while constructing an event ledger all pass.
  The locked classic SAES/layout, accounting, diagnostic, and result-record
  subset passes 77 tests.
- Boundary: this only removes a CPU schema/accounting clean-room dependency.
  It neither supplies the missing sparse S2/S3 execution path nor changes the
  zero-SAEs-savings fail-closed contract, quality gates, calibration, or DL3DV
  execution schedule.

## 28. Retained-Output Hand-Off Contract

- Contract: the paper's T=4 probe-first S2/S3 order is now made explicit at
  the native-output boundary. L0 requests `[0,3,12,15]`; L1 retains that prefix
  and requests `[5,10,1,2]` before a descriptor can enter the staged buffer.
  Each request must receive an upstream native-descriptor confirmation.
- Boundary: the standalone scheduler is intentionally not connected to
  `ScarfTop`, because all three current upstream encoders failed the direct
  dependency gate. It does not create inputs, values, or a synthetic bypass;
  the dependency contract and zero S2/S3 saving remain unchanged.
- Gate: Python layout/reference tests and Chisel request/backpressure tests
  must pass before this hand-off is used by a real sparse producer. A later
  integration additionally needs descriptor packing, bilateral assignment,
  moment matching, buffer/S4 transfer, per-event replay, and an independent
  model-specific sparse-execution proof.

## 29. Multi-Model Finite-Halo Eligibility Audit

- Research question: after the all-nonprobe direct-dependency failures, does a
  fixed one-tile nonprobe perturbation have a bounded retained-probe envelope
  for MVSplat or DepthSplat? A bounded result would only motivate a later exact
  dependency-aware sparse-producer design; an unbounded result closes the
  finite-halo route for that unmodified model.
- Fixed contract: DL3DV sample 0, official selection and checkpoint, one 3x3
  top/center/bottom by left/center/right source-tile raster, twelve nonprobes
  per source tile, raw-head stop before S4/rendering, and no target metrics.
  Target RGB is removed from the batch before context-device transfer and the
  record must prove this provenance.
- Stop condition: any source tile changing retained probes outside a finite
  recorded envelope rejects a direct local bypass for that model. Neither
  outcome changes routing, calibration, quality tolerances, S2/S3 savings, or
  authorization for a DL3DV quality/full protocol.
- Result: MVSplat's fixed record
  `mvsplat_sample0_refine_dependency_locality_v1/results.json` (SHA256
  `2ecd6206d4af77fb59c06add007257ecba76bbba07008c99195b9a14c93cdee6`)
  has a complete 64x64 retained-probe envelope for every source tile; the
  center reaches distance 32 and edge/corner sources reach 63. The direct
  finite-halo route is rejected, matching TranSplat.
- Result: DepthSplat's fixed record
  `depthsplat_sample0_regressor_dependency_locality_v1/results.json` (SHA256
  `bfce94a62940e7faa08320695486e505c714d7f7a689d1f8d964bc9e19ef7d2c`)
  has a bounded one-tile envelope, but the adaptor-footprint contract proves
  its four 3x3 convolutions require every preceding 256x448 position for both
  L0 and L1. Only final-head emission can be sparse; that is not a genuine S3
  producer and cannot change the zero S2/S3 saving or open a quality retry.

## 30. Sparse-Producer Route Decision

- Verdict: reject the direct sparse-producer line for the three unmodified
  upstream checkpoints. TranSplat and MVSplat have full-grid retained-output
  dependence; DepthSplat's bounded raw-head envelope still requires every
  preceding spatial activation through its four-convolution adaptor. The staged
  retained-output scheduler remains correct Functional plumbing but cannot be
  connected to a genuine source under these contracts.
- Rejected alternatives: a constant/dummy descriptor source, zeroing dense
  activations, final-head-only emission presented as S3 sparsity, or training a
  new surrogate/adaptor. The first three contradict execution evidence; the
  last would be a new model-level method outside the submitted mechanism and
  cannot be represented as a reproduction engineering detail.
- Reopen condition: obtain an author-provided probe-first/sparse-compatible
  adaptor or its exact training configuration and checkpoint, then repeat the
  target-free dependency, software/event/RTL, and unchanged quality gates from
  new outputs. Until then, leave SAES S2/S3 savings at zero, do not run DL3DV
  quality/full retries, and retain Results Reproduced as unclaimed.

## 31. Public Source Discovery

- Scope: inspect the official Git remotes, all advertised branch heads/tags,
  and the SCARF public issue/PR history for a probe-first or sparse-compatible
  upstream implementation before treating the missing producer as an author
  hand-off requirement.
- Result: TranSplat exposes only its pinned `main` commit
  `aaa29a40`; MVSplat's official and local-fork remotes expose the same pinned
  `main` commit `01f9a28`; and the local-fork and official DepthSplat remotes
  expose only `main` (the pinned local revision is `1f5e548`). SCARF's public
  issue/PR history contains the original SAES simulator and later cycle-honesty
  work, but no sparse adaptor, probe-first checkpoint, or producer branch.
- Decision: public source discovery does not reopen the direct sparse route.
  The remaining source requirement is specifically author-provided model code
  and weights, or an exact training artifact that is demonstrably compatible
  with the paper's existing S1/L0/L1/Full mechanism.

## 32. Same-Weight Replay And Optical-Mass Gate

- Same-weight selected-output replay is implemented for the classic
  `Conv3x3 -> GELU -> Conv3x3` Gaussian heads. Clean DL3DV sample-0 audits
  prove that the repeated T=4 corner pattern requires a dense first-convolution
  closure, while the second convolution executes exactly the retained outputs.
  The TranSplat and MVSplat records are target-free, stop before rendering,
  and report a head-only MAC reduction of 25.5061%; they do not permit S2 or
  global S3 savings.
- Conditional optical-density moment transport uses one `r_i,p` per receiving
  anchor, C2W ray transport, `tau * sqrt(det(cov + eps I))` mass, PSD checks,
  and a Full fallback. Its clean target-free TranSplat/DL3DV audit preserves
  retained attributes under a skipped-descriptor poison test and has maximum
  mass error `3.8147e-6`.
- Quality gate: the one fixed, predeclared sample-0 quality run at commit
  `a382ea0` failed decisively (baseline/SAES PSNR `34.8381/7.8144`, SSIM
  `0.97360/0.25523`, LPIPS `0.03276/0.70274`). The record is non-claiming and
  must not be expanded to 8, 32, or 140 scenes.
- Root cause and conservative repair: the target-free range diagnostic found
  all L0 tiles expanded covariance beyond their selected-anchor determinant
  envelope. The no-parameter range fallback at commit `93cba9f` correctly
  sends all `8,192/8,192` tiles to Full and executes all `131,072/131,072` S2
  evaluations. This removes the unsupported sparse result but supplies zero
  SAES benefit, so it cannot support Results Reproduced. Any future reopen
  needs a paper-compatible sparse producer or a distinct target-free mechanism
  correction that preserves nonzero sparse work before another quality run.
- Follow-up correction: the determinant-envelope fallback was an additional
  condition not stated in the paper. The paper constrains averaged SH and
  opacity, while its first/second-moment match necessarily permits covariance
  expansion from transported mean dispersion. Keep the historical all-Full
  audit as negative evidence, but remove that envelope condition from the
  candidate implementation. PSD, finite-value, opacity-domain, single-
  assignment, and optical-mass-conservation failures still return Full.

## 33. DL3DV Coordinate-Explicit SAES Gate

- Hypothesis: the paper's fixed `tau_f=0.20` and `tau_d=0.10` compare probe
  statistics in the encoder's normalized feature and S2 inverse-depth candidate
  coordinates, respectively. This preserves the published L0->L1->Full
  hierarchy and thresholds; it only makes their previously implicit units
  explicit. On DL3DV/TranSplat sample 0, the normalized probe-vector standard
  deviation independently yields a 12.915% L0 rate, close to the paper's 12.0%
  DL3DV row, unlike the current statistic's 100% L0 route.
- Pre-quality gate: run exactly one fresh target-free sample-0 audit with
  `probe-normalized-std-first-hit` and
  `inverse-depth-candidate-coordinate-standard-deviation`. It must retain the
  existing single-assignment, skipped-descriptor poison, PSD, opacity/SH-range,
  and finite-value checks and report nonzero sparse work. It must not render,
  load target RGB into the model, compute metrics, or read expected results.
- Stop condition: if that audit still reaches Full fallback for every selected
  tile or otherwise has zero sparse work, record the failure and repair the
  representative materialization before any new quality run. If it passes,
  pre-register one fresh sample-0 quality gate with unchanged global
  thresholds and tolerance.
- Result: the fixed target-free audit at source commit `8fbcb73` is preserved
  as `transplat_sample0_coordinate_explicit_optical_mass_audit_v1/results.json`
  (SHA256 `fca6d6237ef1839358c2ba11475caa50fe530b5c39de8eb9b734588ea1860936`).
  It proves target RGB was removed before device transfer and does not render
  or compute quality metrics. The normalized feature statistic remains 12.915%
  L0, but the inverse-depth candidate coordinate makes every remaining tile
  pass L1 and the covariance-envelope safeguard returns all 8,192 tiles to
  Full. Zero descriptors are skipped. This candidate fails its nonzero-sparse
  pre-quality gate; no quality run is authorized.
- Next candidate: retain the feature interpretation but use the paper-literal
  metric probe-depth standard deviation for L1. The target-free routing ledger
  records 12.915% L0, 34.314% L1, and 52.771% Full on sample 0 before any
  materialization fallback. Its next audit tests only the removal of the
  unsupported covariance envelope; it does not change `tau_f`, `tau_d`,
  datasets, checkpoints, assignment bandwidths, or tolerances.
- Target-free result: `transplat_sample0_metric_depth_optical_mass_audit_v1/
  results.json` (SHA256
  `27ece34902cada6d20a745295dcf6f8c82bcdeff60b8e3265b7cff68fb417183`)
  passes the pre-quality gate. It is target-free, has 1,058 L0, 2,811 L1, and
  4,323 Full tiles, skips 35,184 descriptors (26.8433%), has zero PSD and
  mass-fallback failures, and is invariant to poisoned skipped descriptors.
- Pre-registered quality gate: run exactly once in
  `outputs/ae_dl3dv_repair_diagnostics/
  transplat_sample0_metric_depth_optical_mass_quality_v1/` with the official
  DL3DV index, TranSplat `re10k.ckpt` SHA256
  `89e43c205a04962e427801385d7d18e74cba063d05a76bf8b28e5fa746a4b69a`,
  seed 0, sample/protocol index 0, all four selected target views, diagnostic
  materialization `conditional-optical-mass-diagnostic`, and feature semantics
  `probe-normalized-std-first-hit`. The default metric-depth L1 statistic and
  global thresholds remain fixed. Accept only PSNR loss <= 0.15 dB, SSIM loss
  <= 0.005, and LPIPS increase <= 0.005. Failure stops this line before any
  8/32/140-scene run.
- Quality result: the pre-registered run completed at the same dirty source
  identity and is preserved as
  `transplat_sample0_metric_depth_optical_mass_quality_v1/results.json`
  (SHA256 `88a78ec60326b008cbd8a8f8ccd1ae71babca78ebf2ca2193fe5660e3e8559c2`).
  Its schema/provenance validation passes, but quality fails decisively:
  baseline/SAES PSNR is `34.8381/10.0265` dB, SSIM is
  `0.97360/0.44955`, and LPIPS is `0.03276/0.51326`. This is an improvement
  over the all-L0 optical-mass failure but exceeds every unchanged tolerance.
  Do not run 8/32/140 scenes. The next permitted action is a target-free
  retained-anchor attribute diagnosis, not another quality retry.
- Attribute diagnosis: `transplat_sample0_metric_depth_optical_mass_attribute_
  profile_v1/results.json` (SHA256
  `5f06fc833de6b5d9c786a48626d7f1510bf26a828e958439d65a07d222a65d04`)
  identifies opacity, not SH, as the failure source. Across 26,720 changed
  retained anchors, output/source opacity ratio has p50 `0.01291` and output
  opacity p50 `0.00403` versus source p50 `0.29721`; covariance determinant
  ratio has p95 `1600.92`. The 3D determinant counts transported depth-axis
  spread as rendering footprint and over-attenuates opacity.
- Next candidate: preserve the one-assignment, C2W transport, first/second
  moment, and optical-density construction, but evaluate the optical footprint
  in the context camera's renderer coordinate system as
  `sqrt(det(J * Sigma * J^T))`. The camera Jacobian uses only the producing
  context view's C2W/intrinsics and retained anchor mean, never target camera
  geometry or target RGB. It adds no route input or tunable parameter. Its
  synthetic projected-footprint properties and a fresh target-free attribute
  profile must pass before another pre-registered quality run.
- First projected-footprint audit: preserve
  `transplat_sample0_metric_depth_projected_optical_mass_audit_v1/results.json`
  (SHA256 `4f23429a8b3917a73aa71a5c85934be89ac295d40be0df2e1bf2198ccc9aa1b4`).
  It is target-free and keeps poison/PSD/mass checks, but only 9 L0 tiles pass
  while 3,860 early tiles return Full. The cause is a `1e-8` determinant floor
  incorrectly reused from 3D covariance checks for valid tiny 2D raster-space
  footprints. Correct that numerical validity bound before judging the
  projected-mass mechanism; do not render quality from this result.
- Corrected projected-footprint audit: preserve
  `transplat_sample0_metric_depth_projected_optical_mass_audit_v2/results.json`
  (SHA256 `b6c6d4949ee5ff6d735caed7ffe153dd4bc76070449868bef7e7ec96bfdeb988`).
  It restores the expected 35,184 skipped descriptors and passes all
  target-free numerical/poison checks, but does not repair the result: changed
  anchor opacity ratio remains p50 `0.01236`. Do not run a quality gate for
  projected optical mass.
- Next candidate: use the paper's explicit range-constrained SH/opacity
  averaging with conditional one-assignment C2W transport and first/second
  moments. This is the existing `conditional-anchor-transport-diagnostic`
  equation, not a new router or parameter. Run a target-free attribute audit
  first; its opacity must remain in the selected-anchor range before a single
  new quality gate may be pre-registered.
- Target-free result: preserve
  `transplat_sample0_metric_depth_conditional_anchor_attribute_profile_v1/
  results.json` (SHA256
  `900c8e968076dee17f16ed2d6ef0540e454a98da1c1266b474670c0020e910f0`).
  It passes all target-free checks with the same 35,184 skipped descriptors.
  Across 26,720 changed anchors its opacity ratio is exactly 1.0, SH change is
  only floating-point roundoff, and mean-displacement p95 is `0.06587`; the
  remaining risk is covariance contraction (determinant-ratio p50 `2.25e-4`).
- Pre-registered quality gate: run exactly once in
  `outputs/ae_dl3dv_repair_diagnostics/
  transplat_sample0_metric_depth_conditional_anchor_quality_v1/` using the
  same official sample/checkpoint/seed/target-view contract as the preceding
  quality gate, but materialization
  `conditional-anchor-transport-diagnostic`. Thresholds and quality tolerances
  remain `0.20/0.10` and `0.15/0.005/0.005`. A failure forbids all DL3DV
  expansion and requires a target-free covariance diagnosis.
- Quality result: preserve
  `transplat_sample0_metric_depth_conditional_anchor_quality_v1/results.json`
  (SHA256 `bde1768c7edd077ee000cab60f632c121928182564237d8788ea0acb7abfd9b1`).
  It passes schema/provenance validation but fails the unchanged quality gate:
  PSNR `34.8381 -> 24.2559`, SSIM `0.97360 -> 0.83307`, LPIPS
  `0.03276 -> 0.24604`. This confirms that opacity preservation improves the
  all-L0/optical-mass failures but does not recover sparse quality. Do not
  expand DL3DV. Diagnose the conditional covariance moment equation before any
  future quality run.
- Covariance diagnosis: preserve
  `transplat_sample0_metric_depth_conditional_anchor_covariance_audit_v3/
  results.json` (SHA256
  `8545e6479c7cd496ceaa8ba4aa395b27e16ea055ef32cd02e52dcad7ab818ed1`).
  The corrected determinant ratio is never below `1.00021` (p50 `3.6828`),
  the covariance increment minimum eigenvalue is positive at p50, and source
  covariances are already symmetric/PSD. The former contraction conclusion was
  a diagnostic denominator-floor error, not a mechanism error.
- Current stop condition: the remaining mismatch is routing prevalence. At
  fixed paper thresholds this sample has L0/L1/Full
  `12.915%/34.314%/52.771%`, whereas the paper's DL3DV aggregate is
  `12.0%/10.1%/77.9%` with 17.0% Gaussian saving. Changing the L1 statistic or
  calibrating a new threshold from this evaluation sample would violate the
  global, target-free contract. Retain Results Reproduced as unclaimed; do not
  run another DL3DV quality/full evaluation unless disjoint training
  calibration or author-compatible sparse-adaptor evidence supplies a new
  predeclared route.

### 33.1 Adapter-Offset Transport Diagnostic

- Rationale: TranSplat's Gaussian adapter applies its predicted subpixel image
  offset to the normalized image-plane coordinate before unprojecting and
  normalizing the ray. The prior conditional-anchor diagnostic carried a
  world-space residual, which is not equivalent for nonzero offsets or real
  intrinsics. `conditional-adapter-offset-transport-diagnostic` recovers the
  bounded offset from each retained anchor's native mean/depth and the producing
  context C2W/intrinsics, then applies it before each assigned target-pixel ray
  normalization. It preserves the existing router, `tau_f=0.20`, `tau_d=0.10`,
  assignments, and moment matching, and fail-closes an entire tile to Full when
  any selected anchor cannot satisfy the adapter geometry contract.
- Pre-flight: synthetic direct-adapter equivalence, skipped-descriptor poison,
  PSD/SH/opacity, event-ledger, and fail-closed tests pass. The next fixed run
  is exactly one target-free TranSplat/DL3DV sample-0 attribute audit in
  `outputs/ae_dl3dv_repair_diagnostics/
  transplat_sample0_adapter_offset_transport_audit_v1/`, with seed 0,
  `probe-normalized-std-first-hit`, metric-depth routing, and no target RGB,
  renderer, quality metric, GGU, or hardware simulator. It is non-claim
  diagnostic evidence. A failure preserves the output and forbids a quality
  gate; a pass only authorizes a separately pre-registered single quality gate.
- Target-free result: preserve
  `transplat_sample0_adapter_offset_transport_audit_v1/results.json` (SHA256
  `9fd33bf2a0e1ab4abeacd347d0f5478eac296a8a25d3a0cce7e8d53e9352fa17`).
  It reports `target_rgb_accessed=false`; the native loader's target field was
  removed before context-device transfer and never passed to routing or a
  metric. The route has 26,496 skipped descriptors, zero adapter-geometry
  fallback tiles, zero skipped-S3 reads, identical poisoned-pass events, PSD
  output covariances, unchanged opacity (ratio exactly 1.0), and nonzero sparse
  selected-head work. This passes the target-free attribute gate only.
- Pre-registered quality gate: run exactly once in
  `outputs/ae_dl3dv_repair_diagnostics/
  transplat_sample0_adapter_offset_transport_quality_v1/` using the fixed
  `scripts/saes_selected_output_quality_gate.py` entrypoint bound to
  `conditional-adapter-offset-transport-diagnostic`, seed 0, the same official
  sample/checkpoint/target views, and unchanged `0.15/0.005/0.005` quality
  limits. The output remains non-claiming and does not authorize 8/32/140-scene
  expansion unless all limits and sparse-execution prerequisites pass.
- Quality result: preserve
  `transplat_sample0_adapter_offset_transport_quality_v1/results.json` (SHA256
  `e1e21de6f3d5f5509e676415ecf7314743af1b3aae453d0f17036cc973c858b1`).
  The selected-output trace is internally consistent (same route and retained
  attributes, no adapter fallback, and no target RGB before the mask commits),
  but the fixed quality gate fails: PSNR `34.8391 -> 25.6927` (loss `9.1464`
  dB), SSIM `0.97360 -> 0.85157` (loss `0.12203`), and LPIPS
  `0.03276 -> 0.21826` (increase `0.18550`). The strict decoder comparison is
  also non-bit-identical at max absolute delta `0.001052`, so this remains
  non-claim diagnostic evidence. Do not retry this candidate, change its
  thresholds, or expand DL3DV; a future branch requires an independently
  justified implementation hypothesis and then resumes from target-free tests.

### 33.2 Adapter-Offset Attribute-Transport Diagnostic

- Rationale: the adapter-offset path preserved the receiver-specific C2W-ray
  geometry but constructed every skipped SH/opacity contribution by copying the
  receiving anchor. Its range-constrained average therefore reduced to an
  identity. conditional-adapter-offset-attribute-transport-diagnostic keeps
  that geometry and covariance path unchanged, reconstructs each skipped SH
  and opacity as the existing bilateral selected-anchor convex estimate, then
  absorbs it with the existing receiver assignment. It adds no route level,
  threshold, source tensor, target view, or target RGB access.
- Synthetic gate: constant SH/opacity is preserved; nonuniform selected anchors
  produce the exact two-stage assignment update within the selected-source
  range; skipped-descriptor poison remains observationally irrelevant; L0/L1/
  Full routing, the primary-K L1 depth reference, 2K L1 anchors, adapter
  geometry, PSD covariances, and S2/S3 path counts are unchanged. The analytic
  ledger now charges the new selected-anchor SH/opacity reduction and its
  conservative FP16 reads rather than treating it as free.
- Target-free result: preserve
  transplat_sample0_l1_primary_reference_attribute_transport_target_free_attribute_audit_v2/results.json
  (SHA256 d4ed058d8227be3e371919d7edc32585eaa11b5472084e423041e9d5db7fc4bc,
  source 7c2d180). The fixed sample has 26,496 skipped descriptors, L0/L1/Full
  760/2172/5260, no geometry fallback, no skipped-S3 read, identical poisoned
  events and retained attributes, PSD output covariances, and nonzero retained
  SH/opacity updates of 0.259789/0.0614094 maximum absolute value. The trace
  records 175,488 attribute reconstruction pairs, 350,976 analytic reduction
  cycles, and 26,674,176 charged FP16 attribute bytes. It reads no target RGB,
  renderer, decoder, quality metric, or hardware cycle simulator and remains
  non-claim evidence.
- Pre-registered quality gate: run exactly once in
  outputs/ae_dl3dv_repair_diagnostics/
  transplat_sample0_adapter_offset_attribute_transport_quality_v1/ with
  scripts/saes_adapter_offset_attribute_transport_quality_gate.py. The
  entrypoint fixes TranSplat/DL3DV sample 0, seed 0, two context/four target
  views, tau_f=0.20, tau_d=0.10, primary-K L1 reference, 2K anchors, the
  unchanged 0.15/0.005/0.005 limits, and this materialization; it exposes no
  materialization, threshold, scene, or seed override. Target RGB may be read
  only after the selected-output mask and target-free semantic checks commit.
  A failure is preserved and prohibits an 8/32/140-scene expansion; a pass
  still cannot claim S2/S3 savings until the separate real-execution contract
  passes.
- Quality result: preserve
  transplat_sample0_adapter_offset_attribute_transport_quality_v1/results.json
  (SHA256 ef7535183d8c794b5a8458c0325363501b312f93b277912accaed09b2cd37308,
  source 52cf399). Target RGB stayed outside the encoder and route until the
  sparse mask committed. The selected-head replay has an equal route mask and
  equivalent retained attributes, but the strict decoder comparison remains
  non-bit-identical (maximum absolute delta 0.001048). More importantly, the
  unchanged quality gate fails: PSNR 34.8391 -> 25.6526 (loss 9.1865 dB),
  SSIM 0.97360 -> 0.85004 (loss 0.12356), and LPIPS 0.03276 -> 0.22120
  (increase 0.18844). This candidate is non-claim failure evidence. Do not
  retry it, change its fixed contract, or launch 8/32/140-scene work; the next
  route must begin with a new target-free implementation diagnosis.

### 33.3 Guard-Partition Oracle Audit

- Research question: does the existing diagnostic-only probe-attribute guard
  identify tiles whose selected-anchor materialization is locally faithful, or
  does the same retained-attribute error persist in both accepted and rejected
  partitions? The quality result cannot answer this because it must not be used
  to tune a guard or route.
- Fixed design: a new target-free entrypoint reuses only the v2 sidecar,
  TranSplat/DL3DV sample 0, seed 0, tile size 4, tau_f=0.20, tau_d=0.10,
  normalized-probe feature statistic, metric depth standard deviation, primary-K
  L1 reference, 2K native anchors, and adapter-offset attribute transport. It
  runs a guarded canonical clone and an unguarded shadow clone, records only
  per-tile route/guard scalars, and compares their already-committed retained
  descriptors against full encoder attributes post hoc. It may not render,
  decode, read target RGB, compute quality metrics, expose a threshold or
  sample override, or select a subsequent parameter.
- Acceptance: trace counts must agree with runtime guard counters; each trace
  carries no raw non-probe attributes; poisoning skipped descriptors must leave
  route, trace, stats, and retained output unchanged; covariance PSD,
  transmittance bounds, and assignment normalization must hold. The report
  separates guard-accepted, L0-rejected/L1-accepted, and rejected-to-Full
  outcomes, charging no new result or saving claim. With the current nested
  L1 anchors and identical guard predicate, an L0 rejection necessarily also
  rejects L1; retain the L0-rejected/L1-accepted partition with an explicit
  zero-by-construction count rather than inferring a measured absence.
- Decision rule: if the unguarded shadow has comparable posthoc covariance
  scale, transported-mean, SH, or opacity error in guard-accepted and
  guard-rejected partitions, no additional guard threshold is a valid repair.
  The next candidate must instead repair pseudo-descriptor geometry. If the
  partitions clearly separate, retain the finding as diagnostic-only and wait
  for disjoint calibration before any guard policy is considered.
- Result (2026-07-18): the fixed run at source `42edcce` completed at
  `outputs/ae_dl3dv_repair_diagnostics/
  transplat_sample0_l1_primary_reference_guard_partition_audit_v1/` with
  results SHA256 `dfb94a26923345ab8de0fb1d112a365160a40fc0816b47001f5286d628aa9dde`.
  It records `8,192` tiles, canonical L0/L1/Full `760/2,172/5,260`, `1,058`
  L0 and `2,878` L1 checks, zero skipped-S3 reads, and exact poisoned-clone
  mask/trace/stats/retained-attribute invariance. The guard partitions are
  `2,932` accepted, `937` rejected-to-Full, `4,323` noncandidate-Full, and
  zero L0-rejected/L1-accepted by the declared nested-anchor guard structure.
  Canonical Full passthroughs deliberately carry no sparse oracle samples.
- Shadow comparison: labels remain tied to the canonical guard trace. For L0,
  accepted versus rejected-to-Full shadow covariance/mean/SH/opacity p50 are
  `0.8158/0.00373/0.01456/0.00838` versus
  `0.9775/0.00784/0.06862/0.02798`; for L1 they are
  `0.3523/0.000635/0.00661/0.00342` versus
  `0.4421/0.000820/0.01776/0.00513`. SH/opacity rise for rejected subsets, but
  dominant covariance and optical-depth errors remain the same order (L1
  covariance p95 `0.812/0.913`; optical-depth p50 `0.50039/0.49836`). This is
  target-free diagnostic evidence, not a claim of quality or speed.
- Decision: `bad/stop` for guard-threshold repair and `good/iterate` for
  pseudo-descriptor geometry diagnosis. Do not sweep or change the existing
  guard, thresholds, seed, scene, or quality gate. Guard policy remains frozen
  pending disjoint calibration; the only next implementation route is a
  separately pre-registered pseudo-descriptor geometry correction.

### 33.4 Assignment-Consensus Adapter Pseudo Descriptor

- Hypothesis: the retained-attribute failure arises from representative
  geometry/coverage rather than SH or opacity identity. For each skipped
  position, the existing bilateral selected-anchor weights can form one
  pseudo descriptor: assignment-weighted selected-anchor depth and bounded
  adapter image-plane offset, lifted through that skipped position's C2W ray;
  covariance uses one selected-anchor first/second moment. This is a new
  virtual-output diagnostic, not a paper-result-eligible replacement for the
  frozen sparse representative merge: applying its consensus descriptor again
  to every receiving anchor would introduce an undeclared double-assignment
  term.
- Fixed target-free gate: retain sample 0, seed 0, tile size 4, tau_f=.20,
  tau_d=.10, normalized feature statistic, metric depth, primary-K L1
  reference, 2K anchors, and all existing numerical fail-closed checks. The
  candidate may use only S1, selected-anchor S2/S3, static context camera
  geometry, and the already-declared assignment; it must not read target RGB,
  skipped S3 descriptors, paper results, or quality metrics. It writes only
  virtual skipped outputs, leaves selected anchors untouched, reports zero
  Gaussian compression, and is barred from claim/Functional modes and any
  quality retry. Test constant, one-hot, PSD, C2W, assignment, and poison
  properties before deciding whether a separately pre-registered target-free
  DL3DV audit is justified.
- Algebra and access contract: for selected anchors `q`, existing bilateral
  weights `r_iq`, selected S2 depths `d_q`, recovered bounded adapter offsets
  `o_q`, and skipped position `x_i`, construct
  `d_i=sum_q r_iq*d_q`, `o_i=sum_q r_iq*o_q`, and
  `m_i=LiftC2W(x_i,d_i,o_i)`. Let
  `m_iq=LiftC2W(x_i,d_q,o_q)` and construct exactly one PSD covariance moment
  `C_i=sym(sum_q r_iq*(C_q+(m_iq-m_i)(m_iq-m_i)^T))`, followed by the existing
  eig-floor. SH and opacity remain their selected-anchor convex estimates.
  The diagnostic writes `(m_i,C_i,SH_i,alpha_i)` only at skipped virtual
  outputs and never applies `r_iq` again to a retained anchor, eliminating an
  `r_ip*r_iq` term by construction. Its helper accepts only selected-anchor
  tensors, selected positions, assignments, target positions, and context
  camera geometry, so full depth/Gaussian tensors cannot be read accidentally.
- Synthetic-only gate: prove constant and one-hot exactness; offset-before-ray
  normalization under nonidentity C2W/intrinsics; PSD/finite and full-tile
  fail-closed behavior; assignment simplex and anchor-permutation invariance;
  selected-S2/S3 and skipped-S2/S3 poison invariance for L0 and L1; unchanged
  route, mask, selected-anchor counts, and existing S2/S3 events. Record a
  distinct nonzero consensus geometry arithmetic/traffic counter; do not
  recycle the SH/opacity-only accounting counter or call the work free.
- Abandonment: any selected-anchor access violation, non-finite/PSD failure,
  routing drift, or non-positive direct full-S3 attribute evidence stops this
  branch. A synthetic pass only permits a new target-free-audit preregistration;
  a target-free pass alone does not authorize a quality retry, which remains
  blocked on the fixed failure record and disjoint calibration.
- Synthetic result correction (2026-07-18): the source `43dfd37` pass is
  invalidated as selected-only evidence. Its atomic Full fallback first cloned
  all tile descriptors, which read skipped raw S3 attributes even though later
  writes restored them. The corrected `92dec5c` implementation builds
  selected-anchor-only virtual plans without writing outputs, and commits them
  only after every primitive slot validates; `968563d` also accepts the valid
  no-output case where the anchor layout covers an entire tile. The full SAES
  collection now passes `164` tests (one upstream `skvideo` deprecation
  warning), including constant and one-hot geometry, exact alpha `1.0`
  constants, nonidentity TranSplat adapter geometry, assignment-squaring
  rejection, anchor permutation, L0/L1 selected/skipped poison, selected-read
  guards with default materialization guard enabled, route/event equivalence,
  PSD/finite checks, and atomic multi-slot Full fallback. The analytic ledger
  separately charges successful virtual outputs and a conservative full virtual
  attempt before any fail-closed fallback. No DL3DV audit, render, quality
  metric, target RGB read, or sparse execution claim was run or created by
  either state.

## 34. DL3DV Training-Calibration Preparation

- The author-side training calibration path now has an executable, fail-closed
  archive contract. `data/download_dl3dv_calibration.py --write-plan` lists the
  pinned gated `DL3DV/DL3DV-ALL-480P` tree and commits a 24+8
  evaluation-disjoint selection before data download. Its execution phase
  rereads that tree, verifies the plan byte-for-byte, downloads only the 32
  selected ZIPs, checks size plus the bound upstream object id, records each
  actual archive SHA256, and safely extracts all 32 scenes while binding the
  24 training and eight holdout scene sets to the prepared tree.
- `data/prepare_dl3dv_calibration_inputs.py` converts both fixed splits into
  native and Re10K-compatible target-free sidecars. It permits only selected
  context image bytes and camera geometry, binds source/prepared-tree hashes,
  rejects missing, overlapping, extra, or evaluation scenes, and routes the
  three models to their required representation in `scripts/calibration_sweep.py`.
- Current external blocker: the configured account has a valid Hugging Face
  token but has not been granted data access to `DL3DV-ALL-480P`. On 2026-07-18
  the revision-pinned archive resolve request returned the upstream explicit
  `403 ... not in the authorized list`; no archive was downloaded and no
  permission boundary was bypassed. This is not a GPU or Vivado constraint;
  the RTX 3060 is idle after Vivado was stopped.
- Current technical blocker: all three fixed-model direct-S2/S3 sparse paths
  are fail-closed with zero verified SAES savings. Therefore calibration and
  any 140-scene DL3DV evaluation remain prohibited until a paper-compatible
  sparse implementation passes the existing synthetic and one-sample quality
  gates. The new data path is preparation work, not a Results Reproduced claim.
- Probe-only guard checkpoint: the first fresh target-free TranSplat/DL3DV
  sample-0 run at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_materialization_guard_v1/`
  executed 1,058 L0 and 2,878 L1 guard checks without reading a non-probe S3
  attribute, but an initialization bug serialized
  `materialization_guard_enabled` as `0`. Preserve it as invalid diagnostic
  evidence. The fixed v2 rerun preserves all routing counters exactly, records
  `materialization_guard_enabled=true`, and builds the charged ledger with
  27,256 guard descriptor events and 2,507,552 bytes of guard traffic. It is
  target-free and stops before GGU, rendering, quality metrics, and hardware
  simulation. The ledger remains analytic, not RTL-cycle-equivalent; this does
  not change the zero verified S2/S3 saving or authorize quality expansion.

## 35. Selected-Output Quality-Pilot Contract

- Run ID: `saes-selected-output-quality-pilot-v1`.
- Research question: can the exact same-weight TranSplat Gaussian-head replay
  execute the real L0/L1/Full retained-output mask through the native Gaussian
  adapter and decoder without changing the fixed target-view quality contract?
- Fixed setup: canonical DL3DV sample 0, its committed two context and four
  target views, TranSplat `re10k.ckpt`, seed 0, `tau_f=0.20`, `tau_d=0.10`,
  metric probe-depth standard deviation, the existing probe-only guard, and
  conditional-anchor moment materialization. No target RGB may enter routing,
  mask selection, or head execution; it is read only after the sparse output
  is fixed to render the four already selected target views and evaluate them.
- Execution design: the dense reference/control pass records actual S1/S2
  tensors but contributes no SAES saving. A first selected-head pass produces
  only the conservative L0/L1 anchor closure needed for the guard. A second
  selected-head pass uses the guard-resolved retained mask; every Full-tile
  output and every retained L0/L1 anchor is executed with original weights,
  while omitted second-convolution outputs are not executed. Both replay event
  ledgers are charged. Dense S1/S2/refinement and first-convolution closure
  remain explicitly charged, and global `s2_s3_sparse_execution_verified`
  remains false unless a later complete stage proof is obtained.
- Minimal gate: synthetic batched-mask replay equivalence and a new DL3DV
  sample-0 output directory. Acceptance requires identical guard-resolved
  masks across the two sparse passes, a complete replay event record, and
  PSNR loss <= 0.15 dB, SSIM loss <= 0.005, LPIPS increase <= 0.005 over the
  same dense reference. Any failure is preserved as non-claim evidence; it
  does not authorize an 8/32/140-scene run or a change to thresholds, seed,
  scene, or tolerances.
- `v1` result: preserve
  `outputs/ae_dl3dv_repair_diagnostics/saes_selected_output_quality_pilot_v1/`
  as non-claim failure evidence. Its fixed quality verdict fails (PSNR loss
  `9.1452 dB`, SSIM loss `0.12201`, LPIPS increase `0.18548`), and source
  inspection also found that it transferred target RGB before the sparse mask
  was committed. The result must not be promoted or overwritten.
- `v2` repair contract: repeat the same one-sample command in a new output
  directory with no change to scene, seed, checkpoint, thresholds, routing,
  materialization, or quality limits. Target RGB remains in the native batch
  until after the final sparse mask, selected-head event ledgers, retained
  descriptor equivalence, and target-free decoder equivalence are all fixed.
  The run must prove that the selected-head output and a dense-head SAES
  control have equal routing masks and FP32-equivalent retained attributes and
  decoder colors before target RGB is read. A quality failure after those
  execution checks is a mechanism-fidelity failure, not an execution-boundary
  failure, and still prohibits expansion.
- `v2` target-free decoder finding: the repaired execution reached the
  semantic control before RGB access, but its native rasterizer output was not
  FP32-equivalent while it retained SAES-removed, zero-opacity descriptors
  whose raw head entries were intentionally absent. Retained attributes and
  the route mask were already equivalent. This is an S4 input-contract issue,
  not a quality verdict; preserve
  `saes_selected_output_quality_pilot_v2.run.log` as the failed audit trace.
- `v3` repair contract: pass only the post-SAes retained descriptor set to the
  native decoder on both the dense-SAes control and selected-head path. This
  removes descriptors SAES has already assigned zero opacity and changes no
  retained descriptor, router decision, materialization equation, checkpoint,
  sample, seed, or threshold. Require the same target-free attribute and
  decoder equivalence checks before the one fixed quality evaluation.
- `v3` target-free decoder finding: compaction removed the deleted descriptors
  but the strict decoder equivalence still failed before RGB access and before
  metrics. Preserve `saes_selected_output_quality_pilot_v3.run.log`; it is not
  a quality result. The follow-up remains target-free: record the selected-vs-
  dense render delta together with a repeat render of the same dense descriptor
  set, so CUDA rasterizer repeatability is distinguished from selected-head
  numerical drift before selecting an execution contract.
- `v4` target-free result: the dense control is bit-stable on immediate repeat;
  selected-vs-dense has mean absolute color delta `1.028e-7` and maximum
  `0.001052`, induced by the already audited FP32 selected-head accumulation
  delta (maximum `2.670e-5`). Therefore the next fixed quality measurement
  records this nonzero numerical boundary explicitly instead of falsely
  calling it bit-equivalent. It remains non-claiming unless both strict
  decoder equivalence and all original quality gates pass; no threshold,
  scene, seed, or routing change is authorized by this diagnostic.
- `v5` clean quality result: preserve
  `outputs/ae_dl3dv_repair_diagnostics/saes_selected_output_quality_pilot_v5/`
  (results SHA256
  `d2c51c3e1f93bad4d488f4366f1640f93045683a5e47b5695348561e1fcc4269`).
  Target RGB provenance is valid: it stayed in the native batch until sparse
  output commitment, then was removed for the fixed four-view metrics. The
  dense and selected-head paths have equal masks and FP32-equivalent retained
  attributes, but their strict decoder images differ by max `0.001052` from
  selected-head accumulation. More importantly, the unchanged quality gate
  fails: PSNR loss `9.1452 dB`, SSIM loss `0.12201`, and LPIPS increase
  `0.18548`. It is non-claim evidence and forbids any 8/32/140-scene DL3DV
  expansion. The only valid next routes are an evaluation-disjoint training
  calibration after official access is granted, or a new target-free,
  paper-compatible materialization hypothesis that first passes synthetic and
  descriptor-property gates; neither route may use evaluation RGB, metrics,
  sample choice, or threshold fitting.

## 36. Fixed K/2K Render-Teacher Capacity Diagnostic

- Research question: with the already frozen DL3DV/TranSplat sample-0 mask and
  output budget, is the failure caused by insufficient K/2K representation
  capacity, or by the current target-free representative construction?
- Fixed contract: use the evaluation-index SHA256
  `eab21290cfab8eff12e208377b089b2a65e15f7ba44f7bb085963511354863f4`,
  checkpoint SHA256
  `89e43c205a04962e427801385d7d18e74cba063d05a76bf8b28e5fa746a4b69a`,
  scene `032dee...60ca50ac04ec7`, context `[0,9]`, target indices `[1,3,5,7]`,
  seed 0, and the committed K/2K/Full partition. Target RGB is removed before
  encoding; detached dense renderer outputs, not RGB, are the fixed post-hoc
  teacher. Only the 4,792 representative slots are optimized for exactly 128
  Adam steps. The route, Full descriptors, deleted slots, target-camera set,
  and output count are immutable. This is capacity evidence only, never a
  runtime implementation, paper result, or quality retry.
- Binding result: preserve
  `outputs/ae_dl3dv_repair_diagnostics/
  transplat_sample0_same_budget_render_teacher_oracle_v3_frozen_contract/`
  (results SHA256
  `e4c110a087c8f19126f235a88091383e9b85e34d713ffd70a5960ad91e5ae295`,
  source `ce9349c`). Input identity, the modified-mask SHA256
  `8e5df2...e8e4333e`, the representative-mask SHA256
  `13384c...c414726`, and `131072 / 9432 / 4792 / 116848` dense / removed /
  representative / Full counts all match their frozen values. Every Full
  descriptor is exact before and after compaction and immutable throughout the
  optimizer. The loss falls from `4.978138e-4` to `4.405215e-6`; final dense
  teacher fidelity is `53.5922 dB`, `0.998050` SSIM, and `0.006741` LPIPS.
  Optimized-representative and teacher file hashes are verified against the
  result record. No target RGB, quality metric, sparse execution, or saving
  claim is present.
- Interpretation: K/2K capacity is sufficient under this bounded,
  target-camera post-hoc teacher, while the fixed drop-only / merge-only /
  drop+merge attribution establishes that removal coverage and occlusion, not
  representative parameter capacity alone, dominate the current loss. This is
  not a target-free construction and does not prove that a paper-compatible
  sparse path can attain that fidelity.
- Next gate: before any quality retry, pre-register exactly one target-free,
  target-camera-free multi-context representative-moment audit. It must
  preserve the current route, sample, seed, thresholds, K/2K mask, Full
  fallback, and event accounting; compare the current single-producer merge
  with a deterministic context-projected/tangent-plane first/second-moment
  construction using only selected-anchor S3, S1/S2, existing assignments, and
  static context-camera geometry. The candidate must reject target camera
  metadata, RGB, teacher renders, metrics, parameter optimization, sample
  selection, and threshold fitting. Dense skipped S3 is permitted only after
  the candidate output is committed as a post-hoc reference, never as a
  candidate input. Synthetic gates must prove constant/one-hot behavior,
  PSD/finite covariance, single assignment, context-view permutation or
  equivariance where applicable, selected/skipped poison invariance, unchanged
  route/event counts, and atomic Full fallback. A synthetic pass permits at
  most one separately registered target-free descriptor audit, not a DL3DV
  quality or full-protocol run.
- Implementation checkpoint (2026-07-18): the candidate is now materialized
  only as `multicontext-tangent-plane-diagnostic`. It reuses the current
  conditional selected-anchor construction and can replace only a retained
  representative covariance with the context-projected tangent-plane fit.
  The router, thresholds, K/2K mask, means, SH, opacity, Full passthrough,
  and output count remain unchanged; its counters explicitly report
  `runtime_eligible=false`. Synthetic helper tests cover projected moments,
  camera order and rigid-transform equivariance, invalid/duplicate cameras,
  PSD/finite, and one-hot/constant fallbacks. The materialization test covers
  skipped S2/S3 poison invariance on valid slots, unchanged route/event
  counts, immutable Full slots, and valid-slot PSD. The focused tests plus all
  `tests/test_saes_*.py` pass (`181 passed`), but no target-free DL3DV
  descriptor audit, render, metric computation, or quality retry has run.

## 37. Pre-Registered Multi-Context Directional Descriptor Audit

- Audit id: `multicontext-tangent-target-free-audit-v1`. This is one fixed
  TranSplat/DL3DV sample-0 descriptor audit, not a renderer or quality run.
  It uses the fixed checkpoint SHA256
  `89e43c205a04962e427801385d7d18e74cba063d05a76bf8b28e5fa746a4b69a`,
  source-index SHA256
  `eab21290cfab8eff12e208377b089b2a65e15f7ba44f7bb085963511354863f4`,
  seed `0`, `tau_f=0.20`, `tau_d=0.10`, normalized-probe first-hit routing,
  metric probe-depth routing, and the existing K/2K/Full policy. It may not
  change sample, route, mask, threshold, seed, output count, or Full policy.
- Input boundary: a new immutable sidecar contains only the two fixed context
  RGB inputs and their two context camera records. It contains no target RGB,
  target camera record, target index, renderer input, teacher artifact, or
  quality-metric input. The returned audit batch has no `target` mapping before
  context-device transfer; the audit entrypoint and pure metric helpers accept
  context geometry only.
- Two-phase execution: phase A materializes and hashes the current
  `conditional-adapter-offset-attribute-transport-diagnostic` output and the
  `multicontext-tangent-plane-diagnostic` output separately. It verifies equal
  route mask, K/2K counts, Full slots, means, SH, opacity, and event counts,
  plus PSD/finite and selected/skipped poison isolation, then writes both
  commit manifests. Only after both manifests exist may phase B read dense
  skipped S3 as a read-only post-hoc reference. No phase may invoke a decoder,
  renderer, target camera, teacher oracle, expected-results file, optimizer,
  quality metric, sample selection, or threshold fitting.
- Post-hoc observables: for every populated L0/L1 tile group and each of the
  two context cameras, compare current and tangent against the same dense tile
  reference for projected optical mass (zeroth moment), coverage center (first
  moment), projected covariance (second moment), footprint determinant and
  log-ratio, covariance PSD, condition number, and tangent fit residual. The
  projection is an unclipped positive-depth descriptor surrogate, not alpha
  compositing or a rendering claim. Invalid depth, covariance, opacity, mass,
  or determinant is explicit failure, never silently clamped into a pass.
  Each observable records count, p50, p95, maximum, and candidate-minus-current
  change.
- Fixed directional gate: every populated L0/L1-by-context group must have no
  p50 or p95 dense-reference error worse by more than 1%; the aggregate
  projected-covariance error must decrease by at least 20%; projected
  optical-mass error may not worsen; all structural, PSD/finite, poison,
  Full-passthrough, and route/event invariants must pass; and no local helper
  fallback may be treated as a successful fit. Empty groups or a no-op tangent
  output are inconclusive and do not pass.
- Branch rule: a pass is not quality evidence, but authorizes exactly one
  separately pre-registered fixed DL3DV sample-0 quality gate. A second-moment
  improvement with optical-mass regression permits only a deterministic
  multi-context mass-scale constraint diagnosis. No dense-reference
  second-moment improvement terminates this tangent candidate; substantial
  local fallback requires geometry/identifiability repair. No branch permits
  a threshold, seed, sample, or protocol change.
- Execution record (2026-07-18): the one launched run is preserved at
  `outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_multicontext_tangent_target_free_audit_v1/`
  with `results.json` SHA256
  `a98371e3755fc568f6baef1b94934c3fc61fa65c5108a63c72cb19f4ec2044f8` and
  phase-A manifest SHA256
  `18169c7800378a79ed20ca2f077ca5f05d706f45ecfb538ab53db324f239e015`.
  It did not render, decode, access target RGB/camera metadata, access a
  teacher or expected results, or compute a quality metric. The two committed
  outputs had the same route-mask SHA256
  `d495532166ea723923d1c6f00e8b60b940544d99eaf1aa680a3d6afc474c3a11`, and
  the recorded Full, means, SH, opacity, route/event, PSD, and poison checks
  passed before the post-hoc dense read.
- Negative outcome and stop decision: the post-hoc aggregate projected
  covariance relative error changed from `0.0542566969` to `0.0542694710`
  (`-0.02354%` decrease, rather than the required `>=20%` decrease), although
  optical-mass relative error narrowly improved from `0.1721075829` to
  `0.1720729075`. It also recorded `1,158` local
  `invalid-context-projection` fallbacks. `quality_gate_authorized=false`.
  The tangent candidate is therefore retired: do not tune its residual,
  threshold, seed, sample, or mask, and do not run a quality retry.
- Evidence-integrity boundary: this negative run is retained as a
  non-authorizing diagnostic, not promoted as a clean pre-quality gate. The
  input sidecar itself was context-only and its recorded tree SHA256 was
  `a177ebdf7b4871b834bddfa6bf29ab5029ebaa4c2d421ec5a5ac3f8bd09f5605`, but
  the runner only recorded rather than pre-enforced the fixed input and
  checkpoint bindings, and its selected-only S3 proof used a single poison
  behavior check rather than an instrumented access boundary. Before any
  distinct future candidate receives a new descriptor audit, harden those
  controls with an exact input allowlist/source binding, pre-load hash checks,
  and two-sentinel selected-only access instrumentation. This does not reopen
  or rerun the retired tangent candidate.
