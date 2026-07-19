# SCARF Artifact Claim Contract

SCARF applies for Artifact Available, Artifacts Evaluated - Functional, and
Results Reproduced. The complete Evaluation-section inventory is machine
readable in `artifact/evaluation_catalog.json`. A claim passes only after a
non-author evaluator runs the documented workflow and `scripts/validate_ae.py
--require-key-results` reports PASS.

## Submission Intent Versus Evidence State

Claim intent and completed evidence are separate. Figure 8 is a mandatory key
result with submission state `CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION`:
SCARF requests reproduction of the result, but it is neither scheduled on this
host nor marked `PASS` until a non-author runs the documented real Jetson Orin
NX workflow. Table 1 and Figure 11 are the remaining mandatory deterministic
results. Tables 2-3 are Figure 11 supporting evidence from the same bound
mechanism trace; they remain required for Figure 11 to pass, but are not
separate mandatory results. Figure 10, Figures 12-16, and the public physical
proxy are paused by the current scope. The machine-readable state must never be
promoted merely because a result is listed here.

Figure 9 and Table 4 have two explicitly separate layers: the paper's
commercial TSMC28 values are comparison targets, while the released workflow
generates an ASAP7 predictive 7 nm implementation and a DeepScale 28
nm-equivalent estimate. The proxy can validate the public hardware workflow but
is not the original TSMC28 post-layout measurement.

## Evidence Classes

- `independent_measurement`: a non-author evaluator measures the result on the
  required real hardware. Figure 8 requires this class.
- `deterministic_execution`: a non-author evaluator regenerates the result from
  released code, data indices, checkpoints, and simulator semantics.
- `public_physical_proxy`: public predictive-PDK or technology-normalization
  evidence with its limitations preserved.
- `paper_comparison_target`: a manuscript value used only by validators and
  comparison plots. It is never a generated result.

Only the first two classes directly support Results Reproduced. The generator
must label every figure/table and reject a class that is weaker than the catalog
requires.

## Mandatory Key Results

| Result | Entry point | Required evidence | Acceptance |
|---|---|---|---|
| Figure 8 | `run_ae.sh performance --device orin` | Real Orin CUDA events, Nsight stage records, and positive ASIC cycles | Nine pairs and geometric mean within 5% of 2.94x |
| Table 1 | `run_ae.sh quality` | All selected target views for nine pairs | PSNR 0.15 dB; SSIM/LPIPS 0.005 |
| Figure 11 | `run_ae.sh mechanisms` | No-opt, FSDR, SAES, combined cycles, and Tables 2-3 counters from one bound trace | Three geometric means within 5%; supporting rates/counts within their fixed tolerances |

The quick synthetic fixture, bounded pilots, dense diagnostics, partial
matrices, workstation timing, and manuscript CSV files cannot satisfy these
rows.

## Supporting Results

Tables 2-3 are the active supporting records for Figure 11 and must be complete
for it to pass. Figure 10, Figures 12-16, Figure 9, and Table 4 are paused by
scope; their manual entry points remain available but are excluded from default
execution and validation. No paused result can turn an absent key-result
execution into `PASS`.

## Aggregate Rule

A single-scene pilot is a diagnostic only. Its L0/L1/Full route mix, quality,
or speed cannot be compared with a paper dataset aggregate or used to select
configuration. Only the frozen protocol aggregate over every required scene
and target view can be compared with Table 1, Figure 11, or Tables 2-3.

## Hardware Assignment

The MICRO submission declares Jetson Orin NX 16 GB as a special dependency.
The AE reviewing guidance states that evaluators bid using declared hardware
and software dependencies. If no assigned evaluator owns an Orin NX, the AE FAQ
allows exceptional remote-machine access for rare hardware. Chairs must confirm
one of those paths before Figure 8 is marked ready. A script without a real
independent run is Functional evidence only.

## Strict Execution Rules

- `artifact/expected_results.json` is validator-only. Simulator, sampler,
  runner, aggregation, and hardware code must not read it.
- FSDR uses a recorded signed random-hyperplane projection, discrete full-search
  Top-1 evidence, and event counts. `in_window_rate` is not Top-1 Coverage.
- SAES claim execution uses the published probe-only L0-to-L1-to-Full decision
  order. Dense or probe-spread diagnostics are permanently non-claiming.
- Missing data, views, checkpoints, counters, cycles, hardware reports, or raw
  logs fail closed. No reference constant, random sampler, target fitting, seed
  sweep, or edited generated JSON is allowed.
- Dataset aggregates retain every sample result hash and the exact canonical
  scene/context/target selection hash.
- Calibration is governed by `artifact/CALIBRATION.md`. Its training scenes
  must be disjoint from the evaluation protocol, and its process must not read
  target RGB, ground truth, expected results, manuscript CSV files, or prior
  evaluation outputs.
- Claim runs use one SHA256-bound global `artifact/mechanism_config.json` for
  all pairs. Per-pair parameters and any configuration change after evaluation
  begins invalidate the evidence.
- The FSDR local depth-validity check may only reject an LSH/CAM hit using
  probe/neighbor depths already available to the published schedule. SAES
  routing remains feature variance then depth standard deviation; numerical
  safeguards, C2W-ray moment construction, and lightweight materialization do
  not add a routing criterion or read non-probe Stage-3 attributes.

## SAES Candidate Contract

The current unweighted SAES candidate is a fixed 4-by-4, 12-anchor, guard-on
diagnostic until evaluation-disjoint calibration freezes it in
`artifact/mechanism_config.json`.
For a 4 by 4 tile, L0 retains the four primary probes `(0,0)`, `(0,3)`,
`(3,0)`, and `(3,3)`. L1 retains those four probes plus the eight deterministic
edge anchors `(0,1)`, `(0,2)`, `(1,0)`, `(1,3)`, `(2,0)`, `(2,3)`, `(3,1)`, and
`(3,2)`. L1 routing still uses only the four primary probe depths. Full tiles
must preserve means, covariances, harmonics, and opacities bit-for-bit.

The context safety guard may read only S1 features, primary probe depths,
selected probe Gaussians, context cameras, and projected footprints. Missing or
invalid geometry must fall back to Full. The public `saes-quality` selector is
calibration-gated and any DL3DV sample-0 output from it is non-claim evidence.
A guard-on route becomes claim-eligible only when the frozen configuration and
calibration protocol bind that exact setting.

The current v5 sample-0 quality record takes the Full path for every tile. It
demonstrates fallback fidelity, not SAES reduction, and cannot support Table 3
or Figure 11.

ACID v5 is the only active frozen author-side materialization epoch. Its
live-validated local provenance is not execution proof. With no registered
deterministic verifier or external authority, its candidate runtime, promotion,
and result paths remain fail-closed and have no cache, train, runtime, or
result evidence. It remains `paper_result_eligible=false` and cannot authorize
a DL3DV target-RGB quality gate or any Results Reproduced claim; all earlier
ACID contracts and their cache, smoke, runtime, and result artifacts are
superseded and excluded from release.

Table 1, Figure 11, and Table 3 must be generated from records sharing one
mechanism-config SHA256, checkpoint SHA256, canonical selection SHA256, and
execution-trace-set SHA256. A quality record and a performance record with any
different binding are not combinable evidence.

## Public Hardware Boundary

The commercial TSMC 28 nm PDK, memory compiler, Liberty files, and LPDDR PHY
cannot be redistributed. Public PPA uses iFlow/ASAP7 with explicitly labeled
SRAM proxies and excludes unmodeled analog I/O. DeepScale retains raw 7 nm
values, factors, 28 nm-equivalent estimates, and uncertainty notes. Similarity
to the paper TSMC28 numbers is never a pass/fail criterion.

## Result States

- `PASS`: required raw evidence exists and every structural/numeric gate passes.
- `FAIL`: execution completed but at least one required gate failed.
- `NOT_RUN`: required execution has not completed.
- `BLOCKED`: an external legal, account, or physical-hardware dependency is
  unavailable and no valid evidence exists.
- `NOT_CLAIMED`: context or proxy evidence outside the Results Reproduced set.

`CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION` is a submission-intent state,
not an execution outcome. It requires the same independent Orin evidence as
`PASS` and must never cause a workstation run, release staging, or validator to
treat Figure 8 as complete.

The public LPDDR5 Ramulator/DRAMPower smoke trace remains Functional evidence
until full workload traces reproduce the paper's stated memory configuration.
