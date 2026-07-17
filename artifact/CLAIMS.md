# SCARF Artifact Claim Contract

SCARF applies for Artifact Available, Artifacts Evaluated - Functional, and
Results Reproduced. The complete Evaluation-section inventory is machine
readable in `artifact/evaluation_catalog.json`. A claim passes only after a
non-author evaluator runs the documented workflow and `scripts/validate_ae.py
--require-key-results` reports PASS.

## Submission Intent Versus Evidence State

Claim intent and completed evidence are separate. Figure 8 is a mandatory key
result, but it remains blocked until a real Jetson Orin NX evaluator run exists.
Table 1, Figure 10, Figure 11, Tables 2-3, Figure 12, and Figures 13-16 are
mandatory deterministic results whose current simulator and data gaps must be
closed before release. The machine-readable state must never be promoted merely
because a result is listed here.

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
| Figure 10 | `run_ae.sh worstcase` | Full per-view ranking, source images, and stage cycles | Displayed views are the deterministic global FSDR/SAES worst cases |
| Figure 11 | `run_ae.sh mechanisms` | No-opt, FSDR, SAES, and combined event cycles | Three geometric means within 5% |
| Tables 2-3 | `run_ae.sh mechanisms` | Discrete mechanism and work counters | Rates within 0.02 absolute; counts within 5% relative |
| Figure 12 | `run_ae.sh utilization` | Useful and scheduled MMCU slots | S1-S3 bars within two percentage points |
| Figures 13-16 | `run_ae.sh sensitivity` | Five-point grids for all nine pairs | Fixed peaks, ratios, monotonicity, and quality gates |

The quick synthetic fixture, bounded pilots, dense diagnostics, partial
matrices, workstation timing, and manuscript CSV files cannot satisfy these
rows.

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

The public LPDDR5 Ramulator/DRAMPower smoke trace remains Functional evidence
until full workload traces reproduce the paper's stated memory configuration.
