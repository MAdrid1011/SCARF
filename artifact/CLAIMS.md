# SCARF Artifact Claim Contract

SCARF requests Artifact Available, Artifacts Evaluated - Functional, and
Results Reproduced. `artifact/evaluation_catalog.json` is the authoritative
machine-readable mapping from each Evaluation figure/table to its command,
evidence class, raw inputs, acceptance rule, and generated status.

## Evidence Classes

- `independent_measurement`: a Jetson Orin NX device record with CUDA events,
  Nsight traces, thermal samples, and runtime identity.
- `deterministic_execution`: a regenerated result from the released source,
  selected data, checkpoints, and simulator semantics.
- `public_physical_proxy`: public predictive-PDK or technology-normalization
  evidence with source process and model metadata.
- `paper_comparison_target`: an archived reference value used for comparison
  and validator tolerance checks.

Only generated records in the required evidence class satisfy a catalog row.
The runner and simulator do not read paper reference values.

## Key Results

| Result | Entry point | Acceptance |
|---|---|---|
| Figure 8 | `run_ae.sh performance --device orin` | Nine pairs and geometric-mean ASIC speedup within 5% of 2.94x |
| Table 1 | `run_ae.sh quality` | PSNR within 0.15 dB; SSIM and LPIPS within 0.005 |
| Figure 11 | `run_ae.sh mechanisms` | Three geometric means within 5%; same-trace Tables 2-3 counters within fixed tolerances |

Tables 2-3 are the Figure 11 supporting records. The full aggregate uses the
canonical scene, context, and target-view selection specified in
`artifact/evaluation_protocol.json`.

## Execution Rules

- `artifact/expected_results.json` is validator-only. The simulator, sampler,
  runner, aggregation, and hardware code do not read it.
- Every record binds source revision, submodule revisions, mechanism
  configuration, environment, data, checkpoint, selection, command, and raw
  output hashes.
- FSDR records the signed random-hyperplane projection, candidate search, depth
  validity decision, and reuse events.
- SAES uses the probe-only L0-to-L1-to-Full decision order and records retained
  descriptors, route decisions, and Gaussian materialization.
- One SHA256-bound global mechanism configuration is used across the matrix.
- Quality, performance, and mechanism records are combined only when their
  source, checkpoint, selection, mechanism, and execution-trace bindings match.

## Orin Comparison

`artifact/reference_results/orin_nx_reference.csv` is the archived normalized
Figure 8 comparison table. It provides a stable baseline for reviewers without
an Orin NX. The Orin workflow produces the device record used for independent
measurement and preserves its CUDA-event, Nsight, tegrastats, power-mode, and
clock-state inputs.

## Public Hardware Scope

The public physical workflow uses iFlow/ASAP7 and labels the resulting data as
ASAP7 raw or DeepScale 28 nm-equivalent. The commercial TSMC28 and LPDDR4X
collateral are not redistributed. `artifact/HARDWARE_SCOPE.md` defines the
hierarchy, scaling model, and required output bindings.

## Result States

`PASS`, `FAIL`, `NOT_RUN`, `BLOCKED`, `NOT_CLAIMED`, and
`CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION` are explicit catalog states.
The validator derives `PASS` from the required generated records and their
structural and numeric checks.
