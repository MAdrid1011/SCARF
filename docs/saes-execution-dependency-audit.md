# SAES S2/S3 Dependency Audit

This target-free diagnostic tests a prerequisite for a real probe-only SAES
schedule on an upstream dense encoder. It is not a quality experiment and does
not select a threshold, bandwidth, sample, or paper claim.

## Question

For a fixed DL3DV context-only sample, can the retained probe Gaussian-head
outputs remain bit-identical if non-probe activations at the full-resolution
depth-refinement input are removed? If they change, a schedule that simply
skips those non-probe S2/S3 values cannot reproduce the pretrained dense path;
it must account for its receptive-field dependencies before its cycle savings
can be treated as executable.

## Protocol

`scripts/saes_dependency_audit.py` supports this first ladder rung for each
DL3DV model at sample 0. It performs two deterministic encoder executions on
context images only:

1. capture the native raw Gaussian-head output;
2. zero every non-probe position in each 4x4 tile at the native dense
   full-resolution input, keep the four corner probes unchanged, and capture
   the same raw head.

TranSplat and MVSplat use their `refine_unet` input. DepthSplat uses its
`gaussian_regressor` input. The audit removes target RGB from the batch before
the context tensors move to the device.

The capture hook raises an internal sentinel immediately after the raw head, so
the Gaussian adapter/S4, renderer, target RGB, and quality metrics never run.
The report compares only raw-head values at the retained probe positions.

## Interpretation

- Any nonzero probe delta proves a dense dependency at this boundary. It does
  not measure image quality or refute SAES as a paper mechanism; it invalidates
  only the claimed direct non-probe bypass for this unmodified execution path.
- An exact zero delta is necessary but not sufficient evidence for a sparse
  schedule. Cost-volume and other upstream dependencies still require separate
  tests.

The report is always `paper_result_eligible=false`. It must be preserved in a
new directory and cannot authorize a quality retry or a full DL3DV run.

## Fixed Result And Accounting Policy

All three fixed DL3DV sample-0 results report `dependency_detected=true` after
target RGB removal:

| Model | Dense input | Changed / retained raw-head values | Result SHA256 |
| --- | --- | --- | --- |
| TranSplat | `refine_unet` | 2,752,512 / 2,752,512 | `8a73027989cfaafce2145b6370d2209450725eb33dfbd7c9afa0e5c5ead82679` |
| MVSplat | `refine_unet` | 2,752,511 / 2,752,512 | `ac5610772240887f7f5004c050fbc0ad08261337dd18b02f3430c7cf00174ea5` |
| DepthSplat | `gaussian_regressor` | 2,121,728 / 2,121,728 | `893747ecb7d3336f90b9f7afdf052cd3d946d8728b37ec5ebf558533a4befd76` |

`saes.execution_dependency` is the explicit per-model cycle-accounting
contract. It currently gives all three supported models a zero SAES S2/S3
bypass fraction:

- All three supported models are `dense_dependency_detected`, each bound to
  its corresponding immutable result above.
- A caller without model identity is also unverified. Unknown model names
  raise an error instead of inheriting another model's evidence.

SAES route counts and the analytic control/assignment/moment/storage ledger
remain recorded, but they cannot reduce S2 or S3 cycles unless a model-specific
contract is positively verified. FSDR continues to apply to every position not
removed by an *executed* SAES S2 bypass, so an unverified SAES route cannot
silently change its denominator. This policy is an accounting safety gate, not
a quality result and not a substitute for the missing RTL numeric datapath.

Every schema-v2.1 result carries the exact contract at
`events.saes.execution_dependency` and its requested fractions at
`events.saes.s2_s3_saving`. `scripts/validate_result.py` resolves the contract
again from `provenance.model`; it rejects a missing or substituted contract and
rejects any positive S2/S3 SAES fraction while that contract is unverified.

## Fixed Locality Follow-Up

`scripts/saes_dependency_locality_audit.py` tests whether a bounded spatial
halo could rescue the rejected direct bypass without changing SAES routing. It
uses the same context-only raw-head boundary, explicitly removes target RGB
before context device transfer, and zeros the twelve non-probe positions of
exactly nine predeclared source tiles: top/center/bottom crossed with
left/center/right. It reports only the changed retained-probe envelope. The
same fixed sample-0 protocol supports TranSplat, MVSplat, and DepthSplat; each
new result must use its own non-overwriting directory.

The fixed result is
`outputs/ae_dl3dv_repair_diagnostics/transplat_sample0_refine_dependency_locality_v1/results.json`
with SHA256
`bb5c24557a39da455869ebdac8804cf3a9c883f9c0df1592297c051ce2583a56`.
Every source-tile perturbation changed all 16,384 retained-probe spatial
positions, whose tile bounds were the complete 64 by 64 grid. The center tile
reached a Chebyshev distance of 32 and every edge/corner tile reached 63. This
rejects a finite local-halo direct-bypass implementation for the current
unmodified TranSplat path. It does not measure quality, establish a sparse
replacement, authorize S2/S3 savings, or justify a DL3DV quality/full retry.

The historical TranSplat record predates the explicit deletion field above. It
transferred only `batch["context"]`, so target RGB never entered the model, but
it did not record removal as a separate provenance assertion. It remains
non-claim negative evidence and is not retroactively promoted. New MVSplat and
DepthSplat locality records use the stricter explicit-deletion contract.

## DepthSplat Local-Adaptor Coverage Gate

DepthSplat's fixed sample has a one-tile retained-probe envelope, unlike the
global TranSplat and MVSplat envelopes. This alone does not create useful
sparse execution. The submitted adaptor is four consecutive same-resolution
3x3 convolutions: `gaussian_regressor.0`, `gaussian_regressor.2`,
`gaussian_head.0`, and `gaussian_head.2`. The pure
`saes.depthsplat_s3_footprint` contract propagates the fixed L0/L1 retained
positions backward through exactly that chain with replicate-padding geometry.

For the native 256x448 DL3DV output, L0 requests 28,672 final-head positions
and L1 requests 57,344, but one backward 3x3 expansion already covers all
114,688 spatial positions for either route. Therefore every preceding adaptor
convolution must execute densely for bit-identical retained outputs. Only the
last Gaussian-head convolution could be locally emitted, which is insufficient
to claim a sparse S3 producer and does not affect the zero S2/S3 saving gate.
No local-adaptor bypass or quality retry is authorized from the finite-halo
measurement.
