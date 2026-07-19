# SAES RTL Data-Path Contract

This document defines the staged, reviewer-visible contract for replacing the
current SAES control-only RTL with a real retained-descriptor data path.  It
does not authorize a timing, PPA, or speedup claim by itself.

## Scope and status

The emitted RTL currently has a tested L0-to-L1-to-Full classifier and, after
classification, conservatively executes the ordinary S2/S3 path.  This is
intentional until the following three components are connected and reconciled
to the software event trace:

1. probe and selected-L1-anchor execution,
2. bilateral assignment plus first/second-moment matching, and
3. retained-descriptor storage and S4 hand-off.

The first implementation increment, `SAESDescriptorBuffer`, is complete as a
standalone emitted-and-tested storage primitive. It is not yet connected to the
top-level pipeline, so it is not evidence that sparse work is already
implemented.

## Paper-to-RTL boundary

SAES preserves its published decision order: probe feature variance, then
probe depth standard deviation, then Full.  Descriptor storage and numeric
stability checks do not feed a fourth routing condition. The buffer receives
only descriptors selected by that route. It never restores or reads a skipped
non-probe descriptor.

The software path represents a retained descriptor as world-space mean,
symmetric covariance, SH coefficients, and opacity.  The event ledger fixes
the externally visible compact layout:

| Field | Storage format | Bytes |
|---|---:|---:|
| Mean | three FP32 values | 12 |
| Covariance | six FP32 upper-triangular values | 24 |
| SH | `3*(degree+1)^2` FP16 values | `6*(degree+1)^2` |
| Opacity | FP16 | 2 |

The descriptor buffer stores this packed representation as 128-bit words.  It
uses 6 words at SH degree 2 and 12 words at degree 4.  These counts are derived
from the layout above and the existing 128-bit public memory width. They are
not fitted to a quality result.

## Buffer interface contract

`SAESDescriptorBuffer` has 32 descriptor slots, matching the existing GGU
array width.  A descriptor is visible to a reader only after its terminal beat
is written at the expected packed length.  Reads are synchronous, one 128-bit
beat at a time, and an invalid index, beat, or incomplete descriptor returns
`readValid=false`.  A new first beat invalidates a prior completed descriptor
at that slot, preventing stale output reuse.

The buffer must be driven by selected L0/L1 anchors only. The intended path
counts are four L0 descriptors and twelve L1 descriptors for a 4x4 tile with
one primitive per pixel. These are interface bounds, not a claim that the
surrounding pipeline already executes the sparse paths.

## Verification and claim boundary

Tests must prove:

- degree-dependent packed beat counts and invalid-range rejection,
- write/read ordering and completion visibility,
- overwrite invalidation, and
- agreement between buffer beat counts and `saes.hardware_accounting` traffic.

`SAESDescriptorBufferTest` currently proves the degree-2 and degree-4 beat
counts, ordered completion, overwrite invalidation, and malformed-stream
rejection. SystemVerilog emission is covered by `VerilogEmitTest`. A later
Python/RTL replay will bind dynamic retained-anchor counts and buffer traffic to
the same per-tile event trace before the final bullet can pass.

The separate stage-event simulator is not that replay. It accepts caller-defined
event counts, resource lanes, cycle costs, and dependencies, and it has no demo
call site. Current runtime statistics also omit direct primary-probe,
secondary-probe, and Full replay counts. Its schedule length cannot be reported
as RTL-cycle-equivalent timing.

## Retained-output scheduler contract

`SAESRetainedOutputScheduler` is the next control increment for the submitted
T=4 configuration. After an already-completed L0 route it requests native
S2/S3 descriptors in the exact software order `[0, 3, 12, 15]`. After L1 it
preserves that prefix and requests the eight boundary descriptors
`[1, 2, 4, 7, 8, 11, 13, 14]`, for twelve descriptors in total. The complete L1
order is `[0, 3, 12, 15, 1, 2, 4, 7, 8, 11, 13, 14]`. The scheduler advances
only after an upstream producer asserts `upstreamDescriptorValid`. It has no
descriptor-data input, no interpolation, and no route decision input beyond the
accepted L0/L1 result.

The scheduler is deliberately not wired into `ScarfTop`: the current dense
models have not supplied a sparse S2/S3 producer, and connecting a constant or
unproven descriptor source would misrepresent execution. Chisel tests prove
the fixed request sequence, request backpressure, L0-prefix preservation, and
rejection of Full/overlapping starts. The pure Python reference uses the same
`saes.probe_layout` coordinates. This is an executable hand-off contract, not
evidence of sparse execution, RTL timing, PPA, or a quality result.

## Scalar moment-core contract

`SAESScalarMomentAccumulator` is now implemented and emitted as a standalone
reusable numeric lane for the existing first/second-moment operation. It has no
feature, depth, tile, or target-RGB input and cannot make a routing decision.
Its input values
are signed fixed-point integers in caller-defined units: means use one unit and
variances use its square. Weights are nonnegative integers in one shared unit.
The representation is scale-invariant as long as all weights in one merge use
the same unit.

For a base descriptor and each accepted pseudo descriptor, the core accumulates

\[
M_0=\sum_j w_j,\qquad
M_1=\sum_jw_j\mu_j,\qquad
M_2=\sum_jw_j(\Sigma_j+\mu_j^2).
\]

On a separate `finish` cycle it emits

\[
\mu=\operatorname{trunc}(M_1/M_0),\qquad
\Sigma=\max(0,\operatorname{trunc}(M_2/M_0)-\mu^2).
\]

Division truncates toward zero, and the nonnegative clamp protects a covariance
diagonal against finite-precision cancellation. The Python reference uses the
identical integer rule. A vector/descriptor wrapper may replicate these lanes
for the three means, six covariance terms, SH averages, and opacity average only
after the scalar replay passes.

Each update consumes one core cycle. `finish` produces a one-cycle `done`
record. Chisel and Python tests share an exact synthetic merge vector and a
constant-descriptor case. These local core cycles are not yet the analytic-ledger
cycle count: assignment generation, descriptor packing, and top-level
scheduling remain unintegrated.

## Assignment-normalization contract

`SAESAssignmentNormalizer` accepts between one and eight nonnegative bilateral
kernel scores after their spatial/feature (and, for L1, depth-reliability)
terms have been computed. It performs no routing and does not inspect feature
or depth values itself. The output is a deterministic Q0.16 distribution:

\[
q_p=\left\lfloor \frac{s_p 2^{16}}{\sum_j s_j}\right\rfloor.
\]

The integer residual is added to the lowest-index maximum-score anchor. This
makes the emitted weights sum exactly to \(2^{16}\), fixes a deterministic tie
rule, and avoids a hidden renormalization pass. Invalid anchor counts and an
all-zero score vector fail closed. The reference module accepts integer scores
only, so it does not choose kernel bandwidths or inspect any evaluation data.

The normalizer is a standalone lane until the bilateral-score producer, scalar
moment lanes, descriptor packer, and tile FSM are integrated. Its one-cycle
completion record is not included in the current analytic ledger.

Later work will add a fixed-point or floating-point moment core only after its
format, rounding rule, accumulator bounds, and Python bit-equivalent reference
are documented.  Until that core and the buffer are integrated into the tile
pipeline and checked against per-path event traces, the analytic ledger remains
`analytic_no_overlap_not_rtl_cycle_equivalent` and no SAES RTL saving may be
reported.
