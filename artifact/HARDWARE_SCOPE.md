# Public Hardware Evaluation Scope

## Evidence Layers

SCARF reports hardware evidence in two distinct layers:

1. **ASAP7 raw**: synthesis, placement, routing, timing, and power reports
   generated with iFlow and the ASAP7 predictive 7 nm platform.
2. **28 nm equivalent estimate**: the raw ASAP7 metrics normalized with the
   DeepScaleTool model cited by the paper.

ASAP7 is a predictive research PDK. Neither layer is described as a TSMC 28 nm
post-layout measurement. Every generated JSON record includes `evidence_type`,
`source_process`, `target_process`, and tool/library hashes.

## Current Execution Status

The release's clean iFlow `04b4d98`/ASAP7/container/collateral dry-run passes.
The routed flow is `NOT_CLAIMED_RESOURCE_LIMIT`. A clean, unchanged
low-memory attempt completed synthesis through filler and reached global
routing, then was stopped after a 15-second observation recorded 172,774
swap-in pages with less than 2 GiB available memory. Its `attempt-outcome.json`,
stage snapshots, logs, and hashes are retained as failure evidence, not PPA.
The default full-run preflight recommends 48 GiB of `MemAvailable`, while the
explicit `--allow-low-memory-attempt` mode runs the same design and labels the
complete host resource record. Consequently DeepScale execution remains
`NOT_CLAIMED_NO_PHYSICAL_INPUT` until a structurally complete PPA exists. Unit
tests still verify the published node table, examples, round trips, and 7-to-28
formulas; no paper number is substituted for missing PPA.

## DeepScaleTool Convention

For a metric with table values `v[current]` and `v[target]`, the scaling factor
and target estimate are:

```text
factor = v[current] / v[target]
target_metric = current_metric / factor
```

The pinned 7-to-28 nm factors are:

| Metric | Factor | 28 nm estimate from 7 nm |
|---|---:|---:|
| Area | 0.011 / 0.35 | 31.8182 x area |
| Delay | 0.53 / 0.67 | 1.2642 x delay |
| Energy | 0.11 / 0.37 | 3.3636 x energy |
| Power | 0.21 / 0.56 | 2.6667 x power |
| Throughput | 1.89 / 1.49 | 0.7884 x throughput |
| Throughput/area | 188.59 / 4.25 | 0.02254 x throughput/area |

The DeepScaleTool paper reports approximately 1% area, 2.5% delay, and 5%
power error against the cited TSMC scaling comparison. It does not establish an
equivalent bound for energy, and multi-node extrapolation can accumulate error.
The artifact therefore publishes nominal estimates and model notes, not a false
precision interval.

## Included and Excluded Components

The public physical run reports logic separately from SRAM proxies. SRAMs are
abstract macros with explicit provenance rather than expanded flip-flop arrays.
The LPDDR PHY and pad ring are not modeled by ASAP7 and are excluded from the
public die-area and power totals. Paper values for those blocks are never used
to fill gaps in generated reports.

## Paper Table 4 Compatibility

The public hierarchy uses the exact final-paper row sequence: MVU and its five
children; GGU Array and its three children; FSDR Subsystem and its three
children; On-chip Buffers and its three children; Control & Clock, Control +
Interconnect, PLL + Clock tree; I/O & PHY, I/O + LPDDR4X PHY; Routing / filler;
and Total die. Group totals are derived from their child rows. Publicly
unmodeled PLL and I/O/PHY rows remain present with null ASAP7 and DeepScale
values, while the paper's TSMC28 values remain comparison targets only.

The released SAES software moment construction uses static C2W camera geometry,
probe depths, and the already selected assignment weights. Its geometry
operations map to the existing `GGU Array` children `PositionCalc` and
`CovBuilder`; it adds no new hierarchy node, Table 4 row, or hidden area/power
bucket. The current emitted RTL implements only the decision controller and a
conservative full S2/S3 fallback for L0/L1; it does not yet implement the
assignment, moment-matching, or retained-descriptor data path. Consequently no
RTL timing, PPA, or speedup record may charge those software events.

For the public proxy, the complete-measurement groups (MVU, GGU Array, FSDR
Subsystem, and On-chip Buffers) are summed from their children. `Control &
Clock` remains null because its paper parent includes the unavailable PLL, and
`I/O & PHY` remains null because the LPDDR4X PHY is commercial collateral. This
preserves the paper's row structure without presenting partial rows as complete
measurements.

Each measured logic mapping additionally records the number of routed DEF leaf
instances it matched. This is a fail-closed provenance check: an empty module
binding invalidates the public PPA record, while the count remains metadata and
does not add a tool-specific row to the paper-compatible hierarchy.

## Valid Physical Run

`physical_valid` is true only when the selected stages complete, timing and
area reports parse successfully, required hashes are present, and the route
status is recorded. A failed or partial route remains useful diagnostic data
but cannot satisfy claim C7.

`physical_valid` means that the public routed evidence is structurally complete.
The separate `wns_ns` value states whether the 1 GHz target closes. A complete
routed run with negative slack may reproduce the public implementation result,
but it does not support the commercial-process frequency claim in the paper.
The zero-DRC field counts OpenROAD detailed-routing violations. It is not a
foundry signoff DRC result.
