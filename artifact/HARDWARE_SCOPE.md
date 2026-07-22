# Public Hardware Evaluation Scope

## Evidence Layers

SCARF presents hardware data in two explicit layers:

1. **ASAP7 raw:** synthesis, placement, routing, timing, and power reports
   from iFlow and the ASAP7 predictive 7 nm platform.
2. **28 nm-equivalent estimate:** ASAP7 metrics transformed with the
   DeepScaleTool model cited by the paper.

Every generated hardware record carries evidence type, source process, target
process, tool revision, library identity, and input hashes. The public flow
does not substitute manuscript values for generated reports.

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

## Scope

The physical flow reports logic separately from SRAM proxies. SRAMs are
represented as macros with explicit provenance; the LPDDR PHY and pad ring are
outside the public ASAP7 die-area and power totals. Raw and scaled values remain
separate fields in the generated result schema.

## Paper-Compatible Hierarchy

The hierarchy follows the final-paper row order: MVU and its children; GGU
Array and its children; FSDR Subsystem and its children; On-chip Buffers and
its children; Control and Clock; I/O and PHY; Routing/filler; and Total die.
Group totals are derived from child rows. Publicly unavailable PLL and I/O/PHY
components remain explicitly represented in the hierarchy metadata.

The released RTL implements the decision controller and the S2/S3 fallback
path. SCARF software materialization records its geometry operations against
the existing GGU Array `PositionCalc` and `CovBuilder` hierarchy. Hardware
records charge only the events implemented by their corresponding RTL or
architectural model.

## Valid Physical Run

`physical_valid` is true when the requested stages complete, timing and area
reports parse, required hashes are present, and route status is recorded. The
result records `wns_ns` and detailed-routing violation counts alongside the
measurement. The public output is therefore directly traceable to the selected
iFlow/ASAP7 run and its DeepScale transformation.
