# SCARF Public Hardware Flow

The public hardware flow validates the SCARF RTL with Chisel and Verilator,
implements the emitted SystemVerilog with iFlow/ASAP7, and optionally normalizes
the resulting metrics to a 28 nm equivalent estimate with DeepScaleTool.

## Scope

- ASAP7 results are predictive 7 nm research results.
- DeepScaleTool output is a cross-technology estimate.
- Neither is a replacement for foundry-specific TSMC 28 nm signoff.
- LPDDR PHY, pads, and proprietary compiled SRAMs are excluded from public PPA.

The machine-readable scope and factors are in
`artifact/HARDWARE_SCOPE.md`.

The staged SAES retained-descriptor RTL contract is in
[`docs/saes-rtl-contract.md`](../docs/saes-rtl-contract.md). It explicitly
separates the current classifier-only evidence from unimplemented assignment,
moment-matching, and buffer timing.

## Entry Points

```bash
# Chisel tests, SystemVerilog emission, and Verilator lint
bash scripts/run_ae.sh rtl

# Full public physical-design flow
bash hardware/iflow/run.sh \
  --platform asap7 \
  --stage all \
  --output-dir outputs/physical/asap7

# Normalize a parsed ASAP7 PPA record
python hardware/scaling/deepscale.py \
  --source-node 7 \
  --target-node 28 \
  --input outputs/physical/asap7/ppa.json \
  --output outputs/physical/asap7/ppa_28nm_estimated.json
```

`hardware/iflow/run.sh --dry-run` validates paths and prints the exact external
commands without starting synthesis or physical design.

## Required External Tools

- JDK 11 or newer and sbt 1.9 or newer
- Verilator 5 or newer
- iFlow commit `04b4d98` with its ASAP7 platform
- Yosys/OpenROAD versions selected by that pinned iFlow revision

Set `IFLOW_ROOT` to a clean iFlow checkout. The wrapper rejects a dirty checkout
by default because local modifications make provenance ambiguous. Use a separate
checkout for experimentation.

## Outputs

The physical output directory contains:

```text
manifest.json
ppa.json
reports/
logs/
results/
ppa_28nm_estimated.json  # only after scaling
```

`ppa.json` preserves raw ASAP7 values. It is never overwritten by normalization.
The manifest records the SCARF commit, iFlow commit, library hashes, commands,
stage status, report hashes, and excluded blocks.

## Paper Table 4 Mapping

`hardware/iflow/paper_table4.py` fixes the row order and names to the final
area-and-power table in the paper. No tool-specific bucket is emitted in place
of a paper component:

| Paper rows | Public-source mapping |
|---|---|
| MVU: MMCU, VectorALU, BilinearUnit, NormUnit, ActivationUnit | Preserved RTL instance hierarchy; the MVU parent is the sum of these five rows. |
| GGU Array: PositionCalc, CovBuilder, SH_OPGenerator | The corresponding subinstances under all 32 GGU PEs; the parent is their sum. C2W-ray pseudo-mean construction and first/second-moment covariance updates map to the existing `PositionCalc` and `CovBuilder` paper rows, never to a new public SAES row. They are currently software/event-model work only: no emitted RTL datapath, PPA, or timing claim may count them until assignment, moment matching, and retained-descriptor buffering are implemented and reconciled. |
| FSDR Subsystem: LSHHashUnit, CAM Array, FSDR Controller | `lshHash`, `fsdrCache`, and `fsdrCtrl`; the parent is their sum. |
| On-chip Buffers: Weight, Feature, Tile | Explicit abstract SRAM proxies, area only. |
| Control + Interconnect | Remaining synthesized SCARF control and interconnect logic after the three named logic groups. |
| PLL + Clock tree; I/O + LPDDR4X PHY | Explicit `N/A` public-proxy rows. The commercial collateral is unavailable and no paper value fills the gap. Consequently the `Control & Clock` parent is also `N/A` rather than a partial total. |
| Routing / filler | Routed DEF `DIEAREA` minus placed-cell area, area only. |
| Total die | Routed DEF `DIEAREA`; public power remains vectorless logic power and excludes SRAM/PLL/I/O power. |

The output retains both group and child rows, including `I/O & PHY` and
`I/O + LPDDR4X PHY`, so its hierarchy matches the paper table even where the
public implementation cannot supply a measurement.

For every measured logic row, the final OpenROAD report also records a positive
routed leaf-instance count matched by the same mapping. A missing or empty
binding makes `physical_valid=false`; the count is provenance metadata and is
not emitted as an additional Table 4 row.

## SRAM Policy

Large SCARF buffers must be represented by documented abstract SRAM macros.
They must have timing/area views with recorded provenance and may not be expanded
into registers merely to force routing completion. Reports separate standard-cell
logic and SRAM proxy contributions.

## Power Policy

Power is reportable only when the timing library, clock constraint, activity
source, and tool command are present in the manifest. Vectorless or default-
toggle power is labeled as such. Representative VCD-based power is preferred.

## Host Resource Modes

The physical flow never starts while another Vivado process is active. By
default, it also requires at least 48 GiB of `MemAvailable`, which is the
recommended unconstrained host condition for a full routed run. When that
condition is unavailable, an explicit attempt may run the unchanged design:

```bash
bash hardware/iflow/run.sh \
  --platform asap7 \
  --stage all \
  --output-dir outputs/physical/asap7-low-memory \
  --allow-low-memory-attempt
```

The attempt records its initial resource mode, memory/swap state, per-stage
before/after snapshots, `/usr/bin/time -v` logs, and ordinary stage reports.
When an attempt is intentionally stopped for a verified resource limit,
`hardware/iflow/attempt_outcome.py` writes a hashed `attempt-outcome.json`.
It does not reduce the design or alter the physical flow. A complete run is
evaluated only from routing, GDS, STA, power, DRC, and hash checks; an exit
status 137, missing report, or incomplete GDS remains unclaimed evidence.
