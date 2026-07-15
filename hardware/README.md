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

## SRAM Policy

Large SCARF buffers must be represented by documented abstract SRAM macros.
They must have timing/area views with recorded provenance and may not be expanded
into registers merely to force routing completion. Reports separate standard-cell
logic and SRAM proxy contributions.

## Power Policy

Power is reportable only when the timing library, clock constraint, activity
source, and tool command are present in the manifest. Vectorless or default-
toggle power is labeled as such. Representative VCD-based power is preferred.

## Host Resource Gate

The physical flow must not start while another Vivado process is active. The
host preflight also requires at least 48 GiB of available memory. A resource
failure, exit status 137, missing routed report, or incomplete GDS leaves the
physical claim unexecuted. The artifact does not reduce the design or replace
missing reports with pilot values.
