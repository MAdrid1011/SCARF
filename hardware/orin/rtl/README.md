# Emitted Orin RTL

`ScarfTop.sv` is the monolithic SystemVerilog source emitted by
`scripts/run_rtl.sh` from this release commit. The supporting split modules
are also retained under `split/`, together with a filelist that excludes the
empty emitter placeholder for the top-level module. The monolithic file is the
source RTL input
for a source-bound timing export; the emitter, Chisel tests, and Verilator lint
can be rerun with:

```bash
bash scripts/run_rtl.sh --output-dir outputs/rtl
```

For claim workflows, the platform timing run must bind this source (or a
newly emitted byte-identical copy) and provide one hashed trace for every
model/dataset/sample and all four variants in
`source-bound-timing-backend-v1`. Install that export with
`scripts/install_claim_prerequisites.py` before invoking the quality,
mechanisms, and performance commands. The diagnostic VCD produced by the RTL
validation run is an engineering check and is not substituted for those
per-sample timing traces.
