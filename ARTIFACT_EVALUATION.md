# SCARF Artifact Guide

SCARF provides a hardware-oriented execution path for TranSplat, MVSplat, and
DepthSplat. The repository contains the model adapters, Python execution
pipeline, Chisel RTL, and a small redistributable quick fixture. Model
checkpoints and evaluation datasets are downloaded separately.

## Quick Start

Build and run the container on a CUDA-capable host:

```bash
docker build -t scarf-ae:1.0.4 .
mkdir -p outputs/quick
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/outputs/quick:/results" scarf-ae:1.0.4
```

The container prepares the quick fixture and runs the default functional
workflow. Generated files are written below the mounted `outputs/quick`
directory.

## Native Setup

Clone the repository with its evaluated model submodules and create the two
supported Python environments:

```bash
git clone --recursive https://github.com/MAdrid1011/SCARF.git
cd SCARF
bash install.sh --profile classic --venv .venv/classic
bash install.sh --profile depthsplat --venv .venv/depthsplat
```

TranSplat and MVSplat use the classic environment. DepthSplat uses the
DepthSplat environment because its PyTorch and CUDA requirements differ. If
the environments live elsewhere, point the runner at their interpreters:

```bash
export SCARF_PYTHON_CLASSIC=/path/to/classic/bin/python
export SCARF_PYTHON_DEPTHSPLAT=/path/to/depthsplat/bin/python
```

To run the bundled quick fixture without a full dataset:

```bash
bash data/download_checkpoints.sh --profile quick
.venv/classic/bin/python data/build_quick_dataset.py --output datasets/quick-re10k
bash scripts/run_ae.sh quick --output-root outputs/ae
```

## External Assets

The source package does not redistribute checkpoints or datasets. Download the
assets required for a full model run:

```bash
bash data/download_checkpoints.sh --profile all
bash data/download_re10k.sh
```

Optional dataset helpers are available for ACID and DL3DV:

```bash
bash data/download_acid.sh
bash data/download_dl3dv.sh
```

The asset manifests in `artifact/manifests/` list the source URL, license, and
destination expected by each adapter. DL3DV access requires credentials
accepted by its upstream host.

## Run Models

Use the demo entry point to run an individual model:

```bash
python scripts/demo.py --model transplat
python scripts/demo.py --model mvsplat
python scripts/demo.py --model depthsplat
```

FSDR and SAES are enabled by default. Either mechanism can be disabled for an
ablation run:

```bash
python scripts/demo.py --model transplat --no-fsdr
python scripts/demo.py --model transplat --no-saes
python scripts/demo.py --model transplat --no-fsdr --no-saes
```

## RTL

The Chisel source is under `chisel/`, and generated SystemVerilog is under
`hardware/orin/rtl/`. Regenerate the checked-in RTL after a Chisel change:

```bash
cd chisel
sbt 'runMain scarf.VerilogEmitter'
cp generated/ScarfTop.sv ../hardware/orin/rtl/ScarfTop.sv
cp generated/split/*.sv ../hardware/orin/rtl/split/
verilator --lint-only -Wno-fatal --top-module ScarfTop ../hardware/orin/rtl/ScarfTop.sv
```

## Documentation

- `README.md`: repository overview and environment setup
- `docs/architecture.md`: accelerator architecture
- `docs/pipeline-architecture.md`: stage-by-stage pipeline
- `docs/fsdr-saes-mechanisms.md`: FSDR and SAES behavior
- `docs/multi-model-demo-guide.md`: model-specific asset and demo commands
