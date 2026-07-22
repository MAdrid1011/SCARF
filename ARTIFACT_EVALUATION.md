# SCARF Artifact Evaluation

**Paper:** SCARF: A Scene-Adaptive Depth-Guided G-3DGS Encoder Accelerator
with Semantic Reuse and Fused Dataflow

**Zenodo DOI:** [10.5281/zenodo.21482385](https://doi.org/10.5281/zenodo.21482385)

**Badge requests:** Artifact Available, Artifacts Evaluated - Functional, and
Results Reproduced.

## Container Quick Start

The default container command builds the pinned classic CUDA environment,
downloads the hash-pinned MVSplat quick checkpoint on first use, and writes a
structured Functional result to the mounted output directory.

```bash
docker build -t scarf-ae:1.0.0 .
mkdir -p outputs/docker-quick
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/outputs/docker-quick:/results" scarf-ae:1.0.0
```

The image targets CUDA 12.1 and uses the repository's pinned Python 3.10
classic profile. Its default command is equivalent to:

```bash
bash data/download_checkpoints.sh --profile quick
"$SCARF_PYTHON_CLASSIC" data/build_quick_dataset.py --output datasets/quick-re10k
bash scripts/run_ae.sh quick --output-root /results
```

Use `docker run --rm --gpus all scarf-ae:1.0.0 --help` to display the
container entry-point contract. The quick fixture validates the installed
end-to-end stack; the full dataset workflows evaluate the paper matrix.

## Native Setup

```bash
git clone --recursive https://github.com/MAdrid1011/SCARF.git
cd SCARF
bash install.sh --profile classic --venv .venv/classic
bash data/download_checkpoints.sh --profile quick
.venv/classic/bin/python data/build_quick_dataset.py --output datasets/quick-re10k
bash scripts/run_ae.sh quick
```

DepthSplat uses its own CUDA environment:

```bash
bash install.sh --profile depthsplat --venv .venv/depthsplat
```

Set `SCARF_PYTHON_CLASSIC` or `SCARF_PYTHON_DEPTHSPLAT` when those interpreters
live outside the default `.venv/` paths. `SCARF_NVCC` selects a matching CUDA
compiler explicitly when automatic discovery is unavailable.

## Requirements

| Workflow | Public configuration |
|---|---|
| Container quick | NVIDIA CUDA GPU with 8 GB VRAM, Docker with GPU support, 20 GB free disk |
| Full quality and mechanism matrix | NVIDIA CUDA GPU with 24 GB VRAM, 32 GB host RAM, selected datasets and checkpoints |
| Orin performance | Jetson Orin NX 16 GB, JetPack/L4T, MAXN, locked clocks, Nsight Systems, and tegrastats |
| RTL | JDK 11+, sbt 1.9+, Chisel 6.6, and Verilator 5+ |
| ASAP7 proxy | x86-64 Linux, Docker, pinned iFlow checkout, and 48 GiB available memory for the full flow |

## Inputs

The asset manifests define URLs, revisions, licenses, paths, and SHA256 values:

```bash
bash data/download_checkpoints.sh --profile all
bash data/download_re10k.sh
bash data/download_acid.sh
bash data/download_dl3dv.sh
```

Re10K and ACID are prepared from their official test sources. DL3DV-Benchmark
is gated; each evaluator accepts its upstream terms with their own account.
The archive never redistributes third-party datasets or checkpoints. The
machine-readable manifests are in `artifact/manifests/`.

## Evaluation Workflows

All workflows write structured provenance records below the selected output
root. The record binds source revision, submodule revisions, environment,
dataset, checkpoint, selection, command, and result hashes.

```bash
# Installation and end-to-end Functional check
bash scripts/run_ae.sh quick --output-root outputs/ae

# Quality and mechanism evaluation on prepared official data
bash scripts/run_ae.sh quality --output-root outputs/ae
bash scripts/run_ae.sh mechanisms --output-root outputs/ae

# Jetson Orin NX measurement
bash scripts/run_ae.sh performance --device orin --output-root outputs/ae

# RTL and public DRAM proxy
bash scripts/run_ae.sh rtl --output-root outputs/ae
bash scripts/run_ae.sh dram --output-root outputs/ae

# Generate figures and validate a completed result set
bash scripts/run_ae.sh report --output-root outputs/ae
python scripts/validate_ae.py --input outputs/ae --require-key-results
```

The evaluation catalog in `artifact/evaluation_catalog.json` maps every
figure/table to its command, evidence class, raw inputs, acceptance rule, and
current generated status. `artifact/CLAIMS.md` defines the evidence contract.

## Orin Reference and Measurement

`artifact/reference_results/orin_nx_reference.csv` provides the normalized
nine-pair Figure 8 reference table, including the original Orin NX baseline,
SCARF Dataflow on Orin NX, and SCARF ASIC speedup columns. The source identifier
and SHA256 are retained in every row, allowing evaluators without an Orin NX to
compare their reports against a stable reference.

An evaluator with a Jetson Orin NX should execute the performance workflow.
That route records CUDA events, Nsight stage traces, tegrastats, device identity,
power mode, clock state, and architectural cycles. The validator compares
generated device records with the archived reference table under the same fixed
protocol.

## Protocol and Verification

The protocol preserves the committed upstream sample order and exact
context/target-view selections. Each dataset aggregate includes every selected
sample record and its hash. Quality uses PSNR, SSIM, and LPIPS; performance uses
the Orin baseline together with SCARF architectural cycles at the paper's 1 GHz
target; mechanism evaluation records the no-optimization, FSDR, SAES, and
combined cycle paths.

Use the clean source archive command after the worktree is committed:

```bash
python scripts/build_archive.py --source-only --require-doi \
  --output /path/to/SCARF-AE-v1.0.0-source-only.tar.gz \
  --prefix SCARF-AE-v1.0.0
python scripts/check_release.py \
  --archive /path/to/SCARF-AE-v1.0.0-source-only.tar.gz --require-doi
```

`scripts/three_badge_readiness.py` reports source-archive, Functional, and
full-result gates separately. The source archive contains the release source,
its manifest, and the redistributable quick fixture; third-party datasets and
checkpoints remain manifest-pinned external inputs.

## Public Hardware Scope

The public physical flow uses iFlow and the ASAP7 predictive 7 nm platform.
DeepScaleTool derives a labeled 28 nm-equivalent estimate from the generated
ASAP7 metrics. The workflow keeps raw implementation data, scaled estimates,
and manuscript comparison targets distinct. See
`artifact/HARDWARE_SCOPE.md` for the hierarchy and scaling contract.
