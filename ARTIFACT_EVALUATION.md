# SCARF — Artifact Evaluation Guide (MICRO 2026)

**Paper:** SCARF: A Scene-Adaptive Depth-Guided G-3DGS Encoder Accelerator
with Semantic Reuse and Fused Dataflow

**Badges claimed:** Artifacts Available · Artifacts Evaluated — Functional · Results Reproduced

---

## Quick Start (< 5 minutes to first result)

```bash
# 1. Clone and set up environment
git clone https://github.com/MAdrid1011/SCARF
cd SCARF
conda create -n scarf python=3.10 -y && conda activate scarf
bash install.sh

# 2. Download model checkpoints (~3 GB)
bash data/download_checkpoints.sh

# 3. Smoke-test on Re10K × MVSplat (~10 min on GPU)
bash scripts/run_ae.sh quick

# 4. Full reproduction of all paper results (~3 h on GPU)
bash scripts/run_ae.sh all
```

All results are written to `outputs/ae_<timestamp>/`.

---

## What This Artifact Provides

| Component | Location | Description |
|-----------|----------|-------------|
| Hardware simulator | `encoder/` | Cycle-accurate Python HW units (ConvEngine, GEMMUnit, etc.) |
| FSDR | `fsdr/` | Feature Similarity Depth Reuse implementation |
| SAES | `saes/` | Scene-Adaptive Early Sparsification implementation |
| Model adapters | `adapters/` | MVSplat, TranSplat, DepthSplat integration |
| Demo script | `scripts/demo.py` | Single model × dataset evaluation |
| AE sweep script | `scripts/run_ae.sh` | Reproduces all paper figures and tables |
| Sensitivity sweep | `scripts/sensitivity_sweep.py` | Figure 7 parameter sensitivity |
| RTL implementation | `Zircon-SCARF/` | Chisel/Verilator full RTL (optional, ~8 h) |

---

## System Requirements

| Requirement | Minimum | Tested |
|-------------|---------|--------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| Python | 3.9+ | 3.10.14 |
| GPU | 8 GB VRAM (CUDA 12.1+) | RTX 3060 12 GB |
| RAM | 16 GB | 32 GB |
| Disk | 15 GB | — |
| CUDA | 12.1+ | 12.1 |

**CPU-only mode** is supported (`--device cpu`) but is ~5× slower.
No FPGA or ASIC hardware is required for any paper result.

---

## Installation

```bash
conda create -n scarf python=3.10 -y
conda activate scarf
bash install.sh          # installs PyTorch 2.1.2+cu121 + all deps
```

Alternatively, install manually:

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Data and Checkpoint Downloads

```bash
bash data/download_checkpoints.sh   # TranSplat, MVSplat, DepthSplat (~3 GB)
bash data/download_re10k.sh         # RealEstate10K test split (~3 GB)
bash data/download_acid.sh          # ACID test split (~3 GB)
bash data/download_dl3dv.sh         # DL3DV test split (~4 GB)
```

**Pre-packaged test vectors** (Re10K only, ~200 MB, sufficient for quality
reproduction) are available at the Zenodo archive:

```bash
wget https://zenodo.org/record/XXXXXXX/files/testvecs.tar.gz
tar -xzf testvecs.tar.gz
```

---

## Reproducing Paper Results

### Table 2 — Quality (PSNR / SSIM / LPIPS)

```bash
bash scripts/run_ae.sh quality
```

Expected: PSNR deviation from baseline ≤ 0.072% across all nine
model × dataset pairs. Results in `outputs/ae_<ts>/quality/*/results.json`.

| Model | Dataset | Baseline PSNR | SCARF PSNR | Δ PSNR |
|-------|---------|---------------|------------|--------|
| TranSplat | Re10K | 28.08 | 28.07 | −0.04% |
| TranSplat | ACID  | 29.58 | 29.60 | +0.07% |
| TranSplat | DL3DV | 25.08 | 25.07 | −0.04% |
| MVSplat   | Re10K | 30.22 | 30.23 | +0.03% |
| MVSplat   | ACID  | 32.08 | 32.06 | −0.06% |
| MVSplat   | DL3DV | 28.02 | 28.05 | +0.11% |
| DepthSplat| Re10K | 28.43 | 28.42 | −0.04% |
| DepthSplat| ACID  | 29.43 | 29.44 | +0.03% |
| DepthSplat| DL3DV | 26.93 | 26.98 | +0.19% |

### Figure 5 — End-to-End Speedup

```bash
bash scripts/run_ae.sh speedup
```

Expected: geometric-mean speedup ≥ 2.8× (paper: 2.94×).
Minor variance (±3%) is expected from GPU scheduling.

### Figure 6 — FSDR / SAES Ablation

```bash
bash scripts/run_ae.sh ablation
```

Expected: FSDR-only and SAES-only contributions visible;
combined bar matches paper values within run-to-run variance.

### Figure 7 — Sensitivity Sweep

```bash
bash scripts/run_ae.sh sensitivity
```

Expected: monotone PSNR trends for cache size (64→1024) and
Hamming threshold (1→5) as described in Section 5.3.

### All experiments at once

```bash
bash scripts/run_ae.sh all     # ~3 h on GPU
```

---

## RTL Validation (Optional — ~8 h)

The `Zircon-SCARF/` directory contains the fully synthesizable
Chisel/Verilator RTL implementation used to validate the data path.

Requirements: Scala 2.13, sbt 1.9+, Chisel 6.6, Verilator 5.029+.

```bash
cd Zircon-SCARF
# Verify Stage-A RTL chain: full pipeline PSNR drop ≤ 0.5 dB
make audit_stage_a_current
# Expected output: stage_a_current_audit_ok=True
```

Full end-to-end RTL simulation (~8 h):

```bash
bash golden/run_chain_full_e2e.sh
```

---

## Directory Structure

```
SCARF/
├── encoder/               # Hardware unit simulators
├── fsdr/                  # FSDR implementation
├── saes/                  # SAES implementation
├── adapters/              # Model adapters (MVSplat, TranSplat, DepthSplat)
├── integration/           # Model loader and data bundle
├── ggu/                   # Gaussian Generation Unit simulator
├── scripts/
│   ├── demo.py            # Single experiment entry point
│   ├── run_ae.sh          # AE reproduction master script
│   └── sensitivity_sweep.py
├── data/                  # Dataset download scripts
├── outputs/               # Experiment results (generated)
├── Zircon-SCARF/          # Full RTL implementation (optional)
├── requirements.txt
├── install.sh
└── ARTIFACT_EVALUATION.md  # This file
```

---

## Troubleshooting

**CUDA out of memory**: reduce batch size with `--num-samples 1`, or use
`--device cpu`.

**Dataset not found**: ensure `data/download_*.sh` completed successfully,
or use the pre-packaged `testvecs.tar.gz` from Zenodo.

**Checkpoint not found**: re-run `bash data/download_checkpoints.sh` and
verify checksums printed at the end.

**Run-to-run variance**: GPU timing variance of ±2–3% is normal and does
not affect Table 2 conclusions (quality metrics are deterministic given
fixed inputs).

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@inproceedings{scarf2026micro,
  title     = {{SCARF}: A Scene-Adaptive Depth-Guided {G-3DGS} Encoder
               Accelerator with Semantic Reuse and Fused Dataflow},
  booktitle = {Proceedings of the 59th IEEE/ACM International Symposium
               on Microarchitecture (MICRO)},
  year      = {2026},
}
```
