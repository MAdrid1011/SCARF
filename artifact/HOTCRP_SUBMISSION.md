# MICRO 2026 AE Submission Fields

This document defines the intended three-badge submission. Do not mark HotCRP
ready until `scripts/validate_ae.py --require-key-results` passes and the DOI
archive has passed the final clean-room check.

## Badges Applied For

- Artifact Available
- Artifacts Evaluated - Functional
- Results Reproduced

## DOI URL

Pending the final Zenodo upload. The DOI must resolve to the exact source and
evidence archives named in `SHA256SUMS` and must match the appendix and release
metadata.

## Key Results to be Reproduced

1. **Figure 8, end-to-end performance.** On a real Jetson Orin NX 16 GB,
   reproduce the nine-pair original-GPU and SCARF-Dataflow CUDA-event timings,
   combine them with positive SCARF architectural cycles at 1 GHz, and validate
   the reported 2.94x geometric-mean ASIC speedup and S1-S4 breakdown.
2. **Table 1 and Figure 10, quality.** Reproduce the nine model/dataset quality
   rows and deterministically regenerate the global worst FSDR/SAES views,
   stage cycles, rendered images, and RGB error maps from all evaluated views.
3. **Figure 11 and Tables 2-3, mechanisms.** Reproduce no-optimization,
   FSDR-only, SAES-only, and combined cycles plus discrete FSDR and SAES event
   counters for all nine model/dataset pairs.
4. **Figure 12, utilization.** Reproduce S1-S3 useful/scheduled MMCU slot ratios
   from simulator event traces.
5. **Figures 13-16, sensitivity.** Reproduce the complete five-point cache,
   Hamming, feature-threshold, depth-threshold, and tile-size grids using fixed
   per-sample traces and all nine model/dataset pairs.
6. **Public hardware proxy.** Regenerate routed ASAP7 PPA, the deterministic
   DeepScale 7-to-28 nm normalization, and the public counterparts to Figure 9
   and Table 4. These are not described as the paper's commercial TSMC28
   post-layout measurements.

All commands emit structured records with git, submodule, environment, data,
checkpoint, selection, raw-artifact, and command hashes. Paper constants are
validator-only comparison targets.

All software results use one frozen global mechanism configuration. Its
engineering bandwidths and conservative FSDR hit-validity tolerance are chosen
from a pre-registered finite grid on 64 official training scenes that are
hash-proven disjoint from evaluation. Calibration cannot access target RGB,
ground truth, manuscript result files, or evaluator outputs. The reviewer may
run the deterministic reviewer profile or shard the full profile; both retain
the same canonical views and result schema.

## Hardware Dependencies

- **Mandatory Figure 8:** Jetson Orin NX 16 GB in MAXN mode with GPU locked at
  918 MHz, active cooling, and temperature below 80 C. At least one evaluator
  must run this workflow on real hardware. If no assigned evaluator has the
  device, the AE FAQ permits chairs to broker exceptional remote access for
  rare hardware.
- **Quality and mechanism experiments:** NVIDIA CUDA GPU with at least 24 GB
  VRAM, x86-64 host with 32 GB RAM, and sufficient disk for the selected data
  and checkpoints. Deterministic shards can run on multiple equivalent GPUs.
- **Quick Functional check:** NVIDIA CUDA GPU with at least 8 GB VRAM, 16 GB
  host RAM, and 20 GB free disk.
- **CPU/schema and RTL:** x86-64 Linux with 16 GB RAM; JDK 11+, sbt 1.9+,
  Chisel 6.6, and Verilator 5+ for RTL.
- **ASAP7 physical proxy:** x86-64 Linux with 128 GB RAM recommended, 100 GB
  free disk, Docker, and the pinned iFlow checkout. This workflow will not run
  while Vivado is active. Its default full-run preflight requires 48 GiB
  available memory; the documented low-memory attempt retains resource logs
  and never promotes incomplete reports.

## Software Dependencies

- Ubuntu 22.04 and the pinned Python 3.10 classic, DepthSplat, and Orin profiles.
- NVIDIA JetPack/L4T, CUDA, PyTorch, Nsight Systems, `tegrastats`,
  `nvpmodel`, and `jetson_clocks` on the Orin evaluator.
- Official TranSplat, MVSplat, and DepthSplat revisions and checkpoints named
  in the manifests.
- iFlow commit `04b4d98` and its ASAP7 platform; Ramulator 2 and DRAMPower for
  the public memory workflow.
- No proprietary tool is required for the released proxy. The commercial TSMC
  process and memory-compiler collateral are not distributed.

## Data Dependencies

- Re10K and ACID are downloaded from their official prepared test sources and
  verified against the committed tree hashes.
- DL3DV-Benchmark is official auto-gated data. Each evaluator accepts the
  upstream terms with their own account; the artifact downloads only required
  scenes and never redistributes the dataset in Zenodo.
- Six official model checkpoints and pinned LPIPS/DINO runtime assets are
  downloaded and SHA256-verified. No runtime code or weight download is
  permitted during a claim run.

## Submission Safety Checks

- The abstract and appendix say post-layout design, not fabricated chip.
- Figure 9/Table 4 distinguish ASAP7 raw, DeepScale estimate, and paper TSMC28
  target.
- Figure 8 cannot pass with RTX timing, handwritten data, a paper constant, or
  a result missing CUDA events, Nsight/thermal evidence, selection hashes, or
  real Orin device identity.
- The source/evidence archives, release tag, submodule revisions, DOI, and
  `SHA256SUMS` must describe one clean commit.
