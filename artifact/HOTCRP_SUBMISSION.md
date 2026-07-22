# MICRO 2026 AE Submission Metadata

## Badges Requested

- Artifact Available
- Artifacts Evaluated - Functional
- Results Reproduced

## Zenodo Archive

DOI: [10.5281/zenodo.21482385](https://doi.org/10.5281/zenodo.21482385)

The DOI, release metadata, archive manifest, source revision, submodule
revisions, and SHA256 checksums identify one release. The archive metadata is
stored in `artifact/release.json`.

## Reviewer Entry Points

```bash
# CUDA Functional workflow
docker build -t scarf-ae:1.0.0 .
docker run --rm --gpus all scarf-ae:1.0.0

# Native Functional workflow
bash scripts/run_ae.sh quick --output-root outputs/ae

# Full result validation after the evaluation matrix completes
python scripts/validate_ae.py --input outputs/ae --require-key-results
```

## Key Results

| Result | Entry point | Record surface |
|---|---|---|
| Figure 8 performance | `run_ae.sh performance --device orin` | Orin CUDA events, Nsight, thermal/device state, and ASIC cycles |
| Table 1 quality | `run_ae.sh quality` | Canonical target-view PSNR, SSIM, and LPIPS records |
| Figure 11 mechanisms | `run_ae.sh mechanisms` | No-opt, FSDR, SAES, combined cycles, and Tables 2-3 counters |

The execution contract is machine readable in `artifact/evaluation_catalog.json`.
Every generated record includes source, environment, dataset, checkpoint,
selection, raw-artifact, and command hashes.

## Orin Reference

`artifact/reference_results/orin_nx_reference.csv` is the nine-pair normalized
Figure 8 comparison table. Reviewers without an Orin NX can use it as the
archived comparison baseline. Reviewers with an Orin NX can run the measurement
entry point above; the generated device record is compared against the same
table by the validator.

## Dependencies

- Container quick: CUDA 12.1-compatible NVIDIA GPU with 8 GB VRAM.
- Quality/mechanism matrix: CUDA GPU with 24 GB VRAM and official data.
- Figure 8: Jetson Orin NX 16 GB in MAXN with locked clocks.
- RTL: JDK 11+, sbt 1.9+, Chisel 6.6, and Verilator 5+.
- ASAP7 proxy: x86-64 Linux, Docker, pinned iFlow checkout, and 48 GiB
  available memory for the complete flow.

Dataset access, checkpoint identities, hardware scope, and archive commands
are documented in `ARTIFACT_EVALUATION.md`.
