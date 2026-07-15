# SCARF Artifact Evaluation Guide (MICRO 2026)

**Paper:** SCARF: A Scene-Adaptive Depth-Guided G-3DGS Encoder
Accelerator with Semantic Reuse and Fused Dataflow

**Badges requested:** Artifacts Available and Artifacts Evaluated (Functional).
Results Reproduced is not requested because the real sparse-SAES probes do not
meet the paper's Table 1/Table 3 contracts and the public FSDR implementation
cannot reproduce Table 2's LSH-guided set.

The exact claim scope, commands, and tolerances are defined in
[`artifact/CLAIMS.md`](artifact/CLAIMS.md). The final archival DOI must be added
to this guide only after the Zenodo release passes clean-room validation.

## Quick Start

```bash
git clone --recursive https://github.com/MAdrid1011/SCARF.git
cd SCARF

# TranSplat and MVSplat environment
bash install.sh --profile classic --venv .venv/classic

# Download the quick-test checkpoint and build the redistributable synthetic fixture
bash data/download_checkpoints.sh --profile quick
.venv/classic/bin/python data/build_quick_dataset.py \
  --output datasets/quick-re10k

# First structured result
bash scripts/run_ae.sh quick
```

Generated files are placed below the shared `outputs/ae/` root unless
`--output-root` is supplied. A successful quick
run creates `results.json` and passes schema validation. Preparation and runtime
measurements in the final release manifest supersede estimates in this guide.
The quick fixture is labeled `re10k-synthetic-functional-v1` and is never used
for a paper-result claim. Full quality runs require the real prepared datasets.

## Components

| Component | Location | Purpose |
|---|---|---|
| Hardware simulator | `encoder/`, `depth_predictor/`, `ggu/` | Functional and cycle models |
| FSDR | `fsdr/` | Feature Similarity Depth Reuse |
| SAES | `saes/` | Scene-Adaptive Early Sparsification |
| Model integration | `adapters/`, `integration/` | TranSplat, MVSplat, and DepthSplat adapters |
| Experiment driver | `scripts/run_ae.py` | Matrix execution, aggregation, and validation |
| Compatibility wrapper | `scripts/run_ae.sh` | Stable shell entry point |
| RTL | `chisel/` | Chisel implementation and tests |
| Public physical flow | `hardware/iflow/` | iFlow/ASAP7 predictive implementation |
| Technology scaling | `hardware/scaling/` | DeepScaleTool normalization |
| AE metadata | `artifact/` | Claims, expected results, manifests, and appendix |

## System Requirements

| Workflow | Minimum public configuration |
|---|---|
| Quick strict inference | NVIDIA CUDA GPU with 8 GB VRAM, 16 GB RAM, 20 GB free disk |
| CPU schema smoke | Ubuntu 22.04, x86-64 CPU, 8 GB RAM |
| Optional diagnostic quality matrix | NVIDIA CUDA GPU with 24 GB VRAM, 32 GB RAM |
| Orin baseline | Jetson Orin NX 16 GB, documented JetPack and MAXN state |
| RTL validation | JDK 11+, sbt 1.9+, Verilator 5+ |
| ASAP7 physical proxy | x86-64 Linux, 128 GB RAM recommended, 100 GB free disk, pinned iFlow checkout |

CPU mode validates the CLI, protocol, result schema, DeepScale, and applicable
RTL tooling. The Gaussian rasterizer makes strict model inference a CUDA
workflow, and workstation CUDA timing is not a substitute for archived Orin NX
performance evidence.

Software and CUDA evaluation do not depend on the physical-flow resource gate.
The ASAP7 workflow alone refuses to start while any Vivado process is active or
while less than 48 GiB of host memory is available.

## Environments

The model repositories require two pinned profiles:

```bash
# PyTorch 2.1 profile for TranSplat and MVSplat
bash install.sh --profile classic --venv .venv/classic

# PyTorch 2.4 profile for DepthSplat
bash install.sh --profile depthsplat --venv .venv/depthsplat

# Jetson-specific instructions and dependency checks
bash install.sh --profile orin --check-only
```

Do not mix the classic and DepthSplat packages in one Python environment. The
orchestrator automatically uses `.venv/classic/bin/python` and
`.venv/depthsplat/bin/python`. For environments installed elsewhere, set
`SCARF_PYTHON_CLASSIC` and `SCARF_PYTHON_DEPTHSPLAT` to their Python
interpreters. Before loading a model, every real workflow validates the profile
and saves its environment record below `outputs/ae/environments/`.

## Data and Checkpoints

The machine-readable sources, formats, model mapping, licenses, and hashes live
in `artifact/manifests/`.

```bash
bash data/download_checkpoints.sh --profile all
bash data/download_re10k.sh
bash data/download_acid.sh
bash data/download_dl3dv.sh
```

DL3DV preparation creates two hash-manifested representations. DepthSplat uses
`datasets/dl3dv/native`, which keeps the native 270 by 480 test images.
TranSplat and MVSplat use `datasets/dl3dv/re10k`, which contains deterministic
360 by 640 images for their Re10K-compatible loaders. The second representation
is resized from the higher-resolution `images_4` source with a fixed conversion
recipe. The experiment mapping rejects a representation that does not match the
selected model.

The committed evaluation index has two deliberately distinct ordinals. Its
non-null source-file order defines the stable `sample_index` and selection
SHA256. Actual evaluation follows the prepared dataset loader's sorted chunk
traversal and records that position as `execution_index`. Keeping both values
allows the reviewer to verify the unchanged scene/view contract while the
runner consumes samples in exactly the order produced by the upstream loader.

The pinned DL3DV benchmark is a gated Hugging Face dataset. Before running its
download command, accept the dataset access terms and authenticate with
`hf auth login`. The script downloads only the required `nerfstudio` metadata
and `images_4` and `images_8` trees. Dataset terms are not inferred from access
approval. The manifest records the reviewed terms URL and explicitly forbids
redistribution in the Zenodo package. Gated access is never bypassed.

Download scripts do not install packages implicitly. They verify required tools
before downloading and reject files whose configured SHA256 does not match.
Full third-party datasets are not copied into the archival package unless their
redistribution terms permit it.

The checkpoint command also prepares hash-pinned runtime assets. These assets
include VGG16 weights for LPIPS, the CC-BY-NC-4.0 Depth Anything V2 Base
backbone required only by TranSplat, and a fixed DINOv2 source snapshot. The
full DINO backbone weights are already contained in the pinned DepthSplat
checkpoints, so the local source constructs them with `pretrained=false` and
never fetches a second copy. Runtime assets are downloaded from pinned public
revisions and are not redistributed in the release archives. Model evaluation
does not download weights or hub code at run time.

DepthSplat ACID evaluation follows the upstream zero-shot protocol. It uses the
official large Re10K checkpoint with the ACID data root and evaluation index,
as documented by the pinned DepthSplat submodule. TranSplat and MVSplat use
their official ACID checkpoints.

The large DepthSplat checkpoint is composed with the upstream evaluation
overrides `num_scales=2`, `upsample_factor=2`,
`lowest_feature_resolution=4`, and `monodepth_vit_type=vitl`. The unclaimed
native DL3DV workflow retains the upstream base-model ViT-B overrides. These
variant choices are part of the experiment manifest rather than inferred by a
permissive checkpoint load.

### Evaluation Sample Protocol

The artifact recovers the committed evaluation indices shared by the three
upstream model repositories. Re10K has 6,474 executable entries, ACID has 1,595,
and DL3DV has 140. Source index counts and rejected null entries are also
recorded. Claim runs use these exact context and target views through the
evaluation sampler. A one-sample run remains Functional evidence only. See the
evaluation protocol document.

## Experiment Status

### Table 1: Rendering Quality

```bash
bash scripts/run_ae.sh quality
```

Current state: `NOT_CLAIMED_SAES_SPARSE_QUALITY_MISMATCH`. The claim-aware
command records an explicit `NO_CLAIMED_PAIRS` no-op; focused diagnosis can run
`scripts/demo.py` directly, but its sparse-SAES result is not part of the badge
claim. Corrected TranSplat and MVSplat probes keep the no-optimization path
numerically identical to the pinned model and preserve FSDR quality, while
sparse SAES exceeds the Table 1 tolerances. Each
sample result covers every target view selected by the configured sampler and
records per-view metrics. The sample metric is their arithmetic mean. Signed
change, degradation, and absolute change are separate fields. The maximum
degradation statement in the paper must not be interpreted as a maximum
absolute deviation.

The optional `--saes-materialization dense-diagnostic` path retains every
Gaussian and is rejected by claim/Functional runs. Its closer image quality is
not accepted as evidence for sparse Gaussian pruning.

### Figure 8: End-to-End Speedup

Current state: `NOT_CLAIMED_NO_ORIN_EVIDENCE`. The command below is retained
for a future evidence-bearing Orin run and produces no current claim work.

```bash
bash scripts/run_ae.sh speedup --num-samples N
```

Simulator cycles are deterministic. The GPU baseline uses archived measurements
from the documented Jetson Orin NX state. The AE path does not estimate Orin
latency from the peak TFLOPS of another GPU.

Figure 8 uses the 1 GHz architectural clock target in the paper. When executed,
the public ASAP7 flow reports achieved timing independently. A routed ASAP7
result that misses 1 GHz remains a visible timing failure and does not validate
the unavailable commercial TSMC28 implementation.

### Figure 11 and Tables 2-3: Ablation and Mechanisms

```bash
bash scripts/run_ae.sh ablation
```

When rows are claimed, the output includes FSDR-only, SAES-only, combined, and
no-optimization results, plus guided-rate, Top-1 coverage, L0/L1, Gaussian, and
memory statistics. Current state: `NOT_CLAIMED_SAES_PROTOCOL_MISMATCH`; the
claim-aware command is therefore an explicit no-op. Real probes produced zero
L1 tiles instead of the nonzero paper targets. Separately, all six FSDR pilots
miss the Guided Rate targets and the tracked RTL projection ROM is all zero;
the software's seed-0 hyperplanes are not paper collateral. The RTL MAC also
interprets its 16-bit operands as unsigned values, so populating the ROM alone
would not establish signed random-hyperplane equivalence. Six fixed 32-sample
prefixes and a separate DepthSplat ViT-L feature diagnostic fail the unchanged
mechanism checks. Tables 2--3 and
Figure 11 are therefore not claimed.

### Figures 13-16: Sensitivity

Current state: `NOT_CLAIMED_INCOMPLETE_NINE_PAIR_MATRIX`. The trace-and-replay
implementation remains available, but the current claim does not run it.

```bash
bash scripts/run_ae.sh sensitivity --num-samples N
```

The configured grids include cache sizes 8-128, Hamming thresholds 1-5, the
feature and depth thresholds from the paper, and tile sizes 2-32. Every grid point runs
all protocol samples and produces a strict dataset aggregate before plotting.

### Complete Declared Workflow

```bash
bash scripts/run_ae.sh all
bash scripts/run_ae.sh validate
```

`all` follows the machine-readable claim status. The six-pair software matrix
is currently diagnostic rather than required claim work; it must not be used to
turn the dense path into sparse evidence. The declared workflow runs RTL, the
public DRAM proxy, report generation, and validation. Physical design and
scaling are also skipped because their machine-readable states are
`NOT_CLAIMED_RESOURCE_LIMIT` and `NOT_CLAIMED_NO_PHYSICAL_INPUT`.
`validate`
returns nonzero when a claimed result is missing, structurally invalid, outside
its tolerance, or based on an unfinalized sample protocol.

## RTL Validation

```bash
bash scripts/run_ae.sh rtl
```

This runs Chisel tests, emits SystemVerilog, and invokes Verilator lint. The
generated RTL and reports include source commit provenance.

## Public DRAM Functional Proxy

```bash
export RAMULATOR_ROOT=/path/to/ramulator2-v2.1.0
export DRAMPOWER_ROOT=/path/to/DRAMPower-v6.0.2
bash scripts/run_ae.sh dram
```

This mode runs a labeled representative smoke trace through pinned Ramulator 2
and DRAMPower. It proves that the public timing and energy chain functions. It
uses LPDDR5 and does not reproduce the LPDDR4X workload result in the paper.
Full workload runs can pass measured address events directly to
`hardware/dram/run.sh`.

## ASAP7 Physical Proxy

Current state: `NOT_CLAIMED_RESOURCE_LIMIT`. The pinned clean-worktree,
container, command, and collateral-hash dry-run passed, but a concurrent
180-design Vivado sweep left less than the required 48 GiB available memory.
No routed PPA is included or claimed. The command below remains the documented
workflow for a sufficiently provisioned host.

```bash
export IFLOW_ROOT=/path/to/clean/iFlow

bash hardware/iflow/run.sh \
  --platform asap7 \
  --stage all \
  --output-dir outputs/physical/asap7
```

The required iFlow commit is `04b4d98`. Use `--dry-run` first to verify the
external checkout and commands without running physical design.

The overlay loads the pinned ASAP7 layer resistance and capacitance table in
global placement, resize, clock-tree synthesis, global routing, and final PPA
reporting. Stage validation requires VDD/VSS special nets and macro-grid
insertion evidence for all four SRAM proxies. `physical_valid=true` still
requires routing, GDS, complete timing and power reports, and zero reported DRC
violations from the detailed router. This route check is not foundry signoff DRC.

ASAP7 is a predictive 7 nm research PDK. The public result is not described as
a TSMC 28 nm measurement. SRAM proxies are reported separately. LPDDR PHY and
pads are excluded. The public hardware scope document defines the boundary.

## DeepScaleTool Normalization

Current state: `NOT_CLAIMED_NO_PHYSICAL_INPUT`. The node-table, published
examples, round-trip, and formula tests are Functional evidence. The command
below intentionally fails without a valid routed ASAP7 `ppa.json` and is not
used to synthesize a result from manuscript constants.

```bash
python hardware/scaling/deepscale.py \
  --source-node 7 \
  --target-node 28 \
  --input outputs/physical/asap7/ppa.json \
  --output outputs/physical/asap7/ppa_28nm_estimated.json
```

The output retains every raw ASAP7 metric and records each scaling factor. It is
labeled `28nm_equivalent_estimate`, not `TSMC28_postlayout`.

## Output Layout

```text
outputs/ae/
|-- manifest-<mode>.json
|-- quick/<model>_<dataset>/samples/sample_00000/results.json
|-- quality/<model>_<dataset>/samples/sample_NNNNN/results.json
|-- quality/<model>_<dataset>/results.json
|-- speedup/<model>_<dataset>/samples/sample_NNNNN/orin-evidence/
|-- speedup/<model>_<dataset>/results.json
|-- ablation/<model>_<dataset>/results.json
|-- sensitivity/results.json
|-- rtl/results.json
|-- dram/results.json
|-- physical/asap7/ppa.json
|-- physical/asap7/ppa_28nm_estimated.json
`-- reports/reproduction_report.md
```

Every numeric result must be traceable to a raw record. Plotting code cannot use
manuscript constants as experimental input.
Each pair saves the full GT, baseline, SCARF, and four-configuration ablation
image set for its first executed sample. Pass `--image-output-policy all` to
`demo.py` only when per-sample image files are required.

## Troubleshooting

- **Submodule is empty:** run `git submodule update --init --recursive`.
- **Wrong model environment:** activate the profile named in the experiment
  manifest. Do not resolve incompatible packages by silently upgrading them.
- **Missing data/checkpoint:** run the exact download command printed by the
  driver and verify the configured hash.
- **CUDA out of memory:** use the documented lower-memory quick profile. Do not
  change the full-claim batch or resolution without marking the result custom.
- **Physical flow rejected:** verify `IFLOW_ROOT`, the pinned commit, a clean
  worktree, and ASAP7 collateral hashes.
- **Full mode rejects the sample count:** recompile the finalized upstream
  protocol or pass `--num-samples N` for a non-final diagnostic run.
- **Validation fails:** inspect `reports/reproduction_report.md`. Never edit generated
  results to match the paper.

## Archival Release

Before entering the DOI in HotCRP:

```bash
export RELEASE_ROOT=/path/to/SCARF-AE-release
export STAGED_REFERENCE_RESULTS="$RELEASE_ROOT/reference_results"
python scripts/stage_reference_results.py \
  --input outputs/ae_final --destination "$STAGED_REFERENCE_RESULTS"
python scripts/build_archive.py \
  --output-dir "$RELEASE_ROOT/v1.0.0" --version v1.0.0 --require-doi \
  --reference-results "$STAGED_REFERENCE_RESULTS"
python scripts/check_release.py \
  --archive "$RELEASE_ROOT/v1.0.0/SCARF-AE-source-v1.0.0.tar.gz" --require-doi
python scripts/check_release.py \
  --archive "$RELEASE_ROOT/v1.0.0/SCARF-AE-evidence-v1.0.0.tar.zst" --require-doi
(cd "$RELEASE_ROOT/v1.0.0" && sha256sum -c SHA256SUMS)
```

Upload only an archive that passes this check. Download the DOI archive into a
new directory and rerun `quick` and `validate`. Complete every item in
[`artifact/CHECKLIST.md`](artifact/CHECKLIST.md) before marking HotCRP ready.
The staged evidence payload is intentionally excluded from Git history. Its
packaged manifest binds every payload file, while `--reference-results` lets a
clean source commit consume that separately staged tree without a self-
referential manifest commit. Build final archives outside the repository so
release files do not make the source worktree dirty.
