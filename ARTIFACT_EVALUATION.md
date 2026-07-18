# SCARF Artifact Evaluation Guide (MICRO 2026)

**Paper:** SCARF: A Scene-Adaptive Depth-Guided G-3DGS Encoder
Accelerator with Semantic Reuse and Fused Dataflow

**Badges requested:** Artifact Available, Artifacts Evaluated (Functional), and
Results Reproduced. Figure 8 is a mandatory key result and requires an
independent run on a real Jetson Orin NX. Listing a result in the submission
contract does not mark it complete: every key result remains `NOT_RUN` or
`FAIL` until its raw evidence passes strict validation. Figure 8 is currently
`CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION`: it is included in the requested
badge scope but still has no independent measurement and is not `PASS`.

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
| Full quality/mechanism matrix | NVIDIA CUDA GPU with 24 GB VRAM, 32 GB RAM |
| Mandatory Figure 8 baseline | Jetson Orin NX 16 GB, documented JetPack, MAXN, and 918 MHz state |
| RTL validation | JDK 11+, sbt 1.9+, Verilator 5+ |
| ASAP7 physical proxy | x86-64 Linux, 128 GB RAM recommended, 100 GB free disk, pinned iFlow checkout |

CPU mode validates the CLI, protocol, result schema, DeepScale, and applicable
RTL tooling. The Gaussian rasterizer makes strict model inference a CUDA
workflow, and workstation CUDA timing is not a substitute for archived Orin NX
performance evidence.

Software and CUDA evaluation do not depend on the physical-flow resource gate.
The ASAP7 workflow always refuses to start while any Vivado process is active.
Its default full-run preflight requires 48 GiB of `MemAvailable`; an explicit
`--allow-low-memory-attempt` runs the unchanged flow below that recommendation
and records resource/swap snapshots without promoting incomplete reports.

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
Set `SCARF_NVCC=/path/to/nvcc` only when automatic discovery cannot locate the
compiler matching the locked CUDA release; a mismatched compiler is rejected.

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
`datasets/dl3dv/native`, which keeps the native 270 by 480 test images. At the
pinned `DL3DV-10K-Benchmark` revision, scene source resolutions
vary. The download first reads each pinned `transforms.json`, then emits a
scene-level source plan: native uses the unique 270 by 480 tree (`images_8`
for 2160p scenes and `images_4` for the one 1080p scene). This preserves the
official target pixels without an extra resize.
TranSplat and MVSplat use `datasets/dl3dv/re10k`, which contains deterministic
360 by 640 images for their Re10K-compatible loaders. The second representation
is resized from the unique 540 by 960 tree (`images_4` for 2160p scenes and
`images_2` for the one 1080p scene) with a fixed conversion recipe. The
conversion record embeds and hashes the same source plan, and the experiment
mapping rejects a representation that does not match the selected model.

The committed evaluation index has two deliberately distinct ordinals. Its
non-null source-file order defines the stable `sample_index` and selection
SHA256. Actual evaluation follows the prepared dataset loader's sorted chunk
traversal and records that position as `execution_index`. Keeping both values
allows the reviewer to verify the unchanged scene/view contract while the
runner consumes samples in exactly the order produced by the upstream loader.

The pinned DL3DV benchmark is a gated Hugging Face dataset. Before running its
download command, accept the dataset access terms and authenticate with
`hf auth login`. The script downloads only the required `nerfstudio` metadata
and per-scene image trees named by that source plan. Dataset terms are not inferred from access
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

### Calibration and Reviewer Profiles

The unresolved engineering constants are selected once under the isolated
contract in [`artifact/CALIBRATION.md`](artifact/CALIBRATION.md). The primary
source is 24 official gated DL3DV calibration scenes plus eight DL3DV holdout
scenes, selected by separate domain-separated SHA256 orderings from a pinned
DL3DV archive tree and disjoint from the 140-scene evaluation index. Calibration
cannot read target RGB, ground truth, manuscript tables,
`artifact/expected_results.json`, or completed evaluation outputs. One global
configuration is frozen for all nine pairs before evaluation begins.

Calibration is a pre-submission, author-side provenance operation, not a
reviewer setup step. The selected gated scene archives are neither redistributed
nor required by `quick`, `pilot`, `all-eval --profile reviewer`, or `all-eval
--profile full`. Those workflows consume only the SHA256-bound frozen
`artifact/mechanism_config.json`; the evidence package records the calibration
manifest and candidate-record digests so the selected configuration remains
auditable without redistributing upstream images.

The `full` evidence profile consumes every executable upstream index entry. The
`reviewer` profile uses 512 hash-selected Re10K entries, 512 ACID entries, and
all 140 DL3DV entries while preserving the exact scene/view records. Reviewer
selection is compiled before mechanism evaluation and cannot be changed in
response to results. Both profiles emit the same schema and figure/table
catalog; only `full` is used for the complete author-side evidence set.

Author-side planning, before any gated image download:

```bash
python data/download_dl3dv_calibration.py \
  --write-plan outputs/calibration/dl3dv-download-plan.json \
  --evaluation-index depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
  --revision 5902ed6d707cc13a7779907c1e096676f7707971
```

The plan is intentionally `PLANNED_AWAITING_UPSTREAM_ACCESS` until the account
has accepted the official source terms and the selected archives are downloaded
and prepared. The plan downloader re-reads the official tree before it fetches
the 32 selected archives, validates every byte count and bound upstream object
id, records the actual archive SHA256, extracts all 32 selected scenes with
their train/holdout split provenance, and emits target-free native and
Re10K-compatible sidecars. The legacy Re10K/ACID `run_ae.sh calibrate` command
is not a substitute for this DL3DV plan and cannot produce Results Reproduced
evidence.

Once the author-side protocol exists, `python scripts/run_ae.py calibrate
--calibration-manifest outputs/calibration/dl3dv-protocol/manifest.json
--output-root outputs/calibration/dl3dv-run` executes the fixed train-grid,
exact-tuple holdout, and configuration-freeze sequence. It rejects a missing
holdout, reranking attempt, changed sidecar provenance, or failed holdout
quality gate.

Reviewer and evaluation evidence paths, which never invoke calibration:

```bash
bash scripts/run_ae.sh pilot --pairs all --output-root outputs/ae_pilot
bash scripts/run_ae.sh all-eval --profile reviewer --output-root outputs/ae_reviewer
bash scripts/run_ae.sh all-eval --profile full --output-root outputs/ae
```

FSDR calibration may qualify a Hamming hit using depth evidence already
available from the paper's probe-first tile schedule. SAES calibration may set
the bandwidths used by the published bilateral assignment and depth-reliability
equations. It may not add another SAES routing signal, search a projection seed,
or select pair-specific parameters. These paths remain implementation-in-
progress until their unchanged evaluation gates pass.

## Experiment Status

### Table 1: Rendering Quality

```bash
bash scripts/run_ae.sh quality
```

Target state: mandatory Results Reproduced evidence for all nine pairs. Current
state remains implementation-in-progress while the published FSDR/SAES
semantics and full data matrix are restored. Existing failed sparse-SAES and
LSH pilots remain diagnostic evidence; they are not overwritten or promoted.
Each
sample result covers every target view selected by the configured sampler and
records per-view metrics. The sample metric is their arithmetic mean. Signed
change, degradation, and absolute change are separate fields. The maximum
degradation statement in the paper must not be interpreted as a maximum
absolute deviation.

The optional `--saes-materialization dense-diagnostic` path retains every
Gaussian and is rejected by claim/Functional runs. Its closer image quality is
not accepted as evidence for sparse Gaussian pruning.

### Figure 8: End-to-End Speedup

Target state: mandatory key result. Current state:
`CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION`. This records the requested
result without claiming a completed measurement. The command must run on a real
Jetson Orin NX; an RTX run, calculated baseline, or manuscript constant is
rejected.

```bash
bash scripts/run_ae.sh performance --device orin --output-root outputs/ae
```

Simulator cycles are deterministic. The GPU baseline and SCARF-Dataflow paths
use CUDA events on the evaluator's documented Orin state. Every pair also
archives Nsight Systems, thermal, clock, power-mode, environment, and selection
evidence. The AE path never estimates Orin latency from another GPU.

The MICRO reviewer guidance says evaluators bid using the declared hardware and
software dependencies. If no assigned evaluator owns an Orin NX, the AE FAQ
allows chairs to broker exceptional remote-machine access for rare hardware.
A script alone is Functional evidence; Figure 8 passes only after an independent
evaluator obtains the result.

Figure 8 uses the 1 GHz architectural clock target in the paper. When executed,
the public ASAP7 flow reports achieved timing independently. A routed ASAP7
result that misses 1 GHz remains a visible timing failure and does not validate
the unavailable commercial TSMC28 implementation.

### Figure 10: Worst-Case Error Analysis (Supporting)

```bash
bash scripts/run_ae.sh worstcase --output-root outputs/ae
```

The generator ranks every target view from the completed nine-pair quality
matrix. It selects the worst FSDR-only PSNR loss and SAES-only LPIPS loss using
a deterministic tie-break, then binds the loss trace, stage cycles, source
images, and RGB error map to that same sample/view. Existing plotting metadata
and transformed display values are comparison material only.

### Figure 11 and Tables 2-3: Ablation and Mechanisms

```bash
bash scripts/run_ae.sh mechanisms --output-root outputs/ae
```

For a bounded reviewer shard or a legally available dataset subset, retain the
same mode and canonical protocol but select explicit pairs:

```bash
bash scripts/run_ae.sh mechanisms \
  --pairs transplat/re10k,mvsplat/acid,depthsplat/re10k \
  --num-samples 1 --output-root outputs/ae-pilot
```

`--pairs` never changes an evaluation index or target view. Unknown and
duplicate pairs fail before execution, and omitting it preserves the complete
nine-pair matrix.

The output includes FSDR-only, SAES-only, combined, and no-optimization cycles.
Table 2 uses discrete full-search Top-1 counts, not `in_window_rate`, and derives
depth evaluations and feature-buffer bytes from events. Table 3 derives tile
paths, low-variance agreement, Gaussian savings, and S2 evaluations from the
same event stream. Historical failing pilots remain visible until new complete
aggregates pass the unchanged gates.

### Figure 12: MMCU Utilization (Supporting)

```bash
bash scripts/run_ae.sh utilization --output-root outputs/ae
```

Utilization is `useful_mmcu_slots / scheduled_mmcu_slots` for S1-S3. It cannot
be supplied as a percentage constant or inferred from the manuscript CSV.

### Figures 13-16: Sensitivity (Supporting)

Target state: mandatory full nine-pair trace-and-replay evidence. A model runs
once per sample trace; the five parameter values replay deterministic feature,
depth, Gaussian, cycle, and mechanism inputs.

```bash
bash scripts/run_ae.sh sensitivity --output-root outputs/ae
```

The configured grids include cache sizes 8-128, Hamming thresholds 1-5, the
feature and depth thresholds from the paper, and tile sizes 2-32. Every grid point runs
all protocol samples and produces a strict dataset aggregate before plotting.
These figures are mandatory supporting outputs, not standalone key results.
One-scene route mixes are diagnostics only and are never compared with the
dataset-level Table 3 or Figure 11 aggregate.

### Complete Declared Workflow

```bash
bash scripts/run_ae.sh all-eval --profile full --output-root outputs/ae
bash scripts/run_ae.sh figures --figures all --output-root outputs/ae
bash scripts/run_ae.sh validate --require-key-results --output-root outputs/ae
```

`all-eval` schedules the complete paper matrix except workflows blocked by a
declared external hardware/account gate. `validate --require-key-results`
returns nonzero for every missing, structurally invalid, out-of-tolerance,
wrong-class, or unfinalized key result. It cannot pass merely because a result
is marked not claimed.

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
container, command, and collateral-hash dry-run passed. A clean unchanged
low-memory attempt completed synthesis through filler and reached global
routing, then was stopped after verified persistent swap thrashing. Its hashed
attempt outcome is diagnostic evidence only; no routed PPA is included or
claimed. The default command below is for a sufficiently provisioned host;
the explicitly labeled low-memory attempt is documented in `hardware/README.md`.

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
|-- performance/<model>_<dataset>/samples/sample_NNNNN/orin-evidence/
|-- performance/<model>_<dataset>/results.json
|-- mechanisms/<model>_<dataset>/results.json
|-- utilization/<model>_<dataset>/results.json
|-- worstcase/results.json
|-- sensitivity/results.json
|-- rtl/results.json
|-- dram/results.json
|-- physical/asap7/ppa.json
|-- physical/asap7/ppa_28nm_estimated.json
|-- reports/figure_catalog.json
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
