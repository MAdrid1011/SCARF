# Evaluation Protocol

- Task: reproduce SCARF quality, latency, ablation, and sensitivity results.
- Datasets: Re10K, ACID, and converted DL3DV test data.
- Models: TranSplat, MVSplat, and DepthSplat with the checkpoint mapping in
  `scripts/ae_config.py`.
- Primary quality metrics: PSNR, SSIM, and LPIPS.
- Performance metric: measured Orin NX encoder latency divided by SCARF
  architectural cycles at the 1 GHz target in the paper.
- Selection rule: deterministic file order over the non-null entries in the
  committed upstream evaluation index. Each sample uses the context and target
  indices in that entry. Per-view metrics and their arithmetic mean are
  recorded. Every sample-level result is hashed into the dataset aggregate.

## Evidence

- Paper methodology: `micro59-submit/Sections/section5.tex`.
- Paper results: `micro59-submit/Sections/section6.tex`.
- Experiment mapping: `artifact/CLAIMS.md` and `scripts/ae_config.py`.
- Machine-readable state: `artifact/evaluation_protocol.json`.

## Recovered Contract

The accepted paper names the datasets but does not enumerate every evaluation
view. TranSplat, MVSplat, and DepthSplat contain byte-identical Re10K and ACID
evaluation indices. DepthSplat also contains the 2-context and 4-target DL3DV
index. The artifact treats these committed upstream files as the recoverable
public protocol.

The source indices contain 7,194 Re10K keys, 1,848 ACID keys, and 140 DL3DV
keys. Re10K has 720 null entries and ACID has 253 null entries. Upstream
evaluation samplers reject null entries, so the executable sample counts are
6,474, 1,595, and 140. Both source counts and executable counts are preserved in
the machine-readable protocol.

TranSplat and MVSplat use their dataset-specific checkpoints for Re10K and
ACID. Their converted DL3DV workflows use the official Re10K checkpoints for
zero-shot evaluation. DepthSplat uses its official large Re10K checkpoint for
both Re10K and zero-shot ACID evaluation. This follows the ACID command in the
DepthSplat repository at commit `1f5e5486f005e5b9975cca5cbed3d61acf465707`.
DepthSplat uses its official DL3DV checkpoint for the native DL3DV workflow.
The mapping is machine readable in `artifact/manifests/checkpoints.json` and
`scripts/ae_config.py`.

Before a dataset result can pass, its record must contain the source index
SHA256, the ordered selection SHA256, the prepared dataset tree SHA256, and the
model-compatible representation. The protocol selection is finalized. Dataset
tree hashes for the claimed Re10K and ACID pairs are finalized. The two DL3DV
representations remain evidence gates because gated data is not available.

Full commands use the executable counts. A one-sample smoke run is Functional
evidence only and cannot reproduce a dataset-level paper table. Every aggregate
stores the ordered scene and view-selection records. The final protocol records
the SHA256 of that canonical list.

## Hardware Interpretation

Figure 8 uses the 1 GHz architectural target in the paper. The public ASAP7 physical
run independently reports achieved post-route timing. Failure to close 1 GHz
must remain visible and may not be hidden by the architectural-model result.
ASAP7 and DeepScale results do not validate the unavailable commercial TSMC28
implementation.
