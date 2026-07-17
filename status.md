# Artifact Status

- Branch: `ae/micro2026-three-badges`
- State: three-badge implementation in progress
- Artifacts Available: pending Zenodo DOI and publication
- Artifacts Evaluated (Functional): supported by clean-room evidence
- Results Reproduced: requested; mandatory key-result evidence is incomplete
- Evaluation catalog: all 9 figures and 4 tables have commands, evidence classes,
  raw-input patterns, generators/status outputs, and fixed acceptance rules
- Strict key-result validation: incomplete; Orin, DL3DV quality/sensitivity,
  full Re10K/ACID quality and sensitivity, workload energy, and routed physical
  evidence remain absent
- Repository tests: 294 PASS / 15 SKIP in the CPU schema environment; targeted
  SAES formula tests also pass in the locked classic environment
- RTL evidence: PASS with 11 Chisel tests, SystemVerilog, Verilator lint, VCD,
  and seed-42 FP16/Q1.24 LSH software/RTL signature equivalence
- Physical flow: `NOT_CLAIMED_RESOURCE_LIMIT`
- Table 4 contract: public PPA hierarchy is locked to the final paper's 25
  area/power rows, their parent/child structure, and the corresponding RTL
  instance names; tool-private buckets are rejected.
- Orin NX: `NOT_CLAIMED_NO_ORIN_EVIDENCE`
- DL3DV: official gated data prepared and re-verified for all 140 protocol
  scenes; results remain `NOT_CLAIMED_SAES_SPARSE_QUALITY_MISMATCH` and
  `NOT_CLAIMED_FSDR_LSH_AND_SAES_PROTOCOL_MISMATCH`
- Latest strict quick: schema-v2.1 PASS with exact FSDR local-validity and
  SAES probe-only counts, executed S1-S3 MMCU slots, worst-view image/hash
  retention, and source-tree provenance. Its synthetic fixture is explicitly
  non-claiming.
- Six-pair real pilot: structurally valid but paper-result FAIL. Guided Rate is
  86.51%--94.96%, every L1 rate is zero, and combined PSNR changes by -0.5624
  to -5.0876 dB; no full protocol was launched from this failed gate
- Isolated diagnostics: the 2K(T) L1 path and the paper's explicit bilateral
  assignment formula both fail the unchanged SAES quality gates; the historical
  FSDR depth guard recovers Top-1/quality but not Guided Rate and is absent from
  the paper RTL, so all remain non-claiming
- Physical resource check: the audited low-memory attempt reached global routing
  but was stopped for sustained swap thrashing; it is
  `NOT_CLAIMED_RESOURCE_LIMIT` and cannot be used before complete reports
  exist. The standard flow remains gated on stable 48 GiB `MemAvailable`.
- Next action: continue the official train-split download, compile the
  evaluation-disjoint calibration protocol, freeze one global configuration,
  and rerun mechanism gates before any full matrix. Do not publish or mark
  HotCRP ready until mandatory key results pass.
- Active closure contract: retain the paper's routing mechanisms while adding
  only disclosed engineering details. A disjoint 32-Re10K/32-ACID training
  calibration chooses one global FSDR validity tolerance and SAES bandwidth
  tuple; evaluation targets and manuscript results are inaccessible. A compiled
  target-free sidecar carries only context RGB plus camera geometry, and frozen
  reviewer/full profiles remain blocked on the calibration and pilot gates.
- Calibration preparation: the compiler now consumes only the official prepared
  training subsets and verifies the full-index hash, fixed SHA256-ranked 32
  scenes, selected-index hash, evaluation-disjoint provenance, and target-free
  sidecar tree before any trace is run. The official archive download is still
  incomplete.
