# SCARF MICRO 2026 Artifact Summary

## Recommendation

Prepare and publish the current package for Artifacts Available and Artifacts
Evaluated (Functional). Do not request Results Reproduced from the current
evidence. Reopen that badge only if authentic SAES feature normalization and
materialization semantics, the FSDR projection matrix, or independently valid
paper-result evidence becomes available.

## Claim Ledger

| Claim | State | Evidence and boundary |
|---|---|---|
| Public source and evidence bundles | Supported, pending DOI | Both bundles pass hash, path, manifest, source-identity, and clean-room checks. Zenodo publication still requires author-account authentication. |
| Strict Functional inference | Supported | The extracted source downloads hash-pinned public assets and completes CUDA quick without fallback. The synthetic fixture is explicitly ineligible for paper results. |
| RTL functionality | Supported | Eight Chisel tests, SystemVerilog emission, Verilator lint, and a representative VCD pass. This does not recover the missing FSDR projection collateral. |
| Public DRAM workflow | Supported proxy | Ramulator 2.1 and DRAMPower 6.0.2 produce trace-to-energy evidence for LPDDR5. It is not the paper's LPDDR4X result. |
| Table 1 sparse-SAES quality | Unsupported | Real canonical Re10K/ACID pilots violate the fixed PSNR, SSIM, or LPIPS tolerances. The Gaussian-head audit also fails. |
| Tables 2-3 mechanisms | Unsupported | Six 32-sample FSDR rows fail; the tracked RTL projection ROM is zero. Real SAES probes produce a degenerate L1 path and miss quality tolerances. |
| Figure 8 Orin NX timing | Deferred | No real Orin NX logs exist. Workstation timing is diagnostic only. |
| DL3DV results | Deferred | The gated dataset was not available and is not redistributed. |
| ASAP7 routed PPA | Deferred by resources | The flow requires no Vivado process and at least 48 GiB `MemAvailable`; the current host remains below the memory gate. |
| 7-to-28 DeepScale estimate | Deferred | The deterministic tool and tests pass, but no estimate is emitted without valid routed ASAP7 input. It is never described as TSMC28 post-layout measurement. |

## Strongest Negative Evidence

The clean Gaussian-head diagnostic at commit `5cee926` captured the authentic
`[1,2,163,256,256]` head input for canonical TranSplat/Re10K sample 0. It
produced L0/L1/Full rates of 29.2%/0.0%/70.8%. SAES-only deltas were -0.8090 dB
PSNR, -0.027802 SSIM, and +0.066505 LPIPS. Even a 4.2969% target-free retained
tile slice failed LPIPS at +0.019034. The complete record is retained under
`outputs/ae_failures/diagnostics/` and is never staged as claimed evidence.

## Verified Package State

- Repository tests: 262 passed.
- Extracted source tests: 261 passed, 1 skipped because Git metadata is absent.
- Formal artifact validation: 15/15 checks passed, including explicit
  `NOT_CLAIMED` states for unsupported paper results.
- Extracted source CUDA quick: passed with release-manifest provenance.
- Extracted evidence validation: 15/15 checks passed.
- The authoritative pre-release hashes are stored in the latest external
  `SCARF-AE-release/v1.0.0-<commit>/SHA256SUMS` release directory.

## Resume Packet

Read `artifact/CLAIMS.md`, `PLAN.md`, and `CHECKLIST.md` first. Do not rerun
threshold, seed, feature-source, retention, or projection searches against the
paper targets. Do not edit generated JSON or relax tolerances. The only useful
reopen conditions are new authentic mechanism collateral, access to a real
Orin NX, at least 48 GiB available memory for iFlow, or Zenodo/HotCRP account
authentication for publication.
