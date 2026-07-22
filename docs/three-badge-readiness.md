# Three-Badge Readiness

The requested badge path has three separate gates:

1. Artifact Available: build and verify a clean source archive with
   `python scripts/build_archive.py --source-only --output <archive>`.
   This archive deliberately contains no Results-Reproduced evidence.
2. Artifacts Evaluated - Functional: run the deterministic synthetic fixture
   with `bash scripts/run_ae.sh quick`, then retain its schema-valid aggregate.
3. Results Reproduced: run all active Table 1 and Figure 11 pairs under the
   frozen protocol, obtain the independent Jetson Orin NX Figure 8 evidence,
   stage the evidence bundle, and pass
   `bash scripts/run_ae.sh validate --require-key-results`.

Use the readiness reporter after the first two gates or after a full run:

```bash
python scripts/three_badge_readiness.py \
  --source-archive outputs/SCARF-AE-source.tar.gz \
  --output-root outputs/ae \
  --output outputs/ae/three-badge-readiness.json
```

`THREE_BADGE_COMPLETE` is emitted only when the source archive has a DOI, the
clean Functional aggregate matches the current commit, and the strict key-result
validator passes. Any missing Orin, dataset, or aggregate evidence remains
explicitly non-passing.
