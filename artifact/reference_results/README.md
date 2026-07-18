# Archived Reference Evidence

The final release stores compact, public evidence from the clean-room run in
this directory. Run `scripts/stage_reference_results.py` only after the full AE
validator reports `PASS` with `--require-key-results`. Staging also verifies
that each execution result matches the current source tree, submodule revisions,
mechanism configuration, and calibration provenance; stale Functional evidence
is intentionally rejected rather than rewritten.

The staging command copies structured results, Orin measurement records and raw
logs, RTL and DRAM evidence, public physical reports, scaling output, and report
figures. Full datasets and large routed databases remain external. Their hashes
and regeneration commands stay in the structured records.
