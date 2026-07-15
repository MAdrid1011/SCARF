# Archived Reference Evidence

The final release stores compact, public evidence from the clean-room run in
this directory. Run `scripts/stage_reference_results.py` only after the full AE
validator reports `PASS`.

The staging command copies structured results, Orin measurement records and raw
logs, RTL and DRAM evidence, public physical reports, scaling output, and report
figures. Full datasets and large routed databases remain external. Their hashes
and regeneration commands stay in the structured records.
