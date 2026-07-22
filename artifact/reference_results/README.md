# Reference Results

`orin_nx_reference.csv` is the normalized nine-pair Figure 8 reference table
from the submission. It retains the source identifier and SHA256 so an
evaluator without a Jetson Orin NX can compare a locally generated report with
the archived reference values.

The columns separate the original Orin NX baseline, SCARF Dataflow on Orin NX,
and the SCARF ASIC speedup. The table is a comparison reference, not a device
measurement record. An evaluator with an Orin NX should run
`bash scripts/run_ae.sh performance --device orin`; that workflow emits the
CUDA-event, Nsight, thermal, and device records used by the validator.

Hash-verified execution evidence is staged separately by
`scripts/stage_reference_results.py` after the full validator passes. Full
datasets and large routed databases remain external; their identities and
regeneration commands are carried by the structured result records.
