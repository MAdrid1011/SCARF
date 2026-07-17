# saes_dependency_audit.py

Target-free TranSplat/DL3DV diagnostic for dense S2/S3 dependencies at the
full-resolution refinement boundary.

## External Interface

```bash
python scripts/saes_dependency_audit.py \
  --sample-index 0 --device cuda \
  --output-dir outputs/ae_dl3dv_repair_diagnostics/<new-directory>
```

The output directory must not exist. The script runs two context-only forwards,
captures the raw `to_gaussians` tensor, and terminates each execution at that
hook before S4, rendering, target RGB transfer, or quality metrics.

## Public Helpers

- `build_probe_mask(height, width, tile_size)` returns the four-corner probe
  positions in each full-resolution tile.
- `summarize_probe_dependency(baseline, perturbed, probe_mask)` reports raw
  head changes only at retained probes.
- `collect_transplat_dependency_audit(sample_index, device)` runs the fixed
  DL3DV context-only diagnostic and returns a schema-valid non-claim record.

## Internal Helpers

- `_capture_transplat_raw_head` installs a raw-head sentinel hook and optional
  non-probe refinement-input perturbation, then restores all hooks.
- `_context_on_device` transfers only `batch["context"]`; target RGB remains
  outside execution.
