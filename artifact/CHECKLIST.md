# Final AE Submission Checklist

The submission is ready only when every required item is checked.

- [ ] `bash scripts/run_ae.sh quick` passes from a clean clone.
- [ ] All claimed experiments produce schema-valid `results.json` files.
- [x] The ordered sample and view-selection protocol is finalized and hashed.
- [ ] `bash scripts/run_ae.sh validate` reports no `FAIL` or `NOT_RUN` claim
  from the shared default `outputs/ae` result tree.
- [x] Chisel tests, SystemVerilog emission, and Verilator lint pass.
- [ ] The ASAP7 run records complete tool/library provenance and raw reports.
- [ ] DeepScaleTool normalization preserves raw values and reports factors.
- [x] Claimed Re10K and ACID sources, formats, terms, and hashes are verified.
- [x] DL3DV is marked gated and outside the current claim.
- [x] The LPDDR5 smoke proxy is labeled Functional and not used as a paper result.
- [x] Figure 8 is outside the claim because no real Orin evidence exists.
- [x] No manuscript constant is used as an experimental fallback.
- [x] No absolute author-machine path appears in code or documentation.
- [x] All third-party source and license notices are complete.
- [ ] The release archive was tested after unpacking outside the repository.
- [ ] Passing clean-room evidence is staged and every archived file hash matches.
- [ ] The Zenodo DOI resolves and archive SHA256 matches the release manifest.
- [ ] The DOI is identical in HotCRP, the appendix, and the archive metadata.
- [ ] HotCRP Submission PDF, Key Results, Hardware, Software, and Data fields
      are complete.
- [ ] The HotCRP abstract says post-layout design and does not say fabricated chip.
- [ ] The HotCRP submission is explicitly marked ready before the deadline.
