# Final AE Submission Checklist

The submission is ready only when every required item is checked.

- [x] `bash scripts/run_ae.sh quick` passes from the extracted source archive.
- [x] All claimed experiments produce schema-valid `results.json` files.
- [x] The ordered sample and view-selection protocol is finalized and hashed.
- [x] `bash scripts/run_ae.sh validate` reports no `FAIL` or `NOT_RUN` claim
  from the shared default `outputs/ae` result tree.
- [x] Chisel tests, SystemVerilog emission, and Verilator lint pass.
- [x] ASAP7 is `NOT_CLAIMED_RESOURCE_LIMIT`; its clean preflight and guard
      failure are preserved without partial PPA.
- [x] DeepScale is `NOT_CLAIMED_NO_PHYSICAL_INPUT`; formula/table tests pass and
      no result is synthesized without raw PPA.
- [x] Prepared Re10K and ACID sources, formats, terms, and hashes are verified.
- [x] DL3DV is marked gated and outside the current claim.
- [x] The LPDDR5 smoke proxy is labeled Functional and not used as a paper result.
- [x] Figure 8 is outside the claim because no real Orin evidence exists.
- [x] No manuscript constant is used as an experimental fallback.
- [x] No absolute author-machine path appears in code or documentation.
- [x] All third-party source and license notices are complete.
- [x] The release archives were tested after unpacking outside the repository.
- [x] Passing clean-room evidence is staged and every archived file hash matches.
- [ ] The Zenodo DOI resolves and archive SHA256 matches the release manifest.
- [ ] The DOI is identical in HotCRP, the appendix, and the archive metadata.
- [ ] HotCRP Submission PDF, Key Results, Hardware, Software, and Data fields
      are complete.
- [x] The HotCRP abstract says post-layout design and does not say fabricated chip.
- [ ] The HotCRP submission is explicitly marked ready before the deadline.
