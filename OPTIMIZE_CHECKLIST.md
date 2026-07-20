# V16 Optimization Checklist

- [x] Refresh the local DL3DV quality frontier from durable V13--V15 results.
- [x] Select one primary mode: `loop` / exploit the V15 line.
- [x] Confirm the route: target-free, selected-anchor-only, Full fail-closed.
- [x] Compare the two bounded candidates: V4 center-LOO certificate and V4
  jackknife attribute correction.
- [x] Promote only the certificate first; defer correction so its effect stays
  attributable.
- [x] Record the V16 candidate and fixed smoke/full-evaluation queue.
- [ ] Add the certificate, frozen calibration record, and synthetic tests.
- [ ] Run target-free calibration on DL3DV samples 1--4, excluding sample 0.
- [ ] Run one target-free sample-0 audit using the frozen record.
- [ ] Run one fixed sample-0 quality gate only if the audit is valid.
- [ ] Refresh the frontier and archive or promote V16 from measured evidence.

## Current Route

- Incumbent: V15, `34.20299 dB` versus `34.83912 dB` Full, with
  `128846/131072` descriptors.
- Candidate: V16 center-anchor leave-one-out V4 replay certificate.
- Keep unchanged: V15 S1 p50 record, L1-15 anchor selection, V4 merge,
  source geometry, covariance closure, packet consumer, target access order.
- Stop condition: target-free calibration/audit violation, or a fixed quality
  gate that does not improve V15 enough to justify its additional Full tiles.
