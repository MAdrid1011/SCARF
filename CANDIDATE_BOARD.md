# Local DL3DV Candidate Board

| Candidate ID | Level | Parent | Strategy | Status | Expected Gain | Observed Result | Promote / Archive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V15 | implementation | V13 L1-15 | exploit | succeeded | Reduce high S1-LOO-risk merges | 34.2030 dB; 0.6361 dB below Full | incumbent |
| V16 | implementation | V15 | exploit | proposed | Reject V4-unreplayable L1 merges without GT | pending | run target-free calibration then one audit |
| V17 | brief | V15 | orthogonal | deferred | Correct virtual SH/opacity jackknife bias | pending | only consider after V16 is measured |

## V16 Brief

- Bottleneck: V15 still accepts 2,226 L1 tiles whose selected-anchor V4
  approximation may be locally irreproducible.
- Mechanism: hide each of the three retained center anchors in turn, rebuild it
  from the other 14 selected anchors, and promote the tile to Full when the
  frozen target-free replay risk exceeds its threshold.
- Keep unchanged: all V15 route inputs and threshold, actual omitted-center
  opacity/SH/geometry isolation, V4 materialization, and Full passthrough.
- Risks: 14-to-15 anchor distribution shift or a certificate that becomes too
  conservative. The former is recorded; the latter is an archive condition.
