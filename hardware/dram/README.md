# DRAM Timing and Energy Proxy

This workflow converts measured SCARF address events to a Ramulator 2.1 trace,
runs the pinned public LPDDR5-6400 timing model, and feeds its issued commands to
DRAMPower 6.0.2. It is a functional public memory-system proxy. The paper uses
LPDDR4X, so these numbers are not claimed as reproduced LPDDR4X measurements.

Required external checkouts are pinned in `versions.json`. Build them using
their upstream instructions. This artifact never installs tools at run time.
For DRAMPower v6.0.2, configure with `-DDRAMPOWER_BUILD_CLI=ON` and build the
resulting `build/bin/cli` target. That release fetches DRAMUtils v1.13.4 but its
bundled LPDDR5 fixture predates four required read-to-write timing fields. The
runner verifies the untouched fixture SHA256 and writes a derived memspec with
the missing `RTW_L_32`, `RTW_L_16`, `RTW_S_32`, and `RTW_S_16` fields set to
zero. Both source and derived hashes are retained in `drampower.json`. No
foundry or paper measurement is inferred from this compatibility adaptation.

The input is JSON Lines with one actual address event per line:

```json
{"cycle": 0, "op": "read", "address": "0x80000000", "bytes": 128}
{"cycle": 20, "op": "write", "address": "0x82000000", "bytes": 64}
```

`test_vectors/scarf_smoke_events.jsonl` is a representative functional test
vector for `run_ae.sh dram` and `run_ae.sh all`. It validates the public tool
chain only. It is not a measured paper workload and its energy is not a paper
result. Full workload runs must pass an exported event file with `--events`.
For Figure 9, pass the matching schema-v2 aggregate with `--workload-result`.
Every event must then include its canonical `sample_index`; the exporter rejects
missing or extra sample indices and binds the trace to the aggregate selection
hash and SHA256. Only this path emits per-inference energy. The smoke vector can
never satisfy that contract.

Run:

```bash
bash hardware/dram/run.sh \
  --ramulator-root /path/to/ramulator2-v2.1.0 \
  --drampower-root /path/to/DRAMPower-v6.0.2 \
  --events outputs/software/memory-events.jsonl \
  --workload-result outputs/software/results.json \
  --output-dir outputs/dram
```

Missing events, zero requests, version mismatches, unsupported commands, zero
cycles, or incomplete energy output cause a nonzero exit. The source event
cycles are retained for provenance. The `LoadStoreTrace` frontend in Ramulator queues at most
one request per frontend tick and therefore models memory-side service rather
than the original accelerator issue intervals.
