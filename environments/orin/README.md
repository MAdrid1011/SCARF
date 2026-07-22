# Jetson Orin NX Profile

SCARF does not install or replace JetPack packages. Run the following on the
Jetson Orin NX used for the paper-facing baseline:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
bash install.sh --profile orin --check-only
```

The check records the installed JetPack package state, L4T release, CUDA and
PyTorch versions, GPU clocks, power mode, and device model. This runtime record
is the version lock for the archived Orin run. It is generated automatically.
The measurement wrapper also archives five raw CUDA-event values, Nsight
Systems output, and every thermal sample from `tegrastats`. It rejects a run
above the 80 C contract limit.
