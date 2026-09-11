#!/usr/bin/env python3
"""Create the reviewer-GPU input contract for the Orin NX proxy.

The generated file is a format template only.  It contains no measurements and
is intentionally rejected by ``normalize_orin_proxy.py`` until the reviewer
fills the device, workload, and nine pair records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("transplat", "mvsplat", "depthsplat")
DATASETS = ("re10k", "acid", "dl3dv")


def _breakdown() -> dict[str, object]:
    return {
        "total_ms": 0.0,
        "compute_ms": 0.0,
        "memory_ms": 0.0,
        "overhead_ms": 0.0,
        "event_samples_ms": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def _profile_evidence() -> dict[str, object]:
    return {
        "critical_path_ms": 0.0,
        "nsys_report_sha256": "<fill-nsys-report-sha256>",
        "ncu_report_sha256": "<fill-ncu-report-sha256>",
    }


def template() -> dict[str, object]:
    pairs = {
        f"{model}/{dataset}": {
            "baseline": _breakdown(),
            "scarf_dataflow": _breakdown(),
            "profile_evidence": {
                "method": "nsys_ncu_roofline_critical_path_v1",
                "baseline": _profile_evidence(),
                "scarf_dataflow": _profile_evidence(),
            },
            "asic_cycles": 0,
            "quality_metrics": {
                "source": "run_quality_workflow_separately",
                "note": "Quality is workload/model invariant; do not scale it from GPU peaks.",
            },
            "mechanism_metrics": {
                "source": "source_bound_rtl_trace",
                "note": "Mechanism counters/cycles come from the RTL trace; do not infer them from GPU time.",
            },
        }
        for model in MODELS
        for dataset in DATASETS
    }
    return {
        "schema_version": "scarf-gpu-proxy-input-v1",
        "status": "format_only",
        "claim_eligible": False,
        "same_workload": True,
        "capture": {
            "timing_source": "cuda_events",
            "aggregation": "median",
            "repetitions": 5,
            "warmup_runs": 10,
            "synchronization": "cuda.synchronize_before_and_after",
            "profiler_method": "nsys_ncu_roofline_critical_path_v1",
            "batch_size": 1,
            "precision": "fp32",
            "power_mode": "fill-device-mode",
        },
        "source_device": {
            "device_model": "<fill-device-model>",
            "architecture": "<fill-compute-capability>",
            "fp32_tflops": 0.0,
            "memory_bandwidth_gbps": 0.0,
            "clock_mhz": 0.0,
            "spec_provenance": {
                "source": "<fill-vendor-datasheet-or-device-query>",
                "retrieved_at": "<fill-iso-8601-date>",
                "fp32_basis": "peak FP32 throughput at the recorded clock",
                "bandwidth_basis": "published peak device memory bandwidth",
            },
        },
        "workload": {
            "source_tree_sha256": "<fill-source-tree-sha256>",
            "dataset_tree_sha256": "<fill-dataset-tree-sha256>",
            "selection_sha256": "<fill-selection-sha256>",
            "checkpoint_sha256": "<fill-checkpoint-sha256>",
        },
        "uncertainty_pct": 15.0,
        "pairs": pairs,
    }


def init(output: Path) -> Path:
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="path for reviewer-gpu-proxy.json")
    args = parser.parse_args(argv)
    try:
        print(init(args.output))
    except (OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
