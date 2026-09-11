#!/usr/bin/env python3
"""Normalize a same-workload GPU timing record to an Orin NX proxy.

This is a portability aid for reviewers without Jetson hardware.  It uses a
transparent compute/memory/overhead decomposition and declared peak hardware
specifications.  The output is explicitly non-claim evidence and cannot be
consumed by ``validate_ae.py`` as independent Orin measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_SPEC = ROOT / "hardware/orin/proxy_spec.json"
MODELS = ("transplat", "mvsplat", "depthsplat")
DATASETS = ("re10k", "acid", "dl3dv")
PAIRS = tuple(f"{model}/{dataset}" for model in MODELS for dataset in DATASETS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _positive(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be a nonnegative finite number")
    return float(value)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a populated string")
    normalized = value.strip()
    if not normalized or normalized.startswith("<fill-"):
        raise ValueError(f"{label} must be a populated string")
    return normalized


def _validate_spec(spec: Mapping[str, Any], label: str) -> dict[str, float | str]:
    if spec.get("schema_version") != "scarf-orin-proxy-spec-v1":
        raise ValueError(f"{label} schema_version is invalid")
    if spec.get("claim_eligible") is not False:
        raise ValueError(f"{label} must be marked claim_eligible=false")
    normalized: dict[str, float | str] = {
        "device_model": _required_text(spec.get("device_model"), f"{label}.device_model"),
        "fp32_tflops": _positive(spec.get("fp32_tflops"), f"{label}.fp32_tflops"),
        "memory_bandwidth_gbps": _positive(
            spec.get("memory_bandwidth_gbps"), f"{label}.memory_bandwidth_gbps"
        ),
        "clock_mhz": _positive(spec.get("clock_mhz"), f"{label}.clock_mhz"),
    }
    for field in ("architecture", "source", "method", "clock_basis"):
        value = spec.get(field)
        if value is not None:
            normalized[field] = _required_text(value, f"{label}.{field}")
    if "asic_reference_clock_mhz" in spec:
        normalized["asic_reference_clock_mhz"] = _positive(
            spec.get("asic_reference_clock_mhz"), f"{label}.asic_reference_clock_mhz"
        )
    return normalized


def _validate_event_samples(
    value: Any, expected_median: float, label: str
) -> list[float]:
    """Validate the five raw CUDA-event samples required by the protocol."""
    if (
        not isinstance(value, list)
        or len(value) != 5
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0
            for item in value
        )
    ):
        raise ValueError(f"{label} must contain five positive finite timings")
    samples = [float(item) for item in value]
    if not math.isclose(
        float(statistics.median(samples)), expected_median, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError(f"{label} median does not match total_ms")
    return samples


def _validate_breakdown(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    total = _positive(value.get("total_ms"), f"{label}.total_ms")
    compute = _nonnegative(value.get("compute_ms"), f"{label}.compute_ms")
    memory = _nonnegative(value.get("memory_ms"), f"{label}.memory_ms")
    overhead = _nonnegative(value.get("overhead_ms"), f"{label}.overhead_ms")
    if not math.isclose(total, compute + memory + overhead, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"{label} components do not sum to total_ms")
    event_samples = _validate_event_samples(value.get("event_samples_ms"), total, f"{label}.event_samples_ms")
    return {
        "total_ms": total,
        "compute_ms": compute,
        "memory_ms": memory,
        "overhead_ms": overhead,
        "event_samples_ms": event_samples,
    }


def _validate_profile_evidence(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if value.get("method") != "nsys_ncu_roofline_critical_path_v1":
        raise ValueError(f"{label}.method is invalid")
    normalized: dict[str, Any] = {"method": value["method"]}
    for variant in ("baseline", "scarf_dataflow"):
        item = value.get(variant)
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}.{variant} must be an object")
        normalized[variant] = {
            "critical_path_ms": _positive(
                item.get("critical_path_ms"), f"{label}.{variant}.critical_path_ms"
            ),
            "nsys_report_sha256": _sha256(
                item.get("nsys_report_sha256"), f"{label}.{variant}.nsys_report_sha256"
            ),
            "ncu_report_sha256": _sha256(
                item.get("ncu_report_sha256"), f"{label}.{variant}.ncu_report_sha256"
            ),
        }
    return normalized


def _validate_spec_provenance(value: Any, label: str) -> dict[str, str]:
    """Retain the public source of the review GPU peak specifications."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    normalized: dict[str, str] = {}
    for field in ("source", "retrieved_at", "fp32_basis", "bandwidth_basis"):
        normalized[field] = _required_text(value.get(field), f"{label}.{field}")
    return normalized


def normalize_pair(
    pair: str,
    timings: Mapping[str, Any],
    *,
    source_spec: Mapping[str, float | str],
    target_spec: Mapping[str, float | str],
    uncertainty_pct: float,
) -> dict[str, Any]:
    if not isinstance(timings, Mapping):
        raise ValueError(f"{pair} timing record must be an object")
    baseline = _validate_breakdown(timings.get("baseline"), f"{pair}.baseline")
    scarf = _validate_breakdown(
        timings.get("scarf_dataflow"), f"{pair}.scarf_dataflow"
    )
    profile_evidence = _validate_profile_evidence(
        timings.get("profile_evidence"), f"{pair}.profile_evidence"
    )
    compute_scale = float(source_spec["fp32_tflops"]) / float(target_spec["fp32_tflops"])
    memory_scale = float(source_spec["memory_bandwidth_gbps"]) / float(
        target_spec["memory_bandwidth_gbps"]
    )

    def convert(breakdown: Mapping[str, Any]) -> dict[str, Any]:
        compute_ms = float(breakdown["compute_ms"]) * compute_scale
        memory_ms = float(breakdown["memory_ms"]) * memory_scale
        overhead_ms = float(breakdown["overhead_ms"])
        total_ms = compute_ms + memory_ms + overhead_ms
        converted: dict[str, Any] = {
            "compute_ms": compute_ms,
            "memory_ms": memory_ms,
            "overhead_ms": overhead_ms,
            "total_ms": total_ms,
            "throughput_fps": 1000.0 / total_ms,
            "interval_ms": {
                "lower": total_ms * (1.0 - uncertainty_pct / 100.0),
                "upper": total_ms * (1.0 + uncertainty_pct / 100.0),
            },
            "interval_interpretation": (
                "component-scaling sensitivity band; not a statistical confidence interval"
            ),
        }
        converted["interval_fps"] = {
            "lower": 1000.0 / converted["interval_ms"]["upper"],
            "upper": 1000.0 / converted["interval_ms"]["lower"],
        }
        total_scale = total_ms / float(breakdown["total_ms"])
        converted["event_samples_ms"] = [
            float(sample) * total_scale for sample in breakdown["event_samples_ms"]
        ]
        return converted

    converted_baseline = convert(baseline)
    converted_scarf = convert(scarf)
    result: dict[str, Any] = {
        "source_gpu": {
            "baseline_ms": baseline["total_ms"],
            "scarf_dataflow_ms": scarf["total_ms"],
        },
        "orin_nx_proxy": {
            "baseline": converted_baseline,
            "scarf_dataflow": converted_scarf,
            "dataflow_speedup": converted_baseline["total_ms"]
            / converted_scarf["total_ms"],
        },
        "figure8_indicators": {
            "orin_nx_normalized": 1.0,
            "scarf_dataflow_on_orin_nx_normalized": converted_baseline["total_ms"]
            / converted_scarf["total_ms"],
            "scarf_asic_speedup": None,
        },
        "scaling": {
            "compute_time_multiplier": compute_scale,
            "memory_time_multiplier": memory_scale,
            "overhead_policy": "unchanged",
            "uncertainty_pct": uncertainty_pct,
        },
        "profile_evidence": profile_evidence,
    }
    asic_cycles = timings.get("asic_cycles")
    if asic_cycles is not None:
        cycles = _positive_integer(asic_cycles, f"{pair}.asic_cycles")
        asic_clock_mhz = float(
            target_spec.get("asic_reference_clock_mhz", target_spec["clock_mhz"])
        )
        asic_ms = cycles / (asic_clock_mhz * 1000.0)
        result["orin_nx_proxy"]["asic"] = {
            "cycles": cycles,
            "clock_mhz": asic_clock_mhz,
            "clock_basis": "paper_asic_reference_clock",
            "latency_ms": asic_ms,
            "speedup": converted_baseline["total_ms"] / asic_ms,
        }
        result["figure8_indicators"]["scarf_asic_speedup"] = (
            converted_baseline["total_ms"] / asic_ms
        )
    else:
        result["orin_nx_proxy"]["asic"] = None
    return result


def normalize(
    input_path: Path,
    output_path: Path,
    *,
    target_spec_path: Path = DEFAULT_TARGET_SPEC,
) -> dict[str, Any]:
    source = _load(input_path, "GPU proxy input")
    if source.get("schema_version") != "scarf-gpu-proxy-input-v1":
        raise ValueError("GPU proxy input schema_version is invalid")
    if source.get("claim_eligible") is not False:
        raise ValueError("GPU proxy input must be marked claim_eligible=false")
    if source.get("same_workload") is not True:
        raise ValueError("GPU proxy input must declare same_workload=true")
    capture = source.get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError("capture must be an object")
    if capture.get("timing_source") != "cuda_events":
        raise ValueError("capture.timing_source must be cuda_events")
    if capture.get("repetitions") != 5:
        raise ValueError("capture.repetitions must be five")
    if capture.get("warmup_runs") != 10:
        raise ValueError("capture.warmup_runs must be ten")
    if capture.get("aggregation") != "median":
        raise ValueError("capture.aggregation must be median")
    if capture.get("synchronization") != "cuda.synchronize_before_and_after":
        raise ValueError("capture.synchronization is invalid")
    if capture.get("profiler_method") != "nsys_ncu_roofline_critical_path_v1":
        raise ValueError("capture.profiler_method is invalid")
    if capture.get("batch_size") != 1:
        raise ValueError("capture.batch_size must be one")
    if capture.get("precision") != "fp32":
        raise ValueError(
            "capture.precision must be fp32 because this proxy uses FP32 peaks"
        )
    source_device = source.get("source_device")
    target = _validate_spec(_load(target_spec_path, "Orin proxy specification"), "target spec")
    if not isinstance(source_device, Mapping):
        raise ValueError("source_device is missing")
    source_spec = {
        "device_model": _required_text(
            source_device.get("device_model"), "source_device.device_model"
        ),
        "fp32_tflops": _positive(
            source_device.get("fp32_tflops"), "source_device.fp32_tflops"
        ),
        "memory_bandwidth_gbps": _positive(
            source_device.get("memory_bandwidth_gbps"),
            "source_device.memory_bandwidth_gbps",
        ),
        "clock_mhz": _positive(
            source_device.get("clock_mhz"), "source_device.clock_mhz"
        ),
    }
    for field in ("architecture", "driver", "clock_basis"):
        value = source_device.get(field)
        if value is not None:
            source_spec[field] = _required_text(value, f"source_device.{field}")
    if "spec_provenance" in source_device:
        source_spec["spec_provenance"] = _validate_spec_provenance(
            source_device["spec_provenance"], "source_device.spec_provenance"
        )
    uncertainty_pct = _positive(source.get("uncertainty_pct", 15.0), "uncertainty_pct")
    if uncertainty_pct >= 100:
        raise ValueError("uncertainty_pct must be below 100")
    timings = source.get("pairs")
    if not isinstance(timings, Mapping) or set(timings) != set(PAIRS):
        raise ValueError("GPU proxy input must contain exactly the nine model/dataset pairs")
    workload = source.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("workload provenance is missing")
    for field in ("source_tree_sha256", "dataset_tree_sha256", "selection_sha256", "checkpoint_sha256"):
        _sha256(workload.get(field), f"workload.{field}")
    pair_results = {
        pair: normalize_pair(
            pair,
            timings[pair],
            source_spec=source_spec,
            target_spec=target,
            uncertainty_pct=uncertainty_pct,
        )
        for pair in PAIRS
    }
    output = {
        "schema_version": "scarf-orin-proxy-result-v1",
        "kind": "orin_nx_hardware_proxy",
        "status": "PROXY_ONLY",
        "evidence_class": "hardware_proxy",
        "claim_eligible": False,
        "independent_orin_measurement": False,
        "method": "component_roofline_scaling",
        "indicator_policy": {
            "quality": {
                "mode": "hardware_invariant",
                "procedure": "run the quality workflow on the same checkpoint and sample selection",
            },
            "mechanisms": {
                "mode": "source_bound_cycle_record",
                "procedure": "use combined_stage_cycles and counters from the RTL timing export",
            },
            "performance": {
                "mode": "hardware_proxy",
                "procedure": "scale measured GPU timing components with the declared Orin specification",
            },
        },
        "metric_mapping": {
            "latency_ms": {
                "status": "ESTIMATED",
                "method": "critical-path component scaling",
                "formula": "compute*(P_source/P_orin)+memory*(B_source/B_orin)+overhead",
                "scope": "nine same-workload model/dataset pairs",
            },
            "throughput_fps": {
                "status": "ESTIMATED",
                "method": "1000 / estimated latency_ms",
                "scope": "nine same-workload model/dataset pairs",
            },
            "normalized_speedup": {
                "status": "ESTIMATED",
                "method": "estimated baseline latency / estimated SCARF latency",
                "scope": "trend check only; not an independent Orin measurement",
            },
            "quality_psnr_ssim_lpips": {
                "status": "DIRECT_REEXECUTION_REQUIRED",
                "method": "run the quality workflow with the frozen configuration",
                "scope": "never scaled from GPU specifications",
            },
            "mechanism_counters_and_cycles": {
                "status": "SOURCE_BOUND_RECORD_REQUIRED",
                "method": "validated source-bound RTL timing bundle",
                "scope": "never inferred from GPU timings",
            },
            "power_temperature_occupancy": {
                "status": "UNAVAILABLE_WITHOUT_ORIN",
                "method": "tegrastats/Nsight on the target board",
                "scope": "no proxy value is emitted",
            },
        },
        "source": {
            "input_path": input_path.name,
            "input_sha256": sha256_file(input_path),
            "device": source_spec,
            "same_workload": True,
            "capture": dict(capture),
        },
        "target": {
            **target,
            "spec_path": str(target_spec_path),
            "spec_sha256": sha256_file(target_spec_path),
        },
        "workload": dict(workload),
        "uncertainty_pct": uncertainty_pct,
        "formula": {
            "latency_ms": "compute_ms * (source_fp32_tflops / orin_fp32_tflops) + memory_ms * (source_bandwidth_gbps / orin_bandwidth_gbps) + overhead_ms",
            "throughput_fps": "1000 / latency_ms",
            "asic_latency_ms": "asic_cycles / (asic_reference_clock_mhz * 1000)",
            "normalization": "baseline_latency_ms / variant_latency_ms",
        },
        "assumptions": {
            "compute": "scales inversely with declared FP32 peak throughput",
            "memory": "scales inversely with declared DRAM bandwidth",
            "overhead": "carried unchanged because it is not portable from peak specifications",
            "concurrency": "input components represent one non-overlapping critical path",
            "interval": "uncertainty_pct is a sensitivity band, not a confidence interval",
        },
        "unsupported_metrics": {
            "quality": {
                "status": "RUN_DIRECTLY",
                "reason": "image quality is workload/model output dependent and is not scaled",
            },
            "mechanism_counters": {
                "status": "SOURCE_BOUND_ONLY",
                "reason": "FSDR/SAES counters and stage cycles require the source RTL trace",
            },
            "power_w": {
                "status": "UNAVAILABLE",
                "reason": "GPU peak specifications do not determine Orin power",
            },
            "temperature_c": {
                "status": "UNAVAILABLE",
                "reason": "thermal behavior requires tegrastats on the target board",
            },
            "memory_occupancy": {
                "status": "UNAVAILABLE",
                "reason": "occupancy and contention are not inferred by roofline scaling",
            },
        },
        "pairs": pair_results,
        "limitations": [
            "This is not a Jetson Orin NX CUDA-event measurement.",
            "It does not provide tegrastats, Nsight Systems, thermal, or power evidence.",
            "Peak-spec scaling does not model kernel occupancy, compiler, clocks, or contention.",
            "Run hardware/orin/run.py on Orin NX before claiming Figure 8 or Results Reproduced.",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="filled reviewer-GPU JSON contract"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="proxy report JSON path"
    )
    parser.add_argument(
        "--target-spec",
        type=Path,
        default=DEFAULT_TARGET_SPEC,
        help="declared Orin NX specification JSON",
    )
    args = parser.parse_args(argv)
    try:
        result = normalize(args.input.resolve(), args.output.resolve(), target_spec_path=args.target_spec.resolve())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(
        f"PROXY_ONLY: normalized {len(result['pairs'])} pairs; "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
