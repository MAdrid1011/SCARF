from __future__ import annotations

import json
from pathlib import Path

import pytest


def _input() -> dict:
    pairs = {}
    for model in ("transplat", "mvsplat", "depthsplat"):
        for dataset in ("re10k", "acid", "dl3dv"):
            pairs[f"{model}/{dataset}"] = {
                "baseline": {
                    "total_ms": 10.0,
                    "compute_ms": 4.0,
                    "memory_ms": 5.0,
                    "overhead_ms": 1.0,
                    "event_samples_ms": [9.0, 10.0, 10.0, 10.0, 11.0],
                },
                "scarf_dataflow": {
                    "total_ms": 8.0,
                    "compute_ms": 3.0,
                    "memory_ms": 4.0,
                    "overhead_ms": 1.0,
                    "event_samples_ms": [7.0, 8.0, 8.0, 8.0, 9.0],
                },
                "profile_evidence": {
                    "method": "nsys_ncu_roofline_critical_path_v1",
                    "baseline": {
                        "critical_path_ms": 10.0,
                        "nsys_report_sha256": "e" * 64,
                        "ncu_report_sha256": "f" * 64,
                    },
                    "scarf_dataflow": {
                        "critical_path_ms": 8.0,
                        "nsys_report_sha256": "e" * 64,
                        "ncu_report_sha256": "f" * 64,
                    },
                },
                "asic_cycles": 9180,
            }
    return {
        "schema_version": "scarf-gpu-proxy-input-v1",
        "claim_eligible": False,
        "same_workload": True,
        "capture": {
            "timing_source": "cuda_events",
            "repetitions": 5,
            "warmup_runs": 10,
            "aggregation": "median",
            "synchronization": "cuda.synchronize_before_and_after",
            "profiler_method": "nsys_ncu_roofline_critical_path_v1",
            "batch_size": 1,
            "precision": "fp32",
        },
        "source_device": {
            "device_model": "Test GPU",
            "fp32_tflops": 10.0,
            "memory_bandwidth_gbps": 200.0,
            "clock_mhz": 1500,
        },
        "workload": {
            "source_tree_sha256": "a" * 64,
            "dataset_tree_sha256": "b" * 64,
            "selection_sha256": "c" * 64,
            "checkpoint_sha256": "d" * 64,
        },
        "uncertainty_pct": 10.0,
        "pairs": pairs,
    }


def test_normalize_orin_proxy_uses_component_scaling(tmp_path: Path):
    from scripts.normalize_orin_proxy import normalize

    source = tmp_path / "gpu.json"
    source.write_text(json.dumps(_input()), encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": "scarf-orin-proxy-spec-v1",
                "claim_eligible": False,
                "device_model": "Orin NX",
                "fp32_tflops": 5.0,
                "memory_bandwidth_gbps": 100.0,
                "clock_mhz": 1000,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "proxy.json"
    record = normalize(source, output, target_spec_path=target)

    pair = record["pairs"]["mvsplat/re10k"]
    # compute: 4 * (10 / 5), memory: 5 * (200 / 100), overhead unchanged
    assert pair["orin_nx_proxy"]["baseline"]["total_ms"] == 19.0
    assert pair["orin_nx_proxy"]["baseline"]["throughput_fps"] == pytest.approx(1000 / 19)
    assert pair["orin_nx_proxy"]["baseline"]["interval_fps"]["lower"] < 1000 / 19
    assert (
        pair["orin_nx_proxy"]["baseline"]["interval_interpretation"]
        == "component-scaling sensitivity band; not a statistical confidence interval"
    )
    assert pair["orin_nx_proxy"]["asic"]["latency_ms"] == pytest.approx(0.00918)
    assert record["status"] == "PROXY_ONLY"
    assert record["claim_eligible"] is False
    assert record["independent_orin_measurement"] is False
    assert record["assumptions"]["concurrency"] == (
        "input components represent one non-overlapping critical path"
    )
    assert record["unsupported_metrics"]["power_w"]["status"] == "UNAVAILABLE"
    assert record["unsupported_metrics"]["mechanism_counters"]["status"] == (
        "SOURCE_BOUND_ONLY"
    )
    indicators = pair["figure8_indicators"]
    assert indicators["orin_nx_normalized"] == 1.0
    assert indicators["scarf_dataflow_on_orin_nx_normalized"] == pytest.approx(19.0 / 15.0)
    assert indicators["scarf_asic_speedup"] == pytest.approx(19.0 / 0.00918)


def test_proxy_records_metric_mapping_and_source_spec_provenance(tmp_path: Path):
    from scripts.normalize_orin_proxy import normalize

    value = _input()
    value["source_device"]["spec_provenance"] = {
        "source": "https://example.invalid/gpu-datasheet",
        "retrieved_at": "2026-09-03",
        "fp32_basis": "peak FP32 throughput at the recorded clock",
        "bandwidth_basis": "published peak device memory bandwidth",
    }
    source = tmp_path / "gpu.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    record = normalize(source, tmp_path / "proxy.json")

    assert record["source"]["device"]["spec_provenance"]["source"].startswith("https://")
    mapping = record["metric_mapping"]
    assert mapping["latency_ms"]["status"] == "ESTIMATED"
    assert mapping["throughput_fps"]["status"] == "ESTIMATED"
    assert mapping["normalized_speedup"]["status"] == "ESTIMATED"
    assert mapping["quality_psnr_ssim_lpips"]["status"] == "DIRECT_REEXECUTION_REQUIRED"
    assert mapping["mechanism_counters_and_cycles"]["status"] == "SOURCE_BOUND_RECORD_REQUIRED"
    assert mapping["power_temperature_occupancy"]["status"] == "UNAVAILABLE_WITHOUT_ORIN"


def test_normalize_orin_proxy_validates_five_raw_event_medians(tmp_path: Path):
    from scripts.normalize_orin_proxy import normalize

    value = _input()
    for pair in value["pairs"].values():
        pair["baseline"]["event_samples_ms"] = [9.0, 10.0, 11.0, 10.0, 10.0]
        pair["scarf_dataflow"]["event_samples_ms"] = [7.0, 8.0, 9.0, 8.0, 8.0]
    source = tmp_path / "gpu.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    record = normalize(source, tmp_path / "proxy.json")
    assert record["source"]["capture"]["repetitions"] == 5
    assert record["pairs"]["mvsplat/re10k"]["orin_nx_proxy"]["baseline"]["event_samples_ms"]


def test_normalize_orin_proxy_rejects_bad_event_median_and_fractional_cycles(tmp_path: Path):
    from scripts.normalize_orin_proxy import normalize

    value = _input()
    value["pairs"]["mvsplat/re10k"]["baseline"]["event_samples_ms"] = [1, 1, 1, 1, 1]
    source = tmp_path / "gpu.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="median does not match"):
        normalize(source, tmp_path / "proxy.json")

    value = _input()
    value["pairs"]["mvsplat/re10k"]["asic_cycles"] = 9180.5
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        normalize(source, tmp_path / "proxy.json")


def test_init_orin_proxy_input_is_detailed_format_template(tmp_path: Path):
    from scripts.init_orin_proxy_input import init
    from scripts.normalize_orin_proxy import normalize

    output = init(tmp_path / "reviewer-gpu-proxy.json")
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "format_only"
    assert record["capture"]["repetitions"] == 5
    assert set(record["pairs"]) == {
        f"{model}/{dataset}"
        for model in ("transplat", "mvsplat", "depthsplat")
        for dataset in ("re10k", "acid", "dl3dv")
    }
    assert set(record["pairs"]["transplat/re10k"]) == {
        "baseline",
        "scarf_dataflow",
        "profile_evidence",
        "asic_cycles",
        "quality_metrics",
        "mechanism_metrics",
    }
    with pytest.raises(ValueError, match="populated string"):
        normalize(output, tmp_path / "proxy.json")


def test_normalize_orin_proxy_rejects_missing_profiler_evidence(tmp_path: Path):
    from scripts.normalize_orin_proxy import normalize

    value = _input()
    value["pairs"]["mvsplat/re10k"].pop("profile_evidence")
    source = tmp_path / "gpu.json"
    source.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="profile_evidence must be an object"):
        normalize(source, tmp_path / "proxy.json")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(claim_eligible=True), "claim_eligible"),
        (lambda value: value.update(same_workload=False), "same_workload"),
        (lambda value: value["pairs"].pop("mvsplat/re10k"), "exactly the nine"),
    ],
)
def test_normalize_orin_proxy_rejects_non_portable_input(mutation, match, tmp_path: Path):
    from scripts.normalize_orin_proxy import normalize

    value = _input()
    mutation(value)
    source = tmp_path / "gpu.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        normalize(source, tmp_path / "proxy.json")
