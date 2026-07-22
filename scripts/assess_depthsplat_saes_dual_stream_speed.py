#!/usr/bin/env python3
"""Certify a resource-bound SAES speed model from sealed DL3DV quality runs.

The input records remain immutable dense-capture quality evidence.  This tool
does not load a model, a dataset, or target frames; it only binds each sealed
source route to the coefficient-compatible dual-stream architectural schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from saes.depthsplat_sparse_datapath_projection import (
    DUAL_STREAM_KIND,
    simulate_depthsplat_dual_stream_schedule,
)


KIND = "depthsplat-dl3dv-saes-dual-stream-speed-certificate-v1"
DEFAULT_RESULT_PATHS = (
    ROOT
    / "outputs/ae_dl3dv_repair_diagnostics/"
    "depthsplat_sample007_kernel_closure_psnr05_projection_v1/results.json",
    ROOT
    / "outputs/ae_dl3dv_repair_diagnostics/"
    "depthsplat_sample016_kernel_closure_psnr05_projection_v1/results.json",
    ROOT
    / "outputs/ae_dl3dv_repair_diagnostics/"
    "depthsplat_sample019_kernel_closure_psnr05_projection_v1/results.json",
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/ae_dl3dv_repair_diagnostics/"
    "depthsplat_dl3dv_kernel_closure_dual_stream_speed_certificate_v1/results.json"
)
TARGET_SAES_SPEEDUP = 1.26


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_nonnegative_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be a nonnegative finite number")
    return float(value)


def _require_finite_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def assess_record(path: Path, *, target_saes_speedup: float) -> dict[str, Any]:
    """Bind one sealed route and its PSNR-only quality result to the schedule."""

    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping):
        raise ValueError(f"{path} is not a result record")
    if record.get("status") != "PASS":
        raise ValueError(f"{path} is not a passing quality record")
    if record.get("mechanism") != "soft-mixture-kernel-closure-v3":
        raise ValueError(f"{path} has an unexpected mechanism")

    quality = _require_mapping(record.get("quality"), "quality")
    verdict = _require_mapping(quality.get("verdict"), "quality verdict")
    observed = _require_mapping(verdict.get("observed"), "quality observations")
    psnr_loss_db = _require_finite_float(observed.get("psnr_loss_db"), "PSNR loss")
    limit = _require_nonnegative_float(
        _require_mapping(verdict.get("limits"), "quality limits").get("psnr_loss_db"),
        "PSNR limit",
    )
    if verdict.get("pass") is not True or psnr_loss_db > limit:
        raise ValueError(f"{path} does not satisfy its declared PSNR gate")

    mechanism_cycles = _require_mapping(record.get("mechanism_cycles"), "mechanism cycles")
    projection = _require_mapping(
        mechanism_cycles.get("projected_sparse_datapath_cycles"),
        "sparse-datapath projection",
    )
    if projection.get("measured") is not False:
        raise ValueError(f"{path} projection must remain non-measured")
    stream = _require_mapping(projection.get("streaming_schedule"), "streaming schedule")
    route = _require_mapping(
        _require_mapping(mechanism_cycles.get("table_3_saes"), "SAES route table").get(
            "route_counts"
        ),
        "route counts",
    )
    tile_count = sum(
        _require_nonnegative_int(route.get(level), f"{level} tile count")
        for level in ("L0", "L1", "Full")
    )
    schedule = simulate_depthsplat_dual_stream_schedule(
        dense_total_cycles=_require_positive_int(
            projection.get("dense_total_cycles"), "dense cycle total"
        ),
        s2_probe_pipeline_cycles=_require_positive_int(
            stream.get("s2_probe_pipeline_cycles"), "S2 pipeline cycles"
        ),
        s3_feature_preparation_cycles=_require_positive_int(
            stream.get("s3_feature_preparation_cycles"), "S3 feature cycles"
        ),
        s3_selected_head_cycles=_require_positive_int(
            stream.get("s3_selected_head_cycles"), "S3 head cycles"
        ),
        saes_control_and_materialization_cycles=_require_positive_int(
            projection.get("saes_control_and_materialization_cycles"),
            "SAES control cycles",
        ),
        tile_count=tile_count,
        target_saes_speedup=target_saes_speedup,
    )
    return {
        "source_sample_index": record.get("source_sample_index"),
        "input_result": str(path.relative_to(ROOT)),
        "input_result_sha256": _sha256_file(path),
        "quality": {
            "psnr_loss_db": psnr_loss_db,
            "psnr_limit_db": limit,
            "quality_gate": verdict.get("quality_gate"),
            "pass": True,
        },
        "route_counts": {level: int(route[level]) for level in ("L0", "L1", "Full")},
        "dual_stream_schedule": schedule,
    }


def build_certificate(
    result_paths: Sequence[Path], *, target_saes_speedup: float = TARGET_SAES_SPEEDUP
) -> dict[str, Any]:
    if len(result_paths) < 1:
        raise ValueError("at least one sealed result is required")
    if len(set(result_paths)) != len(result_paths):
        raise ValueError("duplicate sealed result paths are not allowed")
    records = [assess_record(path, target_saes_speedup=target_saes_speedup) for path in result_paths]
    sample_indices = [record["source_sample_index"] for record in records]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in sample_indices):
        raise ValueError("sealed result has an invalid source sample index")
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sealed result sample indices must be unique")

    latencies = [record["dual_stream_schedule"]["latency"] for record in records]
    dense_total = sum(int(latency["dense_baseline_cycles"]) for latency in latencies)
    sparse_total = sum(int(latency["dual_stream_sparse_cycles"]) for latency in latencies)
    per_sample_speedups = [float(latency["dual_stream_speedup"]) for latency in latencies]
    aggregate_speedup = dense_total / sparse_total
    geometric_mean_speedup = math.prod(per_sample_speedups) ** (1.0 / len(records))
    all_quality_pass = all(record["quality"]["pass"] for record in records)
    all_targets_reached = all(
        record["dual_stream_schedule"]["target_speedup_feasibility"]["target_reached"]
        for record in records
    )

    return {
        "kind": KIND,
        "schema_version": "1.0",
        "measured": False,
        "paper_result_eligible": False,
        "timing_class": "resource-bound-architectural-cycle-model",
        "scope": {
            "model": "DepthSplat",
            "dataset": "DL3DV",
            "mechanism": "soft-mixture-kernel-closure-v3",
            "quality_gate": "user-authorized-psnr-only-v1",
            "target_saes_speedup": float(target_saes_speedup),
        },
        "architecture_boundary": {
            "schedule_kind": DUAL_STREAM_KIND,
            "same_weights_as_sealed_source_routes": True,
            "current_scarf_single_mmcu_rtl_can_execute_schedule": False,
            "required_extension": (
                "independent S3 MMCU lane, weight-read bank, and feature scratchpad; "
                "the existing dual-port TileBuffer provides the S2-write/S3-read handoff"
            ),
            "not_a_pytorch_or_gpu_wall_clock_measurement": True,
            "not_a_physical_or_paper_claim": True,
        },
        "records": records,
        "aggregate": {
            "sample_count": len(records),
            "source_sample_indices": sorted(sample_indices),
            "total_dense_cycles": dense_total,
            "total_dual_stream_cycles": sparse_total,
            "aggregate_speedup": aggregate_speedup,
            "geometric_mean_speedup": geometric_mean_speedup,
            "minimum_per_sample_speedup": min(per_sample_speedups),
            "maximum_per_sample_speedup": max(per_sample_speedups),
            "all_psnr_quality_gates_pass": all_quality_pass,
            "all_samples_reach_target_saes_speedup": all_targets_reached,
        },
        "status": (
            "PASS_ARCHITECTURAL_CYCLE_MODEL"
            if all_quality_pass and all_targets_reached
            else "FAIL"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        type=Path,
        default=None,
        help="sealed DL3DV quality result; repeat for every sample",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-saes-speedup", type=float, default=TARGET_SAES_SPEEDUP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = tuple(args.result) if args.result else DEFAULT_RESULT_PATHS
    certificate = build_certificate(paths, target_saes_speedup=args.target_saes_speedup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{certificate['status']}: "
        f"aggregate SAES speedup {certificate['aggregate']['aggregate_speedup']:.4f}x "
        f"from {certificate['aggregate']['sample_count']} sealed DL3DV routes"
    )


if __name__ == "__main__":
    main()
