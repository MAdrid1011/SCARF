#!/usr/bin/env python3
"""Capture an auditable Jetson Orin NX inference measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_environment import validate_orin
from scripts.validate_result import validate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"required Orin measurement tool not found: {name}")
    return path


def parse_tegrastats(text: str) -> dict:
    import re

    temperatures = [float(value) for value in re.findall(r"@([0-9]+(?:\.[0-9]+)?)C", text)]
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not temperatures:
        raise ValueError("tegrastats contains no temperature samples")
    return {
        "sample_lines": len(lines),
        "maximum_temperature_c": max(temperatures),
        "minimum_temperature_c": min(temperatures),
    }


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_measurement_record(
    record: dict,
    *,
    expected_selection_sha256: str,
    measurement_path: Path | None = None,
) -> None:
    if record.get("kind") != "orin_nx_measurement" or record.get("status") != "PASS":
        raise ValueError("Orin measurement status is not PASS")
    if record.get("baseline_source") != "orin_nx_cuda_events":
        raise ValueError("Orin baseline source is not orin_nx_cuda_events")
    environment = record.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("Orin measurement environment is missing")
    contract = environment.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Orin measurement contract is missing")
    expected_device = str(contract.get("device_model_contains", "Jetson Orin NX"))
    if expected_device not in str(environment.get("device_model", "")):
        raise ValueError(f"measurement device is not {expected_device}")

    selection = record.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Orin measurement selection provenance is missing")
    if selection.get("sample_selection_sha256") != expected_selection_sha256:
        raise ValueError("Orin measurement selection SHA256 mismatch")
    for key in ("sample_selection_sha256", "dataset_tree_sha256", "checkpoint_sha256"):
        if not _sha256(selection.get(key)):
            raise ValueError(f"Orin measurement {key} is invalid")

    median = record.get("cuda_event_encoder_median_ms")
    if not isinstance(median, (int, float)) or isinstance(median, bool) or median <= 0:
        raise ValueError("Orin CUDA-event median is missing")
    if record.get("cuda_event_repetitions") != 5:
        raise ValueError("Orin measurement requires five CUDA-event repetitions")
    samples = record.get("cuda_event_encoder_samples_ms")
    if (
        not isinstance(samples, list)
        or len(samples) != 5
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
            for value in samples
        )
    ):
        raise ValueError("Orin measurement requires five raw CUDA-event values")
    if abs(float(median) - float(statistics.median(samples))) > 1e-6:
        raise ValueError("Orin CUDA-event median does not match the raw values")

    thermal = record.get("thermal")
    maximum = thermal.get("maximum_temperature_c") if isinstance(thermal, dict) else None
    allowed = contract.get("maximum_allowed_temperature_c")
    if not isinstance(maximum, (int, float)) or not isinstance(allowed, (int, float)):
        raise ValueError("Orin temperature evidence is missing")
    if maximum > allowed:
        raise ValueError(f"Orin temperature exceeded contract: {maximum} C > {allowed} C")

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Orin raw artifacts are missing")
    for name in ("inference_result", "tegrastats", "nsight_systems", "cuda_events"):
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict) or not _sha256(artifact.get("sha256")):
            raise ValueError(f"Orin raw artifact is missing or invalid: {name}")
        if measurement_path is not None:
            artifact_path = Path(str(artifact.get("path", "")))
            if not artifact_path.is_absolute():
                artifact_path = measurement_path.parent / artifact_path
            if (
                not artifact_path.is_file()
                or sha256_file(artifact_path) != artifact["sha256"]
            ):
                raise ValueError(f"Orin raw artifact hash mismatch: {name}")


def _profile_command(command: list[str], evidence_dir: Path) -> dict:
    if not command:
        raise ValueError("an inference command is required after --")
    environment = validate_orin()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    tegrastats_log = evidence_dir / "tegrastats.log"
    nsys_prefix = evidence_dir / "nsight"
    tegrastats = subprocess.Popen(
        [required_tool("tegrastats"), "--interval", "100", "--logfile", str(tegrastats_log)],
        cwd=ROOT,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        result = subprocess.run(
            [
                required_tool("nsys"),
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--force-overwrite=true",
                f"--output={nsys_prefix}",
                *command,
            ],
            cwd=ROOT,
            check=False,
        )
    finally:
        tegrastats.send_signal(signal.SIGINT)
        try:
            tegrastats.wait(timeout=10)
        except subprocess.TimeoutExpired:
            tegrastats.terminate()
            tegrastats.wait(timeout=5)
    elapsed = time.monotonic() - start
    if result.returncode:
        raise RuntimeError(f"Orin inference command failed with exit {result.returncode}")
    nsys_reports = sorted(evidence_dir.glob("nsight*.nsys-rep"))
    if not nsys_reports:
        raise FileNotFoundError("Nsight Systems report was not generated")
    if not tegrastats_log.is_file() or tegrastats_log.stat().st_size == 0:
        raise FileNotFoundError("tegrastats log is empty")
    thermal = parse_tegrastats(tegrastats_log.read_text(encoding="utf-8", errors="replace"))
    maximum_allowed = float(environment["contract"]["maximum_allowed_temperature_c"])
    if thermal["maximum_temperature_c"] > maximum_allowed:
        raise RuntimeError(
            f"Orin temperature exceeded contract: {thermal['maximum_temperature_c']} C > "
            f"{maximum_allowed} C"
        )
    return {
        "environment": environment,
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "thermal": thermal,
        "command": command,
        "artifacts": {
            "tegrastats": tegrastats_log,
            "nsight_systems": nsys_reports[-1],
        },
    }


def _timing_from_inference(inference: dict) -> tuple[list[float], float]:
    validate(inference)
    device = inference["provenance"]["device"]
    device_name = str(device.get("name", ""))
    if "Orin" not in device_name:
        raise RuntimeError(f"result was not measured on Orin: {device_name}")
    if inference["performance"].get("baseline_source") != "orin_nx_cuda_events":
        raise RuntimeError("result baseline is not sourced from Orin NX CUDA events")
    if device.get("encoder_timing_source") != "cuda_events":
        raise RuntimeError("Orin result does not use CUDA-event timing")
    samples = device.get("encoder_timing_samples_ms")
    if (
        not isinstance(samples, list)
        or len(samples) != 5
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
            for value in samples
        )
    ):
        raise RuntimeError("Orin result has no five-value CUDA-event trace")
    samples = [float(value) for value in samples]
    median = float(statistics.median(samples))
    measured = device.get("measured_encoder_time_ms")
    if not isinstance(measured, (int, float)) or abs(float(measured) - median) > 1e-6:
        raise RuntimeError("Orin result median does not match its CUDA-event trace")
    return samples, median


def build_measurement_record(
    *,
    inference: dict,
    inference_path: Path,
    expected_selection_sha256: str,
    profile: dict,
    cuda_events_path: Path,
) -> dict:
    samples, median = _timing_from_inference(inference)
    evaluation = inference["provenance"]["evaluation"]
    provenance = inference["provenance"]
    artifacts = {
        "inference_result": inference_path,
        **profile["artifacts"],
        "cuda_events": cuda_events_path,
    }
    record = {
        "schema_version": "1.0",
        "kind": "orin_nx_measurement",
        "status": "PASS",
        "baseline_source": "orin_nx_cuda_events",
        "started_at": profile["started_at"],
        "elapsed_seconds": profile["elapsed_seconds"],
        "command": profile["command"],
        "environment": profile["environment"],
        "selection": {
            "sample_selection_sha256": expected_selection_sha256,
            "dataset_tree_sha256": provenance["dataset"]["tree_sha256"],
            "checkpoint_sha256": provenance["checkpoint"]["sha256"],
            "sample_index": evaluation["sample_index"],
            "execution_index": evaluation["execution_index"],
            "scene": evaluation["scene"],
            "context_indices": evaluation["context_indices"],
            "target_indices": evaluation["target_indices"],
        },
        "cuda_event_encoder_samples_ms": samples,
        "cuda_event_encoder_median_ms": median,
        "cuda_event_repetitions": len(samples),
        "thermal": profile["thermal"],
        "artifacts": {
            name: {
                "path": os.path.relpath(path, cuda_events_path.parent),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }
    return record


def capture_pair(
    command: list[str],
    pair_dir: Path,
    evidence_dir: Path,
    *,
    expected_selection_sha256: str,
    expected_count: int,
) -> dict:
    profile = _profile_command(command, evidence_dir)
    result_paths = sorted(pair_dir.glob("samples/sample_*/results.json"))
    if len(result_paths) != expected_count:
        raise RuntimeError(
            f"Orin pair expected {expected_count} sample results, found {len(result_paths)}"
        )
    measurements = []
    for result_path in result_paths:
        inference = json.loads(result_path.read_text(encoding="utf-8"))
        samples, median = _timing_from_inference(inference)
        sample_evidence = result_path.parent / "orin-evidence"
        sample_evidence.mkdir(parents=False, exist_ok=False)
        cuda_events_path = sample_evidence / "cuda-events.json"
        cuda_events_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "source": "torch.cuda.Event",
                    "samples_ms": samples,
                    "median_ms": median,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        measurement = build_measurement_record(
            inference=inference,
            inference_path=result_path,
            expected_selection_sha256=expected_selection_sha256,
            profile=profile,
            cuda_events_path=cuda_events_path,
        )
        validate_measurement_record(
            measurement, expected_selection_sha256=expected_selection_sha256
        )
        measurement_path = sample_evidence / "measurement.json"
        measurement_path.write_text(
            json.dumps(measurement, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        measurements.append(
            {
                "sample_index": measurement["selection"]["sample_index"],
                "path": str(measurement_path),
                "sha256": sha256_file(measurement_path),
            }
        )
    summary = {
        "schema_version": "1.0",
        "kind": "orin_nx_pair_measurement",
        "status": "PASS",
        "sample_count": len(measurements),
        "sample_selection_sha256": expected_selection_sha256,
        "measurements": measurements,
    }
    (evidence_dir / "pair-measurement.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--sample-selection-sha256", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> int:
    args = parse_args()
    capture_pair(
        args.command,
        args.pair_dir.resolve(),
        args.evidence_dir.resolve(),
        expected_selection_sha256=args.sample_selection_sha256,
        expected_count=args.expected_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
