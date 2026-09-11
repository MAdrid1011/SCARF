#!/usr/bin/env python3
"""Build a claim timing bundle from real RTL simulator event exports.

The input is intentionally a small adapter contract.  The RTL simulator (for
example Verilator plus a platform-specific C++ harness) writes one entry per
sample with four stage events for the ``asic`` variant and total cycles for all
variants.  This script computes file hashes, emits the strict claim trace
schema, and validates the resulting bundle before returning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.claim_timing_backend import STAGES, VARIANTS, load_claim_timing_manifest


INPUT_SCHEMA = "scarf-rtl-simulator-events-v2"
TRACE_SCHEMA = "source-bound-timing-trace-v2"
CLAIM_EVIDENCE_FIELDS = (
    "stimulus_sha256",
    "stimulus_size_bytes",
    "workload",
    "axi_read_count",
    "axi_read_addresses",
    "role_coverage",
    "axi_consumption_digest",
    "mechanism_counters",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("simulator event export must be an object")
    if value.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"expected {INPUT_SCHEMA} input")
    if value.get("claim_eligible") is False:
        raise ValueError("simulator export is marked non-claim (structural-only)")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _events(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(STAGES):
        raise ValueError(f"{label} must contain one event for each of {STAGES}")
    expected = 0
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"stage", "start_cycle", "end_cycle", "accepted"}:
            raise ValueError(f"{label} contains an invalid event")
        stage = item["stage"]
        start, end = item["start_cycle"], item["end_cycle"]
        if stage not in STAGES or stage in seen or start != expected or not isinstance(start, int) or not isinstance(end, int) or end <= start or item["accepted"] is not True:
            raise ValueError(f"{label} has non-contiguous or invalid stage events")
        result.append({"stage": stage, "start_cycle": start, "end_cycle": end, "accepted": True})
        seen.add(stage)
        expected = end
    return result


def export(events_path: Path, source_rtl: Path, output_dir: Path) -> Path:
    source_rtl = source_rtl.resolve()
    if not source_rtl.is_file():
        raise FileNotFoundError(f"source RTL not found: {source_rtl}")
    source = _load(events_path.resolve())
    claim_mode = source.get("claim_eligible") is True
    clock_mhz = _positive_int(source.get("clock_mhz"), "clock_mhz")
    samples = source.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("simulator event export has no samples")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rtl_dest = output_dir / "rtl" / source_rtl.name
    rtl_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_rtl, rtl_dest)
    source_digest = sha256_file(rtl_dest)
    manifest_samples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in samples:
        if not isinstance(item, Mapping):
            raise ValueError("sample entry must be an object")
        model, dataset, index = item.get("model"), item.get("dataset"), item.get("sample_index")
        if not isinstance(model, str) or not model or not isinstance(dataset, str) or not dataset or isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("sample identity is invalid")
        key = (model, dataset, index)
        if key in seen:
            raise ValueError(f"duplicate sample: {key}")
        seen.add(key)
        if claim_mode and (
            not isinstance(item.get("stimulus_sha256"), str)
            or len(item["stimulus_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in item["stimulus_sha256"])
            or not isinstance(item.get("stimulus_size_bytes"), int)
            or item["stimulus_size_bytes"] <= 0
        ):
            raise ValueError(
                f"{model}/{dataset}/{index} claim event has no stimulus binding"
            )
        if claim_mode:
            missing = [field for field in CLAIM_EVIDENCE_FIELDS if field not in item]
            if missing:
                raise ValueError(
                    f"{model}/{dataset}/{index} claim event lacks v2 evidence: "
                    + ", ".join(missing)
                )
        events_value = _events(item.get("events"), f"{model}/{dataset}/{index}.events")
        stage_cycles = {event["stage"]: event["end_cycle"] - event["start_cycle"] for event in events_value}
        variants = item.get("variants")
        if not isinstance(variants, Mapping) or set(variants) != set(VARIANTS):
            raise ValueError(f"{model}/{dataset}/{index} must provide all timing variants")
        normalized_variants = {variant: {"total_cycles": _positive_int(variants[variant].get("total_cycles") if isinstance(variants[variant], Mapping) else None, f"{key}.{variant}.total_cycles")} for variant in VARIANTS}
        trace_rel = Path("timing-trace") / model / dataset / f"sample_{index:05d}.json"
        trace_dest = output_dir / trace_rel
        trace_dest.parent.mkdir(parents=True, exist_ok=True)
        trace = {"schema_version": TRACE_SCHEMA, "model": model, "dataset": dataset, "sample_index": index, "cycle_accurate": True, "source_rtl_sha256": source_digest, "clock_mhz": clock_mhz, "events": events_value}
        if item.get("stimulus_sha256") is not None:
            stimulus_digest = item.get("stimulus_sha256")
            stimulus_size = item.get("stimulus_size_bytes")
            if not isinstance(stimulus_digest, str) or len(stimulus_digest) != 64 or any(c not in "0123456789abcdef" for c in stimulus_digest):
                raise ValueError(f"{model}/{dataset}/{index} has an invalid stimulus SHA256")
            if not isinstance(stimulus_size, int) or stimulus_size <= 0:
                raise ValueError(f"{model}/{dataset}/{index} has an invalid stimulus size")
            trace["stimulus_sha256"] = stimulus_digest
            trace["stimulus_size_bytes"] = stimulus_size
        for field in (
            "workload",
            "axi_read_count",
            "axi_read_addresses",
            "role_coverage",
            "axi_consumption_digest",
            "mechanism_counters",
            "applied_registers",
            "run_selection_sha256",
        ):
            if field in item:
                trace[field] = item[field]
        trace_dest.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_sample = {"model": model, "dataset": dataset, "sample_index": index, "trace": {"path": trace_rel.as_posix(), "sha256": sha256_file(trace_dest)}, "variants": normalized_variants, "combined_stage_cycles": stage_cycles}
        for field in (
            "workload",
            "axi_read_count",
            "axi_read_addresses",
            "role_coverage",
            "axi_consumption_digest",
            "mechanism_counters",
            "applied_registers",
            "run_selection_sha256",
        ):
            if field in item:
                manifest_sample[field] = item[field]
        manifest_samples.append(manifest_sample)
    manifest = {"schema_version": "source-bound-timing-backend-v1", "backend": {"kind": "source_rtl", "source": {"path": (Path("rtl") / source_rtl.name).as_posix(), "sha256": source_digest}}, "clock_mhz": clock_mhz, "samples": manifest_samples}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_claim_timing_manifest(manifest_path, root=output_dir)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True, help=f"{INPUT_SCHEMA} JSON emitted by the RTL adapter")
    parser.add_argument("--source-rtl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(export(args.events, args.source_rtl, args.output_dir))
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
