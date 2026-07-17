#!/usr/bin/env python3
"""Convert measured SCARF memory events into a Ramulator LoadStoreTrace."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _address(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("address must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        result = int(value, 0)
    else:
        raise ValueError("address must be an integer or base-prefixed string")
    if not 0 <= result < 2**64:
        raise ValueError("address is outside the unsigned 64-bit range")
    return result


def export(
    source: Path,
    output: Path,
    line_bytes: int = 32,
    evidence_root: Path | None = None,
    workload_result: Path | None = None,
) -> dict:
    if line_bytes <= 0 or line_bytes & (line_bytes - 1):
        raise ValueError("line_bytes must be a positive power of two")
    events = []
    observed_sample_indices: set[int] = set()
    previous_cycle = -1
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        cycle = event.get("cycle")
        operation = event.get("op")
        size = event.get("bytes")
        if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < previous_cycle:
            raise ValueError(f"line {line_number}: cycles must be nonnegative and nondecreasing")
        if operation not in {"read", "write"}:
            raise ValueError(f"line {line_number}: op must be read or write")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"line {line_number}: bytes must be positive")
        address = _address(event.get("address"))
        if workload_result is not None:
            sample_index = event.get("sample_index")
            if (
                not isinstance(sample_index, int)
                or isinstance(sample_index, bool)
                or sample_index < 0
            ):
                raise ValueError(
                    f"line {line_number}: workload events require a nonnegative sample_index"
                )
            observed_sample_indices.add(sample_index)
        previous_cycle = cycle
        first_line = address // line_bytes * line_bytes
        last_line = (address + size - 1) // line_bytes * line_bytes
        for cacheline in range(first_line, last_line + line_bytes, line_bytes):
            events.append(("LD" if operation == "read" else "ST", cacheline))
    if not events:
        raise ValueError("memory event stream is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{op} 0x{address:016x}\n" for op, address in events), encoding="ascii")
    source_path = str(source)
    trace_path = str(output)
    path_base = None
    if evidence_root is not None:
        evidence_root = evidence_root.resolve()
        source_path = source.resolve().relative_to(evidence_root).as_posix()
        trace_path = output.resolve().relative_to(evidence_root).as_posix()
        path_base = "output_dir"
    source_record = {"path": source_path, "sha256": sha256_file(source)}
    trace_record = {"path": trace_path, "sha256": sha256_file(output)}
    if path_base is not None:
        source_record["path_base"] = path_base
        trace_record["path_base"] = path_base
    record = {
        "schema_version": "1.0",
        "evidence_type": "ramulator_load_store_trace",
        "request_count": len(events),
        "read_requests": sum(op == "LD" for op, _ in events),
        "write_requests": sum(op == "ST" for op, _ in events),
        "transaction_bytes": line_bytes,
        "issue_model": "one queued request per Ramulator frontend tick; source cycles retained only in JSONL",
        "source": source_record,
        "trace": trace_record,
    }
    if workload_result is not None:
        workload = json.loads(workload_result.read_text(encoding="utf-8"))
        provenance = workload.get("provenance", {})
        dataset = provenance.get("dataset", {})
        evaluation = provenance.get("evaluation", {})
        sample_indices = evaluation.get("sample_indices")
        selection_hash = evaluation.get("sample_selection_sha256")
        if (
            workload.get("schema_version") != "2.0"
            or workload.get("evidence_class") != "deterministic_execution"
            or dataset.get("paper_result_eligible") is not True
            or evaluation.get("kind") != "dataset_aggregate"
            or not isinstance(sample_indices, list)
            or not sample_indices
            or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in sample_indices
            )
            or len(set(sample_indices)) != len(sample_indices)
            or not isinstance(selection_hash, str)
            or len(selection_hash) != 64
        ):
            raise ValueError("workload result is not an eligible deterministic aggregate")
        if observed_sample_indices != set(sample_indices):
            raise ValueError(
                "memory-event sample indices do not match the software aggregate"
            )
        software_path = str(workload_result)
        software_record = {
            "path": software_path,
            "sha256": sha256_file(workload_result),
        }
        if evidence_root is not None:
            software_record["path"] = workload_result.resolve().relative_to(
                evidence_root.resolve()
            ).as_posix()
            software_record["path_base"] = "output_dir"
        record["workload"] = {
            "model": provenance.get("model"),
            "dataset": dataset.get("name"),
            "sample_count": len(sample_indices),
            "sample_indices": sample_indices,
            "sample_selection_sha256": selection_hash,
            "software_result": software_record,
        }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--line-bytes", type=int, default=32)
    parser.add_argument("--workload-result", type=Path)
    args = parser.parse_args()
    try:
        record = export(
            args.input.resolve(),
            args.output.resolve(),
            args.line_bytes,
            args.manifest.resolve().parent,
            args.workload_result.resolve() if args.workload_result else None,
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
