#!/usr/bin/env python3
"""Convert a model sample record into a deterministic RTL stimulus manifest.

The converter deliberately records references and hashes rather than inventing
tensor values.  A platform RTL adapter consumes this manifest and emits the
cycle events accepted by ``export_claim_timing.py``.
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

from scripts.rtl_payload import parse_payload, workload_descriptor


SCHEMA = "scarf-rtl-stimulus-v1"
PAYLOAD_SCHEMA = "scarf-packed-payload-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"sample record must be an object: {path}")
    return value


def _evaluation(record: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = record.get("provenance")
    evaluation = provenance.get("evaluation") if isinstance(provenance, Mapping) else None
    if not isinstance(evaluation, Mapping) or evaluation.get("kind") != "sample":
        raise ValueError("sample record has no provenance.evaluation sample identity")
    for field in ("sample_index", "execution_index", "scene", "context_indices", "target_indices"):
        if field not in evaluation:
            raise ValueError(f"sample evaluation is missing {field}")
    if (
        isinstance(evaluation["sample_index"], bool)
        or not isinstance(evaluation["sample_index"], int)
        or evaluation["sample_index"] < 0
        or isinstance(evaluation["execution_index"], bool)
        or not isinstance(evaluation["execution_index"], int)
        or evaluation["execution_index"] < 0
        or not isinstance(evaluation["scene"], str)
        or not isinstance(evaluation["context_indices"], list)
        or not isinstance(evaluation["target_indices"], list)
    ):
        raise ValueError("sample evaluation identity has invalid types")
    return evaluation


def _payload_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, list) and all(isinstance(item, (str, Path)) for item in value):
        return [Path(item) for item in value]
    raise ValueError("payload files must be paths")


def _pack_payload(files: list[Path], *, output: Path, input_root: Path) -> dict[str, Any]:
    """Pack target-free bytes and retain per-file provenance and offsets."""
    if not files:
        return {
            "schema_version": PAYLOAD_SCHEMA,
            "required": True,
            "encoding": "raw-concatenated-little-endian-bytes",
            "path": None,
            "sha256": None,
            "size_bytes": 0,
            "files": [],
        }
    resolved: list[Path] = []
    seen: set[Path] = set()
    for item in files:
        path = item.resolve()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"payload must be a regular file: {item}")
        if path in seen:
            raise ValueError(f"duplicate payload file: {item}")
        seen.add(path)
        resolved.append(path)

    payload_dir = output.parent / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)

    # ``demo.py --rtl-payload-output`` already produces the self-describing
    # SCARF container.  Preserve that container byte-for-byte instead of
    # stripping its header and repacking it as anonymous bytes.  Anonymous
    # legacy payloads remain representable for format tests, but cannot pass
    # the claim timing adapter because they have no workload descriptor.
    if len(resolved) == 1:
        try:
            parsed = parse_payload(resolved[0])
        except ValueError:
            parsed = None
        if parsed is not None:
            data = resolved[0].read_bytes()
            payload_digest = hashlib.sha256(data).hexdigest()
            final = payload_dir / f"payload-{payload_digest}.bin"
            if final != resolved[0]:
                shutil.copyfile(resolved[0], final)
            return {
                "schema_version": PAYLOAD_SCHEMA,
                "required": True,
                "encoding": (
                    "scarf-rtl-payload-v2"
                    if parsed.header.get("schema_version") == "scarf-rtl-payload-v2"
                    else "scarf-rtl-payload-v1"
                ),
                "path": final.relative_to(output.parent).as_posix(),
                "sha256": payload_digest,
                "size_bytes": len(data),
                "files": [
                    {
                        "name": resolved[0].name,
                        "sha256": _sha256(resolved[0]),
                        "offset_bytes": 0,
                        "size_bytes": len(data),
                    }
                ],
                "workload": workload_descriptor(parsed),
            }

    temporary = payload_dir / f".{output.stem}.payload.tmp"
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    offset = 0
    with temporary.open("wb") as destination:
        for path in resolved:
            data = path.read_bytes()
            destination.write(data)
            digest.update(data)
            try:
                name = path.relative_to(input_root).as_posix()
            except ValueError:
                name = path.name
            entries.append(
                {
                    "name": name,
                    "sha256": _sha256(path),
                    "offset_bytes": offset,
                    "size_bytes": len(data),
                }
            )
            offset += len(data)
    payload_digest = digest.hexdigest()
    final = payload_dir / f"payload-{payload_digest}.bin"
    temporary.replace(final)
    return {
        "schema_version": PAYLOAD_SCHEMA,
        "required": True,
        "encoding": "raw-concatenated-little-endian-bytes",
        "path": final.relative_to(output.parent).as_posix(),
        "sha256": payload_digest,
        "size_bytes": offset,
        "files": entries,
        "workload": None,
    }


def convert(
    sample_path: Path,
    output: Path,
    *,
    input_root: Path | None = None,
    payload: list[Path] | None = None,
) -> Path:
    sample_path = sample_path.resolve()
    record = _load(sample_path)
    evaluation = _evaluation(record)
    provenance = record.get("provenance")
    provenance_map = provenance if isinstance(provenance, Mapping) else {}
    model = record.get("model") or provenance_map.get("model")
    dataset_value = record.get("dataset") or provenance_map.get("dataset")
    if isinstance(dataset_value, Mapping):
        dataset = dataset_value.get("name")
    else:
        dataset = dataset_value
    if not isinstance(model, str) or not model or not isinstance(dataset, str) or not dataset:
        raise ValueError("sample record must identify model and dataset")
    root = (input_root or sample_path.parent).resolve()
    try:
        relative = sample_path.relative_to(root).as_posix()
    except ValueError:
        relative = sample_path.name
    root = (input_root or sample_path.parent).resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_files = list(payload or [])
    if not payload_files:
        declared = (
            record.get("rtl_payload")
            or record.get("stimulus_payload")
            or provenance_map.get("rtl_payload")
            or provenance_map.get("stimulus_payload")
        )
        if isinstance(declared, Mapping):
            payload_files = _payload_paths(declared.get("files"))
            if not payload_files and declared.get("path"):
                payload_files = _payload_paths(declared.get("path"))
        elif declared is not None:
            payload_files = _payload_paths(declared)
    packed_payload = _pack_payload(payload_files, output=output, input_root=root)
    stimulus = {
        "schema_version": SCHEMA,
        "model": model,
        "dataset": dataset,
        "sample_index": evaluation["sample_index"],
        "execution_index": evaluation["execution_index"],
        "selection": {
            "scene": evaluation["scene"],
            "context_indices": list(evaluation["context_indices"]),
            "target_indices": list(evaluation["target_indices"]),
        },
        "source_record": {"path": relative, "sha256": _sha256(sample_path)},
        "payload": packed_payload,
        "stimulus_policy": {
            "encoding": "adapter-defined-packed-tensors",
            "target_free": True,
            "requires_platform_adapter": False,
            "consumer": "ScarfTop AXI read channel via run_rtl_timing.py",
            "provenance": "source_record_and_runtime_assets",
        },
    }
    run_selection_sha256 = evaluation.get("run_selection_sha256")
    if run_selection_sha256 is not None:
        if (
            not isinstance(run_selection_sha256, str)
            or len(run_selection_sha256) != 64
            or any(character not in "0123456789abcdef" for character in run_selection_sha256)
        ):
            raise ValueError("sample run_selection_sha256 is invalid")
        stimulus["run_selection_sha256"] = run_selection_sha256
        stimulus["selection"]["run_selection_sha256"] = run_selection_sha256
    output.write_text(json.dumps(stimulus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="schema 2.1 sample result")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument(
        "--payload",
        type=Path,
        action="append",
        help="target-free tensor/payload file; may be repeated",
    )
    args = parser.parse_args(argv)
    try:
        print(convert(args.sample, args.output, input_root=args.input_root, payload=args.payload))
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
