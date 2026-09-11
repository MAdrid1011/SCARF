#!/usr/bin/env python3
"""Build and validate a non-claim timing-backend interface fixture.

This fixture is deliberately not accepted by ``claim_timing_backend.py``.  It
is a deterministic path/hash/schema probe for a reviewer or author bringing a
real RTL simulator to an Orin platform.  It must never be used as timing
evidence or copied into ``artifact/reference_results``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.claim_timing_backend import (
    DATASETS,
    MODELS,
    STAGES,
    TIMING_TRACE_ROOT,
    VARIANTS,
    _digest,
    _positive_int,
    _resolve_under,
    _safe_relative,
    sha256_file,
)

FIXTURE_SCHEMA_VERSION = "claim-timing-validation-fixture-v1"
FIXTURE_KIND = "non-claim-validation-fixture"
FIXTURE_SAMPLE = {
    "model": "mvsplat",
    "dataset": "re10k",
    "sample_index": 0,
}
FIXTURE_CYCLES = {
    "asic": {"s1": 60, "s2": 140, "s3": 90, "s4": 110},
    "asic_fsdr": {"s1": 60, "s2": 105, "s3": 70, "s4": 110},
    "asic_saes": {"s1": 60, "s2": 120, "s3": 55, "s4": 110},
    "asic_fsdr_saes": {"s1": 60, "s2": 95, "s3": 40, "s4": 110},
}


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _reject_release_output(path: Path) -> Path:
    """Keep generated validation data outside every release/evidence root."""
    resolved = path.resolve()
    forbidden_roots = (
        ROOT / "artifact",
        ROOT / "outputs",
        ROOT / "evidence",
    )
    if any(
        resolved == root.resolve() or root.resolve() in resolved.parents
        for root in forbidden_roots
    ):
        raise ValueError(
            "validation fixture must be outside artifact/, outputs/, and evidence/"
        )
    return resolved


def _trace_record() -> dict[str, Any]:
    events = []
    cursor = 0
    for stage in STAGES:
        end = cursor + FIXTURE_CYCLES["asic"][stage]
        events.append({"stage": stage, "start_cycle": cursor, "end_cycle": end})
        cursor = end
    return {
        "schema_version": "synthetic-cycle-trace-v1",
        "trace_class": "diagnostic-interface-fixture",
        "cycle_accurate": False,
        "synthetic_fixture": True,
        "claim_eligible": False,
        "model": FIXTURE_SAMPLE["model"],
        "dataset": FIXTURE_SAMPLE["dataset"],
        "sample_index": FIXTURE_SAMPLE["sample_index"],
        "events": events,
    }


def _manifest(source: Path, trace: Path, root: Path) -> dict[str, Any]:
    source_relative = source.resolve().relative_to(root.resolve()).as_posix()
    trace_relative = trace.resolve().relative_to(root.resolve()).as_posix()
    variants = {
        variant: {
            "total_cycles": sum(FIXTURE_CYCLES[variant].values()),
        }
        for variant in VARIANTS
    }
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "bundle_kind": FIXTURE_KIND,
        "claim_eligible": False,
        "synthetic_fixture": True,
        "not_for_release": True,
        "purpose": "interface_validation_only",
        "backend": {
            "kind": "source_rtl",
            "source": {
                "path": source_relative,
                "sha256": sha256_file(source),
            },
        },
        "clock_mhz": 1000,
        "samples": [
            {
                **FIXTURE_SAMPLE,
                "trace": {
                    "path": trace_relative,
                    "sha256": sha256_file(trace),
                },
                "variants": variants,
                "combined_stage_cycles": dict(FIXTURE_CYCLES["asic"]),
            }
        ],
    }


def build_fixture(output_dir: Path) -> Path:
    root = _reject_release_output(output_dir)
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"fixture output is not a directory: {root}")
    source = root / "rtl" / "ScarfTop_validation_fixture.sv"
    trace = root / TIMING_TRACE_ROOT / "mvsplat" / "re10k" / "sample_00000.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "// Synthetic interface-validation source; not SCARF production RTL.\n"
        "module ScarfTop_validation_fixture;\n"
        "endmodule\n",
        encoding="utf-8",
    )
    _write_json(trace, _trace_record())
    manifest = root / "manifest.json"
    _write_json(manifest, _manifest(source, trace, root))
    validate_fixture(manifest, root=root)
    return manifest


def _validate_fixture_backend(
    record: Mapping[str, Any], root: Path
) -> tuple[Path, dict[str, Any]]:
    if set(record) != {"kind", "source"} or record.get("kind") != "source_rtl":
        raise ValueError("fixture backend must describe a source_rtl source")
    source = record.get("source")
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise ValueError("fixture backend source descriptor is invalid")
    relative = _safe_relative(source.get("path"), "fixture.backend.source.path")
    source_path = _resolve_under(root, relative, "fixture backend source")
    expected = _digest(source.get("sha256"), "fixture.backend.source.sha256")
    actual = sha256_file(source_path)
    if actual != expected:
        raise ValueError("fixture backend source SHA256 mismatch")
    return source_path, {"path": relative.as_posix(), "sha256": actual}


def validate_fixture(
    manifest_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"validation fixture manifest not found: {manifest_path}"
        )
    root = Path(root or manifest_path.parent).resolve()
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("validation fixture manifest is not valid JSON") from exc
    if not isinstance(record, Mapping):
        raise ValueError("validation fixture manifest must be a JSON object")
    required = {
        "schema_version",
        "bundle_kind",
        "claim_eligible",
        "synthetic_fixture",
        "not_for_release",
        "purpose",
        "backend",
        "clock_mhz",
        "samples",
    }
    if set(record) != required:
        raise ValueError("validation fixture manifest has an invalid field set")
    if record.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError("validation fixture schema version is invalid")
    if record.get("bundle_kind") != FIXTURE_KIND:
        raise ValueError("validation fixture bundle_kind is invalid")
    expected_markers = {
        "claim_eligible": False,
        "synthetic_fixture": True,
        "not_for_release": True,
    }
    for field, expected in expected_markers.items():
        if record.get(field) is not expected:
            raise ValueError(f"validation fixture marker {field} is invalid")
    if record.get("purpose") != "interface_validation_only":
        raise ValueError("validation fixture purpose is invalid")
    _positive_int(record.get("clock_mhz"), "fixture.clock_mhz")
    _source_path, source_descriptor = _validate_fixture_backend(record["backend"], root)
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("validation fixture samples must be a non-empty list")
    seen: set[tuple[str, str, int]] = set()
    for ordinal, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"fixture samples[{ordinal}] must be an object")
        required_sample = {
            "model",
            "dataset",
            "sample_index",
            "trace",
            "variants",
            "combined_stage_cycles",
        }
        if set(sample) != required_sample:
            raise ValueError(f"fixture samples[{ordinal}] has an invalid field set")
        model, dataset, sample_index = (
            sample.get("model"),
            sample.get("dataset"),
            sample.get("sample_index"),
        )
        if model not in MODELS or dataset not in DATASETS:
            raise ValueError(f"fixture samples[{ordinal}] model/dataset is invalid")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
        ):
            raise ValueError(f"fixture samples[{ordinal}].sample_index is invalid")
        key = (model, dataset, sample_index)
        if key in seen:
            raise ValueError(
                f"duplicate fixture sample: {model}/{dataset}/{sample_index}"
            )
        seen.add(key)
        trace = sample.get("trace")
        if not isinstance(trace, Mapping) or set(trace) != {"path", "sha256"}:
            raise ValueError(f"fixture samples[{ordinal}].trace is invalid")
        trace_relative = _safe_relative(
            trace.get("path"), f"fixture samples[{ordinal}].trace.path"
        )
        if trace_relative.parts[0] != TIMING_TRACE_ROOT:
            raise ValueError("fixture trace must be under timing-trace/")
        trace_path = _resolve_under(
            root, trace_relative, f"fixture samples[{ordinal}] trace"
        )
        if sha256_file(trace_path) != _digest(
            trace.get("sha256"), "fixture trace sha256"
        ):
            raise ValueError(f"fixture samples[{ordinal}] trace SHA256 mismatch")
        try:
            trace_record = json.loads(trace_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"fixture samples[{ordinal}] trace is not valid JSON"
            ) from exc
        if (
            not isinstance(trace_record, Mapping)
            or trace_record.get("synthetic_fixture") is not True
        ):
            raise ValueError("fixture trace is not marked synthetic")
        if trace_record.get("cycle_accurate") is not False:
            raise ValueError("fixture trace must remain diagnostic-only")
        if {
            trace_record.get("model"),
            trace_record.get("dataset"),
            trace_record.get("sample_index"),
        } != {model, dataset, sample_index}:
            raise ValueError("fixture trace identity does not match its sample")
        variants = sample.get("variants")
        if not isinstance(variants, Mapping) or set(variants) != set(VARIANTS):
            raise ValueError(
                f"fixture samples[{ordinal}] must cover all timing variants"
            )
        for variant in VARIANTS:
            item = variants[variant]
            if not isinstance(item, Mapping) or set(item) != {"total_cycles"}:
                raise ValueError(f"fixture variant {variant} is invalid")
            _positive_int(
                item.get("total_cycles"), f"fixture variant {variant}.total_cycles"
            )
        stages = sample.get("combined_stage_cycles")
        if not isinstance(stages, Mapping) or set(stages) != set(STAGES):
            raise ValueError("fixture combined stage cycles are incomplete")
        for stage in STAGES:
            _positive_int(stages.get(stage), f"fixture combined_stage_cycles.{stage}")
        if sum(stages.values()) != variants["asic"]["total_cycles"]:
            raise ValueError("fixture ASIC stage cycles do not match total_cycles")
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "bundle_kind": FIXTURE_KIND,
        "claim_eligible": False,
        "sample_count": len(samples),
        "manifest_sha256": sha256_file(manifest_path),
        "source": source_descriptor,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate a non-claim timing backend fixture"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="create the deterministic fixture")
    build.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate an existing fixture")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_fixture(args.output_dir)
            summary = validate_fixture(manifest, root=manifest.parent)
            print(
                "PASS: built non-claim validation fixture "
                f"({summary['sample_count']} sample, claim_eligible=false) at {manifest}"
            )
        else:
            summary = validate_fixture(args.manifest, root=args.root)
            print(
                "PASS: validated non-claim validation fixture "
                f"({summary['sample_count']} sample, claim_eligible=false)"
            )
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
