"""Load and stage source-bound RTL timing evidence for claim runs.

The claim runner deliberately accepts timing only through this manifest.  The
manifest is a binding between the emitted RTL source, raw timing traces, and
the exact model/dataset/sample workload.  No analytic or diagnostic estimate
can satisfy this interface.
"""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rtl_payload import CLAIM_CRITICAL_ROLES, PAYLOAD_AXI_BASE, descriptor_sha256

SCHEMA_VERSION = "source-bound-timing-backend-v1"
BACKEND_KIND = "source_rtl"
TIMING_TRACE_ROOT = "timing-trace"
VARIANTS = ("asic", "asic_fsdr", "asic_saes", "asic_fsdr_saes")
STAGES = ("s1", "s2", "s3", "s4")
MODELS = ("transplat", "mvsplat", "depthsplat")
DATASETS = ("re10k", "acid", "dl3dv")
SHA256 = re.compile(r"[0-9a-f]{64}")
TRACE_SCHEMA = "source-bound-timing-trace-v2"
MECHANISM_COUNTERS = (
    "fsdr_narrow",
    "fsdr_full",
    "saes_l0",
    "saes_l1",
    "saes_full",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_synthetic_trace(path: Path, label: str) -> None:
    """Keep an explicit validation fixture out of the formal claim path."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(payload, Mapping):
        return
    if (
        payload.get("synthetic_fixture") is True
        or payload.get("claim_eligible") is False
    ):
        raise ValueError(f"{label} is marked synthetic/non-claim")


def _validate_trace_payload(
    path: Path,
    *,
    label: str,
    model: str,
    dataset: str,
    sample_index: int,
    source_sha256: str,
    clock_mhz: int,
    stage_cycles: Mapping[str, int],
) -> None:
    """Validate the raw trace contract, not only its enclosing file hash."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    required = {
        "schema_version",
        "model",
        "dataset",
        "sample_index",
        "cycle_accurate",
        "source_rtl_sha256",
        "clock_mhz",
        "events",
        "stimulus_sha256",
        "stimulus_size_bytes",
        "workload",
        "axi_read_count",
        "axi_read_addresses",
        "role_coverage",
        "axi_consumption_digest",
        "mechanism_counters",
    }
    optional = {
        "applied_registers",
        "run_selection_sha256",
    }
    if not required.issubset(payload) or set(payload) - required - optional:
        raise ValueError(f"{label} has an invalid field set")
    if payload.get("schema_version") != TRACE_SCHEMA:
        raise ValueError(f"{label} has an invalid schema version")
    if (
        payload.get("model") != model
        or payload.get("dataset") != dataset
        or payload.get("sample_index") != sample_index
    ):
        raise ValueError(f"{label} identity does not match its manifest entry")
    if payload.get("cycle_accurate") is not True:
        raise ValueError(f"{label} is not marked cycle_accurate")
    if payload.get("source_rtl_sha256") != source_sha256:
        raise ValueError(f"{label} source RTL SHA256 does not match the manifest")
    if payload.get("clock_mhz") != clock_mhz:
        raise ValueError(f"{label} clock does not match the manifest")
    stimulus_digest = payload.get("stimulus_sha256")
    stimulus_size = payload.get("stimulus_size_bytes")
    if (
        not isinstance(stimulus_digest, str)
        or not SHA256.fullmatch(stimulus_digest)
        or isinstance(stimulus_size, bool)
        or not isinstance(stimulus_size, int)
        or stimulus_size <= 0
    ):
        raise ValueError(f"{label} stimulus binding is incomplete")
    workload_value = payload.get("workload")
    if "workload" in payload:
        workload = workload_value
        if not isinstance(workload, Mapping) or workload.get("schema_version") not in {
            "scarf-rtl-workload-v1",
            "scarf-rtl-workload-v2",
        }:
            raise ValueError(f"{label} workload descriptor is invalid")
        if descriptor_sha256(workload) != workload.get("descriptor_sha256"):
            raise ValueError(f"{label} workload descriptor hash is invalid")
        for field in (
            "image_h", "image_w", "feature_dim", "num_depth_candidates",
            "num_gaussians", "view_count", "tensor_count", "payload_size_bytes",
            "data_offset_bytes", "tensor_data_size_bytes", "primitives_per_pixel",
        ):
            value = workload.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} workload.{field} must be positive")
        digest = workload.get("tensor_data_sha256")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise ValueError(f"{label} workload tensor_data_sha256 is invalid")
        ranges = workload.get("tensor_ranges")
        if not isinstance(ranges, list) or len(ranges) != workload["tensor_count"]:
            raise ValueError(f"{label} workload tensor ranges are incomplete")
        expected_offset = 0
        for ordinal, item in enumerate(ranges):
            expected_range_fields = {
                "name", "dtype", "shape", "offset_bytes", "size_bytes", "end_bytes"
            }
            if workload.get("schema_version") == "scarf-rtl-workload-v2":
                expected_range_fields |= {"layout", "role"}
            if not isinstance(item, Mapping) or set(item) != expected_range_fields:
                raise ValueError(f"{label} workload tensor range {ordinal} is invalid")
            if not isinstance(item["name"], str) or not item["name"]:
                raise ValueError(f"{label} workload tensor range name is invalid")
            offset = item["offset_bytes"]
            size = item["size_bytes"]
            end = item["end_bytes"]
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (offset, size, end)
            ) or size <= 0 or offset != expected_offset or end != offset + size:
                raise ValueError(f"{label} workload tensor ranges are not contiguous")
            expected_offset = end
            if workload.get("schema_version") == "scarf-rtl-workload-v2":
                if not isinstance(item["layout"], str) or not item["layout"]:
                    raise ValueError(f"{label} workload tensor layout is invalid")
                if not isinstance(item["role"], str) or not item["role"]:
                    raise ValueError(f"{label} workload tensor role is invalid")
        if expected_offset != workload["tensor_data_size_bytes"]:
            raise ValueError(f"{label} workload ranges do not cover tensor data")
        if workload["data_offset_bytes"] + expected_offset != workload["payload_size_bytes"]:
            raise ValueError(f"{label} workload ranges do not cover payload size")
    if ("axi_read_count" in payload) != ("axi_read_addresses" in payload):
        raise ValueError(f"{label} AXI read binding is incomplete")
    if "axi_read_count" in payload:
        count = payload.get("axi_read_count")
        addresses = payload.get("axi_read_addresses")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{label} AXI read count is invalid")
        if not isinstance(addresses, list) or len(addresses) != count:
            raise ValueError(f"{label} AXI read address log is incomplete")
        if not isinstance(workload_value, Mapping):
            raise ValueError(f"{label} AXI reads require a workload descriptor")
        payload_size = workload_value["tensor_data_size_bytes"]
        for address in addresses:
            if (
                isinstance(address, bool)
                or not isinstance(address, int)
                or address < PAYLOAD_AXI_BASE
                or address >= PAYLOAD_AXI_BASE + payload_size
            ):
                raise ValueError(f"{label} AXI read address is outside the tensor-data window")
    coverage = payload.get("role_coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != set(CLAIM_CRITICAL_ROLES):
        raise ValueError(f"{label} role coverage is incomplete")
    for role, evidence in coverage.items():
        if not isinstance(evidence, Mapping) or set(evidence) != {"bytes_read", "range_count"}:
            raise ValueError(f"{label} role coverage is invalid for {role}")
        _positive_int(evidence.get("bytes_read"), f"{label}.role_coverage.{role}.bytes_read")
        _positive_int(evidence.get("range_count"), f"{label}.role_coverage.{role}.range_count")
    _digest(payload.get("axi_consumption_digest"), f"{label}.axi_consumption_digest")
    counters = payload.get("mechanism_counters")
    if not isinstance(counters, Mapping) or set(counters) != set(VARIANTS):
        raise ValueError(f"{label} mechanism counters are incomplete")
    for variant, values in counters.items():
        if not isinstance(values, Mapping) or set(values) != set(MECHANISM_COUNTERS):
            raise ValueError(f"{label} mechanism counters are invalid for {variant}")
        for counter, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} mechanism counter {variant}.{counter} is invalid")
    if "run_selection_sha256" in payload:
        _digest(payload.get("run_selection_sha256"), f"{label}.run_selection_sha256")
    if "applied_registers" in payload:
        registers = payload.get("applied_registers")
        if not isinstance(registers, Mapping) or set(registers) != {
            "asic", "asic_fsdr", "asic_saes", "asic_fsdr_saes"
        }:
            raise ValueError(f"{label} applied register map is incomplete")
        required_registers = {
            "0x00_num_depth_candidates", "0x04_feature_dim", "0x08_image_h",
            "0x0c_image_w", "0x34_saes_enabled", "0x38_fsdr_enabled",
            "0x44_config_valid", "0x4c_payload_base", "0x50_payload_bytes",
            "0x54_payload_tensor_count", "0x58_num_gaussians",
            "0x60_payload_valid", "0x64_payload_feature_offset",
            "0x68_payload_depth_offset", "0x6c_payload_probability_offset",
            "0x70_payload_candidate_offset", "0x74_payload_feature_bytes",
            "0x78_payload_depth_bytes", "0x7c_payload_candidate_bytes",
            "0x80_payload_probability_bytes",
        }
        for variant, applied in registers.items():
            if not isinstance(applied, Mapping) or not required_registers.issubset(applied):
                raise ValueError(f"{label} applied register map is incomplete for {variant}")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != len(STAGES):
        raise ValueError(f"{label} must contain one event for every stage")
    expected_start = 0
    seen: set[str] = set()
    for ordinal, event in enumerate(events):
        if not isinstance(event, Mapping) or set(event) != {
            "stage",
            "start_cycle",
            "end_cycle",
            "accepted",
        }:
            raise ValueError(f"{label} event {ordinal} is invalid")
        stage = event.get("stage")
        if stage not in STAGES or stage in seen:
            raise ValueError(f"{label} has duplicate or unknown stage event")
        seen.add(stage)
        start = event.get("start_cycle")
        end = event.get("end_cycle")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != expected_start
            or end <= start
        ):
            raise ValueError(f"{label} event {stage} has invalid cycle bounds")
        if event.get("accepted") is not True:
            raise ValueError(f"{label} event {stage} was not accepted")
        if end - start != stage_cycles[stage]:
            raise ValueError(
                f"{label} event {stage} duration does not match combined stage cycles"
            )
        expected_start = end
    if seen != set(STAGES):
        raise ValueError(f"{label} does not cover all stages")


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be a normalized relative path")
    if path.as_posix() != value:
        raise ValueError(f"{label} is not normalized")
    return path


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA256 digest")
    return value


def _resolve_under(root: Path, relative: PurePosixPath, label: str) -> Path:
    candidate = (root / Path(*relative.parts)).resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{label} escapes the backend root")
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} not found: {relative}")
    return candidate


@dataclass(frozen=True)
class ClaimTimingBackend:
    """Validated timing backend and its per-sample source-bound records."""

    manifest_path: Path
    manifest_sha256: str
    root: Path
    backend_provenance: dict[str, Any]
    source_path: Path
    source_relative: PurePosixPath
    clock_mhz: int
    samples: dict[tuple[str, str, int], dict[str, Any]]
    trace_paths: dict[tuple[str, str, int], Path]

    def sample(self, model: str, dataset: str, sample_index: int) -> dict[str, Any]:
        key = (model, dataset, sample_index)
        try:
            return copy.deepcopy(self.samples[key])
        except KeyError as exc:
            raise ValueError(
                "timing backend has no entry for "
                f"{model}/{dataset} sample_index={sample_index}"
            ) from exc

    def stage_trace(
        self, entry: Mapping[str, Any], output_dir: Path
    ) -> dict[str, Any]:
        """Copy one validated trace under the result's portable timing path."""
        self.stage_bundle(output_dir)
        trace = entry["trace"]
        relative = _safe_relative(trace["path"], "sample.trace.path")
        source = self.trace_paths[
            (entry["model"], entry["dataset"], entry["sample_index"])
        ]
        destination = Path(output_dir).resolve() / Path(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        actual = sha256_file(destination)
        if actual != trace["sha256"]:
            raise ValueError("staged timing trace hash changed during copy")
        return {"path": relative.as_posix(), "sha256": actual}

    def stage_bundle(self, output_dir: Path) -> None:
        """Copy the manifest, source, and all raw traces beside a result."""
        bundle_root = Path(output_dir).resolve() / "timing-backend"
        manifest_destination = bundle_root / "manifest.json"
        manifest_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.manifest_path, manifest_destination)

        source_destination = bundle_root / Path(*self.source_relative.parts)
        source_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.source_path, source_destination)
        for relative, source in self._all_trace_files():
            destination = bundle_root / Path(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def _all_trace_files(self) -> list[tuple[PurePosixPath, Path]]:
        return [
            (PurePosixPath(entry["trace"]["path"]), self.trace_paths[key])
            for key, entry in self.samples.items()
        ]


def _validate_backend(
    record: Mapping[str, Any], *, bundle_root: Path, release_root: Path
) -> dict[str, Any]:
    if set(record) != {"kind", "source"}:
        raise ValueError("timing backend must contain exactly kind and source")
    if record.get("kind") != BACKEND_KIND:
        raise ValueError(f"timing backend kind must be {BACKEND_KIND!r}")
    source = record.get("source")
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise ValueError("timing backend source descriptor is invalid")
    relative = _safe_relative(source.get("path"), "backend.source.path")
    source_path = _resolve_under(bundle_root, relative, "backend source")
    # A manifest may be passed from an arbitrary caller, but claim evidence
    # must never resolve source RTL outside the checked-out release root.
    if release_root not in source_path.parents and source_path != release_root:
        raise ValueError("backend source escapes the release root")
    expected = _digest(source.get("sha256"), "backend.source.sha256")
    actual = sha256_file(source_path)
    if actual != expected:
        raise ValueError("backend source SHA256 mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": BACKEND_KIND,
        "source": {"path": relative.as_posix(), "sha256": actual},
    }


def load_claim_timing_manifest(
    path: Path, *, root: Path = ROOT
) -> ClaimTimingBackend:
    """Read and validate one source-bound timing manifest.

    Paths in the manifest are resolved relative to the directory containing
    the manifest.  The bundle directory itself must be inside ``root`` (the
    checked-out release), so a manifest cannot silently refer to an
    author-local file.  This convention is also used by staged result bundles
    and keeps the portable ``rtl/...`` and ``timing-trace/...`` paths valid.
    """
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"claim timing manifest not found: {path}")
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"claim timing manifest is not valid JSON: {path}") from exc
    if not isinstance(record, Mapping):
        raise ValueError("claim timing manifest must be a JSON object")
    required = {"schema_version", "backend", "clock_mhz", "samples"}
    if set(record) != required:
        raise ValueError("claim timing manifest has an invalid field set")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("claim timing manifest has an invalid schema version")
    root = Path(root).resolve()
    bundle_root = manifest_path.parent.resolve()
    if bundle_root != root and root not in bundle_root.parents:
        raise ValueError("claim timing manifest must be inside the release root")
    backend = _validate_backend(
        record["backend"], bundle_root=bundle_root, release_root=root
    )
    source_relative = PurePosixPath(backend["source"]["path"])
    source_path = _resolve_under(bundle_root, source_relative, "backend source")
    clock_mhz = _positive_int(record["clock_mhz"], "clock_mhz")
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("claim timing manifest samples must be a non-empty list")

    validated: dict[tuple[str, str, int], dict[str, Any]] = {}
    trace_paths: dict[tuple[str, str, int], Path] = {}
    for ordinal, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"samples[{ordinal}] must be an object")
        required_sample = {
            "model",
            "dataset",
            "sample_index",
            "trace",
            "variants",
            "combined_stage_cycles",
        }
        optional_sample = {
            "workload",
            "axi_read_count",
            "axi_read_addresses",
            "applied_registers",
            "run_selection_sha256",
            "role_coverage",
            "axi_consumption_digest",
            "mechanism_counters",
        }
        if set(sample) - required_sample - optional_sample or not required_sample.issubset(sample):
            raise ValueError(f"samples[{ordinal}] has an invalid field set")
        model = sample.get("model")
        dataset = sample.get("dataset")
        if model not in MODELS or dataset not in DATASETS:
            raise ValueError(f"samples[{ordinal}] has an unsupported model/dataset")
        sample_index = sample.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
            raise ValueError(f"samples[{ordinal}].sample_index must be nonnegative")
        key = (model, dataset, sample_index)
        if key in validated:
            raise ValueError(f"duplicate timing sample: {model}/{dataset}/{sample_index}")

        trace = sample.get("trace")
        if not isinstance(trace, Mapping) or set(trace) != {"path", "sha256"}:
            raise ValueError(f"samples[{ordinal}].trace is invalid")
        trace_relative = _safe_relative(trace.get("path"), f"samples[{ordinal}].trace.path")
        if trace_relative.parts[0] != TIMING_TRACE_ROOT:
            raise ValueError(
                f"samples[{ordinal}].trace.path must be under {TIMING_TRACE_ROOT}/"
            )
        trace_path = _resolve_under(
            bundle_root, trace_relative, f"samples[{ordinal}] trace"
        )
        if root not in trace_path.parents and trace_path != root:
            raise ValueError(
                f"samples[{ordinal}] trace escapes the release root"
            )
        trace_digest = _digest(trace.get("sha256"), f"samples[{ordinal}].trace.sha256")
        if sha256_file(trace_path) != trace_digest:
            raise ValueError(f"samples[{ordinal}] trace SHA256 mismatch")
        _reject_synthetic_trace(trace_path, f"samples[{ordinal}] trace")

        variants = sample.get("variants")
        if not isinstance(variants, Mapping) or set(variants) != set(VARIANTS):
            raise ValueError(f"samples[{ordinal}] must cover all timing variants")
        normalized_variants: dict[str, dict[str, int]] = {}
        for variant_name in VARIANTS:
            variant = variants[variant_name]
            if not isinstance(variant, Mapping) or set(variant) != {"total_cycles"}:
                raise ValueError(f"samples[{ordinal}] variant {variant_name} is invalid")
            normalized_variants[variant_name] = {
                "total_cycles": _positive_int(
                    variant.get("total_cycles"),
                    f"samples[{ordinal}].variants.{variant_name}.total_cycles",
                )
            }
        combined_cycles = normalized_variants["asic_fsdr_saes"]["total_cycles"]
        if (
            combined_cycles >= normalized_variants["asic_fsdr"]["total_cycles"]
            or combined_cycles >= normalized_variants["asic_saes"]["total_cycles"]
        ):
            raise ValueError(
                "combined mechanism variant must be strictly faster than each single-mechanism variant"
            )

        stages = sample.get("combined_stage_cycles")
        if not isinstance(stages, Mapping) or set(stages) != set(STAGES):
            raise ValueError(f"samples[{ordinal}] combined stage cycles are incomplete")
        normalized_stages = {
            stage: _positive_int(
                stages.get(stage),
                f"samples[{ordinal}].combined_stage_cycles.{stage}",
            )
            for stage in STAGES
        }
        _validate_trace_payload(
            trace_path,
            label=f"samples[{ordinal}] trace",
            model=model,
            dataset=dataset,
            sample_index=sample_index,
            source_sha256=backend["source"]["sha256"],
            clock_mhz=clock_mhz,
            stage_cycles=normalized_stages,
        )
        trace_value = json.loads(trace_path.read_text(encoding="utf-8"))
        for field in optional_sample:
            if field in sample and trace_value.get(field) != sample[field]:
                raise ValueError(
                    f"samples[{ordinal}] {field} does not match its raw trace"
                )
            if field in trace_value and field not in sample:
                raise ValueError(
                    f"samples[{ordinal}] raw trace has unbound {field} evidence"
                )
        normalized = {
            "model": model,
            "dataset": dataset,
            "sample_index": sample_index,
            "trace": {"path": trace_relative.as_posix(), "sha256": trace_digest},
            "variants": normalized_variants,
            "combined_stage_cycles": normalized_stages,
        }
        for field in optional_sample:
            if field in sample:
                normalized[field] = copy.deepcopy(sample[field])
        validated[key] = normalized
        trace_paths[key] = trace_path

    return ClaimTimingBackend(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        root=root,
        backend_provenance=backend,
        source_path=source_path,
        source_relative=source_relative,
        clock_mhz=clock_mhz,
        samples=validated,
        trace_paths=trace_paths,
    )


def claim_timing_for_sample(
    backend: ClaimTimingBackend,
    model: str,
    dataset: str,
    sample_index: int,
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return result-record timing and backend provenance for one sample."""
    entry = backend.sample(model, dataset, sample_index)
    trace = backend.stage_trace(entry, output_dir)
    timing = {
        "schema_version": "source-bound-claim-timing-v1",
        "timing_class": "rtl_cycle_equivalent_source_bound",
        "rtl_cycle_equivalent": True,
        "clock_mhz": backend.clock_mhz,
        "trace": trace,
        "variants": copy.deepcopy(entry["variants"]),
        "combined_stage_cycles": copy.deepcopy(entry["combined_stage_cycles"]),
    }
    backend.stage_bundle(output_dir)
    provenance = {
        "schema_version": "source-bound-timing-backend-v1",
        "kind": "source_rtl",
        "source": {
            "path": "timing-backend/" + backend.source_relative.as_posix(),
            "sha256": backend.backend_provenance["source"]["sha256"],
        },
        "manifest": {
            "path": "timing-backend/manifest.json",
            "sha256": backend.manifest_sha256,
        },
    }
    return timing, provenance


def manifest_display_path(path: Path, root: Path = ROOT) -> str:
    """Return a portable manifest path when it belongs to the checkout."""
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a source-bound SCARF claim timing manifest"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Release root used to resolve RTL source and timing-trace paths",
    )
    args = parser.parse_args()
    try:
        backend = load_claim_timing_manifest(args.manifest, root=args.root)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        "PASS: "
        f"{len(backend.samples)} timing samples, "
        f"clock={backend.clock_mhz} MHz, "
        f"source={backend.backend_provenance['source']['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
