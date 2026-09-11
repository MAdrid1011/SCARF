"""Build deterministic, schema-valid SCARF experiment records."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from scripts.execution_contract import (
    ASSIGNMENT_CONSENSUS_PSEUDO_DESCRIPTOR_MATERIALIZATION,
    REPRESENTATIVE_SAES_MATERIALIZATION,
    build_execution_contract,
)
from scripts.mechanism_config import load_mechanism_config


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("feature", "depth", "gaussian", "ggu")
STAGE_COMPONENTS = {
    "s1": "feature",
    "s2": "depth",
    "s3": "gaussian",
    "s4": "ggu",
}
EXECUTION_TRACE_SCHEMA_VERSION = "execution-trace-inputs-v2"
EXECUTION_TRACE_SET_SCHEMA_VERSION = "execution-trace-set-v2"
CLAIM_TIMING_SCHEMA_VERSION = "source-bound-claim-timing-v1"
CLAIM_TIMING_CLASS = "rtl_cycle_equivalent_source_bound"
CLAIM_TIMING_CYCLE_SOURCE = "source_bound_verified_timing_trace_v1"
CLAIM_TIMING_VARIANTS = (
    "asic",
    "asic_fsdr",
    "asic_saes",
    "asic_fsdr_saes",
)

# The execution trace deliberately signs only target-free, non-transient
# evidence.  These names are removed even when they occur inside an otherwise
# useful runtime ledger, such as an ablation record that also carries renders.
_TRACE_QUALITY_KEYS = frozenset(
    {
        "quality",
        "quality_views",
        "psnr",
        "psnr_db",
        "ssim",
        "lpips",
        "loss_db",
        "loss_pct",
        "ground_truth",
        "ground_truth_rgb",
        "target_rgb",
        "gt",
        "gt_rgb",
    }
)
_TRACE_PATH_OR_TRANSIENT_KEYS = frozenset(
    {
        "path",
        "paths",
        "command",
        "commands",
        "output",
        "outputs",
        "output_dir",
        "output_path",
        "results_path",
        "audit_entrypoint",
        "entrypoint",
        "device",
        "device_name",
        "measured_encoder_time_ms",
        "encoder_timing_samples_ms",
        "encoder_timing_repetitions",
        "elapsed_seconds",
        "started_at",
    }
)
_TRACE_OMIT = object()


def _canonical_json_sha256(value: Any, *, label: str) -> str:
    """Hash JSON with a fixed encoding instead of a display serialization."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _trace_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object for execution trace binding")
    return value


def _trace_required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label}.{key} is required for execution trace binding")
    return mapping[key]


def canonical_execution_selection(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the non-quality sample selection used in an execution trace."""
    if evaluation.get("kind") != "sample":
        raise ValueError("execution trace binding requires a sample evaluation")
    sample_index = evaluation.get("sample_index")
    execution_index = evaluation.get("execution_index")
    candidate_count = evaluation.get("candidate_count")
    scene = evaluation.get("scene")
    context_indices = evaluation.get("context_indices")
    target_indices = evaluation.get("target_indices")
    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
        or isinstance(execution_index, bool)
        or not isinstance(execution_index, int)
        or execution_index < 0
        or isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
        or execution_index >= candidate_count
        or not isinstance(scene, str)
        or not scene
        or not isinstance(context_indices, list)
        or not context_indices
        or not isinstance(target_indices, list)
        or not target_indices
        or any(isinstance(value, bool) or not isinstance(value, int) for value in context_indices)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in target_indices)
    ):
        raise ValueError("execution trace binding has an invalid canonical selection")
    return {
        "sample_index": sample_index,
        "execution_index": execution_index,
        "candidate_count": candidate_count,
        "scene": scene,
        "context_indices": list(context_indices),
        "target_indices": list(target_indices),
    }


def _canonical_runtime_asset_identities(
    runtime_assets: Any,
) -> dict[str, dict[str, Any]]:
    """Keep content identities while excluding installation-specific asset paths."""
    if not isinstance(runtime_assets, Mapping) or not runtime_assets:
        raise ValueError("runtime assets must be non-empty for execution trace binding")
    identities: dict[str, dict[str, Any]] = {}
    fields = ("sha256", "archive_sha256", "commit", "license_sha256")
    for name, metadata in runtime_assets.items():
        if not isinstance(name, str) or not name or not isinstance(metadata, Mapping):
            raise ValueError("runtime asset identity is invalid for execution trace binding")
        identity = {
            field: copy.deepcopy(metadata[field]) for field in fields if field in metadata
        }
        if not any(field in identity for field in ("sha256", "archive_sha256")):
            raise ValueError("runtime asset has no content hash for execution trace binding")
        identities[name] = identity
    return identities


def _sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def claim_timing_binding_inputs(
    record: Mapping[str, Any], *, aggregate: bool
) -> dict[str, Any]:
    """Return the source and workload identity a claim timing trace must bind."""
    root = _trace_mapping(record, "result record")
    provenance = _trace_mapping(
        _trace_required(root, "provenance", "result record"), "provenance"
    )
    dataset = _trace_mapping(
        _trace_required(provenance, "dataset", "provenance"), "provenance.dataset"
    )
    checkpoint = _trace_mapping(
        _trace_required(provenance, "checkpoint", "provenance"),
        "provenance.checkpoint",
    )
    evaluation = _trace_mapping(
        _trace_required(provenance, "evaluation", "provenance"),
        "provenance.evaluation",
    )
    execution = _trace_safe_nonquality_value(
        _trace_required(provenance, "execution_contract", "provenance"),
        label="provenance.execution_contract",
    )
    if aggregate:
        if evaluation.get("kind") != "dataset_aggregate":
            raise ValueError("claim timing aggregate requires a dataset aggregate")
        selection = {
            "sample_count": evaluation.get("sample_count"),
            "sample_indices": evaluation.get("sample_indices"),
            "execution_indices": evaluation.get("execution_indices"),
            "sample_selection_sha256": evaluation.get("sample_selection_sha256"),
        }
    else:
        selection = canonical_execution_selection(evaluation)
    return {
        "source": {
            "source_tree_sha256": _trace_required(
                provenance, "source_tree_sha256", "provenance"
            ),
            "submodules": _trace_required(provenance, "submodules", "provenance"),
        },
        "mechanism_config_sha256": provenance.get("mechanism_config_sha256"),
        "saes_execution_identity": _trace_required(
            provenance, "saes_execution_identity", "provenance"
        ),
        "saes_execution_route_sha256": _trace_required(
            provenance, "saes_execution_route_sha256", "provenance"
        ),
        "model": _trace_required(provenance, "model", "provenance"),
        "dataset": {
            "name": _trace_required(dataset, "name", "provenance.dataset"),
            "representation": _trace_required(
                dataset, "representation", "provenance.dataset"
            ),
            "manifest_sha256": _trace_required(dataset, "sha256", "provenance.dataset"),
            "tree_sha256": _trace_required(
                dataset, "tree_sha256", "provenance.dataset"
            ),
        },
        "checkpoint_sha256": _trace_required(
            checkpoint, "sha256", "provenance.checkpoint"
        ),
        "runtime_assets": _canonical_runtime_asset_identities(
            _trace_required(provenance, "runtime_assets", "provenance")
        ),
        "seed": _trace_required(provenance, "seed", "provenance"),
        "environment_digest_sha256": _trace_required(
            _trace_mapping(
                _trace_required(provenance, "environment", "provenance"),
                "provenance.environment",
            ),
            "digest_sha256",
            "provenance.environment",
        ),
        "execution_contract": execution,
        "selection": selection,
    }


def claim_timing_binding_sha256(
    record: Mapping[str, Any], *, aggregate: bool
) -> str:
    """Hash the non-quality source and workload inputs for a claim timing trace."""
    return _canonical_json_sha256(
        {
            "schema_version": CLAIM_TIMING_SCHEMA_VERSION,
            "inputs": claim_timing_binding_inputs(record, aggregate=aggregate),
        },
        label="claim timing input binding",
    )


def claim_timing_from_record(
    record: Mapping[str, Any], *, aggregate: bool
) -> dict[str, Any]:
    """Validate the only timing evidence allowed to support a strict claim."""
    root = _trace_mapping(record, "result record")
    provenance = _trace_mapping(
        _trace_required(root, "provenance", "result record"), "provenance"
    )
    performance = _trace_mapping(
        _trace_required(root, "performance", "result record"), "performance"
    )
    backend = provenance.get("timing_backend")
    if backend is not None:
        if not isinstance(backend, Mapping) or set(backend) != {
            "schema_version",
            "kind",
            "source",
            "manifest",
        }:
            raise ValueError("claim timing backend provenance is invalid")
        if backend.get("schema_version") != "source-bound-timing-backend-v1":
            raise ValueError("claim timing backend provenance has an invalid schema")
        if backend.get("kind") != "source_rtl":
            raise ValueError("claim timing backend provenance has an invalid kind")
        for label in ("source", "manifest"):
            descriptor = backend.get(label)
            if (
                not isinstance(descriptor, Mapping)
                or set(descriptor) != {"path", "sha256"}
                or not isinstance(descriptor.get("path"), str)
                or not _sha256_digest(descriptor.get("sha256"))
            ):
                raise ValueError(f"claim timing backend {label} descriptor is invalid")
    timing = performance.get("claim_timing")
    if not isinstance(timing, Mapping):
        raise ValueError("claim result has no source-bound timing evidence")
    required = {
        "schema_version",
        "timing_class",
        "rtl_cycle_equivalent",
        "clock_mhz",
        "trace",
        "input_binding_sha256",
        "variants",
        "combined_stage_cycles",
    }
    if set(timing) != required:
        raise ValueError("claim timing evidence has an invalid field set")
    if timing["schema_version"] != CLAIM_TIMING_SCHEMA_VERSION:
        raise ValueError("claim timing evidence has an invalid schema version")
    if timing["timing_class"] != CLAIM_TIMING_CLASS or timing[
        "rtl_cycle_equivalent"
    ] is not True:
        raise ValueError("claim timing evidence is not RTL-cycle-equivalent")
    _positive_integer(timing["clock_mhz"], "claim timing clock_mhz")
    trace = timing["trace"]
    if not isinstance(trace, Mapping) or set(trace) != {"path", "sha256"}:
        raise ValueError("claim timing evidence has no raw trace descriptor")
    trace_path = trace["path"]
    pure_trace_path = PurePosixPath(trace_path) if isinstance(trace_path, str) else None
    if (
        pure_trace_path is None
        or pure_trace_path.is_absolute()
        or ".." in pure_trace_path.parts
        or not pure_trace_path.parts
        or pure_trace_path.parts[0] != "timing-trace"
        or not _sha256_digest(trace["sha256"])
    ):
        raise ValueError("claim timing evidence has an invalid raw trace descriptor")
    expected_binding = claim_timing_binding_sha256(record, aggregate=aggregate)
    if timing["input_binding_sha256"] != expected_binding:
        raise ValueError("claim timing input binding does not match the result")
    variants = timing["variants"]
    if not isinstance(variants, Mapping) or set(variants) != set(CLAIM_TIMING_VARIANTS):
        raise ValueError("claim timing evidence does not cover every ablation variant")
    variant_cycles: dict[str, int] = {}
    for name in CLAIM_TIMING_VARIANTS:
        variant = variants[name]
        if not isinstance(variant, Mapping) or set(variant) != {"total_cycles"}:
            raise ValueError(f"claim timing variant {name} is invalid")
        variant_cycles[name] = _positive_integer(
            variant["total_cycles"], f"claim timing variant {name}.total_cycles"
        )
    if performance.get("cycle_source") != CLAIM_TIMING_CYCLE_SOURCE:
        raise ValueError("claim result does not use the source-bound timing source")
    if performance.get("scarf_cycles") != variant_cycles["asic_fsdr_saes"]:
        raise ValueError("claim timing combined cycles do not match performance.scarf_cycles")
    stages = performance.get("stages")
    combined_stages = timing["combined_stage_cycles"]
    if not isinstance(stages, Mapping) or not isinstance(combined_stages, Mapping):
        raise ValueError("claim timing has no complete stage evidence")
    if set(combined_stages) != {"s1", "s2", "s3", "s4"}:
        raise ValueError("claim timing combined stage cycles are incomplete")
    for stage in ("s1", "s2", "s3", "s4"):
        cycles = _positive_integer(
            combined_stages[stage], f"claim timing {stage} cycles"
        )
        if not isinstance(stages.get(stage), Mapping) or stages[stage].get("cycles") != cycles:
            raise ValueError("claim timing stage cycles do not match performance stages")
    ablation = root.get("ablation")
    if not isinstance(ablation, Mapping):
        raise ValueError("claim result has no ablation record")
    if any(
        isinstance(variant, Mapping) and "eff_total" in variant
        for variant in ablation.values()
    ):
        raise ValueError("claim result contains analytic eff_total timing")
    return {
        "clock_mhz": timing["clock_mhz"],
        "trace": dict(trace),
        "input_binding_sha256": timing["input_binding_sha256"],
        "variants": variant_cycles,
        "combined_stage_cycles": dict(combined_stages),
    }


def claim_ablation_cycles(record: Mapping[str, Any]) -> dict[str, int]:
    """Return strict ablation cycles without consulting analytic estimates."""
    provenance = _trace_mapping(
        _trace_required(record, "provenance", "result record"), "provenance"
    )
    evaluation = _trace_mapping(
        _trace_required(provenance, "evaluation", "provenance"), "provenance.evaluation"
    )
    timing = claim_timing_from_record(
        record, aggregate=evaluation.get("kind") == "dataset_aggregate"
    )
    return dict(timing["variants"])


def claim_ablation_speedups(record: Mapping[str, Any]) -> dict[str, float]:
    """Compute strict Figure 11 ratios from verified timing trace variants."""
    cycles = claim_ablation_cycles(record)
    baseline = cycles["asic"]
    return {
        "fsdr": baseline / cycles["asic_fsdr"],
        "saes": baseline / cycles["asic_saes"],
        "combined": baseline / cycles["asic_fsdr_saes"],
    }


def _trace_key_is_excluded(key: str) -> bool:
    """Return whether a nested ledger key is disallowed in a trace payload."""
    lowered = key.lower()
    if lowered in _TRACE_QUALITY_KEYS or lowered in _TRACE_PATH_OR_TRANSIENT_KEYS:
        return True
    if "quality" in lowered:
        return True
    if "ground_truth" in lowered:
        return True
    if lowered.startswith("gt_") or lowered.endswith("_gt"):
        return True
    if lowered.startswith("target_") or lowered.startswith("reference_"):
        return True
    if lowered.endswith("_path") or lowered.endswith("_paths"):
        return True
    if "command" in lowered or "output" in lowered:
        return True
    return lowered.startswith("device_") or lowered.endswith("_device")


def _trace_safe_nonquality_value(value: Any, *, label: str) -> Any:
    """Copy a target-free ledger while dropping quality, paths, and timing."""
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} has an invalid trace key")
            if _trace_key_is_excluded(key):
                continue
            sanitized = _trace_safe_nonquality_value(
                child, label=f"{label}.{key}"
            )
            if sanitized is not _TRACE_OMIT:
                copied[key] = sanitized
        return copied
    if isinstance(value, list):
        copied_items = []
        for index, child in enumerate(value):
            sanitized = _trace_safe_nonquality_value(
                child, label=f"{label}[{index}]"
            )
            if sanitized is not _TRACE_OMIT:
                copied_items.append(sanitized)
        return copied_items
    if isinstance(value, str):
        # Keys cover repository-relative paths.  Keep ordinary technical prose
        # such as ``S2/S3`` in ledger assumptions, but omit unmistakable local
        # paths if one is smuggled through an unfamiliar field name.
        if value.startswith(("/", "./", "../", "~")) or "\\" in value:
            return _TRACE_OMIT
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return copy.deepcopy(value)
    raise ValueError(f"{label} has a non-JSON trace value")


def _claimable_mechanism_metrics(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep the target-free mechanism metrics consumed by claim validation."""
    fsdr_saes = record.get("fsdr_saes", {})
    if not isinstance(fsdr_saes, Mapping):
        raise ValueError("fsdr_saes must be an object for execution trace binding")

    fields = {
        "fsdr": ("guided_rate", "in_window_rate", "top1_coverage"),
        "saes": ("level0_ratio", "level1_ratio", "modification_ratio"),
        "preservation": (
            "saes_low_var_agree",
            "saes_low_var_tiles",
            "saes_early_tiles",
            "saes_low_var_mean_similarity",
            "saes_low_var_threshold",
        ),
    }
    evidence: dict[str, dict[str, Any]] = {}
    for namespace, names in fields.items():
        values = fsdr_saes.get(namespace, {})
        if not isinstance(values, Mapping):
            if values not in ({}, None):
                raise ValueError(
                    f"fsdr_saes.{namespace} must be an object for execution trace binding"
                )
            values = {}
        evidence[namespace] = {
            name: _trace_safe_nonquality_value(
                values[name], label=f"fsdr_saes.{namespace}.{name}"
            )
            for name in names
            if name in values
        }
    return evidence


def execution_trace_performance_evidence_from_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract every claimable target-free performance ledger from a record.

    Baseline CUDA timing and its derived speedup stay outside this payload:
    those transient measurements are verified by the separately hashed Orin
    evidence.  SCARF's static cycle ledger and all mechanism evidence remain
    bound here.
    """
    root = _trace_mapping(record, "result record")
    performance = _trace_mapping(
        _trace_required(root, "performance", "result record"), "performance"
    )
    evidence = {
        "cycle_source": _trace_safe_nonquality_value(
            _trace_required(performance, "cycle_source", "performance"),
            label="performance.cycle_source",
        ),
        "scarf_cycles": _trace_safe_nonquality_value(
            _trace_required(performance, "scarf_cycles", "performance"),
            label="performance.scarf_cycles",
        ),
        "components": _trace_safe_nonquality_value(
            _trace_required(performance, "components", "performance"),
            label="performance.components",
        ),
        "stages": _trace_safe_nonquality_value(
            _trace_required(performance, "stages", "performance"),
            label="performance.stages",
        ),
        "ablation": _trace_safe_nonquality_value(
            _trace_required(root, "ablation", "result record"),
            label="ablation",
        ),
        "events": _trace_safe_nonquality_value(
            _trace_required(root, "events", "result record"), label="events"
        ),
        "mechanism_metrics": _claimable_mechanism_metrics(root),
        "energy": _trace_safe_nonquality_value(
            _trace_required(root, "energy", "result record"), label="energy"
        ),
    }
    if "claim_timing" in performance:
        evidence["claim_timing"] = _trace_safe_nonquality_value(
            performance["claim_timing"], label="performance.claim_timing"
        )
    return evidence


def execution_trace_inputs_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only actual, non-quality execution inputs for a sample trace.

    The payload intentionally has an explicit allowlist.  In particular it does
    not read rendered quality, target RGB, or paper-reference values.
    """
    root = _trace_mapping(record, "result record")
    provenance = _trace_mapping(
        _trace_required(root, "provenance", "result record"), "provenance"
    )
    dataset = _trace_mapping(
        _trace_required(provenance, "dataset", "provenance"), "provenance.dataset"
    )
    checkpoint = _trace_mapping(
        _trace_required(provenance, "checkpoint", "provenance"),
        "provenance.checkpoint",
    )
    evaluation = _trace_mapping(
        _trace_required(provenance, "evaluation", "provenance"),
        "provenance.evaluation",
    )
    return {
        "execution_contract": copy.deepcopy(
            _trace_safe_nonquality_value(
                _trace_required(provenance, "execution_contract", "provenance"),
                label="provenance.execution_contract",
            )
        ),
        "source": {
            "git_commit": copy.deepcopy(
                _trace_required(provenance, "git_commit", "provenance")
            ),
            "git_dirty": copy.deepcopy(
                _trace_required(provenance, "git_dirty", "provenance")
            ),
            "source_identity": copy.deepcopy(
                _trace_required(provenance, "source_identity", "provenance")
            ),
            "source_tree_sha256": copy.deepcopy(
                _trace_required(provenance, "source_tree_sha256", "provenance")
            ),
            "submodules": copy.deepcopy(
                _trace_required(provenance, "submodules", "provenance")
            ),
        },
        "mechanism": {
            "config_sha256": copy.deepcopy(provenance.get("mechanism_config_sha256")),
            "calibration_provenance": _trace_safe_nonquality_value(
                provenance.get("calibration_provenance"),
                label="provenance.calibration_provenance",
            ),
            "saes_execution_identity": _trace_safe_nonquality_value(
                provenance.get("saes_execution_identity"),
                label="provenance.saes_execution_identity",
            ),
            "saes_execution_route_sha256": _trace_safe_nonquality_value(
                provenance.get("saes_execution_route_sha256"),
                label="provenance.saes_execution_route_sha256",
            ),
        },
        "model": copy.deepcopy(_trace_required(provenance, "model", "provenance")),
        "dataset": {
            "name": copy.deepcopy(_trace_required(dataset, "name", "provenance.dataset")),
            "representation": copy.deepcopy(
                _trace_required(dataset, "representation", "provenance.dataset")
            ),
            "manifest_sha256": copy.deepcopy(
                _trace_required(dataset, "sha256", "provenance.dataset")
            ),
            "tree_sha256": copy.deepcopy(
                _trace_required(dataset, "tree_sha256", "provenance.dataset")
            ),
        },
        "checkpoint": {
            "sha256": copy.deepcopy(
                _trace_required(checkpoint, "sha256", "provenance.checkpoint")
            ),
            "load": copy.deepcopy(
                _trace_required(checkpoint, "load", "provenance.checkpoint")
            ),
        },
        "execution": {
            "seed": copy.deepcopy(_trace_required(provenance, "seed", "provenance")),
            "environment_digest_sha256": copy.deepcopy(
                _trace_required(
                    _trace_mapping(
                        _trace_required(provenance, "environment", "provenance"),
                        "provenance.environment",
                    ),
                    "digest_sha256",
                    "provenance.environment",
                )
            ),
            "runtime_assets": _canonical_runtime_asset_identities(
                _trace_required(provenance, "runtime_assets", "provenance")
            ),
        },
        "selection": canonical_execution_selection(evaluation),
        "route_evidence": execution_trace_performance_evidence_from_record(root),
    }


def execution_trace_sha256(inputs: Mapping[str, Any]) -> str:
    """Return the digest for an explicit non-quality execution-input payload."""
    if not isinstance(inputs, Mapping):
        raise ValueError("execution trace inputs must be an object")
    return _canonical_json_sha256(
        {
            "schema_version": EXECUTION_TRACE_SCHEMA_VERSION,
            "inputs": inputs,
        },
        label="execution trace inputs",
    )


def execution_trace_set_sha256(
    trace_set: list[Mapping[str, Any]],
    performance_evidence: Mapping[str, Any],
) -> str:
    """Return the digest for sample traces and their aggregate performance evidence."""
    if not isinstance(trace_set, list):
        raise ValueError("execution trace set must be a list")
    if not isinstance(performance_evidence, Mapping):
        raise ValueError("execution trace set performance evidence must be an object")
    return _canonical_json_sha256(
        {
            "schema_version": EXECUTION_TRACE_SET_SCHEMA_VERSION,
            "traces": trace_set,
            "performance_evidence": performance_evidence,
        },
        label="execution trace set",
    )


def bind_execution_trace(record: dict[str, Any]) -> str:
    """Attach one verified execution trace to a completed sample result record."""
    provenance = record.get("provenance")
    quality = record.get("quality")
    performance = record.get("performance")
    if not isinstance(provenance, dict):
        raise ValueError("result record provenance must be an object")
    if not isinstance(quality, dict) or not isinstance(performance, dict):
        raise ValueError("result record needs quality and performance trace bindings")
    inputs = execution_trace_inputs_from_record(record)
    digest = execution_trace_sha256(inputs)
    provenance["execution_trace"] = {
        "schema_version": EXECUTION_TRACE_SCHEMA_VERSION,
        "inputs": inputs,
    }
    provenance["execution_trace_sha256"] = digest
    quality["execution_trace_sha256"] = digest
    performance["execution_trace_sha256"] = digest
    return digest


@lru_cache(maxsize=2)
def build_environment_provenance(profile: str) -> dict[str, Any]:
    if profile not in {"classic", "depthsplat"}:
        raise ValueError(f"unsupported environment profile: {profile}")
    import torch
    import torchvision

    from scripts.check_environment import PROFILE_CONTRACTS

    lock = ROOT / "environments" / profile / "requirements.lock"
    contract = PROFILE_CONTRACTS[profile]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != contract["python"]:
        raise RuntimeError(
            f"{profile} requires Python {contract['python']}, got {actual_python}"
        )
    if not torch.__version__.startswith(contract["torch"]):
        raise RuntimeError(
            f"{profile} requires PyTorch {contract['torch']}, got {torch.__version__}"
        )
    if not torchvision.__version__.startswith(contract["torchvision"]):
        raise RuntimeError(
            f"{profile} requires TorchVision {contract['torchvision']}, got {torchvision.__version__}"
        )
    if torch.version.cuda is not None and torch.version.cuda != contract["cuda"]:
        raise RuntimeError(
            f"{profile} requires CUDA {contract['cuda']}, got {torch.version.cuda}"
        )
    record = {
        "profile": profile,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "torchvision": version("torchvision"),
        "torch_cuda": torch.version.cuda,
        "lock_sha256": sha256_file(lock),
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return {**record, "digest_sha256": hashlib.sha256(canonical).hexdigest()}


def sha256_file(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required provenance file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _cached_sha256_file(path_text: str) -> str:
    return sha256_file(Path(path_text))


def cached_sha256_file(path: Path) -> str:
    path = Path(path).resolve()
    return _cached_sha256_file(str(path))


def require_positive_cycles(cycles: Mapping[str, Any]) -> dict[str, int]:
    checked: dict[str, int] = {}
    for component in COMPONENTS:
        value = cycles.get(component)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{component} cycle count is missing or non-numeric")
        if not math.isfinite(value) or value <= 0 or int(value) != value:
            raise ValueError(f"{component} cycle count must be a positive integer")
        checked[component] = int(value)
    return checked


def _nonnegative_count(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or int(value) != value
    ):
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _unit_fraction(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{label} must be a finite fraction in [0, 1]")
    return float(value)


def _default_stage_records(cycles: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    return {
        stage: {
            "cycles": int(cycles[component]),
            "useful_mmcu_slots": 0,
            "scheduled_mmcu_slots": 0,
            "mmcu_slots_available": False,
            "source": "component_simulator_without_mmcu_events",
        }
        for stage, component in STAGE_COMPONENTS.items()
    }


def _copy_stage_records(
    records: Mapping[str, Any] | None, cycles: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    if records is None:
        return _default_stage_records(cycles)
    if set(records) != set(STAGE_COMPONENTS):
        raise ValueError("stage records must contain exactly s1-s4")
    copied: dict[str, dict[str, Any]] = {}
    for stage in STAGE_COMPONENTS:
        value = records[stage]
        if not isinstance(value, Mapping):
            raise ValueError(f"stage record {stage} must be an object")
        copied[stage] = dict(value)
    return copied


def _default_event_records(fsdr_saes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fsdr_value = fsdr_saes.get("fsdr", {})
    fsdr = fsdr_value if isinstance(fsdr_value, Mapping) else {}
    total_pixels = _nonnegative_count(fsdr.get("total_pixels", 0), "FSDR total pixels")
    cache_hits = _nonnegative_count(fsdr.get("cache_hits", 0), "FSDR cache hits")
    guided_pixels = _nonnegative_count(fsdr.get("guided", 0), "FSDR guided pixels")
    covered = _nonnegative_count(
        fsdr.get("guided_top1_covered", 0), "FSDR covered Top-1 pixels"
    )
    missed = _nonnegative_count(
        fsdr.get("guided_top1_missed", 0), "FSDR missed Top-1 pixels"
    )
    discrete_top1_available = (
        fsdr.get("discrete_candidate_evidence") is True
        and covered + missed == guided_pixels
    )
    if not discrete_top1_available:
        covered = 0
        missed = 0
    depth_evaluations_available = fsdr.get("depth_evaluations_available") is True
    feature_buffer_bytes_available = (
        fsdr.get("feature_buffer_bytes_available") is True
    )

    saes_value = fsdr_saes.get("saes", {})
    saes = saes_value if isinstance(saes_value, Mapping) else {}
    total_tiles = _nonnegative_count(
        saes.get("total_tiles_processed", 0), "SAES total tiles"
    )
    level0_tiles = _nonnegative_count(saes.get("level0_tiles", 0), "SAES L0 tiles")
    level1_tiles = _nonnegative_count(saes.get("level1_tiles", 0), "SAES L1 tiles")
    full_tiles = _nonnegative_count(saes.get("full_tiles", 0), "SAES full tiles")
    tile_path_available = (
        total_tiles > 0 and level0_tiles + level1_tiles + full_tiles == total_tiles
    )
    effective_gaussians = _nonnegative_count(
        saes.get("effective_gaussians", 0), "SAES effective Gaussians"
    )
    zeroed_gaussians = _nonnegative_count(
        saes.get("zeroed_gaussians", 0), "SAES zeroed Gaussians"
    )
    gaussian_counts_available = "effective_gaussians" in saes and "zeroed_gaussians" in saes
    s2_evaluations_available = saes.get("s2_evaluations_available") is True
    full_s2_evaluations = (
        _nonnegative_count(
            saes.get("full_s2_evaluations", 0), "SAES full S2 evaluations"
        )
        if s2_evaluations_available
        else 0
    )
    executed_s2_evaluations = (
        _nonnegative_count(
            saes.get("executed_s2_evaluations", 0),
            "SAES executed S2 evaluations",
        )
        if s2_evaluations_available
        else 0
    )
    if executed_s2_evaluations > full_s2_evaluations:
        raise ValueError("SAES executed S2 evaluations exceed the full search")
    hardware_accounting = saes.get("hardware_accounting")
    if hardware_accounting is not None and not isinstance(hardware_accounting, Mapping):
        raise ValueError("SAES hardware accounting must be an object when available")
    execution_dependency = saes.get("execution_dependency")
    if execution_dependency is not None and not isinstance(execution_dependency, Mapping):
        raise ValueError("SAES execution dependency must be an object when available")
    requested_saving = saes.get("s2_s3_saving", {})
    if not isinstance(requested_saving, Mapping):
        raise ValueError("SAES S2/S3 saving must be an object when available")

    return {
        "fsdr": {
            "total_pixels": total_pixels,
            "cache_hits": cache_hits,
            "guided_pixels": guided_pixels,
            "hamming_hits": _nonnegative_count(
                fsdr.get("hamming_hits", cache_hits), "FSDR Hamming hits"
            ),
            "local_valid_hits": _nonnegative_count(
                fsdr.get("local_valid_hits", guided_pixels),
                "FSDR local-valid hits",
            ),
            "local_invalid_fallbacks": _nonnegative_count(
                fsdr.get(
                    "local_invalid_fallbacks", max(0, cache_hits - guided_pixels)
                ),
                "FSDR local-invalid fallbacks",
            ),
            "guided_top1_covered": covered,
            "guided_top1_missed": missed,
            "discrete_top1_available": discrete_top1_available,
            "full_depth_evaluations": _nonnegative_count(
                fsdr.get("full_depth_evaluations", 0),
                "FSDR full depth evaluations",
            )
            if depth_evaluations_available
            else 0,
            "executed_depth_evaluations": _nonnegative_count(
                fsdr.get("executed_depth_evaluations", 0),
                "FSDR executed depth evaluations",
            )
            if depth_evaluations_available
            else 0,
            "depth_evaluations_available": depth_evaluations_available,
            "feature_buffer_bytes_baseline": _nonnegative_count(
                fsdr.get("feature_buffer_bytes_baseline", 0),
                "FSDR baseline feature-buffer bytes",
            )
            if feature_buffer_bytes_available
            else 0,
            "feature_buffer_bytes_actual": _nonnegative_count(
                fsdr.get("feature_buffer_bytes_actual", 0),
                "FSDR actual feature-buffer bytes",
            )
            if feature_buffer_bytes_available
            else 0,
            "feature_buffer_bytes_available": feature_buffer_bytes_available,
            "source": "fsdr_path_event_counter",
        },
        "saes": {
            "total_tiles": total_tiles if tile_path_available else 0,
            "level0_tiles": level0_tiles if tile_path_available else 0,
            "level1_tiles": level1_tiles if tile_path_available else 0,
            "full_tiles": full_tiles if tile_path_available else 0,
            "tile_path_available": tile_path_available,
            "baseline_gaussians": (
                effective_gaussians + zeroed_gaussians
                if gaussian_counts_available
                else 0
            ),
            "actual_gaussians": effective_gaussians if gaussian_counts_available else 0,
            "gaussian_counts_available": gaussian_counts_available,
            "full_s2_evaluations": full_s2_evaluations,
            "executed_s2_evaluations": executed_s2_evaluations,
            "s2_evaluations_available": s2_evaluations_available,
            "l0_representatives": _nonnegative_count(
                saes.get("l0_representatives", level0_tiles * 4),
                "SAES L0 representatives",
            ),
            "l1_lightweight_anchors": _nonnegative_count(
                saes.get("l1_lightweight_anchors", level1_tiles * 12),
                "SAES L1 lightweight anchors",
            ),
            "full_stage3_gaussians": _nonnegative_count(
                saes.get("full_stage3_gaussians", full_tiles * 16),
                "SAES full Stage-3 Gaussians",
            ),
            "assignment_weight_sum_error_max": float(
                saes.get("assignment_weight_sum_error_max", 0.0)
            ),
            "opacity_transmittance_error_max": float(
                saes.get("opacity_transmittance_error_max", 0.0)
            ),
            "covariance_psd_violations": _nonnegative_count(
                saes.get("covariance_psd_violations", 0),
                "SAES covariance PSD violations",
            ),
            "hardware_accounting": (
                dict(hardware_accounting)
                if isinstance(hardware_accounting, Mapping)
                else None
            ),
            "execution_dependency": (
                dict(execution_dependency)
                if isinstance(execution_dependency, Mapping)
                else None
            ),
            "s2_s3_saving": {
                "s2": _unit_fraction(
                    requested_saving.get("s2", 0.0), "SAES S2 saving"
                ),
                "s3": _unit_fraction(
                    requested_saving.get("s3", 0.0), "SAES S3 saving"
                ),
            },
            "source": "runtime_summary",
        },
    }


def _copy_event_records(
    records: Mapping[str, Any] | None, fsdr_saes: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if records is None:
        return _default_event_records(fsdr_saes)
    if set(records) != {"fsdr", "saes"}:
        raise ValueError("event records must contain exactly fsdr and saes")
    copied: dict[str, dict[str, Any]] = {}
    for namespace in ("fsdr", "saes"):
        value = records[namespace]
        if not isinstance(value, Mapping):
            raise ValueError(f"event record {namespace} must be an object")
        copied[namespace] = dict(value)
    return copied


def _upgrade_v21_events(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fsdr = records["fsdr"]
    cache_hits = _nonnegative_count(fsdr.get("cache_hits", 0), "FSDR cache hits")
    guided = _nonnegative_count(fsdr.get("guided_pixels", 0), "FSDR guided pixels")
    hamming_hits = _nonnegative_count(
        fsdr.get("hamming_hits", cache_hits), "FSDR Hamming hits"
    )
    local_valid = _nonnegative_count(
        fsdr.get("local_valid_hits", guided), "FSDR local-valid hits"
    )
    local_fallbacks = _nonnegative_count(
        fsdr.get("local_invalid_fallbacks", hamming_hits - local_valid),
        "FSDR local-invalid fallbacks",
    )
    fsdr.update(
        {
            "hamming_hits": hamming_hits,
            "local_valid_hits": local_valid,
            "local_invalid_fallbacks": local_fallbacks,
        }
    )

    saes = records["saes"]
    level0_tiles = _nonnegative_count(saes.get("level0_tiles", 0), "SAES L0 tiles")
    level1_tiles = _nonnegative_count(saes.get("level1_tiles", 0), "SAES L1 tiles")
    full_tiles = _nonnegative_count(saes.get("full_tiles", 0), "SAES full tiles")
    saes.update(
        {
            "l0_representatives": _nonnegative_count(
                saes.get("l0_representatives", level0_tiles * 4),
                "SAES L0 representatives",
            ),
            "l1_lightweight_anchors": _nonnegative_count(
                saes.get("l1_lightweight_anchors", level1_tiles * 12),
                "SAES L1 lightweight anchors",
            ),
            "full_stage3_gaussians": _nonnegative_count(
                saes.get("full_stage3_gaussians", full_tiles * 16),
                "SAES full Stage-3 Gaussians",
            ),
            "assignment_weight_sum_error_max": float(
                saes.get("assignment_weight_sum_error_max", 0.0)
            ),
            "opacity_transmittance_error_max": float(
                saes.get("opacity_transmittance_error_max", 0.0)
            ),
            "covariance_psd_violations": _nonnegative_count(
                saes.get("covariance_psd_violations", 0),
                "SAES covariance PSD violations",
            ),
        }
    )
    requested_saving = saes.get("s2_s3_saving", {})
    if not isinstance(requested_saving, Mapping):
        raise ValueError("SAES S2/S3 saving must be an object")
    saes["s2_s3_saving"] = {
        "s2": _unit_fraction(requested_saving.get("s2", 0.0), "SAES S2 saving"),
        "s3": _unit_fraction(requested_saving.get("s3", 0.0), "SAES S3 saving"),
    }
    return records


def _bind_saes_execution_dependency(
    records: dict[str, dict[str, Any]], model: str
) -> None:
    """Bind v2.1 SAES cycle savings to the shipped per-model contract."""
    from saes.execution_dependency import resolve_s2_s3_execution_contract

    expected = resolve_s2_s3_execution_contract(model)
    observed = records["saes"].get("execution_dependency")
    if observed is not None:
        if not isinstance(observed, Mapping) or dict(observed) != expected:
            raise ValueError(
                "SAES execution dependency does not match the model contract"
            )
    records["saes"]["execution_dependency"] = expected
    saving = records["saes"]["s2_s3_saving"]
    if not expected["s2_s3_sparse_execution_verified"] and any(
        saving[stage] != 0.0 for stage in ("s2", "s3")
    ):
        raise ValueError(
            "unverified SAES execution dependency cannot claim S2/S3 savings"
        )


def strict_stage_error(stage: str, error: BaseException) -> RuntimeError:
    """Build a strict-run failure that preserves the actionable root cause."""
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("strict stage name must be non-empty")
    return RuntimeError(
        f"strict run {stage.strip()} failed; GPU fallback is forbidden: "
        f"{type(error).__name__}: {error}"
    )


def build_quality_record(
    baseline: Mapping[str, float],
    scarf: Mapping[str, float],
) -> dict[str, Any]:
    required = ("psnr_db", "ssim", "lpips")
    for label, values in (("baseline", baseline), ("scarf", scarf)):
        for key in required:
            value = values.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{label}.{key} must be finite")
    baseline_psnr = float(baseline["psnr_db"])
    if baseline_psnr <= 0:
        raise ValueError("baseline.psnr_db must be positive")
    signed = (float(scarf["psnr_db"]) - baseline_psnr) / baseline_psnr * 100.0
    return {
        "baseline": {key: float(baseline[key]) for key in required},
        "scarf": {key: float(scarf[key]) for key in required},
        "change": {
            "psnr_signed_pct": signed,
            "psnr_degradation_pct": max(0.0, -signed),
            "psnr_absolute_pct": abs(signed),
        },
    }


def mean_view_quality(
    quality_views: list[Mapping[str, Any]], variant: str
) -> dict[str, float]:
    """Return the exact arithmetic mean of one variant's per-view metrics."""
    if variant not in {"baseline", "scarf"}:
        raise ValueError(f"unsupported quality variant: {variant}")
    if not quality_views:
        raise ValueError("quality view records cannot be empty")
    aggregate: dict[str, float] = {}
    for metric in ("psnr_db", "ssim", "lpips"):
        values = []
        for view in quality_views:
            metrics = view.get(variant)
            value = metrics.get(metric) if isinstance(metrics, Mapping) else None
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"quality view {variant}.{metric} must be finite")
            values.append(float(value))
        aggregate[metric] = sum(values) / len(values)
    return aggregate


def _git(*args: str, allow_empty: bool = False, root: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or (not allow_empty and not result.stdout.strip()):
        raise RuntimeError(f"cannot determine git provenance: {' '.join(args)}")
    return result.stdout.strip()


def _submodule_commits(root: Path = ROOT) -> dict[str, str]:
    commits: dict[str, str] = {}
    for name in ("transplat", "mvsplat", "depthsplat"):
        commit = _git("-C", str(root / name), "rev-parse", "HEAD", root=root)
        if len(commit) != 40:
            raise RuntimeError(f"invalid {name} submodule commit: {commit}")
        commits[name] = commit
    return commits


def _archive_source_identity(root: Path) -> dict[str, Any]:
    path = root / "release-manifest.json"
    if not path.is_file():
        raise RuntimeError("cannot determine source provenance from Git or release manifest")
    record = json.loads(path.read_text(encoding="utf-8"))
    commit = record.get("git_commit")
    submodules = record.get("submodules")
    files = record.get("files")
    if (
        record.get("bundle_kind") != "source"
        or record.get("validation", {}).get("pass") is not True
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(submodules, dict)
        or set(submodules) != {"transplat", "mvsplat", "depthsplat"}
        or any(
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
            for value in submodules.values()
        )
        or not isinstance(files, dict)
        or not files
    ):
        raise RuntimeError("release manifest has invalid source provenance")
    for relative, expected in files.items():
        pure = PurePosixPath(relative)
        source = root / pure
        if (
            not isinstance(relative, str)
            or pure.is_absolute()
            or ".." in pure.parts
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not source.is_file()
            or sha256_file(source) != expected
        ):
            raise RuntimeError(f"release source hash mismatch: {relative}")
    source_tree = hashlib.sha256(
        json.dumps(
            {"git_commit": commit, "submodules": submodules, "files": files},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "git_commit": commit,
        "git_dirty": False,
        "submodules": dict(submodules),
        "source": "release_manifest",
        "source_tree_sha256": source_tree,
    }


def _worktree_source_sha256(
    root: Path, commit: str, submodules: Mapping[str, str]
) -> str:
    digest = hashlib.sha256()
    digest.update(commit.encode("ascii") + b"\0")
    digest.update(
        json.dumps(dict(submodules), sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "."],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    digest.update(b"\0tracked-diff\0" + diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.split(b"\0")
    for raw_path in sorted(path for path in untracked if path):
        path = root / raw_path.decode("utf-8")
        if path.is_file():
            digest.update(b"\0untracked\0" + raw_path + b"\0")
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@lru_cache(maxsize=2)
def source_identity(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    if (root / "release-manifest.json").is_file():
        return _archive_source_identity(root)
    try:
        commit = _git("rev-parse", "HEAD", root=root)
        submodules = _submodule_commits(root)
        return {
            "git_commit": commit,
            "git_dirty": bool(_git("status", "--porcelain", allow_empty=True, root=root)),
            "submodules": submodules,
            "source": "git",
            "source_tree_sha256": _worktree_source_sha256(
                root, commit, submodules
            ),
        }
    except RuntimeError:
        return _archive_source_identity(root)


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return f"<external>/{resolved.name}"


def portable_command(command: list[str]) -> list[str]:
    normalized = []
    for index, value in enumerate(command):
        text = str(value)
        candidate = Path(text)
        if index == 0 and candidate.name.startswith("python"):
            normalized.append("python")
        elif candidate.is_absolute():
            normalized.append(_display_path(candidate))
        else:
            normalized.append(text)
    return normalized


def _strict_saes_runtime_binding(
    fsdr_saes: Mapping[str, Any],
    *,
    run_class: str,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the exact runtime route binding required by strict records."""
    if run_class not in {"claim", "functional"}:
        return None
    saes = fsdr_saes.get("saes")
    if not isinstance(saes, Mapping):
        raise ValueError("strict result records require runtime SAES route evidence")
    runtime_identity = saes.get("saes_execution_identity")
    runtime_route_sha256 = saes.get("route_sha256")
    from scripts.saes_execution_identity import validate_saes_execution_identity

    expected = validate_saes_execution_identity(expected_identity)
    if runtime_identity != expected:
        raise ValueError("strict result records require runtime SAES execution identity")
    if runtime_route_sha256 != expected["route_sha256"]:
        raise ValueError("strict result records require runtime SAES route SHA256")
    return {
        "identity": copy.deepcopy(expected),
        "route_sha256": expected["route_sha256"],
    }


def build_result_record(
    *,
    model: str,
    dataset: str,
    checkpoint: Path,
    checkpoint_load: Mapping[str, Any],
    environment: Mapping[str, Any],
    dataset_manifest: Path,
    dataset_representation: str,
    dataset_tree_sha256: str,
    device: Mapping[str, Any],
    seed: int,
    quality: Mapping[str, Mapping[str, float]],
    quality_views: list[Mapping[str, Any]],
    baseline_cycles: int,
    cycles: Mapping[str, Any],
    cycle_source: str,
    ablation: Mapping[str, Any],
    fsdr_saes: Mapping[str, Any],
    command: list[str],
    runtime_assets: Mapping[str, Any],
    sample_identity: Mapping[str, Any],
    scarf_cycles: int | None = None,
    sample_index: int = 0,
    execution_index: int | None = None,
    num_samples: int = 1,
    baseline_source: str = "diagnostic_device_timing",
    fallback_stages: list[str] | None = None,
    paper_result_eligible: bool | None = None,
    run_class: str = "diagnostic",
    saes_materialization: str = REPRESENTATIVE_SAES_MATERIALIZATION,
    evidence_class: str = "deterministic_execution",
    stage_records: Mapping[str, Any] | None = None,
    event_records: Mapping[str, Any] | None = None,
    energy_record: Mapping[str, Any] | None = None,
    claim_timing: Mapping[str, Any] | None = None,
    claim_timing_backend: Mapping[str, Any] | None = None,
    claim_workflow: str | None = None,
) -> dict[str, Any]:
    if not dataset_representation:
        raise ValueError("dataset representation must be recorded")
    execution = build_execution_contract(
        run_class=run_class,
        saes_materialization=saes_materialization,
    )
    if (
        execution["saes_materialization"]
        == ASSIGNMENT_CONSENSUS_PSEUDO_DESCRIPTOR_MATERIALIZATION
    ):
        raise ValueError(
            "assignment-consensus pseudo descriptors cannot produce a normal "
            "quality/performance result record"
        )
    if (
        execution["run_class"] in {"claim", "functional"}
        and execution["saes_materialization"]
        != REPRESENTATIVE_SAES_MATERIALIZATION
    ):
        raise ValueError(
            "claim and functional result records require representative SAES "
            "materialization"
        )
    if execution["run_class"] == "claim":
        if claim_workflow is None:
            # Older callers did not name their claim surface. Preserve their
            # timing-backed behavior while allowing the new explicit quality
            # path to omit timing evidence.
            if isinstance(claim_timing, Mapping):
                claim_workflow = "performance"
            elif paper_result_eligible is True:
                raise ValueError("claim result records require source-bound timing evidence")
            else:
                claim_workflow = "quality"
        if claim_workflow not in {"quality", "mechanisms", "performance"}:
            raise ValueError(
                "claim result records must declare claim_workflow as quality, "
                "mechanisms, or performance"
            )
        if claim_workflow != "quality" and not isinstance(claim_timing, Mapping):
            raise ValueError(
                "mechanism and performance claim records require source-bound timing evidence"
            )
    elif claim_workflow is not None:
        raise ValueError("claim_workflow is only valid for claim result records")
    if execution["run_class"] == "claim" and claim_timing_backend is not None:
        if not isinstance(claim_timing_backend, Mapping):
            raise ValueError("claim timing backend provenance must be an object")
        if claim_timing_backend.get("schema_version") != (
            "source-bound-timing-backend-v1"
        ):
            raise ValueError("claim timing backend provenance has an invalid schema")
        if claim_timing_backend.get("kind") != "source_rtl":
            raise ValueError("claim timing backend provenance has an invalid kind")
        source = claim_timing_backend.get("source")
        manifest = claim_timing_backend.get("manifest")
        if (
            not isinstance(source, Mapping)
            or set(source) != {"path", "sha256"}
            or not isinstance(manifest, Mapping)
            or set(manifest) != {"path", "sha256"}
        ):
            raise ValueError("claim timing backend provenance is incomplete")
        if not _sha256_digest(source.get("sha256")) or not _sha256_digest(
            manifest.get("sha256")
        ):
            raise ValueError("claim timing backend provenance has invalid hashes")
    if execution["run_class"] != "claim" and claim_timing is not None:
        raise ValueError("only claim result records may include claim_timing")
    if execution["run_class"] != "claim" and claim_timing_backend is not None:
        raise ValueError("only claim result records may include timing backend provenance")
    environment_digest = environment.get("digest_sha256")
    if (
        not isinstance(environment_digest, str)
        or len(environment_digest) != 64
        or any(character not in "0123456789abcdef" for character in environment_digest)
    ):
        raise ValueError("environment provenance has no SHA256 digest")
    matched_tensors = checkpoint_load.get("matched_tensors")
    matched_fraction = checkpoint_load.get("matched_checkpoint_numel_fraction")
    if (
        not isinstance(matched_tensors, int)
        or isinstance(matched_tensors, bool)
        or matched_tensors <= 0
        or not isinstance(matched_fraction, (int, float))
        or isinstance(matched_fraction, bool)
        or not math.isfinite(matched_fraction)
        or not 0 < matched_fraction <= 1
    ):
        raise ValueError("checkpoint load report has no positive matched coverage")
    if len(dataset_tree_sha256) != 64:
        raise ValueError("dataset tree SHA256 must contain 64 hexadecimal characters")
    try:
        int(dataset_tree_sha256, 16)
    except ValueError as exc:
        raise ValueError("dataset tree SHA256 must be hexadecimal") from exc
    checked_cycles = require_positive_cycles(cycles)
    checked_stages = _copy_stage_records(stage_records, checked_cycles)
    checked_events = _upgrade_v21_events(
        _copy_event_records(event_records, fsdr_saes)
    )
    _bind_saes_execution_dependency(checked_events, model)
    if baseline_cycles <= 0:
        raise ValueError("baseline cycle count must be positive")
    if scarf_cycles is None:
        scarf_cycles = sum(checked_cycles.values())
    if scarf_cycles <= 0:
        raise ValueError("SCARF cycle count must be positive")
    if execution_index is None:
        execution_index = sample_index
    if (
        num_samples <= 0
        or sample_index < 0
        or execution_index < 0
        or execution_index >= num_samples
    ):
        raise ValueError("invalid sample selection")
    if baseline_source not in {
        "workstation_cuda_events",
        "cpu_perf_counter",
        "orin_nx_cuda_events",
        "diagnostic_device_timing",
    }:
        raise ValueError(f"invalid baseline timing source: {baseline_source}")
    fallback_stages = list(fallback_stages or [])
    if any(
        stage not in COMPONENTS or fallback_stages.count(stage) != 1
        for stage in fallback_stages
    ):
        raise ValueError("fallback stages must be unique SCARF component names")
    scene = sample_identity.get("scene")
    context_indices = sample_identity.get("context_indices")
    target_indices = sample_identity.get("target_indices")
    if not isinstance(scene, str) or not scene:
        raise ValueError("sample identity requires a scene key")
    if not isinstance(context_indices, list) or not context_indices:
        raise ValueError("sample identity requires context indices")
    if not isinstance(target_indices, list) or not target_indices:
        raise ValueError("sample identity requires target indices")
    if len(quality_views) != len(target_indices):
        raise ValueError("quality view records must cover every target index")
    if [view.get("target_index") for view in quality_views] != target_indices:
        raise ValueError("quality view records do not match selected target indices")
    for variant in ("baseline", "scarf"):
        aggregate = mean_view_quality(quality_views, variant)
        for metric, mean in aggregate.items():
            if not math.isclose(
                mean, float(quality[variant][metric]), rel_tol=1e-5, abs_tol=1e-6
            ):
                raise ValueError(
                    f"quality {variant}.{metric} is not the target-view mean"
                )
    quality_record = build_quality_record(quality["baseline"], quality["scarf"])
    quality_record["views"] = [
        {
            "target_index": int(view["target_index"]),
            "baseline": {
                metric: float(view["baseline"][metric])
                for metric in ("psnr_db", "ssim", "lpips")
            },
            "scarf": {
                metric: float(view["scarf"][metric])
                for metric in ("psnr_db", "ssim", "lpips")
            },
        }
        for view in quality_views
    ]
    source = source_identity()
    functional_fixture = dataset_representation == "re10k-synthetic-functional-v1"
    if paper_result_eligible is None:
        paper_result_eligible = (
            not functional_fixture
            and execution["run_class"] == "claim"
            and execution["saes_materialization"]
            == REPRESENTATIVE_SAES_MATERIALIZATION
        )
    elif not isinstance(paper_result_eligible, bool):
        raise ValueError("paper_result_eligible must be a boolean")
    paper_result_eligible = paper_result_eligible and not functional_fixture
    if paper_result_eligible and (
        execution["run_class"] != "claim"
        or execution["saes_materialization"]
        != REPRESENTATIVE_SAES_MATERIALIZATION
    ):
        raise ValueError(
            "paper-result-eligible records require claim run class and "
            "representative SAES materialization"
        )
    if (
        paper_result_eligible
        and checked_events["saes"]["execution_dependency"].get(
            "s2_s3_sparse_execution_verified"
        )
        is not True
    ):
        raise ValueError(
            "paper-result-eligible records require verified whole-pipeline "
            "SAES S2/S3 sparse execution"
        )
    mechanism_config, mechanism = load_mechanism_config()
    strict_saes_binding = _strict_saes_runtime_binding(
        fsdr_saes,
        run_class=execution["run_class"],
        expected_identity=mechanism_config["saes_execution_identity"],
    )
    if mechanism["status"] != "calibrated":
        paper_result_eligible = False
    elif (
        mechanism.get("protocol") not in {
            "dl3dv_train_holdout_v1",
            "acid_train_holdout_v1",
        }
        or mechanism.get("train_holdout_scene_disjoint") is not True
        or not isinstance(mechanism.get("train"), dict)
        or not isinstance(mechanism.get("holdout"), dict)
    ):
        raise RuntimeError(
            "calibrated mechanism provenance has no verified train/holdout evidence"
        )
    calibration_provenance = {
        key: value
        for key, value in mechanism.items()
        if key != "mechanism_config_sha256"
    }
    record = {
        "schema_version": "2.1",
        "evidence_class": evidence_class,
        "provenance": {
            "git_commit": source["git_commit"],
            "git_dirty": source["git_dirty"],
            "source_identity": source["source"],
            "source_tree_sha256": source["source_tree_sha256"],
            "mechanism_config_sha256": mechanism["mechanism_config_sha256"],
            "calibration_provenance": calibration_provenance,
            **(
                {"timing_backend": copy.deepcopy(dict(claim_timing_backend))}
                if claim_timing_backend is not None
                else {}
            ),
            **(
                {
                    "saes_execution_identity": strict_saes_binding["identity"],
                    "saes_execution_route_sha256": strict_saes_binding[
                        "route_sha256"
                    ],
                }
                if strict_saes_binding is not None
                else {}
            ),
            "submodules": source["submodules"],
            "command": portable_command(command),
            "runtime_assets": dict(runtime_assets),
            "execution_contract": execution,
            **(
                {"claim_workflow": claim_workflow}
                if claim_workflow is not None
                else {}
            ),
            "seed": int(seed),
            "model": model,
            "environment": dict(environment),
            "device": dict(device),
            "dataset": {
                "name": dataset,
                "representation": dataset_representation,
                "functional_fixture": functional_fixture,
                "paper_result_eligible": paper_result_eligible,
                "manifest": _display_path(dataset_manifest),
                "sha256": sha256_file(dataset_manifest),
                "tree_sha256": dataset_tree_sha256,
            },
            "checkpoint": {
                "path": _display_path(checkpoint),
                "sha256": cached_sha256_file(checkpoint),
                "load": dict(checkpoint_load),
            },
            "evaluation": {
                "kind": "sample",
                "sample_index": int(sample_index),
                "execution_index": int(execution_index),
                "candidate_count": int(num_samples),
                "scene": scene,
                "context_indices": [int(value) for value in context_indices],
                "target_indices": [int(value) for value in target_indices],
                "target_view_count": len(target_indices),
                "target_view_aggregation": "arithmetic mean over selected target views",
            },
        },
        "quality": quality_record,
        "performance": {
            "baseline_cycles": int(baseline_cycles),
            "scarf_cycles": int(scarf_cycles),
            "speedup": float(baseline_cycles) / float(scarf_cycles),
            "baseline_source": baseline_source,
            "cycle_source": cycle_source,
            "components": checked_cycles,
            "stages": checked_stages,
        },
        "events": checked_events,
        "energy": dict(
            energy_record
            if energy_record is not None
            else {"available": False, "source": "not_measured"}
        ),
        "ablation": dict(ablation),
        "fsdr_saes": dict(fsdr_saes),
        "hardware": {
            "physical_ppa_included": False,
            "source": "run hardware/iflow/run.sh separately",
        },
        "validation": {
            "reproducible": not fallback_stages,
            "reference_fallback_used": bool(fallback_stages),
            "fallback_stages": fallback_stages,
        },
    }
    if claim_timing is not None:
        record["performance"]["claim_timing"] = copy.deepcopy(dict(claim_timing))
    if execution["run_class"] == "claim" and claim_timing is not None:
        timing = record["performance"]["claim_timing"]
        supplied_binding = timing.get("input_binding_sha256")
        expected_binding = claim_timing_binding_sha256(record, aggregate=False)
        if supplied_binding not in (None, expected_binding):
            raise ValueError("claim timing input binding does not match the result")
        timing["input_binding_sha256"] = expected_binding
    bind_execution_trace(record)
    if execution["run_class"] == "claim" and claim_timing is not None:
        claim_timing_from_record(record, aggregate=False)
    return record


def write_result(record: Mapping[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
