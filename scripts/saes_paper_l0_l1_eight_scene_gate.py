#!/usr/bin/env python3
"""Run the fixed eight-scene V16 DL3DV development quality gate.

Each scene is independently prepared without target data, audited, then rendered
only after the exact target-free audit is bound to its packet.  This is a
development gate for the V16 compact-materialization simulator, not paper
quality, sparse-execution, timing, or Figure 11 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.context_only_audit_input import (
    prepare_context_only_audit_input,
    validate_context_only_audit_input,
)
from data.prepare_dl3dv_target_free_audit_inputs import (
    DEFAULT_RAW_ROOT,
    prepare_inputs,
)
from data.verify_prepared_dataset import prepared_scene_order
from saes.evaluation_disjoint_l1_calibration import (
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
)
from scripts.result_record import source_identity, write_result
from scripts.ae_config import (
    resolve_claim_selection,
    resolve_experiment,
    validate_claim_dataset_tree,
    validate_prepared_dataset,
)
from scripts.compile_protocol import canonicalize_index
from scripts.saes_incremental_selected_output_audit import (
    collect_incremental_selected_output_audit,
)
from scripts.saes_paper_l0_l1_compact_packet_pilot import (
    SEED,
    collect_paper_compact_packet_pilot,
)
from scripts.saes_selected_output_quality_gate import _mean_metrics, _quality_verdict


GATE_KIND = "saes-paper-l0-l1-v16-fixed-eight-scene-quality-gate"
FIXED_SAMPLE_INDICES = tuple(range(8))
EXPECTED_SOURCE_INDEX_SHA256 = (
    "eab21290cfab8eff12e208377b089b2a65e15f7ba44f7bb085963511354863f4"
)
EXPECTED_SAMPLE_SELECTION_SHA256 = (
    "a2b432a5513cec757bc7f5a03c8f424029a856b02fb6a22125a21a61b71cb63e"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "89e43c205a04962e427801385d7d18e74cba063d05a76bf8b28e5fa746a4b69a"
)
EXPECTED_V15_SHA256 = (
    "821ce995a27f84a45721a2089afd52b8f3656e0a968c22036913d7d8e88e9596"
)
EXPECTED_V16_SHA256 = (
    "a2786fd07717d9c08ee654553346e4f88525668e389ad953d20a9e1c2c1ba1ba"
)
EXPECTED_MECHANISM_SHA256 = (
    "7dae4d088aa9d5c4340cc88531e8265d0f33c1c3836b93fbd180aa8b298b01f5"
)
EXPECTED_PREPARED_TREE_SHA256 = (
    "4ea2b8a5794ffac079c986a2a3c94756f27bc90aa287524022fd866c4b6c6ef9"
)
EXPECTED_RAW_SOURCE_RECORD_SHA256 = (
    "92a112fd87ba4d0884bdab3855c5f93dd26b95c866385c39acf6868d871a191e"
)
EXPECTED_RAW_SOURCE_REVISION = "9684e8382278c5e18173c1e72bd246daf2874539"
EXPECTED_RAW_BENCHMARK_METADATA_SHA256 = (
    "ac69c85f30fb208feb8dc1993eccdb6a69d3f2921a0abeeb6381074204e827e3"
)
EXPECTED_RAW_FILELIST_SHA256 = (
    "e310b69b2b72fafd3f7900e14f4301beea721f50ca7326e27d3298baeea2043a"
)
EXPECTED_RAW_SCENE_SOURCE_PLANS_SHA256 = (
    "9283d6840a774acf893639259a17fb8029622fc214003e8772851015701a1b3b"
)
_QUALITY_KEYS = ("psnr_db", "ssim", "lpips")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _require_finite_metric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _set_seed(device: torch.device) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def _release_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _sample_root(output_dir: Path, sample_index: int) -> Path:
    return output_dir / "samples" / f"sample_{sample_index:05d}"


def _fixed_selections() -> tuple[Any, Any, list[dict[str, Any]]]:
    """Resolve fixed source ordinals to the prepared dataloader's execution order."""
    selection = resolve_claim_selection("transplat", "dl3dv", ROOT)
    experiment = resolve_experiment("transplat", "dl3dv", ROOT)
    if (
        selection.source_index_sha256 != EXPECTED_SOURCE_INDEX_SHA256
        or selection.sample_selection_sha256 != EXPECTED_SAMPLE_SELECTION_SHA256
    ):
        raise ValueError("fixed eight-scene canonical selection changed")
    source_rows, summary = canonicalize_index(
        selection.index_path, selection.source_index_sha256
    )
    if summary["sample_count"] != selection.sample_count:
        raise ValueError("canonical selection sample count changed")
    execution_order = prepared_scene_order(
        experiment.dataset_root, {row["scene"] for row in source_rows}
    )
    rows, execution_summary = canonicalize_index(
        selection.index_path,
        selection.source_index_sha256,
        execution_scene_order=execution_order,
    )
    if execution_summary != summary:
        raise ValueError("prepared execution order changed the canonical selection")
    by_source = {row["sample_index"]: row for row in rows}
    fixed = [by_source[index] for index in FIXED_SAMPLE_INDICES if index in by_source]
    if len(fixed) != len(FIXED_SAMPLE_INDICES):
        raise ValueError("fixed eight-scene source ordinal is unavailable")
    return selection, experiment, fixed


def _validate_prepared_contract(experiment: Any) -> dict[str, Any]:
    manifest = Path(experiment.dataset_root) / ".scarf-manifest.json"
    prepared = validate_prepared_dataset(
        experiment, experiment.dataset_root, manifest
    )
    validate_claim_dataset_tree(
        "transplat", "dl3dv", prepared["tree_sha256"], ROOT
    )
    if prepared["tree_sha256"] != EXPECTED_PREPARED_TREE_SHA256:
        raise ValueError("prepared DL3DV tree changed")
    return prepared


def _selection_identity(selection: Mapping[str, Any]) -> dict[str, Any]:
    source_sample_index = selection.get("sample_index")
    execution_index = selection.get("execution_index")
    scene = selection.get("scene")
    context_indices = selection.get("context_indices")
    target_indices = selection.get("target_indices")
    if (
        isinstance(source_sample_index, bool)
        or not isinstance(source_sample_index, int)
        or source_sample_index < 0
        or isinstance(execution_index, bool)
        or not isinstance(execution_index, int)
        or execution_index < 0
        or not isinstance(scene, str)
        or not scene
        or not isinstance(context_indices, list)
        or not isinstance(target_indices, list)
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in [*context_indices, *target_indices])
        or len(context_indices) != 2
        or len(target_indices) != 4
        or len(set(context_indices)) != len(context_indices)
        or len(set(target_indices)) != len(target_indices)
        or set(context_indices) & set(target_indices)
    ):
        raise ValueError("fixed eight-scene canonical row is invalid")
    return {
        "source_sample_index": source_sample_index,
        "execution_index": execution_index,
        "scene": scene,
        "context_indices": list(context_indices),
        "target_indices": list(target_indices),
    }


def _source_contract(
    source: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _selection_identity(selection)
    if (
        source.get("source_sample_index") != expected["source_sample_index"]
        or source.get("target_rgb_included") is not False
        or source.get("target_rgb_opened") is not False
    ):
        raise ValueError("target-free source input identity changed")
    protocol = _require_mapping(source.get("canonical_protocol"), "source protocol")
    source_index = _require_sha256(protocol.get("source_index_sha256"), "source index")
    selection_sha256 = _require_sha256(
        protocol.get("sample_selection_sha256"), "source sample selection"
    )
    if source_index != EXPECTED_SOURCE_INDEX_SHA256:
        raise ValueError("fixed eight-scene source index changed")
    if selection_sha256 != EXPECTED_SAMPLE_SELECTION_SHA256:
        raise ValueError("fixed eight-scene sample selection changed")
    if protocol.get("dataset_tree_sha256") != EXPECTED_PREPARED_TREE_SHA256:
        raise ValueError("target-free source dataset tree changed")
    canonical_selection = _require_mapping(
        source.get("canonical_selection"), "source canonical selection"
    )
    expected_source_selection = {
        key: expected[key]
        for key in ("source_sample_index", "scene", "context_indices", "target_indices")
    }
    if dict(canonical_selection) != expected_source_selection:
        raise ValueError("target-free source canonical selection changed")
    selected = _require_mapping(source.get("selected_sample"), "source selected sample")
    if any(selected.get(key) != value for key, value in expected_source_selection.items() if key != "source_sample_index"):
        raise ValueError("target-free source selected sample changed")
    raw_source = _require_mapping(source.get("source"), "raw DL3DV source")
    if raw_source.get("source_record_sha256") != EXPECTED_RAW_SOURCE_RECORD_SHA256:
        raise ValueError("raw DL3DV source record changed")
    if raw_source.get("revision") != EXPECTED_RAW_SOURCE_REVISION:
        raise ValueError("raw DL3DV source revision changed")
    if raw_source.get("benchmark_metadata_sha256") != EXPECTED_RAW_BENCHMARK_METADATA_SHA256:
        raise ValueError("raw DL3DV benchmark metadata changed")
    if raw_source.get("filelist_sha256") != EXPECTED_RAW_FILELIST_SHA256:
        raise ValueError("raw DL3DV file list changed")
    if raw_source.get("scene_source_plans_sha256") != EXPECTED_RAW_SCENE_SOURCE_PLANS_SHA256:
        raise ValueError("raw DL3DV scene source plans changed")
    return {
        "source_index_sha256": source_index,
        "sample_selection_sha256": selection_sha256,
        "canonical_selection": expected_source_selection,
        "raw_source": dict(raw_source),
    }


def _validate_target_free_audit(
    audit: Mapping[str, Any],
    *,
    context_identity: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    if audit.get("status") != "PASS":
        raise ValueError("target-free selected-output audit did not pass")
    if audit.get("input_identity") != dict(context_identity):
        raise ValueError("target-free audit input identity changed")
    if audit.get("target_mapping_present") is not False:
        raise ValueError("target-free audit constructed a target mapping")
    for key in ("target_rgb_accessed", "target_camera_metadata_accessed"):
        if audit.get(key) is not False:
            raise ValueError(f"target-free audit crossed {key}")
    access = _require_mapping(audit.get("access_evidence"), "target-free audit access")
    if any(
        access.get(key) is not False
        for key in (
            "target_mapping_present",
            "target_rgb_accessed",
            "target_camera_metadata_accessed",
            "target_index_accessed",
        )
    ):
        raise ValueError("target-free audit access evidence changed")
    expected = _selection_identity(selection)
    if (
        context_identity.get("source_sample_index") != expected["source_sample_index"]
        or context_identity.get("scene") != expected["scene"]
        or context_identity.get("context_indices") != expected["context_indices"]
    ):
        raise ValueError("target-free audit sample index changed")


def _validate_quality_record(
    quality: Mapping[str, Any],
    *,
    context_identity: Mapping[str, Any],
    selection: Mapping[str, Any],
    native_sample_count: int,
) -> None:
    expected = _selection_identity(selection)
    if quality.get("sample_index") != expected["source_sample_index"]:
        raise ValueError("quality record sample index changed")
    if quality.get("execution_index") != expected["execution_index"]:
        raise ValueError("quality record execution index changed")
    if quality.get("native_sample_count") != native_sample_count:
        raise ValueError("quality record native sample count changed")
    if any(
        quality.get(key) != expected[key]
        for key in ("scene", "context_indices", "target_indices")
    ):
        raise ValueError("quality record canonical selection changed")
    if quality.get("paper_result_eligible") is not False:
        raise ValueError("development quality gate became paper-eligible")
    if quality.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("quality checkpoint changed")
    if _require_mapping(quality.get("adaptive_l1_calibration"), "V15").get(
        "sha256"
    ) != EXPECTED_V15_SHA256:
        raise ValueError("quality V15 calibration changed")
    if _require_mapping(quality.get("adaptive_l1_v4_attribute_loo_calibration"), "V16").get(
        "sha256"
    ) != EXPECTED_V16_SHA256:
        raise ValueError("quality V16 calibration changed")
    if _require_mapping(quality.get("paper_identity"), "mechanism").get(
        "sha256"
    ) != EXPECTED_MECHANISM_SHA256:
        raise ValueError("quality mechanism identity changed")
    gate = _require_mapping(quality.get("target_free_quality_gate"), "quality gate")
    if gate.get("sample_index") != expected["source_sample_index"]:
        raise ValueError("quality gate sample index changed")
    if gate.get("context_input_identity") != dict(context_identity):
        raise ValueError("quality gate context identity changed")
    for key in (
        "source_selection_mask_sha256",
        "selected_output_mask_sha256",
        "packed_source_trace_sha256",
    ):
        _require_sha256(gate.get(key), f"quality gate {key}")
    boundary = _require_mapping(quality.get("execution_boundary"), "quality boundary")
    if (
        boundary.get("whole_pipeline_s2_s3_sparse_execution_verified") is not False
        or boundary.get("s2_s3_saving") != 0.0
        or boundary.get("timing_claim") is not False
    ):
        raise ValueError("quality record crossed its sparse-execution boundary")


def _quality_views(record: Mapping[str, Any]) -> list[dict[str, float]]:
    quality = _require_mapping(record.get("quality"), "sample quality")
    views = quality.get("views")
    if not isinstance(views, list) or len(views) != 4:
        raise ValueError("fixed eight-scene gate requires four target views per scene")
    result: list[dict[str, float]] = []
    for position, view in enumerate(views):
        item = _require_mapping(view, f"quality view {position}")
        for field in ("baseline", "compact"):
            metric = _require_mapping(item.get(field), f"quality view {field}")
            for key in _QUALITY_KEYS:
                _require_finite_metric(metric.get(key), f"quality view {field}.{key}")
        result.append(
            {
                f"baseline_{key}": _require_finite_metric(
                    _require_mapping(item["baseline"], "baseline").get(key),
                    f"baseline.{key}",
                )
                for key in _QUALITY_KEYS
            }
            | {
                f"compact_{key}": _require_finite_metric(
                    _require_mapping(item["compact"], "compact").get(key),
                    f"compact.{key}",
                )
                for key in _QUALITY_KEYS
            }
        )
    return result


def _mean_quality(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    if not records:
        raise ValueError("cannot aggregate an empty quality record set")
    baseline_views: list[dict[str, float]] = []
    compact_views: list[dict[str, float]] = []
    for record in records:
        for view in _quality_views(record):
            baseline_views.append({key: view[f"baseline_{key}"] for key in _QUALITY_KEYS})
            compact_views.append({key: view[f"compact_{key}"] for key in _QUALITY_KEYS})
    return _mean_metrics(baseline_views), _mean_metrics(compact_views)


def _macro_quality(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    baseline: dict[str, list[float]] = {key: [] for key in _QUALITY_KEYS}
    compact: dict[str, list[float]] = {key: [] for key in _QUALITY_KEYS}
    for record in records:
        quality = _require_mapping(record.get("quality"), "sample quality")
        for label, target in (("baseline", baseline), ("compact", compact)):
            metric = _require_mapping(quality.get(label), f"sample {label} quality")
            for key in _QUALITY_KEYS:
                target[key].append(_require_finite_metric(metric.get(key), f"{label}.{key}"))
    return (
        {key: sum(values) / len(values) for key, values in baseline.items()},
        {key: sum(values) / len(values) for key, values in compact.items()},
    )


def _route_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = {"L0": 0, "L1": 0, "Full": 0}
    for record in records:
        compact_route = _require_mapping(record.get("compact_route"), "compact route")
        final_route = _require_mapping(compact_route.get("final_route"), "final route")
        counts = _require_mapping(final_route.get("route_counts"), "route counts")
        for key in total:
            value = counts.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"route count {key} is invalid")
            total[key] += value
    return total


def _quality_deltas(record: Mapping[str, Any]) -> dict[str, float]:
    quality = _require_mapping(record.get("quality"), "sample quality")
    baseline = _require_mapping(quality.get("baseline"), "sample baseline")
    compact = _require_mapping(quality.get("compact"), "sample compact")
    return {
        "psnr_loss_db": _require_finite_metric(baseline.get("psnr_db"), "baseline psnr")
        - _require_finite_metric(compact.get("psnr_db"), "compact psnr"),
        "ssim_loss": _require_finite_metric(baseline.get("ssim"), "baseline ssim")
        - _require_finite_metric(compact.get("ssim"), "compact ssim"),
        "lpips_increase": _require_finite_metric(compact.get("lpips"), "compact lpips")
        - _require_finite_metric(baseline.get("lpips"), "baseline lpips"),
    }


def _aggregate_completed_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    quality_records = [sample["quality_record"] for sample in samples]
    if not all(isinstance(record, Mapping) for record in quality_records):
        raise ValueError("completed samples lack quality records")
    records = [dict(record) for record in quality_records]
    pooled_baseline, pooled_compact = _mean_quality(records)
    macro_baseline, macro_compact = _macro_quality(records)
    pooled_verdict = _quality_verdict(pooled_baseline, pooled_compact)
    macro_verdict = _quality_verdict(macro_baseline, macro_compact)
    deltas = [
        {"sample_index": sample["sample_index"], "scene": sample["scene"], **_quality_deltas(record)}
        for sample, record in zip(samples, records)
    ]
    return {
        "view_count": len(records) * 4,
        "scene_count": len(records),
        "pooled": {
            "baseline": pooled_baseline,
            "compact": pooled_compact,
            "verdict": pooled_verdict,
        },
        "scene_macro": {
            "baseline": macro_baseline,
            "compact": macro_compact,
            "verdict": macro_verdict,
        },
        "per_scene_deltas": deltas,
        "worst_scene": {
            "psnr_loss_db": max(deltas, key=lambda item: item["psnr_loss_db"]),
            "ssim_loss": max(deltas, key=lambda item: item["ssim_loss"]),
            "lpips_increase": max(deltas, key=lambda item: item["lpips_increase"]),
        },
        "route_totals": _route_totals(records),
        "pass": bool(pooled_verdict["pass"] and macro_verdict["pass"]),
    }


def _sample_summary(
    *,
    selection: Mapping[str, Any],
    native_sample_count: int,
    source: Mapping[str, Any],
    context_identity: Mapping[str, Any],
    audit_path: Path,
    audit: Mapping[str, Any],
    quality_path: Path,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _selection_identity(selection)
    return {
        "sample_index": expected["source_sample_index"],
        "execution_index": expected["execution_index"],
        "native_sample_count": native_sample_count,
        "scene": quality.get("scene"),
        "source_protocol": _source_contract(source, selection),
        "context_input_identity": dict(context_identity),
        "target_free_audit": {
            "path": str(audit_path.relative_to(audit_path.parents[3])),
            "file_sha256": _sha256_file(audit_path),
            "record_sha256": audit.get("sha256"),
            "status": audit.get("status"),
        },
        "quality": {
            "path": str(quality_path.relative_to(quality_path.parents[3])),
            "file_sha256": _sha256_file(quality_path),
            "status": quality.get("status"),
            "verdict": _require_mapping(quality.get("quality"), "sample quality").get(
                "verdict"
            ),
        },
        "quality_record": dict(quality),
    }


def _run_sample(
    *,
    output_dir: Path,
    raw_root: Path,
    device: torch.device,
    selection: Mapping[str, Any],
    native_sample_count: int,
    v15_calibration_record: Path,
    v16_calibration_record: Path,
    acid_plan_path: Path,
    acid_materialization_root: Path,
) -> dict[str, Any]:
    expected = _selection_identity(selection)
    sample_index = expected["source_sample_index"]
    root = _sample_root(output_dir, sample_index)
    source_root = root / "target-free-source"
    context_root = root / "context-only"
    audit_root = root / "audit"
    quality_root = root / "quality"
    source = prepare_inputs(raw_root, output_dir=source_root, sample_index=sample_index)
    source_contract = _source_contract(source, selection)
    prepare_context_only_audit_input(source_root, output_root=context_root)
    context_identity = validate_context_only_audit_input(context_root)
    if (
        context_identity.get("source_sample_index") != sample_index
        or context_identity.get("scene") != expected["scene"]
        or context_identity.get("context_indices") != expected["context_indices"]
    ):
        raise ValueError("context-only input canonical selection changed")
    source_binding = _require_mapping(context_identity.get("source_binding"), "context binding")
    if source_binding.get("canonical_index_sha256") != EXPECTED_SOURCE_INDEX_SHA256:
        raise ValueError("context-only source index changed")
    if source_binding.get("canonical_sample_selection_sha256") != EXPECTED_SAMPLE_SELECTION_SHA256:
        raise ValueError("context-only sample selection changed")
    if source_binding.get("canonical_selection_sha256") != _canonical_sha256(
        source_contract["canonical_selection"]
    ):
        raise ValueError("context-only canonical selection changed")

    _set_seed(device)
    audit = collect_incremental_selected_output_audit(
        input_root=context_root,
        device=device,
        v15_calibration_record=v15_calibration_record,
        v16_calibration_record=v16_calibration_record,
        acid_calibration_plan=acid_plan_path,
        acid_materialization_root=acid_materialization_root,
    )
    audit_path = audit_root / "results.json"
    write_result(audit, audit_path)
    _validate_target_free_audit(
        audit, context_identity=context_identity, selection=selection
    )

    _set_seed(device)
    quality = collect_paper_compact_packet_pilot(
        device=device,
        v15_calibration_record=v15_calibration_record,
        v16_calibration_record=v16_calibration_record,
        target_free_audit_artifact=audit_path,
        target_free_input_root=context_root,
        sample_index=sample_index,
        execution_index=expected["execution_index"],
        native_sample_count=native_sample_count,
        acid_plan_path=acid_plan_path,
        acid_materialization_root=acid_materialization_root,
    )
    quality_path = quality_root / "results.json"
    write_result(quality, quality_path)
    _validate_quality_record(
        quality,
        context_identity=context_identity,
        selection=selection,
        native_sample_count=native_sample_count,
    )
    return _sample_summary(
        selection=selection,
        native_sample_count=native_sample_count,
        source=source,
        context_identity=context_identity,
        audit_path=audit_path,
        audit=audit,
        quality_path=quality_path,
        quality=quality,
    )


def run_fixed_eight_scene_gate(
    *,
    output_dir: Path,
    raw_root: Path,
    device: torch.device,
    v15_calibration_record: Path,
    v16_calibration_record: Path,
    acid_plan_path: Path = DEFAULT_PLAN_PATH,
    acid_materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Execute the fixed V16 quality contract serially for DL3DV indices 0..7."""
    claim_selection, experiment, fixed_selections = _fixed_selections()
    prepared_dataset = _validate_prepared_contract(experiment)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"eight-scene output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for selection in fixed_selections:
        source_sample_index = _selection_identity(selection)["source_sample_index"]
        try:
            sample = _run_sample(
                output_dir=output_dir,
                raw_root=raw_root,
                device=device,
                selection=selection,
                native_sample_count=claim_selection.sample_count,
                v15_calibration_record=v15_calibration_record,
                v16_calibration_record=v16_calibration_record,
                acid_plan_path=acid_plan_path,
                acid_materialization_root=acid_materialization_root,
            )
            samples.append(sample)
        except Exception as exc:
            failures.append(
                {
                    "sample_index": source_sample_index,
                    "execution_index": _selection_identity(selection)["execution_index"],
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        finally:
            _release_cuda(device)

    completed = len(samples) == len(FIXED_SAMPLE_INDICES)
    aggregate = _aggregate_completed_samples(samples) if completed else None
    all_sample_pass = completed and all(
        sample["quality"]["status"] == "PASS"
        and _require_mapping(sample["quality"]["verdict"], "sample verdict").get("pass")
        is True
        for sample in samples
    )
    status = "PASS" if all_sample_pass and aggregate and aggregate["pass"] else "FAILED"
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": GATE_KIND,
        "status": status,
        "paper_result_eligible": False,
        "fixed_sample_indices": list(FIXED_SAMPLE_INDICES),
        "fixed_selections": [
            _selection_identity(selection) for selection in fixed_selections
        ],
        "expected_identity": {
            "source_index_sha256": EXPECTED_SOURCE_INDEX_SHA256,
            "sample_selection_sha256": EXPECTED_SAMPLE_SELECTION_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "v15_calibration_sha256": EXPECTED_V15_SHA256,
            "v16_calibration_sha256": EXPECTED_V16_SHA256,
            "mechanism_sha256": EXPECTED_MECHANISM_SHA256,
            "prepared_tree_sha256": EXPECTED_PREPARED_TREE_SHA256,
            "raw_source_record_sha256": EXPECTED_RAW_SOURCE_RECORD_SHA256,
            "raw_source_revision": EXPECTED_RAW_SOURCE_REVISION,
            "raw_benchmark_metadata_sha256": EXPECTED_RAW_BENCHMARK_METADATA_SHA256,
            "raw_filelist_sha256": EXPECTED_RAW_FILELIST_SHA256,
            "raw_scene_source_plans_sha256": EXPECTED_RAW_SCENE_SOURCE_PLANS_SHA256,
        },
        "sample_count_completed": len(samples),
        "prepared_dataset": prepared_dataset,
        "samples": samples,
        "failures": failures,
        "quality": aggregate,
        "execution_boundary": {
            "per_scene_target_free_audit_required": True,
            "target_loaded_only_after_packet_and_audit_binding": True,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "s2_s3_saving": 0.0,
            "timing_claim": False,
            "global_route_ratio_cycle_scaling_used": False,
        },
        "source": source_identity(),
    }
    record["sha256"] = _canonical_sha256(record)
    write_result(record, output_dir / "results.json")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--v15-calibration-record", type=Path, required=True)
    parser.add_argument("--v16-calibration-record", type=Path, required=True)
    parser.add_argument("--acid-plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument(
        "--acid-materialization-root",
        type=Path,
        default=DEFAULT_MATERIALIZATION_ROOT,
    )
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("the fixed eight-scene V16 gate requires CUDA")
    try:
        record = run_fixed_eight_scene_gate(
            output_dir=args.output_dir,
            raw_root=args.raw_root,
            device=device,
            v15_calibration_record=args.v15_calibration_record,
            v16_calibration_record=args.v16_calibration_record,
            acid_plan_path=args.acid_plan_path,
            acid_materialization_root=args.acid_materialization_root,
        )
    except Exception as exc:
        parser.error(f"eight-scene V16 gate could not start: {exc}")
    print(args.output_dir / "results.json")
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
