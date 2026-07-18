#!/usr/bin/env python3
"""Run an evaluation-disjoint SCARF calibration grid and fixed holdout gate.

The DL3DV protocol is deliberately two-stage.  The training split may rank the
registered global grid exactly once.  The holdout split is then invoked with
that one committed tuple and is allowed only to accept or reject it.  Neither
the holdout trace nor this aggregation code can rerank a second grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ae_config import resolve_experiment
from scripts.calibration_contract import (
    PARAMETER_GRID,
    candidate_is_feasible,
    candidate_sha256,
    canonical_parameters,
    canonical_sha256,
    parameters_sha256,
    select_global_candidate,
    validate_candidate,
)
from scripts.calibration_inputs import validate_target_free_input_root
from scripts.compile_protocol import canonicalize_index
from scripts.run_ae import _python_for


LEGACY_PAIRS = tuple(
    (model, dataset)
    for model in ("transplat", "mvsplat", "depthsplat")
    for dataset in ("re10k", "acid")
)

DL3DV_PAIRS = (
    ("transplat", "dl3dv", "re10k"),
    ("mvsplat", "dl3dv", "re10k"),
    ("depthsplat", "dl3dv", "native"),
)
DL3DV_SPLITS = ("calibration_train", "calibration_holdout")
DL3DV_SPLIT_COUNTS = {"calibration_train": 24, "calibration_holdout": 8}
TRAIN_SPLIT = "calibration_train"
HOLDOUT_SPLIT = "calibration_holdout"
SPLIT_PROTOCOL = "dl3dv_train_holdout_v1"

REGISTERED_GRID_SIZE = math.prod(len(values) for values in PARAMETER_GRID.values())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA256 digest")
    return value


def _parameter_key(parameters: Mapping[str, Any]) -> tuple[float, ...]:
    normalized = canonical_parameters(parameters)
    return tuple(normalized[name] for name in PARAMETER_GRID)


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def aggregate_candidate_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate target-free per-sample traces into global candidate records."""
    if not traces:
        raise ValueError("calibration trace set is empty")
    grouped: dict[tuple[float, ...], dict[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, Mapping) or trace.get("kind") != "calibration_sample_trace":
            raise ValueError("calibration input contains a non-trace record")
        trace_details = trace.get("trace")
        if not isinstance(trace_details, Mapping):
            raise ValueError("calibration trace has no execution details")
        if trace_details.get("neural_forward_passes") != 1:
            raise ValueError("calibration sample did not execute one neural forward pass")
        model, dataset = trace.get("model"), trace.get("dataset")
        if not isinstance(model, str) or not model or not isinstance(dataset, str) or not dataset:
            raise ValueError("calibration trace has no model/dataset identity")
        candidates = trace.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("calibration trace has no candidates")
        pair = f"{model}/{dataset}"
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("calibration trace candidate is invalid")
            validate_candidate(candidate)
            key = _parameter_key(candidate.get("parameters", {}))
            quality_record = candidate.get("quality")
            if not isinstance(quality_record, Mapping):
                raise ValueError("calibration candidate has no quality record")
            quality = {
                metric: _finite_nonnegative(
                    quality_record.get(metric), f"{pair}.{metric}"
                )
                for metric in ("psnr_loss_db", "ssim_loss", "lpips_increase")
            }
            entry = grouped.setdefault(
                key,
                {
                    "parameters": canonical_parameters(candidate["parameters"]),
                    "quality": {},
                    "work": [],
                    "compression_values": [],
                },
            )
            pair_quality = entry["quality"].setdefault(pair, [])
            pair_quality.append(quality)
            entry["work"].append(
                _finite_nonnegative(candidate.get("work_reduction"), "work_reduction")
            )
            entry["compression_values"].append(
                _finite_nonnegative(candidate.get("compression"), "compression")
            )

    output = []
    for key in sorted(grouped):
        entry = grouped[key]
        quality = {
            pair: {
                metric: sum(float(row[metric]) for row in rows) / len(rows)
                for metric in ("psnr_loss_db", "ssim_loss", "lpips_increase")
            }
            for pair, rows in sorted(entry["quality"].items())
        }
        output.append(
            {
                "parameters": entry["parameters"],
                "quality": quality,
                "work_reduction": sum(entry["work"]) / len(entry["work"]),
                "compression": sum(entry["compression_values"])
                / len(entry["compression_values"]),
            }
        )
    return output


def _with_candidate_hashes(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        copied = dict(candidate)
        copied["candidate_sha256"] = candidate_sha256(copied)
        result.append(copied)
    return result


def _candidate_parameter_set(candidates: list[Mapping[str, Any]]) -> list[str]:
    return sorted(parameters_sha256(candidate.get("parameters", {})) for candidate in candidates)


def _registered_parameter_set() -> list[str]:
    names = tuple(PARAMETER_GRID)
    values: list[dict[str, float]] = [{}]
    for name in names:
        values = [
            {**partial, name: float(value)}
            for partial in values
            for value in PARAMETER_GRID[name]
        ]
    return sorted(parameters_sha256(value) for value in values)


def _manifest_pair_records(
    manifest: Mapping[str, Any], split: str | None = None
) -> list[tuple[str, str, dict[str, Any]]]:
    """Resolve model-specific sidecars while preserving legacy protocol support."""
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("calibration manifest has no datasets")
    if manifest.get("kind") == "calibration_protocol":
        records = []
        for model, dataset in LEGACY_PAIRS:
            record = datasets.get(dataset)
            if not isinstance(record, dict):
                raise ValueError(
                    f"calibration protocol has no target-free input root for {dataset}"
                )
            records.append((model, dataset, record))
        return records
    if manifest.get("kind") == "dl3dv_calibration_protocol":
        dataset = datasets.get("dl3dv")
        if not isinstance(dataset, Mapping):
            raise ValueError("DL3DV calibration protocol has no dataset record")
        splits = dataset.get("splits")
        if isinstance(splits, Mapping):
            split = TRAIN_SPLIT if split is None else split
            if split not in DL3DV_SPLITS:
                raise ValueError(f"DL3DV calibration split is invalid: {split}")
            split_record = splits.get(split)
            if not isinstance(split_record, Mapping):
                raise ValueError(f"DL3DV calibration protocol has no {split} record")
            representations = split_record.get("representations")
        else:
            # Accept only the old in-progress manifest shape for helper-level
            # compatibility.  A real split-aware plan below still requires both
            # frozen split records.
            representations = dataset.get("representations")
        if not isinstance(representations, Mapping):
            raise ValueError("DL3DV calibration protocol has no representation records")
        records = []
        for model, dataset_name, representation in DL3DV_PAIRS:
            record = representations.get(representation)
            if not isinstance(record, dict):
                raise ValueError(
                    f"DL3DV calibration protocol has no {representation} sidecar"
                )
            records.append((model, dataset_name, record))
        return records
    raise ValueError("calibration manifest kind is invalid")


def _manifest_split_record(manifest: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    datasets = manifest.get("datasets")
    dataset = datasets.get("dl3dv") if isinstance(datasets, Mapping) else None
    splits = dataset.get("splits") if isinstance(dataset, Mapping) else None
    record = splits.get(split) if isinstance(splits, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"DL3DV calibration protocol has no {split} record")
    return record


def _safe_relative_path(root: Path, relative: str, label: str) -> Path:
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"{label} escapes the protocol directory")
    path = (root / Path(*pure.parts)).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"{label} escapes the protocol directory")
    return path


def _sidecar_pair(
    *,
    manifest_path: Path,
    model: str,
    dataset: str,
    representation: str | None,
    dataset_record: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    protocol_root = manifest_path.parent.resolve()
    relative_root = dataset_record.get("calibration_input_root")
    if not isinstance(relative_root, str) or not relative_root:
        raise ValueError("calibration protocol has no target-free input root")
    prepared_root = _safe_relative_path(
        protocol_root, relative_root, "calibration input root"
    )
    if dataset_record.get("target_rgb_included") is not False:
        raise ValueError("calibration input tree contains target RGB")
    index_file = dataset_record.get("index_file")
    if not isinstance(index_file, str) or not index_file:
        raise ValueError("calibration protocol has no index file")
    index_path = _safe_relative_path(protocol_root, index_file, "calibration index")
    index_sha256 = _sha256(
        dataset_record.get("index_sha256"), "calibration index SHA256"
    )
    selections, summary = canonicalize_index(index_path, index_sha256)
    sample_count = dataset_record.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("calibration protocol has an invalid sample count")
    if len(selections) != sample_count:
        raise ValueError("calibration sample count does not match its index")
    input_identity = validate_target_free_input_root(prepared_root, dataset)
    expected_identity = {
        "calibration_input_tree_sha256": input_identity["tree_sha256"],
        "calibration_input_manifest_sha256": input_identity["manifest_sha256"],
        "calibration_input_provenance_sha256": input_identity[
            "input_provenance_sha256"
        ],
    }
    for field, actual in expected_identity.items():
        if dataset_record.get(field) != actual:
            raise ValueError(f"calibration input {field} mismatch")
    selection_sha256 = summary["sample_selection_sha256"]
    if dataset_record.get("selection_sha256") != selection_sha256:
        raise ValueError("calibration input selection hash mismatch")
    if input_identity["selection_sha256"] != selection_sha256:
        raise ValueError("calibration input selection hash mismatch")
    if input_identity["selected_scene_count"] != len(selections):
        raise ValueError("calibration input scene count mismatch")
    scene_names = {selection["scene"] for selection in selections}
    if len(scene_names) != len(selections):
        raise ValueError("calibration index repeats a scene")
    return (
        {
            "model": model,
            "dataset": dataset,
            "representation": representation,
            "sample_count": len(selections),
            "selection_sha256": selection_sha256,
            "scene_set_sha256": canonical_sha256(sorted(scene_names)),
            "calibration_input_root": relative_root,
            **expected_identity,
            "index_file": index_file,
            "index_sha256": index_sha256,
        },
        scene_names,
    )


def _pair_command(
    pair: Mapping[str, Any],
    output_dir: Path,
    *,
    split: str | None,
    committed_parameters: Mapping[str, Any] | None,
) -> list[str]:
    model = pair["model"]
    dataset = pair["dataset"]
    experiment = resolve_experiment(model, dataset, ROOT)
    python = _python_for(experiment.environment_profile, None)
    trace_parts = ["traces"]
    if split is not None:
        trace_parts.append(split)
    if committed_parameters is not None:
        trace_parts.append(f"committed-{parameters_sha256(committed_parameters)}")
    trace_parts.append(f"{model}_{dataset}")
    pair_dir = output_dir.joinpath(*trace_parts)
    demo = [
        python,
        str(ROOT / "scripts/demo.py"),
        "--model",
        model,
        "--dataset",
        dataset,
        "--checkpoint",
        str(experiment.checkpoint),
        "--dataset-root",
        str(_safe_relative_path(Path(pair["protocol_root"]), pair["calibration_input_root"], "calibration input root")),
        "--evaluation-index",
        str(_safe_relative_path(Path(pair["protocol_root"]), pair["index_file"], "calibration index")),
        "--diagnostic-run",
        "--calibration-trace",
        "--image-output-policy",
        "none",
        "--seed",
        "0",
    ]
    if committed_parameters is not None:
        demo.extend(
            (
                "--calibration-parameters",
                json.dumps(
                    canonical_parameters(committed_parameters),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
    return [
        python,
        str(ROOT / "scripts/run_pair.py"),
        "--evaluation-index",
        str(_safe_relative_path(Path(pair["protocol_root"]), pair["index_file"], "calibration index")),
        "--source-index-sha256",
        pair["index_sha256"],
        "--dataset-root",
        str(_safe_relative_path(Path(pair["protocol_root"]), pair["calibration_input_root"], "calibration input root")),
        "--output-dir",
        str(pair_dir),
        "--num-samples",
        str(pair["sample_count"]),
        "--resume",
        "--",
        *demo,
    ]


def _plan_pair(pair: dict[str, Any], protocol_root: Path) -> dict[str, Any]:
    return {**pair, "protocol_root": str(protocol_root)}


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    digest = _sha256(manifest.get("calibration_manifest_sha256"), "calibration manifest SHA256")
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key != "calibration_manifest_sha256"
    }
    if canonical_sha256(unsigned) != digest:
        raise ValueError("calibration manifest SHA256 does not match its content")
    return digest


def _build_legacy_plan(manifest_path: Path, manifest: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    pairs = []
    for model, dataset, dataset_record in _manifest_pair_records(manifest):
        pair, _ = _sidecar_pair(
            manifest_path=manifest_path,
            model=model,
            dataset=dataset,
            representation=None,
            dataset_record=dataset_record,
        )
        planned = _plan_pair(pair, manifest_path.parent)
        planned["command"] = _pair_command(
            planned, output_dir, split=None, committed_parameters=None
        )
        pairs.append(planned)
    return {
        "schema_version": "1.0",
        "kind": "calibration_sweep_plan",
        "manifest": str(manifest_path),
        "calibration_manifest_sha256": _manifest_digest(manifest),
        "pairs": pairs,
        "required_pairs": [f"{pair['model']}/{pair['dataset']}" for pair in pairs],
    }


def _build_dl3dv_plan(
    manifest_path: Path, manifest: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    if manifest.get("status") != "PASS":
        raise ValueError("DL3DV calibration protocol has not passed preparation")
    _sha256(manifest.get("download_plan_sha256"), "DL3DV download-plan SHA256")
    _sha256(
        manifest.get("archive_preparation_sha256"),
        "DL3DV archive-preparation SHA256",
    )
    if manifest.get("calibration_scene_count") != DL3DV_SPLIT_COUNTS[TRAIN_SPLIT]:
        raise ValueError("DL3DV calibration protocol has an invalid train scene count")
    if manifest.get("holdout_scene_count") != DL3DV_SPLIT_COUNTS[HOLDOUT_SPLIT]:
        raise ValueError("DL3DV calibration protocol has an invalid holdout scene count")
    split_plans: dict[str, dict[str, Any]] = {}
    split_scene_names: dict[str, set[str]] = {}
    for split in DL3DV_SPLITS:
        split_record = _manifest_split_record(manifest, split)
        expected_count = DL3DV_SPLIT_COUNTS[split]
        if split_record.get("scene_count") != expected_count:
            raise ValueError(
                f"DL3DV {split} must contain exactly {expected_count} scenes"
            )
        declared_scene_digest = _sha256(
            split_record.get("scene_set_sha256"), f"DL3DV {split} scene-set SHA256"
        )
        pairs: list[dict[str, Any]] = []
        pair_scene_names: list[set[str]] = []
        for model, dataset, dataset_record in _manifest_pair_records(manifest, split):
            representation = next(
                value
                for pair_model, _pair_dataset, value in DL3DV_PAIRS
                if pair_model == model
            )
            pair, scene_names = _sidecar_pair(
                manifest_path=manifest_path,
                model=model,
                dataset=dataset,
                representation=representation,
                dataset_record=dataset_record,
            )
            if pair["sample_count"] != expected_count:
                raise ValueError(
                    f"DL3DV {split} sidecar does not contain {expected_count} scenes"
                )
            if pair["scene_set_sha256"] != declared_scene_digest:
                raise ValueError("DL3DV split scene-set hash does not match its sidecar")
            pairs.append(_plan_pair(pair, manifest_path.parent))
            pair_scene_names.append(scene_names)
        if not pair_scene_names or any(scene_set != pair_scene_names[0] for scene_set in pair_scene_names[1:]):
            raise ValueError("DL3DV model sidecars do not select the same split scenes")
        selection_hashes = {pair["selection_sha256"] for pair in pairs}
        if len(selection_hashes) != 1:
            raise ValueError("DL3DV model sidecars do not share one split selection")
        split_scene_names[split] = pair_scene_names[0]
        split_plans[split] = {
            "split": split,
            "sample_count": expected_count,
            "selection_sha256": selection_hashes.pop(),
            "scene_set_sha256": declared_scene_digest,
            "required_pairs": [f"{pair['model']}/{pair['dataset']}" for pair in pairs],
            "pairs": pairs,
            "execution_mode": (
                "registered_global_grid"
                if split == TRAIN_SPLIT
                else "deferred_exact_train_selected_tuple"
            ),
        }
    overlap = split_scene_names[TRAIN_SPLIT] & split_scene_names[HOLDOUT_SPLIT]
    if overlap:
        raise ValueError(f"DL3DV train/holdout overlap: {sorted(overlap)[0]}")
    return {
        "schema_version": "2.0",
        "kind": "split_calibration_sweep_plan",
        "calibration_protocol": SPLIT_PROTOCOL,
        "manifest": str(manifest_path),
        "calibration_manifest_sha256": _manifest_digest(manifest),
        "evaluation_disjoint": True,
        "train_holdout_scene_disjoint": True,
        "splits": split_plans,
    }


def build_plan(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("calibration manifest must be an object")
    if manifest.get("evaluation_disjoint") is not True:
        raise ValueError("calibration manifest is not evaluation-disjoint")
    if manifest.get("kind") == "calibration_protocol":
        return _build_legacy_plan(manifest_path, manifest, output_dir)
    if manifest.get("kind") == "dl3dv_calibration_protocol":
        return _build_dl3dv_plan(manifest_path, manifest, output_dir)
    raise ValueError("calibration manifest kind is invalid")


def _stable_selection(record: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        key: record.get(key)
        for key in ("sample_index", "scene", "context_indices", "target_indices")
    }
    if (
        isinstance(values["sample_index"], bool)
        or not isinstance(values["sample_index"], int)
        or not isinstance(values["scene"], str)
        or not values["scene"]
        or not isinstance(values["context_indices"], list)
        or not isinstance(values["target_indices"], list)
    ):
        raise ValueError("calibration trace has an invalid sample selection")
    return values


def _pair_binding(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: pair[key]
        for key in (
            "model",
            "dataset",
            "representation",
            "sample_count",
            "selection_sha256",
            "scene_set_sha256",
            "calibration_input_root",
            "calibration_input_tree_sha256",
            "calibration_input_manifest_sha256",
            "calibration_input_provenance_sha256",
            "index_file",
            "index_sha256",
        )
    }


def _validate_trace_sidecar_identity(
    pair: Mapping[str, Any], trace: Mapping[str, Any]
) -> None:
    """Bind a replayed trace to the sidecar/index bytes it actually opened."""
    identity = trace.get("calibration_input")
    if not isinstance(identity, Mapping):
        raise ValueError("calibration trace has no sidecar provenance")
    required = (
        ("tree_sha256", "calibration_input_tree_sha256", "tree"),
        ("manifest_sha256", "calibration_input_manifest_sha256", "manifest"),
        (
            "input_provenance_sha256",
            "calibration_input_provenance_sha256",
            "provenance",
        ),
        ("selection_sha256", "selection_sha256", "selection"),
        ("index_sha256", "index_sha256", "index"),
        ("index_selection_sha256", "selection_sha256", "index selection"),
    )
    for trace_field, pair_field, label in required:
        if identity.get(trace_field) != pair[pair_field]:
            raise ValueError(f"calibration trace sidecar {label} hash mismatch")
    if identity.get("target_rgb_accessed") is not False:
        raise ValueError("calibration trace sidecar accessed target RGB")
    if identity.get("target_rgb_included") is not False:
        raise ValueError("calibration trace sidecar includes target RGB")
    if identity.get("selected_scene_count") != pair["sample_count"]:
        raise ValueError("calibration trace sidecar scene count mismatch")


def _validate_pair_trace_scope(
    pair: Mapping[str, Any],
    traces: list[dict[str, Any]],
    *,
    committed_parameters: Mapping[str, Any] | None,
) -> None:
    expected_scope = (
        "exact_committed_train_tuple"
        if committed_parameters is not None
        else "registered_global_grid"
    )
    expected_parameters = (
        canonical_parameters(committed_parameters)
        if committed_parameters is not None
        else None
    )
    expected_parameter_digest = (
        parameters_sha256(expected_parameters) if expected_parameters is not None else None
    )
    for trace_record in traces:
        if trace_record.get("kind") != "calibration_sample_trace":
            raise ValueError("calibration input contains a non-trace record")
        if (
            trace_record.get("model") != pair["model"]
            or trace_record.get("dataset") != pair["dataset"]
        ):
            raise ValueError("calibration trace model/dataset does not match its sidecar")
        if trace_record.get("calibration_scope") != expected_scope:
            raise ValueError("calibration trace has the wrong train/holdout scope")
        trace = trace_record.get("trace")
        if not isinstance(trace, Mapping):
            raise ValueError("calibration trace has no execution details")
        if trace.get("target_rgb_accessed") is not False:
            raise ValueError("calibration trace accessed target RGB")
        if trace.get("neural_forward_passes") != 1:
            raise ValueError("calibration sample did not execute one neural forward pass")
        if trace.get("candidate_scope") != expected_scope:
            raise ValueError("calibration trace candidate scope is inconsistent")
        _validate_trace_sidecar_identity(pair, trace)
        candidates = trace_record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("calibration trace has no candidates")
        if trace.get("candidate_count") != len(candidates):
            raise ValueError("calibration trace candidate count is inconsistent")
        candidate_parameters = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("calibration trace candidate is invalid")
            candidate_parameters.append(
                canonical_parameters(candidate.get("parameters", {}))
            )
        if committed_parameters is None:
            if len(candidates) != REGISTERED_GRID_SIZE or _candidate_parameter_set(candidates) != _registered_parameter_set():
                raise ValueError("training calibration trace does not contain the full registered grid")
            if trace_record.get("committed_parameters") is not None:
                raise ValueError("training calibration trace unexpectedly commits a tuple")
            if trace.get("candidate_parameters_sha256") is not None:
                raise ValueError("training calibration trace unexpectedly hashes one tuple")
        else:
            if len(candidates) != 1 or candidate_parameters != [expected_parameters]:
                raise ValueError(
                    "holdout calibration trace must contain exactly the committed train tuple"
                )
            if trace_record.get("committed_parameters") != expected_parameters:
                raise ValueError("holdout calibration trace does not bind the committed tuple")
            if trace.get("candidate_parameters_sha256") != expected_parameter_digest:
                raise ValueError("holdout calibration trace committed-tuple hash mismatch")


def _collect_split_traces(
    split_plan: Mapping[str, Any],
    output_dir: Path,
    *,
    committed_parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    split = split_plan["split"]
    traces: list[dict[str, Any]] = []
    trace_files: list[dict[str, str]] = []
    bindings: list[dict[str, Any]] = []
    for pair in split_plan["pairs"]:
        command = _pair_command(
            pair,
            output_dir,
            split=split,
            committed_parameters=committed_parameters,
        )
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError(
                f"calibration trace failed for {pair['model']}/{pair['dataset']}: "
                f"exit {result.returncode}"
            )
        trace_dir = Path(command[command.index("--output-dir") + 1])
        paths = sorted(trace_dir.glob("samples/sample_*/results.json"))
        if len(paths) != pair["sample_count"]:
            raise ValueError("calibration pair has an incomplete trace set")
        pair_traces = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        _validate_pair_trace_scope(
            pair, pair_traces, committed_parameters=committed_parameters
        )
        selection_rows = sorted(
            [_stable_selection(record) for record in pair_traces],
            key=lambda row: row["sample_index"],
        )
        if canonical_sha256(selection_rows) != pair["selection_sha256"]:
            raise ValueError("calibration trace selection hash mismatch")
        if canonical_sha256(sorted(row["scene"] for row in selection_rows)) != pair["scene_set_sha256"]:
            raise ValueError("calibration trace scene-set hash mismatch")
        binding = _pair_binding(pair)
        bindings.append(binding)
        for path, record in zip(paths, pair_traces):
            trace_files.append(
                {
                    "pair": f"{pair['model']}/{pair['dataset']}",
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256_file(path),
                }
            )
            traces.append(record)
    candidates = _with_candidate_hashes(aggregate_candidate_traces(traces))
    required_pairs = set(split_plan["required_pairs"])
    expected_candidate_count = 1 if committed_parameters is not None else REGISTERED_GRID_SIZE
    if len(candidates) != expected_candidate_count or any(
        set(candidate["quality"]) != required_pairs for candidate in candidates
    ):
        raise ValueError("calibration candidate matrix is incomplete")
    if committed_parameters is not None and _parameter_key(candidates[0]["parameters"]) != _parameter_key(committed_parameters):
        raise ValueError("holdout aggregation did not preserve the committed train tuple")
    return {
        "split": split,
        "execution_mode": (
            "exact_train_selected_tuple_no_rerank"
            if committed_parameters is not None
            else "registered_global_grid"
        ),
        "sample_count": split_plan["sample_count"],
        "selection_sha256": split_plan["selection_sha256"],
        "scene_set_sha256": split_plan["scene_set_sha256"],
        "required_pairs": list(split_plan["required_pairs"]),
        "pair_bindings": bindings,
        "pair_bindings_sha256": canonical_sha256(bindings),
        "trace_count": len(traces),
        "trace_set_sha256": canonical_sha256(traces),
        "trace_files": trace_files,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_set_sha256": canonical_sha256(
            [candidate["candidate_sha256"] for candidate in candidates]
        ),
        **(
            {
                "validated_parameters": canonical_parameters(committed_parameters),
                "validated_parameters_sha256": parameters_sha256(committed_parameters),
            }
            if committed_parameters is not None
            else {}
        ),
    }


def validate_exact_holdout_candidate(
    candidates: list[Mapping[str, Any]], committed_parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Accept/reject one committed tuple without selecting among holdout rows."""
    committed = canonical_parameters(committed_parameters)
    if len(candidates) != 1:
        raise ValueError(
            "holdout calibration must validate exactly one committed train tuple; reranking is forbidden"
        )
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise ValueError("holdout calibration candidate is invalid")
    if canonical_parameters(candidate.get("parameters", {})) != committed:
        raise ValueError("holdout calibration candidate does not match the committed train tuple")
    digest = candidate.get("candidate_sha256")
    if digest != candidate_sha256(candidate):
        raise ValueError("holdout calibration candidate hash mismatch")
    return candidate


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _execute_legacy(plan: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    pairs = []
    for pair in plan["pairs"]:
        command = list(pair["command"])
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError(
                f"calibration trace failed for {pair['model']}/{pair['dataset']}: "
                f"exit {result.returncode}"
            )
        trace_dir = Path(command[command.index("--output-dir") + 1])
        paths = sorted(trace_dir.glob("samples/sample_*/results.json"))
        if len(paths) != pair["sample_count"]:
            raise ValueError("calibration pair has an incomplete trace set")
        pairs.extend(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    candidates = aggregate_candidate_traces(pairs)
    required_pairs = set(plan["required_pairs"])
    if len(candidates) != REGISTERED_GRID_SIZE or any(
        set(candidate["quality"]) != required_pairs for candidate in candidates
    ):
        raise ValueError("calibration candidate matrix is incomplete")
    record = {
        "schema_version": "1.0",
        "kind": "calibration_candidate_records",
        "evaluation_disjoint": True,
        "calibration_manifest_sha256": plan["calibration_manifest_sha256"],
        "trace_count": len(pairs),
        "trace_set_sha256": canonical_sha256(pairs),
        "candidates": candidates,
    }
    _write_json(output_dir / "candidates.json", record)
    return record


def execute(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "plan.json", plan)
    if plan.get("kind") == "calibration_sweep_plan":
        return _execute_legacy(plan, output_dir)
    if plan.get("kind") != "split_calibration_sweep_plan":
        raise ValueError("calibration sweep plan kind is invalid")

    train = _collect_split_traces(
        plan["splits"][TRAIN_SPLIT], output_dir, committed_parameters=None
    )
    _write_json(output_dir / "train-candidates.json", train)
    base_record: dict[str, Any] = {
        "schema_version": "2.0",
        "kind": "split_calibration_candidate_records",
        "calibration_protocol": SPLIT_PROTOCOL,
        "evaluation_disjoint": True,
        "train_holdout_scene_disjoint": plan["train_holdout_scene_disjoint"],
        "calibration_manifest_sha256": plan["calibration_manifest_sha256"],
        "splits": {TRAIN_SPLIT: train},
    }
    try:
        selected = select_global_candidate(train["candidates"])
    except ValueError as exc:
        base_record.update(
            {
                "status": "TRAIN_NO_FEASIBLE_CANDIDATE",
                "train_selection": None,
            }
        )
        _write_json(output_dir / "candidates.json", base_record)
        raise ValueError("training calibration did not select a feasible tuple") from exc
    committed = canonical_parameters(selected["parameters"])
    train_selection = {
        "parameters": committed,
        "parameters_sha256": parameters_sha256(committed),
        "candidate_sha256": selected["candidate_sha256"],
        "selection_rule": "quality constraints, maximum event work reduction",
    }
    _write_json(output_dir / "train-selection.json", train_selection)
    holdout_request = {
        "schema_version": "1.0",
        "kind": "calibration_holdout_request",
        "calibration_manifest_sha256": plan["calibration_manifest_sha256"],
        "train_selection": train_selection,
        "execution_mode": "exact_train_selected_tuple_no_rerank",
    }
    _write_json(output_dir / "holdout-request.json", holdout_request)
    holdout = _collect_split_traces(
        plan["splits"][HOLDOUT_SPLIT],
        output_dir,
        committed_parameters=committed,
    )
    holdout_candidate = validate_exact_holdout_candidate(
        holdout["candidates"], committed
    )
    base_record.update(
        {
            "train_selection": train_selection,
            "splits": {TRAIN_SPLIT: train, HOLDOUT_SPLIT: holdout},
        }
    )
    if not candidate_is_feasible(holdout_candidate):
        base_record["status"] = "HOLDOUT_QUALITY_FAILED"
        _write_json(output_dir / "candidates.json", base_record)
        raise ValueError("holdout calibration rejected the committed tuple on quality")
    base_record["status"] = "PASS"
    _write_json(output_dir / "candidates.json", base_record)
    return base_record


def _safe_trace_path(records_dir: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise ValueError("calibration trace path is invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError("calibration trace path escapes the candidate directory")
    path = (records_dir / Path(*pure.parts)).resolve()
    if records_dir.resolve() not in path.parents:
        raise ValueError("calibration trace path escapes the candidate directory")
    return path


def _validate_pair_binding(binding: Any) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise ValueError("calibration sidecar binding is invalid")
    required = (
        "model",
        "dataset",
        "representation",
        "sample_count",
        "selection_sha256",
        "scene_set_sha256",
        "calibration_input_root",
        "calibration_input_tree_sha256",
        "calibration_input_manifest_sha256",
        "calibration_input_provenance_sha256",
        "index_file",
        "index_sha256",
    )
    if any(key not in binding for key in required):
        raise ValueError("calibration sidecar binding is incomplete")
    if not isinstance(binding["model"], str) or not isinstance(binding["dataset"], str):
        raise ValueError("calibration sidecar binding has no model/dataset")
    if binding["representation"] not in {"native", "re10k"}:
        raise ValueError("calibration sidecar binding representation is invalid")
    if (
        isinstance(binding["sample_count"], bool)
        or not isinstance(binding["sample_count"], int)
        or binding["sample_count"] <= 0
    ):
        raise ValueError("calibration sidecar binding sample count is invalid")
    for key in (
        "selection_sha256",
        "scene_set_sha256",
        "calibration_input_tree_sha256",
        "calibration_input_manifest_sha256",
        "calibration_input_provenance_sha256",
        "index_sha256",
    ):
        _sha256(binding[key], f"calibration sidecar {key}")
    if not isinstance(binding["calibration_input_root"], str) or not isinstance(binding["index_file"], str):
        raise ValueError("calibration sidecar binding paths are invalid")
    return {key: binding[key] for key in required}


def _validate_split_evidence(
    split_record: Any,
    records_dir: Path,
    *,
    expected_split: str,
    committed_parameters: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], set[str]]:
    if not isinstance(split_record, Mapping):
        raise ValueError(f"{expected_split} calibration evidence is missing")
    if split_record.get("split") != expected_split:
        raise ValueError("calibration split identity is invalid")
    expected_mode = (
        "exact_train_selected_tuple_no_rerank"
        if committed_parameters is not None
        else "registered_global_grid"
    )
    if split_record.get("execution_mode") != expected_mode:
        raise ValueError("calibration split execution mode is invalid")
    selection_sha = _sha256(
        split_record.get("selection_sha256"), "calibration split selection SHA256"
    )
    scene_sha = _sha256(
        split_record.get("scene_set_sha256"), "calibration split scene-set SHA256"
    )
    sample_count = split_record.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count != DL3DV_SPLIT_COUNTS[expected_split]
    ):
        raise ValueError("calibration split sample count is invalid")
    required_pairs = split_record.get("required_pairs")
    bindings_value = split_record.get("pair_bindings")
    if not isinstance(required_pairs, list) or not isinstance(bindings_value, list):
        raise ValueError("calibration split pair bindings are missing")
    bindings = [_validate_pair_binding(binding) for binding in bindings_value]
    pairs = [f"{binding['model']}/{binding['dataset']}" for binding in bindings]
    if len(bindings) != len(set(pairs)) or required_pairs != pairs:
        raise ValueError("calibration split pair bindings are inconsistent")
    if set(pairs) != {"transplat/dl3dv", "mvsplat/dl3dv", "depthsplat/dl3dv"}:
        raise ValueError("DL3DV calibration split does not cover every required pair")
    if any(binding["sample_count"] != sample_count for binding in bindings):
        raise ValueError("calibration split sidecar sample count mismatch")
    if any(binding["selection_sha256"] != selection_sha for binding in bindings):
        raise ValueError("calibration split sidecar selection mismatch")
    if any(binding["scene_set_sha256"] != scene_sha for binding in bindings):
        raise ValueError("calibration split sidecar scene-set mismatch")
    if split_record.get("pair_bindings_sha256") != canonical_sha256(bindings):
        raise ValueError("calibration split sidecar binding hash mismatch")

    trace_files = split_record.get("trace_files")
    if not isinstance(trace_files, list) or not trace_files:
        raise ValueError("calibration split trace files are missing")
    seen_paths: set[str] = set()
    traces: list[dict[str, Any]] = []
    expected_pair_traces: dict[str, list[dict[str, Any]]] = {pair: [] for pair in pairs}
    for item in trace_files:
        if not isinstance(item, Mapping):
            raise ValueError("calibration trace file record is invalid")
        pair = item.get("pair")
        relative = item.get("path")
        if not isinstance(pair, str) or pair not in expected_pair_traces or relative in seen_paths:
            raise ValueError("calibration trace file record is inconsistent")
        seen_paths.add(relative)
        path = _safe_trace_path(records_dir, relative)
        if not path.is_file() or item.get("sha256") != _sha256_file(path):
            raise ValueError("calibration trace file hash mismatch")
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("calibration trace file is invalid JSON") from exc
        if not isinstance(trace, dict):
            raise ValueError("calibration trace file is not an object")
        traces.append(trace)
        expected_pair_traces[pair].append(trace)
    if split_record.get("trace_count") != len(traces):
        raise ValueError("calibration split trace count mismatch")
    if split_record.get("trace_set_sha256") != canonical_sha256(traces):
        raise ValueError("calibration split trace-set hash mismatch")

    observed_scenes: set[str] = set()
    for binding in bindings:
        pair = f"{binding['model']}/{binding['dataset']}"
        pair_traces = expected_pair_traces[pair]
        if len(pair_traces) != sample_count:
            raise ValueError("calibration pair has an incomplete trace set")
        _validate_pair_trace_scope(
            binding, pair_traces, committed_parameters=committed_parameters
        )
        selections = sorted(
            [_stable_selection(trace) for trace in pair_traces],
            key=lambda row: row["sample_index"],
        )
        if canonical_sha256(selections) != binding["selection_sha256"]:
            raise ValueError("calibration trace selection hash mismatch")
        scenes = {selection["scene"] for selection in selections}
        if len(scenes) != sample_count or canonical_sha256(sorted(scenes)) != binding["scene_set_sha256"]:
            raise ValueError("calibration trace scene-set hash mismatch")
        if not observed_scenes:
            observed_scenes = scenes
        elif observed_scenes != scenes:
            raise ValueError("calibration pair traces do not share one scene set")

    recomputed = _with_candidate_hashes(aggregate_candidate_traces(traces))
    candidates = split_record.get("candidates")
    if not isinstance(candidates, list) or canonical_sha256(candidates) != canonical_sha256(recomputed):
        raise ValueError("calibration candidate records do not match their traces")
    if split_record.get("candidate_count") != len(candidates):
        raise ValueError("calibration candidate count mismatch")
    if split_record.get("candidate_set_sha256") != canonical_sha256(
        [candidate["candidate_sha256"] for candidate in candidates]
    ):
        raise ValueError("calibration candidate-set hash mismatch")
    expected_count = 1 if committed_parameters is not None else REGISTERED_GRID_SIZE
    if len(candidates) != expected_count:
        raise ValueError("calibration candidate matrix is incomplete")
    if any(set(candidate.get("quality", {})) != set(pairs) for candidate in candidates):
        raise ValueError("calibration candidate pair matrix is incomplete")
    if committed_parameters is not None:
        committed = canonical_parameters(committed_parameters)
        if split_record.get("validated_parameters") != committed:
            raise ValueError("holdout evidence does not bind the committed train tuple")
        if split_record.get("validated_parameters_sha256") != parameters_sha256(committed):
            raise ValueError("holdout committed-tuple hash mismatch")
        validate_exact_holdout_candidate(candidates, committed)
    return dict(split_record), observed_scenes


def validate_split_candidate_records(
    source: Mapping[str, Any], records_dir: Path
) -> dict[str, Any]:
    """Validate split-aware evidence before a mechanism configuration is frozen.

    This reopens the recorded target-free traces, recomputes aggregate candidate
    rows, and then checks the one holdout row directly.  It intentionally never
    invokes the global selector on holdout data.
    """
    if not isinstance(source, Mapping) or source.get("kind") != "split_calibration_candidate_records":
        raise ValueError("calibration candidate record is not a split-aware DL3DV record")
    if source.get("schema_version") != "2.0":
        raise ValueError("split calibration candidate record schema is invalid")
    if source.get("calibration_protocol") != SPLIT_PROTOCOL:
        raise ValueError("split calibration candidate protocol is invalid")
    if source.get("evaluation_disjoint") is not True:
        raise ValueError("calibration manifest is not evaluation-disjoint")
    if source.get("train_holdout_scene_disjoint") is not True:
        raise ValueError("calibration train/holdout split is not disjoint")
    _sha256(source.get("calibration_manifest_sha256"), "calibration manifest SHA256")
    splits = source.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("calibration candidate record has no split evidence")
    train, train_scenes = _validate_split_evidence(
        splits.get(TRAIN_SPLIT),
        records_dir,
        expected_split=TRAIN_SPLIT,
        committed_parameters=None,
    )
    selected = select_global_candidate(train["candidates"])
    committed = canonical_parameters(selected["parameters"])
    train_selection = source.get("train_selection")
    if not isinstance(train_selection, Mapping):
        raise ValueError("calibration candidate record has no committed train selection")
    if train_selection.get("parameters") != committed:
        raise ValueError("calibration train selection does not match the fixed grid result")
    if train_selection.get("parameters_sha256") != parameters_sha256(committed):
        raise ValueError("calibration train selection hash mismatch")
    if train_selection.get("candidate_sha256") != selected.get("candidate_sha256"):
        raise ValueError("calibration train selected-candidate hash mismatch")
    if train_selection.get("selection_rule") != "quality constraints, maximum event work reduction":
        raise ValueError("calibration train selection rule is invalid")
    holdout, holdout_scenes = _validate_split_evidence(
        splits.get(HOLDOUT_SPLIT),
        records_dir,
        expected_split=HOLDOUT_SPLIT,
        committed_parameters=committed,
    )
    if train_scenes & holdout_scenes:
        raise ValueError("calibration train/holdout traces overlap")
    holdout_candidate = validate_exact_holdout_candidate(
        holdout["candidates"], committed
    )
    if source.get("status") != "PASS":
        raise ValueError("calibration holdout did not pass")
    if not candidate_is_feasible(holdout_candidate):
        raise ValueError("calibration holdout quality gate failed")
    return {
        "parameters": committed,
        "selected_train_candidate": selected,
        "validated_holdout_candidate": holdout_candidate,
        "train": train,
        "holdout": holdout,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_plan(args.manifest, args.output_dir.resolve())
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        execute(plan, args.output_dir.resolve())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
