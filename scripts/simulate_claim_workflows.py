#!/usr/bin/env python3
"""Build explicitly non-claim rehearsals of the SCARF AE workflows.

The compact layout checks pair/sample coverage with a small number of files.
The file-tree layout additionally materializes the per-sample paths used by
the real export, but its values remain deterministic predictions rather than
measurements. Neither layout computes evidence hashes or passes claim checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("transplat", "mvsplat", "depthsplat")
DATASETS = ("re10k", "acid", "dl3dv")
WORKFLOWS = ("quality", "mechanisms", "speedup")
MARKERS = {
    "simulation_only": True,
    "claim_eligible": False,
    "hashes": "not_computed",
}
DEFAULT_QUALITY = {
    "baseline": {"psnr_db": 28.0, "ssim": 0.900, "lpips": 0.100},
    "scarf": {"psnr_db": 27.9, "ssim": 0.898, "lpips": 0.102},
}
DEFAULT_MECHANISM = {
    "guided_rate": 0.50,
    "level0_rate": 0.35,
    "level1_rate": 0.25,
}
DEFAULT_PERFORMANCE = {
    "orin_nx_normalized": 1.0,
    "scarf_dataflow_on_orin_nx_normalized": 1.25,
    "scarf_asic_speedup": 2.90,
}
CALIBRATION_SELECTED = {
    "gamma_depth": 0.075,
    "beta_x": 0.5,
    "beta_f": 0.1,
    "beta_d": 1.0,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _counts(root: Path, profile: str) -> dict[str, int]:
    if profile == "full":
        protocol = _load(root / "artifact/evaluation_protocol.json")
        return {
            dataset: int(protocol["pairs"][f"transplat/{dataset}"]["sample_count"])
            for dataset in DATASETS
        }
    if profile == "reviewer":
        manifest = _load(root / "artifact/protocol/reviewer/manifest.json")
        return {
            dataset: int(manifest["datasets"][dataset]["sample_count"])
            for dataset in DATASETS
        }
    raise ValueError(f"unsupported profile: {profile}")


def _pair_offset(model: str, dataset: str) -> int:
    return 1 + MODELS.index(model) * len(DATASETS) + DATASETS.index(dataset)


def _paper_targets(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load optional non-claim reference performance, with safe defaults.

    Rehearsal generation must not read paper expected-result files.  Quality
    and mechanism values therefore remain deterministic rehearsal defaults;
    only the explicitly named reference performance CSV is consulted when it
    is present.
    """
    quality: dict[str, dict[str, Any]] = {}
    mechanism: dict[str, dict[str, Any]] = {}
    performance: dict[str, dict[str, Any]] = {}
    reference_path = root / "artifact/reference_results/orin_nx_reference.csv"
    if reference_path.is_file():
        with reference_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                pair = row.get("pair")
                if not pair:
                    continue
                try:
                    performance[pair] = {
                        "orin_nx_normalized": float(row["orin_nx_normalized"]),
                        "scarf_dataflow_on_orin_nx_normalized": float(
                            row["scarf_dataflow_on_orin_nx_normalized"]
                        ),
                        "scarf_asic_speedup": float(row["scarf_asic_speedup"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
    return {"quality": quality, "mechanism": mechanism, "performance": performance}


def _quality_for_pair(pair: str, targets: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    values = targets["quality"].get(pair)
    if not values:
        return json.loads(json.dumps(DEFAULT_QUALITY))
    baseline = values["baseline"]
    scarf = values["scarf"]
    return {
        "baseline": {
            "psnr_db": float(baseline[0]),
            "ssim": float(baseline[1]),
            "lpips": float(baseline[2]),
        },
        "scarf": {
            "psnr_db": float(scarf[0]),
            "ssim": float(scarf[1]),
            "lpips": float(scarf[2]),
        },
    }


def _mechanism_for_pair(
    pair: str, targets: dict[str, dict[str, dict[str, float]]]
) -> dict[str, float]:
    values = targets["mechanism"].get(pair, DEFAULT_MECHANISM)
    return {key: float(values[key]) for key in ("guided_rate", "level0_rate", "level1_rate")}


def _performance_for_pair(
    pair: str, targets: dict[str, dict[str, dict[str, float]]]
) -> dict[str, float]:
    values = targets["performance"].get(pair, DEFAULT_PERFORMANCE)
    return {
        key: float(values[key])
        for key in (
            "orin_nx_normalized",
            "scarf_dataflow_on_orin_nx_normalized",
            "scarf_asic_speedup",
        )
    }


def _sample(
    model: str,
    dataset: str,
    sample_index: int,
    targets: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    pair = f"{model}/{dataset}"
    offset = _pair_offset(model, dataset)
    wobble = sample_index % 11
    stages = {
        "s1": 60 + offset,
        "s2": 140 + offset + wobble,
        "s3": 90 + offset,
        "s4": 110 + (sample_index % 3),
    }
    asic = sum(stages.values())
    return {
        "model": model,
        "dataset": dataset,
        "sample_index": sample_index,
        "scene": f"simulation/{dataset}/{sample_index:05d}",
        "context_indices": [0, 1],
        "target_indices": [2, 3, 4],
        **MARKERS,
        "quality": _quality_for_pair(pair, targets),
        "mechanism": _mechanism_for_pair(pair, targets),
        "performance_prediction": _performance_for_pair(pair, targets),
        "timing": {
            "timing_class": "simulation_prediction",
            "variants": {
                "asic": {"total_cycles": asic},
                "asic_fsdr": {"total_cycles": asic - 55},
                "asic_saes": {"total_cycles": asic - 65},
                "asic_fsdr_saes": {"total_cycles": asic - 95},
            },
            "combined_stage_cycles": stages,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def _reject_release_path(output: Path, *, root: Path = ROOT) -> Path:
    output = output.resolve()
    checkout_roots = {ROOT.resolve(), Path(root).resolve()}
    forbidden = tuple(
        checkout_root / name
        for checkout_root in checkout_roots
        for name in ("artifact", "outputs", "evidence")
    )
    if any(path == output or path in output.parents for path in forbidden):
        raise ValueError("simulation output must be outside artifact/, outputs/, and evidence/")
    return output


def _calibration_config() -> dict[str, Any]:
    return {
        "schema_version": "scarf-claim-simulation-v1",
        "kind": "simulation_mechanism_config",
        "status": "predicted",
        **MARKERS,
        "selected": CALIBRATION_SELECTED,
        "calibration": {
            "protocol": "dl3dv_train_holdout_v1",
            "train_count": 24,
            "holdout_count": 8,
            "source": "deterministic_simulation",
        },
    }


def _sample_name(index: int) -> str:
    return f"sample_{index:05d}"


def _trace_record(sample: dict[str, Any]) -> dict[str, Any]:
    timing = sample["timing"]
    cursor = 0
    events = []
    for stage, cycles in timing["combined_stage_cycles"].items():
        events.append(
            {
                "stage": stage,
                "start_cycle": cursor,
                "end_cycle": cursor + cycles,
                "accepted": True,
            }
        )
        cursor += cycles
    return {
        "schema_version": "scarf-claim-simulation-v1",
        "kind": "simulation_timing_trace",
        "cycle_accurate": False,
        **MARKERS,
        "model": sample["model"],
        "dataset": sample["dataset"],
        "sample_index": sample["sample_index"],
        "clock_mhz": 1000,
        "events": events,
        "variants": timing["variants"],
        "combined_stage_cycles": timing["combined_stage_cycles"],
    }


def _sample_result(workflow: str, sample: dict[str, Any], trace_path: str) -> dict[str, Any]:
    return {
        "schema_version": "scarf-claim-simulation-v1",
        "kind": "simulation_sample_result",
        "workflow": workflow,
        **MARKERS,
        "identity": {
            key: sample[key]
            for key in (
                "model",
                "dataset",
                "sample_index",
                "scene",
                "context_indices",
                "target_indices",
            )
        },
        "quality": sample["quality"],
        "mechanism": sample["mechanism"],
        "performance": sample["performance_prediction"],
        "timing": sample["timing"],
        "timing_trace": {"path": trace_path, "sha256": None},
    }


def _write_calibration_tree(output: Path) -> None:
    config = _calibration_config()
    _write_json(
        output / "calibration/manifest.json",
        {
            "schema_version": "scarf-claim-simulation-v1",
            "kind": "simulation_calibration_protocol",
            **MARKERS,
            "calibration_train_count": 24,
            "calibration_holdout_count": 8,
            "models": list(MODELS),
        },
    )
    for name in (
        "candidates.json",
        "train-candidates.json",
        "train-selection.json",
        "holdout-request.json",
        "candidate-template.json",
        "plan.json",
        "mechanism_config.json",
        "calibration-result.json",
    ):
        _write_json(output / "calibration" / name, config)
    for split, count in (("calibration_train", 24), ("calibration_holdout", 8)):
        for model in MODELS:
            for index in range(count):
                _write_json(
                    output
                    / "calibration/traces"
                    / split
                    / f"{model}_dl3dv"
                    / "samples"
                    / _sample_name(index)
                    / "results.json",
                    {
                        "schema_version": "scarf-claim-simulation-v1",
                        "kind": "simulation_calibration_trace",
                        **MARKERS,
                        "split": split,
                        "model": model,
                        "dataset": "dl3dv",
                        "sample_index": index,
                        "selected": CALIBRATION_SELECTED,
                    },
                )


def _write_environment_tree(output: Path) -> None:
    for profile in ("classic", "depthsplat", "orin"):
        _write_json(
            output / "environments" / f"{profile}.json",
            {
                "schema_version": "scarf-claim-simulation-v1",
                "kind": "simulation_environment",
                **MARKERS,
                "profile": profile,
                "runtime_capture": "not_run",
            },
        )


def _write_validation_fixtures(output: Path) -> None:
    _write_json(
        output / "validation-fixtures/calibration/calibration-fixture.json",
        {"schema_version": "scarf-claim-simulation-v1", **MARKERS},
    )
    _write_json(
        output / "validation-fixtures/calibration/selection.json",
        {
            "schema_version": "scarf-claim-simulation-v1",
            **MARKERS,
            "selected": CALIBRATION_SELECTED,
        },
    )
    _write_json(
        output / "validation-fixtures/timing/manifest.json",
        {"schema_version": "scarf-claim-simulation-v1", **MARKERS},
    )
    fixture_rtl = output / "validation-fixtures/timing/rtl/ScarfTop_validation_fixture.sv"
    fixture_rtl.parent.mkdir(parents=True, exist_ok=True)
    fixture_rtl.write_text("module ScarfTop_validation_fixture; endmodule\n", encoding="utf-8")
    _write_json(
        output / "validation-fixtures/timing/timing-trace/mvsplat/re10k/sample_00000.json",
        {"schema_version": "scarf-claim-simulation-v1", **MARKERS},
    )


def _write_tree_metadata(output: Path, counts: dict[str, int], profile: str) -> None:
    _write_json(
        output / "input-manifest.json",
        {
            "schema_version": "scarf-claim-simulation-v1",
            "kind": "simulation_input_manifest",
            **MARKERS,
            "profile": profile,
            "counts": counts,
            "pairs": [f"{model}/{dataset}" for model in MODELS for dataset in DATASETS],
            "workflows": list(WORKFLOWS),
        },
    )
    (output / "FORMAT_SPEC.md").write_text(
        "# Simulation File Tree\n\n"
        "This is a deterministic rehearsal layout. It is not a measurement export.\n",
        encoding="utf-8",
    )
    (output / "FIELD_REFERENCE.md").write_text(
        "# Simulation Fields\n\n"
        "All numeric values are deterministic predictions and hashes are not computed.\n",
        encoding="utf-8",
    )


def _write_orin_pair_files(output: Path, pair: str) -> None:
    profile = output / "speedup" / pair / "orin-profile"
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "tegrastats.log").write_text(
        "simulation_only=true; no Jetson device was sampled\n", encoding="utf-8"
    )
    (profile / "nsight.nsys-rep").write_bytes(b"simulation_only=true\n")


def _write_file_tree(
    output: Path,
    counts: dict[str, int],
    targets: dict[str, dict[str, dict[str, float]]],
    profile: str,
) -> dict[str, int]:
    _write_calibration_tree(output)
    _write_environment_tree(output)
    _write_validation_fixtures(output)
    _write_tree_metadata(output, counts, profile)
    rtl = output / "timing-backend/rtl/ScarfTop.sv"
    rtl.parent.mkdir(parents=True, exist_ok=True)
    rtl.write_text("module ScarfTop; endmodule\n", encoding="utf-8")
    manifest_path = output / "timing-backend/manifest.json"
    timing_entries: list[dict[str, Any]] = []
    result_file_count = 0
    orin_file_count = 0

    for model in MODELS:
        for dataset in DATASETS:
            pair = f"{model}_{dataset}"
            count = counts[dataset]
            samples: list[dict[str, Any]] = []
            trace_refs: list[dict[str, Any]] = []
            _write_orin_pair_files(output, pair)
            for index in range(count):
                sample = _sample(model, dataset, index, targets)
                sample_name = _sample_name(index)
                canonical_trace = (
                    output / "timing-backend/timing-trace" / model / dataset / f"{sample_name}.json"
                )
                _write_json(canonical_trace, _trace_record(sample))
                trace_rel = str(canonical_trace.relative_to(output))
                timing_entries.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "sample_index": index,
                        "trace": {"path": trace_rel, "sha256": None},
                        "variants": sample["timing"]["variants"],
                        "combined_stage_cycles": sample["timing"]["combined_stage_cycles"],
                    }
                )
                sample_ref = {"sample_index": index, "path": f"samples/{sample_name}/results.json"}
                samples.append(sample_ref)
                trace_refs.append({"sample_index": index, "path": trace_rel})
                for workflow in WORKFLOWS:
                    sample_root = output / workflow / pair / "samples" / sample_name
                    _write_json(
                        sample_root / "results.json",
                        _sample_result(workflow, sample, trace_rel),
                    )
                    result_file_count += 1
                    _write_link(
                        sample_root / "timing-trace" / model / dataset / f"{sample_name}.json",
                        canonical_trace,
                    )
                    _write_link(sample_root / "timing-backend/manifest.json", manifest_path)
                    _write_link(sample_root / "timing-backend/rtl/ScarfTop.sv", rtl)
                    _write_link(
                        sample_root
                        / "timing-backend/timing-trace"
                        / model
                        / dataset
                        / f"{sample_name}.json",
                        canonical_trace,
                    )
                    if workflow == "speedup":
                        orin_root = sample_root / "orin-evidence"
                        base_ms = 1.0 + _pair_offset(model, dataset) * 0.01 + (index % 7) * 0.001
                        cuda_samples = [
                            round(base_ms + offset, 3)
                            for offset in (-0.002, -0.001, 0, 0.001, 0.002)
                        ]
                        _write_json(
                            orin_root / "cuda-events.json",
                            {
                                "schema_version": "scarf-claim-simulation-v1",
                                "kind": "simulation_cuda_events",
                                **MARKERS,
                                "samples_ms": cuda_samples,
                                "median_ms": cuda_samples[2],
                            },
                        )
                        _write_json(
                            orin_root / "measurement.json",
                            {
                                "schema_version": "scarf-claim-simulation-v1",
                                "kind": "simulation_orin_measurement",
                                **MARKERS,
                                "sample_index": index,
                                "encoder_median_ms": cuda_samples[2],
                                "cuda_events": {"path": "cuda-events.json", "sha256": None},
                                "tegrastats": {
                                    "path": "../../../orin-profile/tegrastats.log",
                                    "sha256": None,
                                },
                                "nsight": {
                                    "path": "../../../orin-profile/nsight.nsys-rep",
                                    "sha256": None,
                                },
                            },
                        )
                        orin_file_count += 2
            for workflow in WORKFLOWS:
                pair_root = output / workflow / pair
                _write_json(
                    pair_root / "results.json",
                    {
                        "schema_version": "scarf-claim-simulation-v1",
                        "kind": "simulation_dataset_aggregate",
                        "workflow": workflow,
                        **MARKERS,
                        "model": model,
                        "dataset": dataset,
                        "profile": profile,
                        "sample_count": count,
                        "quality": _quality_for_pair(f"{model}/{dataset}", targets),
                        "mechanism": _mechanism_for_pair(f"{model}/{dataset}", targets),
                        "performance": _performance_for_pair(f"{model}/{dataset}", targets),
                        "samples": samples,
                    },
                )
                _write_json(
                    pair_root / "pair-execution.json",
                    {
                        "schema_version": "scarf-claim-simulation-v1",
                        "kind": "simulation_pair_execution",
                        **MARKERS,
                        "workflow": workflow,
                        "model": model,
                        "dataset": dataset,
                        "sample_count": count,
                        "executed": count,
                        "resumed": 0,
                        "complete": count,
                    },
                )
                (pair_root / "progress.jsonl").write_text(
                    json.dumps({"event": "pair_start", **MARKERS, "sample_count": count})
                    + "\n"
                    + json.dumps({"event": "pair_complete", **MARKERS, "complete": count})
                    + "\n",
                    encoding="utf-8",
                )
                _write_json(
                    pair_root / "timing-trace/aggregate.json",
                    {
                        "schema_version": "scarf-claim-simulation-v1",
                        "kind": "simulation_timing_aggregate",
                        **MARKERS,
                        "model": model,
                        "dataset": dataset,
                        "samples": trace_refs,
                    },
                )
            _write_json(
                output / "speedup" / pair / "orin-profile/pair-measurement.json",
                {
                    "schema_version": "scarf-claim-simulation-v1",
                    "kind": "simulation_orin_pair_measurement",
                    **MARKERS,
                    "model": model,
                    "dataset": dataset,
                    "sample_count": count,
                },
            )

    _write_json(
        manifest_path,
        {
            "schema_version": "scarf-claim-simulation-v1",
            "kind": "simulation_timing_backend",
            **MARKERS,
            "clock_mhz": 1000,
            "source": {"path": "rtl/ScarfTop.sv", "sha256": None},
            "sample_count": len(timing_entries),
            "samples": timing_entries,
        },
    )
    _write_json(
        output / "validation.json",
        {
            "schema_version": "scarf-claim-simulation-v1",
            "kind": "simulation_validation",
            "status": "SIMULATION_PASS",
            **MARKERS,
            "profile": profile,
            "checks": ["path_materialization", "sample_count_coverage"],
        },
    )
    _write_json(
        output / "progress-event-examples.json",
        {
            "schema_version": "scarf-claim-simulation-v1",
            **MARKERS,
            "events": ["pair_start", "pair_complete"],
        },
    )
    _write_json(
        output / "FILE_MANIFEST.json",
        {
            "schema_version": "scarf-claim-simulation-v1",
            "kind": "simulation_file_tree_manifest",
            **MARKERS,
            "profile": profile,
            "counts": counts,
            "workflows": list(WORKFLOWS),
            "sample_path": "<workflow>/<model>_<dataset>/samples/sample_<index:05d>/results.json",
            "stage_copies": "relative_symlinks",
        },
    )
    return {
        "sample_result_files": result_file_count,
        "orin_measurement_files": orin_file_count,
        "timing_trace_files": len(timing_entries),
    }


def build(
    output: Path,
    *,
    root: Path = ROOT,
    profile: str = "full",
    layout: str = "compact",
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = _reject_release_path(output, root=root)
    counts = _counts(root, profile)
    targets = _paper_targets(root)
    output.mkdir(parents=True, exist_ok=True)
    pair_files: dict[str, dict[str, str]] = {}
    predicted_quality: dict[str, Any] = {}
    predicted_mechanism: dict[str, Any] = {}
    predicted_performance: dict[str, Any] = {}
    timing_samples: list[dict[str, Any]] = []
    total_samples = 0
    for model in MODELS:
        for dataset in DATASETS:
            pair = f"{model}/{dataset}"
            count = counts[dataset]
            samples = [_sample(model, dataset, index, targets) for index in range(count)]
            total_samples += count
            result = {
                "schema_version": "scarf-claim-simulation-v1",
                "pair": pair,
                "profile": profile,
                "sample_count": count,
                **MARKERS,
                "samples": samples,
            }
            result_path = output / "results" / f"{model}_{dataset}.json"
            _write_json(result_path, result)
            pair_files[pair] = {"path": str(result_path.relative_to(output))}
            predicted_quality[pair] = samples[0]["quality"]
            predicted_mechanism[pair] = samples[0]["mechanism"]
            predicted_performance[pair] = samples[0]["performance_prediction"]
            timing_samples.extend(
                {
                    "model": item["model"],
                    "dataset": item["dataset"],
                    "sample_index": item["sample_index"],
                    "timing": item["timing"],
                    **MARKERS,
                }
                for item in samples
            )

    timing_path = output / "timing-backend" / "predicted-manifest.json"
    _write_json(
        timing_path,
        {
            "schema_version": "scarf-claim-simulation-v1",
            "kind": "simulation_prediction",
            **MARKERS,
            "clock_mhz": 1000,
            "sample_count": len(timing_samples),
            "samples": timing_samples,
        },
    )
    calibration_path = output / "calibration" / "predicted-mechanism-config.json"
    _write_json(calibration_path, _calibration_config())
    predictions_path = output / "workflow-predictions.json"
    _write_json(
        predictions_path,
        {
            "schema_version": "scarf-claim-simulation-v1",
            "kind": "predicted_workflow_summary",
            **MARKERS,
            "quality": predicted_quality,
            "mechanism": predicted_mechanism,
            "performance": predicted_performance,
        },
    )
    file_tree = (
        _write_file_tree(output, counts, targets, profile)
        if layout == "file-tree"
        else None
    )
    manifest = {
        "schema_version": "scarf-claim-simulation-v1",
        "status": "simulation_only",
        "profile": profile,
        "layout": layout,
        "expected_sample_counts": counts,
        "total_samples": total_samples,
        **MARKERS,
        "timing_manifest": {"path": str(timing_path.relative_to(output))},
        "calibration_config": {"path": str(calibration_path.relative_to(output))},
        "workflow_predictions": {"path": str(predictions_path.relative_to(output))},
        "pair_results": pair_files,
        "file_tree": file_tree,
        "replacement_rule": (
            "Replace every simulation file with a real platform record, then run the formal validators."
        ),
    }
    _write_json(output / "simulation-manifest.json", manifest)
    (output / "README.md").write_text(
        "# SCARF claim workflow simulation\n\n"
        "This tree is a deterministic local rehearsal only. Its values are\n"
        "predictions seeded from checked-in target tables when available, not\n"
        "measurements. Hashes are deliberately not computed. The formal claim\n"
        "loader rejects every record because `simulation_only` is true and\n"
        "`claim_eligible` is false.\n\n"
        f"Layout: `{layout}`. The `file-tree` layout uses relative symlinks for\n"
        "repeated staged timing files to avoid copying the same simulation data.\n",
        encoding="utf-8",
    )
    return manifest


def _require_simulation_record(path: Path, description: str) -> dict[str, Any]:
    record = _load(path)
    if record.get("simulation_only") is not True or record.get("claim_eligible") is not False:
        raise ValueError(f"simulation markers missing: {description}")
    return record


def _validate_file_tree(path: Path, counts: dict[str, int], total_samples: int) -> None:
    metadata = _require_simulation_record(path / "input-manifest.json", "input manifest")
    if metadata.get("profile") not in {"full", "reviewer"}:
        raise ValueError("file-tree input manifest profile is invalid")
    _require_simulation_record(path / "validation.json", "simulation validation")
    top_level = (
        "README.md",
        "FORMAT_SPEC.md",
        "FIELD_REFERENCE.md",
        "FILE_MANIFEST.json",
        "progress-event-examples.json",
    )
    if not all((path / name).is_file() for name in top_level):
        raise ValueError("file-tree top-level metadata is incomplete")
    calibration_files = (
        "manifest.json",
        "candidates.json",
        "train-candidates.json",
        "train-selection.json",
        "holdout-request.json",
        "candidate-template.json",
        "plan.json",
        "mechanism_config.json",
        "calibration-result.json",
    )
    if not all((path / "calibration" / name).is_file() for name in calibration_files):
        raise ValueError("file-tree calibration metadata is incomplete")
    if not all(
        (path / "environments" / f"{profile}.json").is_file()
        for profile in ("classic", "depthsplat", "orin")
    ):
        raise ValueError("file-tree environment records are incomplete")
    fixtures = (
        "calibration/calibration-fixture.json",
        "calibration/selection.json",
        "timing/manifest.json",
        "timing/rtl/ScarfTop_validation_fixture.sv",
        "timing/timing-trace/mvsplat/re10k/sample_00000.json",
    )
    if not all((path / "validation-fixtures" / fixture).is_file() for fixture in fixtures):
        raise ValueError("file-tree validation fixtures are incomplete")
    timing = _require_simulation_record(path / "timing-backend/manifest.json", "timing manifest")
    if timing.get("sample_count") != total_samples:
        raise ValueError("file-tree timing sample count mismatch")
    for model in MODELS:
        for dataset in DATASETS:
            pair = f"{model}_{dataset}"
            count = counts[dataset]
            for workflow in WORKFLOWS:
                pair_root = path / workflow / pair
                aggregate = _require_simulation_record(
                    pair_root / "results.json", f"{workflow}/{pair}"
                )
                if aggregate.get("sample_count") != count:
                    raise ValueError(f"file-tree aggregate sample count mismatch: {workflow}/{pair}")
                if not (
                    (pair_root / "pair-execution.json").is_file()
                    and (pair_root / "progress.jsonl").is_file()
                ):
                    raise ValueError(f"file-tree execution lineage missing: {workflow}/{pair}")
                for index in range(count):
                    sample_name = _sample_name(index)
                    sample_root = pair_root / "samples" / sample_name
                    required = (
                        sample_root / "results.json",
                        sample_root / "timing-trace" / model / dataset / f"{sample_name}.json",
                        sample_root / "timing-backend/manifest.json",
                        sample_root / "timing-backend/rtl/ScarfTop.sv",
                        sample_root
                        / "timing-backend/timing-trace"
                        / model
                        / dataset
                        / f"{sample_name}.json",
                    )
                    if not all(item.is_file() for item in required):
                        raise ValueError(f"file-tree sample lineage missing: {workflow}/{pair}/{sample_name}")
                    if workflow == "speedup" and not all(
                        item.is_file()
                        for item in (
                            sample_root / "orin-evidence/measurement.json",
                            sample_root / "orin-evidence/cuda-events.json",
                        )
                    ):
                        raise ValueError(f"file-tree Orin rehearsal missing: {pair}/{sample_name}")
                    canonical = (
                        path
                        / "timing-backend/timing-trace"
                        / model
                        / dataset
                        / f"{sample_name}.json"
                    )
                    if not canonical.is_file():
                        raise ValueError(f"file-tree timing trace missing: {model}/{dataset}/{sample_name}")
            profile = path / "speedup" / pair / "orin-profile"
            pair_files = ("tegrastats.log", "nsight.nsys-rep", "pair-measurement.json")
            if not all((profile / name).is_file() for name in pair_files):
                raise ValueError(f"file-tree Orin pair evidence missing: {pair}")
    for split, count in (("calibration_train", 24), ("calibration_holdout", 8)):
        for model in MODELS:
            for index in range(count):
                trace = (
                    path
                    / "calibration/traces"
                    / split
                    / f"{model}_dl3dv"
                    / "samples"
                    / _sample_name(index)
                    / "results.json"
                )
                if not trace.is_file():
                    raise ValueError(f"file-tree calibration trace missing: {split}/{model}/{index}")


def validate(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest = _load(path / "simulation-manifest.json")
    if manifest.get("status") != "simulation_only" or manifest.get("claim_eligible") is not False:
        raise ValueError("simulation manifest is missing non-claim markers")
    if manifest.get("hashes") != "not_computed":
        raise ValueError("simulation manifest must not carry evidence hashes")
    counts = manifest.get("expected_sample_counts")
    if not isinstance(counts, dict) or set(counts) != set(DATASETS):
        raise ValueError("simulation sample-count contract is incomplete")
    pair_results = manifest.get("pair_results")
    if not isinstance(pair_results, dict) or len(pair_results) != 9:
        raise ValueError("simulation does not cover all nine pairs")
    for pair, descriptor in pair_results.items():
        result = _require_simulation_record(path / descriptor["path"], pair)
        expected = int(counts[pair.split("/", 1)[1]])
        if result.get("sample_count") != expected or len(result.get("samples", [])) != expected:
            raise ValueError(f"simulation sample count mismatch: {pair}")
    timing = _require_simulation_record(
        path / manifest["timing_manifest"]["path"], "timing prediction"
    )
    if timing.get("sample_count") != manifest["total_samples"]:
        raise ValueError("simulation timing sample count mismatch")
    calibration = _require_simulation_record(
        path / manifest["calibration_config"]["path"], "calibration prediction"
    )
    if calibration.get("status") != "predicted":
        raise ValueError("simulation calibration marker is invalid")
    predictions = _require_simulation_record(
        path / manifest["workflow_predictions"]["path"], "workflow prediction"
    )
    for field in ("quality", "mechanism", "performance"):
        values = predictions.get(field)
        if not isinstance(values, dict) or set(values) != set(pair_results):
            raise ValueError(f"simulation workflow prediction coverage is incomplete: {field}")
    if manifest.get("layout") == "file-tree":
        _validate_file_tree(path, counts, int(manifest["total_samples"]))
    return {
        "schema_version": "scarf-claim-simulation-v1",
        "status": "PASS",
        "profile": manifest["profile"],
        "layout": manifest["layout"],
        "pairs": len(pair_results),
        "total_samples": manifest["total_samples"],
        "claim_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--root", type=Path, default=ROOT)
    build_parser.add_argument("--profile", choices=("full", "reviewer"), default="full")
    build_parser.add_argument("--layout", choices=("compact", "file-tree"), default="compact")
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            manifest = build(
                args.output_dir,
                root=args.root,
                profile=args.profile,
                layout=args.layout,
            )
            print(
                f"PASS: built {args.layout} simulation-only rehearsal "
                f"({manifest['total_samples']} samples, claim_eligible=false, hashes=not_computed)"
            )
        else:
            result = validate(args.input_dir)
            print(
                f"PASS: validated {result['layout']} simulation-only rehearsal "
                f"({result['total_samples']} samples, claim_eligible=false)"
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
