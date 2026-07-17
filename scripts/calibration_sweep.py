#!/usr/bin/env python3
"""Run and aggregate evaluation-disjoint SCARF calibration traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ae_config import resolve_experiment
from scripts.calibration_contract import PARAMETER_GRID, canonical_sha256
from scripts.calibration_inputs import validate_target_free_input_root
from scripts.compile_protocol import canonicalize_index
from scripts.run_ae import _python_for


PAIRS = tuple(
    (model, dataset)
    for model in ("transplat", "mvsplat", "depthsplat")
    for dataset in ("re10k", "acid")
)


def _parameter_key(parameters: dict[str, Any]) -> tuple[float, ...]:
    if set(parameters) != set(PARAMETER_GRID):
        raise ValueError("calibration trace has an incomplete parameter tuple")
    return tuple(float(parameters[name]) for name in PARAMETER_GRID)


def aggregate_candidate_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not traces:
        raise ValueError("calibration trace set is empty")
    grouped: dict[tuple[float, ...], dict[str, Any]] = {}
    for trace in traces:
        if trace.get("kind") != "calibration_sample_trace":
            raise ValueError("calibration input contains a non-trace record")
        if trace.get("trace", {}).get("neural_forward_passes") != 1:
            raise ValueError("calibration sample did not execute one neural forward pass")
        pair = f"{trace.get('model')}/{trace.get('dataset')}"
        for candidate in trace.get("candidates", []):
            key = _parameter_key(candidate.get("parameters", {}))
            entry = grouped.setdefault(
                key,
                {
                    "parameters": dict(candidate["parameters"]),
                    "quality": {},
                    "work": [],
                    "compression_values": [],
                },
            )
            pair_quality = entry["quality"].setdefault(pair, [])
            pair_quality.append(dict(candidate["quality"]))
            entry["work"].append(float(candidate["work_reduction"]))
            entry["compression_values"].append(float(candidate["compression"]))

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


def build_plan(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "calibration_protocol":
        raise ValueError("calibration manifest kind is invalid")
    if manifest.get("evaluation_disjoint") is not True:
        raise ValueError("calibration manifest is not evaluation-disjoint")
    pairs = []
    for model, dataset in PAIRS:
        dataset_record = manifest["datasets"][dataset]
        relative_root = dataset_record.get("calibration_input_root")
        if not isinstance(relative_root, str) or not relative_root:
            raise ValueError("calibration protocol has no target-free input root")
        prepared_root = (manifest_path.parent / relative_root).resolve()
        if manifest_path.parent not in prepared_root.parents:
            raise ValueError("calibration input root escapes the protocol directory")
        if dataset_record.get("target_rgb_included") is not False:
            raise ValueError("calibration input tree contains target RGB")
        index_file = dataset_record.get("index_file")
        if not isinstance(index_file, str) or not index_file:
            raise ValueError("calibration protocol has no index file")
        index_path = manifest_path.parent / index_file
        selections, summary = canonicalize_index(
            index_path, dataset_record["index_sha256"]
        )
        if len(selections) != dataset_record["sample_count"]:
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
        if input_identity["selection_sha256"] != summary["sample_selection_sha256"]:
            raise ValueError("calibration input selection hash mismatch")
        if input_identity["selected_scene_count"] != len(selections):
            raise ValueError("calibration input scene count mismatch")
        experiment = resolve_experiment(model, dataset, ROOT)
        python = _python_for(experiment.environment_profile, None)
        pair_dir = output_dir / "traces" / f"{model}_{dataset}"
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
            str(prepared_root),
            "--evaluation-index",
            str(index_path),
            "--diagnostic-run",
            "--calibration-trace",
            "--image-output-policy",
            "none",
            "--seed",
            "0",
        ]
        command = [
            python,
            str(ROOT / "scripts/run_pair.py"),
            "--evaluation-index",
            str(index_path),
            "--source-index-sha256",
            dataset_record["index_sha256"],
            "--dataset-root",
            str(prepared_root),
            "--output-dir",
            str(pair_dir),
            "--num-samples",
            str(len(selections)),
            "--resume",
            "--",
            *demo,
        ]
        pairs.append(
            {
                "model": model,
                "dataset": dataset,
                "sample_count": len(selections),
                "selection_sha256": summary["sample_selection_sha256"],
                "trace_dir": str(pair_dir),
                "command": command,
            }
        )
    return {
        "schema_version": "1.0",
        "kind": "calibration_sweep_plan",
        "manifest": str(manifest_path),
        "calibration_manifest_sha256": manifest["calibration_manifest_sha256"],
        "pairs": pairs,
    }


def execute(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    traces = []
    trace_files = []
    for pair in plan["pairs"]:
        result = subprocess.run(pair["command"], cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError(
                f"calibration trace failed for {pair['model']}/{pair['dataset']}: "
                f"exit {result.returncode}"
            )
        paths = sorted(Path(pair["trace_dir"]).glob("samples/sample_*/results.json"))
        if len(paths) != pair["sample_count"]:
            raise ValueError("calibration pair has an incomplete trace set")
        pair_traces = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if any(
            record.get("trace", {}).get("target_rgb_accessed") is not False
            for record in pair_traces
        ):
            raise ValueError("calibration trace accessed target RGB")
        selection_rows = sorted(
            [
                {
                    key: record[key]
                    for key in (
                        "sample_index",
                        "scene",
                        "context_indices",
                        "target_indices",
                    )
                }
                for record in pair_traces
            ],
            key=lambda row: row["sample_index"],
        )
        if canonical_sha256(selection_rows) != pair["selection_sha256"]:
            raise ValueError("calibration trace selection hash mismatch")
        for path, record in zip(paths, pair_traces):
            trace_files.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            traces.append(record)
    candidates = aggregate_candidate_traces(traces)
    required_pairs = {f"{model}/{dataset}" for model, dataset in PAIRS}
    if len(candidates) != 108 or any(
        set(candidate["quality"]) != required_pairs for candidate in candidates
    ):
        raise ValueError("calibration candidate matrix is incomplete")
    record = {
        "schema_version": "1.0",
        "kind": "calibration_candidate_records",
        "evaluation_disjoint": True,
        "calibration_manifest_sha256": plan["calibration_manifest_sha256"],
        "trace_count": len(traces),
        "trace_set_sha256": canonical_sha256(traces),
        "trace_files": trace_files,
        "candidates": candidates,
    }
    (output_dir / "candidates.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


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
