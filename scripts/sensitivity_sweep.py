#!/usr/bin/env python3
"""Run the paper's complete SCARF sensitivity matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ae_config import (
    CLAIMED_MATRIX,
    resolve_claim_selection,
    resolve_experiment,
)
from scripts.aggregate_results import _mean_tree
from scripts.result_record import build_quality_record, sha256_file
from scripts.run_ae import _protocol_sample_count, _python_for


STUDIES: dict[str, dict[str, Any]] = {
    "fsdr_cache_size": {
        "option": "--fsdr-cache-size",
        "values": (8, 16, 32, 64, 128),
        "default": 32,
    },
    "fsdr_hamming_threshold": {
        "option": "--fsdr-hamming",
        "values": (1, 2, 3, 4, 5),
        "default": 3,
    },
    "saes_feature_variance": {
        "option": "--saes-fv",
        "values": (0.1, 0.2, 0.3, 0.4, 0.5),
        "default": 0.2,
    },
    "saes_depth_variance": {
        "option": "--saes-ds",
        "values": (0.01, 0.05, 0.1, 0.2, 0.5),
        "default": 0.1,
    },
    "saes_tile_size": {
        "option": "--tile-size",
        "values": (2, 4, 8, 16, 32),
        "default": 4,
    },
}


def build_plan(
    output_dir: Path,
    *,
    python_override: str | None = None,
    num_samples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    pair_traces = []
    for model, dataset in CLAIMED_MATRIX:
        sample_count = (
            num_samples
            if num_samples is not None
            else _protocol_sample_count(model, dataset)
        )
        experiment = resolve_experiment(model, dataset, ROOT)
        selection = resolve_claim_selection(model, dataset, ROOT)
        profile_python = _python_for(experiment.environment_profile, python_override)
        trace_dir = output_dir / "traces" / f"{model}_{dataset}"
        demo_command = [
            profile_python,
            str(SCRIPT_DIR / "demo.py"),
            "--model",
            model,
            "--dataset",
            dataset,
            "--checkpoint",
            str(experiment.checkpoint),
            "--dataset-root",
            str(experiment.dataset_root),
            "--evaluation-index",
            str(selection.index_path),
            "--claim-run",
            "--device",
            "auto",
            "--seed",
            str(seed),
            "--sensitivity-trace",
        ]
        command = [
            profile_python,
            str(SCRIPT_DIR / "run_pair.py"),
            "--evaluation-index",
            str(selection.index_path),
            "--source-index-sha256",
            selection.source_index_sha256,
            "--output-dir",
            str(trace_dir),
            "--num-samples",
            str(sample_count),
            "--resume",
            "--",
            *demo_command,
        ]
        pair_traces.append(
            {
                "model": model,
                "dataset": dataset,
                "environment_profile": experiment.environment_profile,
                "sample_count": sample_count,
                "declared_sample_selection_sha256": selection.sample_selection_sha256,
                "trace_dir": str(trace_dir),
                "command": command,
            }
        )

    runs = [
        {
            "study": study,
            "parameter": config["option"],
            "value": value,
            "is_default": value == config["default"],
            "model": model,
            "dataset": dataset,
        }
        for study, config in STUDIES.items()
        for value in config["values"]
        for model, dataset in CLAIMED_MATRIX
    ]
    return {
        "schema_version": "1.0",
        "kind": "sensitivity_plan",
        "grids": {
            name: {"values": list(cfg["values"]), "default": cfg["default"]}
            for name, cfg in STUDIES.items()
        },
        "expected_runs": len(runs),
        "pair_traces": pair_traces,
        "runs": runs,
    }


def _mean(records: list[dict[str, Any]], *path: str) -> float:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(float(value))
    return sum(values) / len(values)


def aggregate_pair_traces(pair: dict[str, Any]) -> tuple[dict[tuple[str, Any], Any], dict]:
    trace_dir = Path(pair["trace_dir"])
    paths = sorted(trace_dir.glob("samples/sample_*/results.json"))
    if len(paths) != pair["sample_count"]:
        raise ValueError(
            f"{pair['model']}/{pair['dataset']} expected {pair['sample_count']} traces, "
            f"found {len(paths)}"
        )
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(record.get("kind") != "sensitivity_sample_trace" for record in traces):
        raise ValueError("sensitivity input contains a non-trace result")
    execution_indices = sorted(record.get("execution_index") for record in traces)
    if execution_indices != list(range(pair["sample_count"])):
        raise ValueError("sensitivity trace execution indices are incomplete")
    selections = [
        {
            key: record[key]
            for key in ("sample_index", "scene", "context_indices", "target_indices")
        }
        for record in traces
    ]
    selections.sort(key=lambda item: item["sample_index"])
    import hashlib

    selection_hash = hashlib.sha256(
        json.dumps(selections, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    full_claim = pair["sample_count"] == _protocol_sample_count(
        pair["model"], pair["dataset"]
    )
    if full_claim and selection_hash != pair["declared_sample_selection_sha256"]:
        raise ValueError("sensitivity trace selection hash does not match the protocol")
    if any(record["trace"].get("neural_forward_passes") != 1 for record in traces):
        raise ValueError("sensitivity trace did not preserve one neural forward pass")

    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for trace in traces:
        if trace.get("model") != pair["model"] or trace.get("dataset") != pair["dataset"]:
            raise ValueError("sensitivity trace model/dataset provenance mismatch")
        for replay in trace.get("replays", []):
            grouped.setdefault((replay["study"], replay["value"]), []).append(replay)
    expected_keys = {
        (study, value)
        for study, config in STUDIES.items()
        for value in config["values"]
    }
    if set(grouped) != expected_keys:
        raise ValueError("sensitivity trace parameter grid is incomplete")

    aggregates = {}
    for key, records in grouped.items():
        if len(records) != pair["sample_count"]:
            raise ValueError(f"sensitivity replay {key} has an incomplete sample set")
        baseline_quality = {
            metric: _mean(records, "quality", "baseline", metric)
            for metric in ("psnr_db", "ssim", "lpips")
        }
        scarf_quality = {
            metric: _mean(records, "quality", "scarf", metric)
            for metric in ("psnr_db", "ssim", "lpips")
        }
        baseline_cycles = _mean(records, "performance", "baseline_cycles")
        scarf_cycles = _mean(records, "performance", "scarf_cycles")
        aggregates[key] = {
            "evaluation": {
                "kind": "sensitivity_trace_aggregate",
                "sample_count": pair["sample_count"],
                "sample_selection_sha256": selection_hash,
                "claim_protocol_complete": full_claim,
                "trace_results": [
                    {"path": str(path), "sha256": sha256_file(path)} for path in paths
                ],
            },
            "quality": build_quality_record(baseline_quality, scarf_quality),
            "performance": {
                "baseline_cycles": baseline_cycles,
                "scarf_cycles": scarf_cycles,
                "speedup": baseline_cycles / scarf_cycles,
                "baseline_source": "scarf_no_optimization_cycles",
                "cycle_source": "sensitivity_trace_replay:sample_mean",
            },
            "fsdr_saes": _mean_tree([record["fsdr_saes"] for record in records]),
        }
    audit = {
        "sample_count": len(traces),
        "selection_hash": selection_hash,
        "neural_forward_passes": len(traces),
        "parameter_replays": sum(record["trace"]["replay_count"] for record in traces),
    }
    return aggregates, audit


def execute(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pair_aggregates = {}
    audits = {}
    for index, pair in enumerate(plan["pair_traces"], start=1):
        print(
            f"[{index}/{len(plan['pair_traces'])}]",
            " ".join(pair["command"]),
            flush=True,
        )
        result = subprocess.run(pair["command"], cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError(
                f"sensitivity trace failed ({pair['model']}/{pair['dataset']}): "
                f"exit {result.returncode}"
            )
        aggregates, audit = aggregate_pair_traces(pair)
        pair_key = (pair["model"], pair["dataset"])
        pair_aggregates[pair_key] = aggregates
        audits[f"{pair['model']}/{pair['dataset']}"] = audit

    completed = [
        {
            **run,
            "metrics": pair_aggregates[(run["model"], run["dataset"])][
                (run["study"], run["value"])
            ],
        }
        for run in plan["runs"]
    ]

    final = {
        "schema_version": "1.0",
        "kind": "sensitivity_results",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grids": plan["grids"],
        "runs": completed,
        "validation": {
            "complete": len(completed) == plan["expected_runs"],
            "expected_runs": plan["expected_runs"],
            "completed_runs": len(completed),
            "trace_generation_count": len(plan["pair_traces"]),
            "pair_audits": audits,
        },
    }
    if not final["validation"]["complete"]:
        raise RuntimeError("sensitivity matrix is incomplete")
    (output_dir / "results.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", help="Override the locked profile interpreters")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    return args


def main() -> int:
    args = parse_args()
    plan = build_plan(
        args.output_dir,
        python_override=args.python,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    execute(plan, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
