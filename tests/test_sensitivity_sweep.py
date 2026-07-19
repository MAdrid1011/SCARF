import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_paper_sensitivity_grids_and_defaults():
    from scripts.sensitivity_sweep import STUDIES

    assert STUDIES["fsdr_cache_size"]["values"] == (8, 16, 32, 64, 128)
    assert STUDIES["fsdr_cache_size"]["default"] == 32
    assert STUDIES["fsdr_hamming_threshold"]["values"] == (1, 2, 3, 4, 5)
    assert STUDIES["fsdr_hamming_threshold"]["default"] == 3
    assert STUDIES["saes_feature_variance"]["values"] == (0.1, 0.2, 0.3, 0.4, 0.5)
    assert STUDIES["saes_feature_variance"]["default"] == 0.2
    assert STUDIES["saes_depth_variance"]["values"] == (0.01, 0.05, 0.1, 0.2, 0.5)
    assert STUDIES["saes_depth_variance"]["default"] == 0.1
    assert STUDIES["saes_tile_size"]["values"] == (2, 4, 8, 16, 32)
    assert STUDIES["saes_tile_size"]["default"] == 4


def test_sensitivity_dry_run_covers_every_grid_point_and_pair(tmp_path):
    environment = {
        **os.environ,
        "SCARF_PYTHON_CLASSIC": sys.executable,
        "SCARF_PYTHON_DEPTHSPLAT": sys.executable,
    }
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sensitivity_sweep.py"),
            "--output-dir",
            str(tmp_path / "sensitivity"),
            "--dry-run",
            "--num-samples",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["expected_runs"] == 5 * 5 * 9
    assert len(plan["runs"]) == plan["expected_runs"]
    assert not (tmp_path / "sensitivity").exists()
    assert len(plan["pair_traces"]) == 9
    assert len(plan["execution_provenance"]["source_tree_sha256"]) == 64
    assert len(plan["execution_provenance"]["mechanism_config_sha256"]) == 64
    for pair in plan["pair_traces"]:
        command = pair["command"]
        assert command[1].endswith("scripts/run_pair.py")
        assert "--sensitivity-trace" in command
        assert "--dataset-root" in command
        assert pair["sample_count"] == 2
    assert all("command" not in run for run in plan["runs"])


def test_reviewer_sensitivity_uses_the_frozen_reviewer_profile(tmp_path, monkeypatch):
    from scripts.sensitivity_sweep import build_plan

    monkeypatch.setenv("SCARF_PYTHON_CLASSIC", sys.executable)
    monkeypatch.setenv("SCARF_PYTHON_DEPTHSPLAT", sys.executable)
    plan = build_plan(tmp_path, evidence_profile="reviewer")

    assert plan["evidence_profile"] == "reviewer"
    assert {pair["sample_count"] for pair in plan["pair_traces"]} == {512, 140}
    for pair in plan["pair_traces"]:
        assert "--dataset-root" in pair["command"]
        assert "artifact/protocol/reviewer/" in pair["evaluation_index"]


def test_trace_aggregation_replays_grid_without_rerunning_models(tmp_path: Path):
    from scripts.sensitivity_sweep import STUDIES, aggregate_pair_traces

    trace_dir = tmp_path / "traces" / "mvsplat_re10k"
    provenance = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "source_identity": "git",
        "source_tree_sha256": "b" * 64,
        "submodules": {
            "transplat": "c" * 40,
            "mvsplat": "d" * 40,
            "depthsplat": "e" * 40,
        },
        "mechanism_config_sha256": "f" * 64,
        "calibration_provenance": {
            "status": "calibrated",
            "manifest_sha256": "0" * 64,
            "candidate_records_sha256": "1" * 64,
            "evaluation_disjoint": True,
            "expected_results_accessed": False,
            "global_configuration": True,
        },
    }
    for execution_index, source_index in enumerate((0, 3)):
        replays = []
        for study, config in STUDIES.items():
            for value in config["values"]:
                replays.append(
                    {
                        "study": study,
                        "value": value,
                        "quality": {
                            "baseline": {"psnr_db": 28.0, "ssim": 0.9, "lpips": 0.1},
                            "scarf": {"psnr_db": 27.9, "ssim": 0.89, "lpips": 0.11},
                        },
                        "performance": {
                            "baseline_cycles": 1000,
                            "scarf_cycles": 500,
                        },
                        "fsdr_saes": {
                            "fsdr": {"guided_rate": 0.4},
                            "saes": {"level0_ratio": 0.2},
                        },
                    }
                )
        record = {
            "kind": "sensitivity_sample_trace",
            "sample_index": source_index,
            "execution_index": execution_index,
            "scene": f"scene-{source_index}",
            "context_indices": [0, 2],
            "target_indices": [1],
            "model": "mvsplat",
            "dataset": "re10k",
            "execution_provenance": provenance,
            "trace": {"neural_forward_passes": 1, "replay_count": 25},
            "replays": replays,
        }
        path = trace_dir / "samples" / f"sample_{source_index:05d}" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record), encoding="utf-8")

    aggregates, audit = aggregate_pair_traces(
        {
            "model": "mvsplat",
            "dataset": "re10k",
            "sample_count": 2,
            "declared_sample_count": 6474,
            "declared_sample_selection_sha256": "0" * 64,
            "trace_dir": str(trace_dir),
            "execution_provenance": provenance,
        }
    )

    assert len(aggregates) == 25
    assert audit["neural_forward_passes"] == 2
    assert audit["parameter_replays"] == 50
    assert audit["execution_provenance"] == provenance
    assert aggregates[("fsdr_cache_size", 32)]["performance"]["speedup"] == 2.0

    stale_path = trace_dir / "samples/sample_00000/results.json"
    stale = json.loads(stale_path.read_text(encoding="utf-8"))
    stale["execution_provenance"]["source_tree_sha256"] = "0" * 64
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="trace provenance"):
        aggregate_pair_traces(
            {
                "model": "mvsplat",
                "dataset": "re10k",
                "sample_count": 2,
                "declared_sample_count": 6474,
                "declared_sample_selection_sha256": "0" * 64,
                "trace_dir": str(trace_dir),
                "execution_provenance": provenance,
            }
        )
