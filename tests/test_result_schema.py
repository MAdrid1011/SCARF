import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_result.py"


def valid_result() -> dict:
    return {
        "schema_version": "2.0",
        "evidence_class": "deterministic_execution",
        "provenance": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_identity": "git",
            "source_tree_sha256": "9" * 64,
            "submodules": {
                "transplat": "b" * 40,
                "mvsplat": "c" * 40,
                "depthsplat": "d" * 40,
            },
            "command": ["python", "scripts/demo.py"],
            "runtime_assets": {"VGG16": {"sha256": "1" * 64}},
            "environment": {"profile": "classic", "digest_sha256": "2" * 64},
            "seed": 0,
            "device": {"type": "cuda", "name": "test"},
            "dataset": {
                "name": "re10k",
                "representation": "re10k-native",
                "functional_fixture": False,
                "paper_result_eligible": True,
                "sha256": "e" * 64,
                "tree_sha256": "1" * 64,
            },
            "checkpoint": {
                "path": "re10k.ckpt",
                "sha256": "f" * 64,
                "load": {
                    "matched_tensors": 10,
                    "matched_checkpoint_numel_fraction": 0.99,
                },
            },
            "evaluation": {
                "kind": "sample",
                "sample_index": 0,
                "execution_index": 0,
                "candidate_count": 1,
                "scene": "scene-0000",
                "context_indices": [0, 10],
                "target_indices": [0, 1, 2],
                "target_view_count": 3,
                "target_view_aggregation": "arithmetic mean over selected target views",
            },
        },
        "quality": {
            "baseline": {"psnr_db": 28.08, "ssim": 0.919, "lpips": 0.128},
            "scarf": {"psnr_db": 28.07, "ssim": 0.917, "lpips": 0.130},
            "change": {
                "psnr_signed_pct": -0.0356,
                "psnr_degradation_pct": 0.0356,
                "psnr_absolute_pct": 0.0356,
            },
            "views": [
                {
                    "target_index": target_index,
                    "baseline": {"psnr_db": 28.08, "ssim": 0.919, "lpips": 0.128},
                    "scarf": {"psnr_db": 28.07, "ssim": 0.917, "lpips": 0.130},
                }
                for target_index in (0, 1, 2)
            ],
        },
        "performance": {
            "baseline_cycles": 1000,
            "scarf_cycles": 400,
            "speedup": 2.5,
            "cycle_source": "simulated",
            "baseline_source": "diagnostic_device_timing",
            "stages": {
                stage: {
                    "cycles": 100,
                    "useful_mmcu_slots": 80,
                    "scheduled_mmcu_slots": 100,
                    "mmcu_slots_available": True,
                    "source": "event_simulator",
                }
                for stage in ("s1", "s2", "s3", "s4")
            },
        },
        "events": {
            "fsdr": {
                "total_pixels": 100,
                "cache_hits": 80,
                "guided_pixels": 70,
                "guided_top1_covered": 69,
                "guided_top1_missed": 1,
                "discrete_top1_available": True,
                "full_depth_evaluations": 6400,
                "executed_depth_evaluations": 2200,
                "depth_evaluations_available": True,
                "feature_buffer_bytes_baseline": 1024,
                "feature_buffer_bytes_actual": 512,
                "feature_buffer_bytes_available": True,
                "source": "event_simulator",
            },
            "saes": {
                "total_tiles": 10,
                "level0_tiles": 2,
                "level1_tiles": 3,
                "full_tiles": 5,
                "tile_path_available": True,
                "baseline_gaussians": 100,
                "actual_gaussians": 70,
                "gaussian_counts_available": True,
                "full_s2_evaluations": 6400,
                "executed_s2_evaluations": 4000,
                "s2_evaluations_available": True,
                "source": "event_simulator",
            },
        },
        "energy": {"available": False, "source": "not_measured"},
        "ablation": {},
        "fsdr": {},
        "saes": {},
        "hardware": {},
        "validation": {"reproducible": True, "reference_fallback_used": False},
    }


def valid_v21_result() -> dict:
    from saes.execution_dependency import resolve_s2_s3_execution_contract

    record = valid_result()
    record["schema_version"] = "2.1"
    record["provenance"].update(
        {
            "mechanism_config_sha256": "3" * 64,
            "model": "transplat",
            "calibration_provenance": {
                "status": "calibrated",
                "manifest_sha256": "4" * 64,
                "candidate_records_sha256": "5" * 64,
                "evaluation_disjoint": True,
                "expected_results_accessed": False,
                "global_configuration": True,
                "protocol": "dl3dv_train_holdout_v1",
                "train_holdout_scene_disjoint": True,
                "train": {
                    "selection_sha256": "6" * 64,
                    "scene_set_sha256": "7" * 64,
                    "pair_bindings_sha256": "8" * 64,
                    "trace_set_sha256": "9" * 64,
                    "candidate_set_sha256": "a" * 64,
                    "selected_candidate_sha256": "b" * 64,
                },
                "holdout": {
                    "selection_sha256": "c" * 64,
                    "scene_set_sha256": "d" * 64,
                    "pair_bindings_sha256": "e" * 64,
                    "trace_set_sha256": "f" * 64,
                    "candidate_set_sha256": "0" * 64,
                    "validated_parameters_sha256": "1" * 64,
                    "validated_candidate_sha256": "2" * 64,
                },
            },
        }
    )
    record["events"]["fsdr"].update(
        {
            "hamming_hits": 80,
            "local_valid_hits": 70,
            "local_invalid_fallbacks": 10,
            "source": "fsdr_path_event_counter",
        }
    )
    record["events"]["saes"].update(
        {
            "l0_representatives": 8,
            "l1_lightweight_anchors": 24,
            "full_stage3_gaussians": 50,
            "assignment_weight_sum_error_max": 1e-7,
            "opacity_transmittance_error_max": 1e-6,
            "covariance_psd_violations": 0,
            "execution_dependency": resolve_s2_s3_execution_contract("transplat"),
            "s2_s3_saving": {"s2": 0.0, "s3": 0.0},
        }
    )
    return record


def test_v21_result_requires_calibration_and_faithful_event_evidence(tmp_path):
    result = run_validator(tmp_path, valid_v21_result())
    assert result.returncode == 0, result.stderr

    mutations = (
        lambda r: r["provenance"].pop("mechanism_config_sha256"),
        lambda r: r["provenance"]["calibration_provenance"].update(
            evaluation_disjoint=False
        ),
        lambda r: r["provenance"]["calibration_provenance"].pop("holdout"),
        lambda r: r["events"]["fsdr"].update(local_valid_hits=81),
        lambda r: r["events"]["saes"].update(covariance_psd_violations=1),
        lambda r: r["events"]["saes"].pop("execution_dependency"),
        lambda r: r["events"]["saes"]["s2_s3_saving"].update(s2=0.1),
    )
    for mutation in mutations:
        payload = valid_v21_result()
        mutation(payload)
        assert run_validator(tmp_path, payload).returncode != 0


def run_validator(tmp_path: Path, payload: dict):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        capture_output=True,
        text=True,
    )


def test_valid_result_passes(tmp_path):
    result = run_validator(tmp_path, valid_result())
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout


def test_synthetic_functional_result_is_explicitly_not_paper_eligible(tmp_path):
    payload = valid_result()
    payload["provenance"]["dataset"].update(
        {
            "representation": "re10k-synthetic-functional-v1",
            "functional_fixture": True,
            "paper_result_eligible": False,
        }
    )
    assert run_validator(tmp_path, payload).returncode == 0

    payload["provenance"]["dataset"]["paper_result_eligible"] = True
    result = run_validator(tmp_path, payload)
    assert result.returncode != 0
    assert "eligibility" in result.stderr


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda r: r.pop("provenance"), "provenance"),
        (lambda r: r["quality"]["scarf"].pop("lpips"), "lpips"),
        (lambda r: r["performance"].update(scarf_cycles=0), "scarf_cycles"),
        (lambda r: r.pop("evidence_class"), "evidence_class"),
        (lambda r: r["events"].pop("fsdr"), "events.fsdr"),
        (lambda r: r.pop("energy"), "energy"),
        (
            lambda r: r["validation"].update(reference_fallback_used=True),
            "reference_fallback_used",
        ),
        (
            lambda r: r["provenance"].update(
                command=["/" + "home/author/python"]
            ),
            "portable command",
        ),
    ],
)
def test_invalid_result_fails(tmp_path, mutation, expected):
    payload = valid_result()
    mutation(payload)
    result = run_validator(tmp_path, payload)
    assert result.returncode != 0
    assert expected in result.stderr
