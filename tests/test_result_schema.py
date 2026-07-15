import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_result.py"


def valid_result() -> dict:
    return {
        "schema_version": "1.0",
        "provenance": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_identity": "git",
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
        },
        "ablation": {},
        "fsdr": {},
        "saes": {},
        "hardware": {},
        "validation": {"reproducible": True, "reference_fallback_used": False},
    }


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
