import copy
import json
from pathlib import Path

import pytest


def sample_record(index: int, *, dataset: str = "re10k") -> dict:
    offset = float(index)
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
            "model": "mvsplat",
            "device": {"type": "cuda", "name": "test"},
            "dataset": {
                "name": dataset,
                "representation": f"{dataset}-native",
                "functional_fixture": False,
                "paper_result_eligible": True,
                "manifest": "manifest.json",
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
                "sample_index": index,
                "execution_index": index,
                "candidate_count": 2,
                "scene": f"scene-{index:04d}",
                "context_indices": [0, 10],
                "target_indices": [0, 1, 2],
                "target_view_count": 3,
                "target_view_aggregation": "arithmetic mean over selected target views",
            },
        },
        "quality": {
            "baseline": {"psnr_db": 28.0 + offset, "ssim": 0.90, "lpips": 0.12},
            "scarf": {"psnr_db": 27.9 + offset, "ssim": 0.89, "lpips": 0.13},
            "change": {
                "psnr_signed_pct": -0.1,
                "psnr_degradation_pct": 0.1,
                "psnr_absolute_pct": 0.1,
            },
            "views": [
                {
                    "target_index": target_index,
                    "baseline": {
                        "psnr_db": 28.0 + offset,
                        "ssim": 0.90,
                        "lpips": 0.12,
                    },
                    "scarf": {
                        "psnr_db": 27.9 + offset,
                        "ssim": 0.89,
                        "lpips": 0.13,
                    },
                }
                for target_index in (0, 1, 2)
            ],
        },
        "performance": {
            "baseline_cycles": 1000 + index * 100,
            "scarf_cycles": 400 + index * 50,
            "speedup": (1000 + index * 100) / (400 + index * 50),
            "baseline_source": "workstation_cuda_events",
            "cycle_source": "simulator",
            "components": {"feature": 100, "depth": 200, "gaussian": 50, "ggu": 50},
        },
        "ablation": {"asic": {"eff_total": 500 + index * 10}},
        "fsdr_saes": {"fsdr": {"guided_rate": 0.5 + index * 0.1}},
        "hardware": {"physical_ppa_included": False},
        "validation": {"reproducible": True, "reference_fallback_used": False},
    }


def write_samples(tmp_path: Path, records: list[dict]) -> list[Path]:
    paths = []
    for ordinal, record in enumerate(records):
        path = tmp_path / f"sample_{ordinal:05d}" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    return paths


def test_aggregate_results_preserves_evidence_and_computes_dataset_means(tmp_path):
    from scripts.aggregate_results import aggregate
    from scripts.validate_result import validate

    paths = write_samples(tmp_path, [sample_record(0), sample_record(1)])
    record = aggregate(paths, 2)
    validate(record)

    assert record["provenance"]["evaluation"]["kind"] == "dataset_aggregate"
    assert record["provenance"]["evaluation"]["sample_indices"] == [0, 1]
    assert all(item["sha256"] for item in record["provenance"]["evaluation"]["sample_results"])
    assert len(record["provenance"]["evaluation"]["sample_selection_sha256"]) == 64
    assert record["provenance"]["evaluation"]["sample_selection"][1]["scene"] == "scene-0001"
    assert not Path(
        record["provenance"]["evaluation"]["sample_results"][0]["path"]
    ).is_absolute()
    assert record["quality"]["baseline"]["psnr_db"] == 28.5
    assert record["performance"]["baseline_cycles"] == 1050
    assert record["performance"]["scarf_cycles"] == 425
    assert record["performance"]["speedup"] == 1050 / 425
    assert record["ablation"]["asic"]["eff_total"] == 505
    assert record["validation"]["dispersion"]["baseline.psnr_db"]["population_stddev"] == 0.5


def test_aggregate_results_rejects_missing_or_duplicate_samples(tmp_path):
    from scripts.aggregate_results import aggregate

    paths = write_samples(tmp_path, [sample_record(0)])
    with pytest.raises(ValueError, match="expected 2"):
        aggregate(paths, 2)

    paths = write_samples(tmp_path / "duplicate", [sample_record(0), sample_record(0)])
    with pytest.raises(ValueError, match="source indices must be unique"):
        aggregate(paths, 2)


def test_aggregate_accepts_noncontiguous_source_ordinals(tmp_path):
    from scripts.aggregate_results import aggregate
    from scripts.validate_result import validate

    second = sample_record(3)
    second["provenance"]["evaluation"]["execution_index"] = 1
    second["provenance"]["evaluation"]["candidate_count"] = 2
    paths = write_samples(tmp_path, [sample_record(0), second])
    record = aggregate(paths, 2)
    validate(record)

    assert record["provenance"]["evaluation"]["sample_indices"] == [0, 3]
    assert record["provenance"]["evaluation"]["execution_indices"] == [0, 1]


def test_aggregate_results_rejects_mixed_dataset_provenance(tmp_path):
    from scripts.aggregate_results import aggregate

    other = copy.deepcopy(sample_record(1))
    other["provenance"]["dataset"]["name"] = "acid"
    paths = write_samples(tmp_path, [sample_record(0), other])
    with pytest.raises(ValueError, match="dataset"):
        aggregate(paths, 2)
