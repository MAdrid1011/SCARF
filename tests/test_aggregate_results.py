import copy
import hashlib
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


def v2_sample_record(index: int) -> dict:
    record = sample_record(index)
    record["schema_version"] = "2.0"
    record["evidence_class"] = "deterministic_execution"
    record["performance"]["stages"] = {
        stage: {
            "cycles": (ordinal + 1) * 10 + index * 2,
            "useful_mmcu_slots": (ordinal + 1) * 7 + index,
            "scheduled_mmcu_slots": (ordinal + 1) * 8 + index,
            "mmcu_slots_available": True,
            "source": "event_simulator",
        }
        for ordinal, stage in enumerate(("s1", "s2", "s3", "s4"))
    }
    record["events"] = {
        "fsdr": {
            "total_pixels": 100 + index,
            "cache_hits": 80 + index,
            "guided_pixels": 70 + index,
            "guided_top1_covered": 69 + index,
            "guided_top1_missed": 1,
            "discrete_top1_available": True,
            "full_depth_evaluations": 6400 + index,
            "executed_depth_evaluations": 2200 + index,
            "depth_evaluations_available": True,
            "feature_buffer_bytes_baseline": 1024 + index,
            "feature_buffer_bytes_actual": 512 + index,
            "feature_buffer_bytes_available": True,
            "source": "event_simulator",
        },
        "saes": {
            "total_tiles": 10 + index,
            "level0_tiles": 2 + index,
            "level1_tiles": 3,
            "full_tiles": 5,
            "tile_path_available": True,
            "baseline_gaussians": 100 + index,
            "actual_gaussians": 70 + index,
            "gaussian_counts_available": True,
            "full_s2_evaluations": 6400 + index,
            "executed_s2_evaluations": 4000 + index,
            "s2_evaluations_available": True,
            "source": "event_simulator",
        },
    }
    record["energy"] = {"available": False, "source": "not_measured"}
    return record


def v21_sample_record(index: int) -> dict:
    from saes.execution_dependency import resolve_s2_s3_execution_contract

    record = v2_sample_record(index)
    record["schema_version"] = "2.1"
    record["provenance"].update(
        {
            "mechanism_config_sha256": "3" * 64,
            "calibration_provenance": {
                "manifest_sha256": "4" * 64,
                "status": "preregistered",
                "candidate_records_sha256": None,
                "evaluation_disjoint": False,
                "expected_results_accessed": False,
                "global_configuration": True,
            },
        }
    )
    record["provenance"]["dataset"]["paper_result_eligible"] = False
    record["events"]["fsdr"].update(
        {
            "hamming_hits": 80 + index,
            "local_valid_hits": 70 + index,
            "local_invalid_fallbacks": 10,
        }
    )
    record["events"]["saes"].update(
        {
            "l0_representatives": 8 + index,
            "l1_lightweight_anchors": 12,
            "full_stage3_gaussians": 80,
            "covariance_psd_violations": 0,
            "assignment_weight_sum_error_max": 0.0,
            "opacity_transmittance_error_max": 0.0,
            "execution_dependency": resolve_s2_s3_execution_contract("mvsplat"),
            "s2_s3_saving": {"s2": 0.0, "s3": 0.0},
        }
    )
    return record


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


def test_quality_claim_aggregate_does_not_require_timing_backend(tmp_path):
    from scripts.aggregate_results import aggregate
    from scripts.validate_result import validate

    records = [sample_record(0), sample_record(1)]
    for record in records:
        record["provenance"]["execution_contract"] = {
            "run_class": "claim",
            "saes_materialization": "representative",
        }
        record["provenance"]["claim_workflow"] = "quality"

    aggregate_record = aggregate(write_samples(tmp_path, records), 2)
    validate(aggregate_record)
    assert aggregate_record["provenance"]["claim_workflow"] == "quality"
    assert "claim_timing" not in aggregate_record["performance"]


def test_v2_aggregate_sums_discrete_events_and_mmcu_slots(tmp_path):
    from scripts.aggregate_results import aggregate
    from scripts.validate_result import validate

    paths = write_samples(tmp_path, [v2_sample_record(0), v2_sample_record(1)])
    record = aggregate(paths, 2)
    validate(record)

    assert record["schema_version"] == "2.0"
    assert record["evidence_class"] == "deterministic_execution"
    assert record["performance"]["stages"]["s1"]["cycles"] == 11
    assert record["performance"]["stages"]["s1"]["useful_mmcu_slots"] == 15
    assert record["performance"]["stages"]["s1"]["scheduled_mmcu_slots"] == 17
    assert record["events"]["fsdr"]["total_pixels"] == 201
    assert record["events"]["fsdr"]["guided_top1_covered"] == 139
    assert record["events"]["saes"]["total_tiles"] == 21
    assert record["events"]["saes"]["level0_tiles"] == 5


def test_v21_aggregate_preserves_execution_dependency_contract(tmp_path):
    from scripts.aggregate_results import aggregate
    from scripts.validate_result import validate

    paths = write_samples(tmp_path, [v21_sample_record(0), v21_sample_record(1)])
    record = aggregate(paths, 2)
    validate(record)

    assert record["schema_version"] == "2.1"
    assert record["events"]["saes"]["execution_dependency"]["model"] == "mvsplat"
    assert record["events"]["saes"]["s2_s3_saving"] == {"s2": 0.0, "s3": 0.0}


def test_aggregate_zero_fills_sparse_saes_reason_counters(tmp_path):
    from scripts.aggregate_results import aggregate
    from scripts.validate_result import validate

    records = [v21_sample_record(0), v21_sample_record(1)]
    identity = {"tile_size": 4, "l1_anchor_count": 12}
    for record, reasons in zip(
        records,
        ({}, {"nonzero_source_opacity": 4}),
    ):
        record["fsdr_saes"] = {
            "fsdr": {"guided_rate": 0.5},
            "saes": {
                "deletion_certificate_rejection_reasons": reasons,
                "same_budget_dense_oracle_failure_reasons": {},
                "multicontext_tangent_fallback_reasons": {},
                "saes_execution_identity": identity,
                "route_sha256": "a" * 64,
            },
        }

    aggregate_record = aggregate(write_samples(tmp_path, records), 2)
    validate(aggregate_record)

    assert aggregate_record["fsdr_saes"]["saes"][
        "deletion_certificate_rejection_reasons"
    ] == {"nonzero_source_opacity": 2.0}
    assert aggregate_record["fsdr_saes"]["saes"][
        "saes_execution_identity"
    ] == identity


def test_aggregate_binds_trace_set_to_quality_and_performance(tmp_path):
    from scripts.aggregate_results import aggregate
    from scripts.result_record import (
        bind_execution_trace,
        execution_trace_performance_evidence_from_record,
        execution_trace_set_sha256,
    )
    from scripts.validate_result import validate

    records = [v21_sample_record(0), v21_sample_record(1)]
    for record in records:
        record["provenance"]["execution_contract"] = {
            "run_class": "diagnostic",
            "saes_materialization": "representative",
        }
        bind_execution_trace(record)

    aggregate_record = aggregate(write_samples(tmp_path, records), 2)
    evaluation = aggregate_record["provenance"]["evaluation"]
    digest = evaluation["execution_trace_set_sha256"]
    performance_evidence = evaluation["execution_trace_performance_evidence"]

    assert (
        execution_trace_set_sha256(
            evaluation["execution_trace_set"], performance_evidence
        )
        == digest
    )
    assert performance_evidence == execution_trace_performance_evidence_from_record(
        aggregate_record
    )
    assert aggregate_record["quality"]["execution_trace_set_sha256"] == digest
    assert aggregate_record["performance"]["execution_trace_set_sha256"] == digest
    assert [entry["sample_index"] for entry in evaluation["execution_trace_set"]] == [0, 1]
    assert all(
        entry["execution_trace_sha256"]
        == sample["provenance"]["execution_trace_sha256"]
        for entry, sample in zip(evaluation["execution_trace_set"], records)
    )
    validate(aggregate_record)

    aggregate_record["quality"]["execution_trace_set_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="quality execution trace set binding"):
        validate(aggregate_record)

    aggregate_record = aggregate(write_samples(tmp_path / "tamper", records), 2)
    mutations = (
        lambda result: result["performance"].update(
            scarf_cycles=426,
            speedup=result["performance"]["baseline_cycles"] / 426,
        ),
        lambda result: result["ablation"]["asic"].update(eff_total=506),
        lambda result: result["events"]["saes"].update(source="tampered-events"),
        lambda result: result["fsdr_saes"]["fsdr"].update(guided_rate=0.9),
    )
    for mutation in mutations:
        tampered = copy.deepcopy(aggregate_record)
        mutation(tampered)
        with pytest.raises(ValueError, match="trace performance evidence"):
            validate(tampered)


def test_aggregate_trace_set_binds_orin_measurement_hash_without_its_path(tmp_path):
    from scripts.aggregate_results import aggregate
    from scripts.result_record import bind_execution_trace
    from scripts.validate_result import validate

    record = v21_sample_record(0)
    record["provenance"]["execution_contract"] = {
        "run_class": "diagnostic",
        "saes_materialization": "representative",
    }
    bind_execution_trace(record)
    sample_path = write_samples(tmp_path, [record])[0]
    measurement_path = sample_path.parent / "orin-evidence" / "measurement.json"
    measurement_path.parent.mkdir()
    measurement_path.write_text('{"opaque":"external timing evidence"}\n', encoding="utf-8")
    measurement_sha256 = hashlib.sha256(measurement_path.read_bytes()).hexdigest()

    aggregate_record = aggregate([sample_path], 1)
    trace_entry = aggregate_record["provenance"]["evaluation"][
        "execution_trace_set"
    ][0]

    assert trace_entry["orin_measurement_sha256"] == measurement_sha256
    assert "path" not in trace_entry
    validate(aggregate_record)

    aggregate_record["provenance"]["evaluation"]["sample_results"][0][
        "orin_measurement"
    ]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Orin sample evidence does not match"):
        validate(aggregate_record)


def test_aggregate_rejects_assignment_consensus_result_records(tmp_path):
    from scripts.aggregate_results import aggregate

    records = [sample_record(0), sample_record(1)]
    for record in records:
        record["provenance"]["execution_contract"] = {
            "run_class": "diagnostic",
            "saes_materialization": (
                "assignment-consensus-adapter-pseudo-descriptor-diagnostic"
            ),
        }

    with pytest.raises(ValueError, match="assignment-consensus pseudo descriptors"):
        aggregate(write_samples(tmp_path, records), 2)


def test_aggregate_rejects_reference_only_sample_markers(tmp_path):
    from scripts.aggregate_results import aggregate
    from scripts.validate_result import validate

    marked = sample_record(0)
    marked["artifact_class"] = "PAPER_REFERENCE_ONLY"

    with pytest.raises(ValueError, match="PAPER_REFERENCE_ONLY"):
        validate(marked)
    with pytest.raises(ValueError, match="PAPER_REFERENCE_ONLY"):
        aggregate(write_samples(tmp_path, [marked]), 1)


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
