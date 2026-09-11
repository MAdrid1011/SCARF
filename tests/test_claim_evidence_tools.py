from __future__ import annotations

import json
from pathlib import Path


def test_init_claim_evidence_creates_explicit_pending_template(tmp_path: Path) -> None:
    from scripts.init_claim_evidence import init

    output = init(tmp_path / "evidence-input")
    manifest = json.loads((output / "input-manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "pending_validation"
    assert manifest["claim_eligible"] is False
    assert len(manifest["required_pairs"]) == 9
    assert (output / "README.md").is_file()
    assert (output / "timing-backend/timing-trace").is_dir()
    assert (output / "validation-fixtures/calibration/calibration-fixture.json").is_file()
    assert (output / "validation-fixtures/timing/manifest.json").is_file()


def test_init_claim_evidence_materializes_complete_external_format_tree(tmp_path: Path) -> None:
    from scripts.init_claim_evidence import DATASETS, MODELS, init

    output = init(tmp_path / "format-only")
    assert (output / "calibration/plan.json").is_file()
    assert (output / "calibration/train-candidates.json").is_file()
    assert (output / "calibration/candidate-template.json").is_file()
    assert (output / "calibration/mechanism_config.json").is_file()
    assert (output / "calibration/calibration-result.json").is_file()
    assert (output / "FILE_MANIFEST.json").is_file()
    assert (output / "timing-backend/rtl/ScarfTop.sv").is_file()
    assert (output / "environments/classic.json").is_file()
    assert (output / "environments/depthsplat.json").is_file()
    assert (output / "environments/orin.json").is_file()

    timing = json.loads((output / "timing-backend/manifest.json").read_text())
    assert len(timing["samples"]) == len(MODELS) * len(DATASETS)
    for model in MODELS:
        for dataset in DATASETS:
            pair = f"{model}_{dataset}"
            trace = output / "timing-backend/timing-trace" / model / dataset / "sample_00000.json"
            assert trace.is_file()
            for workflow in ("quality", "mechanisms", "speedup"):
                root = output / workflow / pair
                assert (root / "results.json").is_file()
                assert (root / "samples/sample_00000/results.json").is_file()
                assert (root / "pair-execution.json").is_file()
                assert (root / "progress.jsonl").is_file()
                assert (root / "timing-trace/aggregate.json").is_file()
                assert (
                    root
                    / "samples/sample_00000/timing-trace"
                    / model
                    / dataset
                    / "sample_00000.json"
                ).is_file()
                assert (
                    root / "samples/sample_00000/timing-backend/manifest.json"
                ).is_file()
                assert (
                    root
                    / "samples/sample_00000/timing-backend/rtl/ScarfTop.sv"
                ).is_file()
                assert (
                    root
                    / "samples/sample_00000/timing-backend/timing-trace"
                    / model
                    / dataset
                    / "sample_00000.json"
                ).is_file()
                execution = json.loads((root / "pair-execution.json").read_text())
                assert execution["kind"] == "pair_execution"
                assert set(execution) == {
                    "schema_version",
                    "kind",
                    "sample_count",
                    "executed",
                    "resumed",
                    "complete",
                    "source_index",
                }
            speedup = output / "speedup" / pair
            assert (speedup / "samples/sample_00000/orin-evidence/measurement.json").is_file()
            assert (speedup / "samples/sample_00000/orin-evidence/cuda-events.json").is_file()
            assert (speedup / "orin-profile/pair-measurement.json").is_file()
            assert (speedup / "orin-profile/tegrastats.log").is_file()
            assert (speedup / "orin-profile/nsight.nsys-rep").is_file()

    calibration_manifest = json.loads(
        (output / "calibration/manifest.json").read_text()
    )
    for split, count in (("calibration_train", 24), ("calibration_holdout", 8)):
        split_record = calibration_manifest["datasets"]["dl3dv"]["splits"][split]
        assert split_record["scene_count"] == count
        assert set(split_record["representations"]) == {"native", "re10k"}
        assert all(
            entry["sample_count"] == count
            for entry in split_record["representations"].values()
        )

    train_trace = json.loads(
        (output / "calibration/traces/calibration_train/transplat_dl3dv/samples/sample_00000/results.json").read_text()
    )
    holdout_trace = json.loads(
        (output / "calibration/traces/calibration_holdout/transplat_dl3dv/samples/sample_00000/results.json").read_text()
    )
    assert train_trace["calibration_scope"] == "registered_global_grid"
    assert holdout_trace["calibration_scope"] == "exact_committed_train_tuple"

    aggregate = json.loads(
        (output / "quality/transplat_re10k/results.json").read_text()
    )
    assert aggregate["provenance"]["evaluation"][
        "execution_trace_set_schema_version"
    ] == "execution-trace-set-v2"
    assert aggregate["provenance"]["evaluation"]["sample_results"][0][
        "execution_trace_sha256"
    ].startswith("<fill-")

    measurement = json.loads(
        (output / "speedup/transplat_re10k/samples/sample_00000/orin-evidence/measurement.json").read_text()
    )
    assert measurement["status"] == "pending_validation"
    assert measurement["cuda_event_repetitions"] == 5
    assert set(measurement["artifacts"]) == {
        "inference_result",
        "tegrastats",
        "nsight_systems",
        "cuda_events",
    }

    mechanism = json.loads(
        (output / "calibration/mechanism_config.json").read_text()
    )
    assert mechanism["schema_version"] == "1.0"
    assert mechanism["status"] == "pending_validation"
    assert mechanism["global_configuration"] is True
    assert set(mechanism["selected"]) == {
        "gamma_depth",
        "beta_x",
        "beta_f",
        "beta_d",
    }
    assert mechanism["saes_execution_identity"]["route_sha256"]

    manifest = json.loads((output / "FILE_MANIFEST.json").read_text())
    assert manifest["profiles"]["full"] == {
        "re10k": 6474,
        "acid": 1595,
        "dl3dv": 140,
    }
    assert len(manifest["pairs"]) == 9
    progress = json.loads((output / "progress-event-examples.json").read_text())
    assert progress["status"] == "format_only"
    assert {event["event"] for event in progress["events"]} == {
        "pair_start",
        "sample_start",
        "sample_complete",
        "sample_error",
        "pair_complete",
    }


def test_init_claim_evidence_rejects_release_directories() -> None:
    from scripts.init_claim_evidence import init

    import pytest

    with pytest.raises(ValueError, match="outside artifact"):
        init(Path(__file__).resolve().parents[1] / "artifact" / "format-template")
