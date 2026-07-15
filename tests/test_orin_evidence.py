import copy

import pytest


def valid_measurement() -> dict:
    return {
        "schema_version": "1.0",
        "kind": "orin_nx_measurement",
        "status": "PASS",
        "baseline_source": "orin_nx_cuda_events",
        "environment": {
            "contract": {
                "device_model_contains": "Jetson Orin NX",
                "maximum_allowed_temperature_c": 80.0,
            },
            "device_model": "NVIDIA Jetson Orin NX",
        },
        "selection": {
            "sample_selection_sha256": "a" * 64,
            "dataset_tree_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
        },
        "cuda_event_encoder_median_ms": 12.5,
        "cuda_event_encoder_samples_ms": [12.1, 12.4, 12.5, 12.8, 13.0],
        "cuda_event_repetitions": 5,
        "thermal": {"maximum_temperature_c": 61.0},
        "artifacts": {
            "inference_result": {"sha256": "d" * 64},
            "tegrastats": {"sha256": "e" * 64},
            "nsight_systems": {"sha256": "f" * 64},
            "cuda_events": {"sha256": "1" * 64},
        },
    }


def test_orin_evidence_accepts_complete_raw_measurement():
    from hardware.orin.run import validate_measurement_record

    validate_measurement_record(valid_measurement(), expected_selection_sha256="a" * 64)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r["environment"].update(device_model="RTX 3060"), "Orin NX"),
        (lambda r: r["artifacts"].pop("tegrastats"), "tegrastats"),
        (lambda r: r["artifacts"].pop("cuda_events"), "cuda_events"),
        (lambda r: r["thermal"].update(maximum_temperature_c=81.0), "temperature"),
        (lambda r: r["selection"].update(sample_selection_sha256="0" * 64), "selection"),
        (lambda r: r.update(baseline_source="workstation_cuda_events"), "baseline source"),
        (lambda r: r.update(cuda_event_encoder_median_ms=99.0), "median"),
    ],
)
def test_orin_evidence_rejects_unverifiable_measurement(mutation, match):
    from hardware.orin.run import validate_measurement_record

    record = copy.deepcopy(valid_measurement())
    mutation(record)
    with pytest.raises(ValueError, match=match):
        validate_measurement_record(record, expected_selection_sha256="a" * 64)


def test_figure8_rejects_workstation_baseline_even_with_measurement_pointer():
    from scripts.validate_ae import orin_evidence_complete

    result = {
        "performance": {"baseline_source": "workstation_cuda_events"},
        "provenance": {
            "evaluation": {
                "sample_results": [
                    {
                        "sample_index": 0,
                        "orin_measurement": {"path": "unused", "sha256": "0" * 64},
                    }
                ]
            }
        },
    }
    assert not orin_evidence_complete(result, "a" * 64)
