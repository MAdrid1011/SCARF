import hashlib
import json
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    from scripts.rtl_payload import descriptor_sha256

    source = tmp_path / "rtl" / "ScarfTop.sv"
    trace = tmp_path / "timing-trace" / "mvsplat" / "re10k" / "sample.json"
    source.parent.mkdir(parents=True)
    trace.parent.mkdir(parents=True)
    source.write_text("module ScarfTop; endmodule\n", encoding="utf-8")
    trace.write_text(
        json.dumps(
            {
                "schema_version": "source-bound-timing-trace-v2",
                "model": "mvsplat",
                "dataset": "re10k",
                "sample_index": 0,
                "cycle_accurate": True,
                "source_rtl_sha256": _sha256(source),
                "clock_mhz": 1000,
                "events": [
                    {"stage": "s1", "start_cycle": 0, "end_cycle": 50, "accepted": True},
                    {"stage": "s2", "start_cycle": 50, "end_cycle": 150, "accepted": True},
                    {"stage": "s3", "start_cycle": 150, "end_cycle": 225, "accepted": True},
                    {"stage": "s4", "start_cycle": 225, "end_cycle": 300, "accepted": True},
                ],
                "stimulus_sha256": "a" * 64,
                "stimulus_size_bytes": 164,
                "workload": {},
                "axi_read_count": 4,
                "axi_read_addresses": [0x90000000, 0x90000010, 0x90000020, 0x90000030],
                "role_coverage": {
                    role: {"bytes_read": 16, "range_count": 1}
                    for role in ("pipeline_features", "pipeline_depths", "depth_candidates", "depth_probabilities")
                },
                "axi_consumption_digest": "b" * 64,
                "mechanism_counters": {
                    variant: {"fsdr_narrow": 0, "fsdr_full": 1, "saes_l0": 0, "saes_l1": 0, "saes_full": 1}
                    for variant in ("asic", "asic_fsdr", "asic_saes", "asic_fsdr_saes")
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    workload = {
        "schema_version": "scarf-rtl-workload-v2",
        "image_h": 1,
        "image_w": 1,
        "feature_dim": 1,
        "num_depth_candidates": 1,
        "num_gaussians": 1,
        "view_count": 1,
        "primitives_per_pixel": 1,
        "tensor_count": 4,
        "payload_size_bytes": 164,
        "data_offset_bytes": 100,
        "tensor_data_size_bytes": 64,
        "tensor_data_sha256": "c" * 64,
        "tensor_ranges": [
            {
                "name": role,
                "dtype": "float32",
                "shape": [4],
                "offset_bytes": ordinal * 16,
                "size_bytes": 16,
                "end_bytes": (ordinal + 1) * 16,
                "layout": "A",
                "role": role,
            }
            for ordinal, role in enumerate(("pipeline_features", "pipeline_depths", "depth_candidates", "depth_probabilities"))
        ],
    }
    workload["descriptor_sha256"] = descriptor_sha256(workload)
    trace_value = json.loads(trace.read_text(encoding="utf-8"))
    trace_value["workload"] = workload
    trace.write_text(json.dumps(trace_value) + "\n", encoding="utf-8")
    record = {
        "schema_version": "source-bound-timing-backend-v1",
        "backend": {
            "kind": "source_rtl",
            "source": {"path": "rtl/ScarfTop.sv", "sha256": _sha256(source)},
        },
        "clock_mhz": 1000,
        "samples": [
            {
                "model": "mvsplat",
                "dataset": "re10k",
                "sample_index": 0,
                "trace": {
                    "path": "timing-trace/mvsplat/re10k/sample.json",
                    "sha256": _sha256(trace),
                },
                "variants": {
                    "asic": {"total_cycles": 500},
                    "asic_fsdr": {"total_cycles": 400},
                    "asic_saes": {"total_cycles": 375},
                    "asic_fsdr_saes": {"total_cycles": 300},
                },
                "combined_stage_cycles": {
                    "s1": 50,
                    "s2": 100,
                    "s3": 75,
                    "s4": 75,
                },
                "workload": workload,
                "axi_read_count": 4,
                "axi_read_addresses": [0x90000000, 0x90000010, 0x90000020, 0x90000030],
                "role_coverage": trace_value["role_coverage"],
                "axi_consumption_digest": "b" * 64,
                "mechanism_counters": trace_value["mechanism_counters"],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, record


def test_load_and_stage_source_bound_trace(tmp_path: Path):
    from scripts.claim_timing_backend import (
        claim_timing_for_sample,
        load_claim_timing_manifest,
    )

    manifest, _ = _manifest(tmp_path)
    backend = load_claim_timing_manifest(manifest, root=tmp_path)
    timing, provenance = claim_timing_for_sample(
        backend,
        "mvsplat",
        "re10k",
        0,
        output_dir=tmp_path / "result",
    )
    assert timing["variants"]["asic_fsdr_saes"]["total_cycles"] == 300
    assert timing["trace"]["path"] == "timing-trace/mvsplat/re10k/sample.json"
    assert provenance["source"]["sha256"] == _sha256(tmp_path / "rtl/ScarfTop.sv")
    assert (tmp_path / "result" / timing["trace"]["path"]).is_file()
    assert (tmp_path / "result" / "timing-backend/manifest.json").is_file()
    assert (tmp_path / "result" / "timing-backend/rtl/ScarfTop.sv").is_file()
    assert (
        tmp_path / "result" / "timing-backend/timing-trace/mvsplat/re10k/sample.json"
    ).is_file()


def test_manifest_rejects_v1_trace_and_non_strict_combined_cycles(tmp_path: Path):
    from scripts.claim_timing_backend import load_claim_timing_manifest

    manifest, record = _manifest(tmp_path)
    trace_path = tmp_path / record["samples"][0]["trace"]["path"]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["schema_version"] = "source-bound-timing-trace-v1"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    record["samples"][0]["trace"]["sha256"] = _sha256(trace_path)
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_claim_timing_manifest(manifest, root=tmp_path)

    manifest, record = _manifest(tmp_path / "cycles")
    record["samples"][0]["variants"]["asic_fsdr_saes"]["total_cycles"] = 375
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly faster"):
        load_claim_timing_manifest(manifest, root=tmp_path / "cycles")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda item: item["backend"]["source"].update({"sha256": "0" * 64}), "source SHA256"),
        (lambda item: item["samples"][0]["trace"].update({"sha256": "0" * 64}), "trace SHA256"),
        (lambda item: item["samples"][0]["variants"].pop("asic_saes"), "all timing variants"),
        (lambda item: item["samples"][0]["combined_stage_cycles"].pop("s4"), "stage cycles"),
        (lambda item: item["samples"][0]["trace"].update({"path": "timing-trace/../escape.json"}), "normalized relative path"),
    ],
)
def test_manifest_rejects_invalid_evidence(tmp_path: Path, mutate, match):
    from scripts.claim_timing_backend import load_claim_timing_manifest

    manifest, record = _manifest(tmp_path)
    mutate(record)
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises((OSError, ValueError), match=match):
        load_claim_timing_manifest(manifest, root=tmp_path)


def test_manifest_rejects_duplicate_sample(tmp_path: Path):
    from scripts.claim_timing_backend import load_claim_timing_manifest

    manifest, record = _manifest(tmp_path)
    record["samples"].append(record["samples"][0])
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate timing sample"):
        load_claim_timing_manifest(manifest, root=tmp_path)


def test_nested_bundle_resolves_paths_relative_to_manifest_directory(tmp_path: Path):
    from scripts.claim_timing_backend import load_claim_timing_manifest

    bundle = tmp_path / "outputs" / "timing-backend"
    source = bundle / "rtl" / "ScarfTop.sv"
    trace = bundle / "timing-trace" / "mvsplat" / "re10k" / "sample.json"
    source.parent.mkdir(parents=True)
    trace.parent.mkdir(parents=True)
    source.write_text("module ScarfTop; endmodule\n", encoding="utf-8")
    trace.write_text(
        json.dumps(
            {
                "schema_version": "source-bound-timing-trace-v1",
                "model": "mvsplat",
                "dataset": "re10k",
                "sample_index": 0,
                "cycle_accurate": True,
                "source_rtl_sha256": _sha256(source),
                "clock_mhz": 1000,
                "events": [
                    {"stage": "s1", "start_cycle": 0, "end_cycle": 50, "accepted": True},
                    {"stage": "s2", "start_cycle": 50, "end_cycle": 150, "accepted": True},
                    {"stage": "s3", "start_cycle": 150, "end_cycle": 225, "accepted": True},
                    {"stage": "s4", "start_cycle": 225, "end_cycle": 300, "accepted": True},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = bundle / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "source-bound-timing-backend-v1",
                "backend": {
                    "kind": "source_rtl",
                    "source": {"path": "rtl/ScarfTop.sv", "sha256": _sha256(source)},
                },
                "clock_mhz": 1000,
                "samples": [
                    {
                        "model": "mvsplat",
                        "dataset": "re10k",
                        "sample_index": 0,
                        "trace": {
                            "path": "timing-trace/mvsplat/re10k/sample.json",
                            "sha256": _sha256(trace),
                        },
                        "variants": {
                            "asic": {"total_cycles": 500},
                            "asic_fsdr": {"total_cycles": 400},
                            "asic_saes": {"total_cycles": 375},
                            "asic_fsdr_saes": {"total_cycles": 300},
                        },
                        "combined_stage_cycles": {
                            "s1": 50,
                            "s2": 100,
                            "s3": 75,
                            "s4": 75,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    template_manifest, template_record = _manifest(tmp_path / "template")
    template_trace = template_manifest.parent / template_record["samples"][0]["trace"]["path"]
    trace_value = json.loads(template_trace.read_text(encoding="utf-8"))
    trace_value["source_rtl_sha256"] = _sha256(source)
    trace.write_text(json.dumps(trace_value), encoding="utf-8")
    sample = json.loads(json.dumps(template_record["samples"][0]))
    sample["trace"] = {
        "path": "timing-trace/mvsplat/re10k/sample.json",
        "sha256": _sha256(trace),
    }
    manifest.write_text(
        json.dumps({
            "schema_version": "source-bound-timing-backend-v1",
            "backend": {
                "kind": "source_rtl",
                "source": {"path": "rtl/ScarfTop.sv", "sha256": _sha256(source)},
            },
            "clock_mhz": 1000,
            "samples": [sample],
        }),
        encoding="utf-8",
    )
    backend = load_claim_timing_manifest(manifest, root=tmp_path)
    assert backend.source_relative.as_posix() == "rtl/ScarfTop.sv"
    assert backend.sample("mvsplat", "re10k", 0)["trace"]["path"] == (
        "timing-trace/mvsplat/re10k/sample.json"
    )


def test_manifest_outside_release_root_is_rejected(tmp_path: Path):
    from scripts.claim_timing_backend import load_claim_timing_manifest

    manifest, _ = _manifest(tmp_path / "external")
    with pytest.raises(ValueError, match="inside the release root"):
        load_claim_timing_manifest(manifest, root=tmp_path / "release")
