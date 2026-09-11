from __future__ import annotations

import json
from pathlib import Path


def _workload() -> dict:
    from scripts.rtl_payload import descriptor_sha256

    roles = ("pipeline_features", "pipeline_depths", "depth_candidates", "depth_probabilities")
    value = {
        "schema_version": "scarf-rtl-workload-v2",
        "image_h": 1, "image_w": 1, "feature_dim": 1,
        "num_depth_candidates": 1, "num_gaussians": 1, "view_count": 1,
        "primitives_per_pixel": 1, "tensor_count": 4,
        "payload_size_bytes": 164, "data_offset_bytes": 100,
        "tensor_data_size_bytes": 64, "tensor_data_sha256": "c" * 64,
        "tensor_ranges": [
            {
                "name": role, "dtype": "float32", "shape": [4],
                "offset_bytes": ordinal * 16, "size_bytes": 16,
                "end_bytes": (ordinal + 1) * 16, "layout": "A", "role": role,
            }
            for ordinal, role in enumerate(roles)
        ],
    }
    value["descriptor_sha256"] = descriptor_sha256(value)
    return value


def _sample() -> dict:
    return {
        "model": "mvsplat",
        "dataset": "re10k",
        "provenance": {
            "evaluation": {
                "kind": "sample",
                "sample_index": 0,
                "execution_index": 0,
                "scene": "scene-000",
                "context_indices": [0, 1],
                "target_indices": [2],
            }
        },
    }


def _events() -> dict:
    return {
        "schema_version": "scarf-rtl-simulator-events-v2",
        "clock_mhz": 1000,
        "claim_eligible": True,
        "samples": [
            {
                "model": "mvsplat",
                "dataset": "re10k",
                "sample_index": 0,
                "events": [
                    {"stage": "s1", "start_cycle": 0, "end_cycle": 10, "accepted": True},
                    {"stage": "s2", "start_cycle": 10, "end_cycle": 30, "accepted": True},
                    {"stage": "s3", "start_cycle": 30, "end_cycle": 45, "accepted": True},
                    {"stage": "s4", "start_cycle": 45, "end_cycle": 55, "accepted": True},
                ],
                "variants": {
                    "asic": {"total_cycles": 55},
                    "asic_fsdr": {"total_cycles": 45},
                    "asic_saes": {"total_cycles": 40},
                    "asic_fsdr_saes": {"total_cycles": 35},
                },
                "stimulus_sha256": "a" * 64,
                "stimulus_size_bytes": 164,
                "workload": _workload(),
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
        ],
    }


def test_sample_to_rtl_stimulus_binds_result_hash(tmp_path: Path) -> None:
    from scripts.export_rtl_stimulus import convert

    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(_sample()), encoding="utf-8")
    output = convert(sample, tmp_path / "stimulus.json", input_root=tmp_path)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema_version"] == "scarf-rtl-stimulus-v1"
    assert record["source_record"]["sha256"]
    assert record["selection"]["scene"] == "scene-000"


def test_sample_to_rtl_stimulus_packs_and_hashes_payload(tmp_path: Path) -> None:
    from scripts.export_rtl_stimulus import convert

    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(_sample()), encoding="utf-8")
    payload_a = tmp_path / "features.bin"
    payload_b = tmp_path / "depth.bin"
    payload_a.write_bytes(b"feature-bytes")
    payload_b.write_bytes(b"depth-bytes")
    output = convert(
        sample,
        tmp_path / "stimulus.json",
        input_root=tmp_path,
        payload=[payload_a, payload_b],
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    descriptor = record["payload"]
    packed = output.parent / descriptor["path"]
    assert descriptor["required"] is True
    assert descriptor["size_bytes"] == len(b"feature-bytesdepth-bytes")
    assert packed.read_bytes() == b"feature-bytesdepth-bytes"
    assert descriptor["sha256"]
    assert [item["offset_bytes"] for item in descriptor["files"]] == [0, 13]


def test_rtl_event_export_is_accepted_by_claim_loader(tmp_path: Path) -> None:
    from scripts.claim_timing_backend import load_claim_timing_manifest
    from scripts.export_claim_timing import export

    source = tmp_path / "ScarfTop.sv"
    source.write_text("module ScarfTop(input clock); endmodule\n", encoding="utf-8")
    events = tmp_path / "events.json"
    events.write_text(json.dumps(_events()), encoding="utf-8")
    bundle = tmp_path / "timing-backend"
    manifest = export(events, source, bundle)
    loaded = load_claim_timing_manifest(manifest, root=bundle)
    assert loaded.sample("mvsplat", "re10k", 0)["variants"]["asic"]["total_cycles"] == 55
    assert loaded.sample("mvsplat", "re10k", 0)["combined_stage_cycles"] == {
        "s1": 10,
        "s2": 20,
        "s3": 15,
        "s4": 10,
    }


def test_rtl_event_export_preserves_stimulus_binding(tmp_path: Path) -> None:
    from scripts.claim_timing_backend import load_claim_timing_manifest
    from scripts.export_claim_timing import export

    source = tmp_path / "ScarfTop.sv"
    source.write_text("module ScarfTop; endmodule\n", encoding="utf-8")
    events = _events()
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events), encoding="utf-8")
    manifest = export(path, source, tmp_path / "bundle")
    load_claim_timing_manifest(manifest, root=tmp_path / "bundle")
    trace = tmp_path / "bundle" / "timing-trace/mvsplat/re10k/sample_00000.json"
    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["stimulus_sha256"] == "a" * 64


def test_claim_event_export_requires_stimulus_binding(tmp_path: Path) -> None:
    from scripts.export_claim_timing import export

    source = tmp_path / "ScarfTop.sv"
    source.write_text("module ScarfTop; endmodule\n", encoding="utf-8")
    events = _events()
    events["samples"][0].pop("stimulus_sha256")
    events["samples"][0].pop("stimulus_size_bytes")
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events), encoding="utf-8")
    try:
        export(path, source, tmp_path / "bundle")
    except ValueError as exc:
        assert "stimulus binding" in str(exc)
    else:
        raise AssertionError("claim export accepted an unbound stimulus")


def test_structural_event_export_is_rejected(tmp_path: Path) -> None:
    from scripts.export_claim_timing import export

    source = tmp_path / "ScarfTop.sv"
    source.write_text("module ScarfTop; endmodule\n", encoding="utf-8")
    events = _events()
    events["claim_eligible"] = False
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events), encoding="utf-8")
    try:
        export(path, source, tmp_path / "bundle")
    except ValueError as exc:
        assert "non-claim" in str(exc)
    else:
        raise AssertionError("structural event export entered claim path")


def test_rtl_event_export_rejects_noncontiguous_events(tmp_path: Path) -> None:
    from scripts.export_claim_timing import export

    source = tmp_path / "ScarfTop.sv"
    source.write_text("module ScarfTop; endmodule\n", encoding="utf-8")
    data = _events()
    data["samples"][0]["events"][1]["start_cycle"] = 11
    events = tmp_path / "events.json"
    events.write_text(json.dumps(data), encoding="utf-8")
    try:
        export(events, source, tmp_path / "bundle")
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("invalid stage event was accepted")
