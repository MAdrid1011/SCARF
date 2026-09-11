from __future__ import annotations

import struct
from pathlib import Path


def _write_payload(path: Path) -> None:
    import json

    data = struct.pack("<4f", 0.25, 0.5, 0.75, 1.0)
    header = {
        "schema_version": "scarf-rtl-payload-v1",
        "target_free": True,
        "byte_order": "little-endian",
        "data_offset_bytes": 0,
        "tensor_count": 1,
        "tensors": [{
            "name": "feature", "dtype": "float32", "shape": [4],
            "offset_bytes": 0, "size_bytes": len(data),
        }],
    }
    for _ in range(4):
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        header["data_offset_bytes"] = 18 + len(encoded)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(b"SCARFRTL1\0" + struct.pack("<Q", len(encoded)) + encoded + data)


def test_rtl_memory_image_excludes_payload_container_header(tmp_path: Path) -> None:
    from scripts.rtl_payload import parse_payload
    from scripts.run_rtl_timing import tensor_data_image

    payload_path = tmp_path / "payload.bin"
    _write_payload(payload_path)
    parsed = parse_payload(payload_path)
    image = tensor_data_image(parsed)
    assert image == parsed.read_tensor("feature")
    assert not image.startswith(b"SCARFRTL1\0")
    assert len(image) == payload_path.stat().st_size - parsed.data_offset_bytes


def test_axi_consumption_evidence_is_role_bound_and_content_sensitive() -> None:
    from scripts.run_rtl_timing import consumption_evidence

    descriptor = {
        "tensor_data_size_bytes": 64,
        "tensor_ranges": [
            {
                "name": role, "role": role, "offset_bytes": ordinal * 16,
                "size_bytes": 16, "end_bytes": (ordinal + 1) * 16,
            }
            for ordinal, role in enumerate(("pipeline_features", "pipeline_depths", "depth_candidates", "depth_probabilities"))
        ],
    }
    addresses = [0x90000000, 0x90000010, 0x90000020, 0x90000030]
    first = consumption_evidence(descriptor, bytes(range(64)), addresses)
    changed = bytearray(range(64))
    changed[1] ^= 0xFF
    second = consumption_evidence(descriptor, bytes(changed), addresses)
    assert set(first["role_coverage"]) == {
        "pipeline_features", "pipeline_depths", "depth_candidates", "depth_probabilities",
    }
    assert all(value["bytes_read"] == 16 for value in first["role_coverage"].values())
    assert first["axi_consumption_digest"] != second["axi_consumption_digest"]


def test_axi_consumption_evidence_accepts_exact_role_offsets() -> None:
    from scripts.run_rtl_timing import consumption_evidence

    descriptor = {
        "tensor_ranges": [{
            "name": "pipeline_features", "role": "pipeline_features",
            "offset_bytes": 3, "size_bytes": 16, "end_bytes": 19,
        }],
    }
    evidence = consumption_evidence(
        descriptor, bytes(range(32)), [0x90000003]
    )
    assert evidence["role_coverage"]["pipeline_features"] == {
        "bytes_read": 16, "range_count": 1
    }
