import json
import math
import struct
from pathlib import Path

import pytest


def _write_payload(path: Path, *, seed: int = 0, specs=None) -> None:
    specs = specs or [
        ("context_image", [1, 3, 4, 4]),
        ("pipeline_features", [1, 4, 4, 4]),
        ("depth_candidates", [8]),
        ("gaussian_means", [1, 5, 3]),
    ]
    entries = []
    chunks = []
    offset = 0
    for name, shape in specs:
        elements = 1
        for dimension in shape:
            elements *= dimension
        data = bytes((seed + index) % 256 for index in range(elements * 4))
        entries.append(
            {
                "name": name,
                "dtype": "float32",
                "shape": shape,
                "offset_bytes": offset,
                "size_bytes": len(data),
            }
        )
        chunks.append(data)
        offset += len(data)
    header = {
        "schema_version": "scarf-rtl-payload-v1",
        "target_free": True,
        "byte_order": "little-endian",
        "data_offset_bytes": 0,
        "tensor_count": len(entries),
        "tensors": entries,
    }
    for _ in range(5):
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        header["data_offset_bytes"] = 18 + len(encoded)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(b"SCARFRTL1\0" + struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def _write_v2_claim_payload(
    path: Path,
    *,
    model: str = "mvsplat",
    views: int = 2,
    height: int = 2,
    width: int = 3,
    feature_dim: int = 4,
    depth_candidates: int = 8,
    mutate: dict[str, list[float]] | None = None,
) -> None:
    """Write a minimal runtime-shaped v2 payload with all claim-critical roles."""
    shapes = {
        "context_image": [1, views, 3, height, width],
        "pipeline_features": [1, views, feature_dim, height, width],
        "pipeline_depths": [1, views, height, width],
        "depth_candidates": [1, views, depth_candidates],
        "depth_probabilities": [1, views, depth_candidates, height, width],
        "gaussian_means": [1, views * height * width * 2, 3],
    }
    layouts = {
        "context_image": ("BVCHW", "context_image"),
        "pipeline_features": ("BVCHW", "pipeline_features"),
        "pipeline_depths": ("BVHW", "pipeline_depths"),
        "depth_candidates": ("BVD", "depth_candidates"),
        "depth_probabilities": ("BVDHW", "depth_probabilities"),
        "gaussian_means": ("BNC", "gaussian_means"),
    }
    entries, chunks, bindings = [], [], {}
    offset = 0
    for ordinal, (name, shape) in enumerate(shapes.items()):
        count = math.prod(shape)
        values = (mutate or {}).get(name)
        if values is None:
            values = [0.125 + ordinal + index / 100.0 for index in range(count)]
        assert len(values) == count
        data = struct.pack(f"<{count}f", *values)
        layout, role = layouts[name]
        entries.append({
            "name": name, "dtype": "float32", "shape": shape,
            "offset_bytes": offset, "size_bytes": len(data),
            "layout": layout, "role": role,
        })
        bindings[name] = {"layout": layout, "role": role}
        chunks.append(data)
        offset += len(data)
    header = {
        "schema_version": "scarf-rtl-payload-v2",
        "target_free": True,
        "byte_order": "little-endian",
        "data_offset_bytes": 0,
        "tensor_count": len(entries),
        "tensors": entries,
        "workload": {
            "schema_version": "scarf-rtl-workload-v2",
            "model": model,
            "image_h": height,
            "image_w": width,
            "feature_dim": feature_dim,
            "num_depth_candidates": depth_candidates,
            "num_gaussians": views * height * width * 2,
            "view_count": views,
            "primitives_per_pixel": 2,
            "tensor_bindings": bindings,
        },
    }
    for _ in range(8):
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        header["data_offset_bytes"] = 18 + len(encoded)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(b"SCARFRTL1\0" + struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def test_payload_descriptor_is_self_describing_and_content_bound(tmp_path: Path):
    from scripts.rtl_payload import parse_payload, workload_descriptor

    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    _write_payload(first, seed=0)
    _write_payload(second, seed=1)
    first_descriptor = workload_descriptor(parse_payload(first))
    second_descriptor = workload_descriptor(parse_payload(second))
    assert first_descriptor["image_h"] == 4
    assert first_descriptor["feature_dim"] == 4
    assert first_descriptor["num_depth_candidates"] == 8
    assert first_descriptor["num_gaussians"] == 5
    assert first_descriptor["data_offset_bytes"] > 18
    assert first_descriptor["descriptor_sha256"] != second_descriptor["descriptor_sha256"]


def test_payload_descriptor_tracks_workload_dimensions(tmp_path: Path):
    from scripts.rtl_payload import parse_payload, workload_descriptor

    small = tmp_path / "small.bin"
    large = tmp_path / "large.bin"
    _write_payload(small)
    _write_payload(
        large,
        specs=[
            ("context_image", [1, 3, 8, 8]),
            ("pipeline_features", [1, 8, 8, 8]),
            ("depth_candidates", [16]),
            ("gaussian_means", [1, 10, 3]),
        ],
    )
    small_workload = workload_descriptor(parse_payload(small))
    large_workload = workload_descriptor(parse_payload(large))
    assert (small_workload["image_h"], small_workload["image_w"]) == (4, 4)
    assert (large_workload["image_h"], large_workload["image_w"]) == (8, 8)
    assert large_workload["feature_dim"] == 8
    assert large_workload["num_depth_candidates"] == 16
    assert large_workload["num_gaussians"] == 10
    assert large_workload["descriptor_sha256"] != small_workload["descriptor_sha256"]


def test_mvsplat_four_dimensional_depth_layout_keeps_candidate_axis_explicit():
    from scripts.rtl_payload import infer_tensor_layout

    assert infer_tensor_layout("depth_candidates", (1, 128, 64, 64)) == (
        "BDHW",
        "depth_candidates",
    )
    assert infer_tensor_layout("depth_candidates", (1, 2, 128, 1, 1)) == (
        "BVDHW",
        "depth_candidates",
    )


def test_mvsplat_runtime_candidate_vector_keeps_view_and_candidate_axes_explicit():
    from scripts.rtl_payload import infer_tensor_layout

    assert infer_tensor_layout("depth_candidates", (1, 2, 32)) == (
        "BVD",
        "depth_candidates",
    )
    assert infer_tensor_layout("depth_candidates", (2, 32)) == (
        "VD",
        "depth_candidates",
    )


def test_mvsplat_probability_volume_keeps_candidate_axis_explicit():
    from scripts.rtl_payload import infer_tensor_layout

    assert infer_tensor_layout("depth_probabilities", (1, 2, 128, 64, 64)) == (
        "BVDHW",
        "depth_probabilities",
    )


def test_runtime_flattened_depths_keep_pixel_axis_and_critical_role_explicit():
    from scripts.rtl_payload import infer_tensor_layout

    assert infer_tensor_layout("pipeline_depths", (1, 2, 65536, 1, 1)) == (
        "BVPHW",
        "pipeline_depths",
    )


def test_auxiliary_layout_accepts_unreserved_axis_labels():
    from scripts.rtl_payload import _validate_layout

    assert _validate_layout("ABCD", (1, 2, 3, 4), "auxiliary.layout") == "ABCD"


@pytest.mark.parametrize("model,views,shape", [
    ("transplat", 2, (1, 2, 4, 2, 3)),
    ("mvsplat", 2, (1, 2, 4, 2, 3)),
    ("depthsplat", 3, (1, 3, 4, 2, 3)),
])
def test_claim_binding_table_describes_runtime_tensor_layouts(tmp_path: Path, model, views, shape):
    from scripts.rtl_payload import parse_payload, rtl_tensor_bindings

    payload_path = tmp_path / f"{model}.bin"
    _write_v2_claim_payload(payload_path, model=model, views=views)
    bindings = rtl_tensor_bindings(parse_payload(payload_path))
    feature = bindings["pipeline_features"]
    depths = bindings["pipeline_depths"]
    probabilities = bindings["depth_probabilities"]
    assert feature["shape"] == list(shape)
    assert feature["strides_bytes"] == [views * 4 * 2 * 3 * 4, 4 * 2 * 3 * 4, 2 * 3 * 4, 3 * 4, 4]
    assert feature["data_offset_bytes"] == 1 * views * 3 * 2 * 3 * 4
    assert depths["layout"] == "BVHW"
    assert probabilities["layout"] == "BVDHW"
    assert probabilities["role"] == "depth_probabilities"


def test_claim_binding_table_ignores_repeated_noncritical_auxiliary_roles(tmp_path: Path):
    from scripts.rtl_payload import parse_payload, rtl_tensor_bindings

    payload_path = tmp_path / "auxiliary.bin"
    _write_v2_claim_payload(payload_path)
    raw = payload_path.read_bytes()
    header_length = struct.unpack("<Q", raw[10:18])[0]
    header = json.loads(raw[18 : 18 + header_length].decode())
    tensor_data = raw[header["data_offset_bytes"] :]
    for name in ("context_image", "gaussian_means"):
        binding = header["workload"]["tensor_bindings"][name]
        binding["role"] = "auxiliary"
        next(item for item in header["tensors"] if item["name"] == name)["role"] = "auxiliary"
    for _ in range(8):
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        header["data_offset_bytes"] = 18 + len(encoded)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    payload_path.write_bytes(
        b"SCARFRTL1\0" + struct.pack("<Q", len(encoded)) + encoded + tensor_data
    )

    bindings = rtl_tensor_bindings(parse_payload(payload_path))
    assert set(bindings) == {
        "pipeline_features", "pipeline_depths", "depth_candidates", "depth_probabilities"
    }


@pytest.mark.parametrize("role", [
    "pipeline_features", "pipeline_depths", "depth_candidates", "depth_probabilities",
])
def test_claim_payload_accepts_constant_critical_role_as_diagnostic(tmp_path: Path, role: str):
    from scripts.rtl_payload import constant_claim_roles, parse_payload, validate_claim_payload

    path = tmp_path / f"constant-{role}.bin"
    counts = {
        "pipeline_features": 1 * 2 * 4 * 2 * 3,
        "pipeline_depths": 1 * 2 * 2 * 3,
        "depth_candidates": 1 * 2 * 8,
        "depth_probabilities": 1 * 2 * 8 * 2 * 3,
    }
    _write_v2_claim_payload(path, mutate={role: [0.0] * counts[role]})
    payload = parse_payload(path)
    validate_claim_payload(payload)
    assert role in constant_claim_roles(payload)


def test_claim_payload_rejects_nonfinite_critical_value(tmp_path: Path):
    from scripts.rtl_payload import parse_payload, validate_claim_payload

    path = tmp_path / "nan.bin"
    values = [0.25 + index for index in range(1 * 2 * 4 * 2 * 3)]
    values[3] = float("nan")
    _write_v2_claim_payload(path, mutate={"pipeline_features": values})
    with pytest.raises(ValueError, match="pipeline_features.*non-finite"):
        validate_claim_payload(parse_payload(path))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda header: header.update({"target_free": False}), "target_free"),
        (lambda header: header.update({"data_offset_bytes": 17}), "data_offset_bytes"),
    ],
)
def test_payload_parser_rejects_invalid_header(tmp_path: Path, mutation, match):
    from scripts.rtl_payload import parse_payload

    path = tmp_path / "payload.bin"
    _write_payload(path)
    raw = path.read_bytes()
    header_length = struct.unpack("<Q", raw[10:18])[0]
    header = json.loads(raw[18 : 18 + header_length].decode())
    mutation(header)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(b"SCARFRTL1\0" + struct.pack("<Q", len(encoded)) + encoded + raw[18 + header_length :])
    with pytest.raises(ValueError, match=match):
        parse_payload(path)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("empty", "shorter than its binary header"),
        ("short", "shorter than its binary header"),
        ("magic", "magic is invalid"),
        ("trailing", "trailing or missing bytes"),
    ],
)
def test_payload_parser_rejects_non_payload_bytes(tmp_path: Path, kind, expected):
    from scripts.rtl_payload import parse_payload

    path = tmp_path / f"{kind}.bin"
    if kind == "empty":
        path.write_bytes(b"")
    elif kind == "short":
        path.write_bytes(b"SCARFRTL1\0")
    else:
        _write_payload(path)
        raw = bytearray(path.read_bytes())
        if kind == "magic":
            raw[0] ^= 0x01
        else:
            raw.append(0)
        path.write_bytes(raw)
    with pytest.raises(ValueError, match=expected):
        parse_payload(path)
