import io
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image


torch = pytest.importorskip("torch", reason="DL3DV chunk tests require a locked model profile")

def _write_image(path: Path, shape: tuple[int, int], color: tuple[int, int, int]) -> None:
    height, width = shape
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color).save(path, format="JPEG", quality=95)


def _make_scene(
    root: Path,
    key: str,
    *,
    native_shape: tuple[int, int] = (270, 480),
    re10k_shape: tuple[int, int] = (540, 960),
) -> None:
    scene = root / key / "nerfstudio"
    frames = []
    for index in range(2):
        name = f"frame_{index:05d}.jpg"
        _write_image(scene / "images_4" / name, native_shape, (32 + index, 64, 96))
        _write_image(scene / "images_2" / name, re10k_shape, (32 + index, 64, 96))
        transform = [
            [1.0, 0.0, 0.0, float(index)],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append({"file_path": f"images_4/{name}", "transform_matrix": transform})
    metadata = {
        "h": 270,
        "w": 480,
        "fl_x": 240.0,
        "fl_y": 240.0,
        "cx": 240.0,
        "cy": 135.0,
        "frames": frames,
    }
    (scene / "transforms.json").write_text(json.dumps(metadata), encoding="utf-8")


def _decode_size(tensor: torch.Tensor) -> tuple[int, int]:
    with Image.open(io.BytesIO(tensor.numpy().tobytes())) as image:
        return image.size


def _make_planned_scene(
    root: Path,
    key: str,
    *,
    image_shapes: dict[str, tuple[int, int]],
    metadata_shape: tuple[int, int],
) -> None:
    scene = root / key / "nerfstudio"
    frames = []
    for index in range(2):
        name = f"frame_{index:05d}.jpg"
        for subdir, shape in image_shapes.items():
            _write_image(scene / subdir / name, shape, (32 + index, 64, 96))
        frames.append(
            {
                "file_path": f"images_8/{name}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, float(index)],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    (scene / "transforms.json").write_text(
        json.dumps(
            {
                "h": metadata_shape[0],
                "w": metadata_shape[1],
                "fl_x": metadata_shape[1] / 2,
                "fl_y": metadata_shape[0] / 2,
                "cx": metadata_shape[1] / 2,
                "cy": metadata_shape[0] / 2,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def test_dl3dv_converter_emits_both_required_image_schemas(tmp_path: Path):
    from data.convert_dl3dv import convert
    from data.generate_chunk_index import generate

    raw = tmp_path / "raw"
    _make_scene(raw, "scene-0001")
    _make_scene(
        raw,
        "scene-0002",
        native_shape=(540, 960),
        re10k_shape=(1080, 1920),
    )

    native_root = tmp_path / "native"
    native = convert(
        raw,
        native_root,
        image_subdir="images_4",
        target_shape=(270, 480),
        allow_resize=True,
        accepted_source_shapes=((270, 480), (540, 960)),
        schema="depthsplat-native-270x480-v1",
        source="fixture",
        revision="fixture-v1",
        chunk_bytes=1_000_000,
    )
    re10k_root = tmp_path / "re10k"
    re10k = convert(
        raw,
        re10k_root,
        image_subdir="images_2",
        target_shape=(360, 640),
        allow_resize=True,
        accepted_source_shapes=((540, 960), (1080, 1920)),
        schema="re10k-compatible-360x640-v1",
        source="fixture",
        revision="fixture-v1",
        chunk_bytes=1_000_000,
    )

    native_chunk = torch.load(native_root / "test/000000.torch", map_location="cpu")
    re10k_chunk = torch.load(re10k_root / "test/000000.torch", map_location="cpu")
    assert native["target_image_shape"] == [270, 480]
    assert native["accepted_source_image_shapes"] == [[270, 480], [540, 960]]
    assert re10k["target_image_shape"] == [360, 640]
    assert re10k["accepted_source_image_shapes"] == [[540, 960], [1080, 1920]]
    assert _decode_size(native_chunk[0]["images"][0]) == (480, 270)
    assert _decode_size(re10k_chunk[0]["images"][0]) == (640, 360)
    assert _decode_size(native_chunk[1]["images"][0]) == (480, 270)
    assert _decode_size(re10k_chunk[1]["images"][0]) == (640, 360)
    assert native_chunk[0]["cameras"].shape == (2, 18)
    assert native_chunk[0]["cameras"][0, :4].tolist() == pytest.approx(
        [0.5, 240 / 270, 0.5, 0.5]
    )
    expected_index = {
        "scene-0001": "000000.torch",
        "scene-0002": "000000.torch",
    }
    assert generate(native_root / "test") == expected_index
    assert generate(re10k_root / "test") == expected_index


def test_dl3dv_converter_obeys_the_hashed_scene_source_plan(tmp_path: Path):
    from data.convert_dl3dv import convert, load_scene_image_sources

    raw = tmp_path / "raw"
    _make_planned_scene(
        raw,
        "scene-high",
        image_shapes={"images_8": (270, 480), "images_4": (540, 960)},
        metadata_shape=(2160, 3840),
    )
    _make_planned_scene(
        raw,
        "scene-low",
        image_shapes={
            "images_8": (135, 240),
            "images_4": (270, 480),
            "images_2": (540, 960),
        },
        metadata_shape=(1080, 1920),
    )
    source_plan = tmp_path / "source.json"
    plans = {
        "scene-high": {
            "native": {
                "image_subdir": "images_8",
                "source_image_shape": [270, 480],
            },
            "re10k": {
                "image_subdir": "images_4",
                "source_image_shape": [540, 960],
            },
        },
        "scene-low": {
            "native": {
                "image_subdir": "images_4",
                "source_image_shape": [270, 480],
            },
            "re10k": {
                "image_subdir": "images_2",
                "source_image_shape": [540, 960],
            },
        },
    }
    plan_hash = hashlib.sha256(
        json.dumps(plans, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_plan.write_text(
        json.dumps(
            {
                "scene_source_plans": plans,
                "scene_source_plans_sha256": plan_hash,
            }
        ),
        encoding="utf-8",
    )

    native = convert(
        raw,
        tmp_path / "native",
        image_subdir=None,
        target_shape=(270, 480),
        allow_resize=False,
        schema="native",
        source="fixture",
        revision="fixture-v1",
        scene_image_sources=load_scene_image_sources(source_plan, "native"),
    )
    re10k = convert(
        raw,
        tmp_path / "re10k",
        image_subdir=None,
        target_shape=(360, 640),
        allow_resize=True,
        schema="re10k",
        source="fixture",
        revision="fixture-v1",
        scene_image_sources=load_scene_image_sources(source_plan, "re10k"),
    )
    native_chunk = torch.load(tmp_path / "native/test/000000.torch", map_location="cpu")
    re10k_chunk = torch.load(tmp_path / "re10k/test/000000.torch", map_location="cpu")
    assert all(_decode_size(item["images"][0]) == (480, 270) for item in native_chunk)
    assert all(_decode_size(item["images"][0]) == (640, 360) for item in re10k_chunk)
    assert native["source_image_subdir"] is None
    assert native["scene_image_sources"]["scene-high"]["image_subdir"] == "images_8"
    assert native["scene_image_sources"]["scene-low"]["image_subdir"] == "images_4"
    assert re10k["scene_image_sources"]["scene-high"]["image_subdir"] == "images_4"
    assert re10k["scene_image_sources"]["scene-low"]["image_subdir"] == "images_2"
    assert len(native["scene_image_sources_sha256"]) == 64


def test_dl3dv_converter_rejects_a_tampered_scene_source_plan(tmp_path: Path):
    from data.convert_dl3dv import load_scene_image_sources

    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(
            {
                "scene_source_plans": {
                    "scene-a": {
                        "native": {
                            "image_subdir": "images_8",
                            "source_image_shape": [270, 480],
                        }
                    }
                },
                "scene_source_plans_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="plan SHA256 mismatch"):
        load_scene_image_sources(path, "native")


def test_dl3dv_converter_rejects_nonempty_output(tmp_path: Path):
    from data.convert_dl3dv import convert

    raw = tmp_path / "raw"
    _make_scene(raw, "scene-0001")
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        convert(
            raw,
            output,
            image_subdir="images_4",
            target_shape=(270, 480),
            allow_resize=False,
            schema="native",
            source="fixture",
            revision="fixture-v1",
        )


def test_dl3dv_converter_rejects_the_wrong_resize_source_resolution(tmp_path: Path):
    from data.convert_dl3dv import convert

    raw = tmp_path / "raw"
    _make_scene(raw, "scene-0001")
    index = tmp_path / "index.json"
    index.write_text('{"scene-0001": {}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected one of 540x960, 1080x1920"):
        convert(
            raw,
            tmp_path / "prepared",
            image_subdir="images_4",
            target_shape=(360, 640),
            allow_resize=True,
            accepted_source_shapes=((540, 960), (1080, 1920)),
            schema="re10k",
            source="fixture",
            revision="fixture-v1",
            scene_index=index,
        )


def test_dl3dv_converter_uses_only_the_fixed_scene_index(tmp_path: Path):
    from data.convert_dl3dv import convert

    raw = tmp_path / "raw"
    _make_scene(raw, "scene-keep")
    _make_scene(raw, "scene-ignore")
    index = tmp_path / "index.json"
    index.write_text('{"scene-keep": {}}\n', encoding="utf-8")

    record = convert(
        raw,
        tmp_path / "prepared",
        image_subdir="images_4",
        target_shape=(270, 480),
        allow_resize=False,
        schema="native",
        source="fixture",
        revision="fixture-v1",
        scene_index=index,
    )

    assert record["scene_keys"] == ["scene-keep"]
    assert record["scene_index"] == str(index)
    assert len(record["scene_index_sha256"]) == 64


def test_dl3dv_converter_rejects_an_indexed_scene_missing_from_source(tmp_path: Path):
    from data.convert_dl3dv import convert

    raw = tmp_path / "raw"
    _make_scene(raw, "scene-present")
    index = tmp_path / "index.json"
    index.write_text('{"scene-missing": {}}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing 1 indexed scenes"):
        convert(
            raw,
            tmp_path / "prepared",
            image_subdir="images_4",
            target_shape=(270, 480),
            allow_resize=False,
            schema="native",
            source="fixture",
            revision="fixture-v1",
            scene_index=index,
        )
