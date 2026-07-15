import io
import json
from pathlib import Path

import pytest
from PIL import Image


torch = pytest.importorskip("torch", reason="DL3DV chunk tests require a locked model profile")

def _write_image(path: Path, shape: tuple[int, int], color: tuple[int, int, int]) -> None:
    height, width = shape
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color).save(path, format="JPEG", quality=95)


def _make_scene(root: Path, key: str) -> None:
    scene = root / key / "nerfstudio"
    frames = []
    for index in range(2):
        name = f"frame_{index:05d}.jpg"
        _write_image(scene / "images_8" / name, (270, 480), (32 + index, 64, 96))
        _write_image(scene / "images_4" / name, (540, 960), (32 + index, 64, 96))
        transform = [
            [1.0, 0.0, 0.0, float(index)],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append({"file_path": f"images_8/{name}", "transform_matrix": transform})
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


def test_dl3dv_converter_emits_both_required_image_schemas(tmp_path: Path):
    from data.convert_dl3dv import convert
    from data.generate_chunk_index import generate

    raw = tmp_path / "raw"
    _make_scene(raw, "scene-0001")

    native_root = tmp_path / "native"
    native = convert(
        raw,
        native_root,
        image_subdir="images_8",
        target_shape=(270, 480),
        allow_resize=False,
        schema="depthsplat-native-270x480-v1",
        source="fixture",
        revision="fixture-v1",
        chunk_bytes=1_000_000,
    )
    re10k_root = tmp_path / "re10k"
    re10k = convert(
        raw,
        re10k_root,
        image_subdir="images_4",
        target_shape=(360, 640),
        allow_resize=True,
        schema="re10k-compatible-360x640-v1",
        source="fixture",
        revision="fixture-v1",
        chunk_bytes=1_000_000,
    )

    native_chunk = torch.load(native_root / "test/000000.torch", map_location="cpu")
    re10k_chunk = torch.load(re10k_root / "test/000000.torch", map_location="cpu")
    assert native["target_image_shape"] == [270, 480]
    assert re10k["target_image_shape"] == [360, 640]
    assert _decode_size(native_chunk[0]["images"][0]) == (480, 270)
    assert _decode_size(re10k_chunk[0]["images"][0]) == (640, 360)
    assert native_chunk[0]["cameras"].shape == (2, 18)
    assert native_chunk[0]["cameras"][0, :4].tolist() == pytest.approx(
        [0.5, 240 / 270, 0.5, 0.5]
    )
    assert generate(native_root / "test") == {"scene-0001": "000000.torch"}
    assert generate(re10k_root / "test") == {"scene-0001": "000000.torch"}


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
            image_subdir="images_8",
            target_shape=(270, 480),
            allow_resize=False,
            schema="native",
            source="fixture",
            revision="fixture-v1",
        )
