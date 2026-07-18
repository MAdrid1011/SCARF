import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image


torch = pytest.importorskip("torch", reason="target-free audit sidecars require torch")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raw_scene(root: Path, scene: str) -> None:
    nerfstudio = root / scene / "nerfstudio"
    frames = []
    for view in range(6):
        name = f"frame_{view:05d}.jpg"
        image_path = nerfstudio / "images_4" / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (960, 540), (view, 17, 31)).save(image_path, format="JPEG")
        frames.append(
            {
                "file_path": f"images_4/{name}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, float(view)],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    (nerfstudio / "transforms.json").write_text(
        json.dumps(
            {
                "h": 540,
                "w": 960,
                "fl_x": 480.0,
                "fl_y": 270.0,
                "cx": 480.0,
                "cy": 270.0,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    from data.convert_dl3dv import _canonical_sha256
    from scripts.compile_protocol import canonicalize_index

    scene = "scene-fixed"
    raw_root = tmp_path / "raw"
    _write_raw_scene(raw_root, scene)
    source_plans = {
        scene: {
            "re10k": {
                "image_subdir": "images_4",
                "source_image_shape": [540, 960],
            }
        }
    }
    (raw_root / ".scarf-dl3dv-source.json").write_text(
        json.dumps(
            {
                "revision": "a" * 40,
                "benchmark_metadata_sha256": "b" * 64,
                "filelist_sha256": "c" * 64,
                "scene_source_plans": source_plans,
                "scene_source_plans_sha256": _canonical_sha256(source_plans),
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "official-index.json"
    index.write_text(
        json.dumps({scene: {"context": [0, 5], "target": [1, 2, 3, 4]}}),
        encoding="utf-8",
    )
    index_sha256 = _sha256(index)
    _, summary = canonicalize_index(index, index_sha256)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "pairs": {
                    "transplat/dl3dv": {
                        "index_path": str(index),
                        "source_index_sha256": index_sha256,
                        "sample_selection_sha256": summary["sample_selection_sha256"],
                        "sample_count": 1,
                        "dataset_tree_sha256": "d" * 64,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return raw_root, protocol


def test_target_free_audit_preparation_opens_only_context_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from data import convert_dl3dv
    from data.prepare_dl3dv_target_free_audit_inputs import prepare_inputs
    from data.verify_prepared_dataset import verify_tree_manifest
    from scripts.calibration_inputs import load_target_free_record, validate_target_free_input_root

    raw_root, protocol = _fixture(tmp_path)
    original_open = convert_dl3dv.Image.open
    opened: list[str] = []

    def tracked_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            opened.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(convert_dl3dv.Image, "open", tracked_open)
    output = tmp_path / "audit-input"
    record = prepare_inputs(raw_root, output_dir=output, protocol_path=protocol)

    assert sorted(opened) == ["frame_00000.jpg", "frame_00005.jpg"]
    assert record["target_rgb_included"] is False
    assert record["target_rgb_opened"] is False
    assert record["target_rgb_paths_passed_to_encoder"] is False
    assert record["opened_source_file_count"] == 3
    assert [entry["role"] for entry in record["opened_source_files"]] == [
        "camera_geometry",
        "context_rgb",
        "context_rgb",
    ]
    identity = validate_target_free_input_root(output / "sidecar", "dl3dv")
    assert identity["target_rgb_accessed"] is False
    sidecar = load_target_free_record(output / "sidecar", "scene-fixed")
    assert len(sidecar["context_images"]) == 2
    assert all(key not in sidecar for key in ("images", "target_images", "target_rgb"))
    tree = verify_tree_manifest(output, output / ".scarf-manifest.json")
    assert tree["tree_sha256"] == record["output_tree_sha256"]


def test_target_free_audit_preparation_rejects_context_target_overlap(tmp_path: Path):
    from data.prepare_dl3dv_target_free_audit_inputs import AuditInputError, prepare_inputs

    raw_root, protocol = _fixture(tmp_path)
    source = json.loads(protocol.read_text(encoding="utf-8"))
    index = Path(source["pairs"]["transplat/dl3dv"]["index_path"])
    index.write_text(
        json.dumps({"scene-fixed": {"context": [0, 1], "target": [1, 2, 3, 4]}}),
        encoding="utf-8",
    )
    source["pairs"]["transplat/dl3dv"]["source_index_sha256"] = _sha256(index)
    protocol.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(AuditInputError, match="context and target views overlap"):
        prepare_inputs(raw_root, output_dir=tmp_path / "audit-input", protocol_path=protocol)


def test_target_free_audit_cli_has_no_scene_or_threshold_override(
    monkeypatch: pytest.MonkeyPatch,
):
    from data.prepare_dl3dv_target_free_audit_inputs import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_dl3dv_target_free_audit_inputs.py",
            "--output-dir",
            "outputs/new-audit-input",
            "--sample-index",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
