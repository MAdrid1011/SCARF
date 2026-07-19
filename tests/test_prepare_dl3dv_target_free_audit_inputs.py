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


def _fixture(tmp_path: Path, *, sample_count: int = 1) -> tuple[Path, Path]:
    from data.convert_dl3dv import _canonical_sha256
    from scripts.compile_protocol import canonicalize_index

    if sample_count not in {1, 2}:
        raise ValueError("fixture supports one or two samples")
    scenes = ["scene-fixed", "scene-second"][:sample_count]
    raw_root = tmp_path / "raw"
    for scene in scenes:
        _write_raw_scene(raw_root, scene)
    source_plans = {
        scene: {
            "re10k": {
                "image_subdir": "images_4",
                "source_image_shape": [540, 960],
            }
        }
        for scene in scenes
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
        json.dumps(
            {
                scene: {
                    "context": [0, 5] if scene == "scene-fixed" else [1, 4],
                    "target": [1, 2, 3, 4]
                    if scene == "scene-fixed"
                    else [0, 2, 3, 5],
                }
                for scene in scenes
            }
        ),
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
                        "sample_count": sample_count,
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
    assert record["source_sample_index"] == 0
    assert record["source"]["scene_source_plans_sha256"] == convert_dl3dv._canonical_sha256(
        json.loads((raw_root / ".scarf-dl3dv-source.json").read_text(encoding="utf-8"))[
            "scene_source_plans"
        ]
    )
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


def test_target_free_audit_preparation_binds_nonzero_sample_to_context_only_identity(
    tmp_path: Path,
):
    from data.context_only_audit_input import (
        prepare_context_only_audit_input,
        validate_context_only_audit_input,
    )
    from data.prepare_dl3dv_target_free_audit_inputs import prepare_inputs

    raw_root, protocol = _fixture(tmp_path, sample_count=2)
    source_output = tmp_path / "source-audit-input"
    source_record = prepare_inputs(
        raw_root,
        output_dir=source_output,
        protocol_path=protocol,
        sample_index=1,
    )
    context_output = tmp_path / "context-only"
    context_record = prepare_context_only_audit_input(
        source_output, output_root=context_output
    )
    identity = validate_context_only_audit_input(context_output)

    assert source_record["source_sample_index"] == 1
    assert source_record["selected_sample"]["scene"] == "scene-second"
    assert source_record["selected_sample"]["context_indices"] == [1, 4]
    assert context_record["source_sample_index"] == 1
    assert identity["source_sample_index"] == 1
    assert context_record["target_rgb_included"] is False
    assert context_record["target_camera_metadata_included"] is False
    assert context_record["target_index_included"] is False


def test_target_free_audit_preparation_binds_the_requested_classic_model(
    tmp_path: Path,
):
    from data.context_only_audit_input import (
        prepare_context_only_audit_input,
        validate_context_only_audit_input,
    )
    from data.prepare_dl3dv_target_free_audit_inputs import prepare_inputs

    raw_root, protocol = _fixture(tmp_path)
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    payload["pairs"]["mvsplat/dl3dv"] = dict(
        payload["pairs"]["transplat/dl3dv"]
    )
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    source_output = tmp_path / "mvsplat-source-audit-input"
    source_record = prepare_inputs(
        raw_root,
        output_dir=source_output,
        protocol_path=protocol,
        model="mvsplat",
    )
    context_output = tmp_path / "mvsplat-context-only"
    context_record = prepare_context_only_audit_input(
        source_output, output_root=context_output, model="mvsplat"
    )

    assert source_record["model"] == "mvsplat"
    assert source_record["canonical_protocol"]["pair"] == "mvsplat/dl3dv"
    assert context_record["model"] == "mvsplat"
    validate_context_only_audit_input(context_output, model="mvsplat")
    with pytest.raises(ValueError, match="isolation contract"):
        validate_context_only_audit_input(context_output)


def test_target_free_audit_preparation_binds_depthsplat_protocol_identity(
    tmp_path: Path,
):
    from data.context_only_audit_input import (
        prepare_context_only_audit_input,
        validate_context_only_audit_input,
    )
    from data.prepare_dl3dv_target_free_audit_inputs import prepare_inputs

    raw_root, protocol = _fixture(tmp_path)
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    payload["pairs"]["depthsplat/dl3dv"] = dict(
        payload["pairs"]["transplat/dl3dv"]
    )
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    source_output = tmp_path / "depthsplat-source-audit-input"
    source_record = prepare_inputs(
        raw_root,
        output_dir=source_output,
        protocol_path=protocol,
        model="depthsplat",
    )
    context_output = tmp_path / "depthsplat-context-only"
    context_record = prepare_context_only_audit_input(
        source_output, output_root=context_output, model="depthsplat"
    )

    assert source_record["model"] == "depthsplat"
    assert source_record["canonical_protocol"]["pair"] == "depthsplat/dl3dv"
    assert context_record["model"] == "depthsplat"
    validate_context_only_audit_input(context_output, model="depthsplat")


def test_target_free_audit_preparation_rejects_negative_sample_index(tmp_path: Path):
    from data.prepare_dl3dv_target_free_audit_inputs import AuditInputError, prepare_inputs

    raw_root, protocol = _fixture(tmp_path)
    with pytest.raises(AuditInputError, match="sample index"):
        prepare_inputs(
            raw_root,
            output_dir=tmp_path / "audit-input",
            protocol_path=protocol,
            sample_index=-1,
        )


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
            "--scene",
            "scene-override",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_target_free_audit_cli_forwards_nonzero_sample_index(
    monkeypatch: pytest.MonkeyPatch,
):
    from data import prepare_dl3dv_target_free_audit_inputs as module

    captured: dict[str, object] = {}

    def fake_prepare_inputs(raw_root, *, output_dir, model, sample_index):
        captured.update(
            {
                "raw_root": raw_root,
                "output_dir": output_dir,
                "model": model,
                "sample_index": sample_index,
            }
        )
        return {"output_tree_sha256": "a" * 64}

    monkeypatch.setattr(module, "prepare_inputs", fake_prepare_inputs)
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

    assert module.main() == 0
    assert captured["model"] == "transplat"
    assert captured["sample_index"] == 1
