import json
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _png(value: int) -> bytes:
    from PIL import Image

    stream = BytesIO()
    Image.new("RGB", (8, 8), (value, value, value)).save(stream, format="PNG")
    return stream.getvalue()


def _source_input(tmp_path: Path, *, source_sample_index: int = 0) -> Path:
    from data.build_manifest import build
    from scripts.calibration_inputs import materialize_target_free_inputs, sha256_file
    from scripts.compile_protocol import canonicalize_index

    root = tmp_path / "source"
    root.mkdir()
    scene = "scene-fixed"
    selection = {scene: {"context": [0, 4], "target": [1, 2, 3, 5]}}
    selection_path = root / "audit-selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    _, summary = canonicalize_index(selection_path, sha256_file(selection_path))
    cameras = torch.zeros(6, 18)
    cameras[:, 0] = 1.0
    cameras[:, 1] = 1.0
    cameras[:, 2] = 0.5
    cameras[:, 3] = 0.5
    cameras[:, 6:] = torch.eye(4)[:3].reshape(1, -1)
    cameras[4, 6 + 3] = 0.25
    materialize_target_free_inputs(
        root / "sidecar",
        dataset="dl3dv",
        examples={
            scene: {
                "key": scene,
                "cameras": cameras,
                "images": [_png(value) for value in range(6)],
            }
        },
        index=selection,
        source={},
        selection_sha256=summary["sample_selection_sha256"],
    )
    source_record = {
        "kind": "dl3dv_target_free_l1_primary_reference_audit_input",
        "status": "PASS",
        "model": "transplat",
        "dataset": "dl3dv",
        "source_sample_index": source_sample_index,
        "target_rgb_included": False,
        "selected_sample": {
            "scene": scene,
            "context_indices": [0, 4],
            "target_indices": [1, 2, 3, 5],
        },
        "canonical_selection": {
            "source_sample_index": source_sample_index,
            "scene": scene,
            "context_indices": [0, 4],
            "target_indices": [1, 2, 3, 5],
        },
        "canonical_protocol": {
            "source_index_sha256": "a" * 64,
            "sample_selection_sha256": "b" * 64,
        },
    }
    (root / "audit-input.json").write_text(json.dumps(source_record), encoding="utf-8")
    manifest = build(root, "source", "fixture", "fixture")
    (root / ".scarf-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_context_only_preparation_removes_target_rows_and_metadata(tmp_path: Path):
    from data.context_only_audit_input import (
        load_context_only_audit_record,
        prepare_context_only_audit_input,
        validate_context_only_audit_input,
    )

    source = _source_input(tmp_path)
    output = tmp_path / "context-only"
    record = prepare_context_only_audit_input(source, output_root=output)
    identity = validate_context_only_audit_input(output)
    payload = load_context_only_audit_record(output)

    assert record["target_rgb_included"] is False
    assert record["target_camera_metadata_included"] is False
    assert record["target_index_included"] is False
    assert record["source_sample_index"] == 0
    assert identity["target_rgb_accessed"] is False
    assert identity["target_camera_metadata_accessed"] is False
    assert identity["source_sample_index"] == 0
    assert set(payload) == {
        "schema_version",
        "kind",
        "key",
        "context_indices",
        "context_cameras",
        "context_images",
        "input_identity",
    }
    assert payload["context_indices"] == [0, 4]
    assert payload["context_cameras"].shape == (2, 18)
    assert payload["context_cameras"][1, 9].item() == pytest.approx(0.25)
    assert len(payload["context_images"]) == 2


def test_context_only_preparation_preserves_nonzero_source_sample_identity(
    tmp_path: Path,
):
    from data.context_only_audit_input import (
        load_context_only_audit_record,
        prepare_context_only_audit_input,
        validate_context_only_audit_input,
    )

    source = _source_input(tmp_path, source_sample_index=7)
    output = tmp_path / "context-only"
    record = prepare_context_only_audit_input(source, output_root=output)
    identity = validate_context_only_audit_input(output)
    payload = load_context_only_audit_record(output)

    assert record["source_sample_index"] == 7
    assert identity["source_sample_index"] == 7
    assert payload["input_identity"]["source_sample_index"] == 7
    assert "target" not in payload
    assert record["target_rgb_included"] is False
    assert record["target_camera_metadata_included"] is False
    assert record["target_index_included"] is False


def test_context_only_preparation_rejects_negative_source_sample_index(tmp_path: Path):
    from data.context_only_audit_input import prepare_context_only_audit_input

    source = _source_input(tmp_path, source_sample_index=-1)
    with pytest.raises(ValueError, match="source sample index"):
        prepare_context_only_audit_input(source, output_root=tmp_path / "context-only")


def test_context_only_preparation_rejects_mismatched_canonical_selection(tmp_path: Path):
    from data.build_manifest import build
    from data.context_only_audit_input import prepare_context_only_audit_input

    source = _source_input(tmp_path, source_sample_index=7)
    record_path = source / "audit-input.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["canonical_selection"]["source_sample_index"] = 6
    record_path.write_text(json.dumps(record), encoding="utf-8")
    manifest = build(source, "source", "fixture", "fixture")
    (source / ".scarf-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical selection"):
        prepare_context_only_audit_input(source, output_root=tmp_path / "context-only")


def test_context_only_loader_never_returns_a_target_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from data.context_only_audit_input import prepare_context_only_audit_input
    from integration.model_loader import load_context_only_audit_data

    source = _source_input(tmp_path)
    output = tmp_path / "context-only"
    prepare_context_only_audit_input(source, output_root=output)

    src = ModuleType("src")
    dataset = ModuleType("src.dataset")
    shims = ModuleType("src.dataset.shims")
    crop = ModuleType("src.dataset.shims.crop_shim")
    patch = ModuleType("src.dataset.shims.patch_shim")
    crop.apply_crop_shim_to_views = lambda views, _shape: views
    patch.apply_patch_shim_to_views = lambda views, _patch_size: views
    src.dataset = dataset
    dataset.shims = shims
    shims.crop_shim = crop
    shims.patch_shim = patch
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.dataset", dataset)
    monkeypatch.setitem(sys.modules, "src.dataset.shims", shims)
    monkeypatch.setitem(sys.modules, "src.dataset.shims.crop_shim", crop)
    monkeypatch.setitem(sys.modules, "src.dataset.shims.patch_shim", patch)

    class Loader:
        def _setup_imports(self):
            return None

        def _restore_cwd(self):
            return None

    bundle = SimpleNamespace(
        encoder=SimpleNamespace(
            cfg=SimpleNamespace(shim_patch_size=1, downscale_factor=1)
        ),
        config=SimpleNamespace(
            dataset=SimpleNamespace(
                image_shape=(8, 8),
                make_baseline_1=False,
                near=1.0,
                far=2.0,
                baseline_scale_bounds=False,
            )
        ),
    )
    batch = load_context_only_audit_data(Loader(), bundle, input_root=output).batch

    assert "target" not in batch
    assert batch["context"]["image"].shape == (1, 2, 3, 8, 8)
    assert batch["context"]["extrinsics"].shape == (1, 2, 4, 4)
    assert batch["calibration"]["target_mapping_present"] is False
    assert batch["calibration"]["target_camera_metadata_accessed"] is False


def test_context_only_validator_rejects_unexpected_payload_fields(tmp_path: Path):
    from data.build_manifest import build
    from data.context_only_audit_input import (
        prepare_context_only_audit_input,
        validate_context_only_audit_input,
    )
    from scripts.calibration_inputs import sha256_file

    source = _source_input(tmp_path)
    output = tmp_path / "context-only"
    prepare_context_only_audit_input(source, output_root=output)
    chunk_path = output / "sidecar" / "test" / "000000.torch"
    chunk = torch.load(chunk_path, map_location="cpu")
    chunk[0]["teacher_artifact"] = "forbidden"
    torch.save(chunk, chunk_path)
    audit_path = output / "audit-input.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["sidecar"]["record_sha256"] = sha256_file(chunk_path)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest = build(output, "dl3dv-context-only-audit", "fixture", "fixture")
    (output / ".scarf-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed contract"):
        validate_context_only_audit_input(output)


def test_context_only_validator_rejects_invalid_source_binding(tmp_path: Path):
    from data.context_only_audit_input import (
        prepare_context_only_audit_input,
        validate_context_only_audit_input,
    )

    source = _source_input(tmp_path)
    output = tmp_path / "context-only"
    prepare_context_only_audit_input(source, output_root=output)
    audit_path = output / "audit-input.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["source_binding"]["canonical_index_sha256"] = "not-a-sha"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="source binding"):
        validate_context_only_audit_input(output)
