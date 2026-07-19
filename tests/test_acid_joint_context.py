import importlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class _ArchiveTorch:
    long = "long"

    @staticmethod
    def load(path, map_location=None):
        return pickle.loads(Path(path).read_bytes())

    @staticmethod
    def tensor(value, dtype=None):
        return _Tensor("tensor")


class _Tensor:
    def __init__(self, name):
        self.name = name

    def unsqueeze(self, dimension):
        return (self.name, "unsqueeze", dimension)


@pytest.fixture
def context_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", _ArchiveTorch)
    sys.modules.pop("integration.acid_joint_context", None)
    module = importlib.import_module("integration.acid_joint_context")
    monkeypatch.setattr(module, "torch", _ArchiveTorch)
    monkeypatch.setattr(
        module,
        "_decode_context_images",
        lambda images: _Tensor("images"),
    )
    monkeypatch.setattr(
        module,
        "_context_camera_geometry",
        lambda cameras: (_Tensor("extrinsics"), _Tensor("intrinsics")),
    )
    return module


def _sidecar(tmp_path: Path, *, record: dict | None = None) -> tuple[Path, Path]:
    root = tmp_path / "materialization"
    sidecar = root / "inputs" / "calibration_train" / "test"
    sidecar.mkdir(parents=True)
    scene = "scene-a"
    payload = record or {
        "schema_version": "1.0",
        "kind": "scarf_acid_joint_context_record_v1",
        "key": scene,
        "source_view_count": 5,
        "context_indices": [0, 4],
        "context_cameras": [[0.0] * 18, [0.0] * 18],
        "context_images": [b"context-0", b"context-4"],
    }
    chunk = sidecar / "000000.torch"
    chunk.write_bytes(pickle.dumps([payload]))
    (sidecar / "index.json").write_text(
        json.dumps({scene: "000000.torch"}), encoding="utf-8"
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "plan_sha256": "a" * 64,
                "partition": {
                    "calibration_train": {"scenes": [scene]},
                    "calibration_holdout": {"scenes": ["scene-b"]},
                },
            }
        ),
        encoding="utf-8",
    )
    return root, plan


def _validated_identity():
    return {
        "sidecars": {
            "calibration_train": {"tree_sha256": "b" * 64},
            "calibration_holdout": {"tree_sha256": "c" * 64},
        }
    }


def test_loader_selects_the_planned_scene_and_returns_only_context(
    tmp_path: Path, context_module, monkeypatch: pytest.MonkeyPatch
):
    root, plan = _sidecar(tmp_path)
    calls = []
    monkeypatch.setattr(
        context_module,
        "validate_materialization",
        lambda materialization_root, *, plan: calls.append((materialization_root, plan))
        or _validated_identity(),
    )

    loaded = context_module.load_acid_joint_context(
        materialization_root=root,
        split="calibration_train",
        sample_index=0,
        plan_path=plan,
    )

    assert len(calls) == 1
    assert set(loaded.context) == {"image", "extrinsics", "intrinsics", "index"}
    assert "target" not in loaded.context
    assert loaded.context["image"] == ("images", "unsqueeze", 0)
    assert loaded.context["extrinsics"] == ("extrinsics", "unsqueeze", 0)
    assert loaded.context["intrinsics"] == ("intrinsics", "unsqueeze", 0)
    assert loaded.context["index"] == ("tensor", "unsqueeze", 0)
    assert loaded.identity["scene"] == "scene-a"
    assert loaded.identity["target_rgb_accessed"] is False
    assert loaded.identity["target_camera_metadata_accessed"] is False
    assert loaded.identity["teacher_artifact_accessed"] is False


def test_loader_rejects_target_fields_even_if_materialization_was_claimed_valid(
    tmp_path: Path, context_module, monkeypatch: pytest.MonkeyPatch
):
    record = {
        "schema_version": "1.0",
        "kind": "scarf_acid_joint_context_record_v1",
        "key": "scene-a",
        "source_view_count": 5,
        "context_indices": [0, 4],
        "context_cameras": [[0.0] * 18, [0.0] * 18],
        "context_images": [b"context-0", b"context-4"],
        "target_indices": [1, 2, 3],
    }
    root, plan = _sidecar(tmp_path, record=record)
    monkeypatch.setattr(
        context_module, "validate_materialization", lambda *_args, **_kwargs: _validated_identity()
    )

    with pytest.raises(ValueError, match="forbidden fields"):
        context_module.load_acid_joint_context(
            materialization_root=root,
            split="calibration_train",
            sample_index=0,
            plan_path=plan,
        )


def test_loader_rejects_out_of_range_sample_before_opening_a_record(
    tmp_path: Path, context_module, monkeypatch: pytest.MonkeyPatch
):
    root, plan = _sidecar(tmp_path)
    monkeypatch.setattr(
        context_module, "validate_materialization", lambda *_args, **_kwargs: _validated_identity()
    )

    with pytest.raises(IndexError, match="out of range"):
        context_module.load_acid_joint_context(
            materialization_root=root,
            split="calibration_train",
            sample_index=1,
            plan_path=plan,
        )


def test_raw_context_loader_has_no_model_loader_or_upstream_src_dependency(context_module):
    source = Path(context_module.__file__).read_text(encoding="utf-8")

    assert "model_loader" not in source
    assert "from src" not in source
    assert "import src" not in source
