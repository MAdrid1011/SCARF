from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _source():
    from integration.acid_joint_context import AcidJointContext

    return AcidJointContext(
        context={
            "image": torch.rand(1, 2, 3, 6, 8),
            "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
            "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
            "index": torch.tensor([[0, 4]], dtype=torch.long),
        },
        identity={"scene": "test", "target_rgb_accessed": False},
    )


def test_model_context_uses_only_context_view_shims(monkeypatch: pytest.MonkeyPatch):
    import integration.acid_joint_model_context as module

    calls = []

    class Crop:
        @staticmethod
        def apply_crop_shim_to_views(context, shape):
            calls.append(("crop", tuple(shape), set(context)))
            return context

    class Patch:
        @staticmethod
        def apply_patch_shim_to_views(context, patch_size):
            calls.append(("patch", patch_size, set(context)))
            return context

    def fake_import(name):
        if name.endswith("crop_shim"):
            return Crop
        if name.endswith("patch_shim"):
            return Patch
        raise AssertionError(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    context, audit = module.prepare_acid_joint_model_context(
        _source(),
        dataset_cfg=SimpleNamespace(
            image_shape=[6, 8],
            near=0.1,
            far=10.0,
            make_baseline_1=False,
            baseline_scale_bounds=True,
        ),
        encoder_cfg=SimpleNamespace(shim_patch_size=2, downscale_factor=4),
        device=torch.device("cpu"),
    )

    assert set(context) == {"image", "extrinsics", "intrinsics", "index", "near", "far"}
    assert calls == [
        ("crop", (6, 8), {"image", "extrinsics", "intrinsics", "index", "near", "far"}),
        ("patch", 8, {"image", "extrinsics", "intrinsics", "index", "near", "far"}),
    ]
    assert audit["target_mapping_present"] is False
    assert audit["target_rgb_accessed"] is False
    assert audit["target_camera_metadata_accessed"] is False
    assert audit["target_index_accessed"] is False


def test_model_context_refuses_invalid_baseline_before_shim_import(monkeypatch: pytest.MonkeyPatch):
    import integration.acid_joint_model_context as module

    source = _source()
    source.context["extrinsics"][:, 1, :3, 3] = source.context["extrinsics"][:, 0, :3, 3]
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(AssertionError("shim import should not occur")),
    )

    with pytest.raises(ValueError, match="baseline"):
        module.prepare_acid_joint_model_context(
            source,
            dataset_cfg=SimpleNamespace(
                image_shape=[6, 8],
                near=0.1,
                far=10.0,
                make_baseline_1=True,
                baseline_scale_bounds=True,
            ),
            encoder_cfg=SimpleNamespace(shim_patch_size=1, downscale_factor=1),
            device=torch.device("cpu"),
        )
