from pathlib import Path

import pytest


def test_claimed_matrix_has_nine_unique_pairs():
    from scripts.ae_config import CLAIMED_MATRIX

    assert len(CLAIMED_MATRIX) == 9
    assert len(set(CLAIMED_MATRIX)) == 9
    assert {model for model, _ in CLAIMED_MATRIX} == {
        "transplat",
        "mvsplat",
        "depthsplat",
    }
    assert {dataset for _, dataset in CLAIMED_MATRIX} == {
        "re10k",
        "acid",
        "dl3dv",
    }


@pytest.mark.parametrize(
    "model,dataset,experiment,checkpoint,profile,dataset_relative,representation",
    [
        ("transplat", "re10k", "re10k", "re10k.ckpt", "classic", "re10k", "re10k-native"),
        ("transplat", "acid", "acid", "acid.ckpt", "classic", "acid", "acid-native"),
        ("transplat", "dl3dv", "re10k", "re10k.ckpt", "classic", "dl3dv/re10k", "re10k-compatible-360x640-v1"),
        ("mvsplat", "acid", "acid", "acid.ckpt", "classic", "acid", "acid-native"),
        ("mvsplat", "dl3dv", "re10k", "re10k.ckpt", "classic", "dl3dv/re10k", "re10k-compatible-360x640-v1"),
        ("depthsplat", "re10k", "re10k", "re10k.ckpt", "depthsplat", "re10k", "re10k-native"),
        ("depthsplat", "acid", "re10k", "re10k.ckpt", "depthsplat", "acid", "acid-native"),
        ("depthsplat", "dl3dv", "dl3dv", "dl3dv.ckpt", "depthsplat", "dl3dv/native", "depthsplat-native-270x480-v1"),
    ],
)
def test_experiment_mapping(
    model, dataset, experiment, checkpoint, profile, dataset_relative, representation
):
    from scripts.ae_config import resolve_experiment

    resolved = resolve_experiment(model, dataset, Path("/repo"))
    assert resolved.experiment == experiment
    assert resolved.checkpoint.name == checkpoint
    assert resolved.environment_profile == profile
    assert resolved.dataset_root == Path("/repo/datasets") / dataset_relative
    assert resolved.dataset_representation == representation


def test_invalid_model_or_dataset_is_rejected():
    from scripts.ae_config import resolve_experiment

    with pytest.raises(ValueError, match="unsupported model"):
        resolve_experiment("unknown", "re10k", Path("/repo"))
    with pytest.raises(ValueError, match="unsupported dataset"):
        resolve_experiment("mvsplat", "unknown", Path("/repo"))


def test_depthsplat_checkpoint_variants_pin_upstream_hydra_overrides():
    from scripts.ae_config import resolve_experiment

    assert resolve_experiment("depthsplat", "re10k", Path("/repo")).hydra_overrides == ()
    assert resolve_experiment("depthsplat", "acid", Path("/repo")).hydra_overrides == ()
    assert resolve_experiment("depthsplat", "dl3dv", Path("/repo")).hydra_overrides == (
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=4",
        "model.encoder.lowest_feature_resolution=8",
        "model.encoder.monodepth_vit_type=vitb",
    )
    assert resolve_experiment("mvsplat", "re10k", Path("/repo")).hydra_overrides == ()


def test_prepared_dl3dv_representation_must_match_experiment(tmp_path):
    import json

    from data.build_manifest import build
    from scripts.ae_config import resolve_experiment, validate_prepared_dataset

    root = tmp_path / "datasets/dl3dv/re10k"
    (root / "test").mkdir(parents=True)
    conversion = {
        "dataset": "dl3dv",
        "representation": "re10k-compatible-360x640-v1",
    }
    (root / "conversion.json").write_text(
        json.dumps(conversion) + "\n", encoding="utf-8"
    )
    (root / "test/index.json").write_text("{}\n", encoding="utf-8")
    manifest = build(root, "dl3dv-re10k", "fixture", "v1")
    manifest_path = root / ".scarf-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config = resolve_experiment("mvsplat", "dl3dv", tmp_path)
    identity = validate_prepared_dataset(config, root, manifest_path)
    assert identity["representation"] == "re10k-compatible-360x640-v1"
    assert identity["tree_sha256"] == manifest["tree_sha256"]

    config = resolve_experiment("depthsplat", "dl3dv", tmp_path)
    with pytest.raises(ValueError, match="identifies"):
        validate_prepared_dataset(config, root, manifest_path)


def test_synthetic_quick_dataset_is_functional_only(tmp_path):
    import json

    from data.build_manifest import build
    from scripts.ae_config import resolve_experiment, validate_prepared_dataset

    root = tmp_path / "quick"
    root.mkdir()
    (root / "fixture.txt").write_text("synthetic\n", encoding="utf-8")
    manifest = build(root, "re10k", "synthetic", "v1")
    manifest.update(
        {
            "functional_fixture": True,
            "representation": "re10k-synthetic-functional-v1",
            "paper_result_eligible": False,
        }
    )
    manifest_path = root / ".scarf-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = resolve_experiment("mvsplat", "re10k", tmp_path)

    with pytest.raises(ValueError, match="cannot be used for a claim"):
        validate_prepared_dataset(config, root, manifest_path)
    identity = validate_prepared_dataset(
        config, root, manifest_path, allow_functional_fixture=True
    )
    assert identity["functional_fixture"] is True
    assert identity["representation"] == "re10k-synthetic-functional-v1"


def test_claim_dataset_tree_requires_protocol_and_manifest_agreement(tmp_path):
    import json

    from scripts.ae_config import validate_claim_dataset_tree

    tree = "a" * 64
    artifact = tmp_path / "artifact"
    (artifact / "manifests").mkdir(parents=True)
    (artifact / "evaluation_protocol.json").write_text(
        json.dumps(
            {"pairs": {"mvsplat/re10k": {"dataset_tree_sha256": tree}}}
        ),
        encoding="utf-8",
    )
    (artifact / "manifests/datasets.json").write_text(
        json.dumps(
            {"datasets": {"re10k": {"expected_tree_sha256": tree}}}
        ),
        encoding="utf-8",
    )

    validate_claim_dataset_tree("mvsplat", "re10k", tree, tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        validate_claim_dataset_tree("mvsplat", "re10k", "b" * 64, tmp_path)


def test_depthsplat_hub_load_is_rewritten_to_pinned_commit(monkeypatch):
    import importlib
    import sys
    from types import SimpleNamespace

    calls = []

    def fake_hub_load(repo, model, *args, **kwargs):
        calls.append((repo, model, args, kwargs))
        return "backbone"

    fake_torch = SimpleNamespace(
        device=type("device", (), {}),
        Tensor=type("Tensor", (), {}),
        hub=SimpleNamespace(load=fake_hub_load),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "integration", raising=False)
    monkeypatch.delitem(sys.modules, "integration.model_loader", raising=False)
    loader = importlib.import_module("integration.model_loader")
    try:
        def fake_get_encoder(config):
            backbone = loader.torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14"
            )
            return backbone, None

        encoder, visualizer = loader.get_depthsplat_encoder(fake_get_encoder, object())

        assert encoder == "backbone"
        assert visualizer is None
        assert calls == [
            (
                str(loader.DINOV2_SOURCE),
                "dinov2_vits14",
                (),
                {"source": "local", "pretrained": False},
            )
        ]
        assert loader.torch.hub.load is fake_hub_load
    finally:
        sys.modules.pop("integration.model_loader", None)
        sys.modules.pop("integration", None)
