import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_manifest_has_real_hashes_and_sizes():
    manifest = json.loads((ROOT / "artifact/manifests/checkpoints.json").read_text())
    assert len(manifest["files"]) == 6
    for record in manifest["files"]:
        assert len(record["sha256"]) == 64
        int(record["sha256"], 16)
        assert record["size"] > 1_000_000
        assert "/resolve/" in record["url"]
        assert record["evaluation_datasets"]
    depthsplat_re10k = next(
        record
        for record in manifest["files"]
        if record["model"] == "depthsplat" and record["dataset"] == "re10k"
    )
    assert depthsplat_re10k["evaluation_datasets"] == ["re10k", "acid"]


def test_runtime_assets_pin_model_backbones_and_metric_downloads():
    manifest = json.loads((ROOT / "artifact/manifests/runtime_assets.json").read_text())
    assert len(manifest["files"]) == 3
    assert {record["name"] for record in manifest["files"]} == {
        "VGG16 weights for LPIPS",
        "Depth Anything V2 Base pretrained weights",
        "DINOv2 torch hub source",
    }
    for record in manifest["files"]:
        assert record["size"] > 1_000_000
        assert len(record["sha256"]) == 64
        int(record["sha256"], 16)
    archive = next(record for record in manifest["files"] if record["kind"] == "tar_gz")
    assert len(archive["commit"]) == 40
    assert len(archive["license_sha256"]) == 64
    depth_anything = next(
        record
        for record in manifest["files"]
        if record["name"] == "Depth Anything V2 Base pretrained weights"
    )
    assert depth_anything["models"] == ["transplat"]
    assert depth_anything["source_revision"] == (
        "a4e71a6c2ce52fe50df0f212066b0d4a87be9b5e"
    )
    assert depth_anything["license"] == "CC-BY-NC-4.0"


def test_dataset_manifest_is_deterministic(tmp_path: Path):
    from data.build_manifest import build

    (tmp_path / "test").mkdir()
    (tmp_path / "test/index.json").write_text("{}\n")
    first = build(tmp_path, "re10k", "source", "revision")
    second = build(tmp_path, "re10k", "source", "revision")
    assert first == second
    assert first["files"]["test/index.json"]["sha256"] == hashlib.sha256(b"{}\n").hexdigest()


def test_download_scripts_do_not_install_packages_implicitly():
    for name in ("download_re10k_acid.sh", "download_dl3dv.sh", "download_checkpoints.sh"):
        text = (ROOT / "data" / name).read_text()
        assert "pip install" not in text


def test_dl3dv_download_is_gated_and_restricted_to_required_trees():
    text = (ROOT / "data/download_dl3dv.sh").read_text()
    assert "get_token" in text
    assert "*/nerfstudio/transforms.json" in text
    assert "*/nerfstudio/images_4/*" in text
    assert "*/nerfstudio/images_8/*" in text


def test_quick_profile_selects_only_mvsplat_re10k_checkpoint(tmp_path, monkeypatch):
    import data.download_assets as assets

    manifest = tmp_path / "checkpoints.json"
    record = {
        "profile": "classic",
        "model": "mvsplat",
        "dataset": "re10k",
        "url": "https://example.invalid/re10k.ckpt",
        "path": "mvsplat/checkpoints/re10k.ckpt",
        "size": 1,
        "sha256": "0" * 64,
    }
    manifest.write_text(json.dumps({"files": [record]}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(assets.subprocess, "run", lambda command, check=False: calls.append(command))

    # Existing correctly sized/hash-matching data avoids a network call.
    target = tmp_path / record["path"]
    target.parent.mkdir(parents=True)
    monkeypatch.setattr(assets, "sha256_file", lambda path: record["sha256"])
    target.write_bytes(b"x")

    assets.download("quick", manifest)

    assert not calls
