import hashlib
import json
from pathlib import Path

import pytest


def make_manifest(root: Path) -> Path:
    from data.build_manifest import build

    record = build(root, "re10k", "test", "test")
    path = root / ".scarf-manifest.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_tree_verifier_rehashes_exact_file_set(tmp_path: Path):
    from data.verify_prepared_dataset import verify_tree_manifest

    data = tmp_path / "test/index.json"
    data.parent.mkdir()
    data.write_text("{}", encoding="utf-8")
    manifest = make_manifest(tmp_path)
    assert verify_tree_manifest(tmp_path, manifest)["file_count"] == 1

    data.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_tree_manifest(tmp_path, manifest)


def test_contract_finalizer_accepts_only_passing_representation_validation(tmp_path: Path):
    from data.finalize_dataset_contract import finalize

    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "artifact/evaluation_protocol.json").read_text())
    manifest = json.loads((root / "artifact/manifests/datasets.json").read_text())
    protocol_path = tmp_path / "protocol.json"
    manifest_path = tmp_path / "datasets.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "kind": "prepared_dataset_validation",
                "status": "PASS",
                "representation": "re10k-native",
                "tree": {
                    "tree_sha256": "a" * 64,
                    "source": manifest["datasets"]["re10k"]["prepared_source"],
                    "revision": "archive-sha256:" + "b" * 64,
                },
                "selection": {
                    "source_index_sha256": protocol["pairs"]["mvsplat/re10k"][
                        "source_index_sha256"
                    ],
                    "sample_selection_sha256": protocol["pairs"]["mvsplat/re10k"][
                        "sample_selection_sha256"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    updated_protocol, updated_manifest = finalize(
        protocol_path,
        manifest_path,
        {"re10k-native": validation},
    )
    assert updated_manifest["datasets"]["re10k"]["expected_tree_sha256"] == "a" * 64
    assert updated_manifest["datasets"]["re10k"]["prepared_source_revision"] == (
        "archive-sha256:" + "b" * 64
    )
    assert all(
        record["dataset_tree_sha256"] == "a" * 64
        for pair, record in updated_protocol["pairs"].items()
        if pair.endswith("/re10k")
    )


def test_contract_finalizer_rejects_dl3dv_source_drift(tmp_path: Path):
    from data.finalize_dataset_contract import finalize

    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "artifact/evaluation_protocol.json").read_text())
    manifest = json.loads((root / "artifact/manifests/datasets.json").read_text())
    protocol_path = tmp_path / "protocol.json"
    manifest_path = tmp_path / "datasets.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = tmp_path / "validation.json"
    pair = protocol["pairs"]["depthsplat/dl3dv"]
    validation.write_text(
        json.dumps(
            {
                "kind": "prepared_dataset_validation",
                "status": "PASS",
                "representation": "depthsplat-native-270x480-v1",
                "tree": {
                    "tree_sha256": "a" * 64,
                    "source": "https://example.invalid/dl3dv",
                    "revision": manifest["datasets"]["dl3dv"]["prepared_source_revision"],
                },
                "selection": {
                    "source_index_sha256": pair["source_index_sha256"],
                    "sample_selection_sha256": pair["sample_selection_sha256"],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="DL3DV native prepared source"):
        finalize(
            protocol_path,
            manifest_path,
            {"depthsplat-native-270x480-v1": validation},
        )


def test_prepared_scene_order_matches_sorted_chunks_and_in_chunk_order(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from data.verify_prepared_dataset import prepared_scene_order

    test_root = tmp_path / "test"
    test_root.mkdir()
    torch.save(
        [{"key": "scene-b"}, {"key": "not-selected"}],
        test_root / "000001.torch",
    )
    torch.save(
        [{"key": "scene-a"}, {"key": "scene-c"}],
        test_root / "000002.torch",
    )
    (test_root / "index.json").write_text(
        json.dumps(
            {
                "scene-a": "000002.torch",
                "scene-b": "000001.torch",
                "scene-c": "000002.torch",
            }
        ),
        encoding="utf-8",
    )

    assert prepared_scene_order(tmp_path, {"scene-a", "scene-b", "scene-c"}) == [
        "scene-b",
        "scene-a",
        "scene-c",
    ]
