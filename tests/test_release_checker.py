import json
from pathlib import Path


def test_local_path_patterns_reject_author_home(tmp_path: Path):
    from scripts.check_release import check_local_paths

    bad = tmp_path / "bad.txt"
    bad.write_text("python /" + "home/author/project/demo.py\n")
    assert check_local_paths([bad])


def test_repository_has_no_tracked_author_local_paths():
    from scripts.check_release import archive_files, check_local_paths

    assert check_local_paths(archive_files()) == []


def test_dataset_sources_require_verified_terms_and_tree_hashes(tmp_path: Path):
    from scripts.check_release import check_dataset_sources

    source = Path(__file__).resolve().parents[1] / "artifact/manifests/datasets.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    record["datasets"]["re10k"]["expected_tree_sha256"] = None
    record["datasets"]["acid"]["expected_tree_sha256"] = None
    path = tmp_path / "datasets.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    failures = check_dataset_sources(path)
    assert "re10k has no verified dataset tree SHA256" in failures
    assert "acid has no verified dataset tree SHA256" in failures
    assert not any("dl3dv" in failure for failure in failures)


def test_repository_dataset_contract_is_release_ready():
    from scripts.check_release import check_dataset_sources

    assert check_dataset_sources() == []


def test_dataset_sources_accept_two_verified_dl3dv_representations(tmp_path: Path):
    from scripts.check_release import check_dataset_sources

    verified_license = {
        "status": "verified",
        "identifier": "test-only",
        "terms_url": "https://example.test/terms",
        "redistribution_in_zenodo": False,
    }
    record = {
        "datasets": {
                "re10k": {
                    "expected_tree_sha256": "1" * 64,
                    "prepared_source_revision": "archive-sha256:" + "a" * 64,
                    "license": verified_license,
                },
                "acid": {
                    "expected_tree_sha256": "2" * 64,
                    "prepared_source_revision": "archive-sha256:" + "b" * 64,
                    "license": verified_license,
                },
            "dl3dv": {
                "representations": {
                    "native": {
                        "path": "datasets/dl3dv/native",
                        "schema": "depthsplat-native-270x480-v1",
                        "expected_tree_sha256": "3" * 64,
                    },
                    "re10k": {
                        "path": "datasets/dl3dv/re10k",
                        "schema": "re10k-compatible-360x640-v1",
                        "expected_tree_sha256": "4" * 64,
                    },
                },
                "license": verified_license,
            },
        }
    }
    path = tmp_path / "datasets.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert check_dataset_sources(path) == []


def test_dataset_sources_accept_verified_terms_without_redistribution(tmp_path: Path):
    from scripts.check_release import check_dataset_sources

    verified_license = {
        "status": "verified_no_redistribution",
        "identifier": "test-only",
        "terms_url": "https://example.test/terms",
        "redistribution_in_zenodo": False,
    }
    record = {
        "datasets": {
            "re10k": {
                "expected_tree_sha256": "1" * 64,
                "prepared_source_revision": "archive-sha256:" + "a" * 64,
                "license": verified_license,
            },
            "acid": {
                "expected_tree_sha256": "2" * 64,
                "prepared_source_revision": "archive-sha256:" + "b" * 64,
                "license": verified_license,
            },
            "dl3dv": {
                "representations": {
                    "native": {
                        "path": "datasets/dl3dv/native",
                        "schema": "depthsplat-native-270x480-v1",
                        "expected_tree_sha256": "3" * 64,
                    },
                    "re10k": {
                        "path": "datasets/dl3dv/re10k",
                        "schema": "re10k-compatible-360x640-v1",
                        "expected_tree_sha256": "4" * 64,
                    },
                },
                "license": verified_license,
            },
        }
    }
    path = tmp_path / "datasets.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert check_dataset_sources(path) == []


def test_deepscale_vendor_assets_are_hash_verified():
    from scripts.check_release import check_deepscale_assets

    assert check_deepscale_assets() == []


def test_runtime_asset_manifest_is_release_checked():
    from scripts.check_release import check_runtime_asset_manifest

    assert check_runtime_asset_manifest() == []


def test_runtime_asset_manifest_rejects_unknown_model_scope(tmp_path: Path):
    from scripts.check_release import check_runtime_asset_manifest

    source = Path(__file__).resolve().parents[1] / "artifact/manifests/runtime_assets.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    record["files"][0]["models"] = ["unknown-model"]
    path = tmp_path / "runtime_assets.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    assert "invalid runtime asset models: VGG16 weights for LPIPS" in (
        check_runtime_asset_manifest(path)
    )


def test_checkpoint_manifest_is_release_checked():
    from scripts.check_release import check_checkpoint_manifest

    assert check_checkpoint_manifest() == []


def test_checkpoint_manifest_rejects_unpinned_source_and_wrong_mapping(tmp_path: Path):
    from scripts.check_release import check_checkpoint_manifest

    source = Path(__file__).resolve().parents[1] / "artifact/manifests/checkpoints.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    record["files"][0]["url"] = "https://huggingface.co/example/latest.ckpt"
    record["files"][4]["evaluation_datasets"] = ["re10k"]
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    failures = check_checkpoint_manifest(path)
    assert "checkpoint source is not commit-pinned HTTPS: transplat/re10k" in failures
    assert "checkpoint evaluation mapping mismatch: depthsplat/re10k" in failures


def test_orin_contract_is_release_checked():
    from scripts.check_release import check_orin_contract

    assert check_orin_contract() == []


def test_protocol_and_staged_reference_evidence_are_release_ready():
    from scripts.check_release import check_evaluation_protocol, check_reference_results

    assert check_evaluation_protocol() == []
    assert check_reference_results() == []


def test_claim_status_is_complete_and_release_checked():
    from scripts.check_release import check_claim_status

    assert check_claim_status() == []
