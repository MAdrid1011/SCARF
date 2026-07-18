import json
import hashlib
from pathlib import Path

import pytest


def evidence_binding(*, calibration_status: str = "preregistered") -> dict:
    from scripts.stage_reference_results import canonical_sha256

    calibration = {
        "status": calibration_status,
        "manifest_sha256": "4" * 64,
        "candidate_records_sha256": "5" * 64
        if calibration_status == "calibrated"
        else None,
        "evaluation_disjoint": calibration_status == "calibrated",
        "expected_results_accessed": False,
        "global_configuration": True,
    }
    return {
        "source": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_identity": "git",
            "source_tree_sha256": "b" * 64,
            "submodules": {
                "transplat": "c" * 40,
                "mvsplat": "d" * 40,
                "depthsplat": "e" * 40,
            },
        },
        "mechanism": {
            "mechanism_config_sha256": "f" * 64,
            "calibration_status": calibration_status,
            "calibration_provenance_sha256": canonical_sha256(calibration),
            "calibration_provenance": calibration,
        },
    }


def provenance_result(binding: dict) -> dict:
    return {
        "provenance": {
            **binding["source"],
            "mechanism_config_sha256": binding["mechanism"]["mechanism_config_sha256"],
            "calibration_provenance": json.loads(
                json.dumps(binding["mechanism"]["calibration_provenance"])
            ),
        }
    }


def write_reference_bundle(tmp_path: Path, binding: dict) -> tuple[Path, Path]:
    from scripts.stage_reference_results import (
        required_categories,
        validate_generated_records,
    )

    reference = tmp_path / "reference"
    result = reference / "evidence/quick/mvsplat_re10k/results.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(provenance_result(binding)), encoding="utf-8")
    generated_records, failures = validate_generated_records(
        {result: Path("quick/mvsplat_re10k/results.json")},
        expected_binding=binding,
    )
    assert failures == []
    manifest = {
        "schema_version": "2.0",
        "status": "complete",
        "validation_status": "PASS",
        "validation_require_key_results": True,
        "categories": sorted(required_categories()),
        "files": {
            "evidence/quick/mvsplat_re10k/results.json": {
                "size": result.stat().st_size,
                "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
            }
        },
        "provenance": {**binding, "generated_records": generated_records},
    }
    (reference / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return reference, result


def add_deepscale_reference_evidence(reference: Path, binding: dict) -> Path:
    from hardware.scaling.deepscale import scale_record, sha256_file
    from scripts.stage_reference_results import validate_generated_records

    raw = {
        "schema_version": "1.0",
        "evidence_type": "asap7_predictive_postroute",
        "physical_valid": True,
        "metrics": {"logic_area_mm2": 1.0},
        "provenance": provenance_result(binding)["provenance"],
    }
    raw_path = reference / "evidence/physical/asap7/ppa.json"
    estimate_path = reference / "evidence/physical/asap7/ppa_28nm_estimated.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    estimate = scale_record(
        raw,
        7,
        28,
        raw_input_sha256=sha256_file(raw_path),
    )
    estimate_path.write_text(json.dumps(estimate), encoding="utf-8")

    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (raw_path, estimate_path):
        relative = path.relative_to(reference).as_posix()
        manifest["files"][relative] = {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    generated_records, failures = validate_generated_records(
        {
            reference / relative: Path(relative).relative_to("evidence")
            for relative in manifest["files"]
            if (reference / relative).is_file()
        },
        expected_binding=binding,
    )
    assert failures == []
    manifest["provenance"]["generated_records"] = generated_records
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return estimate_path


def test_local_path_patterns_reject_author_home(tmp_path: Path):
    from scripts.check_release import check_local_paths

    bad = tmp_path / "bad.txt"
    bad.write_text("python /" + "home/author/project/demo.py\n")
    assert check_local_paths([bad])


def test_repository_has_no_tracked_author_local_paths():
    from scripts.check_release import archive_files, check_local_paths

    assert check_local_paths(archive_files()) == []


def test_archive_file_scan_falls_back_to_source_release_manifest(
    tmp_path: Path, monkeypatch
):
    import scripts.check_release as release

    readme = tmp_path / "README.md"
    readme.write_text("portable\n", encoding="utf-8")
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(
            {
                "bundle_kind": "source",
                "validation": {"pass": True},
                "files": {"README.md": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "ROOT", tmp_path)

    assert release.archive_files() == [readme]


def test_archive_file_scan_uses_the_source_allowlist(tmp_path: Path, monkeypatch):
    import scripts.check_release as release

    readme = tmp_path / "README.md"
    plan = tmp_path / "PLAN.md"
    readme.write_text("portable\n", encoding="utf-8")
    plan.write_text("internal\n", encoding="utf-8")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "git", lambda *_args: "README.md\0PLAN.md")

    assert release.archive_files() == [readme]


@pytest.mark.parametrize(
    ("files", "message"),
    (
        ({"PLAN.md": "0" * 64}, "invalid file"),
        (
            {"README.md": "0" * 64, "./README.md": "0" * 64},
            "duplicate normalized file",
        ),
    ),
)
def test_archive_file_scan_rejects_unreviewed_or_duplicate_manifest_paths(
    tmp_path: Path, monkeypatch, files, message
):
    import scripts.check_release as release

    (tmp_path / "README.md").write_text("portable\n", encoding="utf-8")
    (tmp_path / "PLAN.md").write_text("internal\n", encoding="utf-8")
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(
            {
                "bundle_kind": "source",
                "validation": {"pass": True},
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "ROOT", tmp_path)

    with pytest.raises(RuntimeError, match=message):
        release.archive_files()


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
    if not (Path(__file__).resolve().parents[1] / "artifact/reference_results/evidence").is_dir():
        pytest.skip("reference evidence is distributed in the separate evidence bundle")
    assert check_reference_results() == [
        "reference evidence was not validated with --require-key-results",
        "reference evidence manifest has no provenance binding (legacy Functional-era evidence)",
    ]


def test_reference_evidence_accepts_preregistered_nonclaim_provenance(tmp_path):
    from scripts.check_release import check_reference_results

    binding = evidence_binding(calibration_status="preregistered")
    reference, _ = write_reference_bundle(tmp_path, binding)

    assert (
        check_reference_results(
            reference_results=reference, expected_binding=binding
        )
        == []
    )


def test_reference_evidence_rejects_stale_source_or_calibration_records(tmp_path):
    from scripts.check_release import check_reference_results

    binding = evidence_binding(calibration_status="calibrated")
    reference, result_path = write_reference_bundle(tmp_path, binding)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["provenance"]["source_tree_sha256"] = "0" * 64
    result["provenance"]["calibration_provenance"]["status"] = "preregistered"
    result["provenance"]["calibration_provenance"]["manifest_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["evidence/quick/mvsplat_re10k/results.json"] = {
        "size": result_path.stat().st_size,
        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    failures = check_reference_results(
        reference_results=reference, expected_binding=binding
    )

    assert (
        "reference generated evidence source provenance mismatch source_tree_sha256: "
        "evidence/quick/mvsplat_re10k/results.json" in failures
    )
    assert (
        "reference generated evidence calibration status mismatch: "
        "evidence/quick/mvsplat_re10k/results.json" in failures
    )
    assert (
        "reference generated evidence calibration provenance digest mismatch: "
        "evidence/quick/mvsplat_re10k/results.json" in failures
    )


def test_reference_evidence_rejects_a_deepscale_raw_input_hash_mismatch(tmp_path):
    from scripts.check_release import check_reference_results

    binding = evidence_binding()
    reference, _ = write_reference_bundle(tmp_path, binding)
    estimate_path = add_deepscale_reference_evidence(reference, binding)
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    estimate["raw_asap7_input"]["sha256"] = "0" * 64
    estimate_path.write_text(json.dumps(estimate), encoding="utf-8")
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = estimate_path.relative_to(reference).as_posix()
    manifest["files"][relative] = {
        "size": estimate_path.stat().st_size,
        "sha256": hashlib.sha256(estimate_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    failures = check_reference_results(
        reference_results=reference, expected_binding=binding
    )

    assert (
        "reference DeepScale estimate input binding is invalid: "
        "evidence/physical/asap7/ppa_28nm_estimated.json "
        "(DeepScale result raw ASAP7 input binding does not match)" in failures
    )


def test_reference_evidence_rejects_legacy_functional_manifest_precisely(tmp_path):
    from scripts.check_release import check_reference_results

    binding = evidence_binding()
    reference, _ = write_reference_bundle(tmp_path, binding)
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.0"
    manifest["validation_require_key_results"] = False
    manifest.pop("provenance")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert check_reference_results(
        reference_results=reference, expected_binding=binding
    ) == [
        "reference evidence was not validated with --require-key-results",
        "reference evidence manifest has no provenance binding (legacy Functional-era evidence)",
    ]


def test_claim_status_is_complete_and_release_checked():
    from scripts.check_release import check_claim_status

    assert check_claim_status() == []
