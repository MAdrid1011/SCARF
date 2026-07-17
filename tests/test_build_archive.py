import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_source_bundle_includes_only_synthetic_quick_dataset():
    from scripts.build_archive import include_in_source_release

    fixture_files = {
        "datasets/quick-re10k/.scarf-manifest.json",
        "datasets/quick-re10k/.scarf-source.json",
        "datasets/quick-re10k/test/000000.torch",
        "datasets/quick-re10k/test/index.json",
    }

    assert all((ROOT / relative).is_file() for relative in fixture_files)
    assert all(include_in_source_release(Path(relative)) for relative in fixture_files)
    assert not include_in_source_release(Path("datasets/re10k/test/000000.torch"))
    assert not include_in_source_release(Path("outputs/ae/validation.json"))
    assert not include_in_source_release(Path("mvsplat/checkpoints/re10k.ckpt"))


def test_build_archive_cli_resolves_repository_modules():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_archive.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Build and verify a deterministic SCARF AE source archive" in result.stdout
    assert "--reference-results" in result.stdout


def test_single_source_archive_identifies_itself_as_a_release_source(
    tmp_path, monkeypatch
):
    import scripts.build_archive as archive
    import scripts.check_release as release

    source = tmp_path / "README.md"
    source.write_text("portable\n", encoding="utf-8")
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    monkeypatch.setattr(archive, "git", lambda *_args: "")
    monkeypatch.setattr(archive, "source_release_files", lambda: [source])
    monkeypatch.setattr(
        release,
        "build_manifest",
        lambda *_args, **_kwargs: {
            "schema_version": "1.0",
            "git_commit": "a" * 40,
            "submodules": {
                "transplat": "b" * 40,
                "mvsplat": "c" * 40,
                "depthsplat": "d" * 40,
            },
            "zenodo_doi": None,
            "validation": {"pass": True, "failures": []},
        },
    )

    output = tmp_path / "source.tar.gz"
    result = archive.build(output, "SCARF-AE")

    assert result["status"] == "PASS"
    assert result["bundle_kind"] == "source"


def make_archive(path, *, extra=False, unsafe=False):
    payload = b"artifact\n"
    manifest = {
        "files": {"README.md": hashlib.sha256(payload).hexdigest()},
        "git_commit": "a" * 40,
        "zenodo_doi": None,
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, data in (
            ("SCARF-AE/README.md", payload),
            ("SCARF-AE/release-manifest.json", json.dumps(manifest).encode()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if extra:
            data = b"extra"
            info = tarfile.TarInfo("SCARF-AE/extra.txt")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if unsafe:
            data = b"unsafe"
            info = tarfile.TarInfo("../unsafe.txt")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_verify_release_archive_rehashes_every_manifested_file(tmp_path):
    from scripts.build_archive import verify

    path = tmp_path / "artifact.tar.gz"
    make_archive(path)
    result = verify(path)
    assert result["status"] == "PASS"
    assert result["verified_files"] == 1


def test_verify_release_evidence_tar_zst(tmp_path):
    from scripts.build_archive import verify

    tar_gz = tmp_path / "evidence.tar.gz"
    make_archive(tar_gz)
    tar_path = tmp_path / "evidence.tar"
    with tarfile.open(tar_gz, "r:gz") as source, tarfile.open(tar_path, "w") as target:
        for member in source.getmembers():
            stream = source.extractfile(member) if member.isfile() else None
            target.addfile(member, stream)
    archive = tmp_path / "SCARF-AE-evidence-v1.0.0.tar.zst"
    subprocess.run(
        ["zstd", "--no-progress", "-f", str(tar_path), "-o", str(archive)],
        check=True,
        capture_output=True,
    )

    assert verify(archive)["status"] == "PASS"


def test_evidence_file_set_is_curated_and_hash_checked(tmp_path):
    from scripts.build_archive import evidence_release_files

    reference = tmp_path / "external-reference-results"
    evidence = reference / "evidence/result.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    (reference / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "validation_status": "PASS",
                "files": {
                    "evidence/result.json": {
                        "size": evidence.stat().st_size,
                        "sha256": digest,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    contracts = {
        "artifact/evaluation_protocol.json": "{}",
        "artifact/claim_status.json": "{}",
        "artifact/protocol/compiled.json": "{}",
        "artifact/expected_results.json": "{}",
        "artifact/manifests/datasets.json": "{}",
        "artifact/manifests/checkpoints.json": "{}",
        "artifact/manifests/runtime_assets.json": "{}",
        "THIRD_PARTY.md": "third party",
    }
    for relative, content in contracts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    files = evidence_release_files(tmp_path, reference_results=reference)
    assert "evidence/result.json" in files
    assert "contracts/evaluation_protocol.json" in files
    assert "contracts/checkpoints.json" in files
    assert all("datasets/" not in relative for relative in files)


def test_verify_release_archive_rejects_unmanifested_and_unsafe_files(tmp_path):
    from scripts.build_archive import verify

    extra = tmp_path / "extra.tar.gz"
    make_archive(extra, extra=True)
    with pytest.raises(ValueError, match="unmanifested"):
        verify(extra)

    unsafe = tmp_path / "unsafe.tar.gz"
    make_archive(unsafe, unsafe=True)
    with pytest.raises(ValueError, match="root directory|unsafe archive"):
        verify(unsafe)


def test_source_and_evidence_bundles_require_identical_release_identity():
    from scripts.build_archive import validate_bundle_binding

    source = {
        "git_commit": "a" * 40,
        "submodules": {"mvsplat": "b" * 40},
        "zenodo_doi": "10.5281/zenodo.123",
    }
    evidence = dict(source)
    validate_bundle_binding(source, evidence)

    evidence = {**source, "git_commit": "c" * 40}
    with pytest.raises(ValueError, match="git_commit"):
        validate_bundle_binding(source, evidence)


def test_sha256sums_covers_source_and_evidence_bundles(tmp_path):
    from scripts.build_archive import write_sha256sums

    source = tmp_path / "SCARF-AE-source-v1.0.0.tar.gz"
    evidence = tmp_path / "SCARF-AE-evidence-v1.0.0.tar.zst"
    source.write_bytes(b"source")
    evidence.write_bytes(b"evidence")
    output = tmp_path / "SHA256SUMS"
    write_sha256sums([source, evidence], output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].endswith(f"  {source.name}")
    assert lines[1].endswith(f"  {evidence.name}")
