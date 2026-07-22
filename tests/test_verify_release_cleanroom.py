import hashlib
import io
import json
import shutil
import subprocess
import tarfile

import pytest


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _add_regular(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(payload))


def _write_bundle(path, prefix, members, manifest, *, symlink_name=None):
    with tarfile.open(path, "w:gz") as archive:
        for relative, payload in members.items():
            _add_regular(archive, f"{prefix}/{relative}", payload)
        if symlink_name is not None:
            member = tarfile.TarInfo(f"{prefix}/{symlink_name}")
            member.type = tarfile.SYMTYPE
            member.linkname = "run_ae.sh"
            archive.addfile(member)
        _add_regular(
            archive,
            f"{prefix}/release-manifest.json",
            (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8"),
        )


def _write_sha256sums(path, source, evidence, *, source_digest=None):
    source_digest = source_digest or _sha256(source.read_bytes())
    path.write_text(
        f"{source_digest}  {source.name}\n"
        f"{_sha256(evidence.read_bytes())}  {evidence.name}\n",
        encoding="utf-8",
    )


def _identity(commit="a" * 40):
    return {
        "schema_version": "1.0",
        "git_commit": commit,
        "submodules": {
            "transplat": "b" * 40,
            "mvsplat": "c" * 40,
            "depthsplat": "d" * 40,
        },
        "zenodo_doi": None,
        "validation": {"pass": True, "failures": []},
    }


def _make_bundles(tmp_path, *, evidence_commit=None, bad_reference_digest=False):
    source = tmp_path / "SCARF-AE-source-v1.0.0.tar.gz"
    evidence = tmp_path / "SCARF-AE-evidence-v1.0.0.tar.gz"
    checksums = tmp_path / "SHA256SUMS"
    source_prefix = "SCARF-AE-source-v1.0.0"
    evidence_prefix = "SCARF-AE-evidence-v1.0.0"
    source_members = {
        "scripts/run_ae.sh": b"#!/usr/bin/env bash\n",
        "scripts/run_ae.py": b"# runner\n",
        "scripts/run_rtl.sh": b"#!/usr/bin/env bash\n",
        "scripts/validate_ae.py": b"# validator\n",
        "hardware/dram/run.sh": b"#!/usr/bin/env bash\n",
    }
    source_manifest = {
        **_identity(),
        "bundle_kind": "source",
        "files": {
            relative: _sha256(payload) for relative, payload in source_members.items()
        },
        "archive": {"prefix": source_prefix, "format": "tar.gz"},
    }
    _write_bundle(source, source_prefix, source_members, source_manifest)

    evidence_payload = b'{"seed": true}\n'
    declared_digest = _sha256(evidence_payload)
    if bad_reference_digest:
        declared_digest = "0" * 64
    reference_manifest = {
        "status": "complete",
        "validation_status": "PASS",
        "validation_require_key_results": True,
        "categories": ["quick"],
        "files": {
            "evidence/seed.json": {
                "size": len(evidence_payload),
                "sha256": declared_digest,
            }
        },
    }
    evidence_members = {
        "reference-manifest.json": (
            json.dumps(reference_manifest, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "evidence/seed.json": evidence_payload,
    }
    evidence_manifest = {
        **_identity(evidence_commit or "a" * 40),
        "bundle_kind": "evidence",
        "files": {
            relative: _sha256(payload) for relative, payload in evidence_members.items()
        },
        "archive": {"prefix": evidence_prefix, "format": "tar.gz"},
    }
    _write_bundle(evidence, evidence_prefix, evidence_members, evidence_manifest)
    _write_sha256sums(checksums, source, evidence)
    return source, evidence, checksums


def test_cleanroom_dry_run_verifies_and_plans_without_running_commands(tmp_path):
    from scripts.verify_release_cleanroom import run_cleanroom

    source, evidence, checksums = _make_bundles(tmp_path)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("dry-run must not invoke a release command")

    result = run_cleanroom(
        source,
        evidence,
        checksums,
        execute=False,
        command_runner=must_not_run,
    )

    assert result["status"] == "NOT_RUN"
    assert result["doi_download"]["status"] == "NOT_FETCHED"
    assert result["evidence"]["payload"]["files"] == 1
    assert result["evidence_overlay"]["copied_files"] == 1
    assert [record["name"] for record in result["commands"]] == [
        "quick",
        "rtl",
        "dram",
        "validate",
    ]
    assert all(record["status"] == "NOT_RUN" for record in result["commands"])
    assert result["commands"][-1]["argv"][-1] == "--require-key-results"
    assert all("/tmp/" not in " ".join(record["argv"]) for record in result["commands"])


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is not installed")
def test_cleanroom_dry_run_accepts_the_zstd_evidence_bundle_format(tmp_path):
    from scripts.verify_release_cleanroom import run_cleanroom

    source, evidence, checksums = _make_bundles(tmp_path)
    raw_tar = tmp_path / "evidence.tar"
    with tarfile.open(evidence, "r:gz") as input_archive, tarfile.open(
        raw_tar, "w"
    ) as output_archive:
        for member in input_archive.getmembers():
            stream = input_archive.extractfile(member)
            output_archive.addfile(member, stream)
            if stream is not None:
                stream.close()
    zstd_evidence = tmp_path / "SCARF-AE-evidence-v1.0.0.tar.zst"
    subprocess.run(
        ["zstd", "--no-progress", "-f", str(raw_tar), "-o", str(zstd_evidence)],
        check=True,
        capture_output=True,
    )
    _write_sha256sums(checksums, source, zstd_evidence)

    result = run_cleanroom(source, zstd_evidence, checksums, execute=False)

    assert result["status"] == "NOT_RUN"
    assert result["archives"]["evidence"]["name"] == zstd_evidence.name


def test_cleanroom_rejects_checksum_mismatch_before_extraction_or_commands(tmp_path):
    from scripts.verify_release_cleanroom import (
        CleanroomVerificationError,
        run_cleanroom,
    )

    source, evidence, checksums = _make_bundles(tmp_path)
    _write_sha256sums(checksums, source, evidence, source_digest="0" * 64)

    with pytest.raises(CleanroomVerificationError, match="SHA256 mismatch") as raised:
        run_cleanroom(source, evidence, checksums, execute=False)

    assert raised.value.report["status"] == "FAIL"
    assert raised.value.report["commands"] == []


def test_cleanroom_rejects_an_invalid_future_doi_url_without_running(tmp_path):
    from scripts.verify_release_cleanroom import (
        CleanroomVerificationError,
        run_cleanroom,
    )

    source, evidence, checksums = _make_bundles(tmp_path)

    with pytest.raises(
        CleanroomVerificationError, match="DOI URL validation"
    ) as raised:
        run_cleanroom(
            source, evidence, checksums, doi_url="http://doi.org/example", execute=False
        )

    assert raised.value.report["status"] == "FAIL"
    assert raised.value.report["commands"] == []


def test_cleanroom_rejects_nonregular_archive_members_before_commands(tmp_path):
    from scripts.verify_release_cleanroom import (
        CleanroomVerificationError,
        run_cleanroom,
    )

    source, evidence, checksums = _make_bundles(tmp_path)
    source_prefix = "SCARF-AE-source-v1.0.0"
    members = {
        "scripts/run_ae.sh": b"#!/usr/bin/env bash\n",
        "scripts/run_ae.py": b"# runner\n",
        "scripts/run_rtl.sh": b"#!/usr/bin/env bash\n",
        "scripts/validate_ae.py": b"# validator\n",
        "hardware/dram/run.sh": b"#!/usr/bin/env bash\n",
    }
    manifest = {
        **_identity(),
        "bundle_kind": "source",
        "files": {relative: _sha256(payload) for relative, payload in members.items()},
        "archive": {"prefix": source_prefix, "format": "tar.gz"},
    }
    _write_bundle(
        source,
        source_prefix,
        members,
        manifest,
        symlink_name="scripts/untrusted-link.sh",
    )
    _write_sha256sums(checksums, source, evidence)

    with pytest.raises(
        CleanroomVerificationError, match="non-regular archive member"
    ) as raised:
        run_cleanroom(source, evidence, checksums, execute=False)

    assert raised.value.report["status"] == "FAIL"
    assert raised.value.report["commands"] == []


def test_cleanroom_rejects_mismatched_bundle_identity_before_commands(tmp_path):
    from scripts.verify_release_cleanroom import (
        CleanroomVerificationError,
        run_cleanroom,
    )

    source, evidence, checksums = _make_bundles(tmp_path, evidence_commit="e" * 40)

    with pytest.raises(CleanroomVerificationError, match="identity mismatch") as raised:
        run_cleanroom(source, evidence, checksums, execute=False)

    assert raised.value.report["status"] == "FAIL"
    assert raised.value.report["commands"] == []


def test_cleanroom_rejects_reference_evidence_hash_mismatch(tmp_path):
    from scripts.verify_release_cleanroom import (
        CleanroomVerificationError,
        run_cleanroom,
    )

    source, evidence, checksums = _make_bundles(tmp_path, bad_reference_digest=True)

    with pytest.raises(
        CleanroomVerificationError, match="reference evidence hash mismatch"
    ) as raised:
        run_cleanroom(source, evidence, checksums, execute=False)

    assert raised.value.report["status"] == "FAIL"
    assert raised.value.report["commands"] == []


def test_extracted_release_file_rehash_rejects_corruption(tmp_path):
    from scripts.verify_release_cleanroom import verify_extracted_release_files

    root = tmp_path / "source"
    path = root / "scripts/run_ae.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupted\n")

    with pytest.raises(ValueError, match="extracted release file hash mismatch"):
        verify_extracted_release_files(
            root,
            {"files": {"scripts/run_ae.py": "0" * 64}},
        )


def test_cleanroom_stops_after_first_failed_simulated_command(tmp_path):
    from scripts.verify_release_cleanroom import (
        CleanroomVerificationError,
        run_cleanroom,
    )

    source, evidence, checksums = _make_bundles(tmp_path)
    called = []

    def failing_runner(argv, *, cwd, env):
        called.append((argv, cwd, env))
        return subprocess.CompletedProcess(
            argv, 17, stdout=b"", stderr=b"simulated failure"
        )

    with pytest.raises(
        CleanroomVerificationError, match=r"quick \(exit 17\)"
    ) as raised:
        run_cleanroom(
            source,
            evidence,
            checksums,
            execute=True,
            command_runner=failing_runner,
        )

    report = raised.value.report
    assert report["status"] == "FAIL"
    assert len(called) == 1
    assert report["commands"][0]["name"] == "quick"
    assert report["commands"][0]["status"] == "FAIL"
    assert [record["status"] for record in report["commands"][1:]] == [
        "NOT_RUN",
        "NOT_RUN",
        "NOT_RUN",
    ]
