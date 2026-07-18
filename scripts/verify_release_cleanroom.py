#!/usr/bin/env python3
"""Verify and replay a SCARF release from isolated source and evidence bundles.

The optional DOI URL is deliberately recorded but never fetched.  Downloading a
DOI-hosted bundle is a separate release action; this verifier only operates on
the two local archives named by a checked ``SHA256SUMS`` file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, NoReturn
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_archive import (
    MANIFEST_NAME,
    _readable_tar_path,
    normalize_archive_relative,
    validate_bundle_binding,
    verify as verify_archive,
)


DEFAULT_MAX_EXTRACTED_BYTES = 16 * 1024 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
REQUIRED_SUBMODULES = frozenset({"transplat", "mvsplat", "depthsplat"})
REQUIRED_SOURCE_FILES = (
    "scripts/run_ae.sh",
    "scripts/run_ae.py",
    "scripts/run_rtl.sh",
    "scripts/validate_ae.py",
    "hardware/dram/run.sh",
)


class CleanroomVerificationError(RuntimeError):
    """A release bundle is unsafe, inconsistent, or failed clean-room replay."""

    def __init__(self, message: str, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report) if report is not None else None


def sha256_file(path: Path) -> str:
    """Hash one regular release input without loading a large archive into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise FileNotFoundError(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key in {label}: {key}")
            result[key] = value
        return result

    _require_regular_file(path, label=label)
    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return loaded


def _checksum_entries(path: Path) -> dict[str, str]:
    """Parse the exact two-space format written by ``write_sha256sums``."""
    _require_regular_file(path, label="SHA256SUMS")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read SHA256SUMS: {path}") from exc
    if not lines:
        raise ValueError("SHA256SUMS is empty")
    entries: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("SHA256SUMS has an invalid entry format")
        digest, name = line[:64], line[66:]
        if not SHA256_RE.fullmatch(digest):
            raise ValueError(f"SHA256SUMS has an invalid digest for {name!r}")
        if (
            not name
            or name in {".", ".."}
            or name.startswith("*")
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            raise ValueError(f"SHA256SUMS has an unsafe archive name: {name!r}")
        if name in entries:
            raise ValueError(f"SHA256SUMS has a duplicate archive entry: {name}")
        entries[name] = digest
    return entries


def verify_archive_hashes(
    source_archive: Path, evidence_archive: Path, sha256sums: Path
) -> dict[str, str]:
    """Require SHA256SUMS to name exactly the supplied source and evidence archives."""
    source_archive = Path(source_archive)
    evidence_archive = Path(evidence_archive)
    sha256sums = Path(sha256sums)
    _require_regular_file(source_archive, label="source archive")
    _require_regular_file(evidence_archive, label="evidence archive")
    if source_archive.name == evidence_archive.name:
        raise ValueError("source and evidence archives must have different file names")

    entries = _checksum_entries(sha256sums)
    expected_names = {source_archive.name, evidence_archive.name}
    if set(entries) != expected_names:
        missing = sorted(expected_names - set(entries))
        unexpected = sorted(set(entries) - expected_names)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            "SHA256SUMS must name exactly the two supplied archives: "
            + "; ".join(details)
        )

    actual = {
        source_archive.name: sha256_file(source_archive),
        evidence_archive.name: sha256_file(evidence_archive),
    }
    for name, digest in actual.items():
        if entries[name] != digest:
            raise ValueError(f"SHA256 mismatch for {name}")
    return {
        "source": actual[source_archive.name],
        "evidence": actual[evidence_archive.name],
    }


def _validate_archive_members(
    archive: tarfile.TarFile, *, max_extracted_bytes: int
) -> tuple[str, list[tuple[str, tarfile.TarInfo]]]:
    members = archive.getmembers()
    if not members:
        raise ValueError("archive is empty")
    if max_extracted_bytes <= 0:
        raise ValueError("max extracted bytes must be positive")

    roots: set[str] = set()
    seen: set[str] = set()
    validated: list[tuple[str, tarfile.TarInfo]] = []
    total_size = 0
    for member in members:
        if not member.isreg() or member.issparse():
            raise ValueError(f"non-regular archive member: {member.name}")
        normalized = normalize_archive_relative(member.name)
        if normalized != member.name:
            raise ValueError(f"archive contains non-normalized paths: {member.name}")
        if normalized in seen:
            raise ValueError(f"duplicate normalized archive member: {normalized}")
        parts = PurePosixPath(normalized).parts
        if len(parts) < 2:
            raise ValueError(
                f"archive member is outside the archive root: {member.name}"
            )
        if member.size < 0:
            raise ValueError(f"archive member has an invalid size: {member.name}")
        total_size += member.size
        if total_size > max_extracted_bytes:
            raise ValueError(
                "archive exceeds the configured extracted-size limit "
                f"({max_extracted_bytes} bytes)"
            )
        roots.add(parts[0])
        seen.add(normalized)
        validated.append((normalized, member))
    if len(roots) != 1:
        raise ValueError("archive must have exactly one root directory")
    return next(iter(roots)), validated


def _mkdir_checked(path: Path) -> None:
    if path.exists():
        mode = path.lstat().st_mode
        if path.is_symlink() or not stat.S_ISDIR(mode):
            raise ValueError(f"unsafe extraction path component: {path}")
        return
    path.mkdir(mode=0o700)


def _write_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> None:
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"archive member cannot be read: {member.name}")
    written = 0
    try:
        with destination.open("xb") as output:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                written += len(block)
    finally:
        stream.close()
    if written != member.size:
        raise ValueError(f"archive member size changed while extracting: {member.name}")
    # Release builders use only these portable file modes.  Do not preserve
    # set-id bits or other archive-controlled permissions in the temp tree.
    os.chmod(destination, 0o755 if member.mode & 0o111 else 0o644)


def safely_extract_archive(
    archive_path: Path, destination: Path, *, max_extracted_bytes: int
) -> Path:
    """Extract regular, normalized members without ``TarFile.extractall``."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"extraction destination is not empty: {destination}")
    _mkdir_checked(destination)
    destination_root = destination.resolve()
    try:
        readable_context = _readable_tar_path(archive_path)
        with readable_context as readable, tarfile.open(readable, "r:*") as archive:
            prefix, members = _validate_archive_members(
                archive, max_extracted_bytes=max_extracted_bytes
            )
            for normalized, member in members:
                relative_parts = PurePosixPath(normalized).parts
                target = destination.joinpath(*relative_parts)
                try:
                    target.relative_to(destination)
                except ValueError as exc:
                    raise ValueError(f"unsafe extraction path: {member.name}") from exc
                parent = destination
                for part in relative_parts[:-1]:
                    parent = parent / part
                    _mkdir_checked(parent)
                if target.exists() or target.is_symlink():
                    raise ValueError(f"duplicate extraction target: {member.name}")
                _write_member(archive, member, target)
    except (OSError, RuntimeError, tarfile.TarError, ValueError) as exc:
        raise ValueError(f"cannot safely extract {archive_path.name}: {exc}") from exc

    extracted_root = destination / prefix
    if extracted_root.resolve().parent != destination_root:
        raise ValueError(f"archive prefix escaped extraction directory: {prefix}")
    return extracted_root


def _release_manifest(root: Path, *, bundle_kind: str) -> dict[str, Any]:
    manifest = _load_json_object(root / MANIFEST_NAME, label="release manifest")
    if manifest.get("schema_version") != "1.0":
        raise ValueError("release manifest has an unsupported schema version")
    if manifest.get("bundle_kind") != bundle_kind:
        raise ValueError(f"release manifest is not a {bundle_kind} bundle")
    validation = manifest.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("pass") is not True
        or validation.get("failures") != []
    ):
        raise ValueError(
            "release manifest does not record a passing release validation"
        )
    commit = manifest.get("git_commit")
    submodules = manifest.get("submodules")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise ValueError("release manifest has an invalid git commit")
    if not isinstance(submodules, Mapping) or set(submodules) != REQUIRED_SUBMODULES:
        raise ValueError("release manifest has an invalid submodule binding")
    if any(
        not isinstance(value, str) or not COMMIT_RE.fullmatch(value)
        for value in submodules.values()
    ):
        raise ValueError("release manifest has an invalid submodule revision")
    doi = manifest.get("zenodo_doi")
    if doi is not None and not isinstance(doi, str):
        raise ValueError("release manifest has an invalid Zenodo DOI")
    return manifest


def verify_extracted_release_files(root: Path, manifest: Mapping[str, Any]) -> int:
    """Rehash the extracted tree, not just the archive stream that produced it."""
    declared = manifest.get("files")
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("release manifest has no valid file mapping")
    declared_paths: set[str] = set()
    for relative, expected_digest in declared.items():
        if not isinstance(relative, str):
            raise ValueError("release manifest has a non-string file path")
        normalized = normalize_archive_relative(relative)
        if normalized != relative or normalized == MANIFEST_NAME:
            raise ValueError(f"release manifest has an unsafe file path: {relative}")
        if normalized in declared_paths:
            raise ValueError(f"duplicate normalized release manifest path: {relative}")
        if not isinstance(expected_digest, str) or not SHA256_RE.fullmatch(
            expected_digest
        ):
            raise ValueError(f"release manifest has an invalid file hash: {relative}")
        path = root.joinpath(*PurePosixPath(normalized).parts)
        _require_regular_file(path, label=f"extracted release file {relative}")
        if sha256_file(path) != expected_digest:
            raise ValueError(f"extracted release file hash mismatch: {relative}")
        declared_paths.add(normalized)
    return len(declared_paths)


def _validate_verified_bundle(
    archive_path: Path, *, bundle_kind: str
) -> dict[str, Any]:
    try:
        record = verify_archive(Path(archive_path))
    except (
        OSError,
        RuntimeError,
        tarfile.TarError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"archive manifest verification failed for {Path(archive_path).name}: {exc}"
        ) from exc
    if record.get("status") != "PASS" or record.get("bundle_kind") != bundle_kind:
        raise ValueError(f"archive has the wrong verified bundle kind: {archive_path}")
    return record


def _bind_extracted_manifest_to_archive(
    manifest: Mapping[str, Any], archive_record: Mapping[str, Any], *, label: str
) -> None:
    for field in ("git_commit", "submodules", "zenodo_doi"):
        if manifest.get(field) != archive_record.get(field):
            raise ValueError(
                f"{label} extracted manifest disagrees with verified archive {field}"
            )


def _regular_payload_files(payload_root: Path) -> set[str]:
    if not payload_root.is_dir() or payload_root.is_symlink():
        raise ValueError("evidence bundle has no regular evidence directory")
    files: set[str] = set()
    for current, directories, names in os.walk(payload_root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            path = current_path / directory
            mode = path.lstat().st_mode
            if path.is_symlink() or not stat.S_ISDIR(mode):
                raise ValueError(f"unsafe evidence directory: {path}")
        for name in names:
            path = current_path / name
            mode = path.lstat().st_mode
            if path.is_symlink() or not stat.S_ISREG(mode):
                raise ValueError(f"unsafe evidence file: {path}")
            files.add(path.relative_to(payload_root.parent).as_posix())
    return files


def verify_extracted_evidence(evidence_root: Path) -> dict[str, Any]:
    """Bind the staged reference manifest to every extracted evidence payload file."""
    evidence_root = Path(evidence_root)
    manifest = _load_json_object(
        evidence_root / "reference-manifest.json", label="reference evidence manifest"
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("validation_status") != "PASS"
    ):
        raise ValueError("reference evidence was not staged from a passing validation")
    if manifest.get("validation_require_key_results") is not True:
        raise ValueError(
            "reference evidence was not validated with --require-key-results"
        )
    categories = manifest.get("categories")
    if (
        not isinstance(categories, list)
        or not categories
        or any(not isinstance(category, str) or not category for category in categories)
        or len(set(categories)) != len(categories)
    ):
        raise ValueError("reference evidence manifest has invalid categories")
    declared = manifest.get("files")
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("reference evidence manifest has no valid file mapping")

    declared_paths: set[str] = set()
    for relative, record in declared.items():
        if not isinstance(relative, str):
            raise ValueError("reference evidence manifest has a non-string file path")
        normalized = normalize_archive_relative(relative)
        if normalized != relative or not normalized.startswith("evidence/"):
            raise ValueError(f"unsafe reference evidence path: {relative}")
        if normalized in declared_paths:
            raise ValueError(
                f"duplicate normalized reference evidence path: {relative}"
            )
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid reference evidence record: {relative}")
        expected_size = record.get("size")
        expected_digest = record.get("sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or not isinstance(expected_digest, str)
            or not SHA256_RE.fullmatch(expected_digest)
        ):
            raise ValueError(f"invalid reference evidence hash record: {relative}")
        path = evidence_root.joinpath(*PurePosixPath(normalized).parts)
        _require_regular_file(path, label=f"reference evidence {relative}")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_digest:
            raise ValueError(f"reference evidence hash mismatch: {relative}")
        declared_paths.add(normalized)

    actual_paths = _regular_payload_files(evidence_root / "evidence")
    if actual_paths != declared_paths:
        missing = sorted(declared_paths - actual_paths)
        extra = sorted(actual_paths - declared_paths)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing[:10]))
        if extra:
            details.append("unmanifested " + ", ".join(extra[:10]))
        raise ValueError(
            "reference evidence payload differs from its manifest: "
            + "; ".join(details)
        )
    return {
        "reference_manifest_sha256": sha256_file(
            evidence_root / "reference-manifest.json"
        ),
        "files": len(declared_paths),
        "categories": sorted(categories),
    }


def _copy_evidence_payload(evidence_root: Path, output_root: Path) -> int:
    payload_root = evidence_root / "evidence"
    _mkdir_checked(output_root)
    copied = 0
    for source in sorted(payload_root.rglob("*")):
        if not source.is_file():
            continue
        _require_regular_file(source, label="extracted evidence input")
        relative = source.relative_to(payload_root)
        destination = output_root / relative
        parent = output_root
        for part in relative.parts[:-1]:
            parent = parent / part
            _mkdir_checked(parent)
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o644)
        copied += 1
    return copied


def _require_source_commands(source_root: Path) -> None:
    for relative in REQUIRED_SOURCE_FILES:
        _require_regular_file(source_root / relative, label="extracted source command")


def cleanroom_command_plan(
    source_root: Path, output_root: Path
) -> list[dict[str, Any]]:
    """Return the fixed, ordered replay commands and portable audit rendering."""
    runner = source_root / "scripts/run_ae.sh"
    modes = ("quick", "rtl", "dram")
    plan = []
    for order, mode in enumerate(modes, start=1):
        plan.append(
            {
                "order": order,
                "name": mode,
                "argv": ["bash", str(runner), mode, "--output-root", str(output_root)],
                "display_argv": [
                    "bash",
                    "$SOURCE_ROOT/scripts/run_ae.sh",
                    mode,
                    "--output-root",
                    "$OUTPUT_ROOT",
                ],
            }
        )
    plan.append(
        {
            "order": 4,
            "name": "validate",
            "argv": [
                "bash",
                str(runner),
                "validate",
                "--output-root",
                str(output_root),
                "--require-key-results",
            ],
            "display_argv": [
                "bash",
                "$SOURCE_ROOT/scripts/run_ae.sh",
                "validate",
                "--output-root",
                "$OUTPUT_ROOT",
                "--require-key-results",
            ],
        }
    )
    return plan


def _command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # An inherited checkout on PYTHONPATH would defeat source-archive isolation.
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["SCARF_CLEANROOM"] = "1"
    return environment


def _run_command(
    argv: list[str], *, cwd: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return (
        value if isinstance(value, bytes) else value.encode("utf-8", errors="replace")
    )


def _command_record(
    item: Mapping[str, Any], *, status: str = "NOT_RUN"
) -> dict[str, Any]:
    return {
        "order": item["order"],
        "name": item["name"],
        "cwd": "$SOURCE_ROOT",
        "argv": item["display_argv"],
        "status": status,
    }


def _validate_doi_url(doi_url: str | None) -> str | None:
    if doi_url is None:
        return None
    parsed = urlparse(doi_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("--doi-url must be a credential-free HTTPS URL")
    return doi_url


CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def _fail(message: str, report: dict[str, Any]) -> NoReturn:
    report["status"] = "FAIL"
    report["failure"] = message
    raise CleanroomVerificationError(message, report)


def run_cleanroom(
    source_archive: Path,
    evidence_archive: Path,
    sha256sums: Path,
    *,
    doi_url: str | None = None,
    execute: bool = True,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Verify, extract, and replay a release; ``execute=False`` is audit-only."""
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "NOT_RUN",
        "doi_download": {
            "url": None,
            "status": "NOT_FETCHED",
        },
        "commands": [],
    }
    try:
        report["doi_download"]["url"] = _validate_doi_url(doi_url)
    except ValueError as exc:
        _fail(f"DOI URL validation failed: {exc}", report)
    try:
        archive_hashes = verify_archive_hashes(
            Path(source_archive), Path(evidence_archive), Path(sha256sums)
        )
        report["archives"] = {
            "source": {
                "name": Path(source_archive).name,
                "sha256": archive_hashes["source"],
            },
            "evidence": {
                "name": Path(evidence_archive).name,
                "sha256": archive_hashes["evidence"],
            },
        }
    except (OSError, ValueError) as exc:
        _fail(f"archive hash verification failed: {exc}", report)

    try:
        with tempfile.TemporaryDirectory(
            prefix="scarf-release-cleanroom-"
        ) as temporary:
            temporary_root = Path(temporary)
            source_root = safely_extract_archive(
                Path(source_archive),
                temporary_root / "source",
                max_extracted_bytes=max_extracted_bytes,
            )
            evidence_root = safely_extract_archive(
                Path(evidence_archive),
                temporary_root / "evidence",
                max_extracted_bytes=max_extracted_bytes,
            )

            source_verified = _validate_verified_bundle(
                Path(source_archive), bundle_kind="source"
            )
            evidence_verified = _validate_verified_bundle(
                Path(evidence_archive), bundle_kind="evidence"
            )
            if source_root.name != source_verified["prefix"]:
                raise ValueError(
                    "source archive extraction prefix disagrees with its manifest"
                )
            if evidence_root.name != evidence_verified["prefix"]:
                raise ValueError(
                    "evidence archive extraction prefix disagrees with its manifest"
                )
            source_manifest = _release_manifest(source_root, bundle_kind="source")
            evidence_manifest = _release_manifest(evidence_root, bundle_kind="evidence")
            _bind_extracted_manifest_to_archive(
                source_manifest, source_verified, label="source"
            )
            _bind_extracted_manifest_to_archive(
                evidence_manifest, evidence_verified, label="evidence"
            )
            source_file_count = verify_extracted_release_files(
                source_root, source_manifest
            )
            evidence_file_count = verify_extracted_release_files(
                evidence_root, evidence_manifest
            )
            try:
                validate_bundle_binding(source_manifest, evidence_manifest)
            except ValueError as exc:
                raise ValueError(
                    f"source and evidence bundle identity mismatch: {exc}"
                ) from exc
            _require_source_commands(source_root)
            evidence_summary = verify_extracted_evidence(evidence_root)

            output_root = temporary_root / "combined-evidence"
            copied = _copy_evidence_payload(evidence_root, output_root)
            plan = cleanroom_command_plan(source_root, output_root)
            report.update(
                {
                    "source": {
                        "prefix": source_root.name,
                        "git_commit": source_manifest["git_commit"],
                        "submodules": source_manifest["submodules"],
                        "verified_files": source_file_count,
                    },
                    "evidence": {
                        "prefix": evidence_root.name,
                        "git_commit": evidence_manifest["git_commit"],
                        "submodules": evidence_manifest["submodules"],
                        "verified_files": evidence_file_count,
                        "payload": evidence_summary,
                    },
                    "evidence_overlay": {
                        "source": "$EVIDENCE_ROOT/evidence",
                        "destination": "$OUTPUT_ROOT",
                        "copied_files": copied,
                    },
                    "commands": [_command_record(item) for item in plan],
                }
            )
            if not execute:
                return report

            runner = command_runner or _run_command
            environment = _command_environment()
            for item, record in zip(plan, report["commands"]):
                try:
                    completed = runner(item["argv"], cwd=source_root, env=environment)
                except (OSError, subprocess.SubprocessError) as exc:
                    record["status"] = "FAIL"
                    record["error"] = str(exc)
                    _fail(
                        f"clean-room command could not start: {item['name']}: {exc}",
                        report,
                    )
                stdout = _as_bytes(completed.stdout)
                stderr = _as_bytes(completed.stderr)
                record.update(
                    {
                        "returncode": completed.returncode,
                        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                        "stdout_bytes": len(stdout),
                        "stderr_bytes": len(stderr),
                    }
                )
                if completed.returncode != 0:
                    record["status"] = "FAIL"
                    tail = (stderr or stdout)[-2000:].decode("utf-8", errors="replace")
                    if tail:
                        record["output_tail"] = tail
                    _fail(
                        f"clean-room command failed: {item['name']} "
                        f"(exit {completed.returncode})",
                        report,
                    )
                record["status"] = "PASS"
            report["status"] = "PASS"
            return report
    except CleanroomVerificationError:
        raise
    except (
        OSError,
        RuntimeError,
        tarfile.TarError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as exc:
        _fail(f"clean-room preparation failed: {exc}", report)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--evidence-archive", type=Path, required=True)
    parser.add_argument("--sha256sums", type=Path, required=True)
    parser.add_argument(
        "--doi-url",
        help="Record, but do not fetch, the future HTTPS DOI download URL.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the auditable verification record outside the temporary clean room.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify and stage inputs but mark every replay command NOT_RUN.",
    )
    parser.add_argument(
        "--max-extracted-bytes",
        type=int,
        default=DEFAULT_MAX_EXTRACTED_BYTES,
        help="Maximum uncompressed bytes accepted from either archive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run and args.report is None:
        print(
            "error: --report is required when executing clean-room commands",
            file=sys.stderr,
        )
        return 2
    try:
        result = run_cleanroom(
            args.source_archive.resolve(),
            args.evidence_archive.resolve(),
            args.sha256sums.resolve(),
            doi_url=args.doi_url,
            execute=not args.dry_run,
            max_extracted_bytes=args.max_extracted_bytes,
        )
    except CleanroomVerificationError as exc:
        if args.report and exc.report is not None:
            try:
                _write_report(args.report.resolve(), exc.report)
            except OSError as report_error:
                print(
                    f"error: cannot write clean-room report: {report_error}",
                    file=sys.stderr,
                )
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.report:
        try:
            _write_report(args.report.resolve(), result)
        except OSError as exc:
            print(f"error: cannot write clean-room report: {exc}", file=sys.stderr)
            return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
