#!/usr/bin/env python3
"""Build and verify a deterministic SCARF AE source archive."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST_NAME = "release-manifest.json"
VERSION_RE = __import__("re").compile(r"v[0-9]+\.[0-9]+\.[0-9]+")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_bundle_binding(source: dict[str, Any], evidence: dict[str, Any]) -> None:
    for field in ("git_commit", "submodules", "zenodo_doi"):
        if source.get(field) != evidence.get(field):
            raise ValueError(f"source and evidence bundles disagree on {field}")


def write_sha256sums(paths: list[Path], output: Path) -> None:
    if not paths:
        raise ValueError("at least one release bundle is required")
    resolved = [Path(path).resolve() for path in paths]
    if len({path.name for path in resolved}) != len(resolved):
        raise ValueError("release bundle names must be unique")
    lines = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def release_files() -> list[Path]:
    names = git("ls-files", "--recurse-submodules", "-z").split("\0")
    return sorted(ROOT / name for name in names if name and (ROOT / name).is_file())


def source_release_files() -> list[Path]:
    selected = []
    for path in release_files():
        relative = path.relative_to(ROOT)
        if relative.parts[:1] == ("outputs",):
            continue
        if (
            relative.parts[:1] == ("datasets",)
            and relative.parts[:2] != ("datasets", "quick-re10k")
        ):
            continue
        if "checkpoints" in relative.parts:
            continue
        if relative.parts[:3] == ("artifact", "reference_results", "evidence"):
            continue
        selected.append(path)
    return selected


def evidence_release_files(root: Path = ROOT) -> dict[str, Path]:
    reference_root = root / "artifact/reference_results"
    manifest_path = reference_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("validation_status") != "PASS":
        raise ValueError("reference evidence is not staged from a passing run")
    files = {"reference-manifest.json": manifest_path}
    for relative, declared in manifest.get("files", {}).items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.parts[:1] != ("evidence",):
            raise ValueError(f"unsafe reference evidence path: {relative}")
        source = reference_root / pure
        if not source.is_file():
            raise FileNotFoundError(source)
        if (
            not isinstance(declared, dict)
            or declared.get("size") != source.stat().st_size
            or declared.get("sha256") != hashlib.sha256(source.read_bytes()).hexdigest()
        ):
            raise ValueError(f"reference evidence hash mismatch: {relative}")
        files[relative] = source
    contracts = {
        "contracts/evaluation_protocol.json": root / "artifact/evaluation_protocol.json",
        "contracts/claim_status.json": root / "artifact/claim_status.json",
        "contracts/compiled_protocol.json": root / "artifact/protocol/compiled.json",
        "contracts/expected_results.json": root / "artifact/expected_results.json",
        "contracts/datasets.json": root / "artifact/manifests/datasets.json",
        "contracts/checkpoints.json": root / "artifact/manifests/checkpoints.json",
        "contracts/runtime_assets.json": root / "artifact/manifests/runtime_assets.json",
        "contracts/THIRD_PARTY.md": root / "THIRD_PARTY.md",
    }
    for relative, source in contracts.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        files[relative] = source
    release_path = root / "artifact/release.json"
    if release_path.is_file():
        files["contracts/release.json"] = release_path
    return files


def _write_tar(
    tar_path: Path,
    prefix: str,
    files: dict[str, Path],
    manifest: dict[str, Any],
) -> None:
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for relative, path in sorted(files.items()):
            data = path.read_bytes()
            info = tarfile.TarInfo(f"{prefix}/{relative}")
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
            archive.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo(f"{prefix}/{MANIFEST_NAME}")
        info.size = len(manifest_bytes)
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))


def _compress_tar(tar_path: Path, output: Path, archive_format: str) -> None:
    if archive_format == "tar.gz":
        with tar_path.open("rb") as source, output.open("wb") as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    compressed.write(block)
        return
    if archive_format != "tar.zst":
        raise ValueError(f"unsupported archive format: {archive_format}")
    result = subprocess.run(
        ["zstd", "--threads=1", "--no-progress", "-19", "-f", str(tar_path), "-o", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "zstd compression failed")


def _build_from_mapping(
    output: Path,
    prefix: str,
    manifest: dict[str, Any],
    files: dict[str, Path],
    archive_format: str,
) -> dict[str, Any]:
    manifest = copy.deepcopy(manifest)
    manifest["files"] = {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, path in sorted(files.items())
    }
    manifest["archive"] = {
        "prefix": prefix,
        "format": archive_format,
        "deterministic_metadata": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tar", delete=False) as temporary:
        tar_path = Path(temporary.name)
    try:
        _write_tar(tar_path, prefix, files, manifest)
        _compress_tar(tar_path, output, archive_format)
    finally:
        tar_path.unlink(missing_ok=True)
    verified = verify(output)
    verified["archive_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    verified["archive_size"] = output.stat().st_size
    return verified


def build(output: Path, prefix: str, require_doi: bool = False) -> dict[str, Any]:
    if git("status", "--porcelain"):
        raise ValueError("release archive requires a clean worktree")
    if not prefix or PurePosixPath(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("archive prefix must be one safe path component")
    from scripts.check_release import build_manifest

    manifest = build_manifest(require_doi)
    if not manifest["validation"]["pass"]:
        raise ValueError("release checks failed: " + "; ".join(manifest["validation"]["failures"]))
    files = {
        path.relative_to(ROOT).as_posix(): path for path in source_release_files()
    }
    return _build_from_mapping(output, prefix, manifest, files, "tar.gz")


def build_bundles(
    output_dir: Path, version: str, *, require_doi: bool = False
) -> dict[str, Any]:
    if not VERSION_RE.fullmatch(version):
        raise ValueError("release version must have the form vMAJOR.MINOR.PATCH")
    if git("status", "--porcelain"):
        raise ValueError("release bundles require a clean worktree")
    from scripts.check_release import build_manifest

    identity = build_manifest(require_doi)
    if not identity["validation"]["pass"]:
        raise ValueError(
            "release checks failed: " + "; ".join(identity["validation"]["failures"])
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"SCARF-AE-source-{version}.tar.gz"
    evidence_path = output_dir / f"SCARF-AE-evidence-{version}.tar.zst"
    source_manifest = {**identity, "bundle_kind": "source", "release_version": version}
    evidence_manifest = {
        key: copy.deepcopy(identity.get(key))
        for key in ("schema_version", "git_commit", "submodules", "zenodo_doi")
    }
    evidence_manifest.update(
        {"bundle_kind": "evidence", "release_version": version, "validation": {"pass": True, "failures": []}}
    )
    source_files = {
        path.relative_to(ROOT).as_posix(): path for path in source_release_files()
    }
    source = _build_from_mapping(
        source_path, f"SCARF-AE-source-{version}", source_manifest, source_files, "tar.gz"
    )
    evidence = _build_from_mapping(
        evidence_path,
        f"SCARF-AE-evidence-{version}",
        evidence_manifest,
        evidence_release_files(),
        "tar.zst",
    )
    validate_bundle_binding(source, evidence)
    checksums = output_dir / "SHA256SUMS"
    write_sha256sums([source_path, evidence_path], checksums)
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "version": version,
        "source": source,
        "evidence": evidence,
        "sha256sums": str(checksums),
    }


@contextlib.contextmanager
def _readable_tar_path(archive_path: Path):
    if not archive_path.name.endswith(".tar.zst"):
        yield archive_path
        return
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temporary:
        tar_path = Path(temporary.name)
        result = subprocess.run(
            ["zstd", "-d", "--no-progress", "-c", str(archive_path)],
            stdout=temporary,
            stderr=subprocess.PIPE,
            check=False,
        )
    try:
        if result.returncode:
            raise ValueError("cannot decompress evidence archive")
        yield tar_path
    finally:
        tar_path.unlink(missing_ok=True)


def verify(archive_path: Path) -> dict[str, Any]:
    with _readable_tar_path(archive_path) as readable, tarfile.open(readable, "r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if not members:
            raise ValueError("archive is empty")
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        if len(roots) != 1:
            raise ValueError("archive must have exactly one root directory")
        root = next(iter(roots))
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != root:
                raise ValueError(f"unsafe archive path: {member.name}")
        manifest_member = archive.getmember(f"{root}/{MANIFEST_NAME}")
        manifest_stream = archive.extractfile(manifest_member)
        if manifest_stream is None:
            raise ValueError("release manifest cannot be read")
        manifest = json.loads(manifest_stream.read())
        archived = {member.name: member for member in members}
        failures = []
        for relative, expected_hash in manifest.get("files", {}).items():
            name = f"{root}/{relative}"
            member = archived.get(name)
            if member is None:
                failures.append(f"missing {relative}")
                continue
            stream = archive.extractfile(member)
            actual = sha256_bytes(stream.read()) if stream is not None else "unreadable"
            if actual != expected_hash:
                failures.append(f"hash mismatch {relative}")
        expected_names = {f"{root}/{name}" for name in manifest.get("files", {})}
        extra = sorted(set(archived) - expected_names - {f"{root}/{MANIFEST_NAME}"})
        failures.extend(f"unmanifested {name}" for name in extra)
    if failures:
        raise ValueError("archive verification failed: " + "; ".join(failures[:10]))
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "prefix": root,
        "verified_files": len(manifest["files"]),
        "git_commit": manifest.get("git_commit"),
        "submodules": manifest.get("submodules"),
        "zenodo_doi": manifest.get("zenodo_doi"),
        "bundle_kind": manifest.get("bundle_kind"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix", default="SCARF-AE")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--require-doi", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.output, args.verify, args.output_dir)) != 1:
        parser.error("specify exactly one of --output, --output-dir, or --verify")
    try:
        if args.output_dir:
            result = build_bundles(
                args.output_dir.resolve(), args.version, require_doi=args.require_doi
            )
        elif args.output:
            result = build(args.output.resolve(), args.prefix, args.require_doi)
        else:
            result = verify(args.verify.resolve())
    except (OSError, RuntimeError, tarfile.TarError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
