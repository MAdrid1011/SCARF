#!/usr/bin/env python3
"""Download checkpoint assets and verify published SHA256 values."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected(record: dict, profile: str) -> bool:
    if profile == "all":
        return True
    if profile in record.get("profiles", []):
        return True
    if record.get("profile") == profile:
        return True
    return (
        profile == "quick"
        and record.get("model") == "mvsplat"
        and record.get("dataset") == "re10k"
    )


def _download(curl: str, record: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == record["size"]:
        if sha256_file(target) == record["sha256"]:
            return
    partial = target.with_suffix(target.suffix + ".partial")
    result = subprocess.run(
        [
            curl,
            "-L",
            "--fail",
            "--retry",
            "3",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            record["url"],
        ],
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"download failed: {record['url']}")
    if partial.stat().st_size != record["size"]:
        raise ValueError(f"size mismatch for {record['path']}")
    actual = sha256_file(partial)
    if actual != record["sha256"]:
        raise ValueError(f"SHA256 mismatch for {record['path']}: {actual}")
    partial.replace(target)


def _safe_extract(archive: Path, archive_root: str, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"invalid existing extracted asset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary_text:
        temporary = Path(temporary_text)
        extracted = temporary / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream.getmembers():
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                    raise ValueError(f"unsafe archive member: {member.name}")
                if pure.parts[0] != archive_root or member.issym() or member.islnk():
                    raise ValueError(f"unexpected archive member: {member.name}")
                relative = Path(*pure.parts[1:])
                output = extracted / relative
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    source = stream.extractfile(member)
                    if source is None:
                        raise ValueError(f"cannot read archive member: {member.name}")
                    with output.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
        if not (extracted / "hubconf.py").is_file():
            raise ValueError("DINOv2 archive has no hubconf.py")
        extracted.rename(target)


def _install_archive(curl: str, record: dict) -> None:
    target = ROOT / record["path"]
    marker = target / ".scarf-asset.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        license_path = target / "LICENSE"
        if (
            metadata.get("archive_sha256") == record["sha256"]
            and metadata.get("commit") == record["commit"]
            and license_path.is_file()
            and sha256_file(license_path) == record["license_sha256"]
        ):
            return
    archive = ROOT / "assets/downloads" / f"{record['commit']}.tar.gz"
    _download(curl, record, archive)
    _safe_extract(archive, record["archive_root"], target)
    marker.write_text(
        json.dumps(
            {"archive_sha256": record["sha256"], "commit": record["commit"]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def download(profile: str, manifest_path: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("curl is required; install it before downloading assets")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [
        record
        for record in manifest["files"]
        if _selected(record, profile)
    ]
    if not selected:
        raise ValueError(f"manifest has no files for profile {profile}")
    for record in selected:
        if record.get("kind", "file") == "tar_gz":
            _install_archive(curl, record)
        elif record.get("kind", "file") == "file":
            _download(curl, record, ROOT / record["path"])
        else:
            raise ValueError(f"unsupported asset kind: {record.get('kind')}")
        print(f"verified {record['path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("quick", "classic", "depthsplat", "all"), default="all"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifact/manifests/checkpoints.json",
    )
    args = parser.parse_args()
    try:
        download(args.profile, args.manifest)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
