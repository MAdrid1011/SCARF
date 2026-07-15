"""Validate hash-pinned model assets before any inference imports use them."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifact/manifests/runtime_assets.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(record: dict[str, Any], model: str) -> bool:
    profile = "depthsplat" if model == "depthsplat" else "classic"
    models = record.get("models")
    return profile in record.get("profiles", []) and (
        models is None or model in models
    )


@lru_cache(maxsize=8)
def validate_runtime_assets(
    model: str,
    *,
    root: Path = ROOT,
    manifest_path: Path = MANIFEST,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {}
    for record in manifest["files"]:
        if not _required(record, model):
            continue
        path = root / record["path"]
        kind = record.get("kind", "file")
        if kind == "file":
            if not path.is_file() or path.stat().st_size != record["size"]:
                raise FileNotFoundError(f"runtime asset is missing or truncated: {record['path']}")
            actual = sha256_file(path)
            if actual != record["sha256"]:
                raise ValueError(f"runtime asset SHA256 mismatch: {record['path']}")
            records[record["name"]] = {
                "path": record["path"],
                "sha256": actual,
            }
        elif kind == "tar_gz":
            archive = root / "assets/downloads" / f"{record['commit']}.tar.gz"
            marker = path / ".scarf-asset.json"
            license_path = path / "LICENSE"
            hubconf = path / "hubconf.py"
            if not archive.is_file() or sha256_file(archive) != record["sha256"]:
                raise ValueError("pinned DINOv2 source archive is missing or invalid")
            if not marker.is_file() or not license_path.is_file() or not hubconf.is_file():
                raise FileNotFoundError("pinned DINOv2 source tree is incomplete")
            marker_record = json.loads(marker.read_text(encoding="utf-8"))
            if (
                marker_record.get("archive_sha256") != record["sha256"]
                or marker_record.get("commit") != record["commit"]
                or sha256_file(license_path) != record["license_sha256"]
            ):
                raise ValueError("pinned DINOv2 source provenance mismatch")
            records[record["name"]] = {
                "path": record["path"],
                "archive_sha256": record["sha256"],
                "commit": record["commit"],
                "license_sha256": record["license_sha256"],
            }
        else:
            raise ValueError(f"unsupported runtime asset kind: {kind}")
    if not records:
        raise ValueError(f"runtime asset manifest has no entries for model {model}")
    return records
