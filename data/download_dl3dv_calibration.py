#!/usr/bin/env python3
"""Fetch and prepare only the object-id-pinned DL3DV calibration archives.

The tool has two deliberately separate phases.  ``--write-plan`` reads the
official archive tree and writes the evaluation-disjoint selection without
downloading image data.  ``--plan`` re-reads that tree, verifies it still
matches the immutable plan, then downloads only the selected archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_manifest import build as build_dataset_manifest
from data.plan_dl3dv_calibration import (
    SOURCE_REPOSITORY,
    build_plan,
    canonical_sha256,
)


REPO_TYPE = "dataset"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_REPOSITORY}"
CALIBRATION_SUBSET = "calibration"
HOLDOUT_SUBSET = "holdout"
ALL_SUBSETS = "all"
DEFAULT_OUTPUT_ROOT = ROOT / "downloads" / "calibration" / "dl3dv"


class DownloadContractError(ValueError):
    """Raised when a gated source cannot satisfy the frozen data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _archive_object_id(entry: Any) -> tuple[str, str] | None:
    """Read the strongest stable object id exposed by a Hub tree entry."""
    lfs = getattr(entry, "lfs", None)
    # ``BlobLfsInfo`` can satisfy Mapping while its mapping interface exposes
    # API-field names rather than its parsed attributes.  Prefer attributes.
    # Current clients use ``sha256``; older clients used ``oid``.
    value = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
    if value is None and isinstance(lfs, Mapping):
        value = lfs.get("sha256") or lfs.get("oid")
    if _is_sha256(value):
        return "lfs_sha256", value
    # Some gated Hub API responses deliberately mask LFS SHA256 as asterisks.
    # The revision-pinned Git blob is still a stable source identifier. The
    # local byte SHA256 is recorded after download instead of being invented.
    blob = getattr(entry, "blob_id", None) or getattr(entry, "oid", None)
    if (
        isinstance(blob, str)
        and len(blob) == 40
        and all(character in "0123456789abcdef" for character in blob)
    ):
        return "git_blob", blob
    return None


def list_archive_tree(api: Any, *, revision: str) -> list[dict[str, Any]]:
    """List only valid per-scene ZIPs and bind each to its LFS SHA256."""
    # The full source has thousands of objects. ``expand=False`` keeps Hub
    # pagination at its larger normal page size, and a failed page restarts the
    # enumeration rather than producing a partial tree that could bias ranking.
    last_error: Exception | None = None
    entries: Iterable[Any] | None = None
    for attempt in range(4):
        try:
            entries = list(
                api.list_repo_tree(
                    repo_id=SOURCE_REPOSITORY,
                    repo_type=REPO_TYPE,
                    revision=revision,
                    recursive=True,
                    expand=False,
                )
            )
            break
        except Exception as exc:  # Hub transports expose several error types.
            last_error = exc
            if attempt + 1 < 4:
                time.sleep(2**attempt)
    if entries is None:
        assert last_error is not None
        raise DownloadContractError("could not enumerate the complete DL3DV archive tree") from last_error
    archives: list[dict[str, Any]] = []
    for entry in entries:
        path = getattr(entry, "path", None)
        if not isinstance(path, str) or not path.endswith(".zip"):
            continue
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 2:
            raise DownloadContractError(f"unsafe archive path in upstream tree: {path}")
        size = getattr(entry, "size", None)
        object_id = _archive_object_id(entry)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise DownloadContractError(f"archive has no positive size: {path}")
        if object_id is None:
            raise DownloadContractError(
                f"archive has no stable object id: {path}"
            )
        object_id_kind, oid = object_id
        archives.append(
            {
                "path": pure.as_posix(),
                "size": size,
                "oid": oid,
                "object_id_kind": object_id_kind,
            }
        )
    if not archives:
        raise DownloadContractError("official DL3DV archive tree contains no scene ZIPs")
    if len({item["path"] for item in archives}) != len(archives):
        raise DownloadContractError("official DL3DV archive tree repeats an archive path")
    return sorted(archives, key=lambda item: item["path"])


def load_evaluation_index(path: Path) -> tuple[list[str], str]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not source:
        raise DownloadContractError("DL3DV evaluation index must be a non-empty JSON object")
    scenes = list(source)
    if len(set(scenes)) != len(scenes) or any(
        not isinstance(scene, str) or not scene for scene in scenes
    ):
        raise DownloadContractError("DL3DV evaluation index has invalid scene names")
    return scenes, sha256_file(path)


def build_source_plan(
    tree: list[dict[str, Any]], *, evaluation_index: Path, revision: str
) -> dict[str, Any]:
    scenes, index_sha256 = load_evaluation_index(evaluation_index)
    return build_plan(
        tree,
        evaluation_scenes=scenes,
        evaluation_index_sha256=index_sha256,
        revision=revision,
    )


def load_and_verify_plan(
    path: Path,
    *,
    tree: list[dict[str, Any]],
    evaluation_index: Path,
    revision: str,
) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DownloadContractError(f"DL3DV calibration plan is invalid JSON: {path}") from exc
    expected = build_source_plan(tree, evaluation_index=evaluation_index, revision=revision)
    if canonical_sha256(plan) != canonical_sha256(expected):
        raise DownloadContractError(
            "DL3DV calibration plan does not match the current pinned archive tree "
            "and evaluation index"
        )
    return expected


def _selected_archives(plan: Mapping[str, Any], subset: str) -> list[dict[str, Any]]:
    selection = plan.get("selection")
    if not isinstance(selection, Mapping):
        raise DownloadContractError("DL3DV calibration plan has no selection")
    calibration = selection.get("calibration_train")
    holdout = selection.get("calibration_holdout")
    if not isinstance(calibration, list) or not isinstance(holdout, list):
        raise DownloadContractError("DL3DV calibration plan has invalid archive selections")
    if subset == CALIBRATION_SUBSET:
        selected = calibration
    elif subset == HOLDOUT_SUBSET:
        selected = holdout
    elif subset == ALL_SUBSETS:
        selected = [*calibration, *holdout]
    else:
        raise DownloadContractError(f"unsupported DL3DV archive subset: {subset}")
    if not selected:
        raise DownloadContractError("DL3DV calibration selection is empty")
    normalized: list[dict[str, Any]] = []
    for item in selected:
        if not isinstance(item, Mapping):
            raise DownloadContractError("DL3DV calibration selection has an invalid archive")
        scene, relative, size, oid, object_id_kind = (
            item.get("scene"),
            item.get("path"),
            item.get("size"),
            item.get("oid"),
            item.get("object_id_kind"),
        )
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            not isinstance(scene, str)
            or not scene
            or pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or len(pure.parts) != 2
            or pure.stem != scene
            or pure.suffix != ".zip"
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or object_id_kind not in {"lfs_sha256", "git_blob"}
            or not isinstance(oid, str)
            or len(oid) != (64 if object_id_kind == "lfs_sha256" else 40)
            or any(character not in "0123456789abcdef" for character in oid)
        ):
            raise DownloadContractError("DL3DV calibration selection has invalid archive metadata")
        normalized.append(
            {
                "scene": scene,
                "path": pure.as_posix(),
                "size": size,
                "oid": oid,
                "object_id_kind": object_id_kind,
            }
        )
    if len({item["scene"] for item in normalized}) != len(normalized):
        raise DownloadContractError("DL3DV calibration selection repeats a scene")
    return normalized


def _archive_destination(archive_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    destination = (archive_root / Path(*pure.parts)).resolve()
    if archive_root.resolve() not in destination.parents:
        raise DownloadContractError(f"archive destination escapes output root: {relative}")
    return destination


DownloadFile = Callable[[str], str | Path]


def _verify_archive(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"selected DL3DV archive is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != record["size"]:
        raise DownloadContractError(
            f"DL3DV archive byte count mismatch for {record['path']}: "
            f"expected {record['size']}, got {actual_size}"
        )
    digest = sha256_file(path)
    if record["object_id_kind"] == "lfs_sha256" and digest != record["oid"]:
        raise DownloadContractError(
            f"DL3DV archive SHA256 mismatch for {record['path']}"
        )
    return {
        "scene": record["scene"],
        "path": record["path"],
        "size": actual_size,
        "sha256": digest,
        "source_object_id": record["oid"],
        "source_object_id_kind": record["object_id_kind"],
    }


def download_selected_archives(
    records: list[dict[str, Any]], *, archive_root: Path, download_file: DownloadFile
) -> list[dict[str, Any]]:
    """Download each selected archive once, then verify its advertised bytes."""
    archive_root = archive_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    verified = []
    for offset, record in enumerate(records, start=1):
        destination = _archive_destination(archive_root, record["path"])
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            returned = Path(download_file(record["path"])).resolve()
            if returned != destination:
                # ``hf_hub_download(local_dir=...)`` should materialize the exact
                # repository-relative path.  Refuse cache paths or renamed files.
                raise DownloadContractError(
                    f"downloader returned an unexpected archive path for {record['path']}"
                )
        verified.append(_verify_archive(destination, record))
        print(f"DL3DV calibration archive: {offset}/{len(records)}", flush=True)
    return verified


def _safe_zip_member(info: zipfile.ZipInfo) -> PurePosixPath:
    pure = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if (
        not info.filename
        or "\x00" in info.filename
        or pure.is_absolute()
        or ".." in pure.parts
        or not pure.parts
        or stat.S_IFMT(mode) == stat.S_IFLNK
    ):
        raise DownloadContractError(f"unsafe member in DL3DV archive: {info.filename}")
    return pure


def _single_nerfstudio_root(extracted: Path, scene: str) -> Path:
    matches = sorted(extracted.rglob("nerfstudio/transforms.json"))
    if len(matches) != 1:
        raise DownloadContractError(
            f"DL3DV archive for {scene} must contain exactly one nerfstudio/transforms.json"
        )
    nerfstudio = matches[0].parent
    if nerfstudio.name != "nerfstudio" or not nerfstudio.is_dir():
        raise DownloadContractError(f"DL3DV archive for {scene} has an invalid nerfstudio root")
    return nerfstudio


def extract_calibration_archives(
    records: list[dict[str, Any]], *, archive_root: Path, prepared_root: Path
) -> dict[str, Any]:
    """Safely normalize one frozen archive selection to ``<scene>/nerfstudio``."""
    prepared_root = prepared_root.resolve()
    if prepared_root.exists():
        raise FileExistsError(f"DL3DV calibration prepared root already exists: {prepared_root}")
    prepared_root.mkdir(parents=True)
    try:
        for offset, record in enumerate(records, start=1):
            archive_path = _archive_destination(archive_root.resolve(), record["path"])
            _verify_archive(archive_path, record)
            with tempfile.TemporaryDirectory(prefix="scarf-dl3dv-", dir=prepared_root) as temp:
                temporary = Path(temp)
                with zipfile.ZipFile(archive_path) as archive:
                    members = [info for info in archive.infolist() if not info.is_dir()]
                    if not members:
                        raise DownloadContractError(
                            f"DL3DV archive has no file members: {record['path']}"
                        )
                    for info in members:
                        pure = _safe_zip_member(info)
                        destination = (temporary / Path(*pure.parts)).resolve()
                        if temporary.resolve() not in destination.parents:
                            raise DownloadContractError("DL3DV archive member escapes extraction root")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info, "r") as source, destination.open("xb") as target:
                            shutil.copyfileobj(source, target, length=1024 * 1024)
                nerfstudio = _single_nerfstudio_root(temporary, record["scene"])
                target = prepared_root / record["scene"] / "nerfstudio"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(nerfstudio), str(target))
            print(f"DL3DV calibration preparation: {offset}/{len(records)}", flush=True)
        return build_dataset_manifest(
            prepared_root,
            "dl3dv-calibration-raw",
            SOURCE_URL,
            "archive-plan-v1",
        )
    except BaseException:
        # Keep the original failing archive and any verified downloads.  The
        # partially unpacked tree is never a valid calibration input.
        shutil.rmtree(prepared_root, ignore_errors=True)
        raise


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing DL3DV record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_preparation_record(
    *,
    output_root: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    downloaded: list[dict[str, Any]],
    prepared_manifest: Mapping[str, Any] | None,
) -> Path:
    record = {
        "schema_version": "1.0",
        "kind": "dl3dv_calibration_archive_preparation",
        "status": "PREPARED" if prepared_manifest is not None else "DOWNLOADED",
        "source": plan["source"],
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "downloaded_archives": downloaded,
        "downloaded_archives_sha256": canonical_sha256(downloaded),
        "prepared": (
            {
                "root": str((output_root / "prepared").resolve()),
                "tree_sha256": prepared_manifest["tree_sha256"],
                "file_count": prepared_manifest["file_count"],
                "scene_count": len(_selected_archives(plan, ALL_SUBSETS)),
                "split_scene_sets": {
                    subset: {
                        "scene_count": len(_selected_archives(plan, subset)),
                        "scene_set_sha256": canonical_sha256(
                            sorted(
                                item["scene"]
                                for item in _selected_archives(plan, subset)
                            )
                        ),
                    }
                    for subset in (CALIBRATION_SUBSET, HOLDOUT_SUBSET)
                },
            }
            if prepared_manifest is not None
            else None
        ),
    }
    destination = output_root / ".scarf-dl3dv-calibration-source.json"
    _write_json_new(destination, record)
    return destination


def _hub_clients() -> tuple[Any, str, Callable[..., str]]:
    try:
        from huggingface_hub import HfApi, get_token, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install the locked profile first") from exc
    token = get_token()
    if token is None:
        raise RuntimeError(
            "DL3DV-ALL-480P is gated. Accept its Hugging Face terms and run `hf auth login`."
        )
    return HfApi(token=token), token, hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-plan", type=Path)
    action.add_argument("--plan", type=Path)
    parser.add_argument("--evaluation-index", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--subset",
        choices=(CALIBRATION_SUBSET, HOLDOUT_SUBSET, ALL_SUBSETS),
        # A calibration result is valid only when its 24-scene train set and
        # independent eight-scene holdout are both fetched and prepared.
        default=ALL_SUBSETS,
    )
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    try:
        api, token, download = _hub_clients()
        tree = list_archive_tree(api, revision=args.revision)
        if args.write_plan is not None:
            plan = build_source_plan(
                tree, evaluation_index=args.evaluation_index, revision=args.revision
            )
            _write_json_new(args.write_plan, plan)
            print(args.write_plan)
            return 0

        assert args.plan is not None
        plan = load_and_verify_plan(
            args.plan,
            tree=tree,
            evaluation_index=args.evaluation_index,
            revision=args.revision,
        )
        output_root = args.output_root.resolve()
        selected = _selected_archives(plan, args.subset)
        archive_root = output_root / "archives"

        def fetch(relative: str) -> str:
            return str(
                download(
                    repo_id=SOURCE_REPOSITORY,
                    filename=relative,
                    repo_type=REPO_TYPE,
                    revision=args.revision,
                    token=token,
                    local_dir=archive_root,
                )
            )

        downloaded = download_selected_archives(
            selected, archive_root=archive_root, download_file=fetch
        )
        prepared_manifest = None
        if not args.no_extract and args.subset == ALL_SUBSETS:
            all_selected = _selected_archives(plan, ALL_SUBSETS)
            prepared_manifest = extract_calibration_archives(
                all_selected,
                archive_root=archive_root,
                prepared_root=output_root / "prepared",
            )
            manifest_path = output_root / "prepared" / ".scarf-manifest.json"
            manifest_path.write_text(
                json.dumps(prepared_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        record = write_preparation_record(
            output_root=output_root,
            plan_path=args.plan,
            plan=plan,
            downloaded=downloaded,
            prepared_manifest=prepared_manifest,
        )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
