#!/usr/bin/env python3
"""Download the pinned DL3DV benchmark scenes without walking its full tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping


REPO_ID = "DL3DV/DL3DV-10K-Benchmark"
REPO_TYPE = "dataset"
SOURCE_URL = f"https://huggingface.co/datasets/{REPO_ID}"
FILELIST_PATH = ".cache/filelist.bin"
METADATA_PATH = "benchmark-meta.csv"
EXPECTED_SCENE_COUNT = 140
IMAGE_SUBDIRS = ("images_2", "images_4", "images_8")
NATIVE_TARGET_SHAPE = (270, 480)
RE10K_SOURCE_SHAPE = (540, 960)
IMAGE_SCALES = {"images_2": 2, "images_4": 4, "images_8": 8}


class DownloadContractError(ValueError):
    """Raised when upstream metadata cannot satisfy the fixed protocol."""


class _BuiltinsOnlyUnpickler(pickle.Unpickler):
    """The upstream file list is data-only; reject executable pickle globals."""

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(f"disallowed pickle global: {module}.{name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_protocol_keys(index_path: Path) -> list[str]:
    source = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise DownloadContractError("DL3DV evaluation index must be a JSON object")
    keys = sorted(source)
    if len(keys) != EXPECTED_SCENE_COUNT or any(not isinstance(key, str) or not key for key in keys):
        raise DownloadContractError(
            f"DL3DV evaluation index must contain {EXPECTED_SCENE_COUNT} non-empty scene keys"
        )
    return keys


def load_benchmark_keys(metadata_path: Path) -> set[str]:
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = {row.get("hash", "") for row in rows}
    if len(keys) != EXPECTED_SCENE_COUNT or "" in keys:
        raise DownloadContractError(
            f"benchmark metadata must contain {EXPECTED_SCENE_COUNT} distinct scene hashes"
        )
    return keys


def load_filelist(path: Path) -> dict[str, list[str]]:
    with path.open("rb") as stream:
        source = _BuiltinsOnlyUnpickler(stream).load()
    if not isinstance(source, dict):
        raise DownloadContractError("upstream file list is not a dictionary")
    normalized: dict[str, list[str]] = {}
    for key, paths in source.items():
        if not isinstance(key, str) or not isinstance(paths, list):
            raise DownloadContractError("upstream file list has an invalid scene entry")
        if not all(isinstance(item, str) for item in paths):
            raise DownloadContractError(f"upstream file list has non-string paths for {key}")
        normalized[key] = paths
    return normalized


def _required_kind(path: str, scene: str) -> str | None:
    scene_prefix = f"{scene}/"
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or not path.startswith(scene_prefix):
        raise DownloadContractError(f"unsafe or out-of-scene upstream path: {path}")
    suffix = path[len(scene_prefix) :]
    if not suffix.startswith("nerfstudio/"):
        return None
    suffix = suffix[len("nerfstudio/") :]
    if suffix == "transforms.json":
        return "transforms"
    for image_subdir in IMAGE_SUBDIRS:
        if suffix.startswith(f"{image_subdir}/"):
            return image_subdir
    return None


def _scene_paths_for_kind(
    filelist: Mapping[str, list[str]], scene: str, kind: str
) -> list[str]:
    if kind != "transforms" and kind not in IMAGE_SUBDIRS:
        raise DownloadContractError(f"unsupported required upstream tree: {kind}")
    try:
        entries = filelist[scene]
    except KeyError as exc:
        raise DownloadContractError(f"protocol scene is absent from upstream file list: {scene}") from exc
    paths = [path for path in entries if _required_kind(path, scene) == kind]
    if not paths:
        raise DownloadContractError(
            f"protocol scene has no required upstream files ({kind}): {scene}"
        )
    if kind == "transforms" and len(paths) != 1:
        raise DownloadContractError(f"protocol scene has multiple transforms files: {scene}")
    return paths


def select_paths_for_kinds(
    filelist: Mapping[str, list[str]],
    scene_keys: Iterable[str],
    kinds_by_scene: Mapping[str, Iterable[str]],
) -> list[str]:
    selected: list[str] = []
    for scene in scene_keys:
        kinds = tuple(kinds_by_scene.get(scene, ()))
        if not kinds or len(set(kinds)) != len(kinds):
            raise DownloadContractError(f"protocol scene has invalid required trees: {scene}")
        for kind in kinds:
            selected.extend(_scene_paths_for_kind(filelist, scene, kind))
    if len(selected) != len(set(selected)):
        raise DownloadContractError("upstream file list repeats a required path")
    return selected


def _integer_dimension(value: Any, field: str, scene: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DownloadContractError(f"{scene} transforms has invalid {field}")
    integer = int(value)
    if integer <= 0 or integer != value:
        raise DownloadContractError(f"{scene} transforms has invalid {field}")
    return integer


def build_scene_source_plans(
    filelist: Mapping[str, list[str]],
    scene_keys: Iterable[str],
    raw_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Choose the exact-resolution source tree for each pinned scene."""

    plans: dict[str, dict[str, dict[str, Any]]] = {}
    for scene in scene_keys:
        transform_path = raw_dir / _scene_paths_for_kind(filelist, scene, "transforms")[0]
        metadata = json.loads(transform_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise DownloadContractError(f"{scene} transforms is not an object")
        height = _integer_dimension(metadata.get("h"), "h", scene)
        width = _integer_dimension(metadata.get("w"), "w", scene)
        shapes = {
            subdir: (height // scale, width // scale)
            for subdir, scale in IMAGE_SCALES.items()
            if height % scale == 0 and width % scale == 0
        }

        def choose(target_shape: tuple[int, int], label: str) -> str:
            matches = [subdir for subdir, shape in shapes.items() if shape == target_shape]
            if len(matches) != 1:
                raise DownloadContractError(
                    f"{scene} has no unique {label} source at "
                    f"{target_shape[0]}x{target_shape[1]}"
                )
            return matches[0]

        native_subdir = choose(NATIVE_TARGET_SHAPE, "native")
        re10k_subdir = choose(RE10K_SOURCE_SHAPE, "Re10K")
        _scene_paths_for_kind(filelist, scene, native_subdir)
        _scene_paths_for_kind(filelist, scene, re10k_subdir)
        plans[scene] = {
            "native": {
                "image_subdir": native_subdir,
                "source_image_shape": list(NATIVE_TARGET_SHAPE),
            },
            "re10k": {
                "image_subdir": re10k_subdir,
                "source_image_shape": list(RE10K_SOURCE_SHAPE),
            },
        }
    return plans


def select_planned_paths(
    filelist: Mapping[str, list[str]],
    scene_keys: Iterable[str],
    scene_source_plans: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[str]:
    kinds_by_scene = {}
    for scene in scene_keys:
        try:
            plan = scene_source_plans[scene]
            native_subdir = plan["native"]["image_subdir"]
            re10k_subdir = plan["re10k"]["image_subdir"]
        except (KeyError, TypeError) as exc:
            raise DownloadContractError(f"invalid source plan for scene: {scene}") from exc
        kinds_by_scene[scene] = ("transforms", native_subdir, re10k_subdir)
    return select_paths_for_kinds(filelist, scene_keys, kinds_by_scene)


def build_source_record(
    *,
    revision: str,
    index_path: Path,
    metadata_path: Path,
    filelist_path: Path,
    scene_keys: list[str],
    selected_paths: list[str],
    scene_source_plans: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "source": SOURCE_URL,
        "revision": revision,
        "source_index": str(index_path),
        "source_index_sha256": sha256_file(index_path),
        "benchmark_metadata_sha256": sha256_file(metadata_path),
        "filelist_sha256": sha256_file(filelist_path),
        "selected_scene_count": len(scene_keys),
        "selected_scenes_sha256": canonical_sha256(scene_keys),
        "selected_file_count": len(selected_paths),
        "selected_files_sha256": canonical_sha256(selected_paths),
        "scene_source_plans": scene_source_plans,
        "scene_source_plans_sha256": canonical_sha256(scene_source_plans),
    }


DownloadFile = Callable[[str], str]


def existing_local_file(output_dir: Path, path: str) -> Path | None:
    """Return a completed pinned file already materialized under ``output_dir``."""
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise DownloadContractError(f"unsafe local download path: {path}")
    candidate = output_dir.joinpath(*pure.parts)
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate
    return None


def download_required_paths(
    paths: list[str],
    *,
    download_file: DownloadFile,
    workers: int,
    retries: int,
) -> None:
    if workers < 1 or retries < 1:
        raise ValueError("workers and retries must be positive")

    def download_one(path: str) -> str:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                download_file(path)
                return path
            except Exception as exc:  # Network libraries use several exception types.
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(min(30, 2**attempt))
        assert last_error is not None
        raise RuntimeError(f"failed to download {path}") from last_error

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, path): path for path in paths}
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == len(paths) or completed % 250 == 0:
                print(f"DL3DV download: {completed}/{len(paths)} files", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    try:
        from huggingface_hub import get_token, hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required; install the locked profile first") from exc

    token = get_token()
    if token is None:
        raise SystemExit(
            "DL3DV-10K-Benchmark is gated. Accept its Hugging Face access terms and "
            "run `hf auth login` before this command."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def fetch(path: str) -> str:
        local_file = existing_local_file(output_dir, path)
        if local_file is not None:
            return str(local_file)
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=path,
            repo_type=REPO_TYPE,
            revision=args.revision,
            local_dir=output_dir,
            token=token,
        )

    index_path = args.index
    metadata_path = Path(fetch(METADATA_PATH))
    filelist_path = Path(fetch(FILELIST_PATH))
    scene_keys = load_protocol_keys(index_path)
    benchmark_keys = load_benchmark_keys(metadata_path)
    if set(scene_keys) != benchmark_keys:
        raise DownloadContractError("fixed DL3DV index and benchmark metadata do not name the same scenes")
    filelist = load_filelist(filelist_path)
    transform_paths = select_paths_for_kinds(
        filelist,
        scene_keys,
        {scene: ("transforms",) for scene in scene_keys},
    )
    download_required_paths(
        transform_paths,
        download_file=fetch,
        workers=args.workers,
        retries=args.retries,
    )
    scene_source_plans = build_scene_source_plans(filelist, scene_keys, output_dir)
    selected_paths = select_planned_paths(filelist, scene_keys, scene_source_plans)
    transform_path_set = set(transform_paths)
    image_paths = [path for path in selected_paths if path not in transform_path_set]
    download_required_paths(
        image_paths,
        download_file=fetch,
        workers=args.workers,
        retries=args.retries,
    )
    record = build_source_record(
        revision=args.revision,
        index_path=index_path,
        metadata_path=metadata_path,
        filelist_path=filelist_path,
        scene_keys=scene_keys,
        selected_paths=selected_paths,
        scene_source_plans=scene_source_plans,
    )
    destination = output_dir / ".scarf-dl3dv-source.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
