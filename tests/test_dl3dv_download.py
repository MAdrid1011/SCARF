import json
import pickle
from pathlib import Path

import pytest

from data.download_dl3dv_benchmark import (
    DownloadContractError,
    SOURCE_URL,
    build_scene_source_plans,
    build_source_record,
    existing_local_file,
    load_filelist,
    select_paths_for_kinds,
    select_planned_paths,
)


class _UnsafePickle:
    pass


def _filelist(scene: str) -> dict[str, list[str]]:
    prefix = f"{scene}/nerfstudio"
    return {
        scene: [
            f"{prefix}/transforms.json",
            f"{prefix}/images_2/frame_00000.png",
            f"{prefix}/images_4/frame_00000.png",
            f"{prefix}/images_8/frame_00000.png",
            f"{scene}/gaussian_splat/unused.bin",
        ]
    }


def test_selection_includes_only_converter_required_paths():
    selected = select_paths_for_kinds(
        _filelist("scene-a"),
        ["scene-a"],
        {"scene-a": ("transforms", "images_4", "images_8")},
    )
    assert selected == [
        "scene-a/nerfstudio/transforms.json",
        "scene-a/nerfstudio/images_4/frame_00000.png",
        "scene-a/nerfstudio/images_8/frame_00000.png",
    ]


@pytest.mark.parametrize("missing", ["transforms", "images_4", "images_8"])
def test_selection_rejects_scene_without_a_required_tree(missing: str):
    source = _filelist("scene-a")
    source["scene-a"] = [
        path
        for path in source["scene-a"]
        if missing not in path
        and not (missing == "transforms" and path.endswith("transforms.json"))
    ]
    with pytest.raises(DownloadContractError, match="no required upstream files"):
        select_paths_for_kinds(
            source,
            ["scene-a"],
            {"scene-a": ("transforms", "images_4", "images_8")},
        )


def test_selection_rejects_a_protocol_scene_absent_from_filelist():
    with pytest.raises(DownloadContractError, match="absent from upstream file list"):
        select_paths_for_kinds({}, ["scene-a"], {"scene-a": ("transforms",)})


def test_scene_source_plan_uses_exact_native_and_re10k_tiers(tmp_path: Path):
    filelist = {}
    for scene, shape in (("scene-high", (2160, 3840)), ("scene-low", (1080, 1920))):
        prefix = f"{scene}/nerfstudio"
        filelist[scene] = [
            f"{prefix}/transforms.json",
            f"{prefix}/images_2/frame_00000.png",
            f"{prefix}/images_4/frame_00000.png",
            f"{prefix}/images_8/frame_00000.png",
        ]
        metadata = tmp_path / f"{prefix}/transforms.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps({"h": shape[0], "w": shape[1]}), encoding="utf-8")

    scenes = ["scene-high", "scene-low"]
    plans = build_scene_source_plans(filelist, scenes, tmp_path)
    assert plans["scene-high"] == {
        "native": {"image_subdir": "images_8", "source_image_shape": [270, 480]},
        "re10k": {"image_subdir": "images_4", "source_image_shape": [540, 960]},
    }
    assert plans["scene-low"] == {
        "native": {"image_subdir": "images_4", "source_image_shape": [270, 480]},
        "re10k": {"image_subdir": "images_2", "source_image_shape": [540, 960]},
    }
    assert select_planned_paths(filelist, scenes, plans) == [
        "scene-high/nerfstudio/transforms.json",
        "scene-high/nerfstudio/images_8/frame_00000.png",
        "scene-high/nerfstudio/images_4/frame_00000.png",
        "scene-low/nerfstudio/transforms.json",
        "scene-low/nerfstudio/images_4/frame_00000.png",
        "scene-low/nerfstudio/images_2/frame_00000.png",
    ]


def test_safe_filelist_loader_rejects_pickle_globals(tmp_path: Path):
    path = tmp_path / "filelist.bin"
    path.write_bytes(pickle.dumps(_UnsafePickle()))
    with pytest.raises(pickle.UnpicklingError, match="disallowed pickle global"):
        load_filelist(path)


def test_source_record_binds_the_index_and_selected_paths(tmp_path: Path):
    index = tmp_path / "index.json"
    metadata = tmp_path / "benchmark-meta.csv"
    filelist = tmp_path / "filelist.bin"
    index.write_text('{"scene-a": {}}\n', encoding="utf-8")
    metadata.write_text("hash\nscene-a\n", encoding="utf-8")
    filelist.write_bytes(b"fixture")
    record = build_source_record(
        revision="revision",
        index_path=index,
        metadata_path=metadata,
        filelist_path=filelist,
        scene_keys=["scene-a"],
        selected_paths=["scene-a/nerfstudio/transforms.json"],
        scene_source_plans={
            "scene-a": {
                "native": {
                    "image_subdir": "images_8",
                    "source_image_shape": [270, 480],
                },
                "re10k": {
                    "image_subdir": "images_4",
                    "source_image_shape": [540, 960],
                },
            }
        },
    )
    assert record["source"] == SOURCE_URL
    assert record["selected_scene_count"] == 1
    assert record["selected_file_count"] == 1
    assert len(record["filelist_sha256"]) == 64
    assert len(record["scene_source_plans_sha256"]) == 64


def test_existing_local_file_reuses_only_nonempty_relative_files(tmp_path: Path):
    relative = "scene-a/nerfstudio/images_4/frame_00000.png"
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"image")

    assert existing_local_file(tmp_path, relative) == destination

    destination.write_bytes(b"")
    assert existing_local_file(tmp_path, relative) is None

    with pytest.raises(DownloadContractError, match="unsafe local download path"):
        existing_local_file(tmp_path, "../outside")
