import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def _record(scene: int, payload: bytes) -> dict:
    name = f"{scene:064x}"
    return {
        "scene": name,
        "path": f"{scene // 1000 + 1}K/{name}.zip",
        "size": len(payload),
        "oid": hashlib.sha256(payload).hexdigest(),
    }


def _zip_bytes(tmp_path: Path, scene: str) -> bytes:
    archive = tmp_path / f"{scene}.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(f"wrapped/{scene}/nerfstudio/transforms.json", '{"frames": []}')
        output.writestr(f"wrapped/{scene}/nerfstudio/images_8/frame_00000.jpg", b"image")
    return archive.read_bytes()


def _plan(tree: list[dict], evaluation_path: Path) -> dict:
    from data.download_dl3dv_calibration import build_source_plan

    return build_source_plan(tree, evaluation_index=evaluation_path, revision="b" * 40)


def test_tree_listing_requires_lfs_sha256_and_uses_only_zip_archives():
    from data.download_dl3dv_calibration import DownloadContractError, list_archive_tree

    valid = SimpleNamespace(
        path="1K/" + "a" * 64 + ".zip",
        size=123,
        lfs=SimpleNamespace(sha256="f" * 64),
    )
    api = SimpleNamespace(
        list_repo_tree=lambda **kwargs: [SimpleNamespace(path="README.md"), valid]
    )
    assert list_archive_tree(api, revision="b" * 40) == [
        {
            "path": valid.path,
            "size": 123,
            "oid": "f" * 64,
            "object_id_kind": "lfs_sha256",
        }
    ]

    api = SimpleNamespace(
        list_repo_tree=lambda **kwargs: [
            SimpleNamespace(path=valid.path, size=123, lfs=SimpleNamespace(sha256="x"))
        ]
    )
    with pytest.raises(DownloadContractError, match="stable object id"):
        list_archive_tree(api, revision="b" * 40)


def test_plan_verification_rejects_a_changed_archive_tree(tmp_path: Path):
    from data.download_dl3dv_calibration import DownloadContractError, load_and_verify_plan

    index = tmp_path / "index.json"
    index.write_text(json.dumps({f"{item:064x}": {} for item in range(140)}))
    tree = [_record(item, f"payload-{item}".encode()) for item in range(180)]
    plan = _plan(tree, index)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    changed = [dict(item) for item in tree]
    changed[-1]["oid"] = "e" * 64

    with pytest.raises(DownloadContractError, match="does not match"):
        load_and_verify_plan(
            path,
            tree=changed,
            evaluation_index=index,
            revision="b" * 40,
        )


def test_selected_download_and_safe_extraction_preserve_archive_provenance(tmp_path: Path):
    from data.download_dl3dv_calibration import (
        _selected_archives,
        download_selected_archives,
        extract_calibration_archives,
    )

    index = tmp_path / "index.json"
    index.write_text(json.dumps({f"{item:064x}": {} for item in range(140)}))
    payloads = {}
    tree = []
    for item in range(180):
        scene = f"{item:064x}"
        payload = _zip_bytes(tmp_path, scene)
        payloads[f"{item // 1000 + 1}K/{scene}.zip"] = payload
        tree.append(_record(item, payload))
    plan = _plan(tree, index)
    calibration = _selected_archives(plan, "calibration")
    archive_root = tmp_path / "archives"

    def download(relative: str) -> str:
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[relative])
        return str(destination)

    downloaded = download_selected_archives(
        calibration, archive_root=archive_root, download_file=download
    )
    assert len(downloaded) == 24
    assert all(
        item["sha256"] == hashlib.sha256(payloads[item["path"]]).hexdigest()
        for item in downloaded
    )

    prepared = extract_calibration_archives(
        calibration, archive_root=archive_root, prepared_root=tmp_path / "prepared"
    )
    assert prepared["dataset"] == "dl3dv-calibration-raw"
    assert len(list((tmp_path / "prepared").glob("*/nerfstudio/transforms.json"))) == 24


def test_extraction_rejects_path_traversal(tmp_path: Path):
    from data.download_dl3dv_calibration import DownloadContractError, extract_calibration_archives

    scene = "a" * 64
    relative = f"1K/{scene}.zip"
    archive_root = tmp_path / "archives"
    path = archive_root / relative
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape", b"bad")
    payload = path.read_bytes()
    record = {
        "scene": scene,
        "path": relative,
        "size": len(payload),
        "oid": hashlib.sha256(payload).hexdigest(),
        "object_id_kind": "lfs_sha256",
    }

    with pytest.raises(DownloadContractError, match="unsafe member"):
        extract_calibration_archives(
            [record], archive_root=archive_root, prepared_root=tmp_path / "prepared"
        )
    assert not (tmp_path / "escape").exists()
