import json
import hashlib

import pytest


def test_runtime_asset_preflight_rejects_missing_files(tmp_path):
    from scripts.runtime_assets import validate_runtime_assets

    manifest = tmp_path / "runtime_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "kind": "file",
                        "profiles": ["classic"],
                        "name": "metric weights",
                        "path": "assets/weights.pth",
                        "size": 4,
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing or truncated"):
        validate_runtime_assets("mvsplat", root=tmp_path, manifest_path=manifest)


def test_runtime_assets_can_be_scoped_to_one_model(tmp_path):
    from scripts.runtime_assets import validate_runtime_assets

    shared = tmp_path / "assets/shared.pth"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"shared")
    manifest = tmp_path / "runtime_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "kind": "file",
                        "profiles": ["classic"],
                        "name": "shared metric weights",
                        "path": "assets/shared.pth",
                        "size": shared.stat().st_size,
                        "sha256": hashlib.sha256(b"shared").hexdigest(),
                    },
                    {
                        "kind": "file",
                        "profiles": ["classic"],
                        "models": ["transplat"],
                        "name": "TranSplat backbone weights",
                        "path": "transplat/checkpoints/backbone.pth",
                        "size": 4,
                        "sha256": "0" * 64,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    records = validate_runtime_assets(
        "mvsplat", root=tmp_path, manifest_path=manifest
    )
    assert set(records) == {"shared metric weights"}

    with pytest.raises(FileNotFoundError, match="TranSplat|missing or truncated"):
        validate_runtime_assets("transplat", root=tmp_path, manifest_path=manifest)
