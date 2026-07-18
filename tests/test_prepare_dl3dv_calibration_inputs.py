import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image


torch = pytest.importorskip("torch", reason="DL3DV calibration sidecars require torch")


def _archive(scene: int) -> dict:
    name = f"{scene:064x}"
    return {"scene": name, "path": f"1K/{name}.zip", "size": 1, "oid": name}


def _scene(root: Path, scene: str) -> None:
    nerfstudio = root / scene / "nerfstudio"
    frames = []
    for view in range(5):
        name = f"frame_{view:05d}.jpg"
        image_path = nerfstudio / "images_8" / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (480, 270), (view, 10, 20)).save(image_path, format="JPEG")
        frames.append(
            {
                "file_path": f"images_8/{name}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, float(view)],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    (nerfstudio / "transforms.json").write_text(
        json.dumps(
            {
                "h": 2160,
                "w": 3840,
                "fl_x": 1920.0,
                "fl_y": 1080.0,
                "cx": 1920.0,
                "cy": 1080.0,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    from data.build_manifest import build
    from data.plan_dl3dv_calibration import build_plan

    raw = tmp_path / "raw"
    evaluation = [f"{item:064x}" for item in range(140)]
    tree = [_archive(item) for item in range(180)]
    plan = build_plan(
        tree,
        evaluation_scenes=evaluation,
        evaluation_index_sha256="a" * 64,
        revision="b" * 40,
    )
    for record in [
        *plan["selection"]["calibration_train"],
        *plan["selection"]["calibration_holdout"],
    ]:
        _scene(raw, record["scene"])
    manifest = build(raw, "dl3dv-calibration-raw", "fixture", "fixture")
    manifest_path = raw / ".scarf-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    preparation = {
        "kind": "dl3dv_calibration_archive_preparation",
        "status": "PREPARED",
        "source": plan["source"],
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "downloaded_archives": [
            {
                "scene": record["scene"],
                "path": record["path"],
                "size": record["size"],
                "sha256": record["oid"],
                "source_object_id": record["oid"],
                "source_object_id_kind": record["object_id_kind"],
            }
            for record in [
                *plan["selection"]["calibration_train"],
                *plan["selection"]["calibration_holdout"],
            ]
        ],
        "prepared": {
            "tree_sha256": manifest["tree_sha256"],
            "split_scene_sets": {
                split: {
                    "scene_count": len(plan["selection"][split]),
                    "scene_set_sha256": hashlib.sha256(
                        json.dumps(
                            sorted(record["scene"] for record in plan["selection"][split]),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                }
                for split in ("calibration_train", "calibration_holdout")
            },
        },
    }
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(json.dumps(preparation), encoding="utf-8")
    return raw, plan_path, preparation_path


def test_prepare_dl3dv_calibration_inputs_writes_only_context_images(tmp_path: Path):
    from data.prepare_dl3dv_calibration_inputs import prepare_inputs
    from scripts.calibration_inputs import load_target_free_record, validate_target_free_input_root

    raw, plan, preparation = _fixture(tmp_path)
    manifest = prepare_inputs(
        raw,
        plan_path=plan,
        preparation_path=preparation,
        output_dir=tmp_path / "protocol",
    )

    assert manifest["kind"] == "dl3dv_calibration_protocol"
    assert manifest["calibration_scene_count"] == 24
    train = manifest["datasets"]["dl3dv"]["splits"]["calibration_train"]
    holdout = manifest["datasets"]["dl3dv"]["splits"]["calibration_holdout"]
    re10k = train["representations"]["re10k"]
    native = train["representations"]["native"]
    assert re10k["target_rgb_included"] is False
    assert native["target_shape"] == [270, 480]
    assert re10k["target_shape"] == [360, 640]
    assert native["calibration_input_root"] == "inputs/calibration_train/native"
    assert re10k["calibration_input_root"] == "inputs/calibration_train/re10k"
    assert holdout["scene_count"] == 8
    sidecar = tmp_path / "protocol" / re10k["calibration_input_root"]
    assert validate_target_free_input_root(sidecar, "dl3dv")["target_rgb_accessed"] is False
    scene = next(iter(json.loads((tmp_path / "protocol/calibration_train-re10k.json").read_text())))
    record = load_target_free_record(sidecar, scene)
    assert len(record["context_images"]) == 2
    assert all(key not in record for key in ("images", "target_images", "target_rgb"))


def test_prepare_dl3dv_calibration_inputs_rejects_extra_or_eval_scenes(tmp_path: Path):
    from data.prepare_dl3dv_calibration_inputs import PreparationContractError, prepare_inputs

    raw, plan, preparation = _fixture(tmp_path)
    _scene(raw, f"{1:064x}")
    with pytest.raises(PreparationContractError, match="scene set does not match"):
        prepare_inputs(
            raw,
            plan_path=plan,
            preparation_path=preparation,
            output_dir=tmp_path / "protocol",
        )
