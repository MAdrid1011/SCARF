import hashlib
import json
import pickle
import sys
from pathlib import Path

import pytest


class _ArchiveTorch:
    @staticmethod
    def save(value, path):
        Path(path).write_bytes(pickle.dumps(value))

    @staticmethod
    def load(path, map_location=None):
        return pickle.loads(Path(path).read_bytes())

    @staticmethod
    def is_tensor(value):
        return False


@pytest.fixture(autouse=True)
def _archive_torch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", _ArchiveTorch)


def _prepared_acid_root(tmp_path: Path) -> tuple[Path, list[str]]:
    from data.build_manifest import build
    from scripts.calibration_contract import canonical_sha256, hash_ranked_scene_names

    root = tmp_path / "prepared" / "acid"
    train = root / "train"
    train.mkdir(parents=True)
    full_index = {f"scene-{ordinal:03d}": "000000.torch" for ordinal in range(40)}
    selected = hash_ranked_scene_names(list(full_index), dataset="acid", count=32)
    selected_index = {scene: full_index[scene] for scene in selected}
    (train / "full-index.json").write_text(
        json.dumps(full_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (train / "index.json").write_text(
        json.dumps(selected_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    examples = [
        {
            "key": scene,
            "cameras": [[float(value) for value in range(18)] for _ in range(5)],
            "images": [
                b"context-0",
                b"forbidden-target-1",
                b"forbidden-target-2",
                b"forbidden-target-3",
                b"context-4",
            ],
        }
        for scene in selected
    ]
    _ArchiveTorch.save(examples, train / "000000.torch")
    source = {
        "schema_version": "1.0",
        "dataset": "acid",
        "archive": "acid.zip",
        "archive_sha256": "a" * 64,
        "source_url": "https://example.invalid/acid.zip",
        "full_train_index_sha256": hashlib.sha256(
            (train / "full-index.json").read_bytes()
        ).hexdigest(),
        "selection_domain": "SCARF-AE-calibration-v1",
        "selection_count": 32,
        "selected_scenes": selected,
        "selected_index_sha256": canonical_sha256(selected_index),
        "evaluation_disjoint": True,
    }
    (root / ".scarf-calibration-source.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = build(root, "acid", "fixture", "fixture")
    (root / ".scarf-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root, selected


def _evaluation_rows() -> list[dict]:
    return [
        {
            "scene": f"evaluation-{ordinal:03d}",
            "context_indices": [0, 4],
            "target_indices": [1, 2, 3],
        }
        for ordinal in range(4)
    ]


def _evaluation_contract(rows: list[dict]) -> dict:
    from scripts.calibration_contract import canonical_sha256

    return {
        "index_path": "transplat/assets/evaluation_index_acid.json",
        "source_index_sha256": "b" * 64,
        "sample_selection_sha256": "c" * 64,
        "sample_count": len(rows),
        "source_entry_count": len(rows),
        "null_entry_count": 0,
        "model_pairs": ["transplat/acid", "mvsplat/acid", "depthsplat/acid"],
        "evaluation_scene_set_sha256": canonical_sha256(
            sorted(row["scene"] for row in rows)
        ),
    }


def _plan(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    from data.plan_acid_joint_calibration import build_plan

    root, _ = _prepared_acid_root(tmp_path)
    rows = _evaluation_rows()
    return root, build_plan(root, evaluation_rows=rows, evaluation_contract=_evaluation_contract(rows)), rows


def test_acid_joint_plan_is_deterministic_disjoint_and_author_side_only(tmp_path: Path):
    from data.plan_acid_joint_calibration import deterministic_partition, validate_plan

    root, plan, rows = _plan(tmp_path)
    identity = validate_plan(plan, prepared_root=root, evaluation_rows=rows)
    partition = plan["partition"]

    assert identity["train_scene_count"] == 24
    assert identity["holdout_scene_count"] == 8
    assert identity["evaluation_disjoint"] is True
    assert plan["author_side_prerequisite"] is True
    assert plan["paper_result_eligible"] is False
    assert plan["mechanism_config_write_allowed"] is False
    assert plan["dl3dv_quality_gate_authorized"] is False
    assert plan["context_only_sidecars"]["target_camera_metadata_included"] is False
    assert plan["teacher_runtime_isolation"]["teacher_files_runtime_accessible"] is False
    assert plan["teacher_runtime_isolation"]["runtime_teacher_path"] is None
    assert not (
        set(partition["calibration_train"]["scenes"])
        & set(partition["calibration_holdout"]["scenes"])
    )
    reversed_partition = deterministic_partition(
        list(reversed(
            [
                *partition["calibration_train"]["scenes"],
                *partition["calibration_holdout"]["scenes"],
            ]
        )),
        evaluation_scenes=[row["scene"] for row in rows],
    )
    assert reversed_partition == partition


def test_acid_joint_plan_refuses_evaluation_overlap(tmp_path: Path):
    from data.plan_acid_joint_calibration import build_plan

    root, selected = _prepared_acid_root(tmp_path)
    rows = _evaluation_rows()
    rows[0]["scene"] = selected[0]

    with pytest.raises(ValueError, match="overlap"):
        build_plan(root, evaluation_rows=rows, evaluation_contract=_evaluation_contract(rows))


def test_context_only_materialization_removes_target_camera_and_rgb(tmp_path: Path):
    from data.plan_acid_joint_calibration import (
        materialize_context_only_sidecars,
        validate_materialization,
    )

    root, plan, rows = _plan(tmp_path)
    output = tmp_path / "sidecars"
    materialized = materialize_context_only_sidecars(
        root, plan=plan, output_root=output, evaluation_rows=rows
    )
    identity = validate_materialization(output, plan=plan, evaluation_rows=rows)
    first_scene = plan["partition"]["calibration_train"]["scenes"][0]
    index = json.loads(
        (output / "inputs" / "calibration_train" / "test" / "index.json").read_text(
            encoding="utf-8"
        )
    )
    payload = _ArchiveTorch.load(
        output / "inputs" / "calibration_train" / "test" / index[first_scene],
        map_location="cpu",
    )[0]

    assert materialized["paper_result_eligible"] is False
    assert identity["paper_result_eligible"] is False
    assert set(payload) == {
        "schema_version",
        "kind",
        "key",
        "source_view_count",
        "context_indices",
        "context_cameras",
        "context_images",
    }
    assert payload["context_indices"] == [0, 4]
    assert len(payload["context_cameras"]) == 2
    assert all(len(camera) == 18 for camera in payload["context_cameras"])
    assert payload["context_images"] == [b"context-0", b"context-4"]
    assert not any("target" in key or "teacher" in key for key in payload)


def test_context_only_validator_rejects_refreshed_target_metadata(tmp_path: Path):
    from data.build_manifest import build
    from data.plan_acid_joint_calibration import (
        materialize_context_only_sidecars,
        sha256_file,
        validate_context_only_sidecar,
    )
    from scripts.calibration_contract import canonical_sha256

    root, plan, rows = _plan(tmp_path)
    output = tmp_path / "sidecars"
    materialize_context_only_sidecars(
        root, plan=plan, output_root=output, evaluation_rows=rows
    )
    sidecar = output / "inputs" / "calibration_train"
    index_path = sidecar / "test" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    chunk_path = sidecar / "test" / index[plan["partition"]["calibration_train"]["scenes"][0]]
    chunk = _ArchiveTorch.load(chunk_path, map_location="cpu")
    chunk[0]["target_indices"] = [1, 2, 3]
    _ArchiveTorch.save(chunk, chunk_path)
    provenance_path = sidecar / ".scarf-acid-joint-input.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for item in provenance["opened_file_manifest"]:
        if item["path"] == f"test/{chunk_path.name}":
            item["sha256"] = sha256_file(chunk_path)
    provenance["opened_file_manifest_sha256"] = canonical_sha256(
        provenance["opened_file_manifest"]
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    original_manifest = json.loads((sidecar / ".scarf-manifest.json").read_text(encoding="utf-8"))
    refreshed = build(sidecar, original_manifest["dataset"], original_manifest["source"], original_manifest["revision"])
    (sidecar / ".scarf-manifest.json").write_text(
        json.dumps(refreshed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="forbidden fields"):
        validate_context_only_sidecar(sidecar, plan=plan, split="calibration_train")


def test_context_only_validator_rejects_refreshed_expected_results_artifact(tmp_path: Path):
    from data.build_manifest import build
    from data.plan_acid_joint_calibration import (
        materialize_context_only_sidecars,
        validate_context_only_sidecar,
    )

    root, plan, rows = _plan(tmp_path)
    output = tmp_path / "sidecars"
    materialize_context_only_sidecars(
        root, plan=plan, output_root=output, evaluation_rows=rows
    )
    sidecar = output / "inputs" / "calibration_holdout"
    (sidecar / "expected_results.json").write_text("{}\n", encoding="utf-8")
    original_manifest = json.loads((sidecar / ".scarf-manifest.json").read_text(encoding="utf-8"))
    refreshed = build(sidecar, original_manifest["dataset"], original_manifest["source"], original_manifest["revision"])
    (sidecar / ".scarf-manifest.json").write_text(
        json.dumps(refreshed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="forbidden artifact path"):
        validate_context_only_sidecar(sidecar, plan=plan, split="calibration_holdout")


def _access_audit(plan: dict, *, stage: str) -> dict:
    return {
        "schema_version": "1.0",
        "kind": "acid_joint_calibration_access_audit_v1",
        "plan_sha256": plan["plan_sha256"],
        "stage": stage,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "expected_results_accessed": False,
        "evaluation_scene_accessed": False,
        "descriptor_model_id_accessed": False,
        "descriptor_dataset_id_accessed": False,
        "teacher_files_opened": stage in {"train", "holdout"},
        "teacher_files_runtime_accessible": False,
        "runtime_teacher_path": None,
        "optimizer_executed": stage == "train",
        "asset_updated": stage == "train",
        "rerank_executed": False,
        "partition_reshuffled": False,
    }


def test_access_audit_bars_target_metadata_expected_results_and_runtime_teacher(tmp_path: Path):
    from data.plan_acid_joint_calibration import validate_execution_access_audit

    _, plan, _ = _plan(tmp_path)
    assert validate_execution_access_audit(_access_audit(plan, stage="runtime"), plan=plan)["status"] == "PASS"

    illegal_target = _access_audit(plan, stage="train")
    illegal_target["target_camera_metadata_accessed"] = True
    with pytest.raises(ValueError, match="target_camera_metadata_accessed"):
        validate_execution_access_audit(illegal_target, plan=plan)

    illegal_expected = _access_audit(plan, stage="holdout")
    illegal_expected["expected_results_accessed"] = True
    with pytest.raises(ValueError, match="expected_results_accessed"):
        validate_execution_access_audit(illegal_expected, plan=plan)

    illegal_runtime = _access_audit(plan, stage="runtime")
    illegal_runtime["teacher_files_opened"] = True
    with pytest.raises(ValueError, match="offline boundary"):
        validate_execution_access_audit(illegal_runtime, plan=plan)
