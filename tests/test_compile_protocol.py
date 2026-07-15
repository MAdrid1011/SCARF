import copy
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifact/evaluation_protocol.json"


def test_committed_protocol_sources_compile_to_declared_selections():
    from scripts.compile_protocol import compile_protocol

    compiled = compile_protocol(PROTOCOL, ROOT)

    assert compiled["status"] == "PASS"
    assert compiled["datasets"]["re10k"] == {
        "source_entry_count": 7194,
        "null_entry_count": 720,
        "sample_count": 6474,
        "source_index_sha256": "fbcff9b4b13139227d809af45b84bb5a7a9cdb1c7a55bced85b6c66556a219cc",
        "sample_selection_sha256": "48de7c3120460095d4b706c147ad15a6f07abb76ff7461db552ee2afc4660b59",
    }
    assert compiled["datasets"]["acid"]["sample_count"] == 1595
    assert compiled["datasets"]["acid"]["sample_selection_sha256"] == (
        "03c61c18c3ca6d1fb16f8c92432ebfac3eaa668707fcaf15f9de7f4a018030ac"
    )
    assert compiled["datasets"]["dl3dv"]["sample_count"] == 140


def test_protocol_compiler_rejects_source_hash_drift(tmp_path: Path):
    from scripts.compile_protocol import compile_protocol

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["pairs"]["mvsplat/re10k"]["source_index_sha256"] = "0" * 64
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="source index SHA256"):
        compile_protocol(path, ROOT)


def test_protocol_compiler_rejects_invalid_or_duplicate_views(tmp_path: Path):
    from scripts.compile_protocol import canonicalize_index

    index = {
        "scene-a": {"context": [0, 0], "target": [1]},
        "scene-b": None,
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="duplicate context"):
        canonicalize_index(path, digest)


def test_protocol_compiler_preserves_source_and_execution_ordinals(tmp_path: Path):
    from scripts.compile_protocol import canonicalize_index

    index = {
        "missing": None,
        "scene-a": {"context": [0, 2], "target": [1]},
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    rows, _ = canonicalize_index(path, digest)
    assert rows[0]["sample_index"] == 1
    assert rows[0]["execution_index"] == 0


def test_canonicalizer_separates_stable_source_order_from_dataset_execution_order(
    tmp_path: Path,
):
    from scripts.compile_protocol import canonicalize_index

    path = tmp_path / "index.json"
    path.write_text(
        json.dumps(
            {
                "scene-a": {"context": [0, 2], "target": [1]},
                "scene-b": {"context": [3, 5], "target": [4]},
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    source_rows, source_summary = canonicalize_index(path, digest)
    execution_rows, execution_summary = canonicalize_index(
        path,
        digest,
        execution_scene_order=["scene-b", "scene-a"],
    )

    assert [row["scene"] for row in source_rows] == ["scene-a", "scene-b"]
    assert [row["scene"] for row in execution_rows] == ["scene-b", "scene-a"]
    assert [row["sample_index"] for row in execution_rows] == [1, 0]
    assert [row["execution_index"] for row in execution_rows] == [0, 1]
    assert execution_summary == source_summary


def test_protocol_dataset_bounds_reject_missing_scene_and_view_overflow():
    from scripts.compile_protocol import validate_dataset_bounds

    rows = [
        {
            "sample_index": 0,
            "scene": "scene-a",
            "context_indices": [0, 4],
            "target_indices": [2],
        }
    ]
    with pytest.raises(ValueError, match="missing from prepared dataset"):
        validate_dataset_bounds(rows, {})
    with pytest.raises(ValueError, match="view index 4"):
        validate_dataset_bounds(rows, {"scene-a": 4})


def test_all_pairs_keep_model_specific_dataset_representation():
    from scripts.compile_protocol import compile_protocol

    compiled = compile_protocol(PROTOCOL, ROOT)
    pairs = compiled["pairs"]
    assert pairs["depthsplat/dl3dv"]["dataset_representation"] == (
        "depthsplat-native-270x480-v1"
    )
    assert pairs["mvsplat/dl3dv"]["dataset_representation"] == (
        "re10k-compatible-360x640-v1"
    )


def test_compiled_pairs_preserve_prepared_tree_identity():
    from scripts.compile_protocol import compile_protocol

    compiled = compile_protocol(PROTOCOL, ROOT)
    pairs = compiled["pairs"]
    assert pairs["transplat/re10k"]["dataset_tree_sha256"] == (
        "2866634245989caa455fb46e5991d4e4b51e643c3f3e024e8793ff2b664fadd4"
    )
    assert pairs["mvsplat/acid"]["dataset_tree_sha256"] == (
        "0e21d05f448675e160529881d583701a2d03f19dbaae1af7941f37137688c187"
    )
    assert pairs["depthsplat/dl3dv"]["dataset_tree_sha256"] is None


def test_protocol_compiler_rejects_invalid_prepared_tree_hash(tmp_path: Path):
    from scripts.compile_protocol import compile_protocol

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["pairs"]["mvsplat/re10k"]["dataset_tree_sha256"] = "invalid"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset tree SHA256"):
        compile_protocol(path, ROOT)
