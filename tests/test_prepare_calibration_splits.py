import json
import sys
import zipfile
from pathlib import Path

import pytest


def _index(dataset: str, count: int = 40) -> dict[str, str]:
    return {
        f"{dataset}-train-{number:03d}": "selected.torch"
        for number in range(count)
    }


def _archive(path: Path, dataset: str, index: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dataset}/train/index.json",
            json.dumps(index, sort_keys=True),
        )
        archive.writestr(f"{dataset}/train/selected.torch", b"synthetic-chunk")


def test_preparation_extracts_only_the_fixed_hash_ranked_subset(tmp_path: Path):
    from data.prepare_calibration_splits import prepare_dataset
    from scripts.calibration_contract import hash_ranked_scene_names

    index = _index("re10k")
    archive = tmp_path / "re10k.zip"
    _archive(archive, "re10k", index)

    record = prepare_dataset(
        archive,
        dataset="re10k",
        output_root=tmp_path / "prepared",
        source_url="https://example.invalid/re10k.zip",
        evaluation_rows=[],
    )

    destination = tmp_path / "prepared/re10k"
    selected = hash_ranked_scene_names(list(index), dataset="re10k", count=32)
    selected_index = json.loads((destination / "train/index.json").read_text())
    provenance = json.loads((destination / ".scarf-calibration-source.json").read_text())

    assert set(selected_index) == set(selected)
    assert selected_index == {scene: index[scene] for scene in selected}
    assert json.loads((destination / "train/full-index.json").read_text()) == index
    assert (destination / "train/selected.torch").read_bytes() == b"synthetic-chunk"
    assert record["selected_scene_count"] == 32
    assert provenance["selected_scenes"] == selected
    assert provenance["evaluation_disjoint"] is True


def test_preparation_rejects_selected_training_scene_that_overlaps_evaluation(
    tmp_path: Path,
):
    from data.prepare_calibration_splits import prepare_dataset
    from scripts.calibration_contract import hash_ranked_scene_names

    index = _index("acid")
    archive = tmp_path / "acid.zip"
    _archive(archive, "acid", index)
    overlapping_scene = hash_ranked_scene_names(list(index), dataset="acid", count=32)[0]

    with pytest.raises(ValueError, match="evaluation overlap"):
        prepare_dataset(
            archive,
            dataset="acid",
            output_root=tmp_path / "prepared",
            source_url="https://example.invalid/acid.zip",
            evaluation_rows=[
                {
                    "scene": overlapping_scene,
                    "context_indices": [0],
                    "target_indices": [1],
                }
            ],
        )


def test_preparation_rejects_path_traversal_in_an_official_index(tmp_path: Path):
    from data.prepare_calibration_splits import prepare_dataset

    index = _index("re10k")
    index["re10k-train-000"] = "../escape.torch"
    archive = tmp_path / "re10k.zip"
    _archive(archive, "re10k", index)

    with pytest.raises(ValueError, match="unsafe training index chunk path"):
        prepare_dataset(
            archive,
            dataset="re10k",
            output_root=tmp_path / "prepared",
            source_url="https://example.invalid/re10k.zip",
            evaluation_rows=[],
        )


def test_compiler_requires_the_prepared_subset_to_match_full_index_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from scripts.calibration_contract import canonical_sha256, hash_ranked_scene_names
    from scripts.compile_calibration import _selected_examples
    from scripts.compile_protocol import sha256_file

    dataset = "re10k"
    index = _index(dataset)
    selected = hash_ranked_scene_names(list(index), dataset=dataset, count=32)
    root = tmp_path / dataset
    train = root / "train"
    train.mkdir(parents=True)
    (train / "full-index.json").write_text(json.dumps(index, sort_keys=True))
    selected_index = {scene: index[scene] for scene in selected}
    (train / "index.json").write_text(json.dumps(selected_index, sort_keys=True))
    (train / "selected.torch").write_text(
        json.dumps([{"key": scene, "images": [0, 1, 2, 3, 4]} for scene in selected])
    )
    (root / ".scarf-manifest.json").write_text(
        json.dumps({"tree_sha256": "a" * 64, "source": "official", "revision": "synthetic"})
    )
    (root / ".scarf-calibration-source.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "full_train_index_sha256": sha256_file(train / "full-index.json"),
                "selected_index_sha256": canonical_sha256(selected_index),
                "selected_scenes": selected,
                "evaluation_disjoint": True,
            }
        )
    )

    class FakeTorch:
        @staticmethod
        def load(path, map_location):
            return json.loads(Path(path).read_text())

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    examples, source = _selected_examples(root, dataset)

    assert set(examples) == set(selected)
    assert source["train_index_sha256"] == sha256_file(train / "full-index.json")
    assert source["selected_train_index_sha256"] == sha256_file(train / "index.json")
    assert source["prepared_source_sha256"] == sha256_file(
        root / ".scarf-calibration-source.json"
    )

    selected_index.pop(selected[0])
    replacement = next(scene for scene in index if scene not in selected)
    selected_index[replacement] = index[replacement]
    (train / "index.json").write_text(json.dumps(selected_index, sort_keys=True))
    with pytest.raises(ValueError, match="fixed 32-scene subset"):
        _selected_examples(root, dataset)
