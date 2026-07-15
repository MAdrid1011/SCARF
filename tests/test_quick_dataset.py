import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def test_quick_dataset_is_deterministic_re10k_functional_fixture(tmp_path: Path):
    from data.build_quick_dataset import REPRESENTATION, SCENE, generate

    output = tmp_path / "quick-re10k"
    manifest = generate(output)
    repeated_manifest = generate(output)
    second_manifest = generate(tmp_path / "quick-re10k-second")
    chunk = torch.load(output / "test/000000.torch", map_location="cpu")
    index = json.loads((output / "test/index.json").read_text(encoding="utf-8"))

    assert manifest["functional_fixture"] is True
    assert manifest["paper_result_eligible"] is False
    assert manifest["representation"] == REPRESENTATION
    assert repeated_manifest == manifest
    assert manifest["tree_sha256"] == second_manifest["tree_sha256"]
    assert index == {SCENE: "000000.torch"}
    assert len(chunk) == 1
    assert chunk[0]["cameras"].shape == (5, 18)
    assert len(chunk[0]["images"]) == 5
