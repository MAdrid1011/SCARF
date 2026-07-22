import inspect
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _entry(scene: str, context=(0, 9), target=(1, 3, 5)) -> dict:
    return {
        "scene": scene,
        "context_indices": list(context),
        "target_indices": list(target),
    }


def test_hash_ranked_selection_is_order_independent_and_domain_separated():
    from scripts.calibration_contract import hash_ranked_selection

    entries = [_entry(f"scene-{index:03d}") for index in range(20)]
    forward = hash_ranked_selection(entries, dataset="re10k", count=8)
    reverse = hash_ranked_selection(list(reversed(entries)), dataset="re10k", count=8)
    acid = hash_ranked_selection(entries, dataset="acid", count=8)

    assert forward == reverse
    assert len(forward) == 8
    assert len({item["scene"] for item in forward}) == 8
    assert [item["scene"] for item in forward] != [item["scene"] for item in acid]


def test_calibration_rejects_any_evaluation_scene_or_view_overlap():
    from scripts.calibration_contract import assert_evaluation_disjoint

    evaluation = [_entry("eval-scene", context=(1, 4), target=(2, 3))]
    assert_evaluation_disjoint([_entry("train-scene")], evaluation)

    with pytest.raises(ValueError, match="evaluation overlap"):
        assert_evaluation_disjoint([_entry("eval-scene")], evaluation)


def test_global_candidate_selection_is_quality_first_and_target_independent():
    from scripts.calibration_contract import select_global_candidate

    candidates = [
        {
            "parameters": {
                "gamma_depth": 0.05,
                "beta_x": 0.25,
                "beta_f": 0.05,
                "beta_d": 0.5,
            },
            "quality": {
                "transplat/re10k": {
                    "psnr_loss_db": 0.02,
                    "ssim_loss": 0.001,
                    "lpips_increase": 0.001,
                }
            },
            "work_reduction": 0.20,
            "compression": 0.30,
        },
        {
            "parameters": {
                "gamma_depth": 0.10,
                "beta_x": 0.50,
                "beta_f": 0.10,
                "beta_d": 1.0,
            },
            "quality": {
                "transplat/re10k": {
                    "psnr_loss_db": 0.06,
                    "ssim_loss": 0.001,
                    "lpips_increase": 0.001,
                }
            },
            "work_reduction": 0.90,
            "compression": 0.90,
        },
    ]

    selected = select_global_candidate(candidates)
    assert selected["parameters"]["gamma_depth"] == 0.05


def test_calibration_rejects_pair_specific_parameters_and_seed_search():
    from scripts.calibration_contract import validate_candidate

    with pytest.raises(ValueError, match="pair-specific"):
        validate_candidate({"parameters": {}, "pair_overrides": {"mvsplat/re10k": {}}})
    with pytest.raises(ValueError, match="projection seed"):
        validate_candidate({"parameters": {"projection_seed": 7}})


def test_calibration_implementation_has_no_expected_result_dependency():
    import scripts.calibration_contract as contract

    source = inspect.getsource(contract)
    forbidden = "expected" + "_results"
    assert forbidden not in source
    assert "micro59-submit" not in source


def test_calibration_view_selection_is_deterministic_and_in_bounds():
    from scripts.compile_calibration import deterministic_views

    first = deterministic_views("scene-a", 12)
    second = deterministic_views("scene-a", 12)

    assert first == second
    assert len(first["context"]) == 2
    assert len(first["target"]) == 3
    assert not set(first["context"]) & set(first["target"])
    assert max([*first["context"], *first["target"]]) < 12


def test_calibration_manifest_compiler_rejects_eval_overlap(tmp_path: Path):
    from scripts.compile_calibration import compile_dataset_selection

    evaluation = [{
        "scene": "eval-scene",
        "context_indices": [0, 9],
        "target_indices": [1, 3, 5],
    }]
    counts = {"eval-scene": 10, **{f"train-{index}": 10 for index in range(40)}}

    with pytest.raises(ValueError, match="evaluation overlap"):
        compile_dataset_selection(
            dataset="re10k",
            scene_view_counts=counts,
            evaluation_rows=evaluation,
            count=32,
        )


def test_calibration_manifest_contains_only_hash_ranked_training_scenes():
    from scripts.compile_calibration import compile_dataset_selection

    counts = {f"train-{index:03d}": 12 for index in range(40)}
    record = compile_dataset_selection(
        dataset="acid",
        scene_view_counts=counts,
        evaluation_rows=[],
        count=32,
    )

    assert record["sample_count"] == 32
    assert len(record["index"]) == 32
    assert len(record["selection_sha256"]) == 64
    assert all(scene in counts for scene in record["index"])


def test_calibration_trace_aggregation_builds_one_global_candidate_record():
    from scripts.calibration_sweep import aggregate_candidate_traces

    parameters = {
        "gamma_depth": 0.1,
        "beta_x": 0.5,
        "beta_f": 0.1,
        "beta_d": 1.0,
    }
    traces = []
    for model in ("transplat", "mvsplat"):
        traces.append(
            {
                "kind": "calibration_sample_trace",
                "model": model,
                "dataset": "re10k",
                "sample_index": 0,
                "trace": {"neural_forward_passes": 1},
                "candidates": [
                    {
                        "parameters": parameters,
                        "quality": {
                            "psnr_loss_db": 0.01,
                            "ssim_loss": 0.001,
                            "lpips_increase": 0.001,
                        },
                        "work_reduction": 0.4,
                        "compression": 0.2,
                    }
                ],
            }
        )

    candidates = aggregate_candidate_traces(traces)

    assert len(candidates) == 1
    assert set(candidates[0]["quality"]) == {
        "transplat/re10k",
        "mvsplat/re10k",
    }
    assert candidates[0]["work_reduction"] == pytest.approx(0.4)


def test_calibration_runtime_sources_are_isolated_from_paper_targets():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/calibration_contract.py",
        "scripts/calibration_inputs.py",
        "scripts/calibration_replay.py",
        "scripts/calibration_sweep.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        forbidden = "expected" + "_results"
        assert forbidden not in source
        assert "micro59-submit" not in source


def test_full_training_download_contract_does_not_claim_unsupported_resume():
    root = Path(__file__).resolve().parents[1]
    script = (root / "data" / "download_calibration_splits.sh").read_text(
        encoding="utf-8"
    )

    assert "--continue-at" not in script
    assert "SCARF_CALIBRATION_RESTART_PARTIAL" in script
    assert "does not support byte-range requests" in script


class _ArchiveTorch:
    @staticmethod
    def save(value, path):
        Path(path).write_bytes(pickle.dumps(value))

    @staticmethod
    def load(path, map_location=None):
        return pickle.loads(Path(path).read_bytes())


def test_target_free_calibration_sidecar_excludes_target_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "torch", _ArchiveTorch)

    from scripts.calibration_inputs import (
        calibration_scene_order,
        load_target_free_record,
        materialize_target_free_inputs,
        validate_target_free_input_root,
    )

    scene = "calibration-scene"
    root = tmp_path / "inputs" / "re10k"
    record = materialize_target_free_inputs(
        root,
        dataset="re10k",
        examples={
            scene: {
                "key": scene,
                "cameras": [[0.0] * 18 for _ in range(5)],
                "images": [
                    b"context-0",
                    b"target-rgb-1",
                    b"target-rgb-2",
                    b"target-rgb-3",
                    b"context-4",
                ],
            }
        },
        index={scene: {"context": [0, 4], "target": [1, 2, 3]}},
        source={
            "dataset_tree_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "prepared_source_sha256": "c" * 64,
        },
        selection_sha256="d" * 64,
    )

    assert record["target_rgb_included"] is False
    assert all("target" not in item["path"] for item in record["opened_file_manifest"])
    assert calibration_scene_order(root) == [scene]
    sidecar = load_target_free_record(root, scene)
    assert sidecar["context_images"] == [b"context-0", b"context-4"]
    assert "images" not in sidecar
    assert "target_images" not in sidecar
    assert validate_target_free_input_root(root, "re10k")[
        "target_rgb_accessed"
    ] is False


def test_target_free_calibration_sidecar_has_a_hashed_opened_file_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "torch", _ArchiveTorch)

    from scripts.calibration_inputs import materialize_target_free_inputs, validate_target_free_input_root

    root = tmp_path / "inputs" / "dl3dv"
    materialize_target_free_inputs(
        root,
        dataset="dl3dv",
        examples={
            "train-scene": {
                "key": "train-scene",
                "cameras": [[0.0] * 18 for _ in range(5)],
                "images": [b"context-0", b"target-1", b"target-2", b"target-3", b"context-4"],
            }
        },
        index={"train-scene": {"context": [0, 4], "target": [1, 2, 3]}},
        source={},
        selection_sha256="d" * 64,
    )

    identity = validate_target_free_input_root(root, "dl3dv")
    assert identity["opened_file_manifest_sha256"]
    assert {item["path"] for item in identity["opened_file_manifest"]} == {
        "test/000000.torch",
        "test/index.json",
    }
    assert all("target" not in item["path"] for item in identity["opened_file_manifest"])


def test_target_free_calibration_sidecar_rejects_target_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "torch", _ArchiveTorch)

    from scripts.calibration_inputs import (
        load_target_free_record,
        materialize_target_free_inputs,
    )

    scene = "calibration-scene"
    root = tmp_path / "inputs" / "acid"
    materialize_target_free_inputs(
        root,
        dataset="acid",
        examples={
            scene: {
                "key": scene,
                "cameras": [[0.0] * 18 for _ in range(5)],
                "images": [b"context-0", b"target-1", b"target-2", b"target-3", b"context-4"],
            }
        },
        index={scene: {"context": [0, 4], "target": [1, 2, 3]}},
        source={},
        selection_sha256="d" * 64,
    )
    path = root / "test" / "000000.torch"
    payload = _ArchiveTorch.load(path, map_location="cpu")
    payload[0]["target_images"] = [b"forbidden"]
    _ArchiveTorch.save(payload, path)

    with pytest.raises(ValueError, match="target RGB"):
        load_target_free_record(root, scene)


def test_calibration_sweep_rejects_a_protocol_without_target_free_inputs(
    tmp_path: Path,
):
    from scripts.calibration_sweep import build_plan

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "calibration_protocol",
                "evaluation_disjoint": True,
                "datasets": {"re10k": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target-free input root"):
        build_plan(manifest, tmp_path / "sweep")


def test_target_free_loader_never_decodes_target_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    torch = pytest.importorskip("torch")
    from PIL import Image

    from integration.model_loader import load_target_free_calibration_data
    from scripts.calibration_inputs import materialize_target_free_inputs
    from scripts.compile_protocol import canonicalize_index, sha256_file

    scene = "calibration-scene"
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps({scene: {"context": [0, 4], "target": [1, 2, 3]}}),
        encoding="utf-8",
    )
    _, selection = canonicalize_index(index_path, sha256_file(index_path))

    def png_bytes(value: int) -> torch.Tensor:
        from io import BytesIO

        stream = BytesIO()
        Image.new("RGB", (8, 8), (value, value, value)).save(stream, format="PNG")
        return torch.tensor(list(stream.getvalue()), dtype=torch.uint8)

    cameras = torch.zeros(5, 18)
    cameras[:, 0] = 0.5
    cameras[:, 1] = 0.5
    cameras[:, 2] = 0.5
    cameras[:, 3] = 0.5
    cameras[:, 6:] = torch.eye(4)[:3].reshape(1, -1)
    input_root = tmp_path / "inputs" / "re10k"
    materialize_target_free_inputs(
        input_root,
        dataset="re10k",
        examples={
            scene: {
                "key": scene,
                "cameras": cameras,
                # A target decode would fail immediately; only views 0 and 4 are valid.
                "images": [png_bytes(10), b"invalid-target", b"invalid-target", b"invalid-target", png_bytes(20)],
            }
        },
        index={scene: {"context": [0, 4], "target": [1, 2, 3]}},
        source={},
        selection_sha256=selection["sample_selection_sha256"],
    )

    src = ModuleType("src")
    dataset = ModuleType("src.dataset")
    shims = ModuleType("src.dataset.shims")
    data_module = ModuleType("src.dataset.data_module")
    crop_shim = ModuleType("src.dataset.shims.crop_shim")
    data_module.get_data_shim = lambda encoder: lambda batch: batch
    crop_shim.apply_crop_shim = lambda batch, image_shape: batch
    src.dataset = dataset
    dataset.shims = shims
    shims.crop_shim = crop_shim
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.dataset", dataset)
    monkeypatch.setitem(sys.modules, "src.dataset.shims", shims)
    monkeypatch.setitem(sys.modules, "src.dataset.data_module", data_module)
    monkeypatch.setitem(sys.modules, "src.dataset.shims.crop_shim", crop_shim)

    class Loader:
        def __init__(self):
            self.setup_called = False
            self.restore_called = False

        def _setup_imports(self):
            self.setup_called = True

        def _restore_cwd(self):
            self.restore_called = True

    model_bundle = SimpleNamespace(
        encoder=object(),
        config=SimpleNamespace(
            dataset=SimpleNamespace(
                image_shape=[8, 8],
                make_baseline_1=False,
                near=1.0,
                far=2.0,
                baseline_scale_bounds=False,
            )
        ),
    )
    loader = Loader()
    batch = load_target_free_calibration_data(
        loader,
        model_bundle,
        dataset_name="re10k",
        dataset_root=input_root,
        evaluation_index=index_path,
        sample_index=0,
    ).batch

    assert loader.setup_called and loader.restore_called
    assert batch["context"]["image"].shape == (1, 2, 3, 8, 8)
    assert "image" not in batch["target"]
    assert batch["target"]["extrinsics"].shape == (1, 3, 4, 4)
    assert batch["calibration"]["target_rgb_accessed"] is False
    assert batch["calibration"]["index_sha256"] == sha256_file(index_path)
    assert batch["calibration"]["index_selection_sha256"] == selection[
        "sample_selection_sha256"
    ]
