"""CPU contracts for the fixed DepthSplat target-RGB quality gate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest


torch = pytest.importorskip("torch")


def _sha(character: str) -> str:
    return character * 64


def _identity(gate):
    context_indices = [0, 9]
    target_indices = [1, 3, 5, 7]
    selection = {
        "source_sample_index": 0,
        "scene": "scene-fixed",
        "context_indices": context_indices,
        "target_indices": target_indices,
    }
    return {
        "scene": "scene-fixed",
        "context_indices": context_indices,
        "source_binding": {
            "canonical_selection_sha256": gate.canonical_json_sha256(selection)
        },
    }


def _formal_audit(gate, identity, literal_guard, literal_profile, backend):
    expected_literal = gate._expected_literal_binding(
        literal_guard=literal_guard, literal_profile=literal_profile
    )
    boundary = {
        "encoder_only": True,
        "decoder_constructed": False,
        "renderer_executed": False,
        "target_view_rendered": False,
        "quality_metrics_computed": False,
        "timing_claim": False,
        "whole_pipeline_s2_s3_sparse_execution_verified": False,
        "global_s2_s3_savings_claimed": False,
        "native_dense_depth_predictor_executed": True,
        "native_dense_gaussian_regressor_executed": True,
        "initial_selected_native_rgb_adapter_executed": True,
        "final_selected_native_rgb_adapter_executed": True,
        "nonzero_l0_l1_materializer_executed": True,
    }
    return {
        "schema_version": gate.FORMAL_AUDIT_SCHEMA_VERSION,
        "kind": gate.FORMAL_AUDIT_KIND,
        "status": "PASS",
        "paper_result_eligible": False,
        "formal_target_free_audit": True,
        "formal_target_free_sidecar_used": True,
        "model": gate.MODEL,
        "dataset": gate.DATASET,
        "source_sample_index": gate.SOURCE_SAMPLE_INDEX,
        "scene": identity["scene"],
        "context_indices": identity["context_indices"],
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "execution_boundary": boundary,
        "context_only_input": {
            "identity": identity,
            "loaded_native_preprocessing": {"patch_size": 1},
        },
        "literal_t4_v16": expected_literal,
        "backend_identity": backend,
        "checkpoint_sha256": _sha("c"),
        "route_plan": {},
        "materialization_preflight": {},
        "materialization_preflight_tile_trace": [],
        "final_route": {},
        "native_execution": {},
        "initial_selected_head": {},
        "producer_selected_head": {},
        "initial_packet": {},
        "final_packet": {},
        "materialized_packet": {},
        "geometry_source": {},
    }


def _write_target_fixture(tmp_path: Path, gate):
    from PIL import Image

    from data.build_manifest import build
    from data.convert_dl3dv import _load_metadata
    from scripts.calibration_inputs import (
        materialize_target_free_inputs,
        sha256_file,
        validate_target_free_input_root,
    )
    from scripts.compile_protocol import canonicalize_index

    scene = "scene-fixed"
    context_indices = [0, 9]
    target_indices = [1, 3, 5, 7]
    raw_root = tmp_path / "raw"
    scene_root = raw_root / scene / "nerfstudio"
    image_root = scene_root / "images_8"
    image_root.mkdir(parents=True)
    frames = []
    for index in range(10):
        name = f"frame_{index + 1:05d}.png"
        Image.new("RGB", (480, 270), (index, index + 1, index + 2)).save(
            image_root / name
        )
        frames.append(
            {
                "file_path": f"images/{name}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, float(index)],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    transforms_path = scene_root / "transforms.json"
    transforms_path.write_text(
        json.dumps(
            {
                "h": 270,
                "w": 480,
                "fl_x": 480.0,
                "fl_y": 270.0,
                "cx": 240.0,
                "cy": 135.0,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    plans = {
        scene: {
            "native": {
                "image_subdir": "images_8",
                "source_image_shape": [270, 480],
            }
        }
    }
    raw_source = {
        "revision": "fixture-revision",
        "benchmark_metadata_sha256": "a" * 64,
        "filelist_sha256": "b" * 64,
        "scene_source_plans": plans,
        "scene_source_plans_sha256": hashlib.sha256(
            json.dumps(plans, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    raw_source_path = raw_root / ".scarf-dl3dv-source.json"
    raw_source_path.write_text(
        json.dumps(raw_source, sort_keys=True), encoding="utf-8"
    )
    raw_source_sha256 = sha256_file(raw_source_path)
    metadata, _timestamps = _load_metadata(transforms_path)

    source_root = tmp_path / "source"
    source_root.mkdir()
    selection = {scene: {"context": context_indices, "target": target_indices}}
    selection_path = source_root / "audit-selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    _, selection_summary = canonicalize_index(
        selection_path, sha256_file(selection_path)
    )
    images = [
        (image_root / f"frame_{index + 1:05d}.png").read_bytes()
        for index in range(10)
    ]
    materialize_target_free_inputs(
        source_root / "sidecar",
        dataset="dl3dv",
        examples={
            scene: {
                "key": scene,
                "cameras": metadata["cameras"],
                "images": images,
            }
        },
        index=selection,
        source={"prepared_source_sha256": raw_source_sha256},
        selection_sha256=selection_summary["sample_selection_sha256"],
    )
    sidecar_identity = validate_target_free_input_root(source_root / "sidecar", "dl3dv")
    source_audit = {
        "schema_version": "1.0",
        "kind": "dl3dv_target_free_l1_primary_reference_audit_input",
        "status": "PASS",
        "paper_result_eligible": False,
        "model": "depthsplat",
        "dataset": "dl3dv",
        "source_sample_index": 0,
        "selected_sample": {
            "scene": scene,
            "context_indices": context_indices,
            "target_indices": target_indices,
            "audit_selection_sha256": selection_summary["sample_selection_sha256"],
            "audit_selection_file_sha256": sha256_file(selection_path),
        },
        "canonical_selection": {
            "source_sample_index": 0,
            "scene": scene,
            "context_indices": context_indices,
            "target_indices": target_indices,
        },
        "canonical_protocol": {
            "source_index_sha256": "c" * 64,
            "sample_selection_sha256": "d" * 64,
        },
        "source": {
            "revision": raw_source["revision"],
            "benchmark_metadata_sha256": raw_source["benchmark_metadata_sha256"],
            "filelist_sha256": raw_source["filelist_sha256"],
            "scene_source_plans_sha256": raw_source["scene_source_plans_sha256"],
            "source_record_sha256": raw_source_sha256,
        },
        "target_rgb_included": False,
        "target_rgb_opened": False,
        "target_rgb_paths_passed_to_encoder": False,
    }
    source_audit_path = source_root / "audit-input.json"
    source_audit_path.write_text(
        json.dumps(source_audit, sort_keys=True), encoding="utf-8"
    )
    manifest = build(source_root, "fixture", "fixture", "fixture")
    (source_root / ".scarf-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    selection_payload = source_audit["canonical_selection"]
    identity = {
        "scene": scene,
        "context_indices": context_indices,
        "source_binding": {
            "source_audit_input_sha256": sha256_file(source_audit_path),
            "source_audit_tree_sha256": manifest["tree_sha256"],
            "source_sidecar_tree_sha256": sidecar_identity["tree_sha256"],
            "canonical_selection_sha256": gate.canonical_json_sha256(selection_payload),
            "canonical_index_sha256": "c" * 64,
            "canonical_sample_selection_sha256": "d" * 64,
        },
    }
    return source_root, raw_root, identity, target_indices


def _install_target_shims(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    src = ModuleType("src")
    dataset = ModuleType("src.dataset")
    shims = ModuleType("src.dataset.shims")
    crop = ModuleType("src.dataset.shims.crop_shim")
    patch = ModuleType("src.dataset.shims.patch_shim")

    calls = {"crop": 0, "patch": 0}

    def apply_crop_shim_to_views(views, _shape):
        calls["crop"] += 1
        return views

    def apply_patch_shim_to_views(views, _patch_size):
        calls["patch"] += 1
        return {
            **views,
            "image": views["image"][..., :256, :448],
        }

    crop.apply_crop_shim_to_views = apply_crop_shim_to_views
    patch.apply_patch_shim_to_views = apply_patch_shim_to_views
    src.dataset = dataset
    dataset.shims = shims
    shims.crop_shim = crop
    shims.patch_shim = patch
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.dataset", dataset)
    monkeypatch.setitem(sys.modules, "src.dataset.shims", shims)
    monkeypatch.setitem(sys.modules, "src.dataset.shims.crop_shim", crop)
    monkeypatch.setitem(sys.modules, "src.dataset.shims.patch_shim", patch)
    return calls


def test_quality_gate_parser_is_fixed_to_the_registered_route():
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    parser = gate.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "result"])
    args = parser.parse_args(
        [
            "--output-dir",
            "result",
            "--formal-audit",
            "audit/results.json",
            "--literal-t4-v16-record",
            "frozen/v16t4.json",
            "--source-audit-root",
            "source-target-free",
        ]
    )

    assert args.formal_audit.name == "results.json"
    assert args.literal_t4_v16_record.name == "v16t4.json"
    assert args.source_audit_root.name == "source-target-free"
    assert not hasattr(args, "sample_index")
    assert not hasattr(args, "model")
    assert not hasattr(args, "seed")


def test_formal_audit_must_remain_target_free_before_quality(monkeypatch):
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    identity = _identity(gate)
    literal_guard = {
        "frozen_record_sha256": _sha("a"),
        "frozen_record_kind": "literal-v16",
        "threshold_value": 0.25,
        "threshold_rule": "fixed",
        "risk_metric": "risk",
        "acid_binding_sha256": _sha("b"),
        "application_sha256": _sha("d"),
    }
    literal_profile = {"materialization_profile": "literal"}
    backend = {"model": "depthsplat"}
    audit = _formal_audit(gate, identity, literal_guard, literal_profile, backend)
    monkeypatch.setattr(gate, "_read_formal_audit", lambda _path: (audit, {}))
    monkeypatch.setattr(gate, "_require_formal_audit_runner", lambda _audit: None)

    accepted, _binding = gate._validate_formal_audit(
        "audit/results.json",
        input_identity=identity,
        literal_guard=literal_guard,
        literal_profile=literal_profile,
        backend_identity=backend,
        checkpoint_sha256=_sha("c"),
    )
    assert accepted is audit

    audit["target_rgb_accessed"] = True
    with pytest.raises(ValueError, match="target_rgb_accessed"):
        gate._validate_formal_audit(
            "audit/results.json",
            input_identity=identity,
            literal_guard=literal_guard,
            literal_profile=literal_profile,
            backend_identity=backend,
            checkpoint_sha256=_sha("c"),
        )


def test_isolated_target_batch_bypasses_native_loader_and_opens_only_sample_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    source_root, raw_root, identity, target_indices = _write_target_fixture(tmp_path, gate)
    shim_calls = _install_target_shims(monkeypatch)

    class Loader:
        load_data_called = False

        def _setup_imports(self):
            return None

        def _restore_cwd(self):
            return None

        def load_data(self, *_args, **_kwargs):
            self.load_data_called = True
            raise AssertionError("isolated quality target reader must not call load_data")

    loader = Loader()
    bundle = SimpleNamespace(
        config=SimpleNamespace(
            dataset=SimpleNamespace(
                image_shape=(270, 480),
                make_baseline_1=False,
                baseline_epsilon=1.0e-3,
                near=0.5,
                far=200.0,
                baseline_scale_bounds=False,
            )
        ),
        encoder=SimpleNamespace(
            cfg=SimpleNamespace(shim_patch_size=16, downscale_factor=4)
        ),
    )

    target, actual_indices, provenance = gate._load_isolated_target_batch_after_packet_commit(
        loader,
        bundle,
        scene=identity["scene"],
        context_indices=[0, 9],
        input_identity=identity,
        native_preprocessing={"prepared_image_shape": [256, 448]},
        source_audit_root=source_root,
        raw_root=raw_root,
    )
    assert loader.load_data_called is False
    assert actual_indices == target_indices
    assert target["index"].tolist() == [target_indices]
    assert target["image"].shape == (1, 4, 3, 256, 448)
    assert shim_calls == {"crop": 0, "patch": 1}
    assert provenance["reader"] == gate.ISOLATED_TARGET_READER
    assert provenance["native_dl3dv_chunk_loader_used"] is False
    assert provenance["data_module_used"] is False
    assert provenance["nonselected_dl3dv_sample_data_opened"] is False
    assert [record["role"] for record in provenance["opened_source_files"]] == [
        "source_plan",
        "target_camera_geometry",
        "target_rgb",
        "target_rgb",
        "target_rgb",
        "target_rgb",
    ]
    assert [Path(record["path"]).name for record in provenance["opened_source_files"][2:]] == [
        "frame_00002.png",
        "frame_00004.png",
        "frame_00006.png",
        "frame_00008.png",
    ]


def test_isolated_target_reader_rejects_audited_selection_drift(tmp_path: Path):
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    source_root, _raw_root, identity, _target_indices = _write_target_fixture(
        tmp_path, gate
    )
    identity["source_binding"]["canonical_selection_sha256"] = "f" * 64

    with pytest.raises(RuntimeError, match="selection differs"):
        gate._load_isolated_source_target_record(
            source_audit_root=source_root,
            scene=identity["scene"],
            context_indices=identity["context_indices"],
            input_identity=identity,
        )


def test_isolated_target_reader_rejects_source_audit_tree_drift(tmp_path: Path):
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    source_root, _raw_root, identity, _target_indices = _write_target_fixture(
        tmp_path, gate
    )
    identity["source_binding"]["source_audit_tree_sha256"] = "f" * 64

    with pytest.raises(RuntimeError, match="source tree differs"):
        gate._load_isolated_source_target_record(
            source_audit_root=source_root,
            scene=identity["scene"],
            context_indices=identity["context_indices"],
            input_identity=identity,
        )


def test_quality_gate_requires_an_actual_nonzero_merge():
    from scripts import saes_depthsplat_l0_l1_quality_gate as gate

    empty = SimpleNamespace(update_dense_slots=torch.empty(0, dtype=torch.long))
    with pytest.raises(RuntimeError, match="actual compact nonzero merge"):
        gate._require_nonzero_merge(empty)

    nonempty = SimpleNamespace(update_dense_slots=torch.tensor([7], dtype=torch.long))
    assert gate._require_nonzero_merge(nonempty) == 1
