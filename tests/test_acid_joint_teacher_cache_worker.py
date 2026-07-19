from copy import deepcopy

import pytest


torch = pytest.importorskip("torch")


def test_preparation_contract_rejects_a_model_patch_size_change():
    from scripts.acid_joint_teacher_cache_worker import _validate_preparation_contract

    contract = {
        "preprocessing": {
            "source_image_shape": [360, 640],
            "target_image_shape": [256, 256],
            "per_model_patch_size": {"transplat": 16},
        }
    }
    preparation = {
        "source_image_shape": [360, 640],
        "prepared_image_shape": [256, 256],
        "patch_size": 16,
    }

    _validate_preparation_contract(preparation, contract=contract, model="transplat")

    changed = deepcopy(preparation)
    changed["patch_size"] = 8
    with pytest.raises(RuntimeError, match="patch size"):
        _validate_preparation_contract(changed, contract=contract, model="transplat")


def test_cache_worker_rejects_a_noncanonical_legacy_contract_before_loading_data(tmp_path):
    from scripts import acid_joint_teacher_cache_worker as worker

    with pytest.raises(RuntimeError, match="canonical v5 frozen contract"):
        worker.compile_cache(
            contract_path=tmp_path / "acid_joint_materialization_training_contract.json",
            materialization_root=tmp_path / "inputs",
            model="transplat",
            split="calibration_train",
            output_root=tmp_path / "cache",
            sample_indices=[0],
            device=torch.device("cpu"),
        )


def test_cache_worker_direct_api_requires_a_model_isolation_marker(tmp_path, monkeypatch):
    from data.acid_joint_training_contract import ACID_ISOLATED_MODEL_ENV
    from scripts import acid_joint_teacher_cache_worker as worker

    contract_path = tmp_path / "acid_joint_materialization_training_contract_v5.json"
    monkeypatch.setattr(worker, "DEFAULT_TRAINING_CONTRACT_PATH", contract_path)
    monkeypatch.delenv(ACID_ISOLATED_MODEL_ENV, raising=False)

    with pytest.raises(Exception, match="dedicated isolated subprocess"):
        worker.compile_cache(
            contract_path=contract_path,
            materialization_root=tmp_path / "inputs",
            model="transplat",
            split="calibration_train",
            output_root=tmp_path / "cache",
            sample_indices=[0],
            device=torch.device("cpu"),
        )


def test_scene_record_binds_patch_size_and_cross_check_threshold(monkeypatch):
    from scripts import acid_joint_teacher_cache_worker as worker
    from data.acid_joint_training_contract import (
        PREPROCESSING_CONTRACT,
        SAES_MODEL_ROUTING,
        SAES_ROUTING_PARAMETERS,
    )
    from saes.joint_materialization_teacher import _validate_saes_kwargs
    from scripts.saes_execution_identity import build_saes_execution_identity

    captured = {}

    def fake_teacher(_dense, *, height, width, saes_kwargs):
        captured["height"] = height
        captured["width"] = width
        captured["options"] = _validate_saes_kwargs(
            saes_kwargs, height=height, width=width
        )
        attributes = {
            "means": torch.zeros(3),
            "covariances": torch.eye(3),
            "harmonics": torch.zeros(3, 1),
            "opacities": torch.zeros(1),
        }
        return {
            "packets": [
                {
                    "descriptor": torch.zeros(32),
                    "base": attributes,
                    "teacher": {name: value.clone() for name, value in attributes.items()},
                    "level": "L0",
                    "anchor_index": 0,
                }
            ],
            "route_mask_sha256": "a" * 64,
            "route_summary": {"level0_tiles": 1, "level1_tiles": 0, "full_tiles": 0},
            "controls": {"full_slot_count": 0},
        }

    monkeypatch.setattr(worker, "build_dense_adaptor_teacher_packets", fake_teacher)
    execution_identity = build_saes_execution_identity()
    contract = {
        "contract_sha256": "b" * 64,
        "source_plan": {"plan_sha256": "c" * 64},
        "checkpoint_binding": {"models": {"transplat": {"checkpoint_sha256": "d" * 64}}},
        "model_config_binding": {
            "transplat": {
                "source_manifest_sha256": "e" * 64,
                "runtime_source": {
                    "repository_path": "transplat",
                    "repository_commit": "a" * 40,
                    "src_path": "transplat/src",
                    "src_tree_sha256": "f" * 64,
                    "src_file_count": 1,
                    "runtime_import_roots": [
                        {
                            "path": "transplat/src",
                            "tree_sha256": "f" * 64,
                            "file_count": 1,
                        }
                    ],
                },
                "environment_profile": "classic",
                "profile_interpreter": {
                    "executable_sha256": "9" * 64,
                    "python_version": "3.10.0",
                },
                "resolved_config_sha256": "5" * 64,
            }
        },
        "preprocessing": PREPROCESSING_CONTRACT,
        "saes_routing": {
            "execution_identity": execution_identity,
            "execution_route_sha256": execution_identity["route_sha256"],
            "parameters": SAES_ROUTING_PARAMETERS,
            "per_model": SAES_MODEL_ROUTING,
            "routing_sha256": "f" * 64,
        },
        "context_only_inputs": {
            "splits": {
                "calibration_train": {
                    "tree_sha256": "1" * 64,
                    "manifest_sha256": "2" * 64,
                    "input_provenance_sha256": "3" * 64,
                    "selection_sha256": "4" * 64,
                    "scene_count": 24,
                }
            }
        },
    }
    preparation = {
        "source_image_shape": [360, 640],
        "prepared_image_shape": [256, 256],
        "patch_size": 16,
    }
    context = {
        "image": torch.zeros(1, 2, 3, 256, 256),
        "extrinsics": torch.eye(4).repeat(1, 2, 1, 1),
        "intrinsics": torch.eye(3).repeat(1, 2, 1, 1),
    }

    record = worker._record_for_scene(
        contract=contract,
        model="transplat",
        split="calibration_train",
        sample_index=0,
        source_identity={"scene": "acid-fixture", "sidecar": "000000.pt"},
        preparation=preparation,
        dense=None,
        features=torch.zeros(1),
        depths=torch.zeros(1),
        runtime_binding={
            "config_source_sha256": "e" * 64,
            "checkpoint_sha256": "d" * 64,
            "runtime_source": contract["model_config_binding"]["transplat"]["runtime_source"],
            "environment_profile": "classic",
            "profile_interpreter": {
                "executable_sha256": "9" * 64,
                "python_version": "3.10.0",
            },
            "resolved_config_sha256": "5" * 64,
        },
        context=context,
    )

    assert captured["options"]["cross_check_threshold"] == 0.015
    assert captured["options"]["context_safety_guard"] is True
    assert contract["saes_routing"]["parameters"]["l1_retained_positions"] == 12
    assert record["prepared_patch_size"] == 16
    assert record["routing"]["cross_check_threshold"] == 0.015
    assert record["saes_execution_route_sha256"] == execution_identity["route_sha256"]
    assert record["routing"]["execution_route_sha256"] == execution_identity["route_sha256"]
    assert worker._validate_scene_record(record, contract=contract, model="transplat") == 1
