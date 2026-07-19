from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest


def _digest(character: str) -> str:
    return hashlib.sha256(character.encode("ascii")).hexdigest()


def _contract() -> dict:
    from data.acid_joint_training_contract import (
        CALIBRATOR_ARCHITECTURE,
        CANDIDATE_VERIFICATION_POLICY,
        CHECKPOINT_REQUESTS,
        CONTRACT_ID,
        CONTRACT_KIND,
        CONTRACT_SCHEMA_VERSION,
        CONTRACT_STATUS,
        DEFAULT_ASSET_PATH,
        HOLDOUT_POLICY,
        MODEL_CONFIG_REQUESTS,
        MODEL_ENVIRONMENT_PROFILES,
        MODELS,
        OPTIMIZATION_RECIPE,
        PREPROCESSING_CONTRACT,
        PROTOCOL_ID,
        RESULT_RECORD_PATHS,
        RUNTIME_EVIDENCE_RECORDS,
        IMPLEMENTATION_PATHS,
        SAES_MODEL_ROUTING,
        SAES_ROUTING_PARAMETERS,
        TEACHER_FIDELITY_GATES,
        TEACHER_OBJECTIVE,
    )
    from scripts.calibration_contract import canonical_sha256
    from scripts.saes_execution_identity import build_saes_execution_identity

    source_plan_sha = _digest("a")
    split_identity = {}
    for split, character, scene_count in (
        ("calibration_train", "b", 24),
        ("calibration_holdout", "c", 8),
    ):
        split_identity[split] = {
            "split": split,
            "tree_sha256": _digest(character),
            "manifest_sha256": _digest(chr(ord(character) + 1)),
            "input_provenance_sha256": _digest(chr(ord(character) + 2)),
            "selection_sha256": _digest(chr(ord(character) + 3)),
            "scene_count": scene_count,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "teacher_artifact_accessed": False,
            "expected_results_accessed": False,
        }
    checkpoints = {}
    for ordinal, model in enumerate(MODELS):
        request = CHECKPOINT_REQUESTS[model]
        checkpoints[model] = {
            **request,
            "checkpoint_sha256": _digest(chr(ord("k") + ordinal)),
            "checkpoint_size": ordinal + 1,
        }
    configs = {}
    for ordinal, model in enumerate(MODELS):
        request = MODEL_CONFIG_REQUESTS[model]
        source_files = [
            {"path": path, "sha256": _digest(chr(ord("x") - ordinal))}
            for path in request["source_paths"]
        ]
        source_files[0]["sha256"] = _digest(chr(ord("g") + ordinal))
        entrypoint = next(
            item["sha256"]
            for item in source_files
            if item["path"] == request["entrypoint_path"]
        )
        configs[model] = {
            "entrypoint_path": request["entrypoint_path"],
            "entrypoint_sha256": entrypoint,
            "source_files": source_files,
            "source_manifest_sha256": canonical_sha256(source_files),
            "hydra_overrides": list(request["hydra_overrides"]),
            "environment_profile": MODEL_ENVIRONMENT_PROFILES[model],
            "runtime_source": {
                "repository_path": model,
                "repository_commit": chr(ord("a") + ordinal) * 40,
                "src_path": f"{model}/src",
                "src_tree_sha256": _digest(f"{model}-src"),
                "src_file_count": 1,
                "runtime_import_roots": [
                    {
                        "path": f"{model}/src",
                        "tree_sha256": _digest(f"{model}-src"),
                        "file_count": 1,
                    },
                    *(
                        [
                            {
                                "path": "assets/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
                                "tree_sha256": _digest("depthsplat-dinov2"),
                                "file_count": 1,
                            }
                        ]
                        if model == "depthsplat"
                        else []
                    ),
                ],
            },
            "profile_interpreter": {
                "executable_sha256": _digest(f"python-{model}"),
                "python_version": "3.10.0",
            },
            "resolved_config_sha256": _digest(f"resolved-{model}"),
        }
    preprocessing = {
        **deepcopy(PREPROCESSING_CONTRACT),
        "crop_shim_sha256": {
            model: _digest(f"crop-{model}") for model in MODELS
        },
        "patch_shim_sha256": {
            model: _digest(f"patch-{model}") for model in MODELS
        },
    }
    execution_identity = build_saes_execution_identity()
    routing = {
        "execution_identity": execution_identity,
        "execution_route_sha256": execution_identity["route_sha256"],
        "parameters": deepcopy(SAES_ROUTING_PARAMETERS),
        "per_model": deepcopy(SAES_MODEL_ROUTING),
        "mechanism_config_path": "artifact/mechanism_config.json",
        "mechanism_config_sha256": _digest("mechanism"),
        "implementation_path": "saes/progressive_saes.py",
        "implementation_sha256": _digest("implementation"),
    }
    routing["routing_sha256"] = canonical_sha256(routing)
    implementation_files = [
        {"path": path, "sha256": _digest(f"implementation-{ordinal}")}
        for ordinal, path in enumerate(IMPLEMENTATION_PATHS)
    ]
    implementation_binding = {
        "files": implementation_files,
        "implementation_tree_sha256": canonical_sha256(implementation_files),
    }
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "status": CONTRACT_STATUS,
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "author_side_prerequisite": True,
        "paper_result_eligible": False,
        "mechanism_config_write_allowed": False,
        "dl3dv_quality_gate_authorized": False,
        "source_plan": {
            "path": "artifact/protocol/acid_joint_calibration_plan.json",
            "file_sha256": _digest("q"),
            "plan_sha256": source_plan_sha,
            "protocol_id": PROTOCOL_ID,
        },
        "context_only_inputs": {
            "materialization_root": "outputs/calibration/acid_joint_calibration_v1_context_only",
            "outer_tree_sha256": _digest("r"),
            "outer_manifest_sha256": _digest("s"),
            "materialization_sha256": _digest("t"),
            "splits": split_identity,
        },
        "checkpoint_binding": {
            "checkpoint_manifest_path": "artifact/manifests/checkpoints.json",
            "checkpoint_manifest_sha256": _digest("u"),
            "models": checkpoints,
            "model_order": list(MODELS),
            "shared_asset_model_independent": True,
        },
        "model_config_binding": configs,
        "preprocessing": preprocessing,
        "saes_routing": routing,
        "implementation_binding": implementation_binding,
        "shared_asset": {
            "asset_path": DEFAULT_ASSET_PATH,
            "single_global_asset_required": True,
            "per_model_or_dataset_assets_forbidden": True,
            "runtime_hash_pinning_required": True,
            "architecture": deepcopy(CALIBRATOR_ARCHITECTURE),
        },
        "result_records": deepcopy(RESULT_RECORD_PATHS),
        "runtime_evidence_records": deepcopy(RUNTIME_EVIDENCE_RECORDS),
        "candidate_verification_policy": deepcopy(CANDIDATE_VERIFICATION_POLICY),
        "optimization_recipe": deepcopy(OPTIMIZATION_RECIPE),
        "teacher_objective": deepcopy(TEACHER_OBJECTIVE),
        "teacher_fidelity_gates": deepcopy(TEACHER_FIDELITY_GATES),
        "holdout_policy": deepcopy(HOLDOUT_POLICY),
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def _result(contract: dict, *, stage: str, relative_mse: float, asset_sha: str = "v") -> dict:
    from data.acid_joint_training_contract import (
        ACCESS_AUDIT_KIND,
        CALIBRATOR_ARCHITECTURE,
        CONTRACT_SCHEMA_VERSION,
        DEFAULT_ASSET_PATH,
        MODELS,
        RESULT_KIND,
        RUNTIME_PRECONDITIONS,
    )

    models = {}
    for model in MODELS:
        models[model] = {
            "checkpoint_sha256": contract["checkpoint_binding"]["models"][model][
                "checkpoint_sha256"
            ],
            "config_source_sha256": contract["model_config_binding"][model][
                "source_manifest_sha256"
            ],
            "runtime_source": deepcopy(
                contract["model_config_binding"][model]["runtime_source"]
            ),
            "environment_profile": contract["model_config_binding"][model][
                "environment_profile"
            ],
            "profile_interpreter": deepcopy(
                contract["model_config_binding"][model]["profile_interpreter"]
            ),
            "resolved_config_sha256": contract["model_config_binding"][model][
                "resolved_config_sha256"
            ],
            "preprocessing_id": contract["preprocessing"]["identifier"],
            "prepared_patch_size": contract["preprocessing"]["per_model_patch_size"][model],
            "saes_routing_sha256": contract["saes_routing"]["routing_sha256"],
            "saes_execution_route_sha256": contract["saes_routing"][
                "execution_route_sha256"
            ],
            "input_identity": {
                key: contract["context_only_inputs"]["splits"][stage][key]
                for key in (
                    "tree_sha256",
                    "manifest_sha256",
                    "input_provenance_sha256",
                    "selection_sha256",
                    "scene_count",
                )
            },
            "metrics": {
                component: {"identity_mse": 1.0, "calibrated_mse": relative_mse}
                for component in ("mean", "covariance", "opacity", "sh")
            },
            "controls": dict(RUNTIME_PRECONDITIONS),
            "runtime_evidence": {
                "path": contract["runtime_evidence_records"][stage][model],
                "sha256": _digest(f"runtime-evidence-{stage}-{model}"),
                "selected_head_sha256": _digest(f"selected-head-{stage}-{model}"),
                "ledger_sha256": _digest(f"ledger-{stage}-{model}"),
                "asset_sha256": _digest(asset_sha),
                "asset_state_sha256": _digest("w"),
            },
        }
    is_train = stage == "calibration_train"
    candidate_execution_verification = (
        {
            "provenance": {
                "path": "outputs/calibration/acid_joint_materialization_training_v5/frozen_joint_materialization_calibrator.pt.candidate.provenance.json",
                "sha256": _digest("candidate-provenance"),
                "candidate_asset_sha256": _digest(asset_sha),
                "candidate_asset_state_sha256": _digest("w"),
                "cache_manifests": {
                    model: {
                        "path": f"outputs/calibration/acid_joint_materialization_training_v5/teacher_cache/{model}/calibration_train/manifest.json",
                        "sha256": _digest(f"cache-{model}"),
                    }
                    for model in MODELS
                },
                "local_hash_chain_is_not_cryptographic_proof": True,
            },
            "verification": {
                "path": "outputs/calibration/acid_joint_materialization_training_v5/frozen_joint_materialization_calibrator.pt.candidate.verification.json",
                "sha256": _digest("candidate-verification"),
                "verification_mode": "deterministic_replay_verifier",
                "candidate_provenance_path": "outputs/calibration/acid_joint_materialization_training_v5/frozen_joint_materialization_calibrator.pt.candidate.provenance.json",
                "candidate_provenance_sha256": _digest("candidate-provenance"),
                "candidate_asset_sha256": _digest(asset_sha),
            },
        }
        if is_train
        else None
    )
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "stage": stage,
        "frozen_train_result_sha256": None if is_train else _digest("frozen-train-result"),
        "asset": {
            "asset_path": DEFAULT_ASSET_PATH,
            "schema_version": CALIBRATOR_ARCHITECTURE["schema_version"],
            "kind": CALIBRATOR_ARCHITECTURE["kind"],
            "sha256": _digest(asset_sha),
            "state_sha256": _digest("w"),
            "byte_count": 4096,
            "descriptor_dim": CALIBRATOR_ARCHITECTURE["descriptor_dim"],
            "bottleneck_dim": CALIBRATOR_ARCHITECTURE["bottleneck_dim"],
            "joint_output_dim": CALIBRATOR_ARCHITECTURE["joint_output_dim"],
        },
        "candidate_execution_verification": candidate_execution_verification,
        "models": models,
        "access_audit": {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "kind": ACCESS_AUDIT_KIND,
            "plan_sha256": contract["source_plan"]["plan_sha256"],
            "stage": "train" if is_train else "holdout",
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "expected_results_accessed": False,
            "evaluation_scene_accessed": False,
            "descriptor_model_id_accessed": False,
            "descriptor_dataset_id_accessed": False,
            "teacher_files_opened": True,
            "teacher_files_runtime_accessible": False,
            "runtime_teacher_path": None,
            "optimizer_executed": is_train,
            "asset_updated": is_train,
            "rerank_executed": False,
            "partition_reshuffled": False,
        },
    }


def test_static_contract_rejects_a_changed_optimizer_or_teacher_boundary():
    from data.acid_joint_training_contract import (
        JointTrainingContractError,
        validate_training_contract,
    )
    from scripts.calibration_contract import canonical_sha256

    contract = _contract()
    identity = validate_training_contract(contract)
    assert identity["status"] == "FROZEN_PENDING_EXECUTION"
    assert identity["checkpoint_models"] == ["transplat", "mvsplat", "depthsplat"]

    changed_schedule = deepcopy(contract)
    changed_schedule["optimization_recipe"]["total_updates"] += 1
    changed_schedule["contract_sha256"] = canonical_sha256(
        {key: value for key, value in changed_schedule.items() if key != "contract_sha256"}
    )
    with pytest.raises(JointTrainingContractError, match="optimizer schedule"):
        validate_training_contract(changed_schedule)

    changed_teacher = deepcopy(contract)
    changed_teacher["teacher_objective"]["target_rgb_or_ground_truth_allowed"] = True
    changed_teacher["contract_sha256"] = canonical_sha256(
        {key: value for key, value in changed_teacher.items() if key != "contract_sha256"}
    )
    with pytest.raises(JointTrainingContractError, match="teacher objective"):
        validate_training_contract(changed_teacher)

    changed_route = deepcopy(contract)
    changed_route["saes_routing"]["per_model"]["depthsplat"]["ray_depth_mode"] = "euclidean"
    changed_route["saes_routing"]["routing_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed_route["saes_routing"].items()
            if key != "routing_sha256"
        }
    )
    changed_route["contract_sha256"] = canonical_sha256(
        {key: value for key, value in changed_route.items() if key != "contract_sha256"}
    )
    with pytest.raises(JointTrainingContractError, match="SAES routing"):
        validate_training_contract(changed_route)


def test_model_config_binding_freezes_runtime_source_and_resolved_config(tmp_path, monkeypatch):
    import data.acid_joint_training_contract as module

    expected_digests = {
        model: _digest(f"resolved-{model}") for model in module.MODELS
    }

    for model, request in module.MODEL_CONFIG_REQUESTS.items():
        for relative in request["source_paths"]:
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture:{relative}\n", encoding="utf-8")

    def source_binding(_root, model):
        roots = [
            {
                "path": f"{model}/src",
                "tree_sha256": _digest(f"{model}-source"),
                "file_count": 3,
            }
        ]
        if model == "depthsplat":
            roots.append(
                {
                    "path": "assets/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
                    "tree_sha256": _digest("fixture-dinov2"),
                    "file_count": 2,
                }
            )
        return {
            "repository_path": model,
            "repository_commit": "a" * 40,
            "src_path": f"{model}/src",
            "src_tree_sha256": _digest(f"{model}-source"),
            "src_file_count": 3,
            "runtime_import_roots": roots,
        }

    monkeypatch.setattr(module, "_resolved_model_config_sha256s", lambda _root: expected_digests)
    monkeypatch.setattr(module, "_model_source_binding", source_binding)
    monkeypatch.setattr(
        module,
        "_profile_interpreter_binding",
        lambda profile: {
            "executable_sha256": _digest(f"fixture-{profile}"),
            "python_version": "3.10.0",
        },
    )

    binding = module._model_config_records(tmp_path)

    for model in module.MODELS:
        assert binding[model]["runtime_source"] == source_binding(tmp_path, model)
        assert binding[model]["resolved_config_sha256"] == expected_digests[model]
        assert binding[model]["environment_profile"] == module.MODEL_ENVIRONMENT_PROFILES[model]
        assert binding[model]["profile_interpreter"]["python_version"] == "3.10.0"


def test_runtime_model_binding_rejects_source_or_resolved_config_drift(monkeypatch):
    import data.acid_joint_training_contract as module

    contract = _contract()
    expected = contract["model_config_binding"]["transplat"]
    monkeypatch.setattr(
        module, "_require_canonical_live_contract", lambda value, **_kwargs: dict(value)
    )
    monkeypatch.setattr(
        module,
        "_model_source_binding",
        lambda _root, _model: deepcopy(expected["runtime_source"]),
    )
    monkeypatch.setattr(
        module,
        "_model_config_source_files",
        lambda _root, _model: deepcopy(expected["source_files"]),
    )
    monkeypatch.setattr(
        module,
        "_runtime_profile_interpreter_binding",
        lambda _profile: deepcopy(expected["profile_interpreter"]),
    )

    assert module.validate_runtime_model_binding(
        contract,
        model="transplat",
        resolved_config_sha256=expected["resolved_config_sha256"],
    )["resolved_config_sha256"] == expected["resolved_config_sha256"]

    changed_source = deepcopy(expected["runtime_source"])
    changed_source["src_tree_sha256"] = _digest("changed-source")
    monkeypatch.setattr(module, "_model_source_binding", lambda _root, _model: changed_source)
    with pytest.raises(module.JointTrainingContractError, match="runtime source identity"):
        module.validate_runtime_model_binding(
            contract,
            model="transplat",
            resolved_config_sha256=expected["resolved_config_sha256"],
        )

    monkeypatch.setattr(
        module,
        "_model_source_binding",
        lambda _root, _model: deepcopy(expected["runtime_source"]),
    )
    changed_files = deepcopy(expected["source_files"])
    changed_files[0]["sha256"] = _digest("changed-source-config")
    monkeypatch.setattr(module, "_model_config_source_files", lambda _root, _model: changed_files)
    with pytest.raises(module.JointTrainingContractError, match="config sources"):
        module.validate_runtime_model_binding(
            contract,
            model="transplat",
            resolved_config_sha256=expected["resolved_config_sha256"],
        )

    monkeypatch.setattr(
        module,
        "_model_config_source_files",
        lambda _root, _model: deepcopy(expected["source_files"]),
    )
    with pytest.raises(module.JointTrainingContractError, match="resolved configuration"):
        module.validate_runtime_model_binding(
            contract,
            model="transplat",
            resolved_config_sha256=_digest("changed-config"),
        )


def test_depthsplat_runtime_source_rehashes_the_pinned_dinov2_import_tree(
    tmp_path, monkeypatch
):
    import data.acid_joint_training_contract as module
    from types import SimpleNamespace

    source_root = tmp_path / "depthsplat" / "src"
    dino_root = (
        tmp_path
        / "assets"
        / "torch"
        / "hub"
        / "facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    )
    source_root.mkdir(parents=True)
    dino_root.mkdir(parents=True)
    (source_root / "encoder.py").write_text("encoder = 1\n", encoding="utf-8")
    (dino_root / "hubconf.py").write_text("model = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "MODEL_RUNTIME_IMPORT_ROOTS",
        {
            **module.MODEL_RUNTIME_IMPORT_ROOTS,
            "depthsplat": (
                "depthsplat/src",
                "assets/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
            ),
        },
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="a" * 40),
    )

    baseline = module._model_source_binding(tmp_path, "depthsplat")
    (dino_root / "hubconf.py").write_text("model = 2\n", encoding="utf-8")
    drifted = module._model_source_binding(tmp_path, "depthsplat")

    assert len(baseline["runtime_import_roots"]) == 2
    assert baseline["runtime_import_roots"][1]["path"].startswith("assets/torch/hub/")
    assert drifted != baseline


def test_runtime_model_binding_accepts_a_constrained_namespace_src_package(monkeypatch):
    import data.acid_joint_training_contract as module
    from types import SimpleNamespace

    contract = _contract()
    expected = contract["model_config_binding"]["transplat"]
    monkeypatch.setattr(
        module, "_require_canonical_live_contract", lambda value, **_kwargs: dict(value)
    )
    monkeypatch.setattr(
        module,
        "_model_source_binding",
        lambda _root, _model: deepcopy(expected["runtime_source"]),
    )
    monkeypatch.setattr(
        module,
        "_model_config_source_files",
        lambda _root, _model: deepcopy(expected["source_files"]),
    )
    monkeypatch.setattr(
        module,
        "_runtime_profile_interpreter_binding",
        lambda _profile: deepcopy(expected["profile_interpreter"]),
    )
    expected_src = module.ROOT / expected["runtime_source"]["src_path"]
    monkeypatch.setattr(
        module.sys,
        "modules",
        {"src": SimpleNamespace(__path__=[str(expected_src)])},
    )

    assert module.validate_runtime_model_binding(
        contract,
        model="transplat",
        resolved_config_sha256=expected["resolved_config_sha256"],
        require_loaded_src=True,
    )["resolved_config_sha256"] == expected["resolved_config_sha256"]


@pytest.mark.parametrize(
    "namespace_paths",
    [
        [],
        ["/tmp/external-src"],
        [
            str(Path(__file__).resolve().parents[1] / "transplat" / "src"),
            "/tmp/external-src",
        ],
    ],
)
def test_runtime_model_binding_rejects_unconstrained_src_namespace_paths(
    monkeypatch, namespace_paths
):
    import data.acid_joint_training_contract as module
    from types import SimpleNamespace

    contract = _contract()
    expected = contract["model_config_binding"]["transplat"]
    monkeypatch.setattr(
        module, "_require_canonical_live_contract", lambda value, **_kwargs: dict(value)
    )
    monkeypatch.setattr(
        module,
        "_model_source_binding",
        lambda _root, _model: deepcopy(expected["runtime_source"]),
    )
    monkeypatch.setattr(
        module,
        "_model_config_source_files",
        lambda _root, _model: deepcopy(expected["source_files"]),
    )
    monkeypatch.setattr(
        module,
        "_runtime_profile_interpreter_binding",
        lambda _profile: deepcopy(expected["profile_interpreter"]),
    )
    monkeypatch.setattr(
        module.sys,
        "modules",
        {"src": SimpleNamespace(__path__=namespace_paths)},
    )

    with pytest.raises(module.JointTrainingContractError, match="constrained file or namespace path|escaped"):
        module.validate_runtime_model_binding(
            contract,
            model="transplat",
            resolved_config_sha256=expected["resolved_config_sha256"],
            require_loaded_src=True,
        )


def test_runtime_model_binding_rejects_cached_src_from_another_model(monkeypatch):
    import data.acid_joint_training_contract as module
    from types import SimpleNamespace

    contract = _contract()
    expected = contract["model_config_binding"]["transplat"]
    monkeypatch.setattr(
        module, "_require_canonical_live_contract", lambda value, **_kwargs: dict(value)
    )
    monkeypatch.setattr(
        module,
        "_model_source_binding",
        lambda _root, _model: deepcopy(expected["runtime_source"]),
    )
    monkeypatch.setattr(
        module,
        "_model_config_source_files",
        lambda _root, _model: deepcopy(expected["source_files"]),
    )
    monkeypatch.setattr(
        module,
        "_runtime_profile_interpreter_binding",
        lambda _profile: deepcopy(expected["profile_interpreter"]),
    )
    monkeypatch.setattr(
        module.sys,
        "modules",
        {"src": SimpleNamespace(__file__=str(module.ROOT / "mvsplat/src/encoder.py"))},
    )

    with pytest.raises(module.JointTrainingContractError, match="escaped the declared model source root"):
        module.validate_runtime_model_binding(
            contract,
            model="transplat",
            resolved_config_sha256=expected["resolved_config_sha256"],
            require_loaded_src=True,
        )


def test_runtime_model_binding_rejects_depthsplat_dinov2_tree_drift(monkeypatch):
    import data.acid_joint_training_contract as module

    contract = _contract()
    expected = contract["model_config_binding"]["depthsplat"]
    monkeypatch.setattr(
        module, "_require_canonical_live_contract", lambda value, **_kwargs: dict(value)
    )
    drifted_source = deepcopy(expected["runtime_source"])
    drifted_source["runtime_import_roots"][1]["tree_sha256"] = _digest(
        "changed-dinov2-tree"
    )
    monkeypatch.setattr(
        module, "_model_source_binding", lambda _root, _model: drifted_source
    )
    monkeypatch.setattr(
        module,
        "_model_config_source_files",
        lambda _root, _model: deepcopy(expected["source_files"]),
    )
    monkeypatch.setattr(
        module,
        "_runtime_profile_interpreter_binding",
        lambda _profile: deepcopy(expected["profile_interpreter"]),
    )

    with pytest.raises(module.JointTrainingContractError, match="runtime source identity"):
        module.validate_runtime_model_binding(
            contract,
            model="depthsplat",
            resolved_config_sha256=expected["resolved_config_sha256"],
        )


def test_candidate_provenance_binds_contract_caches_and_updates_but_verification_stays_blocked():
    from data.acid_joint_training_contract import (
        CANDIDATE_PROVENANCE_KIND,
        CANDIDATE_PROVENANCE_SCHEMA_VERSION,
        CANDIDATE_VERIFICATION_KIND,
        CANDIDATE_VERIFICATION_SCHEMA_VERSION,
        JointTrainingContractError,
        candidate_asset_path_for_contract,
        candidate_provenance_path_for_contract,
        validate_candidate_provenance,
        validate_candidate_verification,
    )
    from scripts.calibration_contract import canonical_sha256

    contract = _contract()
    asset = _result(contract, stage="calibration_train", relative_mse=0.5)["asset"]
    cache_manifests = {
        model: {
            "path": (
                f"{contract['teacher_objective']['teacher_cache_root']}/"
                f"{model}/calibration_train/manifest.json"
            ),
            "sha256": _digest(f"candidate-cache-{model}"),
        }
        for model in contract["checkpoint_binding"]["model_order"]
    }
    provenance = {
        "schema_version": CANDIDATE_PROVENANCE_SCHEMA_VERSION,
        "kind": CANDIDATE_PROVENANCE_KIND,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "candidate_asset_path": candidate_asset_path_for_contract(contract),
        "candidate_asset_sha256": asset["sha256"],
        "candidate_asset_state_sha256": asset["state_sha256"],
        "candidate_asset_byte_count": asset["byte_count"],
        "cache_manifests": cache_manifests,
        "optimization_recipe_sha256": canonical_sha256(contract["optimization_recipe"]),
        "seed": contract["optimization_recipe"]["seed"],
        "total_updates": contract["optimization_recipe"]["total_updates"],
        "local_hash_chain_is_not_cryptographic_proof": True,
        "verification_status": "BLOCKED_NO_VERIFIED_EXECUTION",
    }
    identity = validate_candidate_provenance(
        provenance,
        contract=contract,
        candidate_asset=asset,
        cache_manifests=cache_manifests,
    )
    assert identity["candidate_asset_sha256"] == asset["sha256"]

    changed_updates = deepcopy(provenance)
    changed_updates["total_updates"] += 1
    with pytest.raises(JointTrainingContractError, match="frozen execution"):
        validate_candidate_provenance(
            changed_updates,
            contract=contract,
            candidate_asset=asset,
            cache_manifests=cache_manifests,
        )

    changed_cache = deepcopy(cache_manifests)
    changed_cache["transplat"]["sha256"] = _digest("changed-cache")
    with pytest.raises(JointTrainingContractError, match="cache manifests changed"):
        validate_candidate_provenance(
            provenance,
            contract=contract,
            candidate_asset=asset,
            cache_manifests=changed_cache,
        )

    changed_contract = deepcopy(provenance)
    changed_contract["contract_sha256"] = _digest("changed-contract")
    with pytest.raises(JointTrainingContractError, match="frozen execution"):
        validate_candidate_provenance(
            changed_contract,
            contract=contract,
            candidate_asset=asset,
            cache_manifests=cache_manifests,
        )

    verification = {
        "schema_version": CANDIDATE_VERIFICATION_SCHEMA_VERSION,
        "kind": CANDIDATE_VERIFICATION_KIND,
        "verification_mode": "deterministic_replay_verifier",
        "candidate_provenance_path": candidate_provenance_path_for_contract(contract),
        "candidate_provenance_sha256": _digest("candidate-provenance-file"),
        "candidate_asset_sha256": asset["sha256"],
        "contract_sha256": contract["contract_sha256"],
        "cache_manifests": cache_manifests,
        "verifier_identity": {"implementation_sha256": _digest("untrusted-verifier")},
    }
    with pytest.raises(JointTrainingContractError, match="BLOCKED_NO_VERIFIED_EXECUTION"):
        validate_candidate_verification(
            verification,
            contract=contract,
            provenance_path=candidate_provenance_path_for_contract(contract),
            provenance_sha256=_digest("candidate-provenance-file"),
            provenance=identity,
        )


def test_v5_contract_binds_the_registered_twelve_anchor_guard_on_route():
    from data.acid_joint_training_contract import (
        DEFAULT_ASSET_PATH,
        DEFAULT_TRAINING_CONTRACT_PATH,
        DEFAULT_TRAINING_OUTPUT_ROOT,
        JointTrainingContractError,
        validate_training_contract,
    )
    from scripts.calibration_contract import canonical_sha256
    from scripts.saes_execution_identity import build_saes_execution_identity

    contract = _contract()
    identity = build_saes_execution_identity()

    assert DEFAULT_TRAINING_CONTRACT_PATH.name.endswith("_v5.json")
    assert "training_v5" in DEFAULT_TRAINING_OUTPUT_ROOT.as_posix()
    assert "training_v5" in DEFAULT_ASSET_PATH
    assert contract["saes_routing"]["execution_identity"] == identity
    assert contract["saes_routing"]["execution_route_sha256"] == identity[
        "route_sha256"
    ]
    assert contract["saes_routing"]["parameters"]["l1_retained_positions"] == 12
    assert contract["saes_routing"]["parameters"]["cross_check_threshold"] == 0.015
    assert contract["saes_routing"]["parameters"]["context_safety_guard"] is True

    changed = deepcopy(contract)
    changed["saes_routing"]["execution_identity"]["context_safety_guard"] = False
    changed["saes_routing"]["routing_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed["saes_routing"].items()
            if key != "routing_sha256"
        }
    )
    changed["contract_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "contract_sha256"}
    )
    with pytest.raises(JointTrainingContractError, match="execution route"):
        validate_training_contract(changed)


def test_v5_teacher_fidelity_rejects_inline_train_candidate_verification():
    from data.acid_joint_training_contract import (
        JointTrainingContractError,
        validate_teacher_fidelity_result,
    )

    contract = _contract()
    train = _result(contract, stage="calibration_train", relative_mse=0.9)
    with pytest.raises(
        JointTrainingContractError, match="provenance or verification sidecar"
    ):
        validate_teacher_fidelity_result(train, contract=contract)

    # A holdout cannot use an inline train object to skip the blocked train
    # candidate chain either.
    holdout = _result(contract, stage="calibration_holdout", relative_mse=0.95)
    with pytest.raises(
        JointTrainingContractError, match="provenance or verification sidecar"
    ):
        validate_teacher_fidelity_result(
            holdout, contract=contract, frozen_train_result=train
        )


def test_v5_rejects_a_syntactically_valid_inline_train_verifier_after_sidecar_rehash(
    tmp_path,
):
    """A complete local hash chain is still not a v5 execution proof."""

    import data.acid_joint_training_contract as module
    from scripts.calibration_contract import canonical_sha256

    contract = _contract()
    train = _result(contract, stage=module.TRAIN_SPLIT, relative_mse=0.9)
    asset = train["asset"]
    cache_manifests = {}
    for model in module.MODELS:
        relative = (
            f"{contract['teacher_objective']['teacher_cache_root']}/"
            f"{model}/{module.TRAIN_SPLIT}/manifest.json"
        )
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"fixture": true}\n', encoding="utf-8")
        cache_manifests[model] = {
            "path": relative,
            "sha256": module.sha256_file(path),
        }

    provenance = {
        "schema_version": module.CANDIDATE_PROVENANCE_SCHEMA_VERSION,
        "kind": module.CANDIDATE_PROVENANCE_KIND,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "candidate_asset_path": module.candidate_asset_path_for_contract(contract),
        "candidate_asset_sha256": asset["sha256"],
        "candidate_asset_state_sha256": asset["state_sha256"],
        "candidate_asset_byte_count": asset["byte_count"],
        "cache_manifests": cache_manifests,
        "optimization_recipe_sha256": canonical_sha256(contract["optimization_recipe"]),
        "seed": contract["optimization_recipe"]["seed"],
        "total_updates": contract["optimization_recipe"]["total_updates"],
        "local_hash_chain_is_not_cryptographic_proof": True,
        "verification_status": "BLOCKED_NO_VERIFIED_EXECUTION",
    }
    provenance_relative = module.candidate_provenance_path_for_contract(contract)
    provenance_path = tmp_path / provenance_relative
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    provenance_sha256 = module.sha256_file(provenance_path)

    verification = {
        "schema_version": module.CANDIDATE_VERIFICATION_SCHEMA_VERSION,
        "kind": module.CANDIDATE_VERIFICATION_KIND,
        "verification_mode": "deterministic_replay_verifier",
        "candidate_provenance_path": provenance_relative,
        "candidate_provenance_sha256": provenance_sha256,
        "candidate_asset_sha256": asset["sha256"],
        "contract_sha256": contract["contract_sha256"],
        "cache_manifests": cache_manifests,
        "verifier_identity": {"implementation_sha256": _digest("forged-verifier")},
    }
    verification_relative = module.candidate_verification_path_for_contract(contract)
    verification_path = tmp_path / verification_relative
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    verification_sha256 = module.sha256_file(verification_path)
    train["candidate_execution_verification"] = {
        "provenance": {
            "path": provenance_relative,
            "sha256": provenance_sha256,
            "candidate_asset_sha256": asset["sha256"],
            "candidate_asset_state_sha256": asset["state_sha256"],
            "cache_manifests": cache_manifests,
            "local_hash_chain_is_not_cryptographic_proof": True,
        },
        "verification": {
            "path": verification_relative,
            "sha256": verification_sha256,
            "verification_mode": "deterministic_replay_verifier",
            "candidate_provenance_path": provenance_relative,
            "candidate_provenance_sha256": provenance_sha256,
            "candidate_asset_sha256": asset["sha256"],
        },
    }

    with pytest.raises(
        module.JointTrainingContractError, match="BLOCKED_NO_VERIFIED_EXECUTION"
    ):
        module.validate_teacher_fidelity_result(
            train, contract=contract, repository_root=tmp_path
        )


def test_holdout_cannot_execute_an_optimizer_or_omit_selected_only_controls():
    from data.acid_joint_training_contract import (
        JointTrainingContractError,
        validate_teacher_fidelity_result,
    )

    contract = _contract()
    train = _result(contract, stage="calibration_train", relative_mse=0.85)
    holdout = _result(contract, stage="calibration_holdout", relative_mse=0.9)
    holdout["access_audit"]["optimizer_executed"] = True
    with pytest.raises(JointTrainingContractError, match="holdout attempted"):
        validate_teacher_fidelity_result(
            holdout, contract=contract, frozen_train_result=train
        )

    missing_sentinel = _result(contract, stage="calibration_train", relative_mse=0.85)
    missing_sentinel["models"]["depthsplat"]["controls"][
        "two_finite_skipped_s3_sentinels_required"
    ] = False
    with pytest.raises(JointTrainingContractError, match="provenance or verification sidecar"):
        validate_teacher_fidelity_result(missing_sentinel, contract=contract)


def test_implementation_binding_rehashes_every_declared_transitive_module(tmp_path):
    import data.acid_joint_training_contract as module

    for relative in module.IMPLEMENTATION_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")

    baseline = module._implementation_binding(tmp_path)
    assert [item["path"] for item in baseline["files"]] == list(
        module.IMPLEMENTATION_PATHS
    )
    assert {
        "data/plan_acid_joint_calibration.py",
        "data/frozen_audit_contract.py",
        "scripts/calibration_contract.py",
        "scripts/acid_joint_calibration_runner.py",
        "scripts/acid_joint_runtime_control_evidence.py",
    } <= set(module.IMPLEMENTATION_PATHS)

    changed = tmp_path / "scripts" / "calibration_contract.py"
    changed.write_text("fixture:changed\n", encoding="utf-8")
    assert module._implementation_binding(tmp_path) != baseline


def test_live_preflight_rejects_a_rehashed_implementation_mutation(tmp_path, monkeypatch):
    import data.acid_joint_training_contract as module
    from scripts.calibration_contract import canonical_sha256

    root = tmp_path.resolve()
    plan_path = root / "artifact" / "protocol" / "acid_joint_calibration_plan.json"
    contract_path = root / "artifact" / "protocol" / "acid_joint_materialization_training_contract_v5.json"
    checkpoint_path = root / "artifact" / "manifests" / "checkpoints.json"
    materialization_root = root / "outputs" / "calibration" / "acid_joint_calibration_v1_context_only"
    plan_path.parent.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True)
    materialization_root.mkdir(parents=True)
    plan_path.write_text("{}", encoding="utf-8")
    contract_path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_text("{}", encoding="utf-8")
    for relative in module.IMPLEMENTATION_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "DEFAULT_TRAINING_CONTRACT_PATH", contract_path)
    monkeypatch.setattr(module, "DEFAULT_PLAN_PATH", plan_path)
    monkeypatch.setattr(module, "DEFAULT_CHECKPOINT_MANIFEST", checkpoint_path)
    monkeypatch.setattr(module, "DEFAULT_MATERIALIZATION_ROOT", materialization_root)
    monkeypatch.setattr(module, "validate_plan", lambda _plan: None)
    monkeypatch.setattr(module, "require_fixed_file_sha256", lambda *_args, **_kwargs: "ok")

    contract = _contract()
    contract["implementation_binding"] = module._implementation_binding(root)
    contract["contract_sha256"] = canonical_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )

    def read_json(path, _label):
        if Path(path).resolve() == contract_path:
            return contract
        return {"plan_sha256": _digest("a"), "protocol_id": module.PROTOCOL_ID}

    monkeypatch.setattr(module, "_read_json", read_json)
    monkeypatch.setattr(
        module,
        "_context_input_binding",
        lambda *_args, **_kwargs: contract["context_only_inputs"],
    )
    monkeypatch.setattr(
        module,
        "_checkpoint_records",
        lambda *_args, **_kwargs: contract["checkpoint_binding"]["models"],
    )
    monkeypatch.setattr(
        module,
        "_model_config_records",
        lambda *_args, **_kwargs: contract["model_config_binding"],
    )
    monkeypatch.setattr(
        module,
        "_preprocessing_binding",
        lambda *_args, **_kwargs: contract["preprocessing"],
    )
    monkeypatch.setattr(
        module,
        "_saes_routing_binding",
        lambda *_args, **_kwargs: contract["saes_routing"],
    )

    assert module.validate_live_training_contract(contract, repository_root=root)[
        "status"
    ] == "PASS_PREOPTIMIZATION_REHASH"

    (root / "scripts" / "acid_joint_calibration_runner.py").write_text(
        "mutated\n", encoding="utf-8"
    )
    with pytest.raises(
        module.JointTrainingContractError, match="training implementation identity"
    ):
        module.validate_live_training_contract(contract, repository_root=root)


def _install_canonical_contract_fixture(module, root, contract, monkeypatch):
    contract_path = (
        root
        / "artifact"
        / "protocol"
        / "acid_joint_materialization_training_contract_v5.json"
    )
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "DEFAULT_TRAINING_CONTRACT_PATH", contract_path)
    monkeypatch.setattr(
        module,
        "DEFAULT_PLAN_PATH",
        root / "artifact" / "protocol" / "acid_joint_calibration_plan.json",
    )
    monkeypatch.setattr(
        module,
        "DEFAULT_CHECKPOINT_MANIFEST",
        root / "artifact" / "manifests" / "checkpoints.json",
    )
    monkeypatch.setattr(
        module,
        "DEFAULT_MATERIALIZATION_ROOT",
        root / "outputs" / "calibration" / "acid_joint_calibration_v1_context_only",
    )
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path


def test_live_teacher_fidelity_rejects_a_result_object_that_differs_from_its_canonical_file(
    tmp_path, monkeypatch
):
    import data.acid_joint_training_contract as module

    root = tmp_path.resolve()
    contract = _contract()
    _install_canonical_contract_fixture(module, root, contract, monkeypatch)
    monkeypatch.setattr(module, "validate_live_training_contract", lambda *_args, **_kwargs: {})
    canonical_result = _result(contract, stage=module.TRAIN_SPLIT, relative_mse=0.9)
    result_path = root / contract["result_records"][module.TRAIN_SPLIT]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(canonical_result), encoding="utf-8")
    caller_result = deepcopy(canonical_result)
    caller_result["plan_sha256"] = _digest("caller-forged-result")

    with pytest.raises(
        module.JointTrainingContractError,
        match="fidelity result object differs from its canonical file",
    ):
        module.validate_live_teacher_fidelity_result(
            caller_result,
            result_path=result_path,
            contract=contract,
            repository_root=root,
        )


def test_live_holdout_rejects_a_fake_train_and_recursively_blocks_the_canonical_train(
    tmp_path, monkeypatch
):
    import data.acid_joint_training_contract as module

    root = tmp_path.resolve()
    contract = _contract()
    _install_canonical_contract_fixture(module, root, contract, monkeypatch)
    monkeypatch.setattr(module, "validate_live_training_contract", lambda *_args, **_kwargs: {})
    canonical_train = _result(contract, stage=module.TRAIN_SPLIT, relative_mse=0.9)
    canonical_holdout = _result(contract, stage=module.HOLDOUT_SPLIT, relative_mse=0.9)
    train_path = root / contract["result_records"][module.TRAIN_SPLIT]
    holdout_path = root / contract["result_records"][module.HOLDOUT_SPLIT]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(canonical_train), encoding="utf-8")
    holdout_path.write_text(json.dumps(canonical_holdout), encoding="utf-8")
    fake_train = deepcopy(canonical_train)
    fake_train["asset"]["sha256"] = _digest("caller-forged-train")

    with pytest.raises(
        module.JointTrainingContractError,
        match="frozen training-result object differs from its canonical file",
    ):
        module.validate_live_teacher_fidelity_result(
            canonical_holdout,
            result_path=holdout_path,
            contract=contract,
            repository_root=root,
            frozen_train_result=fake_train,
            frozen_train_result_path=train_path,
        )

    with pytest.raises(
        module.JointTrainingContractError,
        match="provenance or verification sidecar is unavailable",
    ):
        module.validate_live_teacher_fidelity_result(
            canonical_holdout,
            result_path=holdout_path,
            contract=contract,
            repository_root=root,
            frozen_train_result=canonical_train,
            frozen_train_result_path=train_path,
        )


def test_runtime_checkpoint_binding_rejects_a_post_load_checkpoint_mutation(
    tmp_path, monkeypatch
):
    import data.acid_joint_training_contract as module
    from scripts.calibration_contract import canonical_sha256

    root = tmp_path.resolve()
    checkpoint = root / "transplat" / "checkpoints" / "acid.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"before")
    contract = _contract()
    binding = contract["checkpoint_binding"]["models"]["transplat"]
    binding["checkpoint_size"] = checkpoint.stat().st_size
    binding["checkpoint_sha256"] = module.sha256_file(checkpoint)
    contract["contract_sha256"] = canonical_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(
        module,
        "DEFAULT_PLAN_PATH",
        root / "artifact" / "protocol" / "acid_joint_calibration_plan.json",
    )
    monkeypatch.setattr(
        module,
        "DEFAULT_CHECKPOINT_MANIFEST",
        root / "artifact" / "manifests" / "checkpoints.json",
    )
    monkeypatch.setattr(
        module,
        "DEFAULT_MATERIALIZATION_ROOT",
        root / "outputs" / "calibration" / "acid_joint_calibration_v1_context_only",
    )

    assert module.validate_runtime_checkpoint_binding(
        contract,
        model="transplat",
        checkpoint_path=checkpoint,
        repository_root=root,
    ) == binding["checkpoint_sha256"]

    checkpoint.write_bytes(b"after!")
    with pytest.raises(RuntimeError, match="SHA256 does not match"):
        module.validate_runtime_checkpoint_binding(
            contract,
            model="transplat",
            checkpoint_path=checkpoint,
            repository_root=root,
        )
