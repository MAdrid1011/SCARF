"""Focused synthetic gates for the ACID joint-calibration runner.

These tests deliberately use fabricated cache tensors only. They never open an
ACID sidecar, model checkpoint, target image, or teacher cache from a real run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _contract() -> dict:
    from test_acid_joint_training_contract import _contract as build_contract

    return build_contract()


def _example(*, anchor_index: int, scale: float = 1.0) -> dict:
    from data.acid_joint_training_contract import (
        TEACHER_CACHE_EXAMPLE_KIND,
        TEACHER_CACHE_SCHEMA_VERSION,
    )

    compact = {
        "means": torch.tensor([0.1, 0.2, 0.3]),
        "covariances": torch.eye(3),
        "harmonics": torch.zeros(3, 1),
        "opacities": torch.tensor([0.2]),
    }
    teacher = {
        "means": compact["means"] + scale,
        "covariances": compact["covariances"] + torch.eye(3) * scale,
        "harmonics": compact["harmonics"] + scale,
        "opacities": compact["opacities"] + scale * 0.01,
    }
    return {
        "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "kind": TEACHER_CACHE_EXAMPLE_KIND,
        "descriptor": torch.zeros(32),
        "compact": compact,
        "teacher": teacher,
        "level": "L0",
        "anchor_index": anchor_index,
    }


def _routing(
    *,
    example_count: int,
    cross_check_threshold: float = 0.015,
    execution_route_sha256: str,
) -> dict:
    summary = {
        "level0_tiles": 1,
        "level1_tiles": 0,
        "full_tiles": 0,
        "level0_pixels": 4,
        "level1_pixels": 0,
        "l0_representatives": example_count,
        "l1_lightweight_anchors": 0,
        "full_stage3_gaussians": 0,
    }
    controls = {
        "route_mask_unchanged_required": True,
        "retained_counts_unchanged_required": True,
        "full_passthrough_required": True,
        "selected_only_s3_required": True,
        "two_finite_skipped_s3_sentinels_required": True,
        "route_mask_unchanged": True,
        "retained_counts_unchanged": True,
        "operational_stats_unchanged": True,
        "tile_trace_unchanged": True,
        "ordinary_representative_output_unchanged": True,
        "full_passthrough": True,
        "selected_only_s3": True,
        "two_finite_skipped_s3_sentinels_passed": True,
        "full_slot_count": 0,
        "sparse_packet_count": example_count,
    }
    return {
        "mask_sha256": _digest("mask"),
        "route_summary": summary,
        "level0_tiles": 1,
        "level1_tiles": 0,
        "full_tiles": 0,
        "retained_representatives": example_count,
        "full_passthrough_gaussians": 0,
        "cross_check_threshold": cross_check_threshold,
        "execution_route_sha256": execution_route_sha256,
        "controls": controls,
    }


def _write_cache(
    root: Path,
    *,
    contract: dict,
    model: str = "transplat",
    split: str = "calibration_train",
    anchors: tuple[int, ...] = (0,),
    cross_check_threshold: float = 0.015,
) -> None:
    from data.acid_joint_training_contract import (
        TEACHER_CACHE_KIND,
        TEACHER_CACHE_SCHEMA_VERSION,
    )
    from scripts.acid_joint_calibration_runner import _sha256_file

    root.mkdir(parents=True)
    identity = contract["context_only_inputs"]["splits"][split]
    routing = _routing(
        example_count=len(anchors),
        cross_check_threshold=cross_check_threshold,
        execution_route_sha256=contract["saes_routing"]["execution_route_sha256"],
    )
    entries = []
    for index in range(identity["scene_count"]):
        record = {
            "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
            "kind": TEACHER_CACHE_KIND,
            "contract_sha256": contract["contract_sha256"],
            "plan_sha256": contract["source_plan"]["plan_sha256"],
            "model": model,
            "split": split,
            "scene": f"scene-{index:02d}",
            "sample_index": index,
            "checkpoint_sha256": contract["checkpoint_binding"]["models"][model]["checkpoint_sha256"],
            "config_source_sha256": contract["model_config_binding"][model]["source_manifest_sha256"],
            "runtime_source": contract["model_config_binding"][model]["runtime_source"],
            "environment_profile": contract["model_config_binding"][model]["environment_profile"],
            "profile_interpreter": contract["model_config_binding"][model]["profile_interpreter"],
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
                key: identity[key]
                for key in (
                    "tree_sha256",
                    "manifest_sha256",
                    "input_provenance_sha256",
                    "selection_sha256",
                    "scene_count",
                )
            },
            "source_identity": {
                "sidecar": identity,
                "target_rgb_accessed": False,
                "target_camera_metadata_accessed": False,
                "target_index_accessed": False,
                "expected_results_accessed": False,
            },
            "preparation": {
                "source_image_shape": contract["preprocessing"]["source_image_shape"],
                "prepared_image_shape": contract["preprocessing"]["target_image_shape"],
                "context_view_count": 2,
                "target_mapping_present": False,
                "target_rgb_accessed": False,
                "target_camera_metadata_accessed": False,
                "target_index_accessed": False,
                "baseline_normalized": True,
                "patch_size": contract["preprocessing"]["per_model_patch_size"][model],
            },
            "routing": routing,
            "teacher": {
                "offline_teacher_only": True,
                "source": "dense_adaptor_output",
                "target": "assignment_aligned_dense_adaptor_nonprobe_aggregation",
                "selected_anchor_only": True,
                "skipped_s3_descriptors_accessed_offline_teacher_only": True,
                "skipped_s3_descriptors_persisted": False,
                "target_rgb_accessed": False,
                "target_camera_metadata_accessed": False,
                "target_index_accessed": False,
            },
            "examples": [_example(anchor_index=anchor) for anchor in anchors],
        }
        filename = f"{index:06d}.pt"
        path = root / filename
        torch.save(record, path)
        entries.append(
            {
                "sample_index": index,
                "scene": record["scene"],
                "path": filename,
                "sha256": _sha256_file(path),
                "byte_count": path.stat().st_size,
                "record_count": len(anchors),
                "routing": routing,
            }
        )
    manifest = {
        "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "kind": TEACHER_CACHE_KIND,
        "status": "PASS_AUTHOR_SIDE_OFFLINE_TEACHER",
        "partial": False,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "model": model,
        "split": split,
        "checkpoint_sha256": contract["checkpoint_binding"]["models"][model]["checkpoint_sha256"],
        "config_source_sha256": contract["model_config_binding"][model]["source_manifest_sha256"],
        "runtime_source": contract["model_config_binding"][model]["runtime_source"],
        "environment_profile": contract["model_config_binding"][model]["environment_profile"],
        "profile_interpreter": contract["model_config_binding"][model]["profile_interpreter"],
        "resolved_config_sha256": contract["model_config_binding"][model][
            "resolved_config_sha256"
        ],
        "preprocessing_id": contract["preprocessing"]["identifier"],
        "prepared_patch_size": contract["preprocessing"]["per_model_patch_size"][model],
        "saes_routing_sha256": contract["saes_routing"]["routing_sha256"],
        "saes_execution_route_sha256": contract["saes_routing"][
            "execution_route_sha256"
        ],
        "cross_check_threshold": cross_check_threshold,
        "input_identity": {
            key: identity[key]
            for key in (
                "tree_sha256",
                "manifest_sha256",
                "input_provenance_sha256",
                "selection_sha256",
                "scene_count",
            )
        },
        "scene_count": identity["scene_count"],
        "record_count": len(entries) * len(anchors),
        "entries": entries,
        "access_audit": {
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "expected_results_accessed": False,
            "evaluation_scene_accessed": False,
            "teacher_source": "dense_adaptor_output",
            "teacher_files_runtime_accessible": False,
            "skipped_s3_descriptors_accessed": False,
            "descriptor_model_id_accessed": False,
            "descriptor_dataset_id_accessed": False,
            "optimizer_executed": False,
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def test_cache_loader_rejects_anchor_order_and_cross_check_drift(tmp_path):
    from scripts.acid_joint_calibration_runner import (
        JointCalibrationRunError,
        load_completed_cache,
    )

    contract = _contract()
    unordered = tmp_path / "unordered"
    _write_cache(unordered, contract=contract, anchors=(1, 0))
    with pytest.raises(JointCalibrationRunError, match="ascending order"):
        load_completed_cache(
            unordered,
            contract=contract,
            model="transplat",
            split="calibration_train",
        )

    drifted = tmp_path / "drifted"
    _write_cache(drifted, contract=contract, cross_check_threshold=0.01)
    with pytest.raises(JointCalibrationRunError, match="cross-check threshold"):
        load_completed_cache(
            drifted,
            contract=contract,
            model="transplat",
            split="calibration_train",
        )


def _runtime_evidence(contract: dict, *, asset: dict, model: str, split: str) -> dict:
    from data.acid_joint_training_contract import (
        RUNTIME_EVIDENCE_KIND,
        RUNTIME_EVIDENCE_SCHEMA_VERSION,
    )

    identity = contract["context_only_inputs"]["splits"][split]
    records = [
        {
            "sample_index": index,
            "scene": f"runtime-scene-{index:02d}",
            "route_mask_sha256": _digest(f"mask-{index}"),
            "selected_head_sha256": _digest(f"head-{index}"),
            "saes_stats_sha256": _digest(f"stats-{index}"),
        }
        for index in range(identity["scene_count"])
    ]
    return {
        "schema_version": RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "kind": RUNTIME_EVIDENCE_KIND,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "model": model,
        "split": split,
        "checkpoint_sha256": contract["checkpoint_binding"]["models"][model]["checkpoint_sha256"],
        "config_source_sha256": contract["model_config_binding"][model]["source_manifest_sha256"],
        "runtime_source": contract["model_config_binding"][model]["runtime_source"],
        "environment_profile": contract["model_config_binding"][model]["environment_profile"],
        "profile_interpreter": contract["model_config_binding"][model]["profile_interpreter"],
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
            key: identity[key]
            for key in (
                "tree_sha256",
                "manifest_sha256",
                "input_provenance_sha256",
                "selection_sha256",
                "scene_count",
            )
        },
        "asset": asset,
        "candidate_execution_verification": None,
        "execution_boundary": {
            "dense_route_prepass_for_validation": True,
            "selected_head_only": True,
            "s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
        },
        "selected_head": {
            "model": model,
            "contract_version": "saes-selected-output-replay-v1",
            "dense_head_macs": 1000,
            "replayed_head_macs": 100,
        },
        "ledger": {
            "ledger_version": "saes-event-ledger-v3",
            "events": {
                "joint_calibrator_calls": 4,
                "joint_calibrator_l0_calls": 4,
                "joint_calibrator_l1_calls": 0,
                "joint_calibrator_full_calls": 0,
                "joint_calibrator_selected_descriptor_reads": 4,
                "joint_calibrator_skipped_head_macs": 900,
            },
        },
        "access_audit": {
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "expected_results_accessed": False,
            "evaluation_scene_accessed": False,
            "teacher_files_opened": False,
            "teacher_files_runtime_accessible": False,
            "runtime_teacher_path": None,
            "descriptor_model_id_accessed": False,
            "descriptor_dataset_id_accessed": False,
            "optimizer_executed": False,
            "asset_updated": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
        },
        "scene_records": records,
    }


def test_runtime_evidence_rejects_target_leaks_and_forged_head_events(tmp_path, monkeypatch):
    from scripts import acid_joint_calibration_runner as runner

    contract = _contract()
    asset = {
        "asset_path": contract["shared_asset"]["asset_path"],
        "schema_version": contract["shared_asset"]["architecture"]["schema_version"],
        "kind": contract["shared_asset"]["architecture"]["kind"],
        "sha256": _digest("asset"),
        "state_sha256": _digest("state"),
        "byte_count": 1,
        "descriptor_dim": 32,
        "bottleneck_dim": 8,
        "joint_output_dim": 40,
    }
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "runtime.json"
    evidence = _runtime_evidence(
        contract, asset=asset, model="transplat", split="calibration_train"
    )
    path.write_text(json.dumps(evidence), encoding="utf-8")
    _, identity = runner._runtime_evidence(
        path,
        contract=contract,
        model="transplat",
        split="calibration_train",
        asset=asset,
        candidate_execution_verification=None,
    )
    assert identity["asset_sha256"] == asset["sha256"]

    with pytest.raises(runner.JointCalibrationRunError, match="differs from the frozen contract"):
        runner._runtime_evidence(
            path,
            contract=contract,
            model="transplat",
            split="calibration_train",
            asset=asset,
            candidate_execution_verification={"provenance": {}, "verification": {}},
        )

    evidence["access_audit"]["target_rgb_accessed"] = True
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(runner.JointCalibrationRunError, match="isolation boundary"):
        runner._runtime_evidence(
            path,
            contract=contract,
            model="transplat",
            split="calibration_train",
            asset=asset,
            candidate_execution_verification=None,
        )

    evidence["access_audit"]["target_rgb_accessed"] = False
    evidence["selected_head"]["replayed_head_macs"] = 1000
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(runner.JointCalibrationRunError, match="MACs are invalid"):
        runner._runtime_evidence(
            path,
            contract=contract,
            model="transplat",
            split="calibration_train",
            asset=asset,
            candidate_execution_verification=None,
        )


def test_asset_snapshot_never_calls_module_cpu(tmp_path, monkeypatch):
    from saes.joint_materialization_calibrator import JointMaterializationCalibrator
    from scripts.acid_joint_calibration_runner import _write_asset_new

    calibrator = JointMaterializationCalibrator()
    monkeypatch.setattr(
        calibrator,
        "cpu",
        lambda: (_ for _ in ()).throw(AssertionError("Module.cpu must not run")),
    )
    _write_asset_new(calibrator, tmp_path / "candidate.pt")
    assert (tmp_path / "candidate.pt").is_file()


def test_global_identity_normalizers_are_not_batch_local():
    from scripts.acid_joint_calibration_runner import CacheExample, ModelCache, _global_identity_normalizers

    def cache(model: str) -> ModelCache:
        examples = []
        for scale in (1.0, 3.0):
            record = _example(anchor_index=len(examples), scale=scale)
            examples.append(
                CacheExample(
                    model=model,
                    descriptor=record["descriptor"],
                    compact=record["compact"],
                    teacher=record["teacher"],
                    sh_degree=0,
                )
            )
        return ModelCache(
            model=model,
            split="calibration_train",
            resolved_config_sha256=_digest(model),
            prepared_patch_size=16,
            examples=tuple(examples),
            controls={
                "route_mask_unchanged_required": True,
                "retained_counts_unchanged_required": True,
                "full_passthrough_required": True,
                "selected_only_s3_required": True,
                "two_finite_skipped_s3_sentinels_required": True,
            },
        )

    caches = {model: cache(model) for model in ("transplat", "mvsplat", "depthsplat")}
    normalizers = _global_identity_normalizers(caches, floor=1e-8)
    assert normalizers["transplat"]["mean"] == pytest.approx(5.0)
    assert normalizers["transplat"]["sh"] == pytest.approx(5.0)


@pytest.mark.parametrize("relative_mse", [0.5, 1.1])
def test_train_candidate_gate_promotes_only_after_fidelity_passes(
    tmp_path, monkeypatch, relative_mse
):
    from saes.joint_materialization_calibrator import JointMaterializationCalibrator
    from scripts import acid_joint_calibration_runner as runner

    contract = _contract()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    asset_path = tmp_path / contract["shared_asset"]["asset_path"]
    candidate_path = asset_path.with_name(asset_path.name + ".candidate")
    runner._write_asset_new(JointMaterializationCalibrator(), candidate_path)
    provenance_path = candidate_path.with_name(candidate_path.name + ".provenance.json")
    provenance_path.write_text("{}", encoding="utf-8")

    cache = runner.ModelCache(
        model="transplat",
        split="calibration_train",
        resolved_config_sha256=_digest("resolved-transplat"),
        prepared_patch_size=16,
        examples=(
            runner.CacheExample(
                model="transplat",
                descriptor=torch.zeros(32),
                compact=_example(anchor_index=0)["compact"],
                teacher=_example(anchor_index=0)["teacher"],
                sh_degree=0,
            ),
        ),
        controls={
            "route_mask_unchanged_required": True,
            "retained_counts_unchanged_required": True,
            "full_passthrough_required": True,
            "selected_only_s3_required": True,
            "two_finite_skipped_s3_sentinels_required": True,
        },
    )
    caches = {
        model: runner.ModelCache(
            model=model,
            split=cache.split,
            resolved_config_sha256=_digest(f"resolved-{model}"),
            prepared_patch_size=cache.prepared_patch_size,
            examples=cache.examples,
            controls=cache.controls,
        )
        for model in ("transplat", "mvsplat", "depthsplat")
    }

    monkeypatch.setattr(runner, "_load_canonical_contract", lambda _path: contract)
    monkeypatch.setattr(
        runner,
        "_load_canonical_caches",
        lambda _roots, *, contract, split: caches,
    )
    monkeypatch.setattr(
        runner,
        "_load_asset",
        lambda _path, *, asset, device: JointMaterializationCalibrator().to(device).eval(),
    )

    def evidence(_paths, *, contract, split, asset, candidate_execution_verification):
        return {
            model: (
                {"resolved_config_sha256": caches[model].resolved_config_sha256},
                {
                    "path": contract["runtime_evidence_records"][split][model],
                    "sha256": _digest(f"runtime-{model}"),
                    "selected_head_sha256": _digest(f"head-{model}"),
                    "ledger_sha256": _digest(f"ledger-{model}"),
                    "asset_sha256": asset["sha256"],
                    "asset_state_sha256": asset["state_sha256"],
                },
            )
            for model in ("transplat", "mvsplat", "depthsplat")
        }

    monkeypatch.setattr(runner, "_load_canonical_runtime_evidence", evidence)
    monkeypatch.setattr(
        runner,
        "_metrics",
        lambda _calibrator, _cache, *, device: {
            component: {"identity_mse": 1.0, "calibrated_mse": relative_mse}
            for component in ("mean", "covariance", "opacity", "sh")
        },
    )
    monkeypatch.setattr(runner, "validate_live_teacher_fidelity_result", lambda *args, **kwargs: {"ok": True})

    def provenance(_path, *, contract, candidate_asset, cache_manifests):
        return {
            "path": candidate_path.relative_to(tmp_path).as_posix() + ".provenance.json",
            "sha256": _digest("verified-provenance"),
            "candidate_asset_sha256": candidate_asset["sha256"],
            "candidate_asset_state_sha256": candidate_asset["state_sha256"],
            "cache_manifests": {},
            "local_hash_chain_is_not_cryptographic_proof": True,
        }

    def verification(_path, *, contract, provenance):
        return {
            "path": candidate_path.relative_to(tmp_path).as_posix() + ".verification.json",
            "sha256": _digest("verified-verification"),
            "verification_mode": "deterministic_replay_verifier",
            "candidate_provenance_path": provenance["path"],
            "candidate_provenance_sha256": provenance["sha256"],
            "candidate_asset_sha256": provenance["candidate_asset_sha256"],
        }

    monkeypatch.setattr(runner, "_candidate_cache_manifest_identities", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_candidate_provenance", provenance)
    monkeypatch.setattr(runner, "_candidate_verification", verification)

    # v3 treats the locally fabricated provenance/verification identities as
    # non-cryptographic. The canonical sidecars are therefore required before
    # a candidate can ever be promoted, regardless of synthetic MSE values.
    with pytest.raises(
        runner.JointTrainingContractError,
        match="candidate provenance or verification sidecar is unavailable",
    ):
        runner.finalize_train_candidate(
            contract_path=tmp_path / "ignored-contract.json",
            cache_roots={},
            runtime_evidence_paths={},
            device=torch.device("cpu"),
            candidate_verification_path=tmp_path / "verified.json",
        )
    assert not asset_path.exists()
    assert candidate_path.is_file()


def test_finalize_rejects_a_handwritten_candidate_without_provenance(tmp_path, monkeypatch):
    from saes.joint_materialization_calibrator import JointMaterializationCalibrator
    from scripts import acid_joint_calibration_runner as runner

    contract = _contract()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    asset_path = tmp_path / contract["shared_asset"]["asset_path"]
    candidate_path = asset_path.with_name(asset_path.name + ".candidate")
    runner._write_asset_new(JointMaterializationCalibrator(), candidate_path)
    monkeypatch.setattr(runner, "_load_canonical_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "_load_canonical_caches", lambda *_args, **_kwargs: {})

    with pytest.raises(runner.JointCalibrationRunError, match="staged candidate asset"):
        runner.finalize_train_candidate(
            contract_path=tmp_path / "ignored-contract.json",
            cache_roots={},
            runtime_evidence_paths={},
            device=torch.device("cpu"),
        )
