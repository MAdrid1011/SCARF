#!/usr/bin/env python3
"""Generate one source-bound ACID runtime-control proof for the joint calibrator.

This command is intentionally separate from cache compilation and optimization.
It opens only a frozen context-only ACID sidecar, replays the native classic
Gaussian head at the retained outputs with its original weights, runs the
hash-pinned calibrator through the frozen SAES route, and derives the analytic
ledger from those runtime counters.  It never opens target RGB/cameras, a
teacher cache, an evaluation index, or paper-result artifacts.

For DepthSplat this is deliberately a final-head-only proof: the shared trunk,
cost volume, and Gaussian-regressor closure remain dense.  The command never
turns a selected-head trace into a global S2/S3 saving claim.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.acid_joint_training_contract import (
    ACID_ISOLATED_MODEL_ENV,
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_TRAINING_CONTRACT_PATH,
    HOLDOUT_SPLIT,
    MODELS,
    RUNTIME_EVIDENCE_KIND,
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    TRAIN_SPLIT,
    JointTrainingContractError,
    require_isolated_model_process,
    validate_live_training_contract,
    validate_runtime_checkpoint_binding,
    validate_runtime_model_binding,
)
from integration.acid_joint_context import load_acid_joint_context
from integration.acid_joint_model_context import prepare_acid_joint_model_context
from integration.model_loader import create_model_loader
from saes.hardware_accounting import LEDGER_VERSION, build_saes_event_ledger
from saes.joint_materialization_calibrator import (
    calibrator_asset_manifest,
    load_calibrator_asset,
)
from saes.probe_first_schedule import retained_mask_from_saes_modified
from saes.progressive_saes import apply_progressive_saes
from saes.selected_output_execution import selected_output_head_execution
from scripts.acid_joint_teacher_cache_worker import (
    _capture_encoder_execution,
    _resolved_config_sha256,
    _routing_options,
)
from scripts.calibration_contract import canonical_sha256
from scripts.ae_config import resolve_experiment


EVIDENCE_KIND = RUNTIME_EVIDENCE_KIND
EVIDENCE_SCHEMA_VERSION = RUNTIME_EVIDENCE_SCHEMA_VERSION
_ATTRIBUTES = ("means", "covariances", "harmonics", "opacities")
_EVENT_FIELDS = (
    "joint_calibrator_calls",
    "joint_calibrator_l0_calls",
    "joint_calibrator_l1_calls",
    "joint_calibrator_full_calls",
    "joint_calibrator_selected_descriptor_reads",
    "joint_calibrator_skipped_head_macs",
)


class RuntimeControlEvidenceError(RuntimeError):
    """Raised when an actual context-only replay cannot prove its controls."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeControlEvidenceError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeControlEvidenceError(f"{label} must be an object")
    return value


def _safe_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeControlEvidenceError(f"{label} path is invalid")
    candidate = (Path(root).resolve() / value).resolve()
    if Path(root).resolve() not in candidate.parents:
        raise RuntimeControlEvidenceError(f"{label} path escapes the repository")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask_sha256(mask: torch.Tensor) -> str:
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise RuntimeControlEvidenceError("runtime replay has an invalid SAES mask")
    return hashlib.sha256(
        mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes()
    ).hexdigest()


def _clone_gaussians(gaussians: Any) -> SimpleNamespace:
    values: dict[str, torch.Tensor] = {}
    for name in _ATTRIBUTES:
        value = getattr(gaussians, name, None)
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise RuntimeControlEvidenceError(f"native Gaussian adaptor has invalid {name}")
        values[name] = value.detach().clone()
    return SimpleNamespace(**values)


def _classic_head(encoder: Any, model: str) -> torch.nn.Module:
    if model == "depthsplat":
        head = getattr(encoder, "gaussian_head", None)
    else:
        predictor = getattr(encoder, "depth_predictor", None)
        head = getattr(predictor, "to_gaussians", None)
    if not isinstance(head, torch.nn.Module):
        raise RuntimeControlEvidenceError(f"{model} exposes no native classic Gaussian head")
    return head


def _apply_runtime_saes(
    gaussians: Any,
    *,
    contract: Mapping[str, Any],
    model: str,
    context: Mapping[str, torch.Tensor],
    features: torch.Tensor,
    depths: torch.Tensor,
    calibrator: Any | None = None,
    selected_head: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    height, width = (int(value) for value in context["image"].shape[-2:])
    options = _routing_options(
        contract, model=model, context=context, geometry_on_cpu=False
    )
    modified, stats, _ = apply_progressive_saes(
        gaussians,
        height,
        width,
        **options,
        features=features,
        depths=depths,
        joint_calibrator=calibrator,
        joint_calibrator_model=model if calibrator is not None else None,
        joint_calibrator_selected_head=selected_head,
    )
    if not torch.is_tensor(modified) or modified.dtype != torch.bool:
        raise RuntimeControlEvidenceError("runtime SAES emitted an invalid modified mask")
    return modified, dict(stats)


def _assert_selected_equivalence(
    dense: Any, sparse: Any, retained: torch.Tensor
) -> None:
    if retained.ndim != 1 or retained.dtype != torch.bool or not bool(retained.any()):
        raise RuntimeControlEvidenceError("runtime replay has no retained Gaussian output")
    for name in _ATTRIBUTES:
        reference = getattr(dense, name)[0, retained]
        candidate = getattr(sparse, name)[0, retained]
        if reference.shape != candidate.shape or not torch.allclose(
            reference, candidate, rtol=1.0e-5, atol=1.0e-5
        ):
            raise RuntimeControlEvidenceError(
                f"same-weight selected head changed retained {name}"
            )


def _sh_degree(gaussians: Any) -> int:
    harmonics = getattr(gaussians, "harmonics", None)
    if not torch.is_tensor(harmonics) or harmonics.ndim != 4 or harmonics.shape[2] != 3:
        raise RuntimeControlEvidenceError("runtime Gaussian harmonics are invalid")
    root = math.isqrt(int(harmonics.shape[-1]))
    if root * root != harmonics.shape[-1] or root < 1:
        raise RuntimeControlEvidenceError("runtime Gaussian SH degree is invalid")
    return root - 1


def _asset_identity(path: Path, *, contract: Mapping[str, Any], device: torch.device) -> tuple[Any, dict[str, Any]]:
    manifest = calibrator_asset_manifest(path)
    try:
        calibrator = load_calibrator_asset(path, manifest).to(device).eval()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeControlEvidenceError("runtime evidence asset cannot be loaded") from exc
    return calibrator, {
        "asset_path": contract["shared_asset"]["asset_path"],
        "state_sha256": calibrator.runtime_state_sha256,
        **manifest,
    }


def _scene_record(
    *,
    contract: Mapping[str, Any],
    model: str,
    sample_index: int,
    context: Mapping[str, torch.Tensor],
    encoder: Any,
    calibrator: Any,
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    dense_raw, features, depths = _capture_encoder_execution(model, encoder, context)
    views, height, width = (
        int(context["image"].shape[1]),
        int(context["image"].shape[-2]),
        int(context["image"].shape[-1]),
    )
    if dense_raw.means.shape[1] != views * height * width:
        raise RuntimeControlEvidenceError("runtime evidence requires exactly one Gaussian per pixel")
    dense_ordinary = _clone_gaussians(dense_raw)
    dense_modified, _ = _apply_runtime_saes(
        dense_ordinary,
        contract=contract,
        model=model,
        context=context,
        features=features,
        depths=depths,
    )
    selection = retained_mask_from_saes_modified(
        dense_modified, views=views, height=height, width=width
    )
    head = _classic_head(encoder, model)
    with selected_output_head_execution(head, selection) as trace:
        sparse_raw, sparse_features, sparse_depths = _capture_encoder_execution(
            model, encoder, context
        )
    if not torch.equal(features, sparse_features) or not torch.equal(depths, sparse_depths):
        raise RuntimeControlEvidenceError("selected head changed an upstream routing tensor")
    head_events = trace.events
    selected_head = {
        "model": model,
        "contract_version": head_events["contract_version"],
        "dense_head_macs": int(head_events["dense_head_macs"]),
        "replayed_head_macs": int(head_events["actual_head_macs"]),
    }
    if selected_head["dense_head_macs"] <= selected_head["replayed_head_macs"]:
        raise RuntimeControlEvidenceError("selected head replay did not reduce any final-head work")

    sparse_ordinary = _clone_gaussians(sparse_raw)
    sparse_modified, _ = _apply_runtime_saes(
        sparse_ordinary,
        contract=contract,
        model=model,
        context=context,
        features=sparse_features,
        depths=sparse_depths,
    )
    if not torch.equal(dense_modified, sparse_modified):
        raise RuntimeControlEvidenceError("selected head changed the frozen SAES route")
    _assert_selected_equivalence(dense_ordinary, sparse_ordinary, ~dense_modified)

    sparse_runtime = _clone_gaussians(sparse_raw)
    runtime_modified, runtime_stats = _apply_runtime_saes(
        sparse_runtime,
        contract=contract,
        model=model,
        context=context,
        features=sparse_features,
        depths=sparse_depths,
        calibrator=calibrator,
        selected_head=selected_head,
    )
    if not torch.equal(dense_modified, runtime_modified):
        raise RuntimeControlEvidenceError("joint calibrator changed the frozen SAES route")
    ledger = build_saes_event_ledger(
        runtime_stats,
        feature_dim=int(features.shape[2]),
        tile_size=int(contract["saes_routing"]["parameters"]["tile_size"]),
        sh_degree=_sh_degree(sparse_runtime),
        primitives_per_pixel=1,
    )
    events = ledger.get("events")
    if not isinstance(events, Mapping) or any(field not in events for field in _EVENT_FIELDS):
        raise RuntimeControlEvidenceError("runtime ledger omitted joint calibrator events")
    if (
        events["joint_calibrator_calls"]
        != events["joint_calibrator_selected_descriptor_reads"]
        or events["joint_calibrator_calls"]
        != events["joint_calibrator_l0_calls"]
        + events["joint_calibrator_l1_calls"]
        + events["joint_calibrator_full_calls"]
        or events["joint_calibrator_full_calls"] != 0
        or events["joint_calibrator_skipped_head_macs"] <= 0
    ):
        raise RuntimeControlEvidenceError("runtime ledger did not charge the joint calibrator")
    record = {
        "sample_index": sample_index,
        "scene": "",
        "route_mask_sha256": _mask_sha256(dense_modified),
        "selected_head_sha256": canonical_sha256(selected_head),
        "saes_stats_sha256": canonical_sha256(runtime_stats),
    }
    return record, selected_head, {field: int(events[field]) for field in _EVENT_FIELDS}


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"runtime evidence partial already exists: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def generate_runtime_evidence(
    *,
    contract_path: Path,
    materialization_root: Path,
    asset_path: Path,
    model: str,
    split: str,
    output_path: Path,
    device: torch.device,
    candidate_verification_path: Path | None = None,
) -> dict[str, Any]:
    """Run every frozen context scene and emit one immutable control proof."""
    if model not in MODELS or split not in {TRAIN_SPLIT, HOLDOUT_SPLIT}:
        raise RuntimeControlEvidenceError("runtime evidence model or split is invalid")
    if Path(contract_path).resolve() != DEFAULT_TRAINING_CONTRACT_PATH:
        raise RuntimeControlEvidenceError("runtime evidence requires the canonical frozen contract")
    require_isolated_model_process(model)
    contract = _read_json(contract_path, "joint training contract")
    validate_live_training_contract(contract, repository_root=ROOT)
    expected_materialization = _safe_path(
        ROOT, contract["context_only_inputs"]["materialization_root"], "context materialization"
    )
    if Path(materialization_root).resolve() != expected_materialization:
        raise RuntimeControlEvidenceError("runtime evidence requires the canonical context materialization")
    expected_output = _safe_path(
        ROOT, contract["runtime_evidence_records"][split][model], "runtime evidence"
    )
    if Path(output_path).resolve() != expected_output:
        raise RuntimeControlEvidenceError("runtime evidence output path is not canonical")
    expected_asset = _safe_path(ROOT, contract["shared_asset"]["asset_path"], "calibrator asset")
    candidate_asset = expected_asset.with_name(expected_asset.name + ".candidate")
    if Path(asset_path).resolve() not in {expected_asset, candidate_asset}:
        raise RuntimeControlEvidenceError("runtime evidence asset path is not a frozen candidate or canonical asset")
    resolved_asset_path = Path(asset_path).resolve()
    calibrator, asset = _asset_identity(resolved_asset_path, contract=contract, device=device)
    candidate_execution_verification: dict[str, Any] | None = None
    if resolved_asset_path == candidate_asset:
        from scripts.acid_joint_calibration_runner import (
            _candidate_cache_manifest_identities,
            _candidate_provenance,
            _candidate_provenance_path,
            _candidate_verification,
        )

        cache_manifests = _candidate_cache_manifest_identities(contract, split=TRAIN_SPLIT)
        provenance = _candidate_provenance(
            _candidate_provenance_path(expected_asset),
            contract=contract,
            candidate_asset=asset,
            cache_manifests=cache_manifests,
        )
        verification = _candidate_verification(
            candidate_verification_path, contract=contract, provenance=provenance
        )
        candidate_execution_verification = {
            "provenance": provenance,
            "verification": verification,
        }
    elif candidate_verification_path is not None:
        raise RuntimeControlEvidenceError(
            "canonical runtime evidence may not consume a candidate verifier record"
        )

    experiment = resolve_experiment(model, "acid", ROOT)
    expected_checkpoint = contract["checkpoint_binding"]["models"][model]
    expected_config = contract["model_config_binding"][model]
    if (
        experiment.checkpoint.relative_to(ROOT).as_posix() != expected_checkpoint["checkpoint_path"]
        or experiment.experiment != expected_checkpoint["checkpoint_dataset"]
        or list(experiment.hydra_overrides) != expected_config["hydra_overrides"]
    ):
        raise RuntimeControlEvidenceError("runtime evidence model experiment differs from the frozen contract")
    loader = create_model_loader(model)
    checkpoint_sha256 = validate_runtime_checkpoint_binding(
        contract,
        model=model,
        checkpoint_path=experiment.checkpoint,
        repository_root=ROOT,
    )
    bundle = loader.load_model(
        str(experiment.checkpoint),
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        encoder_only=True,
    )
    if bundle.decoder is not None:
        raise RuntimeControlEvidenceError("runtime evidence unexpectedly constructed a decoder")
    if (
        validate_runtime_checkpoint_binding(
            contract,
            model=model,
            checkpoint_path=experiment.checkpoint,
            repository_root=ROOT,
        )
        != checkpoint_sha256
    ):
        raise RuntimeControlEvidenceError(
            "runtime checkpoint identity changed during model loading"
        )
    resolved_config_sha256 = _resolved_config_sha256(bundle.config)
    runtime_binding = validate_runtime_model_binding(
        contract,
        model=model,
        resolved_config_sha256=resolved_config_sha256,
        repository_root=ROOT,
        require_loaded_src=True,
    )
    runtime_binding = {**runtime_binding, "checkpoint_sha256": checkpoint_sha256}
    scene_records: list[dict[str, Any]] = []
    head_totals = {"dense_head_macs": 0, "replayed_head_macs": 0}
    ledger_totals = {field: 0 for field in _EVENT_FIELDS}
    try:
        count = int(contract["context_only_inputs"]["splits"][split]["scene_count"])
        for sample_index in range(count):
            raw = load_acid_joint_context(
                materialization_root=expected_materialization,
                split=split,
                sample_index=sample_index,
                plan_path=ROOT / contract["source_plan"]["path"],
            )
            context, preparation = prepare_acid_joint_model_context(
                raw,
                dataset_cfg=bundle.config.dataset,
                encoder_cfg=bundle.config.model.encoder,
                device=device,
            )
            if (
                preparation.get("source_image_shape") != contract["preprocessing"]["source_image_shape"]
                or preparation.get("prepared_image_shape") != contract["preprocessing"]["target_image_shape"]
                or preparation.get("patch_size") != contract["preprocessing"]["per_model_patch_size"][model]
                or set(context) != {"image", "extrinsics", "intrinsics", "index", "near", "far"}
            ):
                raise RuntimeControlEvidenceError("runtime evidence preprocessing differs from contract")
            record, selected_head, ledger_events = _scene_record(
                contract=contract,
                model=model,
                sample_index=sample_index,
                context=context,
                encoder=bundle.encoder,
                calibrator=calibrator,
            )
            record["scene"] = str(raw.identity["scene"])
            scene_records.append(record)
            for field in head_totals:
                head_totals[field] += int(selected_head[field])
            for field in ledger_totals:
                ledger_totals[field] += ledger_events[field]
            del context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if head_totals["dense_head_macs"] <= head_totals["replayed_head_macs"]:
        raise RuntimeControlEvidenceError("complete runtime replay did not reduce selected-head MACs")
    selected_head = {
        "model": model,
        "contract_version": "saes-selected-output-replay-v1",
        **head_totals,
    }
    ledger = {
        "ledger_version": LEDGER_VERSION,
        "events": ledger_totals,
        "per_scene_stats_sha256": [record["saes_stats_sha256"] for record in scene_records],
    }
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "model": model,
        "split": split,
        "checkpoint_sha256": runtime_binding["checkpoint_sha256"],
        "config_source_sha256": runtime_binding["config_source_sha256"],
        "runtime_source": dict(runtime_binding["runtime_source"]),
        "environment_profile": runtime_binding["environment_profile"],
        "profile_interpreter": dict(runtime_binding["profile_interpreter"]),
        "resolved_config_sha256": runtime_binding["resolved_config_sha256"],
        "preprocessing_id": contract["preprocessing"]["identifier"],
        "prepared_patch_size": contract["preprocessing"]["per_model_patch_size"][model],
        "saes_routing_sha256": contract["saes_routing"]["routing_sha256"],
        "saes_execution_route_sha256": contract["saes_routing"][
            "execution_route_sha256"
        ],
        "input_identity": {
            key: contract["context_only_inputs"]["splits"][split][key]
            for key in (
                "tree_sha256",
                "manifest_sha256",
                "input_provenance_sha256",
                "selection_sha256",
                "scene_count",
            )
        },
        "asset": asset,
        "candidate_execution_verification": candidate_execution_verification,
        "execution_boundary": {
            "dense_route_prepass_for_validation": True,
            "selected_head_only": True,
            "s2_s3_sparse_execution_verified": False,
            "global_s2_s3_savings_claimed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
        },
        "selected_head": selected_head,
        "ledger": ledger,
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
        "scene_records": scene_records,
    }
    _write_json_new(expected_output, evidence)
    return {
        "output": expected_output.relative_to(ROOT).as_posix(),
        "sha256": _sha256_file(expected_output),
        "model": model,
        "split": split,
        "scene_count": len(scene_records),
        "paper_result_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION_ROOT)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--candidate-verification", type=Path)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=(TRAIN_SPLIT, HOLDOUT_SPLIT), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if os.environ.get(ACID_ISOLATED_MODEL_ENV) != args.model:
        environment = dict(os.environ)
        environment[ACID_ISOLATED_MODEL_ENV] = args.model
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            env=environment,
            check=False,
        ).returncode
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeControlEvidenceError("CUDA was requested for runtime evidence but is unavailable")
        result = generate_runtime_evidence(
            contract_path=args.contract,
            materialization_root=args.materialization_root,
            asset_path=args.asset,
            candidate_verification_path=args.candidate_verification,
            model=args.model,
            split=args.split,
            output_path=args.output,
            device=device,
        )
    except (JointTrainingContractError, OSError, RuntimeError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
