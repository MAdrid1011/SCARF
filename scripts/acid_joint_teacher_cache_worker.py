#!/usr/bin/env python3
"""Compile one isolated ACID context-only teacher cache for joint calibration.

This entrypoint is intentionally run in one upstream model environment at a
time.  It opens only the validated ACID context sidecars, runs the frozen
encoder to obtain an offline dense-adaptor teacher, and writes examples made
from ordinary selected SAES representatives.  The resulting cache is author
training data, never a runtime input.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

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
    TEACHER_CACHE_EXAMPLE_KIND,
    TEACHER_CACHE_KIND,
    TEACHER_CACHE_SCHEMA_VERSION,
    TRAIN_SPLIT,
    JointTrainingContractError,
    require_isolated_model_process,
    resolved_config_sha256 as _resolved_config_sha256,
    validate_live_training_contract,
    validate_runtime_checkpoint_binding,
    validate_runtime_model_binding,
)
from integration.acid_joint_context import DEFAULT_PLAN_PATH, load_acid_joint_context
from integration.acid_joint_model_context import prepare_acid_joint_model_context
from integration.model_loader import create_model_loader
from saes.joint_materialization_teacher import build_dense_adaptor_teacher_packets
from scripts.ae_config import resolve_experiment
from scripts.calibration_contract import canonical_sha256
from scripts.saes_execution_identity import validate_saes_execution_identity


CACHE_SCHEMA_VERSION = TEACHER_CACHE_SCHEMA_VERSION
CACHE_KIND = TEACHER_CACHE_KIND
EXAMPLE_KIND = TEACHER_CACHE_EXAMPLE_KIND
ATTRIBUTE_FIELDS = ("means", "covariances", "harmonics", "opacities")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_gaussians_to_cpu(gaussians: Any) -> SimpleNamespace:
    copied = {}
    for name in ATTRIBUTE_FIELDS:
        value = getattr(gaussians, name, None)
        if not torch.is_tensor(value) or value.ndim < 2:
            raise RuntimeError(f"dense adaptor emitted invalid {name}")
        copied[name] = value.detach().to(device="cpu", dtype=torch.float32).clone()
        if not bool(torch.isfinite(copied[name]).all()):
            raise RuntimeError(f"dense adaptor emitted non-finite {name}")
    if copied["means"].shape[0] != 1 or copied["means"].shape[-1] != 3:
        raise RuntimeError("dense adaptor has an invalid mean layout")
    count = int(copied["means"].shape[1])
    if copied["covariances"].shape != (1, count, 3, 3):
        raise RuntimeError("dense adaptor has an invalid covariance layout")
    if copied["harmonics"].shape[:3] != (1, count, 3):
        raise RuntimeError("dense adaptor has an invalid harmonic layout")
    if copied["opacities"].shape[:2] != (1, count):
        raise RuntimeError("dense adaptor has an invalid opacity layout")
    return SimpleNamespace(**copied)


def _capture_encoder_execution(
    model_name: str, encoder: Any, context: Mapping[str, torch.Tensor]
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Capture the model's actual routing feature and depth tensors once."""

    predictor = getattr(encoder, "depth_predictor", None)
    if predictor is None or not hasattr(predictor, "register_forward_hook"):
        raise RuntimeError("encoder does not expose its frozen depth predictor")
    captured: dict[str, torch.Tensor] = {}
    batch, views = context["image"].shape[:2]

    def capture(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if model_name in {"transplat", "mvsplat"}:
            if (
                not inputs
                or not torch.is_tensor(inputs[0])
                or inputs[0].ndim != 5
                or not isinstance(output, tuple)
                or not output
                or not torch.is_tensor(output[0])
            ):
                raise RuntimeError("classic depth predictor did not expose S1/S2 tensors")
            captured["features"] = inputs[0].detach().clone()
            captured["depths"] = output[0].detach().clone()
            return
        if model_name != "depthsplat" or not isinstance(output, Mapping):
            raise RuntimeError("DepthSplat depth predictor did not expose a result mapping")
        features = output.get("features_mv")
        depths = output.get("depth_preds")
        if (
            not isinstance(features, (list, tuple))
            or not features
            or not torch.is_tensor(features[-1])
            or features[-1].ndim != 4
            or features[-1].shape[0] != batch * views
            or not isinstance(depths, (list, tuple))
            or not depths
            or not torch.is_tensor(depths[-1])
            or depths[-1].shape[:2] != (batch, views)
        ):
            raise RuntimeError("DepthSplat depth predictor emitted invalid S1/S2 tensors")
        feature = features[-1].detach().clone()
        captured["features"] = feature.reshape(
            batch, views, *feature.shape[1:]
        )
        captured["depths"] = depths[-1].detach().clone()

    handle = predictor.register_forward_hook(capture)
    try:
        with torch.no_grad():
            execution = encoder(context, False, deterministic=True)
    finally:
        handle.remove()
    if set(captured) != {"features", "depths"}:
        raise RuntimeError("encoder execution did not expose exactly one S1/S2 capture")
    gaussians = execution.get("gaussians") if isinstance(execution, Mapping) else execution
    if gaussians is None:
        raise RuntimeError("encoder did not emit dense Gaussian adaptor output")
    return gaussians, captured["features"], captured["depths"]


def _routing_options(
    contract: Mapping[str, Any],
    model: str,
    context: Mapping[str, torch.Tensor],
    *,
    geometry_on_cpu: bool = True,
) -> dict[str, Any]:
    routing = contract["saes_routing"]
    parameters = routing["parameters"]
    per_model = routing["per_model"][model]
    try:
        execution_identity = validate_saes_execution_identity(
            routing.get("execution_identity")
        )
    except ValueError as exc:
        raise RuntimeError("teacher cache has no registered SAES execution route") from exc
    if (
        routing.get("execution_route_sha256")
        != execution_identity["route_sha256"]
        or parameters.get("cross_check_threshold")
        != execution_identity["cross_check_threshold"]
        or parameters.get("l0_retained_positions")
        != execution_identity["l0_anchor_count"]
        or parameters.get("l1_retained_positions")
        != execution_identity["l1_anchor_count"]
        or parameters.get("context_safety_guard")
        is not execution_identity["context_safety_guard"]
    ):
        raise RuntimeError("teacher cache routing differs from the registered execution route")
    extrinsics = context["extrinsics"].detach()
    intrinsics = context["intrinsics"].detach()
    if geometry_on_cpu:
        extrinsics = extrinsics.cpu()
        intrinsics = intrinsics.cpu()
    return {
        "tile_size": int(execution_identity["tile_size"]),
        "gpp": int(per_model["gpp"]),
        "feature_var_threshold": float(execution_identity["feature_threshold"]),
        "depth_std_threshold": float(execution_identity["depth_threshold"]),
        "cross_check_threshold": float(execution_identity["cross_check_threshold"]),
        "decision_semantics": str(execution_identity["decision_semantics"]),
        "beta_x": float(parameters["beta_x"]),
        "beta_f": float(parameters["beta_f"]),
        "beta_d": float(parameters["beta_d"]),
        "num_depth_candidates": int(per_model["num_depth_candidates"]),
        "materialization": str(execution_identity["materialization"]),
        "depth_routing_semantics": str(execution_identity["depth_routing_semantics"]),
        "ray_depth_mode": str(per_model["ray_depth_mode"]),
        "materialization_guard": bool(execution_identity["materialization_guard"]),
        "context_safety_guard": bool(execution_identity["context_safety_guard"]),
        "view_count": int(context["image"].shape[1]),
        "context_extrinsics": extrinsics,
        "context_intrinsics": intrinsics,
    }


def _record_for_scene(
    *,
    contract: Mapping[str, Any],
    model: str,
    split: str,
    sample_index: int,
    source_identity: Mapping[str, Any],
    preparation: Mapping[str, Any],
    dense: SimpleNamespace,
    features: torch.Tensor,
    depths: torch.Tensor,
    runtime_binding: Mapping[str, Any],
    context: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    _validate_preparation_contract(preparation, contract=contract, model=model)
    height, width = (int(value) for value in context["image"].shape[-2:])
    options = _routing_options(contract, model, context)
    packet_record = build_dense_adaptor_teacher_packets(
        dense,
        height=height,
        width=width,
        saes_kwargs={
            **options,
            "features": features,
            "depths": depths,
        },
    )
    packets = packet_record["packets"]
    controls = packet_record["controls"]
    if not packets:
        raise RuntimeError("frozen routing emitted no retained representatives for a cache scene")
    config_binding = contract["model_config_binding"][model]
    input_identity = contract["context_only_inputs"]["splits"][split]
    examples = [
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": EXAMPLE_KIND,
            "descriptor": item["descriptor"].reshape(32),
            "compact": item["base"],
            "teacher": item["teacher"],
            "level": item["level"],
            "anchor_index": item["anchor_index"],
        }
        for item in packets
    ]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": CACHE_KIND,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": contract["source_plan"]["plan_sha256"],
        "model": model,
        "split": split,
        "scene": source_identity["scene"],
        "sample_index": sample_index,
        "checkpoint_sha256": runtime_binding["checkpoint_sha256"],
        "config_source_sha256": runtime_binding["config_source_sha256"],
        "runtime_source": dict(runtime_binding["runtime_source"]),
        "environment_profile": runtime_binding["environment_profile"],
        "profile_interpreter": dict(runtime_binding["profile_interpreter"]),
        "resolved_config_sha256": runtime_binding["resolved_config_sha256"],
        "preprocessing_id": contract["preprocessing"]["identifier"],
        "prepared_patch_size": int(preparation["patch_size"]),
        "saes_routing_sha256": contract["saes_routing"]["routing_sha256"],
        "saes_execution_route_sha256": contract["saes_routing"][
            "execution_route_sha256"
        ],
        "input_identity": {
            key: input_identity[key]
            for key in (
                "tree_sha256",
                "manifest_sha256",
                "input_provenance_sha256",
                "selection_sha256",
                "scene_count",
            )
        },
        "source_identity": {
            "sidecar": source_identity["sidecar"],
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "expected_results_accessed": False,
        },
        "preparation": dict(preparation),
        "routing": {
            "mask_sha256": packet_record["route_mask_sha256"],
            "route_summary": packet_record["route_summary"],
            "level0_tiles": int(packet_record["route_summary"]["level0_tiles"]),
            "level1_tiles": int(packet_record["route_summary"]["level1_tiles"]),
            "full_tiles": int(packet_record["route_summary"]["full_tiles"]),
            "retained_representatives": len(examples),
            "full_passthrough_gaussians": int(controls["full_slot_count"]),
            "cross_check_threshold": float(options["cross_check_threshold"]),
            "execution_route_sha256": contract["saes_routing"][
                "execution_route_sha256"
            ],
            "controls": controls,
        },
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
        "examples": examples,
    }


def _validate_scene_record(
    record: Mapping[str, Any], *, contract: Mapping[str, Any], model: str
) -> int:
    required = {
        "schema_version",
        "kind",
        "contract_sha256",
        "plan_sha256",
        "model",
        "split",
        "scene",
        "sample_index",
        "checkpoint_sha256",
        "config_source_sha256",
        "runtime_source",
        "environment_profile",
        "profile_interpreter",
        "resolved_config_sha256",
        "preprocessing_id",
        "prepared_patch_size",
        "saes_routing_sha256",
        "saes_execution_route_sha256",
        "input_identity",
        "source_identity",
        "preparation",
        "routing",
        "teacher",
        "examples",
    }
    if set(record) != required or record.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError("teacher cache scene record has an invalid schema")
    if (
        record.get("kind") != CACHE_KIND
        or record.get("model") != model
        or record.get("runtime_source")
        != contract["model_config_binding"][model]["runtime_source"]
        or record.get("config_source_sha256")
        != contract["model_config_binding"][model]["source_manifest_sha256"]
        or record.get("environment_profile")
        != contract["model_config_binding"][model]["environment_profile"]
        or record.get("profile_interpreter")
        != contract["model_config_binding"][model]["profile_interpreter"]
        or record.get("resolved_config_sha256")
        != contract["model_config_binding"][model]["resolved_config_sha256"]
        or record.get("saes_routing_sha256")
        != contract["saes_routing"]["routing_sha256"]
        or not isinstance(record.get("examples"), list)
    ):
        raise RuntimeError("teacher cache scene record has an invalid kind")
    patch_size = record.get("prepared_patch_size")
    preparation = record.get("preparation")
    if (
        isinstance(patch_size, bool)
        or not isinstance(patch_size, int)
        or patch_size < 1
        or not isinstance(preparation, Mapping)
        or preparation.get("patch_size") != patch_size
        or patch_size != contract["preprocessing"]["per_model_patch_size"][model]
    ):
        raise RuntimeError("teacher cache scene record has an invalid patch preprocessing binding")
    routing = record.get("routing")
    if (
        not isinstance(routing, Mapping)
        or routing.get("cross_check_threshold")
        != float(contract["saes_routing"]["parameters"]["cross_check_threshold"])
        or routing.get("execution_route_sha256")
        != contract["saes_routing"]["execution_route_sha256"]
        or record.get("saes_execution_route_sha256")
        != contract["saes_routing"]["execution_route_sha256"]
    ):
        raise RuntimeError("teacher cache scene record has an invalid cross-check binding")
    examples = record["examples"]
    if not examples:
        raise RuntimeError("teacher cache scene record has no examples")
    anchors = []
    for item in examples:
        if not isinstance(item, Mapping) or set(item) != {
            "schema_version", "kind", "descriptor", "compact", "teacher", "level", "anchor_index"
        }:
            raise RuntimeError("teacher cache example has an invalid schema")
        if item["schema_version"] != CACHE_SCHEMA_VERSION or item["kind"] != EXAMPLE_KIND:
            raise RuntimeError("teacher cache example has an invalid identity")
        descriptor = item["descriptor"]
        if not torch.is_tensor(descriptor) or descriptor.shape != (32,) or not bool(torch.isfinite(descriptor).all()):
            raise RuntimeError("teacher cache descriptor is invalid")
        for family in ("compact", "teacher"):
            attributes = item[family]
            if not isinstance(attributes, Mapping) or set(attributes) != set(ATTRIBUTE_FIELDS):
                raise RuntimeError("teacher cache attributes have an invalid schema")
            if not all(torch.is_tensor(attributes[name]) and bool(torch.isfinite(attributes[name]).all()) for name in ATTRIBUTE_FIELDS):
                raise RuntimeError("teacher cache attributes are non-finite")
        anchors.append(item["anchor_index"])
    if anchors != sorted(anchors) or len(anchors) != len(set(anchors)):
        raise RuntimeError("teacher cache anchors are not in the frozen unique order")
    return len(examples)


def _cache_manifest(
    *,
    contract: Mapping[str, Any],
    model: str,
    split: str,
    runtime_binding: Mapping[str, Any],
    entries: list[dict[str, Any]],
    partial: bool,
) -> dict[str, Any]:
    expected_identity = contract["context_only_inputs"]["splits"][split]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": CACHE_KIND,
        "status": "SMOKE_NONCLAIMING" if partial else "PASS_AUTHOR_SIDE_OFFLINE_TEACHER",
        "partial": partial,
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
        "cross_check_threshold": float(
            contract["saes_routing"]["parameters"]["cross_check_threshold"]
        ),
        "input_identity": {
            key: expected_identity[key]
            for key in (
                "tree_sha256",
                "manifest_sha256",
                "input_provenance_sha256",
                "selection_sha256",
                "scene_count",
            )
        },
        "scene_count": len(entries),
        "record_count": sum(int(item["record_count"]) for item in entries),
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


def _validate_preparation_contract(
    preparation: Mapping[str, Any], *, contract: Mapping[str, Any], model: str
) -> None:
    """Reject a cache record whose model-native preprocessing drifted."""

    expected = contract["preprocessing"]
    if (
        preparation.get("source_image_shape") != expected["source_image_shape"]
        or preparation.get("prepared_image_shape") != expected["target_image_shape"]
    ):
        raise RuntimeError("model context preprocessing differs from the frozen training contract")
    if preparation.get("patch_size") != expected["per_model_patch_size"][model]:
        raise RuntimeError("model context patch size differs from the frozen training contract")


def compile_cache(
    *,
    contract_path: Path,
    materialization_root: Path,
    model: str,
    split: str,
    output_root: Path,
    sample_indices: Sequence[int] | None = None,
    device: torch.device,
) -> dict[str, Any]:
    """Compile one model/split cache, refusing a partial canonical cache."""

    if model not in MODELS or split not in {TRAIN_SPLIT, HOLDOUT_SPLIT}:
        raise ValueError("model or ACID calibration split is invalid")
    if Path(contract_path).resolve() != DEFAULT_TRAINING_CONTRACT_PATH:
        raise RuntimeError("teacher cache requires the canonical v5 frozen contract")
    require_isolated_model_process(model)
    contract = _read_json(contract_path, "joint training contract")
    validate_live_training_contract(contract, repository_root=ROOT)
    expected_materialization = (
        ROOT / contract["context_only_inputs"]["materialization_root"]
    ).resolve()
    if Path(materialization_root).resolve() != expected_materialization:
        raise RuntimeError("teacher cache requires the canonical context materialization")
    canonical_root = ROOT / contract["teacher_objective"]["teacher_cache_root"] / model / split
    output_root = Path(output_root).resolve()
    all_indices = list(range(int(contract["context_only_inputs"]["splits"][split]["scene_count"])))
    selected = all_indices if sample_indices is None else sorted(set(int(value) for value in sample_indices))
    if not selected or any(value < 0 or value not in all_indices for value in selected):
        raise ValueError("teacher cache sample indices are invalid")
    partial = selected != all_indices
    if partial and output_root == canonical_root.resolve():
        raise ValueError("a partial teacher cache may not occupy the canonical cache path")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite teacher cache output: {output_root}")
    partial_root = output_root.with_name(output_root.name + ".partial")
    if partial_root.exists():
        raise FileExistsError(f"teacher cache partial output already exists: {partial_root}")
    partial_root.mkdir(parents=True, exist_ok=False)

    experiment = resolve_experiment(model, "acid", ROOT)
    expected_binding = contract["checkpoint_binding"]["models"][model]
    expected_config = contract["model_config_binding"][model]
    if (
        experiment.checkpoint.relative_to(ROOT).as_posix() != expected_binding["checkpoint_path"]
        or experiment.experiment != expected_binding["checkpoint_dataset"]
        or list(experiment.hydra_overrides) != expected_config["hydra_overrides"]
    ):
        raise RuntimeError("model experiment mapping differs from frozen training contract")
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
        raise RuntimeError("context-only cache worker unexpectedly constructed a decoder")
    if (
        validate_runtime_checkpoint_binding(
            contract,
            model=model,
            checkpoint_path=experiment.checkpoint,
            repository_root=ROOT,
        )
        != checkpoint_sha256
    ):
        raise RuntimeError("runtime checkpoint identity changed during model loading")
    resolved_config_sha256 = _resolved_config_sha256(bundle.config)
    runtime_binding = validate_runtime_model_binding(
        contract,
        model=model,
        resolved_config_sha256=resolved_config_sha256,
        repository_root=ROOT,
        require_loaded_src=True,
    )
    runtime_binding = {**runtime_binding, "checkpoint_sha256": checkpoint_sha256}
    entries: list[dict[str, Any]] = []
    try:
        for sample_index in selected:
            raw = load_acid_joint_context(
                materialization_root=materialization_root,
                split=split,
                sample_index=sample_index,
                plan_path=DEFAULT_PLAN_PATH,
            )
            context, preparation = prepare_acid_joint_model_context(
                raw,
                dataset_cfg=bundle.config.dataset,
                encoder_cfg=bundle.config.model.encoder,
                device=device,
            )
            _validate_preparation_contract(
                preparation, contract=contract, model=model
            )
            if set(context) != {"image", "extrinsics", "intrinsics", "index", "near", "far"}:
                raise RuntimeError("model worker context contains a target-side field")
            dense_gpu, features_gpu, depths_gpu = _capture_encoder_execution(model, bundle.encoder, context)
            dense = _copy_gaussians_to_cpu(dense_gpu)
            features = features_gpu.detach().to(device="cpu", dtype=torch.float32).clone()
            depths = depths_gpu.detach().to(device="cpu", dtype=torch.float32).clone()
            record = _record_for_scene(
                contract=contract,
                model=model,
                split=split,
                sample_index=sample_index,
                source_identity=raw.identity,
                preparation=preparation,
                dense=dense,
                features=features,
                depths=depths,
                runtime_binding=runtime_binding,
                context=context,
            )
            record_count = _validate_scene_record(
                record, contract=contract, model=model
            )
            filename = f"{sample_index:06d}.pt"
            path = partial_root / filename
            torch.save(record, path)
            entries.append(
                {
                    "sample_index": sample_index,
                    "scene": raw.identity["scene"],
                    "path": filename,
                    "sha256": _sha256_file(path),
                    "byte_count": path.stat().st_size,
                    "record_count": record_count,
                    "routing": record["routing"],
                }
            )
            del dense_gpu, features_gpu, depths_gpu, dense, features, depths, context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        manifest = _cache_manifest(
            contract=contract,
            model=model,
            split=split,
            runtime_binding=runtime_binding,
            entries=entries,
            partial=partial,
        )
        manifest_path = partial_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial_root, output_root)
        return {
            "output_root": str(output_root),
            "manifest_sha256": _sha256_file(output_root / "manifest.json"),
            "scene_count": len(entries),
            "record_count": manifest["record_count"],
            "partial": partial,
        }
    except BaseException:
        # Preserve partial evidence for diagnosis rather than silently deleting
        # a failed author-side compiler output.
        raise
    finally:
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION_ROOT)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=(TRAIN_SPLIT, HOLDOUT_SPLIT), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, action="append")
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
            raise RuntimeError("CUDA was requested for the isolated teacher cache worker")
        result = compile_cache(
            contract_path=args.contract,
            materialization_root=args.materialization_root,
            model=args.model,
            split=args.split,
            output_root=args.output_root,
            sample_indices=args.sample_index,
            device=device,
        )
    except (JointTrainingContractError, OSError, RuntimeError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
