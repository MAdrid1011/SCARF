#!/usr/bin/env python3
"""Audit the fixed DL3DV L1 correction without opening target RGB or rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.verify_prepared_dataset import verify_tree_manifest
from saes.progressive_saes import apply_progressive_saes
from scripts.calibration_inputs import sha256_file, validate_target_free_input_root
from scripts.saes_dependency_audit import _context_on_device
from scripts.saes_diagnostics import materialization_attribute_audit
from scripts.saes_target_free_materialization_audit import (
    _capture_encoder_execution,
    _clone_gaussians,
    _gaussians_on_cpu,
    _mask_sha256,
    _max_attribute_delta,
    _poison_skipped_descriptors,
)


MODEL = "transplat"
DATASET = "dl3dv"
SAMPLE_INDEX = 0
SEED = 0
TILE_SIZE = 4
FEATURE_THRESHOLD = 0.20
DEPTH_THRESHOLD = 0.10
DECISION_SEMANTICS = "probe-normalized-std-first-hit"
DEPTH_ROUTING_SEMANTICS = "metric-depth-standard-deviation"
MATERIALIZATION = "conditional-adapter-offset-transport-diagnostic"
ATTRIBUTE_TRANSPORT_MATERIALIZATION = (
    "conditional-adapter-offset-attribute-transport-diagnostic"
)
AUDIT_KIND = "saes_l1_primary_reference_target_free_attribute_audit"
ATTRIBUTE_TRANSPORT_AUDIT_KIND = (
    "saes_l1_primary_reference_attribute_transport_target_free_attribute_audit"
)
FIXED_AUDIT_MATERIALIZATIONS = frozenset(
    (
        MATERIALIZATION,
        ATTRIBUTE_TRANSPORT_MATERIALIZATION,
    )
)
INPUT_KIND = "dl3dv_target_free_l1_primary_reference_audit_input"


def _load_audit_input(input_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    input_root = input_root.resolve()
    try:
        record = json.loads((input_root / "audit-input.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("target-free audit input record is unavailable or invalid") from exc
    if not isinstance(record, dict) or record.get("kind") != INPUT_KIND:
        raise ValueError("target-free audit input has the wrong kind")
    if (
        record.get("status") != "PASS"
        or record.get("model") != MODEL
        or record.get("dataset") != DATASET
        or record.get("source_sample_index") != SAMPLE_INDEX
        or record.get("target_rgb_included") is not False
        or record.get("target_rgb_opened") is not False
        or record.get("target_rgb_paths_passed_to_encoder") is not False
    ):
        raise ValueError("target-free audit input violates the fixed RGB contract")
    selected = record.get("selected_sample")
    if (
        not isinstance(selected, dict)
        or len(selected.get("context_indices", ())) != 2
        or len(selected.get("target_indices", ())) != 4
        or set(selected["context_indices"]) & set(selected["target_indices"])
    ):
        raise ValueError("target-free audit input has invalid fixed view selection")
    protocol = record.get("canonical_protocol")
    if not isinstance(protocol, dict) or protocol.get("pair") != "transplat/dl3dv":
        raise ValueError("target-free audit input has the wrong protocol binding")
    if not isinstance(protocol.get("source_index_sha256"), str):
        raise ValueError("target-free audit input has no source index hash")
    sidecar_root = input_root / "sidecar"
    sidecar = validate_target_free_input_root(sidecar_root, DATASET)
    if sidecar["target_rgb_included"] is not False:
        raise ValueError("target-free audit sidecar contains target RGB")
    if sidecar["selection_sha256"] != selected.get("audit_selection_sha256"):
        raise ValueError("target-free audit sidecar selection does not match its input")
    selection_path = input_root / "audit-selection.json"
    if not selection_path.is_file():
        raise ValueError("target-free audit input has no fixed selection file")
    tree = verify_tree_manifest(input_root, input_root / ".scarf-manifest.json")
    return record, sidecar, tree


def collect_attribute_audit(
    *,
    input_root: Path,
    device: torch.device,
    materialization: str = MATERIALIZATION,
    audit_kind: str = AUDIT_KIND,
) -> dict[str, Any]:
    """Run the fixed, context-only L1 correction oracle on sample zero."""
    from scripts.ae_config import resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    if materialization not in FIXED_AUDIT_MATERIALIZATIONS:
        raise ValueError("target-free attribute audit has an unsupported materialization")
    if not isinstance(audit_kind, str) or not audit_kind:
        raise ValueError("target-free attribute audit has an invalid kind")
    input_root = input_root.resolve()
    input_record, sidecar_identity, input_tree = _load_audit_input(input_root)
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        MODEL,
        dataset_name=DATASET,
        checkpoint_path=experiment.checkpoint,
        dataset_root=input_root / "sidecar",
        evaluation_index=input_root / "audit-selection.json",
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=1,
        sample_index=SAMPLE_INDEX,
        calibration_target_free=True,
        encoder_only=True,
    )
    if "image" in batch.get("target", {}):
        raise RuntimeError("target-free audit batch contains target RGB before encoder execution")
    if batch.get("scene") != [input_record["selected_sample"]["scene"]]:
        raise RuntimeError("target-free audit batch scene does not match its sidecar")
    model.eval()
    context = _context_on_device(batch, loaded_device)
    source_gaussians, features, depths = _capture_encoder_execution(model, context)
    _, views, _, height, width = context["image"].shape
    native_encoder_device = str(source_gaussians.means.device)
    source_gaussians = _gaussians_on_cpu(source_gaussians)
    features = features.detach().cpu()
    depths = depths.detach().cpu()
    audit_extrinsics = context["extrinsics"].detach().cpu()
    audit_intrinsics = context["intrinsics"].detach().cpu()
    audit_near = context["near"].detach().cpu()
    audit_far = context["far"].detach().cpu()
    del model, context
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    materialized = _clone_gaussians(source_gaussians)
    mask, stats, _ = apply_progressive_saes(
        materialized,
        height,
        width,
        tile_size=TILE_SIZE,
        feature_var_threshold=FEATURE_THRESHOLD,
        depth_std_threshold=DEPTH_THRESHOLD,
        features=features,
        depths=depths,
        view_count=views,
        materialization=materialization,
        decision_semantics=DECISION_SEMANTICS,
        context_extrinsics=audit_extrinsics,
        context_intrinsics=audit_intrinsics,
        depth_routing_semantics=DEPTH_ROUTING_SEMANTICS,
        depth_near=audit_near,
        depth_far=audit_far,
    )
    skipped = mask.nonzero(as_tuple=False).flatten()
    retained = (~mask).nonzero(as_tuple=False).flatten()
    if skipped.numel() == 0:
        raise RuntimeError("fixed target-free audit produced no sparse work")
    poisoned = _clone_gaussians(source_gaussians)
    _poison_skipped_descriptors(poisoned, skipped)
    poisoned_mask, poisoned_stats, _ = apply_progressive_saes(
        poisoned,
        height,
        width,
        tile_size=TILE_SIZE,
        feature_var_threshold=FEATURE_THRESHOLD,
        depth_std_threshold=DEPTH_THRESHOLD,
        features=features,
        depths=depths,
        view_count=views,
        materialization=materialization,
        decision_semantics=DECISION_SEMANTICS,
        context_extrinsics=audit_extrinsics,
        context_intrinsics=audit_intrinsics,
        depth_routing_semantics=DEPTH_ROUTING_SEMANTICS,
        depth_near=audit_near,
        depth_far=audit_far,
    )
    if not torch.equal(mask, poisoned_mask):
        raise RuntimeError("poisoned descriptor audit changed the fixed route")
    retained_delta = _max_attribute_delta(materialized, poisoned, retained)
    if any(value != 0.0 for value in retained_delta.values()):
        raise RuntimeError("retained attributes depend on skipped raw descriptors")
    if stats != poisoned_stats:
        raise RuntimeError("poisoned descriptor audit changed SAES event statistics")
    if (
        stats["covariance_psd_violations"]
        or stats["guard_nonprobe_s3_attribute_reads"]
        or stats["opacity_transmittance_error_max"]
        or stats["assignment_weight_sum_error_max"] > 1.0e-5
    ):
        raise RuntimeError("fixed target-free audit violated a numerical or access invariant")

    oracle = materialization_attribute_audit(
        source_gaussians,
        materialized,
        features=features,
        depths=depths,
        height=height,
        width=width,
        tile_size=TILE_SIZE,
        feature_threshold=FEATURE_THRESHOLD,
        depth_threshold=DEPTH_THRESHOLD,
        view_count=views,
        decision_semantics=DECISION_SEMANTICS,
        materialization=materialization,
        effective_mask=mask,
    )
    if oracle["route_source"] != "committed_sparse_output_mask":
        raise RuntimeError("posthoc attribute oracle did not use the committed sparse mask")
    return {
        "schema_version": "1.0",
        "kind": audit_kind,
        "status": "COMPLETED",
        "paper_result_eligible": False,
        "expected_results_accessed": False,
        "target_rgb_accessed": False,
        "model": MODEL,
        "dataset": DATASET,
        "sample_index": SAMPLE_INDEX,
        "scene": input_record["selected_sample"]["scene"],
        "context_indices": input_record["selected_sample"]["context_indices"],
        "target_indices": input_record["selected_sample"]["target_indices"],
        "execution_boundary": {
            "completed": ("S1", "S2", "S3", "S4", "SAES_attribute_audit"),
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "decoder_executed": False,
            "hardware_cycle_simulator_executed": False,
        },
        "target_rgb_provenance": {
            "native_dataloader_loaded_target_rgb": False,
            "target_image_present_before_encoder": False,
            "target_rgb_in_sidecar": False,
            "target_rgb_passed_to_model": False,
        },
        "fixed_contract": {
            "seed": SEED,
            "tile_size": TILE_SIZE,
            "feature_threshold": FEATURE_THRESHOLD,
            "depth_threshold": DEPTH_THRESHOLD,
            "feature_decision_semantics": DECISION_SEMANTICS,
            "depth_routing_semantics": DEPTH_ROUTING_SEMANTICS,
            "materialization": materialization,
            "l1_depth_reference": "primary-probes",
            "l1_anchor_layout": "2K-native-selected-anchors",
        },
        "input_provenance": {
            "audit_input_sha256": sha256_file(input_root / "audit-input.json"),
            "audit_input_tree_sha256": input_tree["tree_sha256"],
            "audit_input_manifest_sha256": input_tree["manifest_sha256"],
            "sidecar": sidecar_identity,
            "opened_source_files": input_record["opened_source_files"],
        },
        "native_execution": {
            "native_encoder_device": native_encoder_device,
            "saes_attribute_audit_device": "cpu",
            "features_shape": list(features.shape),
            "depths_shape": list(depths.shape),
            "gaussian_count": int(source_gaussians.means.shape[1]),
        },
        "selection": {
            "skipped_descriptor_count": int(skipped.numel()),
            "retained_descriptor_count": int(retained.numel()),
            "skipped_mask_sha256": _mask_sha256(mask),
        },
        "saes_stats": stats,
        "skipped_descriptor_poison_audit": {
            "poison_value": 1.0e4,
            "route_identical": True,
            "event_statistics_identical": True,
            "retained_maximum_absolute_delta": retained_delta,
            "skipped_stage3_attributes_read": False,
        },
        "retained_attribute_transport": {
            "enabled": (
                materialization == ATTRIBUTE_TRANSPORT_MATERIALIZATION
            ),
            "maximum_absolute_update": _max_attribute_delta(
                source_gaussians, materialized, retained
            ),
        },
        "posthoc_full_stage3_oracle": {
            **oracle,
            "reference_use": "posthoc-only",
            "routing_signal_used": False,
            "parameter_selection_used": False,
        },
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "source": source_identity(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    record = collect_attribute_audit(input_root=args.input_root, device=device)
    from scripts.result_record import portable_command, write_result

    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
