#!/usr/bin/env python3
"""Run the fixed non-runtime DL3DV same-budget dense SAES oracle."""

from __future__ import annotations

import argparse
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

from scripts.saes_dependency_audit import _context_on_device
from scripts.saes_selected_output_quality_gate import (
    DECISION_SEMANTICS,
    DEPTH_ROUTING_SEMANTICS,
    _apply_fixed_saes,
    _image_equivalence,
    _mean_metrics,
    _quality_verdict,
    _render,
    _retain_renderable_gaussians,
    _sha256_mask,
    _take_target_rgb_for_metrics,
    _target_camera_inputs,
    _view_metrics,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution
from scripts.saes_target_free_materialization_audit import (
    _capture_encoder_execution,
    _clone_gaussians,
)


MATERIALIZATION = "same-budget-dense-oracle-diagnostic"
KIND = "saes_same_budget_dense_oracle"
SEED = 0


def _oracle_output_accounting(
    *,
    modified: torch.Tensor,
    stats: dict[str, Any],
    full_gaussians: int,
) -> dict[str, Any]:
    """Validate that post-hoc reads still produce only K/2K/Full outputs."""
    if (
        modified.ndim != 1
        or modified.dtype != torch.bool
        or modified.numel() != full_gaussians
    ):
        raise RuntimeError("oracle modified mask does not match the dense descriptor layout")
    retained = int((~modified).sum().item())
    l0_outputs = int(stats["l0_representatives"])
    l1_outputs = int(stats["l1_lightweight_anchors"])
    full_outputs = int(stats["full_stage3_gaussians"])
    expected_retained = l0_outputs + l1_outputs + full_outputs
    oracle_outputs = l0_outputs + l1_outputs
    if retained != expected_retained:
        raise RuntimeError(
            "oracle retained descriptor count disagrees with L0/L1/Full accounting: "
            f"{retained} vs {expected_retained}"
        )
    if int(stats["effective_gaussians"]) != retained:
        raise RuntimeError("oracle effective Gaussian count disagrees with committed mask")
    if int(stats["same_budget_dense_oracle_output_gaussians"]) != oracle_outputs:
        raise RuntimeError("oracle K/2K output counter disagrees with routing counters")
    if stats.get("same_budget_dense_oracle_runtime_eligible") is not False:
        raise RuntimeError("same-budget dense oracle must remain non-runtime")
    return {
        "full_gaussians": full_gaussians,
        "retained_gaussians": retained,
        "removed_gaussians": full_gaussians - retained,
        "effective_gaussians_less_than_full": retained < full_gaussians,
        "l0_output_gaussians": l0_outputs,
        "l1_output_gaussians": l1_outputs,
        "full_output_gaussians": full_outputs,
        "same_budget_oracle_output_gaussians": oracle_outputs,
        "same_budget_output_count_matches_l0_l1": True,
        "retained_count_matches_l0_l1_full": True,
    }


def _oracle_success_verdict(
    quality_gate: dict[str, Any], accounting: dict[str, Any]
) -> dict[str, Any]:
    """Keep a baseline-identical all-Full result from passing the oracle gate."""
    reduction_present = bool(accounting["effective_gaussians_less_than_full"])
    quality_limit_pass = bool(quality_gate["pass"])
    return {
        "quality_limit_pass": quality_limit_pass,
        "nonzero_same_budget_reduction": reduction_present,
        "same_budget_dense_oracle_meets_quality_limit": (
            quality_limit_pass and reduction_present
        ),
    }


def collect_oracle(*, device: torch.device) -> dict[str, Any]:
    """Capture one dense sample, reduce it post-hoc, then score the committed output."""
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    experiment = resolve_experiment("transplat", "dl3dv", ROOT)
    selection = resolve_claim_selection("transplat", "dl3dv", ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        "transplat",
        dataset_name="dl3dv",
        checkpoint_path=experiment.checkpoint,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=1,
        sample_index=0,
    )
    model.eval()
    native_target_rgb_loaded = "image" in batch["target"]
    target = _target_camera_inputs(batch, loaded_device)
    context = _context_on_device(batch, loaded_device)
    _, views, _, height, width = context["image"].shape
    if context["image"].shape[0] != 1:
        raise RuntimeError("same-budget dense oracle requires batch size one")

    # The dense encoder completes before this diagnostic reads all Stage-3
    # descriptors. Target RGB remains untouched until the output below is
    # fully committed and rendered.
    with strict_fp32_convolution_execution() as numerical_execution:
        baseline_gaussians, features, depths = _capture_encoder_execution(model, context)
        full_gaussians = int(baseline_gaussians.means.shape[1])
        if full_gaussians != views * height * width:
            raise RuntimeError("same-budget dense oracle requires one Gaussian per pixel")
        oracle_gaussians = _clone_gaussians(baseline_gaussians)
        modified, stats = _apply_fixed_saes(
            oracle_gaussians,
            features=features,
            depths=depths,
            context=context,
            height=height,
            width=width,
            views=views,
            materialization=MATERIALIZATION,
        )
        accounting = _oracle_output_accounting(
            modified=modified, stats=stats, full_gaussians=full_gaussians
        )
        retained = ~modified
        renderable_oracle = _retain_renderable_gaussians(oracle_gaussians, retained)
        oracle_images = _render(model, renderable_oracle, target, (height, width))
        oracle_repeat_images = _render(
            model, renderable_oracle, target, (height, width)
        )
        baseline_images = _render(model, baseline_gaussians, target, (height, width))
    render_repeat = _image_equivalence(oracle_images, oracle_repeat_images)

    # This is the first target-RGB read. All route, moment, accounting, and
    # renderer work above is committed without target image information.
    target_images = _take_target_rgb_for_metrics(batch, loaded_device)
    baseline_views = _view_metrics(baseline_images, target_images[0])
    oracle_views = _view_metrics(oracle_images, target_images[0])
    baseline_quality = _mean_metrics(baseline_views)
    oracle_quality = _mean_metrics(oracle_views)
    quality_gate = _quality_verdict(baseline_quality, oracle_quality)
    oracle_success = _oracle_success_verdict(quality_gate, accounting)

    return {
        "schema_version": "1.0",
        "kind": KIND,
        "paper_result_eligible": False,
        "quality_retry_authorized": False,
        "model": "transplat",
        "dataset": "dl3dv",
        "sample_index": 0,
        "scene": str(batch["scene"][0]),
        "context_indices": [int(value) for value in batch["context"]["index"][0].tolist()],
        "target_indices": [int(value) for value in batch["target"]["index"][0].tolist()],
        "full_s3_reference_use": "posthoc-diagnostic-only",
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": native_target_rgb_loaded,
            "not_accessed_or_transferred_before_oracle_output_committed": True,
            "passed_to_encoder_or_route": False,
            "removed_from_native_batch_for_metrics": True,
            "first_read_after_oracle_output_committed": True,
        },
        "execution_boundary": {
            "dense_s1_s2_s3_executed": True,
            "posthoc_full_s3_read_allowed_for_diagnostic": True,
            "s2_s3_sparse_execution_verified": False,
            "runtime_execution": False,
            "paper_result_eligible": False,
            "s2_s3_savings_claimed": 0.0,
            "target_rgb_accessed_before_output_commit": False,
        },
        "routing": {
            "feature_threshold": 0.2,
            "depth_threshold": 0.1,
            "tile_size": 4,
            "decision_semantics": DECISION_SEMANTICS,
            "depth_routing_semantics": DEPTH_ROUTING_SEMANTICS,
            "materialization": MATERIALIZATION,
            "committed_modified_mask_sha256": _sha256_mask(modified),
            "stats": stats,
        },
        "same_budget_output_accounting": accounting,
        "renderer": {
            "target_camera_metadata_used": True,
            "target_rgb_used_for_moment_or_route": False,
            "repeat_equivalence": render_repeat,
        },
        "quality": {
            "baseline": baseline_quality,
            "oracle": oracle_quality,
            "views": [
                {
                    "target_index": int(batch["target"]["index"][0, index].item()),
                    "baseline": baseline_views[index],
                    "oracle": oracle_views[index],
                }
                for index in range(len(baseline_views))
            ],
        },
        "quality_gate": quality_gate,
        "oracle_verdict": {
            **oracle_success,
            "paper_result_eligible": False,
            "next_route_if_failed": (
                "improve only projected moment matching or use conservative Full fallback"
            ),
        },
        "provenance": {
            "seed": SEED,
            "checkpoint": {
                "path": str(experiment.checkpoint),
                "sha256": cached_sha256_file(experiment.checkpoint),
            },
            "evaluation_index": {
                "path": str(selection.index_path),
                "sha256": cached_sha256_file(selection.index_path),
            },
            "source": source_identity(),
            "numerical_execution": numerical_execution,
            "device": str(loaded_device),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    record = collect_oracle(device=device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / "results.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
