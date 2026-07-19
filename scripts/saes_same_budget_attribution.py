#!/usr/bin/env python3
"""Attribute fixed DL3DV SAES oracle error without accessing target RGB."""

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

from scripts.saes_dependency_audit import _context_on_device, remove_target_rgb
from scripts.saes_same_budget_dense_oracle import (
    MATERIALIZATION,
    SEED,
    _oracle_output_accounting,
)
from scripts.saes_selected_output_quality_gate import (
    _mean_metrics,
    _render,
    _retain_renderable_gaussians,
    _sha256_mask,
    _target_camera_inputs,
    _view_metrics,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution
from scripts.saes_target_free_materialization_audit import (
    _capture_encoder_execution,
    _clone_gaussians,
)


KIND = "saes_same_budget_three_way_attribution"
ATTRIBUTE_NAMES = ("means", "covariances", "harmonics", "opacities")


def _representative_update_mask(
    baseline: Any, merged: Any, retained: torch.Tensor
) -> torch.Tensor:
    """Return retained slots changed by the oracle's representative update."""
    if (
        retained.ndim != 1
        or retained.dtype != torch.bool
        or retained.numel() != baseline.means.shape[1]
    ):
        raise ValueError("retained mask must match the dense Gaussian layout")
    changed = torch.zeros_like(retained)
    for name in ATTRIBUTE_NAMES:
        before = getattr(baseline, name)[0]
        after = getattr(merged, name)[0]
        delta = (after - before).abs().reshape(before.shape[0], -1)
        changed |= delta.any(dim=1)
    return retained & changed


def _build_attribution_variants(
    baseline: Any,
    merged: Any,
    modified: torch.Tensor,
    *,
    views: int,
    height: int,
    width: int,
    stats: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Materialize exactly the three fixed attribution variants.

    ``drop_only`` compacts the same removed slots but leaves each retained
    descriptor untouched. ``merge_only`` keeps every dense slot but copies
    only representative updates. ``drop_merge`` is the committed oracle
    output. All comparisons therefore share one route and one descriptor mask.
    """
    from scripts.saes_diagnostics import representative_indices

    retained = ~modified
    representative_indices_tensor = representative_indices(
        modified,
        view_count=views,
        height=height,
        width=width,
        tile_size=4,
        primitives_per_pixel=1,
    )
    representative_slots = torch.zeros_like(retained)
    representative_slots[representative_indices_tensor] = True
    full_slots = ~(modified | representative_slots)
    if bool((modified & representative_slots).any()) or bool(
        (modified | representative_slots | full_slots).logical_not().any()
    ):
        raise RuntimeError("same-budget attribution masks are not a partition")
    removed_count = int(modified.sum().item())
    representative_count = int(representative_slots.sum().item())
    full_count = int(full_slots.sum().item())
    if removed_count != int(stats["zeroed_gaussians"]):
        raise RuntimeError("attribution removed mask disagrees with oracle zeroed count")
    if representative_count != int(stats["same_budget_dense_oracle_output_gaussians"]):
        raise RuntimeError("attribution representative mask disagrees with K/2K count")
    if full_count != int(stats["full_stage3_gaussians"]):
        raise RuntimeError("attribution Full mask disagrees with oracle count")
    from saes.probe_layout import (
        compute_lightweight_positions,
        compute_probe_positions,
    )

    tile_positions = 4 * 4
    expected_removed = (
        int(stats["level0_tiles"])
        * (tile_positions - len(compute_probe_positions(4)))
        + int(stats["level1_tiles"])
        * (tile_positions - len(compute_lightweight_positions(4)))
    )
    if removed_count != expected_removed:
        raise RuntimeError("attribution mask does not satisfy the fixed L0/L1 budgets")
    changed_representatives = _representative_update_mask(
        baseline, merged, representative_slots
    )
    merge_only = _clone_gaussians(baseline)
    for name in ATTRIBUTE_NAMES:
        destination = getattr(merge_only, name)
        source = getattr(merged, name)
        destination[:, representative_slots] = source[:, representative_slots]
    variants = {
        "drop_only": _retain_renderable_gaussians(baseline, retained),
        "merge_only": merge_only,
        "drop_merge": _retain_renderable_gaussians(merged, retained),
    }
    for name in ATTRIBUTE_NAMES:
        torch.testing.assert_close(
            getattr(merged, name)[:, full_slots],
            getattr(baseline, name)[:, full_slots],
        )
        if name != "opacities":
            torch.testing.assert_close(
                getattr(merged, name)[:, modified],
                getattr(baseline, name)[:, modified],
            )
        else:
            if torch.count_nonzero(getattr(merged, name)[:, modified]):
                raise RuntimeError("canonical oracle did not zero every removed opacity")
        torch.testing.assert_close(
            getattr(variants["drop_only"], name), getattr(baseline, name)[:, retained]
        )
        torch.testing.assert_close(
            getattr(variants["drop_merge"], name), getattr(merged, name)[:, retained]
        )
        unchanged = ~representative_slots
        torch.testing.assert_close(
            getattr(variants["merge_only"], name)[:, unchanged],
            getattr(baseline, name)[:, unchanged],
        )
    return variants, {
        "full_gaussians": int(retained.numel()),
        "removed_nonprobes": removed_count,
        "retained_gaussians": int(retained.sum().item()),
        "representative_slots": representative_count,
        "representative_slots_with_nonzero_update": int(
            changed_representatives.sum().item()
        ),
        "full_passthrough_gaussians": full_count,
        "unchanged_full_or_retained_gaussians": int((~representative_slots).sum().item()),
    }


def _teacher_fidelity(images: torch.Tensor, teacher: torch.Tensor) -> dict[str, Any]:
    """Measure one committed render against dense renderer outputs only."""
    if images.shape != teacher.shape:
        raise RuntimeError("teacher and attribution renders have different shapes")
    values = _view_metrics(images, teacher)
    mse = (images - teacher).square().mean(dim=(1, 2, 3))
    return {
        "teacher_kind": "dense-baseline-render",
        "mean": _mean_metrics(values),
        "mse": {
            "mean": float(mse.mean().item()),
            "maximum": float(mse.max().item()),
        },
        "views": values,
    }


def collect_attribution(*, device: torch.device) -> dict[str, Any]:
    """Run the one fixed three-way error attribution for TranSplat sample zero."""
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
    native_target_rgb_loaded = remove_target_rgb(batch)
    target = _target_camera_inputs(batch, loaded_device)
    context = _context_on_device(batch, loaded_device)
    _, views, _, height, width = context["image"].shape
    if context["image"].shape[0] != 1:
        raise RuntimeError("same-budget attribution requires batch size one")

    with strict_fp32_convolution_execution() as numerical_execution:
        baseline, features, depths = _capture_encoder_execution(model, context)
        full_gaussians = int(baseline.means.shape[1])
        if full_gaussians != views * height * width:
            raise RuntimeError("same-budget attribution requires one Gaussian per pixel")
        merged = _clone_gaussians(baseline)
        modified, stats = _apply_oracle(
            merged,
            features=features,
            depths=depths,
            context=context,
            height=height,
            width=width,
            views=views,
        )
        accounting = _oracle_output_accounting(
            modified=modified, stats=stats, full_gaussians=full_gaussians
        )
        variants, variant_accounting = _build_attribution_variants(
            baseline,
            merged,
            modified,
            views=views,
            height=height,
            width=width,
            stats=stats,
        )
        teacher = _render(model, baseline, target, (height, width))
        renders = {
            name: _render(model, gaussians, target, (height, width))
            for name, gaussians in variants.items()
        }

    fidelity = {
        name: _teacher_fidelity(images, teacher)
        for name, images in renders.items()
    }
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
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": native_target_rgb_loaded,
            "removed_before_encoder": True,
            "available_to_route_or_teacher": False,
            "used_for_metrics": False,
        },
        "execution_boundary": {
            "dense_s1_s2_s3_executed": True,
            "posthoc_full_s3_read_allowed_for_diagnostic": True,
            "target_camera_metadata_used": True,
            "dense_renderer_output_is_teacher": True,
            "target_rgb_used": False,
            "runtime_execution": False,
            "s2_s3_savings_claimed": 0.0,
        },
        "routing": {
            "feature_threshold": 0.2,
            "depth_threshold": 0.1,
            "tile_size": 4,
            "materialization": MATERIALIZATION,
            "committed_modified_mask_sha256": _sha256_mask(modified),
            "stats": stats,
        },
        "same_budget_output_accounting": accounting,
        "three_way_variant_accounting": variant_accounting,
        "teacher_fidelity": fidelity,
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


def _apply_oracle(
    gaussians: Any,
    *,
    features: torch.Tensor,
    depths: torch.Tensor,
    context: dict[str, Any],
    height: int,
    width: int,
    views: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Keep the attribution route exactly equal to the fixed dense oracle."""
    from scripts.saes_selected_output_quality_gate import _apply_fixed_saes

    return _apply_fixed_saes(
        gaussians,
        features=features,
        depths=depths,
        context=context,
        height=height,
        width=width,
        views=views,
        materialization=MATERIALIZATION,
    )


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
    record = collect_attribution(device=device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / "results.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
