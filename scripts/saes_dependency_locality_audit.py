#!/usr/bin/env python3
"""Measure fixed-tile non-probe influence on retained raw S3 probes."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.saes_dependency_audit import (
    _capture_raw_head,
    _context_on_device,
    _raw_head_modules,
    build_probe_mask,
    remove_target_rgb,
    summarize_probe_dependency,
)


def fixed_locality_tile_origins(
    *, height: int, width: int, tile_size: int
) -> list[tuple[int, int]]:
    """Return the predeclared top/center/bottom by left/center/right tiles."""
    if min(height, width, tile_size) <= 0:
        raise ValueError("height, width, and tile_size must be positive")
    if height % tile_size or width % tile_size:
        raise ValueError("activation dimensions must be divisible by tile_size")
    tile_rows = height // tile_size
    tile_columns = width // tile_size
    if min(tile_rows, tile_columns) < 3:
        raise ValueError("locality audit requires at least three tiles per dimension")
    row_indices = (0, tile_rows // 2, tile_rows - 1)
    column_indices = (0, tile_columns // 2, tile_columns - 1)
    return [
        (tile_row * tile_size, tile_column * tile_size)
        for tile_row in row_indices
        for tile_column in column_indices
    ]


def source_tile_nonprobe_mask(
    *, probe_mask: torch.Tensor, source_y: int, source_x: int, tile_size: int
) -> torch.Tensor:
    """Select exactly the non-probe positions of one fixed full-resolution tile."""
    if probe_mask.ndim != 2 or probe_mask.dtype != torch.bool:
        raise ValueError("probe_mask must be a two-dimensional bool tensor")
    height, width = probe_mask.shape
    if (
        source_y < 0
        or source_x < 0
        or source_y % tile_size
        or source_x % tile_size
        or source_y + tile_size > height
        or source_x + tile_size > width
    ):
        raise ValueError("source tile is outside the full-resolution activation")
    mask = torch.zeros_like(probe_mask)
    mask[source_y : source_y + tile_size, source_x : source_x + tile_size] = True
    return mask & ~probe_mask


def summarize_locality_dependency(
    baseline: torch.Tensor,
    perturbed: torch.Tensor,
    probe_mask: torch.Tensor,
    *,
    source_y: int,
    source_x: int,
    tile_size: int,
) -> dict[str, Any]:
    """Report the changed retained-probe envelope for one source-tile perturbation."""
    if (
        baseline.ndim != 4
        or baseline.shape != perturbed.shape
        or probe_mask.shape != baseline.shape[-2:]
        or probe_mask.dtype != torch.bool
    ):
        raise ValueError("raw-head tensors and probe mask are incompatible")
    source_mask = source_tile_nonprobe_mask(
        probe_mask=probe_mask,
        source_y=source_y,
        source_x=source_x,
        tile_size=tile_size,
    )
    changed_spatial = (perturbed - baseline).abs().amax(dim=(0, 1)) > 0
    changed_probes = changed_spatial & probe_mask.to(changed_spatial.device)
    coordinates = changed_probes.nonzero(as_tuple=False).cpu().tolist()
    source_grid = (source_y // tile_size, source_x // tile_size)
    changed_grids = [(row // tile_size, column // tile_size) for row, column in coordinates]
    if changed_grids:
        rows = [row for row, _ in changed_grids]
        columns = [column for _, column in changed_grids]
        max_distance = max(
            max(abs(row - source_grid[0]), abs(column - source_grid[1]))
            for row, column in changed_grids
        )
        bounds: dict[str, int] | None = {
            "minimum_row": min(rows),
            "maximum_row": max(rows),
            "minimum_column": min(columns),
            "maximum_column": max(columns),
        }
    else:
        max_distance = None
        bounds = None
    probe_summary = summarize_probe_dependency(baseline, perturbed, probe_mask)
    return {
        "source_tile": {
            "origin": [source_y, source_x],
            "grid_index": list(source_grid),
            "nonprobe_positions_zeroed": int(source_mask.sum().item()),
        },
        "probe_dependency": probe_summary,
        "changed_probe_spatial_position_count": len(coordinates),
        "changed_probe_tile_bounds": bounds,
        "maximum_changed_probe_tile_chebyshev_distance": max_distance,
        "changed_probe_outside_source_tile": any(
            grid != source_grid for grid in changed_grids
        ),
    }


def _target_free_context(
    batch: dict[str, Any], device: torch.device
) -> tuple[bool, dict[str, Any]]:
    """Delete target RGB before moving the context-only audit inputs to device."""
    native_target_rgb_loaded = remove_target_rgb(batch)
    return native_target_rgb_loaded, _context_on_device(batch, device)


def collect_locality_audit(
    *, model_name: str, sample_index: int, device: torch.device
) -> dict[str, Any]:
    """Run one model's fixed target-free 3x3 refinement-dependency audit."""
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity

    experiment = resolve_experiment(model_name, "dl3dv", ROOT)
    selection = resolve_claim_selection(model_name, "dl3dv", ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        model_name,
        dataset_name="dl3dv",
        checkpoint_path=experiment.checkpoint,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=sample_index + 1,
        sample_index=sample_index,
    )
    model.eval()
    native_target_rgb_loaded, context = _target_free_context(batch, loaded_device)
    _, _, _, height, width = context["image"].shape
    tile_size = 4
    probe_mask = build_probe_mask(height=height, width=width, tile_size=tile_size)
    _, _, perturb_module_name = _raw_head_modules(model, model_name)
    baseline = _capture_raw_head(
        model, context, probe_mask, model_name=model_name, perturb_nonprobes=False
    )
    tile_reports = []
    for source_y, source_x in fixed_locality_tile_origins(
        height=height, width=width, tile_size=tile_size
    ):
        perturb_mask = source_tile_nonprobe_mask(
            probe_mask=probe_mask,
            source_y=source_y,
            source_x=source_x,
            tile_size=tile_size,
        )
        perturbed = _capture_raw_head(
            model, context, probe_mask, model_name=model_name,
            perturb_nonprobes=True, perturb_mask=perturb_mask,
        )
        tile_reports.append(
            summarize_locality_dependency(
                baseline,
                perturbed,
                probe_mask,
                source_y=source_y,
                source_x=source_x,
                tile_size=tile_size,
            )
        )
    return {
        "schema_version": "1.0",
        "kind": "saes_s2s3_refinement_dependency_locality_audit",
        "paper_result_eligible": False,
        "target_rgb_accessed": False,
        "native_dataloader_loaded_target_rgb": native_target_rgb_loaded,
        "model": model_name,
        "dataset": "dl3dv",
        "sample_index": sample_index,
        "scene": str(batch["scene"][0]),
        "context_indices": [
            int(value) for value in batch["context"]["index"][0].tolist()
        ],
        "probe_pattern": "four-corners-per-4x4-full-resolution-tile",
        "source_tiles": "fixed top-center-bottom by left-center-right 3x3 raster",
        "perturbation": {
            "module": perturb_module_name,
            "input": "full-resolution [VB,C,H,W] refinement activation",
            "action": "set one source tile's 12 non-probe positions to zero",
            "routing_signal_used": False,
        },
        "execution_boundary": {
            "completed": ("S1", "S2", "S3_raw_gaussian_head"),
            "ggu_s4_executed": False,
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "raw_head_capture_terminates_execution": True,
        },
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": native_target_rgb_loaded,
            "removed_before_context_device_transfer": True,
            "passed_to_model": False,
            "used_for_routing_or_metric": False,
        },
        "raw_head_shape": list(baseline.shape),
        "tile_reports": tile_reports,
        "checkpoint_sha256": cached_sha256_file(experiment.checkpoint),
        "source": source_identity(),
    }


def collect_transplat_locality_audit(
    *, sample_index: int, device: torch.device
) -> dict[str, Any]:
    """Compatibility wrapper for the original fixed TranSplat locality audit."""
    return collect_locality_audit(
        model_name="transplat", sample_index=sample_index, device=device
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("transplat", "mvsplat", "depthsplat"), default="transplat"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_index != 0:
        parser.error("the locality audit is predeclared for DL3DV sample index 0")
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    record = collect_locality_audit(
        model_name=args.model, sample_index=args.sample_index, device=device
    )
    from scripts.result_record import portable_command, write_result

    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")


if __name__ == "__main__":
    main()
