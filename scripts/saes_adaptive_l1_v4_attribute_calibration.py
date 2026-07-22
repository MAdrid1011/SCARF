#!/usr/bin/env python3
"""Freeze a target-free V4 selected-anchor attribute replay threshold."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.adaptive_l1_calibration import load_frozen_threshold as load_v15_threshold
from saes.adaptive_l1_v4_attribute_calibration import build_calibration_record
from saes.guarded_selected_route import ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY
from saes.probe_first_schedule import ADAPTIVE_L1_15_ANCHOR_SEMANTICS, build_incremental_probe_first_plan
from scripts.ae_config import resolve_claim_selection, resolve_experiment
from scripts.demo import load_model_and_data
from scripts.result_record import cached_sha256_file
from scripts.saes_incremental_selected_output_audit import (
    DECISION_SEMANTICS,
    DEPTH_THRESHOLD,
    FEATURE_THRESHOLD,
    TILE_SIZE,
    _capture_guarded_incremental_packed_adapter,
    _capture_s1_s2_without_dense_adapter,
    _release_cuda_cache,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution


DEFAULT_SAMPLE_INDICES = (1, 2, 3, 4)


def _preflight_risks(preflight: Any) -> list[float]:
    values: list[float] = []
    for tile in preflight.tile_trace:
        replay = tile.get("selected_anchor_v4_attribute_loo")
        if replay is None:
            continue
        if not isinstance(replay, dict) or replay.get("action") != "observed_only":
            raise RuntimeError("V4 replay calibration has an invalid preflight record")
        risk = replay.get("q75_risk")
        if isinstance(risk, bool) or not isinstance(risk, (int, float)) or risk < 0.0:
            raise RuntimeError("V4 replay calibration risk is invalid")
        values.append(float(risk))
    if not values:
        raise RuntimeError("V4 replay calibration sample has no V15-retained L1 tiles")
    return values


def collect_calibration_record(
    *,
    device: torch.device,
    sample_indices: tuple[int, ...],
    v15_calibration_record: Path,
) -> dict[str, Any]:
    if (
        len(sample_indices) < 2
        or len(set(sample_indices)) != len(sample_indices)
        or any(isinstance(index, bool) or not isinstance(index, int) or index <= 0 for index in sample_indices)
    ):
        raise ValueError("V4 replay calibration indices must be distinct and exclude sample-0")
    experiment = resolve_experiment("transplat", "dl3dv", ROOT)
    selection = resolve_claim_selection("transplat", "dl3dv", ROOT)
    checkpoint_sha256 = cached_sha256_file(experiment.checkpoint)
    # V15 is already frozen. Loading it against sample-0 validates that the
    # record does not contain the eventual V16 evaluation sample; V16's own
    # calibration records the intentional train-on-train V15 filter overlap.
    v15 = load_v15_threshold(
        v15_calibration_record,
        evaluation_sample_index=0,
        checkpoint_sha256=checkpoint_sha256,
        source_index_sha256=selection.source_index_sha256,
        sample_selection_sha256=selection.sample_selection_sha256,
    )
    sample_records: list[dict[str, Any]] = []
    for sample_index in sample_indices:
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
            sample_index=sample_index,
        )
        model.eval()
        context = {
            key: value.to(loaded_device) if torch.is_tensor(value) else value
            for key, value in batch["context"].items()
        }
        if context["image"].shape[0] != 1:
            raise RuntimeError("V4 replay calibration requires batch size one")
        _, _views, _, height, width = context["image"].shape
        with strict_fp32_convolution_execution():
            planning = _capture_s1_s2_without_dense_adapter(model, context)
            plan = build_incremental_probe_first_plan(
                planning["features"],
                planning["depths"],
                height=height,
                width=width,
                tile_size=TILE_SIZE,
                feature_threshold=FEATURE_THRESHOLD,
                depth_threshold=DEPTH_THRESHOLD,
                decision_semantics=DECISION_SEMANTICS,
                l1_anchor_semantics=ADAPTIVE_L1_15_ANCHOR_SEMANTICS,
            )
            capture = _capture_guarded_incremental_packed_adapter(
                model,
                context,
                plan=plan,
                compact_nonzero_materialization=True,
                compact_execution_policy=ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_DEV_POLICY,
                adaptive_l1_maximum_leave_one_out_residual=v15["threshold_value"],
                collect_selected_anchor_v4_attribute_loo_risk=True,
            )
        preflight = capture["compact_materialization_preflight"]
        if preflight is None or preflight.events.get("target_rgb_accessed") is not False or preflight.events.get(
            "skipped_s3_attributes_accessed"
        ) is not False:
            raise RuntimeError("V4 replay calibration crossed its selected-only boundary")
        sample_records.append(
            {
                "sample_index": sample_index,
                "scene": str(batch["scene"][0]),
                "risks": _preflight_risks(preflight),
                "access": {
                    "target_rgb_accessed": False,
                    "target_camera_accessed": False,
                    "skipped_s3_attributes_accessed": False,
                },
            }
        )
        _release_cuda_cache(loaded_device)
    return build_calibration_record(
        sample_records=sample_records,
        checkpoint_sha256=checkpoint_sha256,
        source_index_sha256=selection.source_index_sha256,
        sample_selection_sha256=selection.sample_selection_sha256,
        v15_calibration_sha256=v15["sha256"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v15-calibration-record", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=DEFAULT_SAMPLE_INDICES)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("V4 replay calibration requires CUDA")
    try:
        record = collect_calibration_record(
            device=device,
            sample_indices=tuple(args.sample_indices),
            v15_calibration_record=args.v15_calibration_record,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
