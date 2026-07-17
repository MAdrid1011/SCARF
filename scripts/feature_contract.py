#!/usr/bin/env python3
"""Record the actual S1 tensor supplied to each model's cost-volume path."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def summarize_probe_statistics(
    features: torch.Tensor, *, height: int, width: int, tile_size: int = 4
) -> dict[str, dict[str, float]]:
    """Summarize paper-relevant probe statistics without target RGB access."""
    from saes.progressive_saes import ProgressiveSAES

    statistics = (
        "current-channel-std",
        "raw-probe-vector-variance",
        "raw-probe-mean-channel-variance",
        "normalized-probe-total-variance",
        "normalized-probe-vector-standard-deviation",
    )
    report: dict[str, dict[str, float]] = {}
    for statistic in statistics:
        values, _ = ProgressiveSAES.classify_tiles_by_features(
            features,
            height,
            width,
            tile_size,
            per_view=True,
            statistic=statistic,
        )
        array = np.asarray(list(values.values()), dtype=np.float64)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{statistic} produced no finite tile values")
        report[statistic] = {
            "count": int(array.size),
            "min": float(array.min()),
            "p01": float(np.percentile(array, 1)),
            "p10": float(np.percentile(array, 10)),
            "p25": float(np.percentile(array, 25)),
            "p50": float(np.percentile(array, 50)),
            "p75": float(np.percentile(array, 75)),
            "p90": float(np.percentile(array, 90)),
            "p99": float(np.percentile(array, 99)),
            "max": float(array.max()),
            "fraction_below_tau_f_0_2": float((array < 0.2).mean()),
        }
    return report


def _context_on_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch["context"].items()
    }


def _classic_matching_features(
    model: Any, context: dict[str, Any], model_name: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    from feature_extractor import MVSplatFeatureExtractor, TransplatFeatureExtractor

    extractor_type = (
        TransplatFeatureExtractor if model_name == "transplat" else MVSplatFeatureExtractor
    )
    extractor = extractor_type.from_encoder(model.encoder)
    with torch.no_grad():
        if model_name == "transplat":
            output = extractor.forward(
                context["image"], context["extrinsics"], context["intrinsics"]
            )
        else:
            output = extractor.forward(context["image"])
    pipeline = output.trans_features
    captured: list[torch.Tensor] = []
    predictor = model.encoder.depth_predictor
    predictor_module = importlib.import_module(type(predictor).__module__)

    def capture(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError("cost-volume depth predictor received no feature tensor")
        captured.append(inputs[0].detach().clone())

    handle = predictor.register_forward_pre_hook(capture)
    restore_inner_hook = None
    inner_source: dict[str, Any]
    if model_name == "transplat":
        original_match_two = predictor.match_two
        matched_features: list[torch.Tensor] = []

        def capture_match_two(*args, **kwargs):
            feature = kwargs.get("features", args[-1])
            if not torch.is_tensor(feature):
                raise RuntimeError("TranSplat matcher received no feature tensor")
            matched_features.append(feature.detach().clone())
            return original_match_two(*args, **kwargs)

        predictor.match_two = capture_match_two

        def restore_inner_hook() -> dict[str, Any]:
            predictor.match_two = original_match_two
            if len(matched_features) != 1:
                raise RuntimeError(
                    "expected one TranSplat matching invocation, observed "
                    f"{len(matched_features)}"
                )
            return {
                "cost_volume_source": "DepthPredictorTrans.match_two(..., features)",
                "cost_volume_hook_observed_inputs": len(matched_features),
                "cost_volume_features": matched_features[0],
            }

    else:
        original_prepare = getattr(predictor_module, "prepare_feat_proj_data_lists", None)
        original_warp = getattr(predictor_module, "warp_with_pose_depth_candidates", None)
        if not callable(original_prepare) or not callable(original_warp):
            handle.remove()
            raise RuntimeError("MVSplat cost-volume helpers are unavailable")
        prepared_inputs: list[torch.Tensor] = []
        prepared_feature_lists: list[list[torch.Tensor]] = []
        warped_features: list[torch.Tensor] = []

        def capture_prepare(features, *args, **kwargs):
            result = original_prepare(features, *args, **kwargs)
            feature_lists = result[0]
            if not isinstance(feature_lists, list) or not feature_lists:
                raise RuntimeError("MVSplat matcher returned no feature lists")
            prepared_inputs.append(features.detach().clone())
            prepared_feature_lists.append(
                [feature.detach().clone() for feature in feature_lists]
            )
            return result

        def capture_warp(feature, *args, **kwargs):
            if not torch.is_tensor(feature):
                raise RuntimeError("MVSplat matcher received no warped feature tensor")
            warped_features.append(feature.detach().clone())
            return original_warp(feature, *args, **kwargs)

        setattr(predictor_module, "prepare_feat_proj_data_lists", capture_prepare)
        setattr(predictor_module, "warp_with_pose_depth_candidates", capture_warp)

        def restore_inner_hook() -> dict[str, Any]:
            setattr(predictor_module, "prepare_feat_proj_data_lists", original_prepare)
            setattr(predictor_module, "warp_with_pose_depth_candidates", original_warp)
            if len(prepared_inputs) != 1 or len(prepared_feature_lists) != 1:
                raise RuntimeError("MVSplat cost-volume preparation was not observed once")
            feature_lists = prepared_feature_lists[0]
            if len(warped_features) != len(feature_lists) - 1:
                raise RuntimeError(
                    "MVSplat warped-feature count differs from its matching feature lists"
                )
            for index, (expected, observed) in enumerate(
                zip(feature_lists[1:], warped_features)
            ):
                if not torch.equal(expected, observed):
                    max_abs = float((expected - observed).abs().max().item())
                    raise RuntimeError(
                        "MVSplat recorded matching feature differs from its warp input "
                        f"at pair {index}; max_abs={max_abs}"
                    )
            return {
                "cost_volume_source": (
                    "DepthPredictorMultiView.prepare_feat_proj_data_lists -> "
                    "warp_with_pose_depth_candidates"
                ),
                "cost_volume_hook_observed_inputs": len(warped_features),
                "cost_volume_features": prepared_inputs[0],
            }

    try:
        with torch.no_grad():
            model.encoder(context, False, deterministic=True)
    finally:
        handle.remove()
        if restore_inner_hook is not None:
            inner_source = restore_inner_hook()
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one depth-predictor feature input, observed {len(captured)}"
        )
    actual = captured[0]
    if not torch.equal(pipeline, actual):
        max_abs = float((pipeline - actual).abs().max().item())
        raise RuntimeError(
            f"pipeline feature differs from executed cost-volume input; max_abs={max_abs}"
        )
    cost_volume_features = inner_source.pop("cost_volume_features")
    if not torch.equal(actual, cost_volume_features):
        max_abs = float((actual - cost_volume_features).abs().max().item())
        raise RuntimeError(
            "depth-predictor feature differs from actual cost-volume feature; "
            f"max_abs={max_abs}"
        )
    return pipeline, {
        "source": (
            "encoder.backbone.trans_features -> depth_predictor arg0 -> "
            f"{inner_source['cost_volume_source']}"
        ),
        "hook_observed_inputs": len(captured),
        **inner_source,
        "bitwise_equal": True,
    }


def _depthsplat_matching_features(
    model: Any, context: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    from scripts.depthsplat_execution import extract_depthsplat_execution_tensors

    batch, views, _, height, width = context["image"].shape
    near = context["near"].to(context["image"]).clamp_min(1e-6)
    far = context["far"].to(context["image"]).clamp_min(1e-6)
    predictor = model.encoder.depth_predictor
    module = importlib.import_module(type(predictor).__module__)
    original_batch_features = getattr(module, "batch_features_camera_parameters", None)
    if not callable(original_batch_features):
        raise RuntimeError("DepthSplat cost-volume input helper is unavailable")
    observed_scales: list[torch.Tensor] = []

    def capture_batch_features(features_mv_curr, *args, **kwargs):
        if not isinstance(features_mv_curr, (list, tuple)) or not features_mv_curr:
            raise RuntimeError("DepthSplat cost-volume received no matching features")
        if not all(torch.is_tensor(feature) for feature in features_mv_curr):
            raise RuntimeError("DepthSplat cost-volume matching features are invalid")
        observed_scales.append(torch.stack(list(features_mv_curr), dim=1).detach().clone())
        return original_batch_features(features_mv_curr, *args, **kwargs)

    setattr(module, "batch_features_camera_parameters", capture_batch_features)
    try:
        with torch.no_grad():
            results = predictor(
                context["image"],
                attn_splits_list=[2],
                intrinsics=context["intrinsics"],
                min_depth=1.0 / far,
                max_depth=1.0 / near,
                extrinsics=context["extrinsics"],
            )
    finally:
        setattr(module, "batch_features_camera_parameters", original_batch_features)
    tensors = extract_depthsplat_execution_tensors(
        results,
        batch_size=batch,
        view_count=views,
        image_height=height,
        image_width=width,
    )
    expected_scales = tensors.matching_feature_scales
    if len(observed_scales) != len(expected_scales):
        raise RuntimeError(
            "DepthSplat cost-volume scale count differs from its recorded feature list"
        )
    scale_contracts = []
    for scale_index, (expected, observed, probability) in enumerate(
        zip(expected_scales, observed_scales, results["match_probs"])
    ):
        if not torch.equal(expected, observed):
            max_abs = float((expected - observed).abs().max().item())
            raise RuntimeError(
                "DepthSplat recorded feature differs from executed cost-volume input "
                f"at scale {scale_index}; max_abs={max_abs}"
            )
        scale_contracts.append(
            {
                "scale_index": scale_index,
                "source": (
                    f"depth_predictor.results.features_mv[{scale_index}] "
                    f"-> cost_volume scale {scale_index}"
                ),
                "shape": list(expected.shape),
                "probability_shape": list(probability.shape),
                "sha256": _tensor_sha256(expected),
                "bitwise_equal": True,
            }
        )
    return tensors.matching_features, {
        "source": "depth_predictor.results.features_mv[0] -> first cost_volume",
        "first_probability_shape": list(results["match_probs"][0].shape),
        "first_matching_feature_shape": list(results["features_mv"][0].shape),
        "bitwise_equal": True,
        "all_scales_bitwise_equal": True,
        "cost_volume_scales": scale_contracts,
    }


def collect_feature_contract(
    *,
    model_name: str,
    dataset_name: str,
    sample_index: int,
    device: torch.device,
) -> dict[str, Any]:
    """Load one context-only sample and verify its matching-feature contract."""
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data

    experiment = resolve_experiment(model_name, dataset_name, ROOT)
    selection = resolve_claim_selection(model_name, dataset_name, ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        model_name,
        dataset_name=dataset_name,
        checkpoint_path=experiment.checkpoint,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=sample_index + 1,
        sample_index=sample_index,
    )
    context = _context_on_device(batch, loaded_device)
    _, _, _, height, width = context["image"].shape
    if model_name in {"transplat", "mvsplat"}:
        features, source = _classic_matching_features(model, context, model_name)
    else:
        features, source = _depthsplat_matching_features(model, context)
    if features.dim() != 5 or features.shape[:2] != context["image"].shape[:2]:
        raise RuntimeError("matching feature does not have aligned [B,V,C,H,W] shape")
    target = batch.get("target", {})
    target_rgb_loaded = torch.is_tensor(target.get("image"))
    return {
        "schema_version": "1.0",
        "kind": "cost_volume_feature_contract",
        "model": model_name,
        "dataset": dataset_name,
        "sample_index": sample_index,
        "scene": str(batch["scene"][0]),
        "target_rgb_accessed": False,
        "native_dataloader_loaded_target_rgb": target_rgb_loaded,
        "context_indices": [int(value) for value in batch["context"]["index"][0].tolist()],
        "matching_feature": {
            **source,
            "shape": list(features.shape),
            "dtype": str(features.dtype).replace("torch.", ""),
            "sha256": _tensor_sha256(features),
        },
        "probe_statistics": summarize_probe_statistics(
            features, height=height, width=width
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("transplat", "mvsplat", "depthsplat"), required=True)
    parser.add_argument("--dataset", choices=("dl3dv",), default="dl3dv")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
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
    record = collect_feature_contract(
        model_name=args.model,
        dataset_name=args.dataset,
        sample_index=args.sample_index,
        device=device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "feature_contract.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
