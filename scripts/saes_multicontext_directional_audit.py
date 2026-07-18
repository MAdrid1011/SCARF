#!/usr/bin/env python3
"""Run the fixed target-free directional audit for tangent SAES covariance."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration import create_model_loader, load_context_only_audit_data
from saes.progressive_saes import apply_progressive_saes
from saes.projected_optical_moment_audit import (
    DirectionalTileComparison,
    compare_tile_directionally,
    evaluate_directional_gate,
    project_context_optical_moment,
)
from scripts.ae_config import resolve_experiment
from scripts.result_record import cached_sha256_file, portable_command, source_identity, write_result


AUDIT_ID = "multicontext-tangent-target-free-audit-v1"
MODEL = "transplat"
DATASET = "dl3dv"
SAMPLE_INDEX = 0
SEED = 0
TILE_SIZE = 4
FEATURE_THRESHOLD = 0.20
DEPTH_THRESHOLD = 0.10
DECISION_SEMANTICS = "probe-normalized-std-first-hit"
DEPTH_ROUTING_SEMANTICS = "metric-depth-standard-deviation"
CURRENT_MATERIALIZATION = "conditional-adapter-offset-attribute-transport-diagnostic"
TANGENT_MATERIALIZATION = "multicontext-tangent-plane-diagnostic"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"audit record has an unsupported value: {type(value)!r}")


def _capture_encoder_execution(
    model: Any, context: dict[str, Any]
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Capture S1/S2 while the native encoder emits its dense S3 tensor."""
    predictor = model.encoder.depth_predictor
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 5:
            raise RuntimeError("depth predictor did not receive [B,V,C,H,W] features")
        if not isinstance(output, tuple) or not output or not torch.is_tensor(output[0]):
            raise RuntimeError("depth predictor did not emit a depth tensor")
        captured["features"] = inputs[0].detach().clone()
        captured["depths"] = output[0].detach().clone()

    handle = predictor.register_forward_hook(capture)
    try:
        with torch.no_grad():
            gaussians = model.encoder(context, False, deterministic=True)
    finally:
        handle.remove()
    if set(captured) != {"features", "depths"}:
        raise RuntimeError("encoder execution did not expose one S1/S2 capture")
    return gaussians, captured["features"], captured["depths"]


def _tile_indices(
    *,
    view: int,
    tile_row: int,
    tile_column: int,
    height: int,
    width: int,
    gpp: int,
) -> torch.Tensor:
    values = [
        ((view * height * width + (tile_row * TILE_SIZE + row) * width + tile_column * TILE_SIZE + column) * gpp + slot)
        for row in range(TILE_SIZE)
        for column in range(TILE_SIZE)
        for slot in range(gpp)
    ]
    return torch.tensor(values, dtype=torch.long)


def _full_indices_from_trace(
    mask: torch.Tensor,
    trace: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    gpp: int,
) -> torch.Tensor:
    indices: list[int] = []
    cpu_mask = mask.detach().cpu()
    for entry in trace:
        tile = _tile_indices(
            view=int(entry["view_index"]),
            tile_row=int(entry["tile_row"]),
            tile_column=int(entry["tile_column"]),
            height=height,
            width=width,
            gpp=gpp,
        )
        if not bool(cpu_mask[tile].any()):
            indices.extend(tile.tolist())
    return torch.tensor(indices, dtype=torch.long)


def _build_commit_payload(
    *,
    gaussians: Any,
    mask: torch.Tensor,
    stats: dict[str, Any],
    tile_trace: list[dict[str, Any]],
    materialization: str,
    height: int,
    width: int,
    view_count: int,
) -> dict[str, Any]:
    """Serialize only committed descriptors, never deleted dense S3 slots."""
    if gaussians.means.shape[0] != 1:
        raise ValueError("directional audit supports exactly one batch item")
    gpp, remainder = divmod(
        int(gaussians.means.shape[1]), view_count * height * width
    )
    if remainder or gpp < 1 or mask.numel() != gaussians.means.shape[1]:
        raise ValueError("candidate layout does not match the fixed context grid")
    retained = torch.nonzero(~mask, as_tuple=False).flatten()
    if retained.numel() == 0:
        raise RuntimeError("candidate has no committed descriptors")
    full_indices = _full_indices_from_trace(
        mask,
        tile_trace,
        height=height,
        width=width,
        gpp=gpp,
    )
    payload = {
        "schema_version": "1.0",
        "audit_id": AUDIT_ID,
        "materialization": materialization,
        "height": height,
        "width": width,
        "view_count": view_count,
        "primitives_per_pixel": gpp,
        "modified_mask": mask.detach().cpu().to(dtype=torch.bool).clone(),
        "retained_indices": retained.detach().cpu().to(dtype=torch.long).clone(),
        "full_indices": full_indices,
        "means": gaussians.means[0, retained].detach().cpu().clone(),
        "covariances": gaussians.covariances[0, retained].detach().cpu().clone(),
        "harmonics": gaussians.harmonics[0, retained].detach().cpu().clone(),
        "opacities": gaussians.opacities[0, retained].detach().cpu().clone(),
        "saes_stats": _json_value(stats),
        "tile_trace": _json_value(tile_trace),
    }
    return payload


def _write_descriptor_commit(
    directory: Path, name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    path = directory / f"{name}.pt"
    torch.save(payload, path)
    mask = payload["modified_mask"]
    return {
        "name": name,
        "path": path.name,
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
        "modified_mask_sha256": hashlib.sha256(
            mask.numpy().tobytes()
        ).hexdigest(),
        "retained_descriptor_count": int(payload["retained_indices"].numel()),
        "full_descriptor_count": int(payload["full_indices"].numel()),
        "contains_deleted_dense_s3": False,
    }


def _load_descriptor_commit(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = directory / str(manifest["path"])
    if not path.is_file() or _sha256_file(path) != manifest["sha256"]:
        raise RuntimeError("phase-A descriptor commit hash mismatch")
    payload = torch.load(path, map_location="cpu")
    expected = {
        "schema_version",
        "audit_id",
        "materialization",
        "height",
        "width",
        "view_count",
        "primitives_per_pixel",
        "modified_mask",
        "retained_indices",
        "full_indices",
        "means",
        "covariances",
        "harmonics",
        "opacities",
        "saes_stats",
        "tile_trace",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RuntimeError("phase-A descriptor commit has an invalid schema")
    if payload["audit_id"] != AUDIT_ID:
        raise RuntimeError("phase-A descriptor commit has the wrong audit id")
    return payload


def _lookup_committed(payload: dict[str, Any], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    retained = payload["retained_indices"]
    indices = indices.detach().cpu().to(dtype=torch.long)
    locations = torch.searchsorted(retained, indices)
    if bool((locations >= retained.numel()).any()) or not torch.equal(
        retained[locations], indices
    ):
        raise RuntimeError("committed sparse tile has a missing retained descriptor")
    return {
        name: payload[name][locations]
        for name in ("means", "covariances", "harmonics", "opacities")
    }


def _shared_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stats.items()
        if not key.startswith("multicontext_tangent_")
    }


def _route_signature(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    mask = payload["modified_mask"]
    signature = []
    for entry in payload["tile_trace"]:
        tile = _tile_indices(
            view=int(entry["view_index"]),
            tile_row=int(entry["tile_row"]),
            tile_column=int(entry["tile_column"]),
            height=int(payload["height"]),
            width=int(payload["width"]),
            gpp=int(payload["primitives_per_pixel"]),
        )
        signature.append(
            (
                int(entry["view_index"]),
                int(entry["tile_row"]),
                int(entry["tile_column"]),
                str(entry["routing_level_before_materialization"]),
                bool(mask[tile].any()),
                tuple(
                    (str(check["level"]), bool(check["passed"]))
                    for check in entry["guard_checks"]
                ),
            )
        )
    return signature


def _assert_psd(covariances: torch.Tensor, label: str) -> None:
    symmetric = (covariances + covariances.mT) * 0.5
    tolerance = 64.0 * torch.finfo(covariances.dtype).eps
    if not torch.allclose(covariances, symmetric, rtol=0.0, atol=tolerance):
        raise RuntimeError(f"{label} committed non-symmetric covariance")
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    if not bool(torch.isfinite(eigenvalues).all()) or bool((eigenvalues < -1e-6).any()):
        raise RuntimeError(f"{label} committed non-PSD covariance")


def _maximum_delta(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        raise RuntimeError("descriptor shapes differ")
    return float((first - second).abs().max().item()) if first.numel() else 0.0


def _phase_a_invariants(
    current: dict[str, Any], tangent: dict[str, Any]
) -> dict[str, Any]:
    """Enforce the frozen route/output contract before dense-reference reads."""
    for key in ("height", "width", "view_count", "primitives_per_pixel"):
        if current[key] != tangent[key]:
            raise RuntimeError(f"current and tangent commits differ in {key}")
    if not torch.equal(current["modified_mask"], tangent["modified_mask"]):
        raise RuntimeError("tangent candidate changed the committed route mask")
    if not torch.equal(current["retained_indices"], tangent["retained_indices"]):
        raise RuntimeError("tangent candidate changed the retained output slots")
    if _route_signature(current) != _route_signature(tangent):
        raise RuntimeError("tangent candidate changed route or guard events")
    if _shared_stats(current["saes_stats"]) != _shared_stats(tangent["saes_stats"]):
        raise RuntimeError("tangent candidate changed non-diagnostic SAES events")
    if current["saes_stats"].get("guard_nonprobe_s3_attribute_reads") != 0:
        raise RuntimeError("current merge read skipped S3 before output commit")
    if tangent["saes_stats"].get("guard_nonprobe_s3_attribute_reads") != 0:
        raise RuntimeError("tangent candidate read skipped S3 before output commit")
    for name in ("means", "harmonics", "opacities"):
        if not torch.equal(current[name], tangent[name]):
            raise RuntimeError(f"tangent candidate changed committed {name}")
    if not torch.equal(current["full_indices"], tangent["full_indices"]):
        raise RuntimeError("tangent candidate changed Full slot identity")
    current_full = _lookup_committed(current, current["full_indices"])
    tangent_full = _lookup_committed(tangent, tangent["full_indices"])
    for name in ("means", "covariances", "harmonics", "opacities"):
        if not torch.equal(current_full[name], tangent_full[name]):
            raise RuntimeError(f"tangent candidate changed Full {name}")
    _assert_psd(current["covariances"], "current")
    _assert_psd(tangent["covariances"], "tangent")
    if tangent["saes_stats"].get("multicontext_tangent_runtime_eligible") is not False:
        raise RuntimeError("tangent diagnostic was marked runtime eligible")
    return {
        "route_mask_identical": True,
        "retained_slots_identical": True,
        "route_and_guard_events_identical": True,
        "non_diagnostic_event_counts_identical": True,
        "means_identical": True,
        "harmonics_identical": True,
        "opacities_identical": True,
        "full_slots_identical": True,
        "current_psd_finite": True,
        "tangent_psd_finite": True,
    }


def _poison_skipped_descriptors(gaussians: Any, mask: torch.Tensor) -> None:
    skipped = torch.nonzero(mask, as_tuple=False).flatten()
    for name, value in (
        ("means", 1.0e4),
        ("covariances", -1.0e4),
        ("harmonics", 1.0e4),
        ("opacities", 0.99),
    ):
        getattr(gaussians, name)[0, skipped] = value


def _poison_skipped_depths(
    depths: torch.Tensor, mask: torch.Tensor, *, height: int, width: int, gpp: int
) -> None:
    positions = torch.unique(torch.nonzero(mask, as_tuple=False).flatten() // gpp)
    per_view = height * width
    for position in positions.tolist():
        view, pixel = divmod(position, per_view)
        if depths.ndim == 5:
            depths[0, view, pixel] = 1.0e4
        elif depths.ndim == 4:
            row, column = divmod(pixel, width)
            depths[0, view, row, column] = 1.0e4
        else:
            raise RuntimeError("audit cannot poison an unsupported S2 depth layout")


def _poison_invariants(
    tangent: dict[str, Any], poisoned: dict[str, Any]
) -> dict[str, Any]:
    if not torch.equal(tangent["modified_mask"], poisoned["modified_mask"]):
        raise RuntimeError("skipped poison changed the route mask")
    if not torch.equal(tangent["retained_indices"], poisoned["retained_indices"]):
        raise RuntimeError("skipped poison changed retained output slots")
    if _shared_stats(tangent["saes_stats"]) != _shared_stats(poisoned["saes_stats"]):
        raise RuntimeError("skipped poison changed non-diagnostic event counts")
    deltas = {}
    for name in ("means", "harmonics", "opacities"):
        delta = _maximum_delta(tangent[name], poisoned[name])
        if delta != 0.0:
            raise RuntimeError(f"skipped poison changed retained {name}")
        deltas[name] = delta
    covariance_delta = _maximum_delta(tangent["covariances"], poisoned["covariances"])
    if covariance_delta > 1.0e-5:
        raise RuntimeError("skipped poison changed retained covariance")
    tangent_full = _lookup_committed(tangent, tangent["full_indices"])
    poisoned_full = _lookup_committed(poisoned, poisoned["full_indices"])
    for name in ("means", "covariances", "harmonics", "opacities"):
        if not torch.equal(tangent_full[name], poisoned_full[name]):
            raise RuntimeError(f"skipped poison changed Full {name}")
    return {
        "route_mask_identical": True,
        "retained_slots_identical": True,
        "non_diagnostic_event_counts_identical": True,
        "full_slots_identical": True,
        "retained_maximum_absolute_delta": {
            **deltas,
            "covariances": covariance_delta,
        },
        "skipped_s2_poisoned": True,
        "skipped_s3_poisoned": True,
    }


def _materialize_candidate(
    *,
    model: Any,
    context: dict[str, Any],
    materialization: str,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    gaussians, features, depths = _capture_encoder_execution(model, context)
    _, views, _, height, width = context["image"].shape
    trace: list[dict[str, Any]] = []
    mask, stats, _ = apply_progressive_saes(
        gaussians,
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
        context_extrinsics=context["extrinsics"],
        context_intrinsics=context["intrinsics"],
        depth_routing_semantics=DEPTH_ROUTING_SEMANTICS,
        depth_near=context["near"],
        depth_far=context["far"],
        tile_trace=trace,
    )
    return gaussians, features, depths, mask, stats, trace


def _load_encoder_context(input_root: Path, device: torch.device) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment = resolve_experiment(MODEL, DATASET, ROOT)
    loader = create_model_loader(MODEL)
    bundle = loader.load_model(
        str(experiment.checkpoint),
        device=device,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        encoder_only=True,
    )
    data = load_context_only_audit_data(loader, bundle, input_root=input_root)
    if "target" in data.batch:
        raise RuntimeError("context-only audit loader returned a target mapping")
    model = bundle.model
    model.eval()
    context = {
        key: value.to(bundle.device) if torch.is_tensor(value) else value
        for key, value in data.batch["context"].items()
    }
    return model, context, data.batch["calibration"], {
        "checkpoint_path": str(experiment.checkpoint),
        "environment_profile": experiment.environment_profile,
        "native_encoder_device": str(bundle.device),
    }


def _serialize_moment(moment: Any) -> dict[str, Any]:
    return {
        "valid": moment.valid,
        "reason": moment.reason,
        "mass": moment.mass,
        "determinant": moment.determinant,
        "log_determinant": moment.log_determinant,
        "footprint_log_area": moment.footprint_log_area,
        "psd": moment.psd,
        "condition_number": moment.condition_number,
        "descriptor_count": moment.descriptor_count,
        "offscreen_count": moment.offscreen_count,
    }


def _serialize_comparison(comparison: DirectionalTileComparison) -> dict[str, Any]:
    return {
        "tile": comparison.tile_key,
        "level": comparison.level,
        "context_local_index": comparison.context_index,
        "valid": comparison.valid,
        "reason": comparison.reason,
        "dense": _serialize_moment(comparison.dense),
        "current": asdict(comparison.current) if comparison.current is not None else None,
        "candidate": asdict(comparison.candidate) if comparison.candidate is not None else None,
    }


def _condition_summaries(comparisons: Iterable[DirectionalTileComparison]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[DirectionalTileComparison]] = {}
    for comparison in comparisons:
        grouped.setdefault((comparison.level, comparison.context_index), []).append(comparison)
    result: dict[str, Any] = {}
    for (level, context), records in sorted(grouped.items()):
        values = {"dense": [], "current": [], "candidate": []}
        for record in records:
            if not record.valid:
                continue
            assert record.dense.condition_number is not None
            assert record.current is not None and record.candidate is not None
            values["dense"].append(record.dense.condition_number)
            values["current"].append(record.current.condition_number)
            values["candidate"].append(record.candidate.condition_number)
        if not values["dense"]:
            result[f"{level}:context-{context}"] = {"valid_count": 0}
            continue
        result[f"{level}:context-{context}"] = {
            "valid_count": len(values["dense"]),
            **{
                name: {
                    "p50": float(torch.quantile(torch.tensor(series), 0.50).item()),
                    "p95": float(torch.quantile(torch.tensor(series), 0.95).item()),
                    "maximum": float(max(series)),
                }
                for name, series in values.items()
            },
        }
    return result


def _posthoc_dense_comparisons(
    *,
    dense_gaussians: Any,
    current: dict[str, Any],
    tangent: dict[str, Any],
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    context_indices: list[int],
) -> tuple[DirectionalTileComparison, ...]:
    """Read dense S3 only after phase-A commit validation has completed."""
    mask = current["modified_mask"]
    height = int(current["height"])
    width = int(current["width"])
    gpp = int(current["primitives_per_pixel"])
    dense_means = dense_gaussians.means[0].detach().cpu()
    dense_covariances = dense_gaussians.covariances[0].detach().cpu()
    dense_opacities = dense_gaussians.opacities[0].detach().cpu()
    if dense_means.shape[0] != mask.numel():
        raise RuntimeError("posthoc dense reference has the wrong descriptor count")
    comparisons: list[DirectionalTileComparison] = []
    for entry in current["tile_trace"]:
        tile = _tile_indices(
            view=int(entry["view_index"]),
            tile_row=int(entry["tile_row"]),
            tile_column=int(entry["tile_column"]),
            height=height,
            width=width,
            gpp=gpp,
        )
        if not bool(mask[tile].any()):
            continue
        level = str(entry["routing_level_before_materialization"])
        if level not in {"L0", "L1"}:
            raise RuntimeError("modified tile has no L0/L1 route label")
        retained = tile[~mask[tile]]
        current_tile = _lookup_committed(current, retained)
        tangent_tile = _lookup_committed(tangent, retained)
        for context_local_index in range(len(context_indices)):
            extrinsic = context_extrinsics[context_local_index]
            intrinsic = context_intrinsics[context_local_index]
            dense_moment = project_context_optical_moment(
                means=dense_means[tile],
                covariances=dense_covariances[tile],
                opacities=dense_opacities[tile],
                context_extrinsic=extrinsic,
                context_intrinsic=intrinsic,
            )
            current_moment = project_context_optical_moment(
                means=current_tile["means"],
                covariances=current_tile["covariances"],
                opacities=current_tile["opacities"],
                context_extrinsic=extrinsic,
                context_intrinsic=intrinsic,
            )
            tangent_moment = project_context_optical_moment(
                means=tangent_tile["means"],
                covariances=tangent_tile["covariances"],
                opacities=tangent_tile["opacities"],
                context_extrinsic=extrinsic,
                context_intrinsic=intrinsic,
            )
            comparisons.append(
                compare_tile_directionally(
                    tile_key=(
                        int(entry["view_index"]),
                        int(entry["tile_row"]),
                        int(entry["tile_column"]),
                    ),
                    level=level,
                    context_index=context_local_index,
                    dense=dense_moment,
                    current=current_moment,
                    candidate=tangent_moment,
                )
            )
    return tuple(comparisons)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_directional_audit(
    *, input_root: Path, output_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Run the single pre-registered descriptor audit without rendering."""
    model, context, input_identity, execution = _load_encoder_context(input_root, device)
    _, views, _, height, width = context["image"].shape
    if views != 2 or "target" in context:
        raise RuntimeError("audit context violates its two-view target-free contract")
    phase_dir = output_dir / "phase_a"
    phase_dir.mkdir(parents=True, exist_ok=False)

    current_raw, _current_features, _current_depths, current_mask, current_stats, current_trace = _materialize_candidate(
        model=model, context=context, materialization=CURRENT_MATERIALIZATION
    )
    current_payload = _build_commit_payload(
        gaussians=current_raw,
        mask=current_mask,
        stats=current_stats,
        tile_trace=current_trace,
        materialization=CURRENT_MATERIALIZATION,
        height=height,
        width=width,
        view_count=views,
    )
    current_manifest = _write_descriptor_commit(phase_dir, "current", current_payload)
    del current_raw, _current_features, _current_depths

    tangent_raw, _tangent_features, _tangent_depths, tangent_mask, tangent_stats, tangent_trace = _materialize_candidate(
        model=model, context=context, materialization=TANGENT_MATERIALIZATION
    )
    tangent_payload = _build_commit_payload(
        gaussians=tangent_raw,
        mask=tangent_mask,
        stats=tangent_stats,
        tile_trace=tangent_trace,
        materialization=TANGENT_MATERIALIZATION,
        height=height,
        width=width,
        view_count=views,
    )
    tangent_manifest = _write_descriptor_commit(phase_dir, "tangent", tangent_payload)
    del tangent_raw, _tangent_features, _tangent_depths

    phase_manifest = {
        "schema_version": "1.0",
        "audit_id": AUDIT_ID,
        "status": "COMMITTED",
        "dense_skipped_s3_reference_read": False,
        "current": current_manifest,
        "tangent": tangent_manifest,
    }
    phase_manifest_path = phase_dir / "commits.json"
    _write_json(phase_manifest_path, phase_manifest)

    # Reloading hash-bound sparse commits is the hard boundary before phase B.
    phase_manifest = json.loads(phase_manifest_path.read_text(encoding="utf-8"))
    if phase_manifest.get("status") != "COMMITTED" or phase_manifest.get(
        "dense_skipped_s3_reference_read"
    ) is not False:
        raise RuntimeError("phase-A commit manifest is invalid")
    current = _load_descriptor_commit(phase_dir, phase_manifest["current"])
    tangent = _load_descriptor_commit(phase_dir, phase_manifest["tangent"])
    structural = _phase_a_invariants(current, tangent)

    poison_raw, poison_features, poison_depths = _capture_encoder_execution(model, context)
    _poison_skipped_descriptors(poison_raw, current_mask)
    _poison_skipped_depths(
        poison_depths,
        current_mask,
        height=height,
        width=width,
        gpp=int(current["primitives_per_pixel"]),
    )
    poison_trace: list[dict[str, Any]] = []
    poison_mask, poison_stats, _ = apply_progressive_saes(
        poison_raw,
        height,
        width,
        tile_size=TILE_SIZE,
        feature_var_threshold=FEATURE_THRESHOLD,
        depth_std_threshold=DEPTH_THRESHOLD,
        features=poison_features,
        depths=poison_depths,
        view_count=views,
        materialization=TANGENT_MATERIALIZATION,
        decision_semantics=DECISION_SEMANTICS,
        context_extrinsics=context["extrinsics"],
        context_intrinsics=context["intrinsics"],
        depth_routing_semantics=DEPTH_ROUTING_SEMANTICS,
        depth_near=context["near"],
        depth_far=context["far"],
        tile_trace=poison_trace,
    )
    poisoned = _build_commit_payload(
        gaussians=poison_raw,
        mask=poison_mask,
        stats=poison_stats,
        tile_trace=poison_trace,
        materialization=TANGENT_MATERIALIZATION,
        height=height,
        width=width,
        view_count=views,
    )
    poison = _poison_invariants(tangent, poisoned)
    del poison_raw, poison_features, poison_depths, poisoned
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase B begins only here: the following capture is the first dense-S3
    # reference read and is never passed back into routing or materialization.
    dense_reference, _dense_features, _dense_depths = _capture_encoder_execution(model, context)
    context_extrinsics = context["extrinsics"][0].detach().cpu()
    context_intrinsics = context["intrinsics"][0].detach().cpu()
    comparisons = _posthoc_dense_comparisons(
        dense_gaussians=dense_reference,
        current=current,
        tangent=tangent,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        context_indices=[int(value) for value in context["index"][0].detach().cpu().tolist()],
    )
    del dense_reference, _dense_features, _dense_depths, model, context
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gate = evaluate_directional_gate(comparisons)
    fallback_count = int(tangent["saes_stats"].get("multicontext_tangent_local_fallbacks", 0))
    accepted_count = int(tangent["saes_stats"].get("multicontext_tangent_accepted", 0))
    quality_gate_authorized = bool(gate.passed and accepted_count > 0 and fallback_count == 0)
    result = {
        "schema_version": "1.0",
        "kind": "saes_multicontext_tangent_target_free_directional_audit",
        "audit_id": AUDIT_ID,
        "status": "COMPLETED",
        "paper_result_eligible": False,
        "quality_metrics_computed": False,
        "renderer_executed": False,
        "decoder_executed": False,
        "teacher_oracle_accessed": False,
        "expected_results_accessed": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_mapping_present_before_encoder": False,
        "dense_skipped_s3_reference_read_after_phase_a_commit": True,
        "model": MODEL,
        "dataset": DATASET,
        "sample_index": SAMPLE_INDEX,
        "scene": input_identity["scene"],
        "context_indices": input_identity["context_indices"],
        "input_provenance": input_identity,
        "fixed_contract": {
            "seed": SEED,
            "tile_size": TILE_SIZE,
            "feature_threshold": FEATURE_THRESHOLD,
            "depth_threshold": DEPTH_THRESHOLD,
            "decision_semantics": DECISION_SEMANTICS,
            "depth_routing_semantics": DEPTH_ROUTING_SEMANTICS,
            "current_materialization": CURRENT_MATERIALIZATION,
            "candidate_materialization": TANGENT_MATERIALIZATION,
            "quality_retry_run": False,
        },
        "execution": execution,
        "phase_a": {
            "commit_manifest": {
                "path": str(phase_manifest_path.relative_to(output_dir)),
                "sha256": _sha256_file(phase_manifest_path),
            },
            "current": current_manifest,
            "tangent": tangent_manifest,
            "structural_invariants": structural,
            "poison_invariants": poison,
            "current_stats": current["saes_stats"],
            "tangent_stats": tangent["saes_stats"],
        },
        "posthoc_dense_reference": {
            "reference_use": "read-only-after-phase-a-commit",
            "projection": "unclipped-positive-depth-optical-moment-surrogate",
            "not_alpha_compositing": True,
            "not_renderer_fidelity": True,
            "comparison_count": len(comparisons),
            "comparisons": [_serialize_comparison(item) for item in comparisons],
            "condition_number_summaries": _condition_summaries(comparisons),
            "fit_residual_definition": "dense projected covariance relative Frobenius error",
        },
        "directional_gate": {
            **asdict(gate),
            "cells": [
                {
                    "level": cell.level,
                    "context_local_index": cell.context_index,
                    "record_count": cell.record_count,
                    "valid_count": cell.valid_count,
                    "invalid_count": cell.invalid_count,
                    "current": {
                        name: asdict(summary) for name, summary in cell.current.items()
                    },
                    "candidate": {
                        name: asdict(summary) for name, summary in cell.candidate.items()
                    },
                    "candidate_minus_current": {
                        name: {
                            key: getattr(cell.candidate[name], key) - getattr(cell.current[name], key)
                            for key in ("mean", "p50", "p95", "maximum")
                        }
                        for name in cell.current
                    },
                    "percentile_nonworse": cell.percentile_nonworse,
                    "percentile_failures": list(cell.percentile_failures),
                }
                for cell in gate.cells
            ],
        },
        "tangent_fit": {
            "attempts": int(tangent["saes_stats"].get("multicontext_tangent_attempts", 0)),
            "accepted": accepted_count,
            "local_fallbacks": fallback_count,
            "fallback_reasons": tangent["saes_stats"].get("multicontext_tangent_fallback_reasons", {}),
            "residual_max": tangent["saes_stats"].get("multicontext_tangent_residual_max"),
            "all_attempts_fitted": fallback_count == 0 and accepted_count > 0,
        },
        "quality_gate_authorized": quality_gate_authorized,
        "next_action": (
            "pre-register-one-fixed-sample0-quality-gate"
            if quality_gate_authorized
            else "stop-tangent-candidate-and-record-directional-failure"
            if not gate.passed
            else "repair-local-identifiability-before-any-quality-gate"
        ),
        "checkpoint_sha256": cached_sha256_file(Path(execution["checkpoint_path"])),
        "source": source_identity(),
    }
    return result


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
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        record = collect_directional_audit(
            input_root=args.input_root, output_dir=args.output_dir, device=device
        )
    except Exception as exc:
        record = {
            "schema_version": "1.0",
            "kind": "saes_multicontext_tangent_target_free_directional_audit",
            "audit_id": AUDIT_ID,
            "status": "FAILED",
            "paper_result_eligible": False,
            "quality_metrics_computed": False,
            "renderer_executed": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_result(record, args.output_dir / "results.json")
        print(args.output_dir / "results.json", file=sys.stderr)
        return 2
    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
