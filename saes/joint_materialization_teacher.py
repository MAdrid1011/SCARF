"""Author-side dense-adaptor teacher packets for joint SAES calibration.

This module is deliberately an offline cache compiler primitive. It derives
selected-only ordinary representative packets from the runtime-legal inputs,
then derives aligned targets from a separate dense-adaptor pass. It has no
renderer, target RGB, target camera, model loader, dataset loader, or runtime
asset loader.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch

from saes.progressive_saes import apply_progressive_saes


PACKET_KIND = "acid_joint_dense_adaptor_teacher_packets_v1"
PACKET_SCHEMA_VERSION = "1.0"
_ATTRIBUTES = ("means", "covariances", "harmonics", "opacities")
_ROUTE_STAT_KEYS = (
    "level0_tiles",
    "level1_tiles",
    "full_tiles",
    "level0_pixels",
    "level1_pixels",
    "l0_representatives",
    "l1_lightweight_anchors",
    "full_stage3_gaussians",
)
_CAPTURE_ONLY_STAT_KEYS = frozenset({"offline_joint_calibration_capture_calls"})


def _clone_gaussians(gaussians: Any) -> Any:
    if any(not torch.is_tensor(getattr(gaussians, name, None)) for name in _ATTRIBUTES):
        raise TypeError("dense adaptor output has no complete Gaussian attributes")
    return type(gaussians)(
        **{name: getattr(gaussians, name).detach().clone() for name in _ATTRIBUTES}
    )


def _sha256_tensor(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("teacher packet SHA256 requires a tensor")
    return hashlib.sha256(
        value.detach().to(device="cpu").contiguous().numpy().tobytes()
    ).hexdigest()


def _route_summary(stats: Mapping[str, Any]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for key in _ROUTE_STAT_KEYS:
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"SAES teacher path has an invalid {key}")
        summary[key] = value
    return summary


def _sample_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if set(first) != set(second):
        return False
    for key in first:
        if torch.is_tensor(first[key]):
            if not torch.is_tensor(second[key]) or not torch.equal(first[key], second[key]):
                return False
        elif first[key] != second[key]:
            return False
    return True


def _capture_selected_only_packets(
    source: Any,
    *,
    height: int,
    width: int,
    saes_kwargs: Mapping[str, Any],
) -> tuple[Any, torch.Tensor, Mapping[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    packets: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    selected = _clone_gaussians(source)
    mask, stats, _ = apply_progressive_saes(
        selected,
        height,
        width,
        **dict(saes_kwargs),
        offline_joint_calibration_capture=packets.append,
        tile_trace=trace,
    )
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 1:
        raise RuntimeError("selected-only SAES path emitted an invalid modified mask")
    expected = int(stats["l0_representatives"] + stats["l1_lightweight_anchors"])
    if stats.get("offline_joint_calibration_capture_calls") != expected or len(packets) != expected:
        raise RuntimeError("selected-only SAES capture count differs from retained anchors")
    if any(not isinstance(packet, dict) for packet in packets):
        raise RuntimeError("selected-only SAES capture emitted a malformed packet")
    packets.sort(key=lambda packet: int(packet["anchor_index"]))
    anchors = [int(packet["anchor_index"]) for packet in packets]
    if len(anchors) != len(set(anchors)) or any(index < 0 or index >= mask.numel() for index in anchors):
        raise RuntimeError("selected-only SAES capture has invalid anchor indices")
    if any(bool(mask[index]) for index in anchors):
        raise RuntimeError("selected-only SAES capture included a skipped anchor")
    return selected, mask, stats, packets, trace


def _replay_ordinary_representatives(
    source: Any,
    *,
    height: int,
    width: int,
    saes_kwargs: Mapping[str, Any],
) -> tuple[Any, torch.Tensor, Mapping[str, Any], list[dict[str, Any]]]:
    """Replay native representative materialization without the cache hook."""

    ordinary = _clone_gaussians(source)
    trace: list[dict[str, Any]] = []
    mask, stats, _ = apply_progressive_saes(
        ordinary,
        height,
        width,
        **dict(saes_kwargs),
        tile_trace=trace,
    )
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 1:
        raise RuntimeError("ordinary SAES replay emitted an invalid modified mask")
    return ordinary, mask, stats, trace


def _operational_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Discard only the callback counter before comparing replay execution."""

    if not isinstance(stats, Mapping):
        raise RuntimeError("SAES replay emitted invalid statistics")
    return {
        key: value for key, value in stats.items() if key not in _CAPTURE_ONLY_STAT_KEYS
    }


def _same_gaussian_attributes(first: Any, second: Any) -> bool:
    return all(
        torch.equal(getattr(first, name), getattr(second, name))
        for name in _ATTRIBUTES
    )


def _assignment_aligned_dense_target(
    source: Any, sample: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Aggregate real dense-adaptor slots with the frozen sparse assignments.

    The selected-only capture contains just a retained anchor plus the indices
    and weights that its ordinary representative used.  This offline-only
    helper is the first point that opens the real non-probe dense-adaptor
    attributes.  It does not alter routing, does not invoke a renderer, and
    never returns these dense attributes to the runtime cache.
    """

    required = {
        "descriptor",
        "means",
        "covariances",
        "harmonics",
        "opacities",
        "level",
        "anchor_index",
        "teacher_nonprobe_indices",
        "teacher_assignment_weights",
    }
    if not isinstance(sample, Mapping) or set(sample) != required:
        raise RuntimeError("selected-only capture has an invalid teacher packet schema")
    anchor_index = sample["anchor_index"]
    nonprobe_indices = sample["teacher_nonprobe_indices"]
    assignment = sample["teacher_assignment_weights"]
    count = int(source.means.shape[1])
    if (
        not isinstance(anchor_index, int)
        or anchor_index < 0
        or anchor_index >= count
        or not isinstance(nonprobe_indices, tuple)
        or not nonprobe_indices
        or any(not isinstance(index, int) or index < 0 or index >= count for index in nonprobe_indices)
        or anchor_index in nonprobe_indices
        or len(set(nonprobe_indices)) != len(nonprobe_indices)
        or not torch.is_tensor(assignment)
        or assignment.shape != (len(nonprobe_indices),)
        or not bool(torch.isfinite(assignment).all())
        or bool((assignment < 0.0).any())
    ):
        raise RuntimeError("selected-only capture has invalid offline assignment metadata")

    indices = torch.tensor(
        (anchor_index, *nonprobe_indices), device=source.means.device, dtype=torch.long
    )
    weights = torch.cat(
        (
            torch.ones(1, device=source.means.device, dtype=source.means.dtype),
            assignment.to(device=source.means.device, dtype=source.means.dtype),
        )
    )
    normalizer = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    weights = weights / normalizer
    means = source.means[0, indices]
    covariances = source.covariances[0, indices]
    harmonics = source.harmonics[0, indices]
    opacities = source.opacities[0, indices]
    if (
        means.shape != (len(indices), 3)
        or covariances.shape != (len(indices), 3, 3)
        or harmonics.ndim != 3
        or harmonics.shape[:2] != (len(indices), 3)
        or opacities.shape[0] != len(indices)
    ):
        raise RuntimeError("dense-adaptor teacher has inconsistent selected attributes")

    target_mean = torch.einsum("n,nc->c", weights, means)
    deltas = means - target_mean.unsqueeze(0)
    target_covariance = torch.einsum(
        "n,nij->ij",
        weights,
        (covariances + deltas.unsqueeze(2) * deltas.unsqueeze(1)),
    )
    target_covariance = (target_covariance + target_covariance.mT) * 0.5
    target_harmonics = torch.einsum("n,ncd->cd", weights, harmonics)
    target_opacity = torch.einsum("n,n...->...", weights, opacities).clamp(
        0.0, 1.0 - 1e-6
    )
    result = {
        "means": target_mean,
        "covariances": target_covariance,
        "harmonics": target_harmonics,
        "opacities": target_opacity,
    }
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("assignment-aligned dense teacher is non-finite")
    _assert_psd(result["covariances"].reshape(1, 3, 3), label="assignment-aligned dense teacher")
    return result


def _require_joint_identity_signal(packets: list[Mapping[str, Any]]) -> None:
    """Reject a degenerate teacher that is identical to sparse packets."""

    squared_error = {name: 0.0 for name in _ATTRIBUTES}
    for packet in packets:
        for name in _ATTRIBUTES:
            squared_error[name] += float(
                (packet["base"][name] - packet["teacher"][name]).square().sum().item()
            )
    missing = [name for name, value in squared_error.items() if value == 0.0]
    if missing:
        raise RuntimeError(
            "assignment-aligned dense teacher has no identity target signal for "
            + ", ".join(missing)
        )


def _full_slot_indices(
    trace: list[Mapping[str, Any]],
    *,
    height: int,
    width: int,
    tile_size: int,
) -> torch.Tensor:
    indices: list[int] = []
    for record in trace:
        if record.get("final_route") != "Full":
            continue
        view = record.get("view_index")
        tile_row = record.get("tile_row")
        tile_column = record.get("tile_column")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (view, tile_row, tile_column)):
            raise RuntimeError("SAES teacher trace has invalid Full tile coordinates")
        for local_row in range(tile_size):
            for local_column in range(tile_size):
                row = tile_row * tile_size + local_row
                column = tile_column * tile_size + local_column
                if row >= height or column >= width:
                    raise RuntimeError("SAES teacher trace Full tile exceeds image bounds")
                indices.append(view * height * width + row * width + column)
    return torch.tensor(sorted(set(indices)), dtype=torch.long)


def _assert_full_passthrough(source: Any, candidate: Any, full_indices: torch.Tensor) -> bool:
    if full_indices.numel() == 0:
        return True
    for name in _ATTRIBUTES:
        if not torch.equal(
            getattr(source, name)[0, full_indices], getattr(candidate, name)[0, full_indices]
        ):
            return False
    return True


def _assert_psd(value: torch.Tensor, *, label: str) -> None:
    symmetric = (value + value.mT) * 0.5
    if not bool(torch.isfinite(symmetric).all()) or bool((torch.linalg.eigvalsh(symmetric) < -1.0e-6).any()):
        raise RuntimeError(f"{label} covariance is non-finite or non-PSD")


def _poison_skipped(source: Any, mask: torch.Tensor, *, value: float) -> Any:
    poisoned = _clone_gaussians(source)
    if not bool(mask.any()):
        raise RuntimeError("selected-only sentinel requires at least one skipped descriptor")
    for name in _ATTRIBUTES:
        getattr(poisoned, name)[0, mask] = value
    return poisoned


def _capture_sentinel_packets(
    source: Any,
    mask: torch.Tensor,
    *,
    height: int,
    width: int,
    saes_kwargs: Mapping[str, Any],
    value: float,
) -> tuple[torch.Tensor, Mapping[str, Any], list[dict[str, Any]]]:
    _, sentry_mask, stats, packets, _ = _capture_selected_only_packets(
        _poison_skipped(source, mask, value=value),
        height=height,
        width=width,
        saes_kwargs=saes_kwargs,
    )
    return sentry_mask, stats, packets


def _validate_saes_kwargs(value: Mapping[str, Any], *, height: int, width: int) -> dict[str, Any]:
    allowed = {
        "tile_size",
        "gpp",
        "feature_var_threshold",
        "depth_std_threshold",
        "features",
        "depths",
        "cross_check_threshold",
        "view_count",
        "materialization",
        "decision_semantics",
        "beta_x",
        "beta_f",
        "beta_d",
        "num_depth_candidates",
        "context_extrinsics",
        "context_intrinsics",
        "ray_depth_mode",
        "depth_routing_semantics",
        "depth_near",
        "depth_far",
        "materialization_guard",
        "context_safety_guard",
    }
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise ValueError("offline teacher received unsupported SAES arguments")
    kwargs = dict(value)
    if kwargs.get("materialization") != "representative":
        raise ValueError("offline dense-adaptor teacher requires representative materialization")
    if kwargs.get("gpp") != 1 or kwargs.get("tile_size") != 4:
        raise ValueError("offline dense-adaptor teacher requires the frozen K/2K tile contract")
    if height % kwargs["tile_size"] or width % kwargs["tile_size"]:
        raise ValueError("offline dense-adaptor teacher requires whole SAES tiles")
    if kwargs.get("view_count") is None or not isinstance(kwargs["view_count"], int):
        raise ValueError("offline dense-adaptor teacher has no view count")
    return kwargs


def build_dense_adaptor_teacher_packets(
    gaussians: Any,
    *,
    height: int,
    width: int,
    saes_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one target-free, selected-only packet set with dense teacher targets.

    ``gaussians`` is the dense output of the frozen model's native Gaussian
    adaptor. The selected-only pass never reads skipped S3 descriptors. Once
    its route and assignments are frozen, this offline-only routine aggregates
    the actual dense non-probe attributes with those same assignments. Neither
    path invokes a renderer or receives target-side data.
    """
    if not isinstance(height, int) or not isinstance(width, int) or min(height, width) <= 0:
        raise ValueError("offline dense-adaptor teacher has invalid dimensions")
    kwargs = _validate_saes_kwargs(saes_kwargs, height=height, width=width)
    source = _clone_gaussians(gaussians)
    expected = kwargs["view_count"] * height * width
    if source.means.shape[:2] != (1, expected):
        raise ValueError("dense adaptor output does not match the frozen context resolution")

    selected, selected_mask, selected_stats, selected_packets, selected_trace = (
        _capture_selected_only_packets(
            source, height=height, width=width, saes_kwargs=kwargs
        )
    )
    if not bool(selected_mask.any()):
        raise RuntimeError("offline dense-adaptor teacher has no selected-only sparse descriptors")

    ordinary, ordinary_mask, ordinary_stats, ordinary_trace = (
        _replay_ordinary_representatives(
            source, height=height, width=width, saes_kwargs=kwargs
        )
    )
    selected_full = _full_slot_indices(
        selected_trace, height=height, width=width, tile_size=kwargs["tile_size"]
    )
    ordinary_full = _full_slot_indices(
        ordinary_trace, height=height, width=width, tile_size=kwargs["tile_size"]
    )
    replay_controls = {
        "route_mask_unchanged": torch.equal(selected_mask, ordinary_mask),
        "retained_counts_unchanged": (
            _route_summary(selected_stats) == _route_summary(ordinary_stats)
        ),
        "operational_stats_unchanged": (
            _operational_stats(selected_stats) == _operational_stats(ordinary_stats)
        ),
        "tile_trace_unchanged": selected_trace == ordinary_trace,
        "ordinary_representative_output_unchanged": _same_gaussian_attributes(
            selected, ordinary
        ),
        "full_passthrough": (
            torch.equal(selected_full, ordinary_full)
            and _assert_full_passthrough(source, selected, selected_full)
            and _assert_full_passthrough(source, ordinary, ordinary_full)
        ),
    }
    failed_replay_controls = [
        name for name, passed in replay_controls.items() if passed is not True
    ]
    if failed_replay_controls:
        raise RuntimeError(
            "selected capture differs from ordinary representative replay: "
            + ", ".join(failed_replay_controls)
        )

    plus_mask, plus_stats, plus_packets = _capture_sentinel_packets(
        source,
        selected_mask,
        height=height,
        width=width,
        saes_kwargs=kwargs,
        value=1.0e4,
    )
    minus_mask, minus_stats, minus_packets = _capture_sentinel_packets(
        source,
        selected_mask,
        height=height,
        width=width,
        saes_kwargs=kwargs,
        value=-1.0e4,
    )
    sentinel_packets_unchanged = (
        torch.equal(selected_mask, plus_mask)
        and torch.equal(selected_mask, minus_mask)
        and _route_summary(selected_stats) == _route_summary(plus_stats)
        and _route_summary(selected_stats) == _route_summary(minus_stats)
        and len(selected_packets) == len(plus_packets)
        and len(selected_packets) == len(minus_packets)
        and not any(
            not _sample_equal(sample, plus)
            or not _sample_equal(sample, minus)
            for sample, plus, minus in zip(selected_packets, plus_packets, minus_packets)
        )
    )
    selected_only_s3 = (
        int(selected_stats.get("guard_nonprobe_s3_attribute_reads", -1)) == 0
        and sentinel_packets_unchanged
    )
    if not selected_only_s3:
        raise RuntimeError("offline selected-only teacher packets read a skipped S3 descriptor")

    packets: list[dict[str, Any]] = []
    for sample in selected_packets:
        anchor_index = int(sample["anchor_index"])
        harmonic = sample["harmonics"]
        if harmonic.ndim != 2 or harmonic.shape[0] != 3:
            raise RuntimeError("offline selected-only packet has invalid SH")
        coefficient_count = int(harmonic.shape[1])
        root = math.isqrt(coefficient_count)
        if root * root != coefficient_count or not 1 <= root <= 5:
            raise RuntimeError("offline selected-only packet has unsupported SH degree")
        base_covariance = sample["covariances"].reshape(1, 3, 3)
        teacher_target = _assignment_aligned_dense_target(source, sample)
        teacher_covariance = teacher_target["covariances"].reshape(1, 3, 3)
        _assert_psd(base_covariance, label="offline selected-only packet")
        _assert_psd(teacher_covariance, label="offline dense-adaptor teacher")
        packet = {
            "anchor_index": anchor_index,
            "level": sample["level"],
            "sh_degree": root - 1,
            "descriptor": sample["descriptor"].detach().cpu().contiguous(),
            "base": {
                "means": sample["means"].detach().cpu().contiguous(),
                "covariances": sample["covariances"].detach().cpu().contiguous(),
                "harmonics": harmonic.detach().cpu().contiguous(),
                "opacities": sample["opacities"].detach().cpu().contiguous(),
            },
            "teacher": {
                name: value.detach().cpu().contiguous()
                for name, value in teacher_target.items()
            },
        }
        if not all(bool(torch.isfinite(value).all()) for value in (packet["descriptor"], *packet["base"].values(), *packet["teacher"].values())):
            raise RuntimeError("offline teacher packet is non-finite")
        packets.append(packet)
    if not packets:
        raise RuntimeError("offline dense-adaptor teacher emitted no sparse packets")
    _require_joint_identity_signal(packets)
    return {
        "schema_version": PACKET_SCHEMA_VERSION,
        "kind": PACKET_KIND,
        "packets": packets,
        # This target-free schedule artifact is consumed by the cache worker
        # only to replay the native selected head. It is intentionally omitted
        # from persisted teacher packets and never reaches runtime.
        "retained_output_mask": (~selected_mask).detach().cpu().contiguous(),
        "route_mask_sha256": _sha256_tensor(selected_mask.to(dtype=torch.uint8)),
        "route_summary": _route_summary(selected_stats),
        "controls": {
            "route_mask_unchanged_required": True,
            "retained_counts_unchanged_required": True,
            "full_passthrough_required": True,
            "selected_only_s3_required": True,
            "two_finite_skipped_s3_sentinels_required": True,
            **replay_controls,
            "selected_only_s3": selected_only_s3,
            "full_slot_count": int(selected_full.numel()),
            "sparse_packet_count": len(packets),
            "two_finite_skipped_s3_sentinels_passed": selected_only_s3,
        },
    }


__all__ = [
    "PACKET_KIND",
    "PACKET_SCHEMA_VERSION",
    "build_dense_adaptor_teacher_packets",
]
