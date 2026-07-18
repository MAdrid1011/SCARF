"""Shared S3-access checks for frozen target-free audit runners."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Mapping

import torch


ATTRIBUTE_NAMES = ("means", "covariances", "harmonics", "opacities")
POSTHOC_DENSE_ORACLE_MATERIALIZATION = "same-budget-dense-oracle-diagnostic"
SENTINEL_SPECS = (
    (
        "finite-sentinel-a",
        {
            "means": 1.0e4,
            "covariances": -1.0e4,
            "harmonics": 1.0e4,
            "opacities": 0.99,
            "depth": 1.0e4,
        },
    ),
    (
        "finite-sentinel-b",
        {
            "means": -2.0e4,
            "covariances": 2.0e4,
            "harmonics": -2.0e4,
            "opacities": 0.01,
            "depth": -2.0e4,
        },
    ),
)


class SelectedOnlyS3Tensor:
    """Reject descriptor reads outside an explicit selected-slot allowlist."""

    def __init__(self, tensor: torch.Tensor, allowed_indices: torch.Tensor):
        self._tensor = tensor
        self._allowed_indices = {
            int(index) for index in allowed_indices.detach().cpu().reshape(-1).tolist()
        }
        self.read_indices: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tensor, name)

    def __getitem__(self, index: Any) -> Any:
        if not isinstance(index, tuple) or len(index) != 2 or index[0] != 0:
            raise AssertionError(f"unexpected S3 tensor read index: {index!r}")
        requested = index[1]
        if isinstance(requested, int):
            indices = [requested]
        elif torch.is_tensor(requested):
            indices = [
                int(value) for value in requested.detach().cpu().reshape(-1).tolist()
            ]
        else:
            indices = [int(value) for value in requested]
        if not set(indices) <= self._allowed_indices:
            raise AssertionError(f"skipped S3 descriptor read: {indices!r}")
        self.read_indices.extend(indices)
        return self._tensor[index]

    def __setitem__(self, index: Any, value: Any) -> None:
        self._tensor[index] = value


def wrap_selected_s3_reads(
    gaussians: Any, selected_indices: torch.Tensor
) -> tuple[Any, dict[str, SelectedOnlyS3Tensor]]:
    """Expose Gaussian attributes through the common selected-read guard."""
    fields = {
        name: SelectedOnlyS3Tensor(getattr(gaussians, name), selected_indices)
        for name in ATTRIBUTE_NAMES
    }
    return SimpleNamespace(**fields), fields


def selected_s3_read_evidence(
    fields: Mapping[str, SelectedOnlyS3Tensor], *, label: str
) -> dict[str, dict[str, int | bool]]:
    """Fail closed unless every guarded attribute read selected S3 slots."""
    if set(fields) != set(ATTRIBUTE_NAMES):
        raise RuntimeError(f"{label} has an incomplete selected-S3 guard")
    evidence: dict[str, dict[str, int | bool]] = {}
    for name in ATTRIBUTE_NAMES:
        field = fields[name]
        if not field.read_indices:
            raise RuntimeError(f"{label} did not read selected {name}")
        evidence[name] = {
            "read_count": len(field.read_indices),
            "selected_only": True,
        }
    return evidence


def poison_skipped_s3(
    gaussians: Any, modified_mask: torch.Tensor, *, sentinel: Mapping[str, float]
) -> None:
    """Replace only skipped S3 descriptors with one finite sentinel family."""
    skipped = torch.nonzero(modified_mask, as_tuple=False).flatten()
    for name in ATTRIBUTE_NAMES:
        getattr(gaussians, name)[0, skipped] = float(sentinel[name])


def poison_skipped_depths(
    depths: torch.Tensor,
    modified_mask: torch.Tensor,
    *,
    height: int,
    width: int,
    primitives_per_pixel: int,
    value: float,
) -> None:
    """Replace S2 values only at positions whose S3 slots are skipped."""
    if primitives_per_pixel < 1:
        raise RuntimeError("selected-only replay has an invalid primitives-per-pixel count")
    positions = torch.unique(
        torch.nonzero(modified_mask, as_tuple=False).flatten() // primitives_per_pixel
    )
    per_view = height * width
    for position in positions.tolist():
        view, pixel = divmod(position, per_view)
        if depths.ndim == 5:
            depths[0, view, pixel] = value
        elif depths.ndim == 4:
            row, column = divmod(pixel, width)
            depths[0, view, row, column] = value
        else:
            raise RuntimeError("selected-only replay has an unsupported S2 depth layout")


def selected_only_commit_payload(
    *,
    gaussians: Any,
    modified_mask: torch.Tensor,
    stats: Mapping[str, Any],
    tile_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Capture only committed output slots for a selected-only replay check."""
    if modified_mask.ndim != 1 or modified_mask.dtype != torch.bool:
        raise RuntimeError("selected-only replay has an invalid modified mask")
    if gaussians.means.shape[0] != 1 or gaussians.means.shape[1] != modified_mask.numel():
        raise RuntimeError("selected-only replay descriptor layout differs from its mask")
    retained = torch.nonzero(~modified_mask, as_tuple=False).flatten()
    if retained.numel() == 0:
        raise RuntimeError("selected-only replay has no committed descriptors")
    return {
        "modified_mask": modified_mask.detach().cpu().clone(),
        "retained_indices": retained.detach().cpu().clone(),
        "means": gaussians.means[0, retained].detach().cpu().clone(),
        "covariances": gaussians.covariances[0, retained].detach().cpu().clone(),
        "harmonics": gaussians.harmonics[0, retained].detach().cpu().clone(),
        "opacities": gaussians.opacities[0, retained].detach().cpu().clone(),
        "stats": deepcopy(dict(stats)),
        "tile_trace": deepcopy(tile_trace),
    }


def assert_selected_only_commit_equal(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *, label: str
) -> dict[str, float]:
    """Fail closed on route, counter, trace, or committed-output drift."""
    expected = {
        "modified_mask",
        "retained_indices",
        *ATTRIBUTE_NAMES,
        "stats",
        "tile_trace",
    }
    if set(reference) != expected or set(candidate) != expected:
        raise RuntimeError(f"{label} selected-only replay has an invalid payload schema")
    for name in ("stats", "tile_trace"):
        if reference[name] != candidate[name]:
            raise RuntimeError(f"{label} changed committed {name}")
    for name in ("modified_mask", "retained_indices"):
        if not torch.equal(reference[name], candidate[name]):
            raise RuntimeError(f"{label} changed committed {name}")
    deltas: dict[str, float] = {}
    for name in ATTRIBUTE_NAMES:
        first, second = reference[name], candidate[name]
        if first.shape != second.shape:
            raise RuntimeError(f"{label} changed committed {name} shape")
        delta = float((first - second).abs().max().item()) if first.numel() else 0.0
        if delta != 0.0:
            raise RuntimeError(f"{label} changed committed {name}")
        deltas[name] = delta
    return deltas


def require_posthoc_full_s3_exception(
    *,
    materialization: str,
    runtime_execution: bool,
    paper_result_eligible: bool,
    quality_gate_authorized: bool,
) -> dict[str, Any]:
    """Bind a full-S3 diagnostic to explicit non-runtime, non-quality limits."""
    if materialization != POSTHOC_DENSE_ORACLE_MATERIALIZATION:
        raise RuntimeError("post-hoc full-S3 exception has the wrong materialization")
    if runtime_execution or paper_result_eligible or quality_gate_authorized:
        raise RuntimeError("post-hoc full-S3 exception cannot carry runtime or quality eligibility")
    return {
        "kind": "s3-access-contract-v1",
        "mode": "posthoc-full-s3-diagnostic-exception",
        "materialization": materialization,
        "selected_only_s3_instrumentation_applicable": False,
        "selected_only_s3_instrumentation_verdict": "not-applicable",
        "posthoc_full_s3_read_permitted": True,
        "runtime_execution": False,
        "paper_result_eligible": False,
        "quality_gate_authorized": False,
    }


def bind_posthoc_full_s3_observation(
    contract: Mapping[str, Any], *, full_stage3_reads: int
) -> dict[str, Any]:
    """Require the declared full-S3 exception to observe at least one full read."""
    if contract.get("mode") != "posthoc-full-s3-diagnostic-exception":
        raise RuntimeError("full-S3 observation lacks the required exception contract")
    if isinstance(full_stage3_reads, bool) or not isinstance(full_stage3_reads, int):
        raise RuntimeError("full-S3 observation count is invalid")
    if full_stage3_reads <= 0:
        raise RuntimeError("post-hoc full-S3 diagnostic did not record a full-S3 read")
    return {**contract, "observed_full_stage3_reads": full_stage3_reads}
