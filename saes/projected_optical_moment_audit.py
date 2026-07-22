"""Pure context-camera projected optical-moment diagnostics.

The helpers in this module operate only on already-materialized Gaussian
descriptors and context-camera geometry.  They deliberately have no routing,
renderer, target-view, RGB, or optimization inputs.  A caller must commit a
candidate descriptor set before using these functions to compare it with a
dense reference.

The reported optical mass is a pre-compositing footprint surrogate,
``-log(1-alpha) * sqrt(det(J Sigma J^T))``.  It is useful for directional
descriptor audits, but it is not a rendering-quality or occlusion metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log
from typing import Hashable, Iterable, Mapping

import torch


NONWORSE_PERCENT = 0.01
MIN_COVARIANCE_DECREASE = 0.20
NUMERICAL_TOLERANCE = 1.0e-12

ERROR_FIELDS = (
    "mass_absolute_error",
    "mass_relative_error",
    "center_absolute_error",
    "covariance_absolute_error",
    "covariance_relative_error",
    "footprint_determinant_absolute_error",
    "footprint_log_ratio_absolute",
    "fit_residual",
)


@dataclass(frozen=True)
class ProjectedOpticalMoment:
    """One tile's finite, unclipped projected optical moment for one context."""

    valid: bool
    reason: str | None
    mass: float | None
    center: torch.Tensor | None
    covariance: torch.Tensor | None
    determinant: float | None
    log_determinant: float | None
    footprint_log_area: float | None
    psd: bool
    condition_number: float | None
    descriptor_count: int
    offscreen_count: int


@dataclass(frozen=True)
class DenseErrors:
    """Absolute and scale-normalized errors of one sparse output to dense."""

    mass_absolute_error: float
    mass_relative_error: float
    center_absolute_error: float
    covariance_absolute_error: float
    covariance_relative_error: float
    footprint_determinant_absolute_error: float
    footprint_log_ratio: float
    footprint_log_ratio_absolute: float
    psd: bool
    condition_number: float
    fit_residual: float


@dataclass(frozen=True)
class DirectionalTileComparison:
    """Dense-reference errors for current and candidate committed tile outputs."""

    tile_key: Hashable
    level: str
    context_index: int
    valid: bool
    reason: str | None
    dense: ProjectedOpticalMoment
    current: DenseErrors | None
    candidate: DenseErrors | None


@dataclass(frozen=True)
class ErrorStatistics:
    """Fixed descriptive aggregation for one error field."""

    count: int
    mean: float
    p50: float
    p95: float
    maximum: float


@dataclass(frozen=True)
class DirectionalCellSummary:
    """One populated ``(L0/L1, context)`` cell of directional records."""

    level: str
    context_index: int
    record_count: int
    valid_count: int
    invalid_count: int
    current: Mapping[str, ErrorStatistics]
    candidate: Mapping[str, ErrorStatistics]
    percentile_nonworse: bool
    percentile_failures: tuple[str, ...]


@dataclass(frozen=True)
class DirectionalGateResult:
    """The fixed diagnostic gate result; it never authorizes rendering work."""

    passed: bool
    reason: str
    cells: tuple[DirectionalCellSummary, ...]
    aggregate_current_covariance_error: float | None
    aggregate_candidate_covariance_error: float | None
    aggregate_covariance_decrease: float | None
    aggregate_current_mass_error: float | None
    aggregate_candidate_mass_error: float | None


def _failure(
    reason: str, *, descriptor_count: int, offscreen_count: int = 0
) -> ProjectedOpticalMoment:
    return ProjectedOpticalMoment(
        valid=False,
        reason=reason,
        mass=None,
        center=None,
        covariance=None,
        determinant=None,
        log_determinant=None,
        footprint_log_area=None,
        psd=False,
        condition_number=None,
        descriptor_count=descriptor_count,
        offscreen_count=offscreen_count,
    )


def _dtype_tolerance(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(64.0 * torch.finfo(dtype).eps, dtype=dtype, device=device)


def _validate_tensor_contract(
    means: torch.Tensor,
    covariances: torch.Tensor,
    opacities: torch.Tensor,
    context_extrinsic: torch.Tensor,
    context_intrinsic: torch.Tensor,
) -> tuple[torch.Tensor | None, str | None]:
    """Validate the narrow descriptor/camera interface without coercion."""
    count = int(means.shape[0]) if means.ndim >= 1 else 0
    tensors = (means, covariances, opacities, context_extrinsic, context_intrinsic)
    if not all(torch.is_tensor(value) and torch.is_floating_point(value) for value in tensors):
        return None, "input-contract"
    if (
        means.ndim != 2
        or means.shape[1:] != (3,)
        or covariances.shape != (count, 3, 3)
        or context_extrinsic.shape != (4, 4)
        or context_intrinsic.shape != (3, 3)
        or count < 1
    ):
        return None, "input-contract"
    if opacities.shape == (count,):
        flattened_opacities = opacities
    elif opacities.shape == (count, 1):
        flattened_opacities = opacities[:, 0]
    else:
        return None, "input-contract"
    if (
        any(value.device != means.device for value in tensors)
        or any(value.dtype != means.dtype for value in tensors)
    ):
        return None, "input-contract"
    return flattened_opacities, None


def _valid_context_geometry(
    context_extrinsic: torch.Tensor, context_intrinsic: torch.Tensor
) -> bool:
    """Require finite C2W rigid geometry and an invertible intrinsic matrix."""
    if not bool(torch.isfinite(context_extrinsic).all()) or not bool(
        torch.isfinite(context_intrinsic).all()
    ):
        return False
    tolerance = _dtype_tolerance(context_extrinsic.dtype, context_extrinsic.device)
    expected_last_row = torch.tensor(
        (0.0, 0.0, 0.0, 1.0),
        device=context_extrinsic.device,
        dtype=context_extrinsic.dtype,
    )
    rotation = context_extrinsic[:3, :3]
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    if not torch.allclose(context_extrinsic[3], expected_last_row, rtol=0.0, atol=tolerance):
        return False
    if not torch.allclose(rotation.mT @ rotation, identity, rtol=0.0, atol=tolerance):
        return False
    try:
        rotation_determinant = torch.linalg.det(rotation)
        intrinsic_determinant = torch.linalg.det(context_intrinsic)
    except RuntimeError:
        return False
    return bool(
        torch.isfinite(rotation_determinant)
        and torch.isfinite(intrinsic_determinant)
        and (rotation_determinant - 1.0).abs() <= tolerance
        and intrinsic_determinant.abs() > tolerance
    )


def _validate_covariances(covariances: torch.Tensor) -> tuple[torch.Tensor | None, str | None]:
    tolerance = _dtype_tolerance(covariances.dtype, covariances.device)
    symmetric = (covariances + covariances.mT) * 0.5
    if not torch.allclose(covariances, symmetric, rtol=0.0, atol=tolerance):
        return None, "non-symmetric-covariance"
    try:
        eigenvalues = torch.linalg.eigvalsh(symmetric)
    except RuntimeError:
        return None, "invalid-covariance-psd"
    if not bool(torch.isfinite(eigenvalues).all()) or bool((eigenvalues < -tolerance).any()):
        return None, "invalid-covariance-psd"
    return symmetric, None


def _project_context_points(
    means: torch.Tensor,
    *,
    context_extrinsic: torch.Tensor,
    context_intrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, str]:
    """Project C2W world points without clipping coordinates to the frame."""
    rotation = context_extrinsic[:3, :3]
    origin = context_extrinsic[:3, 3]
    camera_points = (means - origin) @ rotation
    tiny = torch.as_tensor(torch.finfo(means.dtype).eps, device=means.device, dtype=means.dtype)
    if not bool(torch.isfinite(camera_points).all()):
        return None, "nonfinite-camera-point"
    if bool((camera_points[:, 2] <= tiny).any()):
        return None, "nonpositive-camera-depth"
    homogeneous = camera_points @ context_intrinsic.mT
    denominator = homogeneous[:, 2]
    if not bool(torch.isfinite(homogeneous).all()) or bool((denominator.abs() <= tiny).any()):
        return None, "invalid-projection"
    coordinates = homogeneous[:, :2] / denominator.unsqueeze(1)
    if not bool(torch.isfinite(coordinates).all()):
        return None, "invalid-projection"

    derivative_camera = torch.empty(
        (means.shape[0], 2, 3), device=means.device, dtype=means.dtype
    )
    for axis in range(2):
        derivative_camera[:, axis] = (
            context_intrinsic[axis].unsqueeze(0) * denominator.unsqueeze(1)
            - homogeneous[:, axis].unsqueeze(1) * context_intrinsic[2].unsqueeze(0)
        ) / denominator.square().unsqueeze(1)
    jacobians = derivative_camera @ rotation.mT
    if not bool(torch.isfinite(jacobians).all()):
        return None, "invalid-projection-jacobian"
    return coordinates, jacobians


def project_context_optical_moment(
    *,
    means: torch.Tensor,
    covariances: torch.Tensor,
    opacities: torch.Tensor,
    context_extrinsic: torch.Tensor,
    context_intrinsic: torch.Tensor,
) -> ProjectedOpticalMoment:
    """Return the unclipped projected optical moment for one descriptor set.

    All descriptor centres must have finite positive depth in the supplied
    context camera.  Centres outside the normalized image frame are reported
    in ``offscreen_count`` but remain part of the moment; dropping them would
    hide directional losses in a context-specific comparison.
    """
    descriptor_count = int(means.shape[0]) if means.ndim >= 1 else 0
    flattened_opacities, contract_error = _validate_tensor_contract(
        means, covariances, opacities, context_extrinsic, context_intrinsic
    )
    if contract_error is not None or flattened_opacities is None:
        return _failure(contract_error or "input-contract", descriptor_count=descriptor_count)
    values = (means, covariances, flattened_opacities, context_extrinsic, context_intrinsic)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        return _failure("nonfinite-input", descriptor_count=descriptor_count)
    if not _valid_context_geometry(context_extrinsic, context_intrinsic):
        return _failure("invalid-context-geometry", descriptor_count=descriptor_count)
    if bool((flattened_opacities < 0.0).any()) or bool((flattened_opacities >= 1.0).any()):
        return _failure("invalid-alpha", descriptor_count=descriptor_count)
    symmetric_covariances, covariance_error = _validate_covariances(covariances)
    if covariance_error is not None or symmetric_covariances is None:
        return _failure(covariance_error or "invalid-covariance-psd", descriptor_count=descriptor_count)

    projected = _project_context_points(
        means,
        context_extrinsic=context_extrinsic,
        context_intrinsic=context_intrinsic,
    )
    if projected[0] is None:
        return _failure(str(projected[1]), descriptor_count=descriptor_count)
    coordinates, jacobians = projected
    offscreen_count = int(
        ((coordinates < 0.0) | (coordinates > 1.0)).any(dim=1).sum().item()
    )
    projected_covariances = jacobians @ symmetric_covariances @ jacobians.mT
    projected_covariances = (projected_covariances + projected_covariances.mT) * 0.5
    if not bool(torch.isfinite(projected_covariances).all()):
        return _failure(
            "nonfinite-projected-covariance",
            descriptor_count=descriptor_count,
            offscreen_count=offscreen_count,
        )
    try:
        projected_determinants = torch.linalg.det(projected_covariances)
    except RuntimeError:
        return _failure(
            "invalid-projected-determinant",
            descriptor_count=descriptor_count,
            offscreen_count=offscreen_count,
        )
    if not bool(torch.isfinite(projected_determinants).all()) or bool(
        (projected_determinants <= 0.0).any()
    ):
        return _failure(
            "invalid-projected-determinant",
            descriptor_count=descriptor_count,
            offscreen_count=offscreen_count,
        )
    optical_depth = -torch.log1p(-flattened_opacities)
    primitive_mass = optical_depth * torch.sqrt(projected_determinants)
    total_mass = primitive_mass.sum()
    if not bool(torch.isfinite(primitive_mass).all()) or not bool(torch.isfinite(total_mass)):
        return _failure("nonfinite-optical-mass", descriptor_count=descriptor_count, offscreen_count=offscreen_count)
    if bool(total_mass <= 0.0):
        return _failure("zero-optical-mass", descriptor_count=descriptor_count, offscreen_count=offscreen_count)

    weights = primitive_mass / total_mass
    center = torch.einsum("n,ni->i", weights, coordinates)
    centered = coordinates - center.unsqueeze(0)
    covariance = torch.einsum(
        "n,nij->ij",
        weights,
        projected_covariances + torch.einsum("ni,nj->nij", centered, centered),
    )
    covariance = (covariance + covariance.mT) * 0.5
    try:
        eigenvalues = torch.linalg.eigvalsh(covariance)
        determinant = torch.linalg.det(covariance)
    except RuntimeError:
        return _failure(
            "invalid-aggregate-determinant",
            descriptor_count=descriptor_count,
            offscreen_count=offscreen_count,
        )
    if (
        not bool(torch.isfinite(eigenvalues).all())
        or bool((eigenvalues <= 0.0).any())
        or not bool(torch.isfinite(determinant))
        or bool(determinant <= 0.0)
    ):
        return _failure(
            "invalid-aggregate-determinant",
            descriptor_count=descriptor_count,
            offscreen_count=offscreen_count,
        )
    condition = eigenvalues.max() / eigenvalues.min()
    if not bool(torch.isfinite(condition)):
        return _failure(
            "invalid-condition-number",
            descriptor_count=descriptor_count,
            offscreen_count=offscreen_count,
        )
    log_determinant = torch.log(determinant)
    return ProjectedOpticalMoment(
        valid=True,
        reason=None,
        mass=float(total_mass.item()),
        center=center.detach().clone(),
        covariance=covariance.detach().clone(),
        determinant=float(determinant.item()),
        log_determinant=float(log_determinant.item()),
        footprint_log_area=float((0.5 * log_determinant).item()),
        psd=True,
        condition_number=float(condition.item()),
        descriptor_count=descriptor_count,
        offscreen_count=offscreen_count,
    )


def _errors_against_dense(
    dense: ProjectedOpticalMoment, sparse: ProjectedOpticalMoment
) -> DenseErrors:
    if not dense.valid or not sparse.valid:
        raise ValueError("dense-error calculation requires valid projected moments")
    assert dense.mass is not None and sparse.mass is not None
    assert dense.center is not None and sparse.center is not None
    assert dense.covariance is not None and sparse.covariance is not None
    assert dense.determinant is not None and sparse.determinant is not None
    assert dense.log_determinant is not None and sparse.log_determinant is not None
    assert dense.condition_number is not None and sparse.condition_number is not None
    mass_absolute_error = abs(sparse.mass - dense.mass)
    mass_relative_error = mass_absolute_error / dense.mass
    center_absolute_error = float(torch.linalg.vector_norm(sparse.center - dense.center).item())
    covariance_absolute_error = float(
        torch.linalg.matrix_norm(sparse.covariance - dense.covariance, ord="fro").item()
    )
    dense_covariance_norm = float(torch.linalg.matrix_norm(dense.covariance, ord="fro").item())
    if dense_covariance_norm <= 0.0:
        raise ValueError("valid dense projected covariance has zero norm")
    covariance_relative_error = covariance_absolute_error / dense_covariance_norm
    footprint_determinant_absolute_error = abs(sparse.determinant - dense.determinant)
    footprint_log_ratio = sparse.log_determinant - dense.log_determinant
    return DenseErrors(
        mass_absolute_error=mass_absolute_error,
        mass_relative_error=mass_relative_error,
        center_absolute_error=center_absolute_error,
        covariance_absolute_error=covariance_absolute_error,
        covariance_relative_error=covariance_relative_error,
        footprint_determinant_absolute_error=footprint_determinant_absolute_error,
        footprint_log_ratio=footprint_log_ratio,
        footprint_log_ratio_absolute=abs(footprint_log_ratio),
        psd=sparse.psd,
        condition_number=sparse.condition_number,
        # This is the normalized residual of the candidate moment against the
        # dense projected covariance, not a renderer or optimizer residual.
        fit_residual=covariance_relative_error,
    )


def compare_tile_directionally(
    *,
    tile_key: Hashable,
    level: str,
    context_index: int,
    dense: ProjectedOpticalMoment,
    current: ProjectedOpticalMoment,
    candidate: ProjectedOpticalMoment,
) -> DirectionalTileComparison:
    """Compare two committed sparse moments against one dense tile moment."""
    if level not in {"L0", "L1"}:
        return DirectionalTileComparison(
            tile_key=tile_key,
            level=level,
            context_index=context_index,
            valid=False,
            reason="invalid-level",
            dense=dense,
            current=None,
            candidate=None,
        )
    if not isinstance(context_index, int) or context_index < 0:
        return DirectionalTileComparison(
            tile_key=tile_key,
            level=level,
            context_index=context_index,
            valid=False,
            reason="invalid-context-index",
            dense=dense,
            current=None,
            candidate=None,
        )
    for label, moment in (("dense", dense), ("current", current), ("candidate", candidate)):
        if not moment.valid:
            return DirectionalTileComparison(
                tile_key=tile_key,
                level=level,
                context_index=context_index,
                valid=False,
                reason=f"{label}:{moment.reason or 'invalid'}",
                dense=dense,
                current=None,
                candidate=None,
            )
    return DirectionalTileComparison(
        tile_key=tile_key,
        level=level,
        context_index=context_index,
        valid=True,
        reason=None,
        dense=dense,
        current=_errors_against_dense(dense, current),
        candidate=_errors_against_dense(dense, candidate),
    )


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _statistics(values: list[float]) -> ErrorStatistics:
    if not values:
        raise ValueError("error statistics require a nonempty series")
    return ErrorStatistics(
        count=len(values),
        mean=sum(values) / len(values),
        p50=_quantile(values, 0.50),
        p95=_quantile(values, 0.95),
        maximum=max(values),
    )


def _error_mapping(errors: DenseErrors) -> Mapping[str, float]:
    return {name: float(getattr(errors, name)) for name in ERROR_FIELDS}


def _nonworse_percentile(candidate: float, current: float) -> bool:
    return candidate <= current * (1.0 + NONWORSE_PERCENT) + NUMERICAL_TOLERANCE


def aggregate_directional_comparisons(
    comparisons: Iterable[DirectionalTileComparison],
) -> tuple[DirectionalCellSummary, ...]:
    """Aggregate fixed p50/p95/max summaries by level and context camera."""
    grouped: dict[tuple[str, int], list[DirectionalTileComparison]] = {}
    for comparison in comparisons:
        grouped.setdefault((comparison.level, comparison.context_index), []).append(comparison)
    summaries: list[DirectionalCellSummary] = []
    for (level, context_index) in sorted(grouped, key=lambda value: (value[0], value[1])):
        records = grouped[(level, context_index)]
        valid = [record for record in records if record.valid]
        if not valid:
            summaries.append(
                DirectionalCellSummary(
                    level=level,
                    context_index=context_index,
                    record_count=len(records),
                    valid_count=0,
                    invalid_count=len(records),
                    current={},
                    candidate={},
                    percentile_nonworse=False,
                    percentile_failures=("invalid-record",),
                )
            )
            continue
        current_values = {name: [] for name in ERROR_FIELDS}
        candidate_values = {name: [] for name in ERROR_FIELDS}
        for record in valid:
            assert record.current is not None and record.candidate is not None
            for name, value in _error_mapping(record.current).items():
                current_values[name].append(value)
            for name, value in _error_mapping(record.candidate).items():
                candidate_values[name].append(value)
        current_statistics = {name: _statistics(values) for name, values in current_values.items()}
        candidate_statistics = {
            name: _statistics(values) for name, values in candidate_values.items()
        }
        failures = []
        if len(valid) != len(records):
            failures.append("invalid-record")
        for name in ERROR_FIELDS:
            for percentile in ("p50", "p95"):
                if not _nonworse_percentile(
                    getattr(candidate_statistics[name], percentile),
                    getattr(current_statistics[name], percentile),
                ):
                    failures.append(f"{name}:{percentile}")
        summaries.append(
            DirectionalCellSummary(
                level=level,
                context_index=context_index,
                record_count=len(records),
                valid_count=len(valid),
                invalid_count=len(records) - len(valid),
                current=current_statistics,
                candidate=candidate_statistics,
                percentile_nonworse=not failures,
                percentile_failures=tuple(failures),
            )
        )
    return tuple(summaries)


def evaluate_directional_gate(
    comparisons: Iterable[DirectionalTileComparison],
) -> DirectionalGateResult:
    """Evaluate the fixed, diagnostic-only directional comparison gate.

    Each populated level/context cell must keep p50 and p95 errors within one
    percent of the current materialization.  Across all valid cells, mean
    covariance error must decrease by at least twenty percent and mean mass
    error must not increase.  Invalid records are hard failures rather than
    values that can be dropped by aggregation.
    """
    materialized = tuple(comparisons)
    cells = aggregate_directional_comparisons(materialized)
    if not cells:
        return DirectionalGateResult(
            passed=False,
            reason="no-populated-cells",
            cells=(),
            aggregate_current_covariance_error=None,
            aggregate_candidate_covariance_error=None,
            aggregate_covariance_decrease=None,
            aggregate_current_mass_error=None,
            aggregate_candidate_mass_error=None,
        )
    if any(not cell.percentile_nonworse for cell in cells):
        return DirectionalGateResult(
            passed=False,
            reason="percentile-regression-or-invalid-record",
            cells=cells,
            aggregate_current_covariance_error=None,
            aggregate_candidate_covariance_error=None,
            aggregate_covariance_decrease=None,
            aggregate_current_mass_error=None,
            aggregate_candidate_mass_error=None,
        )
    valid = tuple(record for record in materialized if record.valid)
    if len(valid) != len(materialized):
        return DirectionalGateResult(
            passed=False,
            reason="invalid-record",
            cells=cells,
            aggregate_current_covariance_error=None,
            aggregate_candidate_covariance_error=None,
            aggregate_covariance_decrease=None,
            aggregate_current_mass_error=None,
            aggregate_candidate_mass_error=None,
        )
    current_covariance = sum(
        record.current.covariance_relative_error  # type: ignore[union-attr]
        for record in valid
    ) / len(valid)
    candidate_covariance = sum(
        record.candidate.covariance_relative_error  # type: ignore[union-attr]
        for record in valid
    ) / len(valid)
    current_mass = sum(
        record.current.mass_relative_error  # type: ignore[union-attr]
        for record in valid
    ) / len(valid)
    candidate_mass = sum(
        record.candidate.mass_relative_error  # type: ignore[union-attr]
        for record in valid
    ) / len(valid)
    if current_covariance <= NUMERICAL_TOLERANCE:
        return DirectionalGateResult(
            passed=False,
            reason="zero-baseline-covariance-error",
            cells=cells,
            aggregate_current_covariance_error=current_covariance,
            aggregate_candidate_covariance_error=candidate_covariance,
            aggregate_covariance_decrease=None,
            aggregate_current_mass_error=current_mass,
            aggregate_candidate_mass_error=candidate_mass,
        )
    covariance_decrease = 1.0 - candidate_covariance / current_covariance
    if covariance_decrease + NUMERICAL_TOLERANCE < MIN_COVARIANCE_DECREASE:
        return DirectionalGateResult(
            passed=False,
            reason="insufficient-aggregate-covariance-decrease",
            cells=cells,
            aggregate_current_covariance_error=current_covariance,
            aggregate_candidate_covariance_error=candidate_covariance,
            aggregate_covariance_decrease=covariance_decrease,
            aggregate_current_mass_error=current_mass,
            aggregate_candidate_mass_error=candidate_mass,
        )
    if candidate_mass > current_mass + NUMERICAL_TOLERANCE:
        return DirectionalGateResult(
            passed=False,
            reason="aggregate-mass-error-worse",
            cells=cells,
            aggregate_current_covariance_error=current_covariance,
            aggregate_candidate_covariance_error=candidate_covariance,
            aggregate_covariance_decrease=covariance_decrease,
            aggregate_current_mass_error=current_mass,
            aggregate_candidate_mass_error=candidate_mass,
        )
    return DirectionalGateResult(
        passed=True,
        reason="pass",
        cells=cells,
        aggregate_current_covariance_error=current_covariance,
        aggregate_candidate_covariance_error=candidate_covariance,
        aggregate_covariance_decrease=covariance_decrease,
        aggregate_current_mass_error=current_mass,
        aggregate_candidate_mass_error=candidate_mass,
    )


__all__ = (
    "DenseErrors",
    "DirectionalCellSummary",
    "DirectionalGateResult",
    "DirectionalTileComparison",
    "ErrorStatistics",
    "ProjectedOpticalMoment",
    "aggregate_directional_comparisons",
    "compare_tile_directionally",
    "evaluate_directional_gate",
    "project_context_optical_moment",
)
