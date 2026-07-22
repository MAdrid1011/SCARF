"""Posthoc-only dense Gaussian render-domain containment diagnostics.

This module compares an already committed retained Gaussian set against dense
non-probe Gaussian descriptors in a source context camera.  It is deliberately
not a router, preflight, renderer, or threshold-selection surface.  Its fixed
two-sigma test reports a conservative projected-domain containment proxy after
the candidate has already been produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from saes.projected_optical_moment_audit import (
    _project_context_points,
    _validate_covariances,
    _valid_context_geometry,
)


AUDIT_SCHEMA_VERSION = "saes-projected-domain-coverage-audit-v1"
FIXED_SIGMA = 2.0


@dataclass(frozen=True)
class ProjectedDomainCoverage:
    """Coverage of positive-alpha dense non-probe projected ellipse domains."""

    valid: bool
    reason: str | None
    schema_version: str
    sigma: float
    dense_descriptor_count: int
    active_dense_descriptor_count: int
    candidate_descriptor_count: int
    active_candidate_descriptor_count: int
    dense_optical_mass: float | None
    contained_optical_mass: float | None
    mass_weighted_recall: float | None
    count_recall: float | None
    hole_count: int | None
    dense_offscreen_count: int
    candidate_offscreen_count: int


@dataclass(frozen=True)
class _ProjectedEllipses:
    centers: torch.Tensor
    covariances: torch.Tensor
    optical_masses: torch.Tensor
    active: torch.Tensor
    offscreen_count: int


def _failure(
    reason: str,
    *,
    dense_descriptor_count: int = 0,
    active_dense_descriptor_count: int = 0,
    candidate_descriptor_count: int = 0,
    active_candidate_descriptor_count: int = 0,
    dense_offscreen_count: int = 0,
    candidate_offscreen_count: int = 0,
) -> ProjectedDomainCoverage:
    return ProjectedDomainCoverage(
        valid=False,
        reason=reason,
        schema_version=AUDIT_SCHEMA_VERSION,
        sigma=FIXED_SIGMA,
        dense_descriptor_count=dense_descriptor_count,
        active_dense_descriptor_count=active_dense_descriptor_count,
        candidate_descriptor_count=candidate_descriptor_count,
        active_candidate_descriptor_count=active_candidate_descriptor_count,
        dense_optical_mass=None,
        contained_optical_mass=None,
        mass_weighted_recall=None,
        count_recall=None,
        hole_count=None,
        dense_offscreen_count=dense_offscreen_count,
        candidate_offscreen_count=candidate_offscreen_count,
    )


def _flatten_opacities(opacities: torch.Tensor, count: int) -> torch.Tensor | None:
    if opacities.shape == (count,):
        return opacities
    if opacities.shape == (count, 1):
        return opacities[:, 0]
    return None


def _validate_descriptor_set(
    *,
    means: torch.Tensor,
    covariances: torch.Tensor,
    opacities: torch.Tensor,
    context_extrinsic: torch.Tensor,
    context_intrinsic: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
    """Validate one descriptor family without coercing dense inputs."""
    tensors = (means, covariances, opacities)
    if not all(torch.is_tensor(value) and torch.is_floating_point(value) for value in tensors):
        return None, None, "input-contract"
    count = int(means.shape[0]) if means.ndim >= 1 else 0
    if (
        count < 1
        or means.ndim != 2
        or means.shape[1:] != (3,)
        or covariances.shape != (count, 3, 3)
    ):
        return None, None, "input-contract"
    flattened_opacities = _flatten_opacities(opacities, count)
    if flattened_opacities is None:
        return None, None, "input-contract"
    if any(value.device != means.device or value.dtype != means.dtype for value in tensors):
        return None, None, "input-contract"
    if (
        context_extrinsic.device != means.device
        or context_intrinsic.device != means.device
        or context_extrinsic.dtype != means.dtype
        or context_intrinsic.dtype != means.dtype
    ):
        return None, None, "input-contract"
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        return None, None, "nonfinite-input"
    if bool((flattened_opacities < 0.0).any()) or bool((flattened_opacities >= 1.0).any()):
        return None, None, "invalid-alpha"
    symmetric_covariances, covariance_error = _validate_covariances(covariances)
    if covariance_error is not None or symmetric_covariances is None:
        return None, None, covariance_error or "invalid-covariance-psd"
    return flattened_opacities, symmetric_covariances, None


def _project_ellipses(
    *,
    means: torch.Tensor,
    covariances: torch.Tensor,
    opacities: torch.Tensor,
    context_extrinsic: torch.Tensor,
    context_intrinsic: torch.Tensor,
) -> tuple[_ProjectedEllipses | None, str | None]:
    """Project valid world-space Gaussian covariances into one context camera."""
    projected = _project_context_points(
        means,
        context_extrinsic=context_extrinsic,
        context_intrinsic=context_intrinsic,
    )
    if projected[0] is None:
        return None, str(projected[1])
    centers, jacobians = projected
    projected_covariances = jacobians @ covariances @ jacobians.mT
    projected_covariances = (projected_covariances + projected_covariances.mT) * 0.5
    try:
        determinants = torch.linalg.det(projected_covariances)
        eigenvalues = torch.linalg.eigvalsh(projected_covariances)
    except RuntimeError:
        return None, "invalid-projected-covariance"
    if (
        not bool(torch.isfinite(projected_covariances).all())
        or not bool(torch.isfinite(determinants).all())
        or not bool(torch.isfinite(eigenvalues).all())
        or bool((determinants <= 0.0).any())
        or bool((eigenvalues <= 0.0).any())
    ):
        return None, "invalid-projected-covariance"
    optical_depth = -torch.log1p(-opacities)
    optical_masses = optical_depth * torch.sqrt(determinants)
    if not bool(torch.isfinite(optical_masses).all()):
        return None, "nonfinite-optical-mass"
    return (
        _ProjectedEllipses(
            centers=centers,
            covariances=projected_covariances,
            optical_masses=optical_masses,
            active=optical_depth > 0.0,
            offscreen_count=int(
                ((centers < 0.0) | (centers > 1.0)).any(dim=1).sum().item()
            ),
        ),
        None,
    )


def _contains_two_sigma_ellipse(
    *,
    dense_center: torch.Tensor,
    dense_covariance: torch.Tensor,
    candidate_center: torch.Tensor,
    candidate_covariance: torch.Tensor,
) -> tuple[bool | None, str | None]:
    """Return a sufficient containment test for two positive-definite ellipses.

    For ``E(c, C, k) = {x : (x-c)^T C^-1 (x-c) <= k^2}``,
    ``m + k*rho <= k`` is sufficient for ``E_dense`` to lie inside
    ``E_candidate``, where ``m`` is the candidate-whitened center distance and
    ``rho`` is the spectral norm of the covariance transport.
    """
    try:
        candidate_factor = torch.linalg.cholesky(candidate_covariance)
        dense_factor = torch.linalg.cholesky(dense_covariance)
        displacement = (dense_center - candidate_center).unsqueeze(1)
        whitened_displacement = torch.linalg.solve_triangular(
            candidate_factor, displacement, upper=False
        )
        whitened_shape = torch.linalg.solve_triangular(
            candidate_factor, dense_factor, upper=False
        )
        center_distance = torch.linalg.vector_norm(whitened_displacement)
        shape_scale = torch.linalg.matrix_norm(whitened_shape, ord=2)
    except RuntimeError:
        return None, "invalid-containment-factorization"
    if not bool(torch.isfinite(center_distance)) or not bool(torch.isfinite(shape_scale)):
        return None, "invalid-containment-factorization"
    tolerance = torch.as_tensor(
        64.0 * torch.finfo(dense_center.dtype).eps,
        device=dense_center.device,
        dtype=dense_center.dtype,
    )
    return bool(center_distance + FIXED_SIGMA * shape_scale <= FIXED_SIGMA + tolerance), None


def audit_projected_dense_domain_coverage(
    *,
    dense_means: torch.Tensor,
    dense_covariances: torch.Tensor,
    dense_opacities: torch.Tensor,
    candidate_means: torch.Tensor,
    candidate_covariances: torch.Tensor,
    candidate_opacities: torch.Tensor,
    context_extrinsic: torch.Tensor,
    context_intrinsic: torch.Tensor,
) -> ProjectedDomainCoverage:
    """Audit fixed two-sigma dense-domain containment after candidate commit.

    Dense non-probe descriptors are legal here only as a posthoc reference.
    No target-view metadata, RGB, routing signal, or threshold-selection input
    is accepted.  Positive-alpha dense ellipses that are not contained by at
    least one positive-alpha retained ellipse are counted as holes.
    """
    dense_count = int(dense_means.shape[0]) if torch.is_tensor(dense_means) and dense_means.ndim else 0
    candidate_count = (
        int(candidate_means.shape[0])
        if torch.is_tensor(candidate_means) and candidate_means.ndim
        else 0
    )
    camera_tensors = (context_extrinsic, context_intrinsic)
    if (
        not all(torch.is_tensor(value) and torch.is_floating_point(value) for value in camera_tensors)
        or context_extrinsic.shape != (4, 4)
        or context_intrinsic.shape != (3, 3)
    ):
        return _failure(
            "camera-input-contract",
            dense_descriptor_count=dense_count,
            candidate_descriptor_count=candidate_count,
        )
    if not _valid_context_geometry(context_extrinsic, context_intrinsic):
        return _failure(
            "invalid-context-geometry",
            dense_descriptor_count=dense_count,
            candidate_descriptor_count=candidate_count,
        )
    dense_alpha, dense_covariances_checked, dense_error = _validate_descriptor_set(
        means=dense_means,
        covariances=dense_covariances,
        opacities=dense_opacities,
        context_extrinsic=context_extrinsic,
        context_intrinsic=context_intrinsic,
    )
    if dense_error is not None or dense_alpha is None or dense_covariances_checked is None:
        return _failure(
            f"dense:{dense_error or 'input-contract'}",
            dense_descriptor_count=dense_count,
            candidate_descriptor_count=candidate_count,
        )
    candidate_alpha, candidate_covariances_checked, candidate_error = _validate_descriptor_set(
        means=candidate_means,
        covariances=candidate_covariances,
        opacities=candidate_opacities,
        context_extrinsic=context_extrinsic,
        context_intrinsic=context_intrinsic,
    )
    if (
        candidate_error is not None
        or candidate_alpha is None
        or candidate_covariances_checked is None
    ):
        return _failure(
            f"candidate:{candidate_error or 'input-contract'}",
            dense_descriptor_count=dense_count,
            candidate_descriptor_count=candidate_count,
        )
    if candidate_means.device != dense_means.device or candidate_means.dtype != dense_means.dtype:
        return _failure(
            "cross-family-input-contract",
            dense_descriptor_count=dense_count,
            candidate_descriptor_count=candidate_count,
        )
    dense_projection, dense_projection_error = _project_ellipses(
        means=dense_means,
        covariances=dense_covariances_checked,
        opacities=dense_alpha,
        context_extrinsic=context_extrinsic,
        context_intrinsic=context_intrinsic,
    )
    if dense_projection_error is not None or dense_projection is None:
        return _failure(
            f"dense:{dense_projection_error or 'projection-failure'}",
            dense_descriptor_count=dense_count,
            candidate_descriptor_count=candidate_count,
        )
    candidate_projection, candidate_projection_error = _project_ellipses(
        means=candidate_means,
        covariances=candidate_covariances_checked,
        opacities=candidate_alpha,
        context_extrinsic=context_extrinsic,
        context_intrinsic=context_intrinsic,
    )
    if candidate_projection_error is not None or candidate_projection is None:
        return _failure(
            f"candidate:{candidate_projection_error or 'projection-failure'}",
            dense_descriptor_count=dense_count,
            candidate_descriptor_count=candidate_count,
            dense_offscreen_count=dense_projection.offscreen_count,
        )
    active_dense = dense_projection.active.nonzero(as_tuple=False).flatten()
    active_candidate = candidate_projection.active.nonzero(as_tuple=False).flatten()
    if active_dense.numel() == 0:
        return _failure(
            "zero-dense-optical-mass",
            dense_descriptor_count=dense_count,
            candidate_descriptor_count=candidate_count,
            dense_offscreen_count=dense_projection.offscreen_count,
            candidate_offscreen_count=candidate_projection.offscreen_count,
        )
    if active_candidate.numel() == 0:
        return _failure(
            "zero-candidate-optical-mass",
            dense_descriptor_count=dense_count,
            active_dense_descriptor_count=int(active_dense.numel()),
            candidate_descriptor_count=candidate_count,
            dense_offscreen_count=dense_projection.offscreen_count,
            candidate_offscreen_count=candidate_projection.offscreen_count,
        )
    dense_mass = dense_projection.optical_masses[active_dense].sum()
    if not bool(torch.isfinite(dense_mass)) or bool(dense_mass <= 0.0):
        return _failure(
            "zero-dense-optical-mass",
            dense_descriptor_count=dense_count,
            active_dense_descriptor_count=int(active_dense.numel()),
            candidate_descriptor_count=candidate_count,
            active_candidate_descriptor_count=int(active_candidate.numel()),
            dense_offscreen_count=dense_projection.offscreen_count,
            candidate_offscreen_count=candidate_projection.offscreen_count,
        )
    contained_mass = dense_mass.new_zeros(())
    contained_count = 0
    for dense_index in active_dense.tolist():
        contained = False
        for candidate_index in active_candidate.tolist():
            result, containment_error = _contains_two_sigma_ellipse(
                dense_center=dense_projection.centers[dense_index],
                dense_covariance=dense_projection.covariances[dense_index],
                candidate_center=candidate_projection.centers[candidate_index],
                candidate_covariance=candidate_projection.covariances[candidate_index],
            )
            if containment_error is not None or result is None:
                return _failure(
                    f"containment:{containment_error or 'invalid'}",
                    dense_descriptor_count=dense_count,
                    active_dense_descriptor_count=int(active_dense.numel()),
                    candidate_descriptor_count=candidate_count,
                    active_candidate_descriptor_count=int(active_candidate.numel()),
                    dense_offscreen_count=dense_projection.offscreen_count,
                    candidate_offscreen_count=candidate_projection.offscreen_count,
                )
            if result:
                contained = True
                break
        if contained:
            contained_count += 1
            contained_mass = contained_mass + dense_projection.optical_masses[dense_index]
    active_dense_count = int(active_dense.numel())
    hole_count = active_dense_count - contained_count
    return ProjectedDomainCoverage(
        valid=True,
        reason=None,
        schema_version=AUDIT_SCHEMA_VERSION,
        sigma=FIXED_SIGMA,
        dense_descriptor_count=dense_count,
        active_dense_descriptor_count=active_dense_count,
        candidate_descriptor_count=candidate_count,
        active_candidate_descriptor_count=int(active_candidate.numel()),
        dense_optical_mass=float(dense_mass.item()),
        contained_optical_mass=float(contained_mass.item()),
        mass_weighted_recall=float((contained_mass / dense_mass).item()),
        count_recall=float(contained_count / active_dense_count),
        hole_count=hole_count,
        dense_offscreen_count=dense_projection.offscreen_count,
        candidate_offscreen_count=candidate_projection.offscreen_count,
    )


__all__ = (
    "AUDIT_SCHEMA_VERSION",
    "FIXED_SIGMA",
    "ProjectedDomainCoverage",
    "audit_projected_dense_domain_coverage",
)
