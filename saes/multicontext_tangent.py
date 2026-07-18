"""Target-free multi-context tangent-plane covariance reconstruction.

This module is deliberately independent of ``ProgressiveSAES``.  It accepts
only already-selected local contributors and context-camera geometry, and
returns an untouched local covariance whenever the multi-context fit is not
identifiable.  It neither selects routes nor accesses target cameras, RGB, or
skipped Stage-3 descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TangentCovarianceResult:
    """A covariance candidate or an exact fail-closed local fallback."""

    covariance: torch.Tensor
    used_multicontext_fit: bool
    reason: str
    residual_max: float | None


def _fallback(covariance: torch.Tensor, reason: str) -> TangentCovarianceResult:
    return TangentCovarianceResult(
        covariance=covariance.clone(),
        used_multicontext_fit=False,
        reason=reason,
        residual_max=None,
    )


def _valid_inputs(
    output_mean: torch.Tensor,
    local_covariance: torch.Tensor,
    contributor_means: torch.Tensor,
    contributor_covariances: torch.Tensor,
    contributor_weights: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
) -> bool:
    count = contributor_weights.numel()
    tensors = (
        output_mean,
        local_covariance,
        contributor_means,
        contributor_covariances,
        contributor_weights,
        context_extrinsics,
        context_intrinsics,
    )
    return bool(
        output_mean.shape == (3,)
        and local_covariance.shape == (3, 3)
        and contributor_means.shape == (count, 3)
        and contributor_covariances.shape == (count, 3, 3)
        and context_extrinsics.ndim == 3
        and context_intrinsics.ndim == 3
        and context_extrinsics.shape[1:] == (4, 4)
        and context_intrinsics.shape
        == (context_extrinsics.shape[0], 3, 3)
        and count >= 1
        and all(torch.is_floating_point(value) for value in tensors)
        and output_mean.device == local_covariance.device
        == contributor_means.device
        == contributor_covariances.device
        == contributor_weights.device
        == context_extrinsics.device
        == context_intrinsics.device
        and output_mean.dtype == local_covariance.dtype
        == contributor_means.dtype
        == contributor_covariances.dtype
        == contributor_weights.dtype
        == context_extrinsics.dtype
        == context_intrinsics.dtype
    )


def _camera_tolerance(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(64.0 * torch.finfo(dtype).eps, device=device, dtype=dtype)


def _valid_context_geometry(
    context_extrinsics: torch.Tensor, context_intrinsics: torch.Tensor
) -> bool:
    """Require finite C2W rigid transforms and full-rank intrinsics."""
    tolerance = _camera_tolerance(context_extrinsics.dtype, context_extrinsics.device)
    expected_last_row = torch.tensor(
        (0.0, 0.0, 0.0, 1.0),
        device=context_extrinsics.device,
        dtype=context_extrinsics.dtype,
    )
    identity = torch.eye(3, device=context_extrinsics.device, dtype=context_extrinsics.dtype)
    for extrinsic, intrinsic in zip(context_extrinsics, context_intrinsics):
        rotation = extrinsic[:3, :3]
        if (
            not torch.allclose(extrinsic[3], expected_last_row, rtol=0.0, atol=tolerance)
            or not torch.allclose(rotation.mT @ rotation, identity, rtol=0.0, atol=tolerance)
        ):
            return False
        try:
            determinant = torch.linalg.det(rotation)
            intrinsic_rank = int(torch.linalg.matrix_rank(intrinsic).item())
        except RuntimeError:
            return False
        if (
            not bool(torch.isfinite(determinant))
            or bool((determinant - 1.0).abs() > tolerance)
            or intrinsic_rank != 3
        ):
            return False
    return True


def _valid_psd_covariances(
    local_covariance: torch.Tensor, contributor_covariances: torch.Tensor
) -> bool:
    """Reject non-symmetric or indefinite descriptor covariances before fitting."""
    tolerance = _camera_tolerance(local_covariance.dtype, local_covariance.device)
    covariances = torch.cat(
        (local_covariance.unsqueeze(0), contributor_covariances), dim=0
    )
    if not torch.allclose(covariances, covariances.mT, rtol=0.0, atol=tolerance):
        return False
    try:
        eigenvalues = torch.linalg.eigvalsh(covariances)
    except RuntimeError:
        return False
    return bool(torch.isfinite(eigenvalues).all() and (eigenvalues >= -tolerance).all())


def _project_world_points(
    points: torch.Tensor,
    *,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Project world points and return normalized-image Jacobians."""
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    rotation = extrinsic[:3, :3]
    origin = extrinsic[:3, 3]
    camera_points = (points - origin) @ rotation
    homogeneous = camera_points @ intrinsic.mT
    denominator = homogeneous[:, 2]
    tiny = torch.as_tensor(torch.finfo(points.dtype).eps, device=points.device)
    if (
        not bool(torch.isfinite(camera_points).all())
        or not bool(torch.isfinite(homogeneous).all())
        or bool((camera_points[:, 2] <= tiny).any())
        or bool((denominator.abs() <= tiny).any())
    ):
        return None
    coordinates = homogeneous[:, :2] / denominator.unsqueeze(1)
    if (
        not bool(torch.isfinite(coordinates).all())
        or bool(((coordinates < 0.0) | (coordinates > 1.0)).any())
    ):
        return None

    # d((K p)[:2] / (K p)[2]) / d p, followed by d p / d world = R^T.
    derivative_camera = torch.empty(
        (points.shape[0], 2, 3), device=points.device, dtype=points.dtype
    )
    for axis in range(2):
        derivative_camera[:, axis] = (
            intrinsic[axis].unsqueeze(0) * denominator.unsqueeze(1)
            - homogeneous[:, axis].unsqueeze(1) * intrinsic[2].unsqueeze(0)
        ) / denominator.square().unsqueeze(1)
    jacobian_world = derivative_camera @ rotation.mT
    if not bool(torch.isfinite(jacobian_world).all()):
        return None
    return coordinates, jacobian_world


def _projected_second_moment(
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a local mixture's image-plane mean and covariance for one camera."""
    projected = _project_world_points(means, extrinsic=extrinsic, intrinsic=intrinsic)
    if projected is None:
        return None
    coordinates, jacobians = projected
    projected_covariances = jacobians @ covariances @ jacobians.mT
    projected_covariances = (projected_covariances + projected_covariances.mT) * 0.5
    if not bool(torch.isfinite(projected_covariances).all()):
        return None
    mean = torch.einsum("n,ni->i", weights, coordinates)
    centered = coordinates - mean.unsqueeze(0)
    covariance = torch.einsum(
        "n,nij->ij",
        weights,
        projected_covariances + torch.einsum("ni,nj->nij", centered, centered),
    )
    covariance = (covariance + covariance.mT) * 0.5
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(covariance).all()):
        return None
    return mean, covariance


def _tangent_basis(ray: torch.Tensor) -> torch.Tensor | None:
    """Build any orthonormal basis for the plane normal to ``ray``."""
    tiny = torch.as_tensor(torch.finfo(ray.dtype).eps, device=ray.device)
    norm = ray.norm()
    if not bool(torch.isfinite(norm)) or bool(norm <= tiny):
        return None
    direction = ray / norm
    axis = torch.zeros(3, dtype=ray.dtype, device=ray.device)
    axis[int(torch.argmin(direction.abs()).item())] = 1.0
    first = torch.linalg.cross(direction, axis)
    first_norm = first.norm()
    if not bool(torch.isfinite(first_norm)) or bool(first_norm <= tiny):
        return None
    first = first / first_norm
    second = torch.linalg.cross(direction, first)
    second_norm = second.norm()
    if not bool(torch.isfinite(second_norm)) or bool(second_norm <= tiny):
        return None
    return torch.stack((first, second / second_norm), dim=1)


def _symmetric_projection_rows(jacobian: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Map the three tangent covariance coefficients to [xx, xy, yy]."""
    tangent_jacobian = jacobian @ basis
    rows = []
    for first, second in ((0, 0), (0, 1), (1, 1)):
        rows.append(
            torch.stack(
                (
                    tangent_jacobian[first, 0] * tangent_jacobian[second, 0],
                    tangent_jacobian[first, 0] * tangent_jacobian[second, 1]
                    + tangent_jacobian[first, 1] * tangent_jacobian[second, 0],
                    tangent_jacobian[first, 1] * tangent_jacobian[second, 1],
                )
            )
        )
    return torch.stack(rows)


def _symmetric_entries(covariance: torch.Tensor) -> torch.Tensor:
    return torch.stack((covariance[0, 0], covariance[0, 1], covariance[1, 1]))


def _project_psd(covariance: torch.Tensor) -> torch.Tensor | None:
    covariance = (covariance + covariance.mT) * 0.5
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    except RuntimeError:
        return None
    if not bool(torch.isfinite(eigenvalues).all()):
        return None
    floor = torch.as_tensor(torch.finfo(covariance.dtype).eps, device=covariance.device)
    return eigenvectors @ torch.diag(eigenvalues.clamp_min(floor)) @ eigenvectors.mT


def multicontext_tangent_covariance(
    *,
    output_mean: torch.Tensor,
    local_covariance: torch.Tensor,
    contributor_means: torch.Tensor,
    contributor_covariances: torch.Tensor,
    contributor_weights: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
) -> TangentCovarianceResult:
    """Fit a PSD tangent-plane covariance from two or more context cameras.

    The radial variance stays fixed at the local representative value.  This
    leaves three tangent-plane degrees of freedom, which two nondegenerate
    cameras can identify.  All failure conditions return an exact clone of
    ``local_covariance`` so a future tile-level caller can remain atomic.
    """
    if not _valid_inputs(
        output_mean,
        local_covariance,
        contributor_means,
        contributor_covariances,
        contributor_weights,
        context_extrinsics,
        context_intrinsics,
    ):
        return _fallback(local_covariance, "input-contract")
    values = (
        output_mean,
        local_covariance,
        contributor_means,
        contributor_covariances,
        contributor_weights,
        context_extrinsics,
        context_intrinsics,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        return _fallback(local_covariance, "nonfinite-input")
    if not _valid_context_geometry(context_extrinsics, context_intrinsics):
        return _fallback(local_covariance, "invalid-context-geometry")
    if not _valid_psd_covariances(local_covariance, contributor_covariances):
        return _fallback(local_covariance, "invalid-source-covariance")
    if bool((contributor_weights < 0.0).any()) or not bool(
        torch.isclose(
            contributor_weights.sum(),
            torch.ones((), device=contributor_weights.device, dtype=contributor_weights.dtype),
            rtol=1e-5,
            atol=1e-6,
        )
    ):
        return _fallback(local_covariance, "assignment-simplex")
    if context_extrinsics.shape[0] < 2:
        return _fallback(local_covariance, "single-context")
    for first in range(context_extrinsics.shape[0]):
        for second in range(first + 1, context_extrinsics.shape[0]):
            tolerance = _camera_tolerance(
                context_extrinsics.dtype, context_extrinsics.device
            )
            if torch.allclose(
                context_extrinsics[first], context_extrinsics[second], rtol=0.0, atol=tolerance
            ):
                return _fallback(local_covariance, "duplicate-context-pose")
    if int((contributor_weights > 0.0).sum().item()) == 1:
        return _fallback(local_covariance, "one-hot-assignment")
    if torch.equal(contributor_means, output_mean.expand_as(contributor_means)) and torch.equal(
        contributor_covariances,
        local_covariance.expand_as(contributor_covariances),
    ):
        return _fallback(local_covariance, "constant-local-descriptor")

    # A mean context origin makes the tangent/radial split independent of the
    # input context-view order. It is only a coordinate basis; all cameras
    # still contribute their own projected-moment constraints below.
    reference_origin = context_extrinsics[:, :3, 3].mean(dim=0)
    basis = _tangent_basis(output_mean - reference_origin)
    if basis is None:
        return _fallback(local_covariance, "reference-ray")
    radial = output_mean - reference_origin
    radial = radial / radial.norm()
    radial_variance = radial @ local_covariance @ radial
    epsilon = torch.as_tensor(torch.finfo(output_mean.dtype).eps, device=output_mean.device)
    if not bool(torch.isfinite(radial_variance)) or bool(radial_variance < 0.0):
        return _fallback(local_covariance, "radial-variance")
    radial_variance = radial_variance.clamp_min(epsilon)
    tangent_radial = basis.mT @ local_covariance @ radial
    fixed_covariance = (
        radial_variance * torch.outer(radial, radial)
        + basis @ tangent_radial.unsqueeze(1) @ radial.unsqueeze(0)
        + radial.unsqueeze(1) @ tangent_radial.unsqueeze(0) @ basis.mT
    )

    system_rows = []
    system_targets = []
    fitted_jacobians = []
    target_covariances = []
    for view in range(context_extrinsics.shape[0]):
        extrinsic = context_extrinsics[view]
        intrinsic = context_intrinsics[view]
        target = _projected_second_moment(
            contributor_means,
            contributor_covariances,
            contributor_weights,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
        )
        output_projection = _project_world_points(
            output_mean.unsqueeze(0), extrinsic=extrinsic, intrinsic=intrinsic
        )
        if target is None or output_projection is None:
            return _fallback(local_covariance, "invalid-context-projection")
        _, target_covariance = target
        output_coordinates, projected_output_jacobians = output_projection
        if not bool(torch.isfinite(output_coordinates).all()):
            return _fallback(local_covariance, "invalid-output-projection")
        jacobian = projected_output_jacobians[0]
        fixed_projection = jacobian @ fixed_covariance @ jacobian.mT
        system_rows.append(_symmetric_projection_rows(jacobian, basis))
        system_targets.append(_symmetric_entries(target_covariance - fixed_projection))
        fitted_jacobians.append(jacobian)
        target_covariances.append(target_covariance)

    matrix = torch.cat(system_rows, dim=0)
    target = torch.cat(system_targets, dim=0)
    try:
        rank = int(torch.linalg.matrix_rank(matrix).item())
    except RuntimeError:
        return _fallback(local_covariance, "rank-evaluation")
    if rank < 3:
        return _fallback(local_covariance, "rank-deficient-contexts")
    try:
        solution = torch.linalg.lstsq(matrix, target.unsqueeze(1)).solution[:3, 0]
    except RuntimeError:
        return _fallback(local_covariance, "least-squares")
    if not bool(torch.isfinite(solution).all()):
        return _fallback(local_covariance, "nonfinite-fit")
    tangent_covariance = torch.stack(
        (
            torch.stack((solution[0], solution[1])),
            torch.stack((solution[1], solution[2])),
        )
    )
    tangent_covariance = _project_psd(tangent_covariance)
    if tangent_covariance is None:
        return _fallback(local_covariance, "tangent-psd")
    covariance = (
        basis @ tangent_covariance @ basis.mT
        + fixed_covariance
    )
    covariance = _project_psd(covariance)
    if covariance is None:
        return _fallback(local_covariance, "world-psd")
    residual = torch.stack(
        tuple(
            (jacobian @ covariance @ jacobian.mT - target_covariance).abs().max()
            for jacobian, target_covariance in zip(fitted_jacobians, target_covariances)
        )
    )
    return TangentCovarianceResult(
        covariance=covariance,
        used_multicontext_fit=True,
        reason="multicontext-tangent-fit",
        residual_max=float(residual.max().item()),
    )
