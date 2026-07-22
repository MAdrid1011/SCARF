import inspect

import pytest


torch = pytest.importorskip("torch")


def _rotation_y(angle):
    return torch.tensor(
        (
            (torch.cos(angle), 0.0, torch.sin(angle)),
            (0.0, 1.0, 0.0),
            (-torch.sin(angle), 0.0, torch.cos(angle)),
        )
    )


def _contexts(*, duplicate=False, dtype=torch.float64):
    intrinsic = torch.tensor(
        ((1.2, 0.0, 0.5), (0.0, 1.1, 0.5), (0.0, 0.0, 1.0)), dtype=dtype
    )
    first = torch.eye(4, dtype=dtype)
    second = torch.eye(4, dtype=dtype)
    second[:3, 3] = torch.tensor((0.10, 0.0, 0.0), dtype=dtype)
    if duplicate:
        second = first.clone()
    return torch.stack((first, second)), torch.stack((intrinsic, intrinsic.clone()))


def _inputs():
    dtype = torch.float64
    output_mean = torch.tensor((0.05, 0.0, 2.0), dtype=dtype)
    local_covariance = torch.tensor(
        ((0.020, 0.0, 0.006), (0.0, 0.020, -0.004), (0.006, -0.004, 0.005)),
        dtype=dtype,
    )
    desired_covariance = torch.tensor(
        ((0.040, 0.0, 0.006), (0.0, 0.030, -0.004), (0.006, -0.004, 0.005)),
        dtype=dtype,
    )
    return {
        "output_mean": output_mean,
        "local_covariance": local_covariance,
        "contributor_means": output_mean.reshape(1, 3).repeat(2, 1),
        "contributor_covariances": desired_covariance.reshape(1, 3, 3).repeat(2, 1, 1),
        "contributor_weights": torch.tensor((0.5, 0.5), dtype=dtype),
        "desired_covariance": desired_covariance,
    }


def _call(*, contexts=None, **kwargs):
    from saes.multicontext_tangent import multicontext_tangent_covariance

    extrinsics, intrinsics = (
        _contexts(dtype=kwargs["output_mean"].dtype) if contexts is None else contexts
    )
    return multicontext_tangent_covariance(
        context_extrinsics=extrinsics,
        context_intrinsics=intrinsics,
        **kwargs,
    )


def test_multicontext_tangent_fit_matches_two_camera_projected_moments():
    from saes.multicontext_tangent import _project_world_points

    values = _inputs()
    result = _call(**{key: value for key, value in values.items() if key != "desired_covariance"})
    assert result.used_multicontext_fit
    assert result.reason == "multicontext-tangent-fit"
    assert result.residual_max is not None and result.residual_max < 1e-10
    torch.testing.assert_close(result.covariance, values["desired_covariance"], rtol=2e-8, atol=2e-9)

    extrinsics, intrinsics = _contexts()
    for extrinsic, intrinsic in zip(extrinsics, intrinsics):
        _, fitted_jacobian = _project_world_points(
            values["output_mean"].unsqueeze(0), extrinsic=extrinsic, intrinsic=intrinsic
        )
        _, desired_jacobian = _project_world_points(
            values["output_mean"].unsqueeze(0), extrinsic=extrinsic, intrinsic=intrinsic
        )
        fitted = fitted_jacobian[0] @ result.covariance @ fitted_jacobian[0].mT
        desired = desired_jacobian[0] @ values["desired_covariance"] @ desired_jacobian[0].mT
        torch.testing.assert_close(fitted, desired, rtol=3e-8, atol=3e-9)


def test_multicontext_tangent_is_context_order_and_rigid_transform_equivariant():
    values = _inputs()
    kwargs = {key: value for key, value in values.items() if key != "desired_covariance"}
    extrinsics, intrinsics = _contexts()
    original = _call(contexts=(extrinsics, intrinsics), **kwargs)
    reordered = _call(contexts=(extrinsics.flip(0), intrinsics.flip(0)), **kwargs)
    assert original.used_multicontext_fit and reordered.used_multicontext_fit
    torch.testing.assert_close(reordered.covariance, original.covariance, rtol=2e-7, atol=2e-8)

    rotation = _rotation_y(torch.tensor(-0.31, dtype=torch.float64))
    translation = torch.tensor((0.12, -0.08, 0.05), dtype=torch.float64)
    transformed_extrinsics = extrinsics.clone()
    transformed_extrinsics[:, :3, :3] = rotation @ extrinsics[:, :3, :3]
    transformed_extrinsics[:, :3, 3] = (
        extrinsics[:, :3, 3] @ rotation.mT + translation
    )
    transformed = _call(
        contexts=(transformed_extrinsics, intrinsics),
        output_mean=rotation @ values["output_mean"] + translation,
        local_covariance=rotation @ values["local_covariance"] @ rotation.mT,
        contributor_means=values["contributor_means"] @ rotation.mT + translation,
        contributor_covariances=torch.einsum(
            "ij,njk,lk->nil", rotation, values["contributor_covariances"], rotation
        ),
        contributor_weights=values["contributor_weights"],
    )
    assert transformed.used_multicontext_fit
    torch.testing.assert_close(
        transformed.covariance,
        rotation @ original.covariance @ rotation.mT,
        rtol=2e-6,
        atol=2e-7,
    )


def test_multicontext_tangent_falls_back_exactly_without_identifiable_contexts():
    values = _inputs()
    kwargs = {key: value for key, value in values.items() if key != "desired_covariance"}
    one_contexts = tuple(value[:1] for value in _contexts())
    one_context = _call(contexts=one_contexts, **kwargs)
    duplicate = _call(contexts=_contexts(duplicate=True), **kwargs)
    pose_duplicate_extrinsics, pose_duplicate_intrinsics = _contexts(dtype=torch.float64)
    pose_duplicate_extrinsics[1] = pose_duplicate_extrinsics[0]
    pose_duplicate_intrinsics[1, 0, 0] = 1.3
    pose_duplicate = _call(
        contexts=(pose_duplicate_extrinsics, pose_duplicate_intrinsics), **kwargs
    )
    invalid_extrinsics, invalid_intrinsics = _contexts(dtype=torch.float64)
    invalid_extrinsics[1, 0, 0] = 2.0
    invalid_camera = _call(contexts=(invalid_extrinsics, invalid_intrinsics), **kwargs)
    invalid_intrinsic_extrinsics, invalid_intrinsics = _contexts(dtype=torch.float64)
    invalid_intrinsics[1, 1] = torch.tensor((0.0, 0.0, 0.5), dtype=torch.float64)
    invalid_intrinsic = _call(
        contexts=(invalid_intrinsic_extrinsics, invalid_intrinsics), **kwargs
    )
    one_hot = _call(
        **{
            **kwargs,
            "contributor_weights": torch.tensor((1.0, 0.0), dtype=torch.float64),
        }
    )
    constant = _call(
        **{
            **kwargs,
            "contributor_covariances": values["local_covariance"].reshape(1, 3, 3).repeat(2, 1, 1),
        }
    )
    for result in (
        one_context,
        duplicate,
        pose_duplicate,
        invalid_camera,
        invalid_intrinsic,
        one_hot,
        constant,
    ):
        assert not result.used_multicontext_fit
        assert torch.equal(result.covariance, values["local_covariance"])
    assert pose_duplicate.reason == "duplicate-context-pose"
    assert invalid_camera.reason == "invalid-context-geometry"
    assert invalid_intrinsic.reason == "invalid-context-geometry"


def test_multicontext_tangent_is_psd_atomic_and_has_no_target_input():
    from saes.multicontext_tangent import multicontext_tangent_covariance

    values = _inputs()
    kwargs = {key: value.clone() if torch.is_tensor(value) else value for key, value in values.items() if key != "desired_covariance"}
    before = {key: value.clone() for key, value in kwargs.items() if torch.is_tensor(value)}
    result = _call(**kwargs)
    assert result.used_multicontext_fit
    assert bool(torch.isfinite(result.covariance).all())
    assert bool((torch.linalg.eigvalsh(result.covariance) >= -1e-7).all())
    for key, value in before.items():
        assert torch.equal(kwargs[key], value)

    malformed = dict(kwargs)
    malformed["contributor_means"] = malformed["contributor_means"].clone()
    malformed["contributor_means"][0, 0] = float("nan")
    failed = _call(**malformed)
    assert not failed.used_multicontext_fit
    assert failed.reason == "nonfinite-input"
    assert torch.equal(failed.covariance, kwargs["local_covariance"])

    non_psd = dict(kwargs)
    non_psd["contributor_covariances"] = non_psd["contributor_covariances"].clone()
    non_psd["contributor_covariances"][0, 0, 0] = -1.0
    rejected_covariance = _call(**non_psd)
    assert not rejected_covariance.used_multicontext_fit
    assert rejected_covariance.reason == "invalid-source-covariance"
    assert torch.equal(rejected_covariance.covariance, kwargs["local_covariance"])

    out_of_frame = dict(kwargs)
    out_of_frame["contributor_means"] = out_of_frame["contributor_means"].clone()
    out_of_frame["contributor_means"][0, 0] = 10.0
    rejected_projection = _call(**out_of_frame)
    assert not rejected_projection.used_multicontext_fit
    assert rejected_projection.reason == "invalid-context-projection"
    assert torch.equal(rejected_projection.covariance, kwargs["local_covariance"])
    assert set(inspect.signature(multicontext_tangent_covariance).parameters) == {
        "output_mean",
        "local_covariance",
        "contributor_means",
        "contributor_covariances",
        "contributor_weights",
        "context_extrinsics",
        "context_intrinsics",
    }
