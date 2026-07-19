from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def _inputs(*, count: int = 3, degree: int = 4, dtype=torch.float32):
    torch.manual_seed(7)
    descriptor = torch.randn(count, 32, dtype=dtype)
    means = torch.randn(count, 3, dtype=dtype)
    factors = torch.randn(count, 3, 3, dtype=dtype)
    covariances = factors @ factors.mT + torch.eye(3, dtype=dtype).unsqueeze(0) * 0.05
    harmonics = torch.randn(count, 3, (degree + 1) ** 2, dtype=dtype)
    opacities = torch.tensor((0.0, 0.2, 0.8), dtype=dtype)[:count]
    return descriptor, means, covariances, harmonics, opacities


def _calibrator():
    from saes.joint_materialization_calibrator import JointMaterializationCalibrator

    return JointMaterializationCalibrator()


def _psd_3x3(covariances):
    first = covariances[..., 0, 0]
    second = covariances[..., 1, 1]
    third = covariances[..., 2, 2]
    cross01 = covariances[..., 0, 1]
    cross02 = covariances[..., 0, 2]
    cross12 = covariances[..., 1, 2]
    determinant = (
        first * second * third
        + 2.0 * cross01 * cross02 * cross12
        - first * cross12.square()
        - second * cross02.square()
        - third * cross01.square()
    )
    return bool(
        (first >= 0.0).all()
        and (second >= 0.0).all()
        and (third >= 0.0).all()
        and (first * second - cross01.square() >= 0.0).all()
        and (first * third - cross02.square() >= 0.0).all()
        and (second * third - cross12.square() >= 0.0).all()
        and (determinant >= 0.0).all()
    )


def test_fixed_shared_architecture_and_cost_contract_are_exact():
    from saes.joint_materialization_calibrator import (
        BOTTLENECK_DIM,
        DESCRIPTOR_DIM,
        FP16_PARAMETER_BYTES,
        FP16_WEIGHT_BYTES,
        JOINT_OUTPUT_DIM,
        NETWORK_MACS_PER_DESCRIPTOR,
        PARAMETER_COUNT,
        JointMaterializationCalibrator,
        cost_contract,
    )

    calibrator = JointMaterializationCalibrator()
    assert calibrator.bottleneck.in_features == DESCRIPTOR_DIM == 32
    assert calibrator.bottleneck.out_features == BOTTLENECK_DIM == 8
    assert calibrator.joint_head.in_features == BOTTLENECK_DIM
    assert calibrator.joint_head.out_features == JOINT_OUTPUT_DIM == 40
    assert set(calibrator.state_dict()) == {
        "bottleneck.weight",
        "bottleneck.bias",
        "joint_head.weight",
        "joint_head.bias",
    }
    contract = cost_contract()
    assert contract == JointMaterializationCalibrator.cost_contract()
    assert contract["network_macs_per_descriptor"] == NETWORK_MACS_PER_DESCRIPTOR == 576
    assert contract["fp16_weight_bytes"] == FP16_WEIGHT_BYTES == 1152
    assert contract["fp16_parameter_bytes"] == FP16_PARAMETER_BYTES == 1248
    assert contract["parameter_count"] == PARAMETER_COUNT == 624
    assert contract["shared_across_models_and_datasets"] is True
    first_hash = calibrator.state_sha256()
    assert len(first_hash) == 64
    assert first_hash == calibrator.state_sha256()


def test_zero_initialized_joint_head_preserves_inputs_and_does_not_mutate_them():
    descriptor, means, covariances, harmonics, opacities = _inputs()
    originals = tuple(value.clone() for value in (means, covariances, harmonics, opacities))
    corrected = _calibrator()(
        descriptor, means, covariances, harmonics, opacities, sh_degree=4
    )

    for actual, expected in zip(
        (corrected.means, corrected.covariances, corrected.harmonics, corrected.opacities),
        originals,
    ):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip((means, covariances, harmonics, opacities), originals):
        assert torch.equal(actual, expected)


def test_joint_residuals_are_bounded_and_covariance_uses_a_congruence_transform():
    descriptor, means, covariances, harmonics, opacities = _inputs()
    calibrator = _calibrator()
    with torch.no_grad():
        calibrator.joint_head.bias.copy_(torch.linspace(-20.0, 20.0, 40))

    corrected = calibrator(
        descriptor, means, covariances, harmonics, opacities, sh_degree=4
    )
    assert bool(((corrected.means - means).abs() <= calibrator.mean_bound + 1e-6).all())
    assert bool((corrected.opacities >= 0.0).all())
    assert bool((corrected.opacities < 1.0).all())
    positive_opacities = opacities > 0.0
    original_logit = torch.logit(opacities[positive_opacities])
    corrected_logit = torch.logit(corrected.opacities[positive_opacities])
    assert bool(
        ((corrected_logit - original_logit).abs() <= calibrator.opacity_logit_bound + 1e-6).all()
    )
    controls = torch.tanh(calibrator.joint_head.bias[10:40]).reshape(5, 3, 2)
    assert bool((controls[:, :, 0].abs() <= 1.0).all())
    assert bool((controls[:, :, 1].abs() <= 1.0).all())
    assert bool(
        ((1.0 + calibrator.sh_gain_bound * controls[:, :, 0]) > 0.0).all()
    )
    assert bool(
        (calibrator.sh_bias_bound * controls[:, :, 1]).abs().le(
            calibrator.sh_bias_bound + 1e-6
        ).all()
    )

    raw_transform = calibrator.covariance_transform_bound * torch.tanh(
        calibrator.joint_head.bias[3:9]
    )
    transform = torch.zeros(1, 3, 3)
    transform[:, 0, 0] = 1.0 + raw_transform[0]
    transform[:, 1, 0] = raw_transform[1]
    transform[:, 1, 1] = 1.0 + raw_transform[2]
    transform[:, 2, 0] = raw_transform[3]
    transform[:, 2, 1] = raw_transform[4]
    transform[:, 2, 2] = 1.0 + raw_transform[5]
    expected_covariances = transform @ covariances @ transform.mT
    expected_covariances = (expected_covariances + expected_covariances.mT) * 0.5
    torch.testing.assert_close(corrected.covariances, expected_covariances)
    assert _psd_3x3(corrected.covariances)


def test_sh_degree_mask_preserves_unrepresented_coefficients_and_masks_missing_degrees():
    from saes.joint_materialization_calibrator import sh_degree_mask

    degree_one = sh_degree_mask(1)
    assert degree_one.shape == (5, 3, 2)
    assert bool((degree_one[:2] == 1.0).all())
    assert bool((degree_one[2:] == 0.0).all())
    assert bool((sh_degree_mask(4) == 1.0).all())

    descriptor, means, covariances, harmonics, opacities = _inputs(degree=4)
    harmonics.zero_()
    calibrator = _calibrator()
    with torch.no_grad():
        calibrator.joint_head.bias[10:40].fill_(20.0)
    corrected = calibrator(
        descriptor, means, covariances, harmonics, opacities, sh_degree=4
    )
    assert bool((corrected.harmonics > harmonics).all())


def test_degree_two_does_not_consume_higher_degree_sh_bands():
    descriptor, means, covariances, harmonics, opacities = _inputs(degree=2)
    calibrator = _calibrator()
    with torch.no_grad():
        # Bands three and four must be invisible to degree-two descriptors.
        calibrator.joint_head.bias[28:40].fill_(20.0)
    corrected = calibrator(
        descriptor, means, covariances, harmonics, opacities, sh_degree=2
    )
    torch.testing.assert_close(corrected.harmonics, harmonics)


def test_joint_packet_remains_differentiable_for_shared_calibration_training():
    descriptor, means, covariances, harmonics, opacities = _inputs()
    calibrator = _calibrator()
    with torch.no_grad():
        calibrator.joint_head.weight.normal_(mean=0.0, std=0.01)
    corrected = calibrator(
        descriptor, means, covariances, harmonics, opacities, sh_degree=4
    )
    loss = (
        corrected.means.square().mean()
        + corrected.covariances.square().mean()
        + corrected.harmonics.square().mean()
        + corrected.opacities.square().mean()
    )
    loss.backward()
    for parameter in calibrator.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


@pytest.mark.parametrize(
    "field, mutate, error",
    (
        ("descriptor", lambda value: value[:, :-1], "shape"),
        ("descriptor", lambda value: value.fill_(float("nan")), "non-finite"),
        ("covariances", lambda value: value.index_put_((torch.tensor([0]), torch.tensor([0]), torch.tensor([0])), torch.tensor([-1.0])), "PSD"),
        ("opacities", lambda value: value.fill_(1.0), "outside"),
        ("harmonics", lambda value: value[:, :, :-1], "do not match"),
    ),
)
def test_invalid_inputs_fail_closed_without_emitting_attributes(field, mutate, error):
    from saes.joint_materialization_calibrator import CalibrationFailure

    values = list(_inputs())
    names = ("descriptor", "means", "covariances", "harmonics", "opacities")
    index = names.index(field)
    values[index] = mutate(values[index].clone())
    calibrator = _calibrator()
    result = calibrator.try_calibrate(*values, sh_degree=4)
    assert isinstance(result, CalibrationFailure)
    assert error in result.reason
    with pytest.raises((RuntimeError, ValueError), match=error):
        calibrator(*values, sh_degree=4)


def test_asset_manifest_hash_validation_and_strict_state_loading(tmp_path: Path):
    from saes.joint_materialization_calibrator import (
        JointMaterializationCalibrator,
        calibrator_asset_manifest,
        load_calibrator_asset,
        validate_calibrator_asset,
    )

    path = tmp_path / "joint-calibrator.pt"
    source = JointMaterializationCalibrator()
    torch.save({"state_dict": source.state_dict()}, path)
    manifest = calibrator_asset_manifest(path)
    assert validate_calibrator_asset(path, manifest) == manifest
    loaded = load_calibrator_asset(path, manifest)
    assert loaded.training is False
    assert loaded.runtime_asset_sha256 == manifest["sha256"]
    assert dict(loaded.runtime_asset_manifest) == manifest
    assert loaded.runtime_state_sha256 == loaded.state_sha256()
    with pytest.raises(AttributeError):
        loaded.runtime_asset_sha256 = "0" * 64
    for key, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])

    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(RuntimeError, match="truncated|SHA256"):
        validate_calibrator_asset(path, manifest)


def test_asset_manifest_rejects_schema_and_architecture_drift(tmp_path: Path):
    from saes.joint_materialization_calibrator import (
        calibrator_asset_manifest,
        validate_calibrator_asset,
    )

    path = tmp_path / "joint-calibrator.pt"
    torch.save({"state_dict": _calibrator().state_dict()}, path)
    manifest = calibrator_asset_manifest(path)
    malformed = {key: value for key, value in manifest.items() if key != "kind"}
    with pytest.raises(ValueError, match="schema"):
        validate_calibrator_asset(path, malformed)
    changed = {**manifest, "descriptor_dim": 31}
    with pytest.raises(RuntimeError, match="descriptor_dim"):
        validate_calibrator_asset(path, changed)
