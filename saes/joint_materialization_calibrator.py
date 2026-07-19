"""Minimal, shared calibration for already-selected SAES representatives.

The module deliberately has no routing, renderer, dataset, or model imports.
Its caller supplies one fixed-width descriptor per already-selected Gaussian and
the selected Gaussian attributes to be corrected.  Invalid inputs do not
produce a partially corrected descriptor: ``calibrate`` raises and
``try_calibrate`` returns a failure record with no attributes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn


DESCRIPTOR_DIM = 32
BOTTLENECK_DIM = 8
JOINT_OUTPUT_DIM = 40
MAX_CALIBRATED_SH_DEGREE = 4
MAX_CALIBRATED_SH_COEFFICIENTS = (MAX_CALIBRATED_SH_DEGREE + 1) ** 2
SH_BAND_COUNT = MAX_CALIBRATED_SH_DEGREE + 1
SH_BAND_PARAMETER_COUNT = SH_BAND_COUNT * 3 * 2

BOTTLENECK_WEIGHT_COUNT = DESCRIPTOR_DIM * BOTTLENECK_DIM
JOINT_HEAD_WEIGHT_COUNT = BOTTLENECK_DIM * JOINT_OUTPUT_DIM
LINEAR_WEIGHT_COUNT = BOTTLENECK_WEIGHT_COUNT + JOINT_HEAD_WEIGHT_COUNT
LINEAR_BIAS_COUNT = BOTTLENECK_DIM + JOINT_OUTPUT_DIM
FP16_WEIGHT_BYTES = LINEAR_WEIGHT_COUNT * 2
FP16_BIAS_BYTES = LINEAR_BIAS_COUNT * 2
FP16_PARAMETER_BYTES = FP16_WEIGHT_BYTES + FP16_BIAS_BYTES
NETWORK_MACS_PER_DESCRIPTOR = LINEAR_WEIGHT_COUNT
PARAMETER_COUNT = LINEAR_WEIGHT_COUNT + LINEAR_BIAS_COUNT

ASSET_SCHEMA_VERSION = "saes-joint-materialization-calibrator-v1"
ASSET_KIND = "saes-joint-materialization-calibrator-state-dict"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CalibratedGaussianAttributes:
    """Corrected attributes for one selected representative set."""

    means: Tensor
    covariances: Tensor
    harmonics: Tensor
    opacities: Tensor


@dataclass(frozen=True)
class CalibrationFailure:
    """Fail-closed result from ``try_calibrate`` with no output attributes."""

    reason: str


def cost_contract() -> dict[str, int | bool | str]:
    """Return the fixed network-only accounting contract per representative.

    Descriptor construction and Gaussian materialization traffic are not part
    of this small network count and must be charged by their callers.
    """

    return {
        "schema_version": ASSET_SCHEMA_VERSION,
        "descriptor_dim": DESCRIPTOR_DIM,
        "bottleneck_dim": BOTTLENECK_DIM,
        "joint_output_dim": JOINT_OUTPUT_DIM,
        "shared_across_models_and_datasets": True,
        "bottleneck_weight_count": BOTTLENECK_WEIGHT_COUNT,
        "joint_head_weight_count": JOINT_HEAD_WEIGHT_COUNT,
        "linear_weight_count": LINEAR_WEIGHT_COUNT,
        "linear_bias_count": LINEAR_BIAS_COUNT,
        "parameter_count": PARAMETER_COUNT,
        "fp16_weight_bytes": FP16_WEIGHT_BYTES,
        "fp16_bias_bytes": FP16_BIAS_BYTES,
        "fp16_parameter_bytes": FP16_PARAMETER_BYTES,
        "network_macs_per_descriptor": NETWORK_MACS_PER_DESCRIPTOR,
        "network_macs_per_call": NETWORK_MACS_PER_DESCRIPTOR,
    }


def _require_nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _active_sh_bands(sh_degree: int) -> int:
    degree = _require_nonnegative_int(sh_degree, name="sh_degree")
    return min(degree, MAX_CALIBRATED_SH_DEGREE) + 1


def sh_degree_mask(
    sh_degree: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Mask gain/bias controls to the SH degree bands present at this degree.

    The fixed joint packet contains five bands (degrees 0 through 4), each
    with RGB gain and bias controls.  Higher-degree coefficients remain exact
    passthrough rather than receiving an unaccounted wider head.
    """

    active = _active_sh_bands(sh_degree)
    result = torch.zeros(
        (SH_BAND_COUNT, 3, 2),
        device=device,
        dtype=torch.float32 if dtype is None else dtype,
    )
    result[:active] = 1.0
    return result


def sha256_file(path: Path | str) -> str:
    """Return the SHA256 of one regular asset file."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"joint calibrator asset is unavailable: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrator_asset_manifest(path: Path | str) -> dict[str, int | str]:
    """Build the complete hash-pinned manifest for one frozen state asset."""

    resolved = Path(path)
    return {
        "schema_version": ASSET_SCHEMA_VERSION,
        "kind": ASSET_KIND,
        "sha256": sha256_file(resolved),
        "byte_count": resolved.stat().st_size,
        "descriptor_dim": DESCRIPTOR_DIM,
        "bottleneck_dim": BOTTLENECK_DIM,
        "joint_output_dim": JOINT_OUTPUT_DIM,
    }


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash exactly the fixed learned tensors for a runtime state binding."""

    expected_shapes = {
        "bottleneck.weight": (BOTTLENECK_DIM, DESCRIPTOR_DIM),
        "bottleneck.bias": (BOTTLENECK_DIM,),
        "joint_head.weight": (JOINT_OUTPUT_DIM, BOTTLENECK_DIM),
        "joint_head.bias": (JOINT_OUTPUT_DIM,),
    }
    if not isinstance(state_dict, Mapping) or set(state_dict) != set(expected_shapes):
        raise ValueError("joint calibrator state dictionary has an invalid schema")
    digest = hashlib.sha256()
    for name in sorted(expected_shapes):
        value = state_dict[name]
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != expected_shapes[name]
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"joint calibrator state dictionary has an invalid {name}")
        canonical = value.detach().contiguous().to(device="cpu")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(canonical.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(canonical.shape)).encode("ascii") + b"\0")
        digest.update(canonical.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_calibrator_asset(
    path: Path | str, manifest: Mapping[str, Any]
) -> dict[str, int | str]:
    """Fail closed unless an asset exactly matches this fixed architecture."""

    required = {
        "schema_version",
        "kind",
        "sha256",
        "byte_count",
        "descriptor_dim",
        "bottleneck_dim",
        "joint_output_dim",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ValueError("joint calibrator asset manifest has an invalid schema")
    if manifest["schema_version"] != ASSET_SCHEMA_VERSION:
        raise RuntimeError("joint calibrator asset has an unsupported schema version")
    if manifest["kind"] != ASSET_KIND:
        raise RuntimeError("joint calibrator asset has an unsupported kind")
    expected_sha256 = manifest["sha256"]
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("joint calibrator asset manifest has an invalid SHA256")
    byte_count = manifest["byte_count"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError("joint calibrator asset manifest has an invalid byte count")
    architecture = {
        "descriptor_dim": DESCRIPTOR_DIM,
        "bottleneck_dim": BOTTLENECK_DIM,
        "joint_output_dim": JOINT_OUTPUT_DIM,
    }
    for name, expected in architecture.items():
        if manifest[name] != expected:
            raise RuntimeError(f"joint calibrator asset changed {name}")

    resolved = Path(path)
    if not resolved.is_file() or resolved.stat().st_size != byte_count:
        raise RuntimeError("joint calibrator asset is missing or truncated")
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("joint calibrator asset SHA256 mismatch")
    return {
        "schema_version": ASSET_SCHEMA_VERSION,
        "kind": ASSET_KIND,
        "sha256": actual_sha256,
        "byte_count": byte_count,
        **architecture,
    }


def _validate_bound(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _psd_3x3(covariances: Tensor) -> bool:
    """Check symmetric 3x3 PSD inputs through their principal minors."""

    first = covariances[..., 0, 0]
    second = covariances[..., 1, 1]
    third = covariances[..., 2, 2]
    cross01 = covariances[..., 0, 1]
    cross02 = covariances[..., 0, 2]
    cross12 = covariances[..., 1, 2]
    minor01 = first * second - cross01.square()
    minor02 = first * third - cross02.square()
    minor12 = second * third - cross12.square()
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
        and (minor01 >= 0.0).all()
        and (minor02 >= 0.0).all()
        and (minor12 >= 0.0).all()
        and (determinant >= 0.0).all()
    )


class JointMaterializationCalibrator(nn.Module):
    """Shared ``32 -> 8 -> 40`` residual calibrator for selected Gaussians.

    The output layout is ``mean[3] | lower-triangular covariance[6] |
    opacity[1] | five SH degree bands * RGB(gain,bias)[30]``.  Covariance
    correction is a congruence transform, so a valid input covariance remains
    PSD without a spectral projection.
    """

    def __init__(
        self,
        *,
        mean_bound: float = 0.05,
        covariance_transform_bound: float = 0.25,
        opacity_logit_bound: float = 1.0,
        sh_gain_bound: float = 0.25,
        sh_bias_bound: float = 0.05,
    ) -> None:
        super().__init__()
        self.mean_bound = _validate_bound(mean_bound, name="mean_bound")
        self.covariance_transform_bound = _validate_bound(
            covariance_transform_bound, name="covariance_transform_bound"
        )
        if self.covariance_transform_bound >= 1.0:
            raise ValueError("covariance_transform_bound must be less than one")
        self.opacity_logit_bound = _validate_bound(
            opacity_logit_bound, name="opacity_logit_bound"
        )
        self.sh_gain_bound = _validate_bound(sh_gain_bound, name="sh_gain_bound")
        if self.sh_gain_bound >= 1.0:
            raise ValueError("sh_gain_bound must be less than one")
        self.sh_bias_bound = _validate_bound(sh_bias_bound, name="sh_bias_bound")
        self.bottleneck = nn.Linear(DESCRIPTOR_DIM, BOTTLENECK_DIM)
        self.joint_head = nn.Linear(BOTTLENECK_DIM, JOINT_OUTPUT_DIM)
        # A fresh calibrator is exactly the native materialization until its
        # one shared head has been fitted on evaluation-disjoint training data.
        nn.init.zeros_(self.joint_head.weight)
        nn.init.zeros_(self.joint_head.bias)

    @staticmethod
    def cost_contract() -> dict[str, int | bool | str]:
        """Expose the immutable network accounting without importing SAES."""

        return cost_contract()

    def state_sha256(self) -> str:
        """Return the deterministic 64-hex binding of this learned state."""

        return state_dict_sha256(self.state_dict())

    @property
    def runtime_asset_sha256(self) -> str:
        """Return the validated file binding set only by ``load_calibrator_asset``."""

        try:
            return self._runtime_asset_sha256
        except AttributeError as error:
            raise AttributeError("joint calibrator has no validated runtime asset") from error

    @property
    def runtime_asset_manifest(self) -> Mapping[str, int | str]:
        """Return the immutable validated file manifest for runtime execution."""

        try:
            return self._runtime_asset_manifest
        except AttributeError as error:
            raise AttributeError("joint calibrator has no validated runtime asset") from error

    @property
    def runtime_state_sha256(self) -> str:
        """Return the frozen-state binding checked during asset loading."""

        try:
            return self._runtime_state_sha256
        except AttributeError as error:
            raise AttributeError("joint calibrator has no validated runtime asset") from error

    def _validate_inputs(
        self,
        descriptor: Tensor,
        means: Tensor,
        covariances: Tensor,
        harmonics: Tensor,
        opacities: Tensor,
        *,
        sh_degree: int,
    ) -> tuple[int, int]:
        values = {
            "descriptor": descriptor,
            "means": means,
            "covariances": covariances,
            "harmonics": harmonics,
            "opacities": opacities,
        }
        parameter = self.bottleneck.weight
        for name, value in values.items():
            if not torch.is_tensor(value) or not value.is_floating_point():
                raise ValueError(f"joint calibrator {name} must be a floating tensor")
            if value.device != parameter.device or value.dtype != parameter.dtype:
                raise RuntimeError(f"joint calibrator {name} has a device or dtype mismatch")
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"joint calibrator {name} is non-finite")

        if descriptor.ndim != 2 or descriptor.shape[1] != DESCRIPTOR_DIM:
            raise ValueError("joint calibrator descriptor must have shape [N, 32]")
        count = int(descriptor.shape[0])
        if count < 1:
            raise ValueError("joint calibrator requires at least one selected descriptor")
        if means.shape != (count, 3):
            raise ValueError("joint calibrator means must have shape [N, 3]")
        if covariances.shape != (count, 3, 3):
            raise ValueError("joint calibrator covariances must have shape [N, 3, 3]")
        if opacities.shape not in {(count,), (count, 1)}:
            raise ValueError("joint calibrator opacities must have shape [N] or [N, 1]")
        coefficient_count = (
            _require_nonnegative_int(sh_degree, name="sh_degree") + 1
        ) ** 2
        if harmonics.shape != (count, 3, coefficient_count):
            raise ValueError("joint calibrator harmonics do not match sh_degree")
        if not bool(torch.allclose(covariances, covariances.mT, rtol=1e-5, atol=1e-7)):
            raise RuntimeError("joint calibrator covariances are not symmetric")
        if not _psd_3x3(covariances):
            raise RuntimeError("joint calibrator covariances are not PSD")
        if bool((opacities < 0.0).any()) or bool((opacities >= 1.0).any()):
            raise RuntimeError("joint calibrator opacities are outside [0, 1)")
        return count, _active_sh_bands(sh_degree)

    def calibrate(
        self,
        descriptor: Tensor,
        means: Tensor,
        covariances: Tensor,
        harmonics: Tensor,
        opacities: Tensor,
        *,
        sh_degree: int,
    ) -> CalibratedGaussianAttributes:
        """Calibrate selected attributes or raise before emitting any output."""

        count, active_sh_bands = self._validate_inputs(
            descriptor,
            means,
            covariances,
            harmonics,
            opacities,
            sh_degree=sh_degree,
        )
        joint = self.joint_head(torch.tanh(self.bottleneck(descriptor)))
        if joint.shape != (count, JOINT_OUTPUT_DIM) or not bool(torch.isfinite(joint).all()):
            raise RuntimeError("joint calibrator emitted an invalid joint residual")

        mean_delta = self.mean_bound * torch.tanh(joint[:, :3])
        triangular = self.covariance_transform_bound * torch.tanh(joint[:, 3:9])
        transform = torch.zeros(count, 3, 3, dtype=means.dtype, device=means.device)
        transform[:, 0, 0] = 1.0 + triangular[:, 0]
        transform[:, 1, 0] = triangular[:, 1]
        transform[:, 1, 1] = 1.0 + triangular[:, 2]
        transform[:, 2, 0] = triangular[:, 3]
        transform[:, 2, 1] = triangular[:, 4]
        transform[:, 2, 2] = 1.0 + triangular[:, 5]
        corrected_covariances = transform @ covariances @ transform.mT
        corrected_covariances = (
            corrected_covariances + corrected_covariances.mT
        ) * 0.5

        opacity_shape = opacities.shape
        flat_opacities = opacities.reshape(count, 1)
        opacity_delta = self.opacity_logit_bound * torch.tanh(joint[:, 9:10])
        epsilon = torch.finfo(flat_opacities.dtype).eps
        logit_input = flat_opacities.clamp(min=epsilon, max=1.0 - epsilon)
        corrected_opacities = torch.sigmoid(torch.logit(logit_input) + opacity_delta)
        # A zero-opacity representative has no visible mass to calibrate.
        # Keeping it zero also makes a zero-initialized calibrator bit-exact.
        corrected_opacities = torch.where(
            flat_opacities == 0.0, torch.zeros_like(corrected_opacities), corrected_opacities
        )
        corrected_opacities = torch.where(
            opacity_delta == 0.0, flat_opacities, corrected_opacities
        )
        corrected_opacities = corrected_opacities.reshape(opacity_shape)

        sh_controls = torch.tanh(joint[:, 10:40]).reshape(
            count, SH_BAND_COUNT, 3, 2
        )
        sh_controls = sh_controls * sh_degree_mask(
            sh_degree, device=harmonics.device, dtype=harmonics.dtype
        ).unsqueeze(0)
        harmonic_bands: list[Tensor] = []
        for degree in range(sh_degree + 1):
            start, stop = degree**2, (degree + 1) ** 2
            band = harmonics[:, :, start:stop]
            if degree < active_sh_bands:
                gain = self.sh_gain_bound * sh_controls[:, degree, :, 0].unsqueeze(-1)
                bias = self.sh_bias_bound * sh_controls[:, degree, :, 1].unsqueeze(-1)
                band = band * (1.0 + gain) + bias
            harmonic_bands.append(band)
        corrected_harmonics = torch.cat(harmonic_bands, dim=-1)
        corrected_means = means + mean_delta

        corrected = CalibratedGaussianAttributes(
            means=corrected_means,
            covariances=corrected_covariances,
            harmonics=corrected_harmonics,
            opacities=corrected_opacities,
        )
        for name, value in (
            ("means", corrected.means),
            ("covariances", corrected.covariances),
            ("harmonics", corrected.harmonics),
            ("opacities", corrected.opacities),
        ):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"joint calibrator emitted non-finite {name}")
        if not _psd_3x3(corrected.covariances):
            raise RuntimeError("joint calibrator emitted a non-PSD covariance")
        if bool((corrected.opacities < 0.0).any()) or bool(
            (corrected.opacities >= 1.0).any()
        ):
            raise RuntimeError("joint calibrator emitted an invalid opacity")
        return corrected

    def forward(
        self,
        descriptor: Tensor,
        means: Tensor,
        covariances: Tensor,
        harmonics: Tensor,
        opacities: Tensor,
        *,
        sh_degree: int,
    ) -> CalibratedGaussianAttributes:
        return self.calibrate(
            descriptor,
            means,
            covariances,
            harmonics,
            opacities,
            sh_degree=sh_degree,
        )

    def try_calibrate(
        self,
        descriptor: Tensor,
        means: Tensor,
        covariances: Tensor,
        harmonics: Tensor,
        opacities: Tensor,
        *,
        sh_degree: int,
    ) -> CalibratedGaussianAttributes | CalibrationFailure:
        """Return no attributes when validation or correction fails."""

        try:
            return self.calibrate(
                descriptor,
                means,
                covariances,
                harmonics,
                opacities,
                sh_degree=sh_degree,
            )
        except (RuntimeError, ValueError, TypeError) as error:
            return CalibrationFailure(reason=str(error))


def load_calibrator_asset(
    path: Path | str,
    manifest: Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> JointMaterializationCalibrator:
    """Load an exactly hash-pinned state dictionary into the fixed module."""

    verified_manifest = validate_calibrator_asset(path, manifest)
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    except (RuntimeError, ValueError, TypeError) as error:
        raise RuntimeError("joint calibrator asset could not be loaded safely") from error
    if not isinstance(payload, Mapping) or set(payload) != {"state_dict"}:
        raise RuntimeError("joint calibrator asset has an invalid payload schema")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("joint calibrator asset has no state dictionary")
    try:
        state_sha256 = state_dict_sha256(state_dict)
    except ValueError as error:
        raise RuntimeError("joint calibrator asset state dictionary is invalid") from error
    calibrator = JointMaterializationCalibrator()
    try:
        calibrator.load_state_dict(dict(state_dict), strict=True)
    except (RuntimeError, ValueError, TypeError) as error:
        raise RuntimeError("joint calibrator asset state dictionary is incompatible") from error
    calibrator = calibrator.to(map_location).eval()
    # Runtime paths can require this file hash and reject direct construction.
    # A mapping proxy prevents accidental mutation of the validated binding.
    calibrator._runtime_asset_sha256 = verified_manifest["sha256"]
    calibrator._runtime_asset_manifest = MappingProxyType(dict(verified_manifest))
    calibrator._runtime_state_sha256 = state_sha256
    return calibrator


__all__ = [
    "ASSET_KIND",
    "ASSET_SCHEMA_VERSION",
    "BOTTLENECK_DIM",
    "CalibrationFailure",
    "CalibratedGaussianAttributes",
    "DESCRIPTOR_DIM",
    "FP16_BIAS_BYTES",
    "FP16_PARAMETER_BYTES",
    "FP16_WEIGHT_BYTES",
    "JOINT_OUTPUT_DIM",
    "JointMaterializationCalibrator",
    "MAX_CALIBRATED_SH_COEFFICIENTS",
    "MAX_CALIBRATED_SH_DEGREE",
    "NETWORK_MACS_PER_DESCRIPTOR",
    "PARAMETER_COUNT",
    "SH_BAND_COUNT",
    "SH_BAND_PARAMETER_COUNT",
    "calibrator_asset_manifest",
    "cost_contract",
    "load_calibrator_asset",
    "sha256_file",
    "sh_degree_mask",
    "state_dict_sha256",
    "validate_calibrator_asset",
]
