"""Packed consumer for source-bound selected TranSplat raw-head packets.

The native GaussianAdapter can operate on arbitrary leading dimensions.  This
module uses that property to convert only raw descriptors that an executor has
actually produced, together with their same-source depth and mapped-opacity
side inputs.  It deliberately rejects a zero-filled dense raw-head map: such a
map does not prove that omitted descriptors are absent from later consumers.

The first implementation is restricted to batch size one and one fixed number
of surfaces/primitives.  Those limits make the descriptor-to-decoder order
explicit while a later path establishes variable-length batch and GGU parity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch


PACKET_SCHEMA_VERSION = "saes-sparse-raw-gaussian-packet-v1"
CONSUMER_SCHEMA_VERSION = "saes-packed-gaussian-consumer-v1"


@dataclass(frozen=True)
class SparseRawGaussianPacket:
    """Selected raw descriptors plus the side inputs required by GaussianAdapter.

    ``descriptor_keys`` use ``(batch, view, pixel, surface)`` order.  Each
    ``primitive_key`` adds a Gaussian-per-pixel index and points back to one
    raw descriptor. ``coordinates`` and ``dense_slots`` preserve the native per-batch
    ``(view,pixel,surface,primitive)`` decoder order.
    """

    descriptor_keys: torch.Tensor
    raw_descriptors: torch.Tensor
    primitive_keys: torch.Tensor
    primitive_to_descriptor: torch.Tensor
    coordinates: torch.Tensor
    depths: torch.Tensor
    mapped_opacities: torch.Tensor
    dense_slots: torch.Tensor
    source_trace: Mapping[str, Any]


@dataclass(frozen=True)
class PackedGaussianAttributes:
    """Adapter results in the stable selected subset of native decoder order."""

    batch_indices: torch.Tensor
    dense_slots: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor
    source_trace: dict[str, Any]
    source_trace_sha256: str

    def as_single_batch(self, gaussians_type: type[Any]) -> Any:
        """Build a decoder-compatible B=1 Gaussian container without padding."""
        if self.batch_indices.ndim != 1 or not bool((self.batch_indices == 0).all()):
            raise ValueError("packed Gaussian output is not a single batch")
        return gaussians_type(
            means=self.means.unsqueeze(0),
            covariances=self.covariances.unsqueeze(0),
            harmonics=self.harmonics.unsqueeze(0),
            opacities=self.opacities.unsqueeze(0),
        )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("source trace must be JSON-serializable") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_tensor(
    value: Any,
    *,
    name: str,
    ndim: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} has an invalid rank")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    return value


def _strictly_lexicographic(keys: torch.Tensor) -> bool:
    if keys.shape[0] < 2:
        return True
    previous, following = keys[:-1], keys[1:]
    equal_prefix = torch.ones(previous.shape[0], dtype=torch.bool, device=keys.device)
    greater = torch.zeros_like(equal_prefix)
    for column in range(keys.shape[1]):
        greater |= equal_prefix & (following[:, column] > previous[:, column])
        equal_prefix &= following[:, column] == previous[:, column]
    return bool(greater.all())


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite values")


class PackedGaussianConsumer:
    """Convert one source-bound selected packet through the native adapter."""

    def __init__(self, adapter: Any, *, num_surfaces: int = 1, gaussians_per_pixel: int = 1):
        if not callable(adapter):
            raise TypeError("packed Gaussian consumer requires a callable adapter")
        if (
            isinstance(num_surfaces, bool)
            or not isinstance(num_surfaces, int)
            or num_surfaces <= 0
            or isinstance(gaussians_per_pixel, bool)
            or not isinstance(gaussians_per_pixel, int)
            or gaussians_per_pixel <= 0
        ):
            raise ValueError("surface and primitive counts must be positive integers")
        self._adapter = adapter
        self._num_surfaces = num_surfaces
        self._gaussians_per_pixel = gaussians_per_pixel

    def _validate_packet(
        self, packet: SparseRawGaussianPacket, *, height: int, width: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not isinstance(packet, SparseRawGaussianPacket):
            raise TypeError("packed Gaussian consumer requires SparseRawGaussianPacket")
        descriptor_keys = _require_tensor(
            packet.descriptor_keys, name="descriptor_keys", ndim=2, dtype=torch.int64
        )
        primitive_keys = _require_tensor(
            packet.primitive_keys, name="primitive_keys", ndim=2, dtype=torch.int64
        )
        raw_descriptors = _require_tensor(packet.raw_descriptors, name="raw_descriptors", ndim=2)
        primitive_to_descriptor = _require_tensor(
            packet.primitive_to_descriptor,
            name="primitive_to_descriptor",
            ndim=1,
            dtype=torch.int64,
        )
        coordinates = _require_tensor(packet.coordinates, name="coordinates", ndim=2)
        depths = _require_tensor(packet.depths, name="depths", ndim=1)
        mapped_opacities = _require_tensor(packet.mapped_opacities, name="mapped_opacities", ndim=1)
        dense_slots = _require_tensor(packet.dense_slots, name="dense_slots", ndim=1, dtype=torch.int64)
        if descriptor_keys.shape[1] != 4 or primitive_keys.shape[1] != 5:
            raise ValueError("descriptor and primitive keys have invalid widths")
        descriptor_count = descriptor_keys.shape[0]
        primitive_count = primitive_keys.shape[0]
        if (
            descriptor_count == 0
            or raw_descriptors.shape[0] != descriptor_count
            or raw_descriptors.shape[1] < 3
            or primitive_to_descriptor.shape[0] != primitive_count
            or coordinates.shape != (primitive_count, 2)
            or depths.shape[0] != primitive_count
            or mapped_opacities.shape[0] != primitive_count
            or dense_slots.shape[0] != primitive_count
        ):
            raise ValueError("packet tensor dimensions are inconsistent")
        tensors = (raw_descriptors, coordinates, depths, mapped_opacities)
        if any(value.device != raw_descriptors.device for value in tensors):
            raise ValueError("packet tensors must share one device")
        if coordinates.dtype != raw_descriptors.dtype:
            raise ValueError("packet coordinates must match raw descriptor dtype")
        _finite(raw_descriptors, "raw_descriptors")
        _finite(coordinates, "coordinates")
        _finite(depths, "depths")
        _finite(mapped_opacities, "mapped_opacities")
        if not _strictly_lexicographic(descriptor_keys):
            raise ValueError("descriptor keys must be strictly canonical")
        if not _strictly_lexicographic(primitive_keys):
            raise ValueError("primitive keys must be strictly canonical")
        if primitive_to_descriptor.min().item() < 0 or primitive_to_descriptor.max().item() >= descriptor_count:
            raise ValueError("primitive descriptor indices are out of range")
        if not torch.equal(descriptor_keys[primitive_to_descriptor], primitive_keys[:, :4]):
            raise ValueError("primitive keys do not bind their raw descriptors")
        if primitive_keys[:, 0].min().item() < 0 or primitive_keys[:, 0].max().item() != 0:
            raise ValueError("packed Gaussian consumer requires batch index zero")
        if (
            primitive_keys[:, 1].min().item() < 0
            or primitive_keys[:, 2].min().item() < 0
            or primitive_keys[:, 2].max().item() >= height * width
            or primitive_keys[:, 3].min().item() < 0
            or primitive_keys[:, 3].max().item() >= self._num_surfaces
            or primitive_keys[:, 4].min().item() < 0
            or primitive_keys[:, 4].max().item() >= self._gaussians_per_pixel
        ):
            raise ValueError("primitive keys exceed the configured layout")
        if dense_slots.numel() > 1 and not bool((dense_slots[1:] > dense_slots[:-1]).all()):
            raise ValueError("dense slots must be strictly increasing")
        expected_slots = (
            (
                primitive_keys[:, 1] * (height * width) + primitive_keys[:, 2]
            )
            * self._num_surfaces
            + primitive_keys[:, 3]
        ) * self._gaussians_per_pixel + primitive_keys[:, 4]
        if not torch.equal(dense_slots, expected_slots):
            raise ValueError("dense slots do not match canonical primitive keys")
        trace = packet.source_trace
        if not isinstance(trace, Mapping) or trace.get("source_bound") is not True:
            raise ValueError("packet lacks a source-bound producer trace")
        if trace.get("execution_scope") != "s3_raw_gaussian_head_only":
            raise ValueError("packet producer has an unsupported execution scope")
        if trace.get("adapter_side_inputs_source_bound") is not True:
            raise ValueError("packet lacks source-bound Adapter side inputs")
        if trace.get("adapter_side_inputs_same_scoped_invocation") is not True:
            raise ValueError("packet lacks same-invocation Adapter side inputs")
        invocations = trace.get("head_forward_invocations")
        if isinstance(invocations, bool) or invocations != 1:
            raise ValueError("packet requires exactly one scoped raw-head invocation")
        count = trace.get("head_final_positions_executed")
        if isinstance(count, bool) or not isinstance(count, int) or count < primitive_count:
            raise ValueError("packet producer trace has insufficient head dispatch evidence")
        return (
            descriptor_keys,
            raw_descriptors,
            primitive_keys,
            primitive_to_descriptor,
            coordinates,
            depths,
            mapped_opacities,
            dense_slots,
        )

    def convert(
        self,
        packet: SparseRawGaussianPacket,
        *,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> PackedGaussianAttributes:
        """Convert selected raw descriptor records without reading omitted slots."""
        if (
            not isinstance(image_shape, tuple)
            or len(image_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_shape)
        ):
            raise ValueError("image_shape must contain two positive integers")
        height, width = image_shape
        extrinsics = _require_tensor(extrinsics, name="extrinsics", ndim=4)
        intrinsics = _require_tensor(intrinsics, name="intrinsics", ndim=4)
        if extrinsics.shape[0] != 1 or intrinsics.shape[0] != 1:
            raise ValueError("packed Gaussian consumer currently requires batch size one")
        if extrinsics.shape[1] != intrinsics.shape[1] or extrinsics.shape[-2:] != (4, 4) or intrinsics.shape[-2:] != (3, 3):
            raise ValueError("camera tensors have incompatible shapes")
        _finite(extrinsics, "extrinsics")
        _finite(intrinsics, "intrinsics")
        (
            _descriptor_keys,
            raw_descriptors,
            primitive_keys,
            primitive_to_descriptor,
            coordinates,
            depths,
            mapped_opacities,
            dense_slots,
        ) = self._validate_packet(packet, height=height, width=width)
        if raw_descriptors.device != extrinsics.device or raw_descriptors.device != intrinsics.device:
            raise ValueError("packet and camera tensors must share one device")
        if primitive_keys[:, 1].max().item() >= extrinsics.shape[1]:
            raise ValueError("primitive view index exceeds camera views")
        raw_for_primitives = raw_descriptors[primitive_to_descriptor]
        raw_body = raw_for_primitives[:, 2:]
        adapter_input_width = getattr(self._adapter, "d_in", None)
        if isinstance(adapter_input_width, int) and raw_body.shape[1] != adapter_input_width:
            raise ValueError("raw descriptor width does not match the adapter")
        views = primitive_keys[:, 1]
        adapter_result = self._adapter(
            extrinsics[0, views],
            intrinsics[0, views],
            coordinates,
            depths.to(dtype=raw_descriptors.dtype),
            mapped_opacities.to(dtype=raw_descriptors.dtype),
            raw_body,
            image_shape,
        )
        required = ("means", "covariances", "harmonics", "opacities")
        if any(not torch.is_tensor(getattr(adapter_result, name, None)) for name in required):
            raise ValueError("adapter result does not expose Gaussian attributes")
        means = adapter_result.means
        covariances = adapter_result.covariances
        harmonics = adapter_result.harmonics
        opacities = adapter_result.opacities
        count = primitive_keys.shape[0]
        if (
            means.shape != (count, 3)
            or covariances.shape != (count, 3, 3)
            or harmonics.ndim != 3
            or harmonics.shape[:2] != (count, 3)
            or opacities.shape != (count,)
        ):
            raise ValueError("adapter result has incompatible selected attribute shapes")
        for name, value in (
            ("adapter means", means),
            ("adapter covariances", covariances),
            ("adapter harmonics", harmonics),
            ("adapter opacities", opacities),
        ):
            _finite(value, name)
        source_trace = dict(packet.source_trace)
        return PackedGaussianAttributes(
            batch_indices=primitive_keys[:, 0].clone(),
            dense_slots=dense_slots.clone(),
            means=means,
            covariances=covariances,
            harmonics=harmonics,
            opacities=opacities,
            source_trace=source_trace,
            source_trace_sha256=_canonical_sha256(source_trace),
        )
