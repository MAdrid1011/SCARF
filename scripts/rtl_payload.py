"""Strict parser and workload contract for SCARF RTL payloads.

The payload is deliberately a small, self-describing binary container.  This
module is shared by model export, stimulus conversion, and the RTL timing
adapter so that a timing trace cannot silently use a different workload than
the model sample that produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MAGIC = b"SCARFRTL1\0"
SCHEMA = "scarf-rtl-payload-v1"
SCHEMA_V2 = "scarf-rtl-payload-v2"
WORKLOAD_SCHEMA_V2 = "scarf-rtl-workload-v2"
HEADER_PREFIX_BYTES = len(MAGIC) + 8
PAYLOAD_AXI_BASE = 0x90000000

# NumPy/PyTorch spellings used by the model exporters.  Unknown dtypes are
# rejected instead of allowing a size guess to turn into an out-of-range read.
DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
}


def infer_tensor_layout(name: str, shape: tuple[int, ...] | list[int]) -> tuple[str, str]:
    """Return the canonical axis labels and semantic role for one tensor.

    Runtime exports use the layouts emitted by the three model families.  A
    four-dimensional depth tensor is the common flattened-view form
    ``[B,D,H,W]``; keeping ``D`` explicit is essential because using an
    opaque positional fallback would make the timing adapter read ``W`` as the
    candidate count.  Unknown tensors remain auxiliary but still receive
    deterministic, unique ASCII labels.
    """
    dimensions = tuple(int(value) for value in shape)
    semantic = {
        "context_image": ("BVCHW", "context_image"),
        "target_image": ("BVCHW", "target_image"),
        "pipeline_features": ("BVCHW", "pipeline_features"),
        "fsdr_feature_vectors": ("BVHWC", "pipeline_features"),
        "features": ("BVCHW", "pipeline_features"),
        "context_features": ("BVCHW", "pipeline_features"),
        "pipeline_depths": ("BVHW", "pipeline_depths"),
        "depths": ("BVHW", "pipeline_depths"),
        "depth_candidates": ("BVDHW", "depth_candidates"),
        "fsdr_depth_candidates": ("BVDHW", "depth_candidates"),
        "depth_logits": ("BVDHW", "depth_candidates"),
        "depth_probabilities": ("BVDHW", "depth_probabilities"),
        # Final SAES decisions emitted by the same model invocation. One byte
        # encodes each physical tile: Full=0, L0=1, L1=2. The timing adapter
        # may replay the sparse schedule without reinterpreting a scalar
        # feature sample as a tile statistic.
        "saes_routes": ("VHW", "saes_routes"),
        "gaussian_means": ("BNC", "gaussian_means"),
        "gaussians": ("BNC", "gaussian_means"),
    }
    if name in semantic and len(dimensions) == len(semantic[name][0]):
        return semantic[name]
    if name in {
        "pipeline_depths",
        "depths",
    }:
        # Runtime depth heads can preserve a singleton spatial shape after
        # flattening pixels: [B,V,P,1,1].  P is deliberately explicit rather
        # than pretending the flattened pixel axis is H or W.
        if len(dimensions) == 5:
            return "BVPHW", "pipeline_depths"
    if name in {
        "depth_candidates",
        "fsdr_depth_candidates",
        "depth_logits",
        "depth_probabilities",
    }:
        # MVSplat's runtime exports one candidate vector per context view as
        # [B,V,D].  The spatial form [B,V,D,H,W] is used by the full cost
        # volume path.  Keep D explicit in both forms; treating the tensor as
        # anonymous ABC would silently bind V or W as the candidate count.
        if len(dimensions) == 5:
            role = "depth_probabilities" if name == "depth_probabilities" else "depth_candidates"
            return "BVDHW", role
        if len(dimensions) == 3:
            role = "depth_probabilities" if name == "depth_probabilities" else "depth_candidates"
            return "BVD", role
        if len(dimensions) == 2:
            role = "depth_probabilities" if name == "depth_probabilities" else "depth_candidates"
            return "VD", role
        if len(dimensions) == 4:
            role = "depth_probabilities" if name == "depth_probabilities" else "depth_candidates"
            return "BDHW", role
        if len(dimensions) == 1:
            role = "depth_probabilities" if name == "depth_probabilities" else "depth_candidates"
            return "D", role
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(dimensions) > len(alphabet):
        raise ValueError(f"payload tensor {name} has too many dimensions")
    return alphabet[: len(dimensions)], "auxiliary"


@dataclass(frozen=True)
class PayloadTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset_bytes: int
    size_bytes: int

    @property
    def end_bytes(self) -> int:
        return self.offset_bytes + self.size_bytes


@dataclass(frozen=True)
class ParsedPayload:
    path: Path
    header: Mapping[str, Any]
    tensors: tuple[PayloadTensor, ...]
    data_offset_bytes: int
    payload_size_bytes: int
    data_sha256: str

    def tensor(self, name: str) -> PayloadTensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(f"payload tensor is missing: {name}")

    def read_tensor(self, name: str) -> bytes:
        tensor = self.tensor(name)
        with self.path.open("rb") as stream:
            stream.seek(self.data_offset_bytes + tensor.offset_bytes)
            data = stream.read(tensor.size_bytes)
        if len(data) != tensor.size_bytes:
            raise ValueError(f"payload tensor is truncated: {name}")
        return data


CLAIM_CRITICAL_ROLES = (
    "pipeline_features",
    "pipeline_depths",
    "depth_candidates",
    "depth_probabilities",
)


def _c_order_strides(shape: tuple[int, ...], element_bytes: int) -> list[int]:
    """Return byte strides for the packed, contiguous C-order tensor."""
    stride = element_bytes
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return list(reversed(result))


def rtl_tensor_bindings(payload: ParsedPayload) -> dict[str, dict[str, Any]]:
    """Return the canonical data-section-relative RTL binding for each role.

    The payload header is deliberately absent from this interface.  Consumers
    program offsets relative to the first packed tensor byte at
    :data:`PAYLOAD_AXI_BASE`, so an AXI request can never interpret JSON
    metadata as an FP tensor.
    """
    if payload.header.get("schema_version") != SCHEMA_V2:
        raise ValueError("claim payload requires scarf-rtl-payload-v2 bindings")
    result: dict[str, dict[str, Any]] = {}
    for tensor in payload.tensors:
        entry = next(
            item for item in payload.header["tensors"] if item["name"] == tensor.name
        )
        role = entry["role"]
        if role not in CLAIM_CRITICAL_ROLES:
            continue
        if role in result:
            raise ValueError(f"claim payload has multiple tensors for role {role}")
        element_bytes = DTYPE_BYTES[tensor.dtype]
        result[role] = {
            "name": tensor.name,
            "role": role,
            "data_offset_bytes": tensor.offset_bytes,
            "size_bytes": tensor.size_bytes,
            "dtype": tensor.dtype,
            "element_bytes": element_bytes,
            "shape": list(tensor.shape),
            "layout": entry["layout"],
            "strides_bytes": _c_order_strides(tensor.shape, element_bytes),
        }
    return result


def _float_values(payload: ParsedPayload, binding: Mapping[str, Any]) -> list[float]:
    """Decode a critical floating-point tensor for bounded claim validation."""
    dtype = binding["dtype"]
    name = binding["name"]
    raw = payload.read_tensor(name)
    count = math.prod(binding["shape"])
    formats = {"float16": "e", "float32": "f", "float64": "d"}
    try:
        fmt = formats[dtype]
    except KeyError as exc:
        raise ValueError(f"claim-critical tensor {name} must use a supported floating dtype") from exc
    return list(struct.unpack(f"<{count}{fmt}", raw))


def constant_claim_roles(payload: ParsedPayload) -> tuple[str, ...]:
    """Return critical roles whose finite values contain no variation.

    Constant payloads are useful directed RTL diagnostics. Callers use this
    classification to keep their timing traces out of claim exports while
    still running the payload through the exact AXI/RTL path.
    """
    bindings = rtl_tensor_bindings(payload)
    return tuple(
        role for role in CLAIM_CRITICAL_ROLES
        if role in bindings
        and (values := _float_values(payload, bindings[role]))
        and min(values) == max(values)
    )


def validate_claim_payload(payload: ParsedPayload) -> dict[str, dict[str, Any]]:
    """Validate that a v2 payload has finite, bindable mechanism inputs.

    This intentionally avoids a per-tensor hash chain.  Constant tensors are
    valid RTL stimuli: the FSDR datapath observes their lack of feature
    variation and takes its conservative Full route.  The timing harness
    records one ordered consumption digest for the bytes it actually reads.
    """
    bindings = rtl_tensor_bindings(payload)
    missing = [role for role in CLAIM_CRITICAL_ROLES if role not in bindings]
    if missing:
        raise ValueError("claim payload is missing critical role(s): " + ", ".join(missing))
    for role in CLAIM_CRITICAL_ROLES:
        binding = bindings[role]
        values = _float_values(payload, binding)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"claim-critical tensor {role} contains non-finite values")
        if not values:
            raise ValueError(f"claim-critical tensor {role} is empty")
        if role in {"pipeline_depths", "depth_candidates", "depth_probabilities"}:
            if min(values) < 0.0:
                raise ValueError(f"claim-critical tensor {role} contains negative depth/probability values")
    return bindings


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _shape(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = tuple(_integer(item, f"{label}[{index}]", minimum=1) for index, item in enumerate(value))
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def parse_payload(path: Path, *, require_target_free: bool = True) -> ParsedPayload:
    """Parse and fully validate one packed payload.

    The old writer used ``17 + header_length`` even though the prefix is 18
    bytes.  The parser intentionally requires the corrected value and rejects
    that ambiguous legacy layout.
    """

    payload_path = Path(path).resolve()
    if not payload_path.is_file() or payload_path.is_symlink():
        raise ValueError(f"payload must be a regular file: {path}")
    file_size = payload_path.stat().st_size
    if file_size < HEADER_PREFIX_BYTES:
        raise ValueError("payload is shorter than its binary header")
    with payload_path.open("rb") as stream:
        prefix = stream.read(HEADER_PREFIX_BYTES)
        if prefix[: len(MAGIC)] != MAGIC:
            raise ValueError("payload magic is invalid")
        header_length = struct.unpack("<Q", prefix[len(MAGIC) :])[0]
        if header_length <= 0 or header_length > file_size - HEADER_PREFIX_BYTES:
            raise ValueError("payload header length is invalid")
        encoded = stream.read(header_length)
    try:
        header = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("payload header is not valid UTF-8 JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("payload header must be an object")
    schema = header.get("schema_version")
    required = {"schema_version", "target_free", "byte_order", "data_offset_bytes", "tensor_count", "tensors"}
    if schema == SCHEMA:
        if set(header) != required:
            raise ValueError("payload header has an invalid field set")
    elif schema == SCHEMA_V2:
        if set(header) != required | {"workload"}:
            raise ValueError("payload v2 header has an invalid field set")
        _validate_v2_workload_header(header["workload"], header["tensors"])
    else:
        raise ValueError("payload schema version is unsupported")
    if schema not in {SCHEMA, SCHEMA_V2}:
        raise ValueError("payload schema version is unsupported")
    if require_target_free and header["target_free"] is not True:
        raise ValueError("RTL claim payload must be marked target_free")
    if header["byte_order"] != "little-endian":
        raise ValueError("payload byte order must be little-endian")
    expected_data_offset = HEADER_PREFIX_BYTES + header_length
    data_offset = _integer(header["data_offset_bytes"], "data_offset_bytes", minimum=expected_data_offset)
    if data_offset != expected_data_offset:
        raise ValueError(
            f"payload data_offset_bytes must equal {expected_data_offset}, got {data_offset}"
        )
    entries = header["tensors"]
    count = _integer(header["tensor_count"], "tensor_count", minimum=1)
    if not isinstance(entries, list) or len(entries) != count:
        raise ValueError("payload tensor_count does not match tensors")
    tensors: list[PayloadTensor] = []
    names: set[str] = set()
    expected_offset = 0
    for index, entry in enumerate(entries):
        expected_fields = {"name", "dtype", "shape", "offset_bytes", "size_bytes"}
        if schema == SCHEMA_V2:
            expected_fields |= {"layout", "role"}
        if not isinstance(entry, dict) or set(entry) != expected_fields:
            raise ValueError(f"payload tensor entry {index} has an invalid field set")
        name = entry["name"]
        dtype = entry["dtype"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"payload tensor entry {index} has a duplicate/invalid name")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise ValueError(f"payload tensor {name} has unsupported dtype: {dtype!r}")
        shape = _shape(entry["shape"], f"payload tensor {name}.shape")
        offset = _integer(entry["offset_bytes"], f"payload tensor {name}.offset_bytes")
        size = _integer(entry["size_bytes"], f"payload tensor {name}.size_bytes", minimum=1)
        elements = math.prod(shape) if shape else 1
        expected_size = elements * DTYPE_BYTES[dtype]
        if size != expected_size:
            raise ValueError(f"payload tensor {name} size does not match dtype and shape")
        if schema == SCHEMA_V2:
            _validate_layout(entry["layout"], shape, f"payload tensor {name}.layout")
            if not isinstance(entry["role"], str) or not entry["role"]:
                raise ValueError(f"payload tensor {name}.role is invalid")
            binding = header["workload"]["tensor_bindings"].get(name)
            if binding.get("layout") != entry["layout"] or binding.get("role") != entry["role"]:
                raise ValueError(f"payload tensor {name} disagrees with workload binding")
        if offset != expected_offset:
            raise ValueError(f"payload tensor {name} has a gap or overlap in packed data")
        if data_offset + offset + size > file_size:
            raise ValueError(f"payload tensor {name} extends beyond the file")
        names.add(name)
        tensors.append(PayloadTensor(name, dtype, shape, offset, size))
        expected_offset += size
    if data_offset + expected_offset != file_size:
        raise ValueError("payload contains trailing or missing bytes")
    with payload_path.open("rb") as stream:
        stream.seek(data_offset)
        data_digest = hashlib.sha256(stream.read(expected_offset)).hexdigest()
    return ParsedPayload(payload_path, header, tuple(tensors), data_offset, file_size, data_digest)


def _find_shape(payload: ParsedPayload, names: tuple[str, ...]) -> tuple[int, ...] | None:
    for name in names:
        try:
            return payload.tensor(name).shape
        except KeyError:
            continue
    return None


def _validate_layout(value: Any, shape: tuple[int, ...], label: str) -> str:
    if not isinstance(value, str) or not value or len(value) != len(shape):
        raise ValueError(f"{label} must contain one axis label per dimension")
    # Semantic tensors use conventional labels (B/V/C/H/W, D, N, ...),
    # while auxiliary tensors may need any otherwise-unused axis letter.  The
    # schema only requires printable, unique ASCII axis names; semantic
    # extraction below explicitly asks for the labels it understands.
    if any(axis not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for axis in value):
        raise ValueError(f"{label} contains an unknown axis")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicate axes")
    return value


def _validate_v2_workload_header(value: Any, tensors: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("payload v2 workload must be an object")
    required = {
        "schema_version", "image_h", "image_w", "feature_dim",
        "num_depth_candidates", "num_gaussians", "view_count",
        "primitives_per_pixel", "tensor_bindings",
    }
    optional = {"model", "dataset", "rtl_config"}
    if set(value) - required - optional or not required.issubset(value):
        raise ValueError("payload v2 workload has an invalid field set")
    if value["schema_version"] != WORKLOAD_SCHEMA_V2:
        raise ValueError("payload v2 workload schema version is invalid")
    for field in required - {"schema_version", "tensor_bindings"}:
        _integer(value.get(field), f"workload.{field}", minimum=1)
    bindings = value["tensor_bindings"]
    if not isinstance(bindings, dict):
        raise ValueError("workload.tensor_bindings must be an object")
    if not isinstance(tensors, list) or set(bindings) != {
        item.get("name") for item in tensors if isinstance(item, dict)
    }:
        raise ValueError("workload tensor bindings do not match tensors")
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"layout", "role"}:
            raise ValueError(f"workload tensor binding is invalid: {name}")
        if not isinstance(binding["role"], str) or not binding["role"]:
            raise ValueError(f"workload tensor role is invalid: {name}")
    rtl_config = value.get("rtl_config")
    if rtl_config is not None:
        if not isinstance(rtl_config, dict):
            raise ValueError("workload.rtl_config must be an object")
        allowed = {
            "tile_size", "cnn_layers", "transformer_layers", "norm_groups",
            "sh_degree", "has_dinov2", "saes_feature_var", "saes_cross_check",
            "saes_depth_std", "fsdr_cache_size", "fsdr_hamming_threshold",
            "fsdr_depth_valid_threshold", "dinov2_layers",
        }
        required = allowed - {"dinov2_layers"}
        if not required.issubset(rtl_config) or not set(rtl_config).issubset(allowed):
            raise ValueError("workload.rtl_config has an invalid field set")
        for field, number in rtl_config.items():
            if field == "has_dinov2":
                if not isinstance(number, bool):
                    raise ValueError("workload.rtl_config.has_dinov2 must be boolean")
            elif _integer(number, f"workload.rtl_config.{field}", minimum=0) < 0:
                raise ValueError(f"workload.rtl_config.{field} must be nonnegative")


def _image_shape(payload: ParsedPayload) -> tuple[int, int]:
    shape = _find_shape(payload, ("context_image", "target_image"))
    if shape is None or len(shape) < 2:
        raise ValueError("payload must contain context_image with spatial dimensions")
    # Images are exported as BCHW or BVCHW.  The final two dimensions are
    # unambiguous for either convention.
    height, width = shape[-2:]
    return height, width


def _explicit_axis(payload: ParsedPayload, names: tuple[str, ...], axis: str) -> int:
    header = payload.header.get("workload")
    if not isinstance(header, Mapping):
        raise ValueError("payload has no explicit workload metadata")
    bindings = header.get("tensor_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("payload workload has no tensor bindings")
    for name in names:
        try:
            tensor = payload.tensor(name)
        except KeyError:
            continue
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"payload tensor {name} has no layout binding")
        layout = binding.get("layout")
        _validate_layout(layout, tensor.shape, f"payload tensor {name}.layout")
        if axis not in layout:
            raise ValueError(f"payload tensor {name} layout has no {axis} axis")
        return tensor.shape[layout.index(axis)]
    raise ValueError(f"payload has no tensor for semantic axis {axis}")


def _feature_dim(payload: ParsedPayload) -> int:
    shape = _find_shape(payload, ("pipeline_features", "fsdr_feature_vectors", "features", "context_features"))
    if shape is None:
        raise ValueError("payload must contain pipeline_features")
    if len(shape) >= 3:
        # BVCHW/BCHW use the channel axis before the spatial axes.  For a
        # flattened [B,N,C] export, the final axis is the feature dimension.
        if len(shape) >= 4:
            return int(shape[-3])
        return int(shape[-1])
    raise ValueError("pipeline_features has no feature dimension")


def _depth_candidates(payload: ParsedPayload) -> int:
    shape = _find_shape(payload, ("depth_candidates", "fsdr_depth_candidates", "depth_logits"))
    if shape is None:
        raise ValueError("payload must contain depth_candidates")
    if not shape:
        raise ValueError("depth_candidates has no candidate dimension")
    # Candidate vectors are exported as [D] or [B,D].  The final axis is the
    # candidate domain in both forms; continuous per-pixel depths are not a
    # substitute for the registered candidate count.
    return int(shape[-1])


def _validate_probability_candidates(payload: ParsedPayload, candidate_count: int) -> None:
    """Ensure an exported probability volume uses the same candidate domain."""

    try:
        probability = payload.tensor("depth_probabilities")
    except KeyError:
        return
    binding = payload.header.get("workload", {}).get("tensor_bindings", {}).get(
        "depth_probabilities"
    )
    if isinstance(binding, Mapping) and isinstance(binding.get("layout"), str):
        layout = binding["layout"]
        if "D" not in layout:
            raise ValueError("depth_probabilities layout has no D axis")
        probability_count = probability.shape[layout.index("D")]
    elif len(probability.shape) >= 3:
        # Legacy v1 exports have no binding; the runtime probability volume is
        # conventionally [B,V,D,H,W], while a flattened [B,D,H,W] volume has
        # D at the second-to-last spatial position.
        probability_count = (
            probability.shape[-3] if len(probability.shape) >= 5 else probability.shape[1]
        )
    else:
        probability_count = probability.shape[-1]
    if probability_count != candidate_count:
        raise ValueError(
            "depth_probabilities candidate axis disagrees with depth_candidates: "
            f"{probability_count} != {candidate_count}"
        )


def _views(payload: ParsedPayload) -> int:
    shape = _find_shape(payload, ("context_image", "pipeline_features"))
    if shape is None:
        return 1
    return int(shape[1]) if len(shape) >= 5 else 1


def _gaussians(payload: ParsedPayload) -> int:
    shape = _find_shape(payload, ("gaussian_means", "gaussians"))
    if shape is None or len(shape) < 2:
        raise ValueError("payload must contain gaussian_means with a Gaussian axis")
    # [B,N,3] and [V,N,3] are the exported layouts.
    return int(shape[-2])


def workload_descriptor(payload: ParsedPayload) -> dict[str, Any]:
    """Return the canonical descriptor consumed by the RTL adapter."""

    explicit = payload.header.get("schema_version") == SCHEMA_V2
    if explicit:
        workload = payload.header["workload"]
        image_h = _explicit_axis(payload, ("context_image", "target_image"), "H")
        image_w = _explicit_axis(payload, ("context_image", "target_image"), "W")
        feature_dim = _explicit_axis(
            payload, ("pipeline_features", "fsdr_feature_vectors", "features", "context_features"), "C"
        )
        num_depth_candidates = _explicit_axis(
            payload, ("depth_candidates", "fsdr_depth_candidates", "depth_logits"), "D"
        )
        _validate_probability_candidates(payload, num_depth_candidates)
        num_gaussians = _explicit_axis(payload, ("gaussian_means", "gaussians"), "N")
        try:
            view_count = _explicit_axis(payload, ("context_image", "pipeline_features"), "V")
        except ValueError as exc:
            if "has no V axis" not in str(exc):
                raise
            view_count = 1
        primitives_per_pixel = _integer(
            workload.get("primitives_per_pixel"), "workload.primitives_per_pixel", minimum=1
        )
        derived = {
            "image_h": image_h,
            "image_w": image_w,
            "feature_dim": feature_dim,
            "num_depth_candidates": num_depth_candidates,
            "num_gaussians": num_gaussians,
            "view_count": view_count,
        }
        for field, value in derived.items():
            if workload.get(field) != value:
                raise ValueError(
                    f"explicit workload {field} does not match tensor layout: "
                    f"{workload.get(field)!r} != {value!r}"
                )
        expected_primitives = num_gaussians // (view_count * image_h * image_w)
        if expected_primitives <= 0 or expected_primitives != primitives_per_pixel:
            raise ValueError("explicit workload primitives_per_pixel is inconsistent")
    else:
        image_h, image_w = _image_shape(payload)
        feature_dim = _feature_dim(payload)
        num_depth_candidates = _depth_candidates(payload)
        _validate_probability_candidates(payload, num_depth_candidates)
        num_gaussians = _gaussians(payload)
        view_count = _views(payload)
        primitives_per_pixel = num_gaussians // (view_count * image_h * image_w)
    if image_h <= 0 or image_w <= 0 or feature_dim <= 0 or num_depth_candidates <= 0 or num_gaussians <= 0:
        raise ValueError("payload workload dimensions must be positive")
    tensors = [
        {
            "name": tensor.name,
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "offset_bytes": tensor.offset_bytes,
            "size_bytes": tensor.size_bytes,
            "end_bytes": tensor.end_bytes,
        }
        for tensor in payload.tensors
    ]
    if explicit:
        bindings = payload.header["workload"]["tensor_bindings"]
        for tensor in tensors:
            binding = bindings[tensor["name"]]
            tensor["layout"] = binding["layout"]
            tensor["role"] = binding["role"]
    descriptor: dict[str, Any] = {
        "schema_version": "scarf-rtl-workload-v1",
        "image_h": image_h,
        "image_w": image_w,
        "feature_dim": feature_dim,
        "num_depth_candidates": num_depth_candidates,
        "num_gaussians": num_gaussians,
        "view_count": view_count,
        "primitives_per_pixel": primitives_per_pixel,
        "tensor_count": len(payload.tensors),
        "payload_size_bytes": payload.payload_size_bytes,
        "data_offset_bytes": payload.data_offset_bytes,
        "tensor_data_size_bytes": sum(tensor.size_bytes for tensor in payload.tensors),
        "tensor_data_sha256": payload.data_sha256,
        "tensor_ranges": tensors,
    }
    if explicit:
        for field in ("model", "dataset"):
            if field in payload.header["workload"]:
                descriptor[field] = payload.header["workload"][field]
        descriptor["schema_version"] = WORKLOAD_SCHEMA_V2
        if "rtl_config" in payload.header["workload"]:
            descriptor["rtl_config"] = payload.header["workload"]["rtl_config"]
    descriptor["descriptor_sha256"] = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    return descriptor


def descriptor_sha256(descriptor: Mapping[str, Any]) -> str:
    value = dict(descriptor)
    value.pop("descriptor_sha256", None)
    return hashlib.sha256(_canonical_json(value)).hexdigest()
