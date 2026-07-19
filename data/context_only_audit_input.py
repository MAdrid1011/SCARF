"""Build and load a context-camera-only input for descriptor diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch

from data.build_manifest import build as build_dataset_manifest
from data.verify_prepared_dataset import verify_tree_manifest
from scripts.calibration_inputs import (
    load_target_free_record,
    sha256_file,
    validate_target_free_input_root,
)


INPUT_KIND = "scarf_context_only_audit_input_v1"
RECORD_KIND = "scarf_context_only_audit_record_v1"
CLASSIC_MODELS = frozenset(("transplat", "mvsplat"))
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_INPUT_FIELDS = frozenset(
    (
        "schema_version",
        "kind",
        "status",
        "paper_result_eligible",
        "model",
        "dataset",
        "source_sample_index",
        "fixed_context",
        "source_binding",
        "sidecar",
        "target_rgb_included",
        "target_camera_metadata_included",
        "target_index_included",
    )
)
_RECORD_FIELDS = frozenset(
    (
        "schema_version",
        "kind",
        "key",
        "context_indices",
        "context_cameras",
        "context_images",
    )
)
_SOURCE_BINDING_FIELDS = frozenset(
    (
        "source_audit_input_sha256",
        "source_audit_tree_sha256",
        "canonical_index_sha256",
        "canonical_sample_selection_sha256",
        "source_sidecar_tree_sha256",
    )
)
_OPTIONAL_SOURCE_BINDING_FIELDS = frozenset(("canonical_selection_sha256",))
_SIDECAR_FIELDS = frozenset(("index_sha256", "record_sha256"))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _indices(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in value
        )
        or len(value) != len(set(value))
    ):
        raise ValueError(f"context-only audit has invalid {label}")
    return list(value)


def _sample_index(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"context-only audit has an invalid {label}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"context-only audit has an invalid {label}")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _copy_context_image(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.dtype != torch.uint8:
            raise ValueError("context-only audit image must be uint8")
        return value.detach().cpu().clone()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError("context-only audit image has an unsupported type")


def _model_name(value: Any) -> str:
    if value not in CLASSIC_MODELS:
        raise ValueError("context-only audit requires a supported classic model")
    return str(value)


def _source_contract(
    source_root: Path, *, model: str
) -> tuple[dict[str, Any], list[int], str, int, str]:
    source = _load_json(source_root / "audit-input.json", "source audit input")
    source_sample_index = _sample_index(
        source.get("source_sample_index"), "source sample index"
    )
    if (
        source.get("kind") != "dl3dv_target_free_l1_primary_reference_audit_input"
        or source.get("status") != "PASS"
        or source.get("model") != model
        or source.get("dataset") != "dl3dv"
        or source.get("target_rgb_included") is not False
    ):
        raise ValueError("source audit input violates the DL3DV contract")
    selected = source.get("selected_sample")
    if not isinstance(selected, dict) or not isinstance(selected.get("scene"), str):
        raise ValueError("source audit input has no fixed scene")
    context_indices = _indices(selected.get("context_indices"), "context indices")
    target_indices = _indices(selected.get("target_indices"), "target indices")
    if len(context_indices) != 2:
        raise ValueError("context-only audit requires exactly two context views")
    if set(context_indices) & set(target_indices):
        raise ValueError("source audit context and target indices overlap")
    protocol = source.get("canonical_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("source audit input lacks a canonical index hash")
    _sha256(protocol.get("source_index_sha256"), "source canonical index hash")
    _sha256(
        protocol.get("sample_selection_sha256"),
        "source canonical sample-selection hash",
    )
    canonical_selection = source.get("canonical_selection")
    expected_selection = {
        "source_sample_index": source_sample_index,
        "scene": str(selected["scene"]),
        "context_indices": context_indices,
        "target_indices": target_indices,
    }
    if not isinstance(canonical_selection, Mapping) or dict(canonical_selection) != expected_selection:
        raise ValueError("source audit input canonical selection changed")
    return (
        source,
        context_indices,
        str(selected["scene"]),
        source_sample_index,
        _canonical_sha256(expected_selection),
    )


def prepare_context_only_audit_input(
    source_root: Path, *, output_root: Path, model: str = "transplat"
) -> dict[str, Any]:
    """Compile one source-bound sidecar to only its two context camera records.

    The source sidecar is consumed only by this preparation step. The emitted
    tree does not retain target image bytes, target indices, or target camera
    rows, so an audit loader cannot pass them into the encoder by accident.
    """
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    model = _model_name(model)
    if output_root.exists():
        raise FileExistsError(
            f"context-only audit output already exists: {output_root}"
        )
    source_tree = verify_tree_manifest(source_root, source_root / ".scarf-manifest.json")
    (
        source,
        context_indices,
        scene,
        source_sample_index,
        canonical_selection_sha256,
    ) = _source_contract(source_root, model=model)
    source_sidecar_identity = validate_target_free_input_root(
        source_root / "sidecar", "dl3dv"
    )
    dense_record = load_target_free_record(source_root / "sidecar", scene)
    if dense_record.get("context_indices") != context_indices:
        raise ValueError("source context indices do not match the fixed selection")
    cameras = torch.as_tensor(dense_record.get("cameras"), dtype=torch.float32)
    if (
        cameras.ndim != 2
        or cameras.shape[1] != 18
        or max(context_indices) >= cameras.shape[0]
    ):
        raise ValueError("source audit sidecar has invalid camera geometry")
    context_images = dense_record.get("context_images")
    if not isinstance(context_images, list) or len(context_images) != len(
        context_indices
    ):
        raise ValueError("source audit sidecar has invalid context images")

    payload = {
        "schema_version": "1.0",
        "kind": RECORD_KIND,
        "key": scene,
        "context_indices": context_indices,
        "context_cameras": cameras[context_indices].contiguous().clone(),
        "context_images": [_copy_context_image(image) for image in context_images],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    chunk_root = output_root / "sidecar" / "test"
    chunk_root.mkdir(parents=True, exist_ok=False)
    chunk_path = chunk_root / "000000.torch"
    torch.save([payload], chunk_path)
    index_path = chunk_root / "index.json"
    index_path.write_text(
        json.dumps({scene: "000000.torch"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_record = {
        "schema_version": "1.0",
        "kind": INPUT_KIND,
        "status": "PASS",
        "paper_result_eligible": False,
        "model": model,
        "dataset": "dl3dv",
        "source_sample_index": source_sample_index,
        "fixed_context": {"scene": scene, "context_indices": context_indices},
        "source_binding": {
            "source_audit_input_sha256": sha256_file(source_root / "audit-input.json"),
            "source_audit_tree_sha256": source_tree["tree_sha256"],
            "canonical_index_sha256": source["canonical_protocol"][
                "source_index_sha256"
            ],
            "canonical_sample_selection_sha256": source["canonical_protocol"].get(
                "sample_selection_sha256"
            ),
            "canonical_selection_sha256": canonical_selection_sha256,
            "source_sidecar_tree_sha256": source_sidecar_identity["tree_sha256"],
        },
        "sidecar": {
            "index_sha256": sha256_file(index_path),
            "record_sha256": sha256_file(chunk_path),
        },
        "target_rgb_included": False,
        "target_camera_metadata_included": False,
        "target_index_included": False,
    }
    audit_path = output_root / "audit-input.json"
    audit_path.write_text(
        json.dumps(audit_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = build_dataset_manifest(
        output_root,
        "dl3dv-context-only-audit",
        "context-camera-minimized fixed DL3DV audit input",
        sha256_file(source_root / "audit-input.json"),
    )
    manifest_path = output_root / ".scarf-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **audit_record,
        "tree_sha256": manifest["tree_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
    }


def _safe_chunk_path(test_root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("context-only audit index has a non-string chunk path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError("context-only audit index path is unsafe")
    path = (test_root / Path(*pure.parts)).resolve()
    if test_root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError("context-only audit chunk is unavailable")
    return path


def validate_context_only_audit_input(
    root: Path, *, model: str = "transplat"
) -> dict[str, Any]:
    """Validate a context-only input without exposing target-side fields."""
    root = Path(root).resolve()
    model = _model_name(model)
    audit = _load_json(root / "audit-input.json", "context-only audit input")
    source_sample_index = _sample_index(
        audit.get("source_sample_index"), "source sample index"
    )
    if (
        set(audit) != _INPUT_FIELDS
        or audit.get("schema_version") != "1.0"
        or audit.get("kind") != INPUT_KIND
        or audit.get("status") != "PASS"
        or audit.get("paper_result_eligible") is not False
        or audit.get("model") != model
        or audit.get("dataset") != "dl3dv"
        or audit.get("target_rgb_included") is not False
        or audit.get("target_camera_metadata_included") is not False
        or audit.get("target_index_included") is not False
    ):
        raise ValueError("context-only audit input violates its isolation contract")
    fixed = audit.get("fixed_context")
    if (
        not isinstance(fixed, dict)
        or set(fixed) != {"scene", "context_indices"}
        or not isinstance(fixed.get("scene"), str)
        or not fixed["scene"]
    ):
        raise ValueError("context-only audit input has no fixed scene")
    context_indices = _indices(fixed.get("context_indices"), "context indices")
    if len(context_indices) != 2:
        raise ValueError("context-only audit input needs two context indices")
    sidecar = audit.get("sidecar")
    if not isinstance(sidecar, dict) or set(sidecar) != _SIDECAR_FIELDS:
        raise ValueError("context-only audit input has no sidecar identity")
    for key in _SIDECAR_FIELDS:
        _sha256(sidecar.get(key), f"sidecar {key}")
    source_binding = audit.get("source_binding")
    if (
        not isinstance(source_binding, dict)
        or set(source_binding)
        not in {
            _SOURCE_BINDING_FIELDS,
            _SOURCE_BINDING_FIELDS | _OPTIONAL_SOURCE_BINDING_FIELDS,
        }
    ):
        raise ValueError("context-only audit input has an invalid source binding")
    for key in _SOURCE_BINDING_FIELDS:
        _sha256(source_binding.get(key), f"source binding {key}")
    if "canonical_selection_sha256" in source_binding:
        _sha256(
            source_binding.get("canonical_selection_sha256"),
            "source binding canonical selection",
        )
    test_root = root / "sidecar" / "test"
    index = _load_json(test_root / "index.json", "context-only audit sidecar index")
    if set(index) != {fixed["scene"]}:
        raise ValueError("context-only audit sidecar index has the wrong scene")
    chunk_path = _safe_chunk_path(test_root, index[fixed["scene"]])
    if sidecar.get("index_sha256") != sha256_file(
        test_root / "index.json"
    ) or sidecar.get("record_sha256") != sha256_file(chunk_path):
        raise ValueError("context-only audit sidecar identity mismatch")
    chunk = torch.load(chunk_path, map_location="cpu")
    if not isinstance(chunk, list) or len(chunk) != 1 or not isinstance(chunk[0], dict):
        raise ValueError("context-only audit chunk is invalid")
    payload = chunk[0]
    if (
        set(payload) != _RECORD_FIELDS
        or payload.get("schema_version") != "1.0"
        or payload.get("kind") != RECORD_KIND
        or payload.get("key") != fixed["scene"]
        or payload.get("context_indices") != context_indices
    ):
        raise ValueError("context-only audit payload does not match its fixed contract")
    cameras = payload.get("context_cameras")
    images = payload.get("context_images")
    if (
        not torch.is_tensor(cameras)
        or cameras.shape != (len(context_indices), 18)
        or not bool(torch.isfinite(cameras).all())
        or not isinstance(images, list)
        or len(images) != len(context_indices)
    ):
        raise ValueError("context-only audit payload has invalid context inputs")
    for image in images:
        _copy_context_image(image)
    tree = verify_tree_manifest(root, root / ".scarf-manifest.json")
    return {
        "scene": fixed["scene"],
        "context_indices": context_indices,
        "tree_sha256": tree["tree_sha256"],
        "manifest_sha256": tree["manifest_sha256"],
        "audit_input_sha256": sha256_file(root / "audit-input.json"),
        "source_sample_index": source_sample_index,
        "source_binding": dict(source_binding),
        "sidecar": dict(sidecar),
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
    }


def load_context_only_audit_record(root: Path) -> dict[str, Any]:
    """Return the validated, context-only record used by an audit loader."""
    identity = validate_context_only_audit_input(root)
    root = Path(root).resolve()
    chunk = torch.load(root / "sidecar" / "test" / "000000.torch", map_location="cpu")
    return {**chunk[0], "input_identity": identity}
