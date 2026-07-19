"""Source-bound contracts for the native DepthSplat DL3DV backend.

DepthSplat does not share the classic raw-Gaussian layout.  Its color branch
emits ``opacity_logit, offset_xy, adapter_body`` through a regressor followed
by a replicate-padded Gaussian head, and its Adapter consumes source RGB to
initialize SH.  This module deliberately keeps that identity and coordinate
contract separate from :mod:`saes.classic_backend`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


FROZEN_DEPTHSPLAT_BACKEND_IDENTITY_SCHEMA_VERSION = (
    "depthsplat-backend-frozen-identity-v1"
)
_GIT_HEAD_PATTERN = re.compile(r"[0-9a-f]{40}")

_SOURCE_FILES = {
    "encoder": "src/model/encoder/encoder_depthsplat.py",
    "depth_predictor": "src/model/encoder/unimatch/mv_unimatch.py",
    "gaussian_adapter": "src/model/encoder/common/gaussian_adapter.py",
    "projection": "src/geometry/projection.py",
    "sh_rotation": "src/misc/sh_rotation.py",
    "decoder": "src/model/decoder/decoder_splatting_cuda.py",
    "gaussian_types": "src/model/types.py",
    "encoder_config": "config/model/encoder/depthsplat.yaml",
    "experiment_config": "config/experiment/dl3dv.yaml",
}
_RUNTIME_IMPORT_ROOTS = (
    "depthsplat/src",
    "assets/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
)
_SIMULATOR_SOURCE_FILES = (
    "integration/model_loader.py",
    "saes/depthsplat_backend.py",
    "saes/depthsplat_selected_output.py",
    "saes/selected_output_replay.py",
    "scripts/saes_depthsplat_selected_output_audit.py",
)


@dataclass(frozen=True)
class DepthSplatBackendContract:
    """Pinned native boundaries for DepthSplat's DL3DV application route."""

    model: str
    dataset: str
    experiment: str
    checkpoint: Path
    evaluation_index: Path
    environment_profile: str
    gaussian_regressor_module: str
    gaussian_head_module: str
    gaussian_adapter_module: str
    decoder_module: str
    raw_descriptor_layout: str
    coordinate_semantics: str


def resolve_depthsplat_backend_contract(root: Path) -> DepthSplatBackendContract:
    """Resolve only the published native DepthSplat/DL3DV route."""

    from scripts.ae_config import resolve_claim_selection, resolve_experiment

    root = Path(root).resolve()
    experiment = resolve_experiment("depthsplat", "dl3dv", root)
    selection = resolve_claim_selection("depthsplat", "dl3dv", root)
    checkpoint = experiment.checkpoint.resolve()
    model_root = root / "depthsplat"
    if (
        experiment.model != "depthsplat"
        or experiment.dataset != "dl3dv"
        or experiment.experiment != "dl3dv"
        or experiment.environment_profile != "depthsplat"
        or checkpoint != (model_root / "checkpoints" / "dl3dv.ckpt").resolve()
        or selection.index_path
        != (model_root / "assets" / "dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json").resolve()
        or not checkpoint.is_file()
        or not selection.index_path.is_file()
    ):
        raise RuntimeError("DepthSplat DL3DV application identity changed")
    return DepthSplatBackendContract(
        model="depthsplat",
        dataset="dl3dv",
        experiment="dl3dv",
        checkpoint=checkpoint,
        evaluation_index=selection.index_path,
        environment_profile="depthsplat",
        gaussian_regressor_module="encoder.gaussian_regressor",
        gaussian_head_module="encoder.gaussian_head",
        gaussian_adapter_module="encoder.gaussian_adapter",
        decoder_module="decoder",
        raw_descriptor_layout="opacity-logit-offset-xy-adapter-body-v1",
        coordinate_semantics="depthsplat-z-depth-pixel-center-plus-sigmoid-offset-rgb-sh-v1",
    )


def _validated_contract_root(contract: DepthSplatBackendContract) -> Path:
    if not isinstance(contract, DepthSplatBackendContract):
        raise TypeError("DepthSplat identity requires a DepthSplatBackendContract")
    checkpoint = Path(contract.checkpoint).resolve()
    model_root = checkpoint.parent.parent
    if (
        contract.model != "depthsplat"
        or contract.dataset != "dl3dv"
        or contract.experiment != "dl3dv"
        or contract.environment_profile != "depthsplat"
        or checkpoint.name != "dl3dv.ckpt"
        or checkpoint.parent.name != "checkpoints"
        or model_root.name != "depthsplat"
        or not checkpoint.is_file()
        or Path(contract.evaluation_index).resolve()
        != (model_root / "assets" / "dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json").resolve()
        or not Path(contract.evaluation_index).is_file()
        or (
            contract.gaussian_regressor_module,
            contract.gaussian_head_module,
            contract.gaussian_adapter_module,
            contract.decoder_module,
            contract.raw_descriptor_layout,
            contract.coordinate_semantics,
        )
        != (
            "encoder.gaussian_regressor",
            "encoder.gaussian_head",
            "encoder.gaussian_adapter",
            "decoder",
            "opacity-logit-offset-xy-adapter-body-v1",
            "depthsplat-z-depth-pixel-center-plus-sigmoid-offset-rgb-sh-v1",
        )
    ):
        raise ValueError("DepthSplat identity contract fields changed")
    return model_root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"DepthSplat identity cannot read {path.name}") from error
    return digest.hexdigest()


def _git_output(model_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(model_root), *arguments),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError("DepthSplat identity cannot execute git") from error
    if completed.returncode != 0:
        raise RuntimeError("DepthSplat identity cannot resolve submodule git state")
    return completed.stdout.strip()


def _source_file_identity(
    model_root: Path, *, relative_path: str, label: str
) -> dict[str, str]:
    source_path = (model_root / relative_path).resolve()
    if model_root.resolve() not in source_path.parents or not source_path.is_file():
        raise ValueError(f"DepthSplat identity source file is invalid: {label}")
    tracked = _git_output(model_root, "ls-files", "--error-unmatch", "--", relative_path)
    if tracked != relative_path:
        raise RuntimeError(f"DepthSplat identity source file is not tracked: {label}")
    return {
        "path": f"{model_root.name}/{relative_path}",
        "sha256": _sha256_file(source_path),
    }


def _selection_identity(contract: DepthSplatBackendContract) -> dict[str, Any]:
    from scripts.ae_config import resolve_claim_selection

    root = Path(contract.checkpoint).resolve().parents[2]
    selection = resolve_claim_selection("depthsplat", "dl3dv", root)
    if selection.index_path.resolve() != Path(contract.evaluation_index).resolve():
        raise RuntimeError("DepthSplat evaluation selection changed")
    return {
        "path": selection.index_path.relative_to(root).as_posix(),
        "sha256": _sha256_file(selection.index_path),
        "source_index_sha256": selection.source_index_sha256,
        "sample_selection_sha256": selection.sample_selection_sha256,
        "sample_count": selection.sample_count,
    }


def _runtime_import_tree_identity(root: Path, relative_root: str) -> dict[str, Any]:
    """Hash one executable import tree while excluding interpreter cache files."""

    source_root = (root / relative_root).resolve()
    if root.resolve() not in source_root.parents or not source_root.is_dir():
        raise ValueError("DepthSplat runtime import root is invalid")
    files: list[dict[str, str]] = []
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        relative_parts = path.relative_to(source_root).parts
        if (
            "__pycache__" in relative_parts
            or ".git" in relative_parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("DepthSplat runtime import tree contains an unsafe entry")
        resolved = path.resolve()
        if source_root not in resolved.parents:
            raise ValueError("DepthSplat runtime import entry escapes its root")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ValueError("DepthSplat runtime import tree has no files")
    return {
        "path": relative_root,
        "tree_sha256": canonical_json_sha256(files),
        "file_count": len(files),
    }


def _runtime_source_identity(contract: DepthSplatBackendContract) -> dict[str, Any]:
    model_root = Path(contract.checkpoint).resolve().parents[1]
    root = model_root.parent
    import_roots = [
        _runtime_import_tree_identity(root, relative_root)
        for relative_root in _RUNTIME_IMPORT_ROOTS
    ]
    src = import_roots[0]
    return {
        "repository_path": "depthsplat",
        "repository_commit": _git_output(model_root, "rev-parse", "--verify", "HEAD"),
        "src_path": "depthsplat/src",
        "src_tree_sha256": src["tree_sha256"],
        "src_file_count": src["file_count"],
        "runtime_import_roots": import_roots,
    }


def _simulator_source_file_identity(root: Path, relative_path: str) -> dict[str, str]:
    path = (root / relative_path).resolve()
    if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError("DepthSplat simulator source file is invalid")
    return {"path": relative_path, "sha256": _sha256_file(path)}


def freeze_depthsplat_backend_identity(
    contract: DepthSplatBackendContract,
) -> dict[str, Any]:
    """Freeze the checkpoint, native source files, and selected DL3DV index."""

    model_root = _validated_contract_root(contract)
    repository_root = model_root.parent
    submodule_head = _git_output(model_root, "rev-parse", "--verify", "HEAD")
    if not _GIT_HEAD_PATTERN.fullmatch(submodule_head):
        raise RuntimeError("DepthSplat identity submodule HEAD is invalid")
    return {
        "schema_version": FROZEN_DEPTHSPLAT_BACKEND_IDENTITY_SCHEMA_VERSION,
        "model": contract.model,
        "dataset": contract.dataset,
        "experiment": contract.experiment,
        "environment_profile": contract.environment_profile,
        "gaussian_regressor_module": contract.gaussian_regressor_module,
        "gaussian_head_module": contract.gaussian_head_module,
        "gaussian_adapter_module": contract.gaussian_adapter_module,
        "decoder_module": contract.decoder_module,
        "raw_descriptor_layout": contract.raw_descriptor_layout,
        "coordinate_semantics": contract.coordinate_semantics,
        "checkpoint": {
            "path": contract.checkpoint.relative_to(model_root.parent).as_posix(),
            "sha256": _sha256_file(contract.checkpoint),
        },
        "evaluation_index": _selection_identity(contract),
        "runtime_source": _runtime_source_identity(contract),
        "simulator_source_files": [
            _simulator_source_file_identity(repository_root, relative_path)
            for relative_path in _SIMULATOR_SOURCE_FILES
        ],
        "source_files": {
            label: _source_file_identity(
                model_root, relative_path=relative_path, label=label
            )
            for label, relative_path in _SOURCE_FILES.items()
        },
        "submodule_git_head": submodule_head,
    }


def validate_frozen_depthsplat_backend_identity(
    contract: DepthSplatBackendContract, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed when any DepthSplat source or application input drifts."""

    if not isinstance(identity, Mapping):
        raise TypeError("frozen DepthSplat backend identity must be a mapping")
    expected = freeze_depthsplat_backend_identity(contract)
    if dict(identity) != expected:
        raise ValueError("frozen DepthSplat backend identity changed")
    return expected


def _selected_positions(selection_mask: Any) -> tuple[Any, int, int]:
    import torch

    if (
        not torch.is_tensor(selection_mask)
        or selection_mask.ndim != 3
        or selection_mask.dtype != torch.bool
        or any(size < 1 for size in selection_mask.shape)
    ):
        raise ValueError("DepthSplat coordinates require a nonempty [V,H,W] bool mask")
    positions = selection_mask.nonzero(as_tuple=False)
    if positions.numel() == 0:
        raise ValueError("DepthSplat coordinates require at least one descriptor")
    return positions, int(selection_mask.shape[1]), int(selection_mask.shape[2])


def source_native_depthsplat_coordinates(
    raw_head_descriptors: Any,
    selection_mask: Any,
    *,
    sample_image_grid: Callable[..., tuple[Any, Any]],
) -> Any:
    """Mirror ``EncoderDepthSplat``'s pixel-center plus offset expression.

    ``sample_image_grid`` must be the source function imported by the loaded
    DepthSplat encoder module.  Keeping it an explicit argument prevents a
    cached foreign top-level ``src`` package from silently defining geometry.
    The result is a normalized image coordinate; the native Adapter then uses
    it with *z-depth* rays rather than a unit-length ray convention.
    """

    import torch

    positions, height, width = _selected_positions(selection_mask)
    if (
        not torch.is_tensor(raw_head_descriptors)
        or raw_head_descriptors.ndim != 2
        or raw_head_descriptors.shape[0] != positions.shape[0]
        or raw_head_descriptors.shape[1] < 3
        or not raw_head_descriptors.is_floating_point()
    ):
        raise ValueError(
            "DepthSplat selected raw descriptors must be floating [selected_positions,>=3]"
        )
    if not callable(sample_image_grid):
        raise TypeError("DepthSplat coordinates require the loaded source sample_image_grid")
    offsets = raw_head_descriptors[:, 1:3].sigmoid()
    grid, _ = sample_image_grid((height, width), offsets.device)
    if not torch.is_tensor(grid) or tuple(grid.shape) != (height, width, 2):
        raise ValueError("DepthSplat source sample_image_grid returned an invalid grid")
    positions = positions.to(offsets.device)
    pixels = positions[:, 1] * width + positions[:, 2]
    base = grid.reshape(height * width, 2)[pixels].to(dtype=offsets.dtype)
    pixel_size = torch.tensor(
        (1 / width, 1 / height), dtype=offsets.dtype, device=offsets.device
    )
    return base + (offsets - 0.5) * pixel_size


def canonical_json_sha256(value: Any) -> str:
    """Return the canonical digest used to bind DepthSplat producer records."""

    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("DepthSplat identity payload must be JSON-serializable") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
