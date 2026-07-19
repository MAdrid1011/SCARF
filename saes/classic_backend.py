"""Small, source-faithful contracts shared by the classic Gaussian backends.

This module deliberately does not install a hook, alter an encoder, or select
a calibration.  It gives later target-free audits one explicit identity
contract and one source-native way to reconstruct selected Adapter
coordinates for TranSplat and MVSplat.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


ClassicModelName = Literal["transplat", "mvsplat"]


@dataclass(frozen=True)
class ClassicBackendContract:
    """Identity and native boundary locations for one classic DL3DV backend."""

    model: ClassicModelName
    dataset: Literal["dl3dv"]
    experiment: str
    checkpoint: Path
    environment_profile: str
    raw_head_module: str
    gaussian_adapter_module: str
    decoder_module: str
    coordinate_semantics: str


_CLASSIC_MODULES: dict[ClassicModelName, tuple[str, str, str, str]] = {
    "transplat": (
        "encoder.depth_predictor.to_gaussians",
        "encoder.gaussian_adapter",
        "decoder",
        "transplat-shared-raw-offset-helper",
    ),
    "mvsplat": (
        "encoder.depth_predictor.to_gaussians",
        "encoder.gaussian_adapter",
        "decoder",
        "mvsplat-inline-pixel-center-plus-sigmoid-offset",
    ),
}

# These files contain the source implementations named by ``_CLASSIC_MODULES``.
# Keep paths relative to the individual model submodule so frozen identities are
# portable between worktrees.
_CLASSIC_SOURCE_FILES: dict[ClassicModelName, dict[str, str]] = {
    "transplat": {
        "raw_head": "src/model/encoder/matching/depth_predictor_trans.py",
        "gaussian_adapter": "src/model/encoder/common/gaussian_adapter.py",
        "decoder": "src/model/decoder/decoder.py",
        "coordinates": "src/model/encoder/encoder_trans.py",
    },
    "mvsplat": {
        "raw_head": "src/model/encoder/costvolume/depth_predictor_multiview.py",
        "gaussian_adapter": "src/model/encoder/common/gaussian_adapter.py",
        "decoder": "src/model/decoder/decoder.py",
        "coordinates": "src/model/encoder/encoder_costvolume.py",
    },
}

FROZEN_CLASSIC_BACKEND_IDENTITY_SCHEMA_VERSION = (
    "classic-backend-frozen-identity-v1"
)
_GIT_HEAD_PATTERN = re.compile(r"[0-9a-f]{40}")


def resolve_classic_backend_contract(
    model: ClassicModelName | str, root: Path
) -> ClassicBackendContract:
    """Resolve the model-specific DL3DV experiment and native boundaries.

    Both classic models use the Re10K experiment/checkpoint on the converted
    DL3DV representation.  The returned path is intentionally not hashed here:
    a run must bind its own checkpoint digest before using the contract as
    evidence.
    """

    if model not in _CLASSIC_MODULES:
        raise ValueError(f"unsupported classic backend: {model!r}")

    from scripts.ae_config import resolve_experiment

    experiment = resolve_experiment(model, "dl3dv", Path(root))
    if (
        experiment.model != model
        or experiment.dataset != "dl3dv"
        or experiment.experiment != "re10k"
        or experiment.checkpoint.name != "re10k.ckpt"
        or experiment.environment_profile != "classic"
    ):
        raise RuntimeError("classic backend experiment identity changed")
    raw_head, adapter, decoder, coordinate_semantics = _CLASSIC_MODULES[model]
    return ClassicBackendContract(
        model=model,
        dataset="dl3dv",
        experiment=experiment.experiment,
        checkpoint=experiment.checkpoint.resolve(),
        environment_profile=experiment.environment_profile,
        raw_head_module=raw_head,
        gaussian_adapter_module=adapter,
        decoder_module=decoder,
        coordinate_semantics=coordinate_semantics,
    )


def _validated_contract_root(
    contract: ClassicBackendContract,
) -> tuple[ClassicModelName, Path]:
    """Return the model submodule root for one complete classic contract."""

    if not isinstance(contract, ClassicBackendContract):
        raise TypeError("classic backend identity requires a ClassicBackendContract")
    if contract.model not in _CLASSIC_MODULES:
        raise ValueError("classic backend identity has an unsupported model")
    expected_modules = _CLASSIC_MODULES[contract.model]
    if (
        contract.dataset != "dl3dv"
        or contract.experiment != "re10k"
        or contract.environment_profile != "classic"
        or (
            contract.raw_head_module,
            contract.gaussian_adapter_module,
            contract.decoder_module,
            contract.coordinate_semantics,
        )
        != expected_modules
    ):
        raise ValueError("classic backend identity contract fields changed")

    checkpoint = Path(contract.checkpoint).resolve()
    model_root = checkpoint.parent.parent
    if (
        checkpoint.name != "re10k.ckpt"
        or checkpoint.parent.name != "checkpoints"
        or model_root.name != contract.model
        or not checkpoint.is_file()
        or not model_root.is_dir()
    ):
        raise ValueError("classic backend identity checkpoint layout is invalid")
    return contract.model, model_root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"classic backend identity cannot read {path.name}") from error
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
        raise RuntimeError("classic backend identity cannot execute git") from error
    if completed.returncode != 0:
        raise RuntimeError(
            "classic backend identity cannot resolve submodule git state"
        )
    return completed.stdout.strip()


def _source_file_identity(
    model_root: Path, *, relative_path: str, label: str
) -> dict[str, str]:
    source_root = model_root.resolve()
    source_path = (source_root / relative_path).resolve()
    if source_root not in source_path.parents or not source_path.is_file():
        raise ValueError(f"classic backend identity source file is invalid: {label}")
    tracked = _git_output(
        model_root, "ls-files", "--error-unmatch", "--", relative_path
    )
    if tracked != relative_path:
        raise RuntimeError(
            f"classic backend identity source file is not tracked: {label}"
        )
    return {
        "path": f"{model_root.name}/{relative_path}",
        "sha256": _sha256_file(source_path),
    }


def freeze_classic_backend_identity(
    contract: ClassicBackendContract,
) -> dict[str, Any]:
    """Return a portable, JSON-serializable source identity for one backend.

    The function deliberately reads the currently materialized source files and
    the submodule's resolved Git HEAD. It refuses a missing, untracked, or
    malformed source boundary rather than emitting an identity that could be
    reused after source drift.
    """

    model, model_root = _validated_contract_root(contract)
    submodule_head = _git_output(model_root, "rev-parse", "--verify", "HEAD")
    if not _GIT_HEAD_PATTERN.fullmatch(submodule_head):
        raise RuntimeError("classic backend identity submodule HEAD is invalid")
    source_files = {
        label: _source_file_identity(
            model_root, relative_path=relative_path, label=label
        )
        for label, relative_path in _CLASSIC_SOURCE_FILES[model].items()
    }
    return {
        "schema_version": FROZEN_CLASSIC_BACKEND_IDENTITY_SCHEMA_VERSION,
        "model": contract.model,
        "dataset": contract.dataset,
        "experiment": contract.experiment,
        "environment_profile": contract.environment_profile,
        "raw_head_module": contract.raw_head_module,
        "gaussian_adapter_module": contract.gaussian_adapter_module,
        "decoder_module": contract.decoder_module,
        "coordinate_semantics": contract.coordinate_semantics,
        "source_files": source_files,
        "submodule_git_head": submodule_head,
    }


def validate_frozen_classic_backend_identity(
    contract: ClassicBackendContract, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless a frozen identity still matches the live backend."""

    if not isinstance(identity, Mapping):
        raise TypeError("frozen classic backend identity must be a mapping")
    expected = freeze_classic_backend_identity(contract)
    if dict(identity) != expected:
        raise ValueError("frozen classic backend identity changed")
    return expected


def _selected_positions(selection_mask: Any) -> tuple[Any, int, int]:
    """Validate the canonical selected descriptor layout and return positions."""

    import torch

    if (
        not torch.is_tensor(selection_mask)
        or selection_mask.ndim != 3
        or selection_mask.dtype != torch.bool
        or any(size < 1 for size in selection_mask.shape)
    ):
        raise ValueError("selected coordinates require a nonempty [V,H,W] bool mask")
    positions = selection_mask.nonzero(as_tuple=False)
    if positions.numel() == 0:
        raise ValueError("selected coordinates require at least one descriptor")
    return positions, int(selection_mask.shape[1]), int(selection_mask.shape[2])


def _validate_raw_descriptors(
    raw_descriptors: Any, positions: Any
) -> Any:
    import torch

    if (
        not torch.is_tensor(raw_descriptors)
        or raw_descriptors.ndim != 2
        or raw_descriptors.shape[0] != positions.shape[0]
        or raw_descriptors.shape[1] < 2
        or not raw_descriptors.is_floating_point()
    ):
        raise ValueError(
            "selected raw descriptors must be floating [selected_positions,>=2]"
        )
    return raw_descriptors[:, :2].sigmoid()


def _mvsplat_inline_adapter_coordinates(
    offsets: Any, pixel_indices: Any, image_shape: tuple[int, int]
) -> Any:
    """Mirror MVSplat's inline ``EncoderCostVolume`` coordinate expression."""

    import torch
    from einops import rearrange
    from mvsplat.src.geometry.projection import sample_image_grid

    height, width = image_shape
    xy_ray, _ = sample_image_grid((height, width), offsets.device)
    xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
    xy_ray = xy_ray[pixel_indices.to(xy_ray.device), 0]
    pixel_size = 1 / torch.tensor(
        (width, height), dtype=torch.float32, device=offsets.device
    )
    return xy_ray + (offsets - 0.5) * pixel_size


def source_native_adapter_coordinates(
    contract: ClassicBackendContract,
    raw_descriptors: Any,
    selection_mask: Any,
) -> Any:
    """Return native Adapter coordinates for canonical selected descriptors.

    ``raw_descriptors`` must be ordered exactly like
    ``selection_mask.nonzero(as_tuple=False)``: view, row, column.  TranSplat
    delegates to its existing source helper.  MVSplat has no equivalent helper,
    so this follows its inline grid-plus-offset expression verbatim.
    """

    import torch

    positions, height, width = _selected_positions(selection_mask)
    offsets = _validate_raw_descriptors(raw_descriptors, positions)
    pixel_indices = (positions[:, 1] * width + positions[:, 2]).to(
        device=offsets.device
    )
    if contract.model == "transplat":
        from transplat.src.model.encoder.encoder_trans import (
            gaussian_adapter_coordinates_from_raw_offsets,
        )

        return gaussian_adapter_coordinates_from_raw_offsets(
            offsets,
            image_shape=(height, width),
            pixel_indices=pixel_indices.to(dtype=torch.int64),
        )
    if contract.model == "mvsplat":
        return _mvsplat_inline_adapter_coordinates(
            offsets,
            pixel_indices.to(dtype=torch.int64),
            (height, width),
        )
    raise ValueError(f"unsupported classic backend: {contract.model!r}")
