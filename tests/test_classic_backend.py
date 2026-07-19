"""CPU coverage for isolated classic-backend identity and coordinate contracts."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model", ("transplat", "mvsplat"))
def test_classic_backend_contract_binds_model_experiment_and_checkpoint(model: str):
    from saes.classic_backend import resolve_classic_backend_contract

    contract = resolve_classic_backend_contract(model, ROOT)

    assert contract.model == model
    assert contract.dataset == "dl3dv"
    assert contract.experiment == "re10k"
    assert contract.checkpoint == ROOT / model / "checkpoints" / "re10k.ckpt"
    assert contract.environment_profile == "classic"
    assert contract.raw_head_module == "encoder.depth_predictor.to_gaussians"
    assert contract.gaussian_adapter_module == "encoder.gaussian_adapter"
    assert contract.decoder_module == "decoder"


def test_classic_backend_contract_rejects_nonclassic_models():
    from saes.classic_backend import resolve_classic_backend_contract

    with pytest.raises(ValueError, match="unsupported classic backend"):
        resolve_classic_backend_contract("depthsplat", ROOT)


@pytest.mark.parametrize("model", ("transplat", "mvsplat"))
def test_frozen_classic_backend_identity_binds_live_source_files(model: str):
    from saes.classic_backend import (
        FROZEN_CLASSIC_BACKEND_IDENTITY_SCHEMA_VERSION,
        freeze_classic_backend_identity,
        resolve_classic_backend_contract,
        validate_frozen_classic_backend_identity,
    )

    contract = resolve_classic_backend_contract(model, ROOT)
    identity = freeze_classic_backend_identity(contract)

    assert json.loads(json.dumps(identity, sort_keys=True)) == identity
    assert identity["schema_version"] == FROZEN_CLASSIC_BACKEND_IDENTITY_SCHEMA_VERSION
    assert identity["model"] == model
    assert identity["dataset"] == "dl3dv"
    assert identity["experiment"] == "re10k"
    assert identity["environment_profile"] == "classic"
    assert re.fullmatch(r"[0-9a-f]{40}", identity["submodule_git_head"])
    assert set(identity["source_files"]) == {
        "raw_head",
        "gaussian_adapter",
        "decoder",
        "coordinates",
    }
    for source in identity["source_files"].values():
        source_path = ROOT / source["path"]
        assert source_path.is_file()
        assert source["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert validate_frozen_classic_backend_identity(contract, identity) == identity


def _temporary_mvsplat_contract(tmp_path: Path):
    from saes.classic_backend import ClassicBackendContract

    root = tmp_path / "fixture-root"
    model_root = root / "mvsplat"
    source_files = {
        "src/model/encoder/costvolume/depth_predictor_multiview.py": "raw-head\n",
        "src/model/encoder/common/gaussian_adapter.py": "adapter\n",
        "src/model/decoder/decoder.py": "decoder\n",
        "src/model/encoder/encoder_costvolume.py": "coordinates\n",
    }
    for relative_path, contents in source_files.items():
        path = model_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    checkpoint = model_root / "checkpoints" / "re10k.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    for command in (
        ("git", "init", "-q"),
        ("git", "add", "."),
        (
            "git",
            "-c",
            "user.email=identity@example.invalid",
            "-c",
            "user.name=Identity Test",
            "commit",
            "-qm",
            "fixture",
        ),
    ):
        subprocess.run(
            command, cwd=model_root, check=True, capture_output=True, text=True
        )
    return (
        ClassicBackendContract(
            model="mvsplat",
            dataset="dl3dv",
            experiment="re10k",
            checkpoint=checkpoint,
            environment_profile="classic",
            raw_head_module="encoder.depth_predictor.to_gaussians",
            gaussian_adapter_module="encoder.gaussian_adapter",
            decoder_module="decoder",
            coordinate_semantics="mvsplat-inline-pixel-center-plus-sigmoid-offset",
        ),
        model_root / "src/model/encoder/encoder_costvolume.py",
    )


def test_frozen_classic_backend_identity_detects_live_source_drift(tmp_path: Path):
    from saes.classic_backend import (
        freeze_classic_backend_identity,
        validate_frozen_classic_backend_identity,
    )

    contract, coordinate_source = _temporary_mvsplat_contract(tmp_path)
    frozen = freeze_classic_backend_identity(contract)
    coordinate_source.write_text("coordinates changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identity changed"):
        validate_frozen_classic_backend_identity(contract, frozen)
    refreshed = freeze_classic_backend_identity(contract)
    assert (
        refreshed["source_files"]["coordinates"]["sha256"]
        != frozen["source_files"]["coordinates"]["sha256"]
    )


def test_frozen_classic_backend_identity_rejects_invalid_contract_or_payload(
    tmp_path: Path,
):
    from saes.classic_backend import (
        ClassicBackendContract,
        freeze_classic_backend_identity,
        validate_frozen_classic_backend_identity,
    )

    with pytest.raises(TypeError, match="ClassicBackendContract"):
        freeze_classic_backend_identity(object())

    contract, _source = _temporary_mvsplat_contract(tmp_path)
    invalid = ClassicBackendContract(
        **{**contract.__dict__, "environment_profile": "unexpected"}
    )
    with pytest.raises(ValueError, match="contract fields"):
        freeze_classic_backend_identity(invalid)
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_frozen_classic_backend_identity(contract, object())


def _selected_raw_inputs():
    torch = pytest.importorskip("torch")

    mask = torch.zeros(2, 4, 5, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 3, 4] = True
    mask[1, 1, 2] = True
    mask[1, 2, 1] = True
    torch.manual_seed(173)
    raw = torch.randn(int(mask.sum()), 84, dtype=torch.float32)
    return torch, raw, mask


def test_mvsplat_selected_coordinates_match_its_inline_source_formula():
    torch, raw, mask = _selected_raw_inputs()
    from einops import rearrange
    from mvsplat.src.geometry.projection import sample_image_grid
    from saes.classic_backend import (
        resolve_classic_backend_contract,
        source_native_adapter_coordinates,
    )

    contract = resolve_classic_backend_contract("mvsplat", ROOT)
    actual = source_native_adapter_coordinates(contract, raw, mask)

    height, width = mask.shape[-2:]
    positions = mask.nonzero(as_tuple=False)
    offsets = raw[:, :2].sigmoid()
    xy_ray, _ = sample_image_grid((height, width), offsets.device)
    xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
    pixels = positions[:, 1] * width + positions[:, 2]
    expected = xy_ray[pixels, 0] + (offsets - 0.5) * torch.tensor(
        (1 / width, 1 / height), dtype=torch.float32
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_transplat_selected_coordinates_delegate_to_its_source_helper(monkeypatch):
    torch, raw, mask = _selected_raw_inputs()
    from transplat.src.model.encoder import encoder_trans
    from saes.classic_backend import (
        resolve_classic_backend_contract,
        source_native_adapter_coordinates,
    )

    captured = {}

    def fake_source_helper(offsets, *, image_shape, pixel_indices):
        captured["offsets"] = offsets.clone()
        captured["image_shape"] = image_shape
        captured["pixel_indices"] = pixel_indices.clone()
        return offsets + 7.0

    monkeypatch.setattr(
        encoder_trans,
        "gaussian_adapter_coordinates_from_raw_offsets",
        fake_source_helper,
    )
    contract = resolve_classic_backend_contract("transplat", ROOT)
    actual = source_native_adapter_coordinates(contract, raw, mask)

    positions = mask.nonzero(as_tuple=False)
    expected_pixels = positions[:, 1] * mask.shape[-1] + positions[:, 2]
    torch.testing.assert_close(actual, raw[:, :2].sigmoid() + 7.0)
    torch.testing.assert_close(captured["offsets"], raw[:, :2].sigmoid())
    assert captured["image_shape"] == tuple(mask.shape[-2:])
    assert torch.equal(captured["pixel_indices"], expected_pixels)


def test_selected_coordinates_reject_descriptor_mask_cardinality_mismatch():
    torch, raw, mask = _selected_raw_inputs()
    from saes.classic_backend import (
        resolve_classic_backend_contract,
        source_native_adapter_coordinates,
    )

    contract = resolve_classic_backend_contract("mvsplat", ROOT)
    with pytest.raises(ValueError, match="selected raw descriptors"):
        source_native_adapter_coordinates(contract, raw[:-1], mask)
