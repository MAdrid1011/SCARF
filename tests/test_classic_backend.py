"""CPU coverage for isolated classic-backend identity and coordinate contracts."""

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
