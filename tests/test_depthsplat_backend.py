"""CPU contracts for the isolated native DepthSplat DL3DV backend."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _require_depthsplat_identity_assets() -> None:
    required = (
        ROOT / "depthsplat/checkpoints/dl3dv.ckpt",
        ROOT / "depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json",
        ROOT / "assets/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    )
    if any(not path.exists() for path in required):
        pytest.skip(
            "DepthSplat frozen-identity tests require external checkpoints and runtime assets"
        )


def test_depthsplat_backend_contract_binds_the_native_dl3dv_route():
    from saes.depthsplat_backend import resolve_depthsplat_backend_contract

    _require_depthsplat_identity_assets()
    contract = resolve_depthsplat_backend_contract(ROOT)

    assert contract.model == "depthsplat"
    assert contract.dataset == "dl3dv"
    assert contract.experiment == "dl3dv"
    assert contract.checkpoint == ROOT / "depthsplat/checkpoints/dl3dv.ckpt"
    assert contract.evaluation_index == (
        ROOT / "depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json"
    )
    assert contract.environment_profile == "depthsplat"
    assert contract.gaussian_regressor_module == "encoder.gaussian_regressor"
    assert contract.gaussian_head_module == "encoder.gaussian_head"
    assert contract.raw_descriptor_layout == "opacity-logit-offset-xy-adapter-body-v1"
    assert contract.coordinate_semantics.startswith("depthsplat-z-depth-")


def test_frozen_depthsplat_identity_binds_checkpoint_index_and_live_sources():
    from saes.depthsplat_backend import (
        FROZEN_DEPTHSPLAT_BACKEND_IDENTITY_SCHEMA_VERSION,
        freeze_depthsplat_backend_identity,
        resolve_depthsplat_backend_contract,
        validate_frozen_depthsplat_backend_identity,
    )

    _require_depthsplat_identity_assets()
    contract = resolve_depthsplat_backend_contract(ROOT)
    identity = freeze_depthsplat_backend_identity(contract)

    assert json.loads(json.dumps(identity, sort_keys=True)) == identity
    assert identity["schema_version"] == FROZEN_DEPTHSPLAT_BACKEND_IDENTITY_SCHEMA_VERSION
    assert re.fullmatch(r"[0-9a-f]{40}", identity["submodule_git_head"])
    assert identity["checkpoint"]["path"] == "depthsplat/checkpoints/dl3dv.ckpt"
    assert identity["checkpoint"]["sha256"] == hashlib.sha256(
        contract.checkpoint.read_bytes()
    ).hexdigest()
    assert identity["evaluation_index"]["path"] == (
        "depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json"
    )
    assert identity["evaluation_index"]["sample_count"] == 140
    runtime_source = identity["runtime_source"]
    assert runtime_source["repository_path"] == "depthsplat"
    assert runtime_source["repository_commit"] == identity["submodule_git_head"]
    assert runtime_source["src_path"] == "depthsplat/src"
    assert len(runtime_source["runtime_import_roots"]) == 2
    assert runtime_source["runtime_import_roots"][0]["path"] == "depthsplat/src"
    assert runtime_source["runtime_import_roots"][1]["path"].startswith(
        "assets/torch/hub/facebookresearch_dinov2_"
    )
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", item["tree_sha256"])
        and item["file_count"] > 0
        for item in runtime_source["runtime_import_roots"]
    )
    assert [item["path"] for item in identity["simulator_source_files"]] == [
        "integration/model_loader.py",
        "saes/depthsplat_backend.py",
        "saes/depthsplat_l0_l1_materializer.py",
        "saes/depthsplat_selected_output.py",
        "saes/selected_output_replay.py",
        "scripts/saes_depthsplat_selected_output_audit.py",
    ]
    for source in identity["simulator_source_files"]:
        assert source["sha256"] == hashlib.sha256(
            (ROOT / source["path"]).read_bytes()
        ).hexdigest()
    assert set(identity["source_files"]) == {
        "decoder",
        "depth_predictor",
        "encoder",
        "encoder_config",
        "experiment_config",
        "gaussian_adapter",
        "gaussian_types",
        "projection",
        "sh_rotation",
    }
    for source in identity["source_files"].values():
        source_path = ROOT / source["path"]
        assert source["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert validate_frozen_depthsplat_backend_identity(contract, identity) == identity


def test_frozen_depthsplat_identity_rejects_mutated_checkpoint_binding():
    from saes.depthsplat_backend import (
        freeze_depthsplat_backend_identity,
        resolve_depthsplat_backend_contract,
        validate_frozen_depthsplat_backend_identity,
    )

    _require_depthsplat_identity_assets()
    contract = resolve_depthsplat_backend_contract(ROOT)
    identity = freeze_depthsplat_backend_identity(contract)
    mutated = copy.deepcopy(identity)
    mutated["checkpoint"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="identity changed"):
        validate_frozen_depthsplat_backend_identity(contract, mutated)


def test_depthsplat_coordinates_follow_source_pixel_center_and_offset_formula():
    torch = pytest.importorskip("torch")
    from depthsplat.src.geometry.projection import sample_image_grid
    from saes.depthsplat_backend import source_native_depthsplat_coordinates

    mask = torch.zeros(2, 4, 5, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 3, 4] = True
    mask[1, 1, 2] = True
    raw = torch.tensor(
        [
            [0.5, -0.2, 0.3, 9.0],
            [-0.4, 1.0, -0.8, 8.0],
            [0.1, 0.0, 0.0, 7.0],
        ],
        dtype=torch.float32,
    )

    actual = source_native_depthsplat_coordinates(
        raw, mask, sample_image_grid=sample_image_grid
    )
    positions = mask.nonzero(as_tuple=False)
    grid, _ = sample_image_grid((4, 5), raw.device)
    pixels = positions[:, 1] * 5 + positions[:, 2]
    expected = grid.reshape(20, 2)[pixels] + (
        raw[:, 1:3].sigmoid() - 0.5
    ) * torch.tensor((1 / 5, 1 / 4))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "raw,mask,message",
    [
        (None, None, r"nonempty \[V,H,W\]"),
        ("not-a-tensor", None, r"nonempty \[V,H,W\]"),
    ],
)
def test_depthsplat_coordinate_contract_rejects_invalid_selection(raw, mask, message):
    from saes.depthsplat_backend import source_native_depthsplat_coordinates

    with pytest.raises(ValueError, match=message):
        source_native_depthsplat_coordinates(
            raw,
            mask,
            sample_image_grid=lambda *_args, **_kwargs: (None, None),
        )


def test_depthsplat_coordinate_contract_rejects_the_classic_two_channel_layout():
    torch = pytest.importorskip("torch")
    from depthsplat.src.geometry.projection import sample_image_grid
    from saes.depthsplat_backend import source_native_depthsplat_coordinates

    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="selected raw descriptors"):
        source_native_depthsplat_coordinates(
            torch.zeros(1, 2), mask, sample_image_grid=sample_image_grid
        )
