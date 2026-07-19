"""Unit gates for the source-bound ACID runtime-control worker."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


def _head(*, padding_mode: str) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1, padding_mode=padding_mode),
        nn.GELU(),
        nn.Conv2d(4, 2, 3, padding=1, padding_mode=padding_mode),
    )


def test_runtime_worker_selects_the_real_model_specific_final_head_only():
    from scripts.acid_joint_runtime_control_evidence import _classic_head

    transplat = SimpleNamespace(
        depth_predictor=SimpleNamespace(to_gaussians=_head(padding_mode="zeros"))
    )
    mvsplat = SimpleNamespace(
        depth_predictor=SimpleNamespace(to_gaussians=_head(padding_mode="zeros"))
    )
    depthsplat = SimpleNamespace(gaussian_head=_head(padding_mode="replicate"))

    assert _classic_head(transplat, "transplat") is transplat.depth_predictor.to_gaussians
    assert _classic_head(mvsplat, "mvsplat") is mvsplat.depth_predictor.to_gaussians
    assert _classic_head(depthsplat, "depthsplat") is depthsplat.gaussian_head


def test_runtime_worker_rejects_an_unbound_or_invalid_final_head():
    from scripts.acid_joint_runtime_control_evidence import (
        RuntimeControlEvidenceError,
        _classic_head,
    )

    with pytest.raises(RuntimeControlEvidenceError, match="native classic Gaussian head"):
        _classic_head(SimpleNamespace(depth_predictor=SimpleNamespace()), "transplat")
    with pytest.raises(RuntimeControlEvidenceError, match="native classic Gaussian head"):
        _classic_head(SimpleNamespace(gaussian_head=None), "depthsplat")


def test_runtime_worker_uses_the_same_registered_guard_on_route_as_cache_capture():
    from scripts import acid_joint_runtime_control_evidence as runtime
    from test_acid_joint_training_contract import _contract

    contract = _contract()
    identity = contract["saes_routing"]["execution_identity"]
    context = {
        "image": torch.zeros(1, 2, 3, 16, 16),
        "extrinsics": torch.eye(4).repeat(1, 2, 1, 1),
        "intrinsics": torch.eye(3).repeat(1, 2, 1, 1),
    }

    options = runtime._routing_options(
        contract, "transplat", context, geometry_on_cpu=False
    )

    assert options["cross_check_threshold"] == 0.015
    assert options["context_safety_guard"] is True
    assert options["materialization_guard"] is True
    assert contract["saes_routing"]["execution_route_sha256"] == identity[
        "route_sha256"
    ]


def test_runtime_evidence_direct_api_requires_a_model_isolation_marker(tmp_path, monkeypatch):
    from data.acid_joint_training_contract import ACID_ISOLATED_MODEL_ENV
    from scripts import acid_joint_runtime_control_evidence as runtime

    contract_path = tmp_path / "acid_joint_materialization_training_contract_v5.json"
    monkeypatch.setattr(runtime, "DEFAULT_TRAINING_CONTRACT_PATH", contract_path)
    monkeypatch.delenv(ACID_ISOLATED_MODEL_ENV, raising=False)

    with pytest.raises(Exception, match="dedicated isolated subprocess"):
        runtime.generate_runtime_evidence(
            contract_path=contract_path,
            materialization_root=tmp_path / "inputs",
            asset_path=tmp_path / "candidate.pt",
            model="transplat",
            split="calibration_train",
            output_path=tmp_path / "runtime.json",
            device=torch.device("cpu"),
        )
