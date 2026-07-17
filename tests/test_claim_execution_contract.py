import inspect
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_claim_selection_resolves_pinned_upstream_index():
    from scripts.ae_config import resolve_claim_selection

    selection = resolve_claim_selection("mvsplat", "re10k", ROOT)
    assert selection.index_path == ROOT / "transplat/assets/evaluation_index_re10k.json"
    assert selection.sample_count == 6474
    assert len(selection.sample_selection_sha256) == 64


def test_demo_claim_run_requires_evaluation_index(capsys):
    from scripts.demo_cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--model", "mvsplat", "--dataset", "re10k", "--claim-run"])
    assert "evaluation-index" in capsys.readouterr().err


def test_all_model_loaders_accept_an_evaluation_index(monkeypatch):
    fake_torch = SimpleNamespace(
        device=type("device", (), {}),
        Tensor=type("Tensor", (), {}),
        hub=SimpleNamespace(load=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "integration", raising=False)
    monkeypatch.delitem(sys.modules, "integration.model_loader", raising=False)
    module = importlib.import_module("integration.model_loader")
    try:
        for loader in (
            module.TransplatLoader,
            module.MVSplatLoader,
            module.DepthSplatLoader,
        ):
            parameters = inspect.signature(loader.load_model).parameters
            assert "evaluation_index" in parameters
            assert "hydra_overrides" in parameters
    finally:
        sys.modules.pop("integration.model_loader", None)
        sys.modules.pop("integration", None)


def test_full_claim_dry_run_passes_index_to_pair_worker(
    tmp_path: Path, monkeypatch
):
    from argparse import Namespace
    import scripts.run_ae as runner

    status = runner.load_claim_status()
    for pair in status["software_pairs"]:
        if not pair.endswith("/dl3dv"):
            status["software_pairs"][pair] = "CLAIMED"
    monkeypatch.setattr(runner, "load_claim_status", lambda: status)

    plan = runner.build_plan(
        Namespace(mode="quality", output_root=tmp_path, python=None, num_samples=None)
    )
    assert len(plan["experiments"]) == 9
    for experiment in plan["experiments"]:
        command = experiment["command"]
        assert command[1].endswith("scripts/run_pair.py")
        assert "--evaluation-index" in command
        assert experiment["sample_count"] in {6474, 1595, 140}
