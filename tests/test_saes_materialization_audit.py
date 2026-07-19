import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_target_free_materialization_audit_accepts_adapter_offset_transport():
    from scripts.saes_target_free_materialization_audit import build_parser

    args = build_parser().parse_args(
        [
            "--materialization",
            "conditional-adapter-offset-transport-diagnostic",
            "--output-dir",
            str(Path("outputs") / "new-audit"),
        ]
    )

    assert args.materialization == "conditional-adapter-offset-transport-diagnostic"


def test_target_free_materialization_parser_imports_without_torch():
    code = """
import builtins

original_import = builtins.__import__

def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("Torch must not be imported by the audit parser")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_torch
from scripts.saes_target_free_materialization_audit import (
    MATERIALIZATION_CHOICES,
    build_parser,
)

assert "conditional-adapter-offset-transport-diagnostic" in MATERIALIZATION_CHOICES
assert build_parser().parse_args(["--output-dir", "outputs/audit"]).sample_index == 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_capture_encoder_execution_loads_torch_only_at_runtime():
    torch = pytest.importorskip("torch")
    from scripts.saes_target_free_materialization_audit import _capture_encoder_execution

    class Predictor(torch.nn.Module):
        def forward(self, features):
            return (features.mean(dim=2, keepdim=True),)

    class Encoder:
        def __init__(self):
            self.depth_predictor = Predictor()
            self.result = object()

        def __call__(self, context, _unused, *, deterministic):
            assert deterministic is True
            self.depth_predictor(context["features"])
            return self.result

    encoder = Encoder()
    features = torch.randn(1, 2, 3, 4, 4)
    gaussians, captured_features, captured_depths = _capture_encoder_execution(
        SimpleNamespace(encoder=encoder), {"features": features}
    )

    assert gaussians is encoder.result
    torch.testing.assert_close(captured_features, features)
    torch.testing.assert_close(captured_depths, features.mean(dim=2, keepdim=True))
