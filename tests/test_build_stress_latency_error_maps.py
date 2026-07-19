"""Regression tests for the Figure 10 source-data renderer."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_stress_latency_error_maps.py"


def _module():
    spec = spec_from_file_location("build_stress_latency_error_maps", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_view_loss_trace_preserves_raw_values_and_selected_value(tmp_path):
    module = _module()
    selected = tmp_path / "selected"
    rows = [
        {"dir": str(tmp_path / "improved"), "delta": -0.2},
        {"dir": str(selected), "delta": 0.01},
        {"dir": str(tmp_path / "larger"), "delta": 0.5},
    ]

    trace = module.view_loss_trace(rows, selected, "delta")

    assert trace["values"] == [-0.2, 0.01, 0.5]
    assert trace["selected_index"] == 1
    assert trace["actual_selected"] == 0.01
    assert trace["plot_selected"] == 0.01
