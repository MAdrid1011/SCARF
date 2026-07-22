"""Regression tests for target-free SAES route construction."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        (
            "scripts.saes_paper_l0_l1_compact_packet_pilot",
            "collect_paper_compact_packet_pilot",
        ),
        (
            "scripts.saes_sparse_packet_quality_pilot",
            "collect_packet_quality_pilot",
        ),
    ],
)
def test_quality_pilots_defer_target_access_until_packet_commit(
    module_name: str, function_name: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / (module_name.replace(".", "/") + ".py")
    module = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    source = ast.get_source_segment(path.read_text(encoding="utf-8"), function)
    assert source is not None

    packet_commit = source.index('final_packed = capture["final_packed"]')
    packet_render = source.index("render_packed_gaussians(")

    if module_name == "scripts.saes_paper_l0_l1_compact_packet_pilot":
        context_loader = source.index("context_data = load_context_only_audit_data(")
        quality_gate = source.index("quality_gate = _validate_target_free_quality_audit(")
        native_data_load = source.index(
            "target_batch = _load_native_target_batch_after_packet_gate("
        )
        target_access = source.index('target_mapping = target_batch.get("target")')
        assert "load_model_and_data" not in source
        assert context_loader < packet_commit < quality_gate < native_data_load
        assert native_data_load < target_access < packet_render
        assert quality_gate < source.index("_target_cameras(target_batch, loaded_device)")
        assert quality_gate < source.index(
            "_take_target_rgb_for_metrics(target_batch, loaded_device)"
        )
    else:
        target_access = source.index('target_mapping = batch.get("target")')
        assert packet_commit < target_access < packet_render
