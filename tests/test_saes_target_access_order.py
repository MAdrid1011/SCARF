"""Regression tests for target-free SAES route construction."""

from __future__ import annotations

import inspect

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
    module = __import__(module_name, fromlist=[function_name])
    source = inspect.getsource(getattr(module, function_name))

    packet_commit = source.index('final_packed = capture["final_packed"]')
    target_access = source.index('target_mapping = batch.get("target")')
    packet_render = source.index("render_packed_gaussians(")

    assert packet_commit < target_access < packet_render
