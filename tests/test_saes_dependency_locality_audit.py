"""Tests for the fixed target-free SAES refinement locality audit helpers."""

import pytest


torch = pytest.importorskip("torch")


def test_fixed_locality_tile_origins_are_a_predeclared_nine_tile_raster():
    from scripts.saes_dependency_locality_audit import fixed_locality_tile_origins

    origins = fixed_locality_tile_origins(height=16, width=16, tile_size=4)

    assert origins == [
        (0, 0), (0, 8), (0, 12),
        (8, 0), (8, 8), (8, 12),
        (12, 0), (12, 8), (12, 12),
    ]


def test_source_tile_mask_never_selects_a_retained_probe():
    from scripts.saes_dependency_audit import build_probe_mask
    from scripts.saes_dependency_locality_audit import source_tile_nonprobe_mask

    probes = build_probe_mask(height=8, width=8, tile_size=4)
    selected = source_tile_nonprobe_mask(
        probe_mask=probes, source_y=4, source_x=0, tile_size=4
    )

    assert int(selected.sum()) == 12
    assert not bool((selected & probes).any())
    assert bool(selected[5, 1])
    assert not bool(selected[4, 0]) and not bool(selected[7, 3])


def test_locality_audit_removes_target_rgb_before_context_device_transfer():
    from scripts.saes_dependency_locality_audit import _target_free_context

    batch = {
        "context": {"image": torch.ones(1), "index": torch.tensor([0])},
        "target": {"image": torch.full((1,), 7.0)},
    }

    loaded, context = _target_free_context(batch, torch.device("cpu"))

    assert loaded is True
    assert set(context) == {"image", "index"}
    assert "image" not in batch["target"]


def test_locality_summary_reports_retained_probe_distance_only():
    from scripts.saes_dependency_audit import build_probe_mask
    from scripts.saes_dependency_locality_audit import summarize_locality_dependency

    baseline = torch.zeros(1, 2, 8, 8)
    perturbed = baseline.clone()
    perturbed[0, 0, 0, 0] = 1.0
    perturbed[0, 1, 7, 7] = 2.0
    perturbed[0, 1, 1, 1] = 3.0  # Non-probe location: not in the envelope.
    probes = build_probe_mask(height=8, width=8, tile_size=4)

    summary = summarize_locality_dependency(
        baseline,
        perturbed,
        probes,
        source_y=0,
        source_x=0,
        tile_size=4,
    )

    assert summary["source_tile"]["nonprobe_positions_zeroed"] == 12
    assert summary["changed_probe_spatial_position_count"] == 2
    assert summary["probe_dependency"]["probe_changed_value_count"] == 2
    assert summary["changed_probe_outside_source_tile"] is True
    assert summary["maximum_changed_probe_tile_chebyshev_distance"] == 1
    assert summary["changed_probe_tile_bounds"] == {
        "minimum_row": 0,
        "maximum_row": 1,
        "minimum_column": 0,
        "maximum_column": 1,
    }


def test_transplat_wrapper_preserves_the_generic_locality_contract(monkeypatch):
    import scripts.saes_dependency_locality_audit as audit

    expected = {"model": "transplat", "marker": "generic"}
    captured = {}

    def fake_collect(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(audit, "collect_locality_audit", fake_collect)

    assert audit.collect_transplat_locality_audit(sample_index=0, device="cuda") == expected
    assert captured == {"model_name": "transplat", "sample_index": 0, "device": "cuda"}
