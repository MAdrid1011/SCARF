from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _prepare(monkeypatch, tmp_path: Path):
    import scripts.install_claim_prerequisites as prerequisites
    import scripts.claim_timing_backend as timing
    import scripts.mechanism_config as mechanism

    source = tmp_path / "source"
    config = source / "calibration/mechanism_config.json"
    manifest = source / "timing-backend/manifest.json"
    config.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    config.write_text("real calibrated configuration\n", encoding="utf-8")
    manifest.write_text("real timing manifest\n", encoding="utf-8")

    samples = {
        (model, dataset, 0): {}
        for model in prerequisites.MODELS
        for dataset in prerequisites.DATASETS
    }
    backend = SimpleNamespace(
        samples=samples,
        backend_provenance={"source": {"sha256": "a" * 64}},
        clock_mhz=1000,
    )
    monkeypatch.setattr(
        mechanism,
        "require_calibrated_mechanism",
        lambda path: ({}, {"status": "calibrated", "evaluation_disjoint": True}),
    )
    monkeypatch.setattr(timing, "load_claim_timing_manifest", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        prerequisites,
        "_expected_counts",
        lambda _root, _profile: {dataset: 1 for dataset in prerequisites.DATASETS},
    )
    root = tmp_path / "release"
    (root / "artifact").mkdir(parents=True)
    (root / "artifact/mechanism_config.json").write_text("old\n", encoding="utf-8")
    return prerequisites, source, root


def test_check_only_validates_complete_prerequisites_without_writing(monkeypatch, tmp_path):
    prerequisites, source, root = _prepare(monkeypatch, tmp_path)

    report = prerequisites.install(source, root=root, profile="reviewer", check_only=True)

    assert report["status"] == "READY"
    assert report["mode"] == "check-only"
    assert report["timing_backend"]["sample_count"] == 9
    assert (root / "artifact/mechanism_config.json").read_text(encoding="utf-8") == "old\n"
    assert not (root / "outputs/timing-backend").exists()


def test_install_copies_only_validated_prerequisites(monkeypatch, tmp_path):
    prerequisites, source, root = _prepare(monkeypatch, tmp_path)

    report = prerequisites.install(source, root=root, profile="reviewer")

    assert report["mode"] == "installed"
    assert (root / "artifact/mechanism_config.json").read_text(encoding="utf-8") == (
        "real calibrated configuration\n"
    )
    assert (root / "outputs/timing-backend/manifest.json").read_text(encoding="utf-8") == (
        "real timing manifest\n"
    )


def test_install_rejects_incomplete_timing_coverage(monkeypatch, tmp_path):
    prerequisites, source, root = _prepare(monkeypatch, tmp_path)
    import scripts.claim_timing_backend as timing

    timing.load_claim_timing_manifest = lambda *args, **kwargs: SimpleNamespace(
        samples={("mvsplat", "re10k", 0): {}},
        backend_provenance={"source": {"sha256": "a" * 64}},
        clock_mhz=1000,
    )

    with pytest.raises(ValueError, match="does not cover"):
        prerequisites.install(source, root=root, profile="reviewer", check_only=True)
