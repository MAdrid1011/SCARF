"""Keep reviewer-facing SAES documentation aligned with the current evidence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relative_path: str) -> str:
    return " ".join((ROOT / relative_path).read_text(encoding="utf-8").split())


def test_saes_mechanism_docs_keep_the_dense_execution_boundary() -> None:
    mechanisms = _text("docs/fsdr-saes-mechanisms.md")
    guide = _text("docs/multi-model-demo-guide.md")

    assert "all S2/S3 model work completes before SAES" in mechanisms
    assert "does not verify sparse S2/S3 execution" in mechanisms
    assert "twelve anchors total" in mechanisms
    assert "The current demo completes dense model S2/S3 work" in guide
    assert "not a performance benchmark or paper-evidence path" in guide
    assert "The current global mechanism configuration is preregistered" in guide
    assert "Expected Performance" not in guide
    assert "Each model has optimized SAES thresholds" not in guide


def test_saes_rtl_docs_keep_the_twelve_anchor_contract() -> None:
    contract = _text("docs/saes-rtl-contract.md")

    assert "four L0 descriptors and twelve L1 descriptors" in contract
    assert "[0, 3, 12, 15, 1, 2, 4, 7, 8, 11, 13, 14]" in contract
    assert "The separate stage-event simulator is not that replay" in contract
    assert "[5, 10, 1, 2]" not in contract
    assert "for eight descriptors in total" not in contract


def test_appendix_keeps_calibration_and_performance_claims_closed() -> None:
    appendix = _text("artifact/appendix.tex")

    assert "24 DL3DV training scenes and eight disjoint DL3DV holdout scenes" in appendix
    assert "configuration remains preregistered and has no selected tuple" in appendix
    assert "cannot support a Figure~8 or SAES speed claim" in appendix
    assert "32 Re10K and 32 ACID" not in appendix
