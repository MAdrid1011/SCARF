"""Keep reviewer-facing SAES documentation aligned with the current evidence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relative_path: str) -> str:
    return " ".join((ROOT / relative_path).read_text(encoding="utf-8").split())


def test_saes_mechanism_docs_explain_the_public_execution_flow() -> None:
    mechanisms = _text("docs/fsdr-saes-mechanisms.md")
    guide = _text("docs/multi-model-demo-guide.md")

    assert "FSDR changes the S2 candidate search" in mechanisms
    assert "SAES changes the S3 materialization route" in mechanisms
    assert "twelve anchors total" in mechanisms
    assert "The demo runs the selected model encoder" in guide
    assert "The same command shape is used for all three adapters" in guide
    assert "Expected Performance" not in guide
    assert "Each model has optimized SAES thresholds" not in guide
