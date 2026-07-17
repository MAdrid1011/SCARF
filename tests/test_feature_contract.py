import pytest


torch = pytest.importorskip("torch")


def test_probe_statistic_summary_distinguishes_width_dependent_trace():
    from scripts.feature_contract import summarize_probe_statistics

    features = torch.zeros(1, 1, 2, 4, 4)
    for row, column, value in ((0, 0, 0.0), (0, 3, 2.0), (3, 0, 0.0), (3, 3, 2.0)):
        features[0, 0, :, row, column] = value
    wide = features.repeat(1, 1, 8, 1, 1)

    base = summarize_probe_statistics(features, height=4, width=4)
    widened = summarize_probe_statistics(wide, height=4, width=4)

    assert base["raw-probe-mean-channel-variance"]["p50"] == pytest.approx(
        widened["raw-probe-mean-channel-variance"]["p50"]
    )
    assert widened["raw-probe-vector-variance"]["p50"] == pytest.approx(
        8 * base["raw-probe-vector-variance"]["p50"]
    )
