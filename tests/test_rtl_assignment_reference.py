"""Target-free exact-integer tests for the staged SAES assignment normalizer."""

import pytest


def test_assignment_reference_is_exactly_normalized_with_deterministic_residual():
    from saes.rtl_assignment_reference import normalize_assignment_scores

    assert normalize_assignment_scores((2, 3, 5, 0), anchor_count=4, fraction_bits=8) == {
        "weights": (51, 76, 129, 0, 0, 0, 0, 0),
        "weight_sum": 256,
        "residual_anchor": 2,
    }
    assert normalize_assignment_scores((1, 1, 1), anchor_count=3, fraction_bits=8) == {
        "weights": (86, 85, 85, 0, 0, 0, 0, 0),
        "weight_sum": 256,
        "residual_anchor": 0,
    }


def test_assignment_reference_rejects_empty_or_zero_score_inputs():
    from saes.rtl_assignment_reference import normalize_assignment_scores

    with pytest.raises(ValueError, match="anchor_count"):
        normalize_assignment_scores((1,), anchor_count=0)
    with pytest.raises(ValueError, match="sum"):
        normalize_assignment_scores((0, 0), anchor_count=2)
