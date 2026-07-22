"""Target-free reference tests for the staged SAES scalar RTL contract."""

import pytest


def test_scalar_moment_reference_matches_the_documented_merge_vector():
    from saes.rtl_moment_reference import scalar_moment_merge

    assert scalar_moment_merge(
        base_weight=8,
        base_mean=10,
        base_variance=4,
        updates=((2, 16, 9), (6, -2, 1)),
    ) == {
        "total_weight": 16,
        "mean": 6,
        "variance": 51,
        "accepted_updates": 2,
    }


def test_scalar_moment_reference_preserves_constants_and_rejects_bad_mass():
    from saes.rtl_moment_reference import scalar_moment_merge

    assert scalar_moment_merge(
        base_weight=4,
        base_mean=-7,
        base_variance=3,
        updates=((12, -7, 3),),
    ) == {
        "total_weight": 16,
        "mean": -7,
        "variance": 3,
        "accepted_updates": 1,
    }
    with pytest.raises(ValueError, match="base_weight"):
        scalar_moment_merge(
            base_weight=0,
            base_mean=0,
            base_variance=0,
            updates=(),
        )
