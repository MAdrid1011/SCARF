import pytest


def test_retained_scheduler_reference_matches_declared_t4_probe_layout():
    from saes.rtl_retained_scheduler_reference import (
        retained_output_indices,
        retained_output_positions,
    )

    assert retained_output_positions("L0") == ((0, 0), (0, 3), (3, 0), (3, 3))
    assert retained_output_indices("L0") == (0, 3, 12, 15)
    assert retained_output_positions("L1") == (
        (0, 0),
        (0, 3),
        (3, 0),
        (3, 3),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 3),
        (2, 0),
        (2, 3),
        (3, 1),
        (3, 2),
    )
    assert retained_output_indices("L1") == (
        0,
        3,
        12,
        15,
        1,
        2,
        4,
        7,
        8,
        11,
        13,
        14,
    )


@pytest.mark.parametrize("level", ("Full", "L2", "unknown"))
def test_retained_scheduler_reference_rejects_non_sparse_routes(level):
    from saes.rtl_retained_scheduler_reference import retained_output_indices

    with pytest.raises(ValueError, match="L0 or L1"):
        retained_output_indices(level)


def test_retained_scheduler_reference_rejects_undeclared_tile_size():
    from saes.rtl_retained_scheduler_reference import retained_output_indices

    with pytest.raises(ValueError, match="only T=4"):
        retained_output_indices("L0", tile_size=8)
