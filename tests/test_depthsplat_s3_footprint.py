import pytest


def test_l0_footprint_has_only_final_head_sparsity_on_the_dl3dv_shape():
    from saes.depthsplat_s3_footprint import depthsplat_s3_required_footprint

    footprint = depthsplat_s3_required_footprint(height=256, width=448, level="L0")

    assert footprint["retained_final_head_positions"] == 28_672
    assert footprint["total_spatial_positions"] == 114_688
    assert footprint["all_preceding_s3_positions_required"] is True
    assert footprint["reverse_conv3_requirements"] == (
        {
            "layer": "gaussian_head.2",
            "required_output_positions": 28_672,
            "required_input_positions": 114_688,
        },
        {
            "layer": "gaussian_head.0",
            "required_output_positions": 114_688,
            "required_input_positions": 114_688,
        },
        {
            "layer": "gaussian_regressor.2",
            "required_output_positions": 114_688,
            "required_input_positions": 114_688,
        },
        {
            "layer": "gaussian_regressor.0",
            "required_output_positions": 114_688,
            "required_input_positions": 114_688,
        },
    )


def test_l1_footprint_also_requires_every_preceding_s3_position():
    from saes.depthsplat_s3_footprint import depthsplat_s3_required_footprint

    footprint = depthsplat_s3_required_footprint(height=256, width=448, level="L1")

    assert footprint["retained_final_head_positions"] == 57_344
    assert footprint["all_preceding_s3_positions_required"] is True
    assert footprint["reverse_conv3_requirements"][0] == {
        "layer": "gaussian_head.2",
        "required_output_positions": 57_344,
        "required_input_positions": 114_688,
    }


def test_conv3_footprint_clamps_replicate_padded_boundaries():
    from saes.depthsplat_s3_footprint import backward_same_conv3_footprint

    assert backward_same_conv3_footprint({(0, 0)}, height=4, width=4) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"height": 255, "width": 448, "level": "L0"}, "divisible by T=4"),
        ({"height": 256, "width": 448, "level": "Full"}, "L0 or L1"),
    ),
)
def test_depthsplat_s3_footprint_rejects_unsupported_contracts(kwargs, message):
    from saes.depthsplat_s3_footprint import depthsplat_s3_required_footprint

    with pytest.raises(ValueError, match=message):
        depthsplat_s3_required_footprint(**kwargs)
