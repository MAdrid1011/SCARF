import pytest


torch = pytest.importorskip("torch")


def _results():
    batch, views = 1, 2
    return {
        "depth_preds": [torch.zeros(batch, views, 4, 8)],
        "match_probs": [
            torch.ones(batch * views, 8, 1, 2) / 8,
            torch.ones(batch * views, 4, 2, 4) / 4,
        ],
        "features_mv": [
            torch.arange(batch * views * 3 * 1 * 2, dtype=torch.float32).reshape(
                batch * views, 3, 1, 2
            ),
            torch.arange(batch * views * 3 * 2 * 4, dtype=torch.float32).reshape(
                batch * views, 3, 2, 4
            ),
        ],
        "features_mono_intermediate": [torch.ones(batch * views, 5, 2, 4)],
    }


def test_depthsplat_reference_tensors_bind_first_cost_volume_scale():
    from scripts.depthsplat_execution import extract_depthsplat_execution_tensors

    results = _results()
    tensors = extract_depthsplat_execution_tensors(
        results, batch_size=1, view_count=2, image_height=4, image_width=8
    )

    assert tensors.depths.shape == (1, 2, 32, 1, 1)
    assert tensors.densities.shape == (1, 2, 32, 1, 1)
    assert tensors.matching_features.shape == (1, 2, 3, 1, 2)
    assert len(tensors.matching_feature_scales) == 2
    assert tensors.matching_feature_scales[1].shape == (1, 2, 3, 2, 4)
    assert torch.equal(tensors.matching_features[0, 1], results["features_mv"][0][1])
    assert torch.equal(
        tensors.matching_feature_scales[1][0, 1], results["features_mv"][1][1]
    )
    assert tensors.mono_features.shape == (2, 5, 2, 4)


def test_depthsplat_reference_tensors_reject_unaligned_matching_feature():
    from scripts.depthsplat_execution import extract_depthsplat_execution_tensors

    results = _results()
    results["features_mv"][1] = torch.zeros(2, 3, 3, 4)

    with pytest.raises(ValueError, match="scale 1 must align exactly"):
        extract_depthsplat_execution_tensors(
            results, batch_size=1, view_count=2, image_height=4, image_width=8
        )
