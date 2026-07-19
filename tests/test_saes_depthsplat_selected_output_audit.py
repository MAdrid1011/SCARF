"""Focused contracts for the target-free DepthSplat selected-output audit."""

import pytest


torch = pytest.importorskip("torch")


def test_depthsplat_audit_selection_is_per_context_view_and_fixed_to_t4_corners():
    from scripts.saes_depthsplat_selected_output_audit import _selection_mask

    mask = _selection_mask(views=2, height=8, width=12, device=torch.device("cpu"))

    assert mask.shape == (2, 8, 12)
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == 2 * (8 // 4) * (12 // 4) * 4
    assert torch.equal(mask[0], mask[1])
    assert mask[0, 0, 0]
    assert mask[0, 3, 3]
    assert mask[0, 7, 11]


def test_depthsplat_audit_rejects_any_nonpredeclared_dl3dv_sample_before_loading_data():
    from scripts.saes_depthsplat_selected_output_audit import (
        collect_depthsplat_selected_output_audit,
    )

    with pytest.raises(ValueError, match="sample index 0"):
        collect_depthsplat_selected_output_audit(
            sample_index=1, device=torch.device("cpu")
        )
