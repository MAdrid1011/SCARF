"""Contract tests for the DL3DV Full-route packet fidelity pilot."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _capture(*, route_counts, deletion_certificate_required=True):
    from saes.sparse_gaussian_consumer import PackedGaussianAttributes

    count = 16
    slots = torch.arange(count, dtype=torch.int64)
    packed = PackedGaussianAttributes(
        batch_indices=torch.zeros(count, dtype=torch.int64),
        dense_slots=slots,
        means=torch.zeros(count, 3),
        covariances=torch.eye(3).expand(count, -1, -1).clone(),
        harmonics=torch.zeros(count, 3, 1),
        opacities=torch.full((count,), 0.2),
        source_trace={},
        source_trace_sha256="0" * 64,
    )
    return {
        "guarded_route": SimpleNamespace(
            events={
                "route_counts": route_counts,
                "deletion_certificate_required": deletion_certificate_required,
            },
            raw_head_request_mask=torch.ones(1, 4, 4, dtype=torch.bool),
        ),
        "final_packet": SimpleNamespace(dense_slots=slots),
        "final_packed": packed,
    }


def test_packet_quality_pilot_accepts_only_the_frozen_full_route():
    from scripts.saes_sparse_packet_quality_pilot import _require_sample_zero_full_route

    assert _require_sample_zero_full_route(
        _capture(route_counts={"L0": 0, "L1": 0, "Full": 1}),
        views=1,
        height=4,
        width=4,
    ) == 16


def test_packet_quality_pilot_rejects_nonzero_or_uncertified_route():
    from scripts.saes_sparse_packet_quality_pilot import _require_sample_zero_full_route

    with pytest.raises(RuntimeError, match="frozen Full"):
        _require_sample_zero_full_route(
            _capture(route_counts={"L0": 1, "L1": 0, "Full": 0}),
            views=1,
            height=4,
            width=4,
        )
    with pytest.raises(RuntimeError, match="deletion certificate"):
        _require_sample_zero_full_route(
            _capture(
                route_counts={"L0": 0, "L1": 0, "Full": 1},
                deletion_certificate_required=False,
            ),
            views=1,
            height=4,
            width=4,
        )
