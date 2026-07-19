from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def test_final_packet_equivalence_gate_requires_consumed_geometry_and_attributes():
    from scripts.saes_sparse_consumer_render_audit import _final_packet_equivalence_gate

    geometry = {
        "camera_rotation": {"equivalent": True},
        "camera_translation": {"equivalent": True},
        "intrinsics": {"equivalent": True},
        "coordinates": {"equivalent": True},
        "depths": {"equivalent": True},
    }
    assert _final_packet_equivalence_gate(
        packet_adapter_inputs={"equivalent": True},
        geometry_inputs=geometry,
        raw_offset_equivalence={"equivalent": True},
        adapter_equivalence={"equivalent": True},
    )

    geometry["coordinates"] = {"equivalent": False}
    assert not _final_packet_equivalence_gate(
        packet_adapter_inputs={"equivalent": True},
        geometry_inputs=geometry,
        raw_offset_equivalence={"equivalent": True},
        adapter_equivalence={"equivalent": True},
    )


def test_packed_renderer_never_receives_nan_omitted_raw_head_slots():
    from saes.sparse_gaussian_consumer import PackedGaussianAttributes
    from scripts.saes_incremental_selected_output_audit import _selected_head_descriptors
    from scripts.saes_sparse_consumer_render_audit import render_packed_gaussians

    raw_head = torch.full((1, 5, 2, 2), torch.nan)
    raw_head[0, :, 0, 0] = torch.tensor((0.0, 0.0, 1.0, 2.0, 3.0))
    raw_head[0, :, 1, 1] = torch.tensor((0.0, 0.0, 4.0, 5.0, 6.0))
    selected = torch.zeros((1, 2, 2), dtype=torch.bool)
    selected[0, 0, 0] = True
    selected[0, 1, 1] = True
    gathered = _selected_head_descriptors(raw_head, selected)
    assert bool(torch.isfinite(gathered).all())

    packed = PackedGaussianAttributes(
        batch_indices=torch.zeros(2, dtype=torch.int64),
        dense_slots=torch.tensor((0, 3), dtype=torch.int64),
        means=gathered[:, :3],
        covariances=torch.eye(3).expand(2, -1, -1).clone(),
        harmonics=gathered[:, :3].reshape(2, 3, 1),
        opacities=torch.full((2,), 0.2),
        source_trace={},
        source_trace_sha256="0" * 64,
    )

    class Gaussians:
        def __init__(self, *, means, covariances, harmonics, opacities):
            self.means = means
            self.covariances = covariances
            self.harmonics = harmonics
            self.opacities = opacities

    class Decoder:
        def __init__(self):
            self.gaussian_count = None

        def forward(self, gaussians, *_args, **_kwargs):
            self.gaussian_count = gaussians.means.shape[1]
            assert bool(torch.isfinite(gaussians.means).all())
            assert bool(torch.isfinite(gaussians.covariances).all())
            assert bool(torch.isfinite(gaussians.harmonics).all())
            assert bool(torch.isfinite(gaussians.opacities).all())
            return SimpleNamespace(color=torch.ones(1, 1, 3, 2, 2))

    decoder = Decoder()
    _gaussians, color = render_packed_gaussians(
        decoder,
        packed,
        gaussians_type=Gaussians,
        target={
            "extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
            "intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
            "near": torch.ones(1, 1),
            "far": torch.full((1, 1), 10.0),
        },
        image_shape=(2, 2),
        expected_descriptor_count=2,
    )

    assert decoder.gaussian_count == 2
    assert tuple(color.shape) == (1, 1, 3, 2, 2)
