"""Tests for the selected raw-head packet consumer boundary."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


class _Adapter:
    d_in = 3

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        extrinsics,
        intrinsics,
        coordinates,
        depths,
        opacities,
        raw_gaussians,
        image_shape,
    ):
        self.calls.append(
            {
                "extrinsics": extrinsics.detach().clone(),
                "intrinsics": intrinsics.detach().clone(),
                "coordinates": coordinates.detach().clone(),
                "depths": depths.detach().clone(),
                "opacities": opacities.detach().clone(),
                "raw_gaussians": raw_gaussians.detach().clone(),
                "image_shape": image_shape,
            }
        )
        count = depths.numel()
        return SimpleNamespace(
            means=torch.cat((coordinates, depths.unsqueeze(-1)), dim=-1),
            covariances=torch.eye(3).expand(count, -1, -1).clone(),
            harmonics=raw_gaussians.reshape(count, 3, 1),
            opacities=opacities.clone(),
        )


def _packet():
    from saes.sparse_gaussian_consumer import SparseRawGaussianPacket

    descriptor_keys = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 3, 0], [0, 1, 1, 0]], dtype=torch.int64
    )
    primitive_keys = torch.tensor(
        [[0, 0, 0, 0, 0], [0, 0, 3, 0, 0], [0, 1, 1, 0, 0]],
        dtype=torch.int64,
    )
    return SparseRawGaussianPacket(
        descriptor_keys=descriptor_keys,
        raw_descriptors=torch.tensor(
            [[0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 4.0, 5.0, 6.0], [0.0, 0.0, 7.0, 8.0, 9.0]]
        ),
        primitive_keys=primitive_keys,
        primitive_to_descriptor=torch.tensor([0, 1, 2], dtype=torch.int64),
        coordinates=torch.tensor([[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]]),
        depths=torch.tensor([1.0, 2.0, 3.0]),
        mapped_opacities=torch.tensor([0.1, 0.2, 0.3]),
        dense_slots=torch.tensor([0, 3, 5], dtype=torch.int64),
        source_trace={
            "schema_version": "saes-incremental-head-execution-v1",
            "source_bound": True,
            "execution_scope": "s3_raw_gaussian_head_only",
            "adapter_side_inputs_source_bound": True,
            "adapter_side_inputs_same_scoped_invocation": True,
            "head_forward_invocations": 1,
            "head_weight_sha256": "a" * 64,
            "head_final_positions_executed": 3,
        },
    )


def _cameras():
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    extrinsics[0, 0, 0, 3] = 11.0
    extrinsics[0, 1, 1, 3] = 22.0
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    return extrinsics, intrinsics


def test_packed_consumer_uses_selected_packet_and_preserves_dense_slot_order():
    from saes.sparse_gaussian_consumer import PackedGaussianConsumer

    adapter = _Adapter()
    extrinsics, intrinsics = _cameras()
    packed = PackedGaussianConsumer(adapter).convert(
        _packet(), extrinsics=extrinsics, intrinsics=intrinsics, image_shape=(2, 2)
    )

    assert packed.dense_slots.tolist() == [0, 3, 5]
    assert packed.batch_indices.tolist() == [0, 0, 0]
    assert packed.source_trace["source_bound"] is True
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call["image_shape"] == (2, 2)
    assert call["extrinsics"][:, 0, 3].tolist() == [11.0, 11.0, 0.0]
    assert call["extrinsics"][:, 1, 3].tolist() == [0.0, 0.0, 22.0]
    torch.testing.assert_close(
        call["coordinates"],
        torch.tensor([[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]]),
    )
    torch.testing.assert_close(packed.means[:, 2], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(packed.opacities, torch.tensor([0.1, 0.2, 0.3]))


def test_packed_consumer_rejects_noncanonical_slots_and_non_source_bound_packets():
    from saes.sparse_gaussian_consumer import PackedGaussianConsumer, SparseRawGaussianPacket

    adapter = _Adapter()
    extrinsics, intrinsics = _cameras()
    packet = _packet()
    duplicate = SparseRawGaussianPacket(
        **{**packet.__dict__, "dense_slots": torch.tensor([0, 0, 5], dtype=torch.int64)}
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        PackedGaussianConsumer(adapter).convert(
            duplicate, extrinsics=extrinsics, intrinsics=intrinsics, image_shape=(2, 2)
        )

    unbound = SparseRawGaussianPacket(
        **{**packet.__dict__, "source_trace": {"source_bound": False}}
    )
    with pytest.raises(ValueError, match="source-bound"):
        PackedGaussianConsumer(adapter).convert(
            unbound, extrinsics=extrinsics, intrinsics=intrinsics, image_shape=(2, 2)
        )

    unbound_side_inputs = SparseRawGaussianPacket(
        **{
            **packet.__dict__,
            "source_trace": {
                **packet.source_trace,
                "adapter_side_inputs_source_bound": False,
            },
        }
    )
    with pytest.raises(ValueError, match="Adapter side inputs"):
        PackedGaussianConsumer(adapter).convert(
            unbound_side_inputs,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=(2, 2),
        )

    cross_invocation = SparseRawGaussianPacket(
        **{
            **packet.__dict__,
            "source_trace": {
                **packet.source_trace,
                "adapter_side_inputs_same_scoped_invocation": False,
            },
        }
    )
    with pytest.raises(ValueError, match="same-invocation"):
        PackedGaussianConsumer(adapter).convert(
            cross_invocation,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=(2, 2),
        )

    repeated_head = SparseRawGaussianPacket(
        **{
            **packet.__dict__,
            "source_trace": {**packet.source_trace, "head_forward_invocations": 2},
        }
    )
    with pytest.raises(ValueError, match="exactly one"):
        PackedGaussianConsumer(adapter).convert(
            repeated_head,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=(2, 2),
        )


def test_packed_consumer_is_fail_closed_for_variable_batch_packing():
    from saes.sparse_gaussian_consumer import PackedGaussianConsumer

    adapter = _Adapter()
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 2, 1, 1)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 2, 1, 1)
    with pytest.raises(ValueError, match="batch size one"):
        PackedGaussianConsumer(adapter).convert(
            _packet(), extrinsics=extrinsics, intrinsics=intrinsics, image_shape=(2, 2)
        )
