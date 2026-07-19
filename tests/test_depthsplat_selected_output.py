"""CPU contracts for DepthSplat's selected raw-head and RGB Adapter boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _trace():
    from saes.depthsplat_selected_output import DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT

    return {
        "contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
        "source_bound": True,
        "execution_scope": "depthsplat-dense-regressor-selected-gaussian-head-only",
        "adapter_side_inputs_source_bound": True,
        "adapter_side_inputs_same_scoped_invocation": True,
        "source_rgb_keyword": "input_images",
        "source_rgb_sh_initialization": True,
        "z_depth_geometry": True,
        "head_forward_invocations": 1,
    }


def _packet():
    from saes.depthsplat_selected_output import DepthSplatSparseRawPacket

    raw = torch.tensor(
        [
            [0.2, -0.3, 0.4, 1.0, 2.0, 3.0, 4.0],
            [-0.7, 0.5, -0.6, 5.0, 6.0, 7.0, 8.0],
        ],
        dtype=torch.float32,
    )
    return DepthSplatSparseRawPacket(
        descriptor_keys=torch.tensor([[0, 0, 0, 0], [0, 1, 3, 0]], dtype=torch.int64),
        raw_head_descriptors=raw,
        extrinsics=torch.eye(4).reshape(1, 4, 4).repeat(2, 1, 1),
        intrinsics=torch.eye(3).reshape(1, 3, 3).repeat(2, 1, 1),
        coordinates=torch.tensor([[0.1, 0.2], [0.7, 0.8]], dtype=torch.float32),
        depths=torch.tensor([2.0, 3.0], dtype=torch.float32),
        mapped_opacities=raw[:, 0].sigmoid(),
        source_rgb=torch.tensor([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=torch.float32),
        dense_slots=torch.tensor([0, 7], dtype=torch.int64),
        source_trace=_trace(),
    )


class _Adapter:
    d_in = 4

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        extrinsics,
        intrinsics,
        coordinates,
        depths,
        opacities,
        raw_body,
        image_shape,
        *,
        input_images,
    ):
        self.calls.append(
            {
                "extrinsics": extrinsics.detach().clone(),
                "intrinsics": intrinsics.detach().clone(),
                "coordinates": coordinates.detach().clone(),
                "depths": depths.detach().clone(),
                "opacities": opacities.detach().clone(),
                "raw_body": raw_body.detach().clone(),
                "image_shape": image_shape,
                "input_images": input_images.detach().clone(),
            }
        )
        count = raw_body.shape[2]
        rgb = input_images.permute(0, 1, 3, 4, 2).reshape(count, 3)
        means = torch.cat((coordinates, depths.unsqueeze(-1)), dim=-1)
        covariances = torch.eye(3, dtype=raw_body.dtype).reshape(1, 1, 1, 1, 1, 3, 3)
        covariances = covariances.expand(1, 1, count, 1, 1, -1, -1)
        return SimpleNamespace(
            means=means,
            covariances=covariances,
            harmonics=rgb.reshape(1, 1, count, 1, 1, 3, 1),
            opacities=opacities,
        )


def test_selected_head_replay_matches_replicate_padded_dense_outputs_and_keeps_omissions_zero():
    from saes.depthsplat_selected_output import replay_depthsplat_selected_head

    torch.manual_seed(103)
    head = torch.nn.Sequential(
        torch.nn.Conv2d(5, 7, 3, 1, 1, padding_mode="replicate"),
        torch.nn.GELU(),
        torch.nn.Conv2d(7, 7, 3, 1, 1, padding_mode="replicate"),
    )
    inputs = torch.randn(2, 5, 5, 6)
    dense = head(inputs)
    mask = torch.zeros(2, 5, 6, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 4, 5] = True
    mask[0, 2, 3] = True
    mask[1] = True

    replay = replay_depthsplat_selected_head(head, inputs, dense, mask)

    assert replay.equivalence["equivalent"] is True
    torch.testing.assert_close(
        replay.values[0, :, mask[0]], dense[0, :, mask[0]], rtol=1e-5, atol=1e-5
    )
    assert torch.count_nonzero(replay.values[0, :, ~mask[0]]) == 0
    torch.testing.assert_close(replay.values[1], dense[1], rtol=0.0, atol=0.0)
    assert replay.events["per_view"][1]["source_native_dense_head_capture"] is True
    assert replay.events["whole_pipeline_s2_s3_sparse_execution_verified"] is False


def test_depthsplat_packed_consumer_preserves_raw_prefix_rgb_and_native_adapter_shapes():
    from saes.depthsplat_selected_output import DepthSplatPackedGaussianConsumer

    adapter = _Adapter()
    packed = DepthSplatPackedGaussianConsumer(adapter).convert(
        _packet(), image_shape=(2, 4)
    )

    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call["extrinsics"].shape == (1, 1, 2, 1, 1, 4, 4)
    assert call["intrinsics"].shape == (1, 1, 2, 1, 1, 3, 3)
    assert call["coordinates"].shape == (1, 1, 2, 1, 1, 2)
    assert call["input_images"].shape == (1, 1, 3, 1, 2)
    assert call["image_shape"] == (2, 4)
    torch.testing.assert_close(
        call["raw_body"].reshape(2, 4), _packet().raw_head_descriptors[:, 3:]
    )
    rgb = call["input_images"].permute(0, 1, 3, 4, 2).reshape(2, 3)
    torch.testing.assert_close(rgb, _packet().source_rgb)
    torch.testing.assert_close(packed.dense_slots, torch.tensor([0, 7], dtype=torch.int64))
    torch.testing.assert_close(packed.means[:, 2], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(packed.harmonics[:, :, 0], _packet().source_rgb)


def test_depthsplat_packed_consumer_rejects_classic_two_prefix_descriptor_layout():
    from saes.depthsplat_selected_output import (
        DepthSplatPackedGaussianConsumer,
        DepthSplatSparseRawPacket,
    )

    packet = _packet()
    invalid_raw = packet.raw_head_descriptors[:, 1:]
    invalid = DepthSplatSparseRawPacket(
        **{
            **packet.__dict__,
            "raw_head_descriptors": invalid_raw,
            "mapped_opacities": invalid_raw[:, 0].sigmoid(),
        }
    )
    with pytest.raises(ValueError, match="body width plus 3"):
        DepthSplatPackedGaussianConsumer(_Adapter()).convert(invalid, image_shape=(2, 4))


def test_depthsplat_packet_builder_uses_source_z_depth_coordinates_and_selected_rgb():
    from depthsplat.src.geometry.projection import sample_image_grid
    from saes.depthsplat_selected_output import (
        DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT,
        DepthSplatAdapterInputs,
        DepthSplatNativeExecution,
        DepthSplatSelectedHeadReplay,
        build_depthsplat_sparse_raw_packet,
    )
    from saes.depthsplat_backend import source_native_depthsplat_coordinates

    torch.manual_seed(107)
    views, height, width, channels = 2, 2, 3, 7
    dense_raw = torch.randn(views, channels, height, width)
    full_mask = torch.ones(views, height, width, dtype=torch.bool)
    all_raw = dense_raw.permute(0, 2, 3, 1)[full_mask]
    all_coordinates = source_native_depthsplat_coordinates(
        all_raw, full_mask, sample_image_grid=sample_image_grid
    ).reshape(views, height * width, 2)
    depth = torch.rand(1, views, height * width, 1, 1) + 1.0
    rgb = torch.rand(1, views, 3, height, width)
    adapter_inputs = DepthSplatAdapterInputs(
        extrinsics=torch.eye(4).reshape(1, 1, 1, 1, 1, 4, 4).repeat(1, views, 1, 1, 1, 1, 1),
        intrinsics=torch.eye(3).reshape(1, 1, 1, 1, 1, 3, 3).repeat(1, views, 1, 1, 1, 1, 1),
        coordinates=all_coordinates.reshape(1, views, height * width, 1, 1, 2),
        depths=depth,
        opacities=dense_raw[:, 0].sigmoid().reshape(1, views, height * width, 1, 1),
        raw_body=dense_raw[:, 3:].permute(0, 2, 3, 1).reshape(1, views, height * width, 1, 1, 4),
        image_shape=(height, width),
        input_images=rgb,
    )
    slots = views * height * width
    dense_gaussians = SimpleNamespace(
        means=torch.zeros(1, slots, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, slots, 1, 1),
        harmonics=torch.zeros(1, slots, 3, 1),
        opacities=torch.zeros(1, slots),
    )
    execution = DepthSplatNativeExecution(
        dense_gaussians=dense_gaussians,
        gaussian_head_input=torch.zeros(views, 4, height, width),
        dense_raw_head=dense_raw,
        adapter_inputs=adapter_inputs,
        sample_image_grid=sample_image_grid,
        events={
            "gaussian_head": {"weight_sha256": "a" * 64},
            "gaussian_regressor": {"weight_sha256": "b" * 64},
        },
    )
    selection = torch.zeros_like(full_mask)
    selection[0, 0, 1] = True
    selection[1, 1, 2] = True
    sparse_map = torch.zeros_like(dense_raw)
    sparse_map.permute(0, 2, 3, 1)[selection] = dense_raw.permute(0, 2, 3, 1)[selection]
    replay = DepthSplatSelectedHeadReplay(
        values=sparse_map,
        selection_mask=selection,
        events={"contract_version": DEPTHSPLAT_SELECTED_OUTPUT_CONTRACT},
        equivalence={"equivalent": True},
    )

    packet = build_depthsplat_sparse_raw_packet(execution, replay)

    positions = selection.nonzero(as_tuple=False)
    expected_raw = dense_raw.permute(0, 2, 3, 1)[selection]
    expected_rgb = rgb[0, positions[:, 0], :, positions[:, 1], positions[:, 2]]
    torch.testing.assert_close(packet.raw_head_descriptors, expected_raw)
    torch.testing.assert_close(packet.mapped_opacities, expected_raw[:, 0].sigmoid())
    torch.testing.assert_close(packet.source_rgb, expected_rgb)
    torch.testing.assert_close(
        packet.coordinates,
        source_native_depthsplat_coordinates(
            expected_raw, selection, sample_image_grid=sample_image_grid
        ),
    )
    assert packet.dense_slots.tolist() == [1, 11]
