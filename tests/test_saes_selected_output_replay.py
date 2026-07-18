"""Unit tests for exact same-weight selected Gaussian-head replay."""

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


def _head(in_channels=3, hidden_channels=5, out_channels=2):
    torch.manual_seed(17)
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
        nn.GELU(),
        nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
    )


def _four_corner_mask(height, width):
    from scripts.saes_dependency_audit import build_probe_mask

    return build_probe_mask(height=height, width=width, tile_size=4)


def test_replay_matches_dense_head_at_selected_outputs_including_edges():
    from saes.selected_output_replay import (
        compare_selected_outputs,
        replay_two_conv_selected_outputs,
    )

    head = _head()
    torch.manual_seed(23)
    inputs = torch.randn(2, 3, 8, 12)
    mask = _four_corner_mask(8, 12)
    dense = head(inputs)
    replay = replay_two_conv_selected_outputs(head, inputs, mask)
    comparison = compare_selected_outputs(dense, replay)

    assert comparison["equivalent"] is True
    assert comparison["maximum_absolute_delta"] < 1.0e-6
    assert replay.values.shape == (2, 2, int(mask.sum()))


def test_replay_event_ledger_charges_dense_first_conv_closure_and_selected_second_conv():
    from saes.selected_output_replay import replay_two_conv_selected_outputs

    head = _head()
    replay = replay_two_conv_selected_outputs(
        head, torch.ones(1, 3, 8, 8), _four_corner_mask(8, 8)
    )

    events = replay.events
    assert events["dense_spatial_positions"] == 64
    assert events["selected_final_output_positions"] == 16
    assert events["first_conv_required_output_positions"] == 64
    assert events["first_conv_dense_closure"] is True
    assert events["second_conv_selected_only"] is True
    assert events["upstream_s2_saving"] == 0.0
    assert events["replayed_head_macs"] < events["dense_head_macs"]
    assert events["head_mac_saving"] == pytest.approx(0.3)


def test_replay_rejects_unsupported_head_structure():
    from saes.selected_output_replay import replay_two_conv_selected_outputs

    unsupported = nn.Sequential(
        nn.Conv2d(3, 5, 1), nn.GELU(), nn.Conv2d(5, 2, 3, 1, 1)
    )
    with pytest.raises(ValueError, match="3x3"):
        replay_two_conv_selected_outputs(
            unsupported, torch.ones(1, 3, 4, 4), _four_corner_mask(4, 4)
        )


def test_replay_rejects_an_empty_selection():
    from saes.selected_output_replay import replay_two_conv_selected_outputs

    with pytest.raises(ValueError, match="at least one"):
        replay_two_conv_selected_outputs(
            _head(), torch.ones(1, 3, 4, 4), torch.zeros(4, 4, dtype=torch.bool)
        )


def test_batched_replay_uses_per_view_masks_and_zero_fills_unexecuted_outputs():
    from saes.selected_output_replay import replay_two_conv_selected_output_maps

    head = _head()
    torch.manual_seed(29)
    inputs = torch.randn(2, 3, 4, 4)
    masks = torch.zeros(2, 4, 4, dtype=torch.bool)
    masks[0] = True
    masks[1, 0, 0] = True
    masks[1, 3, 3] = True

    replay = replay_two_conv_selected_output_maps(head, inputs, masks)
    dense = head(inputs)
    assert torch.allclose(replay.values[0], dense[0], rtol=1.0e-5, atol=1.0e-5)
    assert torch.allclose(
        replay.values[1, :, 0, 0], dense[1, :, 0, 0], rtol=1.0e-5, atol=1.0e-5
    )
    assert torch.allclose(
        replay.values[1, :, 3, 3], dense[1, :, 3, 3], rtol=1.0e-5, atol=1.0e-5
    )
    assert torch.count_nonzero(replay.values[1, :, 1:3, :]) == 0
    assert replay.events["dense_batch_items"] == 1
    assert replay.events["selected_output_batch_items"] == 1
    assert replay.events["omitted_final_output_positions"] == 14
    assert replay.events["actual_head_macs"] < replay.events["dense_head_macs"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quality_path_shape_map_scatter_matches_direct_per_view_replay():
    """Exercise the two-view 256x256 head layout used by the quality path.

    The map wrapper has to preserve both a different mask for each context
    view and the row-major selected-output placement.  Its exact comparison is
    against independently replayed views, which isolates batch routing and
    scatter from CUDA's shape-dependent convolution accumulation order.  The
    dense native comparison deliberately uses the fixed raw-head FP32 contract
    rather than asserting bitwise equality across different convolution shapes.
    """
    from saes.selected_output_replay import (
        FP32_ATOL,
        FP32_RTOL,
        dense_selected_outputs,
        replay_two_conv_selected_output_maps,
        replay_two_conv_selected_outputs,
    )
    from scripts.saes_selected_output_replay_audit import (
        strict_fp32_convolution_execution,
    )

    device = torch.device("cuda")
    torch.manual_seed(47)
    head = _head(163, 168, 84).to(device).eval()
    inputs = torch.randn(2, 163, 256, 256, device=device)
    masks = torch.ones(2, 256, 256, dtype=torch.bool, device=device)
    # Both views take the selected-output branch but have different omissions.
    # The retained neighborhoods still close the first 3x3 convolution densely.
    masks[0, 1::8, 1::8] = False
    masks[1, 3::8, 5::8] = False

    with torch.no_grad(), strict_fp32_convolution_execution():
        dense_per_view = torch.cat([head(inputs[item : item + 1]) for item in range(2)])
        mapped = replay_two_conv_selected_output_maps(head, inputs, masks)

        for item in range(2):
            direct = replay_two_conv_selected_outputs(
                head, inputs[item : item + 1], masks[item]
            )
            mapped_selected = mapped.values[
                item, :, direct.coordinates[:, 0], direct.coordinates[:, 1]
            ]
            # The wrapper's map/scatter path does not change replay values.
            assert torch.equal(mapped_selected, direct.values[0])
            # Dense and patch convolution kernels are compared under the fixed
            # raw-head contract; this is intentionally not a decoder tolerance.
            assert torch.allclose(
                dense_selected_outputs(dense_per_view[item : item + 1], direct.coordinates),
                direct.values,
                rtol=FP32_RTOL,
                atol=FP32_ATOL,
            )

    assert torch.count_nonzero(
        mapped.values.masked_select((~masks).unsqueeze(1).expand_as(mapped.values))
    ) == 0
    assert mapped.events["batch_size"] == 2
    assert mapped.events["selected_output_batch_items"] == 2
    assert all(event["first_conv_dense_closure"] for event in mapped.events["per_item"])


def test_scoped_selected_output_execution_replaces_one_native_head_call_and_restores_it():
    from saes.selected_output_execution import selected_output_head_execution

    head = _head()
    original_forward = head.forward
    inputs = torch.ones(2, 3, 4, 4)
    masks = torch.zeros(2, 4, 4, dtype=torch.bool)
    masks[:, 0, 0] = True

    with selected_output_head_execution(head, masks) as trace:
        output = head(inputs)
        assert output.shape == (2, 2, 4, 4)
        assert trace.events["selected_output_batch_items"] == 2
        assert trace.events["omitted_final_output_positions"] == 30
    assert torch.allclose(head(inputs), original_forward(inputs))


def test_raw_head_capture_preserves_the_native_vbchw_layout():
    from scripts.saes_selected_output_replay_audit import capture_raw_head_io

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_gaussians = _head()

        def forward(self, inputs):
            return self.to_gaussians(inputs)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_predictor = Predictor()

        def forward(self, context, _global_step, deterministic=True):
            del deterministic
            return self.depth_predictor(context["head_input"])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()

    source = torch.ones(3, 3, 4, 4)
    captured_input, captured_output = capture_raw_head_io(
        Model(), {"head_input": source}, model_name="transplat"
    )

    assert captured_input.shape == (3, 3, 4, 4)
    assert captured_output.shape == (3, 2, 4, 4)


def test_audit_strict_fp32_mode_is_scoped_and_restores_global_settings():
    from scripts.saes_selected_output_replay_audit import (
        strict_fp32_convolution_execution,
    )

    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    with strict_fp32_convolution_execution() as mode:
        assert mode == {
            "cuda_matmul_tf32_enabled": False,
            "cudnn_tf32_enabled": False,
        }
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is previous_matmul
    assert torch.backends.cudnn.allow_tf32 is previous_cudnn


def test_quality_pilot_keeps_target_rgb_in_native_batch_until_metrics_phase():
    from scripts.saes_selected_output_quality_gate import (
        _take_target_rgb_for_metrics,
        _target_camera_inputs,
    )

    image = torch.rand(1, 2, 3, 4, 4)
    batch = {
        "target": {
            "image": image,
            "extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
            "intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
            "near": torch.ones(1, 1),
            "far": torch.ones(1, 1) * 10,
        }
    }

    metadata = _target_camera_inputs(batch, torch.device("cpu"))
    assert set(metadata) == {"extrinsics", "intrinsics", "near", "far"}
    assert batch["target"]["image"] is image
    assert _take_target_rgb_for_metrics(batch, torch.device("cpu")) is image
    assert "image" not in batch["target"]


def test_quality_pilot_semantic_equivalence_ignores_removed_descriptors_only():
    from types import SimpleNamespace

    from scripts.saes_selected_output_quality_gate import _active_attribute_equivalence

    dense = SimpleNamespace(
        means=torch.tensor([[[1.0], [2.0]]]),
        covariances=torch.tensor([[[[1.0]], [[2.0]]]]),
        harmonics=torch.tensor([[[[1.0]], [[2.0]]]]),
        opacities=torch.tensor([[0.2, 0.4]]),
    )
    sparse = SimpleNamespace(
        means=torch.tensor([[[1.0], [99.0]]]),
        covariances=torch.tensor([[[[1.0]], [[99.0]]]]),
        harmonics=torch.tensor([[[[1.0]], [[99.0]]]]),
        opacities=torch.tensor([[0.2, 0.0]]),
    )
    retained = torch.tensor([True, False])

    report = _active_attribute_equivalence(dense, sparse, retained)
    assert report["equivalent"] is True
    assert report["retained_descriptor_count"] == 1

    sparse.means[0, 0] = 1.1
    assert _active_attribute_equivalence(dense, sparse, retained)["equivalent"] is False


def test_quality_pilot_decoder_input_excludes_only_saes_removed_descriptors():
    from types import SimpleNamespace

    from scripts.saes_selected_output_quality_gate import _retain_renderable_gaussians

    source = SimpleNamespace(
        means=torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        covariances=torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3),
        harmonics=torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3),
        opacities=torch.tensor([[0.2, 0.0]]),
    )

    rendered = _retain_renderable_gaussians(source, torch.tensor([True, False]))
    assert rendered.means.shape == (1, 1, 3)
    torch.testing.assert_close(rendered.means, source.means[:, :1])
    torch.testing.assert_close(rendered.opacities, source.opacities[:, :1])
