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
