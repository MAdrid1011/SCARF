import pytest


torch = pytest.importorskip("torch")


def test_hardware_predictor_strict_mode_is_explicit():
    from depth_predictor.hw_depth_predictor import HWDepthPredictor

    predictor = HWDepthPredictor(device=torch.device("cpu"), model_type="mvsplat")
    assert predictor.strict_mode is False
    predictor.set_strict_mode(True)
    assert predictor.strict_mode is True


def test_bilinear_cycle_model_rejects_unknown_modes():
    from encoder import BilinearUnit

    unit = BilinearUnit()
    input_tensor = torch.zeros(1, 1, 2, 2)
    output_tensor = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="unsupported interpolation mode"):
        unit._compute_cycles(input_tensor, output_tensor, mode="unknown")


def test_depthsplat_hardware_feature_pyramid_emits_every_scale():
    from torch import nn

    from depth_predictor.hw_depth_predictor import HWDepthPredictor

    predictor = HWDepthPredictor(device=torch.device("cpu"), model_type="depthsplat")
    pyramid = nn.Module()
    pyramid.stages = nn.ModuleList(
        [
            nn.Sequential(),
            nn.Sequential(
                nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(2, 2, kernel_size=3, padding=1),
            ),
        ]
    )

    outputs, cycles = predictor._hw_ds_feature_pyramid(
        torch.randn(1, 4, 4, 4), pyramid
    )

    assert [tuple(output.shape) for output in outputs] == [
        (1, 4, 4, 4),
        (1, 2, 8, 8),
    ]
    assert cycles > 0


def test_depthsplat_regressor_channels_follow_each_model_scale():
    from types import SimpleNamespace
    from torch import nn

    from depth_predictor.hw_depth_predictor import HWDepthPredictor

    model = SimpleNamespace(
        regressor_residual=nn.ModuleList(
            [nn.Conv2d(16, 128, 1), nn.Conv2d(8, 64, 1)]
        )
    )

    assert HWDepthPredictor._depthsplat_regressor_output_channels(model, 0) == 128
    assert HWDepthPredictor._depthsplat_regressor_output_channels(model, 1) == 64


def test_reference_depth_kwargs_are_model_specific():
    from depth_predictor.hw_depth_predictor import HWDepthPredictor

    common = {
        "da_depth": object(),
        "dino_feature": object(),
        "cnn_features": object(),
        "extra_info": {},
        "deterministic": False,
    }
    transplat = HWDepthPredictor(device=torch.device("cpu"), model_type="transplat")
    mvsplat = HWDepthPredictor(device=torch.device("cpu"), model_type="mvsplat")

    assert "da_depth" in transplat._reference_model_kwargs(common)
    assert "dino_feature" in transplat._reference_model_kwargs(common)
    assert "da_depth" not in mvsplat._reference_model_kwargs(common)
    assert "dino_feature" not in mvsplat._reference_model_kwargs(common)


def test_module_cycle_trace_uses_executed_layer_shapes():
    from torch import nn

    from depth_predictor.module_cycle_trace import run_module_with_cycle_trace

    module = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.GELU(),
        nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2),
    )
    input_tensor = torch.randn(1, 3, 8, 8)

    traced = run_module_with_cycle_trace(module, input_tensor)

    assert traced.output.shape == (1, 2, 16, 16)
    assert traced.total_cycles == sum(traced.breakdown.values())
    assert traced.breakdown["conv2d"] > 0
    assert traced.breakdown["conv_transpose2d"] > 0
    assert traced.breakdown["gelu"] > 0


def test_callable_cycle_trace_counts_only_selected_executed_modules():
    from torch import nn

    from depth_predictor.module_cycle_trace import (
        run_callable_with_module_cycle_trace,
    )

    selected = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.GELU())
    excluded = nn.Conv2d(4, 5, 1)
    input_tensor = torch.randn(1, 3, 8, 8)

    trace = run_callable_with_module_cycle_trace(
        lambda: excluded(selected(input_tensor)), [selected]
    )

    assert trace.output.shape == (1, 5, 8, 8)
    assert trace.total_cycles == sum(trace.breakdown.values())
    assert trace.breakdown["conv2d"] > 0
    assert trace.breakdown["gelu"] > 0


def test_gaussian_head_pre_hook_capture_is_removed_and_view_aligned():
    from torch import nn

    from depth_predictor.hw_depth_predictor import capture_first_tensor_input
    from depth_predictor.types import align_view_major_features

    head = nn.Conv2d(3, 2, kernel_size=1)
    view_major = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(
        4, 3, 2, 2
    )
    with capture_first_tensor_input(head) as captured:
        head(view_major)
    head(torch.zeros_like(view_major))

    assert len(captured) == 1
    aligned = align_view_major_features(captured[0], batch_size=2, view_count=2)
    assert aligned.shape == (2, 2, 3, 2, 2)
    assert torch.equal(aligned[0, 0], view_major[0])
    assert torch.equal(aligned[0, 1], view_major[2])
    assert torch.equal(aligned[1, 0], view_major[1])
    assert torch.equal(aligned[1, 1], view_major[3])


def test_gaussian_head_feature_alignment_rejects_incomplete_views():
    from depth_predictor.types import align_view_major_features

    with pytest.raises(ValueError, match=r"batch_size \* view_count"):
        align_view_major_features(
            torch.zeros(3, 4, 2, 2), batch_size=2, view_count=2
        )
