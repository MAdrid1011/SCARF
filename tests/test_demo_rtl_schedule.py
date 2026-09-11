from __future__ import annotations


class _FeatureBackbone:
    layer1 = (object(), object())
    layer2 = (object(), object())
    layer3 = (object(), object())


class _Transformer:
    layers = tuple(object() for _ in range(6))


class _Backbone:
    backbone = _FeatureBackbone()
    transformer = _Transformer()


class _Encoder:
    backbone = _Backbone()


class _PretrainedDino:
    n_blocks = 12


class _Dino:
    pretrained = _PretrainedDino()


def test_rtl_schedule_uses_loaded_graph_and_executed_auxiliary_path():
    from scripts.demo import _rtl_encoder_schedule

    mvsplat = _rtl_encoder_schedule(_Encoder(), has_executed_dinov2=False)
    trans_encoder = _Encoder()
    trans_encoder.da_model = _Dino()
    transplat = _rtl_encoder_schedule(trans_encoder, has_executed_dinov2=True)

    assert mvsplat == {
        "cnn_layers": 6,
        "transformer_layers": 6,
        "has_dinov2": False,
        "dinov2_layers": 0,
    }
    assert transplat["cnn_layers"] == mvsplat["cnn_layers"]
    assert transplat["transformer_layers"] == mvsplat["transformer_layers"]
    assert transplat["has_dinov2"] is True
    assert transplat["dinov2_layers"] == 12
