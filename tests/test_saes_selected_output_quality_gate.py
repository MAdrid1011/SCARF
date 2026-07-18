def test_quality_pilot_is_bound_to_the_preregistered_adapter_offset_diagnostic():
    from scripts import saes_selected_output_quality_gate as gate

    assert gate.MATERIALIZATION == "conditional-adapter-offset-transport-diagnostic"
    assert gate.DECISION_SEMANTICS == "probe-normalized-std-first-hit"
    assert gate.DEPTH_ROUTING_SEMANTICS == "metric-depth-standard-deviation"
    assert gate.QUALITY_LIMITS == {
        "psnr_loss_db": 0.15,
        "ssim_loss": 0.005,
        "lpips_increase": 0.005,
    }
