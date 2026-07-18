import pytest


def test_attribute_transport_quality_gate_is_fixed_and_rejects_overrides():
    from scripts import saes_adapter_offset_attribute_transport_quality_gate as gate

    assert gate.MATERIALIZATION == (
        "conditional-adapter-offset-attribute-transport-diagnostic"
    )
    assert gate.PILOT_KIND == (
        "saes_adapter_offset_attribute_transport_selected_output_quality_pilot"
    )
    assert gate.SEED == 0
    for option, value in (
        ("--seed", "1"),
        ("--materialization", "conditional-adapter-offset-transport-diagnostic"),
    ):
        with pytest.raises(SystemExit) as exc:
            gate.main(
                [
                    "--output-dir",
                    "outputs/fixed",
                    option,
                    value,
                ]
            )
        assert exc.value.code == 2
