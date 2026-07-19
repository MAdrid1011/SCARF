import pytest
import torch


def _mask(*positions: tuple[int, int]) -> torch.Tensor:
    value = torch.zeros((1, 4, 4), dtype=torch.bool)
    for row, column in positions:
        value[0, row, column] = True
    return value


def _root_values(model: str) -> dict[str, object]:
    from saes.dependency_aware_sparse_execution import root_input_contract

    return {
        field: object()
        for field in root_input_contract(model).required_fields
    }


@pytest.mark.parametrize("model", ("transplat", "mvsplat", "depthsplat"))
def test_root_contract_accepts_only_declared_pre_s2_inputs(model: str):
    from saes.dependency_aware_sparse_execution import (
        validate_sparse_producer_root_inputs,
    )

    validated = validate_sparse_producer_root_inputs(model, _root_values(model))

    assert validated.model == model
    assert set(validated.field_names) == set(_root_values(model))


@pytest.mark.parametrize("forbidden", ("depths", "refine_out", "raw_gaussians", "target_rgb"))
def test_root_contract_rejects_dense_s2_s3_and_target_inputs(forbidden: str):
    from saes.dependency_aware_sparse_execution import (
        validate_sparse_producer_root_inputs,
    )

    values = _root_values("transplat")
    values[forbidden] = torch.zeros((1, 1, 4, 4))

    with pytest.raises(ValueError, match="forbidden"):
        validate_sparse_producer_root_inputs("transplat", values)


def test_root_contract_rejects_missing_or_unrecognized_fields():
    from saes.dependency_aware_sparse_execution import (
        validate_sparse_producer_root_inputs,
    )

    missing = _root_values("mvsplat")
    missing.pop("primary_request_mask")
    with pytest.raises(ValueError, match="missing required"):
        validate_sparse_producer_root_inputs("mvsplat", missing)

    unexpected = _root_values("mvsplat")
    unexpected["dense_intermediate"] = object()
    with pytest.raises(ValueError, match="not permitted"):
        validate_sparse_producer_root_inputs("mvsplat", unexpected)


def _ledger():
    from saes.dependency_aware_sparse_execution import (
        DependencyAwarePhaseLedger,
        validate_sparse_producer_root_inputs,
    )

    return DependencyAwarePhaseLedger(
        validate_sparse_producer_root_inputs("mvsplat", _root_values("mvsplat"))
    )


def test_phase_ledger_charges_only_new_dependency_closure_work():
    ledger = _ledger()
    primary = _mask((0, 0), (0, 3))
    secondary = primary | _mask((1, 1))
    full = torch.ones_like(primary)

    ledger.begin_phase("primary")
    first = ledger.record_operator(
        "s2.selected_depth",
        requested_mask=primary,
        dependency_closure_mask=primary,
        executed_mask=primary,
        reused_mask=torch.zeros_like(primary),
        dependency_kind="local_spatial",
    )
    assert first["executed_positions"] == 2
    assert first["reused_positions"] == 0

    ledger.begin_phase("secondary")
    second = ledger.record_operator(
        "s2.selected_depth",
        requested_mask=secondary,
        dependency_closure_mask=secondary,
        executed_mask=secondary & ~primary,
        reused_mask=primary,
        dependency_kind="local_spatial",
    )
    assert second["executed_positions"] == 1
    assert second["reused_positions"] == 2

    ledger.begin_phase("full")
    third = ledger.record_operator(
        "s2.selected_depth",
        requested_mask=full,
        dependency_closure_mask=full,
        executed_mask=full & ~secondary,
        reused_mask=secondary,
        dependency_kind="local_spatial",
    )
    assert third["executed_positions"] == 13
    assert third["reused_positions"] == 3

    trace = ledger.finalize()
    assert trace["paper_result_eligible"] is False
    assert trace["whole_pipeline_s2_s3_sparse_execution_verified"] is False
    assert trace["strict_s2_s3_savings_claimed"] is False
    assert trace["operator_final_computed_positions"]["s2.selected_depth"] == 16
    assert sum(event["executed_positions"] for event in trace["operator_events"]) == 16


def test_phase_ledger_rejects_replaying_previously_computed_closure():
    ledger = _ledger()
    primary = _mask((0, 0))
    secondary = primary | _mask((0, 1))

    ledger.begin_phase("primary")
    ledger.record_operator(
        "s2.selected_depth",
        requested_mask=primary,
        dependency_closure_mask=primary,
        executed_mask=primary,
        reused_mask=torch.zeros_like(primary),
        dependency_kind="local_spatial",
    )
    ledger.begin_phase("secondary")

    with pytest.raises(ValueError, match="executed_mask"):
        ledger.record_operator(
            "s2.selected_depth",
            requested_mask=secondary,
            dependency_closure_mask=secondary,
            executed_mask=secondary,
            reused_mask=torch.zeros_like(secondary),
            dependency_kind="local_spatial",
        )

    event = ledger.record_operator(
        "s2.selected_depth",
        requested_mask=secondary,
        dependency_closure_mask=secondary,
        executed_mask=secondary & ~primary,
        reused_mask=primary,
        dependency_kind="local_spatial",
    )
    assert event["executed_positions"] == 1


def test_global_attention_closure_is_explicitly_dense_and_never_a_saving():
    ledger = _ledger()
    primary = _mask((0, 0))
    full = torch.ones_like(primary)

    ledger.begin_phase("primary")
    first = ledger.record_operator(
        "s2.cross_view_attention",
        requested_mask=primary,
        dependency_closure_mask=full,
        executed_mask=full,
        reused_mask=torch.zeros_like(full),
        dependency_kind="cross_view_attention",
    )
    assert first["dense_closure"] is True
    assert first["executed_positions"] == 16

    ledger.begin_phase("secondary")
    ledger.record_operator(
        "s2.cross_view_attention",
        requested_mask=primary,
        dependency_closure_mask=full,
        executed_mask=torch.zeros_like(full),
        reused_mask=full,
        dependency_kind="cross_view_attention",
    )
    ledger.begin_phase("full")
    ledger.record_operator(
        "s2.cross_view_attention",
        requested_mask=full,
        dependency_closure_mask=full,
        executed_mask=torch.zeros_like(full),
        reused_mask=full,
        dependency_kind="cross_view_attention",
    )

    trace = ledger.finalize()
    assert trace["operator_final_computed_positions"]["s2.cross_view_attention"] == 16
    assert trace["dense_closure_operators"] == ["s2.cross_view_attention"]
    assert trace["operator_events"][0]["dependency_kind"] == "cross_view_attention"
