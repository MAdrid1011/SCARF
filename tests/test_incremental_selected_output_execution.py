"""Regression tests for source-bound incremental Gaussian-head execution."""

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


def _head(*, padding_mode="zeros"):
    torch.manual_seed(71)
    return nn.Sequential(
        nn.Conv2d(3, 5, 3, 1, 1, padding_mode=padding_mode),
        nn.GELU(),
        nn.Conv2d(5, 2, 3, 1, 1, padding_mode=padding_mode),
    ).eval()


def _tile_mask(*, height, width, tile_size, positions_by_tile):
    mask = torch.zeros(1, height, width, dtype=torch.bool)
    for (tile_y, tile_x), positions in positions_by_tile.items():
        origin_y, origin_x = tile_y * tile_size, tile_x * tile_size
        for row, column in positions:
            mask[0, origin_y + row, origin_x + column] = True
    return mask


def _primary_positions():
    return ((0, 0), (0, 3), (3, 0), (3, 3))


def _l1_positions():
    return (
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 3),
        (2, 0),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    )


def test_incremental_producer_reuses_primary_secondary_and_full_outputs_per_tile():
    from saes.incremental_selected_output_execution import (
        IncrementalSelectedOutputProducer,
    )

    torch.manual_seed(73)
    head = _head()
    inputs = torch.randn(1, 3, 8, 8)
    producer = IncrementalSelectedOutputProducer(head, inputs, tile_size=4)

    primary = _tile_mask(
        height=8,
        width=8,
        tile_size=4,
        positions_by_tile={
            (0, 0): _primary_positions(),
            (0, 1): _primary_positions(),
            (1, 0): _primary_positions(),
            (1, 1): _primary_positions(),
        },
    )
    secondary = _tile_mask(
        height=8,
        width=8,
        tile_size=4,
        positions_by_tile={(0, 0): _l1_positions()},
    )
    full = _tile_mask(
        height=8,
        width=8,
        tile_size=4,
        positions_by_tile={
            (1, 1): tuple((row, column) for row in range(4) for column in range(4))
        },
    )

    primary_event = producer.execute("primary", primary)
    secondary_event = producer.execute("secondary", secondary)
    full_event = producer.execute("full", full)
    required = primary | secondary | full
    replay = producer.finalize(required)
    dense = head(inputs)

    assert primary_event["head_final_positions_executed"] == int(primary.sum())
    assert secondary_event["head_final_positions_reused"] == 4
    assert secondary_event["head_final_positions_executed"] == 8
    assert full_event["head_final_positions_reused"] == 4
    assert full_event["head_final_positions_executed"] == 12
    assert full_event["full_tile_native_identity_verified"] is False
    assert full_event["full_execution_mode"] == "native_dense_head_closure_selected_packet_no_s3_saving"
    assert {
        (entry["batch_item"], entry["tile_y"], entry["tile_x"])
        for entry in full_event["per_tile"]
    } == {(0, 1, 1)}
    assert full_event["per_tile"][0]["phase"] == "full"
    assert len(full_event["tile_trace_sha256"]) == 64
    assert torch.equal(replay.computed_mask, required)
    torch.testing.assert_close(
        replay.values[0, :, required[0]],
        dense[0, :, required[0]],
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert replay.events["head_final_positions_executed"] == int(required.sum())
    assert replay.events["tile_trace_records"] == 6
    assert len(replay.events["phase_trace_sha256"]) == 64
    assert replay.events["whole_pipeline_s2_s3_sparse_execution_verified"] is False


def test_dense_primary_closure_uses_native_first_kernel_and_selected_second_outputs():
    from saes.incremental_selected_output_execution import (
        RAW_HEAD_EXECUTION_CONTRACT,
        IncrementalSelectedOutputProducer,
    )

    torch.manual_seed(75)
    head = _head()
    inputs = torch.randn(2, 3, 8, 8)
    dense = head(inputs)
    primary = torch.zeros(2, 8, 8, dtype=torch.bool)
    for batch_item in range(2):
        for tile_y in (0, 4):
            for tile_x in (0, 4):
                for row, column in _primary_positions():
                    primary[batch_item, tile_y + row, tile_x + column] = True
    zeros = torch.zeros_like(primary)
    native_first_shapes = []
    original_first_forward = head[0].forward

    def capture_native_first(value):
        native_first_shapes.append(tuple(value.shape))
        return original_first_forward(value)

    head[0].forward = capture_native_first
    try:
        producer = IncrementalSelectedOutputProducer(head, inputs, tile_size=4)
        primary_event = producer.execute("primary", primary)
        producer.execute("secondary", zeros)
        producer.execute("full", zeros)
        replay = producer.finalize(primary)
    finally:
        head[0].forward = original_first_forward

    assert native_first_shapes == [(2, 3, 8, 8)]
    assert primary_event["first_conv_execution_mode"] == "native_dense_closure"
    assert primary_event["first_conv_native_kernel_aligned"] is True
    assert primary_event["native_dense_first_conv_positions_executed"] == 2 * 8 * 8
    assert primary_event["first_conv_positions_executed"] == 2 * 8 * 8
    assert primary_event["native_dense_second_conv_positions_executed"] == 2 * 8 * 8
    assert primary_event["second_conv_execution_mode"] == "native_dense_closure"
    assert primary_event["second_conv_native_kernel_aligned"] is True
    assert primary_event["second_conv_positions_executed"] == 2 * 8 * 8
    assert replay.events["raw_head_execution_contract"] == RAW_HEAD_EXECUTION_CONTRACT
    assert replay.events["first_conv_execution_mode"] == "native_dense_closure"
    assert replay.events["first_conv_native_kernel_aligned"] is True
    assert replay.events["second_conv_execution_mode"] == "native_dense_closure"
    assert replay.events["second_conv_native_kernel_aligned"] is True
    assert replay.events["second_conv_positions_executed"] == 2 * 8 * 8
    assert replay.events["actual_head_macs"] == replay.events["dense_head_macs"]
    assert torch.count_nonzero(
        replay.values.masked_select((~primary).unsqueeze(1).expand_as(replay.values))
    ) == 0
    torch.testing.assert_close(
        replay.values[0, :, primary[0]], dense[0, :, primary[0]], rtol=1.0e-5, atol=1.0e-5
    )
    torch.testing.assert_close(
        replay.values[1, :, primary[1]], dense[1, :, primary[1]], rtol=1.0e-5, atol=1.0e-5
    )


def test_non_dense_primary_closure_keeps_patch_first_execution():
    from saes.incremental_selected_output_execution import (
        IncrementalSelectedOutputProducer,
    )

    head = _head()
    inputs = torch.randn(1, 3, 8, 8)
    primary = torch.zeros(1, 8, 8, dtype=torch.bool)
    primary[:, 0, 0] = True
    zeros = torch.zeros_like(primary)
    native_first_calls = []
    original_first_forward = head[0].forward

    def capture_native_first(value):
        native_first_calls.append(tuple(value.shape))
        return original_first_forward(value)

    head[0].forward = capture_native_first
    try:
        producer = IncrementalSelectedOutputProducer(head, inputs, tile_size=4)
        primary_event = producer.execute("primary", primary)
        producer.execute("secondary", zeros)
        producer.execute("full", zeros)
    finally:
        head[0].forward = original_first_forward

    assert native_first_calls == []
    assert primary_event["first_conv_execution_mode"] == "incremental_patch_closure"
    assert primary_event["first_conv_native_kernel_aligned"] is False
    assert primary_event["native_dense_first_conv_positions_executed"] == 0
    assert primary_event["second_conv_execution_mode"] == "incremental_patch_selected_outputs"
    assert primary_event["second_conv_native_kernel_aligned"] is False


def test_incremental_full_fallback_does_not_repeat_first_or_second_convolution_work():
    from saes.incremental_selected_output_execution import (
        IncrementalSelectedOutputProducer,
    )

    torch.manual_seed(79)
    head = _head(padding_mode="replicate")
    inputs = torch.randn(1, 3, 8, 8)
    producer = IncrementalSelectedOutputProducer(head, inputs, tile_size=4)
    primary = _tile_mask(
        height=8,
        width=8,
        tile_size=4,
        positions_by_tile={(0, 0): _primary_positions()},
    )
    whole_view = torch.ones(1, 8, 8, dtype=torch.bool)

    producer.execute("primary", primary)
    producer.execute("secondary", primary)
    full_event = producer.execute("full", whole_view)
    replay = producer.finalize(whole_view)
    dense = head(inputs)

    assert full_event["head_final_positions_reused"] == int(primary.sum())
    assert full_event["head_final_positions_executed"] == 64 - int(primary.sum())
    assert replay.events["first_conv_positions_executed"] == 64
    assert replay.events["head_final_positions_executed"] == 64
    assert replay.events["actual_head_macs"] == replay.events["dense_head_macs"]
    torch.testing.assert_close(replay.values, dense, rtol=1.0e-5, atol=1.0e-5)


def test_incremental_producer_rejects_invalid_phase_order_and_unfinished_output():
    from saes.incremental_selected_output_execution import (
        IncrementalSelectedOutputProducer,
    )

    producer = IncrementalSelectedOutputProducer(_head(), torch.ones(1, 3, 4, 4))
    one = torch.zeros(1, 4, 4, dtype=torch.bool)
    one[:, 0, 0] = True

    with pytest.raises(ValueError, match="primary phase"):
        producer.execute("secondary", one)
    producer.execute("primary", one)
    with pytest.raises(ValueError, match="not been executed"):
        producer.finalize(torch.ones(1, 4, 4, dtype=torch.bool))


def test_scoped_incremental_execution_uses_one_head_invocation_and_restores_forward():
    from saes.incremental_selected_output_execution import (
        incremental_selected_output_head_execution,
    )

    torch.manual_seed(83)
    head = _head()
    original_forward = head.forward
    inputs = torch.randn(1, 3, 8, 8)
    primary = _tile_mask(
        height=8,
        width=8,
        tile_size=4,
        positions_by_tile={
            (0, 0): _primary_positions(),
            (0, 1): _primary_positions(),
            (1, 0): _primary_positions(),
            (1, 1): _primary_positions(),
        },
    )
    secondary = _tile_mask(
        height=8,
        width=8,
        tile_size=4,
        positions_by_tile={(0, 0): _l1_positions()[4:]},
    )
    full = _tile_mask(
        height=8,
        width=8,
        tile_size=4,
        positions_by_tile={
            (1, 1): tuple((row, column) for row in range(4) for column in range(4))
        },
    )
    selected = primary | secondary | full
    dense = original_forward(inputs)

    with incremental_selected_output_head_execution(
        head,
        primary_mask=primary,
        secondary_mask=secondary,
        full_mask=full,
        tile_size=4,
    ) as trace:
        output = head(inputs)
        events = trace.events
        assert events["head_forward_invocations"] == 1
        assert events["phases"][0]["phase"] == "primary"
        assert events["phases"][1]["phase"] == "secondary"
        assert events["phases"][2]["phase"] == "full"
        assert events["phases"][2]["head_final_positions_reused"] == 4
        assert events["head_final_positions_executed"] == int(selected.sum())
        torch.testing.assert_close(
            output[0, :, selected[0]],
            dense[0, :, selected[0]],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        assert torch.count_nonzero(
            output.masked_select((~selected).unsqueeze(1).expand_as(output))
        ) == 0

    torch.testing.assert_close(head(inputs), original_forward(inputs))


def test_probe_first_plan_binds_each_scoped_phase_without_repeated_output_work():
    from saes.incremental_selected_output_execution import (
        incremental_selected_output_head_execution,
    )
    from saes.probe_first_schedule import build_incremental_probe_first_plan

    features = torch.zeros(1, 1, 2, 8, 8)
    depths = torch.ones(1, 1, 8, 8)
    for tile_y in range(2):
        for tile_x in range(2):
            values = (
                ((1.0, 0.0),) * 4
                if (tile_y, tile_x) in {(0, 0), (1, 0)}
                else ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0))
            )
            for (row, column), value in zip(_primary_positions(), values):
                features[0, 0, :, tile_y * 4 + row, tile_x * 4 + column] = torch.tensor(value)
    depths[0, 0, 7, 3] = 2.0
    depths[0, 0, 7, 7] = 2.0
    plan = build_incremental_probe_first_plan(
        features,
        depths,
        height=8,
        width=8,
        tile_size=4,
        feature_threshold=0.2,
        depth_threshold=0.1,
        decision_semantics="probe-normalized-std-first-hit",
    )
    assert (
        int(plan.primary_mask.sum()),
        int(plan.secondary_mask.sum()),
        int(plan.full_mask.sum()),
        int(plan.selection_mask.sum()),
    ) == (16, 16, 16, 44)

    torch.manual_seed(97)
    head = _head()
    inputs = torch.randn(1, 3, 8, 8)
    dense = head(inputs)
    invocations = []
    handle = head.register_forward_hook(lambda *_args: invocations.append(True))
    try:
        with incremental_selected_output_head_execution(
            head,
            plan.primary_mask,
            plan.secondary_mask,
            plan.full_mask,
            tile_size=4,
        ) as trace:
            replay = head(inputs)
            events = trace.events
    finally:
        handle.remove()

    assert len(invocations) == 1
    assert events["head_forward_invocations"] == 1
    for phase, mask_key in zip(
        events["phases"],
        ("primary_mask_sha256", "secondary_mask_sha256", "full_mask_sha256"),
    ):
        assert phase["mask_sha256"] == plan.events[mask_key]
        assert (
            phase["head_final_positions_requested"]
            == phase["head_final_positions_executed"]
            + phase["head_final_positions_reused"]
        )
    assert sum(
        phase["head_final_positions_executed"] for phase in events["phases"]
    ) == int(plan.selection_mask.sum())
    assert events["head_final_positions_executed"] == int(plan.selection_mask.sum())
    assert events["first_conv_positions_executed"] <= events["dense_head_positions"]
    assert events["full_tile_native_identity_verified"] is False
    torch.testing.assert_close(
        replay[0, :, plan.full_mask[0]],
        dense[0, :, plan.full_mask[0]],
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_deferred_full_extension_uses_one_head_invocation_without_replaying_work():
    from saes.incremental_selected_output_execution import (
        incremental_selected_output_head_execution,
    )

    torch.manual_seed(101)
    head = _head()
    original_forward = head.forward
    inputs = torch.randn(1, 3, 4, 4)
    primary = _tile_mask(
        height=4,
        width=4,
        tile_size=4,
        positions_by_tile={(0, 0): _primary_positions()},
    )
    primary_set = set(_primary_positions())
    secondary = _tile_mask(
        height=4,
        width=4,
        tile_size=4,
        positions_by_tile={
            (0, 0): tuple(
                position for position in _l1_positions() if position not in primary_set
            )
        },
    )
    full = torch.zeros_like(primary)
    extension = _tile_mask(
        height=4,
        width=4,
        tile_size=4,
        positions_by_tile={(0, 0): ((1, 1), (1, 2), (2, 1), (2, 2))},
    )
    dense = original_forward(inputs)
    invocations = []
    handle = head.register_forward_hook(lambda *_args: invocations.append(True))
    try:
        with incremental_selected_output_head_execution(
            head,
            primary,
            secondary,
            full,
            tile_size=4,
            defer_full_extension=True,
        ) as trace:
            output = head(inputs)
            initial = trace.initial_events
            assert initial["execution_finalized"] is False
            assert [event["phase"] for event in initial["phases"]] == [
                "primary",
                "secondary",
                "full",
            ]
            assert initial["head_final_positions_executed"] == 12
            with pytest.raises(RuntimeError, match="must be finalized"):
                _ = trace.events

            extension_event = trace.append_full_extension(extension)
            final_replay = trace.finalize()
            events = trace.events
    finally:
        handle.remove()

    assert len(invocations) == 1
    assert extension_event["phase"] == "full_extension"
    assert extension_event["head_final_positions_requested"] == 4
    assert extension_event["head_final_positions_reused"] == 0
    assert extension_event["head_final_positions_executed"] == 4
    assert extension_event["first_conv_positions_executed"] == 0
    assert extension_event["second_conv_positions_executed"] == 0
    assert extension_event["second_conv_execution_mode"] == "native_dense_closure_reuse"
    assert extension_event["full_tile_native_identity_verified"] is False
    assert extension_event["full_execution_mode"] == "native_dense_head_closure_reuse_no_s3_saving"
    assert events["head_forward_invocations"] == 1
    assert events["execution_finalized"] is True
    assert events["full_extension_dispatched"] is True
    assert events["full_extension_positions_executed"] == 4
    assert len(events["phases"]) == 4
    assert events["phases"][-1]["phase"] == "full_extension"
    assert final_replay.events["computed_mask_sha256"] == events["computed_mask_sha256"]
    assert torch.equal(final_replay.computed_mask, torch.ones_like(primary))
    torch.testing.assert_close(output, dense, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(head(inputs), original_forward(inputs))
