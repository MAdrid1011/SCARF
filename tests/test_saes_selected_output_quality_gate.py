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


def test_quality_pilot_forwards_context_safety_guard(monkeypatch):
    from scripts import saes_selected_output_quality_gate as gate

    captured = {}

    def fake_apply(*_args, **kwargs):
        captured.update(kwargs)
        return "modified", {"context_safety_guard_enabled": True}, []

    monkeypatch.setattr(gate, "apply_progressive_saes", fake_apply)

    modified, stats = gate._apply_fixed_saes(
        object(),
        features=object(),
        depths=object(),
        context={"extrinsics": object(), "intrinsics": object()},
        height=4,
        width=4,
        views=1,
        materialization=gate.MATERIALIZATION,
        context_safety_guard=True,
    )

    assert modified == "modified"
    assert stats["context_safety_guard_enabled"] is True
    assert captured["context_safety_guard"] is True
    assert captured["require_deletion_certificate"] is True
    assert captured["cross_check_threshold"] == 0.015


def test_sparse_encoder_pass_loads_missing_runtime_dependencies(monkeypatch):
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    from scripts import saes_selected_output_quality_gate as gate

    loader_calls = 0

    @contextmanager
    def fake_selected_output_execution(head, selection_mask):
        assert head == ("head", "transplat")
        assert selection_mask == "selection"
        yield SimpleNamespace(events={"selected": True})

    def fake_loader():
        nonlocal loader_calls
        loader_calls += 1
        monkeypatch.setattr(gate, "_classic_raw_head", fake_classic_raw_head)
        monkeypatch.setattr(
            gate,
            "selected_output_head_execution",
            fake_selected_output_execution,
        )

    def fake_classic_raw_head(_model, kind):
        return ("head", kind)

    class Model:
        def encoder(self, context, _training, *, deterministic):
            assert context == {"context": True}
            assert _training is False
            assert deterministic is True
            return "gaussians"

    monkeypatch.setattr(gate, "_classic_raw_head", None)
    monkeypatch.setattr(gate, "selected_output_head_execution", None)
    monkeypatch.setattr(gate, "_load_runtime_dependencies", fake_loader)
    monkeypatch.setattr(
        gate,
        "_load_torch",
        lambda: SimpleNamespace(no_grad=lambda: nullcontext()),
    )

    gaussians, events = gate._sparse_encoder_pass(
        Model(), {"context": True}, "selection"
    )

    assert loader_calls == 1
    assert gaussians == "gaussians"
    assert events == {"selected": True}
