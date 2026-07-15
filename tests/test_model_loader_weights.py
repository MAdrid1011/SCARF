import pytest


torch = pytest.importorskip("torch")


def test_checkpoint_loader_records_exact_tensor_coverage():
    from integration.model_loader import load_checkpoint_state

    model = torch.nn.Linear(3, 2)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    report = load_checkpoint_state(model, {"state_dict": state})

    assert report["matched_tensors"] == len(state)
    assert report["unmatched_checkpoint_tensors"] == 0
    assert report["matched_checkpoint_numel_fraction"] == 1.0
    assert model._scarf_checkpoint_load == report


def test_checkpoint_loader_rejects_empty_or_shape_mismatched_state():
    from integration.model_loader import load_checkpoint_state

    model = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="no tensor matching"):
        load_checkpoint_state(model, {"state_dict": {"other": torch.ones(1)}})
    with pytest.raises(ValueError, match="shape mismatch"):
        load_checkpoint_state(
            model, {"state_dict": {"weight": torch.ones(7, 7)}}
        )
