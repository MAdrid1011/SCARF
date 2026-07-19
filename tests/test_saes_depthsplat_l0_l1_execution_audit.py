"""CPU contracts for the DepthSplat source-bound materializer smoke."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


class _TargetMapping(dict):
    """Permit only the RGB removal operations used before target discard."""

    def get(self, key, default=None):
        if key != "image":
            raise AssertionError(f"target field was read: {key}")
        return super().get(key, default)

    def pop(self, key, default=None):
        if key != "image":
            raise AssertionError(f"target field was removed individually: {key}")
        return super().pop(key, default)


def _context():
    return {
        "image": torch.zeros(1, 2, 3, 4, 8),
        "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
        "near": torch.full((1, 2), 0.1),
        "far": torch.full((1, 2), 100.0),
        "index": torch.tensor([[0, 9]], dtype=torch.long),
    }


def test_target_mapping_is_discarded_without_reading_its_camera_or_index():
    from scripts.saes_depthsplat_l0_l1_execution_audit import (
        _discard_target_before_context_transfer,
    )

    target = _TargetMapping(
        image=torch.zeros(1, 4, 3, 4, 8),
        extrinsics=SimpleNamespace(poison="target-camera"),
        index=SimpleNamespace(poison="target-index"),
    )
    batch = {"context": _context(), "target": target, "scene": ["scene-0"]}

    context, loaded = _discard_target_before_context_transfer(batch, torch.device("cpu"))

    assert loaded is True
    assert "target" not in batch
    assert set(context) == set(_context())
    assert torch.equal(context["index"], torch.tensor([[0, 9]], dtype=torch.long))


def test_context_decoder_inputs_cannot_fall_back_to_target_data():
    from scripts.saes_depthsplat_l0_l1_execution_audit import _context_decoder_inputs

    context = _context()
    inputs = _context_decoder_inputs(context)

    assert set(inputs) == {"extrinsics", "intrinsics", "near", "far"}
    assert inputs["extrinsics"] is context["extrinsics"]
    assert inputs["near"] is context["near"]


def test_full_passthrough_certificate_fails_closed_when_one_field_drifts():
    from scripts.saes_depthsplat_l0_l1_execution_audit import (
        _require_bitwise_equivalent,
    )

    assert _require_bitwise_equivalent(
        {"count": 4, "bitwise_equivalent": True, "fields": {}}, "test"
    )["count"] == 4
    with pytest.raises(RuntimeError, match="not bitwise equivalent"):
        _require_bitwise_equivalent(
            {"count": 4, "bitwise_equivalent": False, "fields": {}}, "test"
        )


def test_materializer_smoke_rejects_nonpredeclared_sample_before_loading_data():
    from scripts.saes_depthsplat_l0_l1_execution_audit import (
        collect_depthsplat_l0_l1_execution_audit,
    )

    with pytest.raises(ValueError, match="sample index 0"):
        collect_depthsplat_l0_l1_execution_audit(
            sample_index=1, device=torch.device("cpu")
        )
