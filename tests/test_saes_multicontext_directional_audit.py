import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _trace():
    return [
        {
            "view_index": 0,
            "tile_row": 0,
            "tile_column": 0,
            "routing_level_before_materialization": "L0",
            "guard_checks": [{"level": "L0", "passed": True}],
        },
        {
            "view_index": 0,
            "tile_row": 0,
            "tile_column": 1,
            "routing_level_before_materialization": "Full",
            "guard_checks": [],
        },
    ]


def _gaussians():
    count = 32
    means = torch.arange(count * 3, dtype=torch.float32).reshape(1, count, 3) / 100.0
    covariances = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, count, 1, 1) * 0.01
    harmonics = torch.ones(1, count, 3, 2)
    opacities = torch.full((1, count), 0.2)
    return SimpleNamespace(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )


def _mask():
    mask = torch.zeros(32, dtype=torch.bool)
    probes = {0, 3, 24, 27}
    for row in range(4):
        for column in range(4):
            index = row * 8 + column
            if index not in probes:
                mask[index] = True
    return mask


def _stats(*, tangent=False):
    result = {
        "total_tiles_processed": 2,
        "level0_tiles": 1,
        "level1_tiles": 0,
        "full_tiles": 1,
        "guard_nonprobe_s3_attribute_reads": 0,
    }
    result.update(
        {
            "multicontext_tangent_enabled": tangent,
            "multicontext_tangent_runtime_eligible": False,
            "multicontext_tangent_attempts": 4 if tangent else 0,
            "multicontext_tangent_accepted": 4 if tangent else 0,
            "multicontext_tangent_local_fallbacks": 0,
            "multicontext_tangent_residual_max": 0.0,
            "multicontext_tangent_fallback_reasons": {},
        }
    )
    return result


def _payload(*, tangent=False):
    from scripts.saes_multicontext_directional_audit import _build_commit_payload

    gaussians = _gaussians()
    if tangent:
        gaussians.covariances[0, 0, 0, 0] = 0.02
    return _build_commit_payload(
        gaussians=gaussians,
        mask=_mask(),
        stats=_stats(tangent=tangent),
        tile_trace=_trace(),
        materialization=(
            "multicontext-tangent-plane-diagnostic"
            if tangent
            else "conditional-adapter-offset-attribute-transport-diagnostic"
        ),
        height=4,
        width=8,
        view_count=1,
    )


def test_phase_a_commits_only_retained_descriptors_and_preserves_full_slots(
    tmp_path: Path,
):
    from scripts.saes_multicontext_directional_audit import (
        _load_descriptor_commit,
        _phase_a_invariants,
        _write_descriptor_commit,
    )

    current = _payload()
    tangent = _payload(tangent=True)
    invariants = _phase_a_invariants(current, tangent)
    manifest = _write_descriptor_commit(tmp_path, "current", current)
    loaded = _load_descriptor_commit(tmp_path, manifest)

    assert invariants["route_mask_identical"]
    assert invariants["full_slots_identical"]
    assert loaded["retained_indices"].numel() == 20
    assert loaded["means"].shape[0] == 20
    assert loaded["full_indices"].numel() == 16
    assert "dense_means" not in loaded
    assert "deleted_descriptors" not in loaded


def test_phase_a_rejects_route_event_and_full_slot_drift():
    from scripts.saes_multicontext_directional_audit import _phase_a_invariants

    current = _payload()
    route_drift = _payload(tangent=True)
    route_drift["modified_mask"][1] = False
    with pytest.raises(RuntimeError, match="route mask"):
        _phase_a_invariants(current, route_drift)

    event_drift = _payload(tangent=True)
    event_drift["saes_stats"]["full_tiles"] = 0
    with pytest.raises(RuntimeError, match="event"):
        _phase_a_invariants(current, event_drift)

    full_drift = _payload(tangent=True)
    full_location = int(full_drift["full_indices"][0].item())
    retained_location = int(
        torch.searchsorted(
            full_drift["retained_indices"], torch.tensor(full_location)
        ).item()
    )
    full_drift["covariances"][retained_location, 0, 0] = 0.03
    with pytest.raises(RuntimeError, match="Full covariances"):
        _phase_a_invariants(current, full_drift)


def test_directional_audit_entrypoint_has_only_fixed_input_device_and_output_arguments():
    from scripts.saes_multicontext_directional_audit import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--input-root",
                "inputs/fixed",
                "--output-dir",
                "outputs/fixed",
                "--sample-index",
                "1",
            ]
        )
    assert exc.value.code == 2


def test_psd_check_accepts_roundoff_symmetric_reconstruction():
    from scripts.saes_multicontext_directional_audit import _assert_psd

    basis = torch.tensor(
        ((0.8, -0.6, 0.0), (0.6, 0.8, 0.0), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
    )
    covariance = basis @ torch.diag(torch.tensor((0.01, 0.02, 0.03))) @ basis.mT
    _assert_psd(covariance.unsqueeze(0), "roundoff")


def test_sentinel_payload_equality_rejects_trace_counter_and_output_drift():
    from scripts.saes_multicontext_directional_audit import _assert_payload_identical

    reference = _payload()
    candidate = _payload()
    assert _assert_payload_identical(reference, candidate, label="sentinel") == {
        "means": 0.0,
        "covariances": 0.0,
        "harmonics": 0.0,
        "opacities": 0.0,
    }

    trace_drift = _payload()
    trace_drift["tile_trace"][0]["routing_level_before_materialization"] = "L1"
    with pytest.raises(RuntimeError, match="tile_trace"):
        _assert_payload_identical(reference, trace_drift, label="sentinel")

    counter_drift = _payload()
    counter_drift["saes_stats"]["full_tiles"] = 2
    with pytest.raises(RuntimeError, match="saes_stats"):
        _assert_payload_identical(reference, counter_drift, label="sentinel")

    output_drift = _payload()
    output_drift["covariances"][0, 0, 0] = 0.03
    with pytest.raises(RuntimeError, match="covariances"):
        _assert_payload_identical(reference, output_drift, label="sentinel")


def test_directional_audit_rejects_any_nonfrozen_input_identity(monkeypatch):
    import scripts.saes_multicontext_directional_audit as audit

    identity = copy.deepcopy(audit.FIXED_INPUT_IDENTITY)
    monkeypatch.setattr(
        audit, "validate_context_only_audit_input", lambda _root: identity
    )
    assert audit._require_fixed_audit_input(Path("inputs/fixed")) == identity

    identity["source_binding"]["canonical_index_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="canonical_index_sha256"):
        audit._require_fixed_audit_input(Path("inputs/fixed"))


def test_checkpoint_hash_is_verified_before_model_loading(monkeypatch):
    import scripts.saes_multicontext_directional_audit as audit

    def bad_checkpoint(*_args, **_kwargs):
        raise RuntimeError("bad checkpoint")

    monkeypatch.setattr(
        audit,
        "_require_fixed_audit_input",
        lambda _root: copy.deepcopy(audit.FIXED_INPUT_IDENTITY),
    )
    monkeypatch.setattr(
        audit,
        "resolve_experiment",
        lambda *_args: SimpleNamespace(
            checkpoint=Path("checkpoints/fixed.ckpt"),
            experiment="re10k",
            hydra_overrides=(),
            environment_profile="classic",
        ),
    )
    monkeypatch.setattr(
        audit,
        "require_fixed_file_sha256",
        bad_checkpoint,
    )
    monkeypatch.setattr(
        audit,
        "create_model_loader",
        lambda *_args: pytest.fail("model loader ran before checkpoint validation"),
    )

    with pytest.raises(RuntimeError, match="bad checkpoint"):
        audit._load_encoder_context(Path("inputs/fixed"), torch.device("cpu"))
