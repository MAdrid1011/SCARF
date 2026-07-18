import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch", reason="target-free SAES audit requires torch")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_root(tmp_path: Path) -> Path:
    from data.build_manifest import build
    from scripts.calibration_inputs import materialize_target_free_inputs
    from scripts.compile_protocol import canonicalize_index

    root = tmp_path / "input"
    root.mkdir()
    selection = {"scene-fixed": {"context": [0, 1], "target": [2, 3, 4, 5]}}
    selection_path = root / "audit-selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    _, summary = canonicalize_index(selection_path, _sha256(selection_path))
    sidecar = materialize_target_free_inputs(
        root / "sidecar",
        dataset="dl3dv",
        examples={
            "scene-fixed": {
                "key": "scene-fixed",
                "cameras": torch.zeros(6, 18),
                "images": [b"context-0", b"context-1", None, None, None, None],
            }
        },
        index=selection,
        source={"prepared_source_sha256": "a" * 64},
        selection_sha256=summary["sample_selection_sha256"],
    )
    record = {
        "kind": "dl3dv_target_free_l1_primary_reference_audit_input",
        "status": "PASS",
        "model": "transplat",
        "dataset": "dl3dv",
        "source_sample_index": 0,
        "target_rgb_included": False,
        "target_rgb_opened": False,
        "target_rgb_paths_passed_to_encoder": False,
        "selected_sample": {
            "scene": "scene-fixed",
            "context_indices": [0, 1],
            "target_indices": [2, 3, 4, 5],
            "audit_selection_sha256": summary["sample_selection_sha256"],
        },
        "canonical_protocol": {
            "pair": "transplat/dl3dv",
            "source_index_sha256": "b" * 64,
        },
        "opened_source_files": [],
        "sidecar": sidecar,
    }
    (root / "audit-input.json").write_text(json.dumps(record), encoding="utf-8")
    manifest = build(root, "dl3dv-target-free-audit", "fixture", "fixture")
    (root / ".scarf-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_audit_input_validator_accepts_only_a_target_free_fixed_contract(tmp_path: Path):
    from scripts.saes_l1_primary_reference_attribute_audit import _load_audit_input

    root = _input_root(tmp_path)
    record, sidecar, tree = _load_audit_input(root)

    assert record["selected_sample"]["scene"] == "scene-fixed"
    assert sidecar["target_rgb_included"] is False
    assert tree["dataset"] == "dl3dv-target-free-audit"

    tampered = json.loads((root / "audit-input.json").read_text(encoding="utf-8"))
    tampered["target_rgb_opened"] = True
    (root / "audit-input.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed RGB contract"):
        _load_audit_input(root)


def test_fixed_attribute_audit_uses_a_target_free_loader_and_never_decodes_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import scripts.ae_config as ae_config
    import scripts.demo as demo
    import scripts.saes_l1_primary_reference_attribute_audit as audit

    root = _input_root(tmp_path)
    checkpoint = tmp_path / "transplat.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fake_loader(*args, **kwargs):
        calls.append((args, kwargs))
        context = {
            "image": torch.ones(1, 1, 3, 4, 4),
            "extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
            "intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
            "near": torch.ones(1, 1),
            "far": torch.full((1, 1), 10.0),
        }
        return (
            SimpleNamespace(eval=lambda: None),
            {"context": context, "target": {}, "scene": ["scene-fixed"]},
            None,
            torch.device("cpu"),
        )

    source = SimpleNamespace(
        means=torch.ones(1, 16, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1),
        harmonics=torch.full((1, 16, 3, 1), 0.5),
        opacities=torch.full((1, 16), 0.25),
    )
    features = torch.ones(1, 1, 2, 4, 4)
    depths = torch.ones(1, 1, 16, 1, 1)

    def fake_apply(gaussians, *args, **kwargs):
        mask = torch.ones(16, dtype=torch.bool)
        mask[torch.tensor([0, 3, 12, 15])] = False
        return mask, {
            "covariance_psd_violations": 0,
            "guard_nonprobe_s3_attribute_reads": 0,
            "opacity_transmittance_error_max": 0.0,
            "assignment_weight_sum_error_max": 0.0,
        }, None

    monkeypatch.setattr(demo, "load_model_and_data", fake_loader)
    monkeypatch.setattr(
        ae_config,
        "resolve_experiment",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint=checkpoint, experiment="dl3dv", hydra_overrides=()
        ),
    )
    monkeypatch.setattr(audit, "_context_on_device", lambda batch, _device: batch["context"])
    monkeypatch.setattr(
        audit,
        "_capture_encoder_execution",
        lambda _model, _context: (source, features, depths),
    )
    monkeypatch.setattr(audit, "apply_progressive_saes", fake_apply)

    record = audit.collect_attribute_audit(input_root=root, device=torch.device("cpu"))

    assert calls[0][1]["calibration_target_free"] is True
    assert calls[0][1]["encoder_only"] is True
    assert calls[0][1]["dataset_root"] == root / "sidecar"
    assert record["target_rgb_accessed"] is False
    assert record["execution_boundary"]["decoder_executed"] is False
    assert record["execution_boundary"]["quality_metrics_computed"] is False
    assert record["posthoc_full_stage3_oracle"]["route_source"] == "committed_sparse_output_mask"


def test_fixed_attribute_audit_cli_rejects_model_and_threshold_overrides():
    from scripts.saes_l1_primary_reference_attribute_audit import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--input-root",
                "inputs/fixed",
                "--output-dir",
                "outputs/fixed",
                "--saes-fv",
                "0.3",
            ]
        )
    assert exc.value.code == 2
