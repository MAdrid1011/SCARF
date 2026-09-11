import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_has_portable_claim_provenance():
    from scripts.mechanism_config import load_mechanism_config
    from scripts.saes_execution_identity import build_saes_execution_identity

    config, provenance = load_mechanism_config(
        ROOT / "artifact/mechanism_config.json"
    )

    assert config["status"] == "calibrated"
    assert provenance["status"] == "calibrated"
    assert provenance["global_configuration"] is True
    assert provenance["evaluation_disjoint"] is True
    assert len(provenance["candidate_records_sha256"]) == 64
    assert len(provenance["mechanism_config_sha256"]) == 64
    assert len(provenance["manifest_sha256"]) == 64
    assert config["fixed"]["saes_l1_depth_reference"] == "primary-routing-probes-v1"
    assert config["fixed"]["saes_moment_geometry"] == "c2w-probe-depth-ray-v1"
    assert config["saes_execution_identity"] == build_saes_execution_identity()
    assert config["saes_execution_identity"]["l1_anchor_count"] == 12
    assert config["saes_execution_identity"]["context_safety_guard"] is True
    assert provenance["saes_execution_route_sha256"] == config[
        "saes_execution_identity"
    ]["route_sha256"]


def test_execution_identity_builder_is_fresh_and_hash_bound():
    from scripts.saes_execution_identity import build_saes_execution_identity

    first = build_saes_execution_identity()
    second = build_saes_execution_identity()

    assert first == second
    assert first is not second
    first["l1_anchor_positions"][4][0] = 9
    assert build_saes_execution_identity() == second


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.pop("saes_execution_identity"),
            "SAES execution identity is missing",
        ),
        (
            lambda value: value["saes_execution_identity"].update(
                l1_anchor_count=8
            ),
            "registered 12-anchor",
        ),
        (
            lambda value: value["saes_execution_identity"].update(
                context_safety_guard=False
            ),
            "registered 12-anchor",
        ),
        (
            lambda value: value["saes_execution_identity"].update(
                route_sha256="0" * 64
            ),
            "registered 12-anchor",
        ),
    ),
)
def test_mechanism_config_rejects_execution_identity_drift(tmp_path, mutate, message):
    from scripts.mechanism_config import load_mechanism_config

    payload = json.loads(
        (ROOT / "artifact/mechanism_config.json").read_text(encoding="utf-8")
    )
    mutate(payload)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_mechanism_config(path)


def test_mechanism_config_rejects_pair_overrides_or_projection_seed_changes(tmp_path):
    from scripts.mechanism_config import load_mechanism_config

    source = json.loads(
        (ROOT / "artifact/mechanism_config.json").read_text(encoding="utf-8")
    )
    for mutation, expected in (
        (lambda value: value.update(pair_overrides={}), "pair-specific"),
        (lambda value: value["projection"].update(seed=7), "seed 42"),
    ):
        payload = json.loads(json.dumps(source))
        mutation(payload)
        path = tmp_path / f"{expected.replace(' ', '-')}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            load_mechanism_config(path)


def test_calibrated_config_requires_complete_hash_bound_selection(tmp_path):
    from scripts.mechanism_config import load_mechanism_config

    payload = json.loads(
        (ROOT / "artifact/mechanism_config.json").read_text(encoding="utf-8")
    )
    payload["status"] = "calibrated"
    payload["selected"] = {
        "gamma_depth": 0.05,
        "beta_x": 0.25,
        "beta_f": 0.05,
        "beta_d": 0.5,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="tuple does not match"):
        load_mechanism_config(path)


def test_claim_execution_accepts_the_frozen_config():
    from scripts.mechanism_config import require_calibrated_mechanism

    _config, provenance = require_calibrated_mechanism(
        ROOT / "artifact/mechanism_config.json"
    )
    assert provenance["status"] == "calibrated"
    assert provenance["protocol"] == "acid_train_holdout_v1"


def test_result_builder_emits_v21_and_binds_the_mechanism_config():
    source = (ROOT / "scripts/result_record.py").read_text(encoding="utf-8")

    assert '"schema_version": "2.1"' in source
    assert '"mechanism_config_sha256"' in source
    assert '"calibration_provenance"' in source
