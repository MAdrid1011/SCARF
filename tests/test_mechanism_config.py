import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_preregistered_config_has_portable_nonclaim_provenance():
    from scripts.mechanism_config import load_mechanism_config

    config, provenance = load_mechanism_config(
        ROOT / "artifact/mechanism_config.json"
    )

    assert config["status"] == "preregistered"
    assert provenance["status"] == "preregistered"
    assert provenance["global_configuration"] is True
    assert provenance["evaluation_disjoint"] is False
    assert provenance["candidate_records_sha256"] is None
    assert len(provenance["mechanism_config_sha256"]) == 64
    assert len(provenance["manifest_sha256"]) == 64
    assert config["fixed"]["saes_moment_geometry"] == "c2w-probe-depth-ray-v1"


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

    with pytest.raises(ValueError, match="calibration hashes"):
        load_mechanism_config(path)


def test_claim_execution_rejects_a_preregistered_config():
    from scripts.mechanism_config import require_calibrated_mechanism

    with pytest.raises(RuntimeError, match="has not been calibrated"):
        require_calibrated_mechanism(ROOT / "artifact/mechanism_config.json")


def test_result_builder_emits_v21_and_binds_the_mechanism_config():
    source = (ROOT / "scripts/result_record.py").read_text(encoding="utf-8")

    assert '"schema_version": "2.1"' in source
    assert '"mechanism_config_sha256"' in source
    assert '"calibration_provenance"' in source
