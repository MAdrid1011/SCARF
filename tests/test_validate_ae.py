import json
from pathlib import Path

import pytest


def test_complete_validator_rejects_missing_evidence(tmp_path: Path):
    from scripts.validate_ae import validate_complete

    expected = Path(__file__).resolve().parents[1] / "artifact/expected_results.json"
    with pytest.raises(FileNotFoundError, match="quick/mvsplat_re10k/results.json"):
        validate_complete(tmp_path, expected)


def test_functional_reference_falls_back_to_quality_when_quick_was_not_run(
    tmp_path: Path,
):
    from scripts.validate_ae import find_functional_reference

    quality = tmp_path / "quality" / "transplat_re10k" / "results.json"
    quality.parent.mkdir(parents=True)
    quality.write_text("{}", encoding="utf-8")

    workflow, pair, path = find_functional_reference(tmp_path)

    assert (workflow, pair, path) == ("quality", "transplat/re10k", quality)


def test_mechanism_result_lookup_includes_mechanisms_evidence(tmp_path: Path):
    from scripts.validate_ae import MECHANISM_RESULT_MODES, find_pair_result

    result = tmp_path / "mechanisms" / "transplat_re10k" / "results.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}", encoding="utf-8")

    assert find_pair_result(tmp_path, "transplat/re10k", MECHANISM_RESULT_MODES) == result


def test_evaluator_final_profile_enables_key_result_validation(tmp_path, monkeypatch):
    import scripts.validate_ae as validator

    captured = {}

    def fake_validate_complete(output, expected, **kwargs):
        captured.update(kwargs)
        return {"status": "PASS", "summary": {"passed": 1, "total": 1}, "checks": []}

    monkeypatch.setattr(validator, "validate_complete", fake_validate_complete)
    monkeypatch.setattr(
        validator.sys,
        "argv",
        [
            "validate_ae.py",
            "--input",
            str(tmp_path),
            "--profile",
            "evaluator-final",
        ],
    )

    assert validator.main() == 0
    assert captured == {
        "require_key_results": True,
        "allow_missing_quick": False,
        "validation_profile": "evaluator-final",
    }


def test_validator_default_profile_preserves_author_preflight(tmp_path, monkeypatch):
    import scripts.validate_ae as validator

    captured = {}

    def fake_validate_complete(output, expected, **kwargs):
        captured.update(kwargs)
        return {"status": "PASS", "summary": {"passed": 1, "total": 1}, "checks": []}

    monkeypatch.setattr(validator, "validate_complete", fake_validate_complete)
    monkeypatch.setattr(
        validator.sys,
        "argv",
        ["validate_ae.py", "--input", str(tmp_path)],
    )

    assert validator.main() == 0
    assert captured == {
        "require_key_results": False,
        "allow_missing_quick": False,
        "validation_profile": "author-preflight",
    }


def test_key_result_catalog_gate_rejects_not_run_and_accepts_complete_pass(tmp_path):
    from scripts.validate_ae import validate_result_catalog

    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "artifact/evaluation_catalog.json").read_text())
    rows = [
        {
            "id": item["id"],
            "evidence_class": item["required_evidence_class"],
            "status": "PASS",
            "source_data": [{"path": "raw.json", "sha256": "a" * 64}],
            "exports": ["result.pdf"],
        }
        for item in contract["results"]
    ]
    catalog = {"schema_version": "2.0", "results": rows}

    validate_result_catalog(catalog, require_key_results=True)
    rows[0]["status"] = "NOT_RUN"
    with pytest.raises(ValueError, match="mandatory key result"):
        validate_result_catalog(catalog, require_key_results=True)


def test_figure8_pending_state_is_valid_intent_but_not_completed_evidence():
    from scripts.validate_ae import figure8_pending_state_check

    pending = figure8_pending_state_check(
        "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION"
    )
    assert pending["pass"] is True
    assert "awaiting" in pending["claim"]
    assert figure8_pending_state_check("NOT_CLAIMED_NO_ORIN_EVIDENCE")["pass"] is False


def test_pending_figure8_catalog_pass_is_not_completed_evidence(tmp_path):
    from scripts.validate_ae import figure8_catalog_evidence_check

    catalog = {"results": [{"id": "figure8", "status": "PASS"}]}
    protocol = {"pairs": {}}

    check = figure8_catalog_evidence_check(
        tmp_path,
        catalog,
        protocol,
        "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION",
    )

    assert check["pass"] is False
    assert "independent" in check["target"]


def test_expected_table_has_all_nine_pairs():
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "artifact/expected_results.json").read_text())
    assert len(expected["table1"]) == 9
    assert expected["sensitivity_expected_runs"] == 225


def test_unfinalized_sample_protocol_is_a_failed_claim():
    from scripts.validate_ae import protocol_check

    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "artifact/evaluation_protocol.json").read_text())
    expected = json.loads((root / "artifact/expected_results.json").read_text())
    protocol["pairs"]["depthsplat/dl3dv"]["dataset_tree_sha256"] = None

    check = protocol_check(protocol, set(expected["table1"]))

    assert check["claim"] == "evaluation:sample_protocol_finalized"
    assert not check["pass"]


def test_ae_aggregate_boundary_rejects_assignment_consensus_records():
    from scripts.validate_ae import require_aggregate

    candidate = {
        "provenance": {
            "execution_contract": {
                "run_class": "diagnostic",
                "saes_materialization": (
                    "assignment-consensus-adapter-pseudo-descriptor-diagnostic"
                ),
            },
            "evaluation": {"kind": "dataset_aggregate", "sample_count": 1},
        }
    }

    with pytest.raises(ValueError, match="assignment-consensus pseudo descriptors"):
        require_aggregate(candidate, "transplat/dl3dv")


def test_clean_source_check_rejects_dirty_or_invalid_commits():
    from scripts.validate_ae import clean_source_check

    valid = {"provenance": {"git_commit": "a" * 40, "git_dirty": False}}
    assert clean_source_check(valid, "mvsplat/re10k", "quality")["pass"]

    dirty = {"provenance": {"git_commit": "a" * 40, "git_dirty": True}}
    assert not clean_source_check(dirty, "mvsplat/re10k", "quality")["pass"]

    invalid = {"provenance": {"git_commit": "paper-value", "git_dirty": False}}
    assert not clean_source_check(invalid, "mvsplat/re10k", "quality")["pass"]


def test_source_binding_check_requires_matching_clean_commit_and_submodules():
    from scripts.validate_ae import source_binding_check

    reference = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "submodules": {
            "transplat": "b" * 40,
            "mvsplat": "c" * 40,
            "depthsplat": "d" * 40,
        },
    }
    record = {"provenance": dict(reference)}

    assert source_binding_check(record, reference, "rtl")["pass"]
    record["provenance"]["git_commit"] = "e" * 40
    assert not source_binding_check(record, reference, "rtl")["pass"]
    record["provenance"] = {**reference, "git_dirty": True}
    assert not source_binding_check(record, reference, "rtl")["pass"]


def test_quality_mechanism_binding_requires_all_four_aggregate_hashes():
    from scripts.validate_ae import quality_mechanism_binding_check

    quality = {
        "provenance": {
            "mechanism_config_sha256": "a" * 64,
            "checkpoint": {"sha256": "b" * 64},
            "evaluation": {
                "sample_selection_sha256": "c" * 64,
                "execution_trace_set_sha256": "d" * 64,
            },
        }
    }
    mechanism = json.loads(json.dumps(quality))

    assert quality_mechanism_binding_check(
        quality, mechanism, "transplat/re10k"
    )["pass"]

    mechanism["provenance"]["evaluation"]["execution_trace_set_sha256"] = "e" * 64
    check = quality_mechanism_binding_check(quality, mechanism, "transplat/re10k")
    assert not check["pass"]
    assert set(check["mismatches"]) == {"execution_trace_set_sha256"}

    mechanism["provenance"]["evaluation"].pop("execution_trace_set_sha256")
    check = quality_mechanism_binding_check(quality, mechanism, "transplat/re10k")
    assert not check["pass"]
    assert "execution_trace_set_sha256" in check["mismatches"]


def test_quality_mechanism_presence_requires_both_claim_surfaces():
    from scripts.validate_ae import quality_mechanism_presence_check

    quality_only = quality_mechanism_presence_check(
        "transplat/re10k", quality_claimed=True, mechanism_claimed=False
    )
    assert not quality_only["pass"]
    assert quality_only["actual"] == {
        "quality_claimed": True,
        "mechanism_claimed": False,
    }

    mechanism_only = quality_mechanism_presence_check(
        "transplat/re10k", quality_claimed=False, mechanism_claimed=True
    )
    assert not mechanism_only["pass"]

    paired = quality_mechanism_presence_check(
        "transplat/re10k", quality_claimed=True, mechanism_claimed=True
    )
    assert paired["pass"]


def test_hardware_claim_checks_allow_an_evidence_backed_resource_downgrade(tmp_path):
    from scripts.validate_ae import hardware_claim_checks

    checks = hardware_claim_checks(
        tmp_path,
        {
            "physical_asap7": "NOT_CLAIMED_RESOURCE_LIMIT",
            "deepscale": "NOT_CLAIMED_NO_PHYSICAL_INPUT",
        },
    )

    assert [item["claim"] for item in checks] == [
        "hardware:asap7_not_claimed",
        "hardware:deepscale_not_claimed",
    ]
    assert all(item["pass"] for item in checks)


def test_paused_physical_proxy_does_not_require_a_dram_result():
    from scripts.validate_ae import physical_proxy_active

    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "artifact/evaluation_catalog.json").read_text())
    assert physical_proxy_active(catalog) is False

    for record in catalog["results"]:
        if record["id"] == "figure9":
            record["current_state"] = "BLOCKED"
    assert physical_proxy_active(catalog) is True


def test_hardware_claim_checks_require_real_claimed_files(tmp_path):
    from scripts.validate_ae import hardware_claim_checks

    with pytest.raises(FileNotFoundError, match="ppa.json"):
        hardware_claim_checks(
            tmp_path,
            {"physical_asap7": "CLAIMED", "deepscale": "CLAIMED"},
        )


def test_environment_record_is_bound_to_result_digest(tmp_path):
    import hashlib

    from scripts.validate_ae import environment_provenance_check

    canonical = {
        "profile": "classic",
        "python": "3.10.19",
        "implementation": "CPython",
        "torch": "2.1.2+cu121",
        "torchvision": "0.16.2+cu121",
        "torch_cuda": "12.1",
        "lock_sha256": "a" * 64,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = tmp_path / "environments"
    root.mkdir()
    (root / "classic.json").write_text(
        json.dumps(
            {
                "profile": "classic",
                "contract": {"python": "3.10"},
                "details": {key: value for key, value in canonical.items() if key != "profile"},
            }
        ),
        encoding="utf-8",
    )
    result = {
        "provenance": {
            "environment": {"profile": "classic", "digest_sha256": digest}
        }
    }

    assert environment_provenance_check(result, "mvsplat/re10k", tmp_path)["pass"]
    result["provenance"]["environment"]["digest_sha256"] = "b" * 64
    assert not environment_provenance_check(result, "mvsplat/re10k", tmp_path)["pass"]
