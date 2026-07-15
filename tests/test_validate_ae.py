import json
from pathlib import Path

import pytest


def test_complete_validator_rejects_missing_evidence(tmp_path: Path):
    from scripts.validate_ae import validate_complete

    expected = Path(__file__).resolve().parents[1] / "artifact/expected_results.json"
    with pytest.raises(FileNotFoundError, match="quick/mvsplat_re10k/results.json"):
        validate_complete(tmp_path, expected)


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

    check = protocol_check(protocol, set(expected["table1"]))

    assert check["claim"] == "evaluation:sample_protocol_finalized"
    assert not check["pass"]


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
