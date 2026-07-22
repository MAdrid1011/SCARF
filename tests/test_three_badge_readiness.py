from __future__ import annotations

import json


def _quick_record(commit: str) -> dict:
    return {
        "provenance": {
            "git_commit": commit,
            "git_dirty": False,
            "dataset": {
                "functional_fixture": True,
                "paper_result_eligible": False,
            },
            "evaluation": {"kind": "dataset_aggregate", "sample_count": 1},
        }
    }


def _catalog() -> dict:
    return {
        "results": [
            {"id": "figure8", "current_state": "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION"},
            {"id": "table1", "current_state": "NOT_RUN"},
            {"id": "figure11", "current_state": "NOT_RUN"},
        ]
    }


def test_readiness_keeps_unrun_key_results_explicit(tmp_path, monkeypatch) -> None:
    import scripts.three_badge_readiness as readiness

    commit = "a" * 40
    root = tmp_path / "repo"
    quick = root / "out/quick/mvsplat_re10k/results.json"
    quick.parent.mkdir(parents=True)
    quick.write_text(json.dumps(_quick_record(commit)), encoding="utf-8")
    catalog = root / "artifact/evaluation_catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
    archive = root / "source.tar.gz"
    archive.write_bytes(b"archive")

    monkeypatch.setattr(readiness, "ROOT", root)
    monkeypatch.setattr(readiness, "_current_git_commit", lambda: commit)
    monkeypatch.setattr(
        readiness,
        "verify",
        lambda _path: {
            "status": "PASS",
            "bundle_kind": "source",
            "git_commit": commit,
            "zenodo_doi": None,
        },
    )
    monkeypatch.setattr(readiness, "validate", lambda _record: None)
    monkeypatch.setattr(
        readiness,
        "validate_complete",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("missing full evidence")
        ),
    )

    record = readiness.build_readiness(archive, root / "out")

    assert record["available"]["ready_for_zenodo_submission"] is True
    assert record["available"]["badge_complete"] is False
    assert record["functional"]["pass"] is True
    assert record["results_reproduced"]["pass"] is False
    assert record["results_reproduced"]["validator_status"] == "NOT_READY"
    assert record["status"] == "SELF_VERIFIED_READY_FOR_RESULTS_REPRODUCTION"


def test_readiness_requires_all_three_badge_conditions(tmp_path, monkeypatch) -> None:
    import scripts.three_badge_readiness as readiness

    commit = "b" * 40
    root = tmp_path / "repo"
    quick = root / "out/quick/mvsplat_re10k/results.json"
    quick.parent.mkdir(parents=True)
    quick.write_text(json.dumps(_quick_record(commit)), encoding="utf-8")
    catalog = root / "artifact/evaluation_catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
    archive = root / "source.tar.gz"
    archive.write_bytes(b"archive")

    monkeypatch.setattr(readiness, "ROOT", root)
    monkeypatch.setattr(readiness, "_current_git_commit", lambda: commit)
    monkeypatch.setattr(
        readiness,
        "verify",
        lambda _path: {
            "status": "PASS",
            "bundle_kind": "source",
            "git_commit": commit,
            "zenodo_doi": "10.5281/zenodo.12345",
        },
    )
    monkeypatch.setattr(readiness, "validate", lambda _record: None)
    monkeypatch.setattr(
        readiness,
        "validate_complete",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "summary": {"passed": 4, "total": 4},
        },
    )

    record = readiness.build_readiness(archive, root / "out")

    assert record["available"]["badge_complete"] is True
    assert record["functional"]["pass"] is True
    assert record["results_reproduced"]["pass"] is True
    assert record["status"] == "THREE_BADGE_COMPLETE"
