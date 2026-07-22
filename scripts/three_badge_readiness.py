#!/usr/bin/env python3
"""Report whether one SCARF revision is ready for the requested AE badges.

The report separates locally verifiable source/Functional checks from the
full Results Reproduced validation.  In particular, it never turns a source
archive, a synthetic quick fixture, or a workstation run into independent
Orin evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_archive import verify
from scripts.validate_ae import validate_current_release
from scripts.validate_result import validate


KIND = "scarf-three-badge-readiness-v1"
ACTIVE_RESULT_IDS = ("figure8", "table1", "figure11")


def _current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "cannot resolve git commit")
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("git commit is not a SHA-1")
    return commit


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _availability_record(source_archive: Path, *, commit: str) -> dict[str, Any]:
    try:
        verified = verify(source_archive)
        # ``verify`` returns the release-manifest fields required here.
        source_only = verified.get("bundle_kind") == "source"
        same_commit = verified.get("git_commit") == commit
        archive_verified = verified.get("status") == "PASS"
        doi = verified.get("zenodo_doi")
        return {
            "source_archive": str(source_archive),
            "archive_verified": archive_verified,
            "source_bundle": source_only,
            "archive_commit": verified.get("git_commit"),
            "matches_current_commit": same_commit,
            "doi_present": isinstance(doi, str) and bool(doi),
            "ready_for_zenodo_submission": archive_verified and source_only and same_commit,
            "badge_complete": archive_verified
            and source_only
            and same_commit
            and isinstance(doi, str)
            and bool(doi),
        }
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "source_archive": str(source_archive),
            "archive_verified": False,
            "ready_for_zenodo_submission": False,
            "badge_complete": False,
            "reason": str(exc),
        }


def _functional_record(output_root: Path, *, commit: str) -> dict[str, Any]:
    path = output_root / "quick/mvsplat_re10k/results.json"
    try:
        result = _load_json(path, "quick aggregate")
        validate(result)
        provenance = result.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("quick aggregate has no provenance")
        dataset = provenance.get("dataset")
        evaluation = provenance.get("evaluation")
        if not isinstance(dataset, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError("quick aggregate has incomplete provenance")
        passed = (
            provenance.get("git_commit") == commit
            and provenance.get("git_dirty") is False
            and dataset.get("functional_fixture") is True
            and dataset.get("paper_result_eligible") is False
            and evaluation.get("kind") == "dataset_aggregate"
            and isinstance(evaluation.get("sample_count"), int)
            and evaluation["sample_count"] > 0
        )
        return {
            "quick_result": str(path),
            "schema_valid": True,
            "matches_current_commit": provenance.get("git_commit") == commit,
            "git_dirty": provenance.get("git_dirty"),
            "functional_fixture": dataset.get("functional_fixture"),
            "paper_result_eligible": dataset.get("paper_result_eligible"),
            "sample_count": evaluation.get("sample_count"),
            "pass": passed,
        }
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {
            "quick_result": str(path),
            "schema_valid": False,
            "pass": False,
            "reason": str(exc),
        }


def _catalog_states() -> dict[str, str | None]:
    catalog = _load_json(ROOT / "artifact/evaluation_catalog.json", "evaluation catalog")
    records = catalog.get("results")
    if not isinstance(records, list):
        raise ValueError("evaluation catalog has no result list")
    by_id = {
        record.get("id"): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("id"), str)
    }
    return {
        result_id: (
            str(by_id[result_id].get("current_state"))
            if result_id in by_id and by_id[result_id].get("current_state") is not None
            else None
        )
        for result_id in ACTIVE_RESULT_IDS
    }


def _results_record(output_root: Path) -> dict[str, Any]:
    states = _catalog_states()
    try:
        validation = validate_current_release(
            output_root,
            require_key_results=True,
            validation_profile="evaluator-final",
        )
        passed = validation.get("status") == "PASS"
        return {
            "output_root": str(output_root),
            "catalog_states": states,
            "validator_status": validation.get("status"),
            "validator_summary": validation.get("summary"),
            "pass": passed,
            "external_requirement": (
                "Figure 8 remains an independent Jetson Orin NX evaluator measurement; "
                "the validator must see its raw evidence before this field can pass."
            ),
        }
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {
            "output_root": str(output_root),
            "catalog_states": states,
            "validator_status": "NOT_READY",
            "pass": False,
            "reason": str(exc),
            "external_requirement": (
                "Complete all nine Table 1 and Figure 11 pairs, stage their raw "
                "evidence, and obtain the independent Jetson Orin NX Figure 8 run."
            ),
        }


def build_readiness(source_archive: Path, output_root: Path) -> dict[str, Any]:
    """Build one truthful readiness record without changing evidence state."""

    commit = _current_git_commit()
    available = _availability_record(source_archive, commit=commit)
    functional = _functional_record(output_root, commit=commit)
    reproduced = _results_record(output_root)
    if available["badge_complete"] and functional["pass"] and reproduced["pass"]:
        status = "THREE_BADGE_COMPLETE"
    elif available["ready_for_zenodo_submission"] and functional["pass"]:
        status = "SELF_VERIFIED_READY_FOR_RESULTS_REPRODUCTION"
    else:
        status = "SELF_VERIFICATION_INCOMPLETE"
    return {
        "kind": KIND,
        "schema_version": "1.0",
        "git_commit": commit,
        "available": available,
        "functional": functional,
        "results_reproduced": reproduced,
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_readiness(args.source_archive.resolve(), args.output_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(record["status"])


if __name__ == "__main__":
    main()
