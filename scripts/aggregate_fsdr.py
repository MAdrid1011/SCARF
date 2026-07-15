#!/usr/bin/env python3
"""Aggregate FSDR sample counters and compare them with paper Table 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fsdr_evidence import (
    aggregate_fsdr_records,
    compare_fsdr_aggregate,
    validate_fsdr_record,
    write_json,
)
from scripts.result_record import portable_command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-match", action="store_true")
    args = parser.parse_args()
    try:
        paths = sorted(args.input_dir.glob("sample_*/results.json"))
        records = [
            json.loads(path.read_text(encoding="utf-8")) for path in paths
        ]
        aggregate = aggregate_fsdr_records(
            records, expected_count=args.expected_count
        )
        pair_root = args.input_dir.parent
        aggregate["provenance"]["command"] = portable_command(
            [sys.executable, *sys.argv]
        )
        aggregate["provenance"]["evaluation"]["sample_results"] = [
            {
                "path": str(path.relative_to(pair_root)),
                "sha256": _sha256(path),
                "sample_index": record["provenance"]["evaluation"][
                    "sample_index"
                ],
            }
            for path, record in zip(paths, records)
        ]
        expected = json.loads(args.expected_results.read_text(encoding="utf-8"))
        aggregate["paper_comparison"] = compare_fsdr_aggregate(
            aggregate, expected
        )
        validate_fsdr_record(aggregate)
        write_json(aggregate, args.output)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(args.output)
    if args.require_match and aggregate["paper_comparison"]["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
