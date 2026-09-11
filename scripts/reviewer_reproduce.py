#!/usr/bin/env python3
"""Run the reviewer-facing end-to-end workflow rehearsal.

The command materializes every quality, mechanism, performance, calibration,
and timing path in an external directory and validates the resulting tree.
The generated records are explicitly simulation-only; they are a schema and
command rehearsal, not experimental evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(output: Path, *, profile: str = "reviewer") -> dict:
    from scripts.simulate_claim_workflows import build, validate

    manifest = build(output, root=ROOT, profile=profile, layout="file-tree")
    checked = validate(output)
    summary = {
        "schema_version": "scarf-reviewer-reproduction-v1",
        "status": "PASS",
        "kind": "reviewer_workflow_rehearsal",
        "profile": profile,
        "output": str(Path(output).resolve()),
        "simulation_only": True,
        "claim_eligible": False,
        "total_samples": manifest["total_samples"],
        "validated_pairs": checked["pairs"],
        "replacement": (
            "Replace simulation-only files with real calibration, timing, and "
            "device records before running claim_readiness.py and validate_ae.py."
        ),
    }
    Path(output).resolve().joinpath("reviewer-reproduction.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("reviewer", "full"), default="reviewer")
    args = parser.parse_args(argv)
    try:
        result = run(args.output_dir.resolve(), profile=args.profile)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"PASS: reviewer rehearsal validated {result['validated_pairs']} pairs "
        f"({result['total_samples']} samples); simulation_only=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
