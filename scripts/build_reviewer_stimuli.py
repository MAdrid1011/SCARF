#!/usr/bin/env python3
"""Build source-bound RTL stimuli from one reviewer selection.

This command has no model execution side effects.  It only converts quality
sample records that already contain the runtime payload produced by
``run_pair.py --export-rtl-payload``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_rtl_stimulus import convert
from scripts.reviewer_run_config import load_selection


def build(quality_root: Path, selection_file: Path, output_root: Path) -> dict:
    selection = load_selection(selection_file)
    output_root = Path(output_root).resolve()
    records = []
    for pair, pair_record in sorted(selection["pairs"].items()):
        pair_dir = Path(quality_root).resolve() / pair.replace("/", "_") / "samples"
        for sample in pair_record["samples"]:
            index = int(sample["sample_index"])
            sample_path = pair_dir / f"sample_{index:05d}" / "results.json"
            if not sample_path.is_file():
                raise FileNotFoundError(f"quality sample is missing: {sample_path}")
            evaluation = json.loads(sample_path.read_text(encoding="utf-8")).get(
                "provenance", {}
            ).get("evaluation", {})
            if evaluation.get("run_selection_sha256") != selection["selection_sha256"]:
                raise ValueError(f"quality sample is bound to a different selection: {sample_path}")
            destination = output_root / pair.replace("/", "_") / f"sample_{index:05d}.json"
            convert(sample_path, destination, input_root=quality_root)
            records.append(destination.relative_to(output_root).as_posix())
    manifest = {
        "schema_version": "scarf-reviewer-stimuli-v1",
        "selection_sha256": selection["selection_sha256"],
        "sample_count": len(records),
        "stimuli": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = build(args.quality_root, args.selection_file, args.output_root)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"PASS: built {record['sample_count']} RTL stimuli")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
