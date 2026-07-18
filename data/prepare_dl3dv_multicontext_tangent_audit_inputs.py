#!/usr/bin/env python3
"""Compile the fixed DL3DV audit sidecar to context-only inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.context_only_audit_input import prepare_context_only_audit_input


DEFAULT_SOURCE_ROOT = (
    ROOT
    / "outputs"
    / "ae_dl3dv_repair_diagnostics"
    / "transplat_sample0_l1_primary_reference_target_free_input_v1"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = prepare_context_only_audit_input(
            args.source_input_root, output_root=args.output_dir
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir / "audit-input.json")
    print(record["tree_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
