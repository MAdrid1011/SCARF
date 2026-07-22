#!/usr/bin/env python3
"""Run the fixed 12-anchor, guard-on representative SAES quality diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.saes_execution_identity import build_saes_execution_identity  # noqa: E402
from scripts.saes_selected_output_quality_gate import (  # noqa: E402
    REPRESENTATIVE_QUALITY_PILOT_KIND,
    main as run_quality_pilot,
)


EXECUTION_IDENTITY = build_saes_execution_identity()
MATERIALIZATION = EXECUTION_IDENTITY["materialization"]
PILOT_KIND = REPRESENTATIVE_QUALITY_PILOT_KIND
SEED = 0


def main(argv: list[str] | None = None) -> int:
    return run_quality_pilot(
        argv,
        materialization=MATERIALIZATION,
        pilot_kind=PILOT_KIND,
        command_path=Path(__file__),
        fixed_seed=SEED,
        fixed_context_safety_guard=EXECUTION_IDENTITY["context_safety_guard"],
        execution_identity=EXECUTION_IDENTITY,
    )


if __name__ == "__main__":
    raise SystemExit(main())
