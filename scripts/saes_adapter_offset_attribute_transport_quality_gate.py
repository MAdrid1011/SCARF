#!/usr/bin/env python3
"""Run the single pre-registered quality gate for adapter-offset attribute transport."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.saes_selected_output_quality_gate import (
    ATTRIBUTE_TRANSPORT_MATERIALIZATION,
    ATTRIBUTE_TRANSPORT_QUALITY_PILOT_KIND,
    main as run_quality_pilot,
)


MATERIALIZATION = ATTRIBUTE_TRANSPORT_MATERIALIZATION
PILOT_KIND = ATTRIBUTE_TRANSPORT_QUALITY_PILOT_KIND
SEED = 0


def main(argv: list[str] | None = None) -> int:
    return run_quality_pilot(
        argv,
        materialization=MATERIALIZATION,
        pilot_kind=PILOT_KIND,
        command_path=Path(__file__),
        fixed_seed=SEED,
    )


if __name__ == "__main__":
    raise SystemExit(main())
