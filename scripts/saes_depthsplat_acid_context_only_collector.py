#!/usr/bin/env python3
"""Validate formal DepthSplat ACID 24/8 context-only collector inputs.

This command is intentionally CPU-only.  It opens the validated context-only
sidecars, records their safe identities and shapes, and exits before model
loading or native risk collection.  It cannot create V15D/V16D thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.acid_joint_context import load_acid_joint_context  # noqa: E402
from saes.depthsplat_acid_context_only_collector import (  # noqa: E402
    validate_formal_acid_context_only_inputs,
)
from saes.depthsplat_acid_disjoint_calibration import (  # noqa: E402
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
    build_collection_plan,
    resolve_depthsplat_acid_binding,
)
from saes.depthsplat_backend import (  # noqa: E402
    freeze_depthsplat_backend_identity,
    resolve_depthsplat_backend_contract,
)


def collect_input_validation(
    *, plan_path: Path, materialization_root: Path
) -> dict[str, object]:
    """Build a source-bound plan, then validate all raw context-only inputs."""

    binding = resolve_depthsplat_acid_binding(
        plan_path=plan_path, materialization_root=materialization_root
    )
    backend_identity = freeze_depthsplat_backend_identity(
        resolve_depthsplat_backend_contract(ROOT)
    )
    collection_plan = build_collection_plan(
        binding=binding, backend_identity=backend_identity
    )
    return validate_formal_acid_context_only_inputs(
        collection_plan=collection_plan,
        plan_path=plan_path,
        materialization_root=materialization_root,
        context_loader=load_acid_joint_context,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument(
        "--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION_ROOT
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new file")
    try:
        record = collect_input_validation(
            plan_path=args.plan_path,
            materialization_root=args.materialization_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
