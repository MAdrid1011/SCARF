#!/usr/bin/env python3
"""Write the unrun DepthSplat V15D/V16D ACID collection contract.

This command intentionally has no GPU collection mode yet.  Its output proves
which ACID 24/8 context-only sidecars and which native DepthSplat DL3DV source
identity a later collector must use, while explicitly recording that no GPU
observation or frozen threshold has been produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def collect_plan(
    *, plan_path: Path, materialization_root: Path
) -> dict[str, object]:
    """Bind the live sidecar and native application identities without a GPU."""

    binding = resolve_depthsplat_acid_binding(
        plan_path=plan_path, materialization_root=materialization_root
    )
    backend_identity = freeze_depthsplat_backend_identity(
        resolve_depthsplat_backend_contract(ROOT)
    )
    return build_collection_plan(
        binding=binding, backend_identity=backend_identity
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
        record = collect_plan(
            plan_path=args.plan_path,
            materialization_root=args.materialization_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
