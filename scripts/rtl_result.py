#!/usr/bin/env python3
"""Create the machine-readable result for a completed RTL validation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.result_record import source_identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return source_identity(ROOT)["git_commit"]


def build_result(output_dir: Path) -> dict:
    required = {
        "test_log": output_dir / "sbt-test.log",
        "emit_log": output_dir / "emit.log",
        "lint_log": output_dir / "verilator-lint.log",
        "systemverilog": output_dir / "rtl" / "ScarfTop.sv",
        "switching_vcd": output_dir / "traces" / "fsdr-cache.vcd",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing RTL artifacts: {', '.join(missing)}")
    test_text = required["test_log"].read_text(encoding="utf-8", errors="replace")
    lint_text = required["lint_log"].read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Total number of tests run:\s+(\d+)", test_text)
    if not match or "All tests passed." not in test_text:
        raise ValueError("sbt test log does not report a passing test suite")
    errors = len(re.findall(r"^%Error", lint_text, flags=re.MULTILINE))
    if errors:
        raise ValueError(f"Verilator reported {errors} errors")
    warnings = len(re.findall(r"^%Warning", lint_text, flags=re.MULTILINE))
    source = source_identity(ROOT)
    return {
        "schema_version": "1.0",
        "kind": "rtl_validation",
        "status": "PASS",
        "provenance": {
            "git_commit": source["git_commit"],
            "git_dirty": source["git_dirty"],
            "source_identity": source["source"],
            "submodules": source["submodules"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "commands": [
                "sbt test",
                'sbt "runMain scarf.VerilogEmitter"',
                "verilator --lint-only --Wall -Wno-fatal --top-module ScarfTop",
            ],
        },
        "rtl": {
            "tests_passed": int(match.group(1)),
            "systemverilog_emitted": True,
            "verilator_lint_passed": True,
            "verilator_warning_count": warnings,
            "switching_vcd_included": True,
        },
        "artifacts": {
            name: {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
            for name, path in required.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record = build_result(args.output_dir.resolve())
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
