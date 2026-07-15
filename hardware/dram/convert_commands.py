#!/usr/bin/env python3
"""Convert a Ramulator LPDDR5 command trace to DRAMPower CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


COMMANDS = {
    "ACT2": "ACT",
    "PREpb": "PRE",
    "PREab": "PREA",
    "RD": "RD",
    "WR": "WR",
    "RDA": "RDA",
    "WRA": "WRA",
    "REFab": "REFA",
    "REFpb": "REFB",
}
IGNORED_CA_PHASES = {"ACT1", "CAS_RD", "CAS_WR"}
DATA_COMMANDS = {"RD", "WR", "RDA", "WRA"}


def convert(source: Path, output: Path) -> dict[str, int]:
    rows = []
    ignored = 0
    last_clock = -1
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"clock", "command", "Channel", "Rank", "BankGroup", "Bank", "Row", "Column"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Ramulator command trace header is incomplete")
        for item in reader:
            command = item["command"]
            clock = int(item["clock"])
            if clock < last_clock:
                raise ValueError("Ramulator command clocks are not monotonic")
            last_clock = clock
            if command in IGNORED_CA_PHASES:
                ignored += 1
                continue
            mapped = COMMANDS.get(command)
            if mapped is None:
                raise ValueError(f"unsupported Ramulator command: {command}")
            row = [
                str(clock),
                mapped,
                item["Rank"],
                item["BankGroup"],
                item["Bank"],
                item["Row"],
                item["Column"],
            ]
            if mapped in DATA_COMMANDS:
                row.append("0x0000000000000000")
            rows.append(row)
    if not rows:
        raise ValueError("Ramulator command trace contains no convertible commands")
    rows.append([str(last_clock + 1), "END", "0", "0", "0", "0", "0"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="ascii") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return {"converted_commands": len(rows) - 1, "ignored_ca_phases": ignored}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = convert(args.input.resolve(), args.output.resolve())
    except (OSError, csv.Error, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"converted {summary['converted_commands']} commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
