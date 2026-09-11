#!/usr/bin/env python3
"""Write the explicitly non-claim Figure 8 proxy exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def generate(input_path: Path, output_dir: Path) -> dict:
    record = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if record.get("schema_version") != "scarf-orin-proxy-result-v1":
        raise ValueError("proxy result schema is invalid")
    if record.get("claim_eligible") is not False:
        raise ValueError("proxy report must remain claim_eligible=false")
    pairs = record.get("pairs")
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("proxy result has no pairs")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair, value in sorted(pairs.items()):
        proxy = value["orin_nx_proxy"]
        rows.append({
            "pair": pair,
            "baseline_ms": proxy["baseline"]["total_ms"],
            "scarf_dataflow_ms": proxy["scarf_dataflow"]["total_ms"],
            "normalized_speedup": proxy["dataflow_speedup"],
            "claim_eligible": False,
        })
    csv_path = output_dir / "figure8_proxy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": "scarf-figure8-proxy-report-v1",
        "kind": "figure8_proxy",
        "claim_eligible": False,
        "source": str(Path(input_path).resolve()),
        "rows": rows,
    }
    json_path = output_dir / "figure8_proxy.json"
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        labels = [row["pair"] for row in rows]
        values = [row["normalized_speedup"] for row in rows]
        figure = plt.figure(figsize=(9.0, 4.0))
        axis = figure.add_subplot(1, 1, 1)
        axis.bar(range(len(values)), values, color="#3b82f6")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_ylabel("Normalized speedup (proxy)")
        axis.set_title("Figure 8 proxy (not an Orin NX measurement)")
        figure.tight_layout()
        figure.savefig(output_dir / "figure8_proxy.png", dpi=160)
        figure.savefig(output_dir / "figure8_proxy.pdf")
        plt.close(figure)
    except ImportError:
        metadata["plot_status"] = "matplotlib_unavailable"
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = generate(args.input, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(f"PROXY_ONLY: wrote {len(result['rows'])} Figure 8 rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
