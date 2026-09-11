#!/usr/bin/env python3
"""Generate a non-claim Figure 8 proxy from completed local quality results.

This path is intentionally limited to the local smoke configuration. It uses
the measured workstation CUDA event and the source-bound SCARF cycle record
already emitted by each model run; it never labels the result as an Orin
measurement or fills the reviewer hardware contract with invented values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _sample_record(pair_dir: Path) -> tuple[dict[str, Any], Path]:
    samples = sorted((pair_dir / "samples").glob("sample_*/results.json"))
    if len(samples) != 1:
        raise ValueError(f"expected exactly one completed smoke sample in {pair_dir}")
    path = samples[0]
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"quality result is not an object: {path}")
    return record, path


def generate(quality_root: Path, output_dir: Path) -> dict[str, Any]:
    quality_root = Path(quality_root).resolve()
    output_dir = Path(output_dir).resolve()
    rows: list[dict[str, Any]] = []
    for pair_dir in sorted(quality_root.iterdir()):
        if not pair_dir.is_dir() or "_" not in pair_dir.name:
            continue
        model, dataset = pair_dir.name.split("_", 1)
        record, path = _sample_record(pair_dir)
        performance = record.get("performance")
        provenance = record.get("provenance")
        device = provenance.get("device") if isinstance(provenance, dict) else None
        if not isinstance(performance, dict) or not isinstance(device, dict):
            raise ValueError(f"quality result has no runtime performance evidence: {path}")
        baseline_ms = device.get("measured_encoder_time_ms")
        baseline_cycles = performance.get("baseline_cycles")
        scarf_cycles = performance.get("scarf_cycles")
        speedup = performance.get("speedup")
        if not all(isinstance(value, (int, float)) and value > 0 for value in (baseline_ms, baseline_cycles, scarf_cycles, speedup)):
            raise ValueError(f"quality result has invalid performance values: {path}")
        rows.append(
            {
                "pair": f"{model}/{dataset}",
                "baseline_cuda_ms": float(baseline_ms),
                "baseline_cycles": int(baseline_cycles),
                "scarf_cycles": int(scarf_cycles),
                "normalized_speedup": float(speedup),
                "claim_eligible": False,
                "source_result": str(path),
            }
        )
    if not rows:
        raise ValueError(f"no completed quality results below {quality_root}")
    rows.sort(key=lambda row: row["pair"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "figure8_proxy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "pair",
                "baseline_cuda_ms",
                "baseline_cycles",
                "scarf_cycles",
                "normalized_speedup",
                "claim_eligible",
            ],
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)

    quality_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    mechanism_root = quality_root.parent / "mechanisms"
    for row in rows:
        pair = row["pair"]
        model, dataset = pair.split("/", 1)
        quality_record, _ = _sample_record(quality_root / f"{model}_{dataset}")
        quality = quality_record.get("quality")
        if not isinstance(quality, dict):
            raise ValueError(f"quality metrics are missing for {pair}")
        baseline = quality.get("baseline")
        scarf = quality.get("scarf")
        if not isinstance(baseline, dict) or not isinstance(scarf, dict):
            raise ValueError(f"quality metrics are incomplete for {pair}")
        quality_rows.append(
            {
                "pair": pair,
                "sample_count": 1,
                "baseline_psnr_db": baseline["psnr_db"],
                "baseline_ssim": baseline["ssim"],
                "baseline_lpips": baseline["lpips"],
                "scarf_psnr_db": scarf["psnr_db"],
                "scarf_ssim": scarf["ssim"],
                "scarf_lpips": scarf["lpips"],
                "claim_eligible": False,
            }
        )
        mechanism_record, _ = _sample_record(mechanism_root / f"{model}_{dataset}")
        ablation = mechanism_record.get("ablation")
        if not isinstance(ablation, dict):
            raise ValueError(f"ablation metrics are missing for {pair}")
        totals = {
            key: ablation.get(key, {}).get("total_cycles")
            for key in ("asic", "asic_fsdr", "asic_saes", "asic_fsdr_saes")
        }
        if any(not isinstance(value, (int, float)) or value <= 0 for value in totals.values()):
            raise ValueError(f"ablation totals are incomplete for {pair}")
        ablation_rows.append(
            {
                "pair": pair,
                "fsdr_speedup": totals["asic"] / totals["asic_fsdr"],
                "saes_speedup": totals["asic"] / totals["asic_saes"],
                "combined_speedup": totals["asic"] / totals["asic_fsdr_saes"],
                "claim_eligible": False,
            }
        )
    for name, rows_to_write in (
        ("table1_quality.csv", quality_rows),
        ("figure11_ablation.csv", ablation_rows),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows_to_write[0]))
            writer.writeheader()
            writer.writerows(rows_to_write)
    result = {
        "schema_version": "scarf-figure8-local-proxy-v1",
        "kind": "figure8_proxy",
        "status": "PROXY_ONLY",
        "evidence_class": "local_runtime_proxy",
        "claim_eligible": False,
        "independent_orin_measurement": False,
        "scope": "one Re10K sample per selected model",
        "method": "measured workstation CUDA event versus source-bound SCARF cycle record",
        "rows": rows,
        "table1_quality_rows": quality_rows,
        "figure11_ablation_rows": ablation_rows,
        "csv": str(csv_path),
        "limitations": [
            "This is not a Jetson Orin NX measurement.",
            "It does not provide Orin power, temperature, occupancy, or Nsight evidence.",
            "Run the filled reviewer GPU contract or hardware/orin/run.py before making a Figure 8 claim.",
        ],
    }
    (output_dir / "figure8_proxy.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt

        labels = [row["pair"] for row in rows]
        values = [row["normalized_speedup"] for row in rows]
        figure = plt.figure(figsize=(8.0, 4.0))
        axis = figure.add_subplot(1, 1, 1)
        axis.bar(range(len(values)), values, color="#64748b")
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_ylabel("Normalized speedup (local proxy)")
        axis.set_title("Figure 8 local proxy (not an Orin NX measurement)")
        figure.tight_layout()
        figure.savefig(output_dir / "figure8_proxy.png", dpi=160)
        figure.savefig(output_dir / "figure8_proxy.pdf")
        plt.close(figure)
    except ImportError:
        result["plot_status"] = "matplotlib_unavailable"
        (output_dir / "figure8_proxy.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = generate(args.quality_root, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(f"PROXY_ONLY: wrote {len(result['rows'])} local Figure 8 rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
