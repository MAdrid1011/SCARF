#!/usr/bin/env python3
"""Generate paper-result tables, comparison plots, and an AE report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "artifact/plot_style.mplstyle"
EXPECTED = ROOT / "artifact/expected_results.json"
METRICS = ("psnr_db", "ssim", "lpips")
PAIR_LABELS = {
    "transplat/re10k": "Tran / Re10K",
    "transplat/acid": "Tran / ACID",
    "transplat/dl3dv": "Tran / DL3DV",
    "mvsplat/re10k": "MV / Re10K",
    "mvsplat/acid": "MV / ACID",
    "mvsplat/dl3dv": "MV / DL3DV",
    "depthsplat/re10k": "Depth / Re10K",
    "depthsplat/acid": "Depth / ACID",
    "depthsplat/dl3dv": "Depth / DL3DV",
}
COLORS = {"main": "#6F8F88", "neutral": "#A8A29E", "target": "#B67C6B", "quality": "#7C748C"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_pair(output: Path, pair: str, modes: tuple[str, ...]) -> Path:
    directory = pair.replace("/", "_")
    matches = [output / mode / directory / "results.json" for mode in modes]
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"missing {pair} result in {', '.join(modes)}")
    return matches[0]


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: Any, directory: Path, name: str) -> list[str]:
    paths = []
    for suffix in ("pdf", "png"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path)
        paths.append(str(path))
    return paths


def build_tables(
    output: Path, report_dir: Path, expected: dict[str, Any], claims: dict[str, Any]
) -> dict[str, Any]:
    quality_rows = []
    speedup_rows = []
    ablation_rows = []
    fsdr_rows = []
    saes_rows = []
    sources = []
    quality_pairs = {
        pair for pair, state in claims["software_pairs"].items() if state == "CLAIMED"
    }
    mechanism_pairs = {
        pair for pair, state in claims["mechanism_pairs"].items() if state == "CLAIMED"
    }
    for pair in (item for item in expected["table1"] if item in quality_pairs):
        quality_path = find_pair(output, pair, ("quality", "all", "ablation"))
        ablation_path = find_pair(output, pair, ("ablation", "all", "quality"))
        quality = load(quality_path)
        ablation = load(ablation_path)
        sources.extend((quality_path, ablation_path))
        row: dict[str, Any] = {"pair": pair, "sample_count": quality["provenance"]["evaluation"]["sample_count"]}
        for variant in ("baseline", "scarf"):
            for metric in METRICS:
                row[f"{variant}_{metric}"] = quality["quality"][variant][metric]
        quality_rows.append(row)
        if claims["figure8"] == "CLAIMED":
            speed_path = find_pair(output, pair, ("speedup", "orin"))
            speed = load(speed_path)
            sources.append(speed_path)
            speedup_rows.append({"pair": pair, "speedup": speed["performance"]["speedup"]})

        if pair in mechanism_pairs:
            base = float(ablation["ablation"]["asic"]["eff_total"])
            ablation_rows.append(
                {
                    "pair": pair,
                    "fsdr_speedup": base / float(ablation["ablation"]["asic_fsdr"]["eff_total"]),
                    "saes_speedup": base / float(ablation["ablation"]["asic_saes"]["eff_total"]),
                    "combined_speedup": base / float(ablation["ablation"]["asic_fsdr_saes"]["eff_total"]),
                }
            )
            fsdr = ablation["fsdr_saes"]["fsdr"]
            saes = ablation["fsdr_saes"]["saes"]
            preservation = ablation["fsdr_saes"]["preservation"]
            fsdr_rows.append(
                {"pair": pair, "guided_rate": fsdr["guided_rate"], "top1_coverage": fsdr["in_window_rate"]}
            )
            saes_rows.append(
                {
                    "pair": pair,
                    "level0_rate": saes["level0_ratio"],
                    "level1_rate": saes["level1_ratio"],
                    "low_variance_agreement": preservation["saes_low_var_agree"],
                    "gaussians_saved": saes["modification_ratio"],
                }
            )

    write_csv(report_dir / "table1_quality.csv", list(quality_rows[0]), quality_rows)
    write_csv(report_dir / "figure8_speedup.csv", ["pair", "speedup"], speedup_rows)
    write_csv(report_dir / "figure11_ablation.csv", list(ablation_rows[0]), ablation_rows)
    write_csv(report_dir / "table2_fsdr.csv", list(fsdr_rows[0]), fsdr_rows)
    write_csv(report_dir / "table3_saes.csv", list(saes_rows[0]), saes_rows)
    return {
        "quality": quality_rows,
        "speedup": speedup_rows,
        "ablation": ablation_rows,
        "fsdr": fsdr_rows,
        "saes": saes_rows,
        "sources": sorted(set(sources)),
    }


def sensitivity_summary(output: Path, report_dir: Path, expected: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    path = output / "sensitivity/results.json"
    record = load(path)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for run in record["runs"]:
        grouped[(run["study"], float(run["value"]))].append(run["metrics"])
    summary: dict[str, list[dict[str, float]]] = {}
    rows = []
    for study, grid in expected["sensitivity_grids"].items():
        study_rows = []
        for value in grid["values"]:
            metrics = grouped[(study, float(value))]
            if len(metrics) != 9:
                raise ValueError(f"{study}={value} has {len(metrics)} results instead of 9")
            speedup = geometric_mean([float(item["performance"]["speedup"]) for item in metrics])
            degradation = max(float(item["quality"]["change"]["psnr_degradation_pct"]) for item in metrics)
            item = {"value": float(value), "speedup": speedup, "max_psnr_degradation_pct": degradation}
            study_rows.append(item)
        default_speedup = next(item["speedup"] for item in study_rows if item["value"] == float(grid["default"]))
        for item in study_rows:
            item["normalized_throughput"] = item["speedup"] / default_speedup
            rows.append(
                {
                    "study": study,
                    **item,
                    "is_default": item["value"] == float(grid["default"]),
                }
            )
        summary[study] = study_rows
    write_csv(report_dir / "figures13-16_sensitivity.csv", list(rows[0]), rows)
    summary["_source"] = [str(path)]  # type: ignore[assignment]
    return summary


def render_figures(
    report_dir: Path,
    tables: dict[str, Any],
    sensitivity: dict[str, Any],
    expected: dict[str, Any],
    claims: dict[str, Any],
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    plt.style.use(STYLE)
    catalog = []
    if claims["figure8"] == "CLAIMED":
        labels = [PAIR_LABELS[item["pair"]] for item in tables["speedup"]]
        values = [item["speedup"] for item in tables["speedup"]]
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        target = expected["figure8"]["geometric_mean_speedup"]
        ax.bar(range(len(values)), values, color=COLORS["main"], width=0.68)
        ax.axhline(target, color=COLORS["target"], linestyle="--")
        ax.set_ylabel("End-to-end speedup (x)")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_ylim(0, max(max(values), target) * 1.16)
        catalog.append({"id": "figure8", "exports": save_figure(fig, report_dir, "figure8_speedup"), "claim": "Per-pair Orin speedup and paper geometric-mean target"})
        plt.close(fig)

    if claims["figure11"] == "CLAIMED":
        names = ("fsdr", "saes", "combined")
        actual = [geometric_mean([row[f"{name}_speedup"] for row in tables["ablation"]]) for name in names]
        targets = [expected["ablation"][name] for name in names]
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        x = list(range(3))
        ax.bar(x, actual, color=COLORS["main"], width=0.58, label="Measured aggregate")
        ax.scatter(x, targets, color=COLORS["target"], marker="D", zorder=3, label="Paper target")
        ax.set_xticks(x, ["FSDR", "SAES", "Combined"])
        ax.set_ylabel("Speedup over no optimization (x)")
        ax.legend()
        catalog.append({"id": "figure11", "exports": save_figure(fig, report_dir, "figure11_ablation"), "claim": "Geometric-mean optimization speedups"})
        plt.close(fig)

    if claims["sensitivity"] != "CLAIMED":
        return catalog

    def plot_study(study: str, name: str, ax: Any) -> Any:
        rows = sensitivity[study]
        xvals = [item["value"] for item in rows]
        ax.plot(xvals, [item["normalized_throughput"] for item in rows], marker="o", color=COLORS["main"], label="Normalized throughput")
        ax.axvline(expected["sensitivity_grids"][study]["default"], color=COLORS["target"], linestyle="--", label="Selected default")
        ax.set_xlabel(name)
        ax.set_ylabel("Throughput / default")
        quality_ax = ax.twinx()
        quality_ax.grid(False)
        quality_ax.plot(
            xvals,
            [item["max_psnr_degradation_pct"] for item in rows],
            marker="s",
            linestyle=":",
            color=COLORS["quality"],
            label="Max PSNR degradation",
        )
        quality_ax.set_ylabel("PSNR degradation (%)", color=COLORS["quality"])
        return quality_ax

    for study, figure_name, xlabel in (
        ("fsdr_cache_size", "figure13_cache_sensitivity", "Cache entries"),
        ("fsdr_hamming_threshold", "figure14_hamming_sensitivity", "Hamming threshold"),
        ("saes_tile_size", "figure16_tile_sensitivity", "Tile width"),
    ):
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        quality_ax = plot_study(study, xlabel, ax)
        handles, labels_ = ax.get_legend_handles_labels()
        quality_handles, quality_labels = quality_ax.get_legend_handles_labels()
        ax.legend(handles + quality_handles, labels_ + quality_labels, loc="best")
        catalog.append({"id": figure_name.split("_")[0], "exports": save_figure(fig, report_dir, figure_name), "claim": f"Measured {study} sensitivity"})
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex="col")
    for column, (study, xlabel) in enumerate(
        (("saes_feature_variance", "Feature threshold"), ("saes_depth_variance", "Depth threshold"))
    ):
        rows = sensitivity[study]
        xvals = [item["value"] for item in rows]
        default = expected["sensitivity_grids"][study]["default"]
        axes[0, column].plot(
            xvals,
            [item["normalized_throughput"] for item in rows],
            marker="o",
            color=COLORS["main"],
            label="Normalized throughput",
        )
        axes[1, column].plot(
            xvals,
            [item["max_psnr_degradation_pct"] for item in rows],
            marker="s",
            color=COLORS["quality"],
            label="Max PSNR degradation",
        )
        for row in range(2):
            axes[row, column].axvline(
                default,
                color=COLORS["target"],
                linestyle="--",
                label="Selected default",
            )
        axes[1, column].set_xlabel(xlabel)
    axes[0, 0].set_ylabel("Throughput / default")
    axes[1, 0].set_ylabel("PSNR degradation (%)")
    axes[0, 0].legend(loc="best")
    axes[1, 0].legend(loc="best")
    fig.subplots_adjust(hspace=0.16, wspace=0.22)
    catalog.append({"id": "figure15", "exports": save_figure(fig, report_dir, "figure15_saes_threshold_sensitivity"), "claim": "Measured SAES feature/depth threshold sensitivity"})
    plt.close(fig)
    return catalog


def hardware_table(
    output: Path,
    report_dir: Path,
    expected: dict[str, Any],
    claims: dict[str, Any],
) -> dict[str, Any]:
    if (
        claims.get("physical_asap7") != "CLAIMED"
        or claims.get("deepscale") != "CLAIMED"
    ):
        row = {
            "physical_asap7_status": claims.get("physical_asap7"),
            "deepscale_status": claims.get("deepscale"),
        }
        write_csv(report_dir / "hardware_comparison.csv", list(row), [row])
        return {
            "rows": [],
            "sources": [],
            "status": row,
            "target": expected["hardware_comparison_target"],
        }
    raw_path = output / "physical/asap7/ppa.json"
    scaled_path = output / "physical/asap7/ppa_28nm_estimated.json"
    raw = load(raw_path)
    scaled = load(scaled_path)
    target = expected["hardware_comparison_target"]
    rows = []
    fields = (
        ("area_mm2", "area_mm2"),
        ("total_power_w", "total_power_w"),
        ("max_frequency_mhz", "frequency_mhz"),
    )
    for metric, target_key in fields:
        raw_value = raw["metrics"][metric]
        estimate = scaled["scaled_metrics"][metric]
        paper = target[target_key]
        rows.append(
            {
                "metric": metric,
                "asap7_raw": raw_value,
                "deepscale_28nm_estimate": estimate,
                "paper_tsmc28_target": paper,
                "estimate_minus_target": estimate - paper,
                "estimate_relative_difference": (estimate - paper) / paper,
                "comparison_is_pass_fail": False,
            }
        )
    write_csv(report_dir / "hardware_comparison.csv", list(rows[0]), rows)
    return {
        "rows": rows,
        "sources": [raw_path, scaled_path],
        "status": {
            "physical_asap7_status": "CLAIMED",
            "deepscale_status": "CLAIMED",
        },
        "target": target,
    }


def write_markdown(
    report_dir: Path,
    tables: dict[str, Any],
    hardware: dict[str, Any],
    claims: dict[str, Any],
) -> None:
    lines = [
        "# SCARF Reproduction Report",
        "",
        "All measured values below come from archived result records. Paper values are comparison targets only.",
        "",
        "## Headline Results",
        "",
        f"- Dataset pairs: {len(tables['quality'])}",
        f"- Figure 8: {claims['figure8']}",
        f"- Figure 11: {claims['figure11']}",
        f"- Figures 13-16: {claims['sensitivity']}",
        "- Claimed Table 1 and Tables 2-3 rows are exported as CSV.",
        "",
        "## Hardware Scope",
        "",
    ]
    if hardware["rows"]:
        lines.extend(
            (
                "The ASAP7 values are predictive 7 nm results. The DeepScale values are deterministic 28 nm-equivalent estimates. The paper TSMC28 values include commercial SRAM/PHY collateral and are not a pass/fail target for this public flow.",
                "",
                "| Metric | ASAP7 raw | DeepScale 28 nm estimate | Paper TSMC28 target | Relative difference |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for row in hardware["rows"]:
            lines.append(
                f"| {row['metric']} | {row['asap7_raw']:.6g} | {row['deepscale_28nm_estimate']:.6g} | {row['paper_tsmc28_target']:.6g} | {row['estimate_relative_difference']:+.2%} |"
            )
    else:
        lines.extend(
            (
                f"- ASAP7: {hardware['status']['physical_asap7_status']}",
                f"- DeepScale: {hardware['status']['deepscale_status']}",
                "- No public PPA estimate is reported without a valid routed input.",
            )
        )
    lines.extend(("", "See `validation.json` for claim-level PASS/FAIL decisions."))
    (report_dir / "reproduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(output: Path, report_dir: Path) -> dict[str, Any]:
    output = output.resolve()
    report_dir = report_dir.resolve()
    expected = load(EXPECTED)
    claims = load(ROOT / "artifact/claim_status.json")
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(output, report_dir, expected, claims)
    sensitivity = (
        sensitivity_summary(output, report_dir, expected)
        if claims["sensitivity"] == "CLAIMED"
        else {"_source": []}
    )
    hardware = hardware_table(output, report_dir, expected, claims)
    catalog = render_figures(report_dir, tables, sensitivity, expected, claims)
    write_markdown(report_dir, tables, hardware, claims)
    source_paths = set(tables["sources"])
    source_paths.update(Path(path) for path in sensitivity["_source"])
    source_paths.update(hardware["sources"])
    record = {
        "schema_version": "1.0",
        "surface_class": "appendix",
        "generator": "scripts/generate_report.py",
        "source_data": [
            {
                "path": path.resolve().relative_to(output).as_posix(),
                "path_base": "output_root",
                "sha256": sha256_file(path),
            }
            for path in sorted(source_paths)
        ],
        "figures": catalog,
        "self_review": "Removed the Figure 8 legend overlap, added explicit quality traces to Figures 13-16, and replaced Figure 15 twin axes with a non-overlapping 2x2 layout.",
    }
    (report_dir / "figure_catalog.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        generate(args.input.resolve(), args.output_dir.resolve())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir / "reproduction_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
