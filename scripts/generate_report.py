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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.iflow.paper_table4 import (
    TABLE4_COMPONENTS,
    TABLE4_LABELS,
    TABLE4_METRICS,
    UNMODELED_COMPONENTS,
)


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
EVIDENCE_CLASSES = {
    "independent_measurement",
    "deterministic_execution",
    "public_physical_proxy",
    "paper_comparison_target",
}


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


def find_pair_optional(output: Path, pair: str, modes: tuple[str, ...]) -> Path | None:
    try:
        return find_pair(output, pair, modes)
    except FileNotFoundError:
        return None


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


def validate_figure_catalog(
    generated: dict[str, Any],
    contract: dict[str, Any],
    *,
    require_key_results: bool = False,
) -> None:
    """Validate one-to-one result coverage and evidence-class strength."""
    if generated.get("schema_version") != "2.0":
        raise ValueError("generated figure catalog schema must be 2.0")
    expected = {item["id"]: item for item in contract.get("results", [])}
    rows = generated.get("results")
    if not isinstance(rows, list):
        raise ValueError("generated figure catalog results must be a list")
    actual = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("generated figure catalog has an invalid result row")
        result_id = row["id"]
        if result_id in actual:
            raise ValueError(f"duplicate generated result: {result_id}")
        actual[result_id] = row
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"generated result coverage mismatch: missing={missing}, extra={extra}")

    for result_id, requirement in expected.items():
        row = actual[result_id]
        evidence_class = row.get("evidence_class")
        if evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"{result_id} has an invalid evidence class")
        required_class = requirement["required_evidence_class"]
        if evidence_class != required_class:
            raise ValueError(
                f"{result_id} evidence class {evidence_class} does not satisfy {required_class}"
            )
        status = row.get("status")
        if status not in {"PASS", "FAIL", "NOT_RUN"}:
            raise ValueError(f"{result_id} has an invalid status")
        sources = row.get("source_data")
        if not isinstance(sources, list):
            raise ValueError(f"{result_id} source_data must be a list")
        if status == "PASS" and not sources:
            raise ValueError(f"{result_id} has no raw source data")
        for source in sources:
            digest = source.get("sha256") if isinstance(source, dict) else None
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{result_id} has an invalid source-data SHA256")
        exports = row.get("exports")
        if not isinstance(exports, list) or not exports:
            raise ValueError(f"{result_id} has no exported artifact")
        if (
            require_key_results
            and requirement.get("claim_role") == "mandatory_key_result"
            and row.get("status") != "PASS"
        ):
            raise ValueError(f"mandatory key result did not pass: {result_id}")


def build_tables(
    output: Path, report_dir: Path, expected: dict[str, Any], claims: dict[str, Any]
) -> dict[str, Any]:
    quality_rows = []
    speedup_rows = []
    ablation_rows = []
    fsdr_rows = []
    saes_rows = []
    sources = []
    for pair in expected["table1"]:
        quality_path = find_pair_optional(output, pair, ("quality", "all", "ablation"))
        if quality_path is None:
            continue
        ablation_path = find_pair_optional(
            output, pair, ("mechanisms", "ablation", "all", "quality")
        )
        quality = load(quality_path)
        ablation = load(ablation_path) if ablation_path is not None else None
        sources.append(quality_path)
        if ablation_path is not None:
            sources.append(ablation_path)
        row: dict[str, Any] = {"pair": pair, "sample_count": quality["provenance"]["evaluation"]["sample_count"]}
        for variant in ("baseline", "scarf"):
            for metric in METRICS:
                row[f"{variant}_{metric}"] = quality["quality"][variant][metric]
        quality_rows.append(row)
        speed_path = find_pair_optional(output, pair, ("performance", "speedup", "orin"))
        if speed_path is not None:
            speed = load(speed_path)
            if speed.get("performance", {}).get("baseline_source") == "orin_nx_cuda_events":
                sources.append(speed_path)
                speedup_rows.append(
                    {"pair": pair, "speedup": speed["performance"]["speedup"]}
                )

        if ablation is not None:
            base = float(ablation["ablation"]["asic"]["eff_total"])
            ablation_rows.append(
                {
                    "pair": pair,
                    "fsdr_speedup": base / float(ablation["ablation"]["asic_fsdr"]["eff_total"]),
                    "saes_speedup": base / float(ablation["ablation"]["asic_saes"]["eff_total"]),
                    "combined_speedup": base / float(ablation["ablation"]["asic_fsdr_saes"]["eff_total"]),
                }
            )
            events = ablation.get("events", {})
            fsdr_events = events.get("fsdr", {})
            saes_events = events.get("saes", {})
            fsdr = ablation.get("fsdr_saes", {}).get("fsdr", {})
            saes = ablation.get("fsdr_saes", {}).get("saes", {})
            preservation = ablation.get("fsdr_saes", {}).get("preservation", {})
            guided_pixels = fsdr_events.get("guided_pixels")
            total_pixels = fsdr_events.get("total_pixels")
            top1_total = (
                fsdr_events.get("guided_top1_covered", 0)
                + fsdr_events.get("guided_top1_missed", 0)
                if fsdr_events.get("discrete_top1_available") is True
                else 0
            )
            fsdr_rows.append(
                {
                    "pair": pair,
                    "guided_rate": (
                        float(guided_pixels) / float(total_pixels)
                        if isinstance(guided_pixels, (int, float)) and total_pixels
                        else fsdr.get("guided_rate")
                    ),
                    "top1_coverage": (
                        float(fsdr_events["guided_top1_covered"]) / float(top1_total)
                        if top1_total
                        else None
                    ),
                    "full_depth_evaluations": (
                        fsdr_events.get("full_depth_evaluations")
                        if fsdr_events.get("depth_evaluations_available") is True
                        else None
                    ),
                    "executed_depth_evaluations": (
                        fsdr_events.get("executed_depth_evaluations")
                        if fsdr_events.get("depth_evaluations_available") is True
                        else None
                    ),
                    "feature_buffer_bytes_baseline": (
                        fsdr_events.get("feature_buffer_bytes_baseline")
                        if fsdr_events.get("feature_buffer_bytes_available") is True
                        else None
                    ),
                    "feature_buffer_bytes_actual": (
                        fsdr_events.get("feature_buffer_bytes_actual")
                        if fsdr_events.get("feature_buffer_bytes_available") is True
                        else None
                    ),
                }
            )
            total_tiles = saes_events.get("total_tiles")
            baseline_gaussians = saes_events.get("baseline_gaussians")
            saes_rows.append(
                {
                    "pair": pair,
                    "level0_rate": (
                        float(saes_events["level0_tiles"]) / float(total_tiles)
                        if saes_events.get("tile_path_available") is True and total_tiles
                        else saes.get("level0_ratio")
                    ),
                    "level1_rate": (
                        float(saes_events["level1_tiles"]) / float(total_tiles)
                        if saes_events.get("tile_path_available") is True and total_tiles
                        else saes.get("level1_ratio")
                    ),
                    "full_rate": (
                        float(saes_events["full_tiles"]) / float(total_tiles)
                        if saes_events.get("tile_path_available") is True and total_tiles
                        else saes.get("full_ratio")
                    ),
                    "low_variance_agreement": preservation.get("saes_low_var_agree"),
                    "gaussians_saved": (
                        1.0
                        - float(saes_events["actual_gaussians"])
                        / float(baseline_gaussians)
                        if saes_events.get("gaussian_counts_available") is True
                        and baseline_gaussians
                        else saes.get("modification_ratio")
                    ),
                    "full_s2_evaluations": (
                        saes_events.get("full_s2_evaluations")
                        if saes_events.get("s2_evaluations_available") is True
                        else None
                    ),
                    "executed_s2_evaluations": (
                        saes_events.get("executed_s2_evaluations")
                        if saes_events.get("s2_evaluations_available") is True
                        else None
                    ),
                }
            )

    quality_fields = [
        "pair",
        "sample_count",
        *(f"{variant}_{metric}" for variant in ("baseline", "scarf") for metric in METRICS),
    ]
    ablation_fields = ["pair", "fsdr_speedup", "saes_speedup", "combined_speedup"]
    fsdr_fields = [
        "pair",
        "guided_rate",
        "top1_coverage",
        "full_depth_evaluations",
        "executed_depth_evaluations",
        "feature_buffer_bytes_baseline",
        "feature_buffer_bytes_actual",
    ]
    saes_fields = [
        "pair",
        "level0_rate",
        "level1_rate",
        "full_rate",
        "low_variance_agreement",
        "gaussians_saved",
        "full_s2_evaluations",
        "executed_s2_evaluations",
    ]
    write_csv(report_dir / "table1_quality.csv", quality_fields, quality_rows)
    write_csv(report_dir / "figure8_speedup.csv", ["pair", "speedup"], speedup_rows)
    write_csv(report_dir / "figure11_ablation.csv", ablation_fields, ablation_rows)
    write_csv(report_dir / "table2_fsdr.csv", fsdr_fields, fsdr_rows)
    write_csv(report_dir / "table3_saes.csv", saes_fields, saes_rows)
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
    if tables["speedup"]:
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

    if tables["ablation"]:
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

    if not sensitivity.get("_source"):
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


def _load_worstcase_candidate(path: Path) -> dict[str, Any]:
    record = load(path)
    kind = record.get("kind")
    if kind not in {"fsdr_worstcase_view", "saes_worstcase_view"}:
        raise ValueError(f"invalid worst-case manifest kind: {path}")
    pair_dir = path.parents[2]
    source = record.get("source_result")
    if not isinstance(source, dict):
        raise ValueError(f"worst-case source result is missing: {path}")
    source_path = pair_dir / str(source.get("path", ""))
    if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
        raise ValueError(f"worst-case source result hash mismatch: {path}")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "ground_truth",
        "reference",
        "optimized",
    }:
        raise ValueError(f"worst-case image set is incomplete: {path}")
    artifact_paths = {}
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise ValueError(f"worst-case image record is invalid: {path}")
        image_path = path.parent / str(artifact.get("path", ""))
        if not image_path.is_file() or sha256_file(image_path) != artifact.get("sha256"):
            raise ValueError(f"worst-case image hash mismatch: {image_path}")
        artifact_paths[name] = image_path
    loss = record.get("loss_value")
    if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(loss):
        raise ValueError(f"worst-case loss is invalid: {path}")
    return {
        **record,
        "pair": pair_dir.name.replace("_", "/", 1),
        "manifest_path": path,
        "source_path": source_path,
        "artifact_paths": artifact_paths,
    }


def render_worstcase(output: Path, report_dir: Path) -> dict[str, Any] | None:
    manifests = sorted(output.glob("quality/*/worstcase/*/manifest.json"))
    grouped: dict[str, list[dict[str, Any]]] = {"fsdr": [], "saes": []}
    for path in manifests:
        candidate = _load_worstcase_candidate(path)
        grouped[candidate["kind"].split("_", 1)[0]].append(candidate)
    if any(not grouped[kind] for kind in grouped):
        return None
    selected = {
        kind: max(candidates, key=lambda item: float(item["loss_value"]))
        for kind, candidates in grouped.items()
    }

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use(STYLE)
    fig, axes = plt.subplots(2, 5, figsize=(10.0, 4.3))
    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "figure10_worstcase_evidence",
        "selection_rule": {
            "fsdr": "maximum untransformed psnr_loss_db over every recorded target view",
            "saes": "maximum untransformed lpips_increase over every recorded target view",
        },
        "selected": {},
    }
    for row, kind in enumerate(("fsdr", "saes")):
        item = selected[kind]
        images = {
            name: np.asarray(mpimg.imread(path))[..., :3]
            for name, path in item["artifact_paths"].items()
        }
        error = np.abs(images["optimized"].astype(np.float32) - images["reference"].astype(np.float32))
        for column, (name, title) in enumerate(
            (
                ("ground_truth", "Ground truth"),
                ("reference", "No optimization"),
                ("optimized", kind.upper()),
            )
        ):
            axes[row, column].imshow(images[name])
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
        axes[row, 3].imshow(error.clip(0.0, 1.0))
        axes[row, 3].set_title("Absolute RGB error")
        axes[row, 3].axis("off")

        source_result = load(item["source_path"])
        ablation = source_result["ablation"]
        optimized_key = "asic_fsdr" if kind == "fsdr" else "asic_saes"
        stage_fields = ("eff_feature", "eff_dp_core", "eff_gauss_gen")
        labels = ("S1", "S2", "S3")
        reference = [float(ablation["asic"][field]) for field in stage_fields]
        optimized = [float(ablation[optimized_key][field]) for field in stage_fields]
        x = np.arange(len(labels))
        axes[row, 4].bar(x - 0.18, reference, width=0.36, color=COLORS["neutral"], label="No opt")
        axes[row, 4].bar(x + 0.18, optimized, width=0.36, color=COLORS["main"], label=kind.upper())
        axes[row, 4].set_xticks(x, labels)
        axes[row, 4].set_ylabel("Cycles")
        axes[row, 4].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[row, 4].legend(loc="best")
        axes[row, 0].set_ylabel(
            f"{kind.upper()}\n{item['pair']}\nscene {item['scene']} / view {item['target_index']}"
        )
        evidence["selected"][kind] = {
            "pair": item["pair"],
            "sample_index": item["sample_index"],
            "scene": item["scene"],
            "target_index": item["target_index"],
            "loss_metric": item["loss_metric"],
            "loss_value": item["loss_value"],
            "manifest": {
                "path": item["manifest_path"].resolve().relative_to(output).as_posix(),
                "sha256": sha256_file(item["manifest_path"]),
            },
            "source_result": {
                "path": item["source_path"].resolve().relative_to(output).as_posix(),
                "sha256": sha256_file(item["source_path"]),
            },
            "error_map": {
                "definition": "elementwise absolute RGB difference between optimized and no-optimization images",
                "mean": float(error.mean()),
                "maximum": float(error.max()),
            },
            "latency_cycles": {
                "no_optimization": dict(zip(labels, reference)),
                "optimized": dict(zip(labels, optimized)),
            },
        }
    fig.tight_layout()
    evidence_path = report_dir / "figure10_worstcase.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    exports = save_figure(fig, report_dir, "figure10_worstcase")
    plt.close(fig)
    return {
        "id": "figure10",
        "exports": [*exports, str(evidence_path)],
        "acceptance_pass": True,
        "claim": "Worst views selected from raw per-view losses with matching RGB error maps and cycles",
    }


def render_utilization(
    output: Path, report_dir: Path, expected: dict[str, Any]
) -> dict[str, Any] | None:
    targets = expected["figure12"]["utilization"]
    paths = sorted(output.glob("utilization/*/results.json"))
    records = {}
    source_data = []
    for path in paths:
        record = load(path)
        provenance = record.get("provenance", {})
        model = provenance.get("model")
        dataset = provenance.get("dataset", {}).get("name")
        pair = f"{model}/{dataset}"
        if pair not in targets or pair in records:
            continue
        stages = record.get("performance", {}).get("stages")
        if not isinstance(stages, dict):
            continue
        ratios = {}
        complete = True
        for stage in ("s1", "s2", "s3"):
            value = stages.get(stage, {})
            useful = value.get("useful_mmcu_slots")
            scheduled = value.get("scheduled_mmcu_slots")
            if (
                value.get("mmcu_slots_available") is not True
                or not isinstance(useful, (int, float))
                or not isinstance(scheduled, (int, float))
                or isinstance(useful, bool)
                or isinstance(scheduled, bool)
                or scheduled <= 0
                or useful < 0
                or useful > scheduled
            ):
                complete = False
                break
            ratios[stage] = float(useful) / float(scheduled)
        if complete:
            records[pair] = ratios
            source_data.append(
                {
                    "path": path.resolve().relative_to(output).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    if set(records) != set(targets):
        return None

    tolerance = float(expected["figure12"]["tolerance_absolute"])
    rows = []
    acceptance_pass = True
    for pair in targets:
        for stage in ("s1", "s2", "s3"):
            actual = records[pair][stage]
            target = float(targets[pair][stage])
            passed = abs(actual - target) <= tolerance
            acceptance_pass = acceptance_pass and passed
            rows.append(
                {
                    "pair": pair,
                    "stage": stage,
                    "utilization": actual,
                    "paper_target": target,
                    "absolute_error": abs(actual - target),
                    "pass": passed,
                }
            )
    csv_path = report_dir / "figure12_utilization.csv"
    write_csv(csv_path, list(rows[0]), rows)

    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    pairs = list(targets)
    x = np.arange(len(pairs))
    width = 0.24
    for offset, stage, color in (
        (-width, "s1", COLORS["main"]),
        (0.0, "s2", COLORS["neutral"]),
        (width, "s3", COLORS["quality"]),
    ):
        ax.bar(
            x + offset,
            [records[pair][stage] * 100.0 for pair in pairs],
            width=width,
            label=stage.upper(),
            color=color,
        )
    ax.set_ylabel("MMCU active utilization (%)")
    ax.set_xticks(x, [PAIR_LABELS[pair] for pair in pairs], rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(ncols=3, loc="upper center")
    fig.tight_layout()
    evidence_path = report_dir / "figure12_utilization.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "definition": "useful MMCU slots divided by scheduled MMCU slots",
                "tolerance_absolute": tolerance,
                "acceptance_pass": acceptance_pass,
                "rows": rows,
                "source_data": source_data,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    exports = save_figure(fig, report_dir, "figure12_utilization")
    plt.close(fig)
    return {
        "id": "figure12",
        "exports": [*exports, str(csv_path), str(evidence_path)],
        "acceptance_pass": acceptance_pass,
        "claim": "Useful/scheduled MMCU slots for every stage and pair",
    }


def hardware_table(
    output: Path,
    report_dir: Path,
    expected: dict[str, Any],
    claims: dict[str, Any],
) -> dict[str, Any]:
    raw_path = output / "physical/asap7/ppa.json"
    scaled_path = output / "physical/asap7/ppa_28nm_estimated.json"
    aggregate_fields = [
        "metric",
        "asap7_raw",
        "deepscale_28nm_estimate",
        "paper_tsmc28_target",
        "estimate_minus_target",
        "estimate_relative_difference",
        "comparison_is_pass_fail",
    ]
    hierarchy_fields = [
        "component_key",
        "component",
        "metric",
        "public_proxy_scope",
        "asap7_raw",
        "deepscale_28nm_estimate",
        "paper_tsmc28_target",
        "asap7_evidence_class",
        "estimate_evidence_class",
        "paper_evidence_class",
        "comparison_is_pass_fail",
    ]
    if not raw_path.is_file() or not scaled_path.is_file():
        row = {
            "physical_asap7_status": claims.get("physical_asap7"),
            "deepscale_status": claims.get("deepscale"),
        }
        write_csv(report_dir / "hardware_comparison.csv", list(row), [row])
        write_csv(report_dir / "table4_hierarchy.csv", hierarchy_fields, [])
        return {
            "rows": [],
            "hierarchy_rows": [],
            "sources": [],
            "status": row,
            "target": expected["hardware_comparison_target"],
        }
    raw = load(raw_path)
    scaled = load(scaled_path)
    if raw.get("physical_valid") is not True:
        raise ValueError("ASAP7 PPA must be a complete physical_valid routed record")
    if raw.get("evidence_type") != "asap7_predictive_postroute":
        raise ValueError("ASAP7 PPA has an invalid evidence type")
    if raw.get("scope", {}).get("io_phy_included") is not False:
        raise ValueError("public ASAP7 PPA must explicitly exclude I/O and PHY")
    if scaled.get("validation") != {
        "raw_preserved": True,
        "foundry_measurement": False,
    }:
        raise ValueError("DeepScale result must be a non-foundry estimate preserving raw PPA")
    if scaled.get("raw") != raw:
        raise ValueError("DeepScale result is not bound to the supplied raw ASAP7 PPA")
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
    write_csv(report_dir / "hardware_comparison.csv", aggregate_fields, rows)

    raw_hierarchy = raw.get("metrics", {}).get("hierarchy")
    scaled_hierarchy = scaled.get("scaled_hierarchy")
    target_hierarchy = target.get("hierarchy")
    if not all(
        isinstance(value, dict) and value
        for value in (raw_hierarchy, scaled_hierarchy, target_hierarchy)
    ):
        raise ValueError("Table 4 requires raw, scaled, and paper hierarchy records")
    if tuple(target_hierarchy) != TABLE4_COMPONENTS:
        raise ValueError("paper Table 4 hierarchy is not in the canonical paper order")
    if set(raw_hierarchy) != set(TABLE4_COMPONENTS):
        raise ValueError("ASAP7 hierarchy does not cover every Table 4 component")
    if set(scaled_hierarchy) != set(TABLE4_COMPONENTS):
        raise ValueError("DeepScale hierarchy does not match the raw ASAP7 hierarchy")

    hierarchy_rows = []
    for component in TABLE4_COMPONENTS:
        raw_component = raw_hierarchy.get(component, {})
        estimate_component = scaled_hierarchy.get(component, {})
        paper_component = target_hierarchy[component]
        if component == "on_chip_buffers" or component.startswith(
            "on_chip_buffers_"
        ):
            scope = "abstract_sram_area_only"
        elif component in UNMODELED_COMPONENTS:
            scope = "not_modeled_in_public_proxy"
        elif component == "routing_filler":
            scope = "routed_floorplan_area_only"
        elif component == "total_die":
            scope = "routed_public_proxy_total_excluding_io_phy"
        else:
            scope = "public_logic_proxy"
        for metric in TABLE4_METRICS[component]:
            raw_value = raw_component.get(metric)
            estimate_value = estimate_component.get(metric)
            paper_value = paper_component.get(metric)
            for label, value in (
                ("ASAP7", raw_value),
                ("DeepScale", estimate_value),
                ("paper", paper_value),
            ):
                if value is not None and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        f"Table 4 {label} value is invalid: {component}.{metric}"
                    )
            hierarchy_rows.append(
                {
                    "component_key": component,
                    "component": TABLE4_LABELS.get(component, component),
                    "metric": metric,
                    "public_proxy_scope": scope,
                    "asap7_raw": raw_value,
                    "deepscale_28nm_estimate": estimate_value,
                    "paper_tsmc28_target": paper_value,
                    "asap7_evidence_class": (
                        "not_modeled"
                        if component in UNMODELED_COMPONENTS
                        else "public_physical_proxy"
                    ),
                    "estimate_evidence_class": (
                        "not_modeled"
                        if component in UNMODELED_COMPONENTS
                        else "public_physical_proxy"
                    ),
                    "paper_evidence_class": "paper_comparison_target",
                    "comparison_is_pass_fail": False,
                }
            )
    write_csv(report_dir / "table4_hierarchy.csv", hierarchy_fields, hierarchy_rows)
    return {
        "rows": rows,
        "hierarchy_rows": hierarchy_rows,
        "sources": [raw_path, scaled_path],
        "status": {
            "physical_asap7_status": "CLAIMED",
            "deepscale_status": "CLAIMED",
        },
        "target": target,
    }


def _positive_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def render_efficiency_proxy(
    output: Path,
    report_dir: Path,
    hardware: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    """Render Figure 9 without treating smoke traces as inference energy."""
    if not hardware.get("rows"):
        return None
    raw_path = output / "physical/asap7/ppa.json"
    raw = load(raw_path)
    if raw.get("scope", {}).get("power_activity") != "workload_vcd":
        return None
    activity = raw.get("artifacts", {}).get("activity_vcd")
    if not isinstance(activity, dict) or activity.get("path_base") != "output_dir":
        return None
    activity_path = raw_path.parent / str(activity.get("path", ""))
    if not activity_path.is_file() or sha256_file(activity_path) != activity.get("sha256"):
        raise ValueError("ASAP7 activity VCD is missing or does not match its hash")

    raw_area = _positive_number(raw["metrics"].get("area_mm2"), "ASAP7 area")
    raw_power = _positive_number(
        raw["metrics"].get("total_power_w"), "ASAP7 total power"
    )
    raw_frequency_hz = _positive_number(
        raw["metrics"].get("max_frequency_mhz"), "ASAP7 frequency"
    ) * 1_000_000.0
    from hardware.scaling.deepscale import scale_value

    rows = []
    source_data = [
        {
            "path": path.resolve().relative_to(output).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in hardware["sources"]
    ]
    for pair in expected["table1"]:
        software_path = find_pair_optional(output, pair, ("quality", "all"))
        dram_path = output / "dram/workloads" / pair.replace("/", "_") / "results.json"
        if software_path is None or not dram_path.is_file():
            return None
        software = load(software_path)
        dataset = software.get("provenance", {}).get("dataset", {})
        evaluation = software.get("provenance", {}).get("evaluation", {})
        if (
            software.get("schema_version") != "2.0"
            or software.get("evidence_class") != "deterministic_execution"
            or dataset.get("paper_result_eligible") is not True
            or evaluation.get("kind") != "dataset_aggregate"
        ):
            return None
        selection_hash = evaluation.get("sample_selection_sha256")
        if not isinstance(selection_hash, str) or len(selection_hash) != 64:
            return None
        dram = load(dram_path)
        workload = dram.get("workload", {})
        model, dataset_name = pair.split("/")
        if (
            dram.get("evidence_type") != "public_memory_system_proxy"
            or dram.get("paper_lpddr4x_reproduced") is not False
            or dram.get("scope", {}).get("claim")
            != "per_inference_workload_proxy"
            or workload.get("model") != model
            or workload.get("dataset") != dataset_name
            or workload.get("sample_selection_sha256") != selection_hash
            or workload.get("software_result", {}).get("sha256")
            != sha256_file(software_path)
        ):
            return None
        offchip_energy = _positive_number(
            dram.get("metrics", {}).get("drampower_offchip_energy_per_inference_j"),
            f"{pair} DRAM energy",
        )
        cycles = _positive_number(
            software.get("performance", {}).get("scarf_cycles"),
            f"{pair} SCARF cycles",
        )
        compute_energy = raw_power * cycles / raw_frequency_hz
        raw_energy = compute_energy + offchip_energy
        estimated_compute_energy = scale_value("energy", compute_energy, 7, 28)
        estimated_energy = estimated_compute_energy + offchip_energy
        raw_throughput_per_area = raw_frequency_hz / cycles / raw_area
        estimated_throughput_per_area = scale_value(
            "throughput_per_area", raw_throughput_per_area, 7, 28
        )
        rows.append(
            {
                "pair": pair,
                "cycles": cycles,
                "asap7_compute_energy_j": compute_energy,
                "dram_lpddr5_energy_j": offchip_energy,
                "asap7_total_proxy_energy_j": raw_energy,
                "deepscale_28nm_total_proxy_energy_j": estimated_energy,
                "asap7_energy_efficiency_inferences_per_j": 1.0 / raw_energy,
                "deepscale_28nm_energy_efficiency_inferences_per_j": 1.0
                / estimated_energy,
                "asap7_throughput_per_area": raw_throughput_per_area,
                "deepscale_28nm_throughput_per_area": estimated_throughput_per_area,
                "paper_values_are_normalized_targets_only": True,
            }
        )
        source_data.extend(
            {
                "path": path.resolve().relative_to(output).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in (software_path, dram_path)
        )
    csv_path = report_dir / "figure9_public_proxy.csv"
    write_csv(csv_path, list(rows[0]), rows)

    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6))
    x = np.arange(len(rows))
    width = 0.36
    labels = [PAIR_LABELS[row["pair"]] for row in rows]
    axes[0, 0].bar(
        x - width / 2,
        [row["asap7_energy_efficiency_inferences_per_j"] for row in rows],
        width,
        color=COLORS["main"],
        label="ASAP7 raw proxy",
    )
    axes[0, 0].bar(
        x + width / 2,
        [row["deepscale_28nm_energy_efficiency_inferences_per_j"] for row in rows],
        width,
        color=COLORS["neutral"],
        label="28 nm-equivalent proxy",
    )
    axes[0, 0].set_ylabel("Inferences / J")
    axes[0, 0].legend()
    axes[0, 1].bar(
        x - width / 2,
        [row["asap7_throughput_per_area"] for row in rows],
        width,
        color=COLORS["main"],
        label="ASAP7 raw proxy",
    )
    axes[0, 1].bar(
        x + width / 2,
        [row["deepscale_28nm_throughput_per_area"] for row in rows],
        width,
        color=COLORS["neutral"],
        label="28 nm-equivalent proxy",
    )
    axes[0, 1].set_ylabel("Inferences / s / mm$^2$")
    axes[0, 1].legend()

    targets = expected["figure9"]["paper_comparison_target"]
    for column, metric in enumerate(("energy_efficiency", "throughput_per_area")):
        axis = axes[1, column]
        for offset, key, color, label in (
            (-width, "orin_nx", COLORS["neutral"], "Paper Orin NX"),
            (0.0, "tsmc28", COLORS["main"], "Paper TSMC28"),
            (width, "nm8_equivalent", COLORS["target"], "Paper 8 nm-eq."),
        ):
            axis.bar(
                x + offset,
                [targets[row["pair"]][metric][key] for row in rows],
                width,
                color=color,
                label=label,
            )
        axis.set_ylabel("Paper normalized target (x)")
        axis.legend(ncols=3, fontsize=7)
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 0].set_title("Public energy-efficiency counterpart")
    axes[0, 1].set_title("Public throughput/area counterpart")
    axes[1, 0].set_title("Paper energy-efficiency targets (comparison only)")
    axes[1, 1].set_title("Paper throughput/area targets (comparison only)")
    fig.tight_layout()

    evidence_path = report_dir / "figure9_public_proxy.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evidence_class": "public_physical_proxy",
                "rows": rows,
                "source_data": source_data,
                "definitions": {
                    "compute_energy": "activity-aware ASAP7 power multiplied by simulator cycles divided by routed maximum frequency",
                    "offchip_energy": "per-inference LPDDR5 public proxy from the matching workload trace; not paper LPDDR4X",
                    "deepscale_energy": "logic compute energy scaled with the DeepScale energy table; external DRAM energy is not technology-scaled",
                    "throughput_per_area": "ASAP7 value scaled directly with the DeepScale throughput-per-area table",
                    "paper_targets": "read-only normalized comparison targets; not used for public-proxy PASS/FAIL",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    exports = save_figure(fig, report_dir, "figure9_public_proxy")
    plt.close(fig)
    return {
        "id": "figure9",
        "exports": [*exports, str(csv_path), str(evidence_path)],
        "acceptance_pass": True,
        "claim": "Activity-aware ASAP7 and deterministic 28 nm-equivalent public counterparts",
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
        "- Table 1 and Tables 2-3 exports follow the machine-readable claim status; header-only files mean no row is claimed.",
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


def _relative_export(path: str | Path, report_dir: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(report_dir).as_posix()
    except ValueError:
        return resolved.as_posix()


def _source_records(output: Path, patterns: list[str]) -> list[dict[str, str]]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in output.glob(pattern) if path.is_file())
    return [
        {
            "path": path.resolve().relative_to(output).as_posix(),
            "path_base": "output_root",
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _table1_pass(rows: list[dict[str, Any]], expected: dict[str, Any]) -> bool:
    if {row["pair"] for row in rows} != set(expected["table1"]):
        return False
    tolerances = expected["tolerances"]
    for row in rows:
        targets = expected["table1"][row["pair"]]
        for variant in ("baseline", "scarf"):
            for metric, target in zip(METRICS, targets[variant]):
                if abs(float(row[f"{variant}_{metric}"]) - float(target)) > float(
                    tolerances[metric]
                ):
                    return False
    return True


def _ablation_pass(rows: list[dict[str, Any]], expected: dict[str, Any]) -> bool:
    if len(rows) != len(expected["table1"]):
        return False
    tolerance = float(expected["tolerances"]["speedup_relative"])
    for name in ("fsdr", "saes", "combined"):
        actual = geometric_mean([float(row[f"{name}_speedup"]) for row in rows])
        target = float(expected["ablation"][name])
        if abs(actual - target) / target > tolerance:
            return False
    return True


def _mechanism_pass(
    rows: list[dict[str, Any]], expected: dict[str, Any], fields: tuple[str, ...]
) -> bool:
    if {row["pair"] for row in rows} != set(expected["mechanisms"]):
        return False
    tolerance = float(expected["mechanism_tolerance_absolute"])
    for row in rows:
        targets = expected["mechanisms"][row["pair"]]
        for field in fields:
            value = row.get(field)
            if value is None or abs(float(value) - float(targets[field])) > tolerance:
                return False
    return True


def build_result_catalog(
    *,
    output: Path,
    report_dir: Path,
    contract: dict[str, Any],
    expected: dict[str, Any],
    tables: dict[str, Any],
    sensitivity: dict[str, Any],
    hardware: dict[str, Any],
    figure_exports: list[dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    exports_by_id = {
        item["id"]: [_relative_export(path, report_dir) for path in item["exports"]]
        for item in figure_exports
    }
    acceptance_by_id = {
        item["id"]: item.get("acceptance_pass", True) for item in figure_exports
    }
    exports_by_id.update(
        {
            "table1": ["table1_quality.csv"],
            "table2": ["table2_fsdr.csv"],
            "table3": ["table3_saes.csv"],
            "table4": ["hardware_comparison.csv", "table4_hierarchy.csv"],
        }
    )
    sensitivity_ready = bool(sensitivity.get("_source"))
    readiness = {
        "figure8": len(tables["speedup"]) == 9 and "figure8" in exports_by_id,
        "figure9": bool(hardware["rows"])
        and "figure9" in exports_by_id
        and acceptance_by_id["figure9"],
        "table1": _table1_pass(tables["quality"], expected),
        "figure10": "figure10" in exports_by_id and acceptance_by_id["figure10"],
        "figure11": _ablation_pass(tables["ablation"], expected),
        "table2": _mechanism_pass(
            tables["fsdr"], expected, ("guided_rate", "top1_coverage")
        )
        and all(
            row.get(field) is not None
            for row in tables["fsdr"]
            for field in (
                "full_depth_evaluations",
                "executed_depth_evaluations",
                "feature_buffer_bytes_baseline",
                "feature_buffer_bytes_actual",
            )
        ),
        "table3": _mechanism_pass(
            tables["saes"],
            expected,
            ("level0_rate", "level1_rate", "low_variance_agreement", "gaussians_saved"),
        )
        and all(
            row.get(field) is not None
            for row in tables["saes"]
            for field in ("full_rate", "full_s2_evaluations", "executed_s2_evaluations")
        ),
        "figure12": "figure12" in exports_by_id and acceptance_by_id["figure12"],
        "figure13": sensitivity_ready and "figure13" in exports_by_id,
        "figure14": sensitivity_ready and "figure14" in exports_by_id,
        "figure15": sensitivity_ready and "figure15" in exports_by_id,
        "figure16": sensitivity_ready and "figure16" in exports_by_id,
        "table4": bool(hardware["rows"]),
    }
    statuses = report_dir / "status"
    statuses.mkdir(exist_ok=True)
    rows = []
    for requirement in contract["results"]:
        result_id = requirement["id"]
        selected = result_id in selected_ids
        sources = (
            _source_records(output, requirement["raw_inputs"]) if selected else []
        )
        passed = selected and bool(readiness.get(result_id)) and bool(sources)
        if passed:
            status = "PASS"
            reason = "generated from complete raw evidence and passed fixed checks"
            exports = exports_by_id[result_id]
        else:
            critical_source_present = bool(sources)
            if result_id == "figure8":
                critical_source_present = any(
                    "orin-evidence/measurement.json" in item["path"] for item in sources
                )
            status = "FAIL" if selected and critical_source_present else "NOT_RUN"
            reason = (
                "result was not selected for this report invocation"
                if not selected
                else (
                    "raw evidence is present but incomplete or outside the fixed acceptance gate"
                    if status == "FAIL"
                    else "required raw evidence is not present"
                )
            )
            status_path = statuses / f"{result_id}.json"
            status_path.write_text(
                json.dumps(
                    {
                        "id": result_id,
                        "status": status,
                        "reason": reason,
                        "required_evidence_class": requirement["required_evidence_class"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            exports = [status_path.relative_to(report_dir).as_posix()]
        rows.append(
            {
                "id": result_id,
                "kind": requirement["kind"],
                "title": requirement["title"],
                "claim_role": requirement["claim_role"],
                "selected": selected,
                "evidence_class": requirement["required_evidence_class"],
                "status": status,
                "reason": reason,
                "source_data": sources,
                "exports": exports,
                "command": requirement["command"],
                "acceptance": requirement["acceptance"],
            }
        )
    return rows


def parse_result_selection(value: str, contract: dict[str, Any]) -> set[str]:
    available = {item["id"] for item in contract["results"]}
    if value == "all":
        return available
    selected = {item.strip() for item in value.split(",") if item.strip()}
    if not selected:
        raise ValueError("figure selection cannot be empty")
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(f"unknown result id: {', '.join(unknown)}")
    return selected


def generate(output: Path, report_dir: Path, figures: str = "all") -> dict[str, Any]:
    output = output.resolve()
    report_dir = report_dir.resolve()
    expected = load(EXPECTED)
    contract = load(ROOT / "artifact/evaluation_catalog.json")
    selected_ids = parse_result_selection(figures, contract)
    claims = load(ROOT / "artifact/claim_status.json")
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(output, report_dir, expected, claims)
    sensitivity = (
        sensitivity_summary(output, report_dir, expected)
        if (output / "sensitivity/results.json").is_file()
        else {"_source": []}
    )
    hardware = hardware_table(output, report_dir, expected, claims)
    figure_exports = render_figures(report_dir, tables, sensitivity, expected, claims)
    efficiency_export = render_efficiency_proxy(output, report_dir, hardware, expected)
    if efficiency_export is not None:
        figure_exports.append(efficiency_export)
    worstcase_export = render_worstcase(output, report_dir)
    if worstcase_export is not None:
        figure_exports.append(worstcase_export)
    utilization_export = render_utilization(output, report_dir, expected)
    if utilization_export is not None:
        figure_exports.append(utilization_export)
    write_markdown(report_dir, tables, hardware, claims)
    catalog = build_result_catalog(
        output=output,
        report_dir=report_dir,
        contract=contract,
        expected=expected,
        tables=tables,
        sensitivity=sensitivity,
        hardware=hardware,
        figure_exports=figure_exports,
        selected_ids=selected_ids,
    )
    record = {
        "schema_version": "2.0",
        "surface_class": "appendix",
        "generator": "scripts/generate_report.py",
        "results": catalog,
        "self_review": "Removed the Figure 8 legend overlap, added explicit quality traces to Figures 13-16, and replaced Figure 15 twin axes with a non-overlapping 2x2 layout.",
    }
    validate_figure_catalog(record, contract, require_key_results=False)
    (report_dir / "figure_catalog.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--figures",
        default="all",
        help="Comma-separated figure/table IDs or all",
    )
    args = parser.parse_args()
    try:
        generate(args.input.resolve(), args.output_dir.resolve(), args.figures)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir / "reproduction_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
