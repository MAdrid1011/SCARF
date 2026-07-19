#!/usr/bin/env python3
"""Render preserved paper values as explicitly non-claiming reference artifacts.

The utility is deliberately isolated from simulator, calibration, and claim-report
paths. It only reads immutable paper-reference inputs and never produces evidence.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_ROOT = ROOT / "micro59-submit"
REFERENCE_ONLY_MARKER = "PAPER_REFERENCE_ONLY"
REFERENCE_MANIFEST_NAME = "reference_manifest.json"
PROVENANCE_HEADERS = (
    "artifact_class",
    "source_id",
    "source_path",
    "source_sha256",
)

MODEL_KEYS = {
    "tran": "transplat",
    "transplat": "transplat",
    "mv": "mvsplat",
    "mvsplat": "mvsplat",
    "depth": "depthsplat",
    "depthsplat": "depthsplat",
}
TABLE4_COMPONENTS = (
    ("mvu", "MVU"),
    ("mvu_mmcu", "MMCU"),
    ("mvu_vector_alu", "VectorALU"),
    ("mvu_bilinear_unit", "BilinearUnit"),
    ("mvu_norm_unit", "NormUnit"),
    ("mvu_activation_unit", "ActivationUnit"),
    ("ggu_array", "GGU Array (x32 PEs)"),
    ("ggu_position_calc", "PositionCalc"),
    ("ggu_cov_builder", "CovBuilder"),
    ("ggu_sh_op_generator", "SH_OPGenerator"),
    ("fsdr_subsystem", "FSDR Subsystem"),
    ("fsdr_lsh_hash_unit", "LSHHashUnit"),
    ("fsdr_cam_array", "CAM Array (32-entry)"),
    ("fsdr_controller", "FSDR Controller"),
    ("on_chip_buffers", "On-chip Buffers"),
    ("on_chip_buffers_weight_buffer", "Weight Buffer (128 KB)"),
    ("on_chip_buffers_feature_buffer", "Feature Buffer (256 KB)"),
    ("on_chip_buffers_tile_buffer", "Tile Buffer (64 KB)"),
    ("control_and_clock", "Control & Clock"),
    ("control_interconnect", "Control + Interconnect"),
    ("pll_clock_tree", "PLL + Clock tree"),
    ("io_phy", "I/O & PHY"),
    ("io_lpddr4x_phy", "I/O + LPDDR4X PHY"),
    ("routing_filler", "Routing / filler"),
    ("total_die", "Total die"),
)
FIGURE13_SPECS = (
    ("figure13_cache_top_tran", "TranSplat", "top"),
    ("figure13_cache_top_mv", "MVSplat", "top"),
    ("figure13_cache_top_depth", "DepthSplat", "top"),
    ("figure13_cache_bottom_tran", "TranSplat", "bottom"),
    ("figure13_cache_bottom_mv", "MVSplat", "bottom"),
    ("figure13_cache_bottom_depth", "DepthSplat", "bottom"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_label(source: Path) -> str:
    """Return a portable label without exposing an author-local path."""
    try:
        return source.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "external-input"


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def source_paths(expected_path: Path, paper_root: Path) -> dict[str, Path]:
    figures = paper_root / "figures"
    paths = {
        "expected_results": expected_path,
        "paper_evaluation_tex": paper_root / "Sections/section6.tex",
        "figure8_speedup": figures / "performance_dataflow_speedup.csv",
        "figure8_latency": figures / "performance_breakdown_compose.csv",
        "figure9_energy": figures / "efficiency_energy_process.csv",
        "figure9_area": figures / "efficiency_area_process.csv",
        "figure12_utilization": figures / "mmcu_stage_utilization.csv",
        "figure10_rendered": figures / "StressCase_latency_error_maps_preview.pdf",
        "figure10_quality_metadata": figures / "StressCase_quality_distribution_metadata.json",
        "figure10_error_metadata": figures / "WorstCase_error_maps_metadata.json",
        "figure11_rendered": figures / "Ablation.drawio.pdf",
        "figure14_rendered": figures / "SensitiveFSDRThreshold.drawio.pdf",
        "figure15_rendered": figures / "SensitiveSAESThreshold.drawio.pdf",
        "figure16_rendered": figures / "SensitiveTile.drawio.pdf",
    }
    for source_id, _, _ in FIGURE13_SPECS:
        suffix = source_id.removeprefix("figure13_cache_")
        paths[source_id] = figures / f"fsdr_cache_{suffix}.csv"
    missing = [source_id for source_id, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("missing paper-reference sources: " + ", ".join(missing))
    return paths


def source_records(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        source_id: {
            "id": source_id,
            "path": source_label(path),
            "sha256": sha256_file(path),
        }
        for source_id, path in paths.items()
    }


def provenance_rows(
    rows: list[tuple[str, list[Any]]], sources: Mapping[str, Mapping[str, str]]
) -> list[list[Any]]:
    rendered = []
    for source_id, row in rows:
        source = sources[source_id]
        rendered.append(
            [
                REFERENCE_ONLY_MARKER,
                source_id,
                source["path"],
                source["sha256"],
                *row,
            ]
        )
    return rendered


def write_reference_csv(
    output_dir: Path,
    filename: str,
    headers: list[str],
    rows: list[tuple[str, list[Any]]],
    sources: Mapping[str, Mapping[str, str]],
) -> tuple[Path, tuple[str, ...]]:
    path = output_dir / filename
    write_csv(path, [*PROVENANCE_HEADERS, *headers], provenance_rows(rows, sources))
    return path, tuple(sorted({source_id for source_id, _ in rows}))


def write_reference_json(
    output_dir: Path,
    filename: str,
    values: Any,
    source_ids: tuple[str, ...],
    sources: Mapping[str, Mapping[str, str]],
) -> tuple[Path, tuple[str, ...]]:
    path = output_dir / filename
    payload = {
        "schema_version": "1.0",
        "artifact_class": REFERENCE_ONLY_MARKER,
        "source_ids": list(source_ids),
        "sources": [sources[source_id] for source_id in source_ids],
        "values": values,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path, source_ids


def canonical_pair(model: str, dataset: str) -> str:
    try:
        model_key = MODEL_KEYS[model.strip().lower()]
    except KeyError as error:
        raise ValueError(f"unknown paper model label: {model}") from error
    dataset_key = dataset.strip().lower()
    if dataset_key not in {"re10k", "acid", "dl3dv"}:
        raise ValueError(f"unknown paper dataset label: {dataset}")
    return f"{model_key}/{dataset_key}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def geometric_mean(values: list[str]) -> Decimal:
    decimals = [Decimal(value) for value in values]
    if not decimals or any(value <= 0 for value in decimals):
        raise ValueError("geometric mean requires positive values")
    with localcontext() as context:
        context.prec = 50
        return (sum(value.ln() for value in decimals) / Decimal(len(decimals))).exp()


def parse_number(value: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if match is None:
        raise ValueError(f"missing numeric value: {value}")
    return float(match.group())


def parse_percent(value: str) -> float:
    if "%" not in value:
        raise ValueError(f"missing percent sign: {value}")
    return parse_number(value) / 100.0


def latex_table_block(tex: str, label: str) -> str:
    marker = f"\\label{{{label}}}"
    label_index = tex.find(marker)
    if label_index < 0:
        raise ValueError(f"missing LaTex table label: {label}")
    start = tex.rfind("\\begin{table}", 0, label_index)
    end = tex.find("\\end{table}", label_index)
    if start < 0 or end < 0:
        raise ValueError(f"malformed LaTex table: {label}")
    return tex[start:end]


def clean_latex_cell(value: str) -> str:
    value = value.strip()
    while True:
        cleaned = re.sub(r"\\(?:blue|textbf|textit)\{([^{}]*)\}", r"\1", value)
        if cleaned == value:
            break
        value = cleaned
    value = value.replace(r"\%", "%").replace(r"\,", " ")
    value = value.replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\quad", "").replace(r"\enspace", " ")
    value = value.replace(r"\_", "_")
    value = value.replace("$", "").replace("~", " ").replace("{,}", ",")
    return re.sub(r"\s+", " ", value).strip()


def parse_latex_table(tex: str, label: str, metric_count: int) -> list[list[str]]:
    rows = []
    current_model = ""
    for raw_line in latex_table_block(tex, label).splitlines():
        line = raw_line.strip()
        if "&" not in line or not line.endswith(r"\\"):
            continue
        cells = [cell.strip() for cell in line[:-2].split("&")]
        model_match = re.search(r"\\multirow\{\d+\}\{\*\}\{([^{}]+)\}", cells[0])
        if model_match is not None:
            current_model = clean_latex_cell(model_match.group(1))
        dataset = clean_latex_cell(cells[1]) if len(cells) > 1 else ""
        if not current_model or dataset.lower() not in {"re10k", "acid", "dl3dv"}:
            continue
        values = [clean_latex_cell(cell) for cell in cells[2:]]
        if len(values) != metric_count:
            raise ValueError(f"unexpected column count in {label}: {line}")
        rows.append([canonical_pair(current_model, dataset), current_model, dataset, *values])
    if len(rows) != 9:
        raise ValueError(f"expected nine rows in {label}, found {len(rows)}")
    return rows


def table1_rows(expected: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for pair, values in expected["table1"].items():
        rows.append(
            [
                pair,
                *[f"{value:.3f}" for value in values["baseline"]],
                *[f"{value:.3f}" for value in values["scarf"]],
            ]
        )
    return rows


def table2_rows(tex: str, expected: Mapping[str, Any]) -> list[list[str]]:
    rows = parse_latex_table(tex, "tab:fsdr_detail", 4)
    for pair, _, _, guided, top1, _, _ in rows:
        reference = expected["mechanisms"][pair]
        if not math.isclose(parse_percent(guided), reference["guided_rate"], abs_tol=1e-9):
            raise ValueError(f"Table 2 guided-rate mismatch for {pair}")
        if not math.isclose(parse_percent(top1), reference["top1_coverage"], abs_tol=1e-9):
            raise ValueError(f"Table 2 Top-1 coverage mismatch for {pair}")
    return rows


def table3_rows(tex: str, expected: Mapping[str, Any]) -> list[list[str]]:
    rows = parse_latex_table(tex, "tab:saes_detail", 5)
    for pair, _, _, level0, level1, agreement, saved, _ in rows:
        reference = expected["mechanisms"][pair]
        values = {
            "level0_rate": parse_percent(level0),
            "level1_rate": parse_percent(level1),
            "low_variance_agreement": parse_percent(agreement),
            "gaussians_saved": parse_percent(saved),
        }
        if any(
            not math.isclose(value, reference[key], abs_tol=1e-9)
            for key, value in values.items()
        ):
            raise ValueError(f"Table 3 shared metric mismatch for {pair}")
    return rows


def figure8_speedup_rows(path: Path, expected: Mapping[str, Any]) -> tuple[list[list[str]], float, float]:
    rows = []
    for row in read_csv_rows(path):
        pair = canonical_pair(row["model"], row["dataset"])
        rows.append(
            [
                pair,
                row["dataset"],
                row["model"],
                row["Orin NX"],
                row["SCARF Dataflow"],
                row["SCARF ASIC"],
            ]
        )
    if set(row[0] for row in rows) != set(expected["table1"]):
        raise ValueError("Figure 8 speedup rows do not cover all nine paper pairs")
    asic_gm = geometric_mean([row[-1] for row in rows])
    dataflow_gm = geometric_mean([row[-2] for row in rows])
    if not math.isclose(
        float(asic_gm), expected["figure8"]["geometric_mean_speedup"], abs_tol=0.005
    ):
        raise ValueError("Figure 8 ASIC geometric mean does not match paper reference")
    return rows, asic_gm, dataflow_gm


def figure8_latency_rows(path: Path, expected: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for row in read_csv_rows(path):
        pair = canonical_pair(row["model"], row["dataset"])
        rows.append(
            [pair, row["dataset"], row["model"], row["platform"], row["S1"], row["S2"], row["S3"], row["S4"]]
        )
    if len(rows) != 18 or {row[0] for row in rows} != set(expected["table1"]):
        raise ValueError("Figure 8 latency rows are incomplete")
    return rows


def figure9_rows(
    expected: Mapping[str, Any], energy_path: Path, area_path: Path
) -> tuple[list[list[str]], list[list[str]]]:
    energy = {
        canonical_pair(row["Model"], row["Dataset"]): row
        for row in read_csv_rows(energy_path)
    }
    area = {
        canonical_pair(row["Model"], row["Dataset"]): row
        for row in read_csv_rows(area_path)
    }
    energy_rows = []
    area_rows = []
    for pair, values in expected["figure9"]["paper_comparison_target"].items():
        energy_row = energy.get(pair)
        area_row = area.get(pair)
        if energy_row is None or area_row is None:
            raise ValueError(f"Figure 9 source CSV is missing {pair}")
        for column, key in (("Orin NX", "orin_nx"), ("SCARF 28 nm", "tsmc28"), ("SCARF 8 nm-eq.", "nm8_equivalent")):
            if not math.isclose(float(energy_row[column]), values["energy_efficiency"][key]):
                raise ValueError(f"Figure 9 energy mismatch for {pair}")
            if not math.isclose(float(area_row[column]), values["throughput_per_area"][key]):
                raise ValueError(f"Figure 9 area mismatch for {pair}")
        energy_rows.append([pair, energy_row["Orin NX"], energy_row["SCARF 28 nm"], energy_row["SCARF 8 nm-eq."]])
        area_rows.append([pair, area_row["Orin NX"], area_row["SCARF 28 nm"], area_row["SCARF 8 nm-eq."]])
    return energy_rows, area_rows


def figure12_rows(path: Path, expected: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for row in read_csv_rows(path):
        pair = canonical_pair(row["Model"], row["Dataset"])
        values = expected["figure12"]["utilization"].get(pair)
        if values is None:
            raise ValueError(f"Figure 12 expected reference is missing {pair}")
        for column, key in (("S1_MMCU_Util", "s1"), ("S2_MMCU_Util", "s2"), ("S3_MMCU_Util", "s3")):
            if not math.isclose(float(row[column]) / 100.0, values[key], abs_tol=1e-9):
                raise ValueError(f"Figure 12 source mismatch for {pair}")
        rows.append([pair, row["Model"], row["Dataset"], row["S1_MMCU_Util"], row["S2_MMCU_Util"], row["S3_MMCU_Util"]])
    if len(rows) != 9:
        raise ValueError("Figure 12 source CSV is incomplete")
    return rows


def figure13_rows(
    paths: Mapping[str, Path], expected: Mapping[str, Any]
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    data_rows = []
    annotation_rows = []
    expected_grid = sorted(expected["sensitivity_grids"]["fsdr_cache_size"]["values"])
    observed: dict[tuple[str, str, str], list[int]] = {}
    for source_id, model, panel in FIGURE13_SPECS:
        for row in read_csv_rows(paths[source_id]):
            group = row["group"]
            if group.startswith("__"):
                annotation_rows.append(
                    (source_id, [model, panel, group, json.dumps(row, ensure_ascii=False, sort_keys=True)])
                )
                continue
            pair = canonical_pair(model, group)
            cache_entries = int(row["x"])
            observed.setdefault((model, panel, group), []).append(cache_entries)
            if panel == "top":
                data_rows.append(
                    (
                        source_id,
                        [model, panel, pair, group, str(cache_entries), row["Throughput (inf/s)"], row["Norm. access overhead"], "", "", ""],
                    )
                )
            else:
                data_rows.append(
                    (
                        source_id,
                        [model, panel, pair, group, str(cache_entries), "", "", row["Tile WS coverage (%)"], row["Cap. miss"], row["Sem. miss"]],
                    )
                )
    if any(sorted(values) != expected_grid for values in observed.values()):
        raise ValueError("Figure 13 cache grid does not match the paper reference")
    if len(data_rows) != 90:
        raise ValueError(f"expected 90 Figure 13 data rows, found {len(data_rows)}")
    return data_rows, annotation_rows


def flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten(item, child))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(flatten(item, f"{prefix}[{index}]"))
        return rows
    return [(prefix, value)]


def value_text(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def sensitivity_rows(expected: Mapping[str, Any]) -> list[list[str]]:
    values = {
        "sensitivity_grids": expected["sensitivity_grids"],
        "sensitivity_expected_runs": expected["sensitivity_expected_runs"],
        "sensitivity_claims": expected["sensitivity_claims"],
    }
    return [[path, value_text(value)] for path, value in flatten(values)]


def table4_rows(expected: Mapping[str, Any]) -> list[list[str]]:
    hierarchy = expected["hardware_comparison_target"]["hierarchy"]
    rows = []
    for component, label in TABLE4_COMPONENTS:
        values = hierarchy[component]
        rows.append(
            [
                component,
                label,
                value_text(values.get("area_mm2")),
                value_text(values.get("dynamic_power_w")),
                value_text(values.get("static_power_w")),
                value_text(values.get("total_power_w")),
            ]
        )
    return rows


def hardware_metadata_rows(expected: Mapping[str, Any]) -> list[list[str]]:
    hardware = expected["hardware_comparison_target"]
    return [
        [key, value_text(value)]
        for key, value in hardware.items()
        if key != "hierarchy"
    ]


def table4_source_rows(tex: str) -> list[list[str]]:
    component_by_label = {
        "MMCU": "mvu_mmcu",
        "VectorALU": "mvu_vector_alu",
        "BilinearUnit": "mvu_bilinear_unit",
        "NormUnit": "mvu_norm_unit",
        "ActivationUnit": "mvu_activation_unit",
        "PositionCalc": "ggu_position_calc",
        "CovBuilder": "ggu_cov_builder",
        "SH_OPGenerator": "ggu_sh_op_generator",
        "LSHHashUnit": "fsdr_lsh_hash_unit",
        "CAM Array (32-entry)": "fsdr_cam_array",
        "FSDR Controller": "fsdr_controller",
        "Weight Buffer (128 KB)": "on_chip_buffers_weight_buffer",
        "Feature Buffer (256 KB)": "on_chip_buffers_feature_buffer",
        "Tile Buffer (64 KB)": "on_chip_buffers_tile_buffer",
        "Control + Interconnect": "control_interconnect",
        "PLL + Clock tree": "pll_clock_tree",
        "I/O + LPDDR4X PHY": "io_lpddr4x_phy",
        "Routing / filler": "routing_filler",
        "Total die": "total_die",
    }
    group_markers = (
        ("MVU\\enspace", "mvu"),
        ("GGU Array", "ggu_array"),
        ("FSDR Subsystem", "fsdr_subsystem"),
        ("On-chip Buffers", "on_chip_buffers"),
        ("Control \\& Clock", "control_and_clock"),
        ("I/O \\& PHY", "io_phy"),
    )
    labels = dict(TABLE4_COMPONENTS)
    rows = []
    for raw_line in latex_table_block(tex, "tab:area_power").splitlines():
        line = raw_line.strip()
        if not line.endswith(r"\\"):
            continue
        group = next(
            (component for marker, component in group_markers if marker in line), None
        )
        if group is not None:
            text_start = line.find(r"\textit{") + len(r"\textit{")
            paper_row = line[text_start:-2].rsplit("}}", 1)[0]
            rows.append([group, labels[group], clean_latex_cell(paper_row)])
            continue
        if "&" not in line:
            continue
        cells = [clean_latex_cell(cell) for cell in line[:-2].split("&")]
        label = cells[0]
        component = next(
            (
                key
                for source_label, key in component_by_label.items()
                if label.startswith(source_label)
            ),
            None,
        )
        if component is not None:
            rows.append([component, labels[component], " | ".join(cells)])
    expected_components = {component for component, _ in TABLE4_COMPONENTS}
    if len(rows) != 25 or {row[0] for row in rows} != expected_components:
        raise ValueError("Table 4 LaTex rows are incomplete")
    return rows


def unstructured_notice(sources: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    return {
        "figure10": {
            "status": "SOURCE_ONLY_NOT_TABULATED",
            "source_ids": [
                "figure10_rendered",
                "figure10_quality_metadata",
                "figure10_error_metadata",
            ],
            "reason": (
                "The visual error maps cannot be losslessly tabulated. The retained "
                "metadata includes plot-versus-actual mismatches, so this preview "
                "keeps only source identities and does not normalize or claim those values."
            ),
        },
        "figure11": {
            "status": "PDF_ONLY_POINT_SERIES_UNAVAILABLE",
            "source_ids": ["figure11_rendered"],
            "reason": (
                "No machine-readable per-pair ablation series is available. The "
                "separate reference CSV contains only the three expected geometric means."
            ),
        },
        "figure14": {
            "status": "PDF_ONLY_POINT_SERIES_UNAVAILABLE",
            "source_ids": ["figure14_rendered"],
            "reason": "No source CSV/JSON point series exists; graph geometry is not reverse-engineered.",
        },
        "figure15": {
            "status": "PDF_ONLY_POINT_SERIES_UNAVAILABLE",
            "source_ids": ["figure15_rendered"],
            "reason": (
                "No source CSV/JSON point series exists. The rendered lower-panel axis "
                "also conflicts with the documented depth-threshold grid, so no values are inferred."
            ),
        },
        "figure16": {
            "status": "PDF_ONLY_POINT_SERIES_UNAVAILABLE",
            "source_ids": ["figure16_rendered"],
            "reason": "No source CSV/JSON point series exists; graph geometry is not reverse-engineered.",
        },
        "source_assets": {
            source_id: sources[source_id]
            for source_id in (
                "figure10_rendered",
                "figure10_quality_metadata",
                "figure10_error_metadata",
                "figure11_rendered",
                "figure14_rendered",
                "figure15_rendered",
                "figure16_rendered",
            )
        },
    }


def render(
    expected: dict[str, Any],
    source: Path,
    output_dir: Path,
    *,
    paper_root: Path = DEFAULT_PAPER_ROOT,
) -> None:
    paths = source_paths(source, paper_root)
    sources = source_records(paths)
    tex = paths["paper_evaluation_tex"].read_text(encoding="utf-8")
    table1 = table1_rows(expected)
    table2 = table2_rows(tex, expected)
    table3 = table3_rows(tex, expected)
    figure8_speedup, asic_gm, dataflow_gm = figure8_speedup_rows(
        paths["figure8_speedup"], expected
    )
    figure8_latency = figure8_latency_rows(paths["figure8_latency"], expected)
    figure9_energy, figure9_area = figure9_rows(
        expected, paths["figure9_energy"], paths["figure9_area"]
    )
    figure12 = figure12_rows(paths["figure12_utilization"], expected)
    figure13, figure13_annotations = figure13_rows(paths, expected)
    sensitivity = sensitivity_rows(expected)
    table4 = table4_rows(expected)
    table4_source = table4_source_rows(tex)
    hardware_metadata = hardware_metadata_rows(expected)
    unstructured = unstructured_notice(sources)

    artifacts: list[tuple[Path, tuple[str, ...]]] = []
    artifacts.append(
        write_reference_json(
            output_dir,
            "expected_results_reference.json",
            expected,
            ("expected_results",),
            sources,
        )
    )
    csv_specs = (
        ("table1_reference.csv", ["pair", "Baseline PSNR", "Baseline SSIM", "Baseline LPIPS", "SCARF PSNR", "SCARF SSIM", "SCARF LPIPS"], [("expected_results", row) for row in table1]),
        ("table2_fsdr_reference.csv", ["pair", "paper_model", "dataset", "Guided Rate", "Top-1 Coverage", "Depth Evals Saved in S2 (M, percent)", "Feature Buffer Reduction (MB, percent)"], [("paper_evaluation_tex", row) for row in table2]),
        ("table3_saes_reference.csv", ["pair", "paper_model", "dataset", "L0 Rate", "L1 Rate", "Low-variance Agreement", "Gaussians Saved", "S2 Evals Saved (M)"], [("paper_evaluation_tex", row) for row in table3]),
        ("figure8_speedup_reference.csv", ["pair", "dataset", "paper_model", "Orin NX", "SCARF Dataflow on Orin NX", "SCARF ASIC"], [("figure8_speedup", row) for row in figure8_speedup]),
        ("figure8_latency_reference.csv", ["pair", "dataset", "paper_model", "platform", "S1", "S2", "S3", "S4"], [("figure8_latency", row) for row in figure8_latency]),
        ("figure8_summary_reference.csv", ["metric", "value"], [("figure8_speedup", ["SCARF Dataflow on Orin NX geometric mean", f"{dataflow_gm:.15f}"]), ("figure8_speedup", ["SCARF ASIC geometric mean from source rows", f"{asic_gm:.15f}"]), ("expected_results", ["SCARF ASIC geometric mean paper reference", f"{expected['figure8']['geometric_mean_speedup']:.2f}"])]),
        ("figure9_energy_efficiency_reference.csv", ["pair", "Orin NX", "SCARF 28 nm", "SCARF 8 nm-eq."], [("figure9_energy", row) for row in figure9_energy]),
        ("figure9_area_throughput_reference.csv", ["pair", "Orin NX", "SCARF 28 nm", "SCARF 8 nm-eq."], [("figure9_area", row) for row in figure9_area]),
        ("figure11_ablation_reference.csv", ["configuration", "geometric_mean_speedup"], [("expected_results", ["FSDR-only", f"{expected['ablation']['fsdr']:.2f}"]), ("expected_results", ["SAES-only", f"{expected['ablation']['saes']:.2f}"]), ("expected_results", ["Combined", f"{expected['ablation']['combined']:.2f}"])]),
        ("figure12_mmcu_utilization_reference.csv", ["pair", "paper_model", "dataset", "S1 utilization (%)", "S2 utilization (%)", "S3 utilization (%)"], [("figure12_utilization", row) for row in figure12]),
        ("figure13_fsdr_cache_reference.csv", ["paper_model", "panel", "pair", "dataset", "cache_entries", "throughput_inf_s", "normalized_access_overhead", "tile_working_set_coverage_pct", "capacity_miss_pct", "semantic_miss_pct"], figure13),
        ("figure13_source_annotations_reference.csv", ["paper_model", "panel", "directive", "source_row"], figure13_annotations),
        ("figures13_16_sensitivity_reference.csv", ["expected_path", "value"], [("expected_results", row) for row in sensitivity]),
        ("table4_hierarchy_reference.csv", ["component_key", "component", "area_mm2", "dynamic_power_w", "static_power_w", "total_power_w"], [("expected_results", row) for row in table4]),
        ("table4_paper_cells_reference.csv", ["component_key", "component", "paper_table_row"], [("paper_evaluation_tex", row) for row in table4_source]),
        ("table4_metadata_reference.csv", ["field", "value"], [("expected_results", row) for row in hardware_metadata]),
    )
    for filename, headers, rows in csv_specs:
        artifacts.append(write_reference_csv(output_dir, filename, headers, rows, sources))
    artifacts.append(
        write_reference_json(
            output_dir,
            "unstructured_figure_data_reference.json",
            unstructured,
            (
                "figure10_rendered",
                "figure10_quality_metadata",
                "figure10_error_metadata",
                "figure11_rendered",
                "figure14_rendered",
                "figure15_rendered",
                "figure16_rendered",
            ),
            sources,
        )
    )

    source_rows = [[record["id"], record["path"], record["sha256"]] for record in sources.values()]
    unstructured_rows = [
        [figure.replace("figure", "Figure "), value["status"], ", ".join(value["source_ids"]), value["reason"]]
        for figure, value in unstructured.items()
        if figure.startswith("figure")
    ]
    lines = [
        "---",
        f"artifact_class: {REFERENCE_ONLY_MARKER}",
        f"source_path: {sources['expected_results']['path']}",
        f"source_sha256: {sources['expected_results']['sha256']}",
        f"source_manifest: {REFERENCE_MANIFEST_NAME}",
        "---",
        "",
        "# SCARF 论文参考数据展示",
        "",
        f"> **{REFERENCE_ONLY_MARKER}**：本目录仅展示保存的论文参考数据，",
        "> 不是模型执行、ASIC 仿真、Orin 测量或 AE 复现证据。",
        "",
        "## Provenance",
        "",
        *markdown_table(["ID", "相对来源", "SHA256"], source_rows),
        "",
        "## Table 1",
        "",
        *markdown_table(["pair", "Base PSNR", "Base SSIM", "Base LPIPS", "SCARF PSNR", "SCARF SSIM", "SCARF LPIPS"], table1),
        "",
        "## Figure 8",
        "",
        *markdown_table(["pair", "数据集", "模型", "Orin NX", "Dataflow", "ASIC"], figure8_speedup),
        "",
        *markdown_table(["pair", "数据集", "模型", "平台", "S1", "S2", "S3", "S4"], figure8_latency),
        "",
        "## Figure 9",
        "",
        *markdown_table(["pair", "Orin NX", "SCARF 28 nm", "SCARF 8 nm-eq."], figure9_energy),
        "",
        *markdown_table(["pair", "Orin NX", "SCARF 28 nm", "SCARF 8 nm-eq."], figure9_area),
        "",
        "## Tables 2-3",
        "",
        *markdown_table(["pair", "模型", "数据集", "Guided", "Top-1", "S2 Evals Saved", "Feature Buffer Red."], table2),
        "",
        *markdown_table(["pair", "模型", "数据集", "L0", "L1", "Low Var", "Gaussians Saved", "S2 Evals Saved"], table3),
        "",
        "## Figure 11 and Figure 12",
        "",
        *markdown_table(["configuration", "geometric mean"], [["FSDR-only", f"{expected['ablation']['fsdr']:.2f}"], ["SAES-only", f"{expected['ablation']['saes']:.2f}"], ["Combined", f"{expected['ablation']['combined']:.2f}"]]),
        "",
        *markdown_table(["pair", "模型", "数据集", "S1 (%)", "S2 (%)", "S3 (%)"], figure12),
        "",
        "## Figures 13-16",
        "",
        "- Figure 13 的 90 个数据点和原始 panel annotations 位于 `figure13_*_reference.csv`。",
        "- Figures 14-16 的所有可结构化 contract 值位于 `figures13_16_sensitivity_reference.csv`。",
        "",
        "## Table 4",
        "",
        *markdown_table(["key", "组件", "面积 mm2", "动态 W", "静态 W", "总功耗 W"], table4),
        "",
        "- 原稿中带百分比的完整 cell 文本保留在 `table4_paper_cells_reference.csv`。",
        "",
        "## Not Tabulated",
        "",
        *markdown_table(["Figure", "状态", "source IDs", "原因"], unstructured_rows),
        "",
        "`expected_results_reference.json` 保留完整结构化 validator-only 输入副本；所有运行结果必须单独生成，且不得由本目录反向设置路由、周期或质量输出。",
    ]
    markdown_path = output_dir / "paper_reference.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts.append((markdown_path, tuple(sources)))

    manifest = {
        "schema_version": "2.0",
        "artifact_class": REFERENCE_ONLY_MARKER,
        "source": sources["expected_results"],
        "sources": list(sources.values()),
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "source_ids": list(source_ids),
            }
            for path, source_ids in artifacts
        ],
        "unstructured_figure_data": unstructured,
    }
    (output_dir / REFERENCE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "artifact" / "expected_results.json",
        help="Paper-reference contract to display.",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=DEFAULT_PAPER_ROOT,
        help="Original paper source tree used only for reference rendering.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.expected.resolve()
    paper_root = args.paper_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        parser.error(f"missing paper-reference source: {source}")
    if not paper_root.is_dir():
        parser.error(f"missing original paper source tree: {paper_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = json.loads(source.read_text(encoding="utf-8"))
    render(expected, source, output_dir, paper_root=paper_root)
    print(output_dir / "paper_reference.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
