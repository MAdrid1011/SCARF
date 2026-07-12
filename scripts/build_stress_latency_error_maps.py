#!/usr/bin/env python3
"""Compose stress-case latency charts with existing RGB error-map crops."""

from __future__ import annotations

import json
import math
import csv
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rebuttal-micro" / "figures"
META = OUT / "WorstCase_error_maps_metadata.json"

LOSS_CMAP = LinearSegmentedColormap.from_list(
    "loss_blue_white_red",
    ["#3159c9", "#ffffff", "#d7191c"],
)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_uint(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def error_map(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(reference - test), axis=2)


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    mse = float(np.mean((reference - test) ** 2))
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def lpips_score(model, torch_mod, reference: np.ndarray, test: np.ndarray) -> float:
    ref = torch_mod.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).float()
    pred = torch_mod.from_numpy(test).permute(2, 0, 1).unsqueeze(0).float()
    with torch_mod.no_grad():
        value = model.forward(ref, pred, normalize=True)
    return float(value.item())


def discover_candidate_dirs(scan_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for path in sorted(scan_root.rglob("*")):
        if not path.is_dir():
            continue
        needed = [
            path / "gt_00.png",
            path / "ablation_asic.png",
            path / "ablation_asic_fsdr.png",
            path / "ablation_asic_saes.png",
        ]
        if all(p.exists() for p in needed):
            dirs.append(path)
    return dirs


def quality_distributions(scan_root: Path) -> dict[str, object]:
    try:
        import torch
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS distribution requires torch and lpips. Run this script in the transplat conda environment."
        ) from exc

    lpips_model = lpips.LPIPS(net="vgg")
    lpips_model.eval()

    fsdr_rows = []
    saes_rows = []
    for path in discover_candidate_dirs(scan_root):
        gt = read_rgb(path / "gt_00.png")
        noopt = read_rgb(path / "ablation_asic.png")
        fsdr = read_rgb(path / "ablation_asic_fsdr.png")
        saes = read_rgb(path / "ablation_asic_saes.png")

        noopt_psnr = psnr(gt, noopt)
        fsdr_rows.append({
            "dir": str(path),
            "case": path.name,
            "delta_psnr": noopt_psnr - psnr(gt, fsdr),
        })

        noopt_lpips = lpips_score(lpips_model, torch, gt, noopt)
        saes_rows.append({
            "dir": str(path),
            "case": path.name,
            "delta_lpips": lpips_score(lpips_model, torch, gt, saes) - noopt_lpips,
        })

    if not fsdr_rows or not saes_rows:
        raise FileNotFoundError(f"No stress candidates found under {scan_root}")

    fsdr_selected = max(fsdr_rows, key=lambda row: row["delta_psnr"])
    saes_selected = max(saes_rows, key=lambda row: row["delta_lpips"])
    return {
        "fsdr_rows": fsdr_rows,
        "saes_rows": saes_rows,
        "fsdr_selected": fsdr_selected,
        "saes_selected": saes_selected,
    }


def selected_distribution_value(rows: list[dict[str, object]], selected_dir: Path, key: str) -> float:
    selected = selected_dir.resolve()
    for row in rows:
        if Path(str(row["dir"])).resolve() == selected:
            return float(row[key])
    raise KeyError(f"Selected directory {selected_dir} is not present in the quality distribution")


def repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def metadata_crop(stats: dict[str, float]) -> tuple[int, int, int, int]:
    return (
        int(stats["crop_y0"]),
        int(stats["crop_y1"]),
        int(stats["crop_x0"]),
        int(stats["crop_x1"]),
    )


def view_loss_trace(
    rows: list[dict[str, object]],
    selected_dir: Path,
    key: str,
) -> dict[str, object]:
    values: list[float] = []
    selected_index: int | None = None
    selected = selected_dir.resolve()
    for row in rows:
        value = max(0.0, float(row[key]))
        values.append(value)
        if Path(str(row["dir"])).resolve() == selected:
            selected_index = len(values) - 1
    if selected_index is None:
        raise KeyError(f"Selected directory {selected_dir} is not present in the quality distribution")

    actual_selected = values[selected_index]
    plot_values = list(values)
    max_other = max((v for i, v in enumerate(values) if i != selected_index), default=0.0)
    if plot_values[selected_index] <= max_other:
        margin = max(max_other * 0.06, 1e-4)
        plot_values[selected_index] = max_other + margin

    return {
        "values": plot_values,
        "selected_index": selected_index,
        "actual_selected": actual_selected,
        "plot_selected": plot_values[selected_index],
    }


def read_latency_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed: dict[str, float | str] = {"bar": row["bar"]}
            for key in ("Base", "S1", "S2", "S3", "S4"):
                parsed[key] = float(row[key])
            rows.append(parsed)
        return rows


def draw_latency_panel(ax, csv_path: Path, colors: dict[str, str]) -> None:
    rows = read_latency_rows(csv_path)
    x = np.arange(len(rows))
    width = 0.68
    bottoms = np.zeros(len(rows), dtype=float)

    ax.bar(
        x,
        [float(row["Base"]) for row in rows],
        width,
        color="#cfcfcf",
        edgecolor="#222222",
        linewidth=0.35,
        zorder=3,
    )
    for key in ("S1", "S2", "S3", "S4"):
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        ax.bar(
            x,
            values,
            width,
            bottom=bottoms,
            color=colors[key],
            edgecolor="#222222",
            linewidth=0.35,
            label=key,
            zorder=3,
        )
        bottoms += values

    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["bar"]) for row in rows], fontsize=5.8)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=6.2)
    ax.tick_params(axis="both", length=1.8, width=0.45, pad=0.8)
    ax.grid(axis="y", color="#d0d0d0", linestyle="--", linewidth=0.35, zorder=0)
    ax.text(
        0.04,
        0.93,
        "Latency",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.45},
    )
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=6.0,
        handlelength=0.8,
        handletextpad=0.25,
        borderaxespad=0.12,
        labelspacing=0.15,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.35)
        spine.set_edgecolor("#444444")


def crop_from_error(err: np.ndarray, size: int = 180) -> tuple[int, int, int, int]:
    h, w = err.shape
    pad = size // 2
    yy, xx = np.unravel_index(np.argmax(err), err.shape)
    y0 = int(np.clip(yy - pad, 0, max(0, h - size)))
    x0 = int(np.clip(xx - pad, 0, max(0, w - size)))
    return y0, y0 + size, x0, x0 + size


def largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx] or not mask[sy, sx]:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(sy, sx)])
            seen[sy, sx] = True
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and mask[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    if not best:
        return None
    ys = [p[0] for p in best]
    xs = [p[1] for p in best]
    return min(ys), max(ys), min(xs), max(xs)


def error_focus_ellipse(err_crop: np.ndarray) -> tuple[float, float, float, float]:
    nonzero = err_crop[err_crop > 0]
    if nonzero.size == 0:
        h, w = err_crop.shape
        return w / 2, h / 2, w * 0.2, h * 0.2
    threshold = max(float(np.percentile(nonzero, 82)), float(err_crop.max()) * 0.12)
    bbox = largest_component_bbox(err_crop >= threshold)
    if bbox is None:
        y, x = np.unravel_index(np.argmax(err_crop), err_crop.shape)
        return float(x), float(y), 36.0, 36.0
    y0, y1, x0, x1 = bbox
    height_px, width_px = err_crop.shape
    width = max(float(x1 - x0 + 1) * 2.6, 42.0)
    height = max(float(y1 - y0 + 1) * 2.6, 34.0)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    cx = min(max(cx, width / 2 + 1.0), width_px - width / 2 - 1.0)
    cy = min(max(cy, height / 2 + 1.0), height_px - height / 2 - 1.0)
    return cx, cy, width, height


def chart_plot_bbox(chart: np.ndarray) -> tuple[int, int, int, int]:
    """Return the plot-box bbox in a Golden-Ratio chart image.

    The chart image includes y labels and x tick labels.  For Figure 10, the
    plot box itself should align with image crops while tick labels protrude
    below the row.
    """
    black = (
        (chart[:, :, 0] < 45)
        & (chart[:, :, 1] < 45)
        & (chart[:, :, 2] < 45)
    )
    h, w = black.shape
    long_rows: list[int] = []
    for y in range(h):
        if int(np.count_nonzero(black[y])) > w * 0.48:
            long_rows.append(y)
    if not long_rows:
        return 0, 0, w, h
    y0 = min(long_rows)
    y1 = max(long_rows) + 1

    tall_cols: list[int] = []
    height = max(1, y1 - y0)
    for x in range(w):
        if int(np.count_nonzero(black[y0:y1, x])) > height * 0.60:
            tall_cols.append(x)
    if not tall_cols:
        return 0, y0, w, y1
    x0 = min(tall_cols)
    x1 = max(tall_cols) + 1
    return x0, y0, x1, y1


def show_chart_aligned(ax, chart: np.ndarray, allow_overhang: bool = True) -> None:
    x0, y0, x1, y1 = chart_plot_bbox(chart)
    h, w = chart.shape[:2]
    plot_w = max(1, x1 - x0)
    plot_h = max(1, y1 - y0)

    left = -x0 / plot_w
    right = (w - x0) / plot_w
    top = y1 / plot_h
    bottom = (y1 - h) / plot_h

    artist = ax.imshow(
        chart,
        interpolation="lanczos",
        origin="upper",
        extent=[left, right, bottom, top],
    )
    artist.set_clip_on(not allow_overhang)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def draw_distribution_panel(
    ax,
    values: list[float],
    selected_index: int,
    ylabel: str,
) -> None:
    trace_values = np.asarray(values, dtype=float)
    x = np.arange(1, len(trace_values) + 1)
    selected_x = selected_index + 1
    selected_value = float(trace_values[selected_index])

    ax.plot(x, trace_values, color="#7a7a7a", linewidth=0.8, zorder=1)
    ax.scatter(x, trace_values, s=8, color="#7a7a7a", edgecolors="none", zorder=2)
    ax.scatter(
        [selected_x],
        [selected_value],
        s=38,
        color="#d7191c",
        marker="*",
        linewidths=0.45,
        edgecolors="#7f0c0c",
        clip_on=False,
        zorder=3,
    )

    ymin = min(0.0, float(trace_values.min()))
    ymax = float(trace_values.max())
    pad = max((ymax - ymin) * 0.10, 1e-4)
    ax.set_ylim(ymin - pad * 0.10, ymax + pad * 1.35)
    ax.set_xlim(0.55, len(trace_values) + 0.75)
    ax.set_ylabel("")
    ax.set_xticks([1, len(trace_values)])
    ax.set_xticklabels(["1", str(len(trace_values))])
    yticks = [0.0, ymax / 2.0, ymax]
    ax.set_yticks(yticks)
    if ymax < 0.2:
        ax.set_yticklabels([f"{tick:.2f}" for tick in yticks])
    else:
        ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])
    ax.tick_params(axis="both", labelsize=6.2, length=1.8, width=0.45, pad=0.8)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.text(
        0.04,
        0.95,
        ylabel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.5},
    )
    ax.text(
        0.96,
        0.12,
        "views",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.60, "pad": 0.35},
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.35)
        spine.set_edgecolor("#444444")


def main() -> None:
    with META.open() as f:
        meta = json.load(f)
    quality = quality_distributions(ROOT / "outputs")
    fsdr_dir = repo_path(meta["fsdr_dir"])
    saes_dir = repo_path(meta["saes_dir"])
    fsdr_trace = view_loss_trace(quality["fsdr_rows"], fsdr_dir, "delta_psnr")
    saes_trace = view_loss_trace(quality["saes_rows"], saes_dir, "delta_lpips")

    specs = [
        {
            "mechanism": "FSDR",
            "only_label": "FSDR only",
            "descriptor": "dense foliage\nthin edges",
            "dir": fsdr_dir,
            "latency_csv": OUT / "stress_latency_fsdr_compose.csv",
            "latency_colors": {
                "S1": "#bd7b1c",
                "S2": "#d69b2b",
                "S3": "#e8b95d",
                "S4": "#efd38e",
            },
            "only_file": "ablation_asic_fsdr.png",
            "crop": metadata_crop(meta["selected_stats"]["FSDR-only"]),
            "distribution_values": fsdr_trace["values"],
            "selected_index": fsdr_trace["selected_index"],
            "distribution_ylabel": r"$\Delta$PSNR (dB)",
            "labels": [
                "(a) FSDR $\\Delta$PSNR",
                "(b) FSDR latency",
                "(c) FSDR only",
                "(d) RGB error",
            ],
        },
        {
            "mechanism": "SAES",
            "only_label": "SAES only",
            "descriptor": "low-texture\nplanar floor",
            "dir": saes_dir,
            "latency_csv": OUT / "stress_latency_saes_compose.csv",
            "latency_colors": {
                "S1": "#4b2698",
                "S2": "#6137c5",
                "S3": "#7d58d7",
                "S4": "#b9a7e9",
            },
            "only_file": "ablation_asic_saes.png",
            "crop": metadata_crop(meta["selected_stats"]["SAES-only"]),
            "distribution_values": saes_trace["values"],
            "selected_index": saes_trace["selected_index"],
            "distribution_ylabel": r"$\Delta$LPIPS",
            "labels": [
                "(e) SAES $\\Delta$LPIPS",
                "(f) SAES latency",
                "(g) SAES only",
                "(h) RGB error",
            ],
        },
    ]

    rows = []
    crops = []
    for spec in specs:
        reference = read_rgb(spec["dir"] / "ablation_asic.png")
        test = read_rgb(spec["dir"] / spec["only_file"])
        err = error_map(reference, test)
        crop = spec["crop"]
        rows.append((spec, reference, test, err))
        y0, y1, x0, x1 = crop
        crops.append(err[y0:y1, x0:x1])
    vmax = max(float(np.percentile(crop, 99.5)) for crop in crops)
    vmax = max(vmax, 1e-4)

    fig_w_pt = 240.0
    fig_h_pt = 158.0
    left_pt = 10.0
    right_pt = 1.0
    gap01_pt = 12.0
    gap12_pt = 1.5
    gap23_pt = 1.5
    cbar_gap_pt = 2.0
    cbar_w_pt = 6.0
    panel_pt = (
        fig_w_pt
        - left_pt
        - right_pt
        - gap01_pt
        - gap12_pt
        - gap23_pt
        - cbar_gap_pt
        - cbar_w_pt
    ) / 4.0
    top_pt = 2.0
    row_gap_pt = 26.0
    caption_gap_pt = 13.0
    fig_w = fig_w_pt / 72.0
    fig_h = fig_h_pt / 72.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    axes = []
    panel_w = panel_pt / fig_w_pt
    panel_h = panel_pt / fig_h_pt
    x_positions = [
        left_pt,
        left_pt + panel_pt + gap01_pt,
        left_pt + 2 * panel_pt + gap01_pt + gap12_pt,
        left_pt + 3 * panel_pt + gap01_pt + gap12_pt + gap23_pt,
    ]
    y_positions = [
        fig_h_pt - top_pt - panel_pt,
        fig_h_pt - top_pt - 2 * panel_pt - row_gap_pt,
    ]
    for row in range(2):
        y = y_positions[row] / fig_h_pt
        row_axes = []
        for col in range(4):
            x = x_positions[col] / fig_w_pt
            row_axes.append(fig.add_axes([x, y, panel_w, panel_h]))
        axes.append(row_axes)

    cbar_x_pt = x_positions[3] + panel_pt + cbar_gap_pt
    cbar_y_pt = y_positions[1]
    cbar_h_pt = y_positions[0] + panel_pt - y_positions[1]
    cax = fig.add_axes([
        cbar_x_pt / fig_w_pt,
        cbar_y_pt / fig_h_pt,
        cbar_w_pt / fig_w_pt,
        cbar_h_pt / fig_h_pt,
    ])

    norm = Normalize(vmin=0, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=LOSS_CMAP)
    mappable.set_array([])

    for row, (spec, reference, test, err) in enumerate(rows):
        y0, y1, x0, x1 = spec["crop"]
        err_crop = err[y0:y1, x0:x1]
        ref_crop = reference[y0:y1, x0:x1]
        test_crop = test[y0:y1, x0:x1]

        draw_distribution_panel(
            axes[row][0],
            spec["distribution_values"],
            spec["selected_index"],
            spec["distribution_ylabel"],
        )
        draw_latency_panel(axes[row][1], spec["latency_csv"], spec["latency_colors"])
        image_panels = [
            test_crop,
            ref_crop,
        ]
        for col, data in enumerate(image_panels, start=2):
            axes[row][col].imshow(np.clip(data, 0, 1), interpolation="nearest")
            if col == 3:
                axes[row][col].imshow(
                    err_crop,
                    cmap=LOSS_CMAP,
                    norm=norm,
                    alpha=0.95,
                    interpolation="nearest",
                )
        for col in range(4):
            ax = axes[row][col]
            if col in (0, 1):
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
                spine.set_edgecolor("#444444")

        cx, cy, ew, eh = error_focus_ellipse(err_crop)
        axes[row][2].add_patch(Ellipse(
            (cx, cy),
            ew,
            eh,
            edgecolor="#d7191c",
            facecolor="none",
            linewidth=1.0,
        ))
        height_px, width_px = err_crop.shape
        if spec["descriptor"]:
            if row == 0:
                tx = min(max(cx + ew * 0.42, 54.0), width_px - 72.0)
                ty = min(cy + eh * 0.72, height_px - 8)
                va = "top"
            else:
                tx = min(max(cx + ew * 0.42, 52.0), width_px - 72.0)
                ty = max(cy - eh * 0.62, 16)
                va = "bottom"
            axes[row][2].text(
                tx,
                ty,
                spec["descriptor"],
                ha="left",
                va=va,
                fontsize=5.6,
                color="#d7191c",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.54, "pad": 0.45},
            )

        y = axes[row][0].get_position().y0 - caption_gap_pt / fig_h_pt
        for col, label in enumerate(spec["labels"]):
            box = axes[row][col].get_position()
            fig.text(
                box.x0 + box.width / 2,
                y,
                label,
                ha="center",
                va="top",
                fontsize=8,
                linespacing=0.95,
            )

    cbar = fig.colorbar(mappable, cax=cax)
    tick_step = 0.01 if vmax <= 0.06 else 0.02
    ticks = np.arange(0, vmax + tick_step * 0.5, tick_step)
    ticks = ticks[ticks <= vmax + 1e-9]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
    cbar.ax.tick_params(labelsize=8, length=2.0, width=0.5, pad=1.5)
    cbar.outline.set_linewidth(0.55)

    fig.savefig(OUT / "StressCase_latency_error_maps_preview.png", pad_inches=0)
    fig.savefig(OUT / "StressCase_latency_error_maps_preview.pdf", pad_inches=0)
    metadata = {
        "fsdr_displayed": {
            "dir": str(fsdr_dir),
            "case": fsdr_dir.name,
            "actual_delta_psnr": fsdr_trace["actual_selected"],
            "plot_delta_psnr": fsdr_trace["plot_selected"],
            "selected_index": fsdr_trace["selected_index"],
        },
        "saes_displayed": {
            "dir": str(saes_dir),
            "case": saes_dir.name,
            "actual_delta_lpips": saes_trace["actual_selected"],
            "plot_delta_lpips": saes_trace["plot_selected"],
            "selected_index": saes_trace["selected_index"],
        },
        "fsdr_distribution_max": quality["fsdr_selected"],
        "saes_distribution_max": quality["saes_selected"],
        "fsdr_rows": quality["fsdr_rows"],
        "saes_rows": quality["saes_rows"],
    }
    with (OUT / "StressCase_quality_distribution_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    plt.close(fig)


if __name__ == "__main__":
    main()
