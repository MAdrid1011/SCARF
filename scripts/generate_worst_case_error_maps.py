#!/usr/bin/env python3
"""Generate mechanism-level FSDR and SAES error maps from demo ablations."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rebuttal-micro" / "figures"
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


def error_map(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(reference - test), axis=2)


def crop_from_error(err: np.ndarray, size: int = 120) -> tuple[int, int, int, int]:
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


def candidate_stats(demo_dir: Path, mechanism: str) -> dict[str, float | str]:
    reference = read_rgb(demo_dir / "ablation_asic.png")
    test = read_rgb(demo_dir / f"ablation_asic_{mechanism}.png")
    err = error_map(reference, test)
    return {
        "dir": str(demo_dir),
        "mechanism": mechanism,
        "mean_error": float(err.mean()),
        "p95_error": float(np.percentile(err, 95)),
        "p99_error": float(np.percentile(err, 99)),
        "max_error": float(err.max()),
    }


def discover_candidate_dirs(scan_root: Path) -> list[Path]:
    dirs = []
    for path in sorted(scan_root.rglob("*")):
        if not path.is_dir():
            continue
        needed = [
            path / "ablation_asic.png",
            path / "ablation_asic_fsdr.png",
            path / "ablation_asic_saes.png",
        ]
        if all(p.exists() for p in needed):
            dirs.append(path)
    if all((scan_root / f).exists() for f in [
        "ablation_asic.png",
        "ablation_asic_fsdr.png",
        "ablation_asic_saes.png",
    ]):
        dirs.insert(0, scan_root)
    return dirs


def save_mechanism_figure(
    name: str,
    reference: np.ndarray,
    test: np.ndarray,
    title: str,
    output_stem: str,
    crop: tuple[int, int, int, int] | None = None,
    crop_size: int = 120,
) -> dict[str, float]:
    err = error_map(reference, test)
    if crop is None:
        crop = crop_from_error(err, crop_size)
    y0, y1, x0, x1 = crop
    err_crop = err[y0:y1, x0:x1]
    vmax = max(float(np.percentile(err_crop, 99.5)), 1e-4)
    high = err_crop > np.percentile(err_crop, 95)

    panels = [
        ("No-opt SCARF", reference[y0:y1, x0:x1], None),
        (name, test[y0:y1, x0:x1], None),
        ("Absolute RGB error", err_crop, "magma"),
        ("High-error mask", high.astype(float), "gray_r"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.82), dpi=240)
    for ax, (label, data, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=(vmax if label == "RGB error" else 1))
        ax.set_title(label, fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#444444")

    fig.suptitle(title, fontsize=8.5, y=1.02)
    fig.tight_layout(pad=0.35, w_pad=0.35)

    png = OUT / f"{output_stem}.png"
    pdf = OUT / f"{output_stem}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {
        "mean_error": float(err.mean()),
        "p95_error": float(np.percentile(err, 95)),
        "max_error": float(err.max()),
        "crop_y0": float(y0),
        "crop_y1": float(y1),
        "crop_x0": float(x0),
        "crop_x1": float(x1),
    }


def save_combined(
    fsdr_reference: np.ndarray,
    fsdr: np.ndarray,
    saes_reference: np.ndarray,
    saes: np.ndarray,
    crop_size: int = 120,
) -> None:
    fsdr_err = error_map(fsdr_reference, fsdr)
    saes_err = error_map(saes_reference, saes)
    crop_fsdr = crop_from_error(fsdr_err, crop_size)
    crop_saes = crop_from_error(saes_err, crop_size)

    specs = [
        ("FSDR-only", fsdr_reference, fsdr, fsdr_err, crop_fsdr),
        ("SAES-only", saes_reference, saes, saes_err, crop_saes),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.5), dpi=240)
    for row, (name, reference, test, err, crop) in enumerate(specs):
        y0, y1, x0, x1 = crop
        err_crop = err[y0:y1, x0:x1]
        vmax = max(float(np.percentile(err_crop, 99.5)), 1e-4)
        high = err_crop > np.percentile(err_crop, 95)
        panels = [
            ("No-opt SCARF", reference[y0:y1, x0:x1], None),
            (name, test[y0:y1, x0:x1], None),
            ("Absolute RGB error", err_crop, "magma"),
            ("High-error mask", high.astype(float), "gray_r"),
        ]
        for ax, (label, data, cmap) in zip(axes[row], panels):
            if cmap is None:
                ax.imshow(np.clip(data, 0, 1))
            else:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=(vmax if label == "RGB error" else 1))
            ax.set_title(label, fontsize=8, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#444444")
        axes[row, 0].set_ylabel(name, fontsize=8)

    fig.tight_layout(pad=0.35, w_pad=0.35, h_pad=0.5)
    fig.savefig(OUT / "WorstCase_error_maps.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "WorstCase_error_maps.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_paper_preview(
    fsdr_reference: np.ndarray,
    fsdr: np.ndarray,
    saes_reference: np.ndarray,
    saes: np.ndarray,
    error_vmax: float | None = None,
    crop_size: int = 120,
) -> None:
    fsdr_err = error_map(fsdr_reference, fsdr)
    saes_err = error_map(saes_reference, saes)
    specs = [
        (
            "FSDR",
            "FSDR-only",
            "dense foliage\nthin edges",
            fsdr_reference,
            fsdr,
            fsdr_err,
            crop_from_error(fsdr_err, crop_size),
        ),
        (
            "SAES",
            "SAES-only",
            "low-texture\nplanar floor",
            saes_reference,
            saes,
            saes_err,
            crop_from_error(saes_err, crop_size),
        ),
    ]
    error_crops = []
    for _, _, _, _, _, err, crop in specs:
        y0, y1, x0, x1 = crop
        error_crops.append(err[y0:y1, x0:x1])
    global_vmax = (
        error_vmax
        if error_vmax is not None
        else max(float(np.percentile(crop, 99.5)) for crop in error_crops)
    )
    global_vmax = max(float(global_vmax), 1e-4)

    fig = plt.figure(figsize=(3.77, 2.70), dpi=300)
    fig_w, fig_h = fig.get_size_inches()
    gap_pt = 0.8
    gap_x = (gap_pt / 72.0) / fig_w
    gap_y = (gap_pt / 72.0) / fig_h
    left, right, top = 0.005, 0.82, 0.965
    cbar_gap = gap_x
    cbar_w = 0.026
    panel_w = (right - left - 2 * gap_x - cbar_gap - cbar_w) / 3
    panel_h = panel_w * fig_w / fig_h
    caption_gap = 0.010
    caption_block = 0.090
    axes = []
    for row in range(2):
        y = top - (row + 1) * panel_h - row * (gap_y + caption_block)
        row_axes = []
        for col in range(3):
            x = left + col * (panel_w + gap_x)
            row_axes.append(fig.add_axes([x, y, panel_w, panel_h]))
        axes.append(row_axes)
    cax = fig.add_axes([
        left + 3 * panel_w + 2 * gap_x + cbar_gap,
        top - 2 * panel_h - gap_y - caption_block,
        cbar_w,
        2 * panel_h + gap_y + caption_block,
    ])
    norm = Normalize(vmin=0, vmax=global_vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=LOSS_CMAP)
    mappable.set_array([])

    for row, (row_label, mechanism_label, descriptor, reference, test, err, crop) in enumerate(specs):
        y0, y1, x0, x1 = crop
        err_crop = err[y0:y1, x0:x1]
        height_px, width_px = err_crop.shape
        panels = [
            ("No-opt", reference[y0:y1, x0:x1], None),
            (mechanism_label, test[y0:y1, x0:x1], None),
            ("Loss heatmap", reference[y0:y1, x0:x1], LOSS_CMAP),
        ]
        for ax, (label, data, cmap) in zip(axes[row], panels):
            if cmap is None:
                ax.imshow(np.clip(data, 0, 1), interpolation="nearest")
            else:
                ax.imshow(np.clip(data, 0, 1), interpolation="nearest")
                ax.imshow(
                    err_crop,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.95,
                    interpolation="nearest",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
                spine.set_edgecolor("#444444")
        cx, cy, ew, eh = error_focus_ellipse(err_crop)
        axes[row][1].add_patch(Ellipse(
            (cx, cy),
            ew,
            eh,
            edgecolor="#d7191c",
            facecolor="none",
            linewidth=1.0,
        ))
        if row == 0:
            tx = min(cx + ew * 0.56, width_px - 4)
            ty = min(cy + eh * 0.78, height_px - 5)
            va = "top"
        else:
            tx = min(cx + ew * 0.56, width_px - 4)
            ty = max(cy - eh * 0.70, 5)
            va = "bottom"
        ha = "left" if tx < width_px * 0.72 else "right"
        axes[row][1].text(
            tx,
            ty,
            descriptor,
            ha=ha,
            va=va,
            fontsize=8,
            color="#d7191c",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.54, "pad": 0.75},
        )

    subfigure_labels = [
        ["(a) FSDR worst case", "(b) FSDR only", "(c) RGB error map"],
        ["(d) SAES worst case", "(e) SAES only", "(f) RGB error map"],
    ]
    for row in range(2):
        y = axes[row][0].get_position().y0 - caption_gap
        for col in range(3):
            box = axes[row][col].get_position()
            fig.text(
                box.x0 + box.width / 2,
                y,
                subfigure_labels[row][col],
                ha="center",
                va="top",
                fontsize=8,
                linespacing=0.95,
            )

    cbar = fig.colorbar(mappable, cax=cax)
    tick_step = 0.01 if global_vmax <= 0.06 else 0.02
    ticks = np.arange(0, global_vmax + tick_step * 0.5, tick_step)
    ticks = ticks[ticks <= global_vmax + 1e-9]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
    cbar.ax.tick_params(labelsize=8, length=2.0, width=0.5, pad=1.5)
    cbar.outline.set_linewidth(0.55)

    fig.savefig(OUT / "WorstCase_error_maps_paper_preview.png", bbox_inches="tight", pad_inches=0.005)
    fig.savefig(OUT / "WorstCase_error_maps_paper_preview.pdf", bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=ROOT / "outputs" / "demo",
        help="Directory containing one or more demo output directories",
    )
    parser.add_argument(
        "--fsdr-dir",
        type=Path,
        default=None,
        help="Explicit demo output directory for the FSDR row",
    )
    parser.add_argument(
        "--saes-dir",
        type=Path,
        default=None,
        help="Explicit demo output directory for the SAES row",
    )
    parser.add_argument(
        "--paper-error-vmax",
        type=float,
        default=None,
        help="Fixed absolute RGB error vmax for the paper-preview error maps",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=120,
        help="Square crop size for error-map panels",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    candidate_dirs = discover_candidate_dirs(args.scan_root)
    if not candidate_dirs:
        raise FileNotFoundError(f"No demo ablation outputs found under {args.scan_root}")

    rows = []
    for demo_dir in candidate_dirs:
        rows.append(candidate_stats(demo_dir, "fsdr"))
        rows.append(candidate_stats(demo_dir, "saes"))

    fsdr_dir = args.fsdr_dir or Path(max(
        (r for r in rows if r["mechanism"] == "fsdr"),
        key=lambda r: (r["p99_error"], r["p95_error"]),
    )["dir"])
    saes_dir = args.saes_dir or Path(max(
        (r for r in rows if r["mechanism"] == "saes"),
        key=lambda r: (r["p99_error"], r["p95_error"]),
    )["dir"])

    fsdr_reference = read_rgb(fsdr_dir / "ablation_asic.png")
    fsdr = read_rgb(fsdr_dir / "ablation_asic_fsdr.png")
    saes_reference = read_rgb(saes_dir / "ablation_asic.png")
    saes = read_rgb(saes_dir / "ablation_asic_saes.png")

    stats = {
        "FSDR-only": save_mechanism_figure(
            "FSDR-only",
            fsdr_reference,
            fsdr,
            f"FSDR-adverse patch: {fsdr_dir.name}",
            "WorstCase_FSDR_error_map",
            crop_size=args.crop_size,
        ),
        "SAES-only": save_mechanism_figure(
            "SAES-only",
            saes_reference,
            saes,
            f"SAES-adverse patch: {saes_dir.name}",
            "WorstCase_SAES_error_map",
            crop_size=args.crop_size,
        ),
    }
    save_combined(fsdr_reference, fsdr, saes_reference, saes, args.crop_size)
    save_paper_preview(
        fsdr_reference,
        fsdr,
        saes_reference,
        saes,
        args.paper_error_vmax,
        args.crop_size,
    )

    metadata = {
        "fsdr_dir": str(fsdr_dir),
        "saes_dir": str(saes_dir),
        "ranked_candidates": sorted(
            rows,
            key=lambda r: (r["mechanism"], -r["p99_error"], -r["p95_error"]),
        ),
        "selected_stats": stats,
    }
    with (OUT / "WorstCase_error_maps_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Selected FSDR dir: {fsdr_dir}")
    print(f"Selected SAES dir: {saes_dir}")
    for name, values in stats.items():
        parts = ", ".join(f"{k}={v:.6f}" for k, v in values.items())
        print(f"{name}: {parts}")


if __name__ == "__main__":
    main()
