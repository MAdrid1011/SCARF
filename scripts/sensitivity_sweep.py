#!/usr/bin/env python3
"""
Sensitivity sweep for SCARF evaluation Studies 1–4.
Runs demo.py with varying parameters, captures stdout, and parses metrics.
"""
import subprocess, sys, re, json, os

PYTHON = "/home/madrid/anaconda3/envs/transplat/bin/python"
DEMO   = os.path.join(os.path.dirname(__file__), "demo.py")
MODEL  = "transplat"

STUDIES = {
    "study1_cache_size": {
        "param": "--fsgr-cache-size",
        "values": [64, 128, 256, 512, 1024],
    },
    "study2_hamming": {
        "param": "--fsgr-hamming",
        "values": [1, 2, 3, 4, 5],
    },
    "study3_saes_fv": {
        "param": "--saes-fv",
        "values": [0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    },
    "study4_tile_size": {
        "param": "--tile-size",
        "values": [4, 8, 16],
    },
}

def run_one(extra_args: list[str], tag: str) -> str:
    cmd = [PYTHON, DEMO, "--model", MODEL, "--ablation"] + extra_args
    print(f"\n{'='*60}")
    print(f"[SWEEP] {tag}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = result.stdout + "\n" + result.stderr
    outfile = f"/home/madrid/Desktop/SCARF/scripts/sweep_out/{tag}.txt"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        f.write(out)
    return out

def parse_output(text: str) -> dict:
    """Extract key metrics from demo.py stdout."""
    d = {}

    # FSGR stats
    m = re.search(r"Cache hit rate:\s+([\d.]+)%", text)
    if m: d["fsgr_hit_rate"] = float(m.group(1))
    m = re.search(r"Guided.*?:\s+[\d,]+\s+\(([\d.]+)%\)", text)
    if m: d["fsgr_guided_pct"] = float(m.group(1))
    m = re.search(r"Full compute \(miss\):\s+[\d,]+\s+\(([\d.]+)%\)", text)
    if m: d["fsgr_full_compute_pct"] = float(m.group(1))
    m = re.search(r"S2 saving per guided:\s+([\d.]+)%", text)
    if m: d["fsgr_s2_saving_per_guided"] = float(m.group(1))
    m = re.search(r"In window:\s+[\d,]+\s+\(([\d.]+)%", text)
    if m: d["fsgr_in_window_pct"] = float(m.group(1))
    m = re.search(r"Out of window:\s+(\d+)", text)
    if m: d["fsgr_out_window"] = int(m.group(1))

    # SAES stats
    m = re.search(r"Level 0 \(feat\):\s+(\d+)\s+tiles\s+\(([\d.]+)%\)", text)
    if m:
        d["saes_l0_tiles"] = int(m.group(1))
        d["saes_l0_pct"] = float(m.group(2))
    m = re.search(r"Level 1 \(depth\):\s*(\d+)\s+tiles\s+\(([\d.]+)%\)", text)
    if m:
        d["saes_l1_tiles"] = int(m.group(1))
        d["saes_l1_pct"] = float(m.group(2))
    m = re.search(r"Full \(no skip\):\s+(\d+)\s+tiles\s+\(([\d.]+)%\)", text)
    if m:
        d["saes_full_tiles"] = int(m.group(1))
        d["saes_full_pct"] = float(m.group(2))
    m = re.search(r"Total modified:\s+([\d,]+)/([\d,]+)\s+\(([\d.]+)%\)", text)
    if m:
        d["saes_modified_pixels"] = int(m.group(1).replace(",", ""))
        d["saes_total_pixels"] = int(m.group(2).replace(",", ""))
        d["saes_modified_pct"] = float(m.group(3))
    m = re.search(r"Total tiles:\s+(\d+)", text)
    if m: d["saes_total_tiles"] = int(m.group(1))

    # Quality — look for the ablation table's ASIC + SAES+FSGR line
    m = re.search(r"ASIC \+ SAES\+FSGR\s*:\s*PSNR=([\d.]+)\s*dB,\s*SSIM=([\d.]+),\s*loss=([-+\d.]+)\s*dB\s+\(([\d.]+)%\)", text)
    if m:
        d["psnr"] = float(m.group(1))
        d["ssim"] = float(m.group(2))
        d["psnr_loss_db"] = float(m.group(3))
        d["psnr_loss_pct"] = float(m.group(4))

    # Also capture no-opt PSNR as reference
    m = re.search(r"ASIC \(no opt\)\s*:\s*PSNR=([\d.]+)\s*dB", text)
    if m: d["psnr_noopt"] = float(m.group(1))

    # SAES-only quality
    m = re.search(r"ASIC \+ SAES\s*:\s*PSNR=([\d.]+)\s*dB,\s*SSIM=([\d.]+),\s*loss=([-+\d.]+)\s*dB", text)
    if m:
        d["psnr_saes_only"] = float(m.group(1))
        d["psnr_loss_saes_db"] = float(m.group(3))

    # FSGR-only quality
    m = re.search(r"ASIC \+ FSGR\s*:\s*PSNR=([\d.]+)\s*dB,\s*SSIM=([\d.]+),\s*loss=([-+\d.]+)\s*dB", text)
    if m:
        d["psnr_fsgr_only"] = float(m.group(1))
        d["psnr_loss_fsgr_db"] = float(m.group(3))

    # Combined S2 saving from ablation
    m = re.search(r"FSGR S2.*?saving.*?:\s*([\d.]+)%", text)
    if m: d["fsgr_s2_saving_total"] = float(m.group(1))

    # Speedup from ablation
    m = re.search(r"\+FSGR\+SAES.*?speedup.*?([\d.]+)×", text, re.IGNORECASE)
    if m: d["speedup_combined"] = float(m.group(1))

    return d


def main():
    all_results = {}

    for study_name, study_cfg in STUDIES.items():
        print(f"\n{'#'*70}")
        print(f"# {study_name}")
        print(f"{'#'*70}")
        study_results = []

        for val in study_cfg["values"]:
            tag = f"{study_name}_{val}"
            extra = [study_cfg["param"], str(val)]
            out = run_one(extra, tag)
            metrics = parse_output(out)
            metrics["value"] = val
            study_results.append(metrics)
            print(f"  => Parsed: {json.dumps(metrics, indent=2)}")

        all_results[study_name] = study_results

    # Write consolidated JSON
    outpath = "/home/madrid/Desktop/SCARF/scripts/sweep_out/all_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to {outpath}")

    # Print summary tables
    print("\n\n" + "="*80)
    print("SENSITIVITY SWEEP SUMMARY")
    print("="*80)

    print("\n### Study 1: FSDR Cache Size")
    print(f"{'Entries':>8} {'HitRate':>8} {'Guided%':>8} {'S2Save':>8} {'PSNRloss':>10}")
    for r in all_results.get("study1_cache_size", []):
        print(f"{r['value']:>8} {r.get('fsgr_hit_rate',0):>7.1f}% {r.get('fsgr_guided_pct',0):>7.1f}% "
              f"{r.get('fsgr_s2_saving_per_guided',0):>7.1f}% {r.get('psnr_loss_fsgr_db',0):>+9.3f} dB")

    print("\n### Study 2: FSDR Hamming Threshold")
    print(f"{'τ_h':>8} {'HitRate':>8} {'Guided%':>8} {'PSNRloss':>10}")
    for r in all_results.get("study2_hamming", []):
        print(f"{r['value']:>8} {r.get('fsgr_hit_rate',0):>7.1f}% {r.get('fsgr_guided_pct',0):>7.1f}% "
              f"{r.get('psnr_loss_fsgr_db',0):>+9.3f} dB")

    print("\n### Study 3: SAES Feature Variance Threshold")
    print(f"{'τ_f':>8} {'L0Tiles%':>9} {'ModPx%':>8} {'PSNRloss':>10}")
    for r in all_results.get("study3_saes_fv", []):
        print(f"{r['value']:>8.2f} {r.get('saes_l0_pct',0):>8.1f}% {r.get('saes_modified_pct',0):>7.1f}% "
              f"{r.get('psnr_loss_saes_db',0):>+9.3f} dB")

    print("\n### Study 4: Tile Size")
    print(f"{'TileSize':>8} {'L0Tiles%':>9} {'ModPx%':>8} {'#Tiles':>8} {'PSNRloss':>10}")
    for r in all_results.get("study4_tile_size", []):
        print(f"{r['value']:>7}×{r['value']} {r.get('saes_l0_pct',0):>8.1f}% {r.get('saes_modified_pct',0):>7.1f}% "
              f"{r.get('saes_total_tiles',0):>8} {r.get('psnr_loss_saes_db',0):>+9.3f} dB")


if __name__ == "__main__":
    main()
