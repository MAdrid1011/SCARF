#!/usr/bin/env python3
"""Re-parse sweep output files with corrected regexes."""
import re, json, os, glob

SWEEP_DIR = "/home/madrid/Desktop/SCARF/scripts/sweep_out"

def parse_output(text: str) -> dict:
    d = {}

    # FSGR stats (from the detailed stats section)
    m = re.search(r"Cache hit rate:\s+([\d.]+)%", text)
    if m: d["fsgr_hit_rate"] = float(m.group(1))

    m = re.search(r"Guided \(\d+ cand\.\):\s+([\d,]+)\s+\(([\d.]+)%\)", text)
    if m:
        d["fsgr_guided"] = int(m.group(1).replace(",", ""))
        d["fsgr_guided_pct"] = float(m.group(2))

    m = re.search(r"Full compute \(miss\):\s+([\d,]+)\s+\(([\d.]+)%\)", text)
    if m:
        d["fsgr_full_compute"] = int(m.group(1).replace(",", ""))
        d["fsgr_full_compute_pct"] = float(m.group(2))

    m = re.search(r"S2 saving per guided:\s+([\d.]+)%", text)
    if m: d["fsgr_s2_saving_per_guided"] = float(m.group(1))

    m = re.search(r"In window:\s+([\d,]+)\s+\(([\d.]+)%", text)
    if m:
        d["fsgr_in_window"] = int(m.group(1).replace(",", ""))
        d["fsgr_in_window_pct"] = float(m.group(2))

    m = re.search(r"Out of window:\s+(\d+)", text)
    if m: d["fsgr_out_window"] = int(m.group(1))

    m = re.search(r"Depth inconsistent:\s+([\d,]+)", text)
    if m: d["fsgr_depth_inconsistent"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Pixels processed:\s+([\d,]+)", text)
    if m: d["fsgr_total_pixels"] = int(m.group(1).replace(",", ""))

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
    m = re.search(r"Effective Gaussians.*?:\s+([\d,]+)\s+Zeroed.*?:\s+([\d,]+)", text)
    if m:
        d["saes_effective_gaussians"] = int(m.group(1).replace(",", ""))
        d["saes_zeroed_gaussians"] = int(m.group(2).replace(",", ""))

    # Quality — FSGR+SAES combined
    m = re.search(r"ASIC \+ SAES\+FSGR\s*:\s*PSNR=([\d.]+)\s*dB,\s*SSIM=([\d.]+),\s*loss=([-+]?[\d.]+)\s*dB\s+\(([\d.]+)%\)", text)
    if m:
        d["psnr_combined"] = float(m.group(1))
        d["ssim_combined"] = float(m.group(2))
        d["psnr_loss_combined_db"] = float(m.group(3))

    # ASIC (no opt) reference
    m = re.search(r"ASIC \(no opt\)\s*:\s*PSNR=([\d.]+)\s*dB,\s*SSIM=([\d.]+)", text)
    if m:
        d["psnr_noopt"] = float(m.group(1))
        d["ssim_noopt"] = float(m.group(2))

    # SAES-only quality (note: "ASIC + SAES v4")
    m = re.search(r"ASIC \+ SAES v4\s*:\s*PSNR=([\d.]+)\s*dB,\s*SSIM=([\d.]+),\s*loss=([-+]?[\d.]+)\s*dB", text)
    if m:
        d["psnr_saes_only"] = float(m.group(1))
        d["ssim_saes_only"] = float(m.group(2))
        d["psnr_loss_saes_db"] = float(m.group(3))

    # FSGR-only quality
    m = re.search(r"ASIC \+ FSGR\s*:\s*PSNR=([\d.]+)\s*dB,\s*SSIM=([\d.]+),\s*loss=([-+]?[\d.]+)\s*dB", text)
    if m:
        d["psnr_fsgr_only"] = float(m.group(1))
        d["ssim_fsgr_only"] = float(m.group(2))
        d["psnr_loss_fsgr_db"] = float(m.group(3))

    # S2/S3 savings from ablation
    m = re.search(r"FSGR S2 saving \(alone\):\s*([\d.]+)%", text)
    if m: d["fsgr_s2_saving_alone"] = float(m.group(1))
    m = re.search(r"SAES S2 saving \(alone\):\s*([\d.]+)%", text)
    if m: d["saes_s2_saving_alone"] = float(m.group(1))
    m = re.search(r"SAES S3 saving \(alone\):\s*([\d.]+)%", text)
    if m: d["saes_s3_saving_alone"] = float(m.group(1))
    m = re.search(r"Combined S2 saving:\s*([\d.]+)%", text)
    if m: d["combined_s2_saving"] = float(m.group(1))

    return d


STUDIES = {
    "study1_cache_size": {
        "prefix": "study1_cache_size_",
        "values": [64, 128, 256, 512, 1024],
    },
    "study2_hamming": {
        "prefix": "study2_hamming_",
        "values": [1, 2, 3, 4, 5],
    },
    "study3_saes_fv": {
        "prefix": "study3_saes_fv_",
        "values": [0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    },
    "study4_tile_size": {
        "prefix": "study4_tile_size_",
        "values": [4, 8, 16],
    },
}

all_results = {}
for study_name, cfg in STUDIES.items():
    results = []
    for val in cfg["values"]:
        fname = os.path.join(SWEEP_DIR, f"{cfg['prefix']}{val}.txt")
        if not os.path.exists(fname):
            print(f"  MISSING: {fname}")
            continue
        with open(fname) as f:
            text = f.read()
        metrics = parse_output(text)
        metrics["value"] = val
        results.append(metrics)
    all_results[study_name] = results

# Save
with open(os.path.join(SWEEP_DIR, "all_results_v2.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# Print summary
print("="*90)
print("SENSITIVITY SWEEP RESULTS (re-parsed)")
print("="*90)

print("\n### Study 1: FSDR Cache Size (TranSplat, 256×256)")
print(f"{'Entries':>8} | {'HitRate':>8} | {'Guided%':>8} | {'S2/px':>7} | {'InWin%':>7} | {'OutWin':>6} | {'PSNR_fsgr':>10} | {'PSNR_loss':>10}")
print("-"*90)
for r in all_results["study1_cache_size"]:
    print(f"{r['value']:>8} | {r.get('fsgr_hit_rate',0):>7.1f}% | {r.get('fsgr_guided_pct',0):>7.1f}% | "
          f"{r.get('fsgr_s2_saving_per_guided',0):>6.1f}% | {r.get('fsgr_in_window_pct',0):>6.1f}% | "
          f"{r.get('fsgr_out_window',0):>6} | {r.get('psnr_fsgr_only',0):>9.4f} | "
          f"{r.get('psnr_loss_fsgr_db',0):>+9.4f} dB")

print(f"\n### Study 2: FSDR Hamming Threshold τ_h (TranSplat)")
print(f"{'τ_h':>5} | {'HitRate':>8} | {'Guided%':>8} | {'InWin%':>7} | {'OutWin':>6} | {'PSNR_fsgr':>10} | {'PSNR_loss':>10}")
print("-"*80)
for r in all_results["study2_hamming"]:
    print(f"{r['value']:>5} | {r.get('fsgr_hit_rate',0):>7.1f}% | {r.get('fsgr_guided_pct',0):>7.1f}% | "
          f"{r.get('fsgr_in_window_pct',0):>6.1f}% | {r.get('fsgr_out_window',0):>6} | "
          f"{r.get('psnr_fsgr_only',0):>9.4f} | {r.get('psnr_loss_fsgr_db',0):>+9.4f} dB")

print(f"\n### Study 3: SAES Feature Variance Threshold τ_f (TranSplat)")
print(f"{'τ_f':>6} | {'L0Tiles':>8} | {'L0%':>6} | {'ModPx':>8} | {'ModPx%':>7} | {'Zeroed':>7} | {'PSNR_saes':>10} | {'Loss':>10}")
print("-"*90)
for r in all_results["study3_saes_fv"]:
    print(f"{r['value']:>6.2f} | {r.get('saes_l0_tiles',0):>8} | {r.get('saes_l0_pct',0):>5.1f}% | "
          f"{r.get('saes_modified_pixels',0):>8,} | {r.get('saes_modified_pct',0):>6.1f}% | "
          f"{r.get('saes_zeroed_gaussians',0):>7,} | {r.get('psnr_saes_only',0):>9.4f} | "
          f"{r.get('psnr_loss_saes_db',0):>+9.4f} dB")

print(f"\n### Study 4: Tile Size (TranSplat)")
print(f"{'TileSize':>8} | {'#Tiles':>7} | {'L0Tiles':>8} | {'L0%':>6} | {'ModPx':>8} | {'ModPx%':>7} | {'PSNR_saes':>10} | {'Loss':>10}")
print("-"*95)
for r in all_results["study4_tile_size"]:
    ts = r['value']
    print(f"{ts:>5}×{ts:<2} | {r.get('saes_total_tiles',0):>7} | {r.get('saes_l0_tiles',0):>8} | "
          f"{r.get('saes_l0_pct',0):>5.1f}% | {r.get('saes_modified_pixels',0):>8,} | "
          f"{r.get('saes_modified_pct',0):>6.1f}% | {r.get('psnr_saes_only',0):>9.4f} | "
          f"{r.get('psnr_loss_saes_db',0):>+9.4f} dB")
