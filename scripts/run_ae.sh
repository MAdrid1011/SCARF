#!/usr/bin/env bash
# SCARF Artifact Evaluation — one-command reproduction script
# MICRO 2026
#
# Usage:
#   bash scripts/run_ae.sh [EXPERIMENT]
#
# EXPERIMENT may be one of:
#   all          — run all experiments below (~3 h on GPU, ~12 h CPU-only)
#   quality      — Table 2: PSNR / SSIM / LPIPS for all 9 model×dataset pairs
#   speedup      — Figure 5: end-to-end cycle-count speedup
#   ablation     — Figure 6: FSDR-only / SAES-only / combined ablation
#   sensitivity  — Figure 7: sensitivity sweep (cache size, Hamming, fv, tile)
#   quick        — quick smoke-test on Re10K×MVSplat only (~10 min on GPU)
#
# All results are written to outputs/<experiment>/<timestamp>/results.json
# and a human-readable summary is printed at the end.
#
# Requirements: conda environment 'scarf' must be active, or set PYTHON below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Python interpreter — defaults to the conda 'scarf' environment if active
PYTHON="${PYTHON:-python}"

DEMO="$SCRIPT_DIR/demo.py"
SWEEP="$SCRIPT_DIR/sensitivity_sweep.py"

EXPERIMENT="${1:-all}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTBASE="$ROOT/outputs/ae_${TS}"
mkdir -p "$OUTBASE"

LOG="$OUTBASE/run_ae.log"
echo "SCARF AE run — experiment=$EXPERIMENT  timestamp=$TS" | tee "$LOG"
echo "Output directory: $OUTBASE" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Helper: run one model × dataset pair, both baseline and SCARF
# ---------------------------------------------------------------------------
run_pair() {
    local model="$1"
    local tag="$2"
    local outdir="$OUTBASE/quality/${model}_${tag}"
    echo "  [quality] $model × $tag ..." | tee -a "$LOG"
    "$PYTHON" "$DEMO" \
        --model "$model" \
        --output-dir "$outdir" \
        2>&1 | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# quality — Table 2
# ---------------------------------------------------------------------------
run_quality() {
    echo "=== Table 2: Quality (PSNR / SSIM / LPIPS) ===" | tee -a "$LOG"
    mkdir -p "$OUTBASE/quality"

    for model in transplat mvsplat depthsplat; do
        run_pair "$model" "re10k"
    done
    for model in transplat mvsplat depthsplat; do
        run_pair "$model" "acid"
    done
    for model in transplat mvsplat depthsplat; do
        run_pair "$model" "dl3dv"
    done

    echo "" | tee -a "$LOG"
    echo "Quality results written to: $OUTBASE/quality/" | tee -a "$LOG"
    echo "To compare with Table 2, inspect each results.json for" | tee -a "$LOG"
    echo "  baseline_psnr, scarf_psnr, psnr_delta_pct fields." | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# speedup — Figure 5
# ---------------------------------------------------------------------------
run_speedup() {
    echo "=== Figure 5: End-to-End Speedup ===" | tee -a "$LOG"
    mkdir -p "$OUTBASE/speedup"

    for model in transplat mvsplat depthsplat; do
        echo "  [speedup] $model ..." | tee -a "$LOG"
        "$PYTHON" "$DEMO" \
            --model "$model" \
            --output-dir "$OUTBASE/speedup/${model}" \
            2>&1 | tee -a "$LOG"
    done

    echo "" | tee -a "$LOG"
    echo "Speedup results in: $OUTBASE/speedup/" | tee -a "$LOG"
    echo "Compare 'total_cycles_baseline' vs 'total_cycles_scarf' in results.json." | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# ablation — Figure 6
# ---------------------------------------------------------------------------
run_ablation() {
    echo "=== Figure 6: FSDR / SAES Ablation ===" | tee -a "$LOG"
    mkdir -p "$OUTBASE/ablation"

    for model in transplat mvsplat depthsplat; do
        echo "  [ablation] $model ..." | tee -a "$LOG"
        "$PYTHON" "$DEMO" \
            --model "$model" \
            --ablation \
            --output-dir "$OUTBASE/ablation/${model}" \
            2>&1 | tee -a "$LOG"
    done

    echo "" | tee -a "$LOG"
    echo "Ablation results in: $OUTBASE/ablation/" | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# sensitivity — Figure 7
# ---------------------------------------------------------------------------
run_sensitivity() {
    echo "=== Figure 7: Sensitivity Sweep ===" | tee -a "$LOG"
    mkdir -p "$OUTBASE/sensitivity"

    "$PYTHON" "$SWEEP" \
        --output-dir "$OUTBASE/sensitivity" \
        2>&1 | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "Sensitivity results in: $OUTBASE/sensitivity/" | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# quick smoke-test
# ---------------------------------------------------------------------------
run_quick() {
    echo "=== Quick smoke-test: MVSplat × Re10K ===" | tee -a "$LOG"
    "$PYTHON" "$DEMO" \
        --model mvsplat \
        --num-samples 1 \
        --output-dir "$OUTBASE/quick" \
        2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "Smoke-test complete. Check $OUTBASE/quick/results.json" | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$EXPERIMENT" in
    all)
        run_quality
        run_speedup
        run_ablation
        run_sensitivity
        ;;
    quality)     run_quality ;;
    speedup)     run_speedup ;;
    ablation)    run_ablation ;;
    sensitivity) run_sensitivity ;;
    quick)       run_quick ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Valid options: all quality speedup ablation sensitivity quick"
        exit 1
        ;;
esac

echo "" | tee -a "$LOG"
echo "======================================================" | tee -a "$LOG"
echo "SCARF AE run complete." | tee -a "$LOG"
echo "All outputs: $OUTBASE/" | tee -a "$LOG"
echo "Full log:    $LOG" | tee -a "$LOG"
